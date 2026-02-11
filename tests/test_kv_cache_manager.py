"""Tests for the KV Cache Manager (P1.1 through P1.18).

Covers:
- BlockPool pre-allocation and free queue (P1.1)
- compute_block_hash determinism and uniqueness (P1.2)
- find_cached_prefix full/partial/miss scenarios (P1.3)
- allocate_blocks cache hit, fresh alloc, ref_count tracking (P1.4)
- free_blocks ref_count decrement, block stays in hash table (P1.5)
- evict_lru ordering and skipping in-use blocks (P1.6)
- extract_block shapes and values (P1.7-P1.8)
- inject_blocks roundtrip (P1.9-P1.10)
- TieredKVCache RAM/SSD lookup, evict-to-SSD flow (P1.16-P1.18)
"""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.kv_cache_manager import (
    BlockPool,
    BlockPoolExhaustedError,
    KVCacheManager,
    TieredKVCache,
    _compute_chain_hash,
    compute_block_hash,
    compute_model_fingerprint,
    extract_block,
    inject_blocks,
)
from mlx_lm_server.ssd_cache import SSDCache
from mlx_lm_server.types import KVCacheBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from conftest import make_test_config


def make_config(**overrides) -> ServerConfig:
    """Create a ServerConfig with small defaults suitable for testing."""
    overrides.setdefault("num_blocks", 8)
    return make_test_config(**overrides)


# ---------------------------------------------------------------------------
# P1.1 — BlockPool pre-allocation
# ---------------------------------------------------------------------------

class TestBlockPool:
    """Tests for BlockPool initialization and free queue management."""

    def test_pool_init(self):
        """Pool creates the correct number of KVCacheBlock objects."""
        pool = BlockPool(num_blocks=16)
        assert len(pool.blocks) == 16
        assert all(isinstance(b, KVCacheBlock) for b in pool.blocks)
        # Each block should have a unique, sequential block_id
        assert [b.block_id for b in pool.blocks] == list(range(16))

    def test_pool_size(self):
        """Free queue starts with all blocks available."""
        pool = BlockPool(num_blocks=10)
        assert pool.num_free == 10
        # Allocate one
        block = pool.get_free_block()
        assert pool.num_free == 9
        # Return it
        pool.return_block(block.block_id)
        assert pool.num_free == 10

    def test_pool_exhaustion(self):
        """Raises BlockPoolExhaustedError when all blocks are allocated."""
        pool = BlockPool(num_blocks=2)
        pool.get_free_block()
        pool.get_free_block()
        with pytest.raises(BlockPoolExhaustedError):
            pool.get_free_block()

    def test_pool_zero_blocks_rejected(self):
        """Pool creation with num_blocks <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BlockPool(num_blocks=0)
        with pytest.raises(ValueError):
            BlockPool(num_blocks=-5)

    def test_pool_return_clears_metadata(self):
        """Returning a block to the pool clears its metadata."""
        pool = BlockPool(num_blocks=4)
        block = pool.get_free_block()
        block.block_hash = "hash_12345"
        block.token_ids = [1, 2, 3]
        block.ref_count = 2
        block.kv_data = {"layer0": "data"}

        pool.return_block(block.block_id)
        assert block.block_hash is None
        assert block.token_ids == []
        assert block.ref_count == 0
        assert block.kv_data is None


# ---------------------------------------------------------------------------
# P1.2 — compute_block_hash
# ---------------------------------------------------------------------------

class TestComputeBlockHash:
    """Tests for the block hash function."""

    def test_hash_determinism(self):
        """Same inputs produce the same hash every time."""
        prefix = [10, 20, 30, 40]
        block = [50, 60, 70, 80]
        h1 = compute_block_hash(prefix, block)
        h2 = compute_block_hash(prefix, block)
        h3 = compute_block_hash(prefix, block)
        assert h1 == h2 == h3

    def test_hash_uniqueness(self):
        """Different token sequences produce different hashes."""
        h1 = compute_block_hash([1, 2], [3, 4])
        h2 = compute_block_hash([1, 2], [3, 5])  # Different block tokens
        h3 = compute_block_hash([1, 3], [3, 4])  # Different prefix
        h4 = compute_block_hash([], [99, 98, 97, 96])  # Completely different tokens
        # All should be distinct (hash collisions are theoretically possible
        # but extremely unlikely for these small, distinct inputs)
        assert len({h1, h2, h3, h4}) == 4

    def test_hash_empty_prefix(self):
        """Hash works with an empty prefix (first block in sequence)."""
        h = compute_block_hash([], [1, 2, 3, 4])
        assert isinstance(h, str)

    def test_hash_empty_block_tokens(self):
        """Hash works with empty block tokens (edge case)."""
        h = compute_block_hash([1, 2, 3], [])
        assert isinstance(h, str)

    def test_hash_boundary_between_prefix_and_block(self):
        """Hash differentiates where the prefix/block boundary falls.

        With blake2b and a separator sentinel between prefix and block,
        [1,2,3] prefix + [4] block and [1,2] prefix + [3,4] block
        produce DIFFERENT hashes because the separator position differs.
        This is the correct behavior: blocks at different positions
        in the sequence should have distinct hashes.
        """
        h1 = compute_block_hash([1, 2, 3], [4])
        h2 = compute_block_hash([1, 2], [3, 4])
        # These are now different because the separator position differs
        assert h1 != h2

    def test_hash_via_manager_method(self):
        """KVCacheManager.compute_block_hash delegates correctly."""
        config = make_config()
        mgr = KVCacheManager(config)
        h1 = compute_block_hash([1, 2], [3, 4])
        h2 = mgr.compute_block_hash([1, 2], [3, 4])
        assert h1 == h2


# ---------------------------------------------------------------------------
# P1.3 — find_cached_prefix
# ---------------------------------------------------------------------------

class TestFindCachedPrefix:
    """Tests for prefix cache lookup."""

    def test_prefix_miss(self):
        """Returns 0 when nothing is cached."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        assert mgr.find_cached_prefix(tokens) == 0

    def test_prefix_full(self):
        """Returns full length when all blocks are cached."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # Allocate all blocks first so they are in the hash table
        mgr.allocate_blocks(tokens)

        # Now find_cached_prefix should find them all
        assert mgr.find_cached_prefix(tokens) == 8

    def test_prefix_partial(self):
        """Returns partial match when only first blocks are cached."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        # Cache first 4 tokens only
        short_tokens = [1, 2, 3, 4]
        mgr.allocate_blocks(short_tokens)

        # Query with longer sequence that shares the prefix
        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        assert mgr.find_cached_prefix(long_tokens) == 4

    def test_prefix_not_block_aligned(self):
        """Tokens beyond the last full block are ignored."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        # 6 tokens = 1 full block (4) + 2 leftover
        tokens = [1, 2, 3, 4, 5, 6]
        mgr.allocate_blocks(tokens)

        # Only the first block (4 tokens) can be found
        assert mgr.find_cached_prefix(tokens) == 4

    def test_prefix_empty_tokens(self):
        """Empty token list returns 0."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        assert mgr.find_cached_prefix([]) == 0

    def test_prefix_shorter_than_block(self):
        """Token list shorter than block_size returns 0."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        assert mgr.find_cached_prefix([1, 2]) == 0


# ---------------------------------------------------------------------------
# P1.4 — allocate_blocks
# ---------------------------------------------------------------------------

class TestAllocateBlocks:
    """Tests for block allocation with cache hit/miss handling."""

    def test_alloc_fresh(self):
        """Allocating tokens with empty cache creates new blocks."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]

        block_ids = mgr.allocate_blocks(tokens)
        assert len(block_ids) == 2  # 8 tokens / 4 block_size = 2 blocks
        assert mgr.num_free_blocks == 6  # 8 - 2 = 6

    def test_alloc_hit(self):
        """Second allocation of same tokens reuses cached blocks."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]

        block_ids_1 = mgr.allocate_blocks(tokens)
        block_ids_2 = mgr.allocate_blocks(tokens)

        # Should reuse the same blocks
        assert block_ids_1 == block_ids_2
        # No additional blocks consumed from free pool
        assert mgr.num_free_blocks == 6

    def test_alloc_refcount(self):
        """ref_count increments on cache hit."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [10, 20, 30, 40]

        block_ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[block_ids[0]]
        assert block.ref_count == 1

        # Allocate again (cache hit)
        mgr.allocate_blocks(tokens)
        assert block.ref_count == 2

        # Third time
        mgr.allocate_blocks(tokens)
        assert block.ref_count == 3

    def test_alloc_with_existing_blocks(self):
        """num_existing_blocks skips already-allocated blocks."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # Allocate first block
        first_ids = mgr.allocate_blocks(tokens[:4])
        assert len(first_ids) == 1

        # Allocate remaining blocks, skipping the first
        rest_ids = mgr.allocate_blocks(tokens, num_existing_blocks=1)
        assert len(rest_ids) == 2  # Blocks for tokens 5-8 and 9-12

        # Total: 3 blocks used
        assert mgr.num_free_blocks == 5

    def test_alloc_partial_block_ignored(self):
        """Tokens that don't fill a complete block are not allocated."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4, 5, 6]  # 1 full block + 2 leftover

        block_ids = mgr.allocate_blocks(tokens)
        assert len(block_ids) == 1  # Only the first full block

    def test_alloc_exhaustion_triggers_eviction(self):
        """When pool is exhausted, eviction is attempted."""
        # Pool of 2 blocks, block_size=4
        config = make_config(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        # Fill both blocks
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        ids_a = mgr.allocate_blocks(tokens_a)
        assert len(ids_a) == 2
        assert mgr.num_free_blocks == 0

        # Free them (ref_count -> 0, still in hash table)
        mgr.free_blocks(ids_a)

        # Now allocate different tokens — should evict and reuse
        tokens_b = [100, 200, 300, 400]
        ids_b = mgr.allocate_blocks(tokens_b)
        assert len(ids_b) == 1

        # Verify the evicted blocks are truly reusable
        assert mgr.num_free_blocks >= 0
        # The newly allocated block should be in the hash table
        block = mgr.pool.blocks[ids_b[0]]
        assert block.ref_count == 1
        assert block.token_ids == [100, 200, 300, 400]

    def test_alloc_exhaustion_no_evictable_raises(self):
        """When pool is exhausted and all blocks are in use, raises error."""
        config = make_config(block_size=4, num_blocks=1)
        mgr = KVCacheManager(config)

        # Allocate the only block (ref_count=1, not evictable)
        mgr.allocate_blocks([1, 2, 3, 4])

        with pytest.raises(BlockPoolExhaustedError):
            mgr.allocate_blocks([5, 6, 7, 8])

    def test_alloc_shared_prefix(self):
        """Two sequences sharing a prefix reuse the prefix blocks."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        # Sequence A: [1,2,3,4, 5,6,7,8]
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        ids_a = mgr.allocate_blocks(tokens_a)

        # Sequence B shares prefix: [1,2,3,4, 10,20,30,40]
        tokens_b = [1, 2, 3, 4, 10, 20, 30, 40]
        ids_b = mgr.allocate_blocks(tokens_b)

        # First block should be shared (same block_id)
        assert ids_a[0] == ids_b[0]
        # Second block should be different
        assert ids_a[1] != ids_b[1]

        # Shared block ref_count should be 2
        shared_block = mgr.pool.blocks[ids_a[0]]
        assert shared_block.ref_count == 2

        # 3 unique blocks used total
        assert mgr.num_used_blocks == 3


# ---------------------------------------------------------------------------
# P1.5 — free_blocks
# ---------------------------------------------------------------------------

class TestFreeBlocks:
    """Tests for block reference counting on free."""

    def test_free_refcount(self):
        """free_blocks decrements ref_count correctly."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4]

        # Allocate twice to get ref_count=2
        ids = mgr.allocate_blocks(tokens)
        mgr.allocate_blocks(tokens)

        block = mgr.pool.blocks[ids[0]]
        assert block.ref_count == 2

        mgr.free_blocks(ids)
        assert block.ref_count == 1

        mgr.free_blocks(ids)
        assert block.ref_count == 0

    def test_free_stays_in_hash(self):
        """Block with ref_count=0 remains in hash_table for future reuse."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4]

        ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(ids)

        block = mgr.pool.blocks[ids[0]]
        assert block.ref_count == 0

        # Block should still be in hash_table
        assert block.block_hash in mgr.hash_table

        # And should NOT be in the free pool
        assert ids[0] not in mgr.pool.free_queue

    def test_free_already_zero_is_noop(self):
        """Freeing a block with ref_count=0 is a no-op (logged warning)."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4]

        ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(ids)  # ref_count -> 0

        block = mgr.pool.blocks[ids[0]]
        assert block.ref_count == 0

        # Should not go negative
        mgr.free_blocks(ids)
        assert block.ref_count == 0

    def test_free_enables_reuse(self):
        """After freeing, the same prefix can be found by find_cached_prefix."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(ids)

        # Blocks are still in hash table, so prefix is still findable
        assert mgr.find_cached_prefix(tokens) == 8


# ---------------------------------------------------------------------------
# P1.6 — evict_lru
# ---------------------------------------------------------------------------

class TestEvictLRU:
    """Tests for LRU eviction of unreferenced blocks."""

    def test_evict_order(self):
        """Eviction selects oldest (smallest last_accessed) blocks first."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        # Allocate 3 separate (non-overlapping-prefix) blocks
        # Use distinct prefix structures so each block is unique
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)
        ids_c = mgr.allocate_blocks(tokens_c)

        # Free all
        mgr.free_blocks(ids_a)
        mgr.free_blocks(ids_b)
        mgr.free_blocks(ids_c)

        # Manually set last_accessed to control ordering
        mgr.pool.blocks[ids_a[0]].last_accessed = 100.0  # oldest
        mgr.pool.blocks[ids_b[0]].last_accessed = 200.0  # middle
        mgr.pool.blocks[ids_c[0]].last_accessed = 300.0  # newest

        # Evict 2 — should pick A and B (oldest first)
        evicted = mgr.evict_lru(num_blocks=2)
        assert len(evicted) == 2
        assert evicted[0] == ids_a[0]  # oldest
        assert evicted[1] == ids_b[0]  # next oldest

        # Verify evicted blocks are no longer in cache
        assert mgr.find_cached_prefix(tokens_a) == 0, "Evicted prefix A should not be found"
        assert mgr.find_cached_prefix(tokens_b) == 0, "Evicted prefix B should not be found"
        # Block C should still be cached (not evicted)
        assert mgr.find_cached_prefix(tokens_c) == len(tokens_c), "Non-evicted prefix C should still be cached"

    def test_evict_skips_in_use(self):
        """Eviction skips blocks with ref_count > 0."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)

        # Free only block B
        mgr.free_blocks(ids_b)

        # Block A is still in use (ref_count=1)
        # Block B is free (ref_count=0)
        evicted = mgr.evict_lru(num_blocks=2)

        # Should only evict B, skip A
        assert len(evicted) == 1
        assert evicted[0] == ids_b[0]

    def test_evict_returns_to_free_pool(self):
        """Evicted blocks are returned to the free pool."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        assert mgr.num_free_blocks == 3

        mgr.free_blocks(ids)
        assert mgr.num_free_blocks == 3  # Still 3 — block not in free pool yet

        evicted = mgr.evict_lru(num_blocks=1)
        assert len(evicted) == 1
        assert mgr.num_free_blocks == 4  # Now it's back in the free pool

    def test_evict_removes_from_hash_table(self):
        """Evicted blocks are removed from the hash table."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[ids[0]]
        block_hash = block.block_hash

        assert block_hash in mgr.hash_table
        mgr.free_blocks(ids)
        mgr.evict_lru(num_blocks=1)

        assert block_hash not in mgr.hash_table

    def test_evict_none_available(self):
        """Returns empty list when no blocks are evictable."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # Allocate but don't free — all blocks have ref_count=1
        mgr.allocate_blocks([1, 2, 3, 4])

        evicted = mgr.evict_lru(num_blocks=1)
        assert evicted == []

    def test_evict_empty_cache(self):
        """Returns empty list when hash table is empty."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        evicted = mgr.evict_lru(num_blocks=1)
        assert evicted == []

    def test_evict_then_allocate(self):
        """After eviction, new allocations can use the freed blocks."""
        config = make_config(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        # Fill the pool
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)
        assert mgr.num_free_blocks == 0

        # Free and evict block A
        mgr.free_blocks(ids_a)
        mgr.evict_lru(num_blocks=1)
        assert mgr.num_free_blocks == 1

        # Now we can allocate a new block
        tokens_c = [9, 10, 11, 12]
        ids_c = mgr.allocate_blocks(tokens_c)
        assert len(ids_c) == 1
        assert mgr.num_free_blocks == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Basic thread safety tests for concurrent operations."""

    def test_concurrent_allocate_and_free(self):
        """Multiple threads can allocate and free without corruption."""
        config = make_config(block_size=4, num_blocks=64)
        mgr = KVCacheManager(config)
        errors: list[Exception] = []

        def worker(thread_id: int):
            try:
                for _ in range(10):
                    tokens = list(range(thread_id * 100, thread_id * 100 + 8))
                    ids = mgr.allocate_blocks(tokens)
                    time.sleep(0.001)
                    mgr.free_blocks(ids)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_eviction(self):
        """Concurrent eviction doesn't cause double-free or corruption."""
        config = make_config(block_size=4, num_blocks=32)
        mgr = KVCacheManager(config)
        errors: list[Exception] = []

        # Allocate and free many blocks
        for i in range(8):
            tokens = list(range(i * 4, i * 4 + 4))
            ids = mgr.allocate_blocks(tokens)
            mgr.free_blocks(ids)

        def evictor():
            try:
                for _ in range(4):
                    mgr.evict_lru(num_blocks=1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=evictor) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestKVCacheManagerIntegration:
    """End-to-end scenarios combining multiple operations."""

    def test_full_lifecycle(self):
        """Allocate -> find prefix -> free -> evict -> reallocate."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # 1. Allocate tokens
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = mgr.allocate_blocks(tokens)
        assert len(ids) == 2

        # 2. Find cached prefix
        assert mgr.find_cached_prefix(tokens) == 8

        # 3. Free blocks
        mgr.free_blocks(ids)

        # 4. Blocks still in cache (findable)
        assert mgr.find_cached_prefix(tokens) == 8

        # 5. Evict all
        evicted = mgr.evict_lru(num_blocks=2)
        assert len(evicted) == 2

        # 6. Prefix no longer cached
        assert mgr.find_cached_prefix(tokens) == 0

        # 7. Can reallocate
        new_ids = mgr.allocate_blocks(tokens)
        assert len(new_ids) == 2

    def test_properties(self):
        """num_free_blocks, num_cached_blocks, num_used_blocks are consistent."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        assert mgr.num_free_blocks == 8
        assert mgr.num_cached_blocks == 0
        assert mgr.num_used_blocks == 0

        ids = mgr.allocate_blocks([1, 2, 3, 4, 5, 6, 7, 8])
        assert mgr.num_free_blocks == 6
        assert mgr.num_cached_blocks == 2
        assert mgr.num_used_blocks == 2

        mgr.free_blocks(ids)
        # Blocks still in hash table, not returned to free pool
        assert mgr.num_free_blocks == 6
        assert mgr.num_cached_blocks == 2
        assert mgr.num_used_blocks == 2

        mgr.evict_lru(num_blocks=1)
        assert mgr.num_free_blocks == 7
        assert mgr.num_cached_blocks == 1
        assert mgr.num_used_blocks == 1


# ---------------------------------------------------------------------------
# P1.7-P1.10 — MLX KV Cache Adapter (extract_block / inject_blocks)
# ---------------------------------------------------------------------------


class TestExtractBlock:
    """Tests for extract_block slicing K/V arrays."""

    def test_extract_shapes(self):
        """Extracted block has correct shape (B, H, block_size, D)."""
        B, H, S, D = 1, 4, 32, 16
        keys = mx.random.normal((B, H, S, D))
        values = mx.random.normal((B, H, S, D))

        block = extract_block(keys, values, start_pos=0, block_size=8)
        assert block["keys"].shape == (B, H, 8, D)
        assert block["values"].shape == (B, H, 8, D)

    def test_extract_values(self):
        """Extracted block contains the correct values from the source."""
        B, H, S, D = 1, 2, 16, 4
        keys = mx.arange(B * H * S * D).reshape(B, H, S, D).astype(mx.float32)
        values = mx.arange(B * H * S * D).reshape(B, H, S, D).astype(mx.float32) + 1000

        block = extract_block(keys, values, start_pos=4, block_size=4)
        expected_k = keys[:, :, 4:8, :]
        expected_v = values[:, :, 4:8, :]

        assert mx.allclose(block["keys"], expected_k).item()
        assert mx.allclose(block["values"], expected_v).item()

    def test_extract_different_positions(self):
        """Extract blocks from different positions yields different data."""
        B, H, S, D = 1, 2, 16, 8
        keys = mx.random.normal((B, H, S, D))
        values = mx.random.normal((B, H, S, D))

        block0 = extract_block(keys, values, start_pos=0, block_size=4)
        block1 = extract_block(keys, values, start_pos=4, block_size=4)

        # Different positions should have different data (overwhelmingly likely)
        assert not mx.allclose(block0["keys"], block1["keys"]).item()


class TestInjectBlocks:
    """Tests for inject_blocks concatenation."""

    def test_inject_roundtrip(self):
        """Extract then inject recovers the original K/V tensors."""
        B, H, S, D = 1, 4, 16, 8
        block_size = 4
        keys = mx.random.normal((B, H, S, D))
        values = mx.random.normal((B, H, S, D))

        # Extract 4 blocks of size 4 from seq_len=16
        blocks = []
        for i in range(S // block_size):
            b = extract_block(keys, values, start_pos=i * block_size, block_size=block_size)
            blocks.append(b)

        reconstructed = inject_blocks(blocks)

        assert reconstructed["keys"].shape == keys.shape
        assert reconstructed["values"].shape == values.shape
        assert mx.allclose(reconstructed["keys"], keys).item()
        assert mx.allclose(reconstructed["values"], values).item()

    def test_inject_single_block(self):
        """inject_blocks works with a single block."""
        B, H, D = 1, 2, 8
        block = {
            "keys": mx.random.normal((B, H, 4, D)),
            "values": mx.random.normal((B, H, 4, D)),
        }
        result = inject_blocks([block])
        assert mx.allclose(result["keys"], block["keys"]).item()
        assert mx.allclose(result["values"], block["values"]).item()

    def test_inject_empty_raises(self):
        """inject_blocks raises ValueError on empty list."""
        with pytest.raises(ValueError):
            inject_blocks([])

    def test_roundtrip_generation(self):
        """Full roundtrip: create K/V -> extract blocks -> inject -> verify.

        Simulates what would happen during cache save/restore: a sequence
        is split into blocks, each block is stored separately, then all
        blocks are reassembled.
        """
        B, H, D = 1, 8, 64
        block_size = 16
        num_blocks = 4
        total_seq = block_size * num_blocks

        keys = mx.random.normal((B, H, total_seq, D))
        values = mx.random.normal((B, H, total_seq, D))

        # Extract each block
        extracted = []
        for i in range(num_blocks):
            b = extract_block(
                keys, values, start_pos=i * block_size, block_size=block_size
            )
            extracted.append(b)

        # Inject all blocks back
        reconstructed = inject_blocks(extracted)

        assert reconstructed["keys"].shape == (B, H, total_seq, D)
        assert reconstructed["values"].shape == (B, H, total_seq, D)
        assert mx.allclose(reconstructed["keys"], keys).item()
        assert mx.allclose(reconstructed["values"], values).item()


# ---------------------------------------------------------------------------
# P1.16-P1.18 — TieredKVCache (RAM -> SSD -> miss)
# ---------------------------------------------------------------------------


class TestTieredLookup:
    """Tests for TieredKVCache.lookup()."""

    def _make_kv_data(self) -> dict[str, mx.array]:
        return {
            "keys": mx.random.normal((1, 2, 4, 8)),
            "values": mx.random.normal((1, 2, 4, 8)),
        }

    def test_lookup_ram(self):
        """lookup finds block in RAM hash table."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tiered = TieredKVCache(ram=mgr, ssd=None)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[ids[0]]
        kv_data = self._make_kv_data()
        block.kv_data = kv_data

        result = tiered.lookup(block.block_hash)
        assert result is not None
        assert mx.allclose(result["keys"], kv_data["keys"]).item()

    def test_lookup_ssd(self, tmp_path):
        """lookup falls through to SSD when not in RAM."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        kv_data = self._make_kv_data()
        block_hash = "hash_12345"
        ssd.save_block(block_hash, kv_data)

        result = tiered.lookup(block_hash)
        assert result is not None
        assert mx.allclose(result["keys"], kv_data["keys"]).item()

    def test_lookup_miss(self):
        """lookup returns None when block is in neither RAM nor SSD."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tiered = TieredKVCache(ram=mgr, ssd=None)

        result = tiered.lookup("nonexistent_hash")
        assert result is None

    def test_lookup_miss_with_ssd(self, tmp_path):
        """lookup returns None when SSD exists but block is not there."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        result = tiered.lookup("nonexistent_hash")
        assert result is None


class TestEvictToSSD:
    """Tests for TieredKVCache.evict_to_ssd()."""

    def _make_kv_data(self) -> dict[str, mx.array]:
        return {
            "keys": mx.random.normal((1, 2, 4, 8)),
            "values": mx.random.normal((1, 2, 4, 8)),
        }

    def test_evict_saves_to_ssd(self, tmp_path):
        """Evicting a block with kv_data saves it to SSD."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[ids[0]]
        kv_data = self._make_kv_data()
        block.kv_data = kv_data
        block_hash = block.block_hash

        # Free the block so it's eligible for eviction
        mgr.free_blocks(ids)

        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1

        # Block should now be in SSD
        assert block_hash in ssd.index

        # Load from SSD and verify
        loaded = ssd.load_block(block_hash)
        assert loaded is not None
        assert mx.allclose(loaded["keys"], kv_data["keys"]).item()

    def test_evict_without_ssd(self):
        """Eviction works even without SSD (just frees RAM)."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tiered = TieredKVCache(ram=mgr, ssd=None)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(ids)

        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1
        assert mgr.num_free_blocks == 8


class TestTieredFullFlow:
    """Integration test: allocate -> evict to SSD -> lookup from SSD."""

    def test_tiered_full_flow(self, tmp_path):
        """Full tiered flow: allocate -> attach kv_data -> evict to SSD -> lookup."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        # 1. Allocate blocks
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        ids = mgr.allocate_blocks(tokens)
        assert len(ids) == 2

        # 2. Attach KV data to blocks
        kv_data_0 = {
            "keys": mx.random.normal((1, 2, 4, 8)),
            "values": mx.random.normal((1, 2, 4, 8)),
        }
        kv_data_1 = {
            "keys": mx.random.normal((1, 2, 4, 8)),
            "values": mx.random.normal((1, 2, 4, 8)),
        }
        block_0 = mgr.pool.blocks[ids[0]]
        block_1 = mgr.pool.blocks[ids[1]]
        block_0.kv_data = kv_data_0
        block_1.kv_data = kv_data_1
        hash_0 = block_0.block_hash
        hash_1 = block_1.block_hash

        # 3. Free blocks
        mgr.free_blocks(ids)

        # 4. Verify RAM lookup works
        result = tiered.lookup(hash_0)
        assert result is not None

        # 5. Evict to SSD
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2

        # 6. RAM should no longer have them
        assert hash_0 not in mgr.hash_table
        assert hash_1 not in mgr.hash_table

        # 7. SSD lookup should find them
        result_0 = tiered.lookup(hash_0)
        assert result_0 is not None
        assert mx.allclose(result_0["keys"], kv_data_0["keys"]).item()

        result_1 = tiered.lookup(hash_1)
        assert result_1 is not None
        assert mx.allclose(result_1["keys"], kv_data_1["keys"]).item()

        # 8. Verify blocks were returned to free pool
        assert mgr.num_free_blocks == 4


# ---------------------------------------------------------------------------
# Eviction Exclusion tests (exclude_ids parameter)
# ---------------------------------------------------------------------------


class TestEvictionExclusion:
    """Tests for exclude_ids parameter in eviction."""

    def test_eviction_excludes_protected(self):
        """Blocks in exclude_ids survive eviction even with ref_count == 0."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # Allocate 3 blocks
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)
        ids_c = mgr.allocate_blocks(tokens_c)

        # Free all so they're evictable
        mgr.free_blocks(ids_a)
        mgr.free_blocks(ids_b)
        mgr.free_blocks(ids_c)

        # Make A oldest, B middle, C newest
        mgr.pool.blocks[ids_a[0]].last_accessed = 100.0
        mgr.pool.blocks[ids_b[0]].last_accessed = 200.0
        mgr.pool.blocks[ids_c[0]].last_accessed = 300.0

        # Protect block A from eviction
        protected = {ids_a[0]}

        # Evict 2 -- should skip A (protected), evict B and C instead
        evicted = mgr.evict_lru(num_blocks=2, exclude_ids=protected)
        assert len(evicted) == 2
        assert ids_a[0] not in evicted
        assert ids_b[0] in evicted
        assert ids_c[0] in evicted

        # Block A should still be in hash table
        block_a = mgr.pool.blocks[ids_a[0]]
        assert block_a.block_hash in mgr.hash_table

    def test_eviction_excludes_with_tiered(self, tmp_path):
        """TieredKVCache.evict_to_ssd respects exclude_ids."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)

        mgr.free_blocks(ids_a)
        mgr.free_blocks(ids_b)

        mgr.pool.blocks[ids_a[0]].last_accessed = 100.0
        mgr.pool.blocks[ids_b[0]].last_accessed = 200.0

        # Protect block A
        protected = {ids_a[0]}

        # Evict 2 -- A is protected, only B should be evicted
        evicted = tiered.evict_to_ssd(num_blocks=2, exclude_ids=protected)
        assert len(evicted) == 1
        assert ids_b[0] in evicted
        assert ids_a[0] not in evicted


# ---------------------------------------------------------------------------
# cache_block() — atomic block caching with eviction fallback
# ---------------------------------------------------------------------------


class TestCacheBlock:
    """Tests for KVCacheManager.cache_block()."""

    def test_cache_block_new(self, tmp_path):
        """cache_block() allocates and populates a new block."""
        config = ServerConfig(block_size=4, num_blocks=8, ssd_cache_dir=tmp_path / "ssd")
        mgr = KVCacheManager(config)

        block_hash = mgr.compute_block_hash([], [1, 2, 3, 4])
        block_id = mgr.cache_block(block_hash, [1, 2, 3, 4], [{"keys": "k", "values": "v"}])

        assert block_id is not None
        block = mgr.pool.blocks[block_id]
        assert block.block_hash == block_hash
        assert block.token_ids == [1, 2, 3, 4]
        assert block.ref_count == 1
        assert block.kv_data == [{"keys": "k", "values": "v"}]
        assert block_hash in mgr.hash_table

    def test_cache_block_duplicate_noop(self, tmp_path):
        """cache_block() returns None for already-cached hashes (no-op)."""
        config = ServerConfig(block_size=4, num_blocks=8, ssd_cache_dir=tmp_path / "ssd")
        mgr = KVCacheManager(config)

        block_hash = mgr.compute_block_hash([], [1, 2, 3, 4])
        bid1 = mgr.cache_block(block_hash, [1, 2, 3, 4], [{"keys": "k1"}])
        bid2 = mgr.cache_block(block_hash, [1, 2, 3, 4], [{"keys": "k2"}])

        assert bid1 is not None  # First call allocates
        assert bid2 is None  # Duplicate returns None — caller should not free
        # Original data is preserved, not overwritten
        assert mgr.pool.blocks[bid1].kv_data == [{"keys": "k1"}]

    def test_cache_block_eviction_fallback(self, tmp_path):
        """cache_block() evicts LRU blocks when pool is exhausted."""
        config = ServerConfig(block_size=4, num_blocks=2, ssd_cache_dir=tmp_path / "ssd")
        mgr = KVCacheManager(config)

        # Fill all blocks
        h1 = mgr.compute_block_hash([], [1, 2, 3, 4])
        h2 = mgr.compute_block_hash([1, 2, 3, 4], [5, 6, 7, 8])
        bid1 = mgr.cache_block(h1, [1, 2, 3, 4], [{"keys": "k1"}])
        bid2 = mgr.cache_block(h2, [5, 6, 7, 8], [{"keys": "k2"}])

        # Release protection refs so blocks become evictable
        mgr.free_blocks([bid1])
        mgr.free_blocks([bid2])

        assert mgr.pool.num_free == 0

        # Cache a third block — should evict oldest
        h3 = mgr.compute_block_hash([1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12])
        bid3 = mgr.cache_block(h3, [9, 10, 11, 12], [{"keys": "k3"}])

        assert bid3 is not None
        assert h3 in mgr.hash_table

    def test_cache_block_returns_none_when_full(self, tmp_path):
        """cache_block() returns None when pool is exhausted and nothing evictable."""
        config = ServerConfig(block_size=4, num_blocks=1, ssd_cache_dir=tmp_path / "ssd")
        mgr = KVCacheManager(config)

        # Allocate and pin the only block (ref_count > 0 prevents eviction)
        block_ids = mgr.allocate_blocks([1, 2, 3, 4])
        assert len(block_ids) == 1

        # Try to cache another block — should fail since the only block has ref_count=1
        h2 = mgr.compute_block_hash([1, 2, 3, 4], [5, 6, 7, 8])
        bid = mgr.cache_block(h2, [5, 6, 7, 8], [{"keys": "k2"}])

        assert bid is None


# ---------------------------------------------------------------------------
# H2 fix — cache_block ref_count protection
# ---------------------------------------------------------------------------


class TestCacheBlockRefCount:
    """Tests for cache_block() ref_count=1 protection against immediate eviction."""

    def test_cache_block_sets_ref_count_one(self):
        """Freshly cached blocks should have ref_count=1 to prevent immediate eviction."""
        config = make_config(num_blocks=8, block_size=4)
        mgr = KVCacheManager(config)
        block_id = mgr.cache_block(
            block_hash="test_hash",
            token_ids=[1, 2, 3, 4],
            kv_data=[{"keys": [1.0], "values": [2.0]}],
        )
        assert block_id is not None
        block = mgr.pool.blocks[block_id]
        assert block.ref_count == 1, "Freshly cached block should have ref_count=1"

    def test_cache_block_not_immediately_evictable(self):
        """Freshly cached blocks should NOT be evictable."""
        config = make_config(num_blocks=4, block_size=4)
        mgr = KVCacheManager(config)
        # Fill all blocks with cached data
        for i in range(4):
            bid = mgr.cache_block(
                block_hash=f"hash_{i}",
                token_ids=[i * 4 + j for j in range(4)],
                kv_data=[{"keys": [float(i)], "values": [float(i)]}],
            )
            assert bid is not None
        # All blocks have ref_count=1, so eviction should fail (no evictable blocks)
        evicted = mgr.evict_lru(1)
        assert len(evicted) == 0, "Should not evict blocks with ref_count=1"

    def test_cache_block_evictable_after_free(self):
        """After free_blocks(), cached block should become evictable."""
        config = make_config(num_blocks=4, block_size=4)
        mgr = KVCacheManager(config)
        block_id = mgr.cache_block(
            block_hash="test_hash",
            token_ids=[1, 2, 3, 4],
            kv_data=[{"keys": [1.0], "values": [2.0]}],
        )
        assert block_id is not None
        # Free the block — ref_count drops to 0, pushed to eviction heap
        mgr.free_blocks([block_id])
        block = mgr.pool.blocks[block_id]
        assert block.ref_count == 0
        # Now it should be evictable
        evicted = mgr.evict_lru(1)
        assert len(evicted) == 1
        assert evicted[0] == block_id


# ---------------------------------------------------------------------------
# H3 fix — Eviction Heap Rebuild (stale heap entry accumulation)
# ---------------------------------------------------------------------------


class TestEvictionHeapRebuild:
    """Tests for the heap rebuild mechanism that prevents stale entry accumulation."""

    def test_heap_rebuild_removes_stale_entries(self):
        """Heap rebuild should remove stale entries and keep valid ones."""
        config = make_config(num_blocks=8, block_size=4)
        mgr = KVCacheManager(config)

        # Allocate and free blocks to create heap entries
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]
        tokens_d = [13, 14, 15, 16]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)
        ids_c = mgr.allocate_blocks(tokens_c)
        ids_d = mgr.allocate_blocks(tokens_d)

        block_ids = ids_a + ids_b + ids_c + ids_d
        mgr.free_blocks(block_ids)  # All ref_count=0, pushed to heap

        # Update last_accessed on some blocks to create stale entries
        import heapq

        for bid in block_ids[:2]:
            mgr.pool.blocks[bid].last_accessed = time.time() + 100
            # Push new entry — old one becomes stale
            heapq.heappush(mgr._eviction_heap, (mgr.pool.blocks[bid].last_accessed, bid))

        old_heap_size = len(mgr._eviction_heap)
        assert old_heap_size >= 6  # 4 original + 2 updated

        # Force rebuild
        mgr._rebuild_eviction_heap()

        # After rebuild, only 4 entries (one per block with ref_count=0)
        assert len(mgr._eviction_heap) == 4

    def test_heap_auto_rebuild_on_threshold(self):
        """Heap should auto-rebuild when stale ratio exceeds threshold."""
        config = make_config(num_blocks=16, block_size=4)
        mgr = KVCacheManager(config)

        # Allocate and free 4 blocks
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        block_ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(block_ids)

        # Artificially inflate _heap_pushes to trigger rebuild
        mgr._heap_pushes = 1000  # Way above threshold

        # Eviction should trigger rebuild and still work correctly
        evicted = mgr.evict_lru(1)
        assert len(evicted) == 1
        # After rebuild, _heap_pushes should have been reset (then incremented
        # only by the re-push of excluded blocks, if any)
        assert mgr._heap_pushes <= len(mgr._eviction_heap)

    def test_heap_pushes_incremented_on_free(self):
        """_heap_pushes counter increments when blocks are freed."""
        config = make_config(num_blocks=8, block_size=4)
        mgr = KVCacheManager(config)

        assert mgr._heap_pushes == 0

        ids = mgr.allocate_blocks([1, 2, 3, 4])
        assert mgr._heap_pushes == 0  # No pushes yet (allocation doesn't push)

        mgr.free_blocks(ids)
        assert mgr._heap_pushes == 1  # One block freed -> one push

        # Allocate and free more
        ids2 = mgr.allocate_blocks([5, 6, 7, 8, 9, 10, 11, 12])
        mgr.free_blocks(ids2)
        assert mgr._heap_pushes == 3  # 1 + 2 more blocks freed

    def test_rebuild_resets_pushes_counter(self):
        """_rebuild_eviction_heap resets _heap_pushes to 0."""
        config = make_config(num_blocks=8, block_size=4)
        mgr = KVCacheManager(config)

        ids = mgr.allocate_blocks([1, 2, 3, 4])
        mgr.free_blocks(ids)
        assert mgr._heap_pushes == 1

        mgr._rebuild_eviction_heap()
        assert mgr._heap_pushes == 0

    def test_rebuild_preserves_eviction_correctness(self):
        """After rebuild, eviction still picks the oldest block."""
        config = make_config(num_blocks=8, block_size=4)
        mgr = KVCacheManager(config)

        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)
        ids_c = mgr.allocate_blocks(tokens_c)

        mgr.free_blocks(ids_a)
        mgr.free_blocks(ids_b)
        mgr.free_blocks(ids_c)

        # Set explicit timestamps
        mgr.pool.blocks[ids_a[0]].last_accessed = 100.0
        mgr.pool.blocks[ids_b[0]].last_accessed = 200.0
        mgr.pool.blocks[ids_c[0]].last_accessed = 300.0

        # Force rebuild to clean up timestamps in heap
        mgr._rebuild_eviction_heap()

        # Evict 1 — should still pick the oldest (A at 100.0)
        evicted = mgr.evict_lru(1)
        assert len(evicted) == 1
        assert evicted[0] == ids_a[0]


# ---------------------------------------------------------------------------
# F1 — Chain Hash (O(n) compute_block_hash)
# ---------------------------------------------------------------------------


class TestChainHash:
    """Tests for _compute_chain_hash and the legacy wrapper."""

    def test_chain_hash_determinism(self):
        """Same inputs produce the same chain hash."""
        h1 = _compute_chain_hash([1, 2, 3, 4], None)
        h2 = _compute_chain_hash([1, 2, 3, 4], None)
        assert h1 == h2

    def test_chain_hash_prev_hash_matters(self):
        """Different prev_hash produces different chain hash."""
        h1 = _compute_chain_hash([1, 2, 3, 4], None)
        h2 = _compute_chain_hash([1, 2, 3, 4], "abc123")
        assert h1 != h2

    def test_chain_hash_different_tokens(self):
        """Different tokens produce different hashes."""
        h1 = _compute_chain_hash([1, 2, 3, 4], None)
        h2 = _compute_chain_hash([5, 6, 7, 8], None)
        assert h1 != h2

    def test_chain_hash_empty_tokens(self):
        """Empty token list produces a valid hash."""
        h = _compute_chain_hash([], None)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_legacy_wrapper_consistency(self):
        """compute_block_hash produces consistent results across calls."""
        h1 = compute_block_hash([1, 2, 3, 4], [5, 6, 7, 8])
        h2 = compute_block_hash([1, 2, 3, 4], [5, 6, 7, 8])
        assert h1 == h2

    def test_legacy_wrapper_empty_block_tokens(self):
        """compute_block_hash handles empty block_tokens with sentinel."""
        h1 = compute_block_hash([1, 2, 3], [])
        h2 = compute_block_hash([4, 5, 6], [])
        # Both return the same sentinel hash (prefix-independent)
        assert h1 == h2

    def test_legacy_wrapper_boundary_distinction(self):
        """Different prefix/block boundaries produce different hashes."""
        h1 = compute_block_hash([1, 2, 3], [4])
        h2 = compute_block_hash([1, 2], [3, 4])
        assert h1 != h2

    def test_chain_equals_legacy_for_aligned_prefix(self):
        """Chain hash via wrapper matches manual chain for aligned blocks."""
        block_size = 4
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        prefix = tokens[:block_size]
        block = tokens[block_size:]

        legacy = compute_block_hash(prefix, block)
        prev = _compute_chain_hash(prefix, None)
        chain = _compute_chain_hash(block, prev)
        assert legacy == chain

    def test_internal_callers_match_wrapper(self):
        """find_cached_prefix and allocate_blocks use chain hash consistently."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]

        # Allocate via allocate_blocks (uses chain hash internally)
        ids = mgr.allocate_blocks(tokens)
        assert len(ids) == 2

        # find_cached_prefix should find them (also uses chain hash)
        cached = mgr.find_cached_prefix(tokens)
        assert cached == 8

        # Verify hashes match what compute_block_hash produces
        block0 = mgr.pool.blocks[ids[0]]
        expected_h0 = compute_block_hash([], tokens[:4])
        assert block0.block_hash == expected_h0

        block1 = mgr.pool.blocks[ids[1]]
        expected_h1 = compute_block_hash(tokens[:4], tokens[4:8])
        assert block1.block_hash == expected_h1


# ---------------------------------------------------------------------------
# compute_model_fingerprint — adapter_path inclusion
# ---------------------------------------------------------------------------


class TestComputeModelFingerprint:
    """Tests for compute_model_fingerprint with adapter_path support."""

    class _FakeModel:
        """Minimal model stub with a config attribute."""

        class config:
            num_hidden_layers = 24
            num_key_value_heads = 8
            hidden_size = 2048

    def test_different_adapters_produce_different_fingerprints(self):
        """Different adapter_path values must produce different fingerprints."""
        model = self._FakeModel()
        fp_a = compute_model_fingerprint(
            "model-x", model, 8, 64, adapter_path="/adapters/lora-A"
        )
        fp_b = compute_model_fingerprint(
            "model-x", model, 8, 64, adapter_path="/adapters/lora-B"
        )
        assert fp_a != fp_b

    def test_none_adapter_matches_no_adapter_arg(self):
        """adapter_path=None produces the same fingerprint as omitting it (backwards compat)."""
        model = self._FakeModel()
        fp_default = compute_model_fingerprint("model-x", model, 8, 64)
        fp_none = compute_model_fingerprint(
            "model-x", model, 8, 64, adapter_path=None
        )
        assert fp_default == fp_none

    def test_adapter_vs_no_adapter_differ(self):
        """Providing an adapter_path must differ from no adapter."""
        model = self._FakeModel()
        fp_no_adapter = compute_model_fingerprint("model-x", model, 8, 64)
        fp_with_adapter = compute_model_fingerprint(
            "model-x", model, 8, 64, adapter_path="/adapters/lora-A"
        )
        assert fp_no_adapter != fp_with_adapter

    def test_fingerprint_determinism(self):
        """Same inputs always produce the same fingerprint."""
        model = self._FakeModel()
        fp1 = compute_model_fingerprint(
            "model-x", model, 8, 64, adapter_path="/adapters/lora-A"
        )
        fp2 = compute_model_fingerprint(
            "model-x", model, 8, 64, adapter_path="/adapters/lora-A"
        )
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# SSD-aware prefix lookup and block promote
# ---------------------------------------------------------------------------


class TestSSDPromote:
    """Tests for SSD-aware find_cached_prefix and allocate_blocks promote."""

    def _make_kv_data(self) -> list[dict[str, mx.array]]:
        """Create multi-layer KV data suitable for SSD round-trip."""
        return [
            {
                "keys": mx.random.normal((1, 2, 4, 8)),
                "values": mx.random.normal((1, 2, 4, 8)),
            }
        ]

    def test_find_cached_prefix_finds_ssd_blocks(self, tmp_path):
        """Evict blocks to SSD -> find_cached_prefix still reports them as cached."""
        config = make_config(block_size=4, num_blocks=8)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # Allocate blocks and attach KV data
        ids = mgr.allocate_blocks(tokens)
        assert len(ids) == 2
        for bid in ids:
            mgr.pool.blocks[bid].kv_data = self._make_kv_data()

        # Verify RAM lookup works
        assert mgr.find_cached_prefix(tokens) == 8

        # Free and evict to SSD
        mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2

        # Blocks are gone from RAM hash_table
        assert mgr.num_cached_blocks == 0

        # But find_cached_prefix should still find them via SSD index
        assert mgr.find_cached_prefix(tokens) == 8

    def test_allocate_blocks_promotes_from_ssd(self, tmp_path):
        """Evict blocks to SSD -> allocate_blocks loads from SSD and populates RAM block."""
        config = make_config(block_size=4, num_blocks=8)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # Allocate blocks and attach KV data
        ids = mgr.allocate_blocks(tokens)
        kv_data_0 = self._make_kv_data()
        kv_data_1 = self._make_kv_data()
        mgr.pool.blocks[ids[0]].kv_data = kv_data_0
        mgr.pool.blocks[ids[1]].kv_data = kv_data_1

        # Save the block hashes before eviction
        hash_0 = mgr.pool.blocks[ids[0]].block_hash
        hash_1 = mgr.pool.blocks[ids[1]].block_hash

        # Free and evict to SSD
        mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2

        # Blocks are gone from RAM
        assert hash_0 not in mgr.hash_table
        assert hash_1 not in mgr.hash_table

        # Re-allocate the same tokens — should promote from SSD
        new_ids = mgr.allocate_blocks(tokens)
        assert len(new_ids) == 2

        # Blocks should now be back in RAM hash_table
        assert hash_0 in mgr.hash_table
        assert hash_1 in mgr.hash_table

        # Verify KV data was restored
        for bid in new_ids:
            block = mgr.pool.blocks[bid]
            assert block.kv_data is not None
            assert block.ref_count == 1
            assert block.token_ids is not None
            assert len(block.token_ids) == 4

    def test_ssd_promote_with_partial_hit(self, tmp_path):
        """Some blocks in RAM, some evicted to SSD -> correct hybrid prefix."""
        config = make_config(block_size=4, num_blocks=8)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # Allocate all 3 blocks and attach KV data
        ids = mgr.allocate_blocks(tokens)
        assert len(ids) == 3
        for bid in ids:
            mgr.pool.blocks[bid].kv_data = self._make_kv_data()

        hash_1 = mgr.pool.blocks[ids[1]].block_hash

        # Free only block 1 (middle) and evict it to SSD
        mgr.free_blocks([ids[1]])
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1
        assert hash_1 not in mgr.hash_table

        # Block 0 is in RAM, block 1 is on SSD, block 2 is in RAM
        # find_cached_prefix should find all 3 (block 0 from RAM, block 1 from SSD index,
        # block 2 from RAM)
        cached = mgr.find_cached_prefix(tokens)
        assert cached == 12  # All 12 tokens found

        # Free remaining blocks and re-allocate everything
        mgr.free_blocks([ids[0], ids[2]])

        # allocate_blocks should get block 0 from RAM (cache hit),
        # block 1 promoted from SSD, block 2 from RAM (cache hit)
        new_ids = mgr.allocate_blocks(tokens)
        assert len(new_ids) == 3

        # Block 1 should be back in RAM
        assert hash_1 in mgr.hash_table
        promoted_block = mgr.pool.blocks[mgr.hash_table[hash_1]]
        assert promoted_block.kv_data is not None

    def test_find_cached_prefix_without_ssd(self):
        """ssd=None -> behaves exactly as before (backwards compat)."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)  # No ssd parameter
        assert mgr.ssd is None

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        # Allocate then evict
        ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(ids)
        mgr.evict_lru(num_blocks=2)

        # Without SSD, evicted blocks are gone
        assert mgr.find_cached_prefix(tokens) == 0

        # Fresh allocation works
        new_ids = mgr.allocate_blocks(tokens)
        assert len(new_ids) == 2

    def test_ssd_promote_populates_kv_data(self, tmp_path):
        """Promoted blocks from SSD have correct KV data that matches original."""
        config = make_config(block_size=4, num_blocks=8)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens = [10, 20, 30, 40]

        # Allocate and attach KV data
        ids = mgr.allocate_blocks(tokens)
        original_kv = self._make_kv_data()
        mgr.pool.blocks[ids[0]].kv_data = original_kv

        # Free and evict
        mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1

        # Re-allocate — SSD promote
        new_ids = mgr.allocate_blocks(tokens)
        assert len(new_ids) == 1
        block = mgr.pool.blocks[new_ids[0]]

        # KV data should be a list of dicts with keys/values
        assert isinstance(block.kv_data, list)
        assert len(block.kv_data) == 1
        assert "keys" in block.kv_data[0]
        assert "values" in block.kv_data[0]

        # Verify the data matches (SSD round-trip preserves values)
        assert mx.allclose(
            block.kv_data[0]["keys"], original_kv[0]["keys"]
        ).item()
        assert mx.allclose(
            block.kv_data[0]["values"], original_kv[0]["values"]
        ).item()

    def test_ssd_promote_under_pool_exhaustion(self, tmp_path):
        """SSD promote works even when pool is initially exhausted (triggers eviction)."""
        config = make_config(block_size=4, num_blocks=2)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        # Fill pool with tokens_a, attach KV data
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        ids_a = mgr.allocate_blocks(tokens_a)
        assert len(ids_a) == 2
        for bid in ids_a:
            mgr.pool.blocks[bid].kv_data = self._make_kv_data()

        # Free and evict tokens_a to SSD
        mgr.free_blocks(ids_a)
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2
        assert mgr.num_free_blocks == 2

        # Fill pool with different tokens_b (uses both free blocks)
        tokens_b = [100, 200, 300, 400, 500, 600, 700, 800]
        ids_b = mgr.allocate_blocks(tokens_b)
        assert len(ids_b) == 2
        assert mgr.num_free_blocks == 0

        # Free tokens_b so they're evictable
        mgr.free_blocks(ids_b)

        # Now re-allocate tokens_a — pool is full but tokens_b blocks are
        # evictable, and tokens_a data should be promoted from SSD
        new_ids = mgr.allocate_blocks(tokens_a)
        assert len(new_ids) == 2

        # Verify promoted blocks have KV data
        for bid in new_ids:
            block = mgr.pool.blocks[bid]
            assert block.kv_data is not None
            assert block.ref_count == 1


# ---------------------------------------------------------------------------
# Stale SSD index promote fix (phantom hash entry prevention)
# ---------------------------------------------------------------------------


class TestStaleSSDPromote:
    """Tests for the fix to 'Phantom RAM hash entries after stale SSD index miss'.

    When ssd.has_block() returns True but ssd.load_block() returns None,
    the block should be treated as a COLLISION (hash NOT registered in
    hash_table) so that cache_block() can self-heal later.
    """

    def test_stale_ssd_no_phantom_hash_entry(self):
        """Stale SSD entry must NOT create a hash_table registration."""
        from unittest.mock import MagicMock

        config = make_config(block_size=4, num_blocks=8)
        mock_ssd = MagicMock()
        mgr = KVCacheManager(config, ssd=mock_ssd)

        tokens = [1, 2, 3, 4]

        # SSD index says it has the block, but load returns None (stale)
        mock_ssd.has_block.return_value = True
        mock_ssd.load_block.return_value = None

        block_ids = mgr.allocate_blocks(tokens)
        assert len(block_ids) == 1

        # The block should have been allocated (MISS/COLLISION path)
        block = mgr.pool.blocks[block_ids[0]]

        # CRITICAL: block_hash should be None (COLLISION treatment),
        # NOT registered in hash_table
        assert block.block_hash is None, (
            "Stale SSD promote should NOT register block_hash in hash_table"
        )
        block_hash = _compute_chain_hash([1, 2, 3, 4], None)
        assert block_hash not in mgr.hash_table, (
            "Phantom hash entry found — stale SSD promote should not register hash"
        )

        # kv_promote_fail stat should have been incremented
        stats = mgr.get_stats()
        assert stats["kv_promote_fail"] == 1

    def test_stale_ssd_self_heals_with_cache_block(self):
        """After stale SSD promote, cache_block() should succeed (self-heal)."""
        from unittest.mock import MagicMock

        config = make_config(block_size=4, num_blocks=8)
        mock_ssd = MagicMock()
        mgr = KVCacheManager(config, ssd=mock_ssd)

        tokens = [1, 2, 3, 4]
        block_hash = _compute_chain_hash(tokens, None)

        # Step 1: stale SSD promote
        mock_ssd.has_block.return_value = True
        mock_ssd.load_block.return_value = None

        stale_ids = mgr.allocate_blocks(tokens)
        assert len(stale_ids) == 1

        # Verify hash NOT in hash_table
        assert block_hash not in mgr.hash_table

        # Step 2: Free the stale block (collision blocks go directly to free pool)
        mgr.free_blocks(stale_ids)

        # Step 3: Self-heal via cache_block() — should succeed since
        # block_hash is not in hash_table
        kv_data = [{"keys": "real_k", "values": "real_v"}]
        bid = mgr.cache_block(block_hash, tokens, kv_data)

        assert bid is not None, "cache_block() should succeed after stale SSD promote"
        assert block_hash in mgr.hash_table
        healed_block = mgr.pool.blocks[bid]
        assert healed_block.block_hash == block_hash
        assert healed_block.kv_data == kv_data

    def test_stale_ssd_mixed_with_ram_hits(self):
        """9 RAM hits + 1 stale SSD promote: RAM hits preserved, stale treated as COLLISION."""
        from unittest.mock import MagicMock

        config = make_config(block_size=4, num_blocks=16)
        mock_ssd = MagicMock()
        mgr = KVCacheManager(config, ssd=mock_ssd)

        # Build 10 blocks = 40 tokens
        all_tokens = list(range(1, 41))

        # First allocation: all blocks go to RAM (no SSD involvement)
        mock_ssd.has_block.return_value = False
        first_ids = mgr.allocate_blocks(all_tokens)
        assert len(first_ids) == 10

        # Save all block hashes for reference
        block_hashes = [mgr.pool.blocks[bid].block_hash for bid in first_ids]

        # Verify all 10 are in hash_table
        for bh in block_hashes:
            assert bh in mgr.hash_table

        # Now simulate: evict block index 5 (the 6th block) from RAM
        # and make it appear as "stale SSD" on re-allocation
        target_idx = 5
        target_hash = block_hashes[target_idx]
        target_bid = first_ids[target_idx]

        # Free and evict the target block
        mgr.free_blocks([target_bid])
        mgr.evict_lru(num_blocks=1)
        assert target_hash not in mgr.hash_table

        # Configure SSD mock: only target_hash is "on SSD" (but stale)
        def mock_has_block(h):
            return h == target_hash

        mock_ssd.has_block.side_effect = mock_has_block
        mock_ssd.load_block.return_value = None  # stale

        # Re-allocate all tokens
        new_ids = mgr.allocate_blocks(all_tokens)
        assert len(new_ids) == 10

        # NOTE: allocated_block_ids ordering is:
        # - Pass 1 appends RAM hits in index order (0-4, 6-9 = 9 blocks)
        # - Pass 3 appends the stale SSD block (index 5 = 1 block)
        # So the stale block is the LAST element in new_ids.

        # Count RAM hits: 9 blocks should be reused from first allocation
        ram_hit_ids = set(first_ids) - {target_bid}
        ram_hits_in_new = [bid for bid in new_ids if bid in ram_hit_ids]
        assert len(ram_hits_in_new) == 9, (
            f"Expected 9 RAM hits, got {len(ram_hits_in_new)}"
        )

        # Find the stale block: the one NOT in the original first_ids
        # (it was freshly allocated from the free pool)
        stale_candidates = [
            bid for bid in new_ids
            if mgr.pool.blocks[bid].block_hash is None
        ]
        assert len(stale_candidates) == 1, (
            f"Expected exactly 1 COLLISION block, got {len(stale_candidates)}"
        )
        stale_bid = stale_candidates[0]
        stale_block = mgr.pool.blocks[stale_bid]

        # CRITICAL: block_hash must be None (COLLISION treatment)
        assert stale_block.block_hash is None, (
            "Stale SSD block should have block_hash=None (COLLISION)"
        )
        assert target_hash not in mgr.hash_table, (
            "Target hash should NOT be in hash_table after stale promote"
        )

        # Verify kv_promote_fail incremented
        stats = mgr.get_stats()
        assert stats["kv_promote_fail"] == 1


# ---------------------------------------------------------------------------
# _rollback_allocations_locked helper
# ---------------------------------------------------------------------------


class TestRollbackAllocationsLocked:
    """Tests for the extracted _rollback_allocations_locked() helper."""

    def test_rollback_freshly_allocated(self):
        """Freshly allocated blocks are returned to pool and removed from hash_table."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        # Manually set up a "freshly allocated" block
        with mgr.lock:
            block = mgr.pool.get_free_block()
            block.block_hash = "test_hash"
            block.token_ids = [1, 2, 3, 4]
            block.ref_count = 1
            mgr.hash_table["test_hash"] = block.block_id

            allocated = [block.block_id]
            freshly = [block.block_id]

            free_before = mgr.pool.num_free

            mgr._rollback_allocations_locked(allocated, freshly)

            # Block should be returned to free pool
            assert mgr.pool.num_free == free_before + 1
            # Hash should be removed
            assert "test_hash" not in mgr.hash_table

    def test_rollback_reused_blocks(self):
        """Reused (cache-hit) blocks get ref_count decremented."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[ids[0]]
        # Simulate a second allocation reusing this block
        block.ref_count = 2

        with mgr.lock:
            mgr._rollback_allocations_locked(ids, [])  # Not freshly allocated

        assert block.ref_count == 1  # Decremented back

    def test_rollback_mixed(self):
        """Mix of fresh and reused blocks handled correctly."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        # Allocate a reused block (cache hit)
        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        reused_bid = ids[0]
        mgr.pool.blocks[reused_bid].ref_count = 2  # Simulating 2nd alloc

        # Allocate a fresh block
        with mgr.lock:
            fresh_block = mgr.pool.get_free_block()
            fresh_block.block_hash = "fresh_hash"
            fresh_block.token_ids = [5, 6, 7, 8]
            fresh_block.ref_count = 1
            mgr.hash_table["fresh_hash"] = fresh_block.block_id
            fresh_bid = fresh_block.block_id

            all_allocated = [reused_bid, fresh_bid]
            freshly = [fresh_bid]

            mgr._rollback_allocations_locked(all_allocated, freshly)

        # Reused block: ref_count decremented
        assert mgr.pool.blocks[reused_bid].ref_count == 1
        # Fresh block: returned to pool, hash removed
        assert "fresh_hash" not in mgr.hash_table


class TestDoubleReturnBlock:
    """Tests for BlockPool double-return guard (CACHE-M4)."""

    def test_double_return_block_ignored(self):
        """Returning the same block twice should not create duplicates in free_queue."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # Allocate a block
        with mgr.lock:
            block = mgr.pool.get_free_block()
            bid = block.block_id

        free_before = mgr.pool.num_free

        # Return once — should succeed
        mgr.pool.return_block(bid)
        assert mgr.pool.num_free == free_before + 1

        # Return again — should be ignored (double-return guard)
        mgr.pool.return_block(bid)
        assert mgr.pool.num_free == free_before + 1  # No change

        # Verify free_queue has no duplicates
        assert list(mgr.pool.free_queue).count(bid) == 1


class TestGetBlockAccessor:
    """Tests for KVCacheManager.get_block() accessor (SCHED-13)."""

    def test_get_block_accessor(self):
        """get_block() returns the correct block object."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        bid = ids[0]

        block = mgr.get_block(bid)
        assert block.block_id == bid
        assert block is mgr.pool.blocks[bid]

    def test_get_block_matches_pool(self):
        """get_block() returns same object as pool.blocks[]."""
        config = make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        for i in range(config.num_blocks):
            assert mgr.get_block(i) is mgr.pool.blocks[i]


# ---------------------------------------------------------------------------
# evict_to_ssd heap-based candidate selection
# ---------------------------------------------------------------------------


class TestEvictToSSDHeap:
    """Tests for TieredKVCache.evict_to_ssd() using heap-based LRU selection."""

    def _make_kv_data(self) -> dict[str, mx.array]:
        return {
            "keys": mx.random.normal((1, 2, 4, 8)),
            "values": mx.random.normal((1, 2, 4, 8)),
        }

    def test_evict_to_ssd_uses_heap_efficiently(self, tmp_path):
        """Verify evict_to_ssd produces correct LRU ordering via heap.

        Allocates multiple blocks with explicit last_accessed timestamps,
        evicts a subset, and verifies the oldest blocks are evicted first.
        """
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        # Allocate 5 blocks with distinct tokens
        all_tokens = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]

        all_ids = []
        for tokens in all_tokens:
            ids = mgr.allocate_blocks(tokens)
            assert len(ids) == 1
            block = mgr.pool.blocks[ids[0]]
            block.kv_data = self._make_kv_data()
            all_ids.append(ids[0])

        # Free all blocks so they're evictable
        mgr.free_blocks(all_ids)

        # Set explicit timestamps: block 0 oldest, block 4 newest
        for i, bid in enumerate(all_ids):
            mgr.pool.blocks[bid].last_accessed = 100.0 + i * 100.0

        # Rebuild heap so timestamps are fresh
        mgr._rebuild_eviction_heap()

        # Save hashes before eviction
        hashes = [mgr.pool.blocks[bid].block_hash for bid in all_ids]

        # Evict 3 blocks — should pick the 3 oldest (blocks 0, 1, 2)
        evicted = tiered.evict_to_ssd(num_blocks=3)
        assert len(evicted) == 3

        # Verify the 3 oldest blocks were evicted
        assert all_ids[0] in evicted, "Oldest block (ts=100) should be evicted"
        assert all_ids[1] in evicted, "Second oldest block (ts=200) should be evicted"
        assert all_ids[2] in evicted, "Third oldest block (ts=300) should be evicted"

        # Newest blocks should still be in RAM
        assert hashes[3] in mgr.hash_table, "Block 3 should still be in RAM"
        assert hashes[4] in mgr.hash_table, "Block 4 should still be in RAM"

        # Evicted blocks should be on SSD
        for i in range(3):
            assert ssd.has_block(hashes[i]), f"Block {i} should be on SSD"

    def test_evict_to_ssd_respects_exclude_ids_via_heap(self, tmp_path):
        """evict_to_ssd with heap skips excluded blocks and re-pushes them."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]

        ids_a = mgr.allocate_blocks(tokens_a)
        ids_b = mgr.allocate_blocks(tokens_b)
        ids_c = mgr.allocate_blocks(tokens_c)

        for ids in [ids_a, ids_b, ids_c]:
            mgr.pool.blocks[ids[0]].kv_data = self._make_kv_data()

        mgr.free_blocks(ids_a + ids_b + ids_c)

        # A is oldest, B middle, C newest
        mgr.pool.blocks[ids_a[0]].last_accessed = 100.0
        mgr.pool.blocks[ids_b[0]].last_accessed = 200.0
        mgr.pool.blocks[ids_c[0]].last_accessed = 300.0

        mgr._rebuild_eviction_heap()

        # Protect block A from eviction
        protected = {ids_a[0]}

        # Evict 2 — should skip A, evict B and C
        evicted = tiered.evict_to_ssd(num_blocks=2, exclude_ids=protected)
        assert len(evicted) == 2
        assert ids_a[0] not in evicted
        assert ids_b[0] in evicted
        assert ids_c[0] in evicted

        # Block A should still be in RAM
        block_a = mgr.pool.blocks[ids_a[0]]
        assert block_a.block_hash in mgr.hash_table

    def test_evict_to_ssd_heap_fallback(self, tmp_path):
        """evict_to_ssd falls back to linear scan when heap is empty/stale."""
        config = make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "cache")
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        mgr.pool.blocks[ids[0]].kv_data = self._make_kv_data()
        mgr.free_blocks(ids)

        # Clear the heap to force fallback path
        mgr._eviction_heap.clear()

        # Eviction should still work via fallback linear scan
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1
        assert ids[0] in evicted
