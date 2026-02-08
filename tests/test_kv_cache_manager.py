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
    compute_block_hash,
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
