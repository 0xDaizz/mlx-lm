"""Tests for the KV Cache Manager (P1.1 through P1.6).

Covers:
- BlockPool pre-allocation and free queue (P1.1)
- compute_block_hash determinism and uniqueness (P1.2)
- find_cached_prefix full/partial/miss scenarios (P1.3)
- allocate_blocks cache hit, fresh alloc, ref_count tracking (P1.4)
- free_blocks ref_count decrement, block stays in hash table (P1.5)
- evict_lru ordering and skipping in-use blocks (P1.6)
"""

from __future__ import annotations

import threading
import time

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.kv_cache_manager import (
    BlockPool,
    BlockPoolExhaustedError,
    KVCacheManager,
    compute_block_hash,
)
from mlx_lm_server.types import KVCacheBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> ServerConfig:
    """Create a ServerConfig with small defaults suitable for testing."""
    defaults = {
        "block_size": 4,
        "num_blocks": 8,
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)


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
        block.block_hash = 12345
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
        assert isinstance(h, int)

    def test_hash_empty_block_tokens(self):
        """Hash works with empty block tokens (edge case)."""
        h = compute_block_hash([1, 2, 3], [])
        assert isinstance(h, int)

    def test_hash_boundary_between_prefix_and_block(self):
        """Hash differentiates where the prefix/block boundary falls.

        [1,2,3] prefix + [4] block and [1,2] prefix + [3,4] block
        produce the same hash because the concatenation is identical: (1,2,3,4).
        This is by design: the block hash represents the full context
        up to and including the block.
        """
        h1 = compute_block_hash([1, 2, 3], [4])
        h2 = compute_block_hash([1, 2], [3, 4])
        # These are intentionally the same because (1,2,3,4) == (1,2,3,4)
        assert h1 == h2

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
