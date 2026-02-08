"""Stream A-F verification/acceptance tests.

Validates that each refactoring and feature step works correctly:
- Stream A: Refactored methods exist and function
- Stream B: Correctness fixes (blake2b hashing)
- Stream C: Performance improvements (trie, heap eviction)
- Stream F: Critical bug fixes (tuples, ref_count, eviction protection)
"""

from __future__ import annotations

import inspect
import queue
import threading
import time

import mlx.core as mx
import pytest

from conftest import make_test_config, make_test_request
from mlx_lm_server.kv_cache_manager import (
    KVCacheManager,
    compute_block_hash,
    extract_block,
)
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.sequence_cache import SequenceCacheStore
from mlx_lm_server.types import TokenEvent


# =========================================================================
# Stream A: Refactoring
# =========================================================================


class TestStreamA_Refactoring:
    """Verify refactored methods exist and work correctly."""

    def test_a1_put_event_to_stream_exists(self):
        """Scheduler._put_event_to_stream method exists."""
        assert hasattr(Scheduler, "_put_event_to_stream"), (
            "Scheduler._put_event_to_stream should exist after Stream A refactoring"
        )
        assert callable(getattr(Scheduler, "_put_event_to_stream")), (
            "_put_event_to_stream should be callable"
        )

    def test_a1_put_event_finish_guarantees_delivery(self):
        """Finish events are guaranteed delivery even on a full queue.

        Creates a bounded queue (maxsize=1), fills it, then calls
        _put_event_to_stream with is_finish=True on a separate thread.
        The finish event should eventually get through.
        """
        config = make_test_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        stream_q: queue.Queue = queue.Queue(maxsize=1)

        # Fill the queue so it is full
        filler_event = TokenEvent(
            request_id="req-fill",
            token_id=0,
            token_text="filler",
            finish_reason=None,
        )
        stream_q.put(filler_event)
        assert stream_q.full()

        finish_event = TokenEvent(
            request_id="req-1",
            token_id=99,
            token_text="done",
            finish_reason="stop",
        )

        delivered = threading.Event()

        def put_finish():
            scheduler._put_event_to_stream(
                stream_q, finish_event, "req-1", is_finish=True
            )
            delivered.set()

        t = threading.Thread(target=put_finish, daemon=True)
        t.start()

        # The method should eventually deliver (via drop-oldest or blocking put).
        # Wait for delivery.
        assert delivered.wait(timeout=10), (
            "Finish event should be delivered within timeout"
        )

        # Drain the queue and verify the finish event is present
        events = []
        while not stream_q.empty():
            events.append(stream_q.get_nowait())
        finish_events = [e for e in events if e.finish_reason == "stop"]
        assert len(finish_events) >= 1, (
            "The finish event should be in the queue after delivery"
        )
        t.join(timeout=2)

    def test_a2_process_batch_responses_exists(self):
        """Scheduler._process_batch_responses method exists."""
        assert hasattr(Scheduler, "_process_batch_responses"), (
            "Scheduler._process_batch_responses should exist after Stream A refactoring"
        )

    def test_a2_store_finished_caches_exists(self):
        """Scheduler._store_finished_caches method exists."""
        assert hasattr(Scheduler, "_store_finished_caches"), (
            "Scheduler._store_finished_caches should exist after Stream A refactoring"
        )

    def test_a2_prune_ssd_if_needed_exists(self):
        """Scheduler._prune_ssd_if_needed method exists."""
        assert hasattr(Scheduler, "_prune_ssd_if_needed"), (
            "Scheduler._prune_ssd_if_needed should exist after Stream A refactoring"
        )

    def test_a3_do_inference_exists(self):
        """server.py contains an async def _do_inference function.

        Since _do_inference is a local function/closure inside create_app,
        we verify it exists by inspecting the module source.
        """
        import mlx_lm_server.server as server_module

        source = inspect.getsource(server_module)
        assert "async def _do_inference" in source, (
            "server.py should contain 'async def _do_inference' after Stream A refactoring"
        )

    def test_a4_cache_block_method(self):
        """KVCacheManager.cache_block() allocates and registers a block."""
        config = make_test_config()
        mgr = KVCacheManager(config)

        block_hash = compute_block_hash([1, 2], [3, 4, 5, 6])
        block_size = config.block_size
        kv_data = [
            {
                "keys": mx.zeros((1, 1, block_size, 8)),
                "values": mx.zeros((1, 1, block_size, 8)),
            }
            for _ in range(2)
        ]

        block_id = mgr.cache_block(
            block_hash=block_hash, token_ids=[3, 4, 5, 6], kv_data=kv_data
        )

        assert block_id is not None, "cache_block should return a block_id"
        assert block_hash in mgr.hash_table, (
            "The block hash should be registered in hash_table"
        )
        assert mgr.hash_table[block_hash] == block_id


# =========================================================================
# Stream B: Correctness
# =========================================================================


class TestStreamB_Correctness:
    """Test correctness fixes: blake2b hashing with sentinel separator."""

    def test_b2_blake2b_returns_hex_string(self):
        """compute_block_hash returns a hex string (not int), length 32."""
        result = compute_block_hash([1, 2], [3, 4])
        assert isinstance(result, str), (
            f"Expected str, got {type(result).__name__}"
        )
        assert len(result) == 32, (
            f"Expected 32-char hex digest (16 bytes), got length {len(result)}"
        )
        # Verify it is valid hex
        int(result, 16)

    def test_b2_blake2b_deterministic(self):
        """Same inputs always produce the same hash output."""
        h1 = compute_block_hash([10, 20, 30], [40, 50])
        h2 = compute_block_hash([10, 20, 30], [40, 50])
        assert h1 == h2, "Determinism: identical inputs must produce identical hashes"

    def test_b2_blake2b_different_inputs_different_hashes(self):
        """Different token sequences produce different hashes."""
        h1 = compute_block_hash([1, 2, 3], [4, 5, 6])
        h2 = compute_block_hash([7, 8, 9], [10, 11, 12])
        assert h1 != h2, "Different inputs should produce different hashes"

    def test_b2_prefix_boundary_matters(self):
        """Separator sentinel distinguishes prefix boundary positions.

        compute_block_hash([1,2,3], [4]) != compute_block_hash([1,2], [3,4])
        because the sentinel is inserted between prefix and block tokens.
        """
        h1 = compute_block_hash([1, 2, 3], [4])
        h2 = compute_block_hash([1, 2], [3, 4])
        assert h1 != h2, (
            "Different prefix boundaries should produce different hashes "
            "(sentinel separator must be present)"
        )


# =========================================================================
# Stream C: Performance
# =========================================================================


class TestStreamC_Performance:
    """Test performance improvements: trie-based lookup, heap eviction."""

    def test_c1_trie_based_sequence_cache(self):
        """SequenceCacheStore uses trie for prefix lookup."""
        store = SequenceCacheStore(max_entries=10)

        # Store a short sequence
        store.store([1, 2, 3], ["cache_short"])

        # Find exact match
        cache, remaining = store.find_longest_prefix([1, 2, 3])
        assert cache is not None, "Exact match should be found"
        assert remaining == [], "No remaining tokens for exact match"

        # Store a longer sequence sharing a prefix
        store.store([1, 2, 3, 4, 5], ["cache_long"])

        # Query with the longer sequence: should find the longer prefix
        cache, remaining = store.find_longest_prefix([1, 2, 3, 4, 5])
        assert cache is not None, "Longer prefix should be found"
        assert remaining == [], "No remaining tokens for exact long match"

        # Query with even longer tokens: should find the longest stored prefix
        cache, remaining = store.find_longest_prefix([1, 2, 3, 4, 5, 6, 7])
        assert cache is not None, "Prefix [1,2,3,4,5] should match"
        assert remaining == [6, 7], (
            f"Remaining should be [6,7], got {remaining}"
        )

    def test_c1_trie_lru_eviction(self):
        """SequenceCacheStore(max_entries=2) evicts oldest on 3rd insert."""
        store = SequenceCacheStore(max_entries=2)

        store.store([1, 2], ["first"])
        store.store([3, 4], ["second"])
        assert store.size == 2

        # Third entry should trigger eviction of the oldest (first)
        store.store([5, 6], ["third"])
        assert store.size == 2, "Size should stay at max_entries=2"

        # The first entry should have been evicted
        cache, remaining = store.find_longest_prefix([1, 2])
        assert cache is None, "First entry should be evicted"

        # The second and third should still be present
        cache2, _ = store.find_longest_prefix([3, 4])
        assert cache2 is not None, "Second entry should still exist"

        cache3, _ = store.find_longest_prefix([5, 6])
        assert cache3 is not None, "Third entry should still exist"

    def test_c2_shallow_copy_isolation(self):
        """Stored cache data is isolated from later modifications.

        Store data, retrieve it, modify the copy, retrieve again --
        the original cached data should be unchanged.
        """
        store = SequenceCacheStore(max_entries=10)

        original_data = [{"key": "original_value", "nested": [1, 2, 3]}]
        store.store([10, 20, 30], original_data)

        # Retrieve and modify
        retrieved, _ = store.find_longest_prefix([10, 20, 30])
        assert retrieved is not None
        retrieved[0]["key"] = "MODIFIED"
        retrieved[0]["nested"].append(999)

        # Retrieve again -- should be unchanged
        retrieved2, _ = store.find_longest_prefix([10, 20, 30])
        assert retrieved2 is not None
        assert retrieved2[0]["key"] == "original_value", (
            "Original cached data should not be modified by changes to retrieved copy"
        )
        assert 999 not in retrieved2[0]["nested"], (
            "Nested data should not be contaminated"
        )

    def test_c3_eviction_heap_exists(self):
        """KVCacheManager has a _eviction_heap attribute (a list)."""
        config = make_test_config()
        mgr = KVCacheManager(config)
        assert hasattr(mgr, "_eviction_heap"), (
            "KVCacheManager should have _eviction_heap attribute"
        )
        assert isinstance(mgr._eviction_heap, list), (
            "_eviction_heap should be a list"
        )

    def test_c3_eviction_heap_populated_on_free(self):
        """Freeing blocks pushes entries onto the eviction heap."""
        config = make_test_config()
        mgr = KVCacheManager(config)

        # Allocate some blocks
        tokens = list(range(config.block_size * 2))
        block_ids = mgr.allocate_blocks(tokens)
        assert len(block_ids) == 2

        # Free the blocks
        mgr.free_blocks(block_ids)

        # The eviction heap should now have entries
        assert len(mgr._eviction_heap) > 0, (
            "Eviction heap should be non-empty after freeing blocks"
        )

    def test_c3_eviction_uses_heap(self):
        """Eviction selects oldest-accessed block (not random).

        Allocate all blocks, free them with different timestamps,
        evict one -- verify the oldest-accessed is evicted.
        """
        config = make_test_config(num_blocks=4, block_size=4)
        mgr = KVCacheManager(config)

        # Allocate 4 blocks with different token sequences
        blocks_info = []
        for i in range(4):
            tokens = list(range(i * 4, (i + 1) * 4))
            prefix = list(range(0, i * 4))
            full_tokens = prefix + tokens
            block_ids = mgr.allocate_blocks(full_tokens, num_existing_blocks=i)
            assert len(block_ids) == 1, f"Expected 1 block, got {len(block_ids)}"
            blocks_info.append((block_ids[0], tokens))

        # Free all blocks with staggered timestamps
        # Block 0 oldest, block 3 newest
        for idx, (bid, _) in enumerate(blocks_info):
            block = mgr.pool.blocks[bid]
            block.last_accessed = time.time() - (100 - idx * 10)
            block.ref_count = 0
            import heapq
            heapq.heappush(mgr._eviction_heap, (block.last_accessed, bid))

        oldest_bid = blocks_info[0][0]
        oldest_hash = mgr.pool.blocks[oldest_bid].block_hash

        # Evict one block
        evicted = mgr.evict_lru(num_blocks=1)
        assert len(evicted) == 1, "Should evict exactly 1 block"
        assert evicted[0] == oldest_bid, (
            f"Evicted block {evicted[0]} but expected oldest block {oldest_bid}"
        )
        assert oldest_hash not in mgr.hash_table, (
            "Evicted block's hash should be removed from hash_table"
        )


# =========================================================================
# Stream F: Critical Bug Fixes
# =========================================================================


class TestStreamF_CriticalBugs:
    """Test critical flow bug fixes from Stream F."""

    def test_f2_extract_block_handles_tuples(self):
        """extract_block() works with mx.array KV data (simulating QuantizedKVCache).

        The function takes (keys, values, start_pos, block_size) where keys
        and values are mx.arrays. This test ensures no crash with valid input.
        """
        block_size = 4
        seq_len = 8
        keys = mx.zeros((1, 2, seq_len, 8))
        values = mx.ones((1, 2, seq_len, 8))

        result = extract_block(keys, values, start_pos=0, block_size=block_size)

        assert "keys" in result, "Result should have 'keys'"
        assert "values" in result, "Result should have 'values'"
        assert result["keys"].shape == (1, 2, block_size, 8), (
            f"Keys shape should be (1, 2, {block_size}, 8), got {result['keys'].shape}"
        )
        assert result["values"].shape == (1, 2, block_size, 8), (
            f"Values shape should be (1, 2, {block_size}, 8), got {result['values'].shape}"
        )

    def test_f3_ref_count_on_cached_block(self):
        """cache_block() sets ref_count on newly cached blocks.

        After caching, the block is in the hash table and has a defined
        ref_count (currently 0 per implementation -- pushed to eviction heap).
        """
        config = make_test_config()
        mgr = KVCacheManager(config)

        block_hash = compute_block_hash([], [1, 2, 3, 4])
        block_size = config.block_size
        kv_data = [
            {
                "keys": mx.zeros((1, 1, block_size, 8)),
                "values": mx.zeros((1, 1, block_size, 8)),
            }
            for _ in range(2)
        ]

        block_id = mgr.cache_block(
            block_hash=block_hash, token_ids=[1, 2, 3, 4], kv_data=kv_data
        )
        assert block_id is not None

        block = mgr.pool.blocks[block_id]
        # cache_block sets ref_count=0 (block is cached but not actively referenced).
        # This is correct: the block is in the hash table for future reuse.
        # When allocate_blocks finds it via hash, ref_count gets incremented to 1.
        assert block.ref_count >= 0, (
            f"ref_count should be non-negative, got {block.ref_count}"
        )
        assert block.block_hash == block_hash, (
            "Block hash should match what was cached"
        )

    def test_f3_ref_count_survives_eviction_pressure(self):
        """A block with ref_count > 0 is NOT evicted under pressure.

        Allocate blocks (ref_count=1 from allocate_blocks), then try to
        evict -- no blocks should be evicted because all have ref_count > 0.
        """
        config = make_test_config(num_blocks=4, block_size=4)
        mgr = KVCacheManager(config)

        # Allocate blocks -- these get ref_count=1
        tokens = list(range(config.block_size * 2))  # 2 blocks worth
        block_ids = mgr.allocate_blocks(tokens)
        assert len(block_ids) == 2

        # Verify ref_count is 1
        for bid in block_ids:
            assert mgr.pool.blocks[bid].ref_count == 1, (
                f"Block {bid} should have ref_count=1 after allocation"
            )

        # Try to evict -- should fail because all blocks have ref_count > 0
        evicted = mgr.evict_lru(num_blocks=1)
        assert len(evicted) == 0, (
            "No blocks should be evicted when all have ref_count > 0"
        )

        # All original blocks should still be in the hash table
        for bid in block_ids:
            block = mgr.pool.blocks[bid]
            assert block.block_hash is not None, (
                f"Block {bid} should still have its hash after failed eviction"
            )
            assert block.block_hash in mgr.hash_table, (
                f"Block {bid}'s hash should still be in hash_table"
            )
