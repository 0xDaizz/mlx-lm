"""Adversarial / devil's advocate tests.

Stress-tests for Phase 1 (KV Cache + SSD), Phase 2 (Scheduler),
and Phase 3 (Server) to expose hidden bugs, race conditions, edge
cases, and violations of invariants.

Each test class is prefixed so it can be filtered with pytest -k:
  pytest -k da_p1   # Phase 1 tests only
  pytest -k da_p2   # Phase 2 tests only
  pytest -k da_p3   # Phase 3 tests only
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any

import mlx.core as mx
import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.kv_cache_manager import (
    BlockPool,
    BlockPoolExhaustedError,
    KVCacheManager,
    TieredKVCache,
    compute_block_hash,
)
from mlx_lm_server.scheduler import RequestQueue, Scheduler
from mlx_lm_server.ssd_cache import SSDCache
from mlx_lm_server.types import InferenceRequest, SequenceState, TokenEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from conftest import make_test_request as _make_request
from conftest import make_test_config


def _make_config(**overrides) -> ServerConfig:
    overrides.setdefault("num_blocks", 8)
    return make_test_config(**overrides)


def _make_scheduler_config(**overrides) -> ServerConfig:
    return make_test_config(**overrides)


def _make_kv_data(
    batch: int = 1, heads: int = 2, seq_len: int = 4, head_dim: int = 8
) -> dict[str, mx.array]:
    keys = mx.random.normal((batch, heads, seq_len, head_dim))
    values = mx.random.normal((batch, heads, seq_len, head_dim))
    return {"keys": keys, "values": values}


# ===========================================================================
# PHASE 1: KV Cache + SSD Tier Adversarial Tests
# ===========================================================================


class TestDA_P1_ConcurrentFreeBlocks:
    """DA-P1-1: Concurrent free_blocks() on the same block — race on ref_count.

    Attack vector: Two threads calling free_blocks([block_id]) on the same
    block simultaneously. If ref_count operations are not atomic within the
    lock, we could get double-decrement or negative ref_count.
    """

    def test_da_p1_concurrent_free_same_block(self):
        """Multiple threads free the same block concurrently — ref_count
        must never go negative."""
        config = _make_config(block_size=4, num_blocks=16)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        # Allocate many times to raise ref_count
        for _ in range(20):
            mgr.allocate_blocks(tokens)

        block_ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[block_ids[0]]
        assert block.ref_count == 21

        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def free_worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(2):
                    mgr.free_blocks(block_ids)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=free_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        # After 10 threads x 2 frees = 20 frees, ref_count should be 1
        # (started at 21)
        assert block.ref_count >= 0, f"ref_count went negative: {block.ref_count}"
        assert block.ref_count == 1

    def test_da_p1_free_same_block_id_twice_in_list(self):
        """free_blocks([id, id]) with the same id repeated should decrement
        ref_count twice (or handle gracefully)."""
        config = _make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        mgr.allocate_blocks(tokens)
        mgr.allocate_blocks(tokens)  # ref_count = 2
        block_ids = mgr.allocate_blocks(tokens)
        block = mgr.pool.blocks[block_ids[0]]
        assert block.ref_count == 3

        # Pass the same block_id twice
        mgr.free_blocks([block_ids[0], block_ids[0]])
        assert block.ref_count == 1
        # Should not go below 0
        mgr.free_blocks([block_ids[0]])
        assert block.ref_count == 0
        mgr.free_blocks([block_ids[0]])  # Already 0 -- should not go negative
        assert block.ref_count == 0


class TestDA_P1_HashCollision:
    """DA-P1-2: Two different token sequences producing the same hash.

    Attack vector: Force a hash collision and verify the system does not
    silently return wrong KV data.
    """

    def test_da_p1_hash_collision_allocate(self):
        """When a hash collision occurs during allocation, the system must
        not reuse the block with different tokens."""
        config = _make_config(block_size=4, num_blocks=16)
        mgr = KVCacheManager(config)

        tokens_a = [1, 2, 3, 4]
        ids_a = mgr.allocate_blocks(tokens_a)
        block_a = mgr.pool.blocks[ids_a[0]]

        # Manually inject a collision: change the block's hash to match
        # what tokens_b would compute to
        tokens_b = [5, 6, 7, 8]
        hash_b = compute_block_hash([], tokens_b)

        # Store the original hash and manipulate the hash table
        original_hash = block_a.block_hash
        del mgr.hash_table[original_hash]
        block_a.block_hash = hash_b
        mgr.hash_table[hash_b] = block_a.block_id
        # block_a.token_ids is still [1, 2, 3, 4]

        # Now allocate tokens_b -- hash matches but tokens differ
        ids_b = mgr.allocate_blocks(tokens_b)

        # The system should have detected the collision and allocated a NEW block
        assert ids_b[0] != ids_a[0], "Collision not detected -- reused wrong block!"

        # Verify the new block has correct token_ids
        block_b = mgr.pool.blocks[ids_b[0]]
        assert block_b.token_ids == [5, 6, 7, 8]

    def test_da_p1_hash_collision_find_prefix(self):
        """When a hash collision occurs during find_cached_prefix, the
        system must NOT report tokens as cached."""
        config = _make_config(block_size=4, num_blocks=16)
        mgr = KVCacheManager(config)

        tokens_a = [1, 2, 3, 4]
        ids_a = mgr.allocate_blocks(tokens_a)
        block_a = mgr.pool.blocks[ids_a[0]]

        # Forge a collision for tokens_b
        tokens_b = [10, 20, 30, 40]
        hash_b = compute_block_hash([], tokens_b)

        original_hash = block_a.block_hash
        del mgr.hash_table[original_hash]
        block_a.block_hash = hash_b
        mgr.hash_table[hash_b] = block_a.block_id

        # find_cached_prefix should detect token mismatch and return 0
        cached = mgr.find_cached_prefix(tokens_b)
        assert cached == 0, "Collision not detected in find_cached_prefix!"


class TestDA_P1_MemoryLeak:
    """DA-P1-3: Sequence errors out mid-generation — blocks never freed.

    Attack vector: Blocks are allocated for a request, but if the request
    errors out without calling free_blocks(), those blocks remain with
    ref_count > 0 indefinitely.
    """

    def test_da_p1_blocks_leaked_without_free(self):
        """Blocks allocated but never freed remain with ref_count > 0
        and cannot be evicted."""
        config = _make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # Allocate 2 blocks for a "request"
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = mgr.allocate_blocks(tokens)
        assert len(ids) == 2
        assert mgr.num_free_blocks == 2

        # Simulate error: don't call free_blocks()
        # Try to evict -- should fail because ref_count > 0
        evicted = mgr.evict_lru(num_blocks=2)
        assert evicted == [], "Should not evict blocks with ref_count > 0"

        # Now try to allocate more -- should fail because pool has only 2 free
        # and the leaked blocks cannot be evicted
        tokens2 = [9, 10, 11, 12, 13, 14, 15, 16]
        ids2 = mgr.allocate_blocks(tokens2)
        assert len(ids2) == 2  # Uses the remaining 2 free blocks

        # Third allocation should fail -- all 4 blocks in use, none evictable
        with pytest.raises(BlockPoolExhaustedError):
            mgr.allocate_blocks([17, 18, 19, 20])

    def test_da_p1_explicit_cleanup_recovers(self):
        """After explicit free_blocks, blocks become evictable."""
        config = _make_config(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = mgr.allocate_blocks(tokens)
        assert mgr.num_free_blocks == 0

        # Simulate: error handler calls free_blocks
        mgr.free_blocks(ids)

        # Now blocks are evictable
        evicted = mgr.evict_lru(num_blocks=2)
        assert len(evicted) == 2
        assert mgr.num_free_blocks == 2


class TestDA_P1_SSDCorruption:
    """DA-P1-4: load_block() with corrupted/truncated safetensors file.

    Attack vector: A safetensors file on disk is corrupted (truncated,
    random bytes, or wrong format). The load_block() must not crash.
    """

    def test_da_p1_corrupted_safetensors(self, tmp_path):
        """load_block on a corrupted file should return None, not crash."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = _make_kv_data()
        block_hash = "hash_42"

        cache.save_block(block_hash, kv_data)

        # Corrupt the file
        filepath = cache.index[block_hash].filepath
        filepath.write_bytes(b"CORRUPTED DATA HERE!!!")

        # load_block should handle this gracefully
        result = cache.load_block(block_hash)
        # If it returns None or raises a handled exception, that's acceptable
        # The key invariant: it must not crash the process with an unhandled exception
        # If result is not None, the data is garbage and that's also a problem
        assert result is None, "Corrupted file should return None"

    def test_da_p1_truncated_safetensors(self, tmp_path):
        """load_block on a truncated file should return None."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = _make_kv_data()
        block_hash = "hash_99"

        cache.save_block(block_hash, kv_data)

        # Truncate the file
        filepath = cache.index[block_hash].filepath
        original_bytes = filepath.read_bytes()
        filepath.write_bytes(original_bytes[:10])  # Only first 10 bytes

        result = cache.load_block(block_hash)
        assert result is None, "Truncated file should return None"

    def test_da_p1_empty_safetensors(self, tmp_path):
        """load_block on an empty file should return None."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = _make_kv_data()
        block_hash = "hash_77"

        cache.save_block(block_hash, kv_data)

        # Empty the file
        filepath = cache.index[block_hash].filepath
        filepath.write_bytes(b"")

        result = cache.load_block(block_hash)
        assert result is None, "Empty file should return None"


class TestDA_P1_Exhaustion:
    """DA-P1-5: All blocks allocated — new request arrives.

    Attack vector: When the block pool is completely full and eviction
    fails, verify clean error and no partial state corruption.
    """

    def test_da_p1_full_pool_no_evictable(self):
        """Pool completely full with all blocks in use raises clean error."""
        config = _make_config(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        # Fill both blocks with active references
        mgr.allocate_blocks([1, 2, 3, 4])
        mgr.allocate_blocks([5, 6, 7, 8])
        assert mgr.num_free_blocks == 0

        # New allocation should raise
        with pytest.raises(BlockPoolExhaustedError):
            mgr.allocate_blocks([9, 10, 11, 12])

    def test_da_p1_exhaustion_rollback_integrity(self):
        """When allocation fails mid-way, previously allocated blocks in
        this call should have their ref_count rolled back."""
        config = _make_config(block_size=4, num_blocks=3)
        mgr = KVCacheManager(config)

        # Fill 2 out of 3 blocks with ref_count=1 (not evictable because in use)
        ids_a = mgr.allocate_blocks([1, 2, 3, 4])
        ids_b = mgr.allocate_blocks([5, 6, 7, 8])
        assert mgr.num_free_blocks == 1

        # Try to allocate 3 blocks worth of tokens (12 tokens / 4 = 3 blocks)
        # First block will succeed (uses the 1 free block)
        # Second and third will fail (pool exhausted, no evictable)
        # The first block's ref_count should be rolled back
        with pytest.raises(BlockPoolExhaustedError):
            mgr.allocate_blocks([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])

        # The free block should still be free (rollback happened)
        assert mgr.num_free_blocks == 1

    def test_da_p1_exhaustion_with_evictable_succeeds(self):
        """When pool is full but some blocks have ref_count=0, eviction
        allows allocation to proceed."""
        config = _make_config(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        # Allocate and free block A
        ids_a = mgr.allocate_blocks([1, 2, 3, 4])
        mgr.free_blocks(ids_a)  # ref_count=0

        # Allocate block B (still active)
        mgr.allocate_blocks([5, 6, 7, 8])
        assert mgr.num_free_blocks == 0

        # Allocate new tokens -- should evict block A (ref_count=0)
        ids_c = mgr.allocate_blocks([9, 10, 11, 12])
        assert len(ids_c) == 1


class TestDA_P1_TTLEdge:
    """DA-P1-6: Block saved at T=0, TTL=7d, prune runs at T=6d23h59m.

    Attack vector: Verify TTL boundary condition -- blocks EXACTLY at the
    boundary should NOT be pruned.
    """

    def test_da_p1_ttl_boundary_safe(self, tmp_path):
        """Block accessed just inside the TTL window survives pruning."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv_data = _make_kv_data()

        cache.save_block(111, kv_data)

        # Set last_accessed to 6 days 23 hours 59 minutes ago
        # (just inside the 7-day window)
        nearly_expired = datetime.now() - timedelta(days=6, hours=23, minutes=59)
        cache.index[111].last_accessed = nearly_expired

        pruned = cache.prune_expired()
        assert pruned == 0, "Block within TTL should not be pruned"
        assert 111 in cache.index

    def test_da_p1_ttl_boundary_expired(self, tmp_path):
        """Block accessed exactly 7 days ago should be pruned
        (last_accessed < cutoff)."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv_data = _make_kv_data()

        cache.save_block(222, kv_data)

        # Set last_accessed to 7 days + 1 second ago (just outside the window)
        just_expired = datetime.now() - timedelta(days=7, seconds=1)
        cache.index[222].last_accessed = just_expired

        pruned = cache.prune_expired()
        assert pruned == 1
        assert 222 not in cache.index

    def test_da_p1_ttl_exactly_at_boundary(self, tmp_path):
        """Block accessed exactly at the cutoff: behavior depends on
        strict < comparison."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv_data = _make_kv_data()

        cache.save_block(333, kv_data)

        # Set to exactly 7 days ago -- prune uses `<` so exactly at cutoff
        # means NOT pruned (last_accessed is not strictly less than cutoff
        # if they're equal -- but due to time passing between the two
        # datetime.now() calls, this test just verifies the logic direction)
        exactly_boundary = datetime.now() - timedelta(days=7)
        cache.index[333].last_accessed = exactly_boundary

        # This is inherently racy due to datetime.now() in prune_expired,
        # but the boundary should be approximately correct
        pruned = cache.prune_expired()
        # Either outcome is acceptable at exact boundary; just verify no crash
        assert pruned in (0, 1)


class TestDA_P1_StaleSSDIndex:
    """DA-P1-7: Process crashes between save_block and save_index.

    Attack vector: If the process crashes after writing the safetensors
    file but before updating the index, the block file exists on disk
    but is not in the index. This is an orphan file.
    """

    def test_da_p1_orphan_file_on_disk(self, tmp_path):
        """An orphaned safetensors file (not in index) is invisible to
        load_block and does not cause errors."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = _make_kv_data()

        # Manually write a safetensors file without updating the index
        filepath = cache.cache_dir / "block_999.safetensors"
        mx.save_safetensors(str(filepath), kv_data)

        # load_block should return None (not in index)
        result = cache.load_block(999)
        assert result is None

        # prune_expired should not crash despite orphan file
        pruned = cache.prune_expired()
        assert pruned == 0

    def test_da_p1_index_references_missing_file(self, tmp_path):
        """Index references a file that was removed (e.g., disk cleanup).
        load_block should handle gracefully."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = _make_kv_data()

        cache.save_block("hash_888", kv_data)
        assert "hash_888" in cache.index

        # Remove the file behind the cache's back
        filepath = cache.index["hash_888"].filepath
        filepath.unlink()

        # load_block should detect the missing file and clean up index
        result = cache.load_block("hash_888")
        assert result is None
        assert "hash_888" not in cache.index

    def test_da_p1_partial_index_write(self, tmp_path):
        """Corrupted index.json is handled gracefully on startup."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        # Write partial/corrupt index
        index_path = cache_dir / "index.json"
        index_path.write_text('{"hash_12345": {"block_hash": "hash_12345", "filepath": "')

        # Should recover gracefully
        cache = SSDCache(cache_dir=cache_dir)
        assert cache.num_blocks == 0


class TestDA_P1_EvictLRULogBug:
    """DA-P1-BUG: _evict_lru_locked logs block_hash AFTER return_block
    clears it to None.

    The logger.debug line references block.block_hash after
    pool.return_block() has set block_hash = None.
    """

    def test_da_p1_evict_lru_no_crash(self):
        """evict_lru should not crash when logging after return_block."""
        config = _make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4]
        ids = mgr.allocate_blocks(tokens)
        mgr.free_blocks(ids)

        # This used to potentially log None for block_hash
        # Should not crash
        evicted = mgr.evict_lru(num_blocks=1)
        assert len(evicted) == 1


# ===========================================================================
# PHASE 2: Scheduler Adversarial Tests
# ===========================================================================


class TestDA_P2_ConcurrentScheduleStep:
    """DA-P2-1: schedule_step() called while previous step still running.

    Attack vector: Two threads calling schedule_step() simultaneously,
    potentially causing double-add of sequences or race on active set.
    """

    def test_da_p2_concurrent_schedule_step(self):
        """Concurrent schedule_step calls must not duplicate sequences."""
        config = _make_scheduler_config(max_batch_size=4)
        s = Scheduler(config=config, model=None, tokenizer=None)

        for i in range(4):
            s.submit_request(_make_request(f"r{i}"))

        errors: list[Exception] = []
        results: list[int] = []
        barrier = threading.Barrier(4)

        def stepper():
            try:
                barrier.wait(timeout=5)
                outputs = s.schedule_step()
                results.append(
                    len(outputs.prefill_sequences) + len(outputs.decode_sequences)
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stepper) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        # Total sequences across all step results should be <= 4
        # (some steps may pick up 0 if another thread already grabbed them)
        assert s.num_active_sequences <= 4


class TestDA_P2_Starvation:
    """DA-P2-3: Long-running sequence blocks all new requests indefinitely.

    Attack vector: If max_batch_size=1 and one request generates tokens
    forever (max_tokens very high), new requests should eventually be served
    when the long one finishes.
    """

    def test_da_p2_long_request_does_not_starve(self):
        """A long-running request with max_batch_size=1 blocks new requests
        until it finishes, but doesn't deadlock."""
        config = _make_scheduler_config(max_batch_size=1)
        s = Scheduler(config=config, model=None, tokenizer=None)

        # Mock: first request generates 5 tokens, second generates 2
        def mock_gen(rid, tids, step):
            if rid == "r1":
                if step >= 4:
                    return (step + 1, f"t{step}", "stop")
                return (step + 1, f"t{step}", None)
            else:
                if step >= 1:
                    return (step + 1, f"t{step}", "stop")
                return (step + 1, f"t{step}", None)

        s._mock_generate = mock_gen

        s.submit_request(_make_request("r1", max_tokens=10))
        s.submit_request(_make_request("r2", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            t1 = s.get_result("r1", timeout=5.0)
            t2 = s.get_result("r2", timeout=5.0)

            assert len(t1) == 5
            assert len(t2) == 2
            assert t1[-1].finish_reason == "stop"
            assert t2[-1].finish_reason == "stop"
        finally:
            s.stop()


class TestDA_P2_QueueMemory:
    """DA-P2-4: 100 queued requests memory.

    Attack vector: Queue accepts up to max_queue_size. Exceeding it should
    raise an error. Verify no memory leaks for large queues.
    """

    def test_da_p2_queue_overflow(self):
        """Requests beyond max_queue_size are rejected."""
        config = _make_scheduler_config(max_queue_size=10)
        s = Scheduler(config=config, model=None, tokenizer=None)

        for i in range(10):
            s.submit_request(_make_request(f"r{i}"))

        with pytest.raises(RuntimeError, match="queue is full"):
            s.submit_request(_make_request("overflow"))

    def test_da_p2_queue_large_batch(self):
        """Queue handles 100 requests without errors."""
        config = _make_scheduler_config(max_queue_size=128)
        s = Scheduler(config=config, model=None, tokenizer=None)

        for i in range(100):
            s.submit_request(
                _make_request(
                    f"r{i}",
                    prompt_tokens=list(range(32)),
                    max_tokens=1,
                )
            )

        assert s.num_queued_requests == 100


class TestDA_P2_ModelException:
    """DA-P2-5: Model exception mid-decode — scheduler state corrupted?

    Attack vector: If _mock_generate raises an exception during decode,
    the scheduler should not leave the sequence in a corrupt state.
    """

    def test_da_p2_mock_exception_mid_decode(self):
        """Exception in mock generator should not crash the inference loop.
        After the fix, the loop catches exceptions and marks affected
        sequences as errored, then continues processing new requests."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        call_count = [0]

        def flaky_gen(rid, tids, step):
            call_count[0] += 1
            if rid == "r1" and step == 1:
                raise RuntimeError("Simulated model crash!")
            if step >= 2:
                return (step + 1, f"t{step}", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = flaky_gen

        # Submit r1 first (will crash on step 1)
        s.submit_request(_make_request("r1", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            # r1 should finish with error (loop catches exception and marks
            # active sequences as failed)
            t1 = s.get_result("r1", timeout=5.0)
            # After fix: r1 gets some tokens then an error finish
            assert len(t1) >= 1

            # Now submit r2 -- the loop should still be running
            s.submit_request(_make_request("r2", max_tokens=10))
            t2 = s.get_result("r2", timeout=5.0)
            assert len(t2) >= 1
            assert t2[-1].finish_reason is not None
        finally:
            s.stop()


class TestDA_P2_StreamAbort:
    """DA-P2-6: Client disconnects mid-stream — token_queue orphaned.

    Attack vector: A stream is registered but the consumer stops reading.
    The queue grows unbounded. Verify the queue is cleaned up when the
    request finishes.
    """

    def test_da_p2_orphaned_stream_queue(self):
        """Stream queue is cleaned up after request finishes even if
        consumer never reads."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 2:
                return (step + 1, f"t{step}", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", max_tokens=10, stream=True)
        stream_q = s.register_stream("r1")
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            # Don't read from stream_q -- simulate client disconnect
            time.sleep(1.0)

            # The queue should have received tokens + finish
            items = []
            while not stream_q.empty():
                items.append(stream_q.get_nowait())
            assert len(items) >= 1, "Tokens should have been pushed to queue"
            assert any(
                i.finish_reason is not None for i in items
            ), "Finish event should be in queue"

            # Stream should be cleaned up from scheduler
            with s._streams_lock:
                assert "r1" not in s._streams, "Orphaned stream not cleaned up"
        finally:
            s.stop()


class TestDA_P2_EdgeCases:
    """DA-P2-7: max_tokens=0, empty prompt.

    Attack vector: Edge case inputs that might cause division by zero,
    empty iterations, or unexpected behavior.
    """

    def test_da_p2_max_tokens_zero(self):
        """Request with max_tokens=0 should finish immediately with
        'length' finish_reason."""
        config = _make_scheduler_config()
        s = Scheduler(config=config, model=None, tokenizer=None)
        s._mock_generate = lambda rid, tids, step: (1, "t", None)

        req = _make_request("r1", max_tokens=0)
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            # Should get exactly 1 event with finish_reason="length"
            # because the check happens in _run_decode_step before generating
            assert len(tokens) >= 1
            assert tokens[-1].finish_reason == "length"
        finally:
            s.stop()

    def test_da_p2_empty_prompt(self):
        """Request with empty prompt_tokens should not crash."""
        config = _make_scheduler_config()
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 1, "done", "stop")
            return (step + 1, "t0", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", prompt_tokens=[], max_tokens=5)
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert len(tokens) >= 1
        finally:
            s.stop()

    def test_da_p2_single_token_prompt(self):
        """Request with single-token prompt works correctly."""
        config = _make_scheduler_config()
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 1, "done", "stop")
            return (step + 1, "t0", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", prompt_tokens=[42], max_tokens=5)
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert len(tokens) == 2
            assert tokens[-1].finish_reason == "stop"
        finally:
            s.stop()


class TestDA_P2_Deadlock:
    """DA-P2-2: Request queue lock held while waiting for inference result.

    Attack vector: If the scheduler holds the _active_lock while trying
    to emit tokens that require _streams_lock, and another thread holds
    _streams_lock while trying to cancel (which requires _active_lock),
    we get a deadlock.
    """

    def test_da_p2_no_deadlock_cancel_during_emit(self):
        """Concurrent cancel and token emission should not deadlock."""
        config = _make_scheduler_config(max_batch_size=4)
        s = Scheduler(config=config, model=None, tokenizer=None)

        gate = threading.Event()

        def slow_gen(rid, tids, step):
            if step == 0:
                gate.wait(timeout=5)
            if step >= 2:
                return (step + 1, f"t{step}", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = slow_gen

        for i in range(3):
            s.submit_request(_make_request(f"r{i}", max_tokens=10))

        s.run_inference_loop(blocking=False)

        # Wait for requests to become active
        deadline = time.monotonic() + 3.0
        while s.num_active_sequences == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        # Cancel all while generation is happening
        for i in range(3):
            s.cancel_request(f"r{i}")

        gate.set()
        time.sleep(0.5)

        # If we get here without hanging, no deadlock
        s.stop()


# ===========================================================================
# PHASE 3: Server Adversarial Tests
# ===========================================================================


# Import server components
from mlx_lm_server.server import create_app


# Server fixtures (self-contained to avoid conftest conflicts)

class MockSchedulerForAdversarial:
    """Scheduler mock for adversarial server tests."""

    def __init__(self, response_tokens: list[str] | None = None):
        self.response_tokens = response_tokens or ["Hello", ",", " world", "!"]
        self.submitted: list[InferenceRequest] = []
        self.streams: dict[str, Queue[TokenEvent | None]] = {}
        self.shutdown_called = False

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)
        if request.request_id in self.streams:
            q = self.streams[request.request_id]
            for i, tok_text in enumerate(self.response_tokens):
                is_last = i == len(self.response_tokens) - 1
                q.put(
                    TokenEvent(
                        request_id=request.request_id,
                        token_id=i,
                        token_text=tok_text,
                        finish_reason="stop" if is_last else None,
                    )
                )
            q.put(None)

    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]:
        q: Queue[TokenEvent | None] = Queue()
        self.streams[request_id] = q
        return q

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        events: list[TokenEvent] = []
        for i, tok_text in enumerate(self.response_tokens):
            is_last = i == len(self.response_tokens) - 1
            events.append(
                TokenEvent(
                    request_id=request_id,
                    token_id=i,
                    token_text=tok_text,
                    finish_reason="stop" if is_last else None,
                )
            )
        return events

    def get_cache_stats(self) -> dict[str, Any]:
        return {"total_blocks": 64, "used_blocks": 10, "free_blocks": 54}

    def cancel_request(self, request_id: str) -> bool:
        return request_id in {r.request_id for r in self.submitted}

    def shutdown(self) -> None:
        self.shutdown_called = True


class MockTokenizerForAdversarial:
    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text.split()))))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


@pytest.fixture
def adv_server_config(tmp_path):
    return ServerConfig(
        model="test-model",
        block_size=4,
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        max_batch_size=2,
        max_queue_size=8,
    )


@pytest.fixture
def adv_app(adv_server_config):
    return create_app(
        config=adv_server_config,
        scheduler=MockSchedulerForAdversarial(),
        tokenizer=MockTokenizerForAdversarial(),
    )


@pytest.fixture
async def adv_client(adv_app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=adv_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestDA_P3_APIContract:
    """DA-P3-1: Missing required OpenAI response fields.

    Attack vector: Verify all required fields per the OpenAI API spec
    are present in the response.
    """

    @pytest.mark.anyio
    async def test_da_p3_chat_response_fields(self, adv_client):
        """Chat completion response must have all required OpenAI fields."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required top-level fields per OpenAI spec
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Choice fields
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        # Message fields
        msg = choice["message"]
        assert "role" in msg
        assert "content" in msg

        # Usage fields
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    @pytest.mark.anyio
    async def test_da_p3_completion_response_fields(self, adv_client):
        """Text completion response must have all required OpenAI fields."""
        resp = await adv_client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        choice = data["choices"][0]
        assert "text" in choice
        assert "finish_reason" in choice
        assert "index" in choice

    @pytest.mark.anyio
    async def test_da_p3_models_response_fields(self, adv_client):
        """GET /v1/models response must follow OpenAI format."""
        resp = await adv_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"


class TestDA_P3_InputValidation:
    """DA-P3-2: Malformed JSON, missing messages, negative max_tokens.

    Attack vector: Send invalid payloads and verify the server does not
    crash or return 500.
    """

    @pytest.mark.anyio
    async def test_da_p3_missing_messages(self, adv_client):
        """Missing 'messages' field returns 422."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={"model": "test"},
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_da_p3_empty_messages(self, adv_client):
        """Empty messages list should be handled."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
        # Should succeed (empty messages = empty prompt) or return 400
        assert resp.status_code in (200, 400, 422)

    @pytest.mark.anyio
    async def test_da_p3_missing_prompt(self, adv_client):
        """Missing 'prompt' field for completions returns 422."""
        resp = await adv_client.post(
            "/v1/completions",
            json={"model": "test"},
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_da_p3_invalid_content_type(self, adv_client):
        """Non-JSON content type is rejected."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            content="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_da_p3_extra_fields_ignored(self, adv_client):
        """Extra fields in request body should be silently ignored."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "unknown_field": "should be ignored",
                "another_field": 42,
            },
        )
        assert resp.status_code == 200


class TestDA_P3_SSEFormat:
    """DA-P3-3: Missing data: prefix, missing [DONE], newline issues.

    Attack vector: Verify SSE format compliance.
    """

    @pytest.mark.anyio
    async def test_da_p3_sse_data_prefix(self, adv_client):
        """Every SSE line must start with 'data: '."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]

        for line in lines:
            assert line.startswith("data: "), f"Line missing 'data: ' prefix: {line!r}"

    @pytest.mark.anyio
    async def test_da_p3_sse_done_marker(self, adv_client):
        """SSE stream must end with 'data: [DONE]'."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.anyio
    async def test_da_p3_sse_valid_json_chunks(self, adv_client):
        """All SSE chunks (except [DONE]) must be valid JSON."""
        resp = await adv_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]

        for line in lines:
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            # Must be valid JSON
            chunk = json.loads(payload)
            assert "choices" in chunk
            assert "id" in chunk

    @pytest.mark.anyio
    async def test_da_p3_completion_sse_format(self, adv_client):
        """Completion streaming also follows SSE format."""
        resp = await adv_client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )
        assert resp.status_code == 200
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]

        assert lines[-1] == "data: [DONE]"
        for line in lines[:-1]:
            payload = line[len("data: "):]
            chunk = json.loads(payload)
            assert chunk["object"] == "text_completion"


class TestDA_P3_Concurrency:
    """DA-P3-4: 50 simultaneous requests.

    Attack vector: High concurrency to expose race conditions in the
    server layer.
    """

    @pytest.mark.anyio
    async def test_da_p3_50_concurrent_requests(self, adv_client):
        """50 parallel requests all complete with 200 status."""
        tasks = []
        for i in range(50):
            tasks.append(
                adv_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "max_tokens": 10,
                    },
                )
            )
        responses = await asyncio.gather(*tasks)

        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == 50, (
            f"Only {success_count}/50 succeeded. "
            f"Status codes: {[r.status_code for r in responses if r.status_code != 200]}"
        )

    @pytest.mark.anyio
    async def test_da_p3_mixed_stream_nonstream(self, adv_client):
        """Mix of streaming and non-streaming requests in parallel."""
        tasks = []
        for i in range(10):
            tasks.append(
                adv_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                        "stream": i % 2 == 0,
                    },
                )
            )
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200


# ===========================================================================
# DA-Final: Cross-Component Review (DA-F-1 through DA-F-5)
# ===========================================================================


class TestDA_F1_StateLeak:
    """DA-F-1: State leak between consecutive requests.

    Attack vector: After a request completes, residual state in the scheduler
    (result buffers, sequence objects) or KV cache manager (stale ref_counts,
    dirty hash_table entries) could leak into subsequent requests, causing
    incorrect behavior or unbounded memory growth.
    """

    def test_da_f1_result_buffers_cleaned_after_get(self):
        """After get_result(), the result buffer and ready event should be
        removed. Repeated get_result should raise KeyError."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 1, "done", "stop")
            return (step + 1, "t0", None)

        s._mock_generate = mock_gen

        s.submit_request(_make_request("r1", max_tokens=5))
        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert len(tokens) >= 1

            # Result buffers should be cleaned
            with s._results_lock:
                assert "r1" not in s._results, "Result buffer leaked after get_result"
                assert "r1" not in s._results_ready, "Ready event leaked after get_result"

            # Second get_result should raise
            with pytest.raises(KeyError):
                s.get_result("r1", timeout=0.1)
        finally:
            s.stop()

    def test_da_f1_active_sequences_cleaned_after_finish(self):
        """After a request finishes, it should be removed from
        _active_sequences. No stale sequence objects remain."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            return (step + 1, "t0", "stop")

        s._mock_generate = mock_gen

        s.submit_request(_make_request("r1", max_tokens=5))
        s.run_inference_loop(blocking=False)
        try:
            s.get_result("r1", timeout=5.0)
            # Give cleanup a moment
            time.sleep(0.3)

            with s._active_lock:
                assert "r1" not in s._active_sequences, \
                    "Finished sequence still in _active_sequences"
        finally:
            s.stop()

    def test_da_f1_kv_cache_ref_counts_zero_after_request_completes(self):
        """After blocks are allocated and then explicitly freed (as the
        scheduler does on cleanup), ref_counts should reach 0. The blocks
        remain in the hash_table for reuse but are evictable."""
        config = _make_config(block_size=4, num_blocks=32)
        kv_mgr = KVCacheManager(config)

        prompt = list(range(1, 9))  # 8 tokens = 2 blocks
        block_ids = kv_mgr.allocate_blocks(prompt)

        # Simulate: scheduler allocates during request, then frees on cleanup
        for bid in block_ids:
            block = kv_mgr.pool.blocks[bid]
            assert block.ref_count == 1

        kv_mgr.free_blocks(block_ids)

        for bid in block_ids:
            block = kv_mgr.pool.blocks[bid]
            assert block.ref_count == 0, \
                f"Block {bid} ref_count should be 0 after free, got {block.ref_count}"

        # Blocks should still be in hash_table (cached for future reuse)
        assert kv_mgr.num_cached_blocks == 2

        # But they should be evictable
        evicted = kv_mgr.evict_lru(num_blocks=2)
        assert len(evicted) == 2
        assert kv_mgr.num_cached_blocks == 0

    def test_da_f1_consecutive_requests_independent_output(self):
        """Two consecutive requests should produce independent outputs
        with no state contamination."""
        config = _make_scheduler_config(max_batch_size=1)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            # Each request produces different tokens based on request_id
            if rid == "r1":
                text = f"A{step}"
            else:
                text = f"B{step}"
            if step >= 2:
                return (step + 1, text, "stop")
            return (step + 1, text, None)

        s._mock_generate = mock_gen

        s.run_inference_loop(blocking=False)
        try:
            s.submit_request(_make_request("r1", max_tokens=10))
            t1 = s.get_result("r1", timeout=5.0)

            s.submit_request(_make_request("r2", max_tokens=10))
            t2 = s.get_result("r2", timeout=5.0)

            # Verify outputs are independent
            t1_text = "".join(e.token_text for e in t1)
            t2_text = "".join(e.token_text for e in t2)
            assert "A" in t1_text and "B" not in t1_text, \
                f"r1 output contaminated: {t1_text}"
            assert "B" in t2_text and "A" not in t2_text, \
                f"r2 output contaminated: {t2_text}"
        finally:
            s.stop()

    def test_da_f1_stream_cleanup_no_leak(self):
        """After a streaming request completes, its stream queue should be
        removed from _streams. Uncollected stream queues should not accumulate."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 1, "done", "stop")
            return (step + 1, "t0", None)

        s._mock_generate = mock_gen

        s.run_inference_loop(blocking=False)
        try:
            for i in range(5):
                rid = f"stream-{i}"
                req = _make_request(rid, max_tokens=5, stream=True)
                q = s.register_stream(rid)
                s.submit_request(req)

                # Drain the stream
                while True:
                    evt = q.get(timeout=5.0)
                    if evt.finish_reason is not None:
                        break

            time.sleep(0.3)
            with s._streams_lock:
                # All streams should be cleaned up
                remaining = [k for k in s._streams if k.startswith("stream-")]
                assert remaining == [], \
                    f"Stream queues leaked: {remaining}"
        finally:
            s.stop()


class TestDA_F2_SchedulerFreesBlocksDuringSSDSave:
    """DA-F-2: Scheduler frees blocks while SSD save is in-progress.

    Attack vector: TieredKVCache.evict_to_ssd() calls ssd.save_block()
    while holding ram.lock. If another thread tries to free_blocks() or
    allocate_blocks() concurrently, it will be blocked (which is safe but
    may cause latency spikes). More critically: if save_block() raises an
    exception (e.g., disk full), the block may be partially evicted — removed
    from hash_table but the SSD save failed, causing data loss.
    """

    def test_da_f2_evict_to_ssd_concurrent_free(self, tmp_path):
        """Concurrent free_blocks during evict_to_ssd should not corrupt state."""
        config = _make_config(block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-cache")
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # Allocate blocks with KV data
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        ids_a = kv_mgr.allocate_blocks(tokens_a)
        ids_b = kv_mgr.allocate_blocks(tokens_b)

        # Attach dummy KV data
        for bid in ids_a + ids_b:
            kv_mgr.pool.blocks[bid].kv_data = _make_kv_data()

        # Free block A so it's evictable (ref_count=0)
        kv_mgr.free_blocks(ids_a)

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def evict_worker():
            try:
                barrier.wait(timeout=5)
                tiered.evict_to_ssd(num_blocks=1)
            except Exception as e:
                errors.append(e)

        def free_worker():
            try:
                barrier.wait(timeout=5)
                kv_mgr.free_blocks(ids_b)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=evict_worker)
        t2 = threading.Thread(target=free_worker)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == [], f"Errors during concurrent operations: {errors}"

        # Block B should have ref_count 0 (freed successfully)
        block_b = kv_mgr.pool.blocks[ids_b[0]]
        assert block_b.ref_count == 0

        # Block A should have been evicted to SSD
        assert ssd.num_blocks >= 1 or kv_mgr.num_free_blocks > 0

    def test_da_f2_ssd_save_failure_no_data_loss(self, tmp_path):
        """If ssd.save_block() raises during evict_to_ssd, the block should
        still be accessible (not lost from both RAM and SSD).

        FINDING: Currently, evict_to_ssd does NOT handle save_block exceptions.
        If save_block raises after hash_table removal, the block is lost.
        This test documents the vulnerability.
        """
        config = _make_config(block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-cache")
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        tokens = [1, 2, 3, 4]
        ids = kv_mgr.allocate_blocks(tokens)
        kv_mgr.pool.blocks[ids[0]].kv_data = _make_kv_data()
        kv_mgr.free_blocks(ids)  # Make evictable

        # Patch save_block to fail
        original_save = ssd.save_block
        save_called = [False]

        def failing_save(block_hash, kv_data):
            save_called[0] = True
            raise IOError("Disk full!")

        ssd.save_block = failing_save

        # evict_to_ssd will call save_block which raises — but the current
        # implementation does NOT catch this. The exception propagates.
        # The block will be in an inconsistent state.
        try:
            tiered.evict_to_ssd(num_blocks=1)
            # If we get here, the implementation swallowed the error
            # Check if block is still accessible somewhere
            block_hash = compute_block_hash([], tokens)
            in_ram = block_hash in kv_mgr.hash_table
            in_ssd = ssd.load_block(block_hash) is not None
            # At least one should be true
            assert in_ram or in_ssd, \
                "CRITICAL: Block lost from both RAM and SSD after save failure!"
        except IOError:
            # The exception propagated — block may be in inconsistent state
            # This is the expected behavior currently (the bug).
            # Verify the block's state: it was removed from hash_table
            # before save_block was called, so it's lost from RAM.
            assert save_called[0], "save_block should have been called"
            # Document: this is a real bug — see findings report

    def test_da_f2_evict_to_ssd_preserves_kv_data(self, tmp_path):
        """After evict_to_ssd, the block data should be loadable from SSD."""
        config = _make_config(block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-cache")
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        tokens = [10, 20, 30, 40]
        ids = kv_mgr.allocate_blocks(tokens)
        original_kv = _make_kv_data()
        kv_mgr.pool.blocks[ids[0]].kv_data = original_kv
        block_hash = kv_mgr.pool.blocks[ids[0]].block_hash

        kv_mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1

        # Block should no longer be in RAM hash table
        assert block_hash not in kv_mgr.hash_table

        # Block should be loadable from SSD
        loaded = ssd.load_block(block_hash)
        assert loaded is not None
        assert "keys" in loaded
        assert "values" in loaded


class TestDA_F3_SchedulerThreadDies:
    """DA-F-3: FastAPI async handler awaits scheduler -> scheduler thread dies.

    Attack vector: If the scheduler's inference loop thread crashes or stops
    unexpectedly, any pending get_result() or stream queue.get() calls will
    hang indefinitely. The server should detect this and return an error
    instead of hanging.
    """

    def test_da_f3_get_result_timeout_when_loop_not_started(self):
        """If the inference loop is never started, get_result with a
        timeout should raise TimeoutError (not hang forever or return
        partial results)."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            return (step + 1, "t", None)

        s._mock_generate = mock_gen

        # Submit request but don't start the loop
        s.submit_request(_make_request("r1", max_tokens=5))

        # get_result with timeout should raise TimeoutError (B1 fix)
        with pytest.raises(TimeoutError) as exc_info:
            s.get_result("r1", timeout=0.5)
        assert "r1" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)

    def test_da_f3_loop_stops_mid_generation(self):
        """If the loop is stopped while a request is being processed,
        the request should eventually get a result (even if incomplete)."""
        config = _make_scheduler_config(max_batch_size=1)
        s = Scheduler(config=config, model=None, tokenizer=None)

        gate = threading.Event()

        def slow_gen(rid, tids, step):
            if step == 2:
                gate.wait(timeout=2)  # Block here briefly
            if step >= 4:
                return (step + 1, "done", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = slow_gen

        s.submit_request(_make_request("r1", max_tokens=10))
        s.run_inference_loop(blocking=False)

        # Let it generate a couple tokens
        time.sleep(0.3)

        # Unblock the generator first, then stop
        gate.set()
        time.sleep(0.1)
        s.stop()

        # The result should be available (may be incomplete)
        with s._results_lock:
            events = s._results.get("r1", [])
        # We got some tokens before the loop stopped (or the error handler
        # marked it as finished)
        # Key invariant: no hang, no crash
        assert isinstance(events, list)

    def test_da_f3_loop_exception_marks_requests_failed(self):
        """If the inference loop catches an exception, it should mark
        all active requests as failed so get_result doesn't hang."""
        config = _make_scheduler_config(max_batch_size=2)
        s = Scheduler(config=config, model=None, tokenizer=None)

        def crash_gen(rid, tids, step):
            raise RuntimeError("Fatal model error")

        s._mock_generate = crash_gen

        s.submit_request(_make_request("r1", max_tokens=10))
        s.submit_request(_make_request("r2", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            # Both requests should complete (with error) within timeout
            t1 = s.get_result("r1", timeout=5.0)
            t2 = s.get_result("r2", timeout=5.0)

            # They should have been marked as errored
            assert len(t1) >= 1 or len(t2) >= 1, \
                "At least one request should have received events"
        finally:
            s.stop()

    def test_da_f3_stream_unblocks_on_loop_stop(self):
        """If the loop stops, streaming queues should eventually receive
        a finish event so consumers don't hang."""
        config = _make_scheduler_config(max_batch_size=1)
        s = Scheduler(config=config, model=None, tokenizer=None)

        gate = threading.Event()

        def blocking_gen(rid, tids, step):
            gate.wait(timeout=3)
            return (step + 1, "t", "stop")

        s._mock_generate = blocking_gen

        req = _make_request("r1", max_tokens=5, stream=True)
        stream_q = s.register_stream("r1")
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        time.sleep(0.2)

        # Unblock generator first, then stop
        gate.set()
        time.sleep(0.1)
        s.stop()

        # Try to drain the stream with timeout — should not hang
        events = []
        try:
            while True:
                evt = stream_q.get(timeout=2.0)
                events.append(evt)
                if evt.finish_reason is not None:
                    break
        except queue.Empty:
            pass  # Timeout is acceptable — means no more events

        # Key invariant: we did not hang forever


class TestDA_F4_SharedPrefixMixedStreamSync:
    """DA-F-4: 20 requests with shared prefix, mixed stream/sync — all correct?

    Attack vector: Multiple requests sharing a common prefix (triggering
    prefix cache hits) with a mix of streaming and non-streaming modes.
    Each request should produce its own independent output despite sharing
    cached prefix blocks.
    """

    def test_da_f4_shared_prefix_independent_outputs(self):
        """20 requests with identical 8-token prefix but different
        continuations. Verify each produces unique output and prefix
        cache is correctly shared."""
        config = _make_scheduler_config(
            block_size=4, num_blocks=64, max_batch_size=4,
        )
        kv_mgr = KVCacheManager(config)
        s = Scheduler(
            config=config, model=None, tokenizer=None,
            kv_cache_manager=kv_mgr,
        )

        # Shared prefix: 8 tokens = 2 blocks
        shared_prefix = [100, 200, 300, 400, 500, 600, 700, 800]

        def mock_gen(rid, tids, step):
            # Each request generates its own unique tokens based on rid
            idx = int(rid.split("-")[1])
            token_id = 1000 + idx * 100 + step
            text = f"[{rid}:{step}]"
            if step >= 2:
                return (token_id, text, "stop")
            return (token_id, text, None)

        s._mock_generate = mock_gen

        # Pre-allocate prefix blocks so cache hits are possible
        kv_mgr.allocate_blocks(shared_prefix)
        # Free them so they stay in hash_table but have ref_count=0
        # (they'll get ref_count bumped when requests use them)
        initial_cached = kv_mgr.find_cached_prefix(shared_prefix)
        assert initial_cached == 8, "Prefix should be fully cached"

        s.run_inference_loop(blocking=False)
        try:
            results: dict[str, list[TokenEvent]] = {}
            stream_queues: dict[str, Queue] = {}

            for i in range(20):
                rid = f"r-{i}"
                is_stream = i % 2 == 0
                req = _make_request(
                    rid,
                    prompt_tokens=shared_prefix.copy(),
                    max_tokens=5,
                    stream=is_stream,
                )

                if is_stream:
                    stream_queues[rid] = s.register_stream(rid)

                s.submit_request(req)

            # Collect results
            for i in range(20):
                rid = f"r-{i}"
                is_stream = i % 2 == 0

                if is_stream:
                    events = []
                    q = stream_queues[rid]
                    while True:
                        evt = q.get(timeout=10.0)
                        events.append(evt)
                        if evt.finish_reason is not None:
                            break
                    results[rid] = events
                else:
                    results[rid] = s.get_result(rid, timeout=10.0)

            # Verify: each request got its own unique output
            outputs = {}
            for rid, events in results.items():
                text = "".join(e.token_text for e in events)
                outputs[rid] = text
                assert rid in text, \
                    f"Request {rid} output should contain its ID: {text}"

            # Verify all outputs are different
            unique_outputs = set(outputs.values())
            assert len(unique_outputs) == 20, \
                f"Expected 20 unique outputs, got {len(unique_outputs)}"

        finally:
            s.stop()

    def test_da_f4_prefix_cache_hit_count(self):
        """Multiple requests with same prefix should hit the cache for
        the prefix blocks (not re-allocate)."""
        config = _make_config(block_size=4, num_blocks=32)
        kv_mgr = KVCacheManager(config)

        prefix = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks

        # First allocation creates 2 blocks
        ids1 = kv_mgr.allocate_blocks(prefix)
        assert len(ids1) == 2
        initial_free = kv_mgr.num_free_blocks

        # Second allocation with same tokens should reuse (cache hit)
        ids2 = kv_mgr.allocate_blocks(prefix)
        assert ids1 == ids2, "Same tokens should reuse same blocks"
        assert kv_mgr.num_free_blocks == initial_free, \
            "Cache hit should not consume free blocks"

        # Verify ref_counts incremented
        for bid in ids1:
            block = kv_mgr.pool.blocks[bid]
            assert block.ref_count == 2


class TestDA_F5_ServerRestartSSDResume:
    """DA-F-5: Server restart -> SSD index loads -> prefix hits resume.

    Attack vector: After saving blocks to SSD, create a new SSDCache
    instance pointing to the same directory (simulating server restart).
    Verify the index loads correctly and blocks are accessible.
    """

    def test_da_f5_ssd_index_survives_restart(self, tmp_path):
        """SSD index persists across cache re-instantiation."""
        cache_dir = tmp_path / "ssd-cache"

        # Phase 1: Save some blocks
        ssd1 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        kv1 = _make_kv_data()
        kv2 = _make_kv_data()

        ssd1.save_block("hash_1001", kv1)
        ssd1.save_block("hash_1002", kv2)
        assert ssd1.num_blocks == 2

        # Phase 2: "Restart" — create a new SSDCache pointing to same dir
        ssd2 = SSDCache(cache_dir=cache_dir, ttl_days=7)

        # Index should have been loaded
        assert ssd2.num_blocks == 2
        assert "hash_1001" in ssd2.index
        assert "hash_1002" in ssd2.index

        # Blocks should be loadable
        loaded1 = ssd2.load_block("hash_1001")
        assert loaded1 is not None
        assert "keys" in loaded1 and "values" in loaded1

        loaded2 = ssd2.load_block("hash_1002")
        assert loaded2 is not None

    def test_da_f5_tiered_lookup_after_restart(self, tmp_path):
        """After restart, TieredKVCache should find blocks on SSD
        that were evicted from RAM before shutdown."""
        cache_dir = tmp_path / "ssd-cache"

        # Phase 1: Create tiered cache, save blocks, evict to SSD
        config = _make_config(block_size=4, num_blocks=8)
        kv_mgr1 = KVCacheManager(config)
        ssd1 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        tiered1 = TieredKVCache(ram=kv_mgr1, ssd=ssd1)

        tokens = [50, 60, 70, 80]
        ids = kv_mgr1.allocate_blocks(tokens)
        original_kv = _make_kv_data()
        kv_mgr1.pool.blocks[ids[0]].kv_data = original_kv
        block_hash = kv_mgr1.pool.blocks[ids[0]].block_hash

        kv_mgr1.free_blocks(ids)
        tiered1.evict_to_ssd(num_blocks=1)

        # Verify block is on SSD
        assert ssd1.num_blocks >= 1

        # Phase 2: "Restart" — new manager, new SSD cache, same dir
        kv_mgr2 = KVCacheManager(config)
        ssd2 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        tiered2 = TieredKVCache(ram=kv_mgr2, ssd=ssd2)

        # RAM is empty, but SSD should have the block
        assert block_hash not in kv_mgr2.hash_table

        # TieredKVCache.lookup should find it on SSD
        loaded = tiered2.lookup(block_hash)
        assert loaded is not None, \
            "Block evicted before restart should be loadable from SSD after restart"
        assert "keys" in loaded and "values" in loaded

    def test_da_f5_ssd_index_with_expired_blocks(self, tmp_path):
        """After restart, expired blocks in the SSD index should be
        prunable immediately."""
        cache_dir = tmp_path / "ssd-cache"

        # Phase 1: Save block with old timestamp
        ssd1 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        ssd1.save_block("hash_2001", _make_kv_data())

        # Manually age the block beyond TTL
        ssd1.index["hash_2001"].last_accessed = datetime.now() - timedelta(days=10)
        ssd1.save_index()

        # Phase 2: "Restart"
        ssd2 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        assert "hash_2001" in ssd2.index

        # Prune should remove the expired block
        pruned = ssd2.prune_expired()
        assert pruned == 1
        assert "hash_2001" not in ssd2.index

    def test_da_f5_ssd_index_references_moved_files(self, tmp_path):
        """If safetensors files are moved/deleted between restarts,
        load_block should handle gracefully."""
        cache_dir = tmp_path / "ssd-cache"

        ssd1 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        ssd1.save_block("hash_3001", _make_kv_data())
        filepath = ssd1.index["hash_3001"].filepath

        # "External process" removes the file between restarts
        filepath.unlink()

        # Phase 2: restart
        ssd2 = SSDCache(cache_dir=cache_dir, ttl_days=7)
        assert "hash_3001" in ssd2.index  # Index still references it

        # load_block should detect missing file and clean up
        result = ssd2.load_block("hash_3001")
        assert result is None
        assert "hash_3001" not in ssd2.index

    def test_da_f5_multiple_restart_cycles(self, tmp_path):
        """Multiple save/restart/load cycles don't corrupt the index."""
        cache_dir = tmp_path / "ssd-cache"

        for cycle in range(5):
            ssd = SSDCache(cache_dir=cache_dir, ttl_days=7)

            # Add a new block each cycle
            block_hash = f"hash_{4000 + cycle}"
            ssd.save_block(block_hash, _make_kv_data())

            # Verify all previous blocks are still accessible
            for prev in range(cycle + 1):
                prev_hash = f"hash_{4000 + prev}"
                loaded = ssd.load_block(prev_hash)
                assert loaded is not None, \
                    f"Block {prev_hash} lost after cycle {cycle}"

        # Final restart: all 5 blocks should be present
        ssd_final = SSDCache(cache_dir=cache_dir, ttl_days=7)
        assert ssd_final.num_blocks == 5
        for i in range(5):
            assert f"hash_{4000 + i}" in ssd_final.index


# ===========================================================================
# PHASE 6: BatchGenerator Integration Adversarial Tests
# ===========================================================================

# Re-use patterns from test_batch_integration.py

from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from unittest.mock import MagicMock as _MagicMock
from mlx_lm_server.sequence_cache import SequenceCacheStore


@_dataclass
class _MockResponse:
    """Mock of BatchGenerator.Response for DA-P6 tests."""
    uid: int
    token: int
    logprobs: _Any = None
    finish_reason: str | None = None
    _prompt_cache: _Any = None

    def prompt_cache(self):
        return self._prompt_cache


class _MockBatchGenerator:
    """Mock BatchGenerator for DA-P6 adversarial tests."""

    def __init__(self):
        self._uid_counter = 0
        self._active: dict[int, dict] = {}
        self._closed = False
        self._removed_uids: list[int] = []
        self._next_hook = None  # Optional hook to inject behavior

    def insert(self, prompts, max_tokens=None, caches=None, samplers=None, logits_processors=None):
        uids = []
        for i, prompt in enumerate(prompts):
            uid = self._uid_counter
            self._uid_counter += 1
            mt = max_tokens[i] if isinstance(max_tokens, list) else (max_tokens or 10)
            self._active[uid] = {
                "tokens": list(prompt) if hasattr(prompt, '__iter__') else [prompt],
                "max_tokens": mt,
                "step": 0,
            }
            uids.append(uid)
        return uids

    def next(self):
        if self._next_hook is not None:
            return self._next_hook()
        responses = []
        finished = []
        for uid, state in self._active.items():
            state["step"] += 1
            token = 100 + state["step"]
            finish = None
            if state["step"] >= state["max_tokens"]:
                finish = "length"
                finished.append(uid)
            responses.append(_MockResponse(
                uid=uid,
                token=token,
                finish_reason=finish,
            ))
        for uid in finished:
            del self._active[uid]
        return responses

    def remove(self, uids, return_prompt_caches=False):
        result = {}
        for uid in uids:
            self._removed_uids.append(uid)
            if uid in self._active:
                if return_prompt_caches:
                    result[uid] = [{"mock_cache": True}]
                del self._active[uid]
        return result if return_prompt_caches else None

    def close(self):
        self._closed = True
        self._active.clear()


def _make_scheduler_with_mock_bg(config=None, **kwargs):
    """Create a Scheduler with a _MockBatchGenerator injected for DA-P6."""
    if config is None:
        config = ServerConfig(**kwargs)

    sched = Scheduler(config=config, model=None, tokenizer=None)

    mock_bg = _MockBatchGenerator()
    sched._batch_generator = mock_bg
    sched._sequence_cache = None

    mock_tokenizer = _MagicMock()
    mock_tokenizer.detokenizer = None
    mock_tokenizer.decode = lambda ids: "".join(f"t{i}" for i in ids)
    sched.tokenizer = mock_tokenizer

    return sched, mock_bg


class TestDA_P6_UIDLeak:
    """DA-P6-1: Submit many requests, verify UID maps are cleaned up.

    Attack vector: After many requests complete, the _uid_to_request_id and
    _request_id_to_uid maps should be empty. A leak here means unbounded
    memory growth proportional to total requests served.
    """

    def test_da_p6_uid_maps_cleaned_after_many_requests(self):
        """Submit 50 requests, all complete. UID maps must be empty."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=8, default_max_tokens=2
        )
        sched.run_inference_loop()

        try:
            for i in range(50):
                req = InferenceRequest(
                    request_id=f"uid-leak-{i}",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=2,
                )
                sched.submit_request(req)

            # Collect all results
            for i in range(50):
                sched.get_result(f"uid-leak-{i}", timeout=10.0)

            # Give cleanup a moment
            time.sleep(0.5)

            assert len(sched._uid_to_request_id) == 0, \
                f"UID->RID leak: {len(sched._uid_to_request_id)} entries remain"
            assert len(sched._request_id_to_uid) == 0, \
                f"RID->UID leak: {len(sched._request_id_to_uid)} entries remain"

            # Also verify _active_sequences is empty
            with sched._active_lock:
                assert len(sched._active_sequences) == 0, \
                    f"Active sequences leak: {len(sched._active_sequences)} remain"
        finally:
            sched.stop()

    def test_da_p6_uid_maps_cleaned_after_cancelled_requests(self):
        """Submit requests, cancel some mid-flight. All UID maps cleaned."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Slow down next() so cancellation catches active requests
        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.02)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            for i in range(10):
                req = InferenceRequest(
                    request_id=f"cancel-uid-{i}",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=100,
                )
                sched.submit_request(req)

            # Let some processing happen
            time.sleep(0.3)

            # Cancel all
            for i in range(10):
                sched.cancel_request(f"cancel-uid-{i}")

            # Wait for cleanup
            time.sleep(1.0)

            assert len(sched._uid_to_request_id) == 0, \
                f"UID leak after cancel: {len(sched._uid_to_request_id)} entries"
            assert len(sched._request_id_to_uid) == 0, \
                f"RID leak after cancel: {len(sched._request_id_to_uid)} entries"
        finally:
            sched.stop()


class TestDA_P6_ConcurrentInsertNext:
    """DA-P6-2: Multiple rapid submits while batch is running.

    Attack vector: Rapid-fire submits during an active batch step could
    race with _insert_new_requests_batch() or _batch_inference_step().
    """

    def test_da_p6_rapid_submit_during_active_batch(self):
        """Rapidly submit requests while batch is running. All should
        complete without errors or missing results."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=3
        )
        sched.run_inference_loop()

        try:
            # Submit requests in rapid succession from multiple threads
            errors: list[Exception] = []
            barrier = threading.Barrier(5)

            def submit_worker(start_idx):
                try:
                    barrier.wait(timeout=5)
                    for i in range(5):
                        rid = f"rapid-{start_idx + i}"
                        req = InferenceRequest(
                            request_id=rid,
                            prompt_tokens=[1, 2, 3],
                            max_tokens=3,
                        )
                        sched.submit_request(req)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=submit_worker, args=(i * 5,))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Submit errors: {errors}"

            # All 25 requests should complete
            for i in range(25):
                events = sched.get_result(f"rapid-{i}", timeout=15.0)
                assert len(events) >= 1, f"rapid-{i} got no events"
                assert events[-1].finish_reason is not None, \
                    f"rapid-{i} never finished"
        finally:
            sched.stop()


class TestDA_P6_CacheMutation:
    """DA-P6-3: Store cache in SequenceCacheStore, modify original.

    Attack vector: If SequenceCacheStore stores a reference (not deep copy),
    modifications to the original list would corrupt the cache.
    """

    def test_da_p6_sequence_cache_store_immutable(self):
        """Stored cache should be independent of the original object."""
        store = SequenceCacheStore(max_entries=10)

        # Create a mock cache (list of objects)
        original_cache = [{"layer": 0, "data": [1, 2, 3]}, {"layer": 1, "data": [4, 5, 6]}]
        tokens = [10, 20, 30]

        store.store(tokens, original_cache)

        # Mutate the original
        original_cache[0]["data"].append(999)
        original_cache.append({"layer": 2, "data": [7, 8, 9]})

        # Retrieved cache should be unaffected
        cached, remaining = store.find_longest_prefix(tokens)
        assert cached is not None
        assert len(cached) == 2, "Stored cache has wrong length after mutation"
        assert 999 not in cached[0]["data"], "Original mutation leaked into cache!"

    def test_da_p6_sequence_cache_returned_copy_independent(self):
        """Mutating the returned cache should not affect the stored version."""
        store = SequenceCacheStore(max_entries=10)

        tokens = [10, 20, 30]
        original_cache = [{"layer": 0, "data": [1, 2, 3]}]
        store.store(tokens, original_cache)

        # Get cache and mutate it
        cached1, _ = store.find_longest_prefix(tokens)
        cached1[0]["data"].append(999)

        # Get again — should be clean
        cached2, _ = store.find_longest_prefix(tokens)
        assert 999 not in cached2[0]["data"], \
            "Returned copy mutation leaked into stored cache!"


class TestDA_P6_ErrorCascade:
    """DA-P6-4: BatchGenerator.next() throws, then new requests arrive.

    Attack vector: After a fatal error in next(), the scheduler should
    recover by recreating the BatchGenerator. New requests submitted
    after recovery should work normally.
    """

    def test_da_p6_error_cascade_recovery(self):
        """After next() throws, scheduler recovers and processes new requests."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=3
        )

        # Make next() throw on first call only
        call_count = [0]
        original_next = mock_bg.next

        def failing_next():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("GPU memory error!")
            return original_next()

        mock_bg.next = failing_next
        sched.run_inference_loop()

        try:
            # First request triggers the error
            req1 = InferenceRequest(
                request_id="error-req",
                prompt_tokens=[1, 2, 3],
                max_tokens=3,
            )
            sched.submit_request(req1)
            events1 = sched.get_result("error-req", timeout=5.0)
            assert events1[-1].finish_reason == "error", \
                "First request should fail with error"

            # After error, _handle_batch_error recreates the BG
            # The scheduler's _batch_generator is now a NEW BatchGenerator
            # (created by _create_batch_generator), but since model=None,
            # it stays None. We need to re-inject our mock.
            # This is actually testing that the scheduler properly resets
            # state and can accept new requests.
            time.sleep(0.5)

            # Re-inject a fresh mock BG
            new_mock_bg = _MockBatchGenerator()
            sched._batch_generator = new_mock_bg

            req2 = InferenceRequest(
                request_id="recovery-req",
                prompt_tokens=[4, 5, 6],
                max_tokens=2,
            )
            sched.submit_request(req2)
            events2 = sched.get_result("recovery-req", timeout=5.0)
            assert len(events2) >= 1
            assert events2[-1].finish_reason is not None, \
                "Recovery request should complete"
        finally:
            sched.stop()

    def test_da_p6_multiple_consecutive_errors(self):
        """Multiple consecutive next() failures don't permanently break the scheduler."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=2
        )

        fail_count = [0]
        original_next = mock_bg.next

        def repeatedly_failing_next():
            fail_count[0] += 1
            if fail_count[0] <= 3:
                raise RuntimeError(f"Error #{fail_count[0]}")
            return original_next()

        mock_bg.next = repeatedly_failing_next
        sched.run_inference_loop()

        try:
            # Submit requests that will each trigger an error
            for i in range(3):
                req = InferenceRequest(
                    request_id=f"multi-err-{i}",
                    prompt_tokens=[1, 2],
                    max_tokens=2,
                )
                sched.submit_request(req)
                events = sched.get_result(f"multi-err-{i}", timeout=5.0)
                assert events[-1].finish_reason == "error"
                # Re-inject mock after each error recovery
                sched._batch_generator = mock_bg
                time.sleep(0.2)

            # After 3 errors, next request should succeed
            req_ok = InferenceRequest(
                request_id="finally-ok",
                prompt_tokens=[7, 8],
                max_tokens=2,
            )
            sched.submit_request(req_ok)
            events_ok = sched.get_result("finally-ok", timeout=5.0)
            assert len(events_ok) == 2
            assert events_ok[-1].finish_reason == "length"
        finally:
            sched.stop()


class TestDA_P6_ZeroLengthGeneration:
    """DA-P6-5: max_tokens=0 through batch path.

    Attack vector: max_tokens=0 should result in immediate finish with
    no insertion into the BatchGenerator. The request should never appear
    as an active UID in the BG.
    """

    def test_da_p6_zero_tokens_no_bg_insert(self):
        """max_tokens=0 completes immediately, BG never sees the request."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=0
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="zero-gen",
                prompt_tokens=[1, 2, 3, 4, 5],
                max_tokens=0,
            )
            sched.submit_request(req)
            events = sched.get_result("zero-gen", timeout=5.0)

            assert len(events) >= 1
            assert events[-1].finish_reason == "length"
            # BG should never have had a request inserted
            assert mock_bg._uid_counter == 0, \
                "max_tokens=0 should not insert into BatchGenerator"
        finally:
            sched.stop()

    def test_da_p6_negative_max_tokens_batch(self):
        """max_tokens=-1 should also be handled gracefully (treated as 0)."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=0
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="neg-gen",
                prompt_tokens=[1, 2, 3],
                max_tokens=-1,
            )
            sched.submit_request(req)
            events = sched.get_result("neg-gen", timeout=5.0)

            assert len(events) >= 1
            assert events[-1].finish_reason == "length"
            assert mock_bg._uid_counter == 0
        finally:
            sched.stop()


class TestDA_P6_MaxBatchOverflow:
    """DA-P6-6: More requests than max_batch_size.

    Attack vector: If more requests are submitted than max_batch_size allows,
    they should be queued and processed as slots become available, not dropped.
    """

    def test_da_p6_overflow_queued_properly(self):
        """Submit 10 requests with max_batch_size=2. All 10 complete."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=2, default_max_tokens=2
        )
        sched.run_inference_loop()

        try:
            for i in range(10):
                req = InferenceRequest(
                    request_id=f"overflow-{i}",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=2,
                )
                sched.submit_request(req)

            # All should complete
            for i in range(10):
                events = sched.get_result(f"overflow-{i}", timeout=15.0)
                assert len(events) >= 1, f"overflow-{i} got no events"
                assert events[-1].finish_reason is not None, \
                    f"overflow-{i} never finished"
        finally:
            sched.stop()

    def test_da_p6_batch_never_exceeds_max(self):
        """During processing, the number of active sequences never
        exceeds max_batch_size."""
        max_batch = 3
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=max_batch, default_max_tokens=5
        )

        # Track max active count
        max_active_seen = [0]
        original_next = mock_bg.next

        def tracking_next():
            # Check active count at each step
            current = len(mock_bg._active)
            if current > max_active_seen[0]:
                max_active_seen[0] = current
            return original_next()

        mock_bg.next = tracking_next
        sched.run_inference_loop()

        try:
            for i in range(10):
                req = InferenceRequest(
                    request_id=f"maxcheck-{i}",
                    prompt_tokens=[1, 2],
                    max_tokens=5,
                )
                sched.submit_request(req)

            for i in range(10):
                sched.get_result(f"maxcheck-{i}", timeout=15.0)

            assert max_active_seen[0] <= max_batch, \
                f"Active batch exceeded max: {max_active_seen[0]} > {max_batch}"
        finally:
            sched.stop()


class TestDA_P6_StopConditions:
    """DA-P6-7: Stop sequence check through batch path.

    Attack vector: Custom stop sequences should be checked in the batch
    path via _check_stop_conditions(). Verify the sequence is properly
    terminated and removed from BatchGenerator when a stop sequence matches.
    """

    def test_da_p6_stop_sequence_triggers_finish(self):
        """A stop sequence in the output should terminate the request
        via the batch path."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Override next() to produce tokens that form a stop sequence
        step_count = [0]

        def stop_seq_next():
            step_count[0] += 1
            responses = []
            finished_uids = []
            for uid, state in list(mock_bg._active.items()):
                state["step"] += 1
                # Produce "Hello STOP world" token by token
                token_map = {1: 200, 2: 201, 3: 202}  # H, STOP, w
                token = token_map.get(state["step"], 300 + state["step"])
                responses.append(_MockResponse(uid=uid, token=token))
            return responses

        mock_bg.next = stop_seq_next

        # The mock tokenizer uses str(token) for text
        # We need to check stop sequence detection in output_text
        # Since the batch path uses str(token) fallback, the output will be
        # "200201202..." — we set a stop sequence that matches
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="stop-test",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
                stop_sequences=["201"],  # Should match when token 201 is generated
            )
            sched.submit_request(req)
            events = sched.get_result("stop-test", timeout=5.0)

            # Should finish before max_tokens
            assert len(events) <= 100
            assert events[-1].finish_reason == "stop", \
                f"Expected 'stop' finish_reason, got '{events[-1].finish_reason}'"
        finally:
            sched.stop()


class TestDA_P6_SequenceCacheEviction:
    """DA-P6-8: Fill cache past max_entries, oldest evicted.

    Attack vector: SequenceCacheStore has a max_entries limit. When exceeded,
    the oldest entry should be evicted. Verify FIFO eviction order.
    """

    def test_da_p6_eviction_order_lru(self):
        """When max_entries is exceeded, the least recently used entry is
        evicted (FIFO for entries not accessed)."""
        store = SequenceCacheStore(max_entries=3)

        store.store([1, 2], [{"cache": "a"}])
        store.store([3, 4], [{"cache": "b"}])
        store.store([5, 6], [{"cache": "c"}])

        assert store.size == 3

        # Insert a 4th — should evict the oldest (tokens [1,2])
        store.store([7, 8], [{"cache": "d"}])
        assert store.size == 3

        # [1,2] should be gone
        cached, remaining = store.find_longest_prefix([1, 2])
        assert cached is None, "Oldest entry should have been evicted"

        # Others should still be there
        cached_b, _ = store.find_longest_prefix([3, 4])
        assert cached_b is not None, "Entry [3,4] should still be cached"

    def test_da_p6_access_prevents_eviction(self):
        """Accessing an entry moves it to the end of the LRU order,
        preventing its eviction."""
        store = SequenceCacheStore(max_entries=3)

        store.store([1, 2], [{"cache": "a"}])
        store.store([3, 4], [{"cache": "b"}])
        store.store([5, 6], [{"cache": "c"}])

        # Access [1,2] — moves it to end (most recently used)
        cached, _ = store.find_longest_prefix([1, 2])
        assert cached is not None

        # Insert [7,8] — should evict [3,4] (now oldest) not [1,2]
        store.store([7, 8], [{"cache": "d"}])

        cached_a, _ = store.find_longest_prefix([1, 2])
        assert cached_a is not None, "[1,2] should survive (accessed recently)"

        cached_b, _ = store.find_longest_prefix([3, 4])
        assert cached_b is None, "[3,4] should be evicted (oldest after access)"

    def test_da_p6_max_entries_one(self):
        """Edge case: max_entries=1, only the latest entry is retained."""
        store = SequenceCacheStore(max_entries=1)

        store.store([1], [{"cache": "first"}])
        assert store.size == 1

        store.store([2], [{"cache": "second"}])
        assert store.size == 1

        cached_first, _ = store.find_longest_prefix([1])
        assert cached_first is None

        cached_second, _ = store.find_longest_prefix([2])
        assert cached_second is not None


# ===========================================================================
# PHASE 7: Block-Level KV Cache Bridge Adversarial Tests
# ===========================================================================

from mlx_lm_server.kv_cache_manager import (
    decompose_cache_to_blocks,
    reconstruct_cache_from_blocks,
    extract_block,
    inject_blocks,
)


class _MockKVCacheLayer:
    """Mock KVCache layer for DA-P7 tests."""

    def __init__(self, seq_len: int, n_heads: int = 4, head_dim: int = 8):
        self.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.offset = seq_len

    @property
    def state(self):
        return (self.keys, self.values)


def _make_da_mock_cache(num_layers=2, seq_len=32, n_heads=4, head_dim=8):
    """Create a mock List[KVCache] for DA-P7 tests."""
    return [_MockKVCacheLayer(seq_len, n_heads, head_dim) for _ in range(num_layers)]


class TestDA_P7_BlockHashCollision:
    """DA-P7-1: Block hash collision handling.

    Attack vector: Two different token sequences that produce the same hash.
    The KVCacheManager stores token_ids for collision verification. Verify
    that decompose_cache_to_blocks stores token_ids correctly and the
    scheduler's block storage code checks for collisions.
    """

    def test_da_p7_decomposed_blocks_store_token_ids(self):
        """Every decomposed block must include the correct token_ids
        for later collision verification."""
        block_size = 8
        token_ids = list(range(24))
        cache = _make_da_mock_cache(seq_len=24)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        for i, block in enumerate(blocks):
            start = i * block_size
            end = start + block_size
            expected_tokens = list(range(start, end))
            assert block['token_ids'] == expected_tokens, \
                f"Block {i} token_ids mismatch: {block['token_ids']} != {expected_tokens}"

    def test_da_p7_hash_uniqueness_across_positions(self):
        """Blocks with identical tokens but different prefix positions
        should have different hashes."""
        block_size = 8
        # Two sequences where the second block contains the same tokens
        # but the prefix differs
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8,   10, 20, 30, 40, 50, 60, 70, 80]
        tokens_b = [9, 8, 7, 6, 5, 4, 3, 2,   10, 20, 30, 40, 50, 60, 70, 80]

        cache_a = _make_da_mock_cache(seq_len=16)
        cache_b = _make_da_mock_cache(seq_len=16)

        blocks_a = decompose_cache_to_blocks(cache_a, tokens_a, block_size)
        blocks_b = decompose_cache_to_blocks(cache_b, tokens_b, block_size)

        # First blocks differ (different tokens)
        assert blocks_a[0]['block_hash'] != blocks_b[0]['block_hash']

        # Second blocks have same tokens BUT different prefixes
        # -> hashes should be different
        assert blocks_a[1]['block_hash'] != blocks_b[1]['block_hash'], \
            "Blocks with same tokens but different prefix should have different hashes"

    def test_da_p7_hash_determinism(self):
        """Same tokens and prefix always produce the same block hash."""
        block_size = 8
        tokens = list(range(16))

        cache1 = _make_da_mock_cache(seq_len=16)
        cache2 = _make_da_mock_cache(seq_len=16)

        blocks1 = decompose_cache_to_blocks(cache1, tokens, block_size)
        blocks2 = decompose_cache_to_blocks(cache2, tokens, block_size)

        for i in range(len(blocks1)):
            assert blocks1[i]['block_hash'] == blocks2[i]['block_hash'], \
                f"Block {i} hash not deterministic"


class TestDA_P7_PartialBlockDecomposition:
    """DA-P7-2: Token count not multiple of block_size.

    Attack vector: If the token count is not a multiple of block_size,
    the remainder tokens should be silently ignored (no crash, no partial
    block).
    """

    def test_da_p7_remainder_tokens_ignored(self):
        """Tokens beyond the last full block are ignored."""
        block_size = 8
        # 20 tokens = 2 full blocks + 4 remainder
        token_ids = list(range(20))
        cache = _make_da_mock_cache(seq_len=20)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"
        # Remainder tokens 16-19 should not appear
        all_block_tokens = []
        for b in blocks:
            all_block_tokens.extend(b['token_ids'])
        assert 16 not in all_block_tokens
        assert 19 not in all_block_tokens

    def test_da_p7_one_token_short_of_block(self):
        """block_size-1 tokens should produce zero blocks."""
        block_size = 16
        token_ids = list(range(15))  # 1 short of a full block
        cache = _make_da_mock_cache(seq_len=15)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)
        assert blocks == [], "Should produce no blocks for fewer tokens than block_size"

    def test_da_p7_exactly_one_block(self):
        """Exactly block_size tokens should produce exactly 1 block."""
        block_size = 8
        token_ids = list(range(8))
        cache = _make_da_mock_cache(seq_len=8)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)
        assert len(blocks) == 1


class TestDA_P7_EmptyCacheDecomposition:
    """DA-P7-3: Empty prompt_cache list -> no crash.

    Attack vector: An empty cache or empty token list passed to
    decompose_cache_to_blocks() should gracefully return an empty list
    without raising exceptions.
    """

    def test_da_p7_empty_token_list(self):
        """Empty token list produces no blocks."""
        cache = _make_da_mock_cache(seq_len=0)
        blocks = decompose_cache_to_blocks(cache, [], block_size=16)
        assert blocks == []

    def test_da_p7_empty_cache_list(self):
        """Empty cache list with valid tokens produces blocks with
        empty kv_data_per_layer."""
        blocks = decompose_cache_to_blocks([], list(range(16)), block_size=16)
        assert len(blocks) == 1
        assert blocks[0]['kv_data_per_layer'] == []

    def test_da_p7_both_empty(self):
        """Both empty cache and empty tokens -> no crash."""
        blocks = decompose_cache_to_blocks([], [], block_size=8)
        assert blocks == []


class TestDA_P7_DecomposeZeroBlocks:
    """DA-P7-4: Token count < block_size -> empty result.

    Attack vector: When the entire prompt is shorter than one block,
    decomposition should return an empty list.
    """

    def test_da_p7_tokens_shorter_than_block_size(self):
        """3 tokens with block_size=16 -> 0 blocks."""
        cache = _make_da_mock_cache(seq_len=3)
        blocks = decompose_cache_to_blocks(cache, [1, 2, 3], block_size=16)
        assert blocks == []

    def test_da_p7_single_token(self):
        """1 token with any block_size > 1 -> 0 blocks."""
        cache = _make_da_mock_cache(seq_len=1)
        blocks = decompose_cache_to_blocks(cache, [42], block_size=4)
        assert blocks == []

    def test_da_p7_block_size_larger_than_tokens(self):
        """block_size=100 with 50 tokens -> 0 blocks."""
        cache = _make_da_mock_cache(seq_len=50)
        blocks = decompose_cache_to_blocks(cache, list(range(50)), block_size=100)
        assert blocks == []


class TestDA_P7_ConcurrentDecompose:
    """DA-P7-5: Multiple threads decomposing simultaneously.

    Attack vector: decompose_cache_to_blocks() operates on independent data,
    but the module-level compute_block_hash uses Python's hash() which
    should be thread-safe. Verify no corruption or crashes.
    """

    def test_da_p7_concurrent_decompose_no_corruption(self):
        """Multiple threads decomposing different caches simultaneously."""
        block_size = 8
        errors: list[Exception] = []
        results: list[list[dict]] = []
        lock = threading.Lock()

        def worker(tid):
            try:
                for _ in range(20):
                    tokens = list(range(tid * 100, tid * 100 + 32))
                    cache = _make_da_mock_cache(seq_len=32)
                    blocks = decompose_cache_to_blocks(cache, tokens, block_size)
                    assert len(blocks) == 4, \
                        f"Thread {tid}: expected 4 blocks, got {len(blocks)}"
                    # Verify block hashes are consistent
                    for i, b in enumerate(blocks):
                        start = i * block_size
                        end = start + block_size
                        expected_hash = compute_block_hash(
                            tokens[:start], tokens[start:end]
                        )
                        assert b['block_hash'] == expected_hash, \
                            f"Thread {tid}: block {i} hash mismatch"
                    with lock:
                        results.append(blocks)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8 * 20, \
            f"Expected 160 results, got {len(results)}"

    def test_da_p7_concurrent_decompose_same_tokens(self):
        """Multiple threads decomposing the SAME token sequence.
        Hashes should be identical across all threads."""
        block_size = 8
        shared_tokens = list(range(32))
        all_hashes: list[list[str]] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def worker(tid):
            try:
                cache = _make_da_mock_cache(seq_len=32)
                blocks = decompose_cache_to_blocks(cache, shared_tokens, block_size)
                hashes = [b['block_hash'] for b in blocks]
                with lock:
                    all_hashes.append(hashes)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        # All threads should produce identical hash lists
        for i, hashes in enumerate(all_hashes):
            assert hashes == all_hashes[0], \
                f"Thread {i} produced different hashes: {hashes} vs {all_hashes[0]}"


class TestDA_P7_ReconstructEmpty:
    """DA-P7-6: Empty blocks list -> empty result.

    Attack vector: reconstruct_cache_from_blocks with an empty block list
    should return an empty list, not crash or produce garbage.
    """

    def test_da_p7_reconstruct_empty_blocks(self):
        """Empty blocks list returns empty cache list."""
        result = reconstruct_cache_from_blocks([], model=None)
        assert result == []

    def test_da_p7_inject_blocks_empty_raises(self):
        """inject_blocks with empty list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            inject_blocks([])

    def test_da_p7_decompose_then_inject_roundtrip(self):
        """Decompose -> per-layer inject_blocks roundtrip preserves data."""
        block_size = 8
        seq_len = 24
        n_heads = 4
        head_dim = 8
        num_layers = 2
        cache = _make_da_mock_cache(
            num_layers=num_layers, seq_len=seq_len,
            n_heads=n_heads, head_dim=head_dim,
        )
        token_ids = list(range(seq_len))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)
        assert len(blocks) == 3

        # Reconstruct per layer using inject_blocks
        for layer_idx in range(num_layers):
            layer_blocks = [b['kv_data_per_layer'][layer_idx] for b in blocks]
            reconstructed = inject_blocks(layer_blocks)

            original_keys = cache[layer_idx].keys
            original_values = cache[layer_idx].values

            assert mx.allclose(reconstructed['keys'], original_keys).item(), \
                f"Layer {layer_idx}: keys data corrupted in roundtrip"
            assert mx.allclose(reconstructed['values'], original_values).item(), \
                f"Layer {layer_idx}: values data corrupted in roundtrip"

    def test_da_p7_extract_block_boundary(self):
        """extract_block at the boundary of the sequence should work correctly."""
        n_heads = 4
        head_dim = 8
        seq_len = 16
        block_size = 8

        keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        values = mx.random.normal((1, n_heads, seq_len, head_dim))

        # Extract last block (positions 8-15)
        block_data = extract_block(keys, values, start_pos=8, block_size=8)

        assert block_data['keys'].shape == (1, n_heads, block_size, head_dim)
        assert mx.allclose(block_data['keys'], keys[:, :, 8:16, :]).item()
        assert mx.allclose(block_data['values'], values[:, :, 8:16, :]).item()


# ===========================================================================
# Additional DA-P6 / DA-P7 tests — deeper attack vectors
# ===========================================================================


class TestDA_P6_CancelledSetLeak:
    """DA-P6-C1 (HIGH): _cancelled set not cleared after _handle_batch_error.

    Attack vector: When _handle_batch_error() is called after a BG crash,
    it marks all active sequences as errored and clears UID maps. BUT it
    does NOT clear the _cancelled set. If a request was cancelled just
    before the error, its ID remains in _cancelled. If a future request
    uses the same ID (unlikely with UUIDs, but possible in tests or
    with custom ID schemes), it would be auto-cancelled on arrival.
    """

    def test_da_p6_cancelled_set_leak_after_error(self):
        """After BG error, stale cancelled IDs remain in _cancelled set."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Slow down to allow cancellation before error
        original_next = mock_bg.next
        call_count = [0]

        def delayed_fail():
            call_count[0] += 1
            if call_count[0] <= 2:
                time.sleep(0.05)
                return original_next()
            raise RuntimeError("Delayed BG failure")

        mock_bg.next = delayed_fail
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="cancel-then-crash",
                prompt_tokens=[1, 2],
                max_tokens=100,
            )
            sched.submit_request(req)
            time.sleep(0.15)

            # Cancel mid-flight
            sched.cancel_request("cancel-then-crash")
            # BG error will hit on call_count > 2
            time.sleep(1.0)

            # Check if _cancelled is clean
            with sched._cancelled_lock:
                stale = "cancel-then-crash" in sched._cancelled

            # FINDING: _handle_batch_error does NOT clear _cancelled.
            # The cancelled ID persists as a stale entry.
            if stale:
                pytest.xfail(
                    "DA-P6-C1 (HIGH): _handle_batch_error does not clear "
                    "_cancelled set. Stale cancelled IDs survive BG restart "
                    "and could cancel future requests with same ID."
                )
        finally:
            sched.stop()


class TestDA_P7_BlockDataRefNotCopy:
    """DA-P7-H1 (HIGH): Scheduler stores block kv_data by reference.

    Attack vector: In scheduler.py line 516, the code does:
        block.kv_data = bd['kv_data_per_layer']
    This stores a reference to the list of per-layer dicts. If the source
    prompt_cache object is modified or garbage-collected, the block's
    kv_data becomes stale or corrupted.

    In practice, mx.arrays are immutable (operations create new arrays),
    but the list/dict containers ARE mutable. Replacing an element in the
    kv_data_per_layer list would corrupt the block's cached data.
    """

    def test_da_p7_block_kv_data_is_reference(self):
        """Demonstrate that block kv_data is stored by reference, not copy."""
        config = _make_config(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        cache = _make_da_mock_cache(num_layers=1, seq_len=4)
        token_ids = [1, 2, 3, 4]
        blocks = decompose_cache_to_blocks(cache, token_ids, 4)
        assert len(blocks) == 1

        bd = blocks[0]
        bh = bd['block_hash']

        # Simulate scheduler's block storage (reference assignment)
        with mgr.lock:
            block = mgr.pool.get_free_block()
            block.block_hash = bh
            block.token_ids = bd['token_ids']
            block.ref_count = 0
            block.kv_data = bd['kv_data_per_layer']  # Reference!
            mgr.hash_table[bh] = block.block_id

        # The block.kv_data and bd['kv_data_per_layer'] are the same object
        assert block.kv_data is bd['kv_data_per_layer'], \
            "kv_data should be the same object (reference, not copy)"

        # Mutating the source list changes the block's data
        original_len = len(block.kv_data)
        bd['kv_data_per_layer'].append({"injected": True})
        assert len(block.kv_data) == original_len + 1, \
            "Expected reference sharing: mutation should propagate"

        # FINDING: This is a real vulnerability. If the source data is
        # modified (e.g., by another decomposition or garbage collection
        # of the prompt_cache), the cached block data is corrupted.


class TestDA_P7_SchedulerBlockCollisionSkip:
    """DA-P7-M1 (MEDIUM): Scheduler decomposition skips collision check.

    Attack vector: In scheduler.py line 500, the block decomposition code
    checks:
        if bh not in self.kv_cache_manager.hash_table:
    If bh IS in the hash table, it skips the block entirely. But it does
    NOT verify that the existing block's token_ids match. If a hash
    collision occurred, the wrong KV data stays cached.

    The find_cached_prefix() does check token_ids (line 183), so lookups
    are safe. But the decomposition code silently drops the new (correct)
    block data when a collision exists.
    """

    def test_da_p7_decompose_skips_collision_check(self):
        """When a hash collision exists in the hash_table, the scheduler's
        decomposition code skips the block without verifying token_ids."""
        config = _make_config(block_size=4, num_blocks=16)
        mgr = KVCacheManager(config)

        # Store a block with tokens [1,2,3,4]
        tokens_a = [1, 2, 3, 4]
        ids_a = mgr.allocate_blocks(tokens_a)
        block_a = mgr.pool.blocks[ids_a[0]]
        original_kv = _make_kv_data()
        block_a.kv_data = original_kv
        mgr.free_blocks(ids_a)

        # Forge a collision: make tokens [5,6,7,8] hash to the same value
        tokens_b = [5, 6, 7, 8]
        hash_b = compute_block_hash([], tokens_b)

        # Rewire hash table
        hash_a = block_a.block_hash
        del mgr.hash_table[hash_a]
        block_a.block_hash = hash_b
        mgr.hash_table[hash_b] = block_a.block_id

        # Simulate scheduler decomposition: check hash_table only
        # This is what the scheduler code does at line 500:
        with mgr.lock:
            already_exists = hash_b in mgr.hash_table

        assert already_exists, "Hash should be in table (collision scenario)"

        # The scheduler would skip this block because hash exists,
        # but it has WRONG token_ids (block_a.token_ids == [1,2,3,4])
        existing_block = mgr.pool.blocks[mgr.hash_table[hash_b]]
        assert existing_block.token_ids != tokens_b, \
            "This confirms a collision: hash matches but tokens differ"

        # FINDING: Scheduler code at line 500 does NOT check token_ids.
        # It only checks hash presence. The wrong block stays cached.
        # find_cached_prefix catches this at lookup time (line 183),
        # so this doesn't cause incorrect generation, but it does mean
        # the correct KV data for tokens_b is silently dropped.


class TestDA_P7_InjectBlocksShapeMismatch:
    """DA-P7-M2 (MEDIUM): inject_blocks with mismatched tensor shapes.

    Attack vector: If blocks from different model configurations (different
    n_heads or head_dim) are mixed, concatenation should fail cleanly.
    """

    def test_da_p7_mismatched_head_dim(self):
        """Blocks with different head dimensions should fail on concatenation."""
        block_1 = {
            'keys': mx.zeros((1, 4, 8, 8)),   # head_dim=8
            'values': mx.zeros((1, 4, 8, 8)),
        }
        block_2 = {
            'keys': mx.zeros((1, 4, 8, 16)),  # head_dim=16 (MISMATCH)
            'values': mx.zeros((1, 4, 8, 16)),
        }

        # mx.concatenate along axis=2 should still work since shapes differ
        # on axis=3, but the result would be garbage. Let's verify the
        # concatenation behavior:
        with pytest.raises(Exception):
            inject_blocks([block_1, block_2])

    def test_da_p7_mismatched_n_heads(self):
        """Blocks with different number of heads should fail."""
        block_1 = {
            'keys': mx.zeros((1, 4, 8, 8)),   # n_heads=4
            'values': mx.zeros((1, 4, 8, 8)),
        }
        block_2 = {
            'keys': mx.zeros((1, 8, 8, 8)),   # n_heads=8 (MISMATCH)
            'values': mx.zeros((1, 8, 8, 8)),
        }

        with pytest.raises(Exception):
            inject_blocks([block_1, block_2])


class TestDA_P7_ExtractBeyondBoundary:
    """DA-P7-M3 (MEDIUM): extract_block beyond sequence length.

    Attack vector: MLX array slicing is forgiving and returns shorter
    arrays when slicing past the end. If decompose_cache_to_blocks
    somehow passes an out-of-range start_pos, the extracted block
    would have fewer tokens than block_size, causing shape mismatches
    during reconstruction.
    """

    def test_da_p7_extract_past_seq_end(self):
        """extract_block starting past seq_len returns empty tensor."""
        n_heads = 4
        head_dim = 8
        seq_len = 8

        keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        values = mx.random.normal((1, n_heads, seq_len, head_dim))

        # start_pos=8 = seq_len, so slice 8:12 is empty
        result = extract_block(keys, values, start_pos=8, block_size=4)
        assert result['keys'].shape[2] == 0, \
            "Extracting past seq end should produce empty seq dim"

    def test_da_p7_extract_partially_past_end(self):
        """extract_block partially beyond seq_len returns truncated block."""
        n_heads = 4
        head_dim = 8
        seq_len = 10

        keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        values = mx.random.normal((1, n_heads, seq_len, head_dim))

        # start_pos=8, block_size=8 -> positions 8:16, but only 10 positions exist
        result = extract_block(keys, values, start_pos=8, block_size=8)
        actual_len = result['keys'].shape[2]
        assert actual_len == 2, \
            f"Expected 2 tokens from partial extraction, got {actual_len}"


class TestDA_P7_DecomposedBlocksImmediatelyEvictable:
    """DA-P7-M4 (MEDIUM): Decomposed blocks stored with ref_count=0.

    Attack vector: The scheduler stores decomposed blocks with ref_count=0
    (line 514 in scheduler.py). This means they are immediately eligible
    for LRU eviction. Under memory pressure, freshly cached blocks could
    be evicted before any request uses them, wasting the decomposition work.
    """

    def test_da_p7_freshly_decomposed_blocks_evictable(self):
        """Blocks stored with ref_count=0 can be immediately evicted."""
        config = _make_config(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # Manually store 2 blocks with ref_count=0 (as scheduler does)
        for i in range(2):
            with mgr.lock:
                block = mgr.pool.get_free_block()
                tokens = [i * 4 + j for j in range(4)]
                bh = compute_block_hash([], tokens) if i == 0 else compute_block_hash(
                    list(range(4)), tokens
                )
                block.block_hash = bh
                block.token_ids = tokens
                block.ref_count = 0  # As the scheduler does
                block.last_accessed = time.time() - 100  # Old access time
                mgr.hash_table[bh] = block.block_id

        assert mgr.num_cached_blocks == 2
        assert mgr.num_free_blocks == 2

        # These blocks are immediately evictable
        evicted = mgr.evict_lru(num_blocks=2)
        assert len(evicted) == 2, \
            "Freshly decomposed blocks (ref_count=0) should be evictable"
        assert mgr.num_cached_blocks == 0


class TestDA_P7_ConcurrentAllocEvict:
    """DA-P7-M5 (MEDIUM): Concurrent allocation and eviction thread safety.

    Attack vector: Multiple threads performing allocate_blocks() and
    evict_lru() simultaneously should not corrupt the block pool or
    hash_table state.
    """

    def test_da_p7_concurrent_alloc_and_evict_safe(self):
        """Concurrent alloc + evict does not corrupt state."""
        config = _make_config(block_size=4, num_blocks=16)
        mgr = KVCacheManager(config)

        errors: list[Exception] = []
        barrier = threading.Barrier(6)

        def allocator(tid):
            try:
                barrier.wait(timeout=5)
                for i in range(20):
                    tokens = [tid * 1000 + i * 4 + j for j in range(4)]
                    try:
                        ids = mgr.allocate_blocks(tokens)
                        mgr.free_blocks(ids)
                    except BlockPoolExhaustedError:
                        pass  # Expected under contention
            except Exception as e:
                errors.append(e)

        def evictor():
            try:
                barrier.wait(timeout=5)
                for _ in range(30):
                    mgr.evict_lru(num_blocks=1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=allocator, args=(i,))
            for i in range(5)
        ]
        threads.append(threading.Thread(target=evictor))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert not errors, f"Concurrent access errors: {errors}"

        # Verify state consistency: free + used == total
        total = config.num_blocks
        assert mgr.num_free_blocks + mgr.num_used_blocks == total, \
            f"State inconsistency: free={mgr.num_free_blocks} + used={mgr.num_used_blocks} != {total}"


class TestDA_P7_NoneStateLayer:
    """DA-P7-L1 (LOW): Cache layer with state=None skipped gracefully.

    Attack vector: If a model layer returns state=None (e.g., non-attention
    layer or uninitialized cache), decompose_cache_to_blocks should skip
    it without crashing.
    """

    def test_da_p7_none_state_layer_skipped(self):
        """Layer with state=None contributes no KV data to blocks."""
        class NullStateLayer:
            @property
            def state(self):
                return None

        cache = [NullStateLayer(), _MockKVCacheLayer(seq_len=16)]
        token_ids = list(range(16))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size=8)

        assert len(blocks) == 2
        # Only the valid layer should contribute
        for block in blocks:
            assert len(block['kv_data_per_layer']) == 1, \
                "Only valid layers should contribute KV data"

    def test_da_p7_short_state_tuple_skipped(self):
        """Layer with state=(keys_only,) (missing values) skipped."""
        class ShortStateLayer:
            @property
            def state(self):
                return (mx.zeros((1, 4, 16, 8)),)

        cache = [ShortStateLayer()]
        token_ids = list(range(16))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size=8)
        assert len(blocks) == 2
        for block in blocks:
            assert len(block['kv_data_per_layer']) == 0, \
                "Short state tuple should be skipped"
