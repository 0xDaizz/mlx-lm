"""Tests for SSD write-through cache features (Phases 1-5).

Covers:
- P3.3: Crash recovery via validate_index()
- P2.1+P2.2+P2.3: Dedup guard, num_tokens, collision isolation
- P1.2: Stats counters
- P2.4: Lock / thread-safety
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_lm_server.ssd_cache import SSDCache


class TestValidateIndex:
    """P3.3: Crash recovery via validate_index()."""

    def _make_cache(self, tmp_path: Path) -> SSDCache:
        cache = SSDCache(tmp_path)
        return cache

    def _make_block_data(self) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, 16, 64)), "values": mx.zeros((1, 4, 16, 64))}]

    def test_clean_index_no_changes(self, tmp_path):
        """validate_index on clean state should find nothing."""
        cache = self._make_cache(tmp_path)
        cache.save_block("hash_a", self._make_block_data())
        result = cache.validate_index()
        assert result["orphans_cleaned"] == 0
        assert result["missing_cleaned"] == 0

    def test_orphan_file_cleaned(self, tmp_path):
        """File on disk not in index should be deleted."""
        cache = self._make_cache(tmp_path)
        # Create orphan file
        orphan = tmp_path / "block_orphan123.safetensors"
        orphan.write_bytes(b"fake")
        result = cache.validate_index()
        assert result["orphans_cleaned"] == 1
        assert not orphan.exists()

    def test_missing_file_entry_cleaned(self, tmp_path):
        """Index entry with no backing file should be removed."""
        cache = self._make_cache(tmp_path)
        cache.save_block("hash_b", self._make_block_data())
        # Delete the file but keep index entry
        filepath = cache.index["hash_b"].filepath
        filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        filepath.unlink()
        result = cache.validate_index()
        assert result["missing_cleaned"] == 1
        assert "hash_b" not in cache.index

    def test_both_orphans_and_missing(self, tmp_path):
        """Both orphan files and missing entries at once."""
        cache = self._make_cache(tmp_path)
        cache.save_block("hash_c", self._make_block_data())
        # Orphan
        orphan = tmp_path / "block_orphan456.safetensors"
        orphan.write_bytes(b"fake")
        # Missing
        filepath = cache.index["hash_c"].filepath
        filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        filepath.unlink()
        result = cache.validate_index()
        assert result["orphans_cleaned"] == 1
        assert result["missing_cleaned"] == 1

    def test_corrupted_index_file(self, tmp_path):
        """Corrupted index.json should result in empty index after load."""
        cache = self._make_cache(tmp_path)
        cache.save_block("hash_d", self._make_block_data())
        # Corrupt the index
        index_path = tmp_path / "index.json"
        index_path.write_text("NOT VALID JSON{{{")
        # Reload
        cache2 = SSDCache(tmp_path)
        assert len(cache2.index) == 0
        # validate should clean up orphan file
        result = cache2.validate_index()
        assert result["orphans_cleaned"] == 1


class TestSaveBlockDedup:
    """P2.1+P2.2+P2.3: Dedup guard, num_tokens, collision isolation."""

    def _make_cache(self, tmp_path: Path) -> SSDCache:
        return SSDCache(tmp_path)

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_first_save_returns_saved(self, tmp_path):
        cache = self._make_cache(tmp_path)
        result = cache.save_block("hash1", self._make_block_data())
        assert result == "saved"
        assert cache._stats["ssd_save_success"] == 1

    def test_duplicate_save_returns_dedup(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.save_block("hash2", self._make_block_data())
        result = cache.save_block("hash2", self._make_block_data())
        assert result == "dedup"
        assert cache._stats["ssd_save_dedup_skip"] == 1

    def test_stale_index_resaves(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.save_block("hash3", self._make_block_data())
        # Delete file to make entry stale
        filepath = cache.index["hash3"].filepath
        filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        filepath.unlink()
        result = cache.save_block("hash3", self._make_block_data())
        assert result == "saved"
        assert cache._stats["ssd_stale_index_cleaned"] >= 1

    def test_collision_detected_by_num_tokens(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.save_block("hash4", self._make_block_data(num_tokens=16), num_tokens=16)
        # Same hash, different num_tokens -> collision
        result = cache.save_block("hash4", self._make_block_data(num_tokens=32), num_tokens=32)
        assert result == "collision"
        assert cache._stats["ssd_hash_collision_detected"] == 1

    def test_noncacheable_isolation_ttl(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.save_block("hash5", self._make_block_data(num_tokens=16), num_tokens=16)
        # Trigger collision
        cache.save_block("hash5", self._make_block_data(num_tokens=32), num_tokens=32)
        # Subsequent saves of same hash immediately return collision
        result = cache.save_block("hash5", self._make_block_data(num_tokens=16), num_tokens=16)
        assert result == "collision"

    def test_num_tokens_auto_computed(self, tmp_path):
        cache = self._make_cache(tmp_path)
        data = self._make_block_data(num_tokens=24)
        cache.save_block("hash6", data)  # no explicit num_tokens
        assert cache.index["hash6"].num_tokens == 24


class TestSSDCacheStats:
    """P1.2: Stats counters."""

    def test_initial_stats_zero(self, tmp_path):
        cache = SSDCache(tmp_path)
        stats = cache.get_stats()
        assert all(v == 0 for v in stats.values())

    def test_stats_snapshot_is_copy(self, tmp_path):
        cache = SSDCache(tmp_path)
        stats = cache.get_stats()
        stats["ssd_save_success"] = 999
        assert cache.get_stats()["ssd_save_success"] == 0


class TestSSDCacheThreadSafety:
    """P2.4: Lock protection."""

    def test_has_lock(self, tmp_path):
        cache = SSDCache(tmp_path)
        assert hasattr(cache, "_lock")

    def test_concurrent_save_load(self, tmp_path):
        """Basic concurrent access test."""
        cache = SSDCache(tmp_path)
        data = [{"keys": mx.zeros((1, 4, 16, 64)), "values": mx.zeros((1, 4, 16, 64))}]
        errors = []

        def save_blocks(start):
            for i in range(start, start + 10):
                try:
                    cache.save_block(f"hash_{i}", data)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=save_blocks, args=(0,))
        t2 = threading.Thread(target=save_blocks, args=(10,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0
        assert len(cache.index) == 20


class TestWriteThroughPath:
    """P2.5: Write-through integration via cache_block() -> TieredKVCache."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_cache_block_write_through_saves_to_ssd(self, tmp_path):
        """cache_block with ssd_policy=write_through should save to SSD."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        block_id = kv_mgr.cache_block(
            block_hash="wt_hash_1",
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            kv_data=self._make_block_data(),
            tiered_cache=tiered,
            ssd_policy="write_through",
        )
        assert block_id is not None
        assert ssd.has_block("wt_hash_1")

    def test_cache_block_evict_only_no_ssd(self, tmp_path):
        """cache_block with ssd_policy=evict_only should NOT save to SSD."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        block_id = kv_mgr.cache_block(
            block_hash="eo_hash_1",
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            kv_data=self._make_block_data(),
            tiered_cache=tiered,
            ssd_policy="evict_only",
        )
        assert block_id is not None
        assert not ssd.has_block("eo_hash_1")

    def test_cache_block_duplicate_no_ssd_write(self, tmp_path):
        """Caching the same block twice should not trigger a second SSD write."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)
        tokens = list(range(1, 17))

        kv_mgr.cache_block("dup_hash", tokens, self._make_block_data(),
                          tiered_cache=tiered, ssd_policy="write_through")
        initial_saves = ssd._stats["ssd_save_success"]

        # Second call: already in hash_table -> returns None, no SSD write
        result = kv_mgr.cache_block("dup_hash", tokens, self._make_block_data(),
                                    tiered_cache=tiered, ssd_policy="write_through")
        assert result is None
        assert ssd._stats["ssd_save_success"] == initial_saves


class TestEvictionDedup:
    """P2.6: evict_to_ssd() skips blocks already on SSD."""

    def _make_block_data(self) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, 16, 64)), "values": mx.zeros((1, 4, 16, 64))}]

    def test_evict_skips_ssd_existing(self, tmp_path):
        """Eviction should call save_block even for blocks already on SSD (no has_block pre-check).

        With P0-2, eviction always calls _save_to_ssd_with_durability for every block.
        For blocks already on SSD, save_block returns "dedup" which is accepted as success.
        """
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)
        tokens = list(range(1, 17))

        # Write-through: saves to both RAM and SSD
        block_id = kv_mgr.cache_block("ev_hash_1", tokens, self._make_block_data(),
                                      tiered_cache=tiered, ssd_policy="write_through")
        assert block_id is not None
        assert ssd.has_block("ev_hash_1")
        dedup_before = ssd._stats["ssd_save_dedup_skip"]

        # Free the block (ref_count=0) so it's evictable
        kv_mgr.free_blocks([block_id])

        # Evict — save_block is called (no has_block pre-check) but returns "dedup"
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) >= 1
        # save_block was called and returned "dedup" for the already-saved block
        assert ssd._stats["ssd_save_dedup_skip"] > dedup_before


class TestSSDWriterThread:
    """P4.1-P4.6: Async SSD writer thread tests."""

    def _make_cache(self, tmp_path: Path) -> "SSDCache":
        return SSDCache(tmp_path)

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_basic_enqueue_and_write(self, tmp_path):
        """Enqueued block should be written to SSD."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16)
        try:
            result = writer.enqueue("async_1", self._make_block_data())
            assert result == "queued"
            # Give writer time to process
            time.sleep(0.2)
            assert ssd.has_block("async_1")
        finally:
            writer.stop()

    def test_dedup_pending(self, tmp_path):
        """Duplicate enqueue for same hash should be skipped."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16)
        try:
            writer.enqueue("dup_1", self._make_block_data())
            result = writer.enqueue("dup_1", self._make_block_data())
            assert result == "dedup"
            stats = writer.get_stats()
            assert stats["writer_enqueue_dedup_skip"] >= 1
        finally:
            writer.stop()

    def test_sentinel_shutdown(self, tmp_path):
        """Writer should stop cleanly via sentinel."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16)
        # Enqueue some blocks
        for i in range(5):
            writer.enqueue(f"sent_{i}", self._make_block_data())
        ok = writer.stop(drain_timeout=5.0)
        assert ok is True
        # All blocks should have been written
        stats = writer.get_stats()
        assert stats["writer_save_success"] >= 5

    def test_enqueue_after_stop_returns_false(self, tmp_path):
        """Enqueue after stop() should return False."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16)
        writer.stop()
        result = writer.enqueue("after_stop", self._make_block_data())
        assert result == "closing"

    def test_backpressure_sync_fallback(self, tmp_path):
        """When queue is full, sync fallback should save directly."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        import threading

        ssd = self._make_cache(tmp_path)
        # Very small queue to trigger backpressure
        writer = SSDWriterThread(ssd, queue_size=1)

        # Block the worker thread temporarily
        blocker = threading.Event()
        original_save = ssd.save_block

        call_count = [0]
        def slow_save(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                blocker.wait(timeout=2)
            return original_save(*args, **kwargs)

        ssd.save_block = slow_save
        try:
            # First enqueue fills queue (worker processes it slowly)
            writer.enqueue("bp_0", self._make_block_data())
            time.sleep(0.05)
            # Second should attempt backpressure (queue is 1-deep)
            writer.enqueue("bp_1", self._make_block_data())
            time.sleep(0.05)
            # Third should trigger sync fallback
            writer.enqueue("bp_2", self._make_block_data())
            time.sleep(0.1)
            stats = writer.get_stats()
            # At least some should have been enqueued or fallen back
            assert stats["writer_enqueue_total"] >= 3
        finally:
            blocker.set()
            writer.stop()

    def test_stats_counting_accuracy(self, tmp_path):
        """Stats should accurately count operations."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=64)
        n = 20
        for i in range(n):
            writer.enqueue(f"stat_{i}", self._make_block_data())
        ok = writer.stop(drain_timeout=5.0)
        assert ok
        stats = writer.get_stats()
        assert stats["writer_enqueue_total"] == n
        assert stats["writer_save_success"] == n
        assert stats["writer_save_fail"] == 0

    def test_drain_guarantees_all_written(self, tmp_path):
        """After stop(), all enqueued blocks should be on SSD."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=128)
        hashes = [f"drain_{i}" for i in range(50)]
        for h in hashes:
            writer.enqueue(h, self._make_block_data())
        ok = writer.stop(drain_timeout=10.0)
        assert ok
        for h in hashes:
            assert ssd.has_block(h), f"Block {h} not found on SSD after drain"

    def test_pending_count_property(self, tmp_path):
        """pending_count should reflect in-flight writes."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=64)
        try:
            assert writer.pending_count >= 0
            assert writer.queue_size >= 0
        finally:
            writer.stop()

    def test_persistent_retry_on_failure(self, tmp_path):
        """Persistent mode should retry on save failure (return-value based).

        save_block() swallows I/O errors and returns "error". The writer
        _run() loop detects the "error" return and retries in persistent mode.
        """
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16, durability="persistent", max_retries=3)

        fail_count = [0]
        original_save = ssd.save_block
        def failing_save(*args, **kwargs):
            fail_count[0] += 1
            if fail_count[0] <= 2:
                return "error"  # Simulates save_block() returning "error" on I/O failure
            return original_save(*args, **kwargs)

        ssd.save_block = failing_save
        try:
            writer.enqueue("retry_1", self._make_block_data())
            time.sleep(1)
            stats = writer.get_stats()
            assert stats["writer_retry_attempts"] >= 1
            assert stats["writer_save_success"] >= 1  # Should succeed after retries
        finally:
            ssd.save_block = original_save
            writer.stop()

    def test_persistent_retry_in_sync_fallback(self, tmp_path):
        """Persistent retry should apply in queue-full sync fallback path."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=1, durability="persistent", max_retries=3)

        fail_count = [0]
        original_save = ssd.save_block
        def failing_save(*args, **kwargs):
            fail_count[0] += 1
            if fail_count[0] <= 2:
                return "error"
            return original_save(*args, **kwargs)

        # Block the worker so queue fills up
        blocker = threading.Event()
        def blocked_save(*args, **kwargs):
            blocker.wait(timeout=5)
            return original_save(*args, **kwargs)

        ssd.save_block = blocked_save
        try:
            # Worker picks up fill_0 and blocks on blocker.wait() — queue is now empty
            writer.enqueue("fill_0", self._make_block_data())
            time.sleep(0.05)
            # fill_1 sits in the queue (size=1) — queue is now full
            writer.enqueue("fill_1", self._make_block_data())
            time.sleep(0.05)
            # Now queue is full. Next enqueue will exhaust Level 0 + Level 1, hit sync fallback
            # Switch to failing save for the fallback
            ssd.save_block = failing_save
            writer.enqueue("fallback_retry", self._make_block_data())
            time.sleep(0.3)
            stats = writer.get_stats()
            # Should have retried in fallback
            assert stats["writer_retry_attempts"] >= 1
            assert stats["writer_save_success"] >= 1
            assert stats["writer_retry_final_fail"] == 0
        finally:
            blocker.set()
            ssd.save_block = original_save
            writer.stop()

    def test_persistent_retry_exhaustion_in_fallback(self, tmp_path):
        """When all retries fail in fallback, should count as final fail."""
        from mlx_lm_server.ssd_writer import SSDWriterThread
        ssd = self._make_cache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=1, durability="persistent", max_retries=2)

        original_save = ssd.save_block
        def always_fail(*args, **kwargs):
            return "error"

        blocker = threading.Event()
        def blocked_save(*args, **kwargs):
            blocker.wait(timeout=5)
            return original_save(*args, **kwargs)

        ssd.save_block = blocked_save
        try:
            # Worker picks up fill_1 and blocks — queue empty
            writer.enqueue("fill_1", self._make_block_data())
            time.sleep(0.05)
            # fill_2 sits in queue (size=1) — queue full
            writer.enqueue("fill_2", self._make_block_data())
            time.sleep(0.05)
            # Next enqueue hits sync fallback
            ssd.save_block = always_fail
            writer.enqueue("exhaust_retry", self._make_block_data())
            time.sleep(0.3)
            stats = writer.get_stats()
            assert stats["writer_retry_attempts"] == 2  # max_retries
            assert stats["writer_retry_final_fail"] >= 1
            assert stats["writer_save_fail"] >= 1
        finally:
            blocker.set()
            ssd.save_block = original_save
            writer.stop()


class TestSchedulerWriterInjection:
    """C: Verify ssd_writer is injected via constructor."""

    def test_scheduler_accepts_ssd_writer_param(self):
        """Scheduler should accept ssd_writer as constructor arg."""
        from mlx_lm_server.scheduler import Scheduler
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import MagicMock

        mock_writer = MagicMock()
        scheduler = Scheduler(
            config=ServerConfig(),
            ssd_writer=mock_writer,
        )
        assert scheduler._ssd_writer is mock_writer

    def test_scheduler_stop_calls_writer_stop(self):
        """Scheduler.stop() should call writer.stop() when injected."""
        from mlx_lm_server.scheduler import Scheduler
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import MagicMock

        mock_writer = MagicMock()
        mock_writer.stop.return_value = True
        scheduler = Scheduler(
            config=ServerConfig(),
            ssd_writer=mock_writer,
        )
        scheduler.stop()
        mock_writer.stop.assert_called_once_with(drain_timeout=5.0)

    def test_scheduler_none_writer_default(self):
        """Scheduler with no ssd_writer should have None."""
        from mlx_lm_server.scheduler import Scheduler
        from mlx_lm_server.config import ServerConfig

        scheduler = Scheduler(config=ServerConfig())
        assert scheduler._ssd_writer is None


class TestSyncDurability:
    """P0-1: Sync write paths should apply persistent retry."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_write_through_sync_persistent_retry(self, tmp_path):
        """write_through() with writer=None should retry on persistent."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(kv_mgr, ssd, writer=None, durability="persistent", max_retries=3)

        call_count = [0]
        original_save = ssd.save_block
        def failing_save(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return "error"
            return original_save(*args, **kwargs)

        ssd.save_block = failing_save
        tiered.write_through("sync_retry_1", self._make_block_data())
        # First call + 2 retries = 3 total calls until success
        assert call_count[0] == 3
        ssd.save_block = original_save
        assert ssd.has_block("sync_retry_1")

    def test_write_through_sync_best_effort_no_retry(self, tmp_path):
        """write_through() with best_effort should NOT retry."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(kv_mgr, ssd, writer=None, durability="best_effort")

        call_count = [0]
        def always_fail(*args, **kwargs):
            call_count[0] += 1
            return "error"

        ssd.save_block = always_fail
        tiered.write_through("no_retry_1", self._make_block_data())
        assert call_count[0] == 1  # Only 1 attempt, no retries

    def test_evict_to_ssd_persistent_retry(self, tmp_path):
        """evict_to_ssd() should retry saves in persistent mode."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(kv_mgr, ssd, writer=None, durability="persistent", max_retries=3)

        # Set up a block to evict
        tokens = list(range(1, 17))
        block_id = kv_mgr.cache_block(
            block_hash="evict_retry_hash",
            token_ids=tokens,
            kv_data=self._make_block_data(),
            tiered_cache=tiered,
            ssd_policy="evict_only",
        )
        assert block_id is not None
        kv_mgr.free_blocks([block_id])

        call_count = [0]
        original_save = ssd.save_block
        def failing_save(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return "error"
            return original_save(*args, **kwargs)

        ssd.save_block = failing_save
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) >= 1
        assert call_count[0] >= 3  # retried until success
        ssd.save_block = original_save

    def test_write_through_enqueue_fail_falls_back_to_sync(self, tmp_path):
        """When writer.enqueue() returns 'closing', should fall back to sync save."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import MagicMock

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)

        mock_writer = MagicMock()
        mock_writer.enqueue.return_value = "closing"  # Writer shutting down

        tiered = TieredKVCache(kv_mgr, ssd, writer=mock_writer, durability="best_effort")
        tiered.write_through("fallback_sync_1", self._make_block_data())

        # Should have fallen back to sync save
        assert ssd.has_block("fallback_sync_1")


class TestSSDFlushInterval:
    """P0-2: Verify flush_interval_s is wired and effective."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_flush_interval_s_stored(self, tmp_path):
        """flush_interval_s should be stored on SSDCache."""
        cache = SSDCache(tmp_path, flush_interval_s=2.5)
        assert cache._flush_interval_s == 2.5

    def test_flush_interval_s_default(self, tmp_path):
        """Default flush_interval_s should be 1.0."""
        cache = SSDCache(tmp_path)
        assert cache._flush_interval_s == 1.0

    def test_time_based_flush_triggers(self, tmp_path):
        """_mark_dirty should flush when time threshold exceeded."""
        cache = SSDCache(tmp_path, flush_interval_s=0.0)  # Immediate time-based flush
        # Set mutation count below threshold to ensure only time triggers
        cache._flush_interval = 1000  # Very high mutation threshold
        cache._mutation_count = 0

        # Manually advance the last flush time to simulate elapsed time
        cache._last_flush_time = time.monotonic() - 1.0

        # Test _mark_dirty directly:
        cache._index_dirty = False
        cache._mark_dirty()
        # With flush_interval_s=0, time_due should be True
        assert cache._index_dirty is False  # Was flushed
        assert cache._mutation_count == 0

    def test_mutation_based_flush_still_works(self, tmp_path):
        """Mutation-count based flush should still trigger."""
        cache = SSDCache(tmp_path, flush_interval_s=9999.0)  # Very long time interval
        cache._flush_interval = 3  # Low mutation threshold
        cache._mutation_count = 0
        cache._last_flush_time = time.monotonic()  # Recent, so time won't trigger

        # 2 mutations -- not enough
        cache._mark_dirty()
        cache._mark_dirty()
        assert cache._index_dirty is True  # Not flushed yet

        # 3rd mutation -- threshold reached
        cache._mark_dirty()
        assert cache._index_dirty is False  # Flushed
        assert cache._mutation_count == 0


# ---------------------------------------------------------------------------
# New test classes for P0-1 through P2-1 fixes
# ---------------------------------------------------------------------------


class TestP01WriterShutdown:
    """P0-1: Writer shutdown robustness — polling-based drain, no sentinel."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    @pytest.mark.timeout(10)
    def test_stop_timeout_thread_exits(self, tmp_path):
        """Start writer, enqueue items, verify stop(drain_timeout=5) returns True and thread exits."""
        from mlx_lm_server.ssd_writer import SSDWriterThread

        ssd = SSDCache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=64)
        for i in range(10):
            writer.enqueue(f"shutdown_{i}", self._make_block_data())
        ok = writer.stop(drain_timeout=5.0)
        assert ok is True
        assert not writer._thread.is_alive()

    @pytest.mark.timeout(10)
    def test_stop_with_slow_saves(self, tmp_path):
        """Mock save_block to be slow (0.2s sleep), enqueue 5 items, stop and verify all written."""
        from mlx_lm_server.ssd_writer import SSDWriterThread

        ssd = SSDCache(tmp_path)
        original_save = ssd.save_block

        def slow_save(*args, **kwargs):
            time.sleep(0.2)
            return original_save(*args, **kwargs)

        ssd.save_block = slow_save
        writer = SSDWriterThread(ssd, queue_size=64)
        for i in range(5):
            writer.enqueue(f"slow_{i}", self._make_block_data())
        ok = writer.stop(drain_timeout=3.0)
        ssd.save_block = original_save
        assert ok is True
        stats = writer.get_stats()
        assert stats["writer_save_success"] == 5

    @pytest.mark.timeout(10)
    def test_rapid_stop_after_start(self, tmp_path):
        """Create writer, immediately stop(), verify clean exit."""
        from mlx_lm_server.ssd_writer import SSDWriterThread

        ssd = SSDCache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16)
        ok = writer.stop(drain_timeout=2.0)
        assert ok is True

    @pytest.mark.timeout(10)
    def test_no_zombie_thread_after_stop(self, tmp_path):
        """Start writer, stop(), verify thread.is_alive() is False."""
        from mlx_lm_server.ssd_writer import SSDWriterThread

        ssd = SSDCache(tmp_path)
        writer = SSDWriterThread(ssd, queue_size=16)
        writer.enqueue("zombie_check", self._make_block_data())
        writer.stop(drain_timeout=3.0)
        assert not writer._thread.is_alive()


class TestP02EvictionSafety:
    """P0-2: Eviction always persists — no has_block pre-check."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_stale_index_eviction_resaves(self, tmp_path):
        """Save block to SSD, delete file but keep index, then evict: should re-save."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)
        tokens = list(range(1, 17))

        # Cache block with write-through
        block_id = kv_mgr.cache_block("stale_ev_hash", tokens, self._make_block_data(),
                                      tiered_cache=tiered, ssd_policy="write_through")
        assert block_id is not None
        assert ssd.has_block("stale_ev_hash")

        # Delete the file but keep the index entry (simulate stale index)
        filepath = ssd.index["stale_ev_hash"].filepath
        filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        filepath.unlink()

        # Free the block so it's evictable
        kv_mgr.free_blocks([block_id])

        saves_before = ssd._stats["ssd_save_success"]
        stale_cleaned_before = ssd._stats["ssd_stale_index_cleaned"]

        # Evict — save_block should detect stale index and re-save
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) >= 1
        # Should have re-saved (stale index cleaned + new save)
        assert ssd._stats["ssd_stale_index_cleaned"] > stale_cleaned_before
        assert ssd._stats["ssd_save_success"] > saves_before

    def test_collision_blocks_not_evicted(self, tmp_path):
        """When save_block returns 'collision', block stays in RAM (not evicted)."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import patch

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)
        tokens = list(range(1, 17))

        # Cache block (evict_only, NOT write-through)
        block_id = kv_mgr.cache_block("coll_ev_hash", tokens, self._make_block_data(),
                                      tiered_cache=tiered, ssd_policy="evict_only")
        assert block_id is not None
        kv_mgr.free_blocks([block_id])

        # Mock save_block to return "collision"
        original_save = ssd.save_block
        ssd.save_block = lambda *a, **kw: "collision"
        try:
            evicted = tiered.evict_to_ssd(num_blocks=1)
            # Block should NOT have been evicted (collision means keep in RAM)
            assert len(evicted) == 0
        finally:
            ssd.save_block = original_save

    def test_collision_bumps_last_accessed(self, tmp_path):
        """When eviction gets 'collision', block.last_accessed is updated to prevent hot-loop."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)
        tokens = list(range(1, 17))

        block_id = kv_mgr.cache_block("bump_hash", tokens, self._make_block_data(),
                                      tiered_cache=tiered, ssd_policy="evict_only")
        assert block_id is not None
        kv_mgr.free_blocks([block_id])

        block = kv_mgr.pool.blocks[block_id]
        old_last_accessed = block.last_accessed

        # Mock save_block to return "collision"
        original_save = ssd.save_block
        ssd.save_block = lambda *a, **kw: "collision"
        try:
            time.sleep(0.01)  # Ensure time.time() advances
            tiered.evict_to_ssd(num_blocks=1)
            # last_accessed should have been bumped
            assert block.last_accessed > old_last_accessed
        finally:
            ssd.save_block = original_save

    def test_error_blocks_kept_in_ram(self, tmp_path):
        """When save_block returns 'error', block stays in RAM."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)
        tokens = list(range(1, 17))

        block_id = kv_mgr.cache_block("err_ev_hash", tokens, self._make_block_data(),
                                      tiered_cache=tiered, ssd_policy="evict_only")
        assert block_id is not None
        kv_mgr.free_blocks([block_id])

        # Mock save_block to return "error"
        original_save = ssd.save_block
        ssd.save_block = lambda *a, **kw: "error"
        try:
            evicted = tiered.evict_to_ssd(num_blocks=1)
            # Block should NOT have been evicted (error means keep in RAM)
            assert len(evicted) == 0
        finally:
            ssd.save_block = original_save


class TestP11EnqueueStatus:
    """P1-1: Enqueue returns string status ('queued' | 'dedup' | 'closing')."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_write_through_dedup_no_sync_fallback(self, tmp_path):
        """Writer enqueue returns 'dedup' -> no sync fallback."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import MagicMock

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        mock_writer = MagicMock()
        mock_writer.enqueue.return_value = "dedup"

        tiered = TieredKVCache(kv_mgr, ssd, writer=mock_writer, durability="best_effort")
        tiered.write_through("dedup_test", self._make_block_data())

        # "dedup" means already pending — should NOT fall back to sync
        assert not ssd.has_block("dedup_test")
        # _save_to_ssd_with_durability should NOT have been called
        assert tiered._sync_stats["tiered_sync_save_attempts"] == 0

    def test_write_through_closing_triggers_sync(self, tmp_path):
        """Writer enqueue returns 'closing' -> sync fallback happens."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import MagicMock

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        mock_writer = MagicMock()
        mock_writer.enqueue.return_value = "closing"

        tiered = TieredKVCache(kv_mgr, ssd, writer=mock_writer, durability="best_effort")
        tiered.write_through("closing_test", self._make_block_data())

        # "closing" means writer shutting down — should fall back to sync save
        assert ssd.has_block("closing_test")
        assert tiered._sync_stats["tiered_sync_save_attempts"] == 1

    def test_write_through_queued_returns_immediately(self, tmp_path):
        """Writer enqueue returns 'queued' -> no sync fallback."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig
        from unittest.mock import MagicMock

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        mock_writer = MagicMock()
        mock_writer.enqueue.return_value = "queued"

        tiered = TieredKVCache(kv_mgr, ssd, writer=mock_writer, durability="best_effort")
        tiered.write_through("queued_test", self._make_block_data())

        # "queued" means accepted — should NOT fall back to sync
        assert not ssd.has_block("queued_test")
        assert tiered._sync_stats["tiered_sync_save_attempts"] == 0


class TestP12LockConsistency:
    """P1-2: SSDCache lock consistency — save_index/load_index/num_blocks all lock."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    @pytest.mark.timeout(10)
    def test_num_blocks_locked(self, tmp_path):
        """Concurrent save_block + num_blocks reads should not raise."""
        ssd = SSDCache(tmp_path)
        errors = []

        def do_saves():
            for i in range(20):
                try:
                    ssd.save_block(f"lock_test_{i}", self._make_block_data())
                except Exception as e:
                    errors.append(e)

        def do_reads():
            for _ in range(50):
                try:
                    _ = ssd.num_blocks
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=do_saves)
        t2 = threading.Thread(target=do_reads)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0

    @pytest.mark.timeout(10)
    def test_save_index_load_index_no_deadlock(self, tmp_path):
        """save_index() and load_index() concurrently should not deadlock."""
        ssd = SSDCache(tmp_path)
        ssd.save_block("idx_test", self._make_block_data())
        errors = []
        done = threading.Event()

        def do_save_index():
            for _ in range(20):
                try:
                    ssd.save_index()
                except Exception as e:
                    errors.append(e)
            done.set()

        def do_load_index():
            for _ in range(20):
                try:
                    ssd.load_index()
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=do_save_index)
        t2 = threading.Thread(target=do_load_index)
        t1.start()
        t2.start()
        t1.join(timeout=8)
        t2.join(timeout=8)
        assert not t1.is_alive(), "save_index thread deadlocked"
        assert not t2.is_alive(), "load_index thread deadlocked"
        assert len(errors) == 0

    @pytest.mark.timeout(10)
    def test_concurrent_validate_flush_save(self, tmp_path):
        """validate_index(), flush(), save_index() concurrently, no deadlock."""
        ssd = SSDCache(tmp_path)
        # Seed some data
        for i in range(5):
            ssd.save_block(f"conc_{i}", self._make_block_data())
        errors = []

        def do_validate():
            for _ in range(10):
                try:
                    ssd.validate_index()
                except Exception as e:
                    errors.append(e)

        def do_flush():
            for _ in range(10):
                try:
                    ssd.flush()
                except Exception as e:
                    errors.append(e)

        def do_save_index():
            for _ in range(10):
                try:
                    ssd.save_index()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=do_validate),
            threading.Thread(target=do_flush),
            threading.Thread(target=do_save_index),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8)
        for t in threads:
            assert not t.is_alive(), f"Thread {t.name} deadlocked"
        assert len(errors) == 0


class TestP13FlushTimestamp:
    """P1-3: _save_index() sets _last_flush_time after successful os.replace()."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_save_index_updates_last_flush_time(self, tmp_path):
        """save_index() should update _last_flush_time."""
        ssd = SSDCache(tmp_path)
        ssd.save_block("flush_ts_1", self._make_block_data())
        old_flush = ssd._last_flush_time
        time.sleep(0.01)  # Ensure monotonic clock advances
        ssd.save_index()
        assert ssd._last_flush_time > old_flush

    def test_no_unnecessary_consecutive_flush(self, tmp_path):
        """After a save_block triggers flush, _mark_dirty should NOT re-flush if time hasn't elapsed."""
        ssd = SSDCache(tmp_path, flush_interval_s=9999.0)
        ssd._flush_interval = 1000  # High mutation threshold
        ssd.save_block("flush_guard_1", self._make_block_data())  # This triggers immediate flush

        # Record the flush time
        flush_time_after_save = ssd._last_flush_time

        # _mark_dirty should NOT trigger flush (time hasn't elapsed, mutation count is low)
        ssd._index_dirty = False
        ssd._mutation_count = 0
        ssd._mark_dirty()
        # Time-based flush should NOT have occurred (9999s interval)
        # Mutation-based flush should NOT have occurred (threshold=1000, count=1)
        assert ssd._index_dirty is True  # Still dirty, not flushed
        assert ssd._last_flush_time == flush_time_after_save


class TestP21SyncObservability:
    """P2-1: Sync observability — TieredKVCache._sync_stats with 5 counters."""

    def _make_block_data(self, num_tokens: int = 16) -> list[dict[str, mx.array]]:
        return [{"keys": mx.zeros((1, 4, num_tokens, 64)), "values": mx.zeros((1, 4, num_tokens, 64))}]

    def test_tiered_sync_stats_exist(self, tmp_path):
        """TieredKVCache should have _sync_stats dict with all 5 expected keys."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        expected_keys = {
            "tiered_sync_save_attempts",
            "tiered_sync_save_success",
            "tiered_sync_retry_attempts",
            "tiered_sync_save_fail",
            "tiered_sync_save_collision",
        }
        assert expected_keys == set(tiered._sync_stats.keys())
        assert all(v == 0 for v in tiered._sync_stats.values())

    def test_sync_save_increments_attempts_and_success(self, tmp_path):
        """_save_to_ssd_with_durability should increment attempts + success on 'saved'."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        result = tiered._save_to_ssd_with_durability("sync_stat_1", self._make_block_data())
        assert result == "saved"
        assert tiered._sync_stats["tiered_sync_save_attempts"] == 1
        assert tiered._sync_stats["tiered_sync_save_success"] == 1

    def test_sync_save_error_increments_fail(self, tmp_path):
        """_save_to_ssd_with_durability should increment fail on 'error' (best_effort)."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd, durability="best_effort")

        original_save = ssd.save_block
        ssd.save_block = lambda *a, **kw: "error"
        try:
            result = tiered._save_to_ssd_with_durability("err_stat_1", self._make_block_data())
            assert result == "error"
            assert tiered._sync_stats["tiered_sync_save_attempts"] == 1
            assert tiered._sync_stats["tiered_sync_save_fail"] == 1
        finally:
            ssd.save_block = original_save

    def test_sync_save_collision_counted(self, tmp_path):
        """_save_to_ssd_with_durability should increment collision counter."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        original_save = ssd.save_block
        ssd.save_block = lambda *a, **kw: "collision"
        try:
            result = tiered._save_to_ssd_with_durability("coll_stat_1", self._make_block_data())
            assert result == "collision"
            assert tiered._sync_stats["tiered_sync_save_collision"] == 1
        finally:
            ssd.save_block = original_save

    def test_get_sync_stats_returns_copy(self, tmp_path):
        """get_sync_stats() should return a copy, not the original dict."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        stats = tiered.get_sync_stats()
        stats["tiered_sync_save_attempts"] = 999
        assert tiered._sync_stats["tiered_sync_save_attempts"] == 0

    def test_scheduler_merges_sync_stats(self, tmp_path):
        """Scheduler.get_cache_stats() should include tiered_sync_ keys."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.scheduler import Scheduler
        from mlx_lm_server.config import ServerConfig

        ssd = SSDCache(tmp_path)
        kv_mgr = KVCacheManager(ServerConfig(num_blocks=32), ssd=ssd)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # Perform a save so there's non-zero data
        tiered._save_to_ssd_with_durability("sched_stat_1", self._make_block_data())

        scheduler = Scheduler(
            config=ServerConfig(),
            kv_cache_manager=kv_mgr,
            tiered_cache=tiered,
        )

        stats = scheduler.get_cache_stats()
        # Verify sync stats are merged (via get_sync_stats)
        assert "tiered_sync_save_attempts" in stats
        assert stats["tiered_sync_save_attempts"] == 1
        assert "tiered_sync_save_success" in stats
        assert stats["tiered_sync_save_success"] == 1
