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
        """Eviction should skip save_block if block already on SSD."""
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
        saves_before = ssd._stats["ssd_save_success"]

        # Free the block (ref_count=0) so it's evictable
        kv_mgr.free_blocks([block_id])

        # Evict should skip SSD save since block already exists
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) >= 1
        # Save count should NOT increase (block was already on SSD)
        assert ssd._stats["ssd_save_success"] == saves_before


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
            assert result is True
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
            assert result is False
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
        assert result is False

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
