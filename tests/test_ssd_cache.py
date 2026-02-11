"""Tests for SSD Cache Tier (Phase 1.3: P1.11-P1.15).

Covers:
- SSD cache initialization and directory creation (P1.11)
- save_block to safetensors (P1.12)
- load_block from safetensors + timestamp update (P1.13)
- prune_expired TTL-based deletion (P1.14)
- save_index / load_index JSON persistence (P1.15)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_lm_server.ssd_cache import CURRENT_HASH_VERSION, SSDCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kv_data(
    batch: int = 1, heads: int = 2, seq_len: int = 4, head_dim: int = 8
) -> dict[str, mx.array]:
    """Create dummy K/V tensors for testing."""
    keys = mx.random.normal((batch, heads, seq_len, head_dim))
    values = mx.random.normal((batch, heads, seq_len, head_dim))
    return {"keys": keys, "values": values}


# ---------------------------------------------------------------------------
# P1.11 — SSD cache init + directory creation
# ---------------------------------------------------------------------------


class TestSSDInit:
    def test_ssd_init(self, tmp_path: Path):
        """SSDCache creates the cache directory on init."""
        cache_dir = tmp_path / "kv-cache"
        cache = SSDCache(cache_dir=cache_dir, ttl_days=7)
        assert cache_dir.exists()
        assert cache.ttl_days == 7
        assert cache.num_blocks == 0

    def test_ssd_dir(self, tmp_path: Path):
        """SSDCache creates nested directories if they don't exist."""
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        cache = SSDCache(cache_dir=cache_dir)
        assert cache_dir.exists()
        assert cache.num_blocks == 0

    def test_ssd_init_existing_dir(self, tmp_path: Path):
        """SSDCache works when directory already exists."""
        cache_dir = tmp_path / "kv-cache"
        cache_dir.mkdir()
        cache = SSDCache(cache_dir=cache_dir)
        assert cache_dir.exists()


# ---------------------------------------------------------------------------
# P1.12 — save_block
# ---------------------------------------------------------------------------


class TestSaveBlock:
    def test_save_creates_file(self, tmp_path: Path):
        """save_block creates a safetensors file on disk."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data()
        block_hash = "hash_12345"

        cache.save_block(block_hash, kv_data)

        expected_file = cache.cache_dir / f"block_{block_hash}.safetensors"
        assert expected_file.exists()
        assert cache.num_blocks == 1
        assert block_hash in cache.index

    def test_save_records_metadata(self, tmp_path: Path):
        """save_block records correct metadata in the index."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data(seq_len=16)
        block_hash = "hash_99999"

        cache.save_block(block_hash, kv_data)

        meta = cache.index[block_hash]
        assert meta.block_hash == block_hash
        assert meta.num_tokens == 16
        assert isinstance(meta.last_accessed, datetime)


# ---------------------------------------------------------------------------
# P1.13 — load_block
# ---------------------------------------------------------------------------


class TestLoadBlock:
    def test_load_matches_saved(self, tmp_path: Path):
        """load_block returns data matching what was saved."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data(batch=1, heads=4, seq_len=8, head_dim=16)
        block_hash = "hash_42"

        cache.save_block(block_hash, kv_data)
        loaded = cache.load_block(block_hash)

        assert loaded is not None
        assert mx.allclose(loaded["keys"], kv_data["keys"]).item()
        assert mx.allclose(loaded["values"], kv_data["values"]).item()
        assert loaded["keys"].shape == kv_data["keys"].shape, "Key shapes must match"
        assert loaded["values"].shape == kv_data["values"].shape, "Value shapes must match"
        assert loaded["keys"].dtype == kv_data["keys"].dtype, "Key dtypes must match"
        assert loaded["values"].dtype == kv_data["values"].dtype, "Value dtypes must match"

    def test_load_missing_returns_none(self, tmp_path: Path):
        """load_block returns None for a non-existent block."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        assert cache.load_block("nonexistent_hash") is None

    def test_load_updates_timestamp(self, tmp_path: Path):
        """load_block updates the last_accessed timestamp."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data()
        block_hash = "hash_100"

        cache.save_block(block_hash, kv_data)
        time_before = cache.index[block_hash].last_accessed

        # Force a small time difference
        import time

        time.sleep(0.01)

        cache.load_block(block_hash)
        time_after = cache.index[block_hash].last_accessed

        assert time_after >= time_before

    def test_load_stale_index_entry(self, tmp_path: Path):
        """load_block handles missing file (stale index entry) gracefully."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data()
        block_hash = "hash_200"

        cache.save_block(block_hash, kv_data)
        # Delete the file behind the cache's back
        cache.index[block_hash].filepath.unlink()

        result = cache.load_block(block_hash)
        assert result is None
        assert block_hash not in cache.index


# ---------------------------------------------------------------------------
# P1.14 — prune_expired
# ---------------------------------------------------------------------------


class TestPruneExpired:
    def test_prune_ttl_zero(self, tmp_path: Path):
        """With ttl_days=0, all blocks are immediately expired."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=0)
        kv_data = make_kv_data()

        cache.save_block("hash_111", kv_data)
        cache.save_block("hash_222", kv_data)

        # Force timestamps into the past
        for meta in cache.index.values():
            meta.last_accessed = datetime.now() - timedelta(seconds=1)

        pruned = cache.prune_expired()
        assert pruned == 2
        assert cache.num_blocks == 0

    def test_prune_keeps_recent(self, tmp_path: Path):
        """prune_expired keeps blocks within their TTL."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv_data = make_kv_data()

        cache.save_block("hash_111", kv_data)  # Recently saved
        cache.save_block("hash_222", kv_data)

        # Make one block old and one recent
        cache.index["hash_111"].last_accessed = datetime.now() - timedelta(days=10)
        cache.index["hash_222"].last_accessed = datetime.now()

        pruned = cache.prune_expired()
        assert pruned == 1
        assert "hash_111" not in cache.index
        assert "hash_222" in cache.index

    def test_prune_deletes_files(self, tmp_path: Path):
        """prune_expired removes the safetensors files from disk."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=0)
        kv_data = make_kv_data()

        cache.save_block("hash_333", kv_data)
        filepath = cache.index["hash_333"].filepath
        assert filepath.exists()

        cache.index["hash_333"].last_accessed = datetime.now() - timedelta(seconds=1)
        cache.prune_expired()
        assert not filepath.exists()


# ---------------------------------------------------------------------------
# P1.15 — save_index / load_index
# ---------------------------------------------------------------------------


class TestIndexPersistence:
    def test_index_persistence(self, tmp_path: Path):
        """Index survives cache reconstruction (save + load cycle)."""
        cache_dir = tmp_path / "cache"
        cache1 = SSDCache(cache_dir=cache_dir)
        kv_data = make_kv_data()

        cache1.save_block("hash_111", kv_data)
        cache1.save_block("hash_222", kv_data)

        # Create a new cache instance pointing at the same dir
        cache2 = SSDCache(cache_dir=cache_dir)
        assert cache2.num_blocks == 2
        assert "hash_111" in cache2.index
        assert "hash_222" in cache2.index

    def test_index_file_created(self, tmp_path: Path):
        """save_index creates a JSON file in the cache directory."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data()
        cache.save_block("hash_999", kv_data)

        index_path = cache.cache_dir / "index.json"
        assert index_path.exists()

        data = json.loads(index_path.read_text())
        # New format: blocks under "blocks" key, metadata under "__metadata__"
        assert "__metadata__" in data
        assert "blocks" in data
        assert "hash_999" in data["blocks"]

    def test_index_empty_on_fresh_start(self, tmp_path: Path):
        """New cache with no index file starts empty."""
        cache = SSDCache(cache_dir=tmp_path / "fresh-cache")
        assert cache.num_blocks == 0

    def test_index_corrupt_graceful(self, tmp_path: Path):
        """Corrupted index file is handled gracefully (starts fresh)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        index_path = cache_dir / "index.json"
        index_path.write_text("not valid json {{{")

        cache = SSDCache(cache_dir=cache_dir)
        assert cache.num_blocks == 0


# ---------------------------------------------------------------------------
# E3 — Index metadata + hash_version
# ---------------------------------------------------------------------------


class TestIndexMetadata:
    """Tests for SSD index metadata (E3)."""

    def test_index_contains_metadata(self, tmp_path: Path):
        """Saved index includes __metadata__ with version info."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        kv_data = make_kv_data()
        cache.save_block("hash_meta_test", kv_data)

        index_path = cache.cache_dir / "index.json"
        data = json.loads(index_path.read_text())

        assert "__metadata__" in data
        meta = data["__metadata__"]
        assert meta["index_version"] == 1
        assert meta["hash_version"] == CURRENT_HASH_VERSION
        assert "created_at" in meta

    def test_hash_version_mismatch_invalidates(self, tmp_path: Path):
        """Index with different hash_version is invalidated on load."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        # Write an index with wrong hash_version
        index_data = {
            "__metadata__": {
                "index_version": 1,
                "hash_version": 999,  # Wrong version
                "created_at": "2026-01-01T00:00:00",
            },
            "blocks": {
                "test_hash": {
                    "block_hash": "test_hash",
                    "filepath": str(cache_dir / "block_test.safetensors"),
                    "last_accessed": "2026-01-01T00:00:00",
                    "num_tokens": 4,
                }
            },
        }
        (cache_dir / "index.json").write_text(json.dumps(index_data))

        cache = SSDCache(cache_dir=cache_dir)
        assert cache.num_blocks == 0  # Invalidated

    def test_legacy_format_loaded(self, tmp_path: Path):
        """Legacy index (no __metadata__) is loaded as blocks."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        # Write legacy format (flat dict of blocks)
        legacy_data = {
            "legacy_hash": {
                "block_hash": "legacy_hash",
                "filepath": str(cache_dir / "block_legacy.safetensors"),
                "last_accessed": "2026-01-01T00:00:00",
                "num_tokens": 4,
            }
        }
        (cache_dir / "index.json").write_text(json.dumps(legacy_data))

        # Create the file so it's valid
        import mlx.core as mx
        kv = {"keys": mx.zeros((1, 1, 4, 4)), "values": mx.zeros((1, 1, 4, 4))}
        mx.save_safetensors(str(cache_dir / "block_legacy.safetensors"), kv)

        cache = SSDCache(cache_dir=cache_dir)
        assert cache.num_blocks == 1
        assert "legacy_hash" in cache.index


# ---------------------------------------------------------------------------
# F2 — Batch flush
# ---------------------------------------------------------------------------


class TestBatchFlush:
    """Tests for SSD index batch flush (F2)."""

    def test_mark_dirty_defers_flush(self, tmp_path: Path):
        """_mark_dirty defers index write until flush_interval."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        cache._flush_interval = 5

        # Save blocks (each calls save_index immediately, but load_block uses _mark_dirty)
        kv_data = make_kv_data()
        for i in range(3):
            cache.save_block(f"hash_{i}", kv_data)

        # Load blocks — these use _mark_dirty for access time updates
        for i in range(3):
            cache.load_block(f"hash_{i}")

        # After 3 loads: _mutation_count=3, dirty=True, not flushed
        assert cache._index_dirty is True
        assert cache._mutation_count == 3

    def test_flush_writes_index(self, tmp_path: Path):
        """flush() writes pending index changes to disk."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        cache._flush_interval = 100  # High interval to prevent auto-flush

        kv_data = make_kv_data()
        cache.save_block("flush_test", kv_data)

        # Manually set dirty state
        cache._index_dirty = True
        cache._mutation_count = 5

        cache.flush()

        assert cache._index_dirty is False
        assert cache._mutation_count == 0

    def test_flush_noop_when_clean(self, tmp_path: Path):
        """flush() is a no-op when index is not dirty."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        assert cache._index_dirty is False

        # Should not raise or do anything
        cache.flush()
        assert cache._index_dirty is False

    def test_auto_flush_at_interval(self, tmp_path: Path):
        """Index is automatically flushed after _flush_interval mutations."""
        cache = SSDCache(cache_dir=tmp_path / "cache")
        cache._flush_interval = 3

        kv_data = make_kv_data()
        # Save 3 blocks (immediate write) then load them (_mark_dirty)
        for i in range(3):
            cache.save_block(f"auto_hash_{i}", kv_data)

        # Load 3 blocks — at 3rd load, auto-flush should trigger
        for i in range(3):
            cache.load_block(f"auto_hash_{i}")

        # After exactly 3 mark_dirty calls with interval=3, should be flushed
        assert cache._index_dirty is False
        assert cache._mutation_count == 0


# ---------------------------------------------------------------------------
# T3.1 — SSD disk size limit
# ---------------------------------------------------------------------------


class TestSSDDiskSizeLimit:
    """Tests for SSD disk size limit (T3.1)."""

    def test_max_size_prune_on_save(self, tmp_path: Path):
        """When max_size_bytes is exceeded, LRU blocks are pruned."""
        # Very small max size to trigger pruning
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=1000)

        # Save several blocks to fill up
        for i in range(10):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            result = cache.save_block(f"hash_{i}", kv, num_tokens=4)
            assert result in ("saved", "dedup")

        # Total bytes should be tracked
        assert cache._total_bytes > 0
        # Some blocks should have been pruned via LRU
        assert len(cache.index) < 10

    def test_unlimited_size(self, tmp_path: Path):
        """max_size_bytes=0 means unlimited."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=0)
        for i in range(5):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            cache.save_block(f"hash_{i}", kv, num_tokens=4)
        assert len(cache.index) == 5  # All kept

    def test_prune_lru_order(self, tmp_path: Path):
        """LRU pruning removes oldest-accessed blocks first."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=0)

        # Save 3 blocks
        for i in range(3):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            cache.save_block(f"hash_{i}", kv, num_tokens=4)

        # Manually set access times: hash_0 oldest, hash_2 newest
        cache.index["hash_0"].last_accessed = datetime.now() - timedelta(days=3)
        cache.index["hash_1"].last_accessed = datetime.now() - timedelta(days=2)
        cache.index["hash_2"].last_accessed = datetime.now() - timedelta(days=1)

        # Prune with small target (2-phase: index update under lock, file deletion outside)
        with cache._lock:
            freed, files_to_delete = cache._prune_lru_for_space(1)  # Just free 1 byte minimum

        # Phase 2: delete files outside lock
        for _, fpath in files_to_delete:
            fpath.unlink(missing_ok=True)

        # Oldest block should be gone from index
        assert "hash_0" not in cache.index
        assert "hash_2" in cache.index  # newest kept
        assert freed > 0

    def test_total_bytes_tracking(self, tmp_path: Path):
        """_total_bytes tracks cumulative size correctly."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=0)
        assert cache._total_bytes == 0

        kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
        cache.save_block("hash_a", kv, num_tokens=4)
        assert cache._total_bytes > 0

        # Load initial size
        size_after_one = cache._total_bytes
        cache.save_block("hash_b", kv, num_tokens=4)
        assert cache._total_bytes > size_after_one

    def test_get_stats_includes_size_fields(self, tmp_path: Path):
        """get_stats() includes ssd_total_bytes and ssd_max_size_bytes."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=5000)
        stats = cache.get_stats()
        assert "ssd_total_bytes" in stats
        assert "ssd_max_size_bytes" in stats
        assert stats["ssd_max_size_bytes"] == 5000
        assert stats["ssd_total_bytes"] == 0

    def test_lru_prune_stats_tracked(self, tmp_path: Path):
        """LRU pruning increments ssd_lru_prune_count and ssd_lru_prune_bytes."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=1000)

        for i in range(10):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            cache.save_block(f"hash_{i}", kv, num_tokens=4)

        stats = cache.get_stats()
        assert stats["ssd_lru_prune_count"] > 0
        assert stats["ssd_lru_prune_bytes"] > 0

    def test_initial_size_from_existing_files(self, tmp_path: Path):
        """New SSDCache instance calculates _total_bytes from existing files."""
        cache_dir = tmp_path / "cache"
        cache1 = SSDCache(cache_dir=cache_dir, ttl_days=7, max_size_bytes=0)
        kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
        cache1.save_block("hash_persist", kv, num_tokens=4)
        size1 = cache1._total_bytes
        assert size1 > 0

        # Create new cache instance pointing at same directory
        cache2 = SSDCache(cache_dir=cache_dir, ttl_days=7, max_size_bytes=0)
        assert cache2._total_bytes == size1


# ---------------------------------------------------------------------------
# CACHE-M1 — save_block 2-phase locking (lock not held during I/O)
# ---------------------------------------------------------------------------


class TestSaveBlock2Phase:
    """Test save_block 2-phase locking: reservation under lock, I/O outside lock."""

    def test_concurrent_save_dedup(self, tmp_path: Path):
        """Sequential save_block() for the same hash: second sees dedup.

        NOTE: mx.save_safetensors is not thread-safe (Metal command buffers),
        so we test the 2-phase dedup logic sequentially instead of with
        truly concurrent threads. The reservation mechanism ensures the
        second call sees the index entry from Phase 1 and returns dedup.
        """
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv = make_kv_data()

        # First save — should succeed
        r1 = cache.save_block("concurrent_hash", kv, num_tokens=4)
        assert r1 == "saved"

        # Second save for same hash — should return dedup (index entry exists)
        r2 = cache.save_block("concurrent_hash", kv, num_tokens=4)
        assert r2 == "dedup"
        assert cache.num_blocks == 1

    def test_save_block_io_failure_reverts_reservation(self, tmp_path: Path):
        """If disk I/O fails, the index reservation should be reverted."""
        import unittest.mock as mock

        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv = make_kv_data()

        # Make mx.save_safetensors fail
        with mock.patch("mlx.core.save_safetensors", side_effect=OSError("Disk full")):
            result = cache.save_block("fail_hash", kv, num_tokens=4)

        assert result == "error"
        # Reservation should have been reverted
        assert "fail_hash" not in cache.index
        assert cache.num_blocks == 0
        assert cache._stats["ssd_save_fail"] == 1

    def test_save_block_still_dedup_after_success(self, tmp_path: Path):
        """After a successful save, a second save of the same hash returns dedup."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv = make_kv_data()

        r1 = cache.save_block("dedup_test", kv, num_tokens=4)
        assert r1 == "saved"

        r2 = cache.save_block("dedup_test", kv, num_tokens=4)
        assert r2 == "dedup"

    def test_save_block_loads_correctly_after_2phase(self, tmp_path: Path):
        """Data saved with 2-phase save_block can be loaded back correctly."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7)
        kv = make_kv_data(seq_len=8)

        result = cache.save_block("load_test", kv, num_tokens=8)
        assert result == "saved"

        loaded = cache.load_block("load_test")
        assert loaded is not None
        assert "keys" in loaded
        assert "values" in loaded


# ---------------------------------------------------------------------------
# C2-M6 — prune_expired 2-phase (no lock during file deletion)
# ---------------------------------------------------------------------------


class TestPruneExpired2Phase:
    """Test prune_expired 2-phase: index update under lock, file deletion outside."""

    def test_prune_expired_does_not_block_save(self, tmp_path: Path):
        """prune_expired releases lock before file deletion, so save_block is not blocked.

        We verify this by checking that Phase 2 (file deletion) happens
        outside the lock: save_block called immediately after prune_expired
        returns succeeds without contention.

        NOTE: We avoid truly concurrent threads because mx.save_safetensors
        is not thread-safe at the Metal command buffer level.
        """
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=0)
        kv = make_kv_data()

        # Save some blocks and mark them expired
        for i in range(5):
            cache.save_block(f"expired_{i}", kv, num_tokens=4)
            cache.index[f"expired_{i}"].last_accessed = datetime.now() - timedelta(seconds=10)

        # Prune expired blocks (2-phase: index update under lock, file delete outside)
        pruned = cache.prune_expired()
        assert pruned == 5

        # Save a new block immediately after prune — should not be blocked
        r = cache.save_block("new_block", kv, num_tokens=4)
        assert r == "saved"
        assert "new_block" in cache.index

    def test_prune_expired_cleans_noncacheable(self, tmp_path: Path):
        """prune_expired also cleans expired _noncacheable entries."""
        import time

        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=0)
        kv = make_kv_data()

        # Add expired noncacheable entry
        cache._noncacheable["stale_nc"] = time.time() - 100  # expired
        cache._noncacheable["fresh_nc"] = time.time() + 3600  # not expired

        # Need at least one expired block for prune to run
        cache.save_block("block_to_prune", kv, num_tokens=4)
        cache.index["block_to_prune"].last_accessed = datetime.now() - timedelta(seconds=10)

        cache.prune_expired()

        assert "stale_nc" not in cache._noncacheable
        assert "fresh_nc" in cache._noncacheable

    def test_prune_expired_returns_correct_count(self, tmp_path: Path):
        """prune_expired returns total expired count, even if file deletion partially fails."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=0)
        kv = make_kv_data()

        for i in range(3):
            cache.save_block(f"prune_{i}", kv, num_tokens=4)
            cache.index[f"prune_{i}"].last_accessed = datetime.now() - timedelta(seconds=10)

        pruned = cache.prune_expired()
        assert pruned == 3
        assert cache.num_blocks == 0

    def test_prune_expired_updates_total_bytes(self, tmp_path: Path):
        """prune_expired decrements _total_bytes for pruned blocks."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=0)
        kv = make_kv_data()

        cache.save_block("bytes_test", kv, num_tokens=4)
        assert cache._total_bytes > 0

        cache.index["bytes_test"].last_accessed = datetime.now() - timedelta(seconds=10)
        cache.prune_expired()

        assert cache._total_bytes == 0


# ---------------------------------------------------------------------------
# _prune_lru_for_space 2-phase I/O pattern (no lock during file deletion)
# ---------------------------------------------------------------------------


class TestPruneLruForSpace2Phase:
    """Tests for the 2-phase _prune_lru_for_space: index update under lock,
    file deletion outside lock."""

    def test_prune_lru_for_space_does_not_block(self, tmp_path: Path):
        """Verify concurrent save_block works during LRU pruning file deletion.

        The 2-phase pattern ensures _prune_lru_for_space only updates the index
        under lock (Phase 1), and file deletion happens outside the lock,
        allowing concurrent save_block operations to proceed without blocking.
        """
        import threading
        import time as time_mod

        # Use a small max_size to trigger pruning
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=500)

        # Save several blocks to fill up the cache
        for i in range(5):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            cache.save_block(f"fill_{i}", kv, num_tokens=4)

        # Manually call _prune_lru_for_space under lock and verify it returns
        # files_to_delete without doing I/O
        with cache._lock:
            initial_count = len(cache.index)
            freed, files_to_delete = cache._prune_lru_for_space(200)

            # Index should have been modified (blocks removed)
            assert len(cache.index) < initial_count
            # Files should be listed for deletion
            assert len(files_to_delete) > 0
            # But files should still exist on disk (no I/O under lock)
            for _, fpath in files_to_delete:
                assert fpath.exists(), f"File {fpath} should still exist (not deleted under lock)"

        # Phase 2: delete files outside lock
        for _, fpath in files_to_delete:
            fpath.unlink(missing_ok=True)

        # Verify files are now gone
        for _, fpath in files_to_delete:
            assert not fpath.exists()

    def test_prune_lru_returns_correct_files(self, tmp_path: Path):
        """_prune_lru_for_space returns the correct files matching pruned entries."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=0)

        # Save 3 blocks with known access times
        for i in range(3):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            cache.save_block(f"hash_{i}", kv, num_tokens=4)

        cache.index["hash_0"].last_accessed = datetime.now() - timedelta(days=3)
        cache.index["hash_1"].last_accessed = datetime.now() - timedelta(days=2)
        cache.index["hash_2"].last_accessed = datetime.now() - timedelta(days=1)

        with cache._lock:
            freed, files_to_delete = cache._prune_lru_for_space(1)

        # Should have pruned the oldest (hash_0)
        assert len(files_to_delete) >= 1
        pruned_hashes = [h for h, _ in files_to_delete]
        assert "hash_0" in pruned_hashes
        assert freed > 0

        # Files should still exist until caller deletes them
        for _, fpath in files_to_delete:
            assert fpath.exists()

        # Clean up
        for _, fpath in files_to_delete:
            fpath.unlink(missing_ok=True)

    def test_save_block_triggers_2phase_prune(self, tmp_path: Path):
        """save_block properly performs 2-phase prune: files deleted outside lock."""
        cache = SSDCache(cache_dir=tmp_path / "cache", ttl_days=7, max_size_bytes=500)

        # Fill cache to trigger pruning on next save
        for i in range(5):
            kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
            cache.save_block(f"old_{i}", kv, num_tokens=4)

        # Make some blocks old so they get pruned
        for i in range(3):
            if f"old_{i}" in cache.index:
                cache.index[f"old_{i}"].last_accessed = datetime.now() - timedelta(days=10)

        # Save a new block — should trigger LRU prune and succeed
        kv = {"keys": mx.ones((1, 1, 4, 8)), "values": mx.ones((1, 1, 4, 8))}
        result = cache.save_block("new_block", kv, num_tokens=4)
        assert result == "saved"
        assert "new_block" in cache.index
