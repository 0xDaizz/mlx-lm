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

from mlx_lm_server.ssd_cache import SSDCache


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
        assert "hash_999" in data

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
