"""SSD-backed KV cache tier for persistent block storage.

Evicted RAM blocks are saved as safetensors files keyed by block_hash.
Provides TTL-based pruning to reclaim disk space for stale blocks.

Design:
- Each block is saved as a separate .safetensors file containing K/V arrays.
- A JSON index maps block_hash -> SSDBlockMeta (filepath, timestamps).
- The index is loaded at startup and saved after mutations.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import mlx.core as mx

from mlx_lm_server.types import SSDBlockMeta

logger = logging.getLogger(__name__)

CURRENT_HASH_VERSION = 1


class SSDCache:
    """SSD-backed KV cache for persisting evicted blocks.

    Thread-safe: all public methods are protected by threading.Lock.

    Attributes:
        cache_dir: Directory where safetensors files are stored.
        ttl_days: Time-to-live in days; blocks older than this are pruned.
        index: Maps block_hash (str) -> SSDBlockMeta.
    """

    def __init__(self, cache_dir: Path, ttl_days: int = 7, flush_interval_s: float = 1.0, max_size_bytes: int = 0) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self._lock = threading.Lock()
        # Separate lock for mx.save_safetensors I/O: Metal command buffers
        # are not thread-safe, so concurrent save_safetensors calls crash.
        # This lock serializes only I/O while the main _lock is released.
        self._io_lock = threading.Lock()
        # Tracks block hashes currently being written in Phase 2 (outside _lock).
        # Used to distinguish in-progress writes from stale index entries.
        self._in_progress_writes: set[str] = set()
        self.index: dict[str, SSDBlockMeta] = {}

        # Batch flush state: defer save_index() until flush interval
        self._index_dirty = False
        self._mutation_count = 0
        self._flush_interval = 10
        self._flush_interval_s = flush_interval_s
        self._last_flush_time = time.monotonic()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index if available (no contention in __init__)
        self._load_index()

        # Stats counters (thread-safe reads via get_stats snapshot)
        self._stats: dict[str, int] = {
            "ssd_save_success": 0,
            "ssd_save_fail": 0,
            "ssd_save_dedup_skip": 0,
            "ssd_stale_index_cleaned": 0,
            "ssd_hash_collision_detected": 0,
        }

        # Non-cacheable collision isolation set: block_hash -> expiry timestamp
        self._noncacheable: dict[str, float] = {}

        # Disk size tracking
        self._max_size_bytes = max_size_bytes  # 0 = unlimited
        self._total_bytes = 0
        # Calculate initial total size from existing files
        for meta in self.index.values():
            try:
                p = Path(meta.filepath) if not isinstance(meta.filepath, Path) else meta.filepath
                if p.exists():
                    self._total_bytes += p.stat().st_size
            except OSError:
                pass

        # LRU prune stats
        self._stats["ssd_lru_prune_count"] = 0
        self._stats["ssd_lru_prune_bytes"] = 0

    def save_block(self, block_hash: str, kv_data: list[dict[str, mx.array]] | dict[str, mx.array], num_tokens: int | None = None) -> str:
        """Save K/V tensors for a block to disk as a safetensors file.

        Uses a 3-phase approach to minimize lock hold time:
        Phase 1 (under lock): Validate, check dedup/collision, prepare metadata,
            reserve index entry (acts as "write lock" for this hash).
        Phase 2 (outside lock): Perform disk I/O (write temp file + os.replace).
        Phase 3 (under lock): Finalize — if I/O failed, revert index entry.

        Args:
            block_hash: The hash identifying this block.
            kv_data: Either a list[dict] (one dict per layer, each with 'keys'
                and 'values') or a flat dict with 'keys'/'values' mx.arrays.
                The list[dict] format is flattened to
                ``{"layer_0_keys": ..., "layer_0_values": ..., ...}`` for storage.
            num_tokens: Optional token count for collision verification. If None,
                computed from kv_data shape.

        Returns:
            One of: ``"saved"``, ``"dedup"``, ``"collision"``, ``"error"``.
        """
        # Compute num_tokens from kv_data if not provided (no lock needed — read-only on input)
        computed_num_tokens: int
        if isinstance(kv_data, list):
            computed_num_tokens = 0
            for layer_dict in kv_data:
                if computed_num_tokens == 0 and "keys" in layer_dict:
                    computed_num_tokens = layer_dict["keys"].shape[2]
                    break
        else:
            computed_num_tokens = kv_data["keys"].shape[2] if "keys" in kv_data else 0

        if num_tokens is None:
            num_tokens = computed_num_tokens

        # Flatten list[dict] format to a single dict for safetensors (no lock needed)
        if isinstance(kv_data, list):
            flat: dict[str, mx.array] = {}
            for i, layer_dict in enumerate(kv_data):
                for key, val in layer_dict.items():
                    flat[f"layer_{i}_{key}"] = val
        else:
            flat = kv_data

        # Phase 1: validate, dedup check, reserve index entry
        filename = f"block_{block_hash}.safetensors"
        filepath = self.cache_dir / filename

        with self._lock:
            # Non-cacheable check (collision isolation)
            if block_hash in self._noncacheable:
                if time.time() < self._noncacheable[block_hash]:
                    return "collision"
                else:
                    del self._noncacheable[block_hash]

            # Dedup guard: check if already saved (or in-progress by another thread)
            if block_hash in self.index:
                entry = self.index[block_hash]
                entry_filepath = Path(entry.filepath) if not isinstance(entry.filepath, Path) else entry.filepath

                if not entry_filepath.exists():
                    if block_hash in self._in_progress_writes:
                        # Another thread is writing this block (Phase 2 in progress)
                        self._stats["ssd_save_dedup_skip"] += 1
                        return "dedup"
                    # Stale index entry — clean up, fall through to re-save
                    del self.index[block_hash]
                    self._stats["ssd_stale_index_cleaned"] += 1
                    logger.warning("Stale index entry cleaned: %s", block_hash)
                else:
                    # Collision verification via num_tokens
                    stored_num = entry.num_tokens
                    if stored_num is not None and num_tokens is not None and stored_num != num_tokens:
                        logger.warning(
                            "HASH COLLISION: hash=%s stored=%d new=%d — non-cacheable",
                            block_hash, stored_num, num_tokens,
                        )
                        self._noncacheable[block_hash] = time.time() + 3600
                        self._stats["ssd_hash_collision_detected"] += 1
                        return "collision"

                    # Normal dedup — touch last_accessed
                    entry.last_accessed = datetime.now()
                    self._mark_dirty()
                    self._stats["ssd_save_dedup_skip"] += 1
                    return "dedup"

            # Check capacity and prune if needed (Phase 1 only: index update)
            prune_files: list[tuple[str, Path]] = []
            if self._max_size_bytes > 0 and self._total_bytes >= self._max_size_bytes:
                _, prune_files = self._prune_lru_for_space(self._max_size_bytes // 10)  # free ~10% space

            # Reserve: write index entry now so concurrent save_block() for the
            # same hash sees "dedup" and skips. I/O happens outside the lock.
            meta = SSDBlockMeta(
                block_hash=block_hash,
                filepath=filepath,
                last_accessed=datetime.now(),
                num_tokens=num_tokens,
            )
            self.index[block_hash] = meta

        # Phase 1.5: delete pruned files outside the lock (2-phase I/O pattern)
        for pruned_hash, pruned_path in prune_files:
            try:
                pruned_path.unlink(missing_ok=True)
                logger.info("LRU-pruned block %s", pruned_hash)
            except OSError as e:
                logger.warning("Failed to LRU-prune block file %s: %s", pruned_path, e)

        # Phase 2: disk I/O outside _lock (but under _io_lock to serialize
        # mx.save_safetensors which is not thread-safe at the Metal layer).
        # Track in-progress writes so concurrent save_block() for the same
        # hash returns "dedup" instead of treating the reservation as stale.
        with self._lock:
            self._in_progress_writes.add(block_hash)
        io_success = False
        try:
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.cache_dir), suffix='.safetensors')
            except OSError as e:
                self._stats["ssd_save_fail"] += 1
                logger.error("Failed to create temp file for block %s: %s", block_hash, e)
                return "error"
            try:
                os.close(tmp_fd)
                os.unlink(tmp_path)  # mx.save_safetensors needs to create the file itself
                with self._io_lock:
                    mx.save_safetensors(tmp_path, flat)
                os.replace(tmp_path, str(filepath))
                io_success = True
            except Exception as e:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                self._stats["ssd_save_fail"] += 1
                logger.error("Failed to save block %s to SSD: %s", block_hash, e)
                return "error"
        finally:
            # Clean up reservation on failure; finalize on success
            with self._lock:
                self._in_progress_writes.discard(block_hash)
                if not io_success:
                    self.index.pop(block_hash, None)

        # Phase 3 (success): finalize — update total_bytes, save index
        with self._lock:
            try:
                file_size = filepath.stat().st_size
                self._total_bytes += file_size
            except OSError:
                pass
            self._save_index()  # Immediate flush: new block must be indexed for durability
            self._stats["ssd_save_success"] += 1

        logger.debug("Saved block %s to SSD: %s", block_hash, filepath)
        return "saved"

    def load_block(self, block_hash: str) -> list[dict[str, mx.array]] | dict[str, mx.array] | None:
        """Load K/V tensors for a block from disk.

        Updates the last_accessed timestamp on successful load.
        If the data was saved in flattened ``layer_N_key`` format (from a
        ``list[dict]`` input), it is unflattened back to ``list[dict]``.
        If the data was saved in plain ``dict`` format (``{"keys": ...,
        "values": ...}``), it is returned as a ``dict``.

        Args:
            block_hash: The hash identifying the block to load.

        Returns:
            Either a list of dicts (one per layer) when the multi-layer format
            was saved, a plain dict when the legacy format was saved, or None
            if not found.
        """
        with self._lock:
            if block_hash not in self.index:
                return None

            meta = self.index[block_hash]
            filepath = meta.filepath

            if not filepath.exists():
                # Stale index entry — file was removed externally
                del self.index[block_hash]
                self._save_index()
                self._stats["ssd_stale_index_cleaned"] += 1
                return None

            try:
                arrays = mx.load(str(filepath))
            except Exception as e:
                # Corrupted, truncated, or unreadable file — remove stale entry
                logger.warning(
                    "Failed to load block %s from %s: %s. Removing stale index entry.",
                    block_hash,
                    filepath,
                    e,
                )
                del self.index[block_hash]
                self._save_index()
                self._stats["ssd_stale_index_cleaned"] += 1
                return None

            # mx.load returns a dict-like object; ensure we have standard dict
            raw = dict(arrays)  # type: ignore[arg-type]  # mx.load returns SafeTensors (dict-like)

            # Detect format: "layer_N_key" keys indicate multi-layer format
            layer_keys = [k for k in raw if k.startswith("layer_")]
            if layer_keys:
                # Unflatten back to list[dict]
                layer_keys_sorted = sorted(
                    layer_keys, key=lambda k: (int(k.split("_")[1]), k)
                )
                layer_indices = sorted({int(k.split("_")[1]) for k in layer_keys_sorted})
                kv_data_list: list[dict[str, mx.array]] = []
                for li in layer_indices:
                    prefix = f"layer_{li}_"
                    layer_dict = {
                        k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)
                    }
                    kv_data_list.append(layer_dict)
                result: list[dict[str, mx.array]] | dict[str, mx.array] = kv_data_list
            else:
                # Legacy flat format — return as-is (dict)
                result = raw

            # Update access time
            meta.last_accessed = datetime.now()
            self._mark_dirty()

            logger.debug("Loaded block %s from SSD: %s", block_hash, filepath)
            return result

    def prune_expired(self) -> int:
        """Delete blocks that have exceeded their TTL.

        Uses a 2-phase approach to avoid holding the lock during file I/O:
        Phase 1 (under lock): collect expired entries, remove from index, save index.
        Phase 2 (outside lock): delete actual files on disk.

        Returns:
            Number of blocks pruned.
        """
        # Phase 1: collect expired entries and update index under lock
        to_delete: list[tuple[str, Path, int]] = []  # (hash, filepath, estimated_size)
        with self._lock:
            cutoff = datetime.now() - timedelta(days=self.ttl_days)
            expired_hashes = [
                bh for bh, meta in self.index.items() if meta.last_accessed < cutoff
            ]

            for block_hash in expired_hashes:
                meta = self.index[block_hash]
                filepath = meta.filepath if isinstance(meta.filepath, Path) else Path(meta.filepath)
                # Estimate file size while under lock for _total_bytes tracking
                file_size = 0
                try:
                    if filepath.exists():
                        file_size = filepath.stat().st_size
                except OSError:
                    pass
                to_delete.append((block_hash, filepath, file_size))
                self._total_bytes = max(0, self._total_bytes - file_size)
                del self.index[block_hash]

            # Also clean expired _noncacheable entries while we hold the lock
            now_ts = time.time()
            expired_nc = [h for h, exp in self._noncacheable.items() if now_ts >= exp]
            for h in expired_nc:
                del self._noncacheable[h]

            if expired_hashes:
                self._save_index()

        # Phase 2: delete files outside lock (doesn't block other SSD operations)
        deleted = 0
        for block_hash, filepath, _ in to_delete:
            try:
                filepath.unlink(missing_ok=True)
                deleted += 1
                logger.debug("Pruned expired block %s: %s", block_hash, filepath)
            except OSError as e:
                logger.warning("Failed to delete expired block file %s: %s", filepath, e)

        return len(to_delete)

    def _prune_lru_for_space(self, bytes_to_free: int) -> tuple[int, list[tuple[str, Path]]]:
        """Prune least-recently-accessed blocks to free disk space.

        Uses a 2-phase approach (matching prune_expired()):
        Phase 1 (under lock — this method): Collect LRU entries, remove from
            index, update _total_bytes and stats, save index.
        Phase 2 (caller, outside lock): Delete actual files on disk.

        Called internally when disk usage exceeds max_size_bytes.
        Must be called with self._lock held.

        Args:
            bytes_to_free: Target number of bytes to free.

        Returns:
            Tuple of (bytes_freed, files_to_delete) where files_to_delete is
            a list of (block_hash, filepath) pairs for the caller to unlink
            outside the lock.
        """
        if not self.index:
            return 0, []

        # Sort by last_accessed (oldest first)
        sorted_blocks = sorted(
            self.index.items(),
            key=lambda item: item[1].last_accessed,
        )

        freed = 0
        pruned_hashes: list[str] = []
        files_to_delete: list[tuple[str, Path]] = []
        for block_hash, meta in sorted_blocks:
            if freed >= bytes_to_free:
                break
            p = Path(meta.filepath) if not isinstance(meta.filepath, Path) else meta.filepath
            file_size = 0
            try:
                if p.exists():
                    file_size = p.stat().st_size
            except OSError:
                pass
            freed += file_size
            self._total_bytes = max(0, self._total_bytes - file_size)
            pruned_hashes.append(block_hash)
            files_to_delete.append((block_hash, p))

        for bh in pruned_hashes:
            del self.index[bh]

        if pruned_hashes:
            self._save_index()
            self._stats["ssd_lru_prune_count"] += len(pruned_hashes)
            self._stats["ssd_lru_prune_bytes"] += freed

        return freed, files_to_delete

    def _mark_dirty(self) -> None:
        """Mark index as dirty; flush after mutation count or time threshold."""
        self._mutation_count += 1
        self._index_dirty = True

        now = time.monotonic()
        count_due = self._mutation_count >= self._flush_interval
        time_due = (now - self._last_flush_time) >= self._flush_interval_s
        if count_due or time_due:
            self._save_index()
            # _save_index() resets _mutation_count, _index_dirty, _last_flush_time

    def flush(self) -> None:
        """Flush pending index changes to disk. Call during shutdown."""
        with self._lock:
            if self._index_dirty:
                self._save_index()
                # _save_index() resets _mutation_count, _index_dirty, _last_flush_time

    def save_index(self) -> None:
        """Persist the index to a JSON file in the cache directory (thread-safe)."""
        with self._lock:
            self._save_index()

    def _save_index(self) -> None:
        """Persist the index — caller must hold self._lock.

        Uses atomic write (write to temp file + os.replace) to prevent
        index corruption if the process crashes mid-write.

        Index format includes __metadata__ for version tracking (E3).
        """
        index_path = self.cache_dir / "index.json"
        blocks_serializable = {}
        for bh, meta in self.index.items():
            blocks_serializable[str(bh)] = {
                "block_hash": meta.block_hash,
                "filepath": str(meta.filepath),
                "last_accessed": meta.last_accessed.isoformat(),
                "num_tokens": meta.num_tokens,
            }
        serializable = {
            "__metadata__": {
                "index_version": 1,
                "hash_version": CURRENT_HASH_VERSION,
                "created_at": datetime.now().isoformat(),
            },
            "blocks": blocks_serializable,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.cache_dir), suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(serializable, f, indent=2)
            os.replace(tmp_path, str(index_path))
            self._last_flush_time = time.monotonic()
            self._mutation_count = 0
            self._index_dirty = False
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_index(self) -> None:
        """Load the index from the JSON file (thread-safe)."""
        with self._lock:
            self._load_index()

    def _load_index(self) -> None:
        """Load the index — caller must hold self._lock.

        Handles both new format (with __metadata__ + blocks) and legacy
        format (flat dict of block entries). On hash_version mismatch,
        invalidates the index and returns empty.
        """
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            self.index = {}
            return

        try:
            data = json.loads(index_path.read_text())

            # Detect new vs legacy format
            if "__metadata__" in data and "blocks" in data:
                metadata = data["__metadata__"]
                stored_hash_version = metadata.get("hash_version", 0)
                if stored_hash_version != CURRENT_HASH_VERSION:
                    logger.warning(
                        "SSD index hash_version mismatch (stored=%s, current=%s) "
                        "— invalidating cache",
                        stored_hash_version,
                        CURRENT_HASH_VERSION,
                    )
                    self.index = {}
                    return
                block_entries = data["blocks"]
            else:
                # Legacy format: entire dict is block entries
                block_entries = data

            self.index = {}
            for bh_str, entry in block_entries.items():
                bh = bh_str
                self.index[bh] = SSDBlockMeta(
                    block_hash=entry["block_hash"],
                    filepath=Path(entry["filepath"]),
                    last_accessed=datetime.fromisoformat(entry["last_accessed"]),
                    num_tokens=entry.get("num_tokens", 0),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load SSD cache index, starting fresh: %s", e)
            self.index = {}

    def has_block(self, block_hash: str) -> bool:
        """Check if a block exists in the SSD index (no I/O, index-only).

        Args:
            block_hash: The block hash to check.

        Returns:
            True if the block is in the SSD index, False otherwise.
        """
        with self._lock:
            return block_hash in self.index

    @property
    def num_blocks(self) -> int:
        """Number of blocks stored on SSD."""
        with self._lock:
            return len(self.index)

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of SSD cache statistics."""
        with self._lock:
            snapshot = dict(self._stats)
            snapshot["ssd_total_bytes"] = self._total_bytes
            snapshot["ssd_max_size_bytes"] = self._max_size_bytes
            return snapshot

    def validate_index(self) -> dict[str, int]:
        """Startup crash recovery: clean orphan files + remove missing entries.

        Returns:
            Dict with "orphans_cleaned" and "missing_cleaned" counts.
        """
        with self._lock:
            result = {"orphans_cleaned": 0, "missing_cleaned": 0}

            # 1. Find orphan files (on disk but not in index)
            indexed_files = {str(meta.filepath) for meta in self.index.values()}
            for f in self.cache_dir.glob("block_*.safetensors"):
                if str(f) not in indexed_files:
                    try:
                        f.unlink()
                        result["orphans_cleaned"] += 1
                        logger.info("Cleaned orphan SSD file: %s", f)
                    except OSError as e:
                        logger.warning("Failed to clean orphan file %s: %s", f, e)

            # 2. Find missing entries (in index but file doesn't exist)
            missing = []
            for bh, meta in self.index.items():
                filepath = meta.filepath if isinstance(meta.filepath, Path) else Path(meta.filepath)
                if not filepath.exists():
                    missing.append(bh)

            for bh in missing:
                del self.index[bh]
                result["missing_cleaned"] += 1
                self._stats["ssd_stale_index_cleaned"] += 1
                logger.info("Cleaned missing index entry: %s", bh)

            if missing:
                self._save_index()

            return result
