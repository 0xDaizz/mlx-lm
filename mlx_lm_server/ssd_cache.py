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
from datetime import datetime, timedelta
from pathlib import Path

import mlx.core as mx

from mlx_lm_server.types import SSDBlockMeta

logger = logging.getLogger(__name__)

CURRENT_HASH_VERSION = 1


class SSDCache:
    """SSD-backed KV cache for persisting evicted blocks.

    Attributes:
        cache_dir: Directory where safetensors files are stored.
        ttl_days: Time-to-live in days; blocks older than this are pruned.
        index: Maps block_hash (str) -> SSDBlockMeta.

    Note: This class is NOT thread-safe. All methods (save_block, load_block,
    prune_expired) must be called from a single thread (the inference thread).
    If concurrent access is needed (e.g., for Phase 8 tensor-parallel), add
    threading.Lock protection around index mutations.
    """

    def __init__(self, cache_dir: Path, ttl_days: int = 7) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.index: dict[str, SSDBlockMeta] = {}

        # Batch flush state: defer save_index() until flush interval
        self._index_dirty = False
        self._mutation_count = 0
        self._flush_interval = 10

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index if available
        self.load_index()

    def save_block(self, block_hash: str, kv_data: list[dict[str, mx.array]] | dict[str, mx.array]) -> None:
        """Save K/V tensors for a block to disk as a safetensors file.

        Args:
            block_hash: The hash identifying this block.
            kv_data: Either a list[dict] (one dict per layer, each with 'keys'
                and 'values') or a flat dict with 'keys'/'values' mx.arrays.
                The list[dict] format is flattened to
                ``{"layer_0_keys": ..., "layer_0_values": ..., ...}`` for storage.
        """
        filename = f"block_{block_hash}.safetensors"
        filepath = self.cache_dir / filename

        # Flatten list[dict] format to a single dict for safetensors
        if isinstance(kv_data, list):
            flat: dict[str, mx.array] = {}
            num_tokens = 0
            for i, layer_dict in enumerate(kv_data):
                for key, val in layer_dict.items():
                    flat[f"layer_{i}_{key}"] = val
                if num_tokens == 0 and "keys" in layer_dict:
                    num_tokens = layer_dict["keys"].shape[2]
        else:
            flat = kv_data
            num_tokens = kv_data["keys"].shape[2] if "keys" in kv_data else 0

        # Atomic write: save to temp file, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.cache_dir), suffix='.safetensors')
        try:
            os.close(tmp_fd)
            os.unlink(tmp_path)  # mx.save_safetensors needs to create the file itself
            mx.save_safetensors(tmp_path, flat)
            os.replace(tmp_path, str(filepath))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self.index[block_hash] = SSDBlockMeta(
            block_hash=block_hash,
            filepath=filepath,
            last_accessed=datetime.now(),
            num_tokens=num_tokens,
        )
        self.save_index()  # Immediate flush: new block must be indexed for durability

        logger.debug("Saved block %s to SSD: %s", block_hash, filepath)

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
        if block_hash not in self.index:
            return None

        meta = self.index[block_hash]
        filepath = meta.filepath

        if not filepath.exists():
            # Stale index entry — file was removed externally
            del self.index[block_hash]
            self.save_index()
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
            self.save_index()
            return None

        # mx.load returns a dict-like object; ensure we have standard dict
        raw = dict(arrays)

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

        Returns:
            Number of blocks pruned.
        """
        cutoff = datetime.now() - timedelta(days=self.ttl_days)
        to_prune = [
            bh for bh, meta in self.index.items() if meta.last_accessed < cutoff
        ]

        for block_hash in to_prune:
            meta = self.index[block_hash]
            try:
                if meta.filepath.exists():
                    meta.filepath.unlink()
                    logger.debug("Pruned expired block %s: %s", block_hash, meta.filepath)
            except OSError as e:
                logger.warning("Failed to delete expired block file %s: %s", meta.filepath, e)
            del self.index[block_hash]

        if to_prune:
            self.save_index()

        return len(to_prune)

    def _mark_dirty(self) -> None:
        """Mark index as dirty; flush to disk after _flush_interval mutations."""
        self._mutation_count += 1
        if self._mutation_count >= self._flush_interval:
            self.save_index()
            self._mutation_count = 0
            self._index_dirty = False
        else:
            self._index_dirty = True

    def flush(self) -> None:
        """Flush pending index changes to disk. Call during shutdown."""
        if self._index_dirty:
            self.save_index()
            self._mutation_count = 0
            self._index_dirty = False

    def save_index(self) -> None:
        """Persist the index to a JSON file in the cache directory.

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
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_index(self) -> None:
        """Load the index from the JSON file in the cache directory.

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
        return block_hash in self.index

    @property
    def num_blocks(self) -> int:
        """Number of blocks stored on SSD."""
        return len(self.index)
