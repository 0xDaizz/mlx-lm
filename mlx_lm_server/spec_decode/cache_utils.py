"""Cache utilities for speculative decoding.

Provides batched cache operations for per-sequence variable trimming
and capability detection.

GATE RESULT (2026-02-12): **A — argmax-safe per-sequence trim**

    BatchKVCache now supports trim_per_sequence(n) which correctly
    handles per-sequence variable trimming by adjusting both offset
    and _idx atomically. This is available on the upstream
    batchkvcache-per-seq-trim branch.

    The old approach (uniform trim + offset re-advance) was UNSAFE
    (Result B) because _idx is scalar and make_mask()/update_and_fetch()
    use _idx, not per-sequence offset.

    With trim_per_sequence(), each sequence's offset is decremented
    individually, and _idx is set to max(left_padding + offset),
    ensuring correct attention masking and KV data visibility.
"""

from __future__ import annotations

import logging
from typing import Any, List

import mlx.core as mx

logger = logging.getLogger(__name__)


def batch_variable_trim(
    cache_layers: List[Any],
    trim_amounts: mx.array,
) -> None:
    """Per-sequence variable trim using trim_per_sequence().

    Caller MUST verify trim_per_sequence is available before calling.
    Use can_per_seq_trim() to check at init time.

    Args:
        cache_layers: List of BatchKVCache instances (one per layer).
        trim_amounts: [B] int32 array of positions to trim per sequence.
            trim_amounts[i] = 0 means no trimming for sequence i.
    """
    if int(trim_amounts.max()) == 0:
        return

    for cache_layer in cache_layers:
        cache_layer.trim_per_sequence(trim_amounts)


def can_per_seq_trim(cache_layers: List[Any]) -> bool:
    """Check if cache supports per-sequence trim (Result A).

    Checks EVERY layer and recursively every CacheList child.
    Returns False if ANY leaf cache lacks trim_per_sequence.
    """
    if not cache_layers:
        return False
    for layer in cache_layers:
        if not _layer_supports_per_seq_trim(layer):
            return False
    return True


def _layer_supports_per_seq_trim(layer: Any) -> bool:
    """Check a single cache layer (recurse into CacheList)."""
    try:
        from mlx_lm.models.cache import CacheList
        if isinstance(layer, CacheList):
            return all(_layer_supports_per_seq_trim(c) for c in layer.caches)
    except ImportError:
        caches = getattr(layer, 'caches', None)
        if isinstance(caches, (tuple, list)):
            return all(_layer_supports_per_seq_trim(c) for c in caches)
    return hasattr(layer, 'trim_per_sequence')


def uniform_trim(
    cache_layers: List[Any],
    trim_amount: int,
) -> None:
    """Trim all sequences by the same amount.

    Simple wrapper around BatchKVCache.trim() for uniform trimming.

    Args:
        cache_layers: List of BatchKVCache instances.
        trim_amount: Number of positions to trim from all sequences.
    """
    if trim_amount <= 0:
        return
    for cache_layer in cache_layers:
        cache_layer.trim(trim_amount)


def get_cache_offsets(cache_layers: List[Any]) -> mx.array:
    """Get per-sequence offsets from the first cache layer.

    Useful for debugging and verification.

    Args:
        cache_layers: List of BatchKVCache instances.

    Returns:
        [B] int32 array of per-sequence cache offsets.
    """
    if not cache_layers:
        return mx.array([], dtype=mx.int32)
    return cache_layers[0].offset
