"""Cache utilities for speculative decoding.

Provides batched cache operations that the standard BatchKVCache
does not support, specifically per-sequence variable trimming.

GATE RESULT (2026-02-11): **B — offset manipulation is UNSAFE**

    Investigation findings:
    - BatchKVCache._idx is a scalar (global write cursor shared by all seqs)
    - BatchKVCache.offset is a per-sequence mx.array (logical position)
    - make_mask() uses _idx (scalar), NOT per-sequence offset
    - update_and_fetch() returns keys[..., :_idx, :] — sliced at _idx

    The batch_variable_trim approach (uniform trim + offset re-advance)
    is INVALID because:
    1. After trim(_idx -= max_trim), data beyond _idx is not returned
    2. Re-advancing offset for under-trimmed sequences doesn't restore
       the data — _idx still excludes those positions
    3. The attention mask (based on _idx) doesn't see the "restored" data
    4. Confirmed empirically: seq with deficit>0 gets max logit diff of 9.0
       and different argmax output vs reference

    Architecture decision: Must use uniform_trim(max_rejected) for all
    sequences. For sequences that accepted more tokens than the batch
    minimum, re-forward the delta tokens to rebuild cache entries.

    This function is kept for reference but should NOT be used in the
    speculative decoding engine. Use uniform_trim() instead.
"""

from __future__ import annotations

from typing import Any, List

import mlx.core as mx


def batch_variable_trim(
    cache_layers: List[Any],
    trim_amounts: mx.array,
) -> None:
    """Trim different amounts from each sequence's cache.

    WARNING: GATE test confirmed this approach is UNSAFE (Result B).
    Do NOT use in production. Use uniform_trim() instead, followed by
    re-forwarding tokens for under-trimmed sequences.

    Kept for testing and reference purposes only.

    BatchKVCache.trim(n) trims ALL sequences by the same amount n.
    This function attempts to apply different trim amounts per sequence
    by directly manipulating the per-sequence offset array.

    How it works:
    1. Trim ALL sequences by max(trim_amounts) using the standard trim()
    2. For sequences that needed less trimming, re-advance their offset
       to compensate

    Why it fails:
    - _idx (scalar) is decremented by max_trim for ALL sequences
    - update_and_fetch() returns keys[..., :_idx, :] — stale data beyond
      _idx is excluded regardless of per-sequence offset values
    - make_mask() uses _idx as the global offset, not per-sequence offset
    - Result: under-trimmed sequences see incorrect KV data

    Args:
        cache_layers: List of BatchKVCache instances (one per layer).
        trim_amounts: [B] int32 array of positions to trim per sequence.
            trim_amounts[i] = 0 means no trimming for sequence i.
    """
    if int(trim_amounts.max()) == 0:
        return  # Nothing to trim

    max_trim = int(trim_amounts.max())

    for cache_layer in cache_layers:
        # Approach 1: Uniform trim + selective re-advance
        # Trim all by max amount
        cache_layer.trim(max_trim)

        # Re-advance offsets for sequences that needed less trimming
        # deficit[i] = max_trim - trim_amounts[i]
        deficit = max_trim - trim_amounts
        cache_layer.offset = cache_layer.offset + deficit


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
