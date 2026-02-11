# PR Draft: Respect per-sequence offsets in BatchKVCache update_and_fetch/make_mask

## Target Repository
ml-explore/mlx-lm

## Branch
upstream/batchkvcache-per-seq-trim -> main

## Title
fix(cache): respect per-sequence offsets in BatchKVCache update_and_fetch/make_mask

## Description

### Problem

`BatchKVCache` stores per-sequence `offset` (an array of shape `(B,)`) but its core paths — `update_and_fetch()` and `make_mask()` — only use the scalar `_idx` cursor. This is a structural inconsistency: the data structure tracks per-sequence state that the fetch and mask paths ignore.

When all sequences advance in lockstep (the common case), `_idx` happens to equal all offsets, so this inconsistency is hidden. But when offsets diverge — as happens after per-sequence trimming in batched speculative decoding — the mismatch causes:

1. **`update_and_fetch()`** returns `keys[..., :_idx, :]`, which may truncate valid KV data for sequences whose `left_padding + offset > _idx`.
2. **`make_mask()`** passes `offset=_idx` to `create_causal_mask()`, ignoring sequences with different effective lengths. No `right_padding` is computed, so stale data positions are visible to attention.

### Solution

Make `update_and_fetch()` and `make_mask()` respect per-sequence offsets:

**1. `update_and_fetch()` — use `max(left_padding + offset)` for return slice bounds**

```python
end = int(mx.max(self.left_padding + self.offset).item())
end = max(self._idx, end)
return self.keys[..., :end, :], self.values[..., :end, :]
```

When all offsets are uniform (the normal case), `end == _idx` and behavior is identical.

**2. `make_mask()` — compute per-sequence `right_padding` to exclude stale positions**

```python
end = int(mx.max(self.left_padding + self.offset).item())
end = max(self._idx, end)
right_pad = end - (self.left_padding + self.offset)
if mx.max(right_pad).item() > 0:
    kwargs["right_padding"] = right_pad
return create_causal_mask(N, offset=end, left_padding=self.left_padding, **kwargs)
```

The `right_padding` parameter already exists in `create_causal_mask()` (used by `BatchRotatingKVCache`). When all offsets equal `_idx` (the normal case), `right_padding` is all-zeros and we pass `None` — zero overhead.

**3. `trim_per_sequence()` — new convenience method for per-sequence trimming**

```python
def trim_per_sequence(self, n: mx.array):
    n = mx.minimum(n, self.left_padding + self.offset)
    self.offset -= n
    self._idx = int(mx.max(self.left_padding + self.offset).item())
```

Trims each sequence by a different amount. This is the natural API for batched speculative decoding where different sequences accept different numbers of draft tokens.

### Motivation: Batched Speculative Decoding

The existing `speculative_generate_step()` in `generate.py` only supports single-sequence (batch=1) speculative decoding. Its `_rewind_cache()` calls `trim_prompt_cache()` which calls `trim(n)` uniformly. This works for batch=1 where there is only one `num_accept` value.

For batched generation (which `BatchKVCache` is designed to support via left-padded batching), each sequence may accept a different number of draft tokens. Without this fix, the only options are uniform trimming (which discards valid cache entries for some sequences, requiring an additional batch forward pass to rebuild them) or accepting only the minimum across all sequences (wasting speculative benefit).

With this fix, callers can use `trim_per_sequence()` directly, or implement per-sequence variable trim via offset manipulation:

```python
def batch_variable_trim(cache_layers, trim_amounts: mx.array):
    """Trim different amounts from each sequence in a batch."""
    for cache_layer in cache_layers:
        cache_layer.trim_per_sequence(trim_amounts)
```

After this, `update_and_fetch()` and `make_mask()` correctly handle the divergent offsets: the return slice includes all valid data, and the attention mask excludes stale positions per-sequence. This eliminates the need for an additional batch forward pass when sequences accept different numbers of draft tokens.

### Changes

| File | Lines | Change |
|------|-------|--------|
| `mlx_lm/models/cache.py` | `BatchKVCache.update_and_fetch()` | Use `max(_idx, max(left_padding + offset))` for return slice bounds |
| `mlx_lm/models/cache.py` | `BatchKVCache.make_mask()` | Compute `right_padding` from per-sequence offset divergence and pass to `create_causal_mask()` |
| `mlx_lm/models/cache.py` | `BatchKVCache.trim_per_sequence()` | New method: trim each sequence by a different amount |

No changes to `create_causal_mask()` — it already supports the `right_padding` parameter.

### Backward Compatibility

This change is fully backward-compatible:

- **Normal decode (batch or single):** All offsets remain equal to each other and to `_idx`. `max(left_padding + offset) == _idx`, so the return slice is unchanged. `right_padding` is all-zeros, so we pass `None` — `create_causal_mask()` receives the same arguments as before.
- **Existing `trim(n)` usage:** `trim()` decrements both `_idx` and all offsets uniformly. The invariant `max(left_padding + offset) == _idx` is preserved. No behavior change.
- **`speculative_generate_step()` (batch=1):** Continues to work identically since there is only one sequence.
- **`filter()`, `extend()`, `extract()`, `merge()`:** These methods use `_idx` for buffer management. The `_idx` value is unchanged by this PR; only the return slice and mask computation use `max(left_padding + offset)` when offsets diverge.

### Test Plan

1. **Unit test: per-sequence variable trim correctness**
   - Create a `BatchKVCache` with 3 sequences of different lengths
   - Forward pass extending all by 6 tokens (simulating draft+verify)
   - Apply `trim(5)` + per-sequence offset re-advance with deficits `[2, 0, 4]`
   - Forward pass with 1 new token
   - Verify: attention mask excludes stale positions, output matches sequential single-sequence processing

2. **Regression test: normal decode unchanged**
   - Run existing `BatchKVCache` tests to confirm zero behavior change
   - Specifically verify `update_and_fetch()` return shape and `make_mask()` output match pre-change behavior when all offsets are uniform

3. **Integration test: batched speculative decoding**
   - Run `speculative_generate_step()` on a small model to verify single-sequence spec decode is unaffected

### Performance Impact

- **Normal decode path:** Zero overhead. The `max(left_padding + offset)` reduction adds a single scalar operation that is only meaningful when offsets diverge. When offsets are uniform (the overwhelmingly common case), `right_padding` is all-zeros, we pass `None`, and `create_causal_mask()` receives identical arguments as before.
- **Per-sequence variable trim path:** Without this fix, batched speculative decoding must either trim all sequences to the minimum accepted count (wasting valid cache entries) or perform an additional batch forward pass to recompute over-trimmed entries. This fix eliminates that additional forward pass by allowing each sequence to retain its own accepted cache entries.

## Related Issues
- ml-explore/mlx-lm#499 (server batching support)
- ml-explore/mlx-lm#548 (persistent batch cache)

## Checklist
- [ ] Minimal diff (~25 lines added in cache.py, 0 lines in other files)
- [ ] Backward compatible (non-spec-decode paths identical — `max(left_padding + offset) == _idx` when offsets are uniform)
- [ ] Leverages existing `right_padding` support in `create_causal_mask()` — one new public method (`trim_per_sequence`)
- [ ] Test with real model validates per-sequence trim correctness
- [ ] No performance regression for normal decode (zero-cost when offsets are uniform)
- [ ] Self-contained in one file (`mlx_lm/models/cache.py`)
