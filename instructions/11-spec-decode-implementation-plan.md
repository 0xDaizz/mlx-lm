# Speculative Decoding: Implementation Plan

> **Document ID:** 11-spec-decode-implementation-plan
> **Status:** READY FOR IMPLEMENTATION
> **Target Branch:** `feature/spec-decode`
> **Base Branch:** `develop`
> **Owner:** spec-decode-agent (new role)
> **Prerequisite:** All prior phases complete (954 tests passing)

---

## Implementation Checklist

### Phase 0: Foundation + Cache Investigation (GATE)
- [ ] P0.1: Package skeleton (`mlx_lm_server/spec_decode/` directory structure)
- [ ] P0.2: SpecDecodeConfig dataclass with validation
- [ ] P0.3: CLI arguments + ServerConfig integration
- [ ] P1.1: BaseProposer ABC + ProposalResult + SpecResponse types
- [ ] **P1.5: GATE — BatchKVCache per-sequence trim investigation (real model)**
  - Result A: offset manipulation safe → per-sequence variable trim
  - Result B: unsafe → uniform trim + selective re-forward (interfaces change)

### Phase 1: N-gram Speculative Decoding
- [ ] P1.2: NGramProposer (linear search + tests)
- [ ] P1.3: NGramVerifier with greedy + threshold modes (+ tests)
- [ ] P1.4: SpecDecodeEngine orchestration (+ tests with real model)
- [ ] P1.6: Scheduler integration (_process_spec_responses + _batch_inference_step modification)
- [ ] P1.7: DynamicSpecController (EMA-based adaptive k + tests)
- [ ] P1.8: End-to-end test with Qwen3-4B-4bit (streaming + non-streaming)
- [ ] P1.9: Metrics endpoint (/v1/spec_decode/metrics)

### Phase 2: Draft Model Speculative Decoding
- [ ] P2.1: DraftModelProposer (load + propose with separate model)
- [ ] P2.2: BatchedRejectionSampler (Gumbel-max trick)
- [ ] P2.3: Draft model cache management (separate KV cache lifecycle)
- [ ] P2.4: Engine extension for draft model mode
- [ ] P2.5: CLI args for draft model (--draft-model, --num-draft-tokens)
- [ ] P2.6: TP-aware draft placement (rank 0 local draft, broadcast proposals)
- [ ] P2.7: End-to-end test with real draft + target models

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Snapshot](#2-current-architecture-snapshot)
3. [Conflict Analysis and Resolutions](#3-conflict-analysis-and-resolutions)
4. [Package Structure](#4-package-structure)
5. [Phase 0: Foundation](#5-phase-0-foundation)
6. [Phase 1: N-gram Proposer + Greedy Verifier](#6-phase-1-n-gram-proposer--greedy-verifier)
7. [Phase 2: Draft Model Proposer](#7-phase-2-draft-model-proposer)
8. [2-Node Tensor Parallel Design Considerations](#8-2-node-tensor-parallel-design-considerations)
9. [Testing Strategy](#9-testing-strategy)
10. [Implementation Schedule](#10-implementation-schedule)
11. [Risk Register](#11-risk-register)
12. [Key Design Decisions](#12-key-design-decisions)
13. [File Ownership](#13-file-ownership)
14. [Reference Materials](#14-reference-materials)

---

## 1. Executive Summary

This plan integrates speculative decoding into `mlx-lm-server`, enabling the server
to generate multiple tokens per inference step. The approach follows a phased rollout:

1. **Phase 0** -- Foundation types, config, and package skeleton
2. **Phase 1** -- N-gram proposer with greedy verification, engine, scheduler integration, dynamic controller, metrics (CPU-only proposer, zero GPU overhead)
3. **Phase 2** -- Draft model proposer with rejection sampling, TP-aware draft placement

Phase 1 is the priority. It requires NO changes to `mlx_lm/` (upstream code) and
delivers meaningful speedups for repetitive generation tasks (code, translation,
templated output) with zero additional memory cost. All tests use the real
`mlx-community/Qwen3-4B-4bit` model (locally available at `Qwen3-4B-4bit/`).

### Expected Performance Gains

| Proposer | Acceptance Rate | Tokens/Step | Speedup (low batch) | Memory Overhead |
|----------|----------------|-------------|---------------------|-----------------|
| N-gram   | 40-70% (repetitive) | 2-4 | 1.5-2.5x | ~0 |
| N-gram   | 10-30% (creative) | 1.1-1.5 | 1.0-1.2x | ~0 |
| Draft    | 60-80% | 3-5 | 2.0-3.0x | +draft model size |

---

## 2. Current Architecture Snapshot

Understanding the existing architecture is critical. Spec decode must integrate
without breaking any of the 954 existing tests.

### 2.1 Data Flow (Normal Decode)

```
HTTP Request
    |
    v
FastAPI Server (server.py)
    |  submit_request()
    v
Scheduler (scheduler.py)
    |  _insert_new_requests_batch() --> BatchGenerator.insert()
    |  _batch_inference_step()      --> BatchGenerator.next()
    v
BatchGenerator (mlx_lm/generate.py)
    |  _next() --> model(y[:, None], cache=batch_cache)
    |           --> sample(logits[:, -1, :])
    v
Response(uid, token, logprobs, finish_reason, prompt_cache)
    |
    v
Scheduler._process_batch_responses()
    |  --> TokenEvent per response
    v
Streaming queue --> HTTP response
```

### 2.2 BatchGenerator Internals

**File:** `mlx_lm/generate.py`, line 930

```python
class BatchGenerator:
    class Response:
        uid: int           # Sequence identifier
        token: int         # Single generated token
        logprobs: mx.array # Log probabilities
        finish_reason: Optional[str]
        prompt_cache: Callable[[], List[Any]]

    active_batch: Optional[Batch]  # Current active batch (Batch dataclass)
    model: nn.Module               # The target model

    def next(self) -> List[Response]:
        # 1. Process any unprocessed prompts (prefill)
        # 2. Run ONE decode step: model(y[:, None], cache=batch.cache)
        # 3. Sample ONE token per sequence
        # 4. Return List[Response], one per active sequence
```

**Key constraint:** `next()` always generates exactly 1 token per sequence. The decode
step is hardcoded to `y[:, None]` -- always feeding [B, 1] shaped input.

**The Batch dataclass** (line 843):
```python
@dataclass
class Batch:
    uids: List[int]              # [B] sequence UIDs
    y: mx.array                  # [B] last generated token per sequence
    logprobs: mx.array           # [B, vocab] log probs from last step
    max_tokens: List[int]        # [B] max tokens per sequence
    num_tokens: List[int]        # [B] tokens generated so far
    cache: List[Any]             # Per-layer BatchKVCache instances
    samplers: List[Any]          # Per-sequence samplers
    logits_processors: List[Any] # Per-sequence logit processors
    tokens: List[mx.array]       # Per-sequence full token history
```

### 2.3 BatchKVCache Internals

**File:** `mlx_lm/models/cache.py`, line 806

```python
class BatchKVCache:
    keys: mx.array    # [B, n_kv_heads, S, k_head_dim]  -- S grows in steps of 256
    values: mx.array  # [B, n_kv_heads, S, v_head_dim]
    offset: mx.array  # [B] per-sequence offset (effective sequence length)
    _idx: int         # Global write position in the pre-allocated buffer

    def update_and_fetch(self, keys, values):
        # Extends cache by keys.shape[2] positions
        # Returns ALL cached keys/values up to _idx
        # offset += keys.shape[2]  (per-sequence)

    def trim(self, n):
        # Trims last n positions from ALL sequences uniformly
        # self._idx -= n
        # self.offset -= n  (subtracts n from every element)
        # Returns: number of tokens actually trimmed
```

**CRITICAL:** `trim(n)` trims ALL sequences by the same amount. There is no built-in
per-sequence trim. This is the primary technical challenge for batched spec decode.

### 2.4 Scheduler Internals

**File:** `mlx_lm_server/scheduler.py`, line 883

```python
def _batch_inference_step(self):
    # 1. Distributed bus sync (if TP mode)
    # 2. Process cancellations
    # 3. Insert new requests via BatchGenerator.insert()
    # 4. Call BatchGenerator.next() --> List[Response]
    # 5. Process responses via _process_batch_responses()
    # 6. Emit token events
    # 7. Remove finished UIDs from BatchGenerator
    # 8. Store finished caches (block decomposition)
    # 9. SSD pruning
    # 10. Clean up finished sequences
```

**Key observation:** The scheduler sees one `Response` per sequence per step. Spec
decode must change this to potentially multiple tokens per sequence per step.

### 2.5 Existing Spec Decode in mlx-lm

**File:** `mlx_lm/generate.py`, line 469

The upstream `speculative_generate_step()` function already implements spec decode
for **single-stream (batch=1)** generation. Key patterns we can reuse:

```python
def speculative_generate_step(prompt, model, draft_model, ...):
    # Creates separate model_cache and draft_cache
    # _draft_generate(): runs draft model k times autoregressively
    # _step(model, model_cache, ...): verifies with target model
    # _rewind_cache(num_draft, num_accept): trims caches on rejection
    # Yields: (token, logprobs, is_from_draft) one at a time
```

The `_rewind_cache` function uses `cache.trim_prompt_cache()` which calls the cache's
`trim(n)` method. This works for batch=1 because there is only one sequence.

---

## 3. Conflict Analysis and Resolutions

### Conflict 1: BatchGenerator 1-Token Constraint

**Problem:** `BatchGenerator.next()` generates exactly 1 token per sequence. Spec
decode needs up to k+1 tokens per verification step.

**Resolution:** Bypass `BatchGenerator.next()` for the decode step. Keep using it for
prefill and prompt processing. For spec decode steps, directly call `model()` with
multi-token input using the cache from `BatchGenerator.active_batch`.

```
Normal flow:
  Scheduler --> BatchGenerator.next() --> 1 token/seq

Spec decode flow:
  Scheduler --> BatchGenerator handles prefill via insert()
            --> SpecDecodeEngine.speculative_step()
                --> proposer.propose()       [get draft tokens]
                --> model(draft_tokens, cache=batch.cache)  [verify]
                --> verifier.verify()        [accept/reject]
                --> update batch state       [sync with BatchGenerator]
            --> multiple tokens/seq
```

**Approach in detail:**
1. `BatchGenerator` continues to own the `Batch` object and its cache
2. `SpecDecodeEngine` accesses `batch_generator.active_batch` to get the model cache
3. After spec step, engine updates `batch.y`, `batch.tokens`, `batch.num_tokens`
4. `BatchGenerator.next()` is NOT called during spec decode -- the engine replaces it

### Conflict 2: Per-Sequence KV Cache Rollback

**Problem:** After verification, different sequences in a batch may accept different
numbers of draft tokens. We need to trim different amounts from each sequence's cache.
But `BatchKVCache.trim(n)` trims ALL sequences by the same amount `n`.

**Resolution:** A two-phase rollback approach.

```
Given: k=5 draft tokens, acceptance results [3, 5, 1, 5] for 4 sequences
       Model forward extended cache by (k+1)=6 positions for all sequences
       Need to trim: [3, 0, 5, 0] positions respectively

Phase 1: Uniform trim
  max_trim = max(trim_amounts) = 5
  batch_cache.trim(max_trim)  --> trims ALL sequences by 5
  Now: sequences that needed less trimming lost too many positions

Phase 2: Re-extend under-trimmed sequences
  For sequences that needed less than max_trim trimming:
    deficit[i] = max_trim - trim_amounts[i]
    Re-forward model with the tokens that were over-trimmed
    to rebuild those cache positions

  Example:
    seq 0: needed trim 3, got trimmed 5 --> deficit = 2
           Re-forward 2 tokens to rebuild those cache entries
    seq 1: needed trim 0, got trimmed 5 --> deficit = 5
           Re-forward 5 tokens to rebuild
    seq 2: needed trim 5, got trimmed 5 --> deficit = 0 (no action)
    seq 3: needed trim 0, got trimmed 5 --> deficit = 5
           Re-forward 5 tokens to rebuild
```

**HOWEVER**, this two-phase approach is expensive because re-forwarding negates the
spec decode benefit. A better approach:

**Alternative Resolution: Direct offset manipulation.**

```python
def batch_variable_trim(cache_layers, trim_amounts: mx.array):
    """
    Trim different amounts from each sequence in a BatchKVCache.

    Instead of using the built-in trim() which is uniform,
    we directly manipulate the offset array.

    Args:
        cache_layers: List of BatchKVCache (one per model layer)
        trim_amounts: mx.array [B] -- how many positions to trim per sequence
    """
    for cache_layer in cache_layers:
        cache_layer.offset = cache_layer.offset - trim_amounts
        # _idx remains at the global maximum -- stale data beyond offset
        # is ignored because attention uses offset to compute mask bounds
```

**INVESTIGATION REQUIRED:** We must verify that `BatchKVCache.update_and_fetch()`
and the attention mechanism respect `offset` properly when there is stale data beyond
the offset in the pre-allocated buffer. Looking at the code:

- `update_and_fetch()` writes at position `_idx` and increments both `_idx` and
  `offset`. It returns `keys[..., :_idx, :]` -- ALL data up to `_idx`.
- The attention mask is created based on `offset` values, not `_idx`.
- `make_mask()` uses `self.offset` to determine valid positions per sequence.

If the attention mask correctly excludes positions beyond each sequence's `offset`,
then direct offset manipulation is safe. The stale data at positions between
`offset[i]` and `_idx` will be masked out during attention computation.

**If offset manipulation is NOT sufficient** (because `_idx` determines the returned
slice, not per-sequence offsets), then we need a different approach:

**Fallback Resolution: Trim to minimum and use targeted re-forward.**

```
max_accepted = max(num_accepted)
min_accepted = min(num_accepted)
trim_all = (k + 1) - min_accepted  -- trim to the minimum accepted count

batch_cache.trim(trim_all)

For sequences that accepted MORE than min_accepted:
  Re-forward (num_accepted[i] - min_accepted) tokens
  This rebuilds only the delta between min and actual accepted count
```

This is cheaper than the full two-phase approach because the re-forward is smaller.

**Simplest Resolution for Phase 1 (recommended):** Process spec decode at batch
granularity but with uniform acceptance. Accept the MINIMUM number of tokens across
the batch, trim uniformly, and discard tokens accepted by individual sequences beyond
the batch minimum. This loses some benefit but avoids per-sequence trim entirely.

```python
# Uniform batch acceptance (Phase 1 simplification)
min_accepted = min(num_accepted)
accept_tokens = accepted_tokens[:, :min_accepted + 1]  # +1 for bonus/correction
trim_amount = (k + 1) - (min_accepted + 1)
if trim_amount > 0:
    for layer_cache in batch_cache:
        layer_cache.trim(trim_amount)
```

**Phase 1 FINAL DECISION:** Start with **individual sequence processing** -- remove
the sequence from the batch, apply spec decode (essentially batch=1 per sequence),
then add it back. This avoids the per-sequence trim problem entirely while letting us
validate the full pipeline. Optimize to true batched spec decode in a later phase.

Wait, that is also suboptimal. Let me reconsider.

**ACTUAL Phase 1 FINAL DECISION:** Use the **direct offset manipulation** approach.
The key insight is that `_idx` is the global write cursor for the pre-allocated buffer.
After a multi-token forward pass, `_idx` advances by `max_input_len` for all sequences.
But `offset` is per-sequence, tracking the logical end of each sequence's cache. When
we trim, we only need to adjust `offset[i]` for each sequence. The next forward pass
will write at `_idx` (after the stale data), and attention will correctly mask based
on each sequence's `offset[i]`.

However, there is a subtle issue: `update_and_fetch()` returns `keys[..., :_idx, :]`,
which includes stale data for sequences with lower offsets. The attention mechanism
must use the mask (based on `offset`) to ignore these positions. This is how batched
attention works already -- left-padded sequences have zeros in their padding positions,
and the mask excludes them.

**ACTION ITEM for Sprint 2:** Write a test that verifies this behavior before
implementing the full engine. The test should:
1. Create a `BatchKVCache` with two sequences of different lengths
2. Run a forward pass extending both by k+1 tokens
3. Manually adjust `offset` to trim different amounts per sequence
4. Run another forward pass and verify outputs match non-spec decode

### Conflict 3: Streaming Token Emission

**Problem:** Current streaming emits 1 `TokenEvent` per sequence per step. Spec decode
may produce up to k+1 tokens per sequence per step.

**Resolution:** Emit multiple `TokenEvent` instances in rapid succession for each
accepted token. The existing streaming infrastructure (bounded queue with
`maxsize=256`) can handle bursts. The scheduler's `_emit_tokens()` already accepts a
list of events and iterates through them.

```python
# In _process_spec_responses():
for i, seq in enumerate(active_sequences):
    for j in range(num_accepted[i] + 1):  # +1 for bonus/correction
        token = int(accepted_tokens[i, j])
        if token == PLACEHOLDER_TOKEN_ID:
            break
        events.append(TokenEvent(
            request_id=seq.request_id,
            token_id=token,
            token_text=detokenize(token),
            finish_reason=check_stop(seq, token),
        ))
        if events[-1].finish_reason is not None:
            break  # Stop emitting on finish
```

### Conflict 4: Block Allocation Timing

**Problem:** KV cache blocks are created at sequence completion via
`_store_finished_caches()`. Spec decode changes token generation rate but not the
timing of block allocation.

**Resolution:** No change needed. Block allocation remains at sequence completion.
The spec decode engine generates tokens faster, but the block decomposition still
operates on the full completed cache. The only consideration is that `num_tokens`
tracking must account for multi-token steps.

### Conflict 5: Distributed Mode Consistency

**Problem:** In tensor-parallel mode, all ranks must produce identical results.
Non-deterministic sampling could cause rank divergence.

**Resolution by proposer type:**

| Proposer | Determinism Guarantee | Mechanism |
|----------|----------------------|-----------|
| N-gram | Fully deterministic | CPU-only context lookup, no sampling |
| Draft model | Seed-synchronized | Use fixed `mx.random.key()` per step across ranks |

The spec decode engine operates entirely within the inference thread. No bus
interaction is needed during the speculative step itself -- only the final accepted
tokens need to be communicated via the existing bus pattern. See Section 8 for
detailed 2-node TP design considerations.

### Conflict 6: Draft Model + Target Model Resource Contention

**Problem:** On Apple Silicon unified memory, both models share Metal. Running the
draft model may evict target model data from the Metal buffer pool.

**Resolution:**
- N-gram (Phase 1): CPU-only, zero GPU contention
- Draft model (Phase 2): Load as 4-bit quantized, much smaller than target model
- MLX lazy evaluation: Draft model k steps fused into one compute graph -- minimal
  memory churn
- Monitoring: Track `mx.get_peak_memory()` delta with/without draft model

---

## 4. Package Structure

```
mlx_lm_server/
  spec_decode/
  +-- __init__.py               # Package init, public API exports
  +-- config.py                 # SpecDecodeConfig dataclass
  +-- proposer/
  |   +-- __init__.py           # Proposer sub-package init
  |   +-- base.py               # BaseProposer ABC, ProposalResult, factory function
  |   +-- ngram.py              # NGramProposer (Phase 1)
  |   +-- draft_model.py        # DraftModelProposer (Phase 2)
  +-- verifier.py               # NGramVerifier (greedy/threshold)
  +-- rejection_sampler.py      # BatchedRejectionSampler (Phase 2)
  +-- engine.py                 # SpecDecodeEngine (orchestrator)
  +-- cache_utils.py            # Batched cache trim, rollback helpers
  +-- controller.py             # DynamicSpecController (stats + adaptive k)

tests/
  spec_decode/
  +-- __init__.py
  +-- test_config.py            # SpecDecodeConfig validation
  +-- test_ngram_proposer.py    # N-gram matching correctness
  +-- test_ngram_verifier.py    # Greedy/threshold verification
  +-- test_cache_utils.py       # Batched cache operations
  +-- test_spec_engine.py       # Full propose-verify-accept cycle (real model)
  +-- test_controller.py        # Dynamic control EMA + thresholds
  +-- test_scheduler_integration.py  # End-to-end with real model
  +-- test_rejection_sampler.py # Phase 2: rejection sampling correctness
```

---

## 5. Phase 0: Foundation

### P0.1: Create Package Skeleton

**Task:** Create all directories and `__init__.py` files.

**Files to create:**
- `mlx_lm_server/spec_decode/__init__.py`
- `mlx_lm_server/spec_decode/config.py`
- `mlx_lm_server/spec_decode/proposer/__init__.py`
- `mlx_lm_server/spec_decode/proposer/base.py`
- `mlx_lm_server/spec_decode/proposer/ngram.py`
- `mlx_lm_server/spec_decode/verifier.py`
- `mlx_lm_server/spec_decode/rejection_sampler.py`
- `mlx_lm_server/spec_decode/engine.py`
- `mlx_lm_server/spec_decode/cache_utils.py`
- `mlx_lm_server/spec_decode/controller.py`
- `tests/spec_decode/__init__.py`

**`mlx_lm_server/spec_decode/__init__.py` content:**

```python
"""Speculative decoding module for mlx-lm-server.

Provides proposer-verifier framework for multi-token generation:
- N-gram proposer (CPU-only, zero overhead)
- Draft model proposer (small model generates candidates)
- Dynamic controller (adaptive speculation depth)
"""

from mlx_lm_server.spec_decode.config import SpecDecodeConfig

__all__ = ["SpecDecodeConfig"]
```

### P0.2: SpecDecodeConfig

**File:** `mlx_lm_server/spec_decode/config.py`

```python
"""Configuration for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SpecDecodeConfig:
    """Speculative decoding configuration.

    Controls proposer mode, speculation depth, dynamic control thresholds,
    and per-proposer settings. Can be instantiated from CLI args, config
    file, or per-request override via extra_body.

    Attributes:
        mode: Proposer type. "none" disables spec decode entirely.
        num_speculative_tokens: Number of draft tokens (k) per spec step.
            Higher k = more potential gain but higher overhead on rejection.
            Typical range: 3-7 for n-gram, 2-5 for draft model.
        disable_by_batch_size: Auto-disable spec decode when batch size
            reaches this threshold. At high batch sizes, the overhead of
            verification outweighs the benefit. Set to 0 to never disable.
        ngram_max: Maximum n-gram size for context matching (4 = try
            4-gram, then 3-gram, ..., down to ngram_min).
        ngram_min: Minimum n-gram size. Setting to 1 allows unigram
            matching (most aggressive, lowest precision).
        ngram_prompt_lookup: If True, search the full context (prompt +
            generated tokens). If False, search only generated tokens.
        draft_model_path: HuggingFace repo or local path for draft model.
            Required when mode="draft".
        draft_model_quantize: Quantization for draft model ("4bit", "8bit",
            or None for fp16). Recommend "4bit" to minimize memory impact.
        dynamic_enabled: Enable dynamic speculation control (adaptive k
            and auto-disable based on acceptance rate).
        acceptance_rate_threshold: Minimum EMA acceptance rate to keep
            spec decode active. Below this, falls back to normal decode.
        acceptance_rate_ema_alpha: Smoothing factor for acceptance rate
            exponential moving average. Lower = more smoothing.
        adaptive_k: Adjust k dynamically based on acceptance rate EMA.
            High acceptance -> increase k, low acceptance -> decrease k.
    """

    mode: Literal["none", "ngram", "draft"] = "none"
    num_speculative_tokens: int = 5
    disable_by_batch_size: int = 8

    # N-gram settings
    ngram_max: int = 4
    ngram_min: int = 1
    ngram_prompt_lookup: bool = True

    # Draft model settings (Phase 2)
    draft_model_path: Optional[str] = None
    draft_model_quantize: Optional[str] = None

    # Dynamic control
    dynamic_enabled: bool = True
    acceptance_rate_threshold: float = 0.3
    acceptance_rate_ema_alpha: float = 0.1
    adaptive_k: bool = True

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If required fields are missing for the selected mode,
                or if numeric values are out of valid range.
        """
        if self.mode == "draft" and not self.draft_model_path:
            raise ValueError(
                "--draft-model-path is required when --spec-decode=draft"
            )
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )
        if self.num_speculative_tokens > 20:
            raise ValueError(
                f"num_speculative_tokens must be <= 20, got {self.num_speculative_tokens}"
            )
        if self.ngram_max < self.ngram_min:
            raise ValueError(
                f"ngram_max ({self.ngram_max}) must be >= ngram_min ({self.ngram_min})"
            )
        if self.ngram_min < 1:
            raise ValueError(
                f"ngram_min must be >= 1, got {self.ngram_min}"
            )
        if not (0.0 <= self.acceptance_rate_threshold <= 1.0):
            raise ValueError(
                f"acceptance_rate_threshold must be in [0, 1], got {self.acceptance_rate_threshold}"
            )
        if not (0.0 < self.acceptance_rate_ema_alpha <= 1.0):
            raise ValueError(
                f"acceptance_rate_ema_alpha must be in (0, 1], got {self.acceptance_rate_ema_alpha}"
            )
        if self.disable_by_batch_size < 0:
            raise ValueError(
                f"disable_by_batch_size must be >= 0, got {self.disable_by_batch_size}"
            )
```

**Tests:** `tests/spec_decode/test_config.py`

```
test_default_config_is_valid
test_ngram_mode_no_errors
test_draft_mode_requires_model_path
test_invalid_num_speculative_tokens
test_ngram_max_less_than_min
test_acceptance_rate_bounds
test_ema_alpha_bounds
test_disable_by_batch_size_negative
```

### P0.3: CLI Arguments

**File to modify:** `mlx_lm_server/server.py`

Add a `--spec-decode` argument group to the CLI parser. These arguments construct a
`SpecDecodeConfig` that is passed through `ServerConfig` to the scheduler.

**ServerConfig addition** (in `mlx_lm_server/config.py`):

```python
# Add to ServerConfig dataclass:

    # Speculative Decoding
    spec_decode_mode: str = "none"                 # none | ngram | draft
    spec_decode_num_tokens: int = 5                # k
    spec_decode_disable_batch_size: int = 8        # auto-OFF threshold
    spec_decode_ngram_max: int = 4
    spec_decode_ngram_min: int = 1
    spec_decode_ngram_prompt_lookup: bool = True
    spec_decode_draft_model: str | None = None
    spec_decode_draft_quantize: str | None = None
    spec_decode_dynamic: bool = True
    spec_decode_acceptance_threshold: float = 0.3
    spec_decode_adaptive_k: bool = True
```

**CLI arguments** (add to `build_parser()` in `server.py`):

```python
spec_group = parser.add_argument_group("Speculative Decoding")
spec_group.add_argument(
    "--spec-decode",
    choices=["none", "ngram", "draft"],
    default="none",
    help="Speculative decoding mode (default: none)",
)
spec_group.add_argument(
    "--num-speculative-tokens",
    type=int,
    default=5,
    help="Number of draft tokens per speculation step (default: 5)",
)
spec_group.add_argument(
    "--spec-decode-disable-batch-size",
    type=int,
    default=8,
    help="Auto-disable spec decode when batch size >= this (default: 8)",
)
spec_group.add_argument("--ngram-max", type=int, default=4)
spec_group.add_argument("--ngram-min", type=int, default=1)
spec_group.add_argument(
    "--no-ngram-prompt-lookup",
    dest="ngram_prompt_lookup",
    action="store_false",
    default=True,
)
spec_group.add_argument("--draft-model-path", type=str, default=None)
spec_group.add_argument("--draft-model-quantize", type=str, default=None)
spec_group.add_argument(
    "--no-spec-decode-dynamic",
    dest="spec_decode_dynamic",
    action="store_false",
    default=True,
)
spec_group.add_argument(
    "--spec-decode-acceptance-threshold",
    type=float,
    default=0.3,
)
spec_group.add_argument(
    "--no-adaptive-k",
    dest="adaptive_k",
    action="store_false",
    default=True,
)
```

---

## 6. Phase 1: N-gram Proposer + Greedy Verifier

This is the highest-priority phase. It delivers meaningful speedups with zero memory
overhead and no changes to the upstream `mlx_lm/` codebase.

### Overview

```
+-----------------+    +-----------------+    +------------------+
| NGramProposer   |    | Target Model    |    | NGramVerifier    |
| (CPU, O(n*k))   |    | (GPU, 1 fwd)    |    | (GPU, vectorized)|
|                 |    |                 |    |                  |
| context lookup  |--->| verify_input    |--->| argmax match     |
| match n-gram    |    | [B, k+1] fwd    |    | cumprod mask     |
| return k tokens |    | logits [B,k+1,V]|    | accept/reject    |
+-----------------+    +-----------------+    +------------------+
        |                                              |
        |          +-------------------+               |
        +--------->| SpecDecodeEngine  |<--------------+
                   | orchestrate flow  |
                   | update batch state|
                   | emit token events |
                   +-------------------+
```

### P1.1: BaseProposer and ProposalResult

**File:** `mlx_lm_server/spec_decode/proposer/base.py`

```python
"""Base proposer interface and factory function.

All proposer implementations (n-gram, draft model) must inherit
from BaseProposer and implement the propose() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_lm_server.spec_decode.config import SpecDecodeConfig
    from mlx_lm_server.types import SequenceState


@dataclass
class ProposalResult:
    """Output of a proposer's propose() call.

    Attributes:
        draft_tokens: [batch, k] int32 array of proposed token IDs.
            Padded with 0 for sequences with fewer than k proposals.
        draft_probs: [batch, k, vocab] float32 array of draft token
            probabilities. None for n-gram proposer (no draft probs).
            Required for rejection sampling (draft model).
        proposal_lens: [batch] int32 array of actual proposal length
            per sequence. Sequences with proposal_lens[i] == 0 had no
            viable proposals and should use normal decode for that step.
    """

    draft_tokens: mx.array
    draft_probs: Optional[mx.array]
    proposal_lens: mx.array


class BaseProposer(ABC):
    """Abstract base class for all speculative decoding proposers.

    Subclasses must implement:
    - propose(): Generate draft tokens for a batch of sequences
    - needs_draft_probs: Whether the proposer provides draft probabilities
    - requires_gpu: Whether the proposer uses Metal GPU
    """

    @abstractmethod
    def propose(
        self,
        sequences: List[SequenceState],
        k: int,
    ) -> Optional[ProposalResult]:
        """Generate draft tokens for all sequences in the batch.

        Args:
            sequences: List of SequenceState objects in decode phase.
                Each has token_ids (prompt + generated), output_tokens,
                and request_id attributes.
            k: Number of draft tokens to generate per sequence.

        Returns:
            ProposalResult with draft tokens and metadata, or None if
            no proposals could be generated for ANY sequence in the batch.
            Individual sequences with no proposals have proposal_lens[i] == 0.
        """
        ...

    @property
    @abstractmethod
    def needs_draft_probs(self) -> bool:
        """True if proposer provides draft_probs for rejection sampling.

        N-gram: False (no probabilities, uses greedy verification)
        Draft model: True (provides draft_probs for rejection sampling)
        """
        ...

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """True if proposer uses Metal GPU (affects resource planning).

        N-gram: False (pure CPU, Python list operations)
        Draft model: True (runs a separate model on GPU)
        """
        ...


def create_proposer(
    config: SpecDecodeConfig,
    target_model: object = None,
) -> Optional[BaseProposer]:
    """Factory function to create a proposer based on config.

    Args:
        config: SpecDecodeConfig with mode and proposer-specific settings.
        target_model: The target model (reserved for future use).

    Returns:
        A BaseProposer instance, or None if mode is "none".

    Raises:
        ValueError: If the requested mode is not available.
    """
    if config.mode == "none":
        return None
    elif config.mode == "ngram":
        from mlx_lm_server.spec_decode.proposer.ngram import NGramProposer
        return NGramProposer(
            ngram_max=config.ngram_max,
            ngram_min=config.ngram_min,
            prompt_lookup=config.ngram_prompt_lookup,
        )
    elif config.mode == "draft":
        from mlx_lm_server.spec_decode.proposer.draft_model import DraftModelProposer
        return DraftModelProposer(
            model_path=config.draft_model_path,
            quantize=config.draft_model_quantize,
        )
    else:
        raise ValueError(f"Unknown spec decode mode: {config.mode}")
```

**Tests:** `tests/spec_decode/test_config.py` (extend)

```
test_create_proposer_none_returns_none
test_create_proposer_ngram_returns_instance
test_create_proposer_draft_requires_path
test_create_proposer_unknown_mode_raises
test_proposal_result_dataclass
```

### P1.2: NGramProposer

**File:** `mlx_lm_server/spec_decode/proposer/ngram.py`

This is a CPU-only proposer that searches the existing context (prompt + generated
tokens) for n-gram matches and proposes the tokens following the match.

**Algorithm:**

```
Input: context = [t_0, t_1, ..., t_N], k = num_speculative_tokens
Output: draft_tokens = [d_0, d_1, ..., d_{k'-1}] where k' <= k

For n = ngram_max down to ngram_min:
    key = context[-n:]   // last n tokens of context
    Search context[0 : N-n] for key, scanning RIGHT-TO-LEFT (most recent first)
    If match found at position i:
        proposal = context[i+n : i+n+k]  // up to k tokens after the match
        If len(proposal) >= 1:
            return proposal[:k]

If no match found for any n:
    return []  // empty proposal
```

**Visual example:**

```
Context: [A, B, C, D, E, F, G, A, B, C, D, E, _]
                                                ^
                                         current position

k = 3, ngram_max = 3

Step 1: Try 3-gram key = [D, E, _]
        No match in context[0:-3] -> skip

Step 2: Try 2-gram key = [D, E]
        Found at position 3: context[3:5] = [D, E]
        Proposal = context[5:8] = [F, G, A]
        Return [F, G, A]
```

**Full implementation specification:**

```python
"""N-gram proposer for speculative decoding.

Searches the existing context (prompt + generated tokens) for n-gram
pattern matches and proposes continuation tokens. This is a CPU-only
operation with zero GPU overhead.

Effective for:
- Code generation (repetitive patterns, boilerplate)
- Translation (repeated phrases)
- Templated output (JSON, XML, structured formats)
- Summarization (copying from source)

Less effective for:
- Creative writing (unique token sequences)
- Reasoning (novel logical chains)
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import mlx.core as mx

from mlx_lm_server.spec_decode.proposer.base import BaseProposer, ProposalResult

if TYPE_CHECKING:
    from mlx_lm_server.types import SequenceState


class NGramProposer(BaseProposer):
    """N-gram context-matching proposer.

    Attributes:
        ngram_max: Maximum n-gram size to try (inclusive). Default: 4.
        ngram_min: Minimum n-gram size to try (inclusive). Default: 1.
        prompt_lookup: Whether to include prompt tokens in the search
            context. If False, only searches generated tokens.
    """

    def __init__(
        self,
        ngram_max: int = 4,
        ngram_min: int = 1,
        prompt_lookup: bool = True,
    ) -> None:
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.prompt_lookup = prompt_lookup

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False

    def propose(
        self,
        sequences: List[SequenceState],
        k: int,
    ) -> Optional[ProposalResult]:
        """Generate n-gram proposals for all sequences.

        For each sequence, searches its context for the longest n-gram
        match at the current position and proposes up to k continuation
        tokens.

        Args:
            sequences: List of SequenceState with token_ids and
                output_tokens attributes.
            k: Maximum number of draft tokens per sequence.

        Returns:
            ProposalResult with padded draft_tokens [B, max_proposal_len],
            draft_probs=None, and proposal_lens [B].
            Returns None if no sequence produced any proposals.
        """
        batch_proposals: List[List[int]] = []
        proposal_lens: List[int] = []
        any_found = False

        for seq in sequences:
            tokens = self._propose_single(seq, k)
            batch_proposals.append(tokens)
            proposal_lens.append(len(tokens))
            if tokens:
                any_found = True

        if not any_found:
            return None

        # Pad all proposals to the maximum length
        max_len = max(proposal_lens)
        if max_len == 0:
            return None

        padded = []
        for p in batch_proposals:
            padded.append(p + [0] * (max_len - len(p)))

        return ProposalResult(
            draft_tokens=mx.array(padded, dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.array(proposal_lens, dtype=mx.int32),
        )

    def _propose_single(self, seq: SequenceState, k: int) -> List[int]:
        """Generate n-gram proposal for a single sequence.

        Builds the context from prompt + generated tokens (or generated
        only if prompt_lookup is False). Searches from largest n-gram
        down to smallest, most recent match first.

        Args:
            seq: SequenceState with token_ids (full context) and
                output_tokens (generated tokens only).
            k: Maximum number of draft tokens.

        Returns:
            List of proposed token IDs (may be empty).
        """
        # Build search context
        # seq.token_ids = prompt_tokens + output_tokens (maintained by scheduler)
        if self.prompt_lookup:
            context = seq.token_ids
        else:
            context = seq.output_tokens

        if len(context) < self.ngram_min + 1:
            return []

        return self._linear_search(context, k)

    def _linear_search(self, context: List[int], k: int) -> List[int]:
        """Linear scan for n-gram match in context.

        Tries largest n-gram first. For each n-gram size, scans the
        context right-to-left (most recent match preferred).

        Time complexity: O(ngram_max * len(context))
        Acceptable for contexts up to ~2000 tokens. For longer contexts,
        use the suffix index (build_suffix_index + _index_search).

        Args:
            context: Full token ID list (prompt + generated).
            k: Maximum number of tokens to propose.

        Returns:
            List of proposed token IDs (may be shorter than k).
        """
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(context) < n + 1:
                continue

            key = tuple(context[-n:])
            search_end = len(context) - n  # Exclude current position

            # Reverse scan: most recent match is most relevant
            for i in range(search_end - 1, -1, -1):
                if tuple(context[i : i + n]) == key:
                    start = i + n
                    end = min(start + k, len(context))
                    proposals = list(context[start:end])
                    if len(proposals) >= 1:
                        return proposals[:k]

        return []

    @staticmethod
    def build_suffix_index(
        tokens: List[int], ngram_max: int = 4
    ) -> Dict[tuple, List[int]]:
        """Build an n-gram suffix index for O(1) lookup.

        Creates a dictionary mapping n-gram tuples to lists of positions
        where they occur. Called once after prefill, then updated
        incrementally as tokens are generated.

        Args:
            tokens: Full token ID list.
            ngram_max: Maximum n-gram size to index.

        Returns:
            Dict mapping (token_0, ..., token_{n-1}) -> [pos_0, pos_1, ...]
        """
        index: Dict[tuple, List[int]] = {}
        for n in range(1, ngram_max + 1):
            for i in range(len(tokens) - n):
                key = tuple(tokens[i : i + n])
                if key not in index:
                    index[key] = []
                index[key].append(i)
        return index

    @staticmethod
    def update_suffix_index(
        index: Dict[tuple, List[int]],
        tokens: List[int],
        new_token_count: int,
        ngram_max: int = 4,
    ) -> None:
        """Incrementally update suffix index with newly generated tokens.

        Instead of rebuilding the entire index, only processes the
        region affected by new tokens.

        Args:
            index: Existing suffix index to update in-place.
            tokens: Full token list (including new tokens).
            new_token_count: Number of tokens added since last update.
            ngram_max: Maximum n-gram size.
        """
        start = max(0, len(tokens) - new_token_count - ngram_max)
        for n in range(1, ngram_max + 1):
            for i in range(start, len(tokens) - n):
                key = tuple(tokens[i : i + n])
                if key not in index:
                    index[key] = []
                if not index[key] or index[key][-1] != i:
                    index[key].append(i)

    def _index_search(
        self, index: Dict[tuple, List[int]], context: List[int], k: int
    ) -> List[int]:
        """Search for n-gram match using pre-built suffix index.

        O(1) lookup per n-gram size instead of O(len(context)).

        Args:
            index: Pre-built suffix index from build_suffix_index().
            context: Full token ID list.
            k: Maximum number of tokens to propose.

        Returns:
            List of proposed token IDs (may be shorter than k).
        """
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(context) < n + 1:
                continue

            key = tuple(context[-n:])
            if key not in index:
                continue

            positions = index[key]
            # Reverse: most recent position first
            for pos in reversed(positions):
                if pos + n >= len(context):
                    continue  # Skip current position itself
                start = pos + n
                end = min(start + k, len(context))
                proposals = list(context[start:end])
                if len(proposals) >= 1:
                    return proposals[:k]

        return []
```

**Tests:** `tests/spec_decode/test_ngram_proposer.py`

| Test Name | Description |
|-----------|-------------|
| `test_exact_repeat_pattern` | Context `[1,2,3,4,5,1,2,3,4,5]`, last 2 tokens `[4,5]` match position 3, propose `[1,2,3]` |
| `test_no_match_unique_tokens` | All unique tokens, expect empty proposal |
| `test_largest_ngram_preferred` | 4-gram match and 2-gram match both exist, should return 4-gram result |
| `test_most_recent_match_preferred` | Same n-gram at positions 2 and 7, should return position 7's continuation |
| `test_proposal_length_capped_at_k` | Match has 10 continuation tokens but k=3, only return 3 |
| `test_proposal_shorter_than_k` | Match near end of context, only 2 continuation tokens available with k=5 |
| `test_context_too_short` | Context has fewer tokens than ngram_min, return empty |
| `test_batch_mixed_proposals` | Batch of 3 sequences: one with match, one without, one with short match |
| `test_batch_all_no_match` | All sequences have no match, propose() returns None |
| `test_prompt_lookup_false` | With prompt_lookup=False, only searches generated tokens |
| `test_suffix_index_matches_linear` | Verify suffix index produces same results as linear search |
| `test_suffix_index_incremental_update` | Add tokens, update index, verify correctness |
| `test_empty_context` | Empty token list, return empty |
| `test_single_token_context` | Only 1 token, no n-gram possible |

### P1.3: NGramVerifier

**File:** `mlx_lm_server/spec_decode/verifier.py`

The verifier takes the target model's probability distribution over draft token
positions and determines which draft tokens to accept.

For n-gram proposals (no draft probabilities), we use greedy verification: accept
the draft token if it matches the target model's argmax.

```python
"""Verification strategies for speculative decoding.

NGramVerifier: Greedy or threshold-based verification for n-gram proposals.
Used when the proposer does not provide draft probabilities.
"""

from __future__ import annotations

import mlx.core as mx

PLACEHOLDER_TOKEN_ID = -1


class NGramVerifier:
    """Verifier for n-gram proposals (no draft probabilities).

    Two modes:
    - "greedy": Accept draft token if it matches target model's argmax.
      This is lossless -- the output is identical to non-speculative
      greedy decoding.
    - "threshold": Accept draft token if target model assigns it
      probability >= threshold. This is lossy but more permissive.

    Attributes:
        mode: "greedy" or "threshold"
        threshold: Minimum target probability for acceptance in
            threshold mode. Ignored in greedy mode.
    """

    def __init__(self, mode: str = "greedy", threshold: float = 0.1) -> None:
        if mode not in ("greedy", "threshold"):
            raise ValueError(f"Unknown verification mode: {mode}")
        self.mode = mode
        self.threshold = threshold

    def verify(
        self,
        target_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
        modes: list[str] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Verify draft tokens against target model probabilities.

        Args:
            target_probs: [batch, max_k+1, vocab_size] probability
                distribution from target model. Position i contains the
                distribution for predicting position i+1 given the context
                up to position i.
            draft_tokens: [batch, k] proposed token IDs from proposer.
            proposal_lens: [batch] actual proposal length per sequence.
            modes: Optional per-sequence verification mode list. When
                provided, applies "greedy" or "threshold" per sequence,
                enabling mixed-mode batches (e.g., temp=0 seqs use greedy,
                temp>0 seqs use threshold). When None, uses self.mode for all.

        Returns:
            Tuple of:
            - accepted_tokens: [batch, k+1] int32 array. Accepted positions
              contain token IDs, rejected positions contain PLACEHOLDER_TOKEN_ID.
              The k+1-th position is the "bonus" or "correction" token.
            - num_accepted: [batch] int32 array. Number of accepted draft
              tokens per sequence (not counting the bonus/correction token).
        """
        if modes is None:
            # Uniform mode for all sequences
            if self.mode == "greedy":
                return self._greedy(target_probs, draft_tokens, proposal_lens)
            else:
                return self._threshold(target_probs, draft_tokens, proposal_lens)

        # Per-sequence mixed mode: split into greedy and threshold groups,
        # apply each, then merge results. If all same mode, fast path.
        unique_modes = set(modes)
        if unique_modes == {"greedy"}:
            return self._greedy(target_probs, draft_tokens, proposal_lens)
        elif unique_modes == {"threshold"}:
            return self._threshold(target_probs, draft_tokens, proposal_lens)
        else:
            # Mixed batch: run both, merge per-sequence
            greedy_out, greedy_n = self._greedy(target_probs, draft_tokens, proposal_lens)
            thresh_out, thresh_n = self._threshold(target_probs, draft_tokens, proposal_lens)
            for i, m in enumerate(modes):
                if m == "threshold":
                    greedy_out[i] = thresh_out[i]
                    greedy_n[i] = thresh_n[i]
            return greedy_out, greedy_n

    def _greedy(
        self,
        target_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Greedy verification: accept iff draft == target argmax.

        This produces IDENTICAL output to non-speculative greedy decoding.
        There is no quality loss.

        Algorithm:
        1. Compute target_argmax = argmax(target_probs[:, :k, :])
        2. match = (draft_tokens == target_argmax) & valid_mask
        3. match_cumulative = cumprod(match) -- reject after first mismatch
        4. At first rejection: use target_argmax as correction token
        5. If all accepted: take argmax(target_probs[:, k, :]) as bonus token
        """
        batch_size, k = draft_tokens.shape

        output = mx.full(
            (batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32
        )

        # Target model's top-1 prediction at each draft position
        target_argmax = mx.argmax(target_probs[:, :k, :], axis=-1)  # [B, k]

        # Match: draft token equals target argmax
        match = draft_tokens == target_argmax  # [B, k]

        # Valid mask: only check positions within proposal_lens
        pos = mx.arange(k)[None, :]                    # [1, k]
        valid = pos < proposal_lens[:, None]            # [B, k]
        match = match & valid

        # Left-to-right: reject everything after first mismatch
        # cumprod trick: [T, T, F, T] -> [T, T, F, F]
        match_cum = mx.cumprod(match.astype(mx.float32), axis=1)
        match_mask = match_cum.astype(mx.bool_)  # [B, k]

        # Fill accepted positions with draft tokens
        output[:, :k] = mx.where(match_mask, draft_tokens, PLACEHOLDER_TOKEN_ID)

        # Count accepted tokens per sequence
        num_accepted = match_mask.astype(mx.int32).sum(axis=1)  # [B]

        # Fill correction/bonus token at position num_accepted[i]
        # For sequences with partial acceptance: correction = target_argmax
        # For sequences with full acceptance: bonus = argmax(target_probs[:, k, :])
        # For sequences with no proposals: normal decode token
        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])

            if plen == 0:
                # No proposal -- use target model's top-1 as the one token
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                # Partial acceptance -- correction token at rejection position
                output[b, n] = int(target_argmax[b, n])
            else:
                # Full acceptance -- bonus token from position after last draft
                output[b, plen] = mx.argmax(target_probs[b, plen, :])

        return output, num_accepted

    def _threshold(
        self,
        target_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Threshold verification: accept if target assigns >= threshold prob.

        More permissive than greedy but introduces quality loss.
        Useful when temperature > 0 and exact argmax match is too strict.

        Algorithm:
        1. Extract target_prob[draft_token] for each position
        2. accepted = (target_prob >= threshold) & valid_mask
        3. accepted_cumulative = cumprod(accepted)
        4. Correction/bonus same as greedy
        """
        batch_size, k = draft_tokens.shape

        output = mx.full(
            (batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32
        )

        # Extract target model's probability for each draft token
        target_p = mx.take_along_axis(
            target_probs[:, :k, :], draft_tokens[:, :, None], axis=2
        ).squeeze(-1)  # [B, k]

        accepted = target_p >= self.threshold

        pos = mx.arange(k)[None, :]
        valid = pos < proposal_lens[:, None]
        accepted = accepted & valid

        accepted_cum = mx.cumprod(accepted.astype(mx.float32), axis=1)
        accepted_mask = accepted_cum.astype(mx.bool_)

        output[:, :k] = mx.where(accepted_mask, draft_tokens, PLACEHOLDER_TOKEN_ID)
        num_accepted = accepted_mask.astype(mx.int32).sum(axis=1)

        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])

            if plen == 0:
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                output[b, n] = mx.argmax(target_probs[b, n, :])
            else:
                output[b, plen] = mx.argmax(target_probs[b, plen, :])

        return output, num_accepted
```

**Tests:** `tests/spec_decode/test_ngram_verifier.py`

| Test Name | Description |
|-----------|-------------|
| `test_all_accepted_greedy` | All draft tokens match target argmax, expect k accepted + 1 bonus |
| `test_none_accepted_greedy` | First draft token mismatches, expect 0 accepted + 1 correction |
| `test_partial_accepted_greedy` | First 3 of 5 match, expect 3 accepted + 1 correction |
| `test_batch_mixed_acceptance` | Batch of 3 with different acceptance counts |
| `test_zero_proposal_lens` | Sequences with proposal_lens=0 get normal decode token |
| `test_threshold_mode_permissive` | Draft token has 0.5 prob with threshold=0.1, accepted |
| `test_threshold_mode_strict` | Draft token has 0.05 prob with threshold=0.1, rejected |
| `test_cumprod_masking` | Verify [T,T,F,T] becomes [T,T,F,F] |
| `test_bonus_token_is_argmax` | When all k accepted, bonus = argmax(target_probs[:, k, :]) |
| `test_output_shape` | Output is [B, k+1], num_accepted is [B] |

### P1.4: SpecDecodeEngine

**File:** `mlx_lm_server/spec_decode/engine.py`

This is the core orchestrator. It coordinates the propose-verify-accept cycle and
updates the `BatchGenerator` state after each speculative step.

```
                    SpecDecodeEngine.speculative_step()
                    ===================================

    +-------------------------------------------------------------------+
    |                                                                     |
    |  1. PROPOSE                                                         |
    |     proposer.propose(sequences, k) --> ProposalResult               |
    |     draft_tokens: [B, k], proposal_lens: [B]                       |
    |                                                                     |
    |  2. BUILD VERIFICATION INPUT                                        |
    |     For each sequence i:                                            |
    |       verify_input[i] = [last_token_i, draft_1, ..., draft_k]      |
    |     Pad to [B, max_input_len]                                       |
    |                                                                     |
    |  3. TARGET MODEL FORWARD (single call)                              |
    |     logits = model(verify_input, cache=batch.cache)                 |
    |     logits shape: [B, max_input_len, vocab]                         |
    |     target_probs = softmax(logits)                                  |
    |                                                                     |
    |  4. VERIFY                                                          |
    |     accepted_tokens, num_accepted = verifier.verify(                |
    |         target_probs, draft_tokens, proposal_lens                   |
    |     )                                                               |
    |     accepted_tokens: [B, k+1], num_accepted: [B]                    |
    |                                                                     |
    |  5. CACHE ROLLBACK                                                  |
    |     The forward pass extended cache by max_input_len positions.      |
    |     But we only want to keep (num_accepted[i] + 1) positions.       |
    |     Trim excess positions via cache_utils.batch_variable_trim()     |
    |                                                                     |
    |  6. UPDATE BATCH STATE                                              |
    |     batch.y[i] = last accepted token for sequence i                 |
    |     batch.tokens[i] = append accepted tokens                        |
    |     batch.num_tokens[i] += num_accepted[i] + 1                      |
    |     Check finish conditions per token                               |
    |                                                                     |
    |  7. RETURN SpecResponses                                            |
    |     List of SpecResponse (one per sequence, with multiple tokens)   |
    |                                                                     |
    +-------------------------------------------------------------------+
```

**Data types:**

```python
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx


@dataclass
class SpecResponse:
    """Response from one speculative decoding step for a single sequence.

    Unlike BatchGenerator.Response which contains a single token,
    SpecResponse may contain multiple accepted tokens.

    Attributes:
        uid: Sequence UID (matches BatchGenerator UIDs)
        tokens: List of accepted token IDs (1 to k+1 tokens)
        logprobs: Log probabilities for each accepted token
        finish_reason: "stop", "length", or None if not finished
        prompt_cache: Callable to extract prompt cache (same as Response)
        num_drafted: Number of draft tokens that were proposed
        num_accepted: Number of draft tokens that were accepted
            (not counting the correction/bonus token)
    """

    uid: int
    tokens: List[int]
    logprobs: List[mx.array]
    finish_reason: Optional[str]
    prompt_cache: object
    num_drafted: int
    num_accepted: int
```

**Engine implementation specification:**

```python
class SpecDecodeEngine:
    """Orchestrates speculative decoding steps.

    Works alongside BatchGenerator:
    - BatchGenerator handles prefill (prompt processing)
    - SpecDecodeEngine handles decode (multi-token generation)
    - Both share the same Batch object and its KV cache

    The engine does NOT own the model or cache -- it borrows them
    from BatchGenerator.active_batch for each step.

    Attributes:
        model: Reference to the target model (from BatchGenerator)
        batch_generator: The BatchGenerator instance (for state access)
        proposer: The proposer (n-gram, draft model)
        verifier: The verifier (greedy, threshold, rejection sampler)
        config: SpecDecodeConfig
        controller: DynamicSpecController for adaptive behavior
    """

    def __init__(
        self,
        model,
        batch_generator,
        proposer: BaseProposer,
        verifier,
        config: SpecDecodeConfig,
        controller: DynamicSpecController,
    ) -> None:
        self.model = model
        self.batch_generator = batch_generator
        self.proposer = proposer
        self.verifier = verifier
        self.config = config
        self.controller = controller

    def should_speculate(self, batch_size: int) -> bool:
        """Check if spec decode should be used for this step.

        Delegates to DynamicSpecController. Returns False if:
        - Spec decode is disabled (mode="none")
        - Batch size exceeds threshold
        - Acceptance rate EMA is below threshold
        - No active batch in BatchGenerator
        """
        if self.batch_generator.active_batch is None:
            return False
        return self.controller.should_speculate(batch_size)

    def speculative_step(
        self,
        sequences: List[SequenceState],
    ) -> List[SpecResponse]:
        """Execute one speculative decoding step.

        This replaces BatchGenerator.next() for the decode phase.
        The calling code (scheduler) should call this instead of
        batch_generator.next() when spec decode is active.

        Args:
            sequences: List of active SequenceState objects. Must match
                the sequences currently in BatchGenerator.active_batch
                (same UIDs, same order).

        Returns:
            List of SpecResponse, one per sequence. Each contains
            1 to k+1 accepted tokens.
        """
        batch = self.batch_generator.active_batch
        if batch is None:
            return []

        k = self.controller.get_k(len(sequences))
        if k == 0:
            # Controller says no speculation -- fall back to normal decode
            return self._fallback_normal_decode()

        # --- Step 1: PROPOSE ---
        proposal = self.proposer.propose(sequences, k)
        if proposal is None:
            # No proposals for any sequence -- fall back
            return self._fallback_normal_decode()

        # --- Step 2: BUILD VERIFICATION INPUT ---
        verify_input, verify_lens = self._build_verify_input(
            batch, proposal.draft_tokens, proposal.proposal_lens
        )
        # verify_input: [B, max_input_len] -- padded token IDs
        # verify_lens: [B] -- actual input length per sequence

        # --- Step 3: TARGET MODEL FORWARD ---
        target_probs = self._target_forward(batch, verify_input)
        # target_probs: [B, max_input_len, vocab_size]

        # --- Step 4: VERIFY ---
        # Per-sequence verification mode — avoid batch-level demotion
        # Mixed batches (temp=0 and temp>0 together) must not demote greedy
        # sequences to threshold.
        per_seq_modes = []
        for i in range(len(sequences)):
            temp = getattr(batch.samplers[i], 'temperature', 0.0)
            per_seq_modes.append(self.controller.get_verification_mode(temp))

        accepted_tokens, num_accepted = self.verifier.verify(
            target_probs, proposal.draft_tokens, proposal.proposal_lens,
            modes=per_seq_modes,
        )
        # accepted_tokens: [B, k+1]
        # num_accepted: [B]

        # --- Step 5: CACHE ROLLBACK ---
        max_input_len = verify_input.shape[1]
        self._rollback_cache(batch, num_accepted, max_input_len)

        # --- Step 6: UPDATE BATCH STATE ---
        responses = self._update_batch_state(
            batch, sequences, accepted_tokens, num_accepted,
            proposal.proposal_lens, target_probs,
        )

        # --- Step 7: UPDATE CONTROLLER ---
        total_proposed = int(proposal.proposal_lens.sum())
        total_accepted = int(num_accepted.sum())
        total_bonus = sum(
            1 for i in range(len(sequences))
            if int(num_accepted[i]) >= int(proposal.proposal_lens[i])
            and int(proposal.proposal_lens[i]) > 0
        )
        self.controller.update(total_proposed, total_accepted, total_bonus)

        return responses

    def _build_verify_input(
        self,
        batch,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Build padded verification input for target model.

        For each sequence, the input is:
            [last_token, draft_token_0, ..., draft_token_{plen-1}]

        Sequences with proposal_lens[i] == 0 get just [last_token].
        All sequences are padded to the maximum input length.

        Args:
            batch: The active Batch from BatchGenerator.
            draft_tokens: [B, k] proposed tokens.
            proposal_lens: [B] actual proposal length per sequence.

        Returns:
            Tuple of:
            - verify_input: [B, max_input_len] padded token IDs
            - verify_lens: [B] actual input length per sequence
        """
        batch_size = len(batch.uids)
        k = draft_tokens.shape[1]

        # last token for each sequence
        last_tokens = batch.y  # [B] -- current token from BatchGenerator

        inputs = []
        input_lens = []
        for i in range(batch_size):
            plen = int(proposal_lens[i])
            if plen == 0:
                # Just the last token for normal decode
                inp = [int(last_tokens[i])]
            else:
                # last_token + draft tokens
                inp = [int(last_tokens[i])] + [
                    int(draft_tokens[i, j]) for j in range(plen)
                ]
            inputs.append(inp)
            input_lens.append(len(inp))

        max_len = max(input_lens)

        # Left-pad to max_len (BatchKVCache expects left-padded input)
        # Actually, during decode, BatchKVCache uses sequential processing
        # so we should right-pad and handle with attention mask.
        # Let's use right-padding for the verification input.
        padded = []
        for inp in inputs:
            padded.append(inp + [0] * (max_len - len(inp)))

        verify_input = mx.array(padded, dtype=mx.int32)
        verify_lens = mx.array(input_lens, dtype=mx.int32)

        return verify_input, verify_lens

    def _target_forward(
        self,
        batch,
        verify_input: mx.array,
    ) -> mx.array:
        """Run target model forward pass with multi-token input.

        This directly calls the model with the verification input,
        using the existing KV cache from BatchGenerator.

        The model's forward pass extends the KV cache by
        verify_input.shape[1] positions for ALL sequences.

        Args:
            batch: Active Batch with model cache.
            verify_input: [B, max_input_len] token IDs.

        Returns:
            target_probs: [B, max_input_len, vocab_size] softmax probs.
        """
        # Direct model call using batch cache
        logits = self.model(verify_input, cache=batch.cache)
        # logits: [B, max_input_len, vocab_size]

        # Phase 1: Apply temperature scaling per sequence before softmax.
        # top_p and logits_processors are NOT applied in Phase 1 — they require
        # proper rejection sampling (Phase 2) to preserve the target distribution.
        batch_size = logits.shape[0]
        for i in range(batch_size):
            temp = getattr(batch.samplers[i], 'temperature', 0.0)
            if temp > 0:
                logits[i] = logits[i] / temp
        target_probs = mx.softmax(logits, axis=-1)
        mx.eval(target_probs)

        return target_probs

    def _rollback_cache(
        self,
        batch,
        num_accepted: mx.array,
        max_input_len: int,
    ) -> None:
        """Roll back KV cache to account for rejected tokens.

        The target model forward extended the cache by max_input_len
        positions for all sequences. We need to trim back to keep
        only (num_accepted[i] + 1) positions per sequence.

        Uses batch_variable_trim from cache_utils for per-sequence
        trimming.

        Args:
            batch: Active Batch with model cache.
            num_accepted: [B] accepted draft count per sequence.
            max_input_len: Number of positions added by the forward pass.
        """
        from mlx_lm_server.spec_decode.cache_utils import batch_variable_trim

        # How many positions to keep per sequence: num_accepted + 1
        # (accepted drafts + the correction/bonus token)
        keep_amounts = num_accepted + 1  # [B]

        # How many to trim per sequence
        trim_amounts = max_input_len - keep_amounts  # [B]

        # Ensure non-negative
        trim_amounts = mx.maximum(trim_amounts, 0)

        batch_variable_trim(batch.cache, trim_amounts)

    def _update_batch_state(
        self,
        batch,
        sequences: List[SequenceState],
        accepted_tokens: mx.array,
        num_accepted: mx.array,
        proposal_lens: mx.array,
        target_probs: mx.array,
    ) -> List[SpecResponse]:
        """Update BatchGenerator's Batch state with accepted tokens.

        After spec step, we must sync:
        - batch.y: set to last accepted token per sequence
        - batch.tokens: append all accepted tokens
        - batch.num_tokens: increment by total accepted
        - batch.logprobs: update to last position's logprobs

        Args:
            batch: Active Batch to update.
            sequences: SequenceState list for finish checking.
            accepted_tokens: [B, k+1] accepted token IDs.
            num_accepted: [B] accepted draft count.
            proposal_lens: [B] original proposal lengths.
            target_probs: [B, max_input_len, vocab] for logprobs.

        Returns:
            List of SpecResponse, one per sequence.
        """
        responses = []

        for i, seq in enumerate(sequences):
            uid = batch.uids[i]
            n_accepted = int(num_accepted[i])
            plen = int(proposal_lens[i])

            # Extract valid tokens (not PLACEHOLDER)
            row = accepted_tokens[i]  # [k+1]
            valid_tokens = []
            for j in range(row.shape[0]):
                t = int(row[j])
                if t == PLACEHOLDER_TOKEN_ID:
                    break
                valid_tokens.append(t)

            if not valid_tokens:
                # This shouldn't happen (at minimum we get one token)
                # but handle gracefully
                valid_tokens = [int(mx.argmax(target_probs[i, 0, :]))]

            # Update batch state — single bulk concatenation per sequence
            valid_arr = mx.array(valid_tokens)
            batch.tokens[i] = mx.concatenate([batch.tokens[i], valid_arr])
            batch.num_tokens[i] += len(valid_tokens)
            batch.y[i] = valid_tokens[-1]

            # Extract logprobs for each accepted token
            # The logprobs come from the corresponding positions in target_probs
            token_logprobs = []
            for j, t in enumerate(valid_tokens):
                # Position j in the forward pass corresponds to token j
                lp = mx.log(target_probs[i, j, :] + 1e-10)
                token_logprobs.append(lp)

            # Check finish conditions
            finish_reason = None
            for t in valid_tokens:
                if t in self.batch_generator.stop_tokens:
                    finish_reason = "stop"
                    break
            if finish_reason is None and batch.num_tokens[i] >= batch.max_tokens[i]:
                finish_reason = "length"

            responses.append(SpecResponse(
                uid=uid,
                tokens=valid_tokens,
                logprobs=token_logprobs,
                finish_reason=finish_reason,
                prompt_cache=None,  # Set by caller when sequence finishes
                num_drafted=plen,
                num_accepted=n_accepted,
            ))

        # Update batch.y as mx.array (it may have been modified element-wise)
        new_y = mx.array([int(batch.y[i]) for i in range(len(batch.uids))])
        batch.y = new_y

        return responses

    def _fallback_normal_decode(self) -> List:
        """Fall back to BatchGenerator.next() for normal decode.

        Used when proposer returns no proposals or controller disables
        spec decode.

        Returns:
            List of BatchGenerator.Response objects (1 token each).
        """
        return self.batch_generator.next()
```

**CRITICAL IMPLEMENTATION NOTES for subagent:**

1. The `_target_forward` call uses `self.model(verify_input, cache=batch.cache)`.
   This works because the model's forward pass accepts `[B, S]` input where S > 1
   (it does this during prefill). During decode, S is normally 1, but the model
   architecture supports S > 1.

2. The `_rollback_cache` must handle per-sequence trim amounts. See P1.5 for the
   `batch_variable_trim` implementation and the investigation notes in Section 3.

3. After `_update_batch_state`, the next call to `speculative_step()` or
   `batch_generator.next()` must see consistent state. The key invariant is:
   `batch.y[i]` is the last generated token, and `batch.cache` reflects all tokens
   generated so far (including the newly accepted ones minus rollback).

4. The `_fallback_normal_decode` returns `BatchGenerator.Response` objects, not
   `SpecResponse`. The scheduler must handle both types.

### P1.5: Cache Utilities

**File:** `mlx_lm_server/spec_decode/cache_utils.py`

```python
"""Cache utilities for speculative decoding.

Provides batched cache operations that the standard BatchKVCache
does not support, specifically per-sequence variable trimming.
"""

from __future__ import annotations

from typing import Any, List

import mlx.core as mx


def batch_variable_trim(
    cache_layers: List[Any],
    trim_amounts: mx.array,
) -> None:
    """Trim different amounts from each sequence's cache.

    BatchKVCache.trim(n) trims ALL sequences by the same amount n.
    This function applies different trim amounts per sequence by
    directly manipulating the per-sequence offset array.

    IMPORTANT: This function manipulates cache._idx and cache.offset
    directly. It assumes that the attention mask mechanism correctly
    excludes data beyond each sequence's offset. See the investigation
    notes in Section 3 of the implementation plan.

    How it works:
    1. Trim ALL sequences by max(trim_amounts) using the standard trim()
    2. For sequences that needed less trimming, re-advance their offset
       to compensate

    Alternative (if standard approach fails the investigation test):
    Directly set each sequence's offset without using trim() at all.

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

#### GATE Decision Matrix

| | Result A (offset safe) | Result B (offset unsafe) |
|---|---|---|
| `batch_variable_trim()` | per-sequence offset manipulation | `uniform_trim(max_rejected)` for all |
| Verifier output `num_accepted` | `mx.array [B]` per-sequence | `mx.array [B]` (all same value = batch min) |
| Engine rollback | per-sequence different amounts | uniform trim + re-forward for under-trimmed |
| Performance impact | optimal | acceptance rate loss (conservative) |

#### Phase 1 Decision: min_accepted Uniform (No Re-forward)

**Decision:** Phase 1 uses the simplest Result B strategy — trim all sequences to
`min_accepted + 1` and discard tokens beyond that. No re-forward pass.

**Rationale:**
- **Forward is exactly 1 per spec step.** Only the verification forward, no re-forward.
- **Logic is trivial.** 5 lines in `_rollback_cache()`, no complex delta computation.
- **Accept loss is small in practice.** N-gram acceptance has high intra-batch correlation
  (same domain prompts have similar repetition patterns), so `max_accepted - min_accepted`
  is typically 0-1.
- **Worst case is safe.** High acceptance variance → emit 1 token (same as normal decode).

**Re-forward cost analysis (why NOT re-forward in Phase 1):**
```
acceptance = [1, 5, 3, 5]  (k=5, batch=4)
min_accepted = 1  →  trim_all = 4

Re-forward would need [B, max_delta=4] forward = 2nd forward pass total.
2 forwards ÷ avg 2.75 tokens ≈ 0.73 fwd/tok vs normal decode 1.0 fwd/tok.
Marginal gain does not justify complexity.
```

**Phase 2+ strategy:** When draft model acceptance precision matters more,
consider upstream PR to fix per-sequence trim (`update_and_fetch` using
`max(offset)` instead of `_idx`, plus `make_mask` right-padding support).

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
```

**Tests:** `tests/spec_decode/test_cache_utils.py`

| Test Name | Description |
|-----------|-------------|
| `test_uniform_trim_basic` | Trim 2 from all, verify offsets decreased by 2 |
| `test_variable_trim_different_amounts` | Trim [3, 1, 5] from 3 sequences, verify per-sequence offsets |
| `test_variable_trim_zero_for_some` | Trim [0, 3, 0], sequences 0 and 2 unchanged |
| `test_variable_trim_all_zero` | No-op when all trim amounts are 0 |
| `test_offset_consistency_across_layers` | All layers should have same offset changes |
| `test_subsequent_forward_after_variable_trim` | After trim, run model forward and verify output correctness |
| `test_trim_does_not_corrupt_existing_cache` | Verify that cache data before trim point is unchanged |

**Test `test_subsequent_forward_after_variable_trim` detail:**

This is the critical investigation test mentioned in Section 3. It must:
1. Create a BatchKVCache with 2+ sequences at different lengths
2. Run model forward extending all by 6 tokens (simulating k=5 spec step)
3. Apply `batch_variable_trim([3, 1, 5])` -- different trims per sequence
4. Run another model forward with 1 token per sequence
5. Compare outputs to a reference run without spec decode

If this test FAILS, the `batch_variable_trim` approach is invalid and we must
fall back to the uniform-trim-with-re-forward approach.

### P1.6: Scheduler Integration

**File to modify:** `mlx_lm_server/scheduler.py`

The scheduler's `_batch_inference_step()` is modified to optionally use
`SpecDecodeEngine` instead of `BatchGenerator.next()`.

**Changes to `__init__`:**

```python
# In Scheduler.__init__(), after BatchGenerator creation:

# Speculative decode engine (optional)
self._spec_engine: Optional[SpecDecodeEngine] = None
if config.spec_decode_mode != "none":
    from mlx_lm_server.spec_decode.config import SpecDecodeConfig
    from mlx_lm_server.spec_decode.engine import SpecDecodeEngine
    from mlx_lm_server.spec_decode.proposer.base import create_proposer
    from mlx_lm_server.spec_decode.verifier import NGramVerifier
    from mlx_lm_server.spec_decode.controller import DynamicSpecController

    spec_config = SpecDecodeConfig(
        mode=config.spec_decode_mode,
        num_speculative_tokens=config.spec_decode_num_tokens,
        disable_by_batch_size=config.spec_decode_disable_batch_size,
        ngram_max=config.spec_decode_ngram_max,
        ngram_min=config.spec_decode_ngram_min,
        ngram_prompt_lookup=config.spec_decode_ngram_prompt_lookup,
        draft_model_path=config.spec_decode_draft_model,
        draft_model_quantize=config.spec_decode_draft_quantize,
        dynamic_enabled=config.spec_decode_dynamic,
        acceptance_rate_threshold=config.spec_decode_acceptance_threshold,
        adaptive_k=config.spec_decode_adaptive_k,
    )
    spec_config.validate()

    proposer = create_proposer(spec_config, target_model=self._model)
    verifier = NGramVerifier(mode="greedy")  # Phase 1: greedy only
    controller = DynamicSpecController(spec_config)

    if proposer is not None:
        self._spec_engine = SpecDecodeEngine(
            model=self._model,
            batch_generator=self._batch_generator,
            proposer=proposer,
            verifier=verifier,
            config=spec_config,
            controller=controller,
        )
```

**Changes to `_batch_inference_step()`:**

```python
def _batch_inference_step(self) -> None:
    # ... existing steps 1-2 (bus sync, wait for work) ...

    # 1. Process cancellations
    self._process_cancellations_batch()

    # 2. Insert new requests
    self._insert_new_requests_batch()

    # 3. Check active sequences
    with self._active_lock:
        has_active = bool(self._uid_to_request_id)
        active_count = len(self._uid_to_request_id)
    if not has_active:
        return

    # 4. Choose decode strategy
    use_spec = (
        self._spec_engine is not None
        and self._spec_engine.should_speculate(active_count)
    )

    if use_spec:
        # Speculative decode path
        with self._active_lock:
            sequences = [
                self._active_sequences[rid]
                for rid in self._uid_to_request_id.values()
                if rid in self._active_sequences
            ]
        try:
            responses = self._spec_engine.speculative_step(sequences)
        except Exception as e:
            logger.error("SpecDecodeEngine failed: %s", e, exc_info=True)
            # Fall back to normal decode
            responses = self._batch_generator.next()
            use_spec = False

        if use_spec:
            # Process spec responses (multi-token)
            events, uids_to_remove, finished_caches = (
                self._process_spec_responses(responses)
            )
        else:
            # Fell back to normal
            events, uids_to_remove, finished_caches = (
                self._process_batch_responses(responses)
            )
    else:
        # Normal decode path (unchanged)
        try:
            responses = self._batch_generator.next()
        except Exception as e:
            logger.error("BatchGenerator.next() failed: %s", e, exc_info=True)
            raise

        events, uids_to_remove, finished_caches = (
            self._process_batch_responses(responses)
        )

    # 5-10. Emit, remove, store, prune, cleanup (unchanged)
    self._emit_tokens(events)
    # ... rest unchanged ...
```

**New method `_process_spec_responses()`:**

```python
def _process_spec_responses(
    self, responses
) -> tuple[list[TokenEvent], list[int], dict[int, list]]:
    """Process speculative decode responses with multiple tokens per sequence.

    Similar to _process_batch_responses but handles SpecResponse objects
    which contain a list of tokens instead of a single token.

    Args:
        responses: List of SpecResponse from SpecDecodeEngine.

    Returns:
        Tuple of (events, uids_to_remove, finished_caches).
    """
    uids_to_remove: list[int] = []
    events: list[TokenEvent] = []
    finished_caches: dict[int, list] = {}

    for r in responses:
        # Handle both SpecResponse and BatchGenerator.Response
        if hasattr(r, 'tokens'):
            # SpecResponse -- multiple tokens
            tokens = r.tokens
        else:
            # BatchGenerator.Response -- single token (fallback)
            tokens = [r.token]

        request_id = self._uid_to_request_id.get(r.uid)
        if request_id is None:
            continue

        with self._active_lock:
            seq = self._active_sequences.get(request_id)
        if seq is None:
            continue

        finish_reason = None

        for token in tokens:
            # Detokenize
            detokenizer = getattr(seq, "_detokenizer", None)
            if detokenizer is not None:
                detokenizer.add_token(token)
                token_text = detokenizer.last_segment
            else:
                token_text = str(token)

            seq.output_tokens.append(token)
            seq.token_ids.append(token)
            seq.output_text += token_text

            # Check stop conditions
            if finish_reason is None:
                request = getattr(seq, "_request", None)
                if request is not None:
                    finish_reason = self._check_stop_conditions(seq, request)

            events.append(TokenEvent(
                request_id=request_id,
                token_id=token,
                token_text=token_text,
                finish_reason=finish_reason if finish_reason else None,
                logprobs=None,
            ))

            if finish_reason is not None:
                break

        # Handle finish
        if finish_reason is not None:
            seq.is_finished = True
            seq.finish_reason = finish_reason
            self._inc_stat("requests_completed")
            uids_to_remove.append(r.uid)

            if hasattr(r, 'prompt_cache') and r.prompt_cache is not None:
                try:
                    finished_caches[r.uid] = r.prompt_cache
                except Exception:
                    pass

        # Update spec decode stats
        if hasattr(r, 'num_drafted'):
            self._inc_stat("spec_tokens_drafted", r.num_drafted)
            self._inc_stat("spec_tokens_accepted", r.num_accepted)

    return events, uids_to_remove, finished_caches
```

### P1.7: DynamicSpecController

**File:** `mlx_lm_server/spec_decode/controller.py`

```python
"""Dynamic speculation controller.

Adjusts speculation depth (k) and on/off state based on runtime
statistics. Prevents spec decode overhead from degrading performance
at high batch sizes or when acceptance rate is low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from mlx_lm_server.spec_decode.config import SpecDecodeConfig


@dataclass
class SpecDecodeStats:
    """Cumulative statistics for speculative decoding.

    Tracked across all steps, all sequences. Reset only on server restart.
    """

    total_proposed: int = 0    # Total draft tokens proposed
    total_accepted: int = 0    # Total draft tokens accepted
    total_steps: int = 0       # Total spec decode steps executed
    total_bonus_tokens: int = 0  # Total bonus tokens (all k accepted)
    total_fallback_steps: int = 0  # Steps where spec decode was skipped

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all steps."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def avg_tokens_per_step(self) -> float:
        """Average tokens generated per spec decode step.

        1.0 = no benefit from spec decode.
        k+1 = maximum possible (all drafts accepted + bonus).
        """
        if self.total_steps == 0:
            return 1.0
        total_tokens = self.total_accepted + self.total_bonus_tokens
        # Add 1 per step for the correction/bonus token
        total_tokens += self.total_steps
        return total_tokens / self.total_steps


class DynamicSpecController:
    """Controls speculation depth and activation based on runtime stats.

    Decision flow:
    1. batch_size >= disable_by_batch_size --> spec OFF
    2. acceptance_rate_ema < threshold --> spec OFF
    3. If adaptive_k enabled, adjust k based on acceptance rate:
       - ema > 0.8 --> k = max (aggressive)
       - ema > 0.5 --> k = max - 2 (moderate)
       - ema > 0.3 --> k = 1 (conservative)
       - ema <= 0.3 --> k = 0 (OFF)

    Attributes:
        config: SpecDecodeConfig with thresholds and settings.
        acceptance_rate_ema: Exponential moving average of acceptance rate.
            Initialized to 0.7 (optimistic start).
        stats: Cumulative statistics.
    """

    def __init__(self, config: SpecDecodeConfig) -> None:
        self.config = config
        self.acceptance_rate_ema: float = 0.7  # Optimistic initial value
        self.stats = SpecDecodeStats()
        self._recent_rates: list[float] = []
        self._max_recent: int = 100

    def should_speculate(self, batch_size: int) -> bool:
        """Decide whether to use spec decode for this step.

        Args:
            batch_size: Current number of active sequences.

        Returns:
            True if spec decode should be used.
        """
        if self.config.mode == "none":
            return False

        # Batch size threshold (0 = never disable)
        if (
            self.config.disable_by_batch_size > 0
            and batch_size >= self.config.disable_by_batch_size
        ):
            return False

        # If dynamic control is disabled, always speculate
        if not self.config.dynamic_enabled:
            return True

        # Acceptance rate threshold
        return self.acceptance_rate_ema >= self.config.acceptance_rate_threshold

    def get_k(self, batch_size: int) -> int:
        """Get the speculation depth for this step.

        Args:
            batch_size: Current number of active sequences.

        Returns:
            Number of draft tokens (0 = no speculation).
        """
        if not self.should_speculate(batch_size):
            return 0

        k = self.config.num_speculative_tokens

        if not self.config.adaptive_k:
            return k

        # Adaptive k based on acceptance rate EMA
        if self.acceptance_rate_ema > 0.8:
            return k                    # Full speculation
        elif self.acceptance_rate_ema > 0.5:
            return max(1, k - 2)        # Moderate
        elif self.acceptance_rate_ema > 0.3:
            return 1                    # Conservative
        else:
            return 0                    # Off

    def get_verification_mode(self, temperature: float) -> str:
        """Auto-select verification mode based on sampling temperature.

        temperature=0 → "greedy" (lossless, identical to normal decode)
        temperature>0 → "threshold" (lossy but reasonable acceptance rate)
        """
        if temperature == 0.0:
            return "greedy"
        return "threshold"

    def update(
        self,
        num_proposed: int,
        num_accepted: int,
        num_bonus: int = 0,
    ) -> None:
        """Update statistics after a speculative decode step.

        Args:
            num_proposed: Total draft tokens proposed across batch.
            num_accepted: Total draft tokens accepted across batch.
            num_bonus: Total bonus tokens generated (full acceptance).
        """
        self.stats.total_proposed += num_proposed
        self.stats.total_accepted += num_accepted
        self.stats.total_bonus_tokens += num_bonus
        self.stats.total_steps += 1

        if num_proposed > 0:
            step_rate = num_accepted / num_proposed
            alpha = self.config.acceptance_rate_ema_alpha
            self.acceptance_rate_ema = (
                alpha * step_rate + (1 - alpha) * self.acceptance_rate_ema
            )
            self._recent_rates.append(step_rate)
            if len(self._recent_rates) > self._max_recent:
                self._recent_rates.pop(0)

    def record_fallback(self) -> None:
        """Record that spec decode was skipped for a step."""
        self.stats.total_fallback_steps += 1

    def get_metrics(self) -> Dict:
        """Return metrics dict for monitoring endpoints.

        Suitable for /v1/spec_decode/metrics or inclusion in /health.
        """
        return {
            "spec_decode_enabled": self.config.mode != "none",
            "spec_decode_mode": self.config.mode,
            "acceptance_rate_ema": round(self.acceptance_rate_ema, 4),
            "acceptance_rate_overall": round(self.stats.acceptance_rate, 4),
            "avg_tokens_per_step": round(self.stats.avg_tokens_per_step, 2),
            "total_steps": self.stats.total_steps,
            "total_fallback_steps": self.stats.total_fallback_steps,
            "total_proposed": self.stats.total_proposed,
            "total_accepted": self.stats.total_accepted,
            "total_bonus_tokens": self.stats.total_bonus_tokens,
            "current_k": self.config.num_speculative_tokens,
            "adaptive_k_current": self.get_k(1),  # k at batch_size=1
        }
```

**Tests:** `tests/spec_decode/test_controller.py`

| Test Name | Description |
|-----------|-------------|
| `test_should_speculate_mode_none` | mode="none" always returns False |
| `test_should_speculate_under_batch_threshold` | batch < threshold returns True |
| `test_should_speculate_over_batch_threshold` | batch >= threshold returns False |
| `test_should_speculate_low_acceptance_rate` | EMA below threshold returns False |
| `test_ema_update_smoothing` | Verify EMA converges correctly over updates |
| `test_adaptive_k_high_acceptance` | EMA > 0.8 returns full k |
| `test_adaptive_k_moderate_acceptance` | EMA 0.5-0.8 returns k-2 |
| `test_adaptive_k_low_acceptance` | EMA 0.3-0.5 returns 1 |
| `test_adaptive_k_very_low_acceptance` | EMA < 0.3 returns 0 |
| `test_stats_tracking` | Verify total_proposed, total_accepted increment correctly |
| `test_get_metrics_format` | Verify all required keys present |
| `test_dynamic_disabled` | dynamic_enabled=False always speculates |
| `test_disable_by_batch_size_zero` | disable_by_batch_size=0 never disables |

---

## 7. Phase 2: Draft Model Proposer

### P2.1: DraftModelProposer

**File:** `mlx_lm_server/spec_decode/proposer/draft_model.py`

Load a smaller model (e.g., Qwen-1.5B-4bit alongside Qwen-32B target) and run it
autoregressively k times to generate draft tokens with probabilities.

**Key considerations:**
- Draft model loaded via `mlx_lm.utils.load()` (not from `mlx_lm.__init__`)
- Separate KV cache for draft model
- Draft cache cloned from target cache state at start of each spec step
- MLX lazy evaluation fuses k draft steps into one compute graph

```python
class DraftModelProposer(BaseProposer):
    def __init__(self, model_path: str, quantize: Optional[str] = None):
        self.model_path = model_path
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @property
    def needs_draft_probs(self) -> bool:
        return True  # Required for rejection sampling

    @property
    def requires_gpu(self) -> bool:
        return True  # Uses Metal GPU

    def load(self) -> None:
        """Load draft model into memory. Called once at startup."""
        from mlx_lm.utils import load
        self.model, self.tokenizer = load(self.model_path)
        self._loaded = True

    def propose(self, sequences, k) -> Optional[ProposalResult]:
        """Run draft model autoregressively k times.

        For each step:
        1. Feed last token to draft model
        2. Sample from draft model's distribution
        3. Store both token and probability distribution
        4. Use sampled token as input for next step

        All k steps are evaluated lazily -- mx.eval() only at the end.
        """
        # Implementation details in spec
```

### P2.2: BatchedRejectionSampler

**File:** `mlx_lm_server/spec_decode/rejection_sampler.py`

Full rejection sampling implementation for draft model proposals. Requires both
target and draft probability distributions.

**Algorithm:**

```
For each sequence b in batch:
  For position i = 0 to k-1:
    r = uniform_random()
    if target_prob[b, i, draft_token[b, i]] / draft_prob[b, i, draft_token[b, i]] >= r:
      ACCEPT draft_token[b, i]
    else:
      REJECT -- resample from max(0, target_prob - draft_prob) normalized
      BREAK (subsequent positions all rejected)

  If all k tokens accepted:
    Sample BONUS token from target_prob[b, k, :]
```

**Vectorized implementation** uses the cumprod trick for left-to-right masking,
same as NGramVerifier but with the acceptance criterion being `p_target/p_draft >= r`
instead of exact match.

### P2.3: Draft Model Cache Management

The draft model maintains a separate KV cache that is synchronized with the target
model's cache at the start of each speculative step.

```
Start of spec step:
  target_cache: [B, H, S_target, D]  (contains all prompt + generated tokens)
  draft_cache = clone(target_cache)   (copy of target's cache state)

During draft generation (k steps):
  draft_cache grows by k positions (draft model's own forward passes)

After verification:
  draft_cache is DISCARDED (not needed after verification)
  target_cache is trimmed based on acceptance results
```

On Apple Silicon unified memory, the clone operation can potentially be a shallow
copy (view) since the draft cache is discarded before the target cache is modified.
However, this requires careful analysis of MLX's memory semantics.

---

## 8. 2-Node Tensor Parallel Design Considerations

### 8.1 Deployment Context

- **Current testing:** Single-machine development on MacBook with `mlx-community/Qwen3-4B-4bit` (locally available at `Qwen3-4B-4bit/`)
- **Production target:** 2x Mac Studio M3 Ultra 512GB, connected via Thunderbolt 5 RDMA (JACCL)
- **Production model:** Kimi K2.5 (~120GB 4-bit quantized), distributed across 2 nodes

### 8.2 N-gram Proposer (Phase 1) -- No TP Sync Needed

The N-gram proposer is fully deterministic and CPU-only. Given the same token context,
it produces identical proposals on every rank. Since the token context is synchronized
by construction (all ranks process the same tokens via the existing distributed bus),
no additional TP synchronization is required for N-gram proposals.

### 8.3 Draft Model Proposer (Phase 2) -- TP-Aware Design

The draft model introduces a design decision about placement:

**Option A: Draft on rank 0 only, broadcast proposals**
```
Rank 0: draft_model.propose(k) --> draft_tokens [B, k]
        all_sum(draft_tokens) --> broadcast to all ranks
Rank 1: receive draft_tokens via all_sum
Both:   target_model.forward(draft_tokens) --> verify
```

**Option B: Draft replicated on all ranks (preferred)**
```
All ranks: draft_model.propose(k) with synchronized seed
           --> identical draft_tokens [B, k] on each rank
           target_model.forward(draft_tokens) --> verify
```

Option B is preferred because:
- No additional all_sum calls during proposal
- Draft model is small (4-bit quantized), memory impact is minimal per rank
- Seed synchronization ensures identical proposals without communication
- Matches existing TP pattern where both ranks execute the same operations

### 8.4 Rank Consistency Requirements

All speculative decode decisions must be rank-consistent to avoid divergence:

| Component | Consistency Mechanism |
|-----------|----------------------|
| N-gram proposals | Deterministic (same context = same result) |
| Draft model proposals | Synchronized `mx.random.key()` per step |
| Verification decisions | Deterministic (same logits = same argmax) |
| Cache rollback amounts | Derived from verification (deterministic) |
| Dynamic controller state | Same inputs on all ranks = same EMA/k |

**CRITICAL:** The rejection sampler (Phase 2) uses random numbers. These must be
generated with a synchronized seed across ranks, or the sampler result must be
broadcast from rank 0 after sampling.

### 8.5 Distributed Bus Extension

The existing distributed bus (outbox pattern in `scheduler.py`) must be extended
for spec decode events:

- **New ControlEvent types:** `spec_decode_enable`, `spec_decode_disable`,
  `spec_decode_config_update` (for per-request overrides in future)
- **Spec decode metrics** can be rank-local (no sync needed -- rank 0 serves HTTP)
- **Proposal broadcast** (if using Option A): Add `spec_proposals` event type
  carrying `draft_tokens` array via `mx.distributed.all_sum`

No bus changes are needed for Phase 1 (N-gram is deterministic, no broadcast required).
Phase 2 implementation (P2.6) must add the bus extension for draft model proposals
if Option A is chosen.

### 8.6 Testing Strategy for TP

- Phase 1-2 development and testing: single-machine with `Qwen3-4B-4bit`
- TP correctness tests: use `mlx.launch` with 2 local ranks (as done in
  `tests/test_distributed.py`) to verify rank consistency
- Production validation: 2x Mac Studio with Kimi K2.5 after feature merge to develop

---

## 9. Testing Strategy

### 9.1 Test Hierarchy (no mock models)

```
Pure Logic Tests (no model needed)
    |
    +-- test_config.py            -- Config validation
    +-- test_ngram_proposer.py    -- N-gram matching correctness
    +-- test_controller.py        -- Dynamic control EMA + thresholds
    |
Real Model Tests (mlx-community/Qwen3-4B-4bit)
    |
    +-- test_ngram_verifier.py    -- Greedy/threshold verification
    +-- test_cache_utils.py       -- Cache trim operations
    +-- test_spec_engine.py       -- Full propose-verify-accept cycle
    +-- test_scheduler_integration.py -- Scheduler + SpecDecodeEngine
    +-- test_rejection_sampler.py -- Rejection sampling (Phase 2)
    |
End-to-End Tests (real model, Qwen3-4B-4bit)
    |
    +-- test_e2e_spec_decode.py   -- Server + spec decode with real model
```

**NOTE:** All engine and integration tests use `mlx-community/Qwen3-4B-4bit` (locally
available at `Qwen3-4B-4bit/`). There is no MockModel class. Pure logic tests (config
validation, NGramProposer pattern matching, DynamicSpecController EMA calculations)
do not need any model object.

### 9.2 Key Test Scenarios

| # | Scenario | Expected Behavior |
|---|----------|-------------------|
| 1 | **All accepted** | k draft tokens match target argmax, k+1 tokens emitted |
| 2 | **None accepted** | First draft mismatches, 1 correction token emitted |
| 3 | **Partial accept** | First m of k accepted, m+1 tokens emitted |
| 4 | **Mixed batch** | Different accept counts per sequence, all correct |
| 5 | **EOS during spec** | Draft proposes [a, b, EOS, c], stop after EOS |
| 6 | **max_tokens hit** | Would exceed limit, truncate accepted tokens |
| 7 | **Cache rollback** | After partial accept, cache matches non-spec decode |
| 8 | **Streaming order** | Multiple tokens emitted in correct sequence order |
| 9 | **No proposals** | N-gram finds no match, falls back to normal decode |
| 10 | **Batch size threshold** | Auto-disable when batch >= threshold |
| 11 | **Low acceptance EMA** | Auto-disable when EMA drops below threshold |
| 12 | **Adaptive k** | k adjusts as EMA changes |
| 13 | **Fallback on error** | Engine error falls back to BatchGenerator.next() |
| 14 | **Empty batch** | No active sequences, no-op |
| 15 | **Single sequence** | Batch size 1, spec decode works correctly |

---

## 10. Implementation Schedule

### Sprint 1: Foundation + Cache Investigation GATE (Days 1-2)

| Item | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| P0.1 | Package skeleton (dirs, __init__.py) | None | 1 |
| P0.2 | SpecDecodeConfig + tests | None | 2 |
| P0.3 | CLI args + ServerConfig fields | P0.2 | 2 |
| P1.1 | BaseProposer + ProposalResult + SpecResponse types | P0.1 | 2 |
| P1.5 | cache_utils + INVESTIGATION test (real model) — **GATE** | P0.1 | 4 |

**Sprint 1 deliverable:** Foundation complete. GATE result determines interface for Wave 1+.

### Sprint 2: Proposer + Verifier + Controller (Days 3-5) [parallel]

| Item | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| P1.2 | NGramProposer (linear search) + tests | P1.1 | 4 |
| P1.3 | NGramVerifier (greedy + threshold) + tests | P0.1 | 3 |
| P1.7 | DynamicSpecController (EMA + adaptive k) + tests | P0.2 | 3 |

**Sprint 2 deliverable:** Proposer, verifier, and controller work in isolation. All parallelizable.

### Sprint 3: Engine + Scheduler Integration (Days 6-9)

| Item | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| P1.4 | SpecDecodeEngine + tests (real model) | P1.1-P1.3, P1.5, P1.7 | 8 |
| P1.6 | Scheduler integration + _process_spec_responses | P1.4, P0.3 | 6 |
| P1.8 | End-to-end test with Qwen3-4B-4bit (streaming + non-streaming) | P1.6 | 4 |
| P1.9 | Metrics endpoint (/v1/spec_decode/metrics) | P1.7 | 2 |
| -- | Full test suite regression check (954+ tests) | All Phase 1 | 2 |
| -- | Devil's advocate review (Phase 1) | All Phase 1 | 4 |

**Sprint 3 deliverable:** Production-ready n-gram spec decode. Metrics exposed. All tests passing.

### Sprint 4: Draft Model (Phase 2) (Days 10-15)

| Item | Description | Dependencies | Est. Hours |
|------|-------------|--------------|------------|
| P2.1 | DraftModelProposer (load + propose) | P1.4 | 6 |
| P2.2 | BatchedRejectionSampler + tests | P0.1 | 5 |
| P2.3 | Draft model cache management | P2.1, P1.5 | 4 |
| P2.4 | Engine extension for draft model mode | P2.1-P2.3, P1.4 | 3 |
| P2.5 | CLI args for draft model | P0.3 | 1 |
| P2.6 | TP-aware draft placement (rank 0 or replicated) | P2.1 | 4 |
| P2.7 | End-to-end test with real draft + target models | P2.4 | 4 |
| -- | Devil's advocate review (Phase 2) | All Phase 2 | 4 |

**Sprint 4 deliverable:** Draft model spec decode operational with TP-aware placement.

### Multi-Agent Execution Structure

```
Wave 0: [Agent 1] Foundation + Cache Investigation (GATE)
         P0.1-P0.3, P1.1, P1.5 (real model Qwen3-4B-4bit)
         ↓ GATE: per-sequence offset manipulation success/fail
         ↓ Interface confirmed
    ┌────┼────────┐
    ▼    ▼        ▼
Wave 1: [Agent 2] [Agent 3] [Agent 4]  (parallel)
        NGram     NGram     Dynamic
        Proposer  Verifier  Controller
        P1.2      P1.3      P1.7
              │       │        │
              └───────┴────────┘
                      │
Wave 2: [Agent 5] SpecDecodeEngine P1.4
                      │
Wave 3: [Agent 6] Scheduler Integration P1.6 + P1.8 + P1.9
```

---

## 11. Risk Register

| ID | Risk | Impact | Probability | Mitigation |
|----|------|--------|-------------|------------|
| R1 | BatchKVCache per-sequence trim does not work via offset manipulation | HIGH | MEDIUM | Investigation test in Sprint 1 (P1.5) — GATE Result B confirmed. Phase 1: min_accepted uniform (no re-forward) |
| R2 | Model forward with padded multi-token input produces wrong logits | HIGH | LOW | MLX models already handle padded prefill; verify with attention mask |
| R3 | Cache state diverges between BatchGenerator and SpecDecodeEngine | CRITICAL | MEDIUM | Thorough state sync in _update_batch_state; integration tests |
| R4 | Draft model evicts target model from Metal buffer pool | MEDIUM | MEDIUM | Use 4-bit quantized draft; monitor mx.get_peak_memory() |
| R5 | N-gram linear search too slow for long contexts (>2K tokens) | LOW | LOW | Linear O(n*k) is fast up to ~2K; suffix index in Sprint 3 |
| R6 | Distributed mode rank divergence during spec decode | HIGH | LOW | N-gram is deterministic; draft model needs seed sync (see Section 8) |
| R7 | Spec decode overhead exceeds benefit at batch_size > 4 | MEDIUM | MEDIUM | Dynamic controller auto-disables; benchmark to tune threshold |
| R8 | Streaming token burst overwhelms client | LOW | LOW | Token queue maxsize=256 handles bursts; same as current prefill |
| R9 | EOS token in middle of draft sequence | MEDIUM | HIGH | Check EOS per accepted token, stop emitting at first EOS |
| R10 | max_tokens exceeded by multi-token step | MEDIUM | HIGH | Truncate accepted tokens to not exceed max_tokens |
| R11 | Qwen3 spec decode bug (mlx-lm issue #846) may affect real-model tests | MEDIUM | MEDIUM | Monitor issue; if triggered, pin to known-good mlx-lm commit or use workaround |
| R12 | 2-node TP rank divergence during spec decode -- draft proposals must be broadcast | HIGH | MEDIUM | Phase 1 (N-gram) is deterministic, no risk. Phase 2: use seed sync or rank 0 broadcast (P2.6) |

---

## 12. Key Design Decisions

### Decision 1: Bypass BatchGenerator for Verification, Not Replace It

**Rationale:** BatchGenerator handles prefill, sampling, prompt processing, and batch
management well. Replacing it entirely would duplicate significant functionality.
Instead, we bypass only its `next()` method during speculative decode steps, using
the same model and cache it manages.

**Trade-off:** Tight coupling with BatchGenerator internals (Batch dataclass, cache
structure). Changes to upstream BatchGenerator may break SpecDecodeEngine.

### Decision 2: N-gram First, Draft Model Second

**Rationale:** N-gram proposer is CPU-only with zero GPU overhead. It validates the
entire propose-verify-accept pipeline without any model loading complexity. For
repetitive tasks (code gen, translation), n-gram achieves 40-70% acceptance rates.

**Trade-off:** N-gram has low acceptance for creative/reasoning tasks. Draft model
is needed for universal speedup.

### Decision 3: Shadow Cache Approach Over Block Segments

**Rationale:** Spec decode operates on the in-flight cache managed by BatchGenerator.
We do NOT modify the block system during spec decode. Blocks are still only created
at sequence completion. This avoids complicating the KV cache manager.

**Trade-off:** No incremental block caching during generation. All blocks created
at once when the sequence finishes.

### Decision 4: Greedy Verification for N-gram, Rejection Sampling for Draft

**Rationale:** N-gram has no draft probabilities, so rejection sampling does not
apply. Greedy (argmax match) is simple, fast, and produces output identical to
non-speculative greedy decoding.

**Trade-off:** Greedy verification is strict -- it rejects draft tokens that the
target model would assign high probability but not argmax. Threshold mode is
available as a more permissive alternative.

### Decision 5: Dynamic Control Always On By Default

**Rationale:** Spec decode has overhead (proposing, verification forward pass). At
high batch sizes, this overhead outweighs the benefit of multi-token generation.
Auto-disable prevents performance degradation.

**Trade-off:** May miss speedup opportunities when the workload is suitable. Users
can set `--no-spec-decode-dynamic` to force spec decode always on.

### Decision 6: SpecResponse vs Extending BatchGenerator.Response

**Rationale:** We create a new `SpecResponse` type instead of modifying the upstream
`BatchGenerator.Response`. This keeps the upstream code untouched and makes the
spec decode integration clearly separable.

**Trade-off:** The scheduler must handle two response types. The
`_process_spec_responses()` method duplicates some logic from
`_process_batch_responses()`.

### Decision 7: Temperature-Aware Verification Mode

**Rationale:** N-gram greedy verification produces output identical to non-speculative
greedy decoding (temperature=0). But at temperature>0, greedy rejects any token that
isn't the exact argmax, even if the target model assigns it 35% probability. This
collapses acceptance rate to near-zero.

**Resolution:** Controller auto-switches to threshold mode when temperature>0.
Phase 2 (Draft Model) with proper rejection sampling will be lossless at any temperature.

**Trade-off:** Threshold mode at temperature>0 is lossy (output distribution differs
slightly from non-speculative). Acceptable for Phase 1; resolved properly in Phase 2.

---

## 13. File Ownership

| File | Owner | Branch |
|------|-------|--------|
| `mlx_lm_server/spec_decode/__init__.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/config.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/proposer/__init__.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/proposer/base.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/proposer/ngram.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/proposer/draft_model.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/verifier.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/rejection_sampler.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/engine.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/cache_utils.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/spec_decode/controller.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm_server/config.py` | Team Lead (modify) | develop |
| `mlx_lm_server/server.py` | server-agent (modify for CLI) | feature/spec-decode |
| `mlx_lm_server/scheduler.py` | scheduler-agent (modify for integration) | feature/spec-decode |
| `tests/spec_decode/*.py` | spec-decode-agent | feature/spec-decode |
| `mlx_lm/*` | UPSTREAM (READ ONLY) | -- |

---

## 14. Reference Materials

### 14.1 Codebase Files

| File | Path | Relevance |
|------|------|-----------|
| BatchGenerator | `mlx_lm/generate.py:930` | Primary integration point |
| Batch dataclass | `mlx_lm/generate.py:843` | State to sync with engine |
| speculative_generate_step | `mlx_lm/generate.py:469` | Upstream single-stream spec decode reference |
| BatchKVCache | `mlx_lm/models/cache.py:806` | Cache rollback mechanism |
| trim_prompt_cache | `mlx_lm/models/cache.py:93` | Uniform cache trimming |
| Scheduler | `mlx_lm_server/scheduler.py:883` | _batch_inference_step to modify |
| ServerConfig | `mlx_lm_server/config.py` | Add spec decode config fields |
| Types | `mlx_lm_server/types.py` | SequenceState, TokenEvent |

### 14.2 External References

| Reference | URL | Relevance |
|-----------|-----|-----------|
| vLLM prefix caching design | https://docs.vllm.ai/en/stable/design/prefix_caching/ | Block-based caching concepts |
| vLLM V1 rejection sampler | `vllm/v1/sample/rejection_sampler.py` | Vectorized rejection sampling |
| vLLM ngram worker | `vllm/spec_decode/ngram_worker.py` | N-gram proposal algorithm |
| vLLM spec decode worker | `vllm/spec_decode/spec_decode_worker.py` | disable_by_batch_size pattern |
| mlx-lm continuous batching article | https://medium.com/@clnaveen/mlx-lm-continuous-batching | Batching architecture |
| vllm-mlx paper | https://arxiv.org/html/2601.19139v2 | MLX-specific spec decode considerations |

### 14.3 CUDA vs Apple Silicon Translation Table

| vLLM (CUDA) | mlx-lm-server (Apple Silicon) |
|-------------|-------------------------------|
| PagedAttention block alloc/dealloc | Continuous memory + offset-based management |
| GPU-CPU explicit KV copy | Zero-copy (unified memory) |
| Batch expansion (tensor duplication) | Padding + attention mask |
| CUDA graphs (draft step fusion) | MLX lazy eval --> automatic op fusion |
| Block table management | Not needed -- OS-level virtual memory paging |
| `torch.where` / `torch.scatter` | `mx.where` / `mx.take_along_axis` |
| `torch.multinomial` | `mx.random.categorical` |
| `torch.cumprod` | `mx.cumprod` |
| `torch.softmax` | `mx.softmax` |

---

## Appendix A: BatchKVCache Investigation Protocol

Before implementing `batch_variable_trim()`, run this investigation:

### Test Setup

```python
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
```

### Test Procedure

1. Create a batch of 2 sequences with different prompt lengths
2. Run prefill to populate cache
3. Run 1 normal decode step to establish baseline
4. Run a multi-token forward pass (simulating spec decode verification)
5. Apply `batch_variable_trim` with different amounts per sequence
6. Run another 1-token forward pass
7. Compare output to a reference run without spec decode

### Expected Result

If the attention mask correctly handles per-sequence offsets, the trimmed cache
should produce identical output to a cache that was never extended beyond the trim
point. This validates the direct offset manipulation approach.

### If Test Fails

Switch to the uniform-trim-with-selective-re-forward approach described in
Section 3, Conflict 2.

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **k** | Number of draft tokens proposed per speculative step |
| **Draft token** | Token proposed by the proposer for verification |
| **Bonus token** | Extra token generated when all k drafts are accepted |
| **Correction token** | Token sampled at first rejection position |
| **Acceptance rate** | Fraction of draft tokens accepted by verifier |
| **EMA** | Exponential Moving Average of acceptance rate |
| **Greedy verification** | Accept draft iff it matches target argmax |
| **Rejection sampling** | Accept draft with probability p_target/p_draft |
| **Proposer** | Component that generates draft tokens (n-gram, draft model) |
| **Verifier** | Component that checks draft tokens against target model |
| **PLACEHOLDER_TOKEN_ID** | -1, used to mark rejected positions in output array |
