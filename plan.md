# plan.md — Implementation Plan for mlx-lm-server

## Overview

This plan upgrades Apple's mlx-lm into a production-grade LLM serving engine with:
1. **Continuous batching** — iteration-level scheduling for concurrent requests
2. **Hash-based automatic prefix caching** — vLLM-style block-level KV cache reuse
3. **Tiered KV cache (RAM + SSD)** — with TTL-based pruning
4. **INT8 KV cache quantization** — enabled by default
5. **OpenAI-compatible API server** — FastAPI-based, streaming support
6. **Tensor-parallel compatibility** — works with JACCL RDMA distributed inference

**Test model:** `mlx-community/Qwen3-4B-4bit` (already downloaded locally)
**License:** MIT (inherited from mlx-lm) — all dependencies are MIT or Apache 2.0

### Workflow Per Task (ALL agents follow this)

```
1. Main agent picks task (e.g., P1.3)
2. Main agent spawns subagent with specific prompt
3. Subagent: reads files → implements → writes test → runs test → fixes failures
4. Subagent reports back: summary + test results
5. Main agent reviews output (quick scan)
6. Main agent commits + pushes:
   git add -A
   git commit -m "[P1.3] feat(kv-cache): implement block hash computation"
   git push origin feature/kv-cache-manager
7. Main agent picks next task (P1.4)
```

### Quality Gates

```
Per-Task Gate:      pytest tests/test_<component>.py::<test> -v     → PASS
Per-Feature Gate:   pytest tests/ -v --tb=short                      → ALL PASS
Devil's Adv Gate:   adversarial review → CRITICAL/HIGH findings fixed → ALL PASS
Merge Gate:         Team Lead merges to develop                      → ALL PASS
```

---

## Phase 0: Project Setup & Interface Definition

**Owner:** Team Lead
**Branch:** `develop`
**Duration:** ~30 min
**Blocking:** All other phases depend on this

### Tasks

**P0.1** Fork `ml-explore/mlx-lm` on GitHub

**P0.2** Clone and setup git structure:
```bash
git clone https://github.com/<user>/mlx-lm.git mlx-lm-server
cd mlx-lm-server
git remote add upstream https://github.com/ml-explore/mlx-lm.git
git checkout -b develop main
git push origin develop
```

**P0.3** Create all feature branches from develop:
```bash
for b in feature/kv-cache-manager feature/ssd-cache-tier feature/scheduler feature/api-server feature/tests-benchmarks; do
  git checkout -b $b develop && git push origin $b
done
git checkout develop
```

**P0.4** Create directory structure (`mlx_lm_server/`, `tests/`, `scripts/`)

**P0.5** Define `types.py` — shared data structures:

```python
@dataclass
class InferenceRequest:
    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: list[str]
    stream: bool
    created_at: float

@dataclass
class KVCacheBlock:
    block_id: int
    block_hash: int | None
    token_ids: list[int]       # Stored for hash collision verification
    ref_count: int
    last_accessed: float

@dataclass
class SequenceState:
    request_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int
    is_finished: bool
    finish_reason: str | None
    output_text: str

@dataclass
class SchedulerOutputs:
    prefill_sequences: list[SequenceState]
    decode_sequences: list[SequenceState]
    preempted_sequences: list[SequenceState]

@dataclass
class TokenEvent:
    request_id: str
    token_id: int
    token_text: str
    finish_reason: str | None

@dataclass
class SSDBlockMeta:
    block_hash: int
    filepath: Path
    last_accessed: datetime
    num_tokens: int
```

**P0.6** Define `config.py` (ServerConfig dataclass)

**P0.7** Verify test model and BatchGenerator

**P0.8** Install deps: `pip install -e ".[dev]" fastapi uvicorn pytest httpx`

**P0.9** Commit + push: `[P0] chore: project setup`

---

## Phase 1: KV Cache Manager + SSD Tier

**Owner:** cache-agent
**Branch:** `feature/kv-cache-manager` (1.1–1.2, 1.4), `feature/ssd-cache-tier` (1.3)
**Dependencies:** Phase 0 complete
**Subagent policy:** Delegate ALL implementation/test tasks to subagents

### 1.1 Block Pool & Hash Table

**P1.1** Block pool pre-allocation
→ Subagent: implement + test → commit → push

**P1.2** `compute_block_hash()` with prefix-aware hashing
→ **Note:** Store `token_ids` in `KVCacheBlock` for collision verification (see devil's advocate concerns)
→ Subagent: implement + test → commit → push

**P1.3** `find_cached_prefix()` — returns num cached tokens (block-aligned)
→ Subagent: implement + test (full hit, partial, miss, empty) → commit → push

**P1.4** `allocate_blocks()` with hash table reuse + ref_count
→ **Note:** Use `threading.Lock` around ref_count mutation for thread safety
→ Subagent: implement + test → commit → push

**P1.5** `free_blocks()` — ref_count decrement, block stays in hash table when reaching 0
→ Subagent: implement + test → commit → push

**P1.6** `evict_lru()` — pop oldest ref_count=0 block
→ Subagent: implement + test → commit → push

**GATE:** `pytest tests/test_kv_cache_manager.py -v` → ALL PASS

### 1.2 MLX KV Cache Adapter

**P1.7** Study mlx-lm cache format (subagent reads source, documents shapes)

**P1.8** `extract_block()` — slice per-layer K/V arrays

**P1.9** `inject_blocks()` — reconstruct cache from blocks

**P1.10** Roundtrip validation (extract → inject → generate → compare)

**GATE:** `pytest tests/test_kv_cache_manager.py -v` → ALL PASS

### 1.3 SSD Tier — Branch: `feature/ssd-cache-tier`

**P1.11** SSD cache init + directory creation

**P1.12** `save_block()` to safetensors

**P1.13** `load_block()` from safetensors + update timestamp

**P1.14** `prune_expired()` — TTL-based deletion

**P1.15** `save_index()` / `load_index()` — JSON persistence

**GATE:** `pytest tests/test_ssd_cache.py -v` → ALL PASS

### 1.4 Tiered Lookup — Branch: `feature/kv-cache-manager`

**P1.16** `TieredKVCache.lookup()` — RAM → SSD → miss

**P1.17** Eviction → SSD demotion flow

**P1.18** Full tiered integration test

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

### 1.5 Devil's Advocate Review: Phase 1

**Owner:** devil's-advocate-agent
**Activates:** After P1.18 passes feature gate

Devil's advocate reads all Phase 1 code and writes adversarial tests targeting:

| ID | Domain | Attack Vector |
|----|--------|--------------|
| DA-P1-1 | Concurrency | Concurrent `free_blocks()` on same block — race on ref_count |
| DA-P1-2 | Hash collision | Two different token sequences → same hash → wrong KV returned |
| DA-P1-3 | Memory leak | Sequence errors out mid-generation → blocks never freed |
| DA-P1-4 | SSD corruption | `load_block()` with corrupted/truncated safetensors file |
| DA-P1-5 | Exhaustion | All blocks allocated → new request arrives → what happens? |
| DA-P1-6 | TTL edge | Block saved at T=0, TTL=7d, prune runs at T=6d23h59m → safe? |
| DA-P1-7 | Stale SSD index | Process crashes between save_block and save_index → index stale |

Process:
1. Subagent writes adversarial tests in `tests/test_adversarial.py`
2. Run tests — expect some to FAIL (exposing real bugs)
3. File findings with CRITICAL/HIGH/MEDIUM/LOW severity
4. cache-agent fixes CRITICAL + HIGH (via subagent)
5. Re-run: `pytest tests/test_adversarial.py -v` → ALL PASS
6. → Notify Team Lead: "Phase 1 reviewed + hardened, ready for merge"

---

## Phase 2: Continuous Batching Scheduler

**Owner:** scheduler-agent
**Branch:** `feature/scheduler`
**Dependencies:** Phase 0 + Phase 1 interfaces
**Subagent policy:** Delegate ALL tasks to subagents. Decision points (P2.6) in main context.

### 2.1 Request Queue

**P2.1** `RequestQueue` — thread-safe add/pop_batch with `threading.Lock`

### 2.2 Scheduler Core

**P2.2** Scheduler `__init__()`

**P2.3** `schedule_step()` — remove finished → fill slots → classify prefill/decode

**P2.4** `_init_sequence()` — tokenize, allocate blocks, check prefix cache

**P2.5** `_run_prefill()` — process uncached tokens

**P2.6** `_run_decode_step()` — ⚠️ **DECISION POINT (main agent)**
  - Option A: BatchGenerator with external KV cache
  - Option B: generate_step() with manual batching
  - Test both, document decision

**P2.7** `run_inference_loop()` — main thread loop

**P2.8** `register_stream()` — token queue for SSE

### 2.3 Sequence Lifecycle

**P2.9** Stop sequence + EOS detection

**P2.10** Max tokens enforcement

**P2.11** Request cancellation on disconnect

**P2.12–P2.14** Lifecycle, continuous batching, prefix hit tests

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

### 2.4 Devil's Advocate Review: Phase 2

**Owner:** devil's-advocate-agent
**Activates:** After P2.14 passes feature gate

| ID | Domain | Attack Vector |
|----|--------|--------------|
| DA-P2-1 | Concurrency | schedule_step() called while previous step still running |
| DA-P2-2 | Deadlock | Request queue lock held while waiting for inference result |
| DA-P2-3 | Starvation | Long-running sequence blocks all new requests indefinitely |
| DA-P2-4 | Memory | 100 requests queued but max_batch_size=8 → memory of waiting seqs |
| DA-P2-5 | Error recovery | Model raises exception mid-decode → scheduler state corrupted? |
| DA-P2-6 | Stream abort | Client disconnects mid-stream → token_queue orphaned → leak? |
| DA-P2-7 | Edge case | max_tokens=0, empty prompt, prompt longer than context window |

Same process: adversarial tests → findings → fixes → re-pass → merge ready.

---

## Phase 3: FastAPI Server

**Owner:** server-agent
**Branch:** `feature/api-server`
**Dependencies:** Phase 0 + Phase 2 interfaces
**Subagent policy:** Delegate ALL tasks to subagents

### 3.1 API Endpoints

**P3.1** `POST /v1/chat/completions` (non-streaming)

**P3.2** `POST /v1/chat/completions` (SSE streaming)

**P3.3** `POST /v1/completions`

**P3.4** `GET /v1/models`

**P3.5** `GET /health` with cache stats

### 3.2 Infrastructure

**P3.6** CLI argument parser → ServerConfig

**P3.7** Server startup (load → init → threads → uvicorn)

**P3.8** Graceful shutdown with SSD flush

**P3.9** Error handling + validation

### 3.3 Entry Point

**P3.10** `__main__.py`

**P3.11–P3.13** Concurrent, streaming parity tests

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

### 3.4 Devil's Advocate Review: Phase 3

**Owner:** devil's-advocate-agent
**Activates:** After P3.13 passes feature gate

| ID | Domain | Attack Vector |
|----|--------|--------------|
| DA-P3-1 | API contract | Missing required OpenAI response fields → client SDK crashes |
| DA-P3-2 | Input validation | Malformed JSON, missing `messages`, negative `max_tokens` |
| DA-P3-3 | SSE format | Missing `data: ` prefix, missing `[DONE]`, newline issues |
| DA-P3-4 | Concurrency | 50 simultaneous requests → server hangs or crashes |
| DA-P3-5 | Shutdown | Request in-flight during shutdown → partial response? hang? |
| DA-P3-6 | Memory | Streaming response never consumed → server-side buffer grows |
| DA-P3-7 | Security | Extremely long prompt (1M tokens) → OOM? Timeout? |

Same process: adversarial tests → findings → fixes → re-pass → merge ready.

---

## Phase 4: Integration Tests & Benchmarks

**Owner:** test-agent
**Branch:** `feature/tests-benchmarks`
**Dependencies:** Phase 0 (interfaces); starts parallel with P1–P3

### 4.1 Shared Fixtures

**P4.1** `conftest.py` — mock model, test config, temp dirs, real model fixture

### 4.2 Integration Tests

**P4.2** E2E: server → request → response

**P4.3** E2E: prefix caching verification

**P4.4** E2E: SSD tier roundtrip

**P4.5** E2E: 4+ concurrent requests

### 4.3 Benchmarks

**P4.6** `scripts/benchmark.py` (throughput, TTFT, ITL, cache rates)

**P4.7** Run benchmarks, write `BENCHMARKS.md`

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

---

## Phase 5: Final Integration & Polish

**Owner:** Team Lead
**Branch:** `develop`
**Dependencies:** All phases + all devil's advocate reviews complete

**P5.1** Merge all feature branches to develop (`--no-ff`)

**P5.2** Resolve merge conflicts

**P5.3** `pytest tests/ -v --tb=short` → ALL PASS on develop

**P5.4** Final devil's advocate review on integrated codebase (cross-component interactions)

**P5.5** Run benchmarks on develop

**P5.6** Write `README.md`

**P5.7** Add `pyproject.toml` CLI entry point

**P5.8** Document JACCL distributed setup

**P5.9** Code review: docstrings, dead code

**P5.10** Tag `v0.1.0`: `git tag v0.1.0 && git push origin develop --tags`

### 5.1 Final Devil's Advocate: Cross-Component Review

After all branches are merged, devil's-advocate-agent does one final review focused on **cross-component interactions** that individual phase reviews couldn't catch:

| ID | Domain | Attack Vector |
|----|--------|--------------|
| DA-F-1 | End-to-end | Request → scheduler → cache → model → response: any state leak between requests? |
| DA-F-2 | Cache+Scheduler | Scheduler frees blocks while SSD save is in progress → corruption? |
| DA-F-3 | Server+Scheduler | FastAPI async handler awaits scheduler future → scheduler thread dies → hang? |
| DA-F-4 | Load test | 20 requests, shared prefix, some streaming, some sync → everything works? |
| DA-F-5 | Restart | Server stops → restarts → SSD cache loads correctly → prefix hits resume? |

---

## Phase 6: BatchGenerator Integration

**Owner:** batch-integrator | **Branch:** `develop`
**Dependencies:** Phase 5 complete (v0.1.0)
**Status:** ✅ COMPLETE (all tasks done, 202 tests passing)
**Purpose:** Replace sequential `stream_generate()` path with `BatchGenerator` for real batched inference.

### Design Decisions

1. **Two paths:** `_batch_inference_step()` (model != None) vs `_mock_inference_step()` (model == None). All 183 existing tests hit mock path exclusively.
2. **Single inference thread:** `BatchGenerator.next()` uses `mx.stream(generation_stream)` — must be one thread. Already our design.
3. **Per-request samplers:** `make_sampler(temp, top_p)` passed to `BatchGenerator.insert(samplers=[sampler])`.
4. **Per-request detokenizers:** `copy.copy(tokenizer.detokenizer)` per sequence, `add_token()` + `last_segment` per batch step.

### Tasks

**P6.0** Config additions: `completion_batch_size`, `max_kv_size`, `sequence_cache_size` → `ServerConfig`

**P6.1** `SequenceCacheStore` — thread-safe LRU cache mapping token sequences → `List[KVCache]`. Modeled on upstream `LRUPromptCache`. `OrderedDict`, `threading.Lock`, `deepcopy`, prefix-aware lookup with trim.

**P6.2** BatchGenerator lifecycle: `_create_batch_generator()`, UID maps (`_uid_to_request_id`, `_request_id_to_uid`), auto-init when `model is not None`.

**P6.3** Batch inference loop (`_batch_inference_step`):
1. `_process_cancellations_batch()` — remove cancelled UIDs
2. `_insert_new_requests_batch()` — pop queue, cache lookup, `batch_generator.insert()`
3. `responses = batch_generator.next()` — ONE call, all sequences
4. Process responses: detokenize, update state, check stops, emit events
5. `batch_generator.remove()` early-stopped UIDs, extract prompt caches
6. Store caches in `SequenceCacheStore`
7. Clean up finished sequences

**P6.4** Per-sequence detokenization via `copy.copy(tokenizer.detokenizer)`.

**P6.5** Error recovery: `_handle_batch_error()` marks all active as failed, recreates BatchGenerator, clears UID maps.

**P6.6** Mock path preserved verbatim as `_mock_inference_step()`.

**P6.7** Tests: 12 unit tests for `SequenceCacheStore`, 7 integration tests for batch path using `MockBatchGenerator`.

**P6.8** Wire `__main__.py` with model → scheduler → KVCacheManager → SSD tier.

**P6.9** Update docs (this file + checklist.md).

**P6.10** Devil's advocate review: UID leaks, concurrent insert+next, detokenizer state, cache mutation, error cascades, stop sequences, max_tokens=0, batch overflow.

---

## Phase 7: Block-Level KV Cache Bridge + SSD Wiring

**Owner:** cache-bridge | **Branch:** `develop`
**Dependencies:** Phase 6.3+ complete
**Status:** ✅ COMPLETE (all tasks done, 202 tests passing)
**Purpose:** Connect block-level KVCacheManager to BatchGenerator, wire SSD tier for eviction/reloading.

### Design Decisions

1. **Two-tier cache lookup:** Block-level (finer granularity, sub-sequence sharing) → Sequence-level (faster, no reconstruction) → miss.
2. **Cache format bridge:** `decompose_cache_to_blocks()` slices `List[KVCache]` along seq_len axis into block_size chunks with hashes. `reconstruct_cache_from_blocks()` reverses via `make_prompt_cache()` + `inject_blocks()`.
3. **SSD integration:** On block pool exhaustion, `tiered_cache.evict_to_ssd()` demotes LRU blocks before allocating new ones.
4. **Periodic pruning:** Every N inference steps, call `ssd.prune_expired()`.

### Tasks

**P7.1** Cache format bridge functions in `kv_cache_manager.py`:
- `decompose_cache_to_blocks(prompt_cache, token_ids, block_size)` → list of block dicts with hash, tokens, KV data
- `reconstruct_cache_from_blocks(blocks, model)` → `List[KVCache]` ready for `BatchGenerator.insert(caches=...)`

**P7.2** Wire KVCacheManager into scheduler:
- Block-level cache lookup in `_insert_new_requests_batch()` (before sequence-level fallback)
- Block decomposition of finished caches in `_batch_inference_step()` step 7
- Block cleanup in `_cleanup_finished_batch()` via `free_blocks()`

**P7.3** SSD tier wiring:
- On block pool full during decomposition, `tiered_cache.evict_to_ssd()` before allocating
- Falls back to `_evict_lru_locked()` if no tiered cache

**P7.4** Periodic SSD pruning: counter-based, every `_ssd_prune_interval` steps.

**P7.5** Block bridge tests: 12 tests covering decomposition, roundtrip, hash consistency, prefix sharing, concurrency.

**P7.6** Devil's advocate review: hash collisions, partial blocks, shape mismatches, SSD corruption, ref count leaks, memory pressure, concurrent access.

---

## Phase 8 (Future): Tensor-Parallel Serving

Not in initial scope. Documented for reference:
1. Wrap server startup in `mlx.launch --backend jaccl`
2. KV cache blocks sharded per rank
3. Only rank 0 runs FastAPI + scheduler
4. All ranks participate in forward pass via `mx.distributed`
5. Test with 2x Mac Studio M3 Ultra + Kimi K2.5

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| BatchGenerator no external KV cache | High | Fall back to manual generate_step() |
| Block-level cache slicing complex | Medium | Study cache_prompt.py patterns |
| Hash collisions cause wrong KV | Medium | Store token_ids for verification (devil's advocate finding) |
| Race conditions in cache manager | Medium | Lock around ref_count mutations (devil's advocate finding) |
| INT8 + prefix caching interaction | Low | Thorough adversarial testing |
| SSD I/O spikes | Low | Async I/O; M3 NVMe is 7.4 GB/s |
| Continuous batching overhead | Low | Acceptable; bypass for single request |
