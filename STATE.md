# STATE.md — Project State Snapshot

**Date:** 2026-02-08
**Branch:** `develop`
**Latest commit:** `3ac7115` — `[P6.10+P7.6] feat(adversarial): devil's advocate review for Phase 6-7`
**Tests:** 258 pass, 1 xfail, 0 failures
**Python:** 3.14.3 (venv at `.venv/`)

---

## Phase Completion Status

| Phase | Description | Status | Tests Added |
|:-----:|-------------|:------:|:-----------:|
| P0 | Project setup, types, config | ✅ | — |
| P1 | KV Cache Manager + SSD Tier | ✅ | 70 |
| P2 | Continuous Batching Scheduler | ✅ | 26 |
| P3 | FastAPI Server (OpenAI API) | ✅ | 13 |
| P4 | Integration Tests + Benchmarks | ✅ | 12 |
| P5 | Final Integration + Polish + v0.1.0 | ✅ | 43 |
| P6 | BatchGenerator Integration | ✅ | 35 |
| P7 | Block-Level KV Cache Bridge + SSD | ✅ | 40 |
| P8 | Tensor-Parallel Serving (JACCL) | ⬜ Future | — |

---

## Source Files

| File | Lines | Purpose |
|------|:-----:|---------|
| `mlx_lm_server/scheduler.py` | 1032 | Core: continuous batching scheduler with batch + mock paths |
| `mlx_lm_server/kv_cache_manager.py` | 656 | Block pool, hash table, prefix caching, extract/inject, decompose/reconstruct |
| `mlx_lm_server/server.py` | 549 | FastAPI server, OpenAI-compatible API, CLI parser |
| `mlx_lm_server/ssd_cache.py` | 176 | SSD-backed KV cache tier (safetensors) |
| `mlx_lm_server/sequence_cache.py` | 137 | Sequence-level LRU KV cache store |
| `mlx_lm_server/types.py` | 86 | Shared data types (InferenceRequest, SequenceState, TokenEvent, etc.) |
| `mlx_lm_server/config.py` | 49 | ServerConfig dataclass |
| `mlx_lm_server/__main__.py` | 54 | Entry point: model load, scheduler, KVCacheManager, SSD, FastAPI |
| `mlx_lm_server/__init__.py` | 3 | Package init |

**Total source:** ~2,742 lines

---

## Test Files

| File | Tests | Phase | What |
|------|:-----:|:-----:|------|
| `tests/test_adversarial.py` | 106 | P1-P7 | Devil's advocate adversarial tests (DA-P1 through DA-P7 + DA-F) |
| `tests/test_kv_cache_manager.py` | 54 | P1 | Block pool, hash, prefix, alloc, free, evict, extract, inject, tiered |
| `tests/test_scheduler.py` | 26 | P2 | Queue, scheduling, lifecycle, streaming, cancellation |
| `tests/test_ssd_cache.py` | 16 | P1 | SSD init, save, load, prune, index persistence |
| `tests/test_server.py` | 13 | P3 | API endpoints, CLI, shutdown, streaming |
| `tests/test_integration.py` | 12 | P4 | E2E: basic, prefix cache, SSD tier, concurrent |
| `tests/test_sequence_cache.py` | 12 | P6 | SequenceCacheStore: store, prefix, LRU, thread safety |
| `tests/test_block_bridge.py` | 12 | P7 | decompose/reconstruct, hash consistency, prefix sharing |
| `tests/test_batch_integration.py` | 7 | P6 | MockBatchGenerator: single, concurrent, cancel, stream, error |

**Total project tests:** 258 (+ 1 xfail)

---

## Architecture

```
Client (OpenAI SDK / curl)
    │
    ▼
FastAPI Server (server.py)
  /v1/chat/completions, /v1/completions, /v1/models, /health
    │
    ▼
Scheduler (scheduler.py)
  ├── _batch_inference_step()     ← real model (BatchGenerator)
  │   ├── _process_cancellations_batch()
  │   ├── _insert_new_requests_batch()
  │   │   ├── Block-level cache lookup (KVCacheManager)
  │   │   ├── Sequence-level cache lookup (SequenceCacheStore)
  │   │   └── BatchGenerator.insert(prompts, caches, samplers)
  │   ├── BatchGenerator.next()   ← ONE call, all sequences
  │   ├── Per-response: detokenize, stop check, emit TokenEvent
  │   ├── Store caches (sequence + block decomposition)
  │   ├── Periodic SSD pruning
  │   └── Cleanup finished sequences
  └── _mock_inference_step()      ← testing (model=None)
    │
    ▼
KV Cache Manager (kv_cache_manager.py)
  Block Pool + Hash Table + LRU Eviction
  extract_block() / inject_blocks()
  decompose_cache_to_blocks() / reconstruct_cache_from_blocks()
    │
    ▼
TieredKVCache
  RAM (KVCacheManager) → SSD (SSDCache, safetensors)
    │
    ▼
mlx-lm Engine (BatchGenerator)
  MLX / Metal (Apple Silicon)
```

---

## Cache Lookup Priority

```
1. Block-level cache (KVCacheManager.find_cached_prefix)
   → reconstruct from blocks → BatchGenerator.insert(caches=...)
   Finer granularity, sub-sequence sharing

2. Sequence-level cache (SequenceCacheStore.find_longest_prefix)
   → direct List[KVCache] → BatchGenerator.insert(caches=...)
   Faster, no reconstruction overhead

3. Miss → full prefill by BatchGenerator
```

---

## Config Fields (ServerConfig)

| Field | Type | Default | Phase |
|-------|------|---------|:-----:|
| `model` | str | `mlx-community/Qwen3-4B-4bit` | P0 |
| `adapter_path` | str\|None | None | P0 |
| `host` | str | `0.0.0.0` | P0 |
| `port` | int | 8000 | P0 |
| `block_size` | int | 16 | P1 |
| `num_blocks` | int | 2048 | P1 |
| `kv_bits` | int | 8 | P1 |
| `kv_group_size` | int | 64 | P1 |
| `ssd_cache_dir` | Path | `~/.cache/mlx-lm-server/kv-cache` | P1 |
| `ssd_ttl_days` | int | 7 | P1 |
| `ssd_enabled` | bool | True | P1 |
| `max_batch_size` | int | 8 | P2 |
| `prefill_batch_size` | int | 4 | P2 |
| `prefill_step_size` | int | 2048 | P2 |
| `max_queue_size` | int | 128 | P2 |
| `completion_batch_size` | int | 32 | P6 |
| `max_kv_size` | int\|None | None | P6 |
| `sequence_cache_size` | int | 50 | P6 |
| `default_max_tokens` | int | 512 | P3 |
| `default_temperature` | float | 1.0 | P3 |
| `default_top_p` | float | 1.0 | P3 |
| `use_distributed` | bool | False | P8 |

---

## Git History (chronological)

```
686140b [P0]          chore: project setup
15182dd [P1.1-P1.6]   feat(kv-cache): block pool, hash, alloc/free/evict
b42aaf0 [P1.7-P1.18]  feat(kv-cache): MLX adapter, SSD tier, tiered lookup
e36869a [P2.1-P2.14]  feat(sched): continuous batching scheduler
7ca6778 [P3.1-P3.12]  feat(api): FastAPI server with OpenAI API
3e2b354 [DA-P1-P3]    fix: adversarial review bug fixes + 43 tests
75ab855 [P4.1-P4.6]   feat(tests): integration tests + benchmark script
844fcda [P5.4-P5.10]  feat: DA-Final + polish + README
d02f3ee [P5.11]       chore: tag v0.1.0
beeac7b [P5+]         feat(scheduler): wire up real model generation
6c74cf4 [P5+]         fix(scheduler): handle max_tokens=0 bug
a7b86bf [P6-P7]       feat(scheduler): BatchGenerator + block cache bridge
3ac7115 [P6.10+P7.6]  feat(adversarial): DA review Phase 6-7
```

---

## Devil's Advocate Findings (Fixed)

| ID | Severity | Fix | Phase |
|----|----------|-----|:-----:|
| DA-P1-4 | HIGH | Corrupted safetensors graceful handling | P1 |
| DA-P1-5 | HIGH | Pool exhaustion → evict+retry | P1 |
| DA-P2-5 | HIGH | Model exception → scheduler state recovery | P2 |
| DA-P6-C1 | HIGH | `_cancelled` set cleared in `_handle_batch_error()` | P6 |
| DA-P7-H1 | HIGH | Block kv_data stored by copy not reference | P7 |

---

## What's Next (Phase 8 — Future)

1. Tensor-parallel serving via JACCL RDMA
2. `mlx.launch --backend jaccl` wrapper
3. KV cache blocks sharded per rank
4. Only rank 0 runs FastAPI + scheduler
5. Test with 2x Mac Studio M3 Ultra + Kimi K2.5 (4-bit, ~120GB)

---

## Dev Environment

```bash
# Activate
source .venv/bin/activate

# Run tests (project only)
pytest tests/test_kv_cache_manager.py tests/test_ssd_cache.py tests/test_scheduler.py \
  tests/test_server.py tests/test_integration.py tests/test_adversarial.py \
  tests/test_sequence_cache.py tests/test_batch_integration.py tests/test_block_bridge.py \
  -v --tb=short

# Run server
python -m mlx_lm_server --model mlx-community/Qwen3-4B-4bit --port 8000

# Test with curl
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```
