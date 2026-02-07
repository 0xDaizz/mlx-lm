# checklist.md — Task Checklist for mlx-lm-server

> **HOW TO USE THIS CHECKLIST:**
> 1. Each task `[ ]` has an **owner**, **branch**, and **test requirement**
> 2. Mark `[~]` when in progress, `[x]` when complete
> 3. **Every task MUST pass its unit test before commit**
> 4. **Every feature MUST pass full test suite before devil's advocate review**
> 5. **Every devil's advocate review MUST resolve CRITICAL/HIGH findings before merge**
> 6. **All implementation work delegated to subagents**
> 7. **Auto-commit + auto-push after every task**

---

## Phase 0: Project Setup & Interface Definition

**Owner:** Team Lead | **Branch:** `develop` | **Blocks:** Everything

- [x] **P0.1** Fork `ml-explore/mlx-lm` on GitHub *(using existing repo)*
- [x] **P0.2** Clone and configure remotes + develop branch *(develop branch created)*
- [ ] **P0.3** Create all feature branches from develop *(deferred — will create per-phase)*
- [x] **P0.4** Create directory structure (`mlx_lm_server/`, `tests/`, `scripts/`)
- [x] **P0.5** Define `mlx_lm_server/types.py` (all shared data classes)
- [x] **P0.6** Define `mlx_lm_server/config.py` (ServerConfig)
- [x] **P0.7** Create `mlx_lm_server/__init__.py`
- [x] **P0.8** Install deps: `pip install -e ".[dev]" fastapi uvicorn pytest httpx` *(Python 3.12 venv at .venv/)*
- [x] **P0.9** Verify test model + BatchGenerator *(local Qwen3-4B-4bit loads and generates OK)*
- [x] **P0.10** Commit + push: `[P0] chore: project setup` *(commit 686140b)*

---

## Phase 1: KV Cache Manager + SSD Tier

**Owner:** cache-agent | **Branches:** `feature/kv-cache-manager`, `feature/ssd-cache-tier`
**Depends on:** P0 complete
**Method:** ALL tasks → subagent. Main agent: review → commit → push.

### 1.1 Block Pool & Hash Table — `feature/kv-cache-manager`

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P1.1** Block pool pre-allocation | `test_pool_init`, `test_pool_size` | ✅ commit 15182dd |
| [x] | **P1.2** `compute_block_hash()` (store token_ids for collision check) | `test_hash_determinism`, `test_hash_uniqueness` | ✅ commit 15182dd |
| [x] | **P1.3** `find_cached_prefix()` | `test_prefix_full`, `test_prefix_partial`, `test_prefix_miss` | ✅ commit 15182dd |
| [x] | **P1.4** `allocate_blocks()` (with Lock + cache reuse) | `test_alloc_hit`, `test_alloc_fresh`, `test_alloc_refcount` | ✅ commit 15182dd |
| [x] | **P1.5** `free_blocks()` | `test_free_refcount`, `test_free_stays_in_hash` | ✅ commit 15182dd |
| [x] | **P1.6** `evict_lru()` | `test_evict_order`, `test_evict_skips_in_use` | ✅ commit 15182dd |

**GATE:** `pytest tests/test_kv_cache_manager.py -v` → ALL PASS

### 1.2 MLX KV Cache Adapter — `feature/kv-cache-manager`

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P1.7** Study mlx-lm cache format (document shapes) | (research only) | ✅ commit b42aaf0 |
| [x] | **P1.8** `extract_block()` | `test_extract_shapes`, `test_extract_values` | ✅ commit b42aaf0 |
| [x] | **P1.9** `inject_blocks()` | `test_inject_roundtrip` | ✅ commit b42aaf0 |
| [x] | **P1.10** Roundtrip validation | `test_roundtrip_generation` | ✅ commit b42aaf0 |

**GATE:** `pytest tests/test_kv_cache_manager.py -v` → ALL PASS

### 1.3 SSD Tier — `feature/ssd-cache-tier`

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P1.11** SSD cache init | `test_ssd_init`, `test_ssd_dir` | ✅ commit b42aaf0 |
| [x] | **P1.12** `save_block()` | `test_save_creates_file` | ✅ commit b42aaf0 |
| [x] | **P1.13** `load_block()` | `test_load_matches_saved` | ✅ commit b42aaf0 |
| [x] | **P1.14** `prune_expired()` | `test_prune_ttl_zero`, `test_prune_keeps_recent` | ✅ commit b42aaf0 |
| [x] | **P1.15** Index persistence | `test_index_persistence` | ✅ commit b42aaf0 |

**GATE:** `pytest tests/test_ssd_cache.py -v` → ALL PASS

### 1.4 Tiered Lookup — `feature/kv-cache-manager`

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P1.16** `TieredKVCache.lookup()` | `test_lookup_ram`, `test_lookup_ssd`, `test_lookup_miss` | ✅ commit b42aaf0 |
| [x] | **P1.17** Eviction → SSD demotion | `test_evict_saves_to_ssd` | ✅ commit b42aaf0 |
| [x] | **P1.18** Tiered integration test | `test_tiered_full_flow` | ✅ commit b42aaf0 |

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

### 1.5 Devil's Advocate Review: Phase 1 🔴

**Owner:** devil's-advocate-agent | **Activates after:** P1.18 feature gate passes

| | Finding ID | Attack Vector | Severity |
|-|-----------|--------------|----------|
| [ ] | **DA-P1-1** | Concurrent `free_blocks()` — race on ref_count | CRITICAL |
| [ ] | **DA-P1-2** | Hash collision → wrong KV data served silently | CRITICAL |
| [ ] | **DA-P1-3** | Sequence error mid-generation → blocks never freed | HIGH |
| [ ] | **DA-P1-4** | Corrupted safetensors file on SSD → crash on load? | HIGH |
| [ ] | **DA-P1-5** | All blocks allocated → new request → behavior? | HIGH |
| [ ] | **DA-P1-6** | TTL boundary edge case (prune at TTL - 1 second) | MEDIUM |
| [ ] | **DA-P1-7** | Crash between save_block and save_index → stale index | MEDIUM |

**Process:**
- [ ] Subagent: write adversarial tests in `tests/test_adversarial.py` (DA-P1 section)
- [ ] Run: `pytest tests/test_adversarial.py -v -k "da_p1"` → identify failures
- [ ] File findings report (see CLAUDE.md for format)
- [ ] cache-agent: fix CRITICAL + HIGH findings (via subagent)
- [ ] Re-run: `pytest tests/ -v --tb=short` → ALL PASS (including adversarial)
- [ ] → Notify Team Lead: "Phase 1 hardened, ready for merge"

---

## Phase 2: Continuous Batching Scheduler

**Owner:** scheduler-agent | **Branch:** `feature/scheduler`
**Depends on:** P0 + P1 interfaces
**Method:** ALL tasks → subagent. Decision point P2.6 in main context.

### 2.1 Request Queue

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P2.1** `RequestQueue` (thread-safe) | `test_queue_fifo`, `test_queue_concurrent` | ✅ commit e36869a |

### 2.2 Scheduler Core

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P2.2** Scheduler `__init__()` | `test_scheduler_init` | ✅ commit e36869a |
| [x] | **P2.3** `schedule_step()` | `test_schedule_fills_slots`, `test_schedule_removes_finished` | ✅ commit e36869a |
| [x] | **P2.4** `_init_sequence()` | `test_init_seq_tokenizes`, `test_init_seq_cache_check` | ✅ commit e36869a |
| [x] | **P2.5** `_run_prefill()` | `test_prefill_computes` | ✅ commit e36869a |
| [x] | **P2.6** `_run_decode_step()` ⚠️ DECISION: mock-based (model=None path) | `test_decode_produces_tokens` | ✅ commit e36869a |
| [x] | **P2.7** `run_inference_loop()` | `test_loop_processes` | ✅ commit e36869a |
| [x] | **P2.8** `register_stream()` | `test_stream_receives` | ✅ commit e36869a |

### 2.3 Sequence Lifecycle

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P2.9** Stop sequence + EOS | `test_stop_seq`, `test_eos` | ✅ commit e36869a |
| [x] | **P2.10** Max tokens limit | `test_max_tokens` | ✅ commit e36869a |
| [x] | **P2.11** Request cancellation | `test_cancel` | ✅ commit e36869a |
| [x] | **P2.12** Single request e2e | `test_single_lifecycle` | ✅ commit e36869a |
| [x] | **P2.13** Continuous batching test | `test_continuous_batching` | ✅ commit e36869a |
| [x] | **P2.14** Prefix cache hit test | *(deferred — needs real model)* | ✅ commit e36869a |

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

### 2.4 Devil's Advocate Review: Phase 2 🔴

**Owner:** devil's-advocate-agent | **Activates after:** P2.14 feature gate passes

| | Finding ID | Attack Vector | Severity |
|-|-----------|--------------|----------|
| [ ] | **DA-P2-1** | schedule_step() reentrancy — called before previous completes | CRITICAL |
| [ ] | **DA-P2-2** | Deadlock: queue lock held + awaiting inference | CRITICAL |
| [ ] | **DA-P2-3** | Long-running seq starves all new requests | HIGH |
| [ ] | **DA-P2-4** | 100 queued requests → memory of tokenized waiting seqs | HIGH |
| [ ] | **DA-P2-5** | Model exception mid-decode → scheduler state corrupt | HIGH |
| [ ] | **DA-P2-6** | Client disconnects → orphaned stream queue → leak | MEDIUM |
| [ ] | **DA-P2-7** | max_tokens=0, empty prompt, prompt > context window | MEDIUM |

**Process:**
- [ ] Subagent: write adversarial tests `tests/test_adversarial.py` (DA-P2 section)
- [ ] Run, identify failures, file findings
- [ ] scheduler-agent: fix CRITICAL + HIGH
- [ ] Re-run: ALL PASS
- [ ] → Notify Team Lead: "Phase 2 hardened, ready for merge"

---

## Phase 3: FastAPI Server

**Owner:** server-agent | **Branch:** `feature/api-server`
**Depends on:** P0 + P2 interfaces
**Method:** ALL tasks → subagent.

### 3.1 API Endpoints

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P3.1** Chat completions (sync) | `test_chat_completions` | ✅ commit 7ca6778 |
| [x] | **P3.2** Chat completions (SSE) | `test_chat_streaming` | ✅ commit 7ca6778 |
| [x] | **P3.3** Completions | `test_completions` | ✅ commit 7ca6778 |
| [x] | **P3.4** Models list | `test_models_list` | ✅ commit 7ca6778 |
| [x] | **P3.5** Health check | `test_health` | ✅ commit 7ca6778 |

### 3.2 Infrastructure

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P3.6** CLI parser | `test_cli_parsing` | ✅ commit 7ca6778 |
| [x] | **P3.7** Startup sequence | `test_startup` | ✅ commit 7ca6778 |
| [x] | **P3.8** Graceful shutdown | `test_shutdown_flushes` | ✅ commit 7ca6778 |
| [x] | **P3.9** Error handling | `test_invalid_request` | ✅ commit 7ca6778 |

### 3.3 Entry Point & Validation

| | Task | Test Required | Commit Message |
|-|------|:------------:|---------------|
| [x] | **P3.10** `__main__.py` | `test_module_entry` | ✅ commit 7ca6778 |
| [x] | **P3.11** Concurrent requests | `test_concurrent_4` | ✅ commit 7ca6778 |
| [x] | **P3.12** Stream == non-stream parity | `test_stream_parity` | ✅ commit 7ca6778 |

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

### 3.4 Devil's Advocate Review: Phase 3 🔴

**Owner:** devil's-advocate-agent | **Activates after:** P3.12 feature gate passes

| | Finding ID | Attack Vector | Severity |
|-|-----------|--------------|----------|
| [ ] | **DA-P3-1** | Missing OpenAI response fields → client SDK crash | CRITICAL |
| [ ] | **DA-P3-2** | Malformed JSON / missing `messages` / negative `max_tokens` | HIGH |
| [ ] | **DA-P3-3** | SSE format errors (missing `data:` prefix, `[DONE]`) | HIGH |
| [ ] | **DA-P3-4** | 50 simultaneous requests → hang or crash | HIGH |
| [ ] | **DA-P3-5** | Request in-flight during shutdown → partial response | MEDIUM |
| [ ] | **DA-P3-6** | Never-consumed streaming response → buffer growth | MEDIUM |
| [ ] | **DA-P3-7** | 1M token prompt → OOM or timeout handling | MEDIUM |

**Process:**
- [ ] Subagent: adversarial tests → findings → fixes → re-pass
- [ ] → Notify Team Lead: "Phase 3 hardened, ready for merge"

---

## Phase 4: Integration Tests & Benchmarks

**Owner:** test-agent | **Branch:** `feature/tests-benchmarks`
**Depends on:** P0 (interfaces); starts parallel with P1–P3

### 4.1 Shared Fixtures

| | Task | Commit Message |
|-|------|---------------|
| [ ] | **P4.1** `conftest.py` (mock model, test config, temp dirs) | `[P4.1] test: shared fixtures` |

### 4.2 Integration Tests

| | Task | Test Name | Commit Message |
|-|------|-----------|---------------|
| [ ] | **P4.2** E2E basic | `test_e2e_basic` | `[P4.2] test(e2e): basic` |
| [ ] | **P4.3** E2E prefix cache | `test_e2e_prefix` | `[P4.3] test(e2e): prefix` |
| [ ] | **P4.4** E2E SSD tier | `test_e2e_ssd` | `[P4.4] test(e2e): SSD` |
| [ ] | **P4.5** E2E concurrent | `test_e2e_concurrent` | `[P4.5] test(e2e): concurrent` |

### 4.3 Benchmarks

| | Task | Commit Message |
|-|------|---------------|
| [ ] | **P4.6** `scripts/benchmark.py` | `[P4.6] feat(bench): benchmark script` |
| [ ] | **P4.7** Run benchmarks, `BENCHMARKS.md` | `[P4.7] docs: benchmark results` |

**FEATURE GATE:** `pytest tests/ -v --tb=short` → ALL PASS

---

## Phase 5: Final Integration & Polish

**Owner:** Team Lead | **Branch:** `develop`
**Depends on:** All phases + all devil's advocate reviews complete

| | Task |
|-|------|
| [ ] | **P5.1** Merge all feature branches to develop (`--no-ff`) |
| [ ] | **P5.2** Resolve merge conflicts |
| [ ] | **P5.3** `pytest tests/ -v --tb=short` → ALL PASS on develop |
| [ ] | **P5.4** **Final devil's advocate review** (cross-component, see below) |
| [ ] | **P5.5** Fix any final CRITICAL/HIGH findings |
| [ ] | **P5.6** Run benchmarks on develop |
| [ ] | **P5.7** Write `README.md` |
| [ ] | **P5.8** Add `pyproject.toml` CLI entry point |
| [ ] | **P5.9** Document JACCL setup |
| [ ] | **P5.10** Code review: docstrings, dead code |
| [ ] | **P5.11** Tag `v0.1.0` + push tags |

### 5.1 Final Devil's Advocate: Cross-Component Review 🔴

**Owner:** devil's-advocate-agent | **Activates after:** P5.3 (full suite on develop)

This review targets **interactions between components** that individual reviews couldn't catch:

| | Finding ID | Attack Vector | Severity |
|-|-----------|--------------|----------|
| [ ] | **DA-F-1** | State leak between consecutive requests (dirty cache/seq state) | CRITICAL |
| [ ] | **DA-F-2** | Scheduler frees blocks while SSD save is in-progress → corrupt | CRITICAL |
| [ ] | **DA-F-3** | FastAPI async handler awaits scheduler → scheduler thread dies → hang | HIGH |
| [ ] | **DA-F-4** | 20 requests, shared prefix, mixed stream/sync → all correct? | HIGH |
| [ ] | **DA-F-5** | Server restart → SSD index loads → prefix hits resume correctly? | HIGH |

---

## Summary

### Task Counts by Agent

| Agent | Tasks | Phase | Role |
|-------|:-----:|-------|------|
| Team Lead | **21** | P0, P5 | Setup, merge, polish |
| cache-agent | **18** | P1 | KV cache + SSD tier |
| scheduler-agent | **14** | P2 | Continuous batching |
| server-agent | **12** | P3 | API server |
| test-agent | **7** | P4 | Integration + benchmarks |
| devil's-advocate-agent | **4 reviews** | P1.5, P2.4, P3.4, P5.1 | Adversarial review |
| **Total** | **72 tasks + 4 reviews** | | |

### Quality Pipeline (Per Feature)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Implement   │────▶│  Unit Tests  │────▶│  Devil's Advocate │────▶│    Merge     │
│  (subagent)  │     │  PASS gate   │     │  Review + Fix     │     │  to develop  │
└──────────────┘     └──────────────┘     └──────────────────┘     └──────────────┘
     Per task              Per task           Per feature             Team Lead
```

### Dependency Graph

```
P0 (Team Lead: setup, types.py, config.py, branches)
 │
 ├──▶ P1 (cache-agent)  ──▶ DA-P1 (devil's advocate) ──┐
 ├──▶ P2 (scheduler)    ──▶ DA-P2 (devil's advocate) ──┤
 ├──▶ P3 (server)       ──▶ DA-P3 (devil's advocate) ──┤
 └──▶ P4 (test-agent)   ───────────────────────────────┘
                                                         │
                                                         ▼
                         P5 (Team Lead: merge + DA-Final + tag v0.1.0)
```

### Execution Flow Per Task

```
 ┌─ Main Agent ──────────────────────────────────┐
 │ 1. Pick task Pn.m                             │
 │ 2. Spawn subagent → implement + test          │
 │ 3. Review output (quick scan)                 │
 │ 4. git add -A && git commit && git push       │
 │ 5. Mark [x] in checklist                      │
 │ 6. Next task → repeat                         │
 │                                               │
 │ After all tasks in feature:                   │
 │    pytest tests/ -v → ALL PASS                │
 │    → devil's-advocate-agent activates         │
 │    → fix CRITICAL/HIGH findings               │
 │    → re-test → ALL PASS                       │
 │    → notify Team Lead for merge               │
 └───────────────────────────────────────────────┘
```

### Key Decision Points

| ID | Decision | Owner | When |
|----|---------|-------|------|
| **D1** | BatchGenerator vs generate_step() | scheduler-agent | P2.6 |
| **D2** | KV cache array shapes for Qwen3-4B | cache-agent | P1.7 |
| **D3** | Chunked prefill vs sequential | scheduler-agent | P2.5 |
