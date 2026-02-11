# PLAN.md — mlx-lm-server Project Plan

## Workflow Audit Fix Session (2026-02-08)

### Action Items

- [x] **F-08** Fix deadlock: cache_block() → evict_to_ssd() lock re-entry (two-phase eviction)
- [x] **F-07** Fix collision block leak: block_hash=None blocks never freed (return immediately)
- [x] **F-09** Fix stuck sequences: move active registration after UID assignment
- [x] **F-11** Fix silent token drop: signal error + cancel on backpressure overflow
- [x] **F-18** Fix silent exception swallowing: add logger.exception()
- [x] **F-06** Fix stop text in non-streaming response: use seq.output_text
- [x] **F-03/F-13/F-16** Extract _validate_and_prepare_request() + shutdown/model/empty guards
- [x] **New** Create test_server_app.py: 26 endpoint tests with strong assertions
- [x] **New** Add 24 regression tests for all fixed bugs
- [x] **F-15** Add skipif guard to test_datsets.py
- [x] **SSD** Document single-thread assumption in ssd_cache.py

### Deferred (not in this round)

| Issue | Decision | Reason |
|-------|----------|--------|
| F-10: SSD read path not connected | Defer | Feature addition, not bug fix. Separate PR. |
| Perf: O(seq_len^2) hash recomputation | Defer | 5-10ms negligible vs inference time. Perf PR later. |
| Perf: O(N log N) eviction fallback | Defer | Rare, bounded at 10-50ms. Optimize when benchmarked. |
| Perf: Per-event lock acquisitions | Do nothing | 64us noise vs 10-100ms inference. |
| F-04: Config defaults not wired | Low priority | Pydantic defaults match config defaults in practice. |
| F-17: ServerConfig validation | Low priority | Config set by admins, not users. |

### Decisions Log

| Issue | Choice | Rationale |
|-------|--------|-----------|
| F-08 Deadlock | 1A: Two-phase eviction | Explicit, clean lock scopes. RLock hides design smell. |
| F-07 Block leak | 2A: Return immediately on free | Collision blocks have no cache value. |
| F-10 SSD read | 3C: Defer | Feature, not bug. SSD works as write-through for crash recovery. |
| F-09 Stuck seqs | 4B: Move registration after UID | Fix root cause (ordering), not symptom (cleanup). |
| F-18 Exception log | 5A: logger.exception() | 2 lines, zero risk, massive debuggability gain. |
| F-11 Token drop | 6A: Signal error + cancel | Explicit error > silent data corruption. |
| F-06 Stop text | 7C+B: Fix non-streaming only | Streaming buffer adds latency for rare edge case. |
| DRY + guards | 8A: Extract helper | Eliminates 5 DRY violations across 2 endpoints. |
| Endpoint tests | 9A: New test_server_app.py | Follows 1-file-per-module pattern. |
| Regression tests | 10B: Tests for fixed bugs | TDD: write test, verify bug, fix, verify fix. |
| Assertion quality | 11A: Strong in new tests only | Set pattern forward, don't retroactively audit 500 tests. |
| F-15 skipif | 12B: Only skipif guard | Don't modify upstream setup.py. |
| Hash perf | 13C: Defer | Backward-compat risk warrants focused perf PR. |
| Eviction perf | 14C: Defer | Fallback is safety net, rarely triggers. |
| SSD threads | 16B: Document assumption | Phase 8 threading model may differ. |

### Test Results (post-fix)

```
363 passed, 4 skipped, 1 xfailed, 0 failures (21.13s)
```

| File | Tests | What |
|------|:-----:|------|
| test_adversarial.py | 105+1xfail | DA reviews P1-P7 |
| test_kv_cache_manager.py | 68 | Block pool, hash, prefix, alloc, free, evict |
| test_scheduler.py | 37 | Queue, scheduling, lifecycle, streaming |
| test_server_app.py | 26 | **NEW** FastAPI endpoint tests |
| test_regression.py | 40 | **24 NEW** regression tests for audit fixes |
| test_stream_verification.py | 20 | Stream refactoring verification |
| test_ssd_cache.py | 18 | SSD init, save, load, prune |
| test_integration.py | 20 | E2E, SSD round-trip, batch |
| test_block_bridge.py | 15 | decompose/reconstruct |
| test_sequence_cache.py | 12 | Trie-based prefix search |
| test_batch_integration.py | 7 | MockBatchGenerator |
| test_datsets.py | 4 skipped | **FIXED** skipif for optional deps |

---

## Completed Phases (P0-P7)

All phases complete. See git history for details:
- **P0:** Project setup, types.py, config.py
- **P1:** KV Cache Manager + SSD Tier (53 tests)
- **P2:** Continuous Batching Scheduler (32 tests)
- **P3:** FastAPI Server, OpenAI API (13 tests)
- **P4:** Integration Tests + Benchmarks (12 tests)
- **P5:** Final Integration, v0.1.0 tag (43 tests)
- **P6:** BatchGenerator Integration (35 tests)
- **P7:** Block-Level KV Cache Bridge + SSD (40 tests)

Streams A-F (refactoring, correctness, performance, tests, observability) also complete.
Production review (d92f272): 11 issues fixed.

---

## Phase 8: Tensor-Parallel Serving (JACCL RDMA)

See `phase8_plan.md` for the full plan.

**Summary:** Distribute inference across 2x Mac Studio M3 Ultra via Thunderbolt 5 RDMA using JACCL. Only rank 0 runs FastAPI + scheduler. Model weights split across nodes.

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `mlx_lm_server/kv_cache_manager.py` | Two-phase eviction, collision block return |
| `mlx_lm_server/scheduler.py` | Move active registration, backpressure error signal |
| `mlx_lm_server/server.py` | logger.exception, stop text fix, _validate_and_prepare_request() |
| `mlx_lm_server/ssd_cache.py` | Thread-safety docstring |
| `tests/test_server_app.py` | **NEW** 26 endpoint tests |
| `tests/test_regression.py` | **+24** regression tests (7 new test classes) |
| `tests/test_adversarial.py` | Fixed 7 tests (model name -> "test-model") |
| `tests/test_datsets.py` | Added skipif for datasets package |
