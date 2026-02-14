# 15. Rank Death Prevention & Metal Cleanup

> Comprehensive reference for the rank death prevention system in distributed
> tensor-parallel mode. Covers all death paths, cleanup mechanisms, diagnostics,
> and configuration knobs.

---

## Context

During Phase 1 benchmark of Kimi-K2-Instruct (1T MoE, ~320GB/node TP-2), rank 1
died after a 300s eval timeout. Both nodes leaked ~318 GiB wired Metal memory,
requiring reboots. Root cause: one rank's `mx.eval` hung because its peer died
during `model.shard()`, causing `all_sum` to block indefinitely until the 300s
`eval_with_timeout` watchdog fired.

---

## All Rank Death Paths

### A. Model Load Phase

| # | Trigger | Location | Mechanism | Metal Cleanup? |
|---|---------|----------|-----------|----------------|
| A1 | `mx.eval` hangs >timeout | `utils.py:eval_with_timeout` | Watchdog -> `os._exit(1)` | Best-effort (not thread-safe) |
| A2 | Memory below threshold | `utils.py:_check_memory_guard` | `MemoryGuardError` -> exception handler | YES (via finally block) |
| A3 | Load/init error | `__main__.py` except block | `Exception` -> `sys.exit(1)` | YES (via finally block) |

### B. Inference Phase

| # | Trigger | Location | Mechanism | Metal Cleanup? |
|---|---------|----------|-----------|----------------|
| B1 | 10 consecutive bus errors | `scheduler.py` | `_dist_fatal=True` | YES (loop exit -> finally) |
| B2 | Bus deserialization failure | `scheduler.py` | Immediate `_dist_fatal` | YES (loop exit -> finally) |
| B3 | JACCL connection loss | `distributed_bus.py` | `all_sum` throws/hangs | Throws: YES. Hangs: NO |

### C. External

| # | Trigger | Location | Mechanism | Metal Cleanup? |
|---|---------|----------|-----------|----------------|
| C1 | SSH disconnect | `__main__.py` | SIGHUP -> **IGNORED** in distributed mode | N/A (process continues) |
| C2 | Operator stop | `__main__.py` | SIGTERM -> `sys.exit(0)` -> atexit | YES |
| C3 | OS OOM / kernel panic | kernel | SIGKILL | NO (impossible) |

---

## Configuration Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_EVAL_TIMEOUT` | `300` | Seconds before `eval_with_timeout` watchdog fires `os._exit(1)`. Set higher for very large models (e.g., 600 for 1T+ models). Set lower for debugging (e.g., 120). |
| `MLX_MEMORY_GUARD_GB` | `max(10%, 5GB)` | Memory guard threshold in GB. When remaining memory drops below this during model loading, `MemoryGuardError` is raised. Set to `0` to disable. |

---

## Diagnostic Logging

### Per-Layer Load Diagnostics

During `_EvalShardEvalIter.__iter__` (distributed model loading), each layer logs:

```
[rank 0] Layer 1/182: eval start (active=5.2 GB)
[rank 0] Layer 1/182: eval done in 0.8s (active=8.7 GB)
[rank 0] Memory guard: layer 1/182, remaining=503.3 GB, threshold=51.2 GB
```

This allows identifying the exact layer where a timeout or OOM occurred.

### eval_with_timeout Timeout

```
[rank 1] eval_with_timeout: TIMED OUT after 300s -- calling os._exit(1). This may indicate a blocked collective or tensor parallel issue.
```

### EXIT AUDIT

At process exit (in the `finally` block of `main()`):

```
EXIT AUDIT: rank=0, cause=uvicorn_shutdown, active_mem=310.2 GB, peak_mem=320.1 GB
```

Exit causes:
- `uvicorn_shutdown` -- Normal HTTP server shutdown (rank 0)
- `worker_loop_exit` -- Normal worker loop completion (rank >0)
- `worker_loop_timeout` -- Worker loop hit 1-hour safety timeout
- `SIGTERM` / `SIGINT` -- Signal received
- `SIGHUP` -- Only in non-distributed mode (ignored in distributed)
- `{ExceptionType}: {message}` -- Unhandled exception (e.g., `MemoryGuardError: ...`)
- `unknown` -- Process exited before cause was set

### Memory Guard

```
MEMORY GUARD: aborting model load -- remaining memory 3.2 GB is below safety threshold 51.2 GB (layer 45/182, active=508.8 GB, peak=510.1 GB, total_ram=512.0 GB). Raising MemoryGuardError to trigger cleanup.
```

---

## Honest Assessment: What This Actually Fixes

This PR **reduces the blast radius** and **makes death diagnosable**. It does not
fully prevent rank death — the fundamental blocking-collective problem remains.

### What's fixed

**Path A2 (memory guard)**: The dying rank now cleans up properly. Before,
`sys.exit(1)` ran `_cleanup_metal()` via atexit, but the exception never reached
`__main__.py`'s `finally` block for full cleanup. Now `MemoryGuardError` bubbles
up through the normal exception path, and both the `except` handler and `finally`
block execute. The peer is still stuck in `all_sum` — that's unsolved.

**Path C1 (SSH disconnect)**: Eliminated entirely. SIGHUP is now `SIG_IGN` in
distributed mode. SSH flakiness no longer kills ranks.

**Path B3-hang (peer dies during inference)**: `join_worker_loop` had
`timeout=None`, meaning it blocked forever with no cleanup. Now `timeout=3600`
ensures the process exits and runs Metal cleanup after 1 hour. Bad, but better
than "never."

### What's still broken

The **fundamental problem** is unchanged: `mx.distributed.all_sum()` is a
blocking C++ call that Python cannot interrupt. When a peer dies while you're
inside `all_sum`:

1. SIGTERM arrives but can't interrupt the C++ call
2. The operator escalates to SIGKILL
3. SIGKILL bypasses all cleanup (`atexit`, `finally`, signal handlers)
4. Wired memory leaks (~310 GB per node)

The `os._exit(1)` path (A1) attempts `mx.set_wired_limit(0)` from a background
thread, but this is **not thread-safe** (MLX #2133) — it might work, might
deadlock, might crash. It's a coin flip.

### The biggest practical win is diagnostics

Next time a rank dies, per-layer logging tells us **exactly which layer, at what
memory level, on which rank** the failure occurred:

```
[rank 0] Layer 45/182: eval done in 1.2s (active=318.7 GB)
[rank 1] Layer 46/182: eval start (active=315.1 GB)
[rank 1] eval_with_timeout: TIMED OUT after 300s — calling os._exit(1)
EXIT AUDIT: rank=0, cause=MemoryGuardError: ..., active_mem=318.7 GB, peak_mem=320.1 GB
```

Phase 1 had zero logs — we couldn't even confirm the root cause. Now we can.

### What would actually solve this (future work)

1. **Subprocess isolation** (exo pattern) — run MLX inference in a child process.
   Parent can always kill it and clean up Metal from the outside.
2. **MLX collective timeout** — upstream `all_sum(timeout=30)` so it throws
   instead of blocking forever.
3. **Heartbeat** — TCP ping between ranks to detect peer death without waiting
   for `all_sum` to discover it.

All three require either upstream MLX changes or significant architecture work.
This PR is the pragmatic middle ground: fix the easy death paths, add diagnostics
for the hard ones, and make `MLX_EVAL_TIMEOUT` configurable so operators can tune
the blast radius.

---

## Key Implementation Details

### MemoryGuardError (mlx_lm/utils.py)

Replaces `sys.exit(1)` in `_check_memory_guard`. Subclass of `RuntimeError` so existing
`except RuntimeError` handlers catch it. Bubbles up through:
```
_check_memory_guard -> _EvalShardEvalIter.__iter__ -> model.shard() -> sharded_load() -> __main__.py except block -> finally block (Metal cleanup)
```

### SIGHUP Immunity (mlx_lm_server/__main__.py)

In distributed mode (`dist_ctx.enabled`), SIGHUP is set to `signal.SIG_IGN`. This prevents
SSH disconnects from killing a rank and orphaning its peer. The SIGHUP handler is configured
AFTER `init_distributed()` returns, so it knows whether distributed mode is active.

### join_worker_loop Safety Timeout (mlx_lm_server/__main__.py)

Rank >0 workers call `scheduler.join_worker_loop(timeout=3600)` instead of `timeout=None`.
If the inference thread hangs (stuck in `all_sum` after peer death), the 1-hour timeout ensures
the process eventually exits and runs Metal cleanup. The `scheduler.worker_timed_out` flag
is checked to log the timeout.

### eval_with_timeout Thread-Safety Note (mlx_lm/utils.py)

The `mx.set_wired_limit(0)` call from the watchdog thread while `mx.eval` runs on the main
thread is **not thread-safe** (MLX issue #2133). However, since `os._exit(1)` follows
immediately, deadlock risk is bounded. The call is kept as a best-effort measure.

---

## Files Modified

| File | Changes |
|------|---------|
| `mlx_lm/utils.py` | `MemoryGuardError`, configurable eval timeout, per-layer diagnostics, thread-safety docs |
| `mlx_lm_server/__main__.py` | SIGHUP immunity, `except Exception`, join timeout, EXIT AUDIT, `_exit_cause` tracking |
| `tests/test_distributed.py` | 18 new tests for all above features |
| `tests/test_rank_death_prevention.py` | 46 dedicated tests for rank death prevention |

---

## Future Work

- **Heartbeat monitor**: TCP heartbeat to detect peer death when `all_sum` hangs silently (path B3-hang)
- **Subprocess isolation**: Run inference in child process (exo pattern) so parent can always clean up
- **MLX collective timeout**: Request upstream MLX add `timeout` parameter to `all_sum`
- **Interruptible collectives**: Wrap `all_sum` with timeout or run in thread with cancellation

---

## Related Commits

| Commit | Description |
|--------|-------------|
| `50ab48d` | fix: prevent rank death cascade + reliable Metal cleanup in distributed mode |
| `410e8c0` | fix: remove 300s timeout on non-rank0 worker loop + add bench tooling |
| `e25fa72` | fix: best-effort Metal unwire in cleanup_timer emergency exit |
| `e4cc2d5` | fix: stop_server.sh kills actual server processes, not just mlx.launch |
