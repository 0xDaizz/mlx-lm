# 17. Distributed RDMA Deadlock Investigation Log

> Comprehensive error log from 2026-02-14 Kimi-K2.5 benchmark sessions.
> 4 deadlock events observed across 4 server sessions while running Phase 1 Baseline benchmarks.
> Documents all errors, root cause analysis, fix attempts, and remaining open issues.

## Environment

- **Cluster**: 2-node JACCL Tensor Parallel (TP-2)
  - **hwstudio1**: Mac Studio M3 Ultra, 512GB unified memory (rank 0)
  - **hwstudio2**: Mac Studio M3 Ultra, 512GB unified memory (rank 1)
- **Interconnect**: Thunderbolt 5 RDMA (80 Gb/s x 2 channels)
- **Model**: Kimi-K2.5 (1T MoE, 612GB on disk, ~306GB per node with TP-2)
- **Branch**: develop
- **OS**: macOS (Darwin)

## Deadlock Event Timeline

### Session 1 (v1) -- ~15:26-16:42

- Server launched, model loaded (~160s)
- Discovery: Metal lazy wiring (rank 0 wired 5GB->317GB on first inference)
- Discovery: `footprint -a` at 3113% CPU crushed hwstudio1 (load 32.6) -- NEVER use on production
- Bug: httpx ReadTimeout (default 5s, model needs >>5s for prefill) -- Fixed with 600s timeout
- **12 requests completed, ~1771 tokens generated**
- **DEADLOCK**: Both ranks stuck in `eval_impl()` -> `condition_variable::wait()`, `dist_outbox_size=4`
- Both ranks 0% CPU (blocked on condition variable, not spinning)
- Server killed with SIGKILL -> 636GB wired memory orphaned (318GB per node)
- Required full reboot of both nodes

### Session 2 (v2) -- ~16:35-16:55

- Server restarted after node reboots + RDMA reconfiguration
- Applied fixes: inference loop watchdog, health 503 on stall, incremental bench results
- **3 requests completed, 733 tokens generated**
- Discovery: Rank 0 wired oscillation (5GB->316GB during inference, back to 5GB idle -- `BatchGenerator.close()` restoring wired limit)
- **Bug: Watchdog false positive on idle** -- Reports `inference_stalled` after 5min idle even with 0 active/queued requests
- Health started returning 503 Service Unavailable on idle server
- **DEADLOCK/CRASH**: HTTP 000, server unresponsive
- Rank 1 died first (wired 317GB->5.4GB = cleanup succeeded), then rank 0 died
- Metal cleanup SUCCEEDED this time (both nodes 5.3-5.4GB wired after death)

### Session 3 (v3) -- ~17:08 (relaunch into v4)

- Applied fix: async_eval drain (`mx.eval(bg.active_batch.y, bg.active_batch.logprobs)` before bus sync)
- Applied fix: watchdog false positive guard (only flag stale when `active_sequences > 0` or `queued_requests > 0`)
- This session was relaunched as v4 without completing benchmarks

### Session 4 (v4) -- 17:08-17:58

- Server launched with ALL fixes applied (async_eval drain + watchdog guard)
- **2 requests completed, 306 tokens generated**
- 17:17:47: First inference request (`active=1`)
- 17:18:21: 2 requests completed, 306 tokens, `prefill=56`
- **CRITICAL: hwstudio2 wired dropped 315.95GB -> 5.44GB in 9 seconds** (17:17:50->17:17:59) while rank 1 process still alive at 100% CPU
- Outbox growing: 0 -> 2 -> 3 -> 5
- Stale time growing: 0s -> 83s -> 146s -> 196s -> 2304s
- 17:21-17:55: SSH connection errors (Tailscale disrupted by deadlocked RDMA?)
- 17:56: Watchdog correctly detected (`inference_loop_alive=false`, health returning 503)
- Both ranks at **100% CPU** (busy spinning, different from session 1 at 0% CPU)
- Required SIGKILL to terminate (SIGTERM ineffective -- stuck in C++ code)
- hwstudio1: 316GB wired orphaned, hwstudio2: 5.4GB (already unwired during deadlock)

## Error Catalog

### ERR-01: httpx ReadTimeout on Long Prompts

- **Symptom**: Bench scripts crash with `ReadTimeout` during stream reads
- **Root cause**: All 6 bench scripts used bare `httpx.Client()` with default 5s timeout. 1T MoE prefill >> 5s
- **Fix**: `httpx.Client(timeout=httpx.Timeout(600.0))` in all 6 scripts (8 instances)
- **Status**: FIXED
- **Files**: `bench_single.py`, `bench_batch.py`, `bench_concurrent.py`, `bench_cache.py`, `bench_memory.py`, `bench_ngram.py`

### ERR-02: footprint -a CPU Spike (3113%)

- **Symptom**: `footprint -a` command caused 3113% CPU, `load=32.6` on hwstudio1
- **Root cause**: macOS `footprint` does deep memory introspection across all processes
- **Fix**: Never use `footprint` on production nodes. Use `vm_stat` only
- **Status**: RESOLVED (procedural)

### ERR-03: Watchdog False Positive on Idle Server

- **Symptom**: Health returns 503 (`inference_stalled`) after 5min idle with 0 pending requests
- **Root cause**: Watchdog tracked `_last_inference_step_ts` but didn't check if there was actual work to do
- **Fix**: Guard with `has_work = active_sequences > 0 or queued_requests > 0`; only flag stale when `has_work`
- **Status**: FIXED
- **File**: `mlx_lm_server/scheduler.py` (`get_cache_stats`)

### ERR-04: bench_single Loses All Results on Crash

- **Symptom**: `bench_single` completed 12 requests but wrote results only at end; crash = total data loss
- **Root cause**: Results written in single JSON dump after all tests complete
- **Fix**: Added `_save_partial()` helper that writes `*_partial.json` after each test case + try/except for crash safety
- **Status**: FIXED
- **Files**: All 6 bench scripts

### ERR-05: stop_server.sh Can't Resolve hwStudio2.local

- **Symptom**: `stop_server.sh` fails to SSH to `hwStudio2.local` from Tailscale
- **Root cause**: Bonjour/mDNS doesn't resolve across Tailscale network
- **Fix**: Added `--remote-host` and `--hostfile` args with fallback hostname probing (`hwstudio2`, `hwStudio2.local`, `192.168.0.107`)
- **Status**: FIXED
- **File**: `tests/e2e/benchmark/configs/stop_server.sh`

### ERR-06: Distributed RDMA Collective Deadlock (PRIMARY ISSUE)

- **Symptom**: Server deadlocks after 2-12 requests. Both ranks stuck, outbox grows, no more tokens generated
- **Observed 4 times** across 4 sessions
- **Variants**:
  - Sessions 1, 2: Both ranks at 0% CPU (blocked on `condition_variable::wait`)
  - Session 4: Both ranks at 100% CPU (busy spinning -- different mechanism?)
- **Root cause hypothesis 1 (FAILED)**: async_eval pipeline race between model `all_sum` and bus `all_sum` on separate MLX streams. Fix: drain async_eval before bus sync. RESULT: Deadlock still occurs
- **Root cause hypothesis 2 (UNTESTED)**: Something triggers `set_wired_limit(0)` on rank 1 during inference, unwiring the model. Rank 1 then can't do collective operations properly
- **Evidence**: hwstudio2 wired dropped 316GB->5.4GB while rank 1 alive at 100% CPU (session 4)
- **Status**: UNRESOLVED -- needs deeper investigation

### ERR-07: Mysterious Rank 1 Wired Memory Drop

- **Symptom**: hwstudio2 wired memory drops from 316GB to 5.4GB in 9 seconds while rank 1 process is still running
- **Timing**: Occurs right when first inference starts (lazy wiring on rank 0)
- **Possible causes**:
  1. `BatchGenerator.close()` restoring old wired limit (0) on rank 1
  2. `_cleanup_metal()` triggered by unexpected signal
  3. Some code path calling `mx.set_wired_limit(0)` on the worker
- **Status**: UNRESOLVED

### ERR-08: SSH Connection Failures During Deadlock

- **Symptom**: SSH to hwstudio1 fails with "Undefined error: 0" for 30+ minutes during deadlock
- **Root cause**: Likely the deadlocked RDMA operations saturating Thunderbolt bandwidth or disrupting the network stack
- **Impact**: Monitoring agents can't check health, heartbeat shows ERRGB
- **Status**: UNRESOLVED (consequence of deadlock)

### ERR-09: SIGTERM Ineffective on Deadlocked Server

- **Symptom**: `kill PID` (SIGTERM) doesn't stop the server when stuck in RDMA collective
- **Root cause**: Python signal handlers can't interrupt C++ blocking calls (`condition_variable::wait` or RDMA busy-spin)
- **Fix**: Must use SIGKILL (`-9`), which orphans wired memory
- **Status**: KNOWN LIMITATION -- subprocess isolation needed for proper fix

## Fixes Applied (All in develop branch)

### scheduler.py Changes

1. Inference loop watchdog (`_last_inference_step_ts`, `_inference_watchdog_s`)
2. Health reports `inference_loop_stale_s` and `inference_loop_alive`
3. Watchdog false positive guard (`has_work` check)
4. async_eval drain before bus sync (`mx.eval(bg.active_batch.y, bg.active_batch.logprobs)`) -- DID NOT FIX DEADLOCK

### server.py Changes

1. Health endpoint returns 503 when `inference_stalled`
2. `/readyz` includes `inference_stalled` in reasons

### Bench Script Changes (all 6 scripts)

1. httpx timeout: 600s
2. Incremental result saving (`_save_partial`)
3. try/except for crash safety

### stop_server.sh Changes

1. `--remote-host` and `--hostfile` arguments
2. Fallback hostname probing

## Key Findings

### 1. Deadlock Is Reproducible and Consistent

- Always occurs after 2-12 successful requests
- Always involves stuck RDMA collective (`all_sum`)
- Outbox size grows (stranded messages)
- Not a transient/rare issue -- happens 100% of the time

### 2. async_eval Drain Fix Is Insufficient

- Hypothesis: model `all_sum` races with bus `all_sum` on different streams
- Reality: Deadlock persists even with explicit `mx.eval()` before bus sync
- The race condition is not between model and bus `all_sum`, or the drain doesn't fully resolve it

### 3. Rank 1 Wired Memory Behavior Is Anomalous

- During session 4, rank 1's wired memory dropped 316GB->5.4GB while the process was still alive
- This suggests something is calling `set_wired_limit(0)` on rank 1 during normal operation
- Possible: `BatchGenerator.close()` on rank 1's worker path
- This may be related to the deadlock -- if rank 1's model is unwired, collective operations may malfunction

### 4. Two Different Deadlock Modes

- **Mode A** (sessions 1, 2): Both ranks 0% CPU, blocked on `condition_variable`. Classic RDMA completion wait.
- **Mode B** (session 4): Both ranks 100% CPU, busy spinning. Suggests RDMA retry loop or different failure mode.

### 5. Other Frameworks Avoid This

- **exo**: Uses TCP for coordination, not `all_sum`. No collective conflicts possible.
- **ollama**: C++ runner in subprocess. Go supervisor manages lifecycle.
- Our use of `all_sum` for BOTH model weights AND control bus synchronization is unique and potentially problematic.

## Remaining Investigation Needed

1. **Trace `set_wired_limit` calls on rank 1**: Add logging to find what's unwiring the model
2. **Test with bus disabled**: Run server without `DistributedControlBus` to isolate whether bus sync causes the deadlock
3. **Test with TCP bus**: Replace `all_sum` bus sync with TCP socket coordination (like exo)
4. **Profile RDMA collectives**: Count `all_sum` calls per inference step, check for mismatch between ranks
5. **Test with smaller model**: Try a smaller distributed model to see if the issue is model-size dependent

## References

- [15-rank-death-prevention.md](./15-rank-death-prevention.md) -- Metal cleanup analysis
- [16-kimi-k25-server-launch-troubleshooting.md](./16-kimi-k25-server-launch-troubleshooting.md) -- Initial launch issues
- [MEMORY.md](../.claude/projects/-Users-hw-mlx-lm-server/memory/MEMORY.md) -- Metal wired memory research
