# 16. Kimi K2.5 Server Launch Troubleshooting Log

> Detailed troubleshooting log from the 2026-02-14 session attempting to launch
> Kimi-K2.5 (612GB, 4-bit, 182 safetensors) on a 2-node JACCL Tensor Parallel
> setup. Documents all issues encountered, root causes, and fixes applied.

---

## Environment

| Item | hwStudio1 | hwStudio2 |
|------|-----------|-----------|
| Hardware | Mac Studio M3 Ultra | Mac Studio M3 Ultra |
| Memory | 512GB | 512GB |
| OS | macOS Tahoe 26.3 (Darwin 25.2.0) | macOS Tahoe 26.3 (Darwin 25.2.0) |
| Python | 3.14 | 3.14 |
| Connection | Thunderbolt 5 RDMA (80 Gb/s x 2 channels) | Thunderbolt 5 RDMA (80 Gb/s x 2 channels) |

**Model**: Kimi-K2.5 -- 612GB total, 4-bit quantized, 182 safetensors, 61 layers, 384 experts (~306GB per node with TP-2)

---

## Issues Encountered

### 1. Orphaned Wired Metal Memory (~315GB per node)

**Initial state**: Both nodes had ~315GB wired Metal memory from a previous session before any launch attempt.

**Root cause**: The previous server was not cleanly shut down. `mlx.launch` has no SIGTERM handler -- it dies instantly without propagating SIGTERM to child processes. The actual server processes (which hold the Metal ResidencySet) were never signaled and were eventually killed by SIGKILL, bypassing all cleanup.

**Fix applied**:
- Had to manually find and SIGTERM the actual server process (PID 14489 on hwstudio1), not the `mlx.launch` parent.
- hwstudio2 recovered immediately (~5.6GB wired).
- hwstudio1 recovered to ~5.6GB wired after SIGTERM to the actual server, but the process hung in cleanup state (`Rs+`) for 30+ seconds with port already freed. Had to SIGKILL to terminate it.

**Takeaway**: Always kill the actual server process, not `mlx.launch`. See also [instruction 15](./15-rank-death-prevention.md) for the full death path analysis.

---

### 2. OOM Kernel Panic on hwstudio1 (First Launch Attempt)

**Symptom**: Server log showed progressively worsening free memory warnings:
```
Low free memory: 0.6 GB
Low free memory: 0.1 GB
```
Then hwstudio1 became unreachable via SSH. After reconnection, `uptime` showed 1 minute -- confirming a reboot from OOM kernel panic.

**Root cause**: During model loading, macOS ran out of physical memory and triggered a watchdog-timeout kernel panic.

**Contributing factors**:
- Both nodes had ~493GB free before launch. Model needs ~306GB per node, which should fit.
- The "Low free memory" check uses `vm_stat` "Pages free" count, which can report very low values even when the system has hundreds of GB of "inactive" (reclaimable) memory. This metric is misleading.

**Lost diagnostics**: The first launch had `mlx.launch` capturing logs with Python's default stdout buffering + `nohup`, so output was mostly lost before the crash.

---

### 3. JACCL RDMA Init Failure (errno 22 -- After Reboot)

**Error**:
```
ValueError: [jaccl] Changing queue pair to RTR failed with errno 22
```

**Root cause**: After hwstudio1 rebooted, macOS automatically added the Thunderbolt interfaces (`en3`, `en5`) to `bridge0`. RDMA cannot access interfaces that are bridge members.

**Fix applied** (must run after every reboot on BOTH nodes):

```bash
# hwStudio1:
sudo ifconfig bridge0 down
sudo ifconfig bridge0 deletem en3
sudo ifconfig bridge0 deletem en5
sudo ifconfig en3 10.10.0.1/30 up
sudo ifconfig en5 10.10.1.1/30 up

# hwStudio2:
sudo ifconfig bridge0 down
sudo ifconfig bridge0 deletem en3
sudo ifconfig bridge0 deletem en5
sudo ifconfig en3 10.10.0.2/30 up
sudo ifconfig en5 10.10.1.2/30 up
```

**Verification steps**:
1. Thunderbolt ping works: `10.10.0.1 <-> 10.10.0.2`, ~1.7ms RTT
2. `rdma_ctl status` shows "enabled" on both nodes
3. RDMA kernel extensions loaded: `AppleThunderboltRDMA`, `IORDMAFamily`

**Note**: `rdma_en*` interfaces do not appear in `ifconfig` output after bridge reconfiguration, but JACCL works anyway (confirmed with distributed connectivity test).

**TODO**: Configure a LaunchDaemon for persistence so this survives reboots automatically.

---

### 4. JACCL Connectivity Test Failure (Script Path)

**Error**:
```
can't open file '/private/tmp/test_distributed.py': [Errno 2] No such file or directory
```
(on rank 1 / hwstudio2)

**Root cause**: The test script was written to `/tmp/` on hwstudio1 only. `mlx.launch` SSHes to hwstudio2 for rank 1, where the script does not exist. macOS resolves `/tmp` to `/private/tmp`.

**Fix**: Place test scripts in the shared project directory (`/Users/hw/mlx-lm-server/`) which exists on both nodes at the same path.

---

### 5. Rank 1 Crash During Model Loading (Second Launch Attempt)

**Symptom**: During model loading, hwstudio2 (rank 1) memory grew to ~250GB wired then crashed. Wired memory dropped to ~5.5GB (cleanup ran successfully on this occasion). hwstudio1 (rank 0) continued loading and started Uvicorn as if nothing happened.

**Memory timeline on hwstudio2**:

| Time | Wired Memory | Notes |
|------|-------------|-------|
| 0s | ~5GB | Start |
| 10s | ~107GB | Loading |
| 30s | ~250GB | 256GB inactive, 58MB free |
| 60s | ~5.5GB | Crash + successful cleanup |

**Log output was empty** due to Python stdout buffering in `nohup` mode. Adding `PYTHONUNBUFFERED=1` to the launch command did not help because `mlx.launch` spawns child processes via SSH, so the environment variable is not propagated to rank 1.

**Post-crash complication**: After killing hwstudio1 (rank 0), hwstudio2 showed 316GB wired reappearing -- orphaned Metal memory from the rank 0 process's remote allocations (see issue 6).

---

### 6. Orphaned Metal Memory After Rank 1 Crash (Unrecoverable)

**Symptom**: hwstudio2 showed 316GB wired with no server processes running.

**Attempted fix**: Ran a Python script calling `mx.set_wired_limit(0)` -- did not work because a new process creates its own `MetalAllocator` singleton and cannot affect the dead process's `ResidencySet`.

**Actual fix**: Rebooted hwstudio2.

**Key insight**: Orphaned Metal `ResidencySet` wired pages are unrecoverable without reboot. No userspace API exists to release another process's wired pages. This matches the analysis in [MEMORY.md](../.claude/projects/-Users-hw-mlx-lm-server/memory/MEMORY.md): the `MetalAllocator` is a heap-allocated singleton that is intentionally never deleted.

---

### 7. Asymmetric Model Loading (Third Launch Attempt)

**Observation**: After rebooting hwstudio2 and reconfiguring RDMA, launched again. Loading showed extreme asymmetry:

| Node | Wired Memory | Timeline |
|------|-------------|----------|
| hwstudio2 (rank 1) | ~311GB | Loaded within ~50s |
| hwstudio1 (rank 0) | ~122GB -> ~5.6GB | Loaded partially, then dropped back |

**Resolution**: Server IS functional. Model loaded on both ranks, inference confirmed working. The asymmetry is wired vs non-wired only -- hwstudio1 (rank 0) holds ~310GB in active (non-wired) Metal pages while hwstudio2 (rank 1) holds ~316GB in wired Metal pages. Both ranks have the full sharded model loaded (~306GB each). See "Resolution and Final State" section below for details.

---

## Resolution and Final State

### Server Successfully Started (Third Attempt)
- Server health check passes: `GET /health` returns `{"status":"ok", "distributed": {"enabled": true, "rank": 0, "world_size": 2, "fatal": false}}`
- Inference working correctly: "What is 2+2?" -> "2 + 2 = **4**"
- Kimi K2.5 uses a `<think>...</think>` reasoning feature -- initial "garbled" output was actually visible chain-of-thought tokens being cut off by low `max_tokens`

### Memory Asymmetry Explained (Not a Defect)
- hwStudio1 (rank 0): **310GB in active (non-wired) Metal pages**, only 5.6GB wired
- hwStudio2 (rank 1): **316GB wired Metal pages**
- vm_stat breakdown for hwStudio1: Free=1.6GB, Active=310GB, Inactive=183GB, Wired=5.6GB
- Both ranks have the full sharded model loaded (~306GB each)
- The asymmetry is **wired vs non-wired**, not missing weights
- `set_wired_limit_for_model()` is called on both ranks, but rank 0's buffers are in active (non-wired) Metal memory
- Inference works correctly but rank 0 may be slightly slower due to non-wired memory

### Rank Death Root Cause Analysis
Previous rank deaths during model loading were caused by:
1. **`_eval_shard_eval` peak memory**: Each layer is materialized at full size BEFORE sharding, then the sharded version is created and the original freed. Peak per-layer = ~1 full layer + 1 sharded layer.
2. **OS memory reclamation lag**: At the 30s mark, hwstudio2 had 250GB wired + 256GB inactive + only 58MB free. If the next layer eval needed memory faster than the OS could reclaim inactive pages, it triggers OOM.
3. **Lost diagnostics**: Python stdout buffering in SSH-spawned child processes means crash output was never captured. `PYTHONUNBUFFERED=1` on the launcher doesn't propagate to rank 1 (spawned via `mlx.launch` SSH).

---

## Key Patterns and Lessons Learned

### 1. `mlx.launch` Does Not Propagate SIGTERM

`mlx.launch` is a thin process launcher with no signal handling. Sending SIGTERM to `mlx.launch` kills it immediately without forwarding the signal to child processes. The actual server processes continue running (or are orphaned).

**Workaround**: Always find and kill the actual server processes directly. Use `stop_server.sh` which handles this correctly.

### 2. Thunderbolt RDMA Requires Bridge Reconfiguration After Every Reboot

macOS automatically adds Thunderbolt interfaces to `bridge0` on boot. RDMA fails with `errno 22` when interfaces are bridge members.

**Workaround**: Run the `ifconfig` bridge removal commands after every reboot.
**Permanent fix needed**: Create a LaunchDaemon that runs the bridge reconfiguration at boot.

### 3. Python Output Buffering Hides Critical Errors

When using `nohup` with `mlx.launch`, Python's default stdout buffering means crash messages are lost. `PYTHONUNBUFFERED=1` does not propagate to SSH-spawned child processes.

**Workaround**: Need to set `PYTHONUNBUFFERED=1` in the remote node's shell profile or pass it through the SSH command that `mlx.launch` constructs.

### 4. Orphaned Metal ResidencySet Is Unrecoverable Without Reboot

No userspace API exists to release another process's wired Metal pages. Once a process dies without calling `mx.set_wired_limit(0)`, those pages are wired until reboot.

**Prevention**: Ensure clean shutdown always runs `_cleanup_metal()`. See [instruction 15](./15-rank-death-prevention.md).

### 5. `vm_stat` "Pages Free" Is Misleading

macOS can have hundreds of GB of reclaimable "inactive" pages that are not counted as "free" by `vm_stat`. The "Low free memory" warning in the server can trigger even when the system has plenty of reclaimable memory.

**Improvement needed**: Memory guard should consider inactive pages as available, or use `os_proc_available_memory()` / `sysctl hw.memsize` minus wired for a more accurate picture.

---

## Chronological Timeline

```
2026-02-14 Session

[Pre-launch]
  Both nodes: ~315GB wired (orphaned from previous session)
  Fix: SIGTERM actual server processes
  Result: Both nodes recovered to ~5.6GB wired

[First Launch Attempt]
  Both nodes: ~493GB free
  hwstudio1: "Low free memory: 0.6 GB" -> "0.1 GB" -> OOM kernel panic -> reboot
  hwstudio2: OK but orphaned after hwstudio1 died
  Lost: Log output (Python buffering + nohup)

[Post-reboot Recovery]
  hwstudio1: JACCL errno 22 (Thunderbolt in bridge0)
  Fix: Remove interfaces from bridge0, assign static IPs
  Verify: TB ping OK, rdma_ctl OK, JACCL connectivity test OK

[Second Launch Attempt]
  hwstudio2 (rank 1): Loaded to ~250GB -> crashed at ~60s
  hwstudio1 (rank 0): Continued, started Uvicorn
  hwstudio2: 316GB wired reappeared after hwstudio1 killed
  Fix: Rebooted hwstudio2

[Third Launch Attempt]
  hwstudio2 (rank 1): Loaded to ~311GB within ~50s (OK)
  hwstudio1 (rank 0): Loaded to ~122GB -> dropped to ~5.6GB wired (310GB in active non-wired)
  Uvicorn started, health check OK, inference confirmed working
  Memory asymmetry: wired vs non-wired only, not missing weights
  Status: RESOLVED -- server functional
```
