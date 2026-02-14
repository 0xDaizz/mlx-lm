# Distributed Bus: TCP Migration & Long-term Architecture

## Current State (ERR-06 Fix Applied)

### Problem
The `DistributedControlBus` used `mx.new_stream()` to create a separate MLX stream for bus `all_sum` operations. However, model inference `all_sum` runs on `generation_stream`. The RDMA backend matches collectives by **order**, not by stream identity. When bus and model `all_sum` operations overlapped, ranks cross-matched collectives → deadlock.

**Reproduction**: 100% (4/4 sessions), 2-12 requests before deadlock.

### Fix Applied
- Bus now uses `generation_stream` (stream injection via constructor parameter)
- All `all_sum` operations (bus + model) serialize on the same stream
- `mx.synchronize(generation_stream)` barrier added before bus sync as temporary safety net
- Reference: MLX-LM PR #741 ("It needs to be put in the generation stream to avoid races")

## Remaining Risks

### Rank Divergence
Even with single-stream serialization, `all_sum` is a **collective** — all ranks must call it the same number of times. If one rank takes a different code path (exception, early return, control-flow divergence), the collective counts diverge → deadlock.

Current mitigations:
- `_apply_bus_events` catches exceptions and returns False (shutdown)
- `_handle_batch_error` attempts to keep ranks in sync
- But edge cases remain (e.g., OOM on one rank, timeout differences)

### Blocking Collectives
`mx.distributed.all_sum()` is a blocking C++ call that cannot be interrupted by Python signal handlers. This means:
- SIGTERM cannot interrupt a blocked `all_sum`
- Operators escalate to SIGKILL
- SIGKILL bypasses all cleanup (`atexit`, `_cleanup_metal`)
- Metal wired memory (~310GB per node) is orphaned

## Long-term Goals

### Goal 1: TCP/IPC Control Plane
Separate control-plane communication from `all_sum`:
- Use TCP sockets or Unix domain sockets for bus events
- Reserve `all_sum` exclusively for model tensor operations
- Eliminates collective count divergence risk for control messages
- Reference: exo uses HTTP/gRPC for coordination, not collectives

### Goal 2: Subprocess Isolation (exo pattern)
Run MLX inference in a child subprocess:
- Parent (HTTP server) can always SIGTERM/SIGKILL child
- Child crash = parent spawns new child
- No stuck collectives in parent process
- Reference: exo `ExoProcess`, ollama `llm.Server` subprocess model

### Goal 3: Per-rank Collective Step Counter
Add logging for postmortem deadlock debugging:
- Each rank logs its collective step number
- On deadlock, compare step counters to identify divergence point
- Low overhead (increment + occasional log)

## `mx.synchronize` Removal Criteria

The `mx.synchronize(generation_stream)` barrier in `_batch_inference_step` and `_mock_inference_step` is a temporary safety net. Remove it when:

1. **Stability**: 12+ requests × 3 sessions without deadlock on 2-node TP-2
2. **Confidence**: No new collective ordering issues discovered
3. **Performance**: If synchronize overhead is measurable, removal is more urgent

## References

- MLX-LM PR #741: `_share_object()` stream fix (2026-01-30)
- MLX-LM PR #652: Distributed generation architecture
- llama.cpp PR #11427: Metal ResidencySet wired memory leak
- MLX PR #1510: `set_wired_limit` and ResidencySet behavior
- exo: Subprocess isolation pattern (`ExoProcess`)
- ollama: Go supervisor + subprocess model runner
