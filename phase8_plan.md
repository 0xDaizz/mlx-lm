# Phase 8: Tensor-Parallel RDMA Serving via JACCL

## 1. Overview and Goals

Phase 8 adds multi-node tensor-parallel inference to mlx-lm-server, enabling
large models (e.g., Kimi K2.5 at ~120GB 4-bit) to be served across multiple
Apple Silicon machines connected via Thunderbolt 5 RDMA using JACCL (Jack and
Angelos' Collective Communication Library).

### Goals

1. **Serve models that exceed single-node memory** by sharding weights and KV
   caches across N nodes using tensor parallelism.
2. **Preserve the full serving stack** -- continuous batching, hash-based prefix
   caching, tiered SSD cache, streaming, and the OpenAI-compatible API -- all
   running on rank 0 while model forward passes execute in SPMD across all ranks.
3. **Minimal code changes** -- leverage MLX's existing `sharded_load()` and
   model `shard()` infrastructure. The distributed communication (allreduce) is
   transparent inside the sharded linear layers; no changes to the model code.
4. **Correctness and testability** -- single-node mock testing without JACCL,
   plus a 2-node integration test harness.

### Target Hardware

- 2x Mac Studio M3 Ultra 512GB (1TB aggregate)
- Thunderbolt 5 RDMA via JACCL backend
- macOS Tahoe 26.2+ (required for RDMA over Thunderbolt)
- Production model: `kimi-k25-4bit` (~120GB, has `shard()` support)
- Dev/test model: `mlx-community/Qwen3-4B-4bit` (single-node, has `shard()`)

### Non-Goals (Phase 8)

- Pipeline parallelism (future Phase 9 if needed)
- Expert parallelism for MoE models (separate concern)
- Dynamic rank join/leave (static world size assumed)
- Multi-tenant isolation across ranks

---

## 2. Architecture Design

### 2.1 High-Level Architecture

```
                          ┌────────────────────────────────────┐
  mlx.launch              │         RANK 0 (Leader)            │
  --backend jaccl         │                                    │
  --hostfile hosts.json   │  ┌──────────────────────────────┐  │
  sharded_server.py       │  │   FastAPI (OpenAI API)        │  │
                          │  │   /v1/chat/completions        │  │
                          │  ├──────────────────────────────┤  │
                          │  │   Scheduler                   │  │
                          │  │   Continuous Batching          │  │
                          │  │   Request Queue + Streams      │  │
                          │  ├──────────────────────────────┤  │
                          │  │   KVCacheManager              │  │
                          │  │   Block Pool + Hash Table      │  │
                          │  │   (rank-local KV dimensions)   │  │
                          │  ├──────────────────────────────┤  │
                          │  │   SSD Cache (rank-local)      │  │
                          │  ├──────────────────────────────┤  │
                          │  │   BatchGenerator              │  │
                          │  │   (sharded model shard 0)     │  │
                          │  └──────────┬───────────────────┘  │
                          └─────────────┼──────────────────────┘
                                        │ mx.distributed
                     ┌──────────────────┼──────────────────┐
                     │ allreduce (JACCL │ RDMA over TB5)   │
                     ▼                  ▼                   ▼
              ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
              │   RANK 0     │  │   RANK 1     │   │   RANK N     │
              │              │  │              │   │              │
              │  model shard │  │  model shard │   │  model shard │
              │  KV shard    │  │  KV shard    │   │  KV shard    │
              │  (heads/N)   │  │  (heads/N)   │   │  (heads/N)   │
              └──────────────┘  └──────────────┘   └──────────────┘
```

### 2.2 Rank Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                       RANK 0 (Leader)                           │
│                                                                 │
│  Owns:                                                          │
│    - FastAPI server (uvicorn)                                   │
│    - Scheduler (request queue, sequence state, token streams)   │
│    - Tokenizer (encode/decode)                                  │
│    - KVCacheManager + SSD tier (rank-local shard)               │
│    - BatchGenerator (with sharded model)                        │
│                                                                 │
│  Does NOT own:                                                  │
│    - Model weights for other ranks' shards                      │
│    - KV cache data for other ranks' head shards                 │
│                                                                 │
│  Coordinates:                                                   │
│    - Scheduling decisions (which requests to prefill/decode)    │
│    - Token emission and streaming responses                     │
│    - Block allocation (token-hash-based, same across ranks)     │
│                                                                 │
│  Communication with workers:                                    │
│    - Forward pass: implicit via mx.distributed allreduce        │
│    - Scheduling sync: broadcast token_ids before each step      │
│    - Shutdown: broadcast sentinel to terminate worker loops      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     RANK 1..N-1 (Workers)                       │
│                                                                 │
│  Owns:                                                          │
│    - Sharded model weights (their shard of Q/K/V/O/MLP)        │
│    - Rank-local KV cache (their head shard)                     │
│    - Rank-local SSD cache (optional, their shard)               │
│                                                                 │
│  Does NOT own:                                                  │
│    - FastAPI server (no HTTP)                                   │
│    - Scheduler (no request state)                               │
│    - Tokenizer                                                  │
│                                                                 │
│  Worker loop:                                                   │
│    1. Receive scheduling broadcast from rank 0                  │
│    2. Insert/remove sequences in local BatchGenerator           │
│    3. Call batch_generator.next() (participates in allreduce)   │
│    4. Discard returned tokens (rank 0 handles emission)         │
│    5. Repeat until shutdown sentinel                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Request Lifecycle in Distributed Mode

```
Client POST /v1/chat/completions
         │
         ▼
    ┌─ RANK 0 ──────────────────────────────────────────────────┐
    │                                                            │
    │  1. FastAPI handler receives request                       │
    │  2. Tokenizer encodes prompt → token_ids                   │
    │  3. Scheduler.submit_request(InferenceRequest)             │
    │  4. Scheduler._insert_new_requests_batch():                │
    │     a. Creates SequenceState                               │
    │     b. Broadcasts INSERT command + token_ids to all ranks  │
    │     c. Inserts into local BatchGenerator                   │
    │  5. Scheduler._batch_inference_step():                     │
    │     a. Broadcasts STEP command to all ranks                │
    │     b. All ranks call batch_generator.next() in lockstep   │
    │        (allreduce happens inside sharded linear layers)    │
    │     c. Rank 0 processes responses → TokenEvents            │
    │  6. Repeat step 5 until finish_reason                      │
    │  7. On finish:                                             │
    │     a. Broadcasts REMOVE command for finished UIDs         │
    │     b. Rank 0 stores KV blocks in cache (rank-local)      │
    │     c. Token stream closed → HTTP response completes       │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
         │                              │
         │ (allreduce in forward pass)  │ (scheduling broadcasts)
         ▼                              ▼
    ┌─ RANK 1..N-1 ─────────────────────────────────────────────┐
    │                                                            │
    │  Worker loop receives commands:                            │
    │    INSERT → insert token_ids into local BatchGenerator     │
    │    STEP   → call batch_generator.next() (joins allreduce) │
    │    REMOVE → remove UIDs from local BatchGenerator          │
    │    SHUTDOWN → exit loop                                    │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

### 2.4 Scheduling Broadcast Protocol

Rank 0 must synchronize scheduling decisions so all ranks call
`BatchGenerator.next()` with the same active sequences. This is achieved
via lightweight broadcasts before each inference step.

```
┌─────────────────────────────────────────────────────────────┐
│  SchedulingCommand (broadcast from rank 0 before each step) │
│                                                             │
│  Format: mx.array encoded as:                               │
│    [command_type, num_items, ...payload]                     │
│                                                             │
│  Commands:                                                   │
│    NOOP    = 0  (no changes, proceed to next())             │
│    INSERT  = 1  [uid, num_tokens, token_ids...]             │
│    REMOVE  = 2  [uid1, uid2, ...]                           │
│    STEP    = 3  (all ranks call next())                     │
│    SHUTDOWN = 4  (workers exit)                              │
│                                                             │
│  Protocol per inference iteration:                           │
│    1. Rank 0 builds command array                            │
│    2. mx.distributed.all_sum(cmd) — rank 0 sends, others 0  │
│       (equivalent to broadcast; workers send zeros)          │
│    3. Workers parse and execute command                      │
│    4. All ranks call batch_generator.next() in lockstep      │
│    5. Only rank 0 processes the returned tokens              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation note:** MLX distributed does not have a native `broadcast`
primitive. We simulate it using `all_sum` where rank 0 sends the real data
and all other ranks send an array of zeros. This is a standard pattern in
MLX distributed programs. An alternative is to use the model's forward pass
itself as the synchronization point (since allreduce in linear layers forces
all ranks to be in lockstep), and have rank 0 feed the same token_ids to
each rank's BatchGenerator via a pre-step `all_sum` of token data.

---

## 3. Design Decisions

### D1. Tensor Parallel (Not Pipeline) as Primary Strategy

**Decision:** Use tensor parallelism for Kimi K2.5 serving.

**Rationale:**
- Kimi K2.5's `shard()` method is implemented and tested upstream.
- Tensor parallelism shards every layer across all ranks, so each rank
  processes the same layers but with fewer attention heads and smaller MLP
  dimensions. This keeps latency low (single pass, no pipeline bubbles).
- Pipeline parallelism introduces bubbles and complicates scheduling (need
  micro-batching). It is better suited for very deep models where layers
  don't shard well.
- With 2 nodes, tensor parallel halves memory per node (~60GB each from
  120GB total), well within the 512GB per node.
- MLX's `ShardedToAllLinear` handles the allreduce transparently in `o_proj`
  and `down_proj`, so no changes to model code.

**Trade-off:** Higher inter-node bandwidth requirement (allreduce every layer)
vs. pipeline parallel (send activations once between stages). TB5 RDMA at
~40 Gbps is sufficient for 2-node tensor parallel with 4-bit models.

### D2. Replace `mlx_lm.load()` with `sharded_load()` in Distributed Mode

**Decision:** When `config.use_distributed` is True, call
`mlx_lm.utils.sharded_load()` instead of `mlx_lm.load()`.

**Rationale:**
- `sharded_load()` handles lazy model loading, shard discovery, weight
  download for the local shard only, `model.shard(tensor_group)` call,
  `mx.eval(model.parameters())`, and synchronization barrier.
- Reusing upstream infrastructure means we don't reimplement sharding logic.
- The returned model is already sharded; `BatchGenerator` works unchanged
  because the forward pass is transparent after sharding.

### D3. Entry Point Wrapped in `mlx.launch` Context

**Decision:** Create a new `sharded_server.py` entry script that is invoked
via `mlx.launch`. The existing `__main__.py` remains for single-node use.

**Rationale:**
- `mlx.launch` sets up the distributed runtime, spawns processes on remote
  hosts, and initializes `mx.distributed`. This is the official MLX way.
- We cannot call `mlx.launch` from within Python easily; it is a CLI tool
  that manages process lifecycle.
- A separate entry script keeps the single-node path clean and untouched.

**Usage:**
```bash
mlx.launch \
    --backend jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --hostfile hosts.json \
    -m mlx_lm_server.sharded_main \
    --model kimi-k25-4bit \
    --port 8000
```

### D4. Only Rank 0 Runs FastAPI + Scheduler + Tokenizer

**Decision:** The HTTP server, scheduler, tokenizer, and all client-facing
logic run exclusively on rank 0. Worker ranks run a minimal loop.

**Rationale:**
- Avoids duplicating request state across ranks.
- Tokenization is cheap (CPU-bound, negligible vs. model forward pass).
- Scheduler decisions are lightweight; broadcasting them is simpler than
  distributed consensus.
- Follows vLLM's "driver process + SPMD workers" model.
- Workers only need: sharded model, local BatchGenerator, command receiver.

### D5. KVCacheManager Operates on Rank-Local KV Dimensions

**Decision:** Each rank's KVCacheManager manages blocks with shape
`(n_kv_heads / N, seq_len, head_dim)` per layer, where N is the world size.

**Rationale:**
- After `model.shard()`, `n_kv_heads` is divided by N. The KV cache tensors
  produced by BatchGenerator naturally have the sharded dimension.
- Block hashes are computed from token_ids (content-based), NOT from KV
  tensor data. This means block hashes are identical across all ranks.
- Block allocation decisions made on rank 0 are valid for all ranks because
  the hash is content-based. Workers can compute the same hashes locally.
- Each rank stores only its shard of the KV data per block. No cross-rank
  KV transfer is needed.

### D6. SSD Cache Stores Rank-Local KV Shard Only

**Decision:** Each rank's SSD cache persists its own KV shard independently.

**Rationale:**
- SSD read bandwidth (~7.4 GB/s on M3 Ultra) is per-node, so each node
  benefits from its own local SSD tier.
- Block hashes are the same across ranks, so cache hits/misses are
  synchronized by construction (same prefix = same hash = all ranks hit or
  all ranks miss).
- No cross-node SSD coordination needed.

### D7. Broadcast Scheduling via `all_sum` of Command Arrays

**Decision:** Use `mx.distributed.all_sum()` with rank 0 sending real data
and workers sending zeros to simulate broadcast.

**Rationale:**
- MLX has no `broadcast` primitive. `all_sum` with zeros from non-leaders
  is functionally equivalent and is the idiomatic pattern.
- Command arrays are small (a few hundred int32 values at most for a batch
  of 8 inserts), so bandwidth is negligible.
- Alternative: point-to-point `send`/`recv`, but `send`/`recv` over JACCL
  has a known SIGBUS bug under asymmetric timing. `all_sum` is collective
  and avoids this issue.

### D8. Each Rank Maintains Its Own BatchGenerator

**Decision:** Every rank creates its own `BatchGenerator` with the local
model shard. Rank 0's scheduler drives the scheduling logic; workers mirror
insert/remove/next operations based on broadcast commands.

**Rationale:**
- `BatchGenerator` manages per-sequence KV cache state internally. Each rank
  must maintain its own KV cache (with sharded head dimensions).
- After `model.shard()`, calling `model(tokens)` participates in allreduce
  transparently. So all ranks calling `batch_generator.next()` in lockstep
  produces correct results.
- Rank 0 processes the returned tokens; workers discard them (only rank 0
  has the tokenizer and streaming infrastructure).

---

## 4. Implementation Tasks

### Phase 8.0: Configuration and Types

#### P8.0.1 -- Extend ServerConfig for Distributed Settings

**File:** `mlx_lm_server/config.py`

Add the following fields to `ServerConfig`:

```python
# Distributed (Tensor Parallel)
use_distributed: bool = False
distributed_backend: str = "jaccl"      # "jaccl", "ring", "mpi"
hostfile: str | None = None             # Path to hosts.json for mlx.launch
tensor_parallel_size: int = 1           # World size (auto-detected if 0)
rank: int = 0                           # Set at runtime by distributed init
world_size: int = 1                     # Set at runtime by distributed init
```

**Deliverables:**
- Updated `ServerConfig` dataclass
- Unit test: `test_config_distributed_defaults`

---

#### P8.0.2 -- Add Distributed Types

**File:** `mlx_lm_server/types.py`

Add command types for the scheduling broadcast protocol:

```python
class SchedulingCommandType(IntEnum):
    NOOP     = 0
    INSERT   = 1
    REMOVE   = 2
    STEP     = 3
    SHUTDOWN = 4

@dataclass
class SchedulingCommand:
    command_type: SchedulingCommandType
    uids: list[int] = field(default_factory=list)
    token_ids_per_uid: dict[int, list[int]] = field(default_factory=dict)
    sampler_params_per_uid: dict[int, dict] = field(default_factory=dict)
```

**Deliverables:**
- `SchedulingCommandType` enum
- `SchedulingCommand` dataclass
- Serialization: `encode_command() -> mx.array` and `decode_command(mx.array) -> SchedulingCommand`
- Tests: `test_command_encode_decode_roundtrip`, `test_command_insert_with_tokens`

---

### Phase 8.1: Distributed Entry Point

#### P8.1.1 -- Create Sharded Server Entry Point

**File:** `mlx_lm_server/sharded_main.py` (new)

This is the script invoked by `mlx.launch`. It:
1. Calls `mx.distributed.init()` to get group, rank, world_size.
2. Calls `sharded_load(model, tensor_group=group)` to load the sharded model.
3. If rank == 0: creates ServerConfig, KVCacheManager, Scheduler, FastAPI,
   runs uvicorn.
4. If rank != 0: enters worker loop (see P8.2).
5. On shutdown: rank 0 broadcasts SHUTDOWN, workers exit.

```python
# Pseudocode
def main():
    group = mx.distributed.init()
    rank, world_size = group.rank(), group.size()

    config = parse_args()
    config.rank = rank
    config.world_size = world_size
    config.use_distributed = True

    model, tokenizer = sharded_load(config.model, tensor_group=group)

    if rank == 0:
        run_leader(config, model, tokenizer, group)
    else:
        run_worker(config, model, group)
```

**Deliverables:**
- `sharded_main.py` with `main()`, `run_leader()`, `run_worker()` functions
- Argument parsing compatible with `mlx.launch` (positional script + flags)
- Test: `test_sharded_main_arg_parsing` (mock distributed init)

---

#### P8.1.2 -- Hosts Configuration Helper

**File:** `mlx_lm_server/distributed_utils.py` (new)

Utilities for distributed setup:

```python
def create_hostfile(hosts: list[dict], path: str) -> Path:
    """Generate a hosts.json for mlx.launch."""

def validate_distributed_env() -> dict:
    """Check that mx.distributed is available, return rank/size info."""

def get_rank_local_cache_dir(base_dir: Path, rank: int) -> Path:
    """Return rank-specific SSD cache directory."""
```

**Deliverables:**
- `distributed_utils.py` with helper functions
- Tests: `test_create_hostfile`, `test_validate_env_no_distributed`,
  `test_rank_local_cache_dir`

---

### Phase 8.2: Worker Loop

#### P8.2.1 -- Implement Worker Event Loop

**File:** `mlx_lm_server/worker.py` (new)

The worker loop runs on ranks 1..N-1. It:
1. Creates a local `BatchGenerator` with the sharded model.
2. Enters a loop waiting for scheduling commands via `all_sum`.
3. Executes commands (INSERT, REMOVE, STEP, SHUTDOWN).
4. On STEP: calls `batch_generator.next()`, discards returned tokens.

```python
class DistributedWorker:
    def __init__(self, model, config, group):
        self.model = model
        self.config = config
        self.group = group
        self.rank = group.rank()
        self._batch_generator = BatchGenerator(model, ...)
        self._uid_active: set[int] = set()

    def run(self):
        """Main worker loop. Blocks until SHUTDOWN received."""
        while True:
            cmd = self._receive_command()
            if cmd.command_type == SchedulingCommandType.SHUTDOWN:
                break
            self._execute_command(cmd)

    def _receive_command(self) -> SchedulingCommand:
        """Receive command via all_sum (rank 0 sends, we send zeros)."""
        ...

    def _execute_command(self, cmd: SchedulingCommand):
        """Execute INSERT/REMOVE/STEP on local BatchGenerator."""
        ...
```

**Deliverables:**
- `DistributedWorker` class with `run()`, `_receive_command()`, `_execute_command()`
- INSERT handling: creates prompt cache, inserts into BatchGenerator
- REMOVE handling: removes UIDs from BatchGenerator
- STEP handling: calls `batch_generator.next()`, discards results
- SHUTDOWN handling: closes BatchGenerator, returns
- Tests: `test_worker_insert_step_remove` (with mock BatchGenerator)

---

#### P8.2.2 -- Command Serialization and Transport

**File:** `mlx_lm_server/distributed_utils.py` (extend)

Implement the `all_sum`-based broadcast protocol:

```python
def broadcast_command(cmd: SchedulingCommand, group: mx.distributed.Group) -> None:
    """Rank 0 broadcasts a command. Workers call with NOOP (sends zeros)."""

def receive_command(group: mx.distributed.Group) -> SchedulingCommand:
    """Workers receive command. Rank 0 should not call this."""

def _encode_command(cmd: SchedulingCommand) -> mx.array:
    """Encode command to int32 array: [type, n_items, ...payload]."""

def _decode_command(arr: mx.array) -> SchedulingCommand:
    """Decode int32 array back to SchedulingCommand."""
```

**Encoding format:**
```
INSERT: [1, n_inserts, uid0, n_tokens0, tok0, tok1, ..., uid1, n_tokens1, ...]
REMOVE: [2, n_removes, uid0, uid1, ...]
STEP:   [3, 0]
NOOP:   [0, 0]
SHUTDOWN: [4, 0]
```

**Max array size:** Pre-allocate a fixed-size buffer (e.g., 65536 int32 values)
to avoid dynamic allocation. Pad with zeros. This supports up to ~8 concurrent
inserts of ~8000 tokens each, which exceeds our max_batch_size * typical prompt
length.

**Deliverables:**
- `broadcast_command()` and `receive_command()` functions
- `_encode_command()` and `_decode_command()` with tests
- Tests: `test_broadcast_insert_command`, `test_broadcast_remove_command`,
  `test_broadcast_step_command`, `test_buffer_overflow_raises`

---

### Phase 8.3: Leader-Side Distributed Scheduler

#### P8.3.1 -- Extend Scheduler for Distributed Coordination

**File:** `mlx_lm_server/scheduler.py` (modify)

Add distributed awareness to the existing Scheduler class. When
`config.use_distributed` is True and `config.rank == 0`:

1. Before inserting new requests into BatchGenerator, broadcast INSERT
   commands so workers mirror the insert.
2. Before calling `batch_generator.next()`, broadcast a STEP command.
3. After removing finished/cancelled UIDs, broadcast REMOVE commands.
4. On `stop()`, broadcast SHUTDOWN.

```python
# In Scheduler.__init__:
self._distributed_group: mx.distributed.Group | None = None

# New method:
def set_distributed_group(self, group):
    self._distributed_group = group

# Modified _batch_inference_step():
def _batch_inference_step(self):
    # ... existing cancellation logic ...

    # Broadcast REMOVE for cancelled UIDs
    if self._distributed_group and cancelled_uids:
        broadcast_command(SchedulingCommand(REMOVE, uids=cancelled_uids), ...)

    # ... existing insert logic ...

    # Broadcast INSERT for new requests
    if self._distributed_group and new_inserts:
        broadcast_command(SchedulingCommand(INSERT, ...), ...)

    # Broadcast STEP before next()
    if self._distributed_group:
        broadcast_command(SchedulingCommand(STEP), ...)

    # ... existing next() + response processing (unchanged) ...
```

**Key constraint:** The broadcast must happen BEFORE `batch_generator.next()`
because workers are blocking on `all_sum` to receive the command. If rank 0
calls `next()` first, it would enter the model forward pass which also uses
`all_sum` (in allreduce), causing a deadlock because workers are in a
different `all_sum` context.

**Deliverables:**
- Modified `Scheduler.__init__` with `_distributed_group`
- Modified `_batch_inference_step()` with broadcast calls
- Modified `_insert_new_requests_batch()` to collect inserts for broadcast
- Modified `_process_cancellations_batch()` to collect removes for broadcast
- Modified `stop()` to broadcast SHUTDOWN
- All changes gated behind `self._distributed_group is not None`
- Existing single-node behavior unchanged (no distributed group set)
- Tests: `test_scheduler_distributed_broadcasts` (mock all_sum),
  `test_scheduler_single_node_unchanged`

---

#### P8.3.2 -- Sampler Parameter Synchronization

**File:** `mlx_lm_server/distributed_utils.py` (extend)

Each request may have different temperature/top_p. Workers need these to
create matching samplers for their BatchGenerator inserts. Include sampler
parameters in the INSERT command.

**Encoding extension for INSERT:**
```
[1, n_inserts,
 uid0, n_tokens0, temp_x1000_0, top_p_x1000_0, tok0, tok1, ...,
 uid1, n_tokens1, temp_x1000_1, top_p_x1000_1, tok0, tok1, ...,
 ...]
```

Temperature and top_p are encoded as int32 by multiplying by 1000 and
rounding (e.g., temp=0.7 becomes 700). This avoids float encoding
complexity while preserving sufficient precision.

**Deliverables:**
- Extended INSERT encoding with sampler params
- Worker-side sampler creation from decoded params
- Tests: `test_sampler_param_encoding`, `test_sampler_roundtrip_precision`

---

### Phase 8.4: KV Cache Distributed Awareness

#### P8.4.1 -- Rank-Local KV Dimensions in KVCacheManager

**File:** `mlx_lm_server/kv_cache_manager.py` (modify)

When operating in distributed mode, the KV cache blocks store tensors with
`n_kv_heads / world_size` heads instead of `n_kv_heads`. The block hash
computation is unaffected (it uses token_ids, not KV data).

Changes needed:

1. Add `world_size` parameter to `KVCacheManager.__init__()` (default 1).
2. In `extract_block()` and `inject_blocks()`, the head dimension is
   already dynamic (sliced from whatever the cache layer provides), so
   no changes needed to the slicing logic.
3. In `decompose_cache_to_blocks()` and `reconstruct_cache_from_blocks()`,
   same -- they operate on whatever tensors are in the prompt cache layers.
4. The only real change: validation/logging that reports head counts should
   account for sharding.

**Deliverables:**
- `KVCacheManager.__init__` accepts `world_size` parameter
- Logging/validation updated for sharded head counts
- Tests: `test_kv_cache_sharded_dimensions`, `test_block_hash_same_across_ranks`

---

#### P8.4.2 -- Rank-Local SSD Cache Paths

**File:** `mlx_lm_server/ssd_cache.py` (modify)

When running distributed, each rank needs a separate SSD cache directory to
avoid file conflicts:

```
~/.cache/mlx-lm-server/kv-cache/rank-0/
~/.cache/mlx-lm-server/kv-cache/rank-1/
```

Changes:
1. `SSDCache.__init__` accepts optional `rank` parameter.
2. If rank is provided, append `/rank-{rank}/` to cache_dir.
3. Block hashes are the same across ranks (token-based), so the file naming
   scheme is consistent, but KV data content differs per rank.

**Deliverables:**
- `SSDCache.__init__` with `rank` parameter
- Rank-specific subdirectory creation
- Tests: `test_ssd_cache_rank_isolation`, `test_ssd_cache_rank_dir_creation`

---

### Phase 8.5: Sharded Entry Point Integration

#### P8.5.1 -- Wire Up Leader Path

**File:** `mlx_lm_server/sharded_main.py` (extend)

Complete the `run_leader()` function:

```python
def run_leader(config, model, tokenizer, group):
    # KV cache manager with world_size awareness
    kv_cache_manager = KVCacheManager(config, world_size=group.size())

    # SSD cache with rank-specific directory
    ssd_cache = SSDCache(config.ssd_cache_dir, config.ssd_ttl_days, rank=0)
    tiered_cache = TieredKVCache(kv_cache_manager, ssd_cache)

    # Scheduler with distributed group
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer,
                          kv_cache_manager=kv_cache_manager)
    scheduler._tiered_cache = tiered_cache
    scheduler.set_distributed_group(group)
    scheduler.run_inference_loop()

    # FastAPI app (only rank 0 serves HTTP)
    app = create_app(config=config, scheduler=scheduler, tokenizer=tokenizer)
    uvicorn.run(app, host=config.host, port=config.port)
```

**Deliverables:**
- Complete `run_leader()` implementation
- Shutdown hook: when uvicorn exits, `scheduler.stop()` broadcasts SHUTDOWN
- Test: `test_leader_startup_sequence` (mock distributed group)

---

#### P8.5.2 -- Wire Up Worker Path

**File:** `mlx_lm_server/sharded_main.py` (extend)

Complete the `run_worker()` function:

```python
def run_worker(config, model, group):
    worker = DistributedWorker(model=model, config=config, group=group)
    worker.run()  # Blocks until SHUTDOWN
    logger.info("Rank %d: worker shutdown complete", group.rank())
```

**Deliverables:**
- Complete `run_worker()` implementation
- Graceful shutdown on SIGTERM/SIGINT
- Test: `test_worker_shutdown_on_command`

---

### Phase 8.6: End-to-End Testing

#### P8.6.1 -- Single-Node Distributed Simulation Tests

**File:** `tests/test_distributed.py` (new)

Test the distributed logic without actually running `mlx.launch` or JACCL.
Mock `mx.distributed.Group` and `mx.distributed.all_sum` to simulate
multi-rank behavior in a single process.

```python
class MockDistributedGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size
    def rank(self): return self._rank
    def size(self): return self._size

# Tests:
- test_command_encode_decode_all_types
- test_worker_receives_insert_and_steps
- test_worker_receives_remove
- test_worker_receives_shutdown
- test_scheduler_broadcasts_insert_before_step
- test_scheduler_broadcasts_remove_on_cancel
- test_scheduler_broadcasts_shutdown_on_stop
- test_kv_cache_sharded_head_count
- test_block_hash_invariant_across_ranks
- test_ssd_cache_rank_isolation
- test_full_request_lifecycle_mock_distributed
```

**Deliverables:**
- `MockDistributedGroup` class
- `mock_all_sum()` that captures broadcast data for assertions
- 12+ tests covering the distributed protocol without real multi-node setup
- All tests pass with `pytest tests/test_distributed.py -v`

---

#### P8.6.2 -- Two-Node Integration Test Script

**File:** `scripts/test_distributed_2node.py` (new)

A script that can be run via `mlx.launch` on 2 nodes to verify end-to-end
distributed serving:

```bash
mlx.launch \
    --backend jaccl \
    --hostfile hosts.json \
    scripts/test_distributed_2node.py \
    --model mlx-community/Qwen3-4B-4bit
```

The script:
1. Loads the model sharded across 2 ranks.
2. Rank 0 starts the server on a random port.
3. Rank 0 sends a test request via HTTP to localhost.
4. Verifies response is valid and contains generated tokens.
5. Checks that both ranks participated (via allreduce counter or similar).
6. Shuts down cleanly.

**Deliverables:**
- `test_distributed_2node.py` script
- Runnable via `mlx.launch` on real 2-node TB5 setup
- Prints PASS/FAIL with timing information
- Documents required hosts.json format

---

#### P8.6.3 -- Performance Benchmark Script

**File:** `scripts/benchmark_distributed.py` (new)

Measures distributed overhead vs. single-node:

```
Metrics:
- Time-to-first-token (TTFT) for 1K/2K/4K token prompts
- Decode throughput (tokens/sec) at batch sizes 1, 2, 4, 8
- Allreduce overhead per step (measured via timing instrumentation)
- Prefill throughput (tokens/sec)
- Memory per rank
```

**Deliverables:**
- Benchmark script with configurable prompt lengths and batch sizes
- JSON output format for results
- Comparison mode: run single-node, then distributed, print delta

---

### Phase 8.7: Devil's Advocate Review

#### P8.7.1 -- Adversarial Review of Distributed Code

**File:** `tests/test_adversarial.py` (extend)

After all P8 tasks pass, devil's-advocate-agent reviews for:

| Domain | What to Look For |
|--------|-----------------|
| **Deadlocks** | Rank 0 calls next() before broadcast; worker and leader in different all_sum contexts |
| **Split brain** | Ranks disagree on active UIDs after network hiccup or timing skew |
| **Buffer overflow** | INSERT command with prompt longer than broadcast buffer |
| **Partial failure** | One rank crashes mid-step; other rank hangs in allreduce |
| **KV cache mismatch** | Block hash collision + different rank stores wrong KV shard |
| **SSD cache corruption** | Two ranks write to same SSD directory (rank isolation failure) |
| **Shutdown race** | SHUTDOWN broadcast lost; worker hangs forever |
| **Memory leak** | Worker accumulates KV state for finished sequences |
| **Token mismatch** | Worker's BatchGenerator produces different tokens than leader |

**Deliverables:**
- 10+ adversarial tests in `tests/test_adversarial.py`
- Findings report (CRITICAL / HIGH / MEDIUM / LOW)
- All CRITICAL and HIGH findings fixed before merge

---

### Task Dependency Graph

```
P8.0.1 ──┐
P8.0.2 ──┼──→ P8.1.1 ──→ P8.5.1 ──┐
          │         │                │
          │    P8.1.2               │
          │                         │
          ├──→ P8.2.1 ──→ P8.5.2 ──┤
          │         │                │
          ├──→ P8.2.2 ──┘            ├──→ P8.6.1 ──→ P8.6.2 ──→ P8.6.3 ──→ P8.7.1
          │                         │
          ├──→ P8.3.1 ──────────────┤
          │         │                │
          ├──→ P8.3.2               │
          │                         │
          ├──→ P8.4.1 ──────────────┤
          │                         │
          └──→ P8.4.2 ──────────────┘
```

**Critical path:** P8.0 -> P8.2.2 -> P8.2.1 -> P8.3.1 -> P8.5.1 -> P8.6.1

---

## 5. Testing Strategy

### 5.1 Unit Tests (Single-Process, No JACCL)

All distributed logic is tested by mocking `mx.distributed`:

```python
# Mock pattern for all_sum broadcast
class MockAllSum:
    """Captures broadcast data for assertion, returns it to all 'ranks'."""
    def __init__(self):
        self.calls = []

    def __call__(self, arr, group=None, stream=None):
        self.calls.append(arr)
        return arr  # In real multi-rank, all ranks get the sum
```

**Test files:**
| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_distributed.py` | Command encode/decode, worker logic, scheduler broadcasts, KV sharding | P8.0-P8.5 |
| `tests/test_kv_cache_manager.py` | Extended with `world_size` parameter tests | P8.4.1 |
| `tests/test_ssd_cache.py` | Extended with rank-specific directory tests | P8.4.2 |
| `tests/test_adversarial.py` | Extended with distributed adversarial tests | P8.7.1 |

### 5.2 Integration Tests (2-Node, Requires Hardware)

These tests require 2 Mac Studios connected via TB5:

```bash
# Run from either machine
mlx.launch --backend jaccl --hostfile hosts.json \
    scripts/test_distributed_2node.py --model mlx-community/Qwen3-4B-4bit
```

Tests:
- Model loads and shards correctly across 2 ranks
- Single request generates correct output
- Multiple concurrent requests work
- Cancel mid-generation works
- Graceful shutdown works
- SSD cache stores rank-local data correctly

### 5.3 Performance Validation

Acceptance criteria for 2-node Kimi K2.5:
- TTFT < 2x single-node TTFT (allreduce overhead acceptable)
- Decode throughput > 1.5x single-node (memory relief enables larger batches)
- No deadlocks under sustained load (100 sequential requests)
- Clean shutdown within 5 seconds

### 5.4 Test Execution Order

```
1. pytest tests/test_distributed.py -v           # Unit tests (CI, no hardware)
2. pytest tests/test_kv_cache_manager.py -v       # Regression (existing + new)
3. pytest tests/test_ssd_cache.py -v              # Regression (existing + new)
4. pytest tests/ -v --tb=short                    # Full suite regression
5. scripts/test_distributed_2node.py              # 2-node integration (manual)
6. scripts/benchmark_distributed.py               # Performance (manual)
7. pytest tests/test_adversarial.py -v            # Adversarial (after DA review)
```

---

## 6. Risks and Unknowns

### R1. JACCL Stability (HIGH)

**Risk:** JACCL is relatively new. Known SIGBUS bug with `send`/`recv` under
asymmetric timing. May have undiscovered issues with `all_sum` under load.

**Mitigation:**
- Use only `all_sum` and `all_gather` (collective ops), avoid point-to-point
  `send`/`recv`.
- Add timeout to worker command receive loop; if no command received within
  30 seconds, log warning and continue (heartbeat pattern).
- Test extensively on actual hardware before production deployment.

### R2. BatchGenerator Thread Safety Across Ranks (MEDIUM)

**Risk:** `BatchGenerator.next()` is not thread-safe within a single process.
In distributed mode, each rank has its own `BatchGenerator` in its own
process, so this should not be an issue. But if we ever move to multi-process
per node (`-n 2` with `mlx.launch`), this could break.

**Mitigation:**
- Document that tensor_parallel_size should equal the number of hosts, not
  processes per host.
- Add assertion: `config.tensor_parallel_size == group.size()`.

### R3. Allreduce Latency Under High Batch Size (MEDIUM)

**Risk:** With large batches, the allreduce data volume per step increases
(one allreduce per ShardedToAllLinear per layer). For Kimi K2.5 with 72+
layers, this is 144+ allreduces per forward pass.

**Mitigation:**
- TB5 RDMA provides ~40 Gbps; at 4-bit quantization the data volume per
  allreduce is small (hidden_size / N * batch_size * sizeof(float16)).
- Benchmark and tune: if latency is too high, reduce max_batch_size in
  distributed mode.
- Future: JACCL may add bandwidth-optimal ring/tree allreduce (currently
  full mesh only).

### R4. Scheduling Broadcast Buffer Overflow (LOW)

**Risk:** If a request has an extremely long prompt (>60K tokens), the
broadcast buffer (65536 int32) may be insufficient.

**Mitigation:**
- Validate prompt length against buffer capacity before broadcast.
- For very long prompts, split into multiple INSERT commands.
- Alternatively, broadcast only metadata (uid, length, sampler params) and
  have workers tokenize independently (requires workers to have tokenizer).

### R5. macOS Tahoe Version Dependency (LOW)

**Risk:** JACCL RDMA requires macOS Tahoe 26.2+. Users on older macOS
versions cannot use distributed mode.

**Mitigation:**
- Fall back to `ring` backend (TCP) if JACCL is unavailable.
- Log clear error message with version requirement.
- `ring` backend works on any macOS with Thunderbolt networking.

### R6. Model Does Not Support `shard()` (LOW)

**Risk:** Not all mlx-lm models have a `shard()` method. Loading such a
model in distributed mode will fail.

**Mitigation:**
- `sharded_load()` already raises `ValueError` if model lacks `shard()`.
- Document supported models (llama, qwen3, qwen2, kimi_k25, deepseek_v2,
  deepseek_v3, etc.).

---

## 7. Future Extensions

### 7.1 Pipeline Parallelism (Phase 9)

For extremely deep models or 3+ node setups, pipeline parallelism may be
beneficial. MLX already supports `model.model.pipeline(pipeline_group)`.
Would require:
- Micro-batch scheduling in the Scheduler
- Activation transfer between pipeline stages via `send`/`recv`
- Pipeline bubble management
- Separate pipeline_group and tensor_group for hybrid parallelism

### 7.2 Expert Parallelism for MoE Models

Kimi K2.5 and Deepseek V3 are MoE models. Expert parallelism would distribute
experts across nodes:
- Route tokens to the node holding the relevant expert
- Requires all-to-all communication (token routing)
- MLX's MoE sharding is partially supported in some models

### 7.3 Dynamic Scaling

Add/remove nodes at runtime:
- Requires re-sharding model weights
- KV cache migration between ranks
- Not supported by MLX distributed currently

### 7.4 Fault Tolerance

Handle rank failures gracefully:
- Detect rank crash via heartbeat timeout
- Fall back to reduced world_size
- Requires re-sharding (complex)
- Short term: just restart the full cluster

### 7.5 Multi-Process Per Node

Use `-n 2` with `mlx.launch` to run 2 processes per Mac Studio:
- Each process gets half the GPU (if MLX supports this)
- Doubles tensor_parallel_size to 4 with 2 nodes
- Requires careful memory management (shared GPU memory)
- Currently not recommended due to BatchGenerator single-GPU assumption

### 7.6 Prefill-Decode Disaggregation

Separate prefill and decode to different nodes/processes:
- Prefill is compute-bound; decode is memory-bound
- Could improve overall throughput
- Requires KV cache transfer between prefill and decode nodes

---

## Appendix A: File Change Summary

| File | Action | Owner |
|------|--------|-------|
| `mlx_lm_server/config.py` | Modify (add distributed fields) | Team Lead |
| `mlx_lm_server/types.py` | Modify (add SchedulingCommand types) | Team Lead |
| `mlx_lm_server/sharded_main.py` | **New** (distributed entry point) | server-agent |
| `mlx_lm_server/worker.py` | **New** (worker loop) | scheduler-agent |
| `mlx_lm_server/distributed_utils.py` | **New** (broadcast protocol, helpers) | scheduler-agent |
| `mlx_lm_server/scheduler.py` | Modify (add broadcast calls) | scheduler-agent |
| `mlx_lm_server/kv_cache_manager.py` | Modify (world_size param) | cache-agent |
| `mlx_lm_server/ssd_cache.py` | Modify (rank param) | cache-agent |
| `tests/test_distributed.py` | **New** (distributed unit tests) | test-agent |
| `tests/test_adversarial.py` | Modify (add distributed adversarial tests) | devil's-advocate-agent |
| `scripts/test_distributed_2node.py` | **New** (integration test) | test-agent |
| `scripts/benchmark_distributed.py` | **New** (perf benchmark) | test-agent |
| `mlx_lm/` | **READ ONLY** -- no modifications | -- |

## Appendix B: hosts.json Format

```json
[
    {"host": "mac-studio-1.local", "port": 22},
    {"host": "mac-studio-2.local", "port": 22}
]
```

Hosts must have:
- SSH key-based authentication configured
- Same mlx-lm-server virtualenv at the same path
- macOS Tahoe 26.2+
- Thunderbolt 5 connection (for JACCL backend)

Verify connectivity:
```bash
mlx.distributed_config --hostfile hosts.json
```

## Appendix C: Launch Commands

### Development (single-node, simulated distributed)

```bash
# Single-process, non-distributed (existing path)
python -m mlx_lm_server --model mlx-community/Qwen3-4B-4bit --port 8000

# Single-node, 1 rank (tests distributed code path without JACCL)
python -m mlx_lm_server.sharded_main --model mlx-community/Qwen3-4B-4bit --port 8000
```

### Production (2-node, JACCL)

```bash
mlx.launch \
    --backend jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --hostfile hosts.json \
    -m mlx_lm_server.sharded_main \
    --model kimi-k25-4bit \
    --port 8000 \
    --max-batch-size 4
```

### Testing (2-node)

```bash
mlx.launch \
    --backend jaccl \
    --hostfile hosts.json \
    scripts/test_distributed_2node.py \
    --model mlx-community/Qwen3-4B-4bit
```
