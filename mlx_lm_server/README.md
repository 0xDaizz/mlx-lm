# mlx-lm-server

Production-grade LLM serving for Apple Silicon with continuous batching, prefix caching, and speculative decoding.

Built on top of Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm), mlx-lm-server adds vLLM-level serving features to the MLX ecosystem:

- **Continuous batching** with iteration-level scheduling
- **Hash-based automatic prefix caching** (vLLM-style) with LRU eviction
- **Tiered KV cache** (RAM to SSD) for massive context reuse
- **INT8 KV cache** for 50% memory savings at near-zero quality loss
- **Speculative decoding** (N-gram and Draft Model modes)
- **Tensor parallel** distributed inference via MLX ring or JACCL RDMA
- **OpenAI-compatible API** -- drop-in replacement for existing tooling
- **Production hardening** -- auth, rate limiting, memory pressure admission, health probes, Prometheus metrics

---

## Features

### Continuous Batching

Iteration-level scheduling where finished requests are replaced immediately, maximizing GPU utilization. The scheduler manages prefill and decode phases independently, with configurable batch sizes for each.

### Hash-based Automatic Prefix Caching

Each KV cache block (default: 16 tokens) is identified by `hash(prefix_tokens + block_tokens)`. A global hash table maps block hashes to physical blocks with LRU eviction. Requests sharing common prefixes (e.g., system prompts) automatically reuse cached KV blocks without any configuration.

### Tiered KV Cache (RAM to SSD)

Evicted blocks are persisted to SSD as safetensors files, keyed by block hash. On cache miss, blocks are reloaded from SSD (~7.4 GB/s on M3 Ultra) rather than recomputed from scratch. Supports both `evict_only` and `write_through` policies, with configurable TTL-based pruning and disk size limits.

### INT8 KV Cache

Quantized KV cache with `--kv-bits 8 --kv-group-size 64` by default. Provides approximately 50% memory savings with negligible quality degradation, enabling roughly 2x concurrent sessions on the same hardware.

### Speculative Decoding

Two speculation modes to accelerate generation:
- **N-gram** -- uses prompt context to predict next tokens via n-gram matching; no extra model needed
- **Draft Model** -- uses a smaller model to propose tokens, verified by the target model

Both modes support dynamic k (adaptive speculation length) and temperature-aware verification (greedy at temp=0, threshold-based at temp>0).

### Tensor Parallel

Multi-node distributed inference using MLX's built-in `mx.distributed` primitives. Supports the `ring` backend (single-machine or multi-machine via hostfile) and `jaccl` backend (RDMA over Thunderbolt 5). Only rank 0 serves HTTP; all ranks participate in inference.

### Production Features

- **API key authentication** via `--api-key` or `--api-key-file`
- **Concurrency limiting** with HTTP 429 when exceeded
- **Memory pressure admission control** returning HTTP 503
- **Load-aware health endpoint** (`ok` / `degraded` / `overloaded` / `shutting_down`)
- **Kubernetes probes** (`/livez`, `/readyz`)
- **Prometheus-format metrics** at `/metrics`
- **Request body size limits** to prevent DoS
- **Graceful shutdown** with in-flight request draining

---

## Architecture

```
+--------------------------------------------------+
|          FastAPI Server (OpenAI API)              |
|   /v1/chat/completions, /v1/completions          |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|            Scheduler                             |
|   Continuous Batching + Request Queue Manager    |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|          KV Cache Manager (Hash-based)           |
|   Automatic Prefix Caching + LRU Eviction        |
+--------------------------------------------------+
             |                    ^
             | evict              | reload
             v                    |
+--------------------------------------------------+
|          SSD Cache Tier (safetensors)             |
|   TTL Pruning + Disk Size Limits                 |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|          mlx-lm Engine (BatchGenerator)          |
|   Speculative Decoding (N-gram / Draft Model)    |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
|          MLX / Metal (Apple Silicon)             |
|   mx.distributed (JACCL RDMA Tensor Parallel)   |
+--------------------------------------------------+
```

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
pip install fastapi uvicorn
```

### Run

```bash
python -m mlx_lm_server --model mlx-community/Qwen3-4B-4bit
```

### Test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about MLX."}],
    "max_tokens": 100,
    "stream": true
  }'
```

---

## Configuration Reference

All configuration is done via CLI flags. Defaults are shown below.

### Model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `mlx-community/Qwen3-4B-4bit` | HuggingFace model ID or local path |
| `--adapter-path` | str | None | LoRA adapter weights path |

### Server

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--host` | str | `0.0.0.0` | Bind address |
| `--port` | int | `8000` | Bind port |
| `--api-key` | str | `$MLX_LM_SERVER_API_KEY` | Bearer token for auth |
| `--api-key-file` | str | `$MLX_LM_SERVER_API_KEY_FILE` | File containing API key |
| `--max-request-bytes` | int | `1048576` | Max request body size (bytes) |

### KV Cache

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--block-size` | int | `16` | Tokens per KV cache block |
| `--num-blocks` | int | `2048` | Total blocks in block pool |
| `--kv-bits` | int | `8` | KV cache quantization (8 = INT8) |
| `--kv-group-size` | int | `64` | Elements per quantization scale |
| `--max-kv-size` | int | None | Max total KV cache size |
| `--sequence-cache-size` | int | `50` | Max cached sequence states |

### SSD Cache

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ssd-cache-dir` | str | `~/.cache/mlx-lm-server/kv-cache` | SSD cache directory |
| `--no-ssd` | flag | False | Disable SSD tier |
| `--ssd-ttl-days` | int | `7` | Block TTL before pruning |
| `--ssd-policy` | str | `evict_only` | `evict_only` or `write_through` |
| `--ssd-durability` | str | `best_effort` | `best_effort` or `persistent` |
| `--ssd-async-writes` / `--no-ssd-async-writes` | bool | True | Async SSD writer thread |
| `--ssd-writer-queue-size` | int | `512` | Async writer queue depth |
| `--ssd-persistent-max-retries` | int | `3` | Max retries for persistent writes |
| `--ssd-flush-interval-s` | float | `1.0` | Writer flush interval (seconds) |
| `--ssd-max-size-gb` | float | `50.0` | Max SSD cache size in GB (0 = unlimited) |

### Batching

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-batch-size` | int | `8` | Max concurrent decode sequences |
| `--prefill-batch-size` | int | `4` | Max concurrent prefill sequences |
| `--max-queue-size` | int | `128` | Max pending requests in queue |
| `--completion-batch-size` | int | `32` | Max sequences to BatchGenerator |

### Generation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--default-max-tokens` | int | `512` | Default max_tokens |
| `--max-prompt-tokens` | int | `32768` | Max prompt length (tokens) |
| `--max-generation-tokens` | int | `32768` | Upper bound for max_tokens |
| `--request-timeout-s` | float | `120.0` | Request timeout (seconds) |
| `--first-token-timeout-s` | float | `300.0` | First token timeout (seconds) |

### Admission Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-concurrent-requests` | int | `64` | Max concurrent requests (0 = unlimited, 429 when exceeded) |
| `--memory-pressure-threshold` | float | `0.9` | Block utilization threshold for 503 rejection |

### Speculative Decoding

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--spec-decode` | str | `none` | Mode: `none`, `ngram`, `draft` |
| `--num-speculative-tokens` | int | `5` | Draft tokens per step (k) |
| `--spec-decode-disable-batch-size` | int | `8` | Disable spec decode above this batch size |
| `--ngram-max` | int | `4` | N-gram max size |
| `--ngram-min` | int | `1` | N-gram min size |
| `--no-ngram-prompt-lookup` | flag | True | Disable prompt n-gram lookup |
| `--draft-model-path` | str | None | Path to draft model |
| `--draft-model-quantize` | str | None | (Deprecated) Use pre-quantized model instead |
| `--draft-context-len` | int | `128` | Draft prefill context length (max 512) |
| `--no-spec-decode-dynamic` | flag | True | Disable dynamic k |
| `--spec-decode-acceptance-threshold` | float | `0.3` | Acceptance threshold |
| `--no-adaptive-k` | flag | True | Disable adaptive k |

### Distributed / Tensor Parallel

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--distributed-mode` | str | `off` | `off`, `ring`, or `jaccl` |
| `--distributed-sharding` | str | `tensor` | Sharding strategy |
| `--distributed-strict` / `--no-distributed-strict` | bool | True | Strict validation |
| `--distributed-hostfile` | str | `$MLX_HOSTFILE` | Hostfile for ring backend |
| `--distributed-ibv-devices` | str | `$MLX_IBV_DEVICES` | IBV devices for jaccl |
| `--distributed-jaccl-coordinator` | str | `$MLX_JACCL_COORDINATOR` | JACCL coordinator address |
| `--num-local-ranks` | int | None | Local ranks for single-machine TP |

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming / non-streaming) |
| POST | `/v1/completions` | Text completions (streaming / non-streaming) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Load-aware health status |
| GET | `/livez` | Liveness probe |
| GET | `/readyz` | Readiness probe |
| GET | `/metrics` | Prometheus-format metrics |
| GET | `/v1/spec_decode/metrics` | Speculative decoding metrics |

### Chat Completions (non-streaming)

**Request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is MLX?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1707000000,
  "model": "mlx-community/Qwen3-4B-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "MLX is Apple's machine learning framework..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 42,
    "total_tokens": 66
  }
}
```

### Chat Completions (streaming)

**Request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Count to 5."}],
    "max_tokens": 50,
    "stream": true
  }'
```

**Response (Server-Sent Events):**

```
data: {"id":"chatcmpl-a1b2c3d4e5f6","object":"chat.completion.chunk","created":1707000000,"model":"mlx-community/Qwen3-4B-4bit","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4e5f6","object":"chat.completion.chunk","created":1707000000,"model":"mlx-community/Qwen3-4B-4bit","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

...

data: [DONE]
```

### Text Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0.0
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

Returns one of four states:
- `ok` -- utilization below 72% of threshold (200)
- `degraded` -- utilization between 72%-100% of threshold (200)
- `overloaded` -- utilization at or above threshold (503)
- `shutting_down` -- server is shutting down (503)

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Exposes gauges for `active_sequences`, `queued_requests`, `used_blocks`, `free_blocks`, `cache_hit_rate`, `dist_fatal`, and `shutdown_clean`.

---

## Speculative Decoding

Speculative decoding generates multiple draft tokens per step, then verifies them against the target model in a single forward pass. When drafts are accepted, this produces multiple tokens per iteration, improving throughput.

### N-gram Mode

Uses the existing prompt context to predict next tokens via n-gram matching. No extra model required. Performs well on repetitive or structured text (code, templates, formatted output).

```bash
python -m mlx_lm_server \
  --model mlx-community/Qwen3-4B-4bit \
  --spec-decode ngram \
  --num-speculative-tokens 5
```

### Draft Model Mode

Uses a smaller model (e.g., 0.6B parameters) to propose tokens, which are then verified by the target model. Better for general-purpose text generation.

```bash
python -m mlx_lm_server \
  --model mlx-community/Qwen3-4B-4bit \
  --spec-decode draft \
  --draft-model-path mlx-community/Qwen3-0.6B-4bit \
  --num-speculative-tokens 5
```

The draft and target models must share the same vocabulary. A warning is logged if EOS token IDs differ.

### Verification Behavior

- **temp=0 (greedy):** Lossless verification. Draft tokens are accepted if and only if they match the target model's argmax output.
- **temp>0 (sampling):** Threshold-based acceptance controlled by `--spec-decode-acceptance-threshold`.

### Dynamic k

When enabled (default), the speculation length adapts based on acceptance rate using an exponential moving average. High acceptance rates increase k; low rates decrease it. Disable with `--no-spec-decode-dynamic`.

### Monitoring

```bash
curl http://localhost:8000/v1/spec_decode/metrics
```

Returns acceptance rate, average accepted tokens, mode distribution, and controller state.

---

## Distributed Inference

Tensor parallelism splits model weights across multiple devices. Each device computes its shard of every layer, with all-reduce operations to combine results.

### Single-Machine Tensor Parallel

For machines with multiple GPU dies (e.g., Mac Studio M3 Ultra):

```bash
python -m mlx_lm_server \
  --model mlx-community/Qwen3-4B-4bit \
  --distributed-mode ring \
  --num-local-ranks 2
```

The server auto-relaunches itself under `mlx.launch` with the correct number of ranks.

### Multi-Node Tensor Parallel (JACCL RDMA)

For multiple Macs connected via Thunderbolt 5 RDMA:

```bash
python -m mlx_lm_server \
  --model large-model-4bit \
  --distributed-mode jaccl \
  --distributed-ibv-devices /dev/infiniband/rdma_cm \
  --distributed-jaccl-coordinator 192.168.1.1:29500
```

### Design Notes

- Only rank 0 serves HTTP traffic; all other ranks participate solely in inference.
- HTTP handlers enqueue events to an outbox; the inference thread drains and publishes them via collective operations. This avoids collective deadlocks from cross-thread `all_sum` calls.
- SSD cache directories are namespaced per rank (`rank_0/`, `rank_1/`, etc.) to avoid conflicts.
- LoRA adapters are not supported in distributed mode.

---

## Project Structure

```
mlx_lm_server/
├── __init__.py
├── __main__.py            # Entry point, distributed auto-relaunch
├── config.py              # ServerConfig dataclass
├── types.py               # Request/response types, TokenEvent
├── server.py              # FastAPI app, OpenAI API endpoints
├── scheduler.py           # Continuous batching engine
├── kv_cache_manager.py    # Block pool, prefix caching, LRU eviction
├── sequence_cache.py      # Per-sequence KV cache store (trie-based)
├── ssd_cache.py           # SSD tier (safetensors persistence)
├── ssd_writer.py          # Async SSD writer with backpressure
├── distributed.py         # TP context initialization
├── distributed_bus.py     # Cross-rank control bus (pickle + all_sum)
└── spec_decode/
    ├── __init__.py
    ├── config.py           # SpecDecodeConfig
    ├── engine.py           # 7-step speculative decoding loop
    ├── verifier.py         # Greedy + threshold verification
    ├── controller.py       # Dynamic k controller (EMA-based)
    ├── cache_utils.py      # Per-sequence KV cache trim utilities
    ├── rejection_sampler.py
    └── proposer/
        ├── __init__.py
        ├── base.py         # BaseProposer ABC, ProposalResult
        ├── ngram.py        # N-gram proposer (linear + suffix index)
        └── draft_model.py  # Draft model proposer (greedy argmax)
```

---

## Development

### Running Tests

```bash
# Full test suite
pytest tests/ -x -q

# Server and scheduler tests
pytest tests/test_server.py tests/test_scheduler.py -v

# KV cache tests
pytest tests/test_kv_cache_manager.py -v

# Speculative decoding tests
pytest tests/spec_decode/ -v

# Distributed infrastructure tests
pytest tests/test_distributed.py -v

# Integration tests (requires real model)
pytest tests/ -m integration -v
```

### Code Style

The project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting, and [mypy](https://mypy-lang.org/) for type checking.

```bash
ruff check mlx_lm_server/
mypy mlx_lm_server/
```

---

## License

MIT License. This project is a fork of [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm).
