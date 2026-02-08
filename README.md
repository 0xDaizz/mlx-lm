# mlx-lm-server

Production-grade LLM serving for Apple Silicon, built on [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Overview

mlx-lm-server is a fork of Apple's mlx-lm that adds vLLM-level serving capabilities to the MLX ecosystem. It provides continuous batching, hash-based automatic prefix caching, a multi-tier KV cache (RAM + SSD), and an OpenAI-compatible API — all optimized for Apple Silicon.

The serving layer lives in a cleanly separable `mlx_lm_server/` package. The original mlx-lm code remains untouched.

## Features

- **Continuous Batching** — Iteration-level scheduling; finished requests are replaced immediately without waiting for the full batch to complete.
- **Hash-based Automatic Prefix Caching** — KV cache blocks are identified by BLAKE2b hash of `prefix_tokens + block_tokens`. Matching prefixes are reused across requests automatically with collision detection.
- **Multi-Level KV Cache** — Three-tier caching: block-level (hash table) → sequence-level (trie-based prefix search) → SSD (safetensors persistence). Evicted blocks are persisted to SSD with atomic writes for crash safety.
- **Min-Heap LRU Eviction** — O(log N) block eviction with fallback linear scan. Blocks with ref_count=0 are eligible for eviction; active blocks are protected.
- **INT8 KV Cache Quantization** — ~50% memory savings with negligible quality loss. Default: `--kv-bits 8 --kv-group-size 64`.
- **OpenAI-Compatible API** — Drop-in replacement for OpenAI endpoints. Supports `/v1/chat/completions`, `/v1/completions`, and streaming via SSE.
- **Production Hardening** — Block leak prevention, atomic SSD writes, streaming backpressure (bounded queues), request timeouts, graceful shutdown, hash collision handling.
- **Tensor-Parallel Ready** — Architecture supports distributed inference across multiple Mac nodes via JACCL RDMA over Thunderbolt 5 (Phase 8).

## Quick Start

### Install

```bash
git clone https://github.com/<your-username>/mlx-lm.git mlx-lm-server
cd mlx-lm-server
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install fastapi uvicorn
```

### Run the Server

```bash
python -m mlx_lm_server --model mlx-community/Qwen3-4B-4bit --port 8000
```

### Make a Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "max_tokens": 256,
    "stream": true
  }'
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-4bit",
    "prompt": "The capital of France is",
    "max_tokens": 64
  }'
```

## CLI Options

```
python -m mlx_lm_server [OPTIONS]
```

### Model

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/Qwen3-4B-4bit` | HuggingFace model ID or local path |
| `--adapter-path` | `None` | Path to LoRA adapter weights |

### Server

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |

### KV Cache

| Flag | Default | Description |
|------|---------|-------------|
| `--block-size` | `16` | Tokens per KV cache block |
| `--num-blocks` | `2048` | Total blocks in the KV cache pool |
| `--kv-bits` | `8` | KV cache quantization bits (8 = INT8) |
| `--kv-group-size` | `64` | Elements per quantization scale factor |
| `--max-kv-size` | `None` | Max KV cache size (None = unlimited) |
| `--sequence-cache-size` | `50` | Max entries in the sequence-level cache |

### SSD Tier

| Flag | Default | Description |
|------|---------|-------------|
| `--ssd-cache-dir` | `~/.cache/mlx-lm-server/kv-cache` | SSD cache directory |
| `--ssd-ttl-days` | `7` | Days before unused SSD blocks are pruned |
| `--no-ssd` | `false` | Disable SSD cache tier (RAM-only mode) |

### Batching & Scheduling

| Flag | Default | Description |
|------|---------|-------------|
| `--max-batch-size` | `8` | Max concurrent decode sequences |
| `--prefill-batch-size` | `4` | Max concurrent prefill sequences |
| `--prefill-step-size` | `2048` | Chunk size for prompt processing |
| `--completion-batch-size` | `32` | Max sequences in BatchGenerator |
| `--max-queue-size` | `128` | Max pending requests in queue |

### Request Limits

| Flag | Default | Description |
|------|---------|-------------|
| `--default-max-tokens` | `512` | Default max tokens if not specified in request |
| `--max-prompt-tokens` | `32768` | Maximum input prompt length |
| `--request-timeout-s` | `120.0` | Per-request timeout in seconds |

## API Endpoints

### POST `/v1/chat/completions`

Chat completion (OpenAI-compatible). Supports streaming via `"stream": true`.

**Request:**

```json
{
  "model": "mlx-community/Qwen3-4B-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 128,
  "temperature": 1.0,
  "top_p": 1.0,
  "stream": false,
  "stop": ["\n\n"]
}
```

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mlx-community/Qwen3-4B-4bit",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello! How can I help you?"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 7,
    "total_tokens": 22
  }
}
```

**Streaming** (`"stream": true`): Returns Server-Sent Events (SSE) with `text/event-stream` content type. Each token is sent as `data: {chunk}\n\n`. The stream terminates with `data: [DONE]\n\n`.

### POST `/v1/completions`

Text completion (OpenAI-compatible). Supports streaming.

**Request:**

```json
{
  "model": "mlx-community/Qwen3-4B-4bit",
  "prompt": "The meaning of life is",
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": [".", "\n"]
}
```

**Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "mlx-community/Qwen3-4B-4bit",
  "choices": [
    {
      "index": 0,
      "text": " a question that has been pondered by philosophers",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 9,
    "total_tokens": 14
  }
}
```

### GET `/v1/models`

List available models.

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen3-4B-4bit",
      "object": "model",
      "created": 0,
      "owned_by": "mlx-lm-server"
    }
  ]
}
```

### GET `/health`

Health check with cache statistics.

```json
{
  "status": "ok",
  "cache_stats": {
    "active_sequences": 2,
    "queued_requests": 0,
    "total_blocks": 2048,
    "used_blocks": 128,
    "free_blocks": 1920,
    "cached_blocks": 64,
    "cache_hits_block": 150,
    "cache_hits_sequence": 12,
    "cache_misses": 30,
    "cache_hit_rate": 0.84,
    "cache_effectiveness": 0.72,
    "tokens_generated": 5000,
    "total_prefill_tokens": 8000,
    "total_cached_tokens": 20000,
    "requests_completed": 100,
    "requests_errored": 1,
    "total_requests": 103
  }
}
```

### Error Responses

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Prompt exceeds maximum length of 32768 tokens",
    "type": "invalid_request_error",
    "code": "400"
  }
}
```

| HTTP Code | Condition |
|-----------|-----------|
| 400 | Invalid request (empty prompt, model mismatch, prompt too long) |
| 429 | Request queue full |
| 503 | Server shutting down |

## Architecture

```
+---------------------------------------------------+
|            FastAPI Server (OpenAI API)             |
|         /v1/chat/completions, /v1/completions      |
+---------------------------------------------------+
|                  Scheduler                         |
|    Continuous Batching + Request Queue Manager     |
+---------------------------------------------------+
|               Multi-Level Cache                    |
|                                                    |
|  Sequence Cache ──► Block Cache ──► SSD Tier       |
|  (Trie O(M))       (Hash O(1))    (safetensors)   |
+---------------------------------------------------+
|              mlx-lm Engine                         |
|   BatchGenerator + KV Cache (INT8 quantized)       |
+---------------------------------------------------+
|           MLX / Metal (Apple Silicon)              |
+---------------------------------------------------+
```

### Multi-Level Caching

The cache system has three tiers, checked in order:

1. **Sequence Cache** (SequenceCacheStore) — Trie-based prefix lookup. Finds the longest cached token prefix in O(M) time (M = query length). Stores full sequence-level KV states with LRU eviction. Best for repeated multi-turn conversations.

2. **Block Cache** (KVCacheManager) — Hash-based block lookup. Each block holds `block_size` tokens (default: 16). Blocks are identified by BLAKE2b hash of `prefix_tokens + block_tokens`, enabling automatic prefix sharing across different requests. Hash collisions are detected via token verification and handled safely.

3. **SSD Cache** (SSDCache) — Persistent storage for evicted blocks. Blocks are saved as individual `.safetensors` files with atomic writes (tempfile + `os.replace`) for crash safety. Blocks unused for `ssd_ttl_days` (default: 7) are auto-pruned.

### How Prefix Caching Works

1. Each KV cache block holds a fixed number of tokens (default: 16).
2. A block's identity is `BLAKE2b(prefix_tokens + block_tokens)` — deterministic across processes.
3. When a new request arrives, the scheduler walks the prompt in block-sized chunks, checking the hash table for hits.
4. Cached blocks are reused (ref_count incremented); only uncached suffix tokens need prefill.
5. When memory pressure occurs, min-heap LRU eviction frees blocks with `ref_count == 0`. If SSD is enabled, evicted blocks are persisted to disk first.
6. Hash collisions are detected by comparing stored `token_ids` — collision blocks are allocated but never registered in the hash table.

### Continuous Batching

The scheduler implements iteration-level scheduling:

- **Prefill phase**: New requests have their prompts processed in chunks of `prefill_step_size` tokens.
- **Decode phase**: Active sequences generate one token per iteration.
- **Replacement**: When a sequence finishes (EOS, max_tokens, or stop sequence), its slot is immediately filled with a queued request.
- **Dual-path inference**: Real model path via `BatchGenerator` for production; mock path (`model=None`) for testing.

### Production Hardening

- **Block leak prevention**: `free_blocks()` in exception handlers ensures blocks are returned to pool on errors.
- **Atomic SSD writes**: Temp file + `os.replace()` prevents corruption on crash.
- **Streaming backpressure**: Token queues bounded at 256 entries to prevent OOM with slow clients.
- **Request timeouts**: Per-token timeout on streaming queues; timed-out requests are auto-cancelled.
- **Graceful shutdown**: Server returns 503 during shutdown; in-flight requests are drained.
- **Collision safety**: Hash collisions detected and handled — collision blocks allocated but not cached.
- **SSD failure tolerance**: Save failures skip eviction, keeping the block in RAM.
- **Per-block error recovery**: `prune_expired()` handles individual block errors without aborting.

## Configuration

Key configuration parameters in `ServerConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | `16` | Tokens per KV block. Larger = fewer hash lookups, coarser caching. |
| `num_blocks` | `2048` | Total pool size. Memory ≈ `num_blocks × block_size × kv_size_per_token`. |
| `kv_bits` | `8` | Quantization precision. 8 = INT8 (~50% memory savings). |
| `kv_group_size` | `64` | Elements per quantization scale. Lower = more precise, slightly more memory. |
| `sequence_cache_size` | `50` | Max cached sequence-level KV states (trie entries). |
| `ssd_ttl_days` | `7` | SSD blocks unused for this many days are auto-pruned. |
| `max_batch_size` | `8` | Maximum concurrent decode sequences in a batch. |
| `prefill_batch_size` | `4` | Maximum sequences being prefilled concurrently. |
| `prefill_step_size` | `2048` | Token chunk size for prompt processing. |
| `max_queue_size` | `128` | Requests beyond this limit are rejected with 429. |
| `max_prompt_tokens` | `32768` | Maximum input prompt length (tokens). |
| `request_timeout_s` | `120.0` | Per-request timeout in seconds. |

## Development

### Project Structure

```
mlx_lm_server/              # Serving layer (new code)
  __init__.py
  types.py                   # InferenceRequest, SequenceState, TokenEvent, KVCacheBlock
  config.py                  # ServerConfig dataclass
  kv_cache_manager.py        # Hash-based block manager, prefix caching, tiered cache
  sequence_cache.py          # Trie-based sequence-level cache
  ssd_cache.py               # SSD tier (safetensors persistence)
  scheduler.py               # Continuous batching scheduler
  server.py                  # FastAPI server + OpenAI API
  __main__.py                # Entry point
mlx_lm/                      # Original mlx-lm (unmodified)
tests/                       # 363 tests across 18 test files
  conftest.py                # Shared fixtures (mock tokenizer, test config)
  test_kv_cache_manager.py   # Block pool, hash, prefix caching, eviction (68 tests)
  test_adversarial.py        # Devil's advocate: edge cases, races, leaks (106 tests)
  test_real_model.py         # Real model E2E tests with Qwen3-4B-4bit (48 tests)
  test_regression.py         # Regression tests for 6 production bugs (40 tests)
  test_scheduler.py          # Batching, queuing, streaming, error recovery (37 tests)
  test_server_app.py         # FastAPI endpoint tests (26 tests)
  test_stream_verification.py # SSE format, token ordering (20 tests)
  test_integration.py        # SSD round-trip, cache hit verification (17 tests)
  test_ssd_cache.py          # Safetensors save/load, TTL pruning (16 tests)
  test_block_bridge.py       # Block ↔ sequence cache translation (15 tests)
  test_server.py             # Basic server functionality (13 tests)
  test_sequence_cache.py     # Trie prefix matching, eviction (12 tests)
  test_batch_integration.py  # MockBatchGenerator continuous batching (7 tests)
  ...
scripts/
  benchmark.py               # Real-model benchmark (scheduler/HTTP/cache modes)
  benchmark_optimal.py       # Batch size & block size hyperparameter tuning
  benchmark_comparison.py    # A/B comparison: our server vs baseline mlx_lm.server
  benchmark_comprehensive.py # Parameter sweep + correctness validation
```

### Running Tests

```bash
# Full test suite (363 tests)
pytest tests/ -v --tb=short

# Skip slow tests (real model tests)
pytest tests/ -v -m "not slow"

# Individual components
pytest tests/test_kv_cache_manager.py -v
pytest tests/test_scheduler.py -v
pytest tests/test_server_app.py -v

# Integration & adversarial
pytest tests/test_integration.py -v
pytest tests/test_adversarial.py -v

# Real model tests (requires mlx-community/Qwen3-4B-4bit)
pytest tests/test_real_model.py -v
```

### Benchmarks

```bash
# Real-model benchmark (scheduler, HTTP, and cache modes)
python scripts/benchmark.py --mode all --repeat 3

# Hyperparameter tuning (batch size & block size sweep)
python scripts/benchmark_optimal.py --mode all

# A/B comparison with baseline mlx_lm.server
python scripts/benchmark_comparison.py --scenario all

# Parameter sweep + correctness validation
python scripts/benchmark_comprehensive.py --part all
```

### Target Hardware

- **Development:** MacBook with Apple Silicon (M-series)
- **Production:** Mac Studio M3 Ultra 512GB (single or multi-node)

## Distributed Inference (JACCL)

> **Status: Phase 8 — planned.** The architecture supports it; the code paths are ready to be wired up.

### What is JACCL?

JACCL is Apple's distributed machine learning backend for MLX. It provides RDMA (Remote Direct Memory Access) communication over Thunderbolt 5, enabling tensor-parallel inference across multiple Mac nodes with minimal latency.

### Hardware Requirements

- 2+ Mac Studio with M3 Ultra (or similar Apple Silicon with high unified memory)
- Thunderbolt 5 direct connection between nodes (for RDMA)
- Sufficient unified memory across nodes for the target model (e.g., 2× 512GB for large models)

### How It Works

Tensor parallelism splits model weights across nodes. Each node computes its shard of every layer, then nodes exchange intermediate results via RDMA:

- **Rank 0** runs the full serving stack (FastAPI, scheduler, KV cache manager).
- **All ranks** participate in the model forward pass via `mx.distributed`.
- The server starts within an `mlx.launch` distributed context, so the serving layer is transparent to the distributed engine.

### Future Usage

```bash
# Launch across 2 nodes
mlx.launch --n 2 python -m mlx_lm_server \
  --model kimi-k2.5-4bit \
  --port 8000
```

Only rank 0 exposes the HTTP server. Inference requests are automatically distributed.

## Acknowledgments

- [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple — the foundation this project builds on.
- [vLLM](https://github.com/vllm-project/vllm) — inspiration for prefix caching, continuous batching, and paged KV cache design.
- [MLX](https://github.com/ml-explore/mlx) — Apple's machine learning framework for Apple Silicon.

## License

MIT — see the upstream [mlx-lm license](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).
