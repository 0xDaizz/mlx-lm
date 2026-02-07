# mlx-lm-server

Production-grade LLM serving for Apple Silicon, built on [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Overview

mlx-lm-server is a fork of Apple's mlx-lm that adds vLLM-level serving capabilities to the MLX ecosystem. It provides continuous batching, hash-based automatic prefix caching, a tiered KV cache (RAM + SSD), and an OpenAI-compatible API -- all optimized for Apple Silicon.

The serving layer lives in a cleanly separable `mlx_lm_server/` package. The original mlx-lm code is untouched.

## Features

- **Continuous Batching** -- Iteration-level scheduling; finished requests are replaced immediately without waiting for the full batch to complete.
- **Hash-based Automatic Prefix Caching** -- KV cache blocks are identified by `hash(prefix_tokens + block_tokens)`. Matching prefixes are reused across requests automatically.
- **Tiered KV Cache (RAM + SSD)** -- Evicted blocks are persisted to SSD as safetensors files. SSD reads (~7.4 GB/s on M3 Ultra) are far faster than re-computing prefill.
- **INT8 KV Cache Quantization** -- ~50% memory savings with negligible quality loss. Default: `--kv-bits 8 --kv-group-size 64`.
- **OpenAI-compatible API** -- Drop-in replacement for OpenAI endpoints. Supports `/v1/chat/completions`, `/v1/completions`, and streaming via SSE.
- **Tensor-Parallel Ready** -- Architecture supports distributed inference across multiple Mac nodes via JACCL RDMA over Thunderbolt 5 (Phase 6, not yet implemented).

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

## CLI Usage

```
python -m mlx_lm_server [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/Qwen3-4B-4bit` | HuggingFace model ID or local path |
| `--adapter-path` | `None` | Path to LoRA adapter weights |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--block-size` | `16` | Tokens per KV cache block |
| `--num-blocks` | `2048` | Total blocks in the KV cache pool |
| `--kv-bits` | `8` | KV cache quantization bits (8 = INT8) |
| `--kv-group-size` | `64` | Elements per quantization scale factor |
| `--ssd-cache-dir` | `~/.cache/mlx-lm-server/kv-cache` | SSD cache directory |
| `--ssd-ttl-days` | `7` | Days before unused SSD blocks are pruned |
| `--no-ssd` | `false` | Disable SSD cache tier |
| `--max-batch-size` | `8` | Max concurrent decode sequences |
| `--prefill-batch-size` | `4` | Max concurrent prefill sequences |
| `--max-queue-size` | `128` | Max pending requests in queue |
| `--default-max-tokens` | `512` | Default max tokens if not specified in request |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible). Supports streaming via `"stream": true`. |
| `POST` | `/v1/completions` | Text completion (OpenAI-compatible). Supports streaming via `"stream": true`. |
| `GET` | `/v1/models` | List available models. |
| `GET` | `/health` | Health check with cache statistics. |

All endpoints return OpenAI-compatible JSON. Errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": "400"
  }
}
```

## Architecture

```
+---------------------------------------------------+
|            FastAPI Server (OpenAI API)             |
|         /v1/chat/completions, /v1/completions      |
+---------------------------------------------------+
|                  Scheduler                         |
|    Continuous Batching + Request Queue Manager     |
+---------------------------------------------------+
|          KV Cache Manager (Hash-based)             |
|   Automatic Prefix Caching (vLLM-style)            |
|   Block Pool + LRU Eviction + SSD Tier             |
+---------------------------------------------------+
|              mlx-lm Engine                         |
|   BatchGenerator + KV Cache (INT8 quantized)       |
|   mx.distributed (JACCL RDMA tensor parallel)      |
+---------------------------------------------------+
|           MLX / Metal (Apple Silicon)              |
+---------------------------------------------------+
```

### How Prefix Caching Works

1. Each KV cache block holds a fixed number of tokens (default: 16).
2. A block's identity is `hash(all_preceding_tokens + block_tokens)`.
3. When a new request arrives, the scheduler walks the prompt token-by-token in block-sized chunks, checking the hash table for hits.
4. Cached blocks are reused (ref_count incremented); only uncached suffix tokens need prefill.
5. When memory pressure occurs, LRU eviction frees blocks with `ref_count == 0`. If SSD is enabled, evicted blocks are persisted to disk first.

## Configuration

Key configuration options in `ServerConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | `16` | Tokens per KV block. Larger blocks = fewer hash lookups, coarser caching. |
| `num_blocks` | `2048` | Total pool size. Memory usage = `num_blocks * block_size * kv_size_per_token`. |
| `kv_bits` | `8` | Quantization precision. 8 = INT8 (~50% memory savings). |
| `kv_group_size` | `64` | Elements per quantization scale. Lower = more precise, slightly more memory. |
| `ssd_ttl_days` | `7` | SSD blocks unused for this many days are auto-pruned. |
| `max_batch_size` | `8` | Maximum concurrent decode sequences in a batch. |
| `prefill_batch_size` | `4` | Maximum sequences being prefilled concurrently. |
| `max_queue_size` | `128` | Requests beyond this limit are rejected with an error. |

## Development

### Project Structure

```
mlx_lm_server/          # Serving layer (new code)
  __init__.py
  types.py               # Shared data types
  config.py              # ServerConfig dataclass
  kv_cache_manager.py    # Hash-based block manager + prefix caching
  ssd_cache.py           # SSD tier (safetensors persistence)
  scheduler.py           # Continuous batching scheduler
  server.py              # FastAPI server + OpenAI API
  __main__.py            # Entry point
mlx_lm/                  # Original mlx-lm (unmodified)
tests/
  test_kv_cache_manager.py
  test_ssd_cache.py
  test_scheduler.py
  test_server.py
  test_integration.py
  test_adversarial.py
scripts/
  benchmark.py
```

### Running Tests

```bash
# Full test suite
pytest tests/ -v --tb=short

# Individual component
pytest tests/test_kv_cache_manager.py -v
pytest tests/test_scheduler.py -v
pytest tests/test_server.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Target Hardware

- **Development:** MacBook with Apple Silicon (M-series)
- **Production:** Mac Studio M3 Ultra 512GB (single or multi-node)

## Distributed Inference (JACCL)

> **Status: Phase 6 -- not yet implemented.** The architecture supports it; the code paths are ready to be wired up.

### What is JACCL?

JACCL is Apple's distributed machine learning backend for MLX. It provides RDMA (Remote Direct Memory Access) communication over Thunderbolt 5, enabling tensor-parallel inference across multiple Mac nodes with minimal latency.

### Hardware Requirements

- 2+ Mac Studio with M3 Ultra (or similar Apple Silicon with high unified memory)
- Thunderbolt 5 direct connection between nodes (for RDMA)
- Sufficient unified memory across nodes for the target model (e.g., 2x 512GB for Kimi K2.5 4-bit at ~120GB)

### How It Works

Tensor parallelism splits model weights across nodes. Each node computes its shard of every layer, then nodes exchange intermediate results via RDMA:

- **Rank 0** runs the full serving stack (FastAPI, scheduler, KV cache manager).
- **All ranks** participate in the model forward pass via `mx.distributed`.
- The server starts within an `mlx.launch` distributed context, so the serving layer is transparent to the distributed engine.

### Future Usage (when implemented)

```bash
# Launch across 2 nodes
mlx.launch --n 2 python -m mlx_lm_server \
  --model kimi-k2.5-4bit \
  --port 8000
```

Only rank 0 exposes the HTTP server. Inference requests are automatically distributed.

## Acknowledgments

- [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple -- the foundation this project builds on.
- [vLLM](https://github.com/vllm-project/vllm) -- inspiration for prefix caching, continuous batching, and paged KV cache design.
- [MLX](https://github.com/ml-explore/mlx) -- Apple's machine learning framework for Apple Silicon.

## License

MIT -- see the upstream [mlx-lm license](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).
