# Kimi K2.5 Benchmark Test Plan

> 2-Node JACCL TP (hwStudio1 + hwStudio2) benchmark suite for mlx-lm-server.

---

## Environment

| Item | Value |
|------|-------|
| Model | Kimi K2.5 (612 GB, 4-bit quantized, 1.03T params) |
| Nodes | 2x Mac Studio (512 GB RAM each) |
| Connection | JACCL over TB5 RDMA (80 Gb/s x 2) |
| Server | mlx-lm-server at http://localhost:8080 (rank 0 only) |
| Python | `/Users/hw/mlx-lm-server/.venv/bin/python` |
| Results | `/tmp/kimi-bench-results/` |

---

## Phase 1: Baseline (No Extra Features)

**Script:** `bench_single.py`

**Goal:** Establish baseline throughput with default server config (no spec decode, no SSD write-through, default KV bits).

**Tests:**
1. Non-streaming chat completion (short prompt) -- measure total latency
2. Streaming chat completion (short prompt) -- measure TTFT, tok/s, total time
3. Non-streaming text completion -- measure total latency
4. Streaming text completion -- measure TTFT, tok/s
5. Varying max_tokens (64, 128, 256, 512, 1024) -- measure throughput scaling
6. Long prompt (2k tokens) -- measure prefill time vs short prompt

**Metrics:**
- TTFT (Time to First Token) in seconds
- Generation tok/s (completion_tokens / generation_time)
- Total wall time
- Prompt tokens / completion tokens from usage response

**Expected baseline (from instruction doc):**
- Prompt TPS: ~49.6 tok/s
- Generation TPS: ~23.9 tok/s

---

## Phase 2: KV Cache FP8

**Script:** `bench_memory.py`

**Goal:** Measure memory usage difference between baseline and FP8 KV cache (`--kv-bits 8`).

**Tests:**
1. Health endpoint memory stats (baseline server config)
2. Same prompts as Phase 1 -- compare throughput
3. `/health` endpoint cache utilization after load

**Metrics:**
- Memory stats from `/health` endpoint (used_blocks, free_blocks, utilization)
- Throughput comparison (tok/s) vs baseline
- Cache hit rate

---

## Phase 3: SSD Cache

**Script:** `bench_cache.py`

**Goal:** Test prefix cache hit/miss behavior and SSD write-through cache.

**Tests:**
1. **Cold start**: First request with a unique prompt -- should be cache miss
2. **Warm hit**: Repeat same prompt -- should be prefix cache hit
3. **Partial hit**: Send prompt with shared prefix + different suffix
4. **SSD write-through**: Enable `--ssd-policy write_through`, verify SSD writes via health stats
5. **Cache eviction**: Fill cache, verify eviction and SSD recovery
6. **Multi-turn**: Simulate conversation with accumulating context

**Metrics:**
- Cache hit rate from `/health` endpoint
- TTFT difference between cold and warm requests
- SSD cache stats (if available via health endpoint)

---

## Phase 4: N-gram Speculative Decode

**Script:** `bench_ngram.py`

**Goal:** Measure speculative decoding improvement across diverse prompt categories and parameter combinations.

**Prompt Categories:**
1. **Coding** -- Python function generation (high n-gram repetition expected)
2. **Structured output** -- JSON/YAML generation
3. **Daily conversation** -- casual chat
4. **Technical documentation** -- scientific/technical writing
5. **Translation** -- Korean to English
6. **Math/Logic** -- step-by-step reasoning
7. **Creative writing** -- story generation
8. **Repetitive patterns** -- list generation, counting
9. **Counting** -- enumeration tasks

**Parameter Grid:**
- `--ngram-max`: 2, 3, 4, 5
- `--num-speculative-tokens`: 3, 5, 7

**Note:** Each parameter combination requires a server restart. The benchmark reports which combinations require restarts.

**Metrics:**
- tok/s per category per parameter combination
- Acceptance rate from `/v1/spec_decode/metrics`
- TTFT overhead from speculation
- Speedup ratio vs baseline (Phase 1)

---

## Phase 5: Concurrent Load

**Script:** `bench_concurrent.py`

**Goal:** Measure scheduler behavior under concurrent request load.

**Concurrency Levels:** 1, 3, 5, 8

**Tests:**
1. Same prompt, N concurrent streaming requests
2. Mixed prompts (short + long), N concurrent
3. Measure 429 rate limit responses at high concurrency
4. Health endpoint under load (utilization, queue depth)

**Metrics:**
- Per-request TTFT, tok/s, total time
- Aggregate throughput (total tok/s across all concurrent requests)
- P50 / P95 / P99 latency
- Error rate (429s, 503s, timeouts)
- Queue depth from `/health`

---

## Phase 6: Combined Features

**Script:** Run `bench_single.py` against a server with all features enabled.

**Server Config:**
- `--kv-bits 8`
- `--ssd-policy write_through`
- `--spec-decode ngram --ngram-max 4 --num-speculative-tokens 5`

**Goal:** Measure cumulative improvement of all features together.

---

## Execution Order

The `run_all.sh` script runs phases in order to minimize server restarts:

1. Phase 1: Baseline (`bench_single.py`) -- default server config
2. Phase 2: Memory (`bench_memory.py`) -- same server, reads health stats
3. Phase 3: Cache (`bench_cache.py`) -- same server (may need SSD policy change)
4. Phase 4: N-gram (`bench_ngram.py`) -- multiple server restarts for param grid
5. Phase 5: Concurrent (`bench_concurrent.py`) -- server with best ngram config
6. Phase 6: Combined (`bench_single.py` again with all features)

---

## Output Format

All scripts output JSON to `/tmp/kimi-bench-results/<script_name>_<timestamp>.json`:

```json
{
  "benchmark": "bench_single",
  "timestamp": "2026-02-13T12:00:00",
  "server_url": "http://localhost:8080",
  "results": [
    {
      "test_name": "streaming_chat_short",
      "prompt": "...",
      "max_tokens": 256,
      "ttft_s": 0.45,
      "generation_tok_s": 23.9,
      "total_time_s": 11.2,
      "prompt_tokens": 42,
      "completion_tokens": 256,
      "error": null
    }
  ]
}
```

---

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| Baseline gen tok/s | >= 20 tok/s |
| TTFT (short prompt) | < 5s |
| Spec decode speedup | >= 1.2x over baseline |
| Cache warm TTFT | < 50% of cold TTFT |
| Concurrent (3 req) | No errors, all complete |
| Memory (health) | utilization < 0.9 |
