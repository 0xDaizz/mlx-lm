# 13. Comprehensive Test & Benchmark Plan

## Guiding Principle

**Minimize server restarts.** Each restart costs ~108 seconds of load time and consumes ~620 GB of wired memory across two nodes. All tests are grouped by server configuration so we restart only 5 times total.

---

## Required Metrics (All Benchmarks)

**Every benchmark result MUST include all of the following:**

| Metric | Description |
|--------|-------------|
| `prefill_time_s` | Time to process prompt (first token latency) |
| `prefill_tok_s` | Prompt tokens / prefill time |
| `decode_time_s` | Time from first generated token to last |
| `decode_tok_s` | Generated tokens / decode time |
| `throughput_tok_s` | Total tokens (prompt + generated) / total time |

These 5 metrics are mandatory in all `bench_*.py` JSON output. Any benchmark missing them is considered incomplete.

---

## Server Configuration Groups (5 Restarts Total)

The Kimi K2.5 model (612 GB, 2-node JACCL TP) takes ~108 seconds to load. We group all tests by server configuration to minimize restarts.

---

### Phase 1: Baseline

**Config**: `launch_baseline.sh` -- no SSD, kv-bits=0, no spec decode

**Purpose**: Establish baselines and test core API.

**Tests (sequential)**:

1. **API comprehensive test** (`test_api_comprehensive.py`) -- streaming, multi-turn, temp=0, stop sequences, completions, large tokens
2. **`bench_single.py`** -- TTFT and tok/s baselines (non-streaming, streaming, 5 token lengths)
3. **`bench_batch.py`** -- **Batch inference throughput scaling** (see below)
4. **`bench_concurrent.py`** -- concurrency scaling (1, 3, 5, 8 simultaneous requests)
5. **`bench_cache.py`** -- KV cache hit/miss patterns, prefix sharing, multi-turn warmup
6. **`bench_memory.py --label baseline`** -- memory profile per prompt size

#### Batch Inference Test (`bench_batch.py`)

**Purpose**: Verify that batch inference works correctly and throughput scales meaningfully with batch size.

**Design**:
- **Batch sizes**: k = 1, 2, 3, 4, 5, 6, 7, 8
- **Prompts**: Sufficiently long prompts (~500-1000 tokens each) to make speed differences significant. Use news-article-length passages (e.g., multi-paragraph prompts about distinct topics).
- **Metric per batch size**: prefill_time_s, prefill_tok_s, decode_time_s, decode_tok_s, throughput_tok_s
- **Key questions**:
  - Does throughput (tok/s) increase with batch size?
  - At what batch size does throughput saturate?
  - Is there a latency vs throughput trade-off (per-request latency may increase)?
- **Method**: Send k identical-length requests simultaneously, measure aggregate throughput and per-request latency.
- **Output**: `/tmp/kimi-bench-results/bench_batch_*.json`

**Expected**: ~10 minutes (8 batch sizes × ~1 min each)

**Key metrics to record**:
- Baseline tok/s (non-streaming and streaming)
- TTFT at each token length
- **Batch throughput scaling curve** (k=1..8)
- p95 latency at concurrency=8
- Cache hit rate

---

### Phase 2: KV FP8 Quantization

**Config**: `launch_kv_fp8.sh` -- kv-bits=8, kv-group-size=64

**Purpose**: Measure memory savings vs speed trade-off.

**Tests (sequential)**:

1. **`bench_memory.py --label fp8`** -- memory profile (compare with baseline)
2. **`bench_single.py`** -- tok/s with FP8 KV (is there speed penalty?)
3. **`bench_concurrent.py`** -- does FP8 help concurrency (less memory per request)?
4. **`bench_cache.py`** -- cache efficiency with quantized blocks

**Expected**: ~10-15 minutes

**Key metrics**:
- Memory reduction % vs baseline
- tok/s delta vs baseline
- Max concurrent capacity

---

### Phase 3: N-gram Speculative Decoding

**Config**: `launch_ngram.sh` -- spec-decode=ngram, ngram-max=4, K=5

**Purpose**: Measure speedup across content types.

**Tests (sequential)**:

1. **`bench_ngram.py --label ngram4_k5`** -- 9 content categories (coding, structured, translation, math, creative, repetitive, counting, conversation, technical)
2. **`bench_single.py`** -- overall tok/s improvement
3. **Spec decode metrics check**: `curl /v1/spec_decode/metrics` -- acceptance rate, bonus tokens
4. **Parameter sweep** (manual, if time permits):
   - K values: 3, 5, 7, 10
   - ngram-max: 2, 3, 4, 6
   - Record acceptance_rate and effective tok/s for each combo

**Expected**: ~15-20 minutes (longer if parameter sweep)

**Key metrics**:
- Acceptance rate per content type
- Overall speedup %
- Optimal K and ngram-max

---

### Phase 4: SSD Cache

**Config**: `launch_ssd.sh` -- ssd-policy=write_through, ssd-async-writes

**Purpose**: Test SSD KV offloading and cache persistence.

**Tests (sequential)**:

1. **`bench_cache.py`** -- cold vs warm with SSD (does cache survive between requests?)
2. **`bench_single.py`** -- overhead of SSD writes
3. **Manual**: Send same prompt twice, check TTFT improvement on second request (SSD cache hit)
4. **Check SSD cache directory**: file count, total size, index integrity

**Expected**: ~10-15 minutes

**Key metrics**:
- SSD cache hit rate
- TTFT improvement on warm requests
- Write overhead

---

### Phase 5: All Features Combined

**Config**: `launch_all.sh` -- kv-bits=8, SSD, ngram spec decode

**Purpose**: Measure combined effect and find optimal production config.

**Tests (sequential)**:

1. **`bench_single.py`** -- combined tok/s
2. **`bench_concurrent.py`** -- combined concurrency performance
3. **`bench_ngram.py --label all_features`** -- spec decode with quantized KV + SSD
4. **`bench_memory.py --label all_features`** -- total memory profile
5. **`bench_cache.py`** -- full-stack cache behavior

**Expected**: ~15-20 minutes

**Key metrics**:
- Final production tok/s
- Combined memory efficiency
- Stability

---

## Each Phase Uses the Lifecycle Harness

```bash
# Example: Phase 1 with memory leak check
.venv/bin/python tests/e2e/test_memory_leak.py \
    --run-tests "tests/e2e/test_api_comprehensive.py" \
                "tests/e2e/benchmark/bench_single.py" \
                "tests/e2e/benchmark/bench_concurrent.py" \
                "tests/e2e/benchmark/bench_cache.py" \
                "tests/e2e/benchmark/bench_memory.py --label baseline"
```

For phases 2-5, we do not need the memory leak wrapper every time (tested in phase 1). Just use launch/stop scripts directly.

---

## Total Time Estimate

| Phase | Load Time | Test Time | Total |
|-------|-----------|-----------|-------|
| 1. Baseline | ~108s | ~30min | ~32min |
| 2. KV FP8 | ~108s | ~15min | ~17min |
| 3. N-gram | ~108s | ~20min | ~22min |
| 4. SSD | ~108s | ~15min | ~17min |
| 5. All Features | ~108s | ~20min | ~22min |
| **Total (Phase 1-5)** | **~9min** | **~100min** | **~110min** |
| 6. Draft Spec Decode | ~108s | ~20min | ~22min |
| 7. MTP Spec Decode | ~108s | ~20min | ~22min |
| **Total (Phase 1-7)** | **~13min** | **~140min** | **~154min** |

> Phase 6-7 are future work pending prerequisite models.

---

## Future Phases (Roadmap)

### Phase 6: Draft Model Speculative Decoding (planned)

**Prerequisite**: Acquire or quantize a compatible draft model for Kimi K2.5.
- Draft model should be a smaller MoE variant or distilled dense model (~1-7B params)
- Config: `--spec-decode draft --draft-model-path <path> --num-speculative-tokens 5 --draft-context-len 128`
- **Tests**: bench_single.py (speedup vs baseline), bench_ngram.py categories (acceptance rate comparison with n-gram), parameter sweep on K and draft_context_len
- **Key question**: Does a dedicated draft model achieve higher acceptance rates than n-gram on creative/non-repetitive content?

### Phase 7: MTP (Multi-Token Prediction) Speculative Decoding (planned)

**Prerequisite**: Verify Kimi K2.5 has MTP head support in its architecture. If not, test with a compatible model (e.g., DeepSeek-V3 or similar).
- Config: `--spec-decode mtp --num-speculative-tokens 5`
- **Tests**: bench_single.py (speedup), acceptance rate across content types, memory overhead of hidden-state extraction
- **Key question**: Does MTP outperform n-gram and draft model on general-purpose content? What is the memory overhead?

### Not Planned

- **Ring backend**: JACCL is the target for this 2-node setup; ring (TCP) would be slower.
- **Pipeline sharding**: Not implemented in v1.

---

## Parameter Optimization (Phase 3 Detailed)

For N-gram spec decoding, the key parameters to optimize are:

### K (num_speculative_tokens)

Number of draft tokens per step. Higher K = more potential speedup but also more wasted computation on rejections.

- Start with K=5 (default), try 3, 7, 10
- Optimal K depends on acceptance rate: high acceptance -> use higher K

### ngram-max

Maximum n-gram length to search. Higher = better matches but slower lookup.

- Start with 4 (default), try 2, 3, 6
- For Kimi K2.5 (code/technical), expect 4-6 to be optimal

### acceptance_threshold

Below this rate, spec decode auto-disables.

- Default 0.3 is conservative; try 0.2 for more aggressive speculative behavior

### adaptive_k

Automatically adjusts K based on acceptance rate EMA.

- Keep enabled for production; disable for benchmarking to isolate K effects

---

## Results Storage

All results go to `/tmp/kimi-bench-results/` with timestamps:

```
bench_single_baseline_20260214_120000.json
bench_single_fp8_20260214_121500.json
bench_concurrent_baseline_20260214_122000.json
...
```

After all phases, copy results for permanent storage:

```bash
cp -r /tmp/kimi-bench-results/ /Users/hw/mlx-lm-server/benchmark-results/$(date +%Y%m%d)/
```
