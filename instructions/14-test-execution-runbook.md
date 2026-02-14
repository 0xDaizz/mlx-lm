# 14. Test Execution Runbook

Step-by-step runbook for executing the comprehensive test and benchmark plan. Each step is copy-pasteable. Refer to `13-comprehensive-test-benchmark-plan.md` for rationale and expected metrics.

---

## Prerequisites

```bash
# 1. Post-reboot setup (both nodes)
sudo bash ~/mlx-lm-server/post_reboot_setup.sh
```

```bash
# 2. Rsync code to hwStudio2
rsync -avz --exclude='.git' --exclude='models/' --exclude='.venv/' --exclude='__pycache__' \
    /Users/hw/mlx-lm-server/ hwStudio2.local:/Users/hw/mlx-lm-server/
```

```bash
# 3. Verify TB interfaces
ifconfig en3 | grep inet   # expect 10.10.0.1
ifconfig en5 | grep inet   # expect 10.10.1.1
ssh hwStudio2.local "ifconfig en3 | grep inet"  # expect 10.10.0.2
```

```bash
# 4. Verify wired memory is clean
vm_stat | grep "Pages wired"  # should be ~1.5-2 GB
ssh hwStudio2.local "vm_stat | grep 'Pages wired'"  # same
```

```bash
# 5. Create results directory
mkdir -p /tmp/kimi-bench-results
```

```bash
# 5b. (Optional) Configure eval timeout for large models
# Default is 300s. Increase for very large models (>500GB):
export MLX_EVAL_TIMEOUT=600
# Or decrease for faster failure detection during debugging:
# export MLX_EVAL_TIMEOUT=120
```

---

## Phase 1: Baseline (Full Lifecycle + Memory Check)

```bash
# Launch, test, stop — all wrapped in memory leak harness
.venv/bin/python tests/e2e/test_memory_leak.py \
    --run-tests \
        "tests/e2e/test_api_comprehensive.py" \
        "tests/e2e/benchmark/bench_single.py" \
        "tests/e2e/benchmark/bench_batch.py" \
        "tests/e2e/benchmark/bench_concurrent.py" \
        "tests/e2e/benchmark/bench_cache.py" \
        "tests/e2e/benchmark/bench_memory.py --label baseline"
```

> **Note**: `bench_batch.py` tests batch k=1..8 with news-article-length prompts (~500-1000 tokens) to verify throughput scaling.

```bash
# Check results
ls -la /tmp/kimi-bench-results/
```

---

## Phase 2: KV FP8

```bash
# Start
bash tests/e2e/benchmark/configs/launch_kv_fp8.sh
```

```bash
# Run benchmarks (sequential)
.venv/bin/python tests/e2e/benchmark/bench_memory.py --label fp8
.venv/bin/python tests/e2e/benchmark/bench_single.py
.venv/bin/python tests/e2e/benchmark/bench_concurrent.py
.venv/bin/python tests/e2e/benchmark/bench_cache.py
```

```bash
# Stop
bash tests/e2e/benchmark/configs/stop_server.sh
```

---

## Phase 3: N-gram Spec Decode

```bash
# Start
bash tests/e2e/benchmark/configs/launch_ngram.sh
```

```bash
# Run benchmarks (sequential)
.venv/bin/python tests/e2e/benchmark/bench_ngram.py --label ngram4_k5
.venv/bin/python tests/e2e/benchmark/bench_single.py
curl -s http://localhost:8080/v1/spec_decode/metrics | python3 -m json.tool
```

```bash
# Stop
bash tests/e2e/benchmark/configs/stop_server.sh
```

---

## Phase 4: SSD Cache

```bash
# Start
bash tests/e2e/benchmark/configs/launch_ssd.sh
```

```bash
# Run benchmarks (sequential)
.venv/bin/python tests/e2e/benchmark/bench_cache.py
.venv/bin/python tests/e2e/benchmark/bench_single.py
```

```bash
# Check SSD cache state
ls -la ~/.cache/mlx-lm-server/kv-cache/
du -sh ~/.cache/mlx-lm-server/kv-cache/
```

```bash
# Stop
bash tests/e2e/benchmark/configs/stop_server.sh
```

---

## Phase 5: All Features

```bash
# Start
bash tests/e2e/benchmark/configs/launch_all.sh
```

```bash
# Run ALL benchmarks (sequential)
.venv/bin/python tests/e2e/benchmark/bench_single.py
.venv/bin/python tests/e2e/benchmark/bench_concurrent.py
.venv/bin/python tests/e2e/benchmark/bench_ngram.py --label all_features
.venv/bin/python tests/e2e/benchmark/bench_memory.py --label all_features
.venv/bin/python tests/e2e/benchmark/bench_cache.py
```

```bash
# Stop
bash tests/e2e/benchmark/configs/stop_server.sh
```

---

## Post-Run: Collect Results

```bash
# Create permanent results directory
RESULTS_DIR="/Users/hw/mlx-lm-server/benchmark-results/$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"
cp -r /tmp/kimi-bench-results/* "$RESULTS_DIR/"
```

```bash
# Verify wired memory is fully released
vm_stat | grep "Pages wired"
ssh hwStudio2.local "vm_stat | grep 'Pages wired'"
```

```bash
# Check EXIT AUDIT logs for clean shutdown
# Look for "EXIT AUDIT: rank=X, cause=Y" in server logs
# Expected: cause=uvicorn_shutdown (rank 0) or cause=worker_loop_exit (rank >0)
```

---

## Post-Run: Extract & Compare Results

After collecting results from all phases, run the extraction script to generate a cross-phase comparison report.

```bash
# Print comparison table to stdout
.venv/bin/python tests/e2e/benchmark/extract_results.py \
    --results-dir /tmp/kimi-bench-results
```

```bash
# Generate markdown report file
.venv/bin/python tests/e2e/benchmark/extract_results.py \
    --results-dir /tmp/kimi-bench-results \
    --output /tmp/kimi-bench-results/comparison_report.md
```

```bash
# Or from the permanent results directory after collection
RESULTS_DIR="/Users/hw/mlx-lm-server/benchmark-results/$(date +%Y%m%d)"
.venv/bin/python tests/e2e/benchmark/extract_results.py \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/comparison_report.md"
```

The report includes:
- **Cross-phase comparison table** with all 5 required metrics averaged per phase
- **Delta vs baseline** showing percentage change for each metric (e.g., "+15.2% decode_tok_s")
- **Per-benchmark detail** with averages broken down by benchmark within each phase

> For time metrics (Prefill Time, Decode Time), negative delta = improvement.
> For tok/s metrics (Prefill tok/s, Decode tok/s, Throughput tok/s), positive delta = improvement.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| errno 22 on JACCL init | bridge0 UP or TB interfaces in bridge | `sudo bash post_reboot_setup.sh` on both nodes |
| exit 255 during model load | Insufficient wired memory | Check `vm_stat`, reboot if needed |
| TTFT > 300s | First-token-timeout too short | Use `--first-token-timeout-s 600` |
| Wired memory not freed after stop | Old stop_server.sh bug (fixed in e4cc2d5) | Ensure latest code is deployed |
| SSH unreachable during stop | TB interface instability during JACCL teardown | Wait 10s and retry |
| eval_with_timeout TIMED OUT | Peer died during model loading, `all_sum` blocked | Increase `MLX_EVAL_TIMEOUT=600` or check per-layer logs for death point |
| MemoryGuardError during load | Remaining memory below safety threshold | Adjust `MLX_MEMORY_GUARD_GB` or reduce model size |
| Rank died from SIGHUP | SSH disconnect killed worker (pre-50ab48d) | Update to latest code; SIGHUP now ignored in distributed mode |
| Worker loop timed out after 1h | Inference thread stuck in collective op | Check peer health; EXIT AUDIT log shows memory state at exit |

---

## Required Metrics Checklist

Every benchmark JSON output must contain **all 5 metrics** per test case:

- [ ] `prefill_time_s` — time to process prompt
- [ ] `prefill_tok_s` — prompt tokens / prefill time
- [ ] `decode_time_s` — time from first to last generated token
- [ ] `decode_tok_s` — generated tokens / decode time
- [ ] `throughput_tok_s` — total tokens / total time

Benchmarks missing any of these are incomplete and must be fixed before proceeding.

---

## Future Phases (execute when prerequisites are met)

### Phase 6: Draft Model Spec Decode

```bash
# Prerequisite: acquire/quantize a draft model for Kimi K2.5
# Then create launch_draft.sh with:
#   --spec-decode draft --draft-model-path <path> --num-speculative-tokens 5 --draft-context-len 128

bash tests/e2e/benchmark/configs/launch_draft.sh
.venv/bin/python tests/e2e/benchmark/bench_single.py
.venv/bin/python tests/e2e/benchmark/bench_ngram.py --label draft_k5
curl -s http://localhost:8080/v1/spec_decode/metrics | python3 -m json.tool
bash tests/e2e/benchmark/configs/stop_server.sh
```

### Phase 7: MTP Spec Decode

```bash
# Prerequisite: verify model has MTP head support
# Then create launch_mtp.sh with:
#   --spec-decode mtp --num-speculative-tokens 5

bash tests/e2e/benchmark/configs/launch_mtp.sh
.venv/bin/python tests/e2e/benchmark/bench_single.py
.venv/bin/python tests/e2e/benchmark/bench_ngram.py --label mtp_k5
curl -s http://localhost:8080/v1/spec_decode/metrics | python3 -m json.tool
bash tests/e2e/benchmark/configs/stop_server.sh
```

---

## Team Workflow Notes

- **Planning/documentation**: Can be done in parallel by multiple agents.
- **Benchmarks/tests**: Must run SEQUENTIALLY (one at a time) to avoid performance interference.
- **Exception**: `bench_batch.py` and `bench_concurrent.py` intentionally send parallel requests — this is part of the test, not a workflow violation.
- **Between phases**: Always stop server completely, verify wired memory is released, then start next config.
