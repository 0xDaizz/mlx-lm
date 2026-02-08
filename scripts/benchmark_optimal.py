#!/usr/bin/env python3
"""Benchmark to find optimal batch size AND optimal block size.

Runs two analyses:
  Part 1 (--mode batch): Sweeps batch sizes 1..16, sends concurrent requests
      with 5k+ token prompts, measures aggregate throughput, per-request
      throughput, TTFT, ITL, wall time.
  Part 2 (--mode block): Sweeps block sizes 4..96, creates fresh Scheduler
      per block_size, measures cold/warm TTFT, cache hit speedup, lookup
      time, and throughput under concurrency=4.
  Part 3 (--mode all):   Runs both parts.

Usage:
    python scripts/benchmark_optimal.py --mode all --repeat 2 --warmup 1
    python scripts/benchmark_optimal.py --mode batch --repeat 1
    python scripts/benchmark_optimal.py --mode block --repeat 2
"""

from __future__ import annotations

import argparse
import queue
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_MODEL_PATH = str(PROJECT_ROOT / "Qwen3-4B-4bit")
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"

# Batch sizes and block sizes to sweep
BATCH_SIZES = [1, 2, 4, 6, 8]
BLOCK_SIZES = [4, 8, 16, 32, 48, 64, 96]

# A realistic paragraph repeated to build a 5000+ token prompt
_BASE_PARAGRAPH = (
    "Modern large language models are built on the transformer architecture, "
    "which uses self-attention mechanisms to process input sequences in parallel. "
    "The key innovation of transformers over previous recurrent architectures is "
    "that attention allows each position in the sequence to directly attend to "
    "every other position, eliminating the sequential bottleneck of RNNs and LSTMs. "
    "During inference, the model generates tokens autoregressively: at each step, "
    "it computes attention over all previous tokens and predicts the next one. "
    "To avoid redundant computation, the key-value pairs from the attention layers "
    "are cached in what is known as the KV cache. This cache grows linearly with "
    "sequence length and is one of the primary memory bottlenecks during serving. "
    "Various techniques have been developed to manage KV cache memory efficiently, "
    "including paged attention (which allocates cache in fixed-size blocks), "
    "prefix caching (which reuses cached computations for shared prompt prefixes), "
    "and KV cache quantization (which reduces the precision of cached values from "
    "FP16 to INT8 or even INT4, trading minimal quality loss for significant memory "
    "savings). On Apple Silicon hardware, the MLX framework provides native Metal "
    "acceleration for transformer inference, with unified memory architecture "
    "eliminating the CPU-GPU data transfer overhead found on discrete GPU systems. "
    "The combination of continuous batching, hash-based prefix caching, and INT8 "
    "KV cache quantization enables serving multiple concurrent requests efficiently "
    "even on a single Apple Silicon node with 192GB or 512GB of unified memory. "
    "Distributed inference across multiple nodes connected via Thunderbolt 5 RDMA "
    "further extends capacity, enabling models that exceed single-node memory limits "
    "to be served with tensor parallelism across the interconnect fabric. "
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    """Metrics from a single request."""

    ttft_ms: float = 0.0
    itl_ms_values: list[float] = field(default_factory=list)
    total_time_s: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    error: str | None = None

    @property
    def itl_ms_mean(self) -> float:
        if not self.itl_ms_values:
            return 0.0
        return statistics.mean(self.itl_ms_values)

    @property
    def throughput_tps(self) -> float:
        if self.total_time_s <= 0 or self.tokens_generated <= 0:
            return 0.0
        return self.tokens_generated / self.total_time_s


@dataclass
class BatchSizeResult:
    """Aggregated result for one batch_size configuration."""

    batch_size: int = 0
    agg_throughput_tps: float = 0.0
    per_req_throughput_tps: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_itl_ms: float = 0.0
    wall_time_s: float = 0.0
    total_tokens: int = 0
    errors: int = 0
    cache_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockSizeResult:
    """Aggregated result for one block_size configuration."""

    block_size: int = 0
    num_blocks_created: int = 0
    cold_ttft_ms: float = 0.0
    warm_ttft_ms: float = 0.0
    cache_speedup: float = 0.0
    lookup_time_ms: float = 0.0
    partial_match_cached: int = 0
    partial_match_total: int = 0
    conc4_agg_throughput: float = 0.0
    conc4_wall_time_s: float = 0.0
    cold_total_s: float = 0.0
    warm_total_s: float = 0.0
    cache_stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[Any, Any]] = {}


def load_model(model_path: str) -> tuple[Any, Any]:
    """Load and cache model + tokenizer."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    resolved = model_path
    if Path(LOCAL_MODEL_PATH).is_dir() and model_path == DEFAULT_MODEL:
        resolved = LOCAL_MODEL_PATH
    elif Path(model_path).is_dir():
        resolved = model_path

    print(f"  Loading model from {resolved} ...")
    t0 = time.time()
    from mlx_lm import load
    model, tokenizer = load(resolved)
    elapsed = time.time() - t0
    print(f"  Model loaded in {elapsed:.1f}s")

    _model_cache[model_path] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_long_prompt(tokenizer: Any, min_tokens: int = 5000) -> tuple[str, list[int]]:
    """Build a prompt with at least min_tokens by repeating a paragraph.

    Returns (prompt_text, token_ids).
    """
    prompt = ""
    repeat_count = 0
    while True:
        prompt += _BASE_PARAGRAPH + "\n\n"
        repeat_count += 1
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) >= min_tokens:
            print(f"  Long prompt: {len(token_ids)} tokens ({repeat_count} paragraph repeats)")
            return prompt, token_ids


def build_partial_prompt(
    tokenizer: Any,
    original_tokens: list[int],
    match_fraction: float = 0.8,
) -> tuple[str, list[int]]:
    """Build a prompt that shares match_fraction of tokens with the original,
    then diverges. Returns (decoded_text, token_ids)."""
    cut = int(len(original_tokens) * match_fraction)
    shared_prefix = original_tokens[:cut]
    # Decode the prefix and add a different suffix
    prefix_text = tokenizer.decode(shared_prefix)
    suffix = (
        " However, a completely different approach uses reinforcement learning "
        "with human feedback (RLHF) to align model outputs with human preferences. "
        "This technique, pioneered by OpenAI and adopted widely, has proven crucial "
        "for making language models safer and more useful in practice."
    )
    full_text = prefix_text + suffix
    token_ids = tokenizer.encode(full_text)
    return full_text, token_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler(
    model: Any,
    tokenizer: Any,
    model_path: str,
    max_batch_size: int = 8,
    num_blocks: int = 2048,
    block_size: int = 16,
) -> tuple[Any, Any]:
    """Create Scheduler + KVCacheManager. Returns (scheduler, kv_cache_manager)."""
    from mlx_lm_server.config import ServerConfig
    from mlx_lm_server.kv_cache_manager import KVCacheManager
    from mlx_lm_server.scheduler import Scheduler

    config = ServerConfig(
        model=model_path,
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_queue_size=128,
        ssd_enabled=False,
    )
    kv_mgr = KVCacheManager(config)
    sched = Scheduler(
        config=config,
        model=model,
        tokenizer=tokenizer,
        kv_cache_manager=kv_mgr,
    )
    sched.run_inference_loop()
    return sched, kv_mgr


def _make_request(
    prompt_tokens: list[int],
    max_tokens: int,
    request_id: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Build an InferenceRequest."""
    from mlx_lm_server.types import InferenceRequest

    rid = request_id or f"opt-{uuid.uuid4().hex[:8]}"
    return InferenceRequest(
        request_id=rid,
        prompt_tokens=list(prompt_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )


def _collect_streaming(
    sched: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    timeout: float = 300.0,
) -> RequestMetrics:
    """Submit one streaming request and collect TTFT/ITL/total metrics."""
    req = _make_request(prompt_tokens, max_tokens)
    result = RequestMetrics(prompt_tokens=len(prompt_tokens))

    stream_q = sched.register_stream(req.request_id)
    submit_time = time.perf_counter()
    sched.submit_request(req)

    first_token_time: float | None = None
    prev_token_time = submit_time
    itl_values: list[float] = []
    n_tokens = 0

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev = stream_q.get(timeout=min(10.0, max(0.1, deadline - time.time())))
        except queue.Empty:
            continue
        now = time.perf_counter()
        n_tokens += 1

        if first_token_time is None:
            first_token_time = now
            result.ttft_ms = (first_token_time - submit_time) * 1000
        else:
            itl_values.append((now - prev_token_time) * 1000)

        prev_token_time = now
        if ev.finish_reason is not None:
            break

    end_time = time.perf_counter()
    result.total_time_s = end_time - submit_time
    result.tokens_generated = n_tokens
    result.itl_ms_values = itl_values
    return result


def _run_concurrent(
    sched: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    concurrency: int,
    timeout: float = 300.0,
) -> tuple[list[RequestMetrics], float]:
    """Submit `concurrency` requests concurrently.

    Returns (list_of_metrics, wall_time_s).
    """
    results: list[RequestMetrics] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        try:
            r = _collect_streaming(sched, prompt_tokens, max_tokens, timeout=timeout)
            with lock:
                results.append(r)
        except Exception as e:
            with lock:
                results.append(RequestMetrics(error=str(e)))

    threads = []
    wall_start = time.perf_counter()
    for i in range(concurrency):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=timeout + 30)
    wall_time = time.perf_counter() - wall_start

    return results, wall_time


# ---------------------------------------------------------------------------
# Part 1: Optimal Batch Size
# ---------------------------------------------------------------------------


def bench_batch_size(
    model_path: str,
    warmup: int,
    repeat: int,
) -> list[BatchSizeResult]:
    """Sweep batch sizes at multiple prompt lengths to find the sweet spot."""
    print()
    print("=" * 70)
    print("  Part 1: Optimal Batch Size (multiple prompt lengths)")
    print("=" * 70)

    model, tokenizer = load_model(model_path)

    # Test at 3 prompt lengths to see how batch effectiveness changes
    prompt_configs = [
        ("short", 100, 64),    # ~100 tokens, 64 max_tokens
        ("medium", 500, 64),   # ~500 tokens, 64 max_tokens
        ("long", 2000, 64),    # ~2000 tokens, 64 max_tokens
    ]

    all_results: list[BatchSizeResult] = []

    for label, min_toks, max_tokens in prompt_configs:
        _, prompt_tokens = build_long_prompt(tokenizer, min_tokens=min_toks)
        prompt_len = len(prompt_tokens)
        print(f"\n  === Prompt: {label} ({prompt_len} tokens), max_tokens={max_tokens} ===")

        for batch_size in BATCH_SIZES:
            print(f"\n  --- {label}/batch={batch_size} ---")
            sched, kv_mgr = _make_scheduler(
                model, tokenizer, model_path,
                max_batch_size=batch_size,
                num_blocks=2048,
                block_size=16,
            )

            try:
                # Warmup
                for w in range(warmup):
                    _run_concurrent(sched, prompt_tokens, max_tokens, concurrency=batch_size, timeout=120)
                    time.sleep(0.3)

                best_result: BatchSizeResult | None = None

                for run_i in range(repeat):
                    per_req_results, wall_time = _run_concurrent(
                        sched, prompt_tokens, max_tokens,
                        concurrency=batch_size, timeout=120,
                    )

                    valid = [r for r in per_req_results if r.error is None and r.tokens_generated > 0]
                    if not valid:
                        print(f"    WARNING: No valid results")
                        continue

                    total_tokens = sum(r.tokens_generated for r in valid)
                    agg_tps = total_tokens / wall_time if wall_time > 0 else 0
                    per_req_tps = statistics.mean([r.throughput_tps for r in valid])
                    avg_ttft = statistics.mean([r.ttft_ms for r in valid if r.ttft_ms > 0])
                    itl_all = []
                    for r in valid:
                        itl_all.extend(r.itl_ms_values)
                    avg_itl = statistics.mean(itl_all) if itl_all else 0

                    br = BatchSizeResult(
                        batch_size=batch_size,
                        agg_throughput_tps=agg_tps,
                        per_req_throughput_tps=per_req_tps,
                        avg_ttft_ms=avg_ttft,
                        avg_itl_ms=avg_itl,
                        wall_time_s=wall_time,
                        total_tokens=total_tokens,
                        errors=len(per_req_results) - len(valid),
                    )
                    # Store prompt label in errors field (hack) — we'll tag it
                    if best_result is None or br.agg_throughput_tps > best_result.agg_throughput_tps:
                        best_result = br

                    print(
                        f"    agg={agg_tps:.1f} tok/s, "
                        f"per_req={per_req_tps:.1f} tok/s, "
                        f"ttft={avg_ttft:.0f}ms, "
                        f"itl={avg_itl:.1f}ms, "
                        f"wall={wall_time:.2f}s"
                    )
                    time.sleep(0.3)

                if best_result is not None:
                    best_result.cache_stats = sched.get_cache_stats()
                    all_results.append(best_result)
            finally:
                sched.stop()
                time.sleep(0.3)

    # Print summary table
    print()
    print("=" * 70)
    print("  Part 1 Results: Optimal Batch Size")
    print("=" * 70)

    # Group by prompt length (every len(BATCH_SIZES) entries)
    for idx, (label, _, _) in enumerate(prompt_configs):
        start = idx * len(BATCH_SIZES)
        end = start + len(BATCH_SIZES)
        group = all_results[start:end]
        if not group:
            continue

        print(f"\n  [{label} prompt]")
        header = (
            f"{'Batch':>6}  {'Agg Tok/s':>10}  {'Per-Req Tok/s':>14}  "
            f"{'Avg TTFT(ms)':>13}  {'Avg ITL(ms)':>12}  {'Wall(s)':>8}"
        )
        sep = "-" * len(header)
        print(header)
        print(sep)

        for r in group:
            print(
                f"{r.batch_size:>6}  {r.agg_throughput_tps:>10.1f}  "
                f"{r.per_req_throughput_tps:>14.1f}  "
                f"{r.avg_ttft_ms:>13.0f}  {r.avg_itl_ms:>12.1f}  "
                f"{r.wall_time_s:>8.2f}"
            )
        print(sep)

        # Print cache stats for each batch size
        if any(r.cache_stats for r in group):
            print(f"\n  Cache stats:")
            for r in group:
                cs = r.cache_stats
                if cs:
                    hr = cs.get('cache_hit_rate', 0)
                    ce = cs.get('cache_effectiveness', 0)
                    cached_tok = cs.get('total_cached_tokens', 0)
                    prefill_tok = cs.get('total_prefill_tokens', 0)
                    print(
                        f"    batch={r.batch_size}: hit_rate={hr:.1%}, "
                        f"effectiveness={ce:.1%}, "
                        f"cached_tokens={cached_tok}, prefill_tokens={prefill_tok}, "
                        f"total_requests={cs.get('total_requests', 0)}"
                    )

        if group:
            best = max(group, key=lambda r: r.agg_throughput_tps)
            print(f"  -> Best for {label}: batch_size={best.batch_size} ({best.agg_throughput_tps:.1f} agg tok/s)")

    if all_results:
        overall_best = max(all_results, key=lambda r: r.agg_throughput_tps)
        print(
            f"\n  -> Overall optimal batch size: {overall_best.batch_size} "
            f"({overall_best.agg_throughput_tps:.1f} agg tok/s)"
        )
    print()

    return all_results


# ---------------------------------------------------------------------------
# Part 2: Optimal Block Size
# ---------------------------------------------------------------------------


def bench_block_size(
    model_path: str,
    warmup: int,
    repeat: int,
) -> list[BlockSizeResult]:
    """Sweep block sizes, measure cache effectiveness and throughput."""
    print()
    print("=" * 70)
    print("  Part 2: Optimal Block Size (5k+ token prompt, prefix caching)")
    print("=" * 70)

    model, tokenizer = load_model(model_path)
    prompt_text, prompt_tokens = build_long_prompt(tokenizer, min_tokens=5000)
    partial_text, partial_tokens = build_partial_prompt(tokenizer, prompt_tokens, match_fraction=0.8)
    max_tokens = 64

    print(f"  Original prompt: {len(prompt_tokens)} tokens")
    print(f"  Partial-match prompt: {len(partial_tokens)} tokens "
          f"(~80% shared prefix = ~{int(len(prompt_tokens) * 0.8)} tokens)")

    all_results: list[BlockSizeResult] = []

    for block_size in BLOCK_SIZES:
        print(f"\n  --- Block size: {block_size} ---")

        # Calculate how many blocks needed — use enough to hold the prompt
        tokens_per_block = block_size
        min_blocks_needed = (len(prompt_tokens) // tokens_per_block) + 64
        # Give plenty of headroom for concurrency tests
        num_blocks = max(min_blocks_needed * 6, 2048)

        best_result: BlockSizeResult | None = None

        for run_i in range(repeat):
            print(f"    Run {run_i + 1}/{repeat} ...")

            # Fresh scheduler for each run to get clean cold/warm measurements
            sched, kv_mgr = _make_scheduler(
                model, tokenizer, model_path,
                max_batch_size=8,
                num_blocks=num_blocks,
                block_size=block_size,
            )

            try:
                # Warmup the model (not the cache) — run a short request
                for w in range(warmup):
                    short_tokens = prompt_tokens[:200]
                    _collect_streaming(sched, short_tokens, 8, timeout=60)
                    time.sleep(0.3)

                # --- COLD request: no cache hits expected ---
                cold_result = _collect_streaming(sched, prompt_tokens, max_tokens, timeout=300)
                cold_ttft = cold_result.ttft_ms
                cold_total = cold_result.total_time_s
                print(
                    f"      Cold: TTFT={cold_ttft:.0f}ms, total={cold_total:.3f}s, "
                    f"tokens={cold_result.tokens_generated}"
                )

                # Allow block decomposition
                time.sleep(1.0)

                # Count blocks and measure cache lookup
                num_blocks_created = kv_mgr.num_cached_blocks
                t_lookup_start = time.perf_counter()
                cached_prefix_len = kv_mgr.find_cached_prefix(prompt_tokens)
                t_lookup_end = time.perf_counter()
                lookup_time_ms = (t_lookup_end - t_lookup_start) * 1000

                print(
                    f"      Blocks created: {num_blocks_created}, "
                    f"cached prefix: {cached_prefix_len} tokens, "
                    f"lookup: {lookup_time_ms:.2f}ms"
                )

                # --- WARM request: same prompt, expect cache hits ---
                warm_result = _collect_streaming(sched, prompt_tokens, max_tokens, timeout=300)
                warm_ttft = warm_result.ttft_ms
                warm_total = warm_result.total_time_s
                print(
                    f"      Warm: TTFT={warm_ttft:.0f}ms, total={warm_total:.3f}s, "
                    f"tokens={warm_result.tokens_generated}"
                )

                # Capture cache stats after warm request
                cache_stats = sched.get_cache_stats()

                # Cache speedup
                cache_speedup = cold_ttft / warm_ttft if warm_ttft > 0 else 0

                # Allow decomposition again
                time.sleep(0.5)

                # --- PARTIAL match: ~80% shared prefix ---
                partial_cached = kv_mgr.find_cached_prefix(partial_tokens)
                print(
                    f"      Partial match: {partial_cached}/{len(partial_tokens)} "
                    f"tokens cached ({100 * partial_cached / len(partial_tokens):.1f}%)"
                )

                # --- Concurrent test: 4 requests with same prompt ---
                conc_results, conc_wall = _run_concurrent(
                    sched, prompt_tokens, max_tokens,
                    concurrency=4, timeout=300,
                )
                valid_conc = [r for r in conc_results if r.error is None and r.tokens_generated > 0]
                conc_total_tokens = sum(r.tokens_generated for r in valid_conc)
                conc_agg_tps = conc_total_tokens / conc_wall if conc_wall > 0 else 0
                print(
                    f"      Conc(4): agg={conc_agg_tps:.1f} tok/s, "
                    f"wall={conc_wall:.2f}s, tokens={conc_total_tokens}"
                )

                br = BlockSizeResult(
                    block_size=block_size,
                    num_blocks_created=num_blocks_created,
                    cold_ttft_ms=cold_ttft,
                    warm_ttft_ms=warm_ttft,
                    cache_speedup=cache_speedup,
                    lookup_time_ms=lookup_time_ms,
                    partial_match_cached=partial_cached,
                    partial_match_total=len(partial_tokens),
                    conc4_agg_throughput=conc_agg_tps,
                    conc4_wall_time_s=conc_wall,
                    cold_total_s=cold_total,
                    warm_total_s=warm_total,
                    cache_stats=cache_stats,
                )

                # Keep best by cache speedup
                if best_result is None or br.cache_speedup > best_result.cache_speedup:
                    best_result = br

            finally:
                sched.stop()
                time.sleep(0.3)

        if best_result is not None:
            all_results.append(best_result)

    # Print summary tables
    print()
    print("=" * 70)
    print("  Part 2 Results: Optimal Block Size")
    print("=" * 70)

    # Table A: Cache performance
    print("\n  A. Cache Performance (cold vs warm)")
    header_a = (
        f"{'BlkSz':>6}  {'Blocks':>7}  {'Cold TTFT(ms)':>14}  "
        f"{'Warm TTFT(ms)':>14}  {'Speedup':>8}  {'Lookup(ms)':>11}"
    )
    sep_a = "-" * len(header_a)
    print(header_a)
    print(sep_a)
    for r in all_results:
        print(
            f"{r.block_size:>6}  {r.num_blocks_created:>7}  "
            f"{r.cold_ttft_ms:>14.0f}  {r.warm_ttft_ms:>14.0f}  "
            f"{r.cache_speedup:>7.1f}x  {r.lookup_time_ms:>11.2f}"
        )
    print(sep_a)

    # Table B: Partial match and concurrency
    print("\n  B. Partial Match & Concurrency")
    header_b = (
        f"{'BlkSz':>6}  {'Partial Cached':>15}  {'Partial %':>10}  "
        f"{'Conc4 Tok/s':>12}  {'Conc4 Wall(s)':>14}"
    )
    sep_b = "-" * len(header_b)
    print(header_b)
    print(sep_b)
    for r in all_results:
        pct = 100 * r.partial_match_cached / r.partial_match_total if r.partial_match_total > 0 else 0
        print(
            f"{r.block_size:>6}  {r.partial_match_cached:>15}  "
            f"{pct:>9.1f}%  "
            f"{r.conc4_agg_throughput:>12.1f}  {r.conc4_wall_time_s:>14.2f}"
        )
    print(sep_b)

    # Table C: Total times
    print("\n  C. Total Request Times")
    header_c = (
        f"{'BlkSz':>6}  {'Cold Total(s)':>14}  {'Warm Total(s)':>14}  {'Total Speedup':>14}"
    )
    sep_c = "-" * len(header_c)
    print(header_c)
    print(sep_c)
    for r in all_results:
        total_speedup = r.cold_total_s / r.warm_total_s if r.warm_total_s > 0 else 0
        print(
            f"{r.block_size:>6}  {r.cold_total_s:>14.3f}  "
            f"{r.warm_total_s:>14.3f}  {total_speedup:>13.1f}x"
        )
    print(sep_c)

    # Table D: Cache Effectiveness
    print("\n  D. Cache Effectiveness")
    header_d = (
        f"{'BlkSz':>6}  {'Block Hits':>11}  {'Seq Hits':>9}  "
        f"{'Misses':>7}  {'Hit Rate':>9}  {'Effective':>10}  "
        f"{'Cached Tok':>11}  {'Prefill Tok':>12}  {'Tokens Gen':>11}"
    )
    sep_d = "-" * len(header_d)
    print(header_d)
    print(sep_d)
    for r in all_results:
        cs = r.cache_stats
        if cs:
            print(
                f"{r.block_size:>6}  {cs.get('cache_hits_block', 0):>11}  "
                f"{cs.get('cache_hits_sequence', 0):>9}  "
                f"{cs.get('cache_misses', 0):>7}  "
                f"{cs.get('cache_hit_rate', 0):>8.1%}  "
                f"{cs.get('cache_effectiveness', 0):>9.1%}  "
                f"{cs.get('total_cached_tokens', 0):>11}  "
                f"{cs.get('total_prefill_tokens', 0):>12}  "
                f"{cs.get('tokens_generated', 0):>11}"
            )
        else:
            print(
                f"{r.block_size:>6}  {'N/A':>11}  {'N/A':>9}  {'N/A':>7}  "
                f"{'N/A':>9}  {'N/A':>10}  {'N/A':>11}  {'N/A':>12}  {'N/A':>11}"
            )
    print(sep_d)

    # Recommendations
    if all_results:
        best_cold = min(all_results, key=lambda r: r.cold_ttft_ms)
        best_warm = min(all_results, key=lambda r: r.warm_ttft_ms)
        best_speedup = max(all_results, key=lambda r: r.cache_speedup)
        best_conc = max(all_results, key=lambda r: r.conc4_agg_throughput)
        best_lookup = min(all_results, key=lambda r: r.lookup_time_ms)

        print(f"\n  Recommendations:")
        print(
            f"    Best cold TTFT:          block_size={best_cold.block_size} "
            f"({best_cold.cold_ttft_ms:.0f}ms)"
        )
        print(
            f"    Best warm TTFT:          block_size={best_warm.block_size} "
            f"({best_warm.warm_ttft_ms:.0f}ms)"
        )
        print(
            f"    Best cache speedup:      block_size={best_speedup.block_size} "
            f"({best_speedup.cache_speedup:.1f}x)"
        )
        print(
            f"    Best conc4 throughput:    block_size={best_conc.block_size} "
            f"({best_conc.conc4_agg_throughput:.1f} tok/s)"
        )
        print(
            f"    Fastest cache lookup:    block_size={best_lookup.block_size} "
            f"({best_lookup.lookup_time_ms:.2f}ms)"
        )
    print()

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find optimal batch size and block size for mlx-lm-server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "block", "all"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name or local path",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Number of measured runs per configuration",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup rounds before measuring",
    )

    args = parser.parse_args()

    resolved_model = args.model
    if Path(LOCAL_MODEL_PATH).is_dir() and args.model == DEFAULT_MODEL:
        resolved_model = LOCAL_MODEL_PATH

    print("=" * 70)
    print("  mlx-lm-server: Optimal Batch Size & Block Size Benchmark")
    print(f"  Model:   {resolved_model}")
    print(f"  Mode:    {args.mode}")
    print(f"  Warmup:  {args.warmup}")
    print(f"  Repeat:  {args.repeat}")
    print("=" * 70)

    t0 = time.time()

    modes = []
    if args.mode == "all":
        modes = ["batch", "block"]
    else:
        modes = [args.mode]

    batch_results: list[BatchSizeResult] = []
    block_results: list[BlockSizeResult] = []

    for mode in modes:
        if mode == "batch":
            batch_results = bench_batch_size(args.model, args.warmup, args.repeat)
        elif mode == "block":
            block_results = bench_block_size(args.model, args.warmup, args.repeat)

    total_elapsed = time.time() - t0

    # Final summary
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    if batch_results:
        best_batch = max(batch_results, key=lambda r: r.agg_throughput_tps)
        print(
            f"  Optimal batch size: {best_batch.batch_size} "
            f"({best_batch.agg_throughput_tps:.1f} agg tok/s)"
        )

    if block_results:
        # Pick the block size with best overall balance:
        # weight cache speedup and concurrency throughput
        best_block = max(
            block_results,
            key=lambda r: r.cache_speedup * 0.5 + (r.conc4_agg_throughput / max(1, max(b.conc4_agg_throughput for b in block_results))) * 0.5,
        )
        print(
            f"  Optimal block size: {best_block.block_size} "
            f"(cache speedup: {best_block.cache_speedup:.1f}x, "
            f"conc4: {best_block.conc4_agg_throughput:.1f} tok/s)"
        )

    print(f"\n  Total benchmark time: {total_elapsed:.1f}s")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
