#!/usr/bin/env python3
"""Comprehensive benchmark for mlx-lm-server.

Runs three benchmark categories:
  Part 1 (--part variation):   Parameter variation across 36 combinations of
      block_size, num_blocks, prompt_length, and max_batch_size.
  Part 2 (--part comparison):  Before/after comparisons: cache ON vs OFF,
      trie prefix search scaling, heap eviction scaling, and prefill skip
      verification.
  Part 3 (--part correctness): Correctness validation: token completeness,
      streaming vs non-streaming consistency, cache hit/miss output
      consistency, and concurrent request isolation.
  (--part all): Runs all three parts.

Usage:
    python scripts/benchmark_comprehensive.py --part all
    python scripts/benchmark_comprehensive.py --part variation
    python scripts/benchmark_comprehensive.py --part comparison
    python scripts/benchmark_comprehensive.py --part correctness
    python scripts/benchmark_comprehensive.py --part all --model path/to/model
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_MODEL_PATH = str(PROJECT_ROOT / "Qwen3-4B-4bit")
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"

RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

# A paragraph (~50 tokens when encoded) for building prompts of various lengths
_PARAGRAPH = (
    "Modern large language models are built on the transformer architecture, "
    "which uses self-attention mechanisms to process input sequences in parallel. "
    "The key innovation of transformers is that attention allows each position "
    "in the sequence to directly attend to every other position, eliminating "
    "the sequential bottleneck of RNNs and LSTMs. During inference the model "
    "generates tokens autoregressively. To avoid redundant computation the "
    "key-value pairs from the attention layers are cached in the KV cache. "
)

# Distinct prompts for concurrent request isolation test (Part 3d)
_DISTINCT_PROMPTS = [
    "Explain the process of photosynthesis in plants. How do chloroplasts convert sunlight into chemical energy?",
    "Describe the history of the Roman Empire from its founding to the fall of the Western empire.",
    "What are the fundamental principles of quantum mechanics? Explain wave-particle duality and superposition.",
    "How does a modern CPU execute instructions? Describe the fetch-decode-execute cycle and pipelining.",
]


# ---------------------------------------------------------------------------
# Model loading (cached)
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
# Prompt building helpers
# ---------------------------------------------------------------------------


def build_prompt(tokenizer: Any, target_tokens: int = 200) -> tuple[str, list[int]]:
    """Build a prompt of approximately target_tokens length.

    Returns (prompt_text, token_ids truncated to target_tokens).
    """
    text = ""
    while True:
        text += _PARAGRAPH + " "
        tokens = tokenizer.encode(text)
        if len(tokens) >= target_tokens:
            # Decode truncated tokens to get clean text
            truncated = tokens[:target_tokens]
            return text, truncated


# ---------------------------------------------------------------------------
# Scheduler / KVCacheManager creation helpers
# ---------------------------------------------------------------------------


def make_scheduler(
    model: Any,
    tokenizer: Any,
    block_size: int = 16,
    num_blocks: int = 128,
    max_batch_size: int = 4,
    kv_cache_manager: Any = None,
    use_cache: bool = True,
) -> tuple[Any, Any]:
    """Create a Scheduler (+ optional KVCacheManager). Returns (scheduler, config).

    If use_cache=True and kv_cache_manager is not provided, one is created.
    If use_cache=False, kv_cache_manager is set to None.
    """
    from mlx_lm_server.config import ServerConfig
    from mlx_lm_server.kv_cache_manager import KVCacheManager
    from mlx_lm_server.scheduler import Scheduler

    config = ServerConfig(
        model="Qwen3-4B-4bit",
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_queue_size=128,
        ssd_enabled=False,
    )

    if use_cache:
        if kv_cache_manager is None:
            kv_cache_manager = KVCacheManager(config)
    else:
        kv_cache_manager = None

    sched = Scheduler(
        config=config,
        model=model,
        tokenizer=tokenizer,
        kv_cache_manager=kv_cache_manager,
    )
    sched.run_inference_loop()
    return sched, config


# ---------------------------------------------------------------------------
# Streaming / non-streaming collection helpers
# ---------------------------------------------------------------------------


def collect_streaming(
    sched: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Submit a streaming request and collect all tokens with timing metrics."""
    from mlx_lm_server.types import InferenceRequest

    req = InferenceRequest(
        request_id=f"bench-{uuid.uuid4().hex[:8]}",
        prompt_tokens=list(prompt_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    q = sched.register_stream(req.request_id)
    t0 = time.perf_counter()
    sched.submit_request(req)

    tokens: list[Any] = []
    ttft: float | None = None
    prev_t = t0
    itl_values: list[float] = []

    while True:
        try:
            ev = q.get(timeout=timeout)
        except queue.Empty:
            break
        now = time.perf_counter()
        if ttft is None:
            ttft = (now - t0) * 1000
        else:
            itl_values.append((now - prev_t) * 1000)
        prev_t = now
        tokens.append(ev)
        if ev.finish_reason is not None:
            break

    total_time = time.perf_counter() - t0
    return {
        "tokens": tokens,
        "token_texts": [t.token_text for t in tokens],
        "ttft_ms": ttft or 0.0,
        "itl_ms": statistics.mean(itl_values) if itl_values else 0.0,
        "total_time_s": total_time,
        "throughput_tps": len(tokens) / total_time if total_time > 0 else 0.0,
        "finish_reason": tokens[-1].finish_reason if tokens else None,
        "token_count": len(tokens),
    }


def collect_nonstreaming(
    sched: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Submit a non-streaming request and collect all tokens."""
    from mlx_lm_server.types import InferenceRequest

    req = InferenceRequest(
        request_id=f"bench-{uuid.uuid4().hex[:8]}",
        prompt_tokens=list(prompt_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    t0 = time.perf_counter()
    sched.submit_request(req)
    events = sched.get_result(req.request_id, timeout=timeout)
    total_time = time.perf_counter() - t0
    return {
        "tokens": events,
        "token_texts": [e.token_text for e in events],
        "total_time_s": total_time,
        "finish_reason": events[-1].finish_reason if events else None,
        "token_count": len(events),
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def _ensure_results_dir() -> None:
    """Create benchmark_results/ directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(filename: str, data: Any) -> None:
    """Save data as JSON to benchmark_results/filename."""
    _ensure_results_dir()
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


def _save_summary(text: str) -> None:
    """Append text to benchmark_results/summary.txt."""
    _ensure_results_dir()
    filepath = RESULTS_DIR / "summary.txt"
    with open(filepath, "a") as f:
        f.write(text)
    print(f"  Saved summary to: {filepath}")


# ========================================================================
# Part 1: Parameter Variation (36 combinations)
# ========================================================================


def run_part_variation(model_path: str) -> list[dict[str, Any]]:
    """Sweep 36 parameter combinations and measure performance."""
    print()
    print("=" * 72)
    print("  PART 1: Parameter Variation (36 combinations)")
    print("=" * 72)

    model, tokenizer = load_model(model_path)

    block_sizes = [8, 16, 32]
    num_blocks_list = [32, 64, 128]
    prompt_lengths = [("short", 50), ("long", 500)]
    max_batch_sizes = [1, 4]

    all_results: list[dict[str, Any]] = []
    combo_idx = 0
    total_combos = len(block_sizes) * len(num_blocks_list) * len(prompt_lengths) * len(max_batch_sizes)

    for block_size in block_sizes:
        for num_blocks in num_blocks_list:
            for prompt_label, prompt_target in prompt_lengths:
                for max_batch_size in max_batch_sizes:
                    combo_idx += 1
                    combo_key = (
                        f"bs={block_size} nb={num_blocks} "
                        f"prompt={prompt_label} batch={max_batch_size}"
                    )
                    print(f"\n  [{combo_idx}/{total_combos}] {combo_key}")

                    result: dict[str, Any] = {
                        "block_size": block_size,
                        "num_blocks": num_blocks,
                        "prompt_label": prompt_label,
                        "prompt_target": prompt_target,
                        "max_batch_size": max_batch_size,
                        "error": None,
                    }

                    sched = None
                    try:
                        # Build prompt
                        _, prompt_tokens = build_prompt(tokenizer, target_tokens=prompt_target)
                        result["actual_prompt_tokens"] = len(prompt_tokens)

                        # Check if we have enough blocks for this prompt
                        blocks_needed = len(prompt_tokens) // block_size
                        if blocks_needed > num_blocks:
                            result["error"] = (
                                f"Not enough blocks: need {blocks_needed} "
                                f"but only {num_blocks} available"
                            )
                            print(f"    SKIP: {result['error']}")
                            all_results.append(result)
                            continue

                        # Create scheduler
                        sched, config = make_scheduler(
                            model, tokenizer,
                            block_size=block_size,
                            num_blocks=num_blocks,
                            max_batch_size=max_batch_size,
                        )

                        # Warmup (short, 8 max_tokens)
                        print("    Warmup ...", end="", flush=True)
                        collect_streaming(sched, prompt_tokens, max_tokens=8, timeout=60)
                        print(" done")
                        time.sleep(0.3)

                        # Cold run
                        print("    Cold run ...", end="", flush=True)
                        cold = collect_streaming(sched, prompt_tokens, max_tokens=32, timeout=120)
                        result["cold_ttft_ms"] = cold["ttft_ms"]
                        result["cold_itl_ms"] = cold["itl_ms"]
                        result["cold_total_s"] = cold["total_time_s"]
                        result["cold_throughput_tps"] = cold["throughput_tps"]
                        result["cold_token_count"] = cold["token_count"]
                        result["cold_finish_reason"] = cold["finish_reason"]
                        print(
                            f" TTFT={cold['ttft_ms']:.0f}ms "
                            f"ITL={cold['itl_ms']:.1f}ms "
                            f"throughput={cold['throughput_tps']:.1f}tps"
                        )
                        time.sleep(0.3)

                        # Warm run (same prompt, expect cache)
                        print("    Warm run ...", end="", flush=True)
                        warm = collect_streaming(sched, prompt_tokens, max_tokens=32, timeout=120)
                        result["warm_ttft_ms"] = warm["ttft_ms"]
                        result["warm_itl_ms"] = warm["itl_ms"]
                        result["warm_total_s"] = warm["total_time_s"]
                        result["warm_throughput_tps"] = warm["throughput_tps"]
                        result["warm_token_count"] = warm["token_count"]
                        result["warm_finish_reason"] = warm["finish_reason"]
                        print(
                            f" TTFT={warm['ttft_ms']:.0f}ms "
                            f"ITL={warm['itl_ms']:.1f}ms "
                            f"throughput={warm['throughput_tps']:.1f}tps"
                        )

                        # Cache speedup
                        if warm["ttft_ms"] > 0:
                            result["cache_speedup"] = cold["ttft_ms"] / warm["ttft_ms"]
                        else:
                            result["cache_speedup"] = 0.0

                        # Cache stats
                        cache_stats = sched.get_cache_stats()
                        result["cache_stats"] = {
                            k: v for k, v in cache_stats.items()
                            if isinstance(v, (int, float, str, bool))
                        }

                        print(
                            f"    Cache speedup: {result['cache_speedup']:.1f}x, "
                            f"hit_rate={cache_stats.get('cache_hit_rate', 0):.1%}"
                        )

                    except Exception as e:
                        result["error"] = str(e)
                        print(f"    ERROR: {e}")
                    finally:
                        if sched is not None:
                            sched.stop()
                        time.sleep(0.5)

                    all_results.append(result)

    # Print summary table
    print()
    print("=" * 72)
    print("  Part 1 Summary: Parameter Variation")
    print("=" * 72)
    header = (
        f"{'BlkSz':>6} {'NumBlk':>7} {'Prompt':>7} {'Batch':>6} "
        f"{'Cold TTFT':>10} {'Warm TTFT':>10} {'Speedup':>8} "
        f"{'Throughput':>11} {'Status':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r.get("error"):
            print(
                f"{r['block_size']:>6} {r['num_blocks']:>7} "
                f"{r['prompt_label']:>7} {r['max_batch_size']:>6} "
                f"{'':>10} {'':>10} {'':>8} {'':>11} {'ERROR':>8}"
            )
        else:
            print(
                f"{r['block_size']:>6} {r['num_blocks']:>7} "
                f"{r['prompt_label']:>7} {r['max_batch_size']:>6} "
                f"{r.get('cold_ttft_ms', 0):>9.0f}ms "
                f"{r.get('warm_ttft_ms', 0):>9.0f}ms "
                f"{r.get('cache_speedup', 0):>7.1f}x "
                f"{r.get('warm_throughput_tps', 0):>10.1f}t/s "
                f"{'OK':>8}"
            )
    print()

    _save_json("variation_results.json", all_results)
    return all_results


# ========================================================================
# Part 2: Before vs After Comparison
# ========================================================================


def run_part_comparison(model_path: str) -> dict[str, Any]:
    """Run comparison benchmarks: cache ON/OFF, trie, heap, prefill skip."""
    print()
    print("=" * 72)
    print("  PART 2: Before vs After Comparison")
    print("=" * 72)

    comparison_results: dict[str, Any] = {}

    # --- 2a: Cache ON vs OFF ---
    print("\n  --- 2a: Cache ON vs OFF ---")
    try:
        comparison_results["cache_on_vs_off"] = _bench_cache_on_off(model_path)
    except Exception as e:
        comparison_results["cache_on_vs_off"] = {"error": str(e)}
        print(f"    ERROR: {e}")

    # --- 2b: Trie prefix search micro-benchmark ---
    print("\n  --- 2b: Trie Prefix Search Scaling ---")
    try:
        comparison_results["trie_scaling"] = _bench_trie_scaling()
    except Exception as e:
        comparison_results["trie_scaling"] = {"error": str(e)}
        print(f"    ERROR: {e}")

    # --- 2c: Heap eviction micro-benchmark ---
    print("\n  --- 2c: Heap Eviction Scaling ---")
    try:
        comparison_results["heap_eviction"] = _bench_heap_eviction()
    except Exception as e:
        comparison_results["heap_eviction"] = {"error": str(e)}
        print(f"    ERROR: {e}")

    # --- 2d: Prefill skip verification ---
    print("\n  --- 2d: Prefill Skip Verification ---")
    try:
        comparison_results["prefill_skip"] = _bench_prefill_skip(model_path)
    except Exception as e:
        comparison_results["prefill_skip"] = {"error": str(e)}
        print(f"    ERROR: {e}")

    _save_json("comparison_results.json", comparison_results)
    return comparison_results


def _bench_cache_on_off(model_path: str) -> dict[str, Any]:
    """2a: Compare cache ON vs OFF for warm TTFT."""
    model, tokenizer = load_model(model_path)
    _, prompt_tokens = build_prompt(tokenizer, target_tokens=200)

    result: dict[str, Any] = {"prompt_tokens": len(prompt_tokens)}

    # --- Cache ON ---
    print("    Cache ON: cold run ...")
    sched_on, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=4,
        use_cache=True,
    )
    try:
        cold_on = collect_streaming(sched_on, prompt_tokens, max_tokens=32, timeout=120)
        result["cache_on_cold_ttft_ms"] = cold_on["ttft_ms"]
        time.sleep(0.3)

        print("    Cache ON: warm run ...")
        warm_on = collect_streaming(sched_on, prompt_tokens, max_tokens=32, timeout=120)
        result["cache_on_warm_ttft_ms"] = warm_on["ttft_ms"]
        result["cache_on_stats"] = {
            k: v for k, v in sched_on.get_cache_stats().items()
            if isinstance(v, (int, float, str, bool))
        }
    finally:
        sched_on.stop()
        time.sleep(0.5)

    # --- Cache OFF ---
    print("    Cache OFF: cold run ...")
    sched_off, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=4,
        use_cache=False,
    )
    try:
        cold_off = collect_streaming(sched_off, prompt_tokens, max_tokens=32, timeout=120)
        result["cache_off_cold_ttft_ms"] = cold_off["ttft_ms"]
        time.sleep(0.3)

        print("    Cache OFF: warm run ...")
        warm_off = collect_streaming(sched_off, prompt_tokens, max_tokens=32, timeout=120)
        result["cache_off_warm_ttft_ms"] = warm_off["ttft_ms"]
    finally:
        sched_off.stop()
        time.sleep(0.5)

    # Summary
    print(f"    Cache ON  -> cold={result['cache_on_cold_ttft_ms']:.0f}ms, "
          f"warm={result['cache_on_warm_ttft_ms']:.0f}ms")
    print(f"    Cache OFF -> cold={result['cache_off_cold_ttft_ms']:.0f}ms, "
          f"warm={result['cache_off_warm_ttft_ms']:.0f}ms")
    if result["cache_off_warm_ttft_ms"] > 0:
        speedup = result["cache_off_warm_ttft_ms"] / result["cache_on_warm_ttft_ms"]
        result["warm_ttft_speedup"] = speedup
        print(f"    Warm TTFT speedup (cache ON vs OFF): {speedup:.2f}x")
    else:
        result["warm_ttft_speedup"] = 0.0

    return result


def _bench_trie_scaling() -> dict[str, Any]:
    """2b: Micro-benchmark trie prefix search at various entry counts.

    NO model needed. Uses SequenceCacheStore with dummy cache values.
    Demonstrates O(M) lookup time (scales with query length, not entry count).
    """
    from mlx_lm_server.sequence_cache import SequenceCacheStore

    entry_counts = [100, 500, 1000]
    results: dict[str, Any] = {}

    for n_entries in entry_counts:
        print(f"    Entries: {n_entries}")
        store = SequenceCacheStore(max_entries=n_entries + 100)

        # Insert N random token sequences with dummy cache values
        random.seed(42)
        base_tokens = [random.randint(0, 50000) for _ in range(100)]
        for i in range(n_entries):
            # Create varied sequences sharing a common prefix
            seq_len = random.randint(100, 500)
            # First 80% from base, rest random
            prefix_len = int(seq_len * 0.8)
            seq = base_tokens[:min(prefix_len, len(base_tokens))]
            seq += [random.randint(0, 50000) for _ in range(seq_len - len(seq))]
            store.store(seq, [None])  # Dummy cache value

        # Query that shares ~80% prefix with base_tokens
        query = list(base_tokens[:80]) + [random.randint(0, 50000) for _ in range(20)]

        # Time 1000 lookups
        num_lookups = 1000
        t0 = time.perf_counter()
        for _ in range(num_lookups):
            store.find_longest_prefix(query)
        elapsed = time.perf_counter() - t0

        avg_us = (elapsed / num_lookups) * 1_000_000
        results[str(n_entries)] = {
            "entries": n_entries,
            "lookups": num_lookups,
            "total_time_s": elapsed,
            "avg_lookup_us": avg_us,
            "query_length": len(query),
        }
        print(f"      Avg lookup: {avg_us:.1f}us ({num_lookups} lookups)")

    # Show scaling: time should depend on query length, NOT entry count
    times = [results[str(n)]["avg_lookup_us"] for n in entry_counts]
    if len(times) >= 2 and times[0] > 0:
        ratio = times[-1] / times[0]
        entry_ratio = entry_counts[-1] / entry_counts[0]
        results["scaling_analysis"] = {
            "entry_ratio": entry_ratio,
            "time_ratio": ratio,
            "is_sublinear": ratio < entry_ratio * 0.5,
            "note": (
                f"Entries grew {entry_ratio:.0f}x but time grew {ratio:.1f}x. "
                f"{'Confirms O(M) trie behavior.' if ratio < entry_ratio * 0.5 else 'Scaling is close to linear in entries.'}"
            ),
        }
        print(f"    Scaling: entries {entry_ratio:.0f}x -> time {ratio:.1f}x "
              f"({'O(M) confirmed' if ratio < entry_ratio * 0.5 else 'near-linear'})")

    return results


def _bench_heap_eviction() -> dict[str, Any]:
    """2c: Micro-benchmark heap eviction at various pool sizes.

    NO model needed. Creates KVCacheManager, fills all blocks, times eviction.
    """
    from mlx_lm_server.config import ServerConfig
    from mlx_lm_server.kv_cache_manager import KVCacheManager

    pool_sizes = [64, 256, 1024, 4096]
    results: dict[str, Any] = {}

    for num_blocks in pool_sizes:
        print(f"    Pool size: {num_blocks}")
        config = ServerConfig(
            model="bench",
            block_size=16,
            num_blocks=num_blocks,
            ssd_enabled=False,
        )
        mgr = KVCacheManager(config)

        # Allocate ALL blocks by filling with dummy entries
        block_size = 16
        for i in range(num_blocks):
            token_ids = list(range(i * block_size, (i + 1) * block_size))
            prefix = list(range(0, i * block_size))
            block_hash = mgr.compute_block_hash(prefix, token_ids)
            with mgr.lock:
                try:
                    block = mgr.pool.get_free_block()
                except Exception:
                    break
                block.block_hash = block_hash
                block.token_ids = list(token_ids)
                block.ref_count = 0  # Evictable
                block.last_accessed = time.time() - (num_blocks - i)  # Oldest first
                mgr.hash_table[block_hash] = block.block_id
                import heapq
                heapq.heappush(mgr._eviction_heap, (block.last_accessed, block.block_id))

        # Time 100 evictions
        num_evictions = 100
        t0 = time.perf_counter()
        for _ in range(num_evictions):
            evicted = mgr.evict_lru(num_blocks=1)
            if not evicted:
                break
        elapsed = time.perf_counter() - t0
        actual_evictions = min(num_evictions, num_blocks)

        avg_us = (elapsed / actual_evictions) * 1_000_000 if actual_evictions > 0 else 0
        results[str(num_blocks)] = {
            "num_blocks": num_blocks,
            "evictions": actual_evictions,
            "total_time_s": elapsed,
            "avg_eviction_us": avg_us,
        }
        print(f"      Avg eviction: {avg_us:.1f}us ({actual_evictions} evictions)")

    # Show scaling
    sizes = [64, 4096]
    if str(sizes[0]) in results and str(sizes[1]) in results:
        t0 = results[str(sizes[0])]["avg_eviction_us"]
        t1 = results[str(sizes[1])]["avg_eviction_us"]
        if t0 > 0:
            ratio = t1 / t0
            size_ratio = sizes[1] / sizes[0]
            results["scaling_analysis"] = {
                "size_ratio": size_ratio,
                "time_ratio": ratio,
                "note": (
                    f"Pool grew {size_ratio:.0f}x, eviction time grew {ratio:.1f}x. "
                    f"{'O(log N) heap confirmed.' if ratio < size_ratio * 0.3 else 'Scaling may include fallback path.'}"
                ),
            }
            print(f"    Scaling: pool {size_ratio:.0f}x -> eviction time {ratio:.1f}x")

    return results


def _bench_prefill_skip(model_path: str) -> dict[str, Any]:
    """2d: Verify that warm runs skip prefill via cache hits."""
    model, tokenizer = load_model(model_path)
    _, prompt_tokens = build_prompt(tokenizer, target_tokens=300)

    result: dict[str, Any] = {"prompt_tokens": len(prompt_tokens)}

    sched, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=1,
        use_cache=True,
    )
    try:
        # Cold run
        print("    Cold run ...")
        collect_streaming(sched, prompt_tokens, max_tokens=32, timeout=120)
        time.sleep(0.5)
        cold_stats = sched.get_cache_stats()
        cold_prefill = cold_stats.get("total_prefill_tokens", 0)
        cold_cached = cold_stats.get("total_cached_tokens", 0)
        result["cold_prefill_tokens"] = cold_prefill
        result["cold_cached_tokens"] = cold_cached
        print(f"    Cold: prefill={cold_prefill}, cached={cold_cached}")

        # Warm run (same prompt)
        print("    Warm run ...")
        collect_streaming(sched, prompt_tokens, max_tokens=32, timeout=120)
        time.sleep(0.3)
        warm_stats = sched.get_cache_stats()
        warm_prefill = warm_stats.get("total_prefill_tokens", 0)
        warm_cached = warm_stats.get("total_cached_tokens", 0)
        result["warm_prefill_tokens"] = warm_prefill
        result["warm_cached_tokens"] = warm_cached
        print(f"    Warm (cumulative): prefill={warm_prefill}, cached={warm_cached}")

        # The warm run should have added cached tokens and fewer prefill tokens
        # than the cold run's prefill
        warm_only_prefill = warm_prefill - cold_prefill
        warm_only_cached = warm_cached - cold_cached
        result["warm_only_prefill"] = warm_only_prefill
        result["warm_only_cached"] = warm_only_cached

        prefill_reduced = warm_only_prefill < cold_prefill
        result["prefill_skip_verified"] = prefill_reduced
        print(
            f"    Warm-only prefill={warm_only_prefill} "
            f"(cold was {cold_prefill}): "
            f"{'PASS - prefill reduced' if prefill_reduced else 'FAIL - no reduction'}"
        )
        if warm_only_cached > 0:
            print(f"    Warm-only cached tokens: {warm_only_cached}")
        assert prefill_reduced, (
            f"Warm prefill ({warm_only_prefill}) should be less than "
            f"cold prefill ({cold_prefill})"
        )
    finally:
        sched.stop()
        time.sleep(0.5)

    return result


# ========================================================================
# Part 3: Correctness Validation
# ========================================================================


def run_part_correctness(model_path: str) -> dict[str, Any]:
    """Run correctness validation benchmarks."""
    print()
    print("=" * 72)
    print("  PART 3: Correctness Validation")
    print("=" * 72)

    correctness_results: dict[str, Any] = {}

    # --- 3a: Token completeness ---
    print("\n  --- 3a: Token Completeness ---")
    try:
        correctness_results["token_completeness"] = _test_token_completeness(model_path)
    except Exception as e:
        correctness_results["token_completeness"] = {"error": str(e), "passed": False}
        print(f"    ERROR: {e}")

    # --- 3b: Streaming vs non-streaming consistency ---
    print("\n  --- 3b: Streaming vs Non-Streaming Consistency ---")
    try:
        correctness_results["stream_vs_nonstream"] = _test_stream_vs_nonstream(model_path)
    except Exception as e:
        correctness_results["stream_vs_nonstream"] = {"error": str(e), "passed": False}
        print(f"    ERROR: {e}")

    # --- 3c: Cache hit vs miss output consistency ---
    print("\n  --- 3c: Cache Hit vs Miss Output Consistency (MOST IMPORTANT) ---")
    try:
        correctness_results["cache_consistency"] = _test_cache_consistency(model_path)
    except Exception as e:
        correctness_results["cache_consistency"] = {"error": str(e), "passed": False}
        print(f"    ERROR: {e}")

    # --- 3d: Concurrent request isolation ---
    print("\n  --- 3d: Concurrent Request Isolation ---")
    try:
        correctness_results["concurrent_isolation"] = _test_concurrent_isolation(model_path)
    except Exception as e:
        correctness_results["concurrent_isolation"] = {"error": str(e), "passed": False}
        print(f"    ERROR: {e}")

    # Overall summary
    all_passed = all(
        r.get("passed", False) for r in correctness_results.values()
    )
    correctness_results["all_passed"] = all_passed
    print(f"\n  Correctness overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    _save_json("correctness_results.json", correctness_results)
    return correctness_results


def _test_token_completeness(model_path: str) -> dict[str, Any]:
    """3a: Verify all runs produce same token count, finish_reason='length'."""
    model, tokenizer = load_model(model_path)
    _, prompt_tokens = build_prompt(tokenizer, target_tokens=200)

    max_tokens = 32
    num_runs = 3
    result: dict[str, Any] = {"num_runs": num_runs, "max_tokens": max_tokens}

    sched, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=1,
        use_cache=True,
    )
    try:
        runs = []
        for i in range(num_runs):
            print(f"    Run {i + 1}/{num_runs} ...")
            r = collect_streaming(sched, prompt_tokens, max_tokens=max_tokens, timeout=120)
            runs.append(r)
            time.sleep(0.3)

        counts = [r["token_count"] for r in runs]
        finish_reasons = [r["finish_reason"] for r in runs]
        result["token_counts"] = counts
        result["finish_reasons"] = finish_reasons

        all_same_count = len(set(counts)) == 1
        all_length_finish = all(fr == "length" for fr in finish_reasons)

        result["all_same_count"] = all_same_count
        result["all_length_finish"] = all_length_finish
        result["passed"] = all_same_count and all_length_finish

        print(f"    Token counts: {counts} -> {'SAME' if all_same_count else 'DIFFER'}")
        print(f"    Finish reasons: {finish_reasons} -> "
              f"{'ALL length' if all_length_finish else 'MISMATCH'}")
        print(f"    Result: {'PASS' if result['passed'] else 'FAIL'}")

    finally:
        sched.stop()
        time.sleep(0.5)

    return result


def _test_stream_vs_nonstream(model_path: str) -> dict[str, Any]:
    """3b: Verify streaming and non-streaming produce identical output."""
    model, tokenizer = load_model(model_path)
    _, prompt_tokens = build_prompt(tokenizer, target_tokens=200)

    max_tokens = 32
    result: dict[str, Any] = {"max_tokens": max_tokens}

    sched, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=1,
        use_cache=True,
    )
    try:
        # Streaming
        print("    Streaming run ...")
        stream_result = collect_streaming(
            sched, prompt_tokens, max_tokens=max_tokens, timeout=120
        )
        time.sleep(0.3)

        # Non-streaming
        print("    Non-streaming run ...")
        nonstream_result = collect_nonstreaming(
            sched, prompt_tokens, max_tokens=max_tokens, timeout=120
        )

        stream_texts = stream_result["token_texts"]
        nonstream_texts = nonstream_result["token_texts"]

        result["stream_token_count"] = len(stream_texts)
        result["nonstream_token_count"] = len(nonstream_texts)
        result["stream_text"] = "".join(stream_texts)
        result["nonstream_text"] = "".join(nonstream_texts)

        texts_match = stream_texts == nonstream_texts
        result["texts_match"] = texts_match
        result["passed"] = texts_match

        if texts_match:
            print(f"    Streaming ({len(stream_texts)} tokens) == "
                  f"Non-streaming ({len(nonstream_texts)} tokens)")
            print(f"    Result: PASS")
        else:
            print(f"    Streaming:     {stream_texts[:5]}...")
            print(f"    Non-streaming: {nonstream_texts[:5]}...")
            print(f"    Result: FAIL (token texts differ)")

    finally:
        sched.stop()
        time.sleep(0.5)

    return result


def _test_cache_consistency(model_path: str) -> dict[str, Any]:
    """3c: Verify cold (cache miss) and warm (cache hit) produce same output.

    This is the MOST IMPORTANT correctness test.
    """
    model, tokenizer = load_model(model_path)
    _, prompt_tokens = build_prompt(tokenizer, target_tokens=200)

    max_tokens = 32
    result: dict[str, Any] = {"max_tokens": max_tokens}

    # Cold run: fresh scheduler, no cache
    print("    Cold run (fresh scheduler, no cache) ...")
    sched_cold, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=1,
        use_cache=True,
    )
    try:
        cold = collect_streaming(sched_cold, prompt_tokens, max_tokens=max_tokens, timeout=120)
    finally:
        sched_cold.stop()
        time.sleep(0.5)

    # Warm run: reuse scheduler (should hit cache)
    print("    Warm run (reuse scheduler, should hit cache) ...")
    sched_warm, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=128, max_batch_size=1,
        use_cache=True,
    )
    try:
        # First request to populate cache
        collect_streaming(sched_warm, prompt_tokens, max_tokens=max_tokens, timeout=120)
        time.sleep(0.5)

        # Second request should hit cache
        warm = collect_streaming(sched_warm, prompt_tokens, max_tokens=max_tokens, timeout=120)
    finally:
        sched_warm.stop()
        time.sleep(0.5)

    cold_texts = cold["token_texts"]
    warm_texts = warm["token_texts"]
    result["cold_text"] = "".join(cold_texts)
    result["warm_text"] = "".join(warm_texts)
    result["cold_token_count"] = len(cold_texts)
    result["warm_token_count"] = len(warm_texts)

    outputs_match = cold_texts == warm_texts
    result["outputs_match"] = outputs_match
    result["passed"] = outputs_match

    if outputs_match:
        print(f"    Cold ({len(cold_texts)} tokens) == Warm ({len(warm_texts)} tokens)")
        print(f"    Output: {''.join(cold_texts[:50])}...")
        print(f"    Result: PASS")
    else:
        print(f"    Cold:  {''.join(cold_texts[:50])}...")
        print(f"    Warm:  {''.join(warm_texts[:50])}...")
        print(f"    Result: FAIL (outputs differ)")

    return result


def _test_concurrent_isolation(model_path: str) -> dict[str, Any]:
    """3d: Verify 4 concurrent requests produce isolated, correct outputs."""
    model, tokenizer = load_model(model_path)

    max_tokens = 32
    result: dict[str, Any] = {"max_tokens": max_tokens, "num_prompts": len(_DISTINCT_PROMPTS)}

    # Tokenize each distinct prompt
    prompt_data: list[tuple[str, list[int]]] = []
    for text in _DISTINCT_PROMPTS:
        tokens = tokenizer.encode(text)
        prompt_data.append((text, tokens))

    sched, _ = make_scheduler(
        model, tokenizer,
        block_size=16, num_blocks=256, max_batch_size=4,
        use_cache=True,
    )
    try:
        outputs: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        lock = threading.Lock()

        def worker(idx: int, prompt_text: str, prompt_tokens: list[int]) -> None:
            try:
                r = collect_streaming(sched, prompt_tokens, max_tokens=max_tokens, timeout=120)
                with lock:
                    outputs[f"prompt_{idx}"] = {
                        "prompt_text": prompt_text[:50],
                        "output_text": "".join(r["token_texts"]),
                        "token_count": r["token_count"],
                        "finish_reason": r["finish_reason"],
                    }
            except Exception as e:
                with lock:
                    errors.append(f"prompt_{idx}: {e}")

        print("    Submitting 4 concurrent requests ...")
        threads = []
        for i, (text, tokens) in enumerate(prompt_data):
            t = threading.Thread(target=worker, args=(i, text, tokens))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=120)

        result["outputs"] = outputs
        result["errors"] = errors

        if errors:
            print(f"    Errors: {errors}")
            result["passed"] = False
        else:
            # Verify each response's content does not contain text from other prompts
            # (simple check: each output should be unique)
            output_texts = [v["output_text"] for v in outputs.values()]
            all_unique = len(set(output_texts)) == len(output_texts)
            all_have_tokens = all(v["token_count"] > 0 for v in outputs.values())

            result["all_unique"] = all_unique
            result["all_have_tokens"] = all_have_tokens
            result["passed"] = all_unique and all_have_tokens

            for k, v in outputs.items():
                print(f"    {k}: {v['token_count']} tokens, "
                      f"output='{v['output_text'][:40]}...'")
            print(f"    All unique: {all_unique}")
            print(f"    All have tokens: {all_have_tokens}")
            print(f"    Result: {'PASS' if result['passed'] else 'FAIL'}")

    finally:
        sched.stop()
        time.sleep(0.5)

    return result


# ========================================================================
# CLI Entry Point
# ========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for mlx-lm-server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--part",
        choices=["all", "variation", "comparison", "correctness"],
        default="all",
        help="Which benchmark part to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name or local path",
    )

    args = parser.parse_args()

    resolved_model = args.model
    if Path(LOCAL_MODEL_PATH).is_dir() and args.model == DEFAULT_MODEL:
        resolved_model = LOCAL_MODEL_PATH

    print("=" * 72)
    print("  mlx-lm-server: Comprehensive Benchmark")
    print(f"  Model:   {resolved_model}")
    print(f"  Part:    {args.part}")
    print("=" * 72)

    # Clear summary file
    _ensure_results_dir()
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"mlx-lm-server Comprehensive Benchmark\n")
        f.write(f"Model: {resolved_model}\n")
        f.write(f"Part:  {args.part}\n")
        f.write(f"Date:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

    t0 = time.time()

    parts = []
    if args.part == "all":
        parts = ["variation", "comparison", "correctness"]
    else:
        parts = [args.part]

    summary_lines: list[str] = []

    for part in parts:
        try:
            if part == "variation":
                results = run_part_variation(args.model)
                successes = sum(1 for r in results if r.get("error") is None)
                summary_lines.append(
                    f"Part 1 (Variation): {successes}/{len(results)} combinations succeeded\n"
                )
            elif part == "comparison":
                results = run_part_comparison(args.model)
                summary_lines.append(
                    f"Part 2 (Comparison): completed\n"
                )
                for key, val in results.items():
                    if isinstance(val, dict):
                        err = val.get("error")
                        if err:
                            summary_lines.append(f"  {key}: ERROR - {err}\n")
                        else:
                            summary_lines.append(f"  {key}: OK\n")
            elif part == "correctness":
                results = run_part_correctness(args.model)
                all_passed = results.get("all_passed", False)
                summary_lines.append(
                    f"Part 3 (Correctness): {'ALL PASSED' if all_passed else 'SOME FAILED'}\n"
                )
                for key, val in results.items():
                    if isinstance(val, dict) and "passed" in val:
                        summary_lines.append(
                            f"  {key}: {'PASS' if val['passed'] else 'FAIL'}\n"
                        )
        except Exception as e:
            summary_lines.append(f"Part {part}: FATAL ERROR - {e}\n")
            print(f"\n  FATAL ERROR in {part}: {e}")

    total_elapsed = time.time() - t0
    summary_lines.append(f"\nTotal time: {total_elapsed:.1f}s\n")

    # Write summary
    summary_text = "".join(summary_lines)
    _save_summary(summary_text)

    # Final output
    print()
    print("=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(summary_text)
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
