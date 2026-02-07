#!/usr/bin/env python3
"""Benchmark script for mlx-lm-server.

Measures throughput, TTFT (time to first token), ITL (inter-token latency),
and cache hit rates using a mock model.

Usage:
    python scripts/benchmark.py --num-requests 100 --concurrency 4
    python scripts/benchmark.py --num-requests 50 --concurrency 8 --max-tokens 20
    python scripts/benchmark.py --help
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.kv_cache_manager import KVCacheManager
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.types import InferenceRequest, TokenEvent


# ---------------------------------------------------------------------------
# Benchmark result data class
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Collects and summarizes benchmark metrics."""

    num_requests: int = 0
    concurrency: int = 0
    max_tokens: int = 0
    prompt_length: int = 0

    # Per-request metrics
    ttft_ms: list[float] = field(default_factory=list)
    itl_ms: list[float] = field(default_factory=list)
    request_latency_ms: list[float] = field(default_factory=list)
    tokens_generated: list[int] = field(default_factory=list)

    # Aggregate
    total_time_s: float = 0.0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    def throughput_rps(self) -> float:
        """Requests per second."""
        if self.total_time_s <= 0:
            return 0.0
        return self.num_requests / self.total_time_s

    def throughput_tps(self) -> float:
        """Tokens per second."""
        if self.total_time_s <= 0:
            return 0.0
        return self.total_tokens / self.total_time_s

    def cache_hit_rate(self) -> float:
        """Cache hit ratio (0 to 1)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def print_table(self) -> None:
        """Print a formatted results table."""
        sep = "-" * 60
        print("\n" + sep)
        print("  mlx-lm-server Benchmark Results")
        print(sep)
        print(f"  Requests:            {self.num_requests}")
        print(f"  Concurrency:         {self.concurrency}")
        print(f"  Prompt length:       {self.prompt_length} tokens")
        print(f"  Max tokens:          {self.max_tokens}")
        print(f"  Errors:              {self.errors}")
        print(sep)
        print(f"  Total time:          {self.total_time_s:.3f} s")
        print(f"  Total tokens:        {self.total_tokens}")
        print(f"  Throughput (req/s):   {self.throughput_rps():.2f}")
        print(f"  Throughput (tok/s):   {self.throughput_tps():.2f}")
        print(sep)

        if self.ttft_ms:
            print(f"  TTFT (ms):")
            print(f"    p50:               {_percentile(self.ttft_ms, 50):.2f}")
            print(f"    p95:               {_percentile(self.ttft_ms, 95):.2f}")
            print(f"    p99:               {_percentile(self.ttft_ms, 99):.2f}")
            print(f"    mean:              {statistics.mean(self.ttft_ms):.2f}")

        if self.itl_ms:
            print(f"  ITL (ms):")
            print(f"    p50:               {_percentile(self.itl_ms, 50):.2f}")
            print(f"    p95:               {_percentile(self.itl_ms, 95):.2f}")
            print(f"    p99:               {_percentile(self.itl_ms, 99):.2f}")
            print(f"    mean:              {statistics.mean(self.itl_ms):.2f}")

        if self.request_latency_ms:
            print(f"  Request latency (ms):")
            print(f"    p50:               {_percentile(self.request_latency_ms, 50):.2f}")
            print(f"    p95:               {_percentile(self.request_latency_ms, 95):.2f}")
            print(f"    p99:               {_percentile(self.request_latency_ms, 99):.2f}")
            print(f"    mean:              {statistics.mean(self.request_latency_ms):.2f}")

        print(sep)
        print(f"  Cache hits:          {self.cache_hits}")
        print(f"  Cache misses:        {self.cache_misses}")
        print(f"  Cache hit rate:      {self.cache_hit_rate():.2%}")
        print(sep + "\n")


def _percentile(data: list[float], pct: float) -> float:
    """Calculate percentile from a sorted or unsorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    num_requests: int = 100,
    concurrency: int = 4,
    max_tokens: int = 10,
    prompt_length: int = 32,
    block_size: int = 4,
    num_blocks: int = 256,
    shared_prefix_ratio: float = 0.5,
) -> BenchmarkResult:
    """Run the benchmark.

    Args:
        num_requests: Total number of requests to send.
        concurrency: Maximum concurrent requests.
        max_tokens: Number of tokens to generate per request.
        prompt_length: Number of tokens in each prompt.
        block_size: KV cache block size.
        num_blocks: Total KV cache blocks.
        shared_prefix_ratio: Fraction of prompt tokens shared across requests (0.0-1.0).

    Returns:
        BenchmarkResult with all collected metrics.
    """
    config = ServerConfig(
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_size=concurrency,
        max_queue_size=num_requests + 10,
        prefill_batch_size=concurrency,
    )

    kv_mgr = KVCacheManager(config)
    scheduler = Scheduler(
        config=config,
        model=None,
        tokenizer=None,
        kv_cache_manager=kv_mgr,
    )

    # Track cache hits/misses
    cache_hits = 0
    cache_misses = 0
    cache_lock = threading.Lock()

    # Mock generator with realistic-ish timing
    def mock_gen(request_id, token_ids, step):
        # Small sleep to simulate compute time
        time.sleep(0.0001)
        if step >= max_tokens - 1:
            return (step + 100, f"t{step}", "stop")
        return (step + 100, f"t{step}", None)

    scheduler._mock_generate = mock_gen

    # Generate prompt tokens with shared prefix
    shared_len = int(prompt_length * shared_prefix_ratio)
    # Align to block_size
    shared_len = (shared_len // block_size) * block_size
    shared_prefix = list(range(1, shared_len + 1))

    def make_prompt(idx: int) -> list[int]:
        """Create prompt tokens: shared prefix + unique suffix."""
        unique_len = prompt_length - shared_len
        unique_suffix = list(range(idx * 1000 + 1, idx * 1000 + unique_len + 1))
        return shared_prefix + unique_suffix

    # Pre-populate cache with shared prefix (simulating a previous request)
    if shared_len > 0:
        kv_mgr.allocate_blocks(shared_prefix)
        kv_mgr.free_blocks(
            kv_mgr.allocate_blocks(shared_prefix)
        )

    result = BenchmarkResult(
        num_requests=num_requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        prompt_length=prompt_length,
    )

    # Start inference loop
    scheduler.run_inference_loop(blocking=False)

    # Semaphore to limit concurrency
    sem = threading.Semaphore(concurrency)
    all_done = threading.Event()
    completed = [0]
    completed_lock = threading.Lock()

    def run_request(idx: int):
        nonlocal cache_hits, cache_misses
        sem.acquire()
        try:
            rid = f"bench-{idx}"
            prompt = make_prompt(idx)

            # Check prefix cache before submitting
            cached = kv_mgr.find_cached_prefix(prompt)
            with cache_lock:
                if cached > 0:
                    cache_hits += 1
                else:
                    cache_misses += 1

            req = InferenceRequest(
                request_id=rid,
                prompt_tokens=prompt,
                max_tokens=max_tokens,
            )

            # Register stream for TTFT/ITL measurement
            stream_q = scheduler.register_stream(rid)
            submit_time = time.perf_counter()
            scheduler.submit_request(req)

            # Collect tokens and measure timing
            events: list[TokenEvent] = []
            first_token_time = None
            prev_token_time = submit_time
            itl_values: list[float] = []

            while True:
                ev = stream_q.get(timeout=30.0)
                now = time.perf_counter()

                if first_token_time is None:
                    first_token_time = now
                    result.ttft_ms.append((first_token_time - submit_time) * 1000)
                else:
                    itl_values.append((now - prev_token_time) * 1000)

                prev_token_time = now
                events.append(ev)

                if ev.finish_reason is not None:
                    break

            end_time = time.perf_counter()
            result.request_latency_ms.append((end_time - submit_time) * 1000)
            result.tokens_generated.append(len(events))
            result.itl_ms.extend(itl_values)

        except Exception as e:
            result.errors += 1
            print(f"  Error in request {idx}: {e}", file=sys.stderr)
        finally:
            sem.release()
            with completed_lock:
                completed[0] += 1
                if completed[0] >= num_requests:
                    all_done.set()

    # Launch all request threads
    start_time = time.perf_counter()
    threads = []
    for i in range(num_requests):
        t = threading.Thread(target=run_request, args=(i,), daemon=True)
        threads.append(t)
        t.start()

    # Wait for all to complete
    all_done.wait(timeout=120.0)
    end_time = time.perf_counter()

    # Wait for threads to finish
    for t in threads:
        t.join(timeout=5.0)

    scheduler.stop()

    result.total_time_s = end_time - start_time
    result.total_tokens = sum(result.tokens_generated)
    result.cache_hits = cache_hits
    result.cache_misses = cache_misses

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="mlx-lm-server benchmark (mock model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-requests", type=int, default=100,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Maximum concurrent requests",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=10,
        help="Tokens to generate per request",
    )
    parser.add_argument(
        "--prompt-length", type=int, default=32,
        help="Number of tokens in each prompt",
    )
    parser.add_argument(
        "--block-size", type=int, default=4,
        help="KV cache block size",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=256,
        help="Total KV cache blocks in pool",
    )
    parser.add_argument(
        "--shared-prefix-ratio", type=float, default=0.5,
        help="Fraction of prompt shared across requests (0.0-1.0)",
    )

    args = parser.parse_args()

    print(f"Running benchmark: {args.num_requests} requests, "
          f"concurrency={args.concurrency}, max_tokens={args.max_tokens}")

    result = run_benchmark(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        prompt_length=args.prompt_length,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        shared_prefix_ratio=args.shared_prefix_ratio,
    )

    result.print_table()

    # Exit with error if there were failures
    if result.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
