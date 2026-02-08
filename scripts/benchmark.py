#!/usr/bin/env python3
"""Comprehensive benchmark for mlx-lm-server.

Measures real model performance across three modes:
  1. Direct Scheduler  -- bypasses HTTP, measures raw inference
  2. HTTP Endpoint     -- full stack via /v1/chat/completions and /v1/completions
  3. Cache             -- cold vs warm prefix cache hit latency comparison

Uses mlx-community/Qwen3-4B-4bit (or a local path) by default.

Usage:
    python scripts/benchmark.py --mode scheduler --repeat 3 --warmup 2
    python scripts/benchmark.py --mode http --repeat 3 --warmup 1
    python scripts/benchmark.py --mode cache --repeat 3
    python scripts/benchmark.py --mode all --repeat 2
    python scripts/benchmark.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.kv_cache_manager import KVCacheManager
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.types import InferenceRequest, TokenEvent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"
LOCAL_MODEL_PATH = str(project_root / "Qwen3-4B-4bit")
PROMPT_TEMPLATES = {
    "short": "What is 2+2?",
    "medium": (
        "Explain the concept of machine learning in the context of modern "
        "artificial intelligence systems. Cover supervised learning, "
        "unsupervised learning, and reinforcement learning approaches. "
        "Discuss how neural networks and deep learning have transformed "
        "the field in the past decade."
    ),
    "long": (
        "Write a comprehensive analysis of the following topics in computer science. "
        "First, explain the fundamentals of operating systems including process management, "
        "memory management, file systems, and I/O handling. Then discuss how distributed "
        "systems work, covering consensus algorithms like Raft and Paxos, distributed "
        "hash tables, and eventually consistent data stores. Next, analyze the evolution "
        "of programming languages from assembly to modern high-level languages, covering "
        "type systems, memory safety, concurrency models, and functional programming "
        "paradigms. Also explain how compilers work, including lexing, parsing, semantic "
        "analysis, optimization passes, and code generation. Finally, discuss modern "
        "approaches to software engineering, including continuous integration, test-driven "
        "development, microservices architecture, and DevOps practices."
    ),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SingleRunResult:
    """Metrics from a single benchmark run (one request)."""

    test_name: str = ""
    prompt_len: int = 0
    max_tokens: int = 0
    concurrency: int = 1

    ttft_ms: float = 0.0         # Time to first token
    itl_ms_values: list[float] = field(default_factory=list)  # Per-token ITLs
    total_time_s: float = 0.0
    tokens_generated: int = 0

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
class BenchmarkSuite:
    """Collection of all benchmark results."""

    results: list[SingleRunResult] = field(default_factory=list)

    def add(self, r: SingleRunResult) -> None:
        self.results.append(r)

    def print_table(self) -> None:
        if not self.results:
            print("\nNo benchmark results collected.")
            return

        valid = [r for r in self.results if r.error is None]
        errored = [r for r in self.results if r.error is not None]

        header = (
            f"{'Test':<45} {'PLen':>5} {'MaxT':>5} {'Conc':>5} "
            f"{'TTFT':>9} {'ITL':>9} {'Tok/s':>9} {'Total':>9} {'Toks':>5}"
        )
        units = (
            f"{'':<45} {'':>5} {'':>5} {'':>5} "
            f"{'(ms)':>9} {'(ms)':>9} {'':>9} {'(s)':>9} {'':>5}"
        )
        sep = "-" * len(header)

        print("\n" + sep)
        print("  mlx-lm-server Benchmark Results")
        print(sep)
        print(header)
        print(units)
        print(sep)

        for r in valid:
            itl_str = f"{r.itl_ms_mean:.1f}" if r.itl_ms_values else "n/a"
            row = (
                f"{r.test_name:<45} "
                f"{r.prompt_len:>5} "
                f"{r.max_tokens:>5} "
                f"{r.concurrency:>5} "
                f"{r.ttft_ms:>9.1f} "
                f"{itl_str:>9} "
                f"{r.throughput_tps:>9.1f} "
                f"{r.total_time_s:>9.3f} "
                f"{r.tokens_generated:>5}"
            )
            print(row)

        print(sep)

        if errored:
            print(f"\n  Errors: {len(errored)}")
            for r in errored:
                print(f"    {r.test_name}: {r.error}")

        # Summary statistics
        if valid:
            ttft_vals = [r.ttft_ms for r in valid if r.ttft_ms > 0]
            itl_all = []
            for r in valid:
                itl_all.extend(r.itl_ms_values)
            tps_vals = [r.throughput_tps for r in valid if r.throughput_tps > 0]

            print(f"\n{'Summary':>45}")
            print(sep)
            if ttft_vals:
                print(
                    f"  TTFT (ms):   min={min(ttft_vals):.1f}  "
                    f"avg={statistics.mean(ttft_vals):.1f}  "
                    f"max={max(ttft_vals):.1f}  "
                    f"p50={_percentile(ttft_vals, 50):.1f}  "
                    f"p95={_percentile(ttft_vals, 95):.1f}"
                )
            if itl_all:
                print(
                    f"  ITL  (ms):   min={min(itl_all):.1f}  "
                    f"avg={statistics.mean(itl_all):.1f}  "
                    f"max={max(itl_all):.1f}  "
                    f"p50={_percentile(itl_all, 50):.1f}  "
                    f"p95={_percentile(itl_all, 95):.1f}"
                )
            if tps_vals:
                print(
                    f"  Tok/s:       min={min(tps_vals):.1f}  "
                    f"avg={statistics.mean(tps_vals):.1f}  "
                    f"max={max(tps_vals):.1f}  "
                    f"p50={_percentile(tps_vals, 50):.1f}  "
                    f"p95={_percentile(tps_vals, 95):.1f}"
                )
            print(sep + "\n")


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[Any, Any]] = {}


def load_model(model_path: str) -> tuple[Any, Any]:
    """Load and cache the model and tokenizer."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    # Resolve local path
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
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler(
    model: Any,
    tokenizer: Any,
    model_path: str,
    max_batch_size: int = 8,
    num_blocks: int = 256,
    block_size: int = 16,
) -> tuple[Scheduler, KVCacheManager]:
    """Create a scheduler + KV cache manager with real model."""
    config = ServerConfig(
        model=model_path,
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_queue_size=64,
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


def _make_inference_request(
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    request_id: str | None = None,
    stream: bool = False,
    temperature: float = 0.0,
) -> InferenceRequest:
    """Build an InferenceRequest from a text prompt."""
    rid = request_id or f"bench-{uuid.uuid4().hex[:8]}"
    tokens = tokenizer.encode(prompt)
    return InferenceRequest(
        request_id=rid,
        prompt_tokens=tokens,
        max_tokens=max_tokens,
        stream=stream,
        temperature=temperature,
    )


def _collect_streaming(
    sched: Scheduler,
    req: InferenceRequest,
    timeout: float = 120.0,
) -> SingleRunResult:
    """Submit request in streaming mode, collect TTFT/ITL/total metrics."""
    result = SingleRunResult()

    stream_q = sched.register_stream(req.request_id)
    submit_time = time.perf_counter()
    sched.submit_request(req)

    events: list[TokenEvent] = []
    first_token_time: float | None = None
    prev_token_time = submit_time
    itl_values: list[float] = []

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev = stream_q.get(timeout=min(5.0, deadline - time.time()))
        except queue.Empty:
            continue

        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
            result.ttft_ms = (first_token_time - submit_time) * 1000
        else:
            itl_values.append((now - prev_token_time) * 1000)

        prev_token_time = now
        events.append(ev)

        if ev.finish_reason is not None:
            break

    end_time = time.perf_counter()
    result.total_time_s = end_time - submit_time
    result.tokens_generated = len(events)
    result.itl_ms_values = itl_values
    result.prompt_len = len(req.prompt_tokens)
    result.max_tokens = req.max_tokens

    return result


def _collect_blocking(
    sched: Scheduler,
    req: InferenceRequest,
    timeout: float = 120.0,
) -> SingleRunResult:
    """Submit request in non-streaming mode, measure total latency."""
    result = SingleRunResult()

    submit_time = time.perf_counter()
    sched.submit_request(req)
    events = sched.get_result(req.request_id, timeout=timeout)
    end_time = time.perf_counter()

    result.total_time_s = end_time - submit_time
    result.tokens_generated = len(events)
    result.prompt_len = len(req.prompt_tokens)
    result.max_tokens = req.max_tokens

    # For non-streaming, TTFT is approximately the total time minus
    # (n-1) * avg_itl, but we only have total time. Report total / n as proxy.
    if events:
        result.ttft_ms = result.total_time_s * 1000 / max(len(events), 1)

    return result


# ---------------------------------------------------------------------------
# Mode 1: Direct Scheduler Benchmark
# ---------------------------------------------------------------------------


def bench_scheduler(
    model_path: str,
    warmup: int,
    repeat: int,
    suite: BenchmarkSuite,
) -> None:
    """Run direct scheduler benchmarks (no HTTP overhead)."""
    print("\n" + "=" * 60)
    print("  MODE: Direct Scheduler Benchmark")
    print("=" * 60)

    model, tokenizer = load_model(model_path)

    prompt_configs = [
        ("short", 32),
        ("medium", 64),
        ("long", 128),
    ]

    concurrency_levels = [1, 2, 4, 8]

    for prompt_name, max_tokens in prompt_configs:
        prompt_text = PROMPT_TEMPLATES[prompt_name]
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_tokens)

        for conc in concurrency_levels:
            test_name = f"sched/{prompt_name}(p{prompt_len})/mt{max_tokens}/c{conc}"
            print(f"\n  Running: {test_name} (warmup={warmup}, repeat={repeat})")

            sched, kv_mgr = _make_scheduler(
                model, tokenizer, model_path, max_batch_size=max(conc, 4)
            )

            try:
                # --- Warmup ---
                for w in range(warmup):
                    _run_concurrent_scheduler(
                        sched, tokenizer, prompt_text, max_tokens,
                        conc, tag=f"warmup-{w}"
                    )

                # --- Measured runs ---
                for run_i in range(repeat):
                    run_results = _run_concurrent_scheduler(
                        sched, tokenizer, prompt_text, max_tokens,
                        conc, tag=f"run-{run_i}"
                    )
                    for r in run_results:
                        r.test_name = test_name
                        r.concurrency = conc
                        suite.add(r)
            finally:
                sched.stop()


def _run_concurrent_scheduler(
    sched: Scheduler,
    tokenizer: Any,
    prompt_text: str,
    max_tokens: int,
    concurrency: int,
    tag: str = "",
) -> list[SingleRunResult]:
    """Submit `concurrency` requests concurrently and collect results."""
    results: list[SingleRunResult] = []
    errors: list[str] = []
    lock = threading.Lock()

    def _worker(idx: int) -> None:
        rid = f"bench-{tag}-{idx}-{uuid.uuid4().hex[:6]}"
        req = _make_inference_request(
            tokenizer, prompt_text, max_tokens,
            request_id=rid, stream=True, temperature=0.0,
        )
        try:
            r = _collect_streaming(sched, req)
            with lock:
                results.append(r)
        except Exception as e:
            with lock:
                r = SingleRunResult(error=str(e))
                results.append(r)

    threads = []
    for i in range(concurrency):
        t = threading.Thread(target=_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=180.0)

    return results


# ---------------------------------------------------------------------------
# Mode 2: HTTP Endpoint Benchmark
# ---------------------------------------------------------------------------


def bench_http(
    model_path: str,
    warmup: int,
    repeat: int,
    suite: BenchmarkSuite,
    port: int = 8000,
) -> None:
    """Run HTTP endpoint benchmarks (full stack)."""
    print("\n" + "=" * 60)
    print("  MODE: HTTP Endpoint Benchmark")
    print("=" * 60)

    model, tokenizer = load_model(model_path)

    # Resolve model path for config
    resolved = model_path
    if Path(LOCAL_MODEL_PATH).is_dir() and model_path == DEFAULT_MODEL:
        resolved = LOCAL_MODEL_PATH

    asyncio.run(_bench_http_async(
        model, tokenizer, resolved, warmup, repeat, suite
    ))


async def _bench_http_async(
    model: Any,
    tokenizer: Any,
    model_path: str,
    warmup: int,
    repeat: int,
    suite: BenchmarkSuite,
) -> None:
    """Async HTTP benchmark runner."""
    import httpx
    from httpx import ASGITransport
    from mlx_lm_server.server import create_app

    config = ServerConfig(
        model=model_path,
        block_size=16,
        num_blocks=256,
        max_batch_size=8,
        max_queue_size=64,
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
    app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
    transport = ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    try:
        # Test configurations: (prompt_name, endpoint, stream, max_tokens, concurrency)
        test_configs = [
            # Chat completions, non-streaming
            ("short", "/v1/chat/completions", False, 32, 1),
            ("short", "/v1/chat/completions", False, 32, 4),
            ("medium", "/v1/chat/completions", False, 64, 1),
            ("medium", "/v1/chat/completions", False, 64, 4),
            # Chat completions, streaming
            ("short", "/v1/chat/completions", True, 32, 1),
            ("short", "/v1/chat/completions", True, 32, 4),
            ("medium", "/v1/chat/completions", True, 64, 1),
            # Completions endpoint
            ("short", "/v1/completions", False, 32, 1),
            ("medium", "/v1/completions", False, 64, 1),
            ("short", "/v1/completions", True, 32, 1),
        ]

        for prompt_name, endpoint, stream, max_tokens, conc in test_configs:
            prompt_text = PROMPT_TEMPLATES[prompt_name]
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt_len = len(prompt_tokens)
            ep_short = endpoint.split("/")[-1]
            stream_tag = "stream" if stream else "sync"
            test_name = f"http/{ep_short}/{prompt_name}(p{prompt_len})/{stream_tag}/c{conc}"
            print(f"\n  Running: {test_name} (warmup={warmup}, repeat={repeat})")

            # --- Warmup ---
            for _ in range(warmup):
                await _run_http_concurrent(
                    client, endpoint, prompt_name, prompt_text,
                    model_path, max_tokens, stream, conc, tokenizer
                )

            # --- Measured runs ---
            for run_i in range(repeat):
                run_results = await _run_http_concurrent(
                    client, endpoint, prompt_name, prompt_text,
                    model_path, max_tokens, stream, conc, tokenizer
                )
                for r in run_results:
                    r.test_name = test_name
                    r.concurrency = conc
                    suite.add(r)
    finally:
        await client.aclose()
        sched.stop()


async def _run_http_concurrent(
    client: Any,
    endpoint: str,
    prompt_name: str,
    prompt_text: str,
    model_path: str,
    max_tokens: int,
    stream: bool,
    concurrency: int,
    tokenizer: Any,
) -> list[SingleRunResult]:
    """Fire concurrent HTTP requests and collect metrics."""
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_tokens)

    async def _single_request(idx: int) -> SingleRunResult:
        result = SingleRunResult(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
        )

        if endpoint == "/v1/chat/completions":
            body: dict[str, Any] = {
                "model": model_path,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
                "stream": stream,
                "temperature": 0.0,
            }
        else:
            body = {
                "model": model_path,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "stream": stream,
                "temperature": 0.0,
            }

        try:
            submit_time = time.perf_counter()
            resp = await client.post(endpoint, json=body, timeout=120.0)

            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            if stream:
                # Parse SSE events for streaming
                end_time = time.perf_counter()
                result.total_time_s = end_time - submit_time

                body_text = resp.text
                lines = [l for l in body_text.strip().split("\n") if l.strip()]
                chunks = []
                for line in lines:
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            pass

                result.tokens_generated = len(chunks)
                # For streaming over HTTP with httpx non-streaming read,
                # we measure total time. TTFT approximation: total / tokens
                if result.tokens_generated > 0:
                    per_tok = result.total_time_s * 1000 / result.tokens_generated
                    result.ttft_ms = per_tok  # rough approximation
            else:
                end_time = time.perf_counter()
                result.total_time_s = end_time - submit_time

                data = resp.json()
                usage = data.get("usage", {})
                result.tokens_generated = usage.get("completion_tokens", 0)
                if result.tokens_generated > 0:
                    result.ttft_ms = result.total_time_s * 1000 / result.tokens_generated

        except Exception as e:
            result.error = str(e)

        return result

    tasks = [_single_request(i) for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# Mode 3: Cache Benchmark
# ---------------------------------------------------------------------------


def bench_cache(
    model_path: str,
    warmup: int,
    repeat: int,
    suite: BenchmarkSuite,
) -> None:
    """Benchmark cold start vs warm cache performance."""
    print("\n" + "=" * 60)
    print("  MODE: Cache Benchmark (Cold vs Warm)")
    print("=" * 60)

    model, tokenizer = load_model(model_path)

    prompt_text = PROMPT_TEMPLATES["medium"]
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_tokens)
    max_tokens = 32

    # --- Cold start benchmark ---
    print(f"\n  Cold start: new scheduler, no cached prefixes")
    for run_i in range(repeat):
        sched, kv_mgr = _make_scheduler(model, tokenizer, model_path)
        try:
            test_name = f"cache/cold/p{prompt_len}/mt{max_tokens}"
            req = _make_inference_request(
                tokenizer, prompt_text, max_tokens,
                stream=True, temperature=0.0,
            )
            r = _collect_streaming(sched, req)
            r.test_name = test_name
            r.concurrency = 1

            # Report cache stats
            cached_before = kv_mgr.find_cached_prefix(prompt_tokens)
            print(
                f"    Run {run_i}: TTFT={r.ttft_ms:.1f}ms, "
                f"total={r.total_time_s:.3f}s, "
                f"tokens={r.tokens_generated}, "
                f"prefix_cached={cached_before}"
            )
            suite.add(r)
        finally:
            sched.stop()

    # --- Warm cache benchmark ---
    print(f"\n  Warm cache: same prompt repeated, prefix cache should hit")
    sched, kv_mgr = _make_scheduler(model, tokenizer, model_path)
    try:
        # First request to populate cache
        print("    Populating cache with initial request ...")
        req_init = _make_inference_request(
            tokenizer, prompt_text, max_tokens,
            request_id="cache-init", stream=True, temperature=0.0,
        )
        _collect_streaming(sched, req_init)
        time.sleep(1.0)  # Allow block decomposition

        cached_after_init = kv_mgr.find_cached_prefix(prompt_tokens)
        print(f"    Cache populated: {cached_after_init} tokens cached")

        # Warmup
        for w in range(warmup):
            req_w = _make_inference_request(
                tokenizer, prompt_text, max_tokens,
                stream=True, temperature=0.0,
            )
            _collect_streaming(sched, req_w)
            time.sleep(0.3)

        # Measured warm runs
        for run_i in range(repeat):
            test_name = f"cache/warm/p{prompt_len}/mt{max_tokens}"
            req = _make_inference_request(
                tokenizer, prompt_text, max_tokens,
                stream=True, temperature=0.0,
            )
            r = _collect_streaming(sched, req)
            r.test_name = test_name
            r.concurrency = 1

            cached_now = kv_mgr.find_cached_prefix(prompt_tokens)
            print(
                f"    Run {run_i}: TTFT={r.ttft_ms:.1f}ms, "
                f"total={r.total_time_s:.3f}s, "
                f"tokens={r.tokens_generated}, "
                f"prefix_cached={cached_now}"
            )
            suite.add(r)
            time.sleep(0.3)

        # Summary comparison
        cold_results = [
            r for r in suite.results
            if r.test_name.startswith("cache/cold") and r.error is None
        ]
        warm_results = [
            r for r in suite.results
            if r.test_name.startswith("cache/warm") and r.error is None
        ]
        if cold_results and warm_results:
            cold_ttft_avg = statistics.mean([r.ttft_ms for r in cold_results])
            warm_ttft_avg = statistics.mean([r.ttft_ms for r in warm_results])
            cold_total_avg = statistics.mean([r.total_time_s for r in cold_results])
            warm_total_avg = statistics.mean([r.total_time_s for r in warm_results])

            print(f"\n  Cache Hit Summary:")
            print(f"    Cold avg TTFT:  {cold_ttft_avg:.1f} ms")
            print(f"    Warm avg TTFT:  {warm_ttft_avg:.1f} ms")
            if cold_ttft_avg > 0:
                speedup = cold_ttft_avg / max(warm_ttft_avg, 0.01)
                print(f"    TTFT speedup:   {speedup:.2f}x")
            print(f"    Cold avg total: {cold_total_avg:.3f} s")
            print(f"    Warm avg total: {warm_total_avg:.3f} s")
            if cold_total_avg > 0:
                speedup = cold_total_avg / max(warm_total_avg, 0.001)
                print(f"    Total speedup:  {speedup:.2f}x")

            # Report cache hit rate
            stats = sched.get_cache_stats()
            print(f"\n  Cache stats: {stats}")
    finally:
        sched.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for mlx-lm-server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["scheduler", "http", "cache", "all"],
        default="all",
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name or path",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP benchmarks",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs before measuring",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of measured runs per configuration",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  mlx-lm-server Comprehensive Benchmark")
    print(f"  Model:   {args.model}")
    print(f"  Mode:    {args.mode}")
    print(f"  Warmup:  {args.warmup}")
    print(f"  Repeat:  {args.repeat}")
    print("=" * 60)

    suite = BenchmarkSuite()
    modes = [args.mode] if args.mode != "all" else ["scheduler", "http", "cache"]

    t0 = time.time()

    for mode in modes:
        if mode == "scheduler":
            bench_scheduler(args.model, args.warmup, args.repeat, suite)
        elif mode == "http":
            bench_http(args.model, args.warmup, args.repeat, suite, port=args.port)
        elif mode == "cache":
            bench_cache(args.model, args.warmup, args.repeat, suite)

    total_elapsed = time.time() - t0

    suite.print_table()

    print(f"  Total benchmark time: {total_elapsed:.1f}s")
    print(f"  Results collected: {len(suite.results)} "
          f"({len([r for r in suite.results if r.error])} errors)\n")


if __name__ == "__main__":
    main()
