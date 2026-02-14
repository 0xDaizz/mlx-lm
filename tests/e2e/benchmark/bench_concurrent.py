#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Concurrent request benchmark: measure scheduler under load.

Tests 1, 3, 5, 8 concurrent streaming requests and measures per-request
TTFT, tok/s, and aggregate throughput. Also checks for 429/503 errors.

Usage:
    python bench_concurrent.py [--server-url http://localhost:8080] [--max-tokens 128]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

RESULTS_DIR = Path("/tmp/kimi-bench-results")


def get_model_name(server_url: str) -> str:
    """Fetch the loaded model name from the server."""
    resp = httpx.get(f"{server_url}/v1/models", timeout=10.0)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError("No models loaded on server")
    return models[0]["id"]

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list using merge sort.",
    "What are the main differences between REST and GraphQL?",
    "Describe the water cycle in detail.",
    "Translate the following to Korean: The future of AI is collaborative.",
    "What is the time complexity of binary search? Explain step by step.",
    "Write a haiku about programming.",
    "List the top 10 programming languages in 2025.",
]

CONCURRENCY_LEVELS = [1, 3, 5, 8]


def parse_sse_stream(response: httpx.Response):
    """Yield parsed SSE data events."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def single_streaming_request(
    url: str, prompt: str, max_tokens: int, request_idx: int, model: str = "default"
) -> dict:
    """Execute a single streaming chat request and measure performance."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft = None
    token_count = 0
    finish_reason = None
    usage_info = {}
    error = None

    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
            with client.stream(
                "POST", f"{url}/v1/chat/completions", json=payload, timeout=600.0
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    return {
                        "request_idx": request_idx,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "status_code": resp.status_code,
                        "total_time_s": time.perf_counter() - t0,
                    }

                for chunk in parse_sse_stream(resp):
                    now = time.perf_counter()

                    if "usage" in chunk and not chunk.get("choices"):
                        usage_info = chunk["usage"]
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    fr = choices[0].get("finish_reason")

                    if content:
                        if ttft is None:
                            ttft = now - t0
                        token_count += 1

                    if fr is not None:
                        finish_reason = fr

    except Exception as e:
        error = str(e)

    total_time = time.perf_counter() - t0
    completion_tokens = usage_info.get("completion_tokens", token_count)
    prompt_tokens = usage_info.get("prompt_tokens", 0)
    gen_time = total_time - (ttft or 0)
    gen_tok_s = round(completion_tokens / gen_time, 2) if gen_time > 0 else 0

    return {
        "request_idx": request_idx,
        "prompt_preview": prompt[:50],
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_time_s": round(total_time, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_tok_s": gen_tok_s,
        "finish_reason": finish_reason,
        "error": error,
    }


def get_health(url: str) -> dict | None:
    """Fetch server health stats."""
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
            resp = client.get(f"{url}/health", timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def run_concurrent_batch(
    url: str, concurrency: int, max_tokens: int, model: str = "default"
) -> dict:
    """Run N concurrent requests and collect results."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]

    print(f"\n--- Concurrency={concurrency} ---")

    health_before = get_health(url)
    batch_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(single_streaming_request, url, p, max_tokens, i, model): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            idx = r["request_idx"]
            if r.get("error"):
                print(f"  req[{idx}]: ERROR - {r['error'][:60]}")
            else:
                print(
                    f"  req[{idx}]: TTFT={r['ttft_s']}s, "
                    f"{r['generation_tok_s']} tok/s, "
                    f"{r['completion_tokens']} tokens, "
                    f"{r['total_time_s']}s total"
                )

    batch_time = time.perf_counter() - batch_start
    health_after = get_health(url)

    # Compute aggregates
    successful = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]

    ttfts = [r["ttft_s"] for r in successful if r.get("ttft_s") is not None]
    gen_toks = [r["generation_tok_s"] for r in successful]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    total_times = [r["total_time_s"] for r in successful]

    status_codes = [r.get("status_code", 200) for r in errors]
    count_429 = status_codes.count(429)
    count_503 = status_codes.count(503)

    aggregate = {
        "concurrency": concurrency,
        "batch_time_s": round(batch_time, 4),
        "successful": len(successful),
        "errors": len(errors),
        "count_429": count_429,
        "count_503": count_503,
        "total_completion_tokens": total_tokens,
        "aggregate_tok_s": round(total_tokens / batch_time, 2) if batch_time > 0 else 0,
    }

    if ttfts:
        aggregate["ttft_p50"] = round(statistics.median(ttfts), 4)
        aggregate["ttft_p95"] = round(sorted(ttfts)[int(len(ttfts) * 0.95)], 4) if len(ttfts) >= 2 else ttfts[0]
        aggregate["ttft_mean"] = round(statistics.mean(ttfts), 4)

    if total_times:
        sorted_times = sorted(total_times)
        aggregate["latency_p50"] = round(statistics.median(sorted_times), 4)
        aggregate["latency_p95"] = round(sorted_times[int(len(sorted_times) * 0.95)], 4) if len(sorted_times) >= 2 else sorted_times[0]
        aggregate["latency_p99"] = round(sorted_times[int(len(sorted_times) * 0.99)], 4) if len(sorted_times) >= 2 else sorted_times[0]

    if gen_toks:
        aggregate["per_request_tok_s_mean"] = round(statistics.mean(gen_toks), 2)

    print(f"  Aggregate: {aggregate['aggregate_tok_s']} tok/s, "
          f"{aggregate['successful']}/{concurrency} succeeded, "
          f"{aggregate['batch_time_s']}s batch time")

    return {
        "aggregate": aggregate,
        "per_request": sorted(results, key=lambda r: r["request_idx"]),
        "health_before": health_before,
        "health_after": health_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrent request benchmark")
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--levels", type=str, default=None,
                        help="Comma-separated concurrency levels (default: 1,3,5,8)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    url = args.server_url.rstrip("/")
    levels = [int(x) for x in args.levels.split(",")] if args.levels else CONCURRENCY_LEVELS

    health = get_health(url)
    if health is None:
        print(f"ERROR: Server not reachable at {url}", file=sys.stderr)
        sys.exit(1)
    print(f"Server health: {health.get('status', 'unknown')}")

    # Fetch model name from server
    model_name = get_model_name(url)
    print(f"Model: {model_name}")

    all_results = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def _save_partial():
        pf = RESULTS_DIR / f"bench_concurrent_{ts}_partial.json"
        pf.write_text(json.dumps(
            {"benchmark": "bench_concurrent", "results": all_results, "partial": True},
            indent=2, default=str,
        ))

    try:
        for level in levels:
            batch_result = run_concurrent_batch(url, level, args.max_tokens, model=model_name)
            all_results.append(batch_result)
            _save_partial()
            # Brief pause between batches
            time.sleep(2)
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        _save_partial()
        print(f"Partial results ({len(all_results)} levels) saved", file=sys.stderr)
        raise

    # Save results
    output = {
        "benchmark": "bench_concurrent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "max_tokens": args.max_tokens,
        "concurrency_levels": levels,
        "results": all_results,
    }
    outfile = RESULTS_DIR / f"bench_concurrent_{ts}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {outfile}")

    partial_file = RESULTS_DIR / f"bench_concurrent_{ts}_partial.json"
    if partial_file.exists():
        partial_file.unlink()

    # Summary table
    print("\n=== Concurrency Summary ===")
    print(f"{'Level':>6} | {'Agg tok/s':>10} | {'TTFT p50':>9} | {'Lat p50':>9} | {'OK/Total':>9} | {'429s':>5}")
    print("-" * 65)
    for r in all_results:
        a = r["aggregate"]
        ttft = f"{a.get('ttft_p50', '?'):.3f}s" if isinstance(a.get('ttft_p50'), float) else "N/A"
        lat = f"{a.get('latency_p50', '?'):.3f}s" if isinstance(a.get('latency_p50'), float) else "N/A"
        print(
            f"{a['concurrency']:>6} | "
            f"{a['aggregate_tok_s']:>10.1f} | "
            f"{ttft:>9} | "
            f"{lat:>9} | "
            f"{a['successful']}/{a['concurrency']:>7} | "
            f"{a.get('count_429', 0):>5}"
        )


if __name__ == "__main__":
    main()
