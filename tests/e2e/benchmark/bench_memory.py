#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Memory usage benchmark: compare baseline vs FP8 KV cache.

Collects memory and cache statistics from the /health endpoint before
and after running inference requests. Designed to be run against servers
with different --kv-bits settings for comparison.

Usage:
    python bench_memory.py [--server-url http://localhost:8080] [--max-tokens 256]
    python bench_memory.py --label baseline   # for default kv_bits
    python bench_memory.py --label fp8        # for --kv-bits 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

TEST_PROMPTS = [
    {
        "name": "short",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    },
    {
        "name": "medium",
        "messages": [{"role": "user", "content": (
            "Explain the difference between TCP and UDP protocols. "
            "Cover reliability, ordering, speed, and common use cases."
        )}],
    },
    {
        "name": "long_context",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that gives detailed answers."},
            {"role": "user", "content": (
                "Write a comprehensive guide on distributed computing. Cover: "
                "1) CAP theorem and its implications "
                "2) Consensus algorithms (Paxos, Raft) "
                "3) Distributed hash tables "
                "4) MapReduce paradigm "
                "5) Modern approaches like CRDTs "
                "Be thorough and include examples."
            )},
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def get_health(client: httpx.Client, url: str) -> dict | None:
    """Fetch health stats."""
    try:
        resp = client.get(f"{url}/health", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_metrics(client: httpx.Client, url: str) -> str | None:
    """Fetch Prometheus metrics."""
    try:
        resp = client.get(f"{url}/metrics", timeout=10.0)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def streaming_chat(
    client: httpx.Client, url: str, messages: list[dict], max_tokens: int, model: str = "default"
) -> dict:
    """Run a streaming chat request and collect metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft = None
    token_count = 0
    usage_info = {}

    with client.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=600.0) as resp:
        if resp.status_code != 200:
            resp.read()
            return {
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
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
            if delta.get("content"):
                if ttft is None:
                    ttft = now - t0
                token_count += 1

    total_time = time.perf_counter() - t0
    completion_tokens = usage_info.get("completion_tokens", token_count)
    prompt_tokens = usage_info.get("prompt_tokens", 0)
    gen_time = total_time - (ttft or 0)

    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_time_s": round(total_time, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_tok_s": round(completion_tokens / gen_time, 2) if gen_time > 0 else 0,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory usage benchmark")
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--label", type=str, default="default",
                        help="Label for this config (e.g. 'baseline', 'fp8')")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    url = args.server_url.rstrip("/")
    client = httpx.Client()

    # Initial health check
    health_initial = get_health(client, url)
    if health_initial is None:
        print(f"ERROR: Server not reachable at {url}", file=sys.stderr)
        sys.exit(1)
    print(f"Server health: {health_initial.get('status', 'unknown')}")

    # Fetch model name from server
    model_name = get_model_name(url)
    print(f"Model: {model_name}")

    metrics_initial = get_metrics(client, url)

    print(f"\n=== Memory Benchmark (label={args.label}) ===")
    print(f"Initial cache stats:")
    cache = health_initial.get("cache_stats", {})
    print(f"  used_blocks: {cache.get('used_blocks', 'N/A')}")
    print(f"  free_blocks: {cache.get('free_blocks', 'N/A')}")
    print(f"  total_blocks: {cache.get('total_blocks', 'N/A')}")
    print(f"  utilization: {health_initial.get('utilization', 'N/A')}")

    results = []

    for test in TEST_PROMPTS:
        name = test["name"]
        messages = test["messages"]
        print(f"\n--- {name} ---")

        health_before = get_health(client, url)

        print(f"  Running inference... ", end="", flush=True)
        r = streaming_chat(client, url, messages, args.max_tokens, model=model_name)
        r["test_name"] = name

        health_after = get_health(client, url)
        r["health_before"] = health_before
        r["health_after"] = health_after

        if r.get("error"):
            print(f"ERROR: {r['error'][:60]}")
        else:
            print(
                f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s, "
                f"{r['completion_tokens']} tokens"
            )

        # Show cache delta
        before_used = (health_before or {}).get("cache_stats", {}).get("used_blocks", 0)
        after_used = (health_after or {}).get("cache_stats", {}).get("used_blocks", 0)
        delta = after_used - before_used
        after_util = (health_after or {}).get("utilization", 0)
        print(f"  Blocks: {before_used} -> {after_used} (delta={delta}), utilization={after_util:.3f}")

        results.append(r)

    # Final stats
    health_final = get_health(client, url)
    metrics_final = get_metrics(client, url)
    client.close()

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = {
        "benchmark": "bench_memory",
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "max_tokens": args.max_tokens,
        "health_initial": health_initial,
        "health_final": health_final,
        "metrics_initial": metrics_initial,
        "metrics_final": metrics_final,
        "results": results,
    }
    outfile = RESULTS_DIR / f"bench_memory_{args.label}_{ts}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {outfile}")

    # Summary
    print(f"\n=== Memory Summary (label={args.label}) ===")
    final_cache = (health_final or {}).get("cache_stats", {})
    print(f"  Final used_blocks: {final_cache.get('used_blocks', 'N/A')}")
    print(f"  Final free_blocks: {final_cache.get('free_blocks', 'N/A')}")
    print(f"  Final utilization: {(health_final or {}).get('utilization', 'N/A')}")
    print(f"  Final cache_hit_rate: {final_cache.get('cache_hit_rate', 'N/A')}")

    print(f"\n  Per-test throughput:")
    for r in results:
        name = r.get("test_name", "?")
        if r.get("error"):
            print(f"    {name}: ERROR")
        else:
            print(f"    {name}: {r['generation_tok_s']} tok/s, {r['completion_tokens']} tokens")


if __name__ == "__main__":
    main()
