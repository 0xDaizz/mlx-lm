#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Prefix cache and SSD cache hit/miss benchmark.

Tests cache behavior by sending repeated prompts, shared-prefix prompts,
and multi-turn conversations. Measures TTFT improvement from cache hits.

Usage:
    python bench_cache.py [--server-url http://localhost:8080] [--max-tokens 128]
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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BASE_PROMPT = (
    "The transformer architecture was introduced in the paper 'Attention Is All You Need' "
    "by Vaswani et al. in 2017. It revolutionized natural language processing by replacing "
    "recurrent neural networks with self-attention mechanisms. The key innovation was the "
    "ability to process all positions in a sequence simultaneously, enabling much more "
    "efficient parallelization during training."
)

SHARED_PREFIX = BASE_PROMPT + "\n\nBased on the above, "
SUFFIX_A = "explain how multi-head attention works."
SUFFIX_B = "describe the role of positional encoding."
SUFFIX_C = "what are the limitations of this architecture?"


# Multi-turn conversation
CONVERSATION_TURNS = [
    {"role": "user", "content": "Tell me about the history of machine learning."},
    {"role": "assistant", "content": "Machine learning has a rich history dating back to the 1950s..."},
    {"role": "user", "content": "What about deep learning specifically?"},
    {"role": "assistant", "content": "Deep learning emerged as a subfield in the 2000s..."},
    {"role": "user", "content": "How does it compare to traditional ML approaches?"},
]


# ---------------------------------------------------------------------------
# SSE parser
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


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def streaming_chat_request(
    client: httpx.Client, url: str, messages: list[dict], max_tokens: int, model: str = "default"
) -> dict:
    """Execute a streaming chat request and measure timing."""
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
    finish_reason = None
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
            content = delta.get("content", "")
            fr = choices[0].get("finish_reason")

            if content:
                if ttft is None:
                    ttft = now - t0
                token_count += 1

            if fr is not None:
                finish_reason = fr

    total_time = time.perf_counter() - t0
    completion_tokens = usage_info.get("completion_tokens", token_count)
    prompt_tokens = usage_info.get("prompt_tokens", 0)
    gen_time = total_time - (ttft or 0)
    gen_tok_s = round(completion_tokens / gen_time, 2) if gen_time > 0 else 0

    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_time_s": round(total_time, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_tok_s": gen_tok_s,
        "finish_reason": finish_reason,
        "error": None,
    }


def get_health(client: httpx.Client, url: str) -> dict | None:
    """Fetch health stats."""
    try:
        resp = client.get(f"{url}/health", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cold_vs_warm(client: httpx.Client, url: str, max_tokens: int, model: str = "default") -> list[dict]:
    """Test 1: Cold start vs warm repeat of same prompt."""
    results = []
    messages = [{"role": "user", "content": BASE_PROMPT + "\nSummarize this in 3 bullet points."}]

    print("\n--- Test 1: Cold vs Warm (same prompt repeated) ---")

    # Cold request
    print("  Cold request... ", end="", flush=True)
    health_before = get_health(client, url)
    r = streaming_chat_request(client, url, messages, max_tokens, model=model)
    r["test_name"] = "cold_request"
    r["cache_health_before"] = health_before
    r["cache_health_after"] = get_health(client, url)
    results.append(r)
    if r.get("error"):
        print(f"ERROR: {r['error'][:60]}")
    else:
        print(f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s")

    # Brief pause
    time.sleep(1)

    # Warm request (same prompt -- should hit prefix cache)
    print("  Warm request (same prompt)... ", end="", flush=True)
    health_before = get_health(client, url)
    r = streaming_chat_request(client, url, messages, max_tokens, model=model)
    r["test_name"] = "warm_request"
    r["cache_health_before"] = health_before
    r["cache_health_after"] = get_health(client, url)
    results.append(r)
    if r.get("error"):
        print(f"ERROR: {r['error'][:60]}")
    else:
        print(f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s")

    # Compare
    cold_ttft = results[0].get("ttft_s")
    warm_ttft = results[1].get("ttft_s")
    if cold_ttft and warm_ttft:
        improvement = ((cold_ttft - warm_ttft) / cold_ttft) * 100
        print(f"  TTFT improvement: {improvement:.1f}% (cold={cold_ttft}s, warm={warm_ttft}s)")

    return results


def test_shared_prefix(client: httpx.Client, url: str, max_tokens: int, model: str = "default") -> list[dict]:
    """Test 2: Shared prefix with different suffixes."""
    results = []

    print("\n--- Test 2: Shared Prefix (different suffixes) ---")
    suffixes = [("suffix_a", SUFFIX_A), ("suffix_b", SUFFIX_B), ("suffix_c", SUFFIX_C)]

    for name, suffix in suffixes:
        messages = [{"role": "user", "content": SHARED_PREFIX + suffix}]
        print(f"  {name}... ", end="", flush=True)
        health_before = get_health(client, url)
        r = streaming_chat_request(client, url, messages, max_tokens, model=model)
        r["test_name"] = f"shared_prefix_{name}"
        r["cache_health_before"] = health_before
        r["cache_health_after"] = get_health(client, url)
        results.append(r)
        if r.get("error"):
            print(f"ERROR: {r['error'][:60]}")
        else:
            print(f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s")
        time.sleep(0.5)

    # Compare TTFTs
    ttfts = [(r["test_name"], r.get("ttft_s")) for r in results if r.get("ttft_s")]
    if len(ttfts) >= 2:
        print(f"  TTFTs: {', '.join(f'{n}={t}s' for n, t in ttfts)}")
        print(f"  First was cold, subsequent should benefit from shared prefix cache")

    return results


def test_multi_turn(client: httpx.Client, url: str, max_tokens: int, model: str = "default") -> list[dict]:
    """Test 3: Multi-turn conversation with accumulating context."""
    results = []

    print("\n--- Test 3: Multi-turn Conversation ---")

    for i in range(1, len(CONVERSATION_TURNS) + 1, 2):
        # Send increasing turns
        messages = CONVERSATION_TURNS[:i]
        if messages[-1]["role"] == "assistant":
            # Add a follow-up user message
            messages = messages + [{"role": "user", "content": "Please continue."}]

        turn_num = (i + 1) // 2
        print(f"  Turn {turn_num} ({len(messages)} messages)... ", end="", flush=True)
        health_before = get_health(client, url)
        r = streaming_chat_request(client, url, messages, max_tokens, model=model)
        r["test_name"] = f"multi_turn_{turn_num}"
        r["num_messages"] = len(messages)
        r["cache_health_before"] = health_before
        r["cache_health_after"] = get_health(client, url)
        results.append(r)
        if r.get("error"):
            print(f"ERROR: {r['error'][:60]}")
        else:
            print(
                f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s, "
                f"prompt_tokens={r.get('prompt_tokens', '?')}"
            )
        time.sleep(0.5)

    return results


def test_cache_stats(client: httpx.Client, url: str) -> dict:
    """Test 4: Collect cache statistics from health endpoint."""
    print("\n--- Test 4: Cache Statistics ---")
    health = get_health(client, url)
    if health:
        cache_stats = health.get("cache_stats", {})
        print(f"  used_blocks: {cache_stats.get('used_blocks', 'N/A')}")
        print(f"  free_blocks: {cache_stats.get('free_blocks', 'N/A')}")
        print(f"  total_blocks: {cache_stats.get('total_blocks', 'N/A')}")
        print(f"  cache_hit_rate: {cache_stats.get('cache_hit_rate', 'N/A')}")
        print(f"  utilization: {health.get('utilization', 'N/A')}")
    return {"test_name": "cache_stats", "health": health}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prefix cache / SSD cache benchmark")
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    url = args.server_url.rstrip("/")
    client = httpx.Client()

    health = get_health(client, url)
    if health is None:
        print(f"ERROR: Server not reachable at {url}", file=sys.stderr)
        sys.exit(1)
    print(f"Server health: {health.get('status', 'unknown')}")

    # Fetch model name from server
    model_name = get_model_name(url)
    print(f"Model: {model_name}")

    all_results = []
    all_results.extend(test_cold_vs_warm(client, url, args.max_tokens, model=model_name))
    all_results.extend(test_shared_prefix(client, url, args.max_tokens, model=model_name))
    all_results.extend(test_multi_turn(client, url, args.max_tokens, model=model_name))
    stats = test_cache_stats(client, url)
    all_results.append(stats)

    client.close()

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = {
        "benchmark": "bench_cache",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "max_tokens": args.max_tokens,
        "health_initial": health,
        "results": all_results,
    }
    outfile = RESULTS_DIR / f"bench_cache_{ts}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {outfile}")

    # Summary
    print("\n=== Cache Benchmark Summary ===")
    for r in all_results:
        name = r.get("test_name", "?")
        if r.get("error"):
            print(f"  {name}: ERROR - {r['error'][:60]}")
        elif r.get("ttft_s") is not None:
            print(f"  {name}: TTFT={r['ttft_s']}s, {r.get('generation_tok_s', '?')} tok/s")
        elif "health" in r:
            hr = r.get("health") or {}
            print(f"  {name}: hit_rate={hr.get('cache_stats', {}).get('cache_hit_rate', 'N/A')}")


if __name__ == "__main__":
    main()
