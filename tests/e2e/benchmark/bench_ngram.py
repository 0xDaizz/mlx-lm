#!/Users/hw/mlx-lm-server/.venv/bin/python
"""N-gram speculative decode benchmark with diverse prompts.

Tests different prompt categories and ngram parameter combinations.
NOTE: Different --ngram-max / --num-speculative-tokens combinations
require server restarts. This script tests against the CURRENT server
config and records the spec decode metrics.

To test a full parameter grid, run this script multiple times with
different server configurations, or use run_all.sh which handles restarts.

Usage:
    python bench_ngram.py [--server-url http://localhost:8080] [--max-tokens 256]
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
# Diverse prompt categories (designed to test n-gram acceptance patterns)
# ---------------------------------------------------------------------------

PROMPT_CATEGORIES = {
    "coding": {
        "prompt": (
            "Write a Python function that implements a binary search tree with "
            "insert, search, and delete operations. Include docstrings and type hints."
        ),
        "description": "Code generation (high repetitive structure expected)",
    },
    "structured_output": {
        "prompt": (
            "Generate a JSON schema for a REST API that manages a library system. "
            "Include books, authors, members, and loans. Use proper JSON format."
        ),
        "description": "Structured output generation (JSON/YAML patterns)",
    },
    "daily_conversation": {
        "prompt": (
            "I'm planning a weekend trip to Jeju Island. Can you suggest a 2-day "
            "itinerary including restaurants, scenic spots, and activities?"
        ),
        "description": "Casual conversational response",
    },
    "technical_docs": {
        "prompt": (
            "Explain how transformer attention mechanism works, including "
            "multi-head attention, scaled dot-product attention, and positional "
            "encoding. Use mathematical notation where appropriate."
        ),
        "description": "Technical/scientific writing",
    },
    "translation": {
        "prompt": (
            "Translate the following Korean text to English:\n"
            "인공지능의 발전은 우리 사회에 큰 변화를 가져오고 있습니다. "
            "특히 대규모 언어 모델은 자연어 처리 분야에서 혁신적인 성과를 "
            "보여주고 있으며, 다양한 산업 분야에서 활용되고 있습니다."
        ),
        "description": "Korean to English translation",
    },
    "math_logic": {
        "prompt": (
            "Solve this step by step: A train leaves Station A at 9:00 AM traveling "
            "at 80 km/h. Another train leaves Station B (300 km away) at 10:00 AM "
            "traveling toward Station A at 120 km/h. At what time do they meet? "
            "Show all work."
        ),
        "description": "Mathematical reasoning (step-by-step)",
    },
    "creative_writing": {
        "prompt": (
            "Write the opening chapter of a science fiction novel set on Mars in "
            "2150. The protagonist is an AI researcher who discovers something "
            "unexpected in the Martian soil. Use vivid descriptions."
        ),
        "description": "Creative/narrative writing",
    },
    "repetitive_patterns": {
        "prompt": (
            "Create a multiplication table from 1 to 12. Format each row as: "
            "N x 1 = _, N x 2 = _, ..., N x 12 = _. List all 12 rows."
        ),
        "description": "Highly repetitive/structured output (best for n-gram)",
    },
    "counting": {
        "prompt": (
            "Count from 1 to 50, and for each number, state whether it is "
            "prime, even, odd, or a multiple of 5. Format: 'N: [properties]'"
        ),
        "description": "Enumeration task (repetitive structure)",
    },
}


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
# Benchmark
# ---------------------------------------------------------------------------

def bench_streaming_chat(
    client: httpx.Client, url: str, prompt: str, max_tokens: int, model: str = "default"
) -> dict:
    """Run a single streaming chat request and measure performance."""
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
    tokens_collected = []
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
                tokens_collected.append(content)

            if fr is not None:
                finish_reason = fr

    total_time = time.perf_counter() - t0
    completion_tokens = usage_info.get("completion_tokens", len(tokens_collected))
    prompt_tokens = usage_info.get("prompt_tokens", 0)
    gen_time = total_time - (ttft or 0)
    gen_tok_s = round(completion_tokens / gen_time, 2) if gen_time > 0 else 0

    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_time_s": round(total_time, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_tok_s": gen_tok_s,
        "content_preview": "".join(tokens_collected)[:100],
        "finish_reason": finish_reason,
        "error": None,
    }


def get_spec_decode_metrics(client: httpx.Client, url: str) -> dict | None:
    """Fetch speculative decoding metrics."""
    try:
        resp = client.get(f"{url}/v1/spec_decode/metrics", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_health(client: httpx.Client, url: str) -> dict | None:
    """Fetch server health."""
    try:
        resp = client.get(f"{url}/health", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="N-gram spec decode benchmark")
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--category", type=str, default=None,
                        help="Run only a specific prompt category")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for this run (e.g. 'ngram4_k5')")
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

    # Check spec decode status before benchmarking
    spec_metrics_before = get_spec_decode_metrics(client, url)
    spec_enabled = spec_metrics_before and spec_metrics_before.get("spec_decode_enabled", False)
    print(f"Spec decode enabled: {spec_enabled}")
    if spec_metrics_before:
        print(f"Spec decode metrics (before): {json.dumps(spec_metrics_before, indent=2)}")

    categories = (
        {args.category: PROMPT_CATEGORIES[args.category]}
        if args.category and args.category in PROMPT_CATEGORIES
        else PROMPT_CATEGORIES
    )

    results = []
    for cat_name, cat_info in categories.items():
        prompt = cat_info["prompt"]
        desc = cat_info["description"]
        print(f"\n--- {cat_name}: {desc} ---")

        # Get spec metrics before this category
        spec_before = get_spec_decode_metrics(client, url)

        print(f"  Benchmarking... ", end="", flush=True)
        r = bench_streaming_chat(client, url, prompt, args.max_tokens, model=model_name)
        r["category"] = cat_name
        r["category_description"] = desc
        r["max_tokens"] = args.max_tokens

        # Get spec metrics after to compute delta
        spec_after = get_spec_decode_metrics(client, url)
        r["spec_metrics_after"] = spec_after

        if r.get("error"):
            print(f"ERROR: {r['error'][:60]}")
        else:
            print(
                f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s, "
                f"{r['completion_tokens']} tokens, {r['total_time_s']}s"
            )

        results.append(r)

    # Final spec metrics
    spec_metrics_after = get_spec_decode_metrics(client, url)
    client.close()

    # Save
    label = args.label or "default"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = {
        "benchmark": "bench_ngram",
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "max_tokens": args.max_tokens,
        "spec_decode_enabled": spec_enabled,
        "spec_metrics_before": spec_metrics_before,
        "spec_metrics_after": spec_metrics_after,
        "health": health,
        "results": results,
    }
    outfile = RESULTS_DIR / f"bench_ngram_{label}_{ts}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {outfile}")

    # Summary
    print(f"\n=== N-gram Benchmark Summary (label={label}) ===")
    print(f"{'Category':<22} | {'tok/s':>7} | {'TTFT':>7} | {'Tokens':>7} | {'Time':>7}")
    print("-" * 65)
    for r in results:
        cat = r.get("category", "?")
        if r.get("error"):
            print(f"{cat:<22} | ERROR: {r['error'][:40]}")
        else:
            ttft = f"{r['ttft_s']:.3f}s" if r.get("ttft_s") is not None else "N/A"
            print(
                f"{cat:<22} | "
                f"{r['generation_tok_s']:>7.1f} | "
                f"{ttft:>7} | "
                f"{r.get('completion_tokens', '?'):>7} | "
                f"{r['total_time_s']:>6.1f}s"
            )

    # Show spec decode acceptance rate if available
    if spec_metrics_after and spec_metrics_after.get("spec_decode_enabled"):
        print(f"\nSpec decode metrics (final):")
        for k, v in spec_metrics_after.items():
            if k != "spec_decode_enabled":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
