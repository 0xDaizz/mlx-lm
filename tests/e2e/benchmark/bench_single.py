#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Single-request benchmark: TTFT, tok/s, total time.

Measures baseline throughput with streaming and non-streaming requests.
Outputs JSON results to /tmp/kimi-bench-results/.

Usage:
    python bench_single.py [--server-url http://localhost:8080] [--max-tokens 256]
"""

from __future__ import annotations

import argparse
import json
import os
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

SHORT_PROMPT = "What is 2+2? Answer in one sentence."
MEDIUM_PROMPT = (
    "Explain the difference between TCP and UDP protocols. "
    "Cover reliability, ordering, speed, and common use cases. "
    "Be concise but thorough."
)
LONG_PROMPT = (
    "Write a detailed analysis of the following topics:\n"
    + "\n".join(f"{i+1}. Topic {i+1}: {'Lorem ipsum dolor sit amet. ' * 20}" for i in range(10))
    + "\nSummarize the key points."
)

PROMPTS = {
    "short": SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "long": LONG_PROMPT,
}


# ---------------------------------------------------------------------------
# SSE Parser
# ---------------------------------------------------------------------------

def parse_sse_stream(response: httpx.Response):
    """Yield parsed SSE data events from a streaming response."""
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
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_non_streaming_chat(
    client: httpx.Client, url: str, prompt: str, max_tokens: int, model: str = "default"
) -> dict:
    """Benchmark a non-streaming chat completion request."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = client.post(f"{url}/v1/chat/completions", json=payload, timeout=600.0)
    total_time = time.perf_counter() - t0

    if resp.status_code != 200:
        return {
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "total_time_s": total_time,
        }

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    content = data["choices"][0]["message"]["content"]

    return {
        "ttft_s": None,  # non-streaming, no TTFT
        "total_time_s": round(total_time, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_tok_s": round(completion_tokens / total_time, 2) if total_time > 0 else 0,
        "content_preview": content[:100],
        "finish_reason": data["choices"][0].get("finish_reason"),
        "error": None,
    }


def bench_streaming_chat(
    client: httpx.Client, url: str, prompt: str, max_tokens: int, model: str = "default"
) -> dict:
    """Benchmark a streaming chat completion request."""
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
    token_times = []
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

            # Usage chunk (no choices)
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
                token_times.append(now)

            if fr is not None:
                finish_reason = fr

    total_time = time.perf_counter() - t0

    # Calculate tok/s from token arrival times
    completion_tokens = usage_info.get("completion_tokens", len(tokens_collected))
    prompt_tokens = usage_info.get("prompt_tokens", 0)

    # Generation time = total time minus TTFT (prefill)
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


def bench_streaming_completion(
    client: httpx.Client, url: str, prompt: str, max_tokens: int, model: str = "default"
) -> dict:
    """Benchmark a streaming text completion request."""
    payload = {
        "model": model,
        "prompt": prompt,
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

    with client.stream("POST", f"{url}/v1/completions", json=payload, timeout=600.0) as resp:
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

            text = choices[0].get("text", "")
            fr = choices[0].get("finish_reason")

            if text:
                if ttft is None:
                    ttft = now - t0
                tokens_collected.append(text)

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


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def get_health(client: httpx.Client, url: str) -> dict | None:
    """Fetch server health stats."""
    try:
        resp = client.get(f"{url}/health", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-request benchmark")
    parser.add_argument("--server-url", default="http://localhost:8080")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-key", choices=list(PROMPTS.keys()), default=None,
                        help="Run only a specific prompt (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    url = args.server_url.rstrip("/")
    max_tokens = args.max_tokens
    results = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def _save_partial():
        """Write partial results so data isn't lost on crash."""
        partial_file = RESULTS_DIR / f"bench_single_{ts}_partial.json"
        partial_file.write_text(json.dumps(
            {"benchmark": "bench_single", "results": results, "partial": True},
            indent=2, default=str,
        ))

    client = httpx.Client(timeout=httpx.Timeout(600.0))

    # Check server is up
    health = get_health(client, url)
    if health is None:
        print(f"ERROR: Server not reachable at {url}", file=sys.stderr)
        sys.exit(1)
    print(f"Server health: {health.get('status', 'unknown')}")

    # Fetch model name from server
    model_name = get_model_name(url)
    print(f"Model: {model_name}")

    prompt_keys = [args.prompt_key] if args.prompt_key else list(PROMPTS.keys())

    try:
        for prompt_key in prompt_keys:
            prompt = PROMPTS[prompt_key]
            print(f"\n--- Prompt: {prompt_key} (max_tokens={max_tokens}) ---")

            # Non-streaming chat
            print("  [1/3] Non-streaming chat... ", end="", flush=True)
            r = bench_non_streaming_chat(client, url, prompt, max_tokens, model=model_name)
            r["test_name"] = f"non_streaming_chat_{prompt_key}"
            r["prompt_key"] = prompt_key
            r["max_tokens"] = max_tokens
            results.append(r)
            _save_partial()
            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(f"{r['generation_tok_s']} tok/s, {r['total_time_s']}s total")

            # Streaming chat
            print("  [2/3] Streaming chat... ", end="", flush=True)
            r = bench_streaming_chat(client, url, prompt, max_tokens, model=model_name)
            r["test_name"] = f"streaming_chat_{prompt_key}"
            r["prompt_key"] = prompt_key
            r["max_tokens"] = max_tokens
            results.append(r)
            _save_partial()
            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s, {r['total_time_s']}s total")

            # Streaming completion
            print("  [3/3] Streaming completion... ", end="", flush=True)
            r = bench_streaming_completion(client, url, prompt, max_tokens, model=model_name)
            r["test_name"] = f"streaming_completion_{prompt_key}"
            r["prompt_key"] = prompt_key
            r["max_tokens"] = max_tokens
            results.append(r)
            _save_partial()
            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s, {r['total_time_s']}s total")

        # Throughput scaling: vary max_tokens
        print("\n--- Throughput scaling (streaming chat, short prompt) ---")
        for mt in [64, 128, 256, 512, 1024]:
            print(f"  max_tokens={mt}... ", end="", flush=True)
            r = bench_streaming_chat(client, url, SHORT_PROMPT, mt, model=model_name)
            r["test_name"] = f"scaling_max_tokens_{mt}"
            r["prompt_key"] = "short"
            r["max_tokens"] = mt
            results.append(r)
            _save_partial()
            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(f"TTFT={r['ttft_s']}s, {r['generation_tok_s']} tok/s, {r['completion_tokens']} tokens")
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        _save_partial()
        print(f"Partial results ({len(results)} tests) saved", file=sys.stderr)
        raise

    # Final health
    health_after = get_health(client, url)
    client.close()

    # Save final results
    output = {
        "benchmark": "bench_single",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "health_before": health,
        "health_after": health_after,
        "results": results,
    }
    outfile = RESULTS_DIR / f"bench_single_{ts}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    # Clean up partial file
    partial_file = RESULTS_DIR / f"bench_single_{ts}_partial.json"
    if partial_file.exists():
        partial_file.unlink()
    print(f"\nResults saved to {outfile}")

    # Summary
    print("\n=== Summary ===")
    for r in results:
        name = r.get("test_name", "?")
        if r.get("error"):
            print(f"  {name}: ERROR - {r['error'][:60]}")
        else:
            parts = [f"{r.get('generation_tok_s', '?')} tok/s"]
            if r.get("ttft_s") is not None:
                parts.append(f"TTFT={r['ttft_s']}s")
            parts.append(f"total={r['total_time_s']}s")
            print(f"  {name}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
