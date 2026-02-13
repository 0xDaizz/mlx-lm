#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Comprehensive API test suite for a running mlx-lm-server.

Tests remaining endpoints and behaviors against a live server running
Kimi K2.5 (or any OpenAI-compatible model).

Already verified (NOT re-tested here):
  livez, readyz, health, models, metrics, non-streaming chat, validation(422)

Tests in this script:
  1. Streaming chat completions (SSE format, incremental delivery, [DONE])
  2. Multi-turn conversation (context-aware responses)
  3. Deterministic output (temperature=0, identical outputs)
  4. Stop sequences (generation halts at stop token)
  5. Completions endpoint (/v1/completions with prompt string)
  6. Large token generation (max_tokens=512)

Usage:
    /Users/hw/mlx-lm-server/.venv/bin/python tests/e2e/test_api_comprehensive.py
    /Users/hw/mlx-lm-server/.venv/bin/python tests/e2e/test_api_comprehensive.py --server-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://localhost:8080"
SYSTEM_PROMPT = "You are a helpful assistant."

# Short prompts to minimize generation time on large MoE models
PROMPT_SIMPLE = "What is 2+2? Answer in one word."
PROMPT_CAPITAL = "What is the capital of France? Answer in one word."
PROMPT_MULTITURN_1 = "My name is Alice. Remember it."
PROMPT_MULTITURN_2 = "What is my name? Answer in one word."
PROMPT_STOP_SEQ = "Count from 1 to 20, one number per line."
PROMPT_LONG_GEN = (
    "Write a short essay about the history of computing. "
    "Include at least 5 paragraphs."
)
PROMPT_COMPLETION = "The capital of Japan is"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    detail: str = ""
    error: str = ""


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name} ({result.duration_s:.2f}s)")
        if result.detail:
            for line in result.detail.split("\n"):
                print(f"         {line}")
        if result.error:
            print(f"         ERROR: {result.error}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print("\n" + "=" * 60)
        print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
        print("=" * 60)
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
        if failed > 0:
            print(f"\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error or r.detail}")
        print()
        return failed == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_name(server_url: str) -> str:
    """Fetch the loaded model name from the server."""
    resp = httpx.get(f"{server_url}/v1/models", timeout=10.0)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError("No models loaded on server")
    return models[0]["id"]


def chat_payload(
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    max_tokens: int = 64,
    temperature: float = 0.7,
    stop: list[str] | None = None,
    include_usage: bool = False,
) -> dict:
    """Build a /v1/chat/completions request payload."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stop is not None:
        payload["stop"] = stop
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def simple_messages(user_content: str) -> list[dict]:
    """Build a single-turn message list with system prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_sse_lines(response: httpx.Response):
    """Yield raw SSE data strings from a streaming response."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            yield line[6:]


def collect_streaming_response(response: httpx.Response) -> dict:
    """Parse SSE stream and return collected info.

    Returns dict with keys:
        chunks: list of parsed JSON chunks
        content_pieces: list of content delta strings
        full_content: concatenated content
        finish_reason: last finish_reason seen
        saw_done: whether [DONE] terminator was received
        usage: usage dict if present
        chunk_count: total number of data lines
    """
    chunks = []
    content_pieces = []
    finish_reason = None
    saw_done = False
    usage = {}
    chunk_count = 0

    for data_str in parse_sse_lines(response):
        chunk_count += 1

        if data_str == "[DONE]":
            saw_done = True
            continue

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        chunks.append(chunk)

        # Usage-only chunk (no choices)
        if "usage" in chunk and not chunk.get("choices"):
            usage = chunk["usage"]
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        fr = choices[0].get("finish_reason")

        if content:
            content_pieces.append(content)
        if fr is not None:
            finish_reason = fr

    return {
        "chunks": chunks,
        "content_pieces": content_pieces,
        "full_content": "".join(content_pieces),
        "finish_reason": finish_reason,
        "saw_done": saw_done,
        "usage": usage,
        "chunk_count": chunk_count,
    }


# ---------------------------------------------------------------------------
# Test 1: Streaming chat completions
# ---------------------------------------------------------------------------

def test_streaming_chat(client: httpx.Client, url: str, model: str) -> TestResult:
    """Verify SSE streaming format, incremental content, and [DONE] terminator."""
    t0 = time.perf_counter()
    try:
        payload = chat_payload(
            model,
            simple_messages(PROMPT_SIMPLE),
            stream=True,
            max_tokens=64,
            temperature=0.7,
            include_usage=True,
        )

        with client.stream(
            "POST", f"{url}/v1/chat/completions", json=payload, timeout=60.0
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                return TestResult(
                    name="streaming_chat",
                    passed=False,
                    duration_s=time.perf_counter() - t0,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            # Verify Content-Type header is SSE
            content_type = resp.headers.get("content-type", "")
            is_sse_content_type = "text/event-stream" in content_type

            result = collect_streaming_response(resp)

        elapsed = time.perf_counter() - t0

        # Validation checks
        checks = []
        all_ok = True

        # Check 1: SSE content type
        if is_sse_content_type:
            checks.append("SSE content-type: OK")
        else:
            checks.append(f"SSE content-type: WARN (got '{content_type}')")
            # Not a hard failure; some servers use different content types

        # Check 2: Got multiple content chunks (incremental delivery)
        n_pieces = len(result["content_pieces"])
        if n_pieces >= 2:
            checks.append(f"Incremental delivery: OK ({n_pieces} chunks)")
        else:
            checks.append(f"Incremental delivery: FAIL (only {n_pieces} chunks)")
            all_ok = False

        # Check 3: Non-empty content
        content = result["full_content"].strip()
        if content:
            checks.append(f"Content received: OK ({len(content)} chars)")
        else:
            checks.append("Content received: FAIL (empty)")
            all_ok = False

        # Check 4: [DONE] terminator
        if result["saw_done"]:
            checks.append("[DONE] terminator: OK")
        else:
            checks.append("[DONE] terminator: FAIL (not received)")
            all_ok = False

        # Check 5: finish_reason present
        if result["finish_reason"]:
            checks.append(f"finish_reason: OK ({result['finish_reason']})")
        else:
            checks.append("finish_reason: FAIL (not received)")
            all_ok = False

        # Check 6: Each chunk has valid JSON structure with id, object, choices
        valid_structure = True
        for chunk in result["chunks"][:3]:  # Check first 3
            if "id" not in chunk or "object" not in chunk:
                valid_structure = False
                break
        if valid_structure and result["chunks"]:
            checks.append("Chunk structure (id, object): OK")
        elif not result["chunks"]:
            checks.append("Chunk structure: FAIL (no chunks)")
            all_ok = False
        else:
            checks.append("Chunk structure: FAIL (missing id/object)")
            all_ok = False

        return TestResult(
            name="streaming_chat",
            passed=all_ok,
            duration_s=elapsed,
            detail="\n".join(checks),
            error="" if all_ok else "One or more streaming checks failed",
        )

    except Exception as e:
        return TestResult(
            name="streaming_chat",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test 2: Multi-turn conversation
# ---------------------------------------------------------------------------

def test_multiturn_conversation(
    client: httpx.Client, url: str, model: str
) -> TestResult:
    """Send 2 turns: introduce a name, then ask for it. Verify coherence."""
    t0 = time.perf_counter()
    try:
        # Turn 1: Tell the model a name
        messages_turn1 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_MULTITURN_1},
        ]
        payload1 = chat_payload(model, messages_turn1, max_tokens=64, temperature=0.3)
        resp1 = client.post(
            f"{url}/v1/chat/completions", json=payload1, timeout=60.0
        )
        if resp1.status_code != 200:
            return TestResult(
                name="multiturn_conversation",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"Turn 1 HTTP {resp1.status_code}: {resp1.text[:200]}",
            )

        data1 = resp1.json()
        assistant_reply_1 = data1["choices"][0]["message"]["content"]

        # Turn 2: Ask for the name, with full conversation history
        messages_turn2 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_MULTITURN_1},
            {"role": "assistant", "content": assistant_reply_1},
            {"role": "user", "content": PROMPT_MULTITURN_2},
        ]
        payload2 = chat_payload(model, messages_turn2, max_tokens=64, temperature=0.3)
        resp2 = client.post(
            f"{url}/v1/chat/completions", json=payload2, timeout=60.0
        )
        if resp2.status_code != 200:
            return TestResult(
                name="multiturn_conversation",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"Turn 2 HTTP {resp2.status_code}: {resp2.text[:200]}",
            )

        data2 = resp2.json()
        assistant_reply_2 = data2["choices"][0]["message"]["content"]

        elapsed = time.perf_counter() - t0

        # Check: the model should mention "Alice" in the second response
        reply_lower = assistant_reply_2.lower()
        found_name = "alice" in reply_lower

        detail_lines = [
            f"Turn 1 reply: {assistant_reply_1[:100]}",
            f"Turn 2 reply: {assistant_reply_2[:100]}",
            f"Name 'Alice' in turn 2: {'YES' if found_name else 'NO'}",
        ]

        return TestResult(
            name="multiturn_conversation",
            passed=found_name,
            duration_s=elapsed,
            detail="\n".join(detail_lines),
            error="" if found_name else "Model did not recall the name 'Alice'",
        )

    except Exception as e:
        return TestResult(
            name="multiturn_conversation",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test 3: Deterministic output (temperature=0)
# ---------------------------------------------------------------------------

def test_deterministic_temp0(
    client: httpx.Client, url: str, model: str
) -> TestResult:
    """Send identical prompt twice with temperature=0, verify identical outputs."""
    t0 = time.perf_counter()
    try:
        messages = simple_messages(PROMPT_CAPITAL)
        payload = chat_payload(
            model, messages, max_tokens=32, temperature=0.0
        )

        # First request
        resp1 = client.post(
            f"{url}/v1/chat/completions", json=payload, timeout=60.0
        )
        if resp1.status_code != 200:
            return TestResult(
                name="deterministic_temp0",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"Request 1 HTTP {resp1.status_code}: {resp1.text[:200]}",
            )
        content1 = resp1.json()["choices"][0]["message"]["content"]

        # Second request (identical)
        resp2 = client.post(
            f"{url}/v1/chat/completions", json=payload, timeout=60.0
        )
        if resp2.status_code != 200:
            return TestResult(
                name="deterministic_temp0",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"Request 2 HTTP {resp2.status_code}: {resp2.text[:200]}",
            )
        content2 = resp2.json()["choices"][0]["message"]["content"]

        elapsed = time.perf_counter() - t0

        identical = content1 == content2

        detail_lines = [
            f"Response 1: {content1[:120]}",
            f"Response 2: {content2[:120]}",
            f"Identical: {'YES' if identical else 'NO'}",
        ]

        return TestResult(
            name="deterministic_temp0",
            passed=identical,
            duration_s=elapsed,
            detail="\n".join(detail_lines),
            error="" if identical else "Outputs differ with temperature=0",
        )

    except Exception as e:
        return TestResult(
            name="deterministic_temp0",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test 4: Stop sequences
# ---------------------------------------------------------------------------

def test_stop_sequences(client: httpx.Client, url: str, model: str) -> TestResult:
    """Send a request with stop=[\"\\n\"], verify generation stops at newline."""
    t0 = time.perf_counter()
    try:
        messages = simple_messages(PROMPT_STOP_SEQ)
        payload = chat_payload(
            model,
            messages,
            max_tokens=256,
            temperature=0.3,
            stop=["\n"],
        )

        resp = client.post(
            f"{url}/v1/chat/completions", json=payload, timeout=60.0
        )
        if resp.status_code != 200:
            return TestResult(
                name="stop_sequences",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "")

        elapsed = time.perf_counter() - t0

        # The response should be short (stopped at first newline)
        # and finish_reason should be "stop"
        content_stripped = content.strip()
        has_no_newline = "\n" not in content_stripped
        short_enough = len(content_stripped) < 200  # Should be very short
        reason_is_stop = finish_reason == "stop"

        checks = []
        all_ok = True

        if has_no_newline:
            checks.append("No newline in content: OK")
        else:
            checks.append(f"No newline in content: FAIL (found newline in '{content_stripped[:80]}')")
            all_ok = False

        if short_enough:
            checks.append(f"Content is short: OK ({len(content_stripped)} chars)")
        else:
            checks.append(f"Content is short: WARN ({len(content_stripped)} chars)")
            # Not a hard failure

        if reason_is_stop:
            checks.append("finish_reason='stop': OK")
        else:
            checks.append(f"finish_reason='stop': FAIL (got '{finish_reason}')")
            all_ok = False

        checks.append(f"Content: {content_stripped[:100]}")

        return TestResult(
            name="stop_sequences",
            passed=all_ok,
            duration_s=elapsed,
            detail="\n".join(checks),
            error="" if all_ok else "Stop sequence check failed",
        )

    except Exception as e:
        return TestResult(
            name="stop_sequences",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test 5: Completions endpoint (/v1/completions)
# ---------------------------------------------------------------------------

def test_completions_endpoint(
    client: httpx.Client, url: str, model: str
) -> TestResult:
    """POST /v1/completions with a prompt string, verify response format."""
    t0 = time.perf_counter()
    try:
        payload = {
            "model": model,
            "prompt": PROMPT_COMPLETION,
            "max_tokens": 32,
            "temperature": 0.7,
            "stream": False,
        }

        resp = client.post(
            f"{url}/v1/completions", json=payload, timeout=60.0
        )
        if resp.status_code != 200:
            return TestResult(
                name="completions_endpoint",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        elapsed = time.perf_counter() - t0

        checks = []
        all_ok = True

        # Check 1: Response has correct object type
        obj_type = data.get("object", "")
        if obj_type == "text_completion":
            checks.append("object='text_completion': OK")
        else:
            checks.append(f"object='text_completion': FAIL (got '{obj_type}')")
            all_ok = False

        # Check 2: Has id field
        resp_id = data.get("id", "")
        if resp_id:
            checks.append(f"id present: OK ({resp_id})")
        else:
            checks.append("id present: FAIL (missing)")
            all_ok = False

        # Check 3: Has choices with text field (not message)
        choices = data.get("choices", [])
        if choices and "text" in choices[0]:
            text = choices[0]["text"]
            checks.append(f"choices[0].text present: OK")
            checks.append(f"Text: {text.strip()[:100]}")
            if not text.strip():
                checks.append("Text is empty: WARN")
        else:
            checks.append("choices[0].text present: FAIL")
            all_ok = False

        # Check 4: Has usage
        usage = data.get("usage", {})
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            checks.append(
                f"usage: OK (prompt={usage['prompt_tokens']}, "
                f"completion={usage['completion_tokens']})"
            )
        else:
            checks.append(f"usage: FAIL (got {usage})")
            all_ok = False

        # Check 5: finish_reason
        if choices:
            fr = choices[0].get("finish_reason", "")
            if fr:
                checks.append(f"finish_reason: OK ({fr})")
            else:
                checks.append("finish_reason: WARN (empty)")

        return TestResult(
            name="completions_endpoint",
            passed=all_ok,
            duration_s=elapsed,
            detail="\n".join(checks),
            error="" if all_ok else "Completions response format check failed",
        )

    except Exception as e:
        return TestResult(
            name="completions_endpoint",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test 6: Large token generation
# ---------------------------------------------------------------------------

def test_large_token_generation(
    client: httpx.Client, url: str, model: str
) -> TestResult:
    """Request max_tokens=512, verify we get close to that many tokens."""
    t0 = time.perf_counter()
    try:
        messages = simple_messages(PROMPT_LONG_GEN)
        payload = chat_payload(
            model, messages, max_tokens=512, temperature=0.7
        )

        resp = client.post(
            f"{url}/v1/chat/completions", json=payload, timeout=120.0
        )
        if resp.status_code != 200:
            return TestResult(
                name="large_token_generation",
                passed=False,
                duration_s=time.perf_counter() - t0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        elapsed = time.perf_counter() - t0

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "")

        checks = []
        all_ok = True

        # Check 1: Got a substantial number of tokens
        # We request 512; expect at least 256 (50%) unless the model
        # naturally finishes with "stop"
        min_expected = 256
        if finish_reason == "length":
            # Hit the max_tokens limit -- ideal case
            checks.append(f"finish_reason='length': OK (hit max_tokens)")
            if completion_tokens >= min_expected:
                checks.append(
                    f"Tokens generated: OK ({completion_tokens} >= {min_expected})"
                )
            else:
                checks.append(
                    f"Tokens generated: FAIL ({completion_tokens} < {min_expected})"
                )
                all_ok = False
        elif finish_reason == "stop":
            # Model finished naturally before hitting max_tokens
            # Still valid if we got a decent amount
            if completion_tokens >= 100:
                checks.append(
                    f"Natural stop at {completion_tokens} tokens: OK "
                    f"(model completed response)"
                )
            else:
                checks.append(
                    f"Natural stop at {completion_tokens} tokens: WARN "
                    f"(expected more for this prompt)"
                )
                # Not a hard failure -- model might just be concise
        else:
            checks.append(f"finish_reason='{finish_reason}': unexpected")

        # Check 2: Content is substantial
        content_len = len(content.strip())
        if content_len >= 200:
            checks.append(f"Content length: OK ({content_len} chars)")
        else:
            checks.append(f"Content length: WARN ({content_len} chars, expected more)")

        # Check 3: Token count doesn't exceed max_tokens
        if completion_tokens <= 512:
            checks.append(
                f"Token count <= max_tokens: OK ({completion_tokens} <= 512)"
            )
        else:
            checks.append(
                f"Token count <= max_tokens: FAIL ({completion_tokens} > 512)"
            )
            all_ok = False

        tok_per_sec = round(completion_tokens / elapsed, 2) if elapsed > 0 else 0
        checks.append(f"Performance: {tok_per_sec} tok/s, {elapsed:.2f}s total")

        return TestResult(
            name="large_token_generation",
            passed=all_ok,
            duration_s=elapsed,
            detail="\n".join(checks),
            error="" if all_ok else "Large token generation check failed",
        )

    except Exception as e:
        return TestResult(
            name="large_token_generation",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test 5b: Streaming completions endpoint
# ---------------------------------------------------------------------------

def test_streaming_completions(
    client: httpx.Client, url: str, model: str
) -> TestResult:
    """POST /v1/completions with stream=true, verify SSE format with text field."""
    t0 = time.perf_counter()
    try:
        payload = {
            "model": model,
            "prompt": PROMPT_COMPLETION,
            "max_tokens": 64,
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        with client.stream(
            "POST", f"{url}/v1/completions", json=payload, timeout=60.0
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                return TestResult(
                    name="streaming_completions",
                    passed=False,
                    duration_s=time.perf_counter() - t0,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            # Manually parse since completions uses "text" not "delta.content"
            chunks = []
            text_pieces = []
            saw_done = False
            finish_reason = None
            usage = {}

            for data_str in parse_sse_lines(resp):
                if data_str == "[DONE]":
                    saw_done = True
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                chunks.append(chunk)

                if "usage" in chunk and not chunk.get("choices"):
                    usage = chunk["usage"]
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                text = choices[0].get("text", "")
                fr = choices[0].get("finish_reason")
                if text:
                    text_pieces.append(text)
                if fr is not None:
                    finish_reason = fr

        elapsed = time.perf_counter() - t0

        checks = []
        all_ok = True

        full_text = "".join(text_pieces)
        if text_pieces:
            checks.append(f"Text received: OK ({len(full_text)} chars)")
        else:
            checks.append("Text received: FAIL (empty)")
            all_ok = False

        if saw_done:
            checks.append("[DONE] terminator: OK")
        else:
            checks.append("[DONE] terminator: FAIL")
            all_ok = False

        if finish_reason:
            checks.append(f"finish_reason: OK ({finish_reason})")
        else:
            checks.append("finish_reason: FAIL (missing)")
            all_ok = False

        # Check object type in chunks
        if chunks and chunks[0].get("object") == "text_completion":
            checks.append("object='text_completion': OK")
        elif chunks:
            checks.append(f"object: WARN (got '{chunks[0].get('object', 'missing')}')")
        else:
            checks.append("No chunks received: FAIL")
            all_ok = False

        checks.append(f"Text: {full_text.strip()[:100]}")

        return TestResult(
            name="streaming_completions",
            passed=all_ok,
            duration_s=elapsed,
            detail="\n".join(checks),
            error="" if all_ok else "Streaming completions check failed",
        )

    except Exception as e:
        return TestResult(
            name="streaming_completions",
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive API test suite for mlx-lm-server"
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server URL (default: {DEFAULT_SERVER_URL})",
    )
    args = parser.parse_args()

    url = args.server_url.rstrip("/")
    suite = TestSuite()

    # Pre-flight: check server is reachable
    print(f"Connecting to server at {url}...")
    try:
        health_resp = httpx.get(f"{url}/health", timeout=10.0)
        health_resp.raise_for_status()
        print(f"Server health: {health_resp.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Server not reachable at {url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Get model name
    try:
        model = get_model_name(url)
        print(f"Model: {model}")
    except Exception as e:
        print(f"ERROR: Could not get model name: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print("Running comprehensive API tests")
    print("=" * 60)
    print()

    client = httpx.Client()

    # --- Test 1: Streaming chat ---
    print("[Test 1/7] Streaming chat completions")
    suite.add(test_streaming_chat(client, url, model))
    print()

    # --- Test 2: Multi-turn conversation ---
    print("[Test 2/7] Multi-turn conversation")
    suite.add(test_multiturn_conversation(client, url, model))
    print()

    # --- Test 3: Deterministic output (temp=0) ---
    print("[Test 3/7] Deterministic output (temperature=0)")
    suite.add(test_deterministic_temp0(client, url, model))
    print()

    # --- Test 4: Stop sequences ---
    print("[Test 4/7] Stop sequences")
    suite.add(test_stop_sequences(client, url, model))
    print()

    # --- Test 5: Completions endpoint (non-streaming) ---
    print("[Test 5/7] Completions endpoint (/v1/completions)")
    suite.add(test_completions_endpoint(client, url, model))
    print()

    # --- Test 5b: Streaming completions ---
    print("[Test 6/7] Streaming completions (/v1/completions, stream=true)")
    suite.add(test_streaming_completions(client, url, model))
    print()

    # --- Test 6: Large token generation ---
    print("[Test 7/7] Large token generation (max_tokens=512)")
    suite.add(test_large_token_generation(client, url, model))
    print()

    client.close()

    # Summary
    all_passed = suite.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
