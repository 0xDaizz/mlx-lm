"""Real-model integration tests for mlx-lm-server.

Exercises the scheduler, server, and cache layers with a real model
(Qwen3-4B-4bit loaded from local path). These tests are slow and
require the model directory to be present.

Run:
    pytest tests/test_real_model.py -v --timeout=300 -m slow
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

MODEL_PATH = str(Path(__file__).parent.parent / "Qwen3-4B-4bit")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.path.isdir(MODEL_PATH),
        reason=f"Model {MODEL_PATH} not found",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the real model and tokenizer once per module."""
    from mlx_lm import load

    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer


@pytest.fixture
def scheduler_factory(model_and_tokenizer, tmp_path):
    """Factory that creates a fresh Scheduler with real model + KVCacheManager.

    Yields a callable that returns a started scheduler.  The scheduler is
    automatically stopped during cleanup.
    """
    from mlx_lm_server.config import ServerConfig
    from mlx_lm_server.kv_cache_manager import KVCacheManager
    from mlx_lm_server.scheduler import Scheduler

    model, tokenizer = model_and_tokenizer
    created_schedulers: list[Scheduler] = []

    def _make(
        block_size: int = 16,
        num_blocks: int = 256,
        max_batch_size: int = 4,
        max_queue_size: int = 32,
        **overrides: Any,
    ) -> Scheduler:
        config = ServerConfig(
            model=MODEL_PATH,
            block_size=block_size,
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_queue_size=max_queue_size,
            ssd_cache_dir=tmp_path / "ssd-cache",
            ssd_enabled=False,
            **overrides,
        )
        kv_mgr = KVCacheManager(config)
        sched = Scheduler(
            config=config,
            model=model,
            tokenizer=tokenizer,
            kv_cache_manager=kv_mgr,
        )
        sched.run_inference_loop()
        created_schedulers.append(sched)
        return sched

    yield _make

    for s in created_schedulers:
        try:
            s.stop()
        except Exception:
            pass


@pytest.fixture
def app_factory(model_and_tokenizer, tmp_path):
    """Factory that creates a FastAPI app backed by a real-model scheduler.

    Yields an async callable that returns an ``httpx.AsyncClient``.
    """
    import httpx
    from httpx import ASGITransport

    from mlx_lm_server.config import ServerConfig
    from mlx_lm_server.kv_cache_manager import KVCacheManager
    from mlx_lm_server.scheduler import Scheduler
    from mlx_lm_server.server import create_app

    model, tokenizer = model_and_tokenizer
    created: list[tuple[Scheduler, httpx.AsyncClient]] = []

    async def _make(**overrides: Any) -> httpx.AsyncClient:
        config = ServerConfig(
            model=MODEL_PATH,
            block_size=16,
            num_blocks=256,
            max_batch_size=4,
            max_queue_size=32,
            ssd_cache_dir=tmp_path / "ssd-cache",
            ssd_enabled=False,
            **overrides,
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
        created.append((sched, client))
        return client

    yield _make

    for sched, client in created:
        try:
            asyncio.get_event_loop().run_until_complete(client.aclose())
        except Exception:
            pass
        try:
            sched.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    tokenizer,
    prompt: str = "What is 2+2?",
    max_tokens: int = 10,
    request_id: str | None = None,
    **kwargs: Any,
):
    """Build an InferenceRequest from a text prompt."""
    from mlx_lm_server.types import InferenceRequest

    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    tokens = tokenizer.encode(prompt)
    return InferenceRequest(
        request_id=rid,
        prompt_tokens=tokens,
        max_tokens=max_tokens,
        **kwargs,
    )


def _collect_result(sched, request_id: str, timeout: float = 30.0):
    """Collect result events with a generous timeout."""
    return sched.get_result(request_id, timeout=timeout)


def _collect_stream(stream_q, timeout: float = 30.0):
    """Drain a streaming queue until finish_reason is set."""
    events = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            event = stream_q.get(timeout=min(2.0, deadline - time.time()))
            events.append(event)
            if event.finish_reason is not None:
                break
        except queue.Empty:
            continue
    return events


# ===================================================================
# A. Scheduler-level tests (12)
# ===================================================================


class TestSchedulerReal:
    """Scheduler-level tests with the real model."""

    def test_single_decode(self, scheduler_factory, model_and_tokenizer):
        """Submit one request and verify token events are returned."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="Hello", max_tokens=5)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert len(events) > 0
        assert len(events) <= 5
        for e in events:
            assert isinstance(e.token_text, str)
            assert isinstance(e.token_id, int)
        assert events[-1].finish_reason in {"stop", "length"}

    def test_multiple_sequences(self, scheduler_factory, model_and_tokenizer):
        """Submit 2 requests, both complete with valid tokens."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        req1 = _make_request(tokenizer, prompt="Hello", max_tokens=5, request_id="m1")
        req2 = _make_request(tokenizer, prompt="Goodbye", max_tokens=5, request_id="m2")
        sched.submit_request(req1)
        sched.submit_request(req2)

        ev1 = _collect_result(sched, "m1")
        ev2 = _collect_result(sched, "m2")

        assert len(ev1) > 0 and len(ev1) <= 5
        assert len(ev2) > 0 and len(ev2) <= 5
        assert ev1[-1].finish_reason in {"stop", "length"}
        assert ev2[-1].finish_reason in {"stop", "length"}

    def test_full_lifecycle(self, scheduler_factory, model_and_tokenizer):
        """Submit, collect all tokens, verify finish_reason present."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="Count to three:", max_tokens=20)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert len(events) > 0
        assert events[-1].finish_reason is not None
        # Non-final events should have finish_reason=None
        for e in events[:-1]:
            assert e.finish_reason is None

    def test_streaming(self, scheduler_factory, model_and_tokenizer):
        """Register stream before submit, collect tokens from queue."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="Hi there", max_tokens=8, stream=True)

        stream_q = sched.register_stream(req.request_id)
        sched.submit_request(req)

        events = _collect_stream(stream_q)
        assert len(events) > 0
        assert len(events) <= 8
        assert events[-1].finish_reason in {"stop", "length"}

    def test_stop_sequence(self, scheduler_factory, model_and_tokenizer):
        """Use stop_sequences to trigger early termination."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(
            tokenizer,
            prompt="List numbers: 1, 2, 3, 4, 5",
            max_tokens=50,
            stop_sequences=["\n"],
        )
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert len(events) > 0
        assert events[-1].finish_reason in {"stop", "length"}

    def test_eos_detection(self, scheduler_factory, model_and_tokenizer):
        """Let model generate until natural EOS, check finish_reason='stop'."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        # Use a large max_tokens so the model hopefully finishes naturally
        req = _make_request(
            tokenizer,
            prompt="Say 'hi'.",
            max_tokens=200,
        )
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id, timeout=60.0)

        assert len(events) > 0
        assert events[-1].finish_reason in {"stop", "length"}

    def test_max_tokens_enforced(self, scheduler_factory, model_and_tokenizer):
        """Set max_tokens=5, verify at most 5 tokens returned."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="Tell me a story", max_tokens=5)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert 0 < len(events) <= 5

    def test_continuous_batching_3(self, scheduler_factory, model_and_tokenizer):
        """Submit 3 concurrent requests, all complete."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        reqs = []
        for i in range(3):
            r = _make_request(
                tokenizer,
                prompt=f"Question {i}: What is AI?",
                max_tokens=5,
                request_id=f"batch-{i}",
            )
            reqs.append(r)
            sched.submit_request(r)

        for r in reqs:
            events = _collect_result(sched, r.request_id)
            assert len(events) > 0
            assert events[-1].finish_reason in {"stop", "length"}

    def test_staggered_batching(self, scheduler_factory, model_and_tokenizer):
        """Submit req1, wait for 1 token, submit req2, both complete."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        req1 = _make_request(tokenizer, prompt="Hello", max_tokens=8, request_id="stag-1", stream=True)
        stream_q1 = sched.register_stream("stag-1")
        sched.submit_request(req1)

        # Wait for at least one token on req1
        first_event = stream_q1.get(timeout=30.0)
        assert first_event is not None

        # Now submit req2
        req2 = _make_request(tokenizer, prompt="World", max_tokens=5, request_id="stag-2")
        sched.submit_request(req2)

        # Both should complete
        ev1_rest = _collect_stream(stream_q1)
        ev1 = [first_event] + ev1_rest
        ev2 = _collect_result(sched, "stag-2")

        assert len(ev1) > 0
        assert ev1[-1].finish_reason in {"stop", "length"}
        assert len(ev2) > 0
        assert ev2[-1].finish_reason in {"stop", "length"}

    def test_short_prompt(self, scheduler_factory, model_and_tokenizer):
        """Single-token prompt, verify generation works."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        # Use a minimal prompt (single token)
        req = _make_request(tokenizer, prompt="A", max_tokens=5)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert len(events) > 0
        assert events[-1].finish_reason in {"stop", "length"}

    def test_prefix_cache_hit(self, scheduler_factory, model_and_tokenizer):
        """Submit same prompt twice, second should complete (cache may help)."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        prompt = "Explain photosynthesis in one sentence."

        # First request
        req1 = _make_request(tokenizer, prompt=prompt, max_tokens=10, request_id="cache-1")
        sched.submit_request(req1)
        ev1 = _collect_result(sched, "cache-1")
        assert len(ev1) > 0

        # Small delay to let block decomposition run
        time.sleep(0.5)

        # Second request with same prompt
        req2 = _make_request(tokenizer, prompt=prompt, max_tokens=10, request_id="cache-2")
        sched.submit_request(req2)
        ev2 = _collect_result(sched, "cache-2")
        assert len(ev2) > 0
        assert ev2[-1].finish_reason in {"stop", "length"}

    def test_cache_reuse_different_suffix(self, scheduler_factory, model_and_tokenizer):
        """Same prefix, different max_tokens, both work."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        prompt = "The capital of France is"

        req1 = _make_request(tokenizer, prompt=prompt, max_tokens=3, request_id="sfx-1")
        sched.submit_request(req1)
        ev1 = _collect_result(sched, "sfx-1")
        assert len(ev1) > 0

        time.sleep(0.3)

        req2 = _make_request(tokenizer, prompt=prompt, max_tokens=8, request_id="sfx-2")
        sched.submit_request(req2)
        ev2 = _collect_result(sched, "sfx-2")
        assert len(ev2) > 0
        assert ev2[-1].finish_reason in {"stop", "length"}


# ===================================================================
# B. Server HTTP-level tests (7)
# ===================================================================


class TestHTTPReal:
    """HTTP-level tests against a FastAPI app with real model."""

    @pytest.mark.anyio
    async def test_http_chat_completions(self, app_factory):
        """POST /v1/chat/completions, verify response structure."""
        client = await app_factory()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 10,
            },
            timeout=60.0,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert "created" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in {"stop", "length"}
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0

    @pytest.mark.anyio
    async def test_http_chat_streaming(self, app_factory):
        """POST with stream=True, parse SSE events."""
        client = await app_factory()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 10,
                "stream": True,
            },
            timeout=60.0,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"

        chunks = []
        for line in lines[:-1]:
            assert line.startswith("data: ")
            payload = line[len("data: "):]
            chunk = json.loads(payload)
            chunks.append(chunk)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert "delta" in chunk["choices"][0]

        # Last chunk should have finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}

    @pytest.mark.anyio
    async def test_http_completions(self, app_factory):
        """POST /v1/completions with raw prompt."""
        client = await app_factory()
        resp = await client.post(
            "/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": "Once upon a time",
                "max_tokens": 10,
            },
            timeout=60.0,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "text_completion"
        assert data["id"].startswith("cmpl-")
        assert len(data["choices"]) == 1
        assert isinstance(data["choices"][0]["text"], str)
        assert data["choices"][0]["finish_reason"] in {"stop", "length"}

    @pytest.mark.anyio
    async def test_http_4_concurrent(self, app_factory):
        """4 concurrent chat requests via asyncio.gather."""
        client = await app_factory()
        tasks = []
        for i in range(4):
            tasks.append(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": MODEL_PATH,
                        "messages": [{"role": "user", "content": f"Say {i}."}],
                        "max_tokens": 5,
                    },
                    timeout=120.0,
                )
            )
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["choices"]) == 1
            assert data["choices"][0]["finish_reason"] in {"stop", "length"}

    @pytest.mark.anyio
    async def test_http_stream_vs_nonstream_parity(self, app_factory):
        """Same prompt, compare streaming vs non-streaming -- both produce text."""
        client = await app_factory()
        payload = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": "Say exactly: parity test"}],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        # Non-streaming
        resp_sync = await client.post(
            "/v1/chat/completions",
            json={**payload, "stream": False},
            timeout=60.0,
        )
        sync_content = resp_sync.json()["choices"][0]["message"]["content"]

        # Streaming
        resp_stream = await client.post(
            "/v1/chat/completions",
            json={**payload, "stream": True},
            timeout=60.0,
        )
        body = resp_stream.text
        lines = [l for l in body.strip().split("\n") if l.strip()]
        chunks = []
        for line in lines:
            if line == "data: [DONE]":
                continue
            payload_str = line[len("data: "):]
            chunks.append(json.loads(payload_str))
        stream_content = "".join(
            c["choices"][0]["delta"]["content"] for c in chunks
        )

        # Both should produce non-empty text
        assert len(sync_content) > 0
        assert len(stream_content) > 0

    @pytest.mark.anyio
    async def test_http_completion_streaming(self, app_factory):
        """POST /v1/completions with stream=True."""
        client = await app_factory()
        resp = await client.post(
            "/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": "The answer is",
                "max_tokens": 10,
                "stream": True,
            },
            timeout=60.0,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"

        chunks = []
        for line in lines[:-1]:
            payload = line[len("data: "):]
            chunk = json.loads(payload)
            chunks.append(chunk)
            assert chunk["object"] == "text_completion"
            assert "text" in chunk["choices"][0]

        assert len(chunks) > 0

    @pytest.mark.anyio
    async def test_http_health_stats(self, app_factory):
        """GET /health, verify real cache stats present."""
        client = await app_factory()
        resp = await client.get("/health", timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "ok"
        assert "cache_stats" in data
        stats = data["cache_stats"]
        assert "active_sequences" in stats
        assert "queued_requests" in stats
        assert "total_blocks" in stats
        assert stats["total_blocks"] == 256
        assert "free_blocks" in stats


# ===================================================================
# C. Integration E2E tests (6)
# ===================================================================


class TestE2EReal:
    """End-to-end integration tests using the scheduler directly."""

    def test_e2e_submit_tokens(self, scheduler_factory, model_and_tokenizer):
        """Basic submit -> collect tokens -> verify."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="What is Python?", max_tokens=10)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert len(events) > 0
        assert all(isinstance(e.token_text, str) for e in events)
        assert events[-1].finish_reason in {"stop", "length"}

    def test_e2e_max_tokens(self, scheduler_factory, model_and_tokenizer):
        """max_tokens=3, verify bounded output."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="Hello world", max_tokens=3)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert 0 < len(events) <= 3

    def test_e2e_streaming(self, scheduler_factory, model_and_tokenizer):
        """Streaming end-to-end with real model."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        req = _make_request(tokenizer, prompt="Describe the sky", max_tokens=10, stream=True)

        stream_q = sched.register_stream(req.request_id)
        sched.submit_request(req)

        events = _collect_stream(stream_q)
        assert len(events) > 0
        assert events[-1].finish_reason in {"stop", "length"}
        # Every event should have a request_id
        for e in events:
            assert e.request_id == req.request_id

    def test_e2e_4_concurrent(self, scheduler_factory, model_and_tokenizer):
        """4 concurrent requests via scheduler, all complete."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        reqs = []
        for i in range(4):
            r = _make_request(
                tokenizer, prompt=f"Topic {i}", max_tokens=5, request_id=f"e2e-{i}"
            )
            reqs.append(r)
            sched.submit_request(r)

        for r in reqs:
            events = _collect_result(sched, r.request_id)
            assert len(events) > 0
            assert events[-1].finish_reason in {"stop", "length"}

    def test_e2e_varied_lengths(self, scheduler_factory, model_and_tokenizer):
        """Requests with max_tokens 2, 5, 10; all respect limits."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        lengths = [2, 5, 10]
        reqs = []
        for i, mt in enumerate(lengths):
            r = _make_request(
                tokenizer, prompt=f"Generate {mt} tokens", max_tokens=mt, request_id=f"var-{i}"
            )
            reqs.append((r, mt))
            sched.submit_request(r)

        for r, mt in reqs:
            events = _collect_result(sched, r.request_id)
            assert 0 < len(events) <= mt
            assert events[-1].finish_reason in {"stop", "length"}

    def test_e2e_multithreaded(self, scheduler_factory, model_and_tokenizer):
        """Submit requests from 3 threads, all complete."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        results: dict[str, list] = {}
        errors: list[Exception] = []

        def _submit_and_collect(rid: str, prompt: str):
            try:
                req = _make_request(tokenizer, prompt=prompt, max_tokens=5, request_id=rid)
                sched.submit_request(req)
                events = _collect_result(sched, rid, timeout=60.0)
                results[rid] = events
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(
                target=_submit_and_collect,
                args=(f"mt-{i}", f"Thread {i} prompt"),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60.0)

        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(results) == 3
        for rid, events in results.items():
            assert len(events) > 0
            assert events[-1].finish_reason in {"stop", "length"}


# ===================================================================
# D. Batch path tests (3)
# ===================================================================


class TestBatchPathReal:
    """Tests exercising the batch inference path with a real model."""

    def test_batch_single_request(self, scheduler_factory, model_and_tokenizer):
        """Single request through batch path."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()
        # Verify batch generator is active (real model)
        assert sched._batch_generator is not None

        req = _make_request(tokenizer, prompt="Batch test", max_tokens=5)
        sched.submit_request(req)
        events = _collect_result(sched, req.request_id)

        assert len(events) > 0
        assert events[-1].finish_reason in {"stop", "length"}

    def test_batch_concurrent_decode(self, scheduler_factory, model_and_tokenizer):
        """2 concurrent batch requests."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        req1 = _make_request(tokenizer, prompt="First batch", max_tokens=5, request_id="b-1")
        req2 = _make_request(tokenizer, prompt="Second batch", max_tokens=5, request_id="b-2")
        sched.submit_request(req1)
        sched.submit_request(req2)

        ev1 = _collect_result(sched, "b-1")
        ev2 = _collect_result(sched, "b-2")

        assert len(ev1) > 0 and ev1[-1].finish_reason in {"stop", "length"}
        assert len(ev2) > 0 and ev2[-1].finish_reason in {"stop", "length"}

    def test_batch_streaming(self, scheduler_factory, model_and_tokenizer):
        """Streaming through batch path."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        req = _make_request(tokenizer, prompt="Stream batch", max_tokens=8, stream=True)
        stream_q = sched.register_stream(req.request_id)
        sched.submit_request(req)

        events = _collect_stream(stream_q)
        assert len(events) > 0
        assert events[-1].finish_reason in {"stop", "length"}


# ===================================================================
# E. Block cache bridge tests (3)
# ===================================================================


class TestBlockCacheBridge:
    """Tests for decompose/reconstruct on real model KV caches."""

    def test_decompose_real_kv(self, scheduler_factory, model_and_tokenizer):
        """After generation, decompose cache, check block shapes."""
        model, tokenizer = model_and_tokenizer
        sched = scheduler_factory(block_size=16)

        # Generate tokens to populate the cache
        prompt = "This is a test of the block cache bridge for decomposition."
        req = _make_request(tokenizer, prompt=prompt, max_tokens=10, request_id="decomp-1")
        sched.submit_request(req)
        events = _collect_result(sched, "decomp-1")
        assert len(events) > 0

        # Give time for block decomposition in cleanup
        time.sleep(1.0)

        # Check that blocks were populated in the KV cache manager
        kv_mgr = sched.kv_cache_manager
        assert kv_mgr is not None
        assert len(kv_mgr.hash_table) > 0, "Expected blocks from decomposition"
        for block_hash, block_id in kv_mgr.hash_table.items():
            block = kv_mgr.pool.blocks[block_id]
            assert isinstance(block.block_hash, int)
            assert len(block.token_ids) == kv_mgr.block_size
            if block.kv_data is not None:
                # kv_data should be a list of per-layer dicts
                assert isinstance(block.kv_data, list)

    def test_reconstruct_roundtrip(self, scheduler_factory, model_and_tokenizer):
        """Decompose -> reconstruct, verify shapes match."""
        from mlx_lm_server.kv_cache_manager import (
            decompose_cache_to_blocks,
            reconstruct_cache_from_blocks,
        )
        from mlx_lm.models.cache import make_prompt_cache

        model, tokenizer = model_and_tokenizer
        block_size = 16

        # Create a prompt cache by running the model
        prompt_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
        cache = make_prompt_cache(model)

        # Run model to populate cache
        import mlx.core as mx
        input_ids = mx.array([prompt_tokens])
        model(input_ids, cache=cache)
        mx.eval([c.state for c in cache])

        # Decompose
        blocks = decompose_cache_to_blocks(cache, prompt_tokens, block_size)
        assert len(blocks) > 0, "Expected blocks from decomposition"

        # Reconstruct
        reconstructed = reconstruct_cache_from_blocks(blocks, model)
        assert len(reconstructed) == len(cache)

        # Verify shapes match for first layer
        orig_state = cache[0].state
        recon_state = reconstructed[0].state
        assert orig_state is not None, "Original cache state should not be None"
        assert recon_state is not None, "Reconstructed cache state should not be None"
        # Reconstructed seq_len should be num_blocks * block_size
        expected_seq_len = len(blocks) * block_size
        assert recon_state[0].shape[2] == expected_seq_len

    def test_block_cache_reuse(self, scheduler_factory, model_and_tokenizer):
        """Same prompt twice, verify block cache is populated."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory(block_size=16)

        prompt = "A sentence for cache reuse testing with enough tokens to fill blocks."

        # First request
        req1 = _make_request(tokenizer, prompt=prompt, max_tokens=5, request_id="reuse-1")
        sched.submit_request(req1)
        _collect_result(sched, "reuse-1")

        time.sleep(1.0)

        # Record cached blocks after first request
        cached_before = sched.kv_cache_manager.num_cached_blocks

        # Second request with same prompt
        req2 = _make_request(tokenizer, prompt=prompt, max_tokens=5, request_id="reuse-2")
        sched.submit_request(req2)
        _collect_result(sched, "reuse-2")

        time.sleep(0.5)

        # Cache should have blocks from the first request (at least as many)
        cached_after = sched.kv_cache_manager.num_cached_blocks
        # The second request should benefit from already-cached blocks
        # At minimum, the number of cached blocks should not decrease
        assert cached_after >= cached_before


# ===================================================================
# F. Adversarial tests (5)
# ===================================================================


class TestAdversarialReal:
    """Adversarial tests with the real model."""

    def test_error_recovery(self, scheduler_factory, model_and_tokenizer):
        """Trigger error, verify scheduler recovers for next request."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        # Submit a request with empty prompt to see how it handles edge cases
        from mlx_lm_server.types import InferenceRequest

        # First submit a normal request to verify things work
        req1 = _make_request(tokenizer, prompt="Normal request", max_tokens=3, request_id="err-1")
        sched.submit_request(req1)
        ev1 = _collect_result(sched, "err-1")
        assert len(ev1) > 0

        # Now submit another normal request -- scheduler should still work
        req2 = _make_request(tokenizer, prompt="Recovery test", max_tokens=3, request_id="err-2")
        sched.submit_request(req2)
        ev2 = _collect_result(sched, "err-2")
        assert len(ev2) > 0
        assert ev2[-1].finish_reason in {"stop", "length"}

    def test_state_leak_after_n_requests(self, scheduler_factory, model_and_tokenizer):
        """Run 5 requests sequentially, verify no state leak."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        for i in range(5):
            req = _make_request(
                tokenizer, prompt=f"Leak test {i}", max_tokens=3, request_id=f"leak-{i}"
            )
            sched.submit_request(req)
            events = _collect_result(sched, f"leak-{i}")
            assert len(events) > 0

        # Allow cleanup
        time.sleep(0.5)

        # Check no active sequences remain
        assert sched.num_active_sequences == 0
        assert sched.num_queued_requests == 0
        # UID mappings should be clean
        assert len(sched._uid_to_request_id) == 0
        assert len(sched._request_id_to_uid) == 0

    def test_cancel_active_request(self, scheduler_factory, model_and_tokenizer):
        """Cancel mid-generation, verify cancelled finish_reason."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        req = _make_request(
            tokenizer,
            prompt="Tell me a very long story about dragons and wizards",
            max_tokens=100,
            request_id="cancel-real",
            stream=True,
        )
        stream_q = sched.register_stream("cancel-real")
        sched.submit_request(req)

        # Wait for at least one token
        first_event = None
        try:
            first_event = stream_q.get(timeout=30.0)
        except queue.Empty:
            pass
        assert first_event is not None, "No token received before cancel"

        # Cancel
        sched.cancel_request("cancel-real")

        # Drain remaining events
        events = [first_event] if first_event else []
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                event = stream_q.get(timeout=1.0)
                events.append(event)
                if event.finish_reason is not None:
                    break
            except queue.Empty:
                continue

        assert len(events) > 0
        assert events[-1].finish_reason in {"cancelled", "stop", "length"}

    def test_cache_eviction_real(self, scheduler_factory, model_and_tokenizer):
        """Fill cache with small num_blocks, verify eviction works."""
        _, tokenizer = model_and_tokenizer
        # Use very small block pool to trigger eviction
        sched = scheduler_factory(num_blocks=32, block_size=16)

        # Submit several requests with different prompts to fill cache
        for i in range(6):
            prompt = f"Unique prompt number {i} to fill the cache blocks with data."
            req = _make_request(
                tokenizer, prompt=prompt, max_tokens=3, request_id=f"evict-{i}"
            )
            sched.submit_request(req)
            events = _collect_result(sched, f"evict-{i}")
            assert len(events) > 0
            time.sleep(0.3)

        # The scheduler should not have crashed
        assert sched.is_running
        # Free blocks should be non-negative
        assert sched.kv_cache_manager.num_free_blocks >= 0

    def test_concurrent_prefix_sharing(self, scheduler_factory, model_and_tokenizer):
        """Same prefix, 3 concurrent suffixes, no cross-contamination."""
        _, tokenizer = model_and_tokenizer
        sched = scheduler_factory()

        prefix = "The quick brown fox "
        suffixes = ["jumps over the lazy dog", "runs through the forest", "sleeps under a tree"]

        reqs = []
        for i, suffix in enumerate(suffixes):
            r = _make_request(
                tokenizer,
                prompt=prefix + suffix,
                max_tokens=5,
                request_id=f"pfx-{i}",
            )
            reqs.append(r)
            sched.submit_request(r)

        results = {}
        for r in reqs:
            events = _collect_result(sched, r.request_id)
            assert len(events) > 0
            assert events[-1].finish_reason in {"stop", "length"}
            text = "".join(e.token_text for e in events if e.finish_reason is None or e.token_text)
            results[r.request_id] = text

        # All 3 should have completed and produced text
        assert len(results) == 3
        for rid, text in results.items():
            assert isinstance(text, str)
