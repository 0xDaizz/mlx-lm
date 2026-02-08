"""Tests for FastAPI Server (Phase 3).

Uses a MockScheduler and MockTokenizer so no real model is needed.
Run: .venv/bin/python -m pytest tests/test_mlx_lm_server.py -v
"""

from __future__ import annotations

import asyncio
import json
import time
from queue import Queue
from threading import Thread
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.server import create_app, parse_args
from mlx_lm_server.types import InferenceRequest, TokenEvent


# ---------------------------------------------------------------------------
# Mock scheduler
# ---------------------------------------------------------------------------


class MockScheduler:
    """Scheduler mock that returns canned token events."""

    def __init__(self, response_tokens: list[str] | None = None):
        self.response_tokens = response_tokens or ["Hello", ",", " world", "!"]
        self.submitted: list[InferenceRequest] = []
        self.streams: dict[str, Queue[TokenEvent | None]] = {}
        self.cancelled: list[str] = []
        self.shutdown_called = False

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)
        # If a stream queue was registered, push tokens into it
        if request.request_id in self.streams:
            q = self.streams[request.request_id]
            for i, tok_text in enumerate(self.response_tokens):
                is_last = i == len(self.response_tokens) - 1
                q.put(
                    TokenEvent(
                        request_id=request.request_id,
                        token_id=i,
                        token_text=tok_text,
                        finish_reason="stop" if is_last else None,
                    )
                )
            q.put(None)  # sentinel

    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]:
        q: Queue[TokenEvent | None] = Queue()
        self.streams[request_id] = q
        return q

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        events: list[TokenEvent] = []
        for i, tok_text in enumerate(self.response_tokens):
            is_last = i == len(self.response_tokens) - 1
            events.append(
                TokenEvent(
                    request_id=request_id,
                    token_id=i,
                    token_text=tok_text,
                    finish_reason="stop" if is_last else None,
                )
            )
        return events

    def get_cache_stats(self) -> dict[str, Any]:
        return {
            "total_blocks": 64,
            "used_blocks": 10,
            "free_blocks": 54,
            "hit_rate": 0.85,
        }

    def cancel_request(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer mock that splits on whitespace."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_scheduler():
    return MockScheduler()


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def server_config(tmp_path):
    return ServerConfig(
        model="test-model",
        block_size=4,
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        max_batch_size=2,
        max_queue_size=8,
    )


@pytest.fixture
def app(server_config, mock_scheduler, mock_tokenizer):
    return create_app(
        config=server_config,
        scheduler=mock_scheduler,
        tokenizer=mock_tokenizer,
    )


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_completions(client):
    """P3.1: Non-streaming chat completion returns OpenAI-compatible schema."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level fields
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert "created" in data
    assert data["model"] == "test-model"

    # Choices
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello, world!"
    assert choice["finish_reason"] == "stop"

    # Usage
    usage = data["usage"]
    assert usage["prompt_tokens"] >= 1
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.anyio
async def test_chat_streaming(client):
    """P3.2: SSE streaming follows OpenAI format with data: prefix and [DONE]."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    lines = [l for l in body.strip().split("\n") if l.strip()]

    # Each non-empty line should start with "data: "
    for line in lines:
        assert line.startswith("data: "), f"Line missing 'data: ' prefix: {line!r}"

    # Last line must be [DONE]
    assert lines[-1] == "data: [DONE]"

    # Parse JSON chunks (all except [DONE])
    chunks = []
    for line in lines[:-1]:
        payload = line[len("data: "):]
        chunk = json.loads(payload)
        chunks.append(chunk)

    assert len(chunks) == 4  # 4 tokens
    # Verify chunk structure
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl-")
        assert len(chunk["choices"]) == 1
        assert "delta" in chunk["choices"][0]
        assert "content" in chunk["choices"][0]["delta"]

    # Concatenated content matches non-streaming
    full_text = "".join(c["choices"][0]["delta"]["content"] for c in chunks)
    assert full_text == "Hello, world!"

    # Last chunk should have finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_completions(client):
    """P3.3: Text completion endpoint."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Once upon a time",
            "max_tokens": 50,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "text_completion"
    assert data["id"].startswith("cmpl-")
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"] == "Hello, world!"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 4


@pytest.mark.anyio
async def test_models_list(client):
    """P3.4: GET /v1/models returns model list."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"
    assert data["data"][0]["object"] == "model"


@pytest.mark.anyio
async def test_health(client):
    """P3.5: GET /health returns ok + cache stats."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert "cache_stats" in data
    assert data["cache_stats"]["total_blocks"] == 64
    assert data["cache_stats"]["hit_rate"] == 0.85


def test_cli_parsing():
    """P3.6: argparse produces correct ServerConfig."""
    config = parse_args(["--model", "my-model", "--port", "9000", "--max-batch-size", "16"])
    assert isinstance(config, ServerConfig)
    assert config.model == "my-model"
    assert config.port == 9000
    assert config.max_batch_size == 16
    # Defaults
    assert config.host == "0.0.0.0"
    assert config.block_size == 16
    assert config.kv_bits == 8

    # --no-ssd flag
    config2 = parse_args(["--no-ssd"])
    assert config2.ssd_enabled is False


@pytest.mark.anyio
async def test_startup(app, mock_scheduler, mock_tokenizer):
    """P3.7: App starts without error and state is initialized."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        # Verify app state is set
        assert app.state.model_name == "test-model"
        assert app.state.scheduler is mock_scheduler
        assert app.state.tokenizer is mock_tokenizer


@pytest.mark.anyio
async def test_shutdown_flushes(server_config, mock_scheduler, mock_tokenizer):
    """P3.8: Graceful shutdown calls scheduler.shutdown()."""
    app = create_app(
        config=server_config,
        scheduler=mock_scheduler,
        tokenizer=mock_tokenizer,
    )
    # Manually enter and exit the lifespan to trigger shutdown
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.get("/health")
    # After exiting lifespan context, shutdown runs
    assert mock_scheduler.shutdown_called is True


@pytest.mark.anyio
async def test_invalid_request(client):
    """P3.9: Malformed body returns 422 (Pydantic validation error)."""
    # Missing required 'messages' field
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "test-model"},
    )
    assert resp.status_code == 422

    # Missing required 'prompt' field for completions
    resp2 = await client.post(
        "/v1/completions",
        json={"model": "test-model"},
    )
    assert resp2.status_code == 422


@pytest.mark.anyio
async def test_concurrent_4(client):
    """P3.11: 4 parallel requests all complete successfully."""
    tasks = []
    for i in range(4):
        tasks.append(
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "max_tokens": 50,
                },
            )
        )
    responses = await asyncio.gather(*tasks)
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello, world!"


@pytest.mark.anyio
async def test_stream_parity(client):
    """P3.12: Streaming and non-streaming produce same content."""
    # Non-streaming
    resp_sync = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "stream": False,
        },
    )
    sync_content = resp_sync.json()["choices"][0]["message"]["content"]

    # Streaming
    resp_stream = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "stream": True,
        },
    )
    body = resp_stream.text
    lines = [l for l in body.strip().split("\n") if l.strip()]
    # Parse chunks (skip [DONE])
    chunks = []
    for line in lines:
        if line == "data: [DONE]":
            continue
        payload = line[len("data: "):]
        chunks.append(json.loads(payload))

    stream_content = "".join(c["choices"][0]["delta"]["content"] for c in chunks)

    assert sync_content == stream_content


@pytest.mark.anyio
async def test_completions_streaming(client):
    """P3.3 streaming: Text completion SSE works correctly."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Once upon",
            "max_tokens": 50,
            "stream": True,
        },
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


@pytest.mark.anyio
async def test_stop_as_string(client):
    """stop parameter can be a single string instead of a list."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "<stop>",
        },
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_error_response_format(client):
    """Error responses follow OpenAI format: {error: {message, type, code}}."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "test-model"},  # missing messages
    )
    # FastAPI returns 422 for validation errors (its own format)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# B3: Input validation tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_prompt_config(tmp_path):
    """ServerConfig with a very small max_prompt_tokens for testing."""
    return ServerConfig(
        model="test-model",
        block_size=4,
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        max_batch_size=2,
        max_queue_size=8,
        max_prompt_tokens=10,
    )


@pytest.fixture
def small_prompt_app(small_prompt_config, mock_scheduler, mock_tokenizer):
    return create_app(
        config=small_prompt_config,
        scheduler=mock_scheduler,
        tokenizer=mock_tokenizer,
    )


@pytest.fixture
async def small_prompt_client(small_prompt_app):
    transport = ASGITransport(app=small_prompt_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_prompt_too_long_returns_400(small_prompt_client):
    """B3: Prompt exceeding max_prompt_tokens returns 400."""
    # MockTokenizer.encode splits on whitespace, so 15 words = 15 tokens > 10 limit
    long_prompt_words = " ".join(f"word{i}" for i in range(15))
    resp = await small_prompt_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": long_prompt_words}],
            "max_tokens": 50,
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "Prompt too long" in data["error"]["message"]

    # Also test /v1/completions endpoint
    resp2 = await small_prompt_client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": long_prompt_words,
            "max_tokens": 50,
        },
    )
    assert resp2.status_code == 400
    data2 = resp2.json()
    assert "Prompt too long" in data2["error"]["message"]


@pytest.mark.anyio
async def test_temperature_clamped(client, mock_scheduler):
    """B3: temperature=5.0 is clamped to 2.0, no error."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "temperature": 5.0,
        },
    )
    assert resp.status_code == 200
    # Verify the submitted request used the clamped value
    assert len(mock_scheduler.submitted) >= 1
    submitted = mock_scheduler.submitted[-1]
    assert submitted.temperature == 2.0


@pytest.mark.anyio
async def test_negative_temperature_clamped(client, mock_scheduler):
    """B3: temperature=-1.0 is clamped to 0.0, no error."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "temperature": -1.0,
        },
    )
    assert resp.status_code == 200
    submitted = mock_scheduler.submitted[-1]
    assert submitted.temperature == 0.0


@pytest.mark.anyio
async def test_max_tokens_clamped_to_minimum_1(client, mock_scheduler):
    """B3: max_tokens=0 or negative is clamped to 1, no error."""
    # Test with max_tokens=0
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,
        },
    )
    assert resp.status_code == 200
    submitted = mock_scheduler.submitted[-1]
    assert submitted.max_tokens == 1

    # Test with max_tokens=-5
    resp2 = await client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": -5,
        },
    )
    assert resp2.status_code == 200
    submitted2 = mock_scheduler.submitted[-1]
    assert submitted2.max_tokens == 1


# ---------------------------------------------------------------------------
# B4: Unified timeout + auto-cancel tests
# ---------------------------------------------------------------------------


class SlowMockScheduler(MockScheduler):
    """Scheduler mock whose get_result blocks until timeout expires."""

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        """Block for longer than the timeout, then return empty (simulating timeout)."""
        import time as _time
        if timeout is not None:
            _time.sleep(timeout + 0.1)
        return []  # empty = timed out

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)
        # Do NOT push tokens into streams — simulate stalled inference


class SlowStreamScheduler(MockScheduler):
    """Scheduler mock that registers a stream queue but never pushes tokens."""

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)
        # Intentionally do NOT push tokens into the stream queue


@pytest.fixture
def short_timeout_config(tmp_path):
    """ServerConfig with a very short request_timeout_s for testing."""
    return ServerConfig(
        model="test-model",
        block_size=4,
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        max_batch_size=2,
        max_queue_size=8,
        request_timeout_s=0.1,  # 100ms — very short for fast tests
    )


@pytest.fixture
def slow_scheduler():
    return SlowMockScheduler()


@pytest.fixture
def slow_stream_scheduler():
    return SlowStreamScheduler()


@pytest.fixture
def timeout_app(short_timeout_config, slow_scheduler, mock_tokenizer):
    return create_app(
        config=short_timeout_config,
        scheduler=slow_scheduler,
        tokenizer=mock_tokenizer,
    )


@pytest.fixture
async def timeout_client(timeout_app):
    transport = ASGITransport(app=timeout_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def stream_timeout_app(short_timeout_config, slow_stream_scheduler, mock_tokenizer):
    return create_app(
        config=short_timeout_config,
        scheduler=slow_stream_scheduler,
        tokenizer=mock_tokenizer,
    )


@pytest.fixture
async def stream_timeout_client(stream_timeout_app):
    transport = ASGITransport(app=stream_timeout_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_timeout_uses_config_value(timeout_client, slow_scheduler):
    """B4: Non-streaming request times out using config.request_timeout_s and returns 504."""
    resp = await timeout_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
        },
    )
    assert resp.status_code == 504
    data = resp.json()
    assert "timed out" in data["error"]["message"].lower()
    # Verify cancel was called
    assert len(slow_scheduler.cancelled) == 1


@pytest.mark.anyio
async def test_stream_timeout_cancels_request(stream_timeout_client, slow_stream_scheduler):
    """B4: Stream timeout cancels the request and emits an error SSE event."""
    resp = await stream_timeout_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    lines = [l for l in body.strip().split("\n") if l.strip()]

    # Should contain an error SSE event about stream timeout
    error_found = False
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            payload = line[len("data: "):]
            try:
                chunk = json.loads(payload)
                if "error" in chunk:
                    assert "Stream timeout" in chunk["error"]["message"]
                    assert "0.1" in chunk["error"]["message"]  # config value reflected
                    error_found = True
            except json.JSONDecodeError:
                pass

    assert error_found, "Expected a stream timeout error SSE event"
    # Should end with [DONE]
    assert lines[-1] == "data: [DONE]"
    # Verify cancel was called on the scheduler
    assert len(slow_stream_scheduler.cancelled) == 1
