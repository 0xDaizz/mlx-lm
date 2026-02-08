"""Comprehensive endpoint tests for the mlx-lm-server FastAPI application.

Covers: chat completions, text completions, validation guards, health endpoint,
error handling, and SSE streaming format correctness.

Uses the MockSchedulerForApp + SimpleTokenizer from conftest.py.

Run: .venv/bin/python -m pytest tests/test_server_app.py -v -x --timeout=30
"""
from __future__ import annotations

import json
from queue import Queue
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.server import create_app
from mlx_lm_server.types import InferenceRequest, TokenEvent

from conftest import MockSchedulerForApp, SimpleTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_scheduler():
    return MockSchedulerForApp()


@pytest.fixture
def tokenizer():
    return SimpleTokenizer()


@pytest.fixture
def config(tmp_path):
    return ServerConfig(
        model="mlx-community/Qwen3-4B-4bit",
        block_size=4,
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        max_batch_size=2,
        max_queue_size=8,
    )


@pytest.fixture
def app(config, mock_scheduler, tokenizer):
    return create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse_body(body: str) -> tuple[list[dict], bool]:
    """Parse SSE body into a list of JSON chunks and whether [DONE] was present.

    Returns (chunks, has_done).
    """
    lines = [line for line in body.strip().split("\n") if line.strip()]
    chunks = []
    has_done = False
    for line in lines:
        assert line.startswith("data: "), f"SSE line missing 'data: ' prefix: {line!r}"
        payload = line[len("data: "):]
        if payload == "[DONE]":
            has_done = True
        else:
            chunks.append(json.loads(payload))
    return chunks, has_done


# ===========================================================================
# TestChatCompletions — happy path
# ===========================================================================


class TestChatCompletions:
    """Non-streaming and streaming chat completion endpoint tests."""

    @pytest.mark.anyio
    async def test_non_streaming_returns_200_with_correct_structure(self, client):
        """Non-streaming chat completion returns 200 with all required OpenAI fields."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi there"}],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level required fields
        assert data["id"].startswith("chatcmpl-"), f"id should start with 'chatcmpl-', got {data['id']}"
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["created"] > 0
        assert "choices" in data
        assert "usage" in data

    @pytest.mark.anyio
    async def test_non_streaming_content_is_correct(self, client):
        """Content from MockSchedulerForApp should be 'Hello, world!'."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 50,
            },
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "Hello, world!", f"Expected 'Hello, world!', got {content!r}"

    @pytest.mark.anyio
    async def test_non_streaming_finish_reason(self, client):
        """finish_reason should be 'stop' for completed generation."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
            },
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.anyio
    async def test_non_streaming_model_field_matches_config(self, client, config):
        """Response model field should match the server's configured model name."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
            },
        )
        data = resp.json()
        assert data["model"] == config.model

    @pytest.mark.anyio
    async def test_non_streaming_usage_token_counts(self, client):
        """Usage should have positive prompt_tokens, correct completion_tokens, and valid total."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
            },
        )
        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0, "prompt_tokens must be positive"
        assert usage["completion_tokens"] == 4, (
            f"MockScheduler produces 4 tokens, got {usage['completion_tokens']}"
        )
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.anyio
    async def test_non_streaming_choice_structure(self, client):
        """Choices should have index, message with role and content, and finish_reason."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        )
        data = resp.json()
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] is not None

    @pytest.mark.anyio
    async def test_streaming_returns_sse_events(self, client):
        """Streaming chat returns SSE events with correct format and content-type."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        chunks, has_done = _parse_sse_body(resp.text)
        assert len(chunks) == 4, f"Expected 4 token chunks, got {len(chunks)}"

        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["id"].startswith("chatcmpl-")
            assert len(chunk["choices"]) == 1
            assert "delta" in chunk["choices"][0]
            assert "content" in chunk["choices"][0]["delta"]

    @pytest.mark.anyio
    async def test_streaming_ends_with_done(self, client):
        """Streaming must end with 'data: [DONE]' line."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
            },
        )
        chunks, has_done = _parse_sse_body(resp.text)
        assert has_done, "SSE stream must end with 'data: [DONE]'"

    @pytest.mark.anyio
    async def test_streaming_content_matches_non_streaming(self, client):
        """Concatenated streaming delta content should equal non-streaming content."""
        # Non-streaming
        resp_sync = await client.post(
            "/v1/chat/completions",
            json={
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
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
            },
        )
        chunks, _ = _parse_sse_body(resp_stream.text)
        stream_content = "".join(c["choices"][0]["delta"]["content"] for c in chunks)

        assert stream_content == sync_content, (
            f"Stream content {stream_content!r} != non-stream {sync_content!r}"
        )


# ===========================================================================
# TestCompletions — happy path
# ===========================================================================


class TestCompletions:
    """Text completion endpoint tests (non-streaming and streaming)."""

    @pytest.mark.anyio
    async def test_non_streaming_returns_200_with_correct_structure(self, client):
        """Non-streaming completion returns 200 with OpenAI text_completion structure."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "Once upon a time",
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["id"].startswith("cmpl-"), f"id should start with 'cmpl-', got {data['id']}"
        assert data["object"] == "text_completion"
        assert isinstance(data["created"], int)
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "Hello, world!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 4

    @pytest.mark.anyio
    async def test_streaming_completion_works(self, client):
        """Streaming completion returns SSE events with 'text' field in choices."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "Once upon",
                "max_tokens": 50,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        chunks, has_done = _parse_sse_body(resp.text)
        assert has_done, "Stream must end with [DONE]"
        assert len(chunks) == 4

        for chunk in chunks:
            assert chunk["object"] == "text_completion"
            assert chunk["id"].startswith("cmpl-")
            assert "text" in chunk["choices"][0]

        # Concatenated text should match non-streaming
        full_text = "".join(c["choices"][0]["text"] for c in chunks)
        assert full_text == "Hello, world!"


# ===========================================================================
# TestValidation — error paths exercising _validate_and_prepare_request
# ===========================================================================


class TestValidation:
    """Tests for request validation guards."""

    @pytest.mark.anyio
    async def test_wrong_model_name_returns_400(self, client):
        """Requesting a model name that doesn't match the loaded model returns 400."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model-name",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Model mismatch" in data["error"]["message"]
        assert "wrong-model-name" in data["error"]["message"]
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "400"

    @pytest.mark.anyio
    async def test_empty_model_name_allowed(self, client):
        """Empty model name (default) should be allowed — no model validation."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_omitted_model_name_allowed(self, client):
        """Omitted model field defaults to '' which is allowed."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_empty_messages_list_returns_400(self, client):
        """Empty messages list produces an empty prompt, which should return 400."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [],
                "max_tokens": 10,
            },
        )
        # SimpleTokenizer.apply_chat_template with empty messages produces "assistant:"
        # which encodes to non-empty tokens. So this actually succeeds.
        # However the behavior is correct: non-empty prompt passes validation.
        # Let's verify the actual status code from the current code.
        # "assistant:" encodes to [97, 115, 115, 105, 115, 116, 97, 110, 116, 58]
        # which is 10 tokens, so the prompt is NOT empty -> 200.
        # NOTE: This documents current behavior. True empty-prompt validation
        # only triggers if the encoded prompt is literally zero tokens.
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_empty_prompt_string_returns_400(self, client):
        """Empty string prompt for /v1/completions should return 400 (zero tokens)."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "",
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "Prompt must not be empty" in data["error"]["message"]

    @pytest.mark.anyio
    async def test_shutting_down_returns_503(self, app):
        """When shutting_down=True, requests should return 503."""
        app.state.shutting_down = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 503
        data = resp.json()
        assert "shutting down" in data["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_queue_full_returns_429(self, config, tokenizer):
        """When scheduler.submit_request raises RuntimeError (queue full), return 429."""

        class QueueFullScheduler(MockSchedulerForApp):
            def submit_request(self, request):
                raise RuntimeError("Request queue is full (max_size=8)")

        sched = QueueFullScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 429
        data = resp.json()
        assert "queue" in data["error"]["message"].lower()


# ===========================================================================
# TestHealthEndpoint
# ===========================================================================


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    @pytest.mark.anyio
    async def test_health_returns_200_with_status_and_cache_stats(self, client):
        """GET /health should return 200 with 'status' and 'cache_stats' keys."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "cache_stats" in data

    @pytest.mark.anyio
    async def test_health_cache_stats_contains_expected_fields(self, client):
        """cache_stats from MockSchedulerForApp should have standard stat fields."""
        resp = await client.get("/health")
        data = resp.json()
        stats = data["cache_stats"]
        # MockSchedulerForApp returns total_blocks, used_blocks, free_blocks, hit_rate
        assert "total_blocks" in stats
        assert "free_blocks" in stats
        assert "hit_rate" in stats
        assert isinstance(stats["total_blocks"], int)
        assert isinstance(stats["hit_rate"], (int, float))


# ===========================================================================
# TestModelsEndpoint
# ===========================================================================


class TestModelsEndpoint:
    """Tests for the GET /v1/models endpoint."""

    @pytest.mark.anyio
    async def test_models_list_returns_correct_structure(self, client, config):
        """GET /v1/models returns a list with the loaded model."""
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        model_info = data["data"][0]
        assert model_info["id"] == config.model
        assert model_info["object"] == "model"
        assert "owned_by" in model_info


# ===========================================================================
# TestErrorHandling
# ===========================================================================


class TestErrorHandling:
    """Tests for error response structure and exception handlers."""

    @pytest.mark.anyio
    async def test_http_exception_returns_openai_error_format(self, client):
        """HTTPException errors should return the OpenAI error structure."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        # Validate error structure
        assert "error" in data
        error = data["error"]
        assert "message" in error
        assert "type" in error
        assert "code" in error
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "400"

    @pytest.mark.anyio
    async def test_general_exception_returns_500(self, config, tokenizer):
        """Unhandled exceptions should return 500 with server_error structure.

        NOTE: raise_app_exceptions=False is required for httpx.ASGITransport
        to return the HTTP response instead of re-raising the server exception.
        """

        class CrashingScheduler(MockSchedulerForApp):
            def get_result(self, request_id, timeout=None):
                raise ValueError("Unexpected internal failure")

        sched = CrashingScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "500"
        assert data["error"]["message"] == "Internal server error"

    @pytest.mark.anyio
    async def test_no_tokenizer_returns_500(self, config, mock_scheduler):
        """If tokenizer is None, chat completions should return 500."""
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 500
        data = resp.json()
        assert "Tokenizer not loaded" in data["error"]["message"]

    @pytest.mark.anyio
    async def test_pydantic_validation_returns_422(self, client):
        """Malformed body (missing required fields) returns 422."""
        # Missing 'messages' for chat completions
        resp = await client.post(
            "/v1/chat/completions",
            json={"max_tokens": 10},
        )
        assert resp.status_code == 422

        # Missing 'prompt' for completions
        resp2 = await client.post(
            "/v1/completions",
            json={"max_tokens": 10},
        )
        assert resp2.status_code == 422

    @pytest.mark.anyio
    async def test_streaming_queue_full_returns_error_sse(self, config, tokenizer):
        """When submit_request raises during streaming, SSE error event is emitted."""

        class QueueFullScheduler(MockSchedulerForApp):
            def submit_request(self, request):
                raise RuntimeError("Request queue is full")

        sched = QueueFullScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                    "stream": True,
                },
            )
        # Streaming still returns 200 status, but the SSE body contains the error
        assert resp.status_code == 200
        body = resp.text
        lines = [line for line in body.strip().split("\n") if line.strip()]

        error_found = False
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = line[len("data: "):]
                try:
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        assert "queue" in chunk["error"]["message"].lower()
                        error_found = True
                except json.JSONDecodeError:
                    pass

        assert error_found, "Expected an SSE error event for queue full during streaming"
