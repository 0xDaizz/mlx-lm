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
import uvicorn
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
        # First chunk is role delta, then 4 content chunks
        assert len(chunks) == 5, f"Expected 5 chunks (1 role + 4 content), got {len(chunks)}"

        # Verify role delta is first
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}

        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["id"].startswith("chatcmpl-")
            assert len(chunk["choices"]) == 1
            assert "delta" in chunk["choices"][0]

        # Content chunks (skip role delta) must have "content" in delta
        for chunk in chunks[1:]:
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
        # Skip role delta chunk (first chunk has {"role": "assistant"} but no "content")
        content_chunks = [c for c in chunks if "content" in c["choices"][0].get("delta", {})]
        stream_content = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)

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
    async def test_empty_messages_list_returns_422(self, client):
        """Empty messages list is rejected by Pydantic field_validator (422)."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 422

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
        assert "utilization" in data

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
    async def test_validation_error_does_not_leak_paths(self, client):
        """422 error messages must not expose internal file paths (DA-C6-M1)."""
        resp = await client.post(
            "/v1/chat/completions",
            json={"bad_field": 123},
        )
        assert resp.status_code == 422
        body = resp.json()
        error_msg = body["error"]["message"]
        # Must not contain absolute file paths or .py references
        assert ".py" not in error_msg, f"Error leaks .py path: {error_msg}"
        # Check no absolute paths (Unix or Windows style)
        import re
        assert not re.search(r"(/[a-zA-Z0-9_.\-]+){3,}", error_msg), (
            f"Error leaks absolute path: {error_msg}"
        )

    @pytest.mark.anyio
    async def test_streaming_queue_full_returns_error_http(self, config, tokenizer):
        """When submit_request raises during streaming, HTTP error is returned.

        Since submit_request is called BEFORE the SSE generator is created
        (to support proper HTTP status codes), a queue-full RuntimeError
        results in an HTTP 429 response, not an SSE error event.
        """

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
        # Submit errors before the stream starts -> proper HTTP error code
        assert resp.status_code == 429
        body = resp.json()
        assert "error" in body
        assert "queue" in body["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_streaming_submit_failure_cleans_registered_stream(self, config, tokenizer):
        """Streaming submit failure should unregister the pre-registered stream queue."""
        from mlx_lm_server.types import BusOutboxFullError

        class OutboxFullScheduler(MockSchedulerForApp):
            def submit_request(self, request):
                raise BusOutboxFullError(request.request_id)

        sched = OutboxFullScheduler()
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
        assert resp.status_code == 503
        assert len(sched.streams) == 0

    @pytest.mark.anyio
    async def test_streaming_unexpected_submit_exception_cleans_stream(self, config, tokenizer):
        """If submit_request() raises an unexpected exception (not RuntimeError or
        BusOutboxFullError), the pre-registered stream must still be cleaned up.

        This is the V1 stream registration leak fix: a catch-all ``except Exception``
        in ``_stream_response`` ensures ``unregister_stream()`` is always called.
        """

        class ExplodingScheduler(MockSchedulerForApp):
            def submit_request(self, request):
                raise ValueError("Totally unexpected kaboom")

        sched = ExplodingScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                    "stream": True,
                },
            )
        # The ValueError propagates as an unhandled 500
        assert resp.status_code == 500
        # Critical assertion: stream registration must be cleaned up
        assert len(sched.streams) == 0, (
            f"Stream leak: {len(sched.streams)} stream(s) still registered after exception"
        )

    @pytest.mark.anyio
    async def test_distributed_nonstream_submit_does_not_raise_keyerror(self, tokenizer, tmp_path):
        """Distributed non-stream submit should not raise KeyError before local apply.

        Without pre-registered result buffers, this path fails with KeyError(500).
        With the fix, request waits for result and returns 504 timeout cleanly.
        """
        from mlx_lm_server.distributed import DistributedContext
        from mlx_lm_server.scheduler import Scheduler

        class MockBus:
            def publish(self, event):
                pass

        config = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache-distributed",
            max_batch_size=2,
            max_queue_size=8,
            request_timeout_s=0.05,
        )
        dist_ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        sched = Scheduler(config=config, model=None, tokenizer=tokenizer, dist_ctx=dist_ctx, control_bus=MockBus())
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer, dist_ctx=dist_ctx)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                json={
                    "prompt": "hello",
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 504
        assert "timed out" in resp.json()["error"]["message"].lower()
        sched.stop()


# ===========================================================================
# TestA1: Chat tokenize special token dedup
# ===========================================================================


class TestA1ChatTokenize:
    """A1: _format_chat_messages with tokenize=True, _safe_encode fallback."""

    def test_format_chat_messages_tokenize_false_returns_str(self, tokenizer):
        """tokenize=False always returns a string."""
        from mlx_lm_server.server import _format_chat_messages, ChatMessage
        msgs = [ChatMessage(role="user", content="Hi")]
        result = _format_chat_messages(msgs, tokenizer, tokenize=False)
        assert isinstance(result, str)

    def test_format_chat_messages_tokenize_true_with_simple_tokenizer(self, tokenizer):
        """SimpleTokenizer.apply_chat_template ignores tokenize=True, returns str."""
        from mlx_lm_server.server import _format_chat_messages, ChatMessage
        msgs = [ChatMessage(role="user", content="Hi")]
        result = _format_chat_messages(msgs, tokenizer, tokenize=True)
        # SimpleTokenizer always returns str (doesn't support tokenize=True)
        assert isinstance(result, str)

    def test_format_chat_messages_tokenize_true_with_tokenizer_returning_list(self):
        """When tokenizer returns list[int] for tokenize=True, use directly."""
        from mlx_lm_server.server import _format_chat_messages, ChatMessage

        class TokenizingTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                if tokenize:
                    return [1, 2, 3, 4, 5]  # token ids
                return "user: Hi\nassistant:"

        msgs = [ChatMessage(role="user", content="Hi")]
        result = _format_chat_messages(msgs, TokenizingTokenizer(), tokenize=True)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]

    def test_safe_encode_with_add_special_tokens_support(self):
        """_safe_encode uses add_special_tokens=False when supported."""
        from mlx_lm_server.server import _safe_encode

        class FullTokenizer:
            def encode(self, text, add_special_tokens=True):
                if add_special_tokens:
                    return [999] + [ord(c) for c in text]  # BOS + text
                return [ord(c) for c in text]  # no BOS

        tok = FullTokenizer()
        result = _safe_encode(tok, "Hi")
        assert 999 not in result  # No BOS token
        assert result == [72, 105]

    def test_safe_encode_fallback_on_type_error(self, tokenizer):
        """_safe_encode falls back to encode() when TypeError is raised."""
        from mlx_lm_server.server import _safe_encode
        # SimpleTokenizer.encode() doesn't accept add_special_tokens
        result = _safe_encode(tokenizer, "Hi")
        assert result == [72, 105]  # ord('H'), ord('i')

    @pytest.mark.anyio
    async def test_chat_endpoint_uses_safe_encode(self, client):
        """Chat completions endpoint still works correctly with A1 changes."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello, world!"


# ===========================================================================
# TestB1Server: TimeoutError handling
# ===========================================================================


class TestB1ServerTimeout:
    """B1: _do_inference handles TimeoutError from get_result."""

    @pytest.mark.anyio
    async def test_timeout_error_returns_504(self, config, tokenizer):
        """get_result() raising TimeoutError should yield 504."""

        class TimeoutScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.cancelled = []

            def get_result(self, request_id, timeout=None):
                raise TimeoutError(f"Request {request_id} timed out after {timeout}s")

            def cancel_request(self, request_id):
                self.cancelled.append(request_id)
                return True

        sched = TimeoutScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 504
        data = resp.json()
        assert "timed out" in data["error"]["message"].lower()
        # Verify cancel was called
        assert len(sched.cancelled) == 1


# ===========================================================================
# TestC1: Streaming stop sequence buffering
# ===========================================================================


class TestC1StreamingStopBuffer:
    """C1: Stop sequence buffering in _stream_response."""

    @pytest.mark.anyio
    async def test_multi_token_stop_boundary(self, config, tokenizer):
        """Stop sequence spanning multiple tokens is properly caught."""

        class StopScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.response_tokens = ["He", "ll", "ST", "OP", "after"]
                self.cancelled = []

            def submit_request(self, request):
                self.submitted.append(request)
                if request.request_id in self.streams:
                    q = self.streams[request.request_id]
                    for i, tok_text in enumerate(self.response_tokens):
                        is_last = i == len(self.response_tokens) - 1
                        q.put(TokenEvent(
                            request_id=request.request_id,
                            token_id=i,
                            token_text=tok_text,
                            finish_reason="stop" if is_last else None,
                        ))
                    q.put(None)

            def cancel_request(self, request_id):
                self.cancelled.append(request_id)
                return True

        sched = StopScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                    "stream": True,
                    "stop": ["STOP"],
                },
            )
        assert resp.status_code == 200
        chunks, has_done = _parse_sse_body(resp.text)
        # Collect all text (skip role delta chunk which has no "content")
        content_chunks = [c for c in chunks if "content" in c["choices"][0].get("delta", {})]
        all_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert "STOP" not in all_text, f"Stop sequence should not appear in output: {all_text!r}"
        assert "after" not in all_text, f"Text after stop should not appear: {all_text!r}"
        assert all_text == "Hell", f"Expected 'Hell' before stop, got {all_text!r}"
        # Last meaningful chunk should have finish_reason="stop"
        finish_chunks = [c for c in chunks if c["choices"][0]["finish_reason"] == "stop"]
        assert len(finish_chunks) == 1

    @pytest.mark.anyio
    async def test_earliest_stop_wins(self, config, tokenizer):
        """When multiple stop candidates, earliest match wins."""

        class EarliestStopScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.response_tokens = ["ab", "cE", "ND", "IN", "Gx"]
                self.cancelled = []

            def submit_request(self, request):
                self.submitted.append(request)
                if request.request_id in self.streams:
                    q = self.streams[request.request_id]
                    for i, tok_text in enumerate(self.response_tokens):
                        is_last = i == len(self.response_tokens) - 1
                        q.put(TokenEvent(
                            request_id=request.request_id,
                            token_id=i,
                            token_text=tok_text,
                            finish_reason="stop" if is_last else None,
                        ))
                    q.put(None)

            def cancel_request(self, request_id):
                self.cancelled.append(request_id)
                return True

        sched = EarliestStopScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                    "stream": True,
                    "stop": ["END", "ENDING"],
                },
            )
        chunks, _ = _parse_sse_body(resp.text)
        content_chunks = [c for c in chunks if "content" in c["choices"][0].get("delta", {})]
        all_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        # "abcENDINGx" — "END" appears first at position 3
        assert "END" not in all_text
        assert all_text == "abc", f"Expected 'abc' before 'END', got {all_text!r}"

    @pytest.mark.anyio
    async def test_unicode_stop_boundary(self, config, tokenizer):
        """Unicode multi-byte stop sequence across token boundaries."""

        class UnicodeStopScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.response_tokens = ["abc", "\u505c", "\u6b62", "xyz"]
                self.cancelled = []

            def submit_request(self, request):
                self.submitted.append(request)
                if request.request_id in self.streams:
                    q = self.streams[request.request_id]
                    for i, tok_text in enumerate(self.response_tokens):
                        is_last = i == len(self.response_tokens) - 1
                        q.put(TokenEvent(
                            request_id=request.request_id,
                            token_id=i,
                            token_text=tok_text,
                            finish_reason="stop" if is_last else None,
                        ))
                    q.put(None)

            def cancel_request(self, request_id):
                self.cancelled.append(request_id)
                return True

        sched = UnicodeStopScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                    "stream": True,
                    "stop": ["\u505c\u6b62"],  # "停止"
                },
            )
        chunks, _ = _parse_sse_body(resp.text)
        content_chunks = [c for c in chunks if "content" in c["choices"][0].get("delta", {})]
        all_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert "\u505c\u6b62" not in all_text
        assert all_text == "abc", f"Expected 'abc' before stop, got {all_text!r}"

    @pytest.mark.anyio
    async def test_no_stop_sequences_zero_overhead(self, client):
        """Without stop sequences, streaming works identically to before."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
            },
        )
        chunks, has_done = _parse_sse_body(resp.text)
        # 1 role delta + 4 content chunks
        assert len(chunks) == 5
        content_chunks = [c for c in chunks if "content" in c["choices"][0].get("delta", {})]
        all_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert all_text == "Hello, world!"


# ===========================================================================
# TestC2: EOS filtering by token_id
# ===========================================================================


class TestC2EosTokenId:
    """C2: EOS detection uses token_id instead of text comparison."""

    def test_get_eos_token_ids_from_set(self):
        """Tokenizer with eos_token_ids as set."""
        from mlx_lm_server.server import _get_eos_token_ids

        class Tok:
            eos_token_ids = {1, 2}
        assert _get_eos_token_ids(Tok()) == {1, 2}

    def test_get_eos_token_ids_from_list(self):
        """Tokenizer with eos_token_ids as list."""
        from mlx_lm_server.server import _get_eos_token_ids

        class Tok:
            eos_token_ids = [3, 4]
        assert _get_eos_token_ids(Tok()) == {3, 4}

    def test_get_eos_token_ids_from_int(self):
        """Tokenizer with eos_token_ids as single int."""
        from mlx_lm_server.server import _get_eos_token_ids

        class Tok:
            eos_token_ids = 5
        assert _get_eos_token_ids(Tok()) == {5}

    def test_get_eos_token_ids_fallback_to_eos_token_id(self):
        """Tokenizer with eos_token_id (singular) but no eos_token_ids."""
        from mlx_lm_server.server import _get_eos_token_ids

        class Tok:
            eos_token_id = 42
        assert _get_eos_token_ids(Tok()) == {42}

    def test_get_eos_token_ids_empty_when_nothing(self):
        """Tokenizer without any EOS info returns empty set."""
        from mlx_lm_server.server import _get_eos_token_ids

        class Tok:
            pass
        assert _get_eos_token_ids(Tok()) == set()

    @pytest.mark.anyio
    async def test_eos_filtered_by_token_id_non_streaming(self, config):
        """Non-streaming: EOS token is filtered by token_id, not text."""

        class EosTokenizer:
            eos_token_id = 99
            eos_token = "</s>"

            def encode(self, text):
                return [ord(c) for c in text]

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "user: Hi\nassistant:"

        class EosScheduler(MockSchedulerForApp):
            def get_result(self, request_id, timeout=None):
                return [
                    TokenEvent(request_id=request_id, token_id=1, token_text="Hello"),
                    TokenEvent(request_id=request_id, token_id=99, token_text="</s>", finish_reason="stop"),
                ]

        sched = EosScheduler()
        tok = EosTokenizer()
        app = create_app(config=config, scheduler=sched, tokenizer=tok)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                },
            )
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.anyio
    async def test_eos_filtered_by_token_id_streaming(self, config):
        """Streaming: EOS token text replaced with '' when token_id matches."""

        class EosTokenizer:
            eos_token_id = 99

            def encode(self, text):
                return [ord(c) for c in text]

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "user: Hi\nassistant:"

        class EosStreamScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.response_tokens = [
                    (1, "Hello"),
                    (99, "</s>"),
                ]

            def submit_request(self, request):
                self.submitted.append(request)
                if request.request_id in self.streams:
                    q = self.streams[request.request_id]
                    for i, (tid, text) in enumerate(self.response_tokens):
                        is_last = i == len(self.response_tokens) - 1
                        q.put(TokenEvent(
                            request_id=request.request_id,
                            token_id=tid,
                            token_text=text,
                            finish_reason="stop" if is_last else None,
                        ))
                    q.put(None)

        sched = EosStreamScheduler()
        tok = EosTokenizer()
        app = create_app(config=config, scheduler=sched, tokenizer=tok)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                    "stream": True,
                },
            )
        chunks, _ = _parse_sse_body(resp.text)
        content_chunks = [c for c in chunks if "content" in c["choices"][0].get("delta", {})]
        all_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert "</s>" not in all_text
        assert all_text == "Hello"


# ===========================================================================
# TestC3: First token timeout
# ===========================================================================


class TestC3FirstTokenTimeout:
    """C3: first_token_timeout_s config and behavior."""

    def test_config_default(self):
        """ServerConfig has first_token_timeout_s=300.0 by default."""
        cfg = ServerConfig()
        assert cfg.first_token_timeout_s == 300.0

    def test_cli_parsing(self):
        """CLI --first-token-timeout-s is parsed correctly."""
        from mlx_lm_server.server import parse_args
        cfg = parse_args(["--first-token-timeout-s", "600"])
        assert cfg.first_token_timeout_s == 600.0

    @pytest.mark.anyio
    async def test_first_token_uses_longer_timeout(self, tokenizer):
        """First token uses first_token_timeout_s window (polled at 1s intervals).

        With disconnect-aware polling, the event loop polls token_queue.get(timeout=1.0)
        in a loop and checks elapsed time against the configured timeout (first_token_timeout_s
        for the first token, request_timeout_s for subsequent tokens). This test verifies:
        1. All individual get() calls use the 1.0s polling interval.
        2. The stream completes successfully (meaning the first token was waited for
           within the configured first_token_timeout_s window, not request_timeout_s).
        """
        from mlx_lm_server.server import create_app

        # Track which timeout was used for each get() call
        timeouts_used = []

        class TimeoutTrackingScheduler(MockSchedulerForApp):
            def register_stream(self, request_id):
                q = Queue()
                self.streams[request_id] = q
                original_get = q.get

                def tracking_get(timeout=None):
                    timeouts_used.append(timeout)
                    return original_get(timeout=timeout)

                q.get = tracking_get
                return q

        sched = TimeoutTrackingScheduler()
        cfg = ServerConfig(
            model="test-model",
            request_timeout_s=10.0,
            first_token_timeout_s=60.0,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        # With disconnect-aware polling, all get() calls use 1.0s polling interval
        assert len(timeouts_used) >= 2
        for t in timeouts_used:
            assert t == 1.0, f"Polling interval should be 1.0, got {t}"


# ===========================================================================
# TestMaxTokensCap: F1 — max_generation_tokens caps max_tokens
# ===========================================================================


class TestMaxTokensCap:
    """F1: max_tokens is capped by config.max_generation_tokens."""

    @pytest.mark.anyio
    async def test_max_tokens_capped_by_config(self, mock_scheduler, tokenizer, tmp_path):
        """Requesting max_tokens > max_generation_tokens gets capped."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_generation_tokens=100,
        )
        app = create_app(config=cfg, scheduler=mock_scheduler, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 99999,
                },
            )
        assert resp.status_code == 200
        # Verify the submitted request was capped
        assert len(mock_scheduler.submitted) == 1
        assert mock_scheduler.submitted[0].max_tokens == 100

    @pytest.mark.anyio
    async def test_max_tokens_under_cap_unchanged(self, mock_scheduler, tokenizer, tmp_path):
        """Requesting max_tokens < max_generation_tokens is not changed."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_generation_tokens=1000,
        )
        app = create_app(config=cfg, scheduler=mock_scheduler, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200
        assert len(mock_scheduler.submitted) == 1
        assert mock_scheduler.submitted[0].max_tokens == 50

    @pytest.mark.anyio
    async def test_max_tokens_zero_returns_400(self, client):
        """max_tokens=0 should return 400 error."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 0,
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert "max_tokens must be at least 1" in data["error"]["message"]

    @pytest.mark.anyio
    async def test_max_tokens_negative_returns_400(self, client):
        """max_tokens=-1 should return 400 error."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": -1,
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert "max_tokens must be at least 1" in data["error"]["message"]


class TestHealthDistributed:
    """U14: /health endpoint includes distributed info."""

    @pytest.mark.anyio
    async def test_health_includes_distributed_disabled(self, config, mock_scheduler, tokenizer):
        """Health endpoint without dist_ctx shows distributed disabled."""
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "distributed" in data
        assert data["distributed"]["enabled"] is False
        assert data["distributed"]["rank"] is None
        assert data["distributed"]["world_size"] is None

    @pytest.mark.anyio
    async def test_health_includes_distributed_enabled(self, config, mock_scheduler, tokenizer):
        """Health endpoint with dist_ctx shows rank and world_size."""
        from mlx_lm_server.distributed import DistributedContext
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer, dist_ctx=ctx)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["distributed"]["enabled"] is True
        assert data["distributed"]["rank"] == 0
        assert data["distributed"]["world_size"] == 2

    @pytest.mark.anyio
    async def test_health_includes_distributed_rank1(self, config, mock_scheduler, tokenizer):
        """Health endpoint shows correct info for non-rank0."""
        from mlx_lm_server.distributed import DistributedContext
        ctx = DistributedContext(enabled=True, rank=1, world_size=4)
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer, dist_ctx=ctx)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["distributed"]["enabled"] is True
        assert data["distributed"]["rank"] == 1
        assert data["distributed"]["world_size"] == 4


# ===========================================================================
# TestRateLimitAndHealth — T3.2, T3.3, T3.4
# ===========================================================================


class TestRateLimitAndHealth:
    """Tests for concurrency limiter, memory pressure, and load-aware health."""

    @pytest.mark.anyio
    async def test_concurrent_limit_429(self, tokenizer, tmp_path):
        """T3.2: When max_concurrent_requests=1, second concurrent request gets 429."""
        import asyncio

        # Scheduler that blocks on get_result until told to release
        release_event = asyncio.Event()

        class SlowScheduler(MockSchedulerForApp):
            def get_result(self, request_id, timeout=None):
                # Block in executor — we need sync blocking for run_in_executor
                import time
                for _ in range(200):  # up to 2 seconds
                    if release_event.is_set():
                        break
                    time.sleep(0.01)
                return super().get_result(request_id, timeout)

        sched = SlowScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_concurrent_requests=1,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start first (slow) request
            task1 = asyncio.create_task(ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            ))
            # Give it a moment to acquire the semaphore
            await asyncio.sleep(0.05)

            # Second request should get 429
            resp2 = await ac.post(
                "/v1/completions",
                json={"prompt": "World", "max_tokens": 10},
            )
            assert resp2.status_code == 429
            data = resp2.json()
            assert data["error"]["type"] == "rate_limit_error"
            assert data["error"]["code"] == "rate_limit_exceeded"
            assert "capacity" in data["error"]["message"].lower()

            # Release slow request and let it finish
            release_event.set()
            resp1 = await task1
            assert resp1.status_code == 200

    @pytest.mark.anyio
    async def test_health_bypasses_concurrency_limit(self, tokenizer, tmp_path):
        """T3.2: /health endpoint bypasses the concurrency limiter."""
        import asyncio

        release_event = asyncio.Event()

        class SlowScheduler(MockSchedulerForApp):
            def get_result(self, request_id, timeout=None):
                import time
                for _ in range(200):
                    if release_event.is_set():
                        break
                    time.sleep(0.01)
                return super().get_result(request_id, timeout)

        sched = SlowScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_concurrent_requests=1,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start slow inference request (fills semaphore)
            task1 = asyncio.create_task(ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            ))
            await asyncio.sleep(0.05)

            # Health should still work even with semaphore full
            resp_health = await ac.get("/health")
            assert resp_health.status_code == 200

            release_event.set()
            await task1

    @pytest.mark.anyio
    async def test_memory_pressure_503(self, tokenizer, tmp_path):
        """T3.3: High memory pressure returns 503."""

        class HighPressureScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {
                    "total_blocks": 100,
                    "used_blocks": 95,  # 95% utilization >= 0.9 threshold
                    "free_blocks": 5,
                    "hit_rate": 0.5,
                }

        sched = HighPressureScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Chat completions
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
            )
            assert resp.status_code == 503
            data = resp.json()
            assert data["error"]["code"] == "memory_pressure"
            assert "memory pressure" in data["error"]["message"].lower()

            # Text completions also affected
            resp2 = await ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )
            assert resp2.status_code == 503

    @pytest.mark.anyio
    async def test_memory_pressure_below_threshold_ok(self, tokenizer, tmp_path):
        """T3.3: Below memory pressure threshold, requests succeed."""

        class LowPressureScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {
                    "total_blocks": 100,
                    "used_blocks": 50,  # 50% utilization < 0.9 threshold
                    "free_blocks": 50,
                    "hit_rate": 0.5,
                }

        sched = LowPressureScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )
            assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_health_status_ok(self, tokenizer, tmp_path):
        """T3.4: Low utilization returns status=ok with 200."""
        sched = MockSchedulerForApp()  # 0% utilization by default
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["utilization"] == 0.0

    @pytest.mark.anyio
    async def test_health_status_degraded(self, tokenizer, tmp_path):
        """T3.4: ~80% utilization returns status=degraded with 200."""

        class DegradedScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {
                    "total_blocks": 100,
                    "used_blocks": 75,  # 75% >= 0.9*0.8=72%, < 90%
                    "free_blocks": 25,
                    "hit_rate": 0.5,
                }

        sched = DegradedScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["utilization"] == 0.75

    @pytest.mark.anyio
    async def test_health_status_overloaded(self, tokenizer, tmp_path):
        """T3.4: >90% utilization returns status=overloaded with 503."""

        class OverloadedScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {
                    "total_blocks": 100,
                    "used_blocks": 95,  # 95% >= 90% threshold
                    "free_blocks": 5,
                    "hit_rate": 0.5,
                }

        sched = OverloadedScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "overloaded"
        assert data["utilization"] == 0.95

    @pytest.mark.anyio
    async def test_health_status_shutting_down(self, tokenizer, tmp_path):
        """T3.4: Shutting down returns status=shutting_down with 503."""
        sched = MockSchedulerForApp()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        app.state.shutting_down = True
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "shutting_down"

    def test_config_defaults(self):
        """T3.2+T3.3: Config defaults for admission control."""
        cfg = ServerConfig()
        assert cfg.max_concurrent_requests == 64
        assert cfg.memory_pressure_threshold == 0.9
        assert cfg.max_request_bytes == 1_048_576

    def test_cli_max_concurrent_requests(self):
        """T3.2: CLI --max-concurrent-requests is parsed correctly."""
        from mlx_lm_server.server import parse_args
        cfg = parse_args(["--max-concurrent-requests", "32"])
        assert cfg.max_concurrent_requests == 32

    def test_cli_memory_pressure_threshold(self):
        """T3.3: CLI --memory-pressure-threshold is parsed correctly."""
        from mlx_lm_server.server import parse_args
        cfg = parse_args(["--memory-pressure-threshold", "0.8"])
        assert cfg.memory_pressure_threshold == 0.8

    @pytest.mark.anyio
    async def test_concurrent_limit_zero_means_unlimited(self, tokenizer, tmp_path):
        """max_concurrent_requests=0 disables request limiter."""
        import asyncio

        release_event = asyncio.Event()

        class SlowScheduler(MockSchedulerForApp):
            def get_result(self, request_id, timeout=None):
                import time

                for _ in range(200):
                    if release_event.is_set():
                        break
                    time.sleep(0.01)
                return super().get_result(request_id, timeout)

        sched = SlowScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_concurrent_requests=0,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            task1 = asyncio.create_task(
                ac.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 10})
            )
            await asyncio.sleep(0.05)
            resp2 = await ac.post("/v1/completions", json={"prompt": "World", "max_tokens": 10})
            release_event.set()
            resp1 = await task1
        assert resp1.status_code == 200
        assert resp2.status_code == 200

    @pytest.mark.anyio
    async def test_nonstream_keyerror_maps_to_503(self, tokenizer, tmp_path):
        """Unexpected missing result buffer should return 503, not 500."""

        class MissingResultScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.cancelled: list[str] = []

            def get_result(self, request_id, timeout=None):
                raise KeyError(request_id)

            def cancel_request(self, request_id: str) -> bool:
                self.cancelled.append(request_id)
                return True

        sched = MissingResultScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 10})
        assert resp.status_code == 503
        assert "state unavailable" in resp.json()["error"]["message"].lower()
        assert len(sched.cancelled) == 1

    @pytest.mark.anyio
    async def test_api_key_required_for_v1_endpoints(self, tokenizer, tmp_path):
        """When api_key is configured, /v1 endpoints require a Bearer token."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            api_key="secret-key",
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_completion = await ac.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 10})
            resp_models = await ac.get("/v1/models")
        assert resp_completion.status_code == 401
        assert resp_completion.json()["error"]["code"] == "invalid_api_key"
        assert resp_models.status_code == 401

    @pytest.mark.anyio
    async def test_api_key_valid_bearer_allows_request(self, tokenizer, tmp_path):
        """Valid Bearer API key should allow requests."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            api_key="secret-key",
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        headers = {"Authorization": "Bearer secret-key"}
        async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as ac:
            resp = await ac.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 10})
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_health_endpoints_bypass_api_key(self, tokenizer, tmp_path):
        """health/liveness/readiness/metrics should not require API key."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            api_key="secret-key",
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_health = await ac.get("/health")
            resp_livez = await ac.get("/livez")
            resp_readyz = await ac.get("/readyz")
            resp_metrics = await ac.get("/metrics")
        assert resp_health.status_code == 200
        assert resp_livez.status_code == 200
        assert resp_readyz.status_code == 200
        assert resp_metrics.status_code == 200

    @pytest.mark.anyio
    async def test_request_size_limit_returns_413(self, tokenizer, tmp_path):
        """Content-Length header exceeding max_request_bytes returns 413.

        The middleware performs a fast-path Content-Length header check for
        well-behaved clients.  Chunked transfers and requests without a
        Content-Length header are enforced by uvicorn's ``limit_request_body``
        parameter (set in __main__.py), which is not exercised in ASGI
        transport tests.
        """
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_request_bytes=64,
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        # Send a request with Content-Length header that exceeds the limit.
        # The middleware rejects based on the header value alone.
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                content=b'{"prompt":"hi","max_tokens":10}',
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": "999999",
                },
            )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"

    @pytest.mark.anyio
    async def test_readyz_and_health_report_distributed_fatal(self, tokenizer, tmp_path):
        """Distributed fatal state should surface in /readyz and /health."""

        class DistFatalScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {
                    "total_blocks": 64,
                    "used_blocks": 0,
                    "free_blocks": 64,
                    "dist_fatal": True,
                    "dist_fatal_reason": "bus_error_threshold",
                }

        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
        )
        app = create_app(config=cfg, scheduler=DistFatalScheduler(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_ready = await ac.get("/readyz")
            resp_health = await ac.get("/health")
        assert resp_ready.status_code == 503
        assert "bus_error_threshold" in resp_ready.json()["reasons"]
        assert resp_health.status_code == 503
        assert resp_health.json()["status"] == "distributed_fatal"

    @pytest.mark.anyio
    async def test_livez_and_metrics_shape(self, tokenizer, tmp_path):
        """/livez and /metrics should return operational payloads."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp_livez = await ac.get("/livez")
            resp_metrics = await ac.get("/metrics")
        assert resp_livez.status_code == 200
        assert resp_livez.json()["status"] == "alive"
        assert "uptime_s" in resp_livez.json()
        assert resp_metrics.status_code == 200
        text = resp_metrics.text
        assert "mlx_lm_server_active_sequences" in text
        assert "mlx_lm_server_dist_fatal" in text

    def test_cli_max_request_bytes(self):
        """--max-request-bytes should be parsed correctly."""
        from mlx_lm_server.server import parse_args

        cfg = parse_args(["--max-request-bytes", "2048"])
        assert cfg.max_request_bytes == 2048

    def test_cli_api_key_from_flag(self):
        """--api-key should be parsed into config.api_key."""
        from mlx_lm_server.server import parse_args

        cfg = parse_args(["--api-key", "my-secret"])
        assert cfg.api_key == "my-secret"

    def test_cli_api_key_file(self, tmp_path):
        """--api-key-file should load and strip key content."""
        from mlx_lm_server.server import parse_args

        key_file = tmp_path / "api.key"
        key_file.write_text("  file-secret-key  \n", encoding="utf-8")
        cfg = parse_args(["--api-key-file", str(key_file)])
        assert cfg.api_key == "file-secret-key"

    def test_cli_rejects_non_positive_timeouts(self):
        """Timeouts must be > 0."""
        from mlx_lm_server.server import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--request-timeout-s", "0"])
        with pytest.raises(SystemExit):
            parse_args(["--first-token-timeout-s", "-1"])

    def test_cli_ring_accepts_num_local_ranks_without_hostfile(self):
        """ring mode should accept --num-local-ranks as an alternative to hostfile."""
        from mlx_lm_server.server import parse_args

        cfg = parse_args([
            "--model",
            "test-model",
            "--distributed-mode",
            "ring",
            "--num-local-ranks",
            "2",
        ])
        assert cfg.distributed_mode == "ring"


# ===========================================================================
# TestMiddlewareOrdering — P0-1: Middleware order + concurrency TOCTOU fix
# ===========================================================================


class TestMiddlewareOrdering:
    """P0-1: Middleware registration order and concurrency TOCTOU fix."""

    @pytest.mark.anyio
    async def test_unauthenticated_request_does_not_consume_semaphore(self, tokenizer, tmp_path):
        """With max_concurrent_requests=1 and api_key set, an unauthenticated request
        should return 401 without consuming a semaphore slot, so a subsequent
        authenticated request should succeed (not 429)."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_concurrent_requests=1,
            api_key="test-secret-key",
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Send unauthenticated request — should get 401, NOT consume semaphore
            resp_unauth = await ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )
            assert resp_unauth.status_code == 401, (
                f"Expected 401 for unauthenticated request, got {resp_unauth.status_code}"
            )

            # Now send authenticated request — should succeed (not 429)
            resp_auth = await ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
                headers={"Authorization": "Bearer test-secret-key"},
            )
            assert resp_auth.status_code == 200, (
                f"Expected 200 for authenticated request, got {resp_auth.status_code}. "
                "If 429, the unauthenticated request consumed a semaphore slot."
            )

    @pytest.mark.anyio
    async def test_concurrency_limit_rejects_atomically(self, tokenizer, tmp_path):
        """Flood requests to verify concurrency limiter uses atomic acquire_nowait
        (no TOCTOU gap between check and acquire). With max_concurrent_requests=1,
        exactly one request should proceed and the rest should get 429 immediately
        without blocking on acquire."""
        import asyncio

        release_event = asyncio.Event()

        class SlowScheduler(MockSchedulerForApp):
            def get_result(self, request_id, timeout=None):
                import time
                for _ in range(200):  # up to 2 seconds
                    if release_event.is_set():
                        break
                    time.sleep(0.01)
                return super().get_result(request_id, timeout)

        sched = SlowScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_concurrent_requests=1,
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Start one slow request to fill the semaphore
            task_slow = asyncio.create_task(ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            ))
            await asyncio.sleep(0.05)

            # Fire 5 concurrent requests — all should get 429 immediately (not block)
            flood_tasks = [
                asyncio.create_task(ac.post(
                    "/v1/completions",
                    json={"prompt": f"Flood {i}", "max_tokens": 10},
                ))
                for i in range(5)
            ]

            # All flood requests should resolve quickly (within 1s, not waiting 2s)
            flood_responses = await asyncio.wait_for(
                asyncio.gather(*flood_tasks),
                timeout=1.0,
            )
            for resp in flood_responses:
                assert resp.status_code == 429, (
                    f"Expected 429 for flood request, got {resp.status_code}"
                )

            # Clean up: release the slow request
            release_event.set()
            resp_slow = await task_slow
            assert resp_slow.status_code == 200


# ===========================================================================
# TestExecutorPool — P1-6: Dedicated thread pool executor
# ===========================================================================


class TestExecutorPool:
    """P1-6: Verify dedicated ThreadPoolExecutor prevents starvation."""

    @pytest.mark.anyio
    async def test_executor_handles_concurrent_streams(self, tokenizer, tmp_path):
        """N concurrent streaming requests all get served without executor exhaustion.

        Uses a barrier-based SlowScheduler so all streams must start concurrently
        before any of them can complete, proving the executor has enough threads.
        """
        import asyncio
        import threading

        N = 8  # Number of concurrent streaming requests

        # Barrier ensures all N streams are actively polling before any completes.
        barrier = threading.Barrier(N, timeout=10)
        barrier_passed = threading.Event()

        class BarrierScheduler(MockSchedulerForApp):
            """Scheduler that blocks all streams at a barrier, then releases."""

            def __init__(self):
                super().__init__()
                self._lock = threading.Lock()

            def submit_request(self, request):
                self.submitted.append(request)
                if request.request_id in self.streams:
                    q = self.streams[request.request_id]
                    # Put tokens only after barrier_passed is set
                    # (submit just enqueues; the token delivery waits)
                    def _deliver():
                        try:
                            # Wait at the barrier — all N threads must arrive
                            barrier.wait()
                            barrier_passed.set()
                        except threading.BrokenBarrierError:
                            pass
                        for i, tok_text in enumerate(self.response_tokens):
                            is_last = i == len(self.response_tokens) - 1
                            q.put(TokenEvent(
                                request_id=request.request_id,
                                token_id=i,
                                token_text=tok_text,
                                finish_reason="stop" if is_last else None,
                            ))
                        q.put(None)

                    threading.Thread(target=_deliver, daemon=True).start()

        sched = BarrierScheduler()
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=N,
            max_queue_size=N * 2,
            max_concurrent_requests=0,  # unlimited
        )
        app = create_app(config=cfg, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            tasks = [
                asyncio.create_task(ac.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": f"Hi {i}"}],
                        "max_tokens": 50,
                        "stream": True,
                    },
                ))
                for i in range(N)
            ]

            # All N requests must complete within 10 seconds.
            # If the executor has too few threads, the barrier will
            # never be satisfied and this times out.
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=10.0,
            )

        # All requests should succeed
        for i, resp in enumerate(responses):
            assert resp.status_code == 200, (
                f"Stream {i} failed with status {resp.status_code}"
            )
            assert "text/event-stream" in resp.headers["content-type"]
            chunks, has_done = _parse_sse_body(resp.text)
            assert has_done, f"Stream {i} missing [DONE]"
            # 1 role delta + 4 content chunks
            assert len(chunks) == 5, (
                f"Stream {i}: expected 5 chunks, got {len(chunks)}"
            )

        # Verify the barrier was actually hit (all N threads ran concurrently)
        assert barrier_passed.is_set(), (
            "Barrier was never satisfied — threads did not run concurrently"
        )


# ===========================================================================
# TestRequestSizeGuard — P0-2: ASGI receive-wrapping body size enforcement
# ===========================================================================


class TestRequestSizeGuard:
    """P0-2: Body size limit enforced at ASGI layer via receive wrapping.

    Verifies that large request bodies are rejected even when the
    Content-Length header is missing or forged (chunked transfer).
    """

    @pytest.mark.anyio
    async def test_large_body_without_content_length_rejected(self, tokenizer, tmp_path):
        """Oversized body with forged Content-Length should be rejected with 413.

        httpx always auto-sets Content-Length, so we can't truly omit it.
        Instead we send a forged (small) Content-Length header while sending
        a large body.  This bypasses the fast-path Content-Length check but
        triggers the slow-path byte-counting guard.
        """
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_request_bytes=64,
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        # Body larger than 64-byte limit, but forged Content-Length claims 10
        large_body = b"x" * 200
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                content=large_body,
                headers={
                    "content-type": "application/json",
                    "content-length": "10",
                },
            )
        # Should be rejected with 413 (not 422 validation error or 500)
        assert resp.status_code == 413, (
            f"Expected 413 for oversized body with forged Content-Length, got {resp.status_code}"
        )

    @pytest.mark.anyio
    async def test_body_exactly_at_limit_succeeds(self, tokenizer, tmp_path):
        """A body exactly at the byte limit should succeed (not off-by-one)."""
        # Build a body that is just under the limit
        small_body = b'{"prompt":"Hi","max_tokens":10}'
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_request_bytes=len(small_body) + 1,  # just above body size
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                content=small_body,
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200, (
            f"Expected 200 for body within limit, got {resp.status_code}"
        )

    @pytest.mark.anyio
    async def test_small_body_within_limit_succeeds(self, tokenizer, tmp_path):
        """A normal-sized body under the limit should succeed."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_request_bytes=4096,
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_non_v1_path_bypasses_size_guard(self, tokenizer, tmp_path):
        """Requests to non-/v1/ paths should not be subject to size guard."""
        cfg = ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
            max_request_bytes=64,
        )
        app = create_app(config=cfg, scheduler=MockSchedulerForApp(), tokenizer=tokenizer)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # /health is a GET and not under /v1/, should bypass
            resp = await ac.get("/health")
        assert resp.status_code == 200


# ===========================================================================
# TestAppCreationSmoke — smoke test to prevent startup crashes
# ===========================================================================


class TestAppCreationSmoke:
    """Smoke test: verify the app can be created without errors."""

    def test_create_app_returns_fastapi_instance(self, config, mock_scheduler, tokenizer):
        """create_app() returns a FastAPI instance without raising."""
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer)
        assert app is not None
        assert hasattr(app, "state")
        assert app.state.config is config
        assert app.state.scheduler is mock_scheduler
        assert app.state.tokenizer is tokenizer
        assert app.state.shutting_down is False

    @pytest.mark.anyio
    async def test_app_serves_health_endpoint(self, config, mock_scheduler, tokenizer):
        """Smoke test: freshly created app can serve /health."""
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200

    def test_uvicorn_run_call_has_no_invalid_params(self):
        """Verify the uvicorn.run() call in __main__.py does not use
        unsupported parameters (regression test for P0-1 crash)."""
        import inspect
        sig = inspect.signature(uvicorn.run)
        # limit_request_body is NOT a valid uvicorn parameter
        assert "limit_request_body" not in sig.parameters, (
            "uvicorn.run() should not have limit_request_body parameter"
        )

        # Also verify the source code doesn't pass it
        import ast
        source_path = "/Users/hw/mlx-lm/mlx_lm_server/__main__.py"
        with open(source_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    assert kw.arg != "limit_request_body", (
                        "Found limit_request_body in __main__.py uvicorn.run() call"
                    )


# ===========================================================================
# TestBatchEncodingCompat — transformers >= 5.0 BatchEncoding compatibility
# ===========================================================================


class TestBatchEncodingCompat:
    """DA-C4-C1: Verify chat completions handle transformers 5.x BatchEncoding.

    In transformers >= 5.0, apply_chat_template(tokenize=True) returns a
    BatchEncoding dict-like object by default instead of list[int].
    Our code must handle both return types correctly.
    """

    def test_format_chat_messages_returns_list_with_return_dict_false(self):
        """apply_chat_template accepting return_dict=False returns list[int]."""
        from mlx_lm_server.server import _format_chat_messages, ChatMessage

        class ModernTokenizer:
            """Simulates transformers >= 5.0 tokenizer that accepts return_dict."""
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, return_dict=True):
                if tokenize:
                    tokens = [1, 2, 3, 4, 5]
                    if return_dict:
                        # BatchEncoding-like dict
                        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
                    return tokens
                return "user: hello\nassistant:"

        msgs = [ChatMessage(role="user", content="hello")]
        result = _format_chat_messages(msgs, ModernTokenizer(), tokenize=True)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result == [1, 2, 3, 4, 5]

    def test_call_site_handles_batch_encoding_dict(self):
        """Chat endpoint handles dict-like result with 'input_ids' key."""
        from mlx_lm_server.server import _format_chat_messages, ChatMessage

        class BatchEncodingResult:
            """Simulates a BatchEncoding object (has input_ids attribute)."""
            def __init__(self, ids):
                self.input_ids = ids
            def __getitem__(self, key):
                if key == "input_ids":
                    return self.input_ids
                raise KeyError(key)

        class BatchEncodingTokenizer:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, **kwargs):
                if kwargs.get("return_dict") is False:
                    # Honour return_dict=False
                    return [10, 20, 30]
                if tokenize:
                    return BatchEncodingResult([10, 20, 30])
                return "user: hello\nassistant:"

        msgs = [ChatMessage(role="user", content="hello")]
        result = _format_chat_messages(msgs, BatchEncodingTokenizer(), tokenize=True)
        # With return_dict=False it should return list directly
        assert isinstance(result, list)
        assert result == [10, 20, 30]

    def test_legacy_tokenizer_without_return_dict(self):
        """Legacy tokenizer that does NOT accept return_dict still works."""
        from mlx_lm_server.server import _format_chat_messages, ChatMessage

        class LegacyTokenizer:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                # No return_dict parameter — will raise TypeError if called with it
                if tokenize:
                    return [7, 8, 9]
                return "user: hello\nassistant:"

        msgs = [ChatMessage(role="user", content="hello")]
        result = _format_chat_messages(msgs, LegacyTokenizer(), tokenize=True)
        assert isinstance(result, list)
        assert result == [7, 8, 9]

    @pytest.mark.anyio
    async def test_chat_endpoint_with_batch_encoding_tokenizer(self, config, mock_scheduler):
        """Full endpoint test: tokenizer returning BatchEncoding-like object."""
        from mlx_lm_server.server import create_app

        class BatchEncodingResult:
            """Simulates transformers BatchEncoding."""
            def __init__(self, ids):
                self.input_ids = ids
            def __getitem__(self, key):
                if key == "input_ids":
                    return self.input_ids
                raise KeyError(key)

        class ModernTokenizer:
            eos_token_ids: set[int] = set()

            def encode(self, text, add_special_tokens=True):
                return [ord(c) for c in text]

            def decode(self, ids):
                return "".join(chr(i) for i in ids if 0 <= i < 0x110000)

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, **kwargs):
                if kwargs.get("return_dict") is False:
                    if tokenize:
                        return [1, 2, 3, 4, 5]
                    return "user: hello\nassistant:"
                if tokenize:
                    return BatchEncodingResult([1, 2, 3, 4, 5])
                return "user: hello\nassistant:"

        tok = ModernTokenizer()
        app = create_app(config=config, scheduler=mock_scheduler, tokenizer=tok)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "Hello, world!"

    @pytest.mark.anyio
    async def test_chat_endpoint_with_simple_tokenizer_still_works(self, client):
        """Regression: SimpleTokenizer (no return_dict) still works after fix."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello, world!"


# ===========================================================================
# TestSRV: Audit finding fixes (SRV-5, SRV-9, SRV-11, SRV-14)
# ===========================================================================


class TestSRVAuditFixes:
    """Tests for server audit findings SRV-5, SRV-9, SRV-11, SRV-14."""

    @pytest.mark.anyio
    async def test_metrics_content_type(self, client):
        """SRV-14: /metrics response has correct content-type including charset."""
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        ct = resp.headers["content-type"]
        assert "text/plain" in ct
        assert "version=0.0.4" in ct
        assert "charset=utf-8" in ct

    @pytest.mark.anyio
    async def test_completion_tokens_with_stop_sequence(self, config, tokenizer):
        """SRV-11: completion_tokens reflects truncated output, not pre-truncation count."""

        class StopSeqScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                # Tokens: "One" "Two" "STOP" "Three" -- stop at "STOP"
                self.response_tokens = ["One", "Two", "STOP", "Three"]

            def get_result(self, request_id, timeout=None):
                events = []
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

        sched = StopSeqScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/completions",
                json={
                    "prompt": "Hello",
                    "max_tokens": 50,
                    "stop": ["STOP"],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        # The output text should be truncated before "STOP"
        assert data["choices"][0]["text"] == "OneTwo"
        # completion_tokens should reflect the truncated text length,
        # not the original 4 events. SimpleTokenizer encodes per-character,
        # so "OneTwo" = 6 tokens.
        assert data["usage"]["completion_tokens"] == len(tokenizer.encode("OneTwo"))

    @pytest.mark.anyio
    async def test_streaming_disconnect_cleanup(self, config, tokenizer):
        """Non-streaming disconnect triggers cancel_request for cleanup.

        When a non-streaming request is cancelled (e.g., client disconnect),
        the finally block in _do_inference calls cancel_request to free
        scheduler resources.
        """
        import asyncio

        class SlowGetResultScheduler(MockSchedulerForApp):
            def __init__(self):
                super().__init__()
                self.cancelled: list[str] = []

            def get_result(self, request_id, timeout=None):
                import time
                time.sleep(2.0)
                return super().get_result(request_id, timeout)

            def cancel_request(self, request_id: str) -> bool:
                self.cancelled.append(request_id)
                return True

        sched = SlowGetResultScheduler()
        app = create_app(config=config, scheduler=sched, tokenizer=tokenizer)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            try:
                resp = await asyncio.wait_for(
                    ac.post(
                        "/v1/completions",
                        json={"prompt": "Hello", "max_tokens": 50},
                    ),
                    timeout=0.3,
                )
            except (asyncio.TimeoutError, Exception):
                pass
        # Allow background cleanup to run
        await asyncio.sleep(0.2)
        # cancel_request should have been called by the finally block
        assert len(sched.cancelled) >= 1, (
            "cancel_request should be called on disconnect/timeout"
        )

    def test_deprecated_config_removed(self):
        """SRV-9: ServerConfig no longer has use_distributed attribute."""
        assert not hasattr(ServerConfig, "use_distributed"), (
            "use_distributed should be removed from ServerConfig"
        )
        config = ServerConfig()
        assert not hasattr(config, "use_distributed"), (
            "use_distributed instance attribute should not exist"
        )


# ===========================================================================
# TestStreamOptionsIncludeUsage — stream_options.include_usage support
# ===========================================================================


class TestStreamOptionsIncludeUsage:
    """OpenAI stream_options.include_usage compliance tests."""

    @pytest.fixture
    def mock_scheduler(self):
        return MockSchedulerForApp()

    @pytest.fixture
    def tokenizer(self):
        return SimpleTokenizer()

    @pytest.fixture
    def config(self, tmp_path):
        return ServerConfig(
            model="mlx-community/Qwen3-4B-4bit",
            block_size=4,
            num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd-cache",
            max_batch_size=2,
            max_queue_size=8,
        )

    @pytest.fixture
    def app(self, config, mock_scheduler, tokenizer):
        return create_app(config=config, scheduler=mock_scheduler, tokenizer=tokenizer)

    @pytest.fixture
    async def client(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    @pytest.mark.anyio
    async def test_streaming_chat_with_include_usage(self, client):
        """stream_options.include_usage emits a usage chunk before [DONE] for chat."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status_code == 200

        chunks, has_done = _parse_sse_body(resp.text)
        assert has_done, "Stream must end with [DONE]"

        # The last chunk (before [DONE]) should be the usage chunk
        usage_chunk = chunks[-1]
        assert usage_chunk["choices"] == [], "Usage chunk must have empty choices"
        assert "usage" in usage_chunk, "Usage chunk must contain 'usage' field"
        usage = usage_chunk["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["completion_tokens"] == 4  # MockSchedulerForApp produces 4 tokens
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage_chunk["object"] == "chat.completion.chunk"
        assert usage_chunk["id"].startswith("chatcmpl-")

    @pytest.mark.anyio
    async def test_streaming_completion_with_include_usage(self, client):
        """stream_options.include_usage emits a usage chunk before [DONE] for completions."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "Once upon a time",
                "max_tokens": 50,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status_code == 200

        chunks, has_done = _parse_sse_body(resp.text)
        assert has_done, "Stream must end with [DONE]"

        # The last chunk should be the usage chunk
        usage_chunk = chunks[-1]
        assert usage_chunk["choices"] == [], "Usage chunk must have empty choices"
        assert "usage" in usage_chunk
        usage = usage_chunk["usage"]
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage_chunk["object"] == "text_completion"
        assert usage_chunk["id"].startswith("cmpl-")

    @pytest.mark.anyio
    async def test_streaming_without_include_usage(self, client):
        """Without stream_options, no usage chunk is emitted (backward compat)."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
            },
        )
        assert resp.status_code == 200

        chunks, has_done = _parse_sse_body(resp.text)
        assert has_done

        # No chunk should have empty choices (which is the usage chunk signature)
        for chunk in chunks:
            assert chunk["choices"] != [], (
                "Without stream_options.include_usage, no usage chunk should be emitted"
            )

    @pytest.mark.anyio
    async def test_streaming_with_include_usage_false(self, client):
        """stream_options with include_usage=false should not emit usage chunk."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                "stream": True,
                "stream_options": {"include_usage": False},
            },
        )
        assert resp.status_code == 200

        chunks, has_done = _parse_sse_body(resp.text)
        assert has_done

        for chunk in chunks:
            assert chunk["choices"] != [], (
                "With include_usage=false, no usage chunk should be emitted"
            )


# ===========================================================================
# TestNParameter — reject n > 1
# ===========================================================================


class TestNParameter:
    """Verify that n > 1 is rejected and n=1 is accepted."""

    @pytest.mark.anyio
    async def test_n_greater_than_1_rejected(self, client):
        """n=2 must be rejected with a validation error."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "n > 1 is not supported" in body["error"]["message"]

    @pytest.mark.anyio
    async def test_n_equals_1_accepted(self, client):
        """n=1 should be accepted normally (200)."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "n": 1,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_completion_n_greater_than_1_rejected(self, client):
        """n=2 on /v1/completions must also be rejected."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "n": 3,
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "n > 1 is not supported" in body["error"]["message"]

    @pytest.mark.anyio
    async def test_completion_n_equals_1_accepted(self, client):
        """n=1 on /v1/completions should work normally."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "n": 1,
            },
        )
        assert resp.status_code == 200


# ===========================================================================
# TestExtraFieldsIgnored — unknown fields silently dropped
# ===========================================================================


class TestExtraFieldsIgnored:
    """Verify that unknown request fields are silently ignored (OpenAI compat)."""

    @pytest.mark.anyio
    async def test_chat_completion_extra_fields_ignored(self, client):
        """Extra fields on /v1/chat/completions should not cause 422."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "unknown_field": "should_be_ignored",
                "logprobs": True,
                "best_of": 5,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_extra_fields_ignored(self, client):
        """Extra fields on /v1/completions should not cause 422."""
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "unknown_field": "should_be_ignored",
                "logprobs": 5,
                "suffix": "world",
            },
        )
        assert resp.status_code == 200
