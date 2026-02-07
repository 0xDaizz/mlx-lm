"""Shared fixtures for mlx-lm-server tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from queue import Queue
from typing import Any

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.server import create_app
from mlx_lm_server.types import InferenceRequest, TokenEvent


@pytest.fixture
def test_config(tmp_path: Path) -> ServerConfig:
    """Server config suitable for testing (small pool, temp SSD dir)."""
    return ServerConfig(
        model="mlx-community/Qwen3-4B-4bit",
        block_size=4,  # Small blocks for faster tests
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        ssd_ttl_days=1,
        max_batch_size=2,
        max_queue_size=8,
    )


@pytest.fixture
def tmp_ssd_dir(tmp_path: Path) -> Path:
    """Temporary directory for SSD cache tests."""
    d = tmp_path / "ssd-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# P4.1 — Integration test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Placeholder model (None) for tests that don't need a real model."""
    return None


class SimpleTokenizer:
    """A simple tokenizer mock that encodes strings to token lists.

    Encodes each character as its ordinal value. Provides encode/decode
    plus an eos_token_ids set for stop detection.
    """

    eos_token_ids: set[int] = set()

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of character ordinals."""
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return "".join(chr(i) for i in ids if 0 <= i < 0x110000)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Simple chat template: concatenate role: content lines."""
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


@pytest.fixture
def mock_tokenizer() -> SimpleTokenizer:
    """A simple tokenizer mock that encodes strings to token lists."""
    return SimpleTokenizer()


@pytest.fixture
def scheduler_with_mock(test_config) -> Scheduler:
    """A Scheduler with model=None and a mock generation callback.

    The mock generator produces tokens t0, t1, ... up to max_tokens,
    finishing with reason 'stop' on the last one.
    """
    scheduler = Scheduler(
        config=test_config,
        model=None,
        tokenizer=None,
    )

    def mock_gen(request_id: str, token_ids: list[int], step: int):
        """Generate numbered tokens; stop after 5 steps."""
        if step >= 4:
            return (step + 100, f"t{step}", "stop")
        return (step + 100, f"t{step}", None)

    scheduler._mock_generate = mock_gen
    return scheduler


class MockSchedulerForApp:
    """Scheduler mock for FastAPI app testing.

    Provides the SchedulerProtocol interface that create_app expects.
    Produces a fixed response of ['Hello', ', ', 'world', '!'].
    """

    def __init__(self) -> None:
        self.response_tokens: list[str] = ["Hello", ", ", "world", "!"]
        self.submitted: list[InferenceRequest] = []
        self.streams: dict[str, Queue[TokenEvent | None]] = {}
        self.shutdown_called: bool = False

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)
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
            q.put(None)

    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]:
        q: Queue[TokenEvent | None] = Queue()
        self.streams[request_id] = q
        return q

    def get_result(self, request_id: str) -> list[TokenEvent]:
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
            "used_blocks": 0,
            "free_blocks": 64,
            "hit_rate": 0.0,
        }

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def app_with_mock(test_config, mock_tokenizer):
    """A FastAPI test app using MockSchedulerForApp.

    Returns a tuple of (app, mock_scheduler) so tests can inspect
    submitted requests and other state.
    """
    scheduler = MockSchedulerForApp()
    app = create_app(
        config=test_config,
        scheduler=scheduler,
        tokenizer=mock_tokenizer,
    )
    return app, scheduler
