"""Tests for BatchGenerator integration in Scheduler (P6.7).

Uses mock objects to test the batch path without requiring a real model.
The mock path (model=None) is tested by existing test_scheduler.py.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.types import InferenceRequest, TokenEvent


@dataclass
class MockResponse:
    """Mock of BatchGenerator.Response."""
    uid: int
    token: int
    logprobs: Any = None
    finish_reason: str | None = None
    _prompt_cache: Any = None

    def prompt_cache(self):
        return self._prompt_cache


class MockBatchGenerator:
    """Mock BatchGenerator for testing the batch path without a real model."""

    def __init__(self):
        self._uid_counter = 0
        self._active: dict[int, dict] = {}  # uid -> {tokens, max_tokens, step}
        self._closed = False
        self._removed_uids: list[int] = []

    def insert(self, prompts, max_tokens=None, caches=None, samplers=None, logits_processors=None):
        uids = []
        for i, prompt in enumerate(prompts):
            uid = self._uid_counter
            self._uid_counter += 1
            mt = max_tokens[i] if isinstance(max_tokens, list) else (max_tokens or 10)
            self._active[uid] = {
                "tokens": list(prompt) if hasattr(prompt, '__iter__') else [prompt],
                "max_tokens": mt,
                "step": 0,
            }
            uids.append(uid)
        return uids

    def next(self):
        responses = []
        finished = []
        for uid, state in self._active.items():
            state["step"] += 1
            token = 100 + state["step"]
            finish = None
            if state["step"] >= state["max_tokens"]:
                finish = "length"
                finished.append(uid)
            responses.append(MockResponse(
                uid=uid,
                token=token,
                finish_reason=finish,
            ))
        for uid in finished:
            del self._active[uid]
        return responses

    def remove(self, uids, return_prompt_caches=False):
        result = {}
        for uid in uids:
            self._removed_uids.append(uid)
            if uid in self._active:
                if return_prompt_caches:
                    result[uid] = [{"mock_cache": True}]
                del self._active[uid]
        return result if return_prompt_caches else None

    def close(self):
        self._closed = True
        self._active.clear()


def _make_scheduler_with_mock_bg(config=None, **kwargs):
    """Create a Scheduler with a MockBatchGenerator injected."""
    if config is None:
        config = ServerConfig(**kwargs)

    # Create a scheduler with a fake model so batch path activates
    sched = Scheduler(config=config, model=None, tokenizer=None)

    # Inject mock batch generator
    mock_bg = MockBatchGenerator()
    sched._batch_generator = mock_bg
    sched._sequence_cache = None

    # Create a mock tokenizer with minimal interface.
    # Set detokenizer=None so the batch path uses str(token) fallback
    # instead of trying to use a real detokenizer.
    mock_tokenizer = MagicMock()
    mock_tokenizer.detokenizer = None
    mock_tokenizer.decode = lambda ids: "".join(f"t{i}" for i in ids)
    sched.tokenizer = mock_tokenizer

    return sched, mock_bg


class TestBatchIntegration:
    """Tests for the batch inference path."""

    def test_single_request_batch_path(self):
        """Single request completes through batch path."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=3
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="req-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=3,
            )
            sched.submit_request(req)
            events = sched.get_result("req-1", timeout=5.0)

            assert len(events) == 3
            assert events[-1].finish_reason == "length"
            # Tokens should be 101, 102, 103 from mock
            assert [e.token_id for e in events] == [101, 102, 103]
        finally:
            sched.stop()

    def test_multiple_concurrent_batch(self):
        """Multiple requests run concurrently in one batch."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=2
        )
        sched.run_inference_loop()

        try:
            for i in range(3):
                req = InferenceRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=[10 + i],
                    max_tokens=2,
                )
                sched.submit_request(req)

            results = {}
            for i in range(3):
                results[f"req-{i}"] = sched.get_result(f"req-{i}", timeout=5.0)

            for rid, events in results.items():
                assert len(events) == 2
                assert events[-1].finish_reason == "length"
        finally:
            sched.stop()

    def test_cancel_during_batch(self):
        """Cancellation removes request from batch.

        Uses a slow MockBatchGenerator so the request is still active
        when cancel is called.
        """
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Slow down next() so we can cancel mid-flight
        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="cancel-me",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            stream_q = sched.register_stream("cancel-me")
            sched.submit_request(req)

            # Wait for at least one token, then cancel
            time.sleep(0.2)
            sched.cancel_request("cancel-me")

            # Should get a cancelled event eventually
            deadline = time.time() + 5.0
            got_finish = False
            finish_reason = None
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=0.5)
                    if event.finish_reason is not None:
                        got_finish = True
                        finish_reason = event.finish_reason
                        break
                except queue.Empty:
                    continue

            assert got_finish
            assert finish_reason == "cancelled"
        finally:
            sched.stop()

    def test_uid_mapping_cleanup(self):
        """UID mappings are cleaned up after request completes."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=2
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="cleanup-test",
                prompt_tokens=[1],
                max_tokens=2,
            )
            sched.submit_request(req)
            sched.get_result("cleanup-test", timeout=5.0)

            # Give cleanup time to run
            time.sleep(0.3)

            assert len(sched._uid_to_request_id) == 0
            assert len(sched._request_id_to_uid) == 0
        finally:
            sched.stop()

    def test_zero_max_tokens_batch(self):
        """max_tokens=0 finishes immediately without entering BatchGenerator."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=0
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="zero-tok",
                prompt_tokens=[1, 2, 3],
                max_tokens=0,
            )
            sched.submit_request(req)
            events = sched.get_result("zero-tok", timeout=5.0)

            assert len(events) >= 1
            assert events[-1].finish_reason == "length"
            # Should not have inserted into batch generator
            assert mock_bg._uid_counter == 0
        finally:
            sched.stop()

    def test_error_recovery_batch(self):
        """BatchGenerator error is handled, loop continues."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=5
        )

        # Make next() raise on first call
        original_next = mock_bg.next
        call_count = [0]
        def failing_next():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Mock BG failure")
            return original_next()

        mock_bg.next = failing_next
        sched.run_inference_loop()

        try:
            req1 = InferenceRequest(
                request_id="fail-req",
                prompt_tokens=[1],
                max_tokens=5,
            )
            sched.submit_request(req1)
            events = sched.get_result("fail-req", timeout=5.0)

            # Should get an error finish
            assert events[-1].finish_reason == "error"
        finally:
            sched.stop()

    def test_streaming_batch(self):
        """Streaming works through batch path -- tokens arrive incrementally."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=3
        )
        sched.run_inference_loop()

        try:
            stream_q = sched.register_stream("stream-test")
            req = InferenceRequest(
                request_id="stream-test",
                prompt_tokens=[1, 2],
                max_tokens=3,
                stream=True,
            )
            sched.submit_request(req)

            events = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=1.0)
                    events.append(event)
                    if event.finish_reason is not None:
                        break
                except queue.Empty:
                    continue

            assert len(events) == 3
            assert events[-1].finish_reason == "length"
        finally:
            sched.stop()
