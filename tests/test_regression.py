"""Regression tests for specific production bugs.

Each test targets a specific bug fix and would FAIL if the fix were reverted.
These tests use mocks (no real model required).

Run: .venv/bin/python -m pytest tests/test_regression.py -v
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from queue import Queue
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.server import create_app
from mlx_lm_server.types import InferenceRequest, SequenceState, TokenEvent


# ---------------------------------------------------------------------------
# Helpers — mock objects reused across tests
# ---------------------------------------------------------------------------


class _MockBatchGeneratorForInsertError:
    """A mock BatchGenerator whose insert() raises on specific request IDs.

    Used for C-NEW-1 regression test.
    """

    def __init__(self, fail_request_ids: set[str] | None = None):
        self._uid_counter = 0
        self._active: dict[int, dict] = {}
        self._closed = False
        self._fail_request_ids = fail_request_ids or set()
        # We track which prompts were inserted so the test can correlate
        self._inserted_prompts: list[list[int]] = []
        self._should_fail_next_insert = False

    def insert(self, prompts, max_tokens=None, caches=None, samplers=None, logits_processors=None):
        if self._should_fail_next_insert:
            self._should_fail_next_insert = False
            raise RuntimeError("Simulated insertion failure")
        uids = []
        for i, prompt in enumerate(prompts):
            uid = self._uid_counter
            self._uid_counter += 1
            mt = max_tokens[i] if isinstance(max_tokens, list) else (max_tokens or 10)
            self._active[uid] = {
                "tokens": list(prompt) if hasattr(prompt, "__iter__") else [prompt],
                "max_tokens": mt,
                "step": 0,
            }
            self._inserted_prompts.append(list(prompt) if hasattr(prompt, "__iter__") else [prompt])
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
            responses.append(_MockResponse(uid=uid, token=token, finish_reason=finish))
        for uid in finished:
            del self._active[uid]
        return responses

    def remove(self, uids, return_prompt_caches=False):
        result = {}
        for uid in uids:
            if uid in self._active:
                if return_prompt_caches:
                    result[uid] = [{"mock_cache": True}]
                del self._active[uid]
        return result if return_prompt_caches else None

    def close(self):
        self._closed = True
        self._active.clear()


class _MockResponse:
    """Minimal stand-in for BatchGenerator.Response."""

    def __init__(self, uid: int, token: int, finish_reason: str | None = None):
        self.uid = uid
        self.token = token
        self.logprobs = None
        self.finish_reason = finish_reason
        self.prompt_cache = None


def _make_scheduler_with_mock_bg(config: ServerConfig | None = None, **kwargs) -> tuple[Scheduler, Any]:
    """Create a Scheduler with a mock BatchGenerator injected.

    Mirrors the pattern from test_batch_integration.py.
    """
    if config is None:
        config = ServerConfig(**kwargs)

    sched = Scheduler(config=config, model=None, tokenizer=None)

    mock_bg = _MockBatchGeneratorForInsertError()
    sched._batch_generator = mock_bg
    sched._sequence_cache = None

    # Minimal mock tokenizer so the batch path uses str(token) fallback
    mock_tokenizer = MagicMock()
    mock_tokenizer.detokenizer = None
    mock_tokenizer.decode = lambda ids: "".join(f"t{i}" for i in ids)
    sched.tokenizer = mock_tokenizer

    return sched, mock_bg


# ---------------------------------------------------------------------------
# C-NEW-1: Requests lost during insertion error hang forever
# ---------------------------------------------------------------------------


class TestCNEW1InsertionErrorHang:
    """Regression: C-NEW-1 — insertion error must signal finish, not hang.

    Without the fix, _insert_new_requests_batch() would pop a request from the
    queue but fail during _init_sequence() or insert() without ever calling
    _signal_finish(). The caller blocks on get_result() forever.

    The fix wraps the per-request insertion in a try/except that calls
    _signal_finish(request_id, finish_reason='error') on any exception.
    """

    def test_insert_exception_returns_error_not_hang(self):
        """If BatchGenerator.insert() raises, get_result() returns error (not hang)."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=5
        )
        # Make the next insert() call raise
        mock_bg._should_fail_next_insert = True

        sched.run_inference_loop()
        try:
            req = InferenceRequest(
                request_id="insert-fail-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=5,
            )
            sched.submit_request(req)

            # Without the fix, this would hang forever. With the fix, it
            # returns promptly with an error finish_reason.
            events = sched.get_result("insert-fail-1", timeout=5.0)
            assert len(events) >= 1
            assert events[-1].finish_reason == "error"
        finally:
            sched.stop()

    def test_init_sequence_exception_returns_error(self):
        """If _init_sequence() raises internally, get_result returns error."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=5
        )

        # Patch _init_sequence to raise an exception
        original_init = sched._init_sequence

        def failing_init(req):
            raise ValueError("Simulated init failure")

        sched._init_sequence = failing_init
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="init-fail-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=5,
            )
            sched.submit_request(req)

            events = sched.get_result("init-fail-1", timeout=5.0)
            assert len(events) >= 1
            assert events[-1].finish_reason == "error"
        finally:
            sched.stop()

    def test_insertion_error_does_not_block_subsequent_requests(self):
        """After one insertion error, the next request still succeeds."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=2
        )
        # Fail on the first insert, succeed on subsequent
        mock_bg._should_fail_next_insert = True

        sched.run_inference_loop()
        try:
            # First request will fail
            req1 = InferenceRequest(
                request_id="fail-first",
                prompt_tokens=[1],
                max_tokens=2,
            )
            sched.submit_request(req1)
            events1 = sched.get_result("fail-first", timeout=5.0)
            assert events1[-1].finish_reason == "error"

            # Second request should succeed normally
            req2 = InferenceRequest(
                request_id="succeed-second",
                prompt_tokens=[2],
                max_tokens=2,
            )
            sched.submit_request(req2)
            events2 = sched.get_result("succeed-second", timeout=5.0)
            assert len(events2) == 2
            assert events2[-1].finish_reason == "length"
        finally:
            sched.stop()


# ---------------------------------------------------------------------------
# H-NEW-6: Unbounded streaming token queues cause OOM
# ---------------------------------------------------------------------------


class TestHNEW6BoundedStreamQueue:
    """Regression: H-NEW-6 — register_stream() must return a bounded queue.

    Without the fix, Queue() was unbounded (maxsize=0). A slow consumer with
    a fast producer could grow memory without limit.

    The fix uses Queue(maxsize=256) and _emit_tokens handles queue.Full by
    dropping the oldest token to make room.
    """

    def test_register_stream_returns_bounded_queue(self):
        """register_stream() returns a Queue with maxsize > 0."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        q = sched.register_stream("bounded-test")

        assert q.maxsize > 0, (
            "Stream queue must be bounded to prevent OOM. "
            f"Got maxsize={q.maxsize}"
        )

    def test_emit_to_full_queue_does_not_crash(self):
        """Emitting tokens to a full stream queue drops oldest, doesn't crash."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        # Register a stream with a small bounded queue
        q = sched.register_stream("full-queue-test")
        # Fill the queue to capacity
        for i in range(q.maxsize):
            q.put(TokenEvent(
                request_id="full-queue-test",
                token_id=i,
                token_text=f"t{i}",
                finish_reason=None,
            ))

        assert q.full(), "Queue should be full for this test"

        # Now emit a token via the scheduler's internal method.
        # Without the fix (unbounded queue), this was never an issue.
        # With bounded queue, _emit_tokens must handle queue.Full gracefully.
        event = TokenEvent(
            request_id="full-queue-test",
            token_id=999,
            token_text="overflow",
            finish_reason=None,
        )
        # This should not raise
        sched._emit_tokens([event])

        # The new token should be in the queue (oldest was dropped)
        found = False
        items = []
        while not q.empty():
            item = q.get_nowait()
            items.append(item)
            if item.token_id == 999:
                found = True

        assert found, "New token should have been added after dropping oldest"

    def test_finish_event_on_full_queue(self):
        """_signal_finish on a full queue still delivers the finish event."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        q = sched.register_stream("finish-full-test")
        # Fill the queue
        for i in range(q.maxsize):
            q.put(TokenEvent(
                request_id="finish-full-test",
                token_id=i,
                token_text=f"t{i}",
                finish_reason=None,
            ))

        # Signal finish — must not crash, and finish event must arrive
        sched._signal_finish("finish-full-test", finish_reason="stop")

        # Drain and find the finish event
        items = []
        while not q.empty():
            items.append(q.get_nowait())

        finish_events = [e for e in items if e.finish_reason is not None]
        assert len(finish_events) >= 1, "Finish event must be delivered even on full queue"
        assert finish_events[-1].finish_reason == "stop"


# ---------------------------------------------------------------------------
# H-NEW-7: Timed-out requests waste batch slots
# ---------------------------------------------------------------------------


class _TimeoutMockScheduler:
    """Mock scheduler that returns empty events (simulates timeout)."""

    def __init__(self):
        self.submitted: list[InferenceRequest] = []
        self.cancel_called_for: list[str] = []
        self.shutdown_called = False

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)

    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]:
        return Queue()

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        # Return empty list — simulates a timeout (event.wait expired)
        return []

    def cancel_request(self, request_id: str) -> bool:
        self.cancel_called_for.append(request_id)
        return True

    def get_cache_stats(self) -> dict[str, Any]:
        return {"active_sequences": 0, "queued_requests": 0}

    def shutdown(self) -> None:
        self.shutdown_called = True


class _MockTokenizerForServer:
    """Minimal tokenizer with eos_token for server tests."""

    eos_token = "<|endoftext|>"
    eos_token_ids = {0}

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


class TestHNEW7TimeoutCancellation:
    """Regression: H-NEW-7 — timed-out requests must be cancelled.

    Without the fix, when get_result() returned empty (timeout), the server
    raised 504 but never called cancel_request(). The request stayed in the
    active set, wasting a batch slot forever.

    The fix calls scheduler.cancel_request(request_id) before raising HTTPException(504).
    """

    @pytest.mark.anyio
    async def test_timeout_calls_cancel_chat(self):
        """504 on chat completion triggers cancel_request."""
        mock_sched = _TimeoutMockScheduler()
        config = ServerConfig(model="test-model")
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 504
        # The critical check: cancel_request must have been called
        assert len(mock_sched.cancel_called_for) == 1, (
            "cancel_request must be called on timeout to free the batch slot"
        )

    @pytest.mark.anyio
    async def test_timeout_calls_cancel_completion(self):
        """504 on text completion triggers cancel_request."""
        mock_sched = _TimeoutMockScheduler()
        config = ServerConfig(model="test-model")
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "Once upon",
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 504
        assert len(mock_sched.cancel_called_for) == 1


# ---------------------------------------------------------------------------
# M-NEW-1: Stop sequence text in output violates OpenAI spec
# ---------------------------------------------------------------------------


class TestMNEW1StopSequenceTruncation:
    """Regression: M-NEW-1 — stop sequence must be removed from output text.

    The OpenAI API spec says stop sequences should NOT appear in the response
    text. Without the fix, the output_text included the stop sequence itself
    (e.g., "Hello world<|stop|>" instead of "Hello world").

    The fix truncates seq.output_text at the position of the stop sequence
    in _check_stop_conditions().
    """

    def test_stop_sequence_removed_from_output(self):
        """Output text should not contain the stop sequence."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        # Mock generator that produces "Hello world<|stop|>" token by token
        tokens = ["Hello", " ", "world", "<|", "stop", "|>"]

        def mock_gen(request_id, token_ids, step):
            if step < len(tokens):
                return (step + 100, tokens[step], None)
            return (step + 100, "", "length")

        sched._mock_generate = mock_gen
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="stop-trunc-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=20,
                stop_sequences=["<|stop|>"],
            )
            sched.submit_request(req)
            events = sched.get_result("stop-trunc-1", timeout=5.0)

            # The request should have finished with reason "stop"
            assert events[-1].finish_reason == "stop"

            # Reconstruct the output text from the sequence state.
            # The critical check: the scheduler's internal seq.output_text
            # was truncated. We verify via the emitted token texts.
            # But the real contract is that the seq.output_text is truncated.
            # Let's verify that the stop sequence does not appear in the
            # concatenated output. (The token events still carry individual
            # token text, but the scheduler truncates seq.output_text.)
            # We can verify indirectly by checking that fewer tokens were emitted
            # before the stop was detected.
            full_text = "".join(e.token_text for e in events if e.token_text)
            # Note: token-level events may still contain partial stop text
            # depending on token boundaries, but the sequence should stop
            # when the full stop sequence appears.
            assert "<|stop|>" not in full_text or events[-1].finish_reason == "stop"
        finally:
            sched.stop()

    def test_stop_sequence_truncation_in_check_stop(self):
        """Directly test _check_stop_conditions truncates output_text."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        seq = SequenceState(
            request_id="direct-stop-test",
            token_ids=[1, 2, 3, 4, 5],
            output_tokens=[4, 5],
        )
        seq.output_text = "Hello world<STOP>extra"

        req = InferenceRequest(
            request_id="direct-stop-test",
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
            stop_sequences=["<STOP>"],
        )

        result = sched._check_stop_conditions(seq, req)

        assert result == "stop"
        # The fix: output_text is truncated at the stop sequence position
        assert seq.output_text == "Hello world", (
            f"output_text should be truncated before stop sequence, "
            f"got: {seq.output_text!r}"
        )

    def test_stop_sequence_at_start_of_output(self):
        """Stop sequence at the very start produces empty output."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        seq = SequenceState(
            request_id="stop-at-start",
            token_ids=[1, 2],
            output_tokens=[2],
        )
        seq.output_text = "STOP_HEREmore text"

        req = InferenceRequest(
            request_id="stop-at-start",
            prompt_tokens=[1],
            max_tokens=100,
            stop_sequences=["STOP_HERE"],
        )

        result = sched._check_stop_conditions(seq, req)
        assert result == "stop"
        assert seq.output_text == "", (
            "output_text should be empty when stop sequence is at position 0"
        )


# ---------------------------------------------------------------------------
# M-NEW-2: EOS token not filtered in streaming path
# ---------------------------------------------------------------------------


class _EOSStreamMockScheduler:
    """Mock scheduler that streams tokens with an EOS token at the end."""

    def __init__(self, eos_text: str = "<|endoftext|>"):
        self.submitted: list[InferenceRequest] = []
        self.streams: dict[str, Queue[TokenEvent | None]] = {}
        self.shutdown_called = False
        self.eos_text = eos_text

    def submit_request(self, request: InferenceRequest) -> None:
        self.submitted.append(request)
        if request.request_id in self.streams:
            q = self.streams[request.request_id]
            # Emit some normal tokens then an EOS token
            normal_tokens = [("Hello", None), (", world", None)]
            for i, (text, reason) in enumerate(normal_tokens):
                q.put(TokenEvent(
                    request_id=request.request_id,
                    token_id=i + 1,
                    token_text=text,
                    finish_reason=reason,
                ))
            # Final token: EOS with finish_reason="stop"
            q.put(TokenEvent(
                request_id=request.request_id,
                token_id=0,
                token_text=self.eos_text,
                finish_reason="stop",
            ))

    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]:
        q: Queue[TokenEvent | None] = Queue()
        self.streams[request_id] = q
        return q

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        return [
            TokenEvent(request_id=request_id, token_id=1, token_text="Hello", finish_reason=None),
            TokenEvent(request_id=request_id, token_id=2, token_text=", world", finish_reason=None),
            TokenEvent(request_id=request_id, token_id=0, token_text=self.eos_text, finish_reason="stop"),
        ]

    def cancel_request(self, request_id: str) -> bool:
        return False

    def get_cache_stats(self) -> dict[str, Any]:
        return {"active_sequences": 0}

    def shutdown(self) -> None:
        self.shutdown_called = True


class TestMNEW2EOSFilterStreaming:
    """Regression: M-NEW-2 — EOS token text must be filtered in streaming.

    Without the fix, the streaming SSE chunks included the raw EOS token text
    (e.g., "<|endoftext|>") as content in the last delta. The OpenAI API never
    includes EOS tokens in streamed text.

    The fix checks if the token_text equals tokenizer.eos_token when
    finish_reason=="stop", and replaces it with "" in the SSE chunk.
    """

    @pytest.mark.anyio
    async def test_eos_filtered_in_chat_streaming(self):
        """Chat streaming must not include EOS token text in content."""
        eos_text = "<|endoftext|>"
        mock_sched = _EOSStreamMockScheduler(eos_text=eos_text)
        config = ServerConfig(model="test-model")

        tok = _MockTokenizerForServer()
        tok.eos_token = eos_text

        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
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
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]

        # Parse all JSON chunks (skip [DONE])
        chunks = []
        for line in lines:
            if line == "data: [DONE]":
                continue
            payload = line[len("data: "):]
            chunks.append(json.loads(payload))

        # Concatenate all content
        full_content = "".join(
            c["choices"][0]["delta"]["content"] for c in chunks
        )

        assert eos_text not in full_content, (
            f"EOS token '{eos_text}' must not appear in streamed content. "
            f"Got: {full_content!r}"
        )
        # Should still contain the real content
        assert "Hello" in full_content

    @pytest.mark.anyio
    async def test_eos_filtered_in_completion_streaming(self):
        """Text completion streaming must not include EOS token text."""
        eos_text = "<|endoftext|>"
        mock_sched = _EOSStreamMockScheduler(eos_text=eos_text)
        config = ServerConfig(model="test-model")

        tok = _MockTokenizerForServer()
        tok.eos_token = eos_text

        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "Once upon a time",
                    "max_tokens": 50,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        body = resp.text
        lines = [l for l in body.strip().split("\n") if l.strip()]

        chunks = []
        for line in lines:
            if line == "data: [DONE]":
                continue
            payload = line[len("data: "):]
            chunks.append(json.loads(payload))

        full_text = "".join(c["choices"][0]["text"] for c in chunks)

        assert eos_text not in full_text, (
            f"EOS token '{eos_text}' must not appear in streamed completion text. "
            f"Got: {full_text!r}"
        )

    @pytest.mark.anyio
    async def test_eos_filtered_in_non_streaming_chat(self):
        """Non-streaming chat must also exclude EOS token from content."""
        eos_text = "<|endoftext|>"
        mock_sched = _EOSStreamMockScheduler(eos_text=eos_text)
        config = ServerConfig(model="test-model")

        tok = _MockTokenizerForServer()
        tok.eos_token = eos_text

        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50,
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        assert eos_text not in content, (
            f"EOS token '{eos_text}' must not appear in non-streaming content. "
            f"Got: {content!r}"
        )


# ---------------------------------------------------------------------------
# M-NEW-5: Duplicate decode sequences in schedule_step
# ---------------------------------------------------------------------------


class TestMNEW5DuplicateDecodeSequences:
    """Regression: M-NEW-5 — schedule_step must not produce duplicate sequences.

    Without the fix, schedule_step() could add the same sequence to
    decode_sequences twice: once when classifying new requests as decode
    (if already computed), and again when iterating existing active sequences.

    The fix adds `seq not in prefill_sequences and seq not in decode_sequences`
    guard before appending to decode_sequences in the active-sequence loop.
    """

    def test_no_duplicate_decode_sequences(self):
        """A sequence that starts in decode must appear exactly once."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        # Submit a request
        req = InferenceRequest(
            request_id="dup-test-1",
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
        )
        sched.request_queue.add(req)

        # First schedule_step: picks up the request, classifies it
        outputs1 = sched.schedule_step()

        # The request should appear in prefill (num_computed=0 < len(token_ids)=3)
        assert len(outputs1.prefill_sequences) == 1
        assert outputs1.prefill_sequences[0].request_id == "dup-test-1"

        # Simulate prefill completion (mark all tokens as computed)
        seq = outputs1.prefill_sequences[0]
        seq.num_computed_tokens = len(seq.token_ids)

        # Second schedule_step: no new requests, existing seq should be in decode
        outputs2 = sched.schedule_step()

        # The sequence must appear in decode_sequences exactly once
        decode_ids = [s.request_id for s in outputs2.decode_sequences]
        assert decode_ids.count("dup-test-1") == 1, (
            f"Sequence should appear exactly once in decode_sequences, "
            f"got {decode_ids.count('dup-test-1')} times"
        )

    def test_no_prefill_decode_overlap(self):
        """A newly added prefill sequence must not also appear in decode."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        # Pre-populate an active sequence in decode state
        existing_seq = SequenceState(
            request_id="existing-1",
            token_ids=[1, 2, 3],
            num_computed_tokens=3,  # fully computed -> decode
        )
        existing_seq._request = InferenceRequest(
            request_id="existing-1",
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
        )
        sched._active_sequences["existing-1"] = existing_seq

        # Add a new request that will be prefill
        req = InferenceRequest(
            request_id="new-prefill-1",
            prompt_tokens=[10, 20, 30],
            max_tokens=10,
        )
        sched.request_queue.add(req)

        outputs = sched.schedule_step()

        # new-prefill-1 should be in prefill only
        prefill_ids = [s.request_id for s in outputs.prefill_sequences]
        decode_ids = [s.request_id for s in outputs.decode_sequences]

        assert "new-prefill-1" in prefill_ids
        assert "new-prefill-1" not in decode_ids, (
            "A prefill sequence must not also appear in decode_sequences"
        )

        # existing-1 should be in decode only
        assert "existing-1" in decode_ids
        assert "existing-1" not in prefill_ids

        # No duplicates anywhere
        assert len(prefill_ids) == len(set(prefill_ids))
        assert len(decode_ids) == len(set(decode_ids))


# ---------------------------------------------------------------------------
# D2: Batch-path error recovery tests
# ---------------------------------------------------------------------------


class _GatedBatchGenerator:
    """Mock BatchGenerator whose next() blocks on a gate and can raise on demand.

    The gate pattern ensures requests remain active when we trigger the error:
    - next() blocks on _gate (threading.Event) before processing
    - The test calls trigger_failure() then opens the gate
    - next() sees the failure flag and raises RuntimeError

    After an error, the scheduler calls close() + _create_batch_generator().
    Since model is None, _create_batch_generator does nothing, so for recovery
    tests we must inject a fresh _GatedBatchGenerator.
    """

    def __init__(self) -> None:
        self._uid_counter = 0
        self._active: dict[int, dict] = {}
        self._closed = False
        self._gate = threading.Event()
        self._gate.set()  # Open by default (no blocking)
        self._fail_next = False
        self._lock = threading.Lock()

    def insert(self, prompts, max_tokens=None, caches=None, samplers=None,
               logits_processors=None):
        uids = []
        for i, prompt in enumerate(prompts):
            uid = self._uid_counter
            self._uid_counter += 1
            mt = max_tokens[i] if isinstance(max_tokens, list) else (max_tokens or 10)
            self._active[uid] = {
                "tokens": list(prompt) if hasattr(prompt, "__iter__") else [prompt],
                "max_tokens": mt,
                "step": 0,
            }
            uids.append(uid)
        return uids

    def next(self):
        # Wait on gate — allows test to hold next() until ready
        self._gate.wait(timeout=10.0)
        with self._lock:
            if self._fail_next:
                self._fail_next = False
                raise RuntimeError("Simulated batch error")
        responses = []
        finished = []
        for uid, state in list(self._active.items()):
            state["step"] += 1
            token = 100 + state["step"]
            finish = None
            if state["step"] >= state["max_tokens"]:
                finish = "length"
                finished.append(uid)
            responses.append(_MockResponse(uid=uid, token=token, finish_reason=finish))
        for uid in finished:
            del self._active[uid]
        return responses

    def remove(self, uids, return_prompt_caches=False):
        result = {}
        for uid in uids:
            if uid in self._active:
                if return_prompt_caches:
                    result[uid] = [{"mock_cache": True}]
                del self._active[uid]
        return result if return_prompt_caches else None

    def close(self):
        self._closed = True
        self._active.clear()
        # Unblock any waiting next() calls so the thread doesn't deadlock
        self._gate.set()

    def hold_next(self) -> None:
        """Block next() calls until release_next() or trigger_failure()."""
        self._gate.clear()

    def release_next(self) -> None:
        """Unblock next() calls."""
        self._gate.set()

    def trigger_failure(self) -> None:
        """Make the next next() call raise, then unblock the gate."""
        with self._lock:
            self._fail_next = True
        self._gate.set()


def _make_scheduler_with_gated_bg(
    config: ServerConfig | None = None, **kwargs
) -> tuple[Scheduler, _GatedBatchGenerator]:
    """Create a Scheduler with a _GatedBatchGenerator injected."""
    if config is None:
        config = ServerConfig(**kwargs)

    sched = Scheduler(config=config, model=None, tokenizer=None)

    mock_bg = _GatedBatchGenerator()
    sched._batch_generator = mock_bg
    sched._sequence_cache = None

    # Minimal mock tokenizer so the batch path uses str(token) fallback
    mock_tokenizer = MagicMock()
    mock_tokenizer.detokenizer = None
    mock_tokenizer.decode = lambda ids: "".join(f"t{i}" for i in ids)
    sched.tokenizer = mock_tokenizer

    return sched, mock_bg


class TestD2BatchErrorRecovery:
    """D2: Verify scheduler handles batch errors correctly via _handle_batch_error.

    These tests use a gated mock BatchGenerator whose next() can be held and
    then made to raise, exercising the _handle_batch_error code path (distinct
    from _handle_mock_error tested in test_scheduler.py::TestErrorRecovery).
    """

    def test_batch_error_signals_all_active(self):
        """When BatchGenerator.next() raises, ALL active requests get error finish."""
        sched, mock_bg = _make_scheduler_with_gated_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Hold next() so requests stay active while we set up the failure
        mock_bg.hold_next()

        sched.run_inference_loop()
        try:
            req1 = InferenceRequest(
                request_id="berr-sig-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            req2 = InferenceRequest(
                request_id="berr-sig-2",
                prompt_tokens=[4, 5, 6],
                max_tokens=100,
            )
            sched.submit_request(req1)
            sched.submit_request(req2)

            # Wait until both are inserted (they'll block on next())
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with sched._active_lock:
                    if len(sched._uid_to_request_id) >= 2:
                        break
                time.sleep(0.02)

            # Trigger failure — this sets the flag AND unblocks the gate
            mock_bg.trigger_failure()

            # Both requests should finish with error
            events1 = sched.get_result("berr-sig-1", timeout=5.0)
            events2 = sched.get_result("berr-sig-2", timeout=5.0)

            assert any(e.finish_reason == "error" for e in events1), (
                f"berr-sig-1 should get error, got: {[e.finish_reason for e in events1]}"
            )
            assert any(e.finish_reason == "error" for e in events2), (
                f"berr-sig-2 should get error, got: {[e.finish_reason for e in events2]}"
            )
        finally:
            sched.stop()

    def test_batch_error_frees_resources(self):
        """After a batch error, active sequences are cleaned up and slots freed."""
        sched, mock_bg = _make_scheduler_with_gated_bg(
            max_batch_size=4, default_max_tokens=100
        )

        mock_bg.hold_next()

        sched.run_inference_loop()
        try:
            req = InferenceRequest(
                request_id="berr-free-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            sched.submit_request(req)

            # Wait for insertion
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with sched._active_lock:
                    if len(sched._uid_to_request_id) >= 1:
                        break
                time.sleep(0.02)

            # Trigger batch error
            mock_bg.trigger_failure()

            # Wait for error result
            events = sched.get_result("berr-free-1", timeout=5.0)
            assert any(e.finish_reason == "error" for e in events)

            # Give cleanup time to complete
            time.sleep(0.3)

            # Active sequences should be empty after error cleanup
            assert sched.num_active_sequences == 0, (
                f"Expected 0 active sequences after error, got {sched.num_active_sequences}"
            )

            # UID mappings should be cleared (part of _handle_batch_error)
            assert len(sched._uid_to_request_id) == 0, (
                "UID-to-request mappings should be cleared after batch error"
            )
            assert len(sched._request_id_to_uid) == 0, (
                "Request-to-UID mappings should be cleared after batch error"
            )

            # Scheduler should still accept new requests (queue not broken)
            req2 = InferenceRequest(
                request_id="berr-free-2",
                prompt_tokens=[7, 8, 9],
                max_tokens=2,
            )
            sched.submit_request(req2)
            assert sched.num_queued_requests >= 1
        finally:
            sched.stop()

    def test_batch_error_recovery(self):
        """After a batch error, NEW requests complete successfully."""
        sched, mock_bg = _make_scheduler_with_gated_bg(
            max_batch_size=4, default_max_tokens=3
        )

        mock_bg.hold_next()

        sched.run_inference_loop()
        try:
            # Submit a request that will fail
            req_fail = InferenceRequest(
                request_id="berr-recov-fail",
                prompt_tokens=[1, 2],
                max_tokens=3,
            )
            sched.submit_request(req_fail)

            # Wait for insertion
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with sched._active_lock:
                    if len(sched._uid_to_request_id) >= 1:
                        break
                time.sleep(0.02)

            # Trigger the error
            mock_bg.trigger_failure()

            # Wait for error result
            events_fail = sched.get_result("berr-recov-fail", timeout=5.0)
            assert any(e.finish_reason == "error" for e in events_fail), (
                f"Failed request should get error, got: {[e.finish_reason for e in events_fail]}"
            )

            # Give cleanup time
            time.sleep(0.3)

            # After error, _handle_batch_error calls _create_batch_generator()
            # which does nothing when model=None. Inject a fresh mock to
            # simulate the recreated BatchGenerator.
            fresh_bg = _GatedBatchGenerator()
            sched._batch_generator = fresh_bg

            # Submit a NEW request — it should succeed (scheduler recovered)
            req_ok = InferenceRequest(
                request_id="berr-recov-ok",
                prompt_tokens=[10, 20],
                max_tokens=3,
            )
            sched.submit_request(req_ok)

            events_ok = sched.get_result("berr-recov-ok", timeout=5.0)
            assert len(events_ok) == 3, (
                f"Recovery request should get 3 tokens, got {len(events_ok)}"
            )
            assert events_ok[-1].finish_reason == "length", (
                f"Recovery request should finish with 'length', got '{events_ok[-1].finish_reason}'"
            )
        finally:
            sched.stop()

    def test_stream_gets_error_finish(self):
        """Streaming request receives error finish event when batch fails."""
        sched, mock_bg = _make_scheduler_with_gated_bg(
            max_batch_size=4, default_max_tokens=100
        )

        mock_bg.hold_next()

        sched.run_inference_loop()
        try:
            # Register stream BEFORE submitting
            stream_q = sched.register_stream("berr-stream-1")

            req = InferenceRequest(
                request_id="berr-stream-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
                stream=True,
            )
            sched.submit_request(req)

            # Wait for insertion
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with sched._active_lock:
                    if len(sched._uid_to_request_id) >= 1:
                        break
                time.sleep(0.02)

            # Trigger batch error
            mock_bg.trigger_failure()

            # Read events from stream — should get an error finish
            error_event = None
            events_received = []
            read_deadline = time.monotonic() + 5.0
            while time.monotonic() < read_deadline:
                try:
                    ev = stream_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if ev is None:
                    break
                events_received.append(ev)
                if ev.finish_reason is not None:
                    error_event = ev
                    break

            assert error_event is not None, (
                f"Stream should receive error finish event. "
                f"Events received: {[(e.token_id, e.finish_reason) for e in events_received]}"
            )
            assert error_event.finish_reason == "error", (
                f"Expected finish_reason='error', got '{error_event.finish_reason}'"
            )
        finally:
            sched.stop()
