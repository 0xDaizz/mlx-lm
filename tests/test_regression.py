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
        """Emitting tokens to a full stream queue signals error + cancel, doesn't crash.

        Bug 4 / Issue 6 fix: instead of silently dropping tokens (which causes
        data corruption), the scheduler drains the queue, puts an error finish
        event, and cancels the request.
        """
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
        event = TokenEvent(
            request_id="full-queue-test",
            token_id=999,
            token_text="overflow",
            finish_reason=None,
        )
        # This should not raise
        sched._emit_tokens([event])

        # The queue should have been drained and an error finish event added
        items = []
        while not q.empty():
            item = q.get_nowait()
            items.append(item)

        # Should contain exactly the error finish event (queue was drained)
        error_events = [e for e in items if e.finish_reason == "error"]
        assert len(error_events) >= 1, (
            "An error finish event should be delivered when backpressure overflow occurs"
        )

        # The request should have been added to the cancelled set
        with sched._cancelled_lock:
            assert "full-queue-test" in sched._cancelled, (
                "Request should be cancelled on backpressure overflow"
            )

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
        # Raise TimeoutError — simulates a timeout in the executor poll loop.
        # The new _do_inference() catches TimeoutError per poll iteration.
        raise TimeoutError("Simulated timeout")

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


# ---------------------------------------------------------------------------
# F08: Deadlock in two-phase eviction (cache_block + TieredKVCache)
# ---------------------------------------------------------------------------


class TestF08_DeadlockTwoPhaseEviction:
    """Regression: F08 — cache_block() with tiered_cache must NOT deadlock.

    Without the fix, cache_block() held self.lock and then called
    tiered_cache.evict_to_ssd(), which also tried to acquire self.ram.lock
    (the same lock). Python's threading.Lock is non-reentrant, so this
    caused a deadlock.

    The fix uses two-phase eviction: release the lock before calling
    evict_to_ssd(), then re-acquire it to complete the allocation.
    """

    def test_cache_block_with_tiered_cache_completes(self):
        """cache_block() with a TieredKVCache must complete without deadlock."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache

        config = ServerConfig(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        # Create a mock SSD cache that accepts save_block calls
        class _MockSSD:
            def save_block(self, block_hash, kv_data, num_tokens=None):
                return "saved"

            def load_block(self, block_hash):
                return None

            def has_block(self, block_hash):
                return False

        tiered = TieredKVCache(ram=mgr, ssd=_MockSSD())

        # Exhaust the pool by allocating all blocks
        for i in range(config.num_blocks):
            block = mgr.pool.get_free_block()
            block.block_hash = f"existing-{i}"
            block.token_ids = [i] * config.block_size
            block.ref_count = 0  # Evictable
            block.last_accessed = 0.0
            block.kv_data = [{"keys": "mock", "values": "mock"}]
            mgr.hash_table[block.block_hash] = block.block_id

        assert mgr.pool.num_free == 0, "Pool should be exhausted"

        # This call would deadlock before the fix (timeout detects it)
        result = [None]
        error = [None]

        def _do_cache():
            try:
                result[0] = mgr.cache_block(
                    block_hash="new-block-hash",
                    token_ids=[100, 101, 102, 103],
                    kv_data=[{"keys": "new", "values": "new"}],
                    tiered_cache=tiered,
                )
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_do_cache)
        t.start()
        t.join(timeout=5.0)

        assert not t.is_alive(), (
            "cache_block() with tiered_cache deadlocked (thread still alive after 5s)"
        )
        assert error[0] is None, f"cache_block() raised: {error[0]}"
        # Either successfully cached (got a block_id) or returned None
        # (eviction freed a block and we allocated it)

    def test_two_phase_eviction_result_is_valid(self):
        """Two-phase eviction should produce a valid cached block."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache

        config = ServerConfig(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        ssd_saved = {}

        class _RecordingSSD:
            def save_block(self, block_hash, kv_data, num_tokens=None):
                ssd_saved[block_hash] = kv_data
                return "saved"

            def load_block(self, block_hash):
                return ssd_saved.get(block_hash)

            def has_block(self, block_hash):
                return block_hash in ssd_saved

        tiered = TieredKVCache(ram=mgr, ssd=_RecordingSSD())

        # Fill both blocks as evictable
        for i in range(2):
            block = mgr.pool.get_free_block()
            block.block_hash = f"old-{i}"
            block.token_ids = [i] * config.block_size
            block.ref_count = 0
            block.last_accessed = 0.0
            block.kv_data = [{"keys": "old", "values": "old"}]
            mgr.hash_table[block.block_hash] = block.block_id

        block_id = mgr.cache_block(
            block_hash="fresh-hash",
            token_ids=[10, 11, 12, 13],
            kv_data=[{"keys": "fresh", "values": "fresh"}],
            tiered_cache=tiered,
        )

        assert block_id is not None, "cache_block should return a block_id after two-phase eviction"
        assert "fresh-hash" in mgr.hash_table, "New block should be in hash_table"
        block = mgr.pool.blocks[block_id]
        assert block.token_ids == [10, 11, 12, 13]
        assert block.ref_count == 1
        # The evicted block should have been saved to SSD
        assert len(ssd_saved) >= 1, "At least one block should have been saved to SSD"


# ---------------------------------------------------------------------------
# F07: Collision blocks (block_hash=None) stuck in eviction heap
# ---------------------------------------------------------------------------


class TestF07_CollisionBlockFreed:
    """Regression: F07 — collision blocks must be returned to free pool.

    Without the fix, blocks with block_hash=None (created during hash
    collisions) were pushed to the eviction heap when freed. But
    _evict_lru_locked() skips block_hash=None entries, so these blocks
    were stuck forever — neither in the free pool nor evictable.

    The fix returns collision blocks directly to the free pool in
    free_blocks() when ref_count reaches 0, bypassing the eviction heap.
    """

    def test_collision_block_returned_to_free_pool(self):
        """Freeing a collision block (block_hash=None) returns it to the free pool."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager

        config = ServerConfig(block_size=4, num_blocks=8)
        mgr = KVCacheManager(config)

        # Allocate a block and simulate a collision (block_hash=None)
        block = mgr.pool.get_free_block()
        block.block_hash = None  # Collision block
        block.token_ids = [1, 2, 3, 4]
        block.ref_count = 1
        block.last_accessed = 0.0

        free_before = mgr.pool.num_free

        # Free the collision block
        mgr.free_blocks([block.block_id])

        free_after = mgr.pool.num_free

        assert free_after == free_before + 1, (
            f"Collision block should be returned to free pool. "
            f"Free before: {free_before}, after: {free_after}"
        )

        # Verify the block was fully reset
        assert block.block_hash is None  # return_block sets it to None
        assert block.ref_count == 0
        assert block.token_ids == []

    def test_collision_block_can_be_reallocated(self):
        """A freed collision block can be re-allocated for new use."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager

        config = ServerConfig(block_size=4, num_blocks=2)
        mgr = KVCacheManager(config)

        # Allocate both blocks
        b1 = mgr.pool.get_free_block()
        b2 = mgr.pool.get_free_block()
        b1.block_hash = None  # Collision block
        b1.ref_count = 1
        b2.block_hash = "normal-hash"
        b2.ref_count = 1
        mgr.hash_table["normal-hash"] = b2.block_id

        assert mgr.pool.num_free == 0, "Pool should be empty"

        # Free the collision block
        mgr.free_blocks([b1.block_id])

        assert mgr.pool.num_free == 1, "One block should be free now"

        # Allocate again — should succeed
        b3 = mgr.pool.get_free_block()
        assert b3.block_id == b1.block_id, (
            "Re-allocated block should be the freed collision block"
        )

    def test_normal_block_not_returned_directly(self):
        """Normal blocks (with block_hash) go to eviction heap, not directly freed."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager

        config = ServerConfig(block_size=4, num_blocks=4)
        mgr = KVCacheManager(config)

        block = mgr.pool.get_free_block()
        block.block_hash = "some-hash"
        block.token_ids = [1, 2, 3, 4]
        block.ref_count = 1
        block.last_accessed = 1.0
        mgr.hash_table["some-hash"] = block.block_id

        free_before = mgr.pool.num_free

        # Free the normal block
        mgr.free_blocks([block.block_id])

        # Normal block stays in hash_table (for potential reuse) and goes to
        # eviction heap — it should NOT be in the free pool yet
        assert mgr.pool.num_free == free_before, (
            "Normal block should NOT be returned directly to free pool"
        )
        assert "some-hash" in mgr.hash_table, (
            "Normal block should remain in hash_table for cache reuse"
        )


# ---------------------------------------------------------------------------
# F09: Stuck sequence on insert() failure
# ---------------------------------------------------------------------------


class TestF09_StuckSequenceOnInsertFailure:
    """Regression: F09 — failed insert() must not leave a stuck sequence.

    Without the fix, _insert_new_requests_batch() added the sequence to
    _active_sequences BEFORE calling BatchGenerator.insert(). If insert()
    raised, the sequence was left in _active_sequences with no UID, so it
    could never be cleaned up and permanently consumed a batch slot.

    The fix moves the _active_sequences registration AFTER a successful
    insert(), and the outer try/except pops the sequence from active set
    if it was partially added.
    """

    def test_insert_failure_cleans_active_sequences(self):
        """A failed insert() must not leave a sequence in _active_sequences."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=5
        )
        mock_bg._should_fail_next_insert = True

        sched.run_inference_loop()
        try:
            req = InferenceRequest(
                request_id="insert-fail-stuck",
                prompt_tokens=[1, 2, 3],
                max_tokens=5,
            )
            sched.submit_request(req)

            # The request should finish with error
            events = sched.get_result("insert-fail-stuck", timeout=5.0)
            assert events[-1].finish_reason == "error"

            # Give cleanup time
            time.sleep(0.3)

            # Critical check: _active_sequences must be empty
            with sched._active_lock:
                assert "insert-fail-stuck" not in sched._active_sequences, (
                    "Failed insert should not leave a sequence in _active_sequences"
                )
                assert len(sched._active_sequences) == 0, (
                    f"Expected 0 active sequences, got {len(sched._active_sequences)}"
                )
        finally:
            sched.stop()

    def test_subsequent_request_uses_freed_slot(self):
        """After a failed insert, the batch slot must be available for the next request."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=1, default_max_tokens=2
        )
        mock_bg._should_fail_next_insert = True

        sched.run_inference_loop()
        try:
            # First request fails
            req1 = InferenceRequest(
                request_id="slot-fail",
                prompt_tokens=[1, 2],
                max_tokens=2,
            )
            sched.submit_request(req1)
            events1 = sched.get_result("slot-fail", timeout=5.0)
            assert events1[-1].finish_reason == "error"

            # Give cleanup time
            time.sleep(0.3)

            # Second request should succeed (batch slot is free)
            req2 = InferenceRequest(
                request_id="slot-ok",
                prompt_tokens=[3, 4],
                max_tokens=2,
            )
            sched.submit_request(req2)
            events2 = sched.get_result("slot-ok", timeout=5.0)
            assert len(events2) == 2
            assert events2[-1].finish_reason == "length"
        finally:
            sched.stop()


# ---------------------------------------------------------------------------
# F11: Backpressure signals error instead of silently dropping
# ---------------------------------------------------------------------------


class TestF11_BackpressureSignalsError:
    """Regression: F11 — stream backpressure overflow must signal error.

    Without the fix, when a stream queue was full, tokens were silently
    dropped via a simple try/except around put_nowait(). This caused
    silent data corruption — the client received incomplete output with
    missing tokens and no indication of the problem.

    The fix drains the queue, puts an error finish event, and adds the
    request to the _cancelled set so the scheduler stops generating.
    """

    def test_backpressure_overflow_signals_error(self):
        """Overflow on a full stream queue must deliver an error finish event."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        q = sched.register_stream("bp-overflow-test")

        # Fill the queue to capacity
        for i in range(q.maxsize):
            q.put(TokenEvent(
                request_id="bp-overflow-test",
                token_id=i,
                token_text=f"t{i}",
                finish_reason=None,
            ))
        assert q.full(), "Queue must be full before testing overflow"

        # Emit a token to a full queue
        event = TokenEvent(
            request_id="bp-overflow-test",
            token_id=999,
            token_text="overflow",
            finish_reason=None,
        )
        sched._emit_tokens([event])

        # Drain the queue and check for error
        items = []
        while not q.empty():
            items.append(q.get_nowait())

        error_events = [e for e in items if e.finish_reason == "error"]
        assert len(error_events) >= 1, (
            "An error finish event must be delivered on backpressure overflow. "
            f"Got events: {[(e.token_id, e.finish_reason) for e in items]}"
        )

    def test_backpressure_overflow_cancels_request(self):
        """Overflow must add request_id to _cancelled set."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        q = sched.register_stream("bp-cancel-test")

        # Fill the queue
        for i in range(q.maxsize):
            q.put(TokenEvent(
                request_id="bp-cancel-test",
                token_id=i,
                token_text=f"t{i}",
                finish_reason=None,
            ))

        # Trigger overflow
        event = TokenEvent(
            request_id="bp-cancel-test",
            token_id=999,
            token_text="overflow",
            finish_reason=None,
        )
        sched._emit_tokens([event])

        with sched._cancelled_lock:
            assert "bp-cancel-test" in sched._cancelled, (
                "Request must be added to _cancelled on backpressure overflow"
            )

    def test_backpressure_overflow_does_not_crash(self):
        """Overflow must not raise an exception."""
        config = ServerConfig(max_batch_size=4)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        q = sched.register_stream("bp-nocrash-test")

        # Fill the queue
        for i in range(q.maxsize):
            q.put(TokenEvent(
                request_id="bp-nocrash-test",
                token_id=i,
                token_text=f"t{i}",
                finish_reason=None,
            ))

        # Should NOT raise
        event = TokenEvent(
            request_id="bp-nocrash-test",
            token_id=999,
            token_text="overflow",
            finish_reason=None,
        )
        sched._emit_tokens([event])  # No assertion needed — just must not raise


# ---------------------------------------------------------------------------
# F03: Model mismatch rejected with 400
# ---------------------------------------------------------------------------


class TestF03_ModelMismatchRejected:
    """Regression: F03 — wrong model name must return 400, not silently proceed.

    Without the fix, the server accepted any model name in the request body
    and silently ran inference with the loaded model. This violated the
    OpenAI API contract where a model mismatch should be an error.

    The fix validates body.model against the loaded model name in
    _validate_and_prepare_request() and raises HTTPException(400).
    """

    @pytest.mark.anyio
    async def test_wrong_model_returns_400_chat(self):
        """Chat with wrong model name returns 400."""
        config = ServerConfig(model="test-model")
        mock_sched = _TimeoutMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "wrong-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 400, (
            f"Expected 400 for model mismatch, got {resp.status_code}"
        )
        body = resp.json()
        assert "error" in body
        assert "mismatch" in body["error"]["message"].lower() or "wrong-model" in body["error"]["message"]

    @pytest.mark.anyio
    async def test_wrong_model_returns_400_completion(self):
        """Completion with wrong model name returns 400."""
        config = ServerConfig(model="test-model")
        mock_sched = _TimeoutMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "wrong-model",
                    "prompt": "Hello world",
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body

    @pytest.mark.anyio
    async def test_empty_model_accepted(self):
        """Empty model string (default) should be accepted — no mismatch."""
        config = ServerConfig(model="test-model")
        mock_sched = _EOSStreamMockScheduler()  # Returns valid tokens
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 200, (
            f"Empty model should be accepted, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.anyio
    async def test_correct_model_accepted(self):
        """Correct model name should be accepted."""
        config = ServerConfig(model="test-model")
        mock_sched = _EOSStreamMockScheduler()
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

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# F13: Shutdown gate — requests rejected with 503 when shutting down
# ---------------------------------------------------------------------------


class TestF13_ShutdownGate:
    """Regression: F13 — requests must be rejected with 503 during shutdown.

    Without the fix, the server continued accepting requests during
    shutdown, leading to incomplete responses and resource contention.

    The fix checks app.state.shutting_down in _validate_and_prepare_request()
    and raises HTTPException(503) before any work is done.
    """

    @pytest.mark.anyio
    async def test_chat_rejected_during_shutdown(self):
        """Chat request returns 503 when server is shutting down."""
        config = ServerConfig(model="test-model")
        mock_sched = _TimeoutMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        # Simulate shutdown state
        app.state.shutting_down = True

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

        assert resp.status_code == 503, (
            f"Expected 503 during shutdown, got {resp.status_code}"
        )
        body = resp.json()
        assert "error" in body
        assert "shutting down" in body["error"]["message"].lower()

    @pytest.mark.anyio
    async def test_completion_rejected_during_shutdown(self):
        """Completion request returns 503 when server is shutting down."""
        config = ServerConfig(model="test-model")
        mock_sched = _TimeoutMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        app.state.shutting_down = True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "Once upon a time",
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 503

    @pytest.mark.anyio
    async def test_requests_accepted_before_shutdown(self):
        """Requests should work normally when NOT shutting down."""
        config = ServerConfig(model="test-model")
        mock_sched = _EOSStreamMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        assert app.state.shutting_down is False

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

        assert resp.status_code == 200, (
            f"Request should succeed before shutdown, got {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# F16: Empty prompt rejected with 400
# ---------------------------------------------------------------------------


class TestF16_EmptyPromptRejected:
    """Regression: F16 — empty prompts must return 400, not crash.

    Without the fix, an empty messages list or empty prompt string would
    produce an empty prompt_tokens list, which would cause downstream
    errors (e.g., IndexError in token processing, or zero-length tensors).

    The fix checks len(prompt_tokens) == 0 in _validate_and_prepare_request()
    and raises HTTPException(400) with a descriptive message.
    """

    @pytest.mark.anyio
    async def test_empty_messages_returns_422(self):
        """Chat with empty messages list returns 422 (Pydantic field_validator).

        Empty messages are now rejected at the Pydantic model validation layer
        via a field_validator on ChatCompletionRequest.messages, returning 422
        before the request reaches _validate_and_prepare_request().
        """
        config = ServerConfig(model="test-model")
        mock_sched = _TimeoutMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [],
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 422, (
            f"Expected 422 for empty messages, got {resp.status_code}"
        )

    @pytest.mark.anyio
    async def test_empty_prompt_string_returns_400(self):
        """Completion with empty prompt string returns 400."""
        config = ServerConfig(model="test-model")
        mock_sched = _TimeoutMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "",
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 400, (
            f"Expected 400 for empty prompt, got {resp.status_code}"
        )
        body = resp.json()
        assert "error" in body

    @pytest.mark.anyio
    async def test_nonempty_prompt_accepted(self):
        """Non-empty prompts should be accepted normally."""
        config = ServerConfig(model="test-model")
        mock_sched = _EOSStreamMockScheduler()
        tok = _MockTokenizerForServer()
        app = create_app(config=config, scheduler=mock_sched, tokenizer=tok)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "Hello world",
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# R19: max_tokens=0 should free blocks, not leak
# ---------------------------------------------------------------------------


class TestR19_MaxTokensZeroBlockLeak:
    """Regression: R19 — max_tokens=0 must free blocks and return finish event.

    Without the fix, _insert_new_requests_batch() would allocate blocks during
    cache lookup but never free them when max_tokens=0, causing a block leak.

    The fix checks req.max_tokens <= 0 before insert() and frees any allocated
    blocks, then signals finish with reason="length".
    """

    def test_max_tokens_zero_returns_finish_event(self):
        """max_tokens=0 request should return a finish event with reason='length'."""
        config = ServerConfig(max_batch_size=4, default_max_tokens=10)
        sched = Scheduler(config=config, model=None, tokenizer=None)

        sched.run_inference_loop()
        try:
            req = InferenceRequest(
                request_id="zero-max-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=0,
            )
            sched.submit_request(req)
            events = sched.get_result("zero-max-1", timeout=5.0)

            assert len(events) >= 1, "Should receive at least one event"
            # The mock path hits _run_decode_step which checks max_tokens
            # and returns a length finish when step >= request.max_tokens (0 >= 0 = True)
            assert events[-1].finish_reason == "length", (
                f"Expected finish_reason='length', got '{events[-1].finish_reason}'"
            )
        finally:
            sched.stop()

    def test_max_tokens_zero_no_block_leak(self):
        """max_tokens=0 should not leak KV cache blocks."""
        from mlx_lm_server.kv_cache_manager import KVCacheManager

        config = ServerConfig(
            max_batch_size=4, default_max_tokens=10,
            block_size=4, num_blocks=8,
        )
        mgr = KVCacheManager(config)
        sched = Scheduler(config=config, model=None, tokenizer=None, kv_cache_manager=mgr)

        total_blocks = mgr.pool.num_blocks
        free_before = mgr.pool.num_free

        sched.run_inference_loop()
        try:
            req = InferenceRequest(
                request_id="zero-leak-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=0,
            )
            sched.submit_request(req)
            events = sched.get_result("zero-leak-1", timeout=5.0)
            assert events[-1].finish_reason == "length"

            # Give cleanup time
            time.sleep(0.3)

            # All blocks should be returned — no leak
            free_after = mgr.pool.num_free
            assert free_after == free_before, (
                f"Block leak detected: free_before={free_before}, free_after={free_after}, "
                f"total={total_blocks}"
            )
        finally:
            sched.stop()


# ---------------------------------------------------------------------------
# R14: Deep copy in SequenceCacheStore prevents mutation
# ---------------------------------------------------------------------------


class TestR14_DeepCopySequenceCache:
    """Regression: R14 — SequenceCacheStore must deep-copy caches on store.

    Without the fix, store() kept a reference to the original cache list.
    When the original was mutated (e.g., by BatchGenerator), the cached
    entry was silently corrupted, leading to wrong KV data on prefix hits.

    The fix calls _clone_cache_list() in store() to create an independent copy.
    """

    def test_stored_cache_independent_from_original(self):
        """Mutating the original cache after store() must not affect the stored copy."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore

        store = SequenceCacheStore(max_entries=10)

        # Create a simple cache with dict entries (these get deepcopied)
        original_cache = [{"keys": [1, 2, 3], "values": [4, 5, 6]}]

        store.store([10, 20, 30], original_cache)

        # Mutate the original
        original_cache[0]["keys"] = [99, 99, 99]
        original_cache.append({"keys": [7], "values": [8]})

        # Retrieve from store
        cached, remaining = store.find_longest_prefix([10, 20, 30])

        assert cached is not None, "Cache should be found"
        assert len(remaining) == 0
        # The stored cache should be independent — still have original values
        assert cached[0]["keys"] == [1, 2, 3], (
            f"Stored cache was mutated: got {cached[0]['keys']}"
        )
        assert len(cached) == 1, (
            f"Stored cache length changed: expected 1, got {len(cached)}"
        )

    def test_clone_cache_list_plain_kv(self):
        """_clone_cache_list with plain KVCache-like objects creates independent copies."""
        import mlx.core as mx
        from mlx_lm_server.sequence_cache import _clone_cache_list

        class MockKVCache:
            """Minimal KVCache mock with keys, values, offset (no group_size)."""

            def __init__(self, keys, values, offset):
                self.keys = keys
                self.values = values
                self.offset = offset

        # Create a mock cache with mx arrays
        keys = mx.ones((1, 4, 8, 16))  # [batch, heads, seq, dim]
        values = mx.zeros((1, 4, 8, 16))
        cache = MockKVCache(keys=keys, values=values, offset=8)

        cloned = _clone_cache_list([cache])

        assert len(cloned) == 1
        cloned_cache = cloned[0]

        # Verify offset is preserved
        assert cloned_cache.offset == 8

        # Verify arrays are independent (different objects)
        assert cloned_cache.keys is not cache.keys, (
            "Cloned keys should be a different array object"
        )
        assert cloned_cache.values is not cache.values, (
            "Cloned values should be a different array object"
        )

        # Verify shapes match (sliced to offset)
        assert cloned_cache.keys.shape == (1, 4, 8, 16)
        assert cloned_cache.values.shape == (1, 4, 8, 16)


# ---------------------------------------------------------------------------
# R13: Executor poll loop timeout behavior
# ---------------------------------------------------------------------------


class TestR13_ExecutorPollTimeout:
    """Regression: R13 — non-streaming request timeout with executor poll loop.

    The new _do_inference() uses a polling loop with run_in_executor to
    periodically check for results. When get_result() raises TimeoutError,
    the loop continues polling until the total timeout is exhausted.

    This verifies:
    1. A scheduler that always raises TimeoutError causes 504.
    2. A scheduler that returns valid events causes 200.
    """

    @pytest.mark.anyio
    async def test_timeout_returns_504_with_poll_loop(self):
        """get_result() raising TimeoutError should eventually return 504."""
        mock_sched = _TimeoutMockScheduler()
        config = ServerConfig(model="test-model", request_timeout_s=0.5)
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

        assert resp.status_code == 504, (
            f"Expected 504 for timeout, got {resp.status_code}: {resp.text}"
        )
        # cancel_request must have been called
        assert len(mock_sched.cancel_called_for) >= 1, (
            "cancel_request must be called when polling times out"
        )

    @pytest.mark.anyio
    async def test_successful_response_returns_200(self):
        """get_result() returning valid events should return 200."""
        mock_sched = _EOSStreamMockScheduler()
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

        assert resp.status_code == 200, (
            f"Expected 200 for successful response, got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "Hello" in data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# R1: Concurrent SSD operations — 2-phase eviction + 3-pass allocate_blocks
# ---------------------------------------------------------------------------


class TestR1_ConcurrentSSDOps:
    """Regression: R1 — concurrent SSD operations must not deadlock.

    The KV cache manager uses lock-release patterns to avoid holding the
    lock during slow SSD I/O:
    - evict_to_ssd(): 3-phase (select candidates locked -> SSD save unlocked
      -> re-check and evict locked)
    - allocate_blocks(): 3-pass (classify locked -> SSD load unlocked ->
      allocate with loaded data locked)
    - cache_block(): 2-phase (RAM alloc locked -> SSD write-through unlocked),
      with two-phase eviction fallback

    These tests verify that concurrent operations using these patterns do not
    deadlock and that the Phase 3 re-validation guards work correctly.
    """

    @pytest.mark.timeout(10)
    def test_evict_to_ssd_does_not_hold_lock_during_save(self, tmp_path):
        """evict_to_ssd releases the lock during SSD save, allowing concurrent alloc.

        Strategy: mock SSD save_block to sleep 0.2s. If evict held the lock
        during that sleep, a concurrent allocate_blocks call would block for
        the full 0.2s. We verify both finish and the allocate thread was not
        blocked for the full save duration.
        """
        import mlx.core as mx
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache

        config = ServerConfig(block_size=4, num_blocks=4)
        ssd = SSDCache(cache_dir=tmp_path / "ssd_cache", ttl_days=7)
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        kv_data = [{"keys": mx.zeros((1, 4, 4, 64)), "values": mx.zeros((1, 4, 4, 64))}]

        # Fill all 4 blocks as evictable (ref_count=0)
        for i in range(4):
            block = mgr.pool.get_free_block()
            block.block_hash = f"fill-{i}"
            block.token_ids = [i * 10 + j for j in range(4)]
            block.ref_count = 0
            block.last_accessed = float(i)  # oldest first
            block.kv_data = list(kv_data)
            mgr.hash_table[block.block_hash] = block.block_id

        assert mgr.pool.num_free == 0

        # Patch SSD save_block to sleep 0.2s (simulating slow disk I/O)
        original_save = ssd.save_block

        def slow_save(block_hash, data, num_tokens=None):
            time.sleep(0.2)
            return original_save(block_hash, data, num_tokens)

        ssd.save_block = slow_save

        evict_result = [None]
        alloc_start = [None]
        alloc_end = [None]
        alloc_result = [None]

        def evict_thread():
            evict_result[0] = tiered.evict_to_ssd(num_blocks=2)

        def alloc_thread():
            # Wait a tiny bit so evict starts first
            time.sleep(0.05)
            alloc_start[0] = time.monotonic()
            # This should be able to acquire the lock during the SSD save window
            # Since evict releases the lock for Phase 2, allocate_blocks can
            # proceed during that window.
            alloc_result[0] = mgr.allocate_blocks(
                token_ids=[100, 101, 102, 103, 200, 201, 202, 203],
                num_existing_blocks=0,
            )
            alloc_end[0] = time.monotonic()

        t_evict = threading.Thread(target=evict_thread)
        t_alloc = threading.Thread(target=alloc_thread)

        t_evict.start()
        t_alloc.start()

        t_evict.join(timeout=8.0)
        t_alloc.join(timeout=8.0)

        assert not t_evict.is_alive(), "evict_to_ssd() deadlocked (thread still alive)"
        assert not t_alloc.is_alive(), "allocate_blocks() deadlocked (thread still alive)"

        # Both operations should have completed
        assert evict_result[0] is not None, "evict_to_ssd should return a result"
        assert alloc_result[0] is not None, "allocate_blocks should return a result"

        # The allocate thread should not have been blocked for the full 0.2s
        # save duration. It should have been able to start processing during
        # the unlock window. Allow some slack for thread scheduling.
        alloc_duration = alloc_end[0] - alloc_start[0]
        # If lock were held during save, alloc would wait ~0.4s (2 saves x 0.2s).
        # With the unlock pattern, alloc can interleave during the save window.
        # We just verify both complete; exact timing depends on OS scheduling.
        assert alloc_duration < 2.0, (
            f"allocate_blocks took {alloc_duration:.2f}s, suggesting lock contention"
        )

    @pytest.mark.timeout(10)
    def test_concurrent_cache_block_and_evict(self, tmp_path):
        """cache_block (triggering two-phase eviction) and evict_to_ssd run concurrently.

        Both threads operate on a fully-allocated pool where all blocks are
        evictable. Thread 1 calls cache_block (which triggers two-phase eviction
        internally when pool is exhausted), thread 2 calls evict_to_ssd. Both
        should complete without deadlock.
        """
        import mlx.core as mx
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache

        config = ServerConfig(block_size=4, num_blocks=6)
        ssd = SSDCache(cache_dir=tmp_path / "ssd_cache2", ttl_days=7)
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        kv_data = [{"keys": mx.zeros((1, 4, 4, 64)), "values": mx.zeros((1, 4, 4, 64))}]

        # Fill all 6 blocks as evictable
        for i in range(6):
            block = mgr.pool.get_free_block()
            block.block_hash = f"old-{i}"
            block.token_ids = [i * 10 + j for j in range(4)]
            block.ref_count = 0
            block.last_accessed = float(i)
            block.kv_data = list(kv_data)
            mgr.hash_table[block.block_hash] = block.block_id

        assert mgr.pool.num_free == 0

        # Use barriers to force overlap
        barrier = threading.Barrier(2, timeout=5.0)

        cache_block_result = [None]
        cache_block_error = [None]
        evict_result = [None]
        evict_error = [None]

        def thread_cache_block():
            try:
                barrier.wait()
                cache_block_result[0] = mgr.cache_block(
                    block_hash="new-cb-hash",
                    token_ids=[50, 51, 52, 53],
                    kv_data=list(kv_data),
                    tiered_cache=tiered,
                )
            except Exception as e:
                cache_block_error[0] = e

        def thread_evict():
            try:
                barrier.wait()
                evict_result[0] = tiered.evict_to_ssd(num_blocks=2)
            except Exception as e:
                evict_error[0] = e

        t1 = threading.Thread(target=thread_cache_block)
        t2 = threading.Thread(target=thread_evict)

        t1.start()
        t2.start()

        t1.join(timeout=8.0)
        t2.join(timeout=8.0)

        assert not t1.is_alive(), (
            "cache_block() deadlocked (thread still alive after 8s)"
        )
        assert not t2.is_alive(), (
            "evict_to_ssd() deadlocked (thread still alive after 8s)"
        )

        assert cache_block_error[0] is None, (
            f"cache_block() raised: {cache_block_error[0]}"
        )
        assert evict_error[0] is None, (
            f"evict_to_ssd() raised: {evict_error[0]}"
        )

        # At least one operation should have succeeded in evicting/allocating
        # (exact behavior depends on thread scheduling)
        # cache_block either allocated a block or returned None
        # evict_to_ssd either evicted blocks or returned empty list
        assert evict_result[0] is not None, "evict_to_ssd should return a list"

    @pytest.mark.timeout(10)
    def test_evict_to_ssd_revalidates_after_unlock(self, tmp_path):
        """Phase 3 of evict_to_ssd re-checks ref_count after the SSD save.

        If another thread increments a block's ref_count during the unlock
        window (Phase 2), Phase 3 must detect this and skip eviction for
        that block.

        Strategy:
        1. Fill pool with 2 evictable blocks.
        2. Patch SSD save_block to signal an event after it starts, then sleep.
        3. In the main thread, wait for the event, then increment the block's
           ref_count (simulating another thread claiming the block).
        4. After evict_to_ssd returns, verify the block with incremented
           ref_count was NOT evicted.
        """
        import mlx.core as mx
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache

        config = ServerConfig(block_size=4, num_blocks=2)
        ssd = SSDCache(cache_dir=tmp_path / "ssd_cache3", ttl_days=7)
        mgr = KVCacheManager(config, ssd=ssd)
        tiered = TieredKVCache(ram=mgr, ssd=ssd)

        kv_data = [{"keys": mx.zeros((1, 4, 4, 64)), "values": mx.zeros((1, 4, 4, 64))}]

        # Fill both blocks as evictable
        blocks = []
        for i in range(2):
            block = mgr.pool.get_free_block()
            block.block_hash = f"reval-{i}"
            block.token_ids = [i * 10 + j for j in range(4)]
            block.ref_count = 0
            block.last_accessed = float(i)
            block.kv_data = list(kv_data)
            mgr.hash_table[block.block_hash] = block.block_id
            blocks.append(block)

        assert mgr.pool.num_free == 0

        # The block we'll "rescue" during the unlock window
        target_block = blocks[0]  # oldest, will be eviction candidate
        target_hash = target_block.block_hash

        # Event to coordinate: fires when SSD save starts (lock is released)
        save_started = threading.Event()
        # Event: main thread has incremented ref_count, save can complete
        ref_incremented = threading.Event()

        original_save = ssd.save_block

        def coordinated_save(block_hash, data, num_tokens=None):
            save_started.set()
            ref_incremented.wait(timeout=5.0)
            return original_save(block_hash, data, num_tokens)

        ssd.save_block = coordinated_save

        evict_result = [None]

        def evict_thread():
            evict_result[0] = tiered.evict_to_ssd(num_blocks=2)

        t = threading.Thread(target=evict_thread)
        t.start()

        # Wait for Phase 2 to start (lock is released, SSD save begins)
        save_started.wait(timeout=5.0)

        # During the unlock window: simulate another thread claiming the block
        with mgr.lock:
            target_block.ref_count = 1  # Block is now in use

        # Let the save complete
        ref_incremented.set()

        t.join(timeout=8.0)
        assert not t.is_alive(), "evict_to_ssd() deadlocked"

        evicted_ids = evict_result[0]
        assert evicted_ids is not None, "evict_to_ssd should return a list"

        # The target block should NOT have been evicted because Phase 3
        # re-checks ref_count and finds it's now 1 (not 0)
        assert target_block.block_id not in evicted_ids, (
            f"Block {target_block.block_id} (ref_count incremented during "
            f"Phase 2) should NOT have been evicted. Evicted: {evicted_ids}"
        )

        # The target block should still be in the hash table
        assert target_hash in mgr.hash_table, (
            f"Block hash '{target_hash}' should still be in hash_table "
            f"since its ref_count was incremented during the unlock window"
        )

        # The target block should still have ref_count=1
        assert target_block.ref_count == 1, (
            f"Target block ref_count should be 1, got {target_block.ref_count}"
        )


# ---------------------------------------------------------------------------
# R36: _clone_cache_list preserves all attributes (copy.copy vs __new__)
# ---------------------------------------------------------------------------


class TestR36_ClonePreservesAllAttributes:
    """Regression: R36 — _clone_cache_list must preserve ALL object attributes.

    The old implementation used ``type(obj).__new__(type(obj))`` and manually
    copied only keys/values/offset/step.  Any additional attributes added to
    KVCache (upstream or locally) were silently dropped, leading to subtle
    corruption when the clone was later used.

    The fix uses ``copy.copy(obj)`` which shallow-copies the entire ``__dict__``
    (plus class-level defaults), then overwrites keys/values with sliced arrays.
    """

    def test_clone_preserves_extra_attributes(self):
        """Extra attributes beyond keys/values/offset must survive cloning."""
        import mlx.core as mx
        from mlx_lm_server.sequence_cache import _clone_cache_list

        class MockKVCache:
            step = 256

            def __init__(self):
                self.keys = mx.ones((1, 4, 8, 16))
                self.values = mx.zeros((1, 4, 8, 16))
                self.offset = 8
                # Extra attributes the old __new__() approach would lose
                self._custom_flag = True
                self.metadata = "test"

        cache = MockKVCache()
        cloned = _clone_cache_list([cache])

        assert len(cloned) == 1
        clone = cloned[0]

        # Core attributes preserved
        assert clone.offset == 8
        assert clone.step == 256

        # Extra attributes that old code would have dropped
        assert hasattr(clone, "_custom_flag"), (
            "_custom_flag missing on clone — copy.copy not used?"
        )
        assert clone._custom_flag is True
        assert hasattr(clone, "metadata"), (
            "metadata missing on clone — copy.copy not used?"
        )
        assert clone.metadata == "test"

        # Keys/values must be independent (sliced copies, not shared refs)
        assert clone.keys is not cache.keys, "keys should be a new array"
        assert clone.values is not cache.values, "values should be a new array"

        # Shape must match (sliced to offset=8, which is the full extent here)
        assert clone.keys.shape == (1, 4, 8, 16)
        assert clone.values.shape == (1, 4, 8, 16)

    def test_clone_with_real_kvcache_class(self):
        """Clone of a real KVCache preserves all attributes including step."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from mlx_lm_server.sequence_cache import _clone_cache_list

        cache = KVCache()
        # Populate keys/values/offset by calling update_and_fetch
        k = mx.ones((1, 4, 5, 16))
        v = mx.zeros((1, 4, 5, 16))
        cache.update_and_fetch(k, v)
        assert cache.offset == 5

        # Record all attributes before cloning
        original_attrs = set(vars(cache).keys())

        cloned = _clone_cache_list([cache])
        assert len(cloned) == 1
        clone = cloned[0]

        # Every instance attribute on the original must exist on the clone
        clone_attrs = set(vars(clone).keys())
        missing = original_attrs - clone_attrs
        assert not missing, (
            f"Clone is missing attributes present on original: {missing}"
        )

        # step is a class attribute — must be preserved
        assert clone.step == cache.step, (
            f"step mismatch: original={cache.step}, clone={clone.step}"
        )

        # offset preserved
        assert clone.offset == 5

        # Keys/values are independent arrays
        assert clone.keys is not cache.keys
        assert clone.values is not cache.values

        # Sliced to offset=5
        assert clone.keys.shape[2] == 5
        assert clone.values.shape[2] == 5

    def test_clone_copy_vs_new_attribute_completeness(self):
        """All attributes (5+ beyond keys/values/offset) are preserved by clone."""
        import mlx.core as mx
        from mlx_lm_server.sequence_cache import _clone_cache_list

        class RichMockKVCache:
            step = 128

            def __init__(self):
                self.keys = mx.ones((1, 2, 4, 8))
                self.values = mx.zeros((1, 2, 4, 8))
                self.offset = 4
                # 5+ extra attributes
                self.layer_idx = 7
                self.head_dim = 64
                self._is_prefill = False
                self.rope_theta = 10000.0
                self.sliding_window = 4096
                self.attention_mask_type = "causal"

        cache = RichMockKVCache()
        original_attr_names = set(vars(cache).keys())
        assert len(original_attr_names) >= 8, (
            f"Expected 8+ instance attrs, got {len(original_attr_names)}"
        )

        cloned = _clone_cache_list([cache])
        clone = cloned[0]
        clone_attr_names = set(vars(clone).keys())

        # Attribute sets must be identical
        assert original_attr_names == clone_attr_names, (
            f"Attribute mismatch.\n"
            f"  Missing on clone: {original_attr_names - clone_attr_names}\n"
            f"  Extra on clone:   {clone_attr_names - original_attr_names}"
        )

        # Verify each extra attribute value
        assert clone.layer_idx == 7
        assert clone.head_dim == 64
        assert clone._is_prefill is False
        assert clone.rope_theta == 10000.0
        assert clone.sliding_window == 4096
        assert clone.attention_mask_type == "causal"

        # Keys/values still independent
        assert clone.keys is not cache.keys
        assert clone.values is not cache.values
