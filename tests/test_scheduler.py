"""Tests for Continuous Batching Scheduler (Phase 2).

All tests use mocks -- no real model loading required.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.scheduler import RequestQueue, Scheduler
from mlx_lm_server.types import InferenceRequest, SequenceState, TokenEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    request_id: str = "req-1",
    prompt_tokens: list[int] | None = None,
    max_tokens: int = 10,
    stream: bool = False,
    stop_sequences: list[str] | None = None,
    temperature: float = 1.0,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens or [1, 2, 3, 4],
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences or [],
        stream=stream,
    )


def _make_config(**overrides) -> ServerConfig:
    defaults = dict(
        block_size=4,
        num_blocks=64,
        max_batch_size=4,
        max_queue_size=32,
        prefill_batch_size=2,
    )
    defaults.update(overrides)
    return ServerConfig(**defaults)


def _make_scheduler(config: ServerConfig | None = None, **kwargs) -> Scheduler:
    config = config or _make_config()
    return Scheduler(config=config, model=None, tokenizer=None, **kwargs)


# ---------------------------------------------------------------------------
# P2.1  RequestQueue tests
# ---------------------------------------------------------------------------

class TestRequestQueue:
    def test_queue_fifo(self):
        """Requests come out in FIFO order."""
        q = RequestQueue(max_size=10)
        r1 = _make_request("a")
        r2 = _make_request("b")
        r3 = _make_request("c")
        q.add(r1)
        q.add(r2)
        q.add(r3)

        batch = q.pop_batch(2)
        assert [r.request_id for r in batch] == ["a", "b"]

        batch = q.pop_batch(5)
        assert [r.request_id for r in batch] == ["c"]

    def test_queue_empty_pop(self):
        """Popping from an empty queue returns an empty list."""
        q = RequestQueue(max_size=10)
        assert q.pop_batch(5) == []

    def test_queue_size(self):
        q = RequestQueue(max_size=10)
        assert q.size == 0
        q.add(_make_request("a"))
        assert q.size == 1
        q.pop_batch(1)
        assert q.size == 0

    def test_queue_full(self):
        q = RequestQueue(max_size=2)
        q.add(_make_request("a"))
        q.add(_make_request("b"))
        with pytest.raises(RuntimeError, match="queue is full"):
            q.add(_make_request("c"))

    def test_queue_cancel(self):
        q = RequestQueue(max_size=10)
        q.add(_make_request("a"))
        q.add(_make_request("b"))
        q.add(_make_request("c"))

        assert q.cancel("b") is True
        assert q.size == 2

        batch = q.pop_batch(10)
        assert [r.request_id for r in batch] == ["a", "c"]

    def test_queue_cancel_missing(self):
        q = RequestQueue(max_size=10)
        q.add(_make_request("a"))
        assert q.cancel("nonexistent") is False
        assert q.size == 1

    def test_queue_concurrent(self):
        """Multiple threads can add/pop concurrently without errors."""
        q = RequestQueue(max_size=1000)
        barrier = threading.Barrier(4)
        results: list[list[str]] = [[] for _ in range(2)]

        def adder(start: int):
            barrier.wait()
            for i in range(100):
                q.add(_make_request(f"req-{start + i}"))

        def popper(idx: int):
            barrier.wait()
            for _ in range(50):
                batch = q.pop_batch(2)
                results[idx].extend(r.request_id for r in batch)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=adder, args=(0,)),
            threading.Thread(target=adder, args=(100,)),
            threading.Thread(target=popper, args=(0,)),
            threading.Thread(target=popper, args=(1,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Drain whatever is left
        remaining = q.pop_batch(1000)
        all_ids = results[0] + results[1] + [r.request_id for r in remaining]

        # All 200 requests should be accounted for exactly once
        assert len(set(all_ids)) == 200


# ---------------------------------------------------------------------------
# P2.2-P2.3  Scheduler Core tests
# ---------------------------------------------------------------------------

class TestSchedulerInit:
    def test_scheduler_init(self):
        """Scheduler initializes with correct defaults."""
        config = _make_config()
        s = Scheduler(config=config, model=None, tokenizer=None)
        assert s.config is config
        assert s.model is None
        assert s.tokenizer is None
        assert s.kv_cache_manager is None
        assert s.num_active_sequences == 0
        assert s.num_queued_requests == 0
        assert s.is_running is False


class TestScheduleStep:
    def test_schedule_fills_slots(self):
        """schedule_step pops requests and creates active sequences."""
        s = _make_scheduler()
        s.submit_request(_make_request("r1"))
        s.submit_request(_make_request("r2"))

        outputs = s.schedule_step()
        # Both should be in prefill (they have uncached prompt tokens)
        assert len(outputs.prefill_sequences) == 2
        assert s.num_active_sequences == 2
        assert s.num_queued_requests == 0

    def test_schedule_respects_max_batch(self):
        """schedule_step won't exceed max_batch_size."""
        config = _make_config(max_batch_size=2)
        s = _make_scheduler(config)
        for i in range(5):
            s.submit_request(_make_request(f"r{i}"))

        outputs = s.schedule_step()
        assert s.num_active_sequences == 2
        assert s.num_queued_requests == 3

    def test_schedule_removes_finished(self):
        """Finished sequences are removed in the next schedule_step."""
        s = _make_scheduler()
        s.submit_request(_make_request("r1"))
        s.schedule_step()  # r1 becomes active

        # Mark r1 as finished
        with s._active_lock:
            s._active_sequences["r1"].is_finished = True
            s._active_sequences["r1"].finish_reason = "stop"

        s.submit_request(_make_request("r2"))
        outputs = s.schedule_step()  # r1 removed, r2 added

        assert s.num_active_sequences == 1
        with s._active_lock:
            assert "r2" in s._active_sequences
            assert "r1" not in s._active_sequences


# ---------------------------------------------------------------------------
# P2.4-P2.5  Sequence Management tests
# ---------------------------------------------------------------------------

class TestInitSequence:
    def test_init_seq_tokenizes(self):
        """_init_sequence creates a SequenceState from an InferenceRequest."""
        s = _make_scheduler()
        req = _make_request("r1", prompt_tokens=[10, 20, 30])
        seq = s._init_sequence(req)

        assert seq.request_id == "r1"
        assert seq.token_ids == [10, 20, 30]
        assert seq.num_computed_tokens == 0
        assert seq.is_finished is False

    def test_init_seq_cache_check(self):
        """_init_sequence checks prefix cache when kv_cache_manager is present."""
        # Create a mock KV cache manager
        class MockCacheManager:
            def find_cached_prefix(self, token_ids):
                # Pretend first 8 tokens are cached
                return min(8, len(token_ids))

        config = _make_config()
        s = Scheduler(
            config=config,
            model=None,
            tokenizer=None,
            kv_cache_manager=MockCacheManager(),
        )
        req = _make_request("r1", prompt_tokens=list(range(16)))
        seq = s._init_sequence(req)

        assert seq.num_computed_tokens == 8


class TestPrefill:
    def test_prefill_computes(self):
        """_run_prefill marks all prompt tokens as computed."""
        s = _make_scheduler()
        req = _make_request("r1", prompt_tokens=[1, 2, 3, 4, 5])
        seq = s._init_sequence(req)
        assert seq.num_computed_tokens == 0

        s._run_prefill([seq])
        assert seq.num_computed_tokens == 5


# ---------------------------------------------------------------------------
# P2.6  Decode Step tests
# ---------------------------------------------------------------------------

class TestDecodeStep:
    def test_decode_produces_tokens(self):
        """_run_decode_step produces one token per sequence."""
        s = _make_scheduler()

        # Use mock generator that returns incrementing tokens
        s._mock_generate = lambda rid, tids, step: (
            100 + step,
            f"t{step}",
            None,
        )

        req = _make_request("r1", max_tokens=5)
        seq = s._init_sequence(req)
        s._run_prefill([seq])

        events = s._run_decode_step([seq])
        assert len(events) == 1
        assert events[0].token_id == 100
        assert events[0].token_text == "t0"
        assert events[0].finish_reason is None
        assert seq.output_tokens == [100]

    def test_decode_multiple_sequences(self):
        """_run_decode_step handles multiple sequences in a batch."""
        s = _make_scheduler()
        s._mock_generate = lambda rid, tids, step: (step + 1, f"t{step}", None)

        seqs = []
        for i in range(3):
            req = _make_request(f"r{i}", max_tokens=10)
            seq = s._init_sequence(req)
            s._run_prefill([seq])
            seqs.append(seq)

        events = s._run_decode_step(seqs)
        assert len(events) == 3
        for e in events:
            assert e.token_id == 1  # step=0 for all


# ---------------------------------------------------------------------------
# P2.7-P2.8  Inference Loop & Streaming tests
# ---------------------------------------------------------------------------

class TestInferenceLoop:
    def test_loop_processes(self):
        """Inference loop processes a request end-to-end."""
        config = _make_config(max_batch_size=2)
        s = _make_scheduler(config)

        # Mock: generate 3 tokens then stop
        def mock_gen(rid, tids, step):
            if step >= 2:
                return (step + 100, f"t{step}", "stop")
            return (step + 100, f"t{step}", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", max_tokens=10)
        s.submit_request(req)

        # Run loop in background
        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert len(tokens) == 3
            assert tokens[-1].finish_reason == "stop"
            assert tokens[0].token_id == 100
            assert tokens[1].token_id == 101
            assert tokens[2].token_id == 102
        finally:
            s.stop()

    def test_stream_receives(self):
        """register_stream delivers TokenEvents to a queue."""
        config = _make_config(max_batch_size=2)
        s = _make_scheduler(config)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 50, f"t{step}", "stop")
            return (step + 50, f"t{step}", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", max_tokens=10, stream=True)
        stream_q = s.register_stream("r1")
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            events: list[TokenEvent] = []
            while True:
                ev = stream_q.get(timeout=5.0)
                events.append(ev)
                if ev.finish_reason is not None:
                    break

            assert len(events) == 2
            assert events[0].token_id == 50
            assert events[1].finish_reason == "stop"
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# P2.9-P2.11  Sequence Lifecycle tests
# ---------------------------------------------------------------------------

class TestStopSequence:
    def test_stop_seq(self):
        """Generation stops when output contains a stop sequence."""
        s = _make_scheduler()

        # Mock: produce "hello" then "world" then "STOP_HERE more"
        words = ["hello", " world", " STOP_HERE more"]

        def mock_gen(rid, tids, step):
            if step < len(words):
                return (step + 1, words[step], None)
            return (step + 1, "", "length")

        s._mock_generate = mock_gen

        req = _make_request("r1", max_tokens=10, stop_sequences=["STOP_HERE"])
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            # Should stop after "STOP_HERE" appears in output
            assert any(t.finish_reason == "stop" for t in tokens)
        finally:
            s.stop()

    def test_eos(self):
        """Generation stops on EOS token when tokenizer is provided."""

        class MockTokenizer:
            eos_token_ids = {999}

        config = _make_config()
        s = Scheduler(config=config, model=None, tokenizer=MockTokenizer())

        def mock_gen(rid, tids, step):
            if step == 2:
                return (999, "", None)  # EOS token
            return (step + 1, f"t{step}", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", max_tokens=10)
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert tokens[-1].finish_reason == "stop"
            assert len(tokens) == 3  # 2 normal + 1 eos
        finally:
            s.stop()


class TestMaxTokens:
    def test_max_tokens(self):
        """Generation stops after max_tokens are produced."""
        s = _make_scheduler()
        s._mock_generate = lambda rid, tids, step: (step + 1, f"t{step}", None)

        req = _make_request("r1", max_tokens=3)
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert len(tokens) == 3
            assert tokens[-1].finish_reason == "length"
        finally:
            s.stop()


class TestCancel:
    def test_cancel_queued(self):
        """Cancelling a queued request removes it before processing."""
        config = _make_config(max_batch_size=1)
        s = _make_scheduler(config)

        # Submit 2 requests, only 1 slot available
        s.submit_request(_make_request("r1", max_tokens=100))
        s.submit_request(_make_request("r2", max_tokens=100))

        # Cancel r2 while it's still in the queue
        assert s.cancel_request("r2") is True

        # r2 should be gone from the queue
        assert s.num_queued_requests == 0 or s.request_queue.size <= 1

    def test_cancel_active(self):
        """Cancelling an active request marks it for cleanup."""
        s = _make_scheduler()

        # Use an event to keep the generator blocked so the request stays active
        gate = threading.Event()

        def slow_gen(rid, tids, step):
            # Block on first token to ensure sequence is active when we cancel
            if step == 0:
                gate.wait(timeout=5.0)
            return (step + 1, f"t{step}", None)

        s._mock_generate = slow_gen

        req = _make_request("r1", max_tokens=1000)
        s.submit_request(req)
        s.run_inference_loop(blocking=False)

        # Wait for the request to become active
        deadline = time.monotonic() + 2.0
        while s.num_active_sequences == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert s.num_active_sequences > 0

        # Cancel while active
        assert s.cancel_request("r1") is True

        # Release the gate and wait for cleanup
        gate.set()
        time.sleep(0.5)
        s.stop()


# ---------------------------------------------------------------------------
# P2.12-P2.14  End-to-end Lifecycle tests
# ---------------------------------------------------------------------------

class TestSingleLifecycle:
    def test_single_lifecycle(self):
        """Full lifecycle: submit -> schedule -> prefill -> decode -> finish."""
        s = _make_scheduler()

        token_count = 0

        def mock_gen(rid, tids, step):
            if step >= 4:
                return (step + 10, f"t{step}", "stop")
            return (step + 10, f"t{step}", None)

        s._mock_generate = mock_gen

        req = _make_request("r1", max_tokens=20)
        s.submit_request(req)

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert len(tokens) == 5
            assert tokens[0].token_id == 10
            assert tokens[-1].finish_reason == "stop"
            # All intermediate tokens have no finish_reason
            for t in tokens[:-1]:
                assert t.finish_reason is None
        finally:
            s.stop()


class TestContinuousBatching:
    def test_continuous_batching(self):
        """Multiple requests are batched and processed concurrently.

        Submit 3 requests. They should all complete without blocking each other.
        """
        config = _make_config(max_batch_size=4)
        s = _make_scheduler(config)

        # Each request generates exactly 3 tokens
        def mock_gen(rid, tids, step):
            if step >= 2:
                return (step + 1, f"t{step}", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = mock_gen

        for i in range(3):
            s.submit_request(_make_request(f"r{i}", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            results = {}
            for i in range(3):
                results[f"r{i}"] = s.get_result(f"r{i}", timeout=5.0)

            # All 3 requests should have completed with 3 tokens each
            for rid, tokens in results.items():
                assert len(tokens) == 3, f"{rid} got {len(tokens)} tokens"
                assert tokens[-1].finish_reason == "stop"
        finally:
            s.stop()

    def test_continuous_batching_staggered(self):
        """Requests submitted at different times still get processed."""
        config = _make_config(max_batch_size=4)
        s = _make_scheduler(config)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 1, f"t{step}", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = mock_gen

        s.submit_request(_make_request("r0", max_tokens=10))
        s.run_inference_loop(blocking=False)

        try:
            # Submit second request after a brief delay
            time.sleep(0.1)
            s.submit_request(_make_request("r1", max_tokens=10))

            t0 = s.get_result("r0", timeout=5.0)
            t1 = s.get_result("r1", timeout=5.0)

            assert len(t0) == 2
            assert len(t1) == 2
            assert t0[-1].finish_reason == "stop"
            assert t1[-1].finish_reason == "stop"
        finally:
            s.stop()
