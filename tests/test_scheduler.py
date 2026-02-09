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

from conftest import make_test_request as _make_request
from conftest import make_test_config as _make_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        assert sorted([seq.request_id for seq in outputs.prefill_sequences]) == ["r1", "r2"]
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
                # Verify actual token text content
                token_texts = [t.token_text for t in tokens]
                assert token_texts == ["t0", "t1", "t2"], f"{rid} got unexpected texts {token_texts}"
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


# ---------------------------------------------------------------------------
# Remaining tokens fix tests
# ---------------------------------------------------------------------------


class TestRemainingTokens:
    """Tests for remaining_tokens fix in batch path."""

    def test_init_sequence_no_prefix_check_batch(self):
        """find_cached_prefix is NOT called in _init_sequence when model is set."""
        from unittest.mock import MagicMock

        config = _make_config()
        mock_cache_mgr = MagicMock()
        mock_cache_mgr.find_cached_prefix = MagicMock(return_value=4)

        s = Scheduler(
            config=config,
            model="fake_model",  # Non-None triggers batch path
            tokenizer=None,
            kv_cache_manager=mock_cache_mgr,
        )

        req = _make_request("r1", prompt_tokens=list(range(16)))
        seq = s._init_sequence(req)

        # With model set, _init_sequence should NOT call find_cached_prefix
        mock_cache_mgr.find_cached_prefix.assert_not_called()
        # num_computed_tokens should be 0 (no prefix check in batch path)
        assert seq.num_computed_tokens == 0

    def test_init_sequence_prefix_check_mock_path(self):
        """find_cached_prefix IS called in _init_sequence when model is None."""
        from unittest.mock import MagicMock

        config = _make_config()
        mock_cache_mgr = MagicMock()
        mock_cache_mgr.find_cached_prefix = MagicMock(return_value=4)

        s = Scheduler(
            config=config,
            model=None,
            tokenizer=None,
            kv_cache_manager=mock_cache_mgr,
        )

        req = _make_request("r1", prompt_tokens=list(range(16)))
        seq = s._init_sequence(req)

        mock_cache_mgr.find_cached_prefix.assert_called_once()
        assert seq.num_computed_tokens == 4


# ---------------------------------------------------------------------------
# D2  Error Recovery tests
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """Tests for error recovery in the mock inference path."""

    def test_batch_error_signals_all_active(self):
        """All active requests get finish_reason='error' when _mock_generate crashes."""
        s = _make_scheduler()

        call_count = 0

        def crash_on_step_1(rid, tids, step):
            nonlocal call_count
            call_count += 1
            if step >= 1:
                raise RuntimeError("test crash")
            return (step + 100, f"t{step}", None)

        s._mock_generate = crash_on_step_1

        s.submit_request(_make_request("r1", max_tokens=10))
        s.submit_request(_make_request("r2", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            tokens_r1 = s.get_result("r1", timeout=5.0)
            tokens_r2 = s.get_result("r2", timeout=5.0)

            # Both requests should have received a finish event with error
            assert any(t.finish_reason == "error" for t in tokens_r1), (
                f"r1 should get error finish, got: {[t.finish_reason for t in tokens_r1]}"
            )
            assert any(t.finish_reason == "error" for t in tokens_r2), (
                f"r2 should get error finish, got: {[t.finish_reason for t in tokens_r2]}"
            )
        finally:
            s.stop()

    def test_batch_error_frees_resources(self):
        """After an error, active sequences are cleaned up and resources freed."""
        from unittest.mock import MagicMock

        mock_cache_mgr = MagicMock()
        mock_cache_mgr.find_cached_prefix = MagicMock(return_value=0)

        s = _make_scheduler(kv_cache_manager=mock_cache_mgr)

        def crash_immediately(rid, tids, step):
            raise RuntimeError("crash on step 0")

        s._mock_generate = crash_immediately

        s.submit_request(_make_request("r1", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            tokens = s.get_result("r1", timeout=5.0)
            assert any(t.finish_reason == "error" for t in tokens)

            # Give a moment for cleanup to complete
            time.sleep(0.2)

            # Active sequences should be cleaned up
            assert s.num_active_sequences == 0, (
                f"Expected 0 active sequences, got {s.num_active_sequences}"
            )
            # Queue should be empty
            assert s.request_queue.size == 0
        finally:
            s.stop()

    def test_batch_error_recovery(self):
        """After an error, new requests can still be processed successfully."""
        s = _make_scheduler()

        # First: crash on step 0
        def crash_gen(rid, tids, step):
            raise RuntimeError("crash on step 0")

        s._mock_generate = crash_gen

        s.submit_request(_make_request("req-fail", max_tokens=10))

        s.run_inference_loop(blocking=False)
        try:
            tokens_fail = s.get_result("req-fail", timeout=5.0)
            assert any(t.finish_reason == "error" for t in tokens_fail), (
                f"req-fail should get error, got: {[t.finish_reason for t in tokens_fail]}"
            )

            # Now swap to a working generator: 3 tokens then stop
            def working_gen(rid, tids, step):
                if step >= 2:
                    return (step + 200, f"ok{step}", "stop")
                return (step + 200, f"ok{step}", None)

            s._mock_generate = working_gen

            s.submit_request(_make_request("req-ok", max_tokens=10))

            tokens_ok = s.get_result("req-ok", timeout=5.0)
            assert len(tokens_ok) == 3, (
                f"req-ok should get 3 tokens, got {len(tokens_ok)}"
            )
            assert tokens_ok[-1].finish_reason == "stop", (
                f"req-ok last token should be 'stop', got '{tokens_ok[-1].finish_reason}'"
            )
        finally:
            s.stop()

    def test_stream_gets_error_finish(self):
        """Streaming requests receive a finish event with error on crash."""
        s = _make_scheduler()

        def crash_immediately(rid, tids, step):
            raise RuntimeError("crash on step 0")

        s._mock_generate = crash_immediately

        stream_q = s.register_stream("err-stream")
        s.submit_request(_make_request("err-stream", max_tokens=10, stream=True))

        s.run_inference_loop(blocking=False)
        try:
            # Read events from the stream until we get a finish event
            error_event = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    ev = stream_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if ev is None:
                    # Sentinel — stream ended
                    break
                if ev.finish_reason is not None:
                    error_event = ev
                    break

            assert error_event is not None, "Should have received an error finish event"
            assert error_event.finish_reason == "error", (
                f"Expected finish_reason='error', got '{error_event.finish_reason}'"
            )
            assert error_event.token_text == "", (
                f"Expected empty token_text on error, got '{error_event.token_text}'"
            )
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# E1  Cache Stats tests
# ---------------------------------------------------------------------------


class TestCacheStats:
    """Tests for cache effectiveness counters (E1)."""

    def test_stats_initialized(self):
        """Scheduler starts with zeroed stats."""
        s = _make_scheduler()
        stats = s.get_cache_stats()
        assert stats["cache_misses"] == 0
        assert stats["cache_hits_block"] == 0
        assert stats["cache_hits_sequence"] == 0
        assert stats["requests_completed"] == 0
        assert stats["requests_errored"] == 0
        assert stats["tokens_generated"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_stats_after_completion(self):
        """Stats update after a request completes."""
        s = _make_scheduler()
        s._mock_generate = lambda rid, tids, step: (
            step + 100,
            f"t{step}",
            "stop" if step >= 2 else None,
        )
        req = _make_request("stats-req", max_tokens=10)
        s.submit_request(req)
        s.run_inference_loop(blocking=False)
        try:
            s.get_result("stats-req", timeout=5.0)
            stats = s.get_cache_stats()
            assert stats["tokens_generated"] >= 3
            assert stats["requests_completed"] >= 1
        finally:
            s.stop()

    def test_stats_after_error(self):
        """requests_errored increments after mock error."""
        s = _make_scheduler()
        s._mock_generate = lambda rid, tids, step: (_ for _ in ()).throw(
            RuntimeError("crash")
        )
        req = _make_request("err-req", max_tokens=5)
        s.submit_request(req)
        s.run_inference_loop(blocking=False)
        try:
            result = s.get_result("err-req", timeout=5.0)
            assert any(e.finish_reason == "error" for e in result)
            stats = s.get_cache_stats()
            assert stats["requests_errored"] >= 1
        finally:
            s.stop()

    def test_cache_hit_rate_zero_when_no_lookups(self):
        """cache_hit_rate is 0.0 when no cache lookups have occurred."""
        s = _make_scheduler()
        stats = s.get_cache_stats()
        assert stats["cache_hit_rate"] == 0.0

    def test_tokens_generated_multiple_requests(self):
        """tokens_generated accumulates across multiple requests."""
        s = _make_scheduler()
        # Each request generates exactly 2 tokens then stops
        s._mock_generate = lambda rid, tids, step: (
            step + 1,
            f"t{step}",
            "stop" if step >= 1 else None,
        )

        s.submit_request(_make_request("r1", max_tokens=10))
        s.submit_request(_make_request("r2", max_tokens=10))
        s.run_inference_loop(blocking=False)
        try:
            s.get_result("r1", timeout=5.0)
            s.get_result("r2", timeout=5.0)
            stats = s.get_cache_stats()
            # 2 tokens per request * 2 requests = 4 tokens minimum
            assert stats["tokens_generated"] >= 4
            assert stats["requests_completed"] >= 2
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# SSD flush on shutdown
# ---------------------------------------------------------------------------

class TestSSDFlushOnShutdown:
    """Verify that SSD cache index is flushed when the scheduler stops."""

    def test_stop_flushes_ssd_cache(self):
        """stop() calls tiered_cache.ssd.flush() to persist dirty metadata."""
        from unittest.mock import MagicMock

        mock_ssd = MagicMock()
        mock_tiered = MagicMock()
        mock_tiered.ssd = mock_ssd

        s = _make_scheduler(tiered_cache=mock_tiered)
        s.stop()

        mock_ssd.flush.assert_called_once()

    def test_stop_handles_flush_exception(self):
        """stop() does not raise if ssd.flush() throws."""
        from unittest.mock import MagicMock

        mock_ssd = MagicMock()
        mock_ssd.flush.side_effect = OSError("disk full")
        mock_tiered = MagicMock()
        mock_tiered.ssd = mock_ssd

        s = _make_scheduler(tiered_cache=mock_tiered)
        # Should not raise
        s.stop()
        mock_ssd.flush.assert_called_once()

    def test_stop_skips_flush_when_no_tiered_cache(self):
        """stop() works fine when tiered_cache is None."""
        s = _make_scheduler()
        # Should not raise
        s.stop()

    def test_stop_skips_flush_when_ssd_is_none(self):
        """stop() skips flush when tiered_cache.ssd is None."""
        from unittest.mock import MagicMock

        mock_tiered = MagicMock()
        mock_tiered.ssd = None

        s = _make_scheduler(tiered_cache=mock_tiered)
        # Should not raise
        s.stop()


# ---------------------------------------------------------------------------
# Cancel / get_result race contract tests
# ---------------------------------------------------------------------------


class TestCancelGetResultContract:
    """Test that cancel/get_result race behavior matches documented API contract.

    The documented API contract for get_result() states:
        "After cancel: may raise KeyError (already cleaned up) or return [cancelled_event]"

    The race window: when cancel_request() (for queued requests) or
    _process_cancellations_batch() (for active requests) calls _signal_finish()
    then _cleanup_result_buffers(), a concurrent get_result() may wake up between
    these two calls and see different results depending on lock timing:
        - [cancelled_event] if get_result acquires _results_lock after _signal_finish
          but before _cleanup_result_buffers
        - [] if _cleanup_result_buffers runs before get_result pops the results
        - KeyError if _cleanup_result_buffers already removed _results_ready
        - TimeoutError if the cancel has not yet been processed by the inference loop
    """

    def test_cancel_then_get_result_returns_acceptable_values(self):
        """After cancel, get_result may return cancelled event, empty list,
        raise KeyError, or raise TimeoutError. All are acceptable per the
        API contract.
        """
        config = _make_config(max_batch_size=1)
        s = _make_scheduler(config)

        # Use a slow generator so the request stays active long enough to cancel
        gate = threading.Event()

        def slow_gen(rid, tids, step):
            gate.wait(timeout=10.0)
            return (step + 1, f"t{step}", None)

        s._mock_generate = slow_gen

        req = _make_request("race-req", max_tokens=100)
        s.submit_request(req)
        s.run_inference_loop(blocking=False)

        # Wait for the request to become active (in the batch)
        deadline = time.monotonic() + 3.0
        while s.num_active_sequences == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        # Cancel the active request
        s.cancel_request("race-req")

        # Release the gate so the inference loop can process the cancellation
        gate.set()

        # Try get_result -- any of these outcomes is acceptable per the contract
        try:
            tokens = s.get_result("race-req", timeout=2.0)
            # Acceptable: list with cancelled event, or empty list (race window)
            if len(tokens) > 0:
                assert any(
                    t.finish_reason == "cancelled" for t in tokens
                ), f"Expected cancelled finish_reason, got {[t.finish_reason for t in tokens]}"
            # len(tokens) == 0 is also acceptable (race: cleanup before pop)
        except KeyError:
            # Acceptable: _cleanup_result_buffers already removed the entry
            pass
        except TimeoutError:
            # Acceptable: cancel may not have been processed yet within timeout
            pass
        finally:
            s.stop()

    def test_cancel_queued_then_get_result(self):
        """Cancel a queued (not yet active) request, then call get_result.

        For queued requests, cancel_request() calls _signal_finish() then
        _cleanup_result_buffers() synchronously, so by the time cancel_request()
        returns, the result buffers are already gone. get_result() should
        raise KeyError.
        """
        config = _make_config(max_batch_size=1)
        s = _make_scheduler(config)

        # Block the first request so the second stays queued
        gate = threading.Event()

        def slow_gen(rid, tids, step):
            if step == 0:
                gate.wait(timeout=10.0)
            if step >= 1:
                return (step + 1, f"t{step}", "stop")
            return (step + 1, f"t{step}", None)

        s._mock_generate = slow_gen

        s.submit_request(_make_request("blocker", max_tokens=10))
        s.submit_request(_make_request("queued-cancel", max_tokens=10))
        s.run_inference_loop(blocking=False)

        # Wait for blocker to become active
        deadline = time.monotonic() + 3.0
        while s.num_active_sequences == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        # queued-cancel should still be in the queue
        assert s.cancel_request("queued-cancel") is True

        # After cancel_request returns for a queued request, buffers are cleaned up
        # synchronously, so get_result should raise KeyError
        try:
            tokens = s.get_result("queued-cancel", timeout=1.0)
            # If we get here, the cancelled event was retrieved before cleanup --
            # still acceptable
            if len(tokens) > 0:
                assert any(
                    t.finish_reason == "cancelled" for t in tokens
                ), f"Expected cancelled finish_reason, got {[t.finish_reason for t in tokens]}"
        except KeyError:
            # Expected: buffers already cleaned up
            pass
        except TimeoutError:
            # Also acceptable
            pass
        finally:
            gate.set()
            s.stop()


# ---------------------------------------------------------------------------
# F4  Cancel frees KV cache blocks
# ---------------------------------------------------------------------------


class TestCancelFreesKVBlocks:
    """Test that cancelling a request frees its KV cache blocks (F4)."""

    def test_cancel_frees_kv_blocks(self):
        """When a request is cancelled mid-generation, its KV blocks must be freed."""
        from unittest.mock import MagicMock

        mock_cache_mgr = MagicMock()
        mock_cache_mgr.find_cached_prefix = MagicMock(return_value=0)

        config = _make_config(max_batch_size=2)
        s = Scheduler(
            config=config,
            model=None,
            tokenizer=None,
            kv_cache_manager=mock_cache_mgr,
        )

        # Use a gate to keep the request active long enough to cancel
        gate = threading.Event()

        def slow_gen(rid, tids, step):
            if step == 0:
                gate.wait(timeout=5.0)
            return (step + 1, f"t{step}", None if step < 5 else "stop")

        s._mock_generate = slow_gen

        req = _make_request("cancel-blocks", max_tokens=100)
        s.submit_request(req)
        s.run_inference_loop(blocking=False)

        # Wait for the request to become active
        deadline = time.monotonic() + 3.0
        while s.num_active_sequences == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert s.num_active_sequences > 0

        # Inject block_ids into the active sequence (simulating allocated blocks)
        with s._active_lock:
            seq = s._active_sequences.get("cancel-blocks")
        if seq is not None:
            seq.block_ids = [10, 20, 30]

        # Cancel the request
        s.cancel_request("cancel-blocks")

        # Release the gate so the inference loop processes the cancellation
        gate.set()

        # Wait for cleanup
        deadline = time.monotonic() + 3.0
        while s.num_active_sequences > 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        # Verify free_blocks was called with the block_ids
        mock_cache_mgr.free_blocks.assert_called_with([10, 20, 30])

        s.stop()
