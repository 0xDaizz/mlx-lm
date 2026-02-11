"""End-to-end tests for distributed Tensor Parallel infrastructure.

Tests the integrated flow of scheduler + control bus using mock objects,
since we can't actually launch multi-rank processes in pytest.

Run: .venv/bin/python -m pytest tests/test_distributed_e2e.py -v -x --timeout=60
"""
from __future__ import annotations

import pickle
import queue
import threading
import time

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.distributed import DistributedContext
from mlx_lm_server.distributed_bus import ControlEvent
from mlx_lm_server.scheduler import Scheduler, BUS_ERROR_THRESHOLD
from mlx_lm_server.types import InferenceRequest, TokenEvent

try:
    import mlx.core as mx
    HAS_MX_DISTRIBUTED = hasattr(mx.distributed, 'init') or hasattr(mx, 'distributed')
except ImportError:
    HAS_MX_DISTRIBUTED = False


# =========================================================================
# Helpers
# =========================================================================


class MockControlBus:
    """Mock control bus for E2E testing.

    Simulates rank0 publish + rank>0 recv using a thread-safe queue.
    """

    def __init__(self):
        self._queue: queue.Queue[ControlEvent] = queue.Queue()
        self.published: list[ControlEvent] = []

    def publish(self, event: ControlEvent) -> None:
        self.published.append(event)
        self._queue.put(event)

    def recv(self) -> ControlEvent:
        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            return ControlEvent.noop()


class FailAfterNBus:
    """Mock bus that fails after N successful operations."""

    def __init__(self, fail_after: int):
        self._count = 0
        self._fail_after = fail_after
        self.published: list[ControlEvent] = []

    def publish(self, event: ControlEvent) -> None:
        self._count += 1
        if self._count > self._fail_after:
            raise RuntimeError("Bus connection lost")
        self.published.append(event)

    def recv(self) -> ControlEvent:
        self._count += 1
        if self._count > self._fail_after:
            raise RuntimeError("Bus connection lost")
        return ControlEvent.noop()


# =========================================================================
# E2E Test Cases
# =========================================================================


@pytest.mark.skipif(not HAS_MX_DISTRIBUTED, reason="requires mx.distributed")
class TestDistributedE2E:
    """End-to-end distributed integration tests using mock buses."""

    def _make_rank0_scheduler(self, bus=None):
        """Create a rank0 scheduler with mock bus."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        if bus is None:
            bus = MockControlBus()
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=bus)
        return scheduler, bus

    def _make_worker_scheduler(self, bus=None, rank=1):
        """Create a worker (rank>0) scheduler with mock bus."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=rank, world_size=2)
        if bus is None:
            bus = MockControlBus()
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=bus)
        return scheduler, bus

    def test_full_lifecycle_submit_generate_shutdown(self):
        """E2E: Submit request on rank0 -> generate tokens -> shutdown."""
        scheduler, bus = self._make_rank0_scheduler()

        # 1. Submit a request
        req = InferenceRequest(
            request_id="e2e-lifecycle",
            prompt_tokens=[1, 2, 3],
            max_tokens=3,
        )
        scheduler.submit_request(req)

        # 2. Verify submit event was queued in outbox
        assert not scheduler._bus_outbox.empty()

        # 3. Drain outbox (simulates inference loop step)
        scheduler._drain_bus_outbox()
        assert len(bus.published) == 1
        assert bus.published[0].typ == "batch"
        inner = bus.published[0].unpack_batch()
        assert any(e.typ == "submit" for e in inner)

        # 4. Run mock inference (generate tokens)
        scheduler._mock_generate = lambda rid, tids, step: (
            step + 100,
            f"tok{step}",
            "stop" if step >= 2 else None,
        )
        scheduler.run_inference_loop(blocking=False)

        # Wait for completion
        result = scheduler.get_result("e2e-lifecycle", timeout=5.0)
        assert len(result) == 3
        assert result[-1].finish_reason == "stop"

        # 5. Shutdown
        scheduler.stop()
        # Verify shutdown event was queued
        shutdown_events = [e for e in bus.published if e.typ == "batch"]
        assert len(shutdown_events) >= 1

    def test_cancel_propagation_through_bus(self):
        """E2E: Submit -> cancel via scheduler -> verify cleanup."""
        scheduler, bus = self._make_rank0_scheduler()

        req = InferenceRequest(
            request_id="e2e-cancel",
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
        )
        scheduler.submit_request(req)

        # Cancel the request
        result = scheduler.cancel_request("e2e-cancel")
        assert result is True

        # Verify cancel event was queued
        events_in_outbox = []
        while not scheduler._bus_outbox.empty():
            events_in_outbox.append(scheduler._bus_outbox.get_nowait())
        cancel_events = [e for e in events_in_outbox if e.typ == "cancel"]
        assert len(cancel_events) == 1
        assert cancel_events[0].unpack_request_id() == "e2e-cancel"

        scheduler.stop()

    def test_worker_receives_and_applies_events(self):
        """E2E: Worker receives compound event and applies it."""
        bus = MockControlBus()

        # Pre-load compound event for worker to receive
        compound = ControlEvent.batch([
            ControlEvent.submit(InferenceRequest(
                request_id="worker-e2e-1",
                prompt_tokens=[10, 20],
                max_tokens=5,
            )),
            ControlEvent.submit(InferenceRequest(
                request_id="worker-e2e-2",
                prompt_tokens=[30, 40],
                max_tokens=5,
            )),
        ])
        bus._queue.put(compound)

        scheduler, _ = self._make_worker_scheduler(bus=bus)
        result = scheduler._apply_bus_events()
        assert result is True
        assert scheduler.request_queue.size == 2

        # Worker should NOT have result buffers (U1 fix)
        assert "worker-e2e-1" not in scheduler._results
        assert "worker-e2e-2" not in scheduler._results
        scheduler.stop()

    def test_bus_error_recovery_graceful_shutdown(self):
        """E2E: Bus failures trigger graceful distributed shutdown."""
        bus = FailAfterNBus(fail_after=0)  # Fail immediately
        scheduler, _ = self._make_worker_scheduler(bus=bus)
        scheduler._running = True

        # Apply events should increment error counter
        for _ in range(BUS_ERROR_THRESHOLD - 1):
            result = scheduler._apply_bus_events()
            assert result is True  # Not yet at threshold

        # Threshold-reaching call triggers shutdown
        result = scheduler._apply_bus_events()
        assert result is False
        assert scheduler._dist_fatal is True
        assert scheduler._dist_fatal_reason == "bus_error_threshold"
        assert scheduler._running is False
        scheduler.stop()

    def test_immediate_cancel_on_queue_add_failure(self):
        """E2E: When rank0 local apply fails (queue full), the request is immediately cancelled.

        In the outbox-only pattern, submit_request() in distributed mode only
        enqueues to the outbox — it does NOT touch request_queue. The actual
        queue add happens in _drain_bus_outbox() during local apply. If that
        fails (e.g., queue full), the request is added to _cancelled set and
        _signal_finish is called so blocked callers are unblocked.
        No compensation cancel is enqueued — workers will see the cancel
        naturally via schedule_step at the next drain cycle.
        """
        scheduler, bus = self._make_rank0_scheduler()

        # Fill the queue to max
        for i in range(scheduler.config.max_queue_size):
            scheduler.request_queue.add(InferenceRequest(
                request_id=f"filler-{i}",
                prompt_tokens=[1],
                max_tokens=1,
            ))

        # Submit goes to outbox only (no queue add, no result buffers)
        req = InferenceRequest(
            request_id="overflow-req",
            prompt_tokens=[1, 2],
            max_tokens=5,
        )
        scheduler.submit_request(req)  # Should not raise — outbox-only

        # Drain outbox: publish succeeds, but local apply fails (queue full)
        scheduler._drain_bus_outbox()

        # DIST-6: Result buffers are preserved so get_result() can read the
        # error finish event. They are cleaned up naturally when get_result()
        # pops them.
        assert "overflow-req" in scheduler._results
        assert "overflow-req" in scheduler._results_ready

        # get_result should return the error event
        result = scheduler.get_result("overflow-req", timeout=1.0)
        assert len(result) >= 1
        assert result[-1].finish_reason == "error"

        # After get_result pops, buffers are cleaned up
        assert "overflow-req" not in scheduler._results
        assert "overflow-req" not in scheduler._results_ready

        # The request should be in the cancelled set for schedule_step cleanup
        assert "overflow-req" in scheduler._cancelled

        # No compensation cancel should be in the outbox (new behavior)
        while not scheduler._bus_outbox.empty():
            ev = scheduler._bus_outbox.get_nowait()
            assert not (ev.typ == "cancel" and ev.unpack_request_id() == "overflow-req"), \
                "Should NOT find compensation cancel — immediate cancel used instead"
        scheduler.stop()

    def test_shutdown_propagation_rank0_to_worker(self):
        """E2E: Rank0 shutdown queues event; worker applies it."""
        rank0_bus = MockControlBus()
        scheduler_rank0, _ = self._make_rank0_scheduler(bus=rank0_bus)

        # Stop rank0 scheduler — should queue shutdown event
        scheduler_rank0.stop()

        # Verify shutdown was queued in outbox and drained
        # stop() queues to outbox but doesn't drain (that's the inference loop's job)
        # So we check the outbox directly
        shutdown_found = False
        while not scheduler_rank0._bus_outbox.empty():
            event = scheduler_rank0._bus_outbox.get_nowait()
            if event.typ == "shutdown":
                shutdown_found = True
        assert shutdown_found

    def test_multiple_requests_batch_broadcast(self):
        """E2E: Multiple requests batched into single compound event."""
        scheduler, bus = self._make_rank0_scheduler()

        # Submit 3 requests rapidly
        for i in range(3):
            req = InferenceRequest(
                request_id=f"batch-e2e-{i}",
                prompt_tokens=[i + 1],
                max_tokens=5,
            )
            scheduler.submit_request(req)

        # Drain outbox — should create ONE compound event with 3 submits
        scheduler._drain_bus_outbox()
        assert len(bus.published) == 1
        assert bus.published[0].typ == "batch"
        inner = bus.published[0].unpack_batch()
        submit_events = [e for e in inner if e.typ == "submit"]
        assert len(submit_events) == 3

        request_ids = [e.unpack_request().request_id for e in submit_events]
        assert "batch-e2e-0" in request_ids
        assert "batch-e2e-1" in request_ids
        assert "batch-e2e-2" in request_ids
        scheduler.stop()

    def test_worker_shutdown_via_bus_event(self):
        """E2E: Worker receives shutdown event and stops."""
        bus = MockControlBus()
        bus._queue.put(ControlEvent.batch([ControlEvent.shutdown()]))

        scheduler, _ = self._make_worker_scheduler(bus=bus)
        scheduler._running = True

        result = scheduler._apply_bus_events()
        assert result is False
        assert scheduler._running is False
        scheduler.stop()

    def test_interleaved_submit_cancel_via_bus(self):
        """E2E: Worker processes interleaved submit and cancel in compound event."""
        bus = MockControlBus()
        compound = ControlEvent.batch([
            ControlEvent.submit(InferenceRequest(
                request_id="interleave-1",
                prompt_tokens=[1],
                max_tokens=5,
            )),
            ControlEvent.submit(InferenceRequest(
                request_id="interleave-2",
                prompt_tokens=[2],
                max_tokens=5,
            )),
            ControlEvent.cancel("interleave-1"),
        ])
        bus._queue.put(compound)

        scheduler, _ = self._make_worker_scheduler(bus=bus)
        result = scheduler._apply_bus_events()
        assert result is True

        # interleave-1 was submitted then cancelled from queue
        # interleave-2 should remain
        assert scheduler.request_queue.size == 1
        scheduler.stop()

    def test_rank0_bus_error_threshold_sends_shutdown(self):
        """E2E: Rank0 publish failures at threshold trigger shutdown broadcast attempt."""
        bus = FailAfterNBus(fail_after=0)  # All publishes fail
        scheduler, _ = self._make_rank0_scheduler(bus=bus)
        scheduler._running = True

        # Keep draining until threshold is reached
        for _ in range(BUS_ERROR_THRESHOLD):
            scheduler._drain_bus_outbox()

        assert scheduler._dist_fatal is True
        assert scheduler._running is False
        scheduler.stop()


# =========================================================================
# Deferred E2E Tests — require real multi-node hardware
# =========================================================================


class TestMultiNodeE2E:
    """Multi-node E2E tests — require 2x Mac Studio nodes."""

    @pytest.mark.skip(reason="Requires 2 Mac Studio nodes with Thunderbolt 5")
    def test_2node_ring_inference(self):
        """E2E: 2-node ring TP inference produces correct output."""
        # TODO: Launch mlx_lm_server on 2 nodes with --distributed-mode ring
        # Verify that inference requests produce correct completions
        pass

    @pytest.mark.skip(reason="Requires 2 Mac Studio nodes with JACCL RDMA")
    def test_2node_jaccl_inference(self):
        """E2E: 2-node JACCL RDMA TP inference produces correct output."""
        # TODO: Launch mlx_lm_server on 2 nodes with --distributed-mode jaccl
        # Verify that RDMA tensor parallel inference works correctly
        pass
