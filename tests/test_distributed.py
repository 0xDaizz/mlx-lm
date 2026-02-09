"""Tests for the distributed Tensor Parallel infrastructure.

Covers:
  A. Config & CLI — distributed_* fields and parse_args() validation
  B. DistributedContext — dataclass defaults, properties, init_distributed()
  C. ControlEvent — serialization/deserialization roundtrips
  D. Scheduler bus integration — _apply_bus_events(), join_worker_loop(), etc.
  E. SSD rank namespace — directory separation for multi-rank SSD caches
"""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.distributed import (
    DistributedContext,
    finalize_distributed,
    init_distributed,
)
from mlx_lm_server.distributed_bus import ControlEvent
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.server import parse_args
from mlx_lm_server.types import InferenceRequest


# =========================================================================
# A. Config & CLI Tests
# =========================================================================


class TestDistributedConfig:
    """Test distributed configuration fields and defaults."""

    def test_default_config_distributed_off(self):
        """ServerConfig defaults to distributed_mode='off'."""
        config = ServerConfig()
        assert config.distributed_mode == "off"
        assert config.distributed_sharding == "tensor"
        assert config.distributed_strict is True
        assert config.distributed_hostfile is None
        assert config.distributed_ibv_devices is None
        assert config.distributed_jaccl_coordinator is None

    def test_config_with_distributed_fields(self):
        """ServerConfig accepts distributed fields."""
        config = ServerConfig(
            distributed_mode="ring",
            distributed_hostfile="/path/to/hosts.json",
        )
        assert config.distributed_mode == "ring"
        assert config.distributed_hostfile == "/path/to/hosts.json"

    def test_config_jaccl_fields(self):
        """ServerConfig accepts jaccl-specific fields."""
        config = ServerConfig(
            distributed_mode="jaccl",
            distributed_ibv_devices="/path/to/ibv.json",
            distributed_jaccl_coordinator="10.0.0.1:55000",
        )
        assert config.distributed_mode == "jaccl"
        assert config.distributed_ibv_devices == "/path/to/ibv.json"
        assert config.distributed_jaccl_coordinator == "10.0.0.1:55000"

    def test_config_distributed_strict_false(self):
        """ServerConfig accepts distributed_strict=False."""
        config = ServerConfig(distributed_strict=False)
        assert config.distributed_strict is False

    def test_config_use_distributed_deprecated(self):
        """use_distributed field exists but is deprecated."""
        config = ServerConfig(use_distributed=True)
        assert config.use_distributed is True
        # The new field should still default to "off"
        assert config.distributed_mode == "off"


class TestDistributedCLI:
    """Test CLI argument parsing for distributed mode."""

    def test_parse_args_distributed_off_default(self):
        config = parse_args(["--model", "test-model"])
        assert config.distributed_mode == "off"

    def test_parse_args_ring_mode(self, tmp_path):
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        config = parse_args([
            "--model", "test-model",
            "--distributed-mode", "ring",
            "--distributed-hostfile", str(hostfile),
        ])
        assert config.distributed_mode == "ring"
        assert config.distributed_hostfile == str(hostfile)

    def test_parse_args_jaccl_mode(self, tmp_path):
        ibv_file = tmp_path / "ibv.json"
        ibv_file.write_text("{}")
        config = parse_args([
            "--model", "test-model",
            "--distributed-mode", "jaccl",
            "--distributed-ibv-devices", str(ibv_file),
            "--distributed-jaccl-coordinator", "10.0.0.1:55000",
        ])
        assert config.distributed_mode == "jaccl"
        assert config.distributed_ibv_devices == str(ibv_file)
        assert config.distributed_jaccl_coordinator == "10.0.0.1:55000"

    def test_parse_args_ring_missing_hostfile_errors(self):
        """ring mode without --distributed-hostfile should error."""
        with pytest.raises(SystemExit):
            parse_args(["--model", "test-model", "--distributed-mode", "ring"])

    def test_parse_args_jaccl_missing_ibv_errors(self):
        """jaccl mode without --distributed-ibv-devices should error."""
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "test-model",
                "--distributed-mode", "jaccl",
                "--distributed-jaccl-coordinator", "10.0.0.1:55000",
            ])

    def test_parse_args_jaccl_missing_coordinator_errors(self):
        """jaccl mode without --distributed-jaccl-coordinator should error."""
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "test-model",
                "--distributed-mode", "jaccl",
                "--distributed-ibv-devices", "/tmp/ibv.json",
            ])

    def test_parse_args_jaccl_missing_both_errors(self):
        """jaccl mode without both required flags should error."""
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "test-model",
                "--distributed-mode", "jaccl",
            ])

    def test_parse_args_distributed_with_adapter_errors(self, tmp_path):
        """Distributed mode with --adapter-path should error."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "test-model",
                "--distributed-mode", "ring",
                "--distributed-hostfile", str(hostfile),
                "--adapter-path", "/tmp/adapter",
            ])

    def test_parse_args_pipeline_sharding_errors(self, tmp_path):
        """Pipeline sharding should error (not supported in v1)."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "test-model",
                "--distributed-mode", "ring",
                "--distributed-hostfile", str(hostfile),
                "--distributed-sharding", "pipeline",
            ])

    def test_parse_args_no_distributed_strict(self, tmp_path):
        """--no-distributed-strict should set strict=False."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        config = parse_args([
            "--model", "test-model",
            "--distributed-mode", "ring",
            "--distributed-hostfile", str(hostfile),
            "--no-distributed-strict",
        ])
        assert config.distributed_strict is False

    def test_parse_args_distributed_strict_default_true(self, tmp_path):
        """Default distributed_strict should be True."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        config = parse_args([
            "--model", "test-model",
            "--distributed-mode", "ring",
            "--distributed-hostfile", str(hostfile),
        ])
        assert config.distributed_strict is True

    def test_parse_args_off_mode_ignores_hostfile(self, tmp_path):
        """off mode with distributed flags should warn but not error."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        # This should not raise -- parse_args just warns
        config = parse_args([
            "--model", "test-model",
            "--distributed-mode", "off",
            "--distributed-hostfile", str(hostfile),
        ])
        assert config.distributed_mode == "off"
        # The hostfile is set on the config even though mode is off
        assert config.distributed_hostfile == str(hostfile)

    def test_parse_args_sharding_default_tensor(self, tmp_path):
        """Default sharding should be tensor."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        config = parse_args([
            "--model", "test-model",
            "--distributed-mode", "ring",
            "--distributed-hostfile", str(hostfile),
        ])
        assert config.distributed_sharding == "tensor"


# =========================================================================
# B. DistributedContext Tests
# =========================================================================


class TestDistributedContext:
    """Test DistributedContext dataclass and init_distributed()."""

    def test_default_context_is_disabled(self):
        ctx = DistributedContext()
        assert not ctx.enabled
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.is_rank0 is True
        assert ctx.group is None
        assert ctx.backend == "off"

    def test_is_rank0_property_true(self):
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        assert ctx.is_rank0 is True

    def test_is_rank0_property_false(self):
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        assert ctx.is_rank0 is False

    def test_context_with_all_fields(self):
        mock_group = object()
        ctx = DistributedContext(
            enabled=True,
            group=mock_group,
            rank=3,
            world_size=8,
            pipeline_group=None,
            tensor_group=mock_group,
            backend="ring",
        )
        assert ctx.enabled is True
        assert ctx.group is mock_group
        assert ctx.rank == 3
        assert ctx.world_size == 8
        assert ctx.pipeline_group is None
        assert ctx.tensor_group is mock_group
        assert ctx.backend == "ring"
        assert ctx.is_rank0 is False

    def test_init_distributed_off_mode(self):
        config = ServerConfig()
        ctx = init_distributed(config)
        assert not ctx.enabled
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.backend == "off"
        assert ctx.group is None

    def test_init_distributed_unknown_mode_raises(self):
        config = ServerConfig()
        config.distributed_mode = "unknown_backend"
        with pytest.raises(ValueError, match="Unknown distributed_mode"):
            init_distributed(config)

    def test_finalize_distributed_noop_disabled(self):
        """finalize should not raise for disabled context."""
        ctx = DistributedContext()
        finalize_distributed(ctx)  # Should not raise

    def test_finalize_distributed_noop_enabled(self):
        """finalize should not raise for enabled context (just logs)."""
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        finalize_distributed(ctx)  # Should not raise


# =========================================================================
# C. ControlEvent Tests
# =========================================================================


class TestControlEvent:
    """Test ControlEvent serialization and deserialization."""

    def test_submit_event_roundtrip(self):
        """Submit event should serialize/deserialize InferenceRequest."""
        req = InferenceRequest(
            request_id="test-123",
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
        )
        event = ControlEvent.submit(req)
        assert event.typ == "submit"
        assert event.payload is not None
        unpacked = event.unpack_request()
        assert isinstance(unpacked, InferenceRequest)
        assert unpacked.request_id == "test-123"
        assert unpacked.prompt_tokens == [1, 2, 3]
        assert unpacked.max_tokens == 100

    def test_cancel_event_roundtrip(self):
        event = ControlEvent.cancel("req-456")
        assert event.typ == "cancel"
        assert event.payload is not None
        assert event.unpack_request_id() == "req-456"

    def test_shutdown_event(self):
        event = ControlEvent.shutdown()
        assert event.typ == "shutdown"
        assert event.payload is None

    def test_noop_event(self):
        event = ControlEvent.noop()
        assert event.typ == "noop"
        assert event.payload is None

    def test_unpack_none_payload_request(self):
        event = ControlEvent(typ="noop", payload=None)
        assert event.unpack_request() is None

    def test_unpack_none_payload_request_id(self):
        event = ControlEvent(typ="noop", payload=None)
        assert event.unpack_request_id() is None

    def test_submit_preserves_request_fields(self):
        """All InferenceRequest fields should survive serialization."""
        req = InferenceRequest(
            request_id="full-req",
            prompt_tokens=[10, 20, 30, 40, 50],
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["STOP", "END"],
            stream=True,
        )
        event = ControlEvent.submit(req)
        unpacked = event.unpack_request()
        assert unpacked.request_id == "full-req"
        assert unpacked.prompt_tokens == [10, 20, 30, 40, 50]
        assert unpacked.max_tokens == 200
        assert unpacked.temperature == 0.7
        assert unpacked.top_p == 0.9
        assert unpacked.stop_sequences == ["STOP", "END"]
        assert unpacked.stream is True

    def test_cancel_event_unpack_request_returns_string(self):
        """unpack_request on a cancel event should return the pickled string."""
        event = ControlEvent.cancel("my-id")
        # unpack_request just unpickles the payload -- for cancel it returns a string
        result = event.unpack_request()
        assert result == "my-id"

    def test_event_payload_is_pickle_bytes(self):
        """Verify the payload is valid pickle bytes."""
        req = InferenceRequest(
            request_id="pickle-test",
            prompt_tokens=[1],
            max_tokens=1,
        )
        event = ControlEvent.submit(req)
        # payload should be bytes
        assert isinstance(event.payload, bytes)
        # Should be unpicklable
        obj = pickle.loads(event.payload)
        assert obj.request_id == "pickle-test"

    def test_event_types_are_literal(self):
        """Test that factory methods produce correct type strings."""
        assert ControlEvent.submit(InferenceRequest(
            request_id="x", prompt_tokens=[1], max_tokens=1
        )).typ == "submit"
        assert ControlEvent.cancel("x").typ == "cancel"
        assert ControlEvent.shutdown().typ == "shutdown"
        assert ControlEvent.noop().typ == "noop"


# =========================================================================
# D. Scheduler Bus Integration Tests (mock bus)
# =========================================================================


class TestSchedulerDistributedParams:
    """Test scheduler constructor accepts distributed parameters."""

    def test_scheduler_accepts_dist_params_none(self):
        """Scheduler should accept dist_ctx=None and control_bus=None."""
        config = ServerConfig()
        scheduler = Scheduler(config=config, dist_ctx=None, control_bus=None)
        assert scheduler._dist_ctx is None
        assert scheduler._control_bus is None
        scheduler.stop()

    def test_scheduler_default_no_dist(self):
        """Scheduler without dist params stores None."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        assert scheduler._dist_ctx is None
        assert scheduler._control_bus is None
        scheduler.stop()

    def test_scheduler_with_dist_context(self):
        """Scheduler with DistributedContext stores it correctly."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx)
        assert scheduler._dist_ctx is ctx
        assert scheduler._dist_ctx.is_rank0
        scheduler.stop()

    def test_scheduler_with_dist_context_rank1(self):
        """Scheduler with rank=1 context stores it correctly."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx)
        assert scheduler._dist_ctx is ctx
        assert not scheduler._dist_ctx.is_rank0
        scheduler.stop()


class TestApplyBusEvents:
    """Test scheduler._apply_bus_events() with mock control buses."""

    def _make_scheduler(self, mock_bus, rank=1):
        """Create a scheduler with a mock bus and non-rank0 dist context."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=rank, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=mock_bus)
        return scheduler

    def test_apply_bus_events_noop(self):
        """_apply_bus_events should return True for noop."""

        class MockBus:
            def recv(self):
                return ControlEvent.noop()

        scheduler = self._make_scheduler(MockBus())
        result = scheduler._apply_bus_events()
        assert result is True
        scheduler.stop()

    def test_apply_bus_events_shutdown(self):
        """_apply_bus_events should return False on shutdown and set _running=False."""

        class MockBus:
            def recv(self):
                return ControlEvent.shutdown()

        scheduler = self._make_scheduler(MockBus())
        scheduler._running = True
        result = scheduler._apply_bus_events()
        assert result is False
        assert scheduler._running is False
        scheduler.stop()

    def test_apply_bus_events_cancel(self):
        """_apply_bus_events should add request_id to _cancelled set."""

        class MockBus:
            def recv(self):
                return ControlEvent.cancel("req-to-cancel")

        scheduler = self._make_scheduler(MockBus())
        scheduler._apply_bus_events()
        assert "req-to-cancel" in scheduler._cancelled
        scheduler.stop()

    def test_apply_bus_events_submit(self):
        """_apply_bus_events should add request to queue and results on submit event."""

        class MockBus:
            def recv(self):
                req = InferenceRequest(
                    request_id="remote-req",
                    prompt_tokens=[10, 20],
                    max_tokens=50,
                )
                return ControlEvent.submit(req)

        scheduler = self._make_scheduler(MockBus())
        scheduler._apply_bus_events()
        # Verify request was added to the request queue
        assert scheduler.request_queue.size == 1
        # Verify result buffers were NOT set up (U1: rank>0 workers don't create them)
        assert "remote-req" not in scheduler._results
        assert "remote-req" not in scheduler._results_ready
        # Verify stats incremented
        assert scheduler._stats["total_requests"] == 1
        scheduler.stop()

    def test_apply_bus_events_submit_with_none_payload(self):
        """_apply_bus_events should handle submit event with None payload gracefully."""

        class MockBus:
            def recv(self):
                return ControlEvent(typ="submit", payload=None)

        scheduler = self._make_scheduler(MockBus())
        # Should not raise -- unpack_request returns None, skip
        result = scheduler._apply_bus_events()
        assert result is True
        assert scheduler.request_queue.size == 0
        scheduler.stop()

    def test_apply_bus_events_cancel_with_none_payload(self):
        """_apply_bus_events should handle cancel event with None request_id."""

        class MockBus:
            def recv(self):
                return ControlEvent(typ="cancel", payload=None)

        scheduler = self._make_scheduler(MockBus())
        result = scheduler._apply_bus_events()
        assert result is True
        # No request_id added to cancelled since payload is None
        assert len(scheduler._cancelled) == 0
        scheduler.stop()

    def test_apply_bus_events_no_bus_returns_true(self):
        """_apply_bus_events should return True if no control_bus."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        result = scheduler._apply_bus_events()
        assert result is True
        scheduler.stop()


class TestSchedulerBusBroadcast:
    """Test that rank0 scheduler queues events in _bus_outbox for broadcast.

    With the outbox architecture, HTTP handlers (submit_request, cancel_request,
    stop) queue events into _bus_outbox. The inference loop thread then calls
    _drain_bus_outbox() which publishes one event per step via the control bus.
    This ensures all collective all_sum operations happen in the same thread.
    """

    def test_submit_queues_to_outbox(self):
        """submit_request on rank0 should queue a submit event in _bus_outbox."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)

        class MockBus:
            def publish(self, event):
                pass

        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        req = InferenceRequest(
            request_id="broadcast-test",
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
        )
        scheduler.submit_request(req)

        # Event should be in the outbox, not yet published
        assert not scheduler._bus_outbox.empty()
        event = scheduler._bus_outbox.get_nowait()
        assert event.typ == "submit"
        unpacked = event.unpack_request()
        assert unpacked.request_id == "broadcast-test"
        scheduler.stop()

    def test_cancel_queues_to_outbox(self):
        """cancel_request on rank0 should queue a cancel event in _bus_outbox."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)

        class MockBus:
            def publish(self, event):
                pass

        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Submit a request first so cancel has something to find in queue
        req = InferenceRequest(
            request_id="cancel-me",
            prompt_tokens=[1, 2],
            max_tokens=5,
        )
        scheduler.submit_request(req)

        # Drain the submit event
        scheduler._bus_outbox.get_nowait()

        # Cancel should queue a cancel event
        scheduler.cancel_request("cancel-me")

        assert not scheduler._bus_outbox.empty()
        event = scheduler._bus_outbox.get_nowait()
        assert event.typ == "cancel"
        assert event.unpack_request_id() == "cancel-me"
        scheduler.stop()

    def test_stop_queues_shutdown_to_outbox(self):
        """stop() on rank0 should queue a shutdown event in _bus_outbox."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)

        class MockBus:
            def publish(self, event):
                pass

        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        scheduler.stop()

        # Shutdown event should be in the outbox
        assert not scheduler._bus_outbox.empty()
        event = scheduler._bus_outbox.get_nowait()
        assert event.typ == "shutdown"

    def test_drain_outbox_publishes_event(self):
        """_drain_bus_outbox should publish a compound batch event containing the queued event."""
        published = []

        class MockBus:
            def publish(self, event):
                published.append(event)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        req = InferenceRequest(
            request_id="drain-test",
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
        )
        scheduler.submit_request(req)

        # Nothing published yet
        assert len(published) == 0

        # Drain should publish one compound batch event
        scheduler._drain_bus_outbox()
        assert len(published) == 1
        assert published[0].typ == "batch"
        # Unpack the compound event to verify it contains the submit event
        inner_events = pickle.loads(published[0].payload)
        assert len(inner_events) == 1
        assert inner_events[0].typ == "submit"
        assert inner_events[0].unpack_request().request_id == "drain-test"
        scheduler.stop()

    def test_drain_outbox_sends_noop_when_empty(self):
        """_drain_bus_outbox should send a compound batch event with noop when outbox is empty."""
        published = []

        class MockBus:
            def publish(self, event):
                published.append(event)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Outbox is empty — should send compound batch with noop inside
        scheduler._drain_bus_outbox()
        assert len(published) == 1
        assert published[0].typ == "batch"
        inner_events = pickle.loads(published[0].payload)
        assert len(inner_events) == 1
        assert inner_events[0].typ == "noop"
        scheduler.stop()

    def test_no_bus_means_no_broadcast(self):
        """Without control_bus, submit_request should not try to broadcast."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)

        req = InferenceRequest(
            request_id="local-only",
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
        )
        scheduler.submit_request(req)  # Should not raise
        scheduler.stop()

    def test_rank1_does_not_broadcast_on_submit(self):
        """Non-rank0 scheduler should not broadcast on submit_request."""
        published = []

        class MockBus:
            def publish(self, event):
                published.append(event)

            def recv(self):
                return ControlEvent.noop()

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        req = InferenceRequest(
            request_id="rank1-req",
            prompt_tokens=[1, 2],
            max_tokens=5,
        )
        scheduler.submit_request(req)

        # rank1 should not publish
        assert len(published) == 0
        scheduler.stop()


class TestJoinWorkerLoop:
    """Test join_worker_loop()."""

    def test_join_worker_loop_no_thread(self):
        """join_worker_loop() without inference thread should return immediately."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        # Should not block or raise
        scheduler.join_worker_loop()
        scheduler.stop()

    def test_join_worker_loop_with_thread(self):
        """join_worker_loop() should wait for inference thread to finish."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler.run_inference_loop(blocking=False)
        assert scheduler._inference_thread is not None
        # Stop the loop and join
        scheduler._running = False
        scheduler._new_request_event.set()
        scheduler.join_worker_loop()
        scheduler.stop()


# =========================================================================
# E. SSD Rank Namespace Tests
# =========================================================================


class TestSSDRankNamespace:
    """Test SSD cache directory rank separation."""

    def test_ssd_dir_unchanged_when_not_distributed(self):
        """Non-distributed mode should not modify ssd_cache_dir."""
        base = Path("/tmp/test-ssd-cache")
        ctx = DistributedContext()  # off
        if ctx.enabled and ctx.world_size > 1:
            ssd_dir = base / f"rank_{ctx.rank}"
        else:
            ssd_dir = base
        assert ssd_dir == base

    def test_ssd_dir_has_rank_suffix_when_distributed(self):
        """Distributed mode should add rank_N suffix to ssd_cache_dir."""
        base = Path("/tmp/test-ssd-cache")
        for rank in range(4):
            ctx = DistributedContext(enabled=True, rank=rank, world_size=4)
            if ctx.enabled and ctx.world_size > 1:
                ssd_dir = base / f"rank_{ctx.rank}"
            else:
                ssd_dir = base
            assert ssd_dir == base / f"rank_{rank}"

    def test_ssd_dir_single_rank_no_suffix(self):
        """Distributed with world_size=1 should not add rank suffix."""
        base = Path("/tmp/test-ssd-cache")
        ctx = DistributedContext(enabled=True, rank=0, world_size=1)
        if ctx.enabled and ctx.world_size > 1:
            ssd_dir = base / f"rank_{ctx.rank}"
        else:
            ssd_dir = base
        assert ssd_dir == base

    def test_main_ssd_dir_logic(self):
        """Verify the __main__.py ssd_cache_dir logic pattern."""
        config = ServerConfig()
        base = config.ssd_cache_dir

        # Non-distributed: base unchanged
        ctx_off = DistributedContext()
        ssd_dir_off = base
        if ctx_off.enabled and ctx_off.world_size > 1:
            ssd_dir_off = base / f"rank_{ctx_off.rank}"
        assert ssd_dir_off == base

        # Distributed rank 0
        ctx_r0 = DistributedContext(enabled=True, rank=0, world_size=2)
        ssd_dir_r0 = base
        if ctx_r0.enabled and ctx_r0.world_size > 1:
            ssd_dir_r0 = base / f"rank_{ctx_r0.rank}"
        assert ssd_dir_r0 == base / "rank_0"

        # Distributed rank 1
        ctx_r1 = DistributedContext(enabled=True, rank=1, world_size=2)
        ssd_dir_r1 = base
        if ctx_r1.enabled and ctx_r1.world_size > 1:
            ssd_dir_r1 = base / f"rank_{ctx_r1.rank}"
        assert ssd_dir_r1 == base / "rank_1"

        # Each rank gets a unique directory
        assert ssd_dir_r0 != ssd_dir_r1


# =========================================================================
# F. Compound Drain / Apply Bus Events Tests
# =========================================================================


class TestDrainBusOutboxCompound:
    """Test compound event batching in _drain_bus_outbox."""

    def test_drain_publishes_all_queued_events_as_compound(self):
        """Multiple queued events should be published as single compound batch."""
        published = []

        class MockBus:
            def publish(self, event):
                published.append(event)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Queue 3 submit events
        for i in range(3):
            req = InferenceRequest(request_id=f"r{i}", prompt_tokens=[i], max_tokens=5)
            scheduler.submit_request(req)

        # Drain ALL outbox items
        published.clear()
        scheduler._drain_bus_outbox()

        # Should be ONE compound event containing all 3 submits
        assert len(published) == 1
        assert published[0].typ == "batch"
        inner = pickle.loads(published[0].payload)
        assert len(inner) == 3
        assert all(e.typ == "submit" for e in inner)
        scheduler.stop()

    def test_drain_requeues_on_publish_failure(self):
        """If publish fails, events should be saved in _bus_retry_events for retry."""

        class FailingBus:
            def publish(self, event):
                raise RuntimeError("Network error")

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=FailingBus())

        req = InferenceRequest(request_id="fail-req", prompt_tokens=[1], max_tokens=5)
        scheduler.submit_request(req)

        # Drain should not raise (caught internally)
        scheduler._drain_bus_outbox()

        # Event should be saved in _bus_retry_events (not re-queued to outbox)
        assert len(scheduler._bus_retry_events) > 0
        scheduler.stop()

    def test_drain_outbox_no_bus_is_noop(self):
        """_drain_bus_outbox should do nothing if no control_bus."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler._drain_bus_outbox()  # Should not raise
        scheduler.stop()


class TestApplyBusEventsCompound:
    """Test _apply_bus_events with compound batch events."""

    def test_apply_compound_event_processes_all(self):
        """Compound batch event should apply all inner events."""
        events = [
            ControlEvent.submit(InferenceRequest(
                request_id="batch-r1", prompt_tokens=[1], max_tokens=5
            )),
            ControlEvent.submit(InferenceRequest(
                request_id="batch-r2", prompt_tokens=[2], max_tokens=5
            )),
        ]

        class MockBus:
            def recv(self):
                return ControlEvent(typ="batch", payload=pickle.dumps(events))

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())
        result = scheduler._apply_bus_events()
        assert result is True
        assert scheduler.request_queue.size == 2
        # rank>0 workers don't create result buffers (U1 fix)
        assert "batch-r1" not in scheduler._results
        assert "batch-r2" not in scheduler._results
        scheduler.stop()

    def test_apply_compound_with_shutdown_stops_early(self):
        """Compound event with shutdown should stop processing and return False."""
        events = [
            ControlEvent.submit(InferenceRequest(
                request_id="before-shutdown", prompt_tokens=[1], max_tokens=5
            )),
            ControlEvent.shutdown(),
            ControlEvent.submit(InferenceRequest(
                request_id="after-shutdown", prompt_tokens=[2], max_tokens=5
            )),
        ]

        class MockBus:
            def recv(self):
                return ControlEvent(typ="batch", payload=pickle.dumps(events))

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())
        scheduler._running = True
        result = scheduler._apply_bus_events()
        assert result is False
        assert scheduler._running is False
        # First submit should have been processed (request in queue)
        assert scheduler.request_queue.size == 1
        scheduler.stop()


# =========================================================================
# G. Additional Apply Bus Events Tests (added to existing class via standalone)
# =========================================================================


class TestApplyBusEventsExtended:
    """Extended _apply_bus_events tests: streaming, cancel-from-queue, unknown, recv failure."""

    def _make_scheduler(self, mock_bus, rank=1):
        """Create a scheduler with a mock bus and non-rank0 dist context."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=rank, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=mock_bus)
        return scheduler

    def test_apply_submit_streaming_no_result_buffers(self):
        """Submit event for streaming request should NOT create result buffers."""
        class MockBus:
            def recv(self):
                req = InferenceRequest(
                    request_id="stream-req",
                    prompt_tokens=[1],
                    max_tokens=5,
                    stream=True,
                )
                return ControlEvent.submit(req)

        scheduler = self._make_scheduler(MockBus())
        scheduler._apply_bus_events()

        # Streaming request should NOT have result buffers
        assert "stream-req" not in scheduler._results
        assert "stream-req" not in scheduler._results_ready
        # But should be in queue
        assert scheduler.request_queue.size == 1
        scheduler.stop()

    def test_apply_cancel_removes_from_queue_first(self):
        """Cancel event on rank>0 should try removing from queue before adding to _cancelled."""
        class MockBus:
            call_count = 0
            def recv(self):
                self.call_count += 1
                if self.call_count == 1:
                    return ControlEvent.submit(InferenceRequest(
                        request_id="cancel-target",
                        prompt_tokens=[1],
                        max_tokens=5,
                    ))
                return ControlEvent.cancel("cancel-target")

        bus = MockBus()
        scheduler = self._make_scheduler(bus)

        # First: receive submit
        scheduler._apply_bus_events()
        assert scheduler.request_queue.size == 1

        # Second: receive cancel -- should remove from queue
        scheduler._apply_bus_events()
        assert scheduler.request_queue.size == 0
        # Should NOT be in _cancelled since it was removed from queue
        assert "cancel-target" not in scheduler._cancelled
        scheduler.stop()

    def test_apply_bus_events_unknown_type_returns_true(self):
        """_apply_bus_events should ignore unknown event types gracefully."""
        class MockBus:
            def recv(self):
                return ControlEvent(typ="unknown_type", payload=None)

        scheduler = self._make_scheduler(MockBus())
        result = scheduler._apply_bus_events()
        assert result is True
        scheduler.stop()

    def test_apply_bus_events_recv_failure_continues(self):
        """If recv() raises, _apply_bus_events should return True (continue)."""
        class FailingBus:
            def recv(self):
                raise RuntimeError("Connection lost")

        scheduler = self._make_scheduler(FailingBus())
        result = scheduler._apply_bus_events()
        assert result is True  # Should continue despite error
        scheduler.stop()


# =========================================================================
# H. Cancel Active Path -> Outbox Test
# =========================================================================


class TestCancelActiveQueuesOutbox:
    """Test cancel_request for active requests queues to outbox (rank0)."""

    def test_cancel_active_request_queues_to_outbox(self):
        """cancel_request for an active request should queue cancel to outbox."""
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)

        class MockBus:
            def publish(self, event):
                pass

        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Simulate a request that's already active (not in queue)
        # Need to set up the active sequence properly
        class MockSeq:
            is_finished = False
            block_ids = []
        with scheduler._active_lock:
            scheduler._active_sequences["active-req"] = MockSeq()

        # Drain any existing outbox items from init
        while not scheduler._bus_outbox.empty():
            scheduler._bus_outbox.get_nowait()

        scheduler.cancel_request("active-req")

        # Should have queued a cancel event
        assert not scheduler._bus_outbox.empty()
        event = scheduler._bus_outbox.get_nowait()
        assert event.typ == "cancel"
        assert event.unpack_request_id() == "active-req"
        scheduler.stop()


# =========================================================================
# I. Backward Compatibility / No Distributed Context Tests
# =========================================================================


class TestSchedulerBackwardCompat:
    """Verify scheduler works identically without distributed context."""

    def test_full_lifecycle_without_dist(self):
        """Full submit -> cancel -> stop lifecycle without dist_ctx."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler.run_inference_loop(blocking=False)

        req = InferenceRequest(
            request_id="compat-test",
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )
        scheduler.submit_request(req)
        assert scheduler._bus_outbox.empty()

        scheduler.cancel_request("compat-test")
        assert scheduler._bus_outbox.empty()

        scheduler.stop()
        assert scheduler._bus_outbox.empty()


# =========================================================================
# J. ControlEvent.batch() Factory Tests (F5)
# =========================================================================


class TestControlEventBatch:
    """Test ControlEvent.batch() factory method (F5)."""

    def test_control_event_batch_factory(self):
        """batch() creates an event with typ='batch' and correct payload."""
        events = [
            ControlEvent.submit(InferenceRequest(
                request_id="b1", prompt_tokens=[1], max_tokens=5
            )),
            ControlEvent.cancel("b2"),
            ControlEvent.noop(),
        ]
        batch_event = ControlEvent.batch(events)
        assert batch_event.typ == "batch"
        assert batch_event.payload is not None
        # Verify the payload is the correct list
        unpacked = batch_event.unpack_batch()
        assert len(unpacked) == 3
        assert unpacked[0].typ == "submit"
        assert unpacked[1].typ == "cancel"
        assert unpacked[2].typ == "noop"

    def test_control_event_batch_roundtrip(self):
        """Batch event should survive pickle roundtrip (simulating bus transport)."""
        events = [
            ControlEvent.submit(InferenceRequest(
                request_id="rt1", prompt_tokens=[10, 20], max_tokens=50
            )),
            ControlEvent.shutdown(),
        ]
        batch_event = ControlEvent.batch(events)

        # Simulate pickle roundtrip (as happens in bus transport)
        serialized = pickle.dumps(batch_event)
        deserialized = pickle.loads(serialized)

        assert deserialized.typ == "batch"
        unpacked = deserialized.unpack_batch()
        assert len(unpacked) == 2
        assert unpacked[0].typ == "submit"
        req = unpacked[0].unpack_request()
        assert req.request_id == "rt1"
        assert req.prompt_tokens == [10, 20]
        assert unpacked[1].typ == "shutdown"

    def test_control_event_batch_empty(self):
        """batch() with empty list should still work."""
        batch_event = ControlEvent.batch([])
        assert batch_event.typ == "batch"
        unpacked = batch_event.unpack_batch()
        assert unpacked == []

    def test_unpack_batch_none_payload(self):
        """unpack_batch() with None payload should return empty list."""
        event = ControlEvent(typ="batch", payload=None)
        assert event.unpack_batch() == []


# =========================================================================
# K. Bus Cancel Cleans Result Buffers (U6)
# =========================================================================


class TestBusCancelCleansBuffers:
    """Test that cancel in _apply_bus_events cleans result buffers (U6)."""

    def test_bus_cancel_from_queue_cleans_buffers(self):
        """When cancel event removes a request from queue on worker rank,
        result buffers should be cleaned up."""

        class MockBus:
            call_count = 0
            def recv(self):
                self.call_count += 1
                if self.call_count == 1:
                    # First: submit a non-streaming request
                    return ControlEvent.submit(InferenceRequest(
                        request_id="clean-target",
                        prompt_tokens=[1, 2],
                        max_tokens=5,
                    ))
                # Second: cancel it
                return ControlEvent.cancel("clean-target")

        bus = MockBus()
        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=bus)

        # First apply: submit adds to queue (U1: rank>0 doesn't create result buffers)
        scheduler._apply_bus_events()
        assert "clean-target" not in scheduler._results
        assert "clean-target" not in scheduler._results_ready
        assert scheduler.request_queue.size == 1

        # Second apply: cancel removes from queue
        scheduler._apply_bus_events()
        assert scheduler.request_queue.size == 0
        # Result buffers were never created (U1 fix)
        assert "clean-target" not in scheduler._results
        assert "clean-target" not in scheduler._results_ready
        scheduler.stop()


# =========================================================================
# L. Commit 3 Fixes — U1, U3, U4, U5
# =========================================================================


class TestWorkerResultBufferLeak:
    """U1: Workers (rank>0) should NOT create result buffers on submit events."""

    def test_worker_no_result_buffers_for_submit(self):
        """rank>0 scheduler should not create _results/_results_ready on submit."""
        class MockBus:
            def recv(self):
                req = InferenceRequest(
                    request_id="worker-req",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=10,
                )
                return ControlEvent.submit(req)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())
        scheduler._apply_bus_events()

        # Request should be in queue but no result buffers
        assert scheduler.request_queue.size == 1
        assert "worker-req" not in scheduler._results
        assert "worker-req" not in scheduler._results_ready
        scheduler.stop()

    def test_rank0_still_creates_result_buffers(self):
        """rank0 scheduler should still create result buffers on submit via _apply_bus_events."""
        class MockBus:
            def recv(self):
                req = InferenceRequest(
                    request_id="rank0-req",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=10,
                )
                return ControlEvent.submit(req)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())
        scheduler._apply_bus_events()

        # rank0 should have result buffers
        assert "rank0-req" in scheduler._results
        assert "rank0-req" in scheduler._results_ready
        scheduler.stop()


class TestBusErrorThreshold:
    """U3: Bus consecutive errors should trigger distributed shutdown."""

    def test_bus_error_count_triggers_shutdown(self):
        """Consecutive bus recv failures should set _dist_fatal and stop running."""
        from mlx_lm_server.scheduler import BUS_ERROR_THRESHOLD

        class FailingBus:
            def recv(self):
                raise RuntimeError("Bus connection lost")

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=FailingBus())
        scheduler._running = True

        # Call _apply_bus_events BUS_ERROR_THRESHOLD times
        for i in range(BUS_ERROR_THRESHOLD - 1):
            result = scheduler._apply_bus_events()
            assert result is True  # Not yet at threshold
            assert scheduler._running is True

        # The threshold-reaching call should return False and set _dist_fatal
        result = scheduler._apply_bus_events()
        assert result is False
        assert scheduler._running is False
        assert scheduler._dist_fatal is True
        assert scheduler._dist_fatal_reason == "bus_error_threshold"
        scheduler.stop()

    def test_bus_error_count_resets_on_success(self):
        """Successful recv should reset _bus_error_count to 0."""
        class FlakeyBus:
            call_count = 0
            def recv(self):
                self.call_count += 1
                if self.call_count <= 5:
                    raise RuntimeError("Temporary failure")
                return ControlEvent.noop()

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        bus = FlakeyBus()
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=bus)
        scheduler._running = True

        # 5 failures
        for _ in range(5):
            scheduler._apply_bus_events()
        assert scheduler._bus_error_count == 5

        # 1 success should reset
        scheduler._apply_bus_events()
        assert scheduler._bus_error_count == 0
        scheduler.stop()

    def test_publish_error_threshold_triggers_shutdown(self):
        """Consecutive publish failures should also trigger shutdown."""
        from mlx_lm_server.scheduler import BUS_ERROR_THRESHOLD

        class FailingBus:
            def publish(self, event):
                raise RuntimeError("Publish failed")

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=FailingBus())
        scheduler._running = True

        # Each drain call increments bus_error_count
        for i in range(BUS_ERROR_THRESHOLD):
            scheduler._drain_bus_outbox()

        assert scheduler._dist_fatal is True
        assert scheduler._running is False
        scheduler.stop()

    def test_submit_rejects_when_dist_fatal(self):
        """submit_request should raise RuntimeError when _dist_fatal is True."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler._dist_fatal = True

        req = InferenceRequest(
            request_id="rejected-req",
            prompt_tokens=[1],
            max_tokens=5,
        )
        with pytest.raises(RuntimeError, match="Distributed control plane degraded"):
            scheduler.submit_request(req)
        scheduler.stop()


class TestWorldSize1FailFast:
    """U4: distributed_mode != 'off' but world_size==1 should fail-fast."""

    def test_world_size_1_raises_runtime_error(self):
        """init_distributed should raise when world_size==1 with ring/jaccl mode."""
        from unittest.mock import MagicMock, patch

        config = ServerConfig(
            distributed_mode="ring",
            distributed_hostfile="/tmp/hosts.json",
        )

        # Mock mx.distributed.init to return a group with world_size=1
        mock_group = MagicMock()
        mock_group.rank.return_value = 0
        mock_group.size.return_value = 1

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            import mlx.core as mx_mock
            mx_mock.distributed.init.return_value = mock_group

            with patch("mlx_lm_server.distributed.mx", create=True):
                # We need to patch the import inside init_distributed
                with patch.dict("sys.modules", {"mlx.core": mx_mock}):
                    with pytest.raises(RuntimeError, match="world_size=1"):
                        init_distributed(config)


class TestJoinWorkerLoopTimeout:
    """U5: join_worker_loop with timeout should set worker_timed_out flag."""

    def test_join_worker_loop_timeout_sets_flag(self):
        """join_worker_loop should set worker_timed_out=True when thread doesn't exit."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)

        # Create a thread that never exits (blocked on an event)
        blocker = threading.Event()

        def blocked_loop():
            blocker.wait()  # Block forever until set

        scheduler._inference_thread = threading.Thread(target=blocked_loop, daemon=True)
        scheduler._inference_thread.start()

        # join_worker_loop with very short timeout
        scheduler.join_worker_loop(timeout=0.1)

        assert scheduler.worker_timed_out is True

        # Clean up
        blocker.set()
        scheduler._inference_thread.join(timeout=2.0)
        scheduler.stop()

    def test_join_worker_loop_no_timeout_clears_flag(self):
        """join_worker_loop should set worker_timed_out=False when thread exits normally."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler.run_inference_loop(blocking=False)
        assert scheduler._inference_thread is not None

        # Stop the loop so the thread exits
        scheduler._running = False
        scheduler._new_request_event.set()

        scheduler.join_worker_loop(timeout=5.0)
        assert scheduler.worker_timed_out is False
        scheduler.stop()

    def test_join_worker_loop_no_thread_no_timeout(self):
        """join_worker_loop without inference thread should not set timeout flag."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler.join_worker_loop(timeout=1.0)
        assert scheduler.worker_timed_out is False
        scheduler.stop()


# =========================================================================
# M. Commit 4 Fixes — U7, U8, U9
# =========================================================================


class TestPublishFailurePreservesOrder:
    """U7: Publish failure should preserve event order for retry."""

    def test_publish_failure_preserves_event_order(self):
        """When publish fails, events should be saved for retry in order."""
        fail_count = 0

        class FailOnceBus:
            def publish(self, event):
                nonlocal fail_count
                fail_count += 1
                if fail_count == 1:
                    raise RuntimeError("Network error")
                # Second call succeeds

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=FailOnceBus())

        req = InferenceRequest(request_id="order-test", prompt_tokens=[1], max_tokens=5)
        scheduler.submit_request(req)

        # First drain fails — events should be saved in _bus_retry_events
        scheduler._drain_bus_outbox()
        assert len(scheduler._bus_retry_events) > 0

        # Second drain should retry the saved events (not drain new from outbox)
        scheduler._drain_bus_outbox()
        assert len(scheduler._bus_retry_events) == 0  # Success clears retry
        scheduler.stop()

    def test_retry_backlog_does_not_drain_new_events(self):
        """When retry backlog exists, new outbox items should not be drained."""
        published_events = []

        class TrackingBus:
            call_count = 0
            def publish(self, event):
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("First publish fails")
                published_events.append(event)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        bus = TrackingBus()
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=bus)

        # Submit first request
        req1 = InferenceRequest(request_id="retry-r1", prompt_tokens=[1], max_tokens=5)
        scheduler.submit_request(req1)

        # First drain fails
        scheduler._drain_bus_outbox()
        assert len(scheduler._bus_retry_events) > 0

        # Submit a second request while retry is pending
        req2 = InferenceRequest(request_id="retry-r2", prompt_tokens=[2], max_tokens=5)
        scheduler.submit_request(req2)

        # Second drain should only retry the first batch, NOT drain r2
        scheduler._drain_bus_outbox()
        assert len(published_events) == 1  # Only the retry batch
        # Verify the retried batch contains r1 but not r2
        inner = published_events[0].unpack_batch()
        request_ids = [e.unpack_request().request_id for e in inner if e.typ == "submit"]
        assert "retry-r1" in request_ids
        assert "retry-r2" not in request_ids

        # r2 should still be in the outbox
        assert not scheduler._bus_outbox.empty()
        scheduler.stop()


class TestApplyBusEventsAddException:
    """U8: request_queue.add() exception in _apply_bus_events should be handled."""

    def test_apply_bus_events_add_exception_handled(self):
        """If request_queue.add() fails in _apply_bus_events, should not crash."""
        class MockBus:
            def recv(self):
                req = InferenceRequest(
                    request_id="add-fail-req",
                    prompt_tokens=[1],
                    max_tokens=5,
                )
                return ControlEvent.submit(req)

        config = ServerConfig(max_queue_size=1)  # queue size 1
        ctx = DistributedContext(enabled=True, rank=1, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Fill the queue first
        scheduler.request_queue.add(InferenceRequest(
            request_id="filler", prompt_tokens=[1], max_tokens=1
        ))

        # Apply bus event should not crash even though queue is full
        result = scheduler._apply_bus_events()
        assert result is True  # Should continue despite add failure
        # Queue should still have original item
        assert scheduler.request_queue.size == 1
        scheduler.stop()


class TestBusOutboxBackpressure:
    """U9: Bus outbox backpressure should reject requests."""

    def test_distributed_nonstream_submit_precreates_result_slot(self):
        """Distributed non-stream submit should pre-register result buffers."""
        class MockBus:
            def publish(self, event):
                pass

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        req = InferenceRequest(request_id="precreate-slot", prompt_tokens=[1], max_tokens=5, stream=False)
        scheduler.submit_request(req)

        with scheduler._results_lock:
            assert "precreate-slot" in scheduler._results
            assert "precreate-slot" in scheduler._results_ready
            assert isinstance(scheduler._results_ready["precreate-slot"], threading.Event)
        scheduler.stop()

    def test_distributed_submit_reject_cleans_precreated_result_slot(self):
        """If distributed submit is rejected, pre-registered result buffers are cleaned up."""
        from mlx_lm_server.types import BusOutboxFullError
        from mlx_lm_server.scheduler import BUS_OUTBOX_MAXSIZE, BUS_OUTBOX_CONTROL_RESERVE

        class MockBus:
            def publish(self, event):
                pass

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Fill to the submit rejection threshold
        from mlx_lm_server.distributed_bus import ControlEvent as CE
        for _ in range(BUS_OUTBOX_MAXSIZE - BUS_OUTBOX_CONTROL_RESERVE):
            scheduler._bus_outbox.put_nowait(CE.noop())

        req = InferenceRequest(request_id="reject-cleanup", prompt_tokens=[1], max_tokens=5, stream=False)
        with pytest.raises(BusOutboxFullError):
            scheduler.submit_request(req)

        with scheduler._results_lock:
            assert "reject-cleanup" not in scheduler._results
            assert "reject-cleanup" not in scheduler._results_ready
        scheduler.stop()

    def test_bus_outbox_full_rejects_request(self):
        """When outbox is nearly full, submit should raise BusOutboxFullError."""
        from mlx_lm_server.types import BusOutboxFullError
        from mlx_lm_server.scheduler import BUS_OUTBOX_MAXSIZE, BUS_OUTBOX_CONTROL_RESERVE

        class MockBus:
            def publish(self, event):
                pass

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Fill the outbox to the backpressure threshold
        from mlx_lm_server.distributed_bus import ControlEvent as CE
        for i in range(BUS_OUTBOX_MAXSIZE - BUS_OUTBOX_CONTROL_RESERVE):
            scheduler._bus_outbox.put_nowait(CE.noop())

        # Next submit should be rejected
        req = InferenceRequest(request_id="full-bus-req", prompt_tokens=[1], max_tokens=5)
        with pytest.raises(BusOutboxFullError):
            scheduler.submit_request(req)

        # Request should NOT be in queue (was rolled back)
        assert scheduler.request_queue.size == 0
        scheduler.stop()

    def test_dist_fatal_submit_returns_503(self):
        """After _dist_fatal, submit should raise RuntimeError."""
        config = ServerConfig()
        scheduler = Scheduler(config=config)
        scheduler._dist_fatal = True

        req = InferenceRequest(request_id="fatal-req", prompt_tokens=[1], max_tokens=5)
        with pytest.raises(RuntimeError, match="Distributed control plane degraded"):
            scheduler.submit_request(req)
        scheduler.stop()

    def test_cancel_uses_control_reserve(self):
        """Cancel events should use the control reserve and succeed even when submit would fail."""
        from mlx_lm_server.types import BusOutboxFullError
        from mlx_lm_server.scheduler import BUS_OUTBOX_MAXSIZE, BUS_OUTBOX_CONTROL_RESERVE

        class MockBus:
            def publish(self, event):
                pass

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Fill the outbox to the backpressure threshold
        from mlx_lm_server.distributed_bus import ControlEvent as CE
        for i in range(BUS_OUTBOX_MAXSIZE - BUS_OUTBOX_CONTROL_RESERVE):
            scheduler._bus_outbox.put_nowait(CE.noop())

        # Put a request in queue so cancel has something to find
        scheduler.request_queue.add(InferenceRequest(
            request_id="cancel-reserve", prompt_tokens=[1], max_tokens=5
        ))

        # Cancel should work even though submit would fail (uses control reserve)
        result = scheduler.cancel_request("cancel-reserve")
        assert result is True
        scheduler.stop()

    def test_distributed_cancel_is_outbox_only(self):
        """In distributed mode, cancel should go through outbox."""
        published = []

        class MockBus:
            def publish(self, event):
                published.append(event)

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        # Submit and drain
        req = InferenceRequest(request_id="outbox-cancel", prompt_tokens=[1], max_tokens=5)
        scheduler.submit_request(req)
        # Drain submit event
        while not scheduler._bus_outbox.empty():
            scheduler._bus_outbox.get_nowait()

        # Cancel queued request
        scheduler.cancel_request("outbox-cancel")

        # Cancel event should be in outbox
        assert not scheduler._bus_outbox.empty()
        event = scheduler._bus_outbox.get_nowait()
        assert event.typ == "cancel"
        scheduler.stop()

    def test_rank0_cancel_local_apply_cleans_orphan_buffers(self):
        """Rank0 local cancel apply should clean orphan result buffers for unknown requests."""
        class MockBus:
            def publish(self, event):
                pass

        config = ServerConfig()
        ctx = DistributedContext(enabled=True, rank=0, world_size=2)
        scheduler = Scheduler(config=config, dist_ctx=ctx, control_bus=MockBus())

        request_id = "orphan-cancel"
        with scheduler._results_lock:
            scheduler._results[request_id] = []
            scheduler._results_ready[request_id] = threading.Event()

        scheduler.register_stream(request_id)
        scheduler._bus_outbox.put_nowait(ControlEvent.cancel(request_id))
        scheduler._drain_bus_outbox()

        with scheduler._results_lock:
            assert request_id not in scheduler._results
            assert request_id not in scheduler._results_ready
        with scheduler._streams_lock:
            assert request_id not in scheduler._streams
        scheduler.stop()


# =========================================================================
# N. Auto-Relaunch Tests
# =========================================================================


class TestAutoRelaunch:
    """Test _maybe_relaunch_under_mlx_launch() auto-relaunch logic."""

    def test_skip_if_mlx_rank_set(self, monkeypatch):
        """Should not exec when MLX_RANK is already set (recursion guard)."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.setenv("MLX_RANK", "0")
        with patch("os.execvp") as mock_exec:
            _maybe_relaunch_under_mlx_launch()
            mock_exec.assert_not_called()

    def test_skip_if_mode_off(self, monkeypatch):
        """Should not exec when distributed_mode is 'off'."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        monkeypatch.setattr("sys.argv", ["mlx_lm_server", "--distributed-mode", "off"])
        with patch("os.execvp") as mock_exec:
            _maybe_relaunch_under_mlx_launch()
            mock_exec.assert_not_called()

    def test_skip_if_no_distributed_flag(self, monkeypatch):
        """Should not exec when no --distributed-mode flag (defaults to off)."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        monkeypatch.setattr("sys.argv", ["mlx_lm_server", "--model", "test-model"])
        with patch("os.execvp") as mock_exec:
            _maybe_relaunch_under_mlx_launch()
            mock_exec.assert_not_called()

    def test_error_if_mlx_launch_missing(self, monkeypatch):
        """Should sys.exit(1) if mlx.launch is not found."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        monkeypatch.setattr("sys.argv", ["mlx_lm_server", "--distributed-mode", "ring", "--distributed-hostfile", "/tmp/hosts.json"])
        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                _maybe_relaunch_under_mlx_launch()
            assert exc_info.value.code == 1

    def test_ring_cmd_with_hostfile(self, monkeypatch):
        """Should build correct mlx.launch command for ring + hostfile."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        monkeypatch.setattr("sys.argv", [
            "mlx_lm_server", "--distributed-mode", "ring",
            "--distributed-hostfile", "/tmp/hosts.json",
            "--model", "test-model",
        ])
        exec_calls = []
        with patch("shutil.which", return_value="/usr/local/bin/mlx.launch"):
            with patch("os.execvp", side_effect=lambda p, c: exec_calls.append((p, c))):
                _maybe_relaunch_under_mlx_launch()

        assert len(exec_calls) == 1
        path, cmd = exec_calls[0]
        assert path == "mlx.launch"
        assert "--backend" in cmd
        assert "ring" in cmd
        assert "--hostfile" in cmd
        assert "/tmp/hosts.json" in cmd
        # sys.argv[1:] should be passed through
        assert "--distributed-mode" in cmd
        assert "--model" in cmd

    def test_ring_cmd_with_num_local_ranks(self, monkeypatch):
        """Should build correct mlx.launch command for ring + num-local-ranks (no hostfile)."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        monkeypatch.setattr("sys.argv", [
            "mlx_lm_server", "--distributed-mode", "ring",
            "--num-local-ranks", "4",
        ])
        exec_calls = []
        with patch("shutil.which", return_value="/usr/local/bin/mlx.launch"):
            with patch("os.execvp", side_effect=lambda p, c: exec_calls.append((p, c))):
                _maybe_relaunch_under_mlx_launch()

        assert len(exec_calls) == 1
        _, cmd = exec_calls[0]
        assert "--hosts" in cmd
        assert "localhost" in cmd
        assert "-n" in cmd
        assert "4" in cmd

    def test_jaccl_cmd(self, monkeypatch):
        """Should build correct mlx.launch command for jaccl + set env vars."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        monkeypatch.setattr("sys.argv", [
            "mlx_lm_server", "--distributed-mode", "jaccl",
            "--distributed-ibv-devices", "/dev/ibv0",
            "--distributed-jaccl-coordinator", "192.168.1.1:9000",
        ])
        exec_calls = []
        with patch("shutil.which", return_value="/usr/local/bin/mlx.launch"):
            with patch("os.execvp", side_effect=lambda p, c: exec_calls.append((p, c))):
                _maybe_relaunch_under_mlx_launch()

        assert len(exec_calls) == 1
        _, cmd = exec_calls[0]
        assert "--backend" in cmd
        assert "jaccl" in cmd
        # Env vars should be set
        assert os.environ.get("MLX_IBV_DEVICES") == "/dev/ibv0"
        assert os.environ.get("MLX_JACCL_COORDINATOR") == "192.168.1.1:9000"

    def test_passthrough_all_args(self, monkeypatch):
        """Should forward all original sys.argv[1:] verbatim."""
        from unittest.mock import patch
        from mlx_lm_server.__main__ import _maybe_relaunch_under_mlx_launch

        monkeypatch.delenv("MLX_RANK", raising=False)
        original_args = [
            "mlx_lm_server",
            "--distributed-mode", "ring",
            "--distributed-hostfile", "/tmp/hosts.json",
            "--model", "mlx-community/Qwen3-4B-4bit",
            "--port", "9000",
            "--max-batch-size", "16",
        ]
        monkeypatch.setattr("sys.argv", original_args)
        exec_calls = []
        with patch("shutil.which", return_value="/usr/local/bin/mlx.launch"):
            with patch("os.execvp", side_effect=lambda p, c: exec_calls.append((p, c))):
                _maybe_relaunch_under_mlx_launch()

        _, cmd = exec_calls[0]
        # All original args (except argv[0]) should appear in the command
        for arg in original_args[1:]:
            assert arg in cmd
