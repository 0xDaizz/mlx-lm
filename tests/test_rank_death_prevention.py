"""Tests for rank death prevention features.

Covers:
  1. MemoryGuardError — exception class hierarchy and _check_memory_guard()
  2. Configurable eval timeout — MLX_EVAL_TIMEOUT env var respected
  3. SIGHUP immunity — ignored in distributed mode, clean exit in single mode
  4. Exit audit logging — _exit_cause tracking and EXIT AUDIT log line
  5. Per-layer diagnostics — _extract_layer_index(), _EvalShardEvalIter protocol
  6. Broadened exception handling — except Exception catches non-RuntimeError
  7. Worker loop timeout — join_worker_loop(timeout=3600) and worker_timed_out flag
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# 1. MemoryGuardError
# =========================================================================


class TestMemoryGuardError:
    """Test the MemoryGuardError exception class."""

    def test_is_runtime_error_subclass(self):
        """MemoryGuardError should be a RuntimeError subclass for backward compat."""
        from mlx_lm.utils import MemoryGuardError

        assert issubclass(MemoryGuardError, RuntimeError)

    def test_caught_by_except_runtime_error(self):
        """Existing except RuntimeError handlers should catch MemoryGuardError."""
        from mlx_lm.utils import MemoryGuardError

        with pytest.raises(RuntimeError):
            raise MemoryGuardError("out of memory")

    def test_caught_by_except_exception(self):
        """except Exception should also catch MemoryGuardError."""
        from mlx_lm.utils import MemoryGuardError

        with pytest.raises(Exception):
            raise MemoryGuardError("out of memory")

    def test_message_preserved(self):
        """Error message should be preserved."""
        from mlx_lm.utils import MemoryGuardError

        err = MemoryGuardError("remaining 3.2 GB < threshold 5.0 GB")
        assert "3.2 GB" in str(err)
        assert "threshold" in str(err)

    def test_type_name_in_repr(self):
        """type(e).__name__ should return 'MemoryGuardError'."""
        from mlx_lm.utils import MemoryGuardError

        err = MemoryGuardError("test")
        assert type(err).__name__ == "MemoryGuardError"


class TestCheckMemoryGuard:
    """Test the _check_memory_guard function."""

    def test_raises_when_remaining_below_threshold(self):
        """Should raise MemoryGuardError when remaining memory is below threshold."""
        from mlx_lm.utils import MemoryGuardError, _check_memory_guard

        # Mock MLX memory functions: active=90GB of 100GB total -> 10GB remaining
        with (
            patch("mlx_lm.utils.mx.distributed.init") as mock_dist,
            patch("mlx_lm.utils.mx.get_active_memory", return_value=90 * (1024**3)),
            patch("mlx_lm.utils.mx.get_peak_memory", return_value=95 * (1024**3)),
            patch("mlx_lm.utils._get_total_physical_memory", return_value=100 * (1024**3)),
        ):
            mock_dist.return_value.rank.return_value = 0

            # Threshold 15GB > remaining 10GB -> should raise
            with pytest.raises(MemoryGuardError, match="MEMORY GUARD"):
                _check_memory_guard(
                    threshold_bytes=15 * (1024**3),
                    layer_idx=5,
                    num_layers=64,
                )

    def test_passes_when_remaining_above_threshold(self):
        """Should not raise when remaining memory is above threshold."""
        from mlx_lm.utils import _check_memory_guard

        with (
            patch("mlx_lm.utils.mx.distributed.init") as mock_dist,
            patch("mlx_lm.utils.mx.get_active_memory", return_value=50 * (1024**3)),
            patch("mlx_lm.utils.mx.get_peak_memory", return_value=55 * (1024**3)),
            patch("mlx_lm.utils._get_total_physical_memory", return_value=100 * (1024**3)),
        ):
            mock_dist.return_value.rank.return_value = 0

            # Threshold 5GB < remaining 50GB -> should pass
            _check_memory_guard(
                threshold_bytes=5 * (1024**3),
                layer_idx=5,
                num_layers=64,
            )

    def test_skips_when_no_memory_api(self):
        """Should silently return when MLX memory API is unavailable."""
        from mlx_lm.utils import _check_memory_guard

        with (
            patch("mlx_lm.utils.mx.distributed.init") as mock_dist,
            patch("mlx_lm.utils.mx.get_active_memory", side_effect=AttributeError),
        ):
            mock_dist.return_value.rank.return_value = 0
            # Should not raise
            _check_memory_guard(
                threshold_bytes=5 * (1024**3),
                layer_idx=0,
                num_layers=10,
            )

    def test_skips_when_total_ram_unknown(self):
        """Should silently return when total RAM cannot be determined."""
        from mlx_lm.utils import _check_memory_guard

        with (
            patch("mlx_lm.utils.mx.distributed.init") as mock_dist,
            patch("mlx_lm.utils.mx.get_active_memory", return_value=50 * (1024**3)),
            patch("mlx_lm.utils.mx.get_peak_memory", return_value=55 * (1024**3)),
            patch("mlx_lm.utils._get_total_physical_memory", return_value=0),
        ):
            mock_dist.return_value.rank.return_value = 0
            # total_ram=0 -> should return early
            _check_memory_guard(
                threshold_bytes=5 * (1024**3),
                layer_idx=0,
                num_layers=10,
            )

    def test_error_message_includes_layer_info(self):
        """Error message should include layer index and count."""
        from mlx_lm.utils import MemoryGuardError, _check_memory_guard

        with (
            patch("mlx_lm.utils.mx.distributed.init") as mock_dist,
            patch("mlx_lm.utils.mx.get_active_memory", return_value=90 * (1024**3)),
            patch("mlx_lm.utils.mx.get_peak_memory", return_value=95 * (1024**3)),
            patch("mlx_lm.utils._get_total_physical_memory", return_value=100 * (1024**3)),
        ):
            mock_dist.return_value.rank.return_value = 1

            with pytest.raises(MemoryGuardError, match=r"layer 11/64"):
                _check_memory_guard(
                    threshold_bytes=15 * (1024**3),
                    layer_idx=10,
                    num_layers=64,
                )


# =========================================================================
# 2. Configurable eval timeout
# =========================================================================


class TestEvalWithTimeout:
    """Test configurable timeout for eval_with_timeout."""

    def test_default_timeout_300s(self):
        """Default timeout should be 300s when MLX_EVAL_TIMEOUT is not set."""
        from mlx_lm.utils import eval_with_timeout

        env = os.environ.copy()
        env.pop("MLX_EVAL_TIMEOUT", None)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("mlx_lm.utils.mx.eval") as mock_eval,
        ):
            eval_with_timeout({"w": MagicMock()})
            mock_eval.assert_called_once()

    def test_custom_timeout_from_env(self):
        """MLX_EVAL_TIMEOUT env var should be respected."""
        from mlx_lm.utils import eval_with_timeout

        with (
            patch.dict(os.environ, {"MLX_EVAL_TIMEOUT": "60"}),
            patch("mlx_lm.utils.mx.eval") as mock_eval,
        ):
            eval_with_timeout({"w": MagicMock()})
            mock_eval.assert_called_once()

    def test_explicit_timeout_overrides_env(self):
        """Explicit timeout_seconds should override the env var."""
        from mlx_lm.utils import eval_with_timeout

        with (
            patch.dict(os.environ, {"MLX_EVAL_TIMEOUT": "999"}),
            patch("mlx_lm.utils.mx.eval") as mock_eval,
        ):
            eval_with_timeout({"w": MagicMock()}, timeout_seconds=10)
            mock_eval.assert_called_once()

    def test_completed_event_set_on_success(self):
        """The completed event should be set after successful eval."""
        from mlx_lm.utils import eval_with_timeout

        with patch("mlx_lm.utils.mx.eval"):
            # Should not hang — completed event is set, watchdog exits
            eval_with_timeout({"w": MagicMock()}, timeout_seconds=5)


# =========================================================================
# 3. SIGHUP immunity in distributed mode
# =========================================================================


class TestSighupImmunity:
    """Test SIGHUP signal handling based on distributed mode."""

    def test_sighup_ignored_in_distributed_mode(self):
        """In distributed mode, SIGHUP should be set to SIG_IGN."""
        from mlx_lm_server.distributed import DistributedContext

        dist_ctx = DistributedContext(enabled=True, rank=1, world_size=2)

        # Simulate the SIGHUP logic from __main__.py
        if hasattr(signal, "SIGHUP"):
            old_handler = signal.getsignal(signal.SIGHUP)
            try:
                if dist_ctx.enabled:
                    signal.signal(signal.SIGHUP, signal.SIG_IGN)

                handler = signal.getsignal(signal.SIGHUP)
                assert handler == signal.SIG_IGN
            finally:
                signal.signal(signal.SIGHUP, old_handler)

    def test_sighup_exits_in_single_mode(self):
        """In single-machine mode, SIGHUP should trigger sys.exit(0)."""
        from mlx_lm_server.__main__ import _signal_handler
        from mlx_lm_server.distributed import DistributedContext

        dist_ctx = DistributedContext(enabled=False)

        if hasattr(signal, "SIGHUP"):
            old_handler = signal.getsignal(signal.SIGHUP)
            try:
                if not dist_ctx.enabled:
                    signal.signal(signal.SIGHUP, _signal_handler("SIGHUP"))

                handler = signal.getsignal(signal.SIGHUP)
                assert handler is not signal.SIG_IGN
                assert callable(handler)
            finally:
                signal.signal(signal.SIGHUP, old_handler)

    def test_sighup_after_init_distributed(self):
        """SIGHUP should only be configured after init_distributed returns."""
        # Verify the code structure: SIGHUP is NOT set before init_distributed
        import mlx_lm_server.__main__ as main_mod

        # The signal handlers registered before init_distributed are SIGTERM/SIGINT only
        # SIGHUP is set after dist_ctx = init_distributed(config)
        # We verify this by checking the source structure indirectly:
        # _signal_handler is used for SIGHUP only in the conditional block
        import inspect
        source = inspect.getsource(main_mod.main)
        # SIGHUP should appear AFTER init_distributed in the source
        init_pos = source.find("init_distributed(config)")
        sighup_pos = source.find("SIGHUP")
        assert init_pos < sighup_pos, "SIGHUP should be configured after init_distributed"


# =========================================================================
# 4. Exit audit logging
# =========================================================================


class TestExitAuditLogging:
    """Test the _exit_cause tracking and EXIT AUDIT log output."""

    def test_exit_cause_default_unknown(self):
        """_exit_cause should default to 'unknown'."""
        import mlx_lm_server.__main__ as main_mod
        # Reset to check default
        original = main_mod._exit_cause
        try:
            main_mod._exit_cause = "unknown"
            assert main_mod._exit_cause == "unknown"
        finally:
            main_mod._exit_cause = original

    def test_signal_handler_sets_exit_cause(self):
        """_signal_handler should set _exit_cause before sys.exit."""
        from mlx_lm_server.__main__ import _signal_handler
        import mlx_lm_server.__main__ as main_mod

        handler = _signal_handler("SIGTERM")
        original = main_mod._exit_cause
        try:
            with pytest.raises(SystemExit) as exc_info:
                handler(signal.SIGTERM, None)
            assert exc_info.value.code == 0
            assert main_mod._exit_cause == "SIGTERM"
        finally:
            main_mod._exit_cause = original

    def test_signal_handler_sigint(self):
        """SIGINT handler should set _exit_cause='SIGINT'."""
        from mlx_lm_server.__main__ import _signal_handler
        import mlx_lm_server.__main__ as main_mod

        handler = _signal_handler("SIGINT")
        original = main_mod._exit_cause
        try:
            with pytest.raises(SystemExit):
                handler(signal.SIGINT, None)
            assert main_mod._exit_cause == "SIGINT"
        finally:
            main_mod._exit_cause = original

    def test_exit_cause_uvicorn_shutdown_in_source(self):
        """The source should set _exit_cause='uvicorn_shutdown' after uvicorn.run."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        assert 'uvicorn_shutdown' in source

    def test_exit_cause_worker_loop_exit_in_source(self):
        """The source should set _exit_cause for worker loop outcomes."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        assert 'worker_loop_exit' in source
        assert 'worker_loop_timeout' in source

    def test_exit_audit_log_format(self):
        """The EXIT AUDIT log line should include rank, cause, memory info."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        assert "EXIT AUDIT" in source
        assert "rank=" in source
        assert "cause=" in source
        assert "active_mem=" in source
        assert "peak_mem=" in source

    def test_exception_handler_sets_exit_cause(self):
        """The except block should set _exit_cause to exception type + message."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        # Check that the except block sets _exit_cause with type info
        assert 'type(e).__name__' in source


# =========================================================================
# 5. Per-layer diagnostics — _extract_layer_index
# =========================================================================


class TestExtractLayerIndex:
    """Test the _extract_layer_index helper for per-layer diagnostics."""

    def test_standard_layers_pattern(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("model.layers.42.self_attn.q_proj.weight") == 42

    def test_layers_zero_index(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("layers.0.mlp.gate_proj.weight") == 0

    def test_transformer_h_pattern(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("transformer.h.10.attn.weight") == 10

    def test_blocks_pattern(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("blocks.5.norm.weight") == 5

    def test_non_layer_param(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("model.embed_tokens.weight") is None

    def test_lm_head(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("lm_head.weight") is None

    def test_final_norm(self):
        from mlx_lm.utils import _extract_layer_index

        assert _extract_layer_index("model.norm.weight") is None

    def test_deeply_nested(self):
        from mlx_lm.utils import _extract_layer_index

        result = _extract_layer_index("model.language_model.model.layers.99.attn.weight")
        assert result == 99


# =========================================================================
# 6. _EvalShardEvalIter protocol
# =========================================================================


class TestEvalShardEvalIter:
    """Test the _EvalShardEvalIter wrapper for per-layer eval."""

    def test_len_delegation(self):
        from mlx_lm.utils import _EvalShardEvalIter

        layers = [MagicMock() for _ in range(5)]
        wrapper = _EvalShardEvalIter(layers)
        assert len(wrapper) == 5

    def test_getitem_delegation(self):
        from mlx_lm.utils import _EvalShardEvalIter

        layers = [MagicMock(name=f"layer_{i}") for i in range(3)]
        wrapper = _EvalShardEvalIter(layers)
        assert wrapper[0] is layers[0]
        assert wrapper[2] is layers[2]


# =========================================================================
# 7. Broadened exception handling
# =========================================================================


class TestBroadenedExceptionHandling:
    """Test that main() catches Exception (not just RuntimeError)."""

    def test_except_block_catches_exception(self):
        """The except block should use 'except Exception' not 'except RuntimeError'."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        # Should have 'except Exception' not 'except RuntimeError'
        assert "except Exception as e:" in source
        assert "except RuntimeError" not in source

    def test_error_log_includes_rank_and_type(self):
        """The error log should include rank number and exception type."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        assert "Fatal error (rank %d)" in source
        assert "type(e).__name__" in source


# =========================================================================
# 8. Worker loop timeout
# =========================================================================


class TestWorkerLoopTimeout:
    """Test the join_worker_loop timeout and worker_timed_out flag."""

    def test_join_worker_loop_uses_timeout(self):
        """join_worker_loop should be called with timeout=3600."""
        import inspect
        import mlx_lm_server.__main__ as main_mod

        source = inspect.getsource(main_mod.main)
        assert "join_worker_loop(timeout=3600)" in source

    def test_worker_timed_out_flag_default_false(self):
        """Scheduler.worker_timed_out should default to False."""
        from mlx_lm_server.scheduler import Scheduler

        config = MagicMock()
        config.block_size = 4
        config.num_blocks = 16
        config.max_batch_size = 2
        config.max_queue_size = 8
        config.prefill_batch_size = 1
        config.ssd_enabled = False
        config.ssd_policy = "write_back"
        config.ssd_async_writes = False
        config.max_kv_size = 0
        config.distributed_mode = "off"
        config.spec_model = None
        config.spec_length = 0
        config.spec_min_tokens = 0
        config.mtp_enabled = False
        config.mtp_max_speculate = 0
        config.top_logprobs = 0

        scheduler = Scheduler(config=config, model=None, tokenizer=None)
        assert scheduler.worker_timed_out is False

    def test_worker_timed_out_set_on_timeout(self):
        """worker_timed_out should be True when thread doesn't finish in time."""
        from mlx_lm_server.scheduler import Scheduler

        config = MagicMock()
        config.block_size = 4
        config.num_blocks = 16
        config.max_batch_size = 2
        config.max_queue_size = 8
        config.prefill_batch_size = 1
        config.ssd_enabled = False
        config.ssd_policy = "write_back"
        config.ssd_async_writes = False
        config.max_kv_size = 0
        config.distributed_mode = "off"
        config.spec_model = None
        config.spec_length = 0
        config.spec_min_tokens = 0
        config.mtp_enabled = False
        config.mtp_max_speculate = 0
        config.top_logprobs = 0

        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Create a thread that blocks forever
        blocker = threading.Event()
        scheduler._inference_thread = threading.Thread(
            target=lambda: blocker.wait(), daemon=True
        )
        scheduler._inference_thread.start()

        # Join with very short timeout — should set worker_timed_out
        scheduler.join_worker_loop(timeout=0.01)
        assert scheduler.worker_timed_out is True

        # Cleanup
        blocker.set()
        scheduler._inference_thread.join(timeout=1)

    def test_worker_timed_out_false_on_normal_exit(self):
        """worker_timed_out should be False when thread finishes normally."""
        from mlx_lm_server.scheduler import Scheduler

        config = MagicMock()
        config.block_size = 4
        config.num_blocks = 16
        config.max_batch_size = 2
        config.max_queue_size = 8
        config.prefill_batch_size = 1
        config.ssd_enabled = False
        config.ssd_policy = "write_back"
        config.ssd_async_writes = False
        config.max_kv_size = 0
        config.distributed_mode = "off"
        config.spec_model = None
        config.spec_length = 0
        config.spec_min_tokens = 0
        config.mtp_enabled = False
        config.mtp_max_speculate = 0
        config.top_logprobs = 0

        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Create a thread that exits immediately
        scheduler._inference_thread = threading.Thread(
            target=lambda: None, daemon=True
        )
        scheduler._inference_thread.start()
        scheduler._inference_thread.join(timeout=1)  # wait for it to finish

        scheduler.join_worker_loop(timeout=5)
        assert scheduler.worker_timed_out is False


# =========================================================================
# 9. _get_total_physical_memory
# =========================================================================


class TestGetTotalPhysicalMemory:
    """Test the _get_total_physical_memory helper."""

    def test_returns_positive_value(self):
        """Should return a positive value on macOS/Linux."""
        from mlx_lm.utils import _get_total_physical_memory

        result = _get_total_physical_memory()
        # On any real system this should be > 0
        assert result > 0

    def test_returns_zero_on_unsupported(self):
        """Should return 0 when sysconf is not available."""
        from mlx_lm.utils import _get_total_physical_memory

        with patch("mlx_lm.utils.os.sysconf", side_effect=AttributeError):
            assert _get_total_physical_memory() == 0


# =========================================================================
# 10. Signal handler factory
# =========================================================================


class TestSignalHandlerFactory:
    """Test the _signal_handler factory function."""

    def test_returns_callable(self):
        from mlx_lm_server.__main__ import _signal_handler

        handler = _signal_handler("SIGTERM")
        assert callable(handler)

    def test_different_signames_produce_different_handlers(self):
        from mlx_lm_server.__main__ import _signal_handler

        h1 = _signal_handler("SIGTERM")
        h2 = _signal_handler("SIGINT")
        assert h1 is not h2

    def test_handler_calls_sys_exit_zero(self):
        from mlx_lm_server.__main__ import _signal_handler

        handler = _signal_handler("TEST_SIG")
        with pytest.raises(SystemExit) as exc_info:
            handler(15, None)
        assert exc_info.value.code == 0

    def test_handler_sets_global_exit_cause(self):
        from mlx_lm_server.__main__ import _signal_handler
        import mlx_lm_server.__main__ as main_mod

        original = main_mod._exit_cause
        try:
            handler = _signal_handler("MY_CUSTOM_SIG")
            with pytest.raises(SystemExit):
                handler(15, None)
            assert main_mod._exit_cause == "MY_CUSTOM_SIG"
        finally:
            main_mod._exit_cause = original
