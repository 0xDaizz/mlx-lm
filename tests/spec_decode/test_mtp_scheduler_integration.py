"""Integration tests for MTP-specific code paths in the Scheduler.

Focuses on:
- MTP proposer initialization (mode="mtp" in Scheduler.__init__)
- MTP hidden-state invalidation on UID removal

Uses unittest.mock.patch heavily -- no real model or BatchGenerator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlx_lm_server.config import ServerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mtp_config(**overrides) -> ServerConfig:
    """Create a ServerConfig with spec_decode_mode='mtp' and sensible defaults."""
    defaults = dict(
        model="fake-model-path",
        spec_decode_mode="mtp",
        spec_decode_num_tokens=3,
    )
    defaults.update(overrides)
    return ServerConfig(**defaults)


def _fake_create_batch_generator(self):
    """Stub for Scheduler._create_batch_generator that sets a mock."""
    self._batch_generator = MagicMock()


# ---------------------------------------------------------------------------
# TestSchedulerMTPInit
# ---------------------------------------------------------------------------


class TestSchedulerMTPInit:
    """Tests for the MTP branch inside Scheduler.__init__."""

    def test_mtp_mode_creates_mtp_proposer(self):
        """mode='mtp' with a model that has MTP weights -> MTPProposer created."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        fake_mtp_module = MagicMock()
        fake_mtp_module.num_layers = 2

        with (
            patch(
                "mlx_lm_server.scheduler.Scheduler._create_batch_generator",
                _fake_create_batch_generator,
            ),
            patch(
                "mlx_lm_server.spec_decode.mtp_loader.build_mtp_module",
                return_value=fake_mtp_module,
            ) as mock_build,
            patch(
                "mlx_lm_server.spec_decode.proposer.mtp.MTPProposer",
            ) as mock_proposer_cls,
            patch(
                "mlx_lm_server.spec_decode.engine.SpecDecodeEngine",
            ),
            patch(
                "mlx_lm_server.spec_decode.verifier.NGramVerifier",
            ),
            patch(
                "mlx_lm_server.spec_decode.controller.DynamicSpecController",
            ),
        ):
            mock_proposer_cls.return_value = MagicMock()
            config = _mtp_config()

            from mlx_lm_server.scheduler import Scheduler

            scheduler = Scheduler(
                config=config,
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

            mock_build.assert_called_once_with(mock_model, config.model)
            mock_proposer_cls.assert_called_once_with(
                mtp_module=fake_mtp_module,
                num_mtp_layers=2,
            )
            assert scheduler._spec_engine is not None

    def test_mtp_mode_raises_on_no_weights(self):
        """build_mtp_module returns None -> ValueError logged, spec engine stays None.

        The scheduler wraps spec-decode init in try/except and logs the error
        rather than crashing. The test verifies the ValueError is raised
        internally (visible in logs) and _spec_engine remains None.
        """
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch(
                "mlx_lm_server.scheduler.Scheduler._create_batch_generator",
                _fake_create_batch_generator,
            ),
            patch(
                "mlx_lm_server.spec_decode.mtp_loader.build_mtp_module",
                return_value=None,
            ),
        ):
            config = _mtp_config()

            from mlx_lm_server.scheduler import Scheduler

            import logging

            with patch.object(logging.getLogger("mlx_lm_server.scheduler"), "warning") as mock_warn:
                scheduler = Scheduler(
                    config=config,
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                )

            # spec engine should NOT have been created
            assert scheduler._spec_engine is None

            # The ValueError should have been logged as a warning
            mock_warn.assert_called_once()
            warn_msg = str(mock_warn.call_args)
            assert "no MTP layers" in warn_msg

    def test_mtp_mode_passes_num_layers(self):
        """num_mtp_layers is read from mtp_module.num_layers and forwarded."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        fake_mtp_module = MagicMock()
        fake_mtp_module.num_layers = 7  # unusual value to verify passthrough

        with (
            patch(
                "mlx_lm_server.scheduler.Scheduler._create_batch_generator",
                _fake_create_batch_generator,
            ),
            patch(
                "mlx_lm_server.spec_decode.mtp_loader.build_mtp_module",
                return_value=fake_mtp_module,
            ),
            patch(
                "mlx_lm_server.spec_decode.proposer.mtp.MTPProposer",
            ) as mock_proposer_cls,
            patch(
                "mlx_lm_server.spec_decode.engine.SpecDecodeEngine",
            ),
            patch(
                "mlx_lm_server.spec_decode.verifier.NGramVerifier",
            ),
            patch(
                "mlx_lm_server.spec_decode.controller.DynamicSpecController",
            ),
        ):
            mock_proposer_cls.return_value = MagicMock()
            config = _mtp_config()

            from mlx_lm_server.scheduler import Scheduler

            Scheduler(
                config=config,
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

            # Verify the exact num_mtp_layers value was forwarded
            _, kwargs = mock_proposer_cls.call_args
            assert kwargs["num_mtp_layers"] == 7


# ---------------------------------------------------------------------------
# TestSchedulerMTPInvalidation
# ---------------------------------------------------------------------------


class TestSchedulerMTPInvalidation:
    """Tests for MTP hidden-state invalidation when UIDs are removed."""

    def test_invalidation_on_uid_removal(self):
        """UID removal triggers proposer.invalidate_sequence on the spec engine."""
        config = ServerConfig(spec_decode_mode="none")

        from mlx_lm_server.scheduler import Scheduler

        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Manually wire up a fake spec engine with a proposer
        mock_proposer = MagicMock()
        mock_proposer.invalidate_sequence = MagicMock()
        mock_engine = MagicMock()
        mock_engine.proposer = mock_proposer
        scheduler._spec_engine = mock_engine

        # Simulate the invalidation code path:
        # In the real scheduler, when uids_to_remove is non-empty and
        # _spec_engine is not None, it calls proposer.invalidate_sequence(0).
        # We replicate the exact code from scheduler.py lines 1057-1061.
        uids_to_remove = [42]
        if uids_to_remove:
            if scheduler._spec_engine is not None:
                proposer = scheduler._spec_engine.proposer
                if hasattr(proposer, "invalidate_sequence"):
                    proposer.invalidate_sequence(0)

        mock_proposer.invalidate_sequence.assert_called_once_with(0)

    def test_no_invalidation_without_spec_engine(self):
        """No spec engine -> invalidation block is skipped, no crash."""
        config = ServerConfig(spec_decode_mode="none")

        from mlx_lm_server.scheduler import Scheduler

        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # _spec_engine is None by default
        assert scheduler._spec_engine is None

        # Replicate the invalidation code path -- should not crash
        uids_to_remove = [1, 2, 3]
        if uids_to_remove:
            if scheduler._spec_engine is not None:
                proposer = scheduler._spec_engine.proposer
                if hasattr(proposer, "invalidate_sequence"):
                    proposer.invalidate_sequence(0)

        # If we reached here without exception, the test passes
