"""Tests for DynamicSpecController and SpecDecodeStats."""

import pytest

from mlx_lm_server.spec_decode.config import SpecDecodeConfig
from mlx_lm_server.spec_decode.controller import (
    DynamicSpecController,
    SpecDecodeStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> SpecDecodeConfig:
    """Config with ngram mode and default dynamic settings."""
    return SpecDecodeConfig(
        mode="ngram",
        num_speculative_tokens=5,
        disable_by_batch_size=8,
        dynamic_enabled=True,
        acceptance_rate_threshold=0.3,
        acceptance_rate_ema_alpha=0.1,
        adaptive_k=True,
    )


@pytest.fixture
def controller(default_config: SpecDecodeConfig) -> DynamicSpecController:
    return DynamicSpecController(default_config)


# ---------------------------------------------------------------------------
# SpecDecodeStats
# ---------------------------------------------------------------------------

class TestSpecDecodeStats:
    def test_initial_values(self):
        stats = SpecDecodeStats()
        assert stats.total_proposed == 0
        assert stats.total_accepted == 0
        assert stats.total_steps == 0
        assert stats.total_bonus_tokens == 0
        assert stats.total_fallback_steps == 0

    def test_acceptance_rate_zero_proposed(self):
        stats = SpecDecodeStats()
        assert stats.acceptance_rate == 0.0

    def test_acceptance_rate_calculation(self):
        stats = SpecDecodeStats(total_proposed=10, total_accepted=7)
        assert stats.acceptance_rate == pytest.approx(0.7)

    def test_avg_tokens_per_step_no_steps(self):
        stats = SpecDecodeStats()
        assert stats.avg_tokens_per_step == 1.0

    def test_avg_tokens_per_step_calculation(self):
        # 3 steps, 10 accepted, 2 bonus
        # total_tokens = 10 + 2 + 3 (correction tokens) = 15
        # avg = 15 / 3 = 5.0
        stats = SpecDecodeStats(
            total_steps=3,
            total_accepted=10,
            total_bonus_tokens=2,
        )
        assert stats.avg_tokens_per_step == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# DynamicSpecController — Initialization
# ---------------------------------------------------------------------------

class TestControllerInit:
    def test_initial_ema_optimistic(self, controller: DynamicSpecController):
        """EMA starts at 0.7 (optimistic initial value)."""
        assert controller.acceptance_rate_ema == pytest.approx(0.7)

    def test_initial_stats_empty(self, controller: DynamicSpecController):
        assert controller.stats.total_steps == 0
        assert controller.stats.total_proposed == 0

    def test_recent_rates_empty(self, controller: DynamicSpecController):
        assert controller._recent_rates == []


# ---------------------------------------------------------------------------
# should_speculate
# ---------------------------------------------------------------------------

class TestShouldSpeculate:
    def test_should_speculate_mode_none(self):
        """mode=none always returns False regardless of other settings."""
        config = SpecDecodeConfig(mode="none")
        ctrl = DynamicSpecController(config)
        assert ctrl.should_speculate(1) is False
        assert ctrl.should_speculate(0) is False

    def test_should_speculate_batch_threshold(self, default_config: SpecDecodeConfig):
        """batch >= threshold returns False."""
        ctrl = DynamicSpecController(default_config)
        # threshold is 8
        assert ctrl.should_speculate(8) is False
        assert ctrl.should_speculate(10) is False

    def test_should_speculate_under_batch_threshold(self, controller: DynamicSpecController):
        """batch < threshold returns True (with good EMA)."""
        assert controller.should_speculate(1) is True
        assert controller.should_speculate(7) is True

    def test_should_speculate_batch_threshold_zero(self):
        """disable_by_batch_size=0 never disables based on batch."""
        config = SpecDecodeConfig(
            mode="ngram",
            disable_by_batch_size=0,
            dynamic_enabled=False,
        )
        ctrl = DynamicSpecController(config)
        assert ctrl.should_speculate(100) is True
        assert ctrl.should_speculate(1000) is True

    def test_should_speculate_low_ema(self, default_config: SpecDecodeConfig):
        """EMA below threshold returns False."""
        ctrl = DynamicSpecController(default_config)
        ctrl.acceptance_rate_ema = 0.1  # Below 0.3 threshold
        assert ctrl.should_speculate(1) is False

    def test_should_speculate_dynamic_disabled(self):
        """dynamic_enabled=False always returns True (when mode != none)."""
        config = SpecDecodeConfig(
            mode="ngram",
            dynamic_enabled=False,
            disable_by_batch_size=0,
        )
        ctrl = DynamicSpecController(config)
        ctrl.acceptance_rate_ema = 0.0  # Would fail if dynamic were enabled
        assert ctrl.should_speculate(1) is True

    def test_should_speculate_at_exact_threshold(self, default_config: SpecDecodeConfig):
        """EMA exactly at threshold should speculate."""
        ctrl = DynamicSpecController(default_config)
        ctrl.acceptance_rate_ema = 0.3  # Exactly at threshold
        assert ctrl.should_speculate(1) is True


# ---------------------------------------------------------------------------
# get_k
# ---------------------------------------------------------------------------

class TestGetK:
    def test_get_k_adaptive_high_ema(self, controller: DynamicSpecController):
        """EMA > 0.8 returns max k."""
        controller.acceptance_rate_ema = 0.9
        assert controller.get_k(1) == 5  # Full num_speculative_tokens

    def test_get_k_adaptive_moderate_ema(self, controller: DynamicSpecController):
        """EMA 0.5-0.8 returns k-2."""
        controller.acceptance_rate_ema = 0.65
        assert controller.get_k(1) == 3  # 5 - 2 = 3

    def test_get_k_adaptive_low_ema(self, controller: DynamicSpecController):
        """EMA 0.3-0.5 returns 1."""
        controller.acceptance_rate_ema = 0.4
        assert controller.get_k(1) == 1

    def test_get_k_adaptive_very_low_ema(self, controller: DynamicSpecController):
        """EMA <= 0.3 returns 0."""
        controller.acceptance_rate_ema = 0.25
        assert controller.get_k(1) == 0

    def test_get_k_no_adaptive(self):
        """adaptive_k=False returns full k regardless of EMA."""
        config = SpecDecodeConfig(
            mode="ngram",
            num_speculative_tokens=5,
            adaptive_k=False,
            disable_by_batch_size=0,
        )
        ctrl = DynamicSpecController(config)
        ctrl.acceptance_rate_ema = 0.4  # Would return 1 if adaptive
        assert ctrl.get_k(1) == 5

    def test_get_k_zero_when_should_not_speculate(self):
        """Returns 0 when should_speculate is False."""
        config = SpecDecodeConfig(mode="none")
        ctrl = DynamicSpecController(config)
        assert ctrl.get_k(1) == 0

    def test_get_k_moderate_min_clamp(self):
        """k-2 never goes below 1 in moderate band."""
        config = SpecDecodeConfig(
            mode="ngram",
            num_speculative_tokens=2,  # k-2 = 0, clamped to 1
            adaptive_k=True,
            disable_by_batch_size=0,
        )
        ctrl = DynamicSpecController(config)
        ctrl.acceptance_rate_ema = 0.65
        assert ctrl.get_k(1) == 1  # max(1, 2-2) = 1


# ---------------------------------------------------------------------------
# get_verification_mode
# ---------------------------------------------------------------------------

class TestGetVerificationMode:
    def test_get_verification_mode_greedy(self, controller: DynamicSpecController):
        """temperature=0 returns 'greedy'."""
        assert controller.get_verification_mode(0.0) == "greedy"

    def test_get_verification_mode_threshold(self, controller: DynamicSpecController):
        """temperature>0 returns 'threshold'."""
        assert controller.get_verification_mode(0.7) == "threshold"

    def test_get_verification_mode_various_temps(self, controller: DynamicSpecController):
        """Test several temperature values."""
        assert controller.get_verification_mode(0.0) == "greedy"
        assert controller.get_verification_mode(0.1) == "threshold"
        assert controller.get_verification_mode(0.5) == "threshold"
        assert controller.get_verification_mode(1.0) == "threshold"
        assert controller.get_verification_mode(2.0) == "threshold"

    def test_get_verification_mode_small_positive(self, controller: DynamicSpecController):
        """Even very small positive temp returns threshold."""
        assert controller.get_verification_mode(0.001) == "threshold"


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_ema_calculation(self, controller: DynamicSpecController):
        """Verify EMA formula: alpha * rate + (1-alpha) * prev."""
        initial_ema = controller.acceptance_rate_ema  # 0.7
        alpha = controller.config.acceptance_rate_ema_alpha  # 0.1

        # Update with 100% acceptance rate
        controller.update(num_proposed=5, num_accepted=5)

        expected = alpha * 1.0 + (1 - alpha) * initial_ema
        assert controller.acceptance_rate_ema == pytest.approx(expected)

    def test_update_zero_proposed(self, controller: DynamicSpecController):
        """No proposed tokens means EMA stays unchanged."""
        initial_ema = controller.acceptance_rate_ema
        controller.update(num_proposed=0, num_accepted=0)
        assert controller.acceptance_rate_ema == pytest.approx(initial_ema)
        assert controller.stats.total_steps == 1  # Step still counted

    def test_update_stats_increment(self, controller: DynamicSpecController):
        """Stats are correctly incremented."""
        controller.update(num_proposed=5, num_accepted=3, num_bonus=1)
        assert controller.stats.total_proposed == 5
        assert controller.stats.total_accepted == 3
        assert controller.stats.total_bonus_tokens == 1
        assert controller.stats.total_steps == 1

    def test_update_recent_rates_tracked(self, controller: DynamicSpecController):
        """Recent rates list grows with updates."""
        controller.update(num_proposed=5, num_accepted=3)
        assert len(controller._recent_rates) == 1
        assert controller._recent_rates[0] == pytest.approx(3 / 5)

    def test_update_recent_rates_capped(self, controller: DynamicSpecController):
        """Recent rates list doesn't exceed _max_recent."""
        for i in range(150):
            controller.update(num_proposed=5, num_accepted=3)
        assert len(controller._recent_rates) == controller._max_recent

    def test_update_ema_converges_down(self):
        """EMA converges toward 0 with consistent 0% acceptance."""
        config = SpecDecodeConfig(
            mode="ngram",
            acceptance_rate_ema_alpha=0.3,  # Faster convergence for test
        )
        ctrl = DynamicSpecController(config)
        for _ in range(50):
            ctrl.update(num_proposed=5, num_accepted=0)
        assert ctrl.acceptance_rate_ema < 0.01

    def test_update_ema_converges_up(self):
        """EMA converges toward 1 with consistent 100% acceptance."""
        config = SpecDecodeConfig(
            mode="ngram",
            acceptance_rate_ema_alpha=0.3,
        )
        ctrl = DynamicSpecController(config)
        ctrl.acceptance_rate_ema = 0.1  # Start low
        for _ in range(50):
            ctrl.update(num_proposed=5, num_accepted=5)
        assert ctrl.acceptance_rate_ema > 0.99


# ---------------------------------------------------------------------------
# record_fallback
# ---------------------------------------------------------------------------

class TestRecordFallback:
    def test_record_fallback_increments(self, controller: DynamicSpecController):
        """Fallback counter increments each call."""
        assert controller.stats.total_fallback_steps == 0
        controller.record_fallback()
        assert controller.stats.total_fallback_steps == 1
        controller.record_fallback()
        assert controller.stats.total_fallback_steps == 2

    def test_record_fallback_independent_of_update(self, controller: DynamicSpecController):
        """Fallback counter independent from step counter."""
        controller.record_fallback()
        controller.update(num_proposed=5, num_accepted=3)
        assert controller.stats.total_fallback_steps == 1
        assert controller.stats.total_steps == 1


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------

class TestGetMetrics:
    def test_get_metrics_structure(self, controller: DynamicSpecController):
        """All expected keys are present."""
        metrics = controller.get_metrics()
        expected_keys = {
            "spec_decode_enabled",
            "spec_decode_mode",
            "acceptance_rate_ema",
            "acceptance_rate_overall",
            "avg_tokens_per_step",
            "total_steps",
            "total_fallback_steps",
            "total_proposed",
            "total_accepted",
            "total_bonus_tokens",
            "current_k",
            "adaptive_k_current",
        }
        assert set(metrics.keys()) == expected_keys

    def test_get_metrics_initial_values(self, controller: DynamicSpecController):
        """Metrics reflect initial state correctly."""
        metrics = controller.get_metrics()
        assert metrics["spec_decode_enabled"] is True
        assert metrics["spec_decode_mode"] == "ngram"
        assert metrics["acceptance_rate_ema"] == 0.7
        assert metrics["total_steps"] == 0
        assert metrics["total_proposed"] == 0

    def test_get_metrics_after_updates(self, controller: DynamicSpecController):
        """Metrics update after controller updates."""
        controller.update(num_proposed=10, num_accepted=8, num_bonus=2)
        controller.record_fallback()
        metrics = controller.get_metrics()
        assert metrics["total_steps"] == 1
        assert metrics["total_proposed"] == 10
        assert metrics["total_accepted"] == 8
        assert metrics["total_bonus_tokens"] == 2
        assert metrics["total_fallback_steps"] == 1

    def test_get_metrics_mode_none(self):
        """Metrics show enabled=False when mode is none."""
        config = SpecDecodeConfig(mode="none")
        ctrl = DynamicSpecController(config)
        metrics = ctrl.get_metrics()
        assert metrics["spec_decode_enabled"] is False
        assert metrics["spec_decode_mode"] == "none"


# ---------------------------------------------------------------------------
# Stats accumulation
# ---------------------------------------------------------------------------

class TestStatsAccumulation:
    def test_stats_accumulation(self, controller: DynamicSpecController):
        """Multiple updates accumulate correctly."""
        controller.update(num_proposed=5, num_accepted=3, num_bonus=1)
        controller.update(num_proposed=5, num_accepted=4, num_bonus=0)
        controller.update(num_proposed=5, num_accepted=5, num_bonus=1)

        assert controller.stats.total_proposed == 15
        assert controller.stats.total_accepted == 12
        assert controller.stats.total_bonus_tokens == 2
        assert controller.stats.total_steps == 3
        assert controller.stats.acceptance_rate == pytest.approx(12 / 15)
