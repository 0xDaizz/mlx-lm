"""Dynamic speculation controller.

Adjusts speculation depth (k) and on/off state based on runtime
statistics. Prevents spec decode overhead from degrading performance
at high batch sizes or when acceptance rate is low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from mlx_lm_server.spec_decode.config import SpecDecodeConfig


@dataclass
class SpecDecodeStats:
    """Cumulative statistics for speculative decoding.

    Tracked across all steps, all sequences. Reset only on server restart.
    """

    total_proposed: int = 0    # Total draft tokens proposed
    total_accepted: int = 0    # Total draft tokens accepted
    total_steps: int = 0       # Total spec decode steps executed
    total_bonus_tokens: int = 0  # Total bonus tokens (all k accepted)
    total_fallback_steps: int = 0  # Steps where spec decode was skipped

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all steps."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def avg_tokens_per_step(self) -> float:
        """Average tokens generated per spec decode step.

        1.0 = no benefit from spec decode.
        k+1 = maximum possible (all drafts accepted + bonus).
        """
        if self.total_steps == 0:
            return 1.0
        total_tokens = self.total_accepted + self.total_bonus_tokens
        # Add 1 per step for the correction/bonus token
        total_tokens += self.total_steps
        return total_tokens / self.total_steps


class DynamicSpecController:
    """Controls speculation depth and activation based on runtime stats.

    Decision flow:
    1. batch_size >= disable_by_batch_size --> spec OFF
    2. acceptance_rate_ema < threshold --> spec OFF
    3. If adaptive_k enabled, adjust k based on acceptance rate:
       - ema > 0.8 --> k = max (aggressive)
       - ema > 0.5 --> k = max - 2 (moderate)
       - ema > 0.3 --> k = 1 (conservative)
       - ema <= 0.3 --> k = 0 (OFF)

    Attributes:
        config: SpecDecodeConfig with thresholds and settings.
        acceptance_rate_ema: Exponential moving average of acceptance rate.
            Initialized to 0.7 (optimistic start).
        stats: Cumulative statistics.
    """

    def __init__(self, config: SpecDecodeConfig) -> None:
        self.config = config
        self.acceptance_rate_ema: float = 0.7  # Optimistic initial value
        self.stats = SpecDecodeStats()
        self._recent_rates: list[float] = []
        self._max_recent: int = 100

    def should_speculate(self, batch_size: int) -> bool:
        """Decide whether to use spec decode for this step.

        Args:
            batch_size: Current number of active sequences.

        Returns:
            True if spec decode should be used.
        """
        if self.config.mode == "none":
            return False

        # Batch size threshold (0 = never disable)
        if (
            self.config.disable_by_batch_size > 0
            and batch_size >= self.config.disable_by_batch_size
        ):
            return False

        # If dynamic control is disabled, always speculate
        if not self.config.dynamic_enabled:
            return True

        # Acceptance rate threshold
        return self.acceptance_rate_ema >= self.config.acceptance_rate_threshold

    def get_k(self, batch_size: int) -> int:
        """Get the speculation depth for this step.

        Args:
            batch_size: Current number of active sequences.

        Returns:
            Number of draft tokens (0 = no speculation).
        """
        if not self.should_speculate(batch_size):
            return 0

        k = self.config.num_speculative_tokens

        if not self.config.adaptive_k:
            return k

        # Adaptive k based on acceptance rate EMA
        if self.acceptance_rate_ema > 0.8:
            return k                    # Full speculation
        elif self.acceptance_rate_ema > 0.5:
            return max(1, k - 2)        # Moderate
        elif self.acceptance_rate_ema > 0.3:
            return 1                    # Conservative
        else:
            return 0                    # Off

    def get_verification_mode(self, temperature: float) -> str:
        """Auto-select verification mode based on sampling temperature.

        temperature=0 → "greedy" (lossless, identical to normal decode)
        temperature>0 → "threshold" (lossy but reasonable acceptance rate)
        """
        if temperature == 0.0:
            return "greedy"
        return "threshold"

    def update(
        self,
        num_proposed: int,
        num_accepted: int,
        num_bonus: int = 0,
    ) -> None:
        """Update statistics after a speculative decode step.

        Args:
            num_proposed: Total draft tokens proposed across batch.
            num_accepted: Total draft tokens accepted across batch.
            num_bonus: Total bonus tokens generated (full acceptance).
        """
        self.stats.total_proposed += num_proposed
        self.stats.total_accepted += num_accepted
        self.stats.total_bonus_tokens += num_bonus
        self.stats.total_steps += 1

        if num_proposed > 0:
            step_rate = num_accepted / num_proposed
            alpha = self.config.acceptance_rate_ema_alpha
            self.acceptance_rate_ema = (
                alpha * step_rate + (1 - alpha) * self.acceptance_rate_ema
            )
            self._recent_rates.append(step_rate)
            if len(self._recent_rates) > self._max_recent:
                self._recent_rates.pop(0)

    def record_fallback(self) -> None:
        """Record that spec decode was skipped for a step."""
        self.stats.total_fallback_steps += 1

    def get_metrics(self) -> Dict:
        """Return metrics dict for monitoring endpoints.

        Suitable for /v1/spec_decode/metrics or inclusion in /health.
        """
        return {
            "spec_decode_enabled": self.config.mode != "none",
            "spec_decode_mode": self.config.mode,
            "acceptance_rate_ema": round(self.acceptance_rate_ema, 4),
            "acceptance_rate_overall": round(self.stats.acceptance_rate, 4),
            "avg_tokens_per_step": round(self.stats.avg_tokens_per_step, 2),
            "total_steps": self.stats.total_steps,
            "total_fallback_steps": self.stats.total_fallback_steps,
            "total_proposed": self.stats.total_proposed,
            "total_accepted": self.stats.total_accepted,
            "total_bonus_tokens": self.stats.total_bonus_tokens,
            "current_k": self.config.num_speculative_tokens,
            "adaptive_k_current": self.get_k(1),  # k at batch_size=1
        }
