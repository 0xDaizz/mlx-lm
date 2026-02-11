"""Configuration for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SpecDecodeConfig:
    """Speculative decoding configuration.

    Controls proposer mode, speculation depth, dynamic control thresholds,
    and per-proposer settings. Can be instantiated from CLI args, config
    file, or per-request override via extra_body.

    Attributes:
        mode: Proposer type. "none" disables spec decode entirely.
        num_speculative_tokens: Number of draft tokens (k) per spec step.
            Higher k = more potential gain but higher overhead on rejection.
            Typical range: 3-7 for n-gram, 2-5 for draft model.
        disable_by_batch_size: Auto-disable spec decode when batch size
            reaches this threshold. At high batch sizes, the overhead of
            verification outweighs the benefit. Set to 0 to never disable.
        ngram_max: Maximum n-gram size for context matching (4 = try
            4-gram, then 3-gram, ..., down to ngram_min).
        ngram_min: Minimum n-gram size. Setting to 1 allows unigram
            matching (most aggressive, lowest precision).
        ngram_prompt_lookup: If True, search the full context (prompt +
            generated tokens). If False, search only generated tokens.
        draft_model_path: HuggingFace repo or local path for draft model.
            Required when mode="draft".
        draft_model_quantize: Quantization for draft model ("4bit", "8bit",
            or None for fp16). Recommend "4bit" to minimize memory impact.
        dynamic_enabled: Enable dynamic speculation control (adaptive k
            and auto-disable based on acceptance rate).
        acceptance_rate_threshold: Minimum EMA acceptance rate to keep
            spec decode active. Below this, falls back to normal decode.
        acceptance_rate_ema_alpha: Smoothing factor for acceptance rate
            exponential moving average. Lower = more smoothing.
        adaptive_k: Adjust k dynamically based on acceptance rate EMA.
            High acceptance -> increase k, low acceptance -> decrease k.
    """

    mode: Literal["none", "ngram", "draft"] = "none"
    num_speculative_tokens: int = 5
    disable_by_batch_size: int = 8

    # N-gram settings
    ngram_max: int = 4
    ngram_min: int = 1
    ngram_prompt_lookup: bool = True

    # Draft model settings (Phase 2)
    draft_model_path: Optional[str] = None
    draft_model_quantize: Optional[str] = None
    draft_context_len: int = 128

    # Dynamic control
    dynamic_enabled: bool = True
    acceptance_rate_threshold: float = 0.3
    acceptance_rate_ema_alpha: float = 0.1
    adaptive_k: bool = True

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If required fields are missing for the selected mode,
                or if numeric values are out of valid range.
        """
        if self.mode == "draft" and not self.draft_model_path:
            raise ValueError(
                "--draft-model-path is required when --spec-decode=draft"
            )
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )
        if self.num_speculative_tokens > 20:
            raise ValueError(
                f"num_speculative_tokens must be <= 20, got {self.num_speculative_tokens}"
            )
        if self.ngram_max < self.ngram_min:
            raise ValueError(
                f"ngram_max ({self.ngram_max}) must be >= ngram_min ({self.ngram_min})"
            )
        if self.ngram_min < 1:
            raise ValueError(
                f"ngram_min must be >= 1, got {self.ngram_min}"
            )
        if not (0.0 <= self.acceptance_rate_threshold <= 1.0):
            raise ValueError(
                f"acceptance_rate_threshold must be in [0, 1], got {self.acceptance_rate_threshold}"
            )
        if not (0.0 < self.acceptance_rate_ema_alpha <= 1.0):
            raise ValueError(
                f"acceptance_rate_ema_alpha must be in (0, 1], got {self.acceptance_rate_ema_alpha}"
            )
        if self.disable_by_batch_size < 0:
            raise ValueError(
                f"disable_by_batch_size must be >= 0, got {self.disable_by_batch_size}"
            )
        if self.draft_context_len < 1 or self.draft_context_len > 512:
            raise ValueError(
                f"draft_context_len must be in [1, 512], got {self.draft_context_len}"
            )
