"""Tests for SpecDecodeConfig, ProposalResult, BaseProposer, and create_proposer."""

from __future__ import annotations

import pytest
import mlx.core as mx

from mlx_lm_server.spec_decode.config import SpecDecodeConfig
from mlx_lm_server.spec_decode.proposer.base import (
    BaseProposer,
    ProposalResult,
    SpecResponse,
    create_proposer,
)


# ---------------------------------------------------------------------------
# P0.2: SpecDecodeConfig validation tests
# ---------------------------------------------------------------------------


class TestSpecDecodeConfig:
    """Tests for SpecDecodeConfig dataclass validation."""

    def test_default_config_is_valid(self) -> None:
        """Default config (mode='none') should pass validation."""
        config = SpecDecodeConfig()
        config.validate()  # Should not raise

    def test_ngram_mode_no_errors(self) -> None:
        """N-gram mode with defaults should pass validation."""
        config = SpecDecodeConfig(mode="ngram")
        config.validate()  # Should not raise

    def test_draft_mode_requires_model_path(self) -> None:
        """Draft mode without a model path should raise ValueError."""
        config = SpecDecodeConfig(mode="draft")
        with pytest.raises(ValueError, match="--draft-model-path is required"):
            config.validate()

    def test_draft_mode_with_model_path_valid(self) -> None:
        """Draft mode with a model path should pass validation."""
        config = SpecDecodeConfig(mode="draft", draft_model_path="some/model")
        config.validate()  # Should not raise

    def test_invalid_num_speculative_tokens_too_low(self) -> None:
        """num_speculative_tokens < 1 should raise ValueError."""
        config = SpecDecodeConfig(num_speculative_tokens=0)
        with pytest.raises(ValueError, match="num_speculative_tokens must be >= 1"):
            config.validate()

    def test_invalid_num_speculative_tokens_too_high(self) -> None:
        """num_speculative_tokens > 20 should raise ValueError."""
        config = SpecDecodeConfig(num_speculative_tokens=21)
        with pytest.raises(ValueError, match="num_speculative_tokens must be <= 20"):
            config.validate()

    def test_ngram_max_less_than_min(self) -> None:
        """ngram_max < ngram_min should raise ValueError."""
        config = SpecDecodeConfig(ngram_max=2, ngram_min=3)
        with pytest.raises(ValueError, match="ngram_max.*must be >= ngram_min"):
            config.validate()

    def test_acceptance_rate_bounds(self) -> None:
        """acceptance_rate_threshold outside [0, 1] should raise ValueError."""
        config = SpecDecodeConfig(acceptance_rate_threshold=-0.1)
        with pytest.raises(ValueError, match="acceptance_rate_threshold must be in"):
            config.validate()

        config = SpecDecodeConfig(acceptance_rate_threshold=1.1)
        with pytest.raises(ValueError, match="acceptance_rate_threshold must be in"):
            config.validate()

    def test_ema_alpha_bounds(self) -> None:
        """acceptance_rate_ema_alpha outside (0, 1] should raise ValueError."""
        config = SpecDecodeConfig(acceptance_rate_ema_alpha=0.0)
        with pytest.raises(ValueError, match="acceptance_rate_ema_alpha must be in"):
            config.validate()

        config = SpecDecodeConfig(acceptance_rate_ema_alpha=1.1)
        with pytest.raises(ValueError, match="acceptance_rate_ema_alpha must be in"):
            config.validate()

    def test_disable_by_batch_size_negative(self) -> None:
        """disable_by_batch_size < 0 should raise ValueError."""
        config = SpecDecodeConfig(disable_by_batch_size=-1)
        with pytest.raises(ValueError, match="disable_by_batch_size must be >= 0"):
            config.validate()


# ---------------------------------------------------------------------------
# P1.1: ProposalResult, SpecResponse, BaseProposer, create_proposer tests
# ---------------------------------------------------------------------------


class TestProposalResult:
    """Tests for ProposalResult dataclass."""

    def test_proposal_result_dataclass(self) -> None:
        """ProposalResult should store draft_tokens, draft_probs, proposal_lens."""
        draft_tokens = mx.array([[1, 2, 3], [4, 5, 0]], dtype=mx.int32)
        proposal_lens = mx.array([3, 2], dtype=mx.int32)

        result = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=proposal_lens,
        )

        assert result.draft_tokens.shape == (2, 3)
        assert result.draft_probs is None
        assert result.proposal_lens.shape == (2,)
        assert int(result.proposal_lens[0]) == 3
        assert int(result.proposal_lens[1]) == 2

    def test_proposal_result_with_probs(self) -> None:
        """ProposalResult can include draft_probs for draft model mode."""
        draft_tokens = mx.array([[1, 2]], dtype=mx.int32)
        draft_probs = mx.zeros((1, 2, 100), dtype=mx.float32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        result = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            proposal_lens=proposal_lens,
        )

        assert result.draft_probs is not None
        assert result.draft_probs.shape == (1, 2, 100)


class TestSpecResponse:
    """Tests for SpecResponse dataclass."""

    def test_spec_response_fields(self) -> None:
        """SpecResponse should store all required fields."""
        resp = SpecResponse(
            uid=42,
            tokens=[10, 20, 30],
            logprobs=[mx.array([0.1]), mx.array([0.2]), mx.array([0.3])],
            finish_reason=None,
            prompt_cache=None,
            num_drafted=5,
            num_accepted=2,
        )

        assert resp.uid == 42
        assert resp.tokens == [10, 20, 30]
        assert len(resp.logprobs) == 3
        assert resp.finish_reason is None
        assert resp.num_drafted == 5
        assert resp.num_accepted == 2


class TestCreateProposer:
    """Tests for create_proposer factory function."""

    def test_create_proposer_none_returns_none(self) -> None:
        """mode='none' should return None."""
        config = SpecDecodeConfig(mode="none")
        proposer = create_proposer(config)
        assert proposer is None

    def test_create_proposer_ngram_returns_instance(self) -> None:
        """mode='ngram' should return an NGramProposer instance."""
        from mlx_lm_server.spec_decode.proposer.ngram import NGramProposer

        config = SpecDecodeConfig(mode="ngram", ngram_max=3, ngram_min=2)
        proposer = create_proposer(config)
        assert isinstance(proposer, NGramProposer)
        assert isinstance(proposer, BaseProposer)
        assert proposer.ngram_max == 3
        assert proposer.ngram_min == 2
        assert proposer.needs_draft_probs is False
        assert proposer.requires_gpu is False

    def test_create_proposer_unknown_mode_raises(self) -> None:
        """Unknown mode should raise ValueError."""
        config = SpecDecodeConfig()
        # Force an invalid mode by bypassing Literal type checking
        object.__setattr__(config, "mode", "unknown_mode")
        with pytest.raises(ValueError, match="Unknown spec decode mode"):
            create_proposer(config)
