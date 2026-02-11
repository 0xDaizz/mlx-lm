"""Tests for DraftModelProposer (Phase 2).

Group A: Unit tests with mocked model (no downloads).
Group B: Real model tests (skipped if model not present).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.config import SpecDecodeConfig
from mlx_lm_server.spec_decode.proposer.base import (
    BaseProposer,
    ProposalResult,
    create_proposer,
)
from mlx_lm_server.spec_decode.proposer.draft_model import DraftModelProposer


# ---------------------------------------------------------------------------
# Lightweight mock sequence state
# ---------------------------------------------------------------------------


@dataclass
class MockSequenceState:
    request_id: str = "test"
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


def _make_seq(token_ids: list[int]) -> MockSequenceState:
    return MockSequenceState(token_ids=token_ids, output_tokens=list(token_ids))


# ---------------------------------------------------------------------------
# Mock draft model that returns fixed logits
# ---------------------------------------------------------------------------


class MockDraftModel:
    """Minimal mock that mimics model(input, cache=...) -> logits."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def __call__(self, x, cache=None):
        B, S = x.shape
        # Return logits where token 42 always has the highest score
        logits = mx.zeros((B, S, self.vocab_size))
        # Set token 42 to high value
        indices = mx.array([42])
        for b in range(B):
            for s in range(S):
                logits = logits.at[b, s, 42].add(mx.array(10.0))
        return logits


class SimpleMockDraftModel:
    """Even simpler mock -- always returns zeros (argmax -> token 0)."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def __call__(self, x, cache=None):
        B, S = x.shape
        return mx.zeros((B, S, self.vocab_size))


# ---------------------------------------------------------------------------
# Group A: Unit tests (no real model needed)
# ---------------------------------------------------------------------------


class TestDraftModelProposerInit:
    """Constructor and property tests."""

    def test_init_defaults(self) -> None:
        """Verify default model_path, context_len=128, model=None."""
        p = DraftModelProposer(model_path="some/model")
        assert p.model_path == "some/model"
        assert p.context_len == 128
        assert p.model is None
        assert p.tokenizer is None
        assert p._loaded is False

    def test_init_context_clamp(self) -> None:
        """context_len=1000 should be clamped to 512."""
        p = DraftModelProposer(model_path="x", context_len=1000)
        assert p.context_len == 512

    def test_init_context_within_range(self) -> None:
        """context_len=256 should be kept as-is."""
        p = DraftModelProposer(model_path="x", context_len=256)
        assert p.context_len == 256

    def test_needs_draft_probs_false(self) -> None:
        """Phase 2: needs_draft_probs should be False."""
        p = DraftModelProposer(model_path="x")
        assert p.needs_draft_probs is False

    def test_requires_gpu_true(self) -> None:
        """Draft model requires GPU."""
        p = DraftModelProposer(model_path="x")
        assert p.requires_gpu is True

    def test_is_base_proposer_subclass(self) -> None:
        """DraftModelProposer inherits from BaseProposer."""
        p = DraftModelProposer(model_path="x")
        assert isinstance(p, BaseProposer)


class TestDraftModelProposerGuards:
    """Propose() guard conditions."""

    def test_not_loaded_returns_none(self) -> None:
        """propose() before load() returns None."""
        p = DraftModelProposer(model_path="x")
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=3)
        assert result is None

    def test_empty_sequences_returns_none(self) -> None:
        """propose() with empty sequence list returns None."""
        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel()
        result = p.propose([], k=3)
        assert result is None

    def test_k_zero_returns_none(self) -> None:
        """propose() with k=0 returns None."""
        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel()
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=0)
        assert result is None

    def test_k_negative_returns_none(self) -> None:
        """propose() with k<0 returns None."""
        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel()
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=-1)
        assert result is None


class TestDraftModelProposerLoad:
    """load() method tests with mocks."""

    @patch("mlx_lm.utils.load")
    def test_vocab_mismatch_raises(self, mock_load) -> None:
        """Mismatched vocab_size should raise ValueError."""
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.vocab_size = 100
        mock_load.return_value = (mock_model, mock_tok)

        target_tok = MagicMock()
        target_tok.vocab_size = 200

        p = DraftModelProposer(model_path="x")
        with pytest.raises(ValueError, match="Draft vocab 100 != target vocab 200"):
            p.load(target_tokenizer=target_tok)

    @patch("mlx_lm.utils.load")
    def test_eos_mismatch_warns(self, mock_load, caplog) -> None:
        """Mismatched eos_token_id should log warning but not raise."""
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.vocab_size = 100
        mock_tok.eos_token_id = 1
        mock_load.return_value = (mock_model, mock_tok)

        target_tok = MagicMock()
        target_tok.vocab_size = 100
        target_tok.eos_token_id = 2

        p = DraftModelProposer(model_path="x")
        with caplog.at_level(logging.WARNING):
            p.load(target_tokenizer=target_tok)
        assert p._loaded is True
        assert "eos_token_id" in caplog.text

    @patch("mlx_lm.utils.load")
    def test_load_sets_loaded_flag(self, mock_load) -> None:
        """load() should set _loaded=True and populate model/tokenizer."""
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_load.return_value = (mock_model, mock_tok)

        p = DraftModelProposer(model_path="x")
        p.load()
        assert p._loaded is True
        assert p.model is mock_model
        assert p.tokenizer is mock_tok

    @patch("mlx_lm.utils.load")
    def test_vocab_match_no_error(self, mock_load) -> None:
        """Matching vocab_size should not raise."""
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.vocab_size = 100
        mock_tok.eos_token_id = 1
        mock_load.return_value = (mock_model, mock_tok)

        target_tok = MagicMock()
        target_tok.vocab_size = 100
        target_tok.eos_token_id = 1

        p = DraftModelProposer(model_path="x")
        p.load(target_tokenizer=target_tok)
        assert p._loaded is True


class TestDraftModelProposerPropose:
    """Propose with mocked model (no real model download)."""

    @patch("mlx_lm.models.cache.make_prompt_cache")
    def test_proposal_shape_with_mock(self, mock_make_cache) -> None:
        """Mock model returns zeros -> argmax is 0 for all positions."""
        mock_make_cache.return_value = []  # Empty cache list

        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel(vocab_size=100)

        seq = _make_seq([10, 20, 30])
        result = p.propose([seq], k=3)

        assert result is not None
        assert isinstance(result, ProposalResult)
        assert result.draft_tokens.shape == (1, 3)
        assert result.draft_probs is None
        assert result.proposal_lens.shape == (1,)
        assert int(result.proposal_lens[0]) == 3
        # SimpleMockDraftModel returns all zeros -> argmax = 0
        assert result.draft_tokens.tolist()[0] == [0, 0, 0]

    @patch("mlx_lm.models.cache.make_prompt_cache")
    def test_proposal_batch_shape(self, mock_make_cache) -> None:
        """Batch of 2 sequences produces [2, k] shaped output."""
        mock_make_cache.return_value = []

        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel(vocab_size=100)

        seq1 = _make_seq([1, 2, 3])
        seq2 = _make_seq([4, 5, 6, 7])
        result = p.propose([seq1, seq2], k=2)

        assert result is not None
        assert result.draft_tokens.shape == (2, 2)
        assert result.proposal_lens.shape == (2,)
        assert int(result.proposal_lens[0]) == 2
        assert int(result.proposal_lens[1]) == 2

    @patch("mlx_lm.models.cache.make_prompt_cache")
    def test_empty_token_ids_gets_zero_proposals(self, mock_make_cache) -> None:
        """Sequence with empty token_ids gets proposal_lens=0."""
        mock_make_cache.return_value = []

        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel(vocab_size=100)

        seq_empty = MockSequenceState(token_ids=[], output_tokens=[])
        seq_ok = _make_seq([1, 2, 3])
        result = p.propose([seq_empty, seq_ok], k=2)

        assert result is not None
        lens = result.proposal_lens.tolist()
        assert lens[0] == 0
        assert lens[1] == 2

    @patch("mlx_lm.models.cache.make_prompt_cache")
    def test_single_token_context(self, mock_make_cache) -> None:
        """Single-token context should skip prefill and still work."""
        mock_make_cache.return_value = []

        p = DraftModelProposer(model_path="x")
        p._loaded = True
        p.model = SimpleMockDraftModel(vocab_size=50)

        seq = _make_seq([42])
        result = p.propose([seq], k=2)

        assert result is not None
        assert result.draft_tokens.shape == (1, 2)
        assert int(result.proposal_lens[0]) == 2


# ---------------------------------------------------------------------------
# Config + factory tests
# ---------------------------------------------------------------------------


class TestSpecDecodeConfigDraftContextLen:
    """Tests for draft_context_len config field."""

    def test_draft_context_len_default(self) -> None:
        """Default draft_context_len is 128."""
        config = SpecDecodeConfig()
        assert config.draft_context_len == 128

    def test_draft_context_len_validation_too_low(self) -> None:
        """draft_context_len=0 should raise ValueError."""
        config = SpecDecodeConfig(draft_context_len=0)
        with pytest.raises(ValueError, match="draft_context_len must be in"):
            config.validate()

    def test_draft_context_len_validation_too_high(self) -> None:
        """draft_context_len=513 should raise ValueError."""
        config = SpecDecodeConfig(draft_context_len=513)
        with pytest.raises(ValueError, match="draft_context_len must be in"):
            config.validate()

    def test_draft_context_len_boundary_valid(self) -> None:
        """draft_context_len=1 and 512 should pass validation."""
        config = SpecDecodeConfig(draft_context_len=1)
        config.validate()  # Should not raise

        config = SpecDecodeConfig(draft_context_len=512)
        config.validate()  # Should not raise


class TestCreateProposerDraft:
    """Factory function tests for draft mode."""

    def test_create_proposer_draft_mode(self) -> None:
        """create_proposer with mode='draft' returns DraftModelProposer."""
        config = SpecDecodeConfig(mode="draft", draft_model_path="some/model")
        proposer = create_proposer(config)
        assert isinstance(proposer, DraftModelProposer)
        assert proposer.model_path == "some/model"

    def test_create_proposer_passes_context_len(self) -> None:
        """create_proposer passes draft_context_len to DraftModelProposer."""
        config = SpecDecodeConfig(
            mode="draft",
            draft_model_path="some/model",
            draft_context_len=256,
        )
        proposer = create_proposer(config)
        assert isinstance(proposer, DraftModelProposer)
        assert proposer.context_len == 256

    def test_create_proposer_ngram_unchanged(self) -> None:
        """create_proposer with mode='ngram' still returns NGramProposer."""
        from mlx_lm_server.spec_decode.proposer.ngram import NGramProposer

        config = SpecDecodeConfig(mode="ngram")
        proposer = create_proposer(config)
        assert isinstance(proposer, NGramProposer)

    def test_create_proposer_none_mode(self) -> None:
        """create_proposer with mode='none' returns None."""
        config = SpecDecodeConfig(mode="none")
        proposer = create_proposer(config)
        assert proposer is None
