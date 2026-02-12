"""Tests for MTPProposer (Phase 3).

Uses mock MTPModule and SequenceState — no real model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.proposer.base import BaseProposer, ProposalResult
from mlx_lm_server.spec_decode.proposer.mtp import MTPProposer


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockSequenceState:
    request_id: str = "test"
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


def _make_seq(token_ids: list[int]) -> MockSequenceState:
    return MockSequenceState(token_ids=token_ids, output_tokens=list(token_ids))


class MockMTPModule:
    """Mock MTPModule that returns predictable outputs."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100, num_layers: int = 2):
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers

    def predict(self, depth, hidden, token_embed, mask=None, cache=None):
        """Return (hidden, logits) with predictable shapes."""
        B, S, D = hidden.shape
        new_hidden = mx.zeros((B, S, self._hidden_size))
        # Make logits where token 42 has highest score
        logits = mx.zeros((B, S, self._vocab_size))
        for b in range(B):
            for s in range(S):
                logits = logits.at[b, s, 42].add(mx.array(10.0))
        return new_hidden, logits

    def get_embed(self, token_ids):
        """Return embeddings of shape [B, S, hidden_size]."""
        B, S = token_ids.shape
        return mx.zeros((B, S, self._hidden_size))

    @property
    def num_layers(self):
        return self._num_layers


class SimpleMockMTPModule:
    """Simpler mock — all zeros (argmax -> token 0)."""

    def __init__(self, hidden_size=64, vocab_size=100, num_layers=1):
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers

    def predict(self, depth, hidden, token_embed, mask=None, cache=None):
        B, S, D = hidden.shape
        return mx.zeros((B, S, self._hidden_size)), mx.zeros((B, S, self._vocab_size))

    def get_embed(self, token_ids):
        B, S = token_ids.shape
        return mx.zeros((B, S, self._hidden_size))

    @property
    def num_layers(self):
        return self._num_layers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMTPProposerInit:
    def test_is_base_proposer_subclass(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        assert isinstance(p, BaseProposer)

    def test_needs_draft_probs_false(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        assert p.needs_draft_probs is False

    def test_requires_gpu_true(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        assert p.requires_gpu is True


class TestMTPProposerGuards:
    def test_propose_returns_none_without_hidden(self):
        """First step — no cached hidden -> returns None."""
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=1)
        assert result is None

    def test_propose_empty_sequences_returns_none(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        result = p.propose([], k=1)
        assert result is None

    def test_propose_k_zero_returns_none(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=0)
        assert result is None


class TestMTPProposerPropose:
    def test_propose_shape_single_seq(self):
        """Single sequence with k=1 produces [1, 1] draft tokens."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=1)

        assert result is not None
        assert isinstance(result, ProposalResult)
        assert result.draft_tokens.shape == (1, 1)
        assert result.draft_probs is None
        assert result.proposal_lens.shape == (1,)
        assert int(result.proposal_lens[0]) == 1

    def test_propose_clamps_k_to_num_layers(self):
        """k > num_mtp_layers is clamped to num_mtp_layers."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=5)

        assert result is not None
        # k clamped to 1 (num_mtp_layers=1)
        assert result.draft_tokens.shape == (1, 1)
        assert int(result.proposal_lens[0]) == 1

    def test_propose_multiple_layers(self):
        """With 2 MTP layers and k=2, produces [1, 2] draft tokens."""
        mock_mtp = SimpleMockMTPModule(num_layers=2)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=2)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=2)

        assert result is not None
        assert result.draft_tokens.shape == (1, 2)
        assert int(result.proposal_lens[0]) == 2

    def test_propose_batch(self):
        """Batch of 2 sequences produces [2, k] shaped output."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((2, 1, 64)))
        seq1 = _make_seq([1, 2, 3])
        seq2 = _make_seq([4, 5, 6, 7])
        result = p.propose([seq1, seq2], k=1)

        assert result is not None
        assert result.draft_tokens.shape == (2, 1)
        assert result.proposal_lens.shape == (2,)
        assert int(result.proposal_lens[0]) == 1
        assert int(result.proposal_lens[1]) == 1

    def test_propose_with_mock42(self):
        """MockMTPModule produces token 42 as argmax."""
        mock_mtp = MockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=1)

        assert result is not None
        assert int(result.draft_tokens[0, 0]) == 42


class TestMTPProposerHiddenState:
    def test_set_hidden_states(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        assert p._cached_hidden is None

        hidden = mx.zeros((1, 1, 64))
        p.set_hidden_states(hidden)
        assert p._cached_hidden is not None

    def test_invalidate_sequence(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        assert p._cached_hidden is not None

        p.invalidate_sequence(0)
        assert p._cached_hidden is None

    def test_propose_after_invalidation_returns_none(self):
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))
        p.invalidate_sequence(0)

        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=1)
        assert result is None


class TestMTPProposerBatchGuard:
    """Tests for batch size mismatch guard in propose()."""

    def test_batch_size_mismatch_returns_none(self):
        """cached_hidden [2,1,D] + 3 sequences → None."""
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((2, 1, 64)))
        seqs = [_make_seq([1, 2]), _make_seq([3, 4]), _make_seq([5, 6])]
        result = p.propose(seqs, k=1)
        assert result is None
        # Cache should be cleared
        assert p._cached_hidden is None

    def test_batch_size_match_succeeds(self):
        """cached_hidden [2,1,D] + 2 sequences → ProposalResult."""
        mock_mtp = SimpleMockMTPModule()
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((2, 1, 64)))
        seqs = [_make_seq([1, 2]), _make_seq([3, 4])]
        result = p.propose(seqs, k=1)
        assert result is not None
        assert result.draft_tokens.shape == (2, 1)
