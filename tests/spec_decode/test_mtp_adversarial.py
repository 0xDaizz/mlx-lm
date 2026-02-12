"""Adversarial / edge-case tests for MTPProposer.

Covers empty inputs, NaN propagation, invalidation races,
bootstrap flow, and uniform proposal lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.proposer.base import ProposalResult
from mlx_lm_server.spec_decode.proposer.mtp import MTPProposer


# ---------------------------------------------------------------------------
# Mock helpers (reused patterns from test_mtp_proposer.py)
# ---------------------------------------------------------------------------


@dataclass
class MockSequenceState:
    request_id: str = "test"
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


def _make_seq(token_ids: list[int]) -> MockSequenceState:
    return MockSequenceState(token_ids=token_ids, output_tokens=list(token_ids))


class SimpleMockMTPModule:
    """Simpler mock -- all zeros (argmax -> token 0)."""

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


class MockMTPModule:
    """Mock MTPModule that returns predictable outputs (argmax -> token 42)."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100, num_layers: int = 2):
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers

    def predict(self, depth, hidden, token_embed, mask=None, cache=None):
        B, S, D = hidden.shape
        new_hidden = mx.zeros((B, S, self._hidden_size))
        logits = mx.zeros((B, S, self._vocab_size))
        for b in range(B):
            for s in range(S):
                logits = logits.at[b, s, 42].add(mx.array(10.0))
        return new_hidden, logits

    def get_embed(self, token_ids):
        B, S = token_ids.shape
        return mx.zeros((B, S, self._hidden_size))

    @property
    def num_layers(self):
        return self._num_layers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMTPAdversarial:
    def test_empty_sequences_list(self):
        """sequences=[] with valid hidden -> returns None."""
        mock_mtp = SimpleMockMTPModule(num_layers=2)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=2)
        p.set_hidden_states(mx.zeros((1, 1, 64)))

        result = p.propose([], k=2)
        assert result is None

    def test_single_token_sequence(self):
        """token_ids=[5] (single token) -> normal operation, not a crash."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))

        seq = _make_seq([5])
        result = p.propose([seq], k=1)

        assert result is not None
        assert isinstance(result, ProposalResult)
        assert result.draft_tokens.shape == (1, 1)
        assert result.proposal_lens.shape == (1,)
        assert int(result.proposal_lens[0]) == 1

    def test_very_large_k(self):
        """k=100 with num_mtp_layers=1 -> clamped to 1 draft token."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))

        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=100)

        assert result is not None
        # k clamped to num_mtp_layers=1
        assert result.draft_tokens.shape == (1, 1)
        assert int(result.proposal_lens[0]) == 1

    def test_hidden_state_nan_propagation(self):
        """NaN hidden states -> proposer doesn't crash (may produce NaN tokens)."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)

        nan_hidden = mx.array([[[float("nan")] * 64]])  # [1, 1, 64]
        p.set_hidden_states(nan_hidden)

        seq = _make_seq([1, 2, 3])
        # Should not raise -- the mock ignores hidden content anyway
        result = p.propose([seq], k=1)
        assert result is not None
        assert result.draft_tokens.shape == (1, 1)

    def test_concurrent_set_hidden_invalidate(self):
        """set_hidden_states then invalidate_sequence -> cached_hidden is None."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)

        p.set_hidden_states(mx.zeros((1, 1, 64)))
        assert p._cached_hidden is not None

        p.invalidate_sequence(0)
        assert p._cached_hidden is None

        # propose after invalidation returns None
        seq = _make_seq([1, 2, 3])
        result = p.propose([seq], k=1)
        assert result is None

    def test_repeated_propose_same_hidden(self):
        """Calling propose twice with same cached hidden -> both return results."""
        mock_mtp = MockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((1, 1, 64)))

        seq = _make_seq([1, 2, 3])

        result1 = p.propose([seq], k=1)
        assert result1 is not None
        assert int(result1.draft_tokens[0, 0]) == 42

        result2 = p.propose([seq], k=1)
        assert result2 is not None
        assert int(result2.draft_tokens[0, 0]) == 42

    def test_bootstrap_then_normal_flow(self):
        """None hidden -> set_hidden -> propose -> ProposalResult lifecycle."""
        mock_mtp = MockMTPModule(num_layers=2)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=2)

        seq = _make_seq([10, 20, 30])

        # Step 1: no hidden yet -> None (bootstrap)
        result = p.propose([seq], k=2)
        assert result is None

        # Step 2: set hidden from target forward
        p.set_hidden_states(mx.zeros((1, 1, 64)))

        # Step 3: now propose works
        result = p.propose([seq], k=2)
        assert result is not None
        assert isinstance(result, ProposalResult)
        assert result.draft_tokens.shape == (1, 2)
        assert result.draft_probs is None  # v1: greedy
        assert int(result.proposal_lens[0]) == 2

    def test_all_sequences_finish_simultaneously(self):
        """Empty batch after all sequences finish -> None."""
        mock_mtp = SimpleMockMTPModule(num_layers=1)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=1)
        p.set_hidden_states(mx.zeros((2, 1, 64)))

        # Simulate: all sequences done, empty list passed
        result = p.propose([], k=1)
        assert result is None

    def test_mixed_proposal_lens_zero_and_nonzero(self):
        """MTP always produces uniform proposal_lens (all equal to clamped k).

        Unlike n-gram proposer which may have mixed lens, MTP produces
        the same number of draft tokens for every sequence in the batch.
        """
        mock_mtp = SimpleMockMTPModule(num_layers=2)
        p = MTPProposer(mtp_module=mock_mtp, num_mtp_layers=2)
        p.set_hidden_states(mx.zeros((3, 1, 64)))

        seq1 = _make_seq([1, 2])
        seq2 = _make_seq([3, 4, 5])
        seq3 = _make_seq([6])

        result = p.propose([seq1, seq2, seq3], k=2)

        assert result is not None
        assert result.draft_tokens.shape == (3, 2)
        assert result.proposal_lens.shape == (3,)
        # All proposal lengths are uniform
        for i in range(3):
            assert int(result.proposal_lens[i]) == 2
