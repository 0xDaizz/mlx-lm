"""Adversarial tests for speculative decoding implementation.

Devil's Advocate tests that stress-test edge cases, boundary conditions,
and potential failure modes across all spec decode components.

Categories:
- NGramProposer: extreme contexts, degenerate patterns
- NGramVerifier: boundary probabilities, mixed-mode edge cases
- DynamicSpecController: rapid oscillation, EMA stability, boundary thresholds
- SpecDecodeEngine: integration edge cases, state corruption detection
- Config validation: exhaustive boundary checks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.cache_utils import uniform_trim, batch_variable_trim
from mlx_lm_server.spec_decode.config import SpecDecodeConfig
from mlx_lm_server.spec_decode.controller import DynamicSpecController, SpecDecodeStats
from mlx_lm_server.spec_decode.engine import SpecDecodeEngine
from mlx_lm_server.spec_decode.proposer.base import (
    BaseProposer,
    ProposalResult,
    SpecResponse,
    create_proposer,
)
from mlx_lm_server.spec_decode.proposer.ngram import NGramProposer
from mlx_lm_server.spec_decode.verifier import PLACEHOLDER_TOKEN_ID, NGramVerifier


# ===========================================================================
# Mock objects (reusable across test classes)
# ===========================================================================

@dataclass
class MockSequenceState:
    request_id: str = "test"
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


@dataclass
class MockSampler:
    temperature: float = 0.0


@dataclass
class MockCacheLayer:
    _offset: int = 10
    _batch_size: int = 1
    trim_calls: list[int] = field(default_factory=list)
    trim_per_seq_calls: list = field(default_factory=list)

    def __post_init__(self):
        self.left_padding = mx.zeros((self._batch_size,), dtype=mx.int32)
        self.offset = mx.array([self._offset] * self._batch_size, dtype=mx.int32)

    def trim(self, n: int) -> None:
        self._offset -= n
        self.offset = self.offset - n
        self.trim_calls.append(n)

    def trim_per_sequence(self, n) -> None:
        n = mx.minimum(n, self.left_padding + self.offset)
        self.offset = self.offset - n
        self._offset = int(mx.max(self.left_padding + self.offset).item())
        self.trim_per_seq_calls.append(n)


@dataclass
class MockBatch:
    uids: List[int] = field(default_factory=list)
    y: mx.array = field(default_factory=lambda: mx.array([0]))
    tokens: List[mx.array] = field(default_factory=list)
    num_tokens: List[int] = field(default_factory=list)
    max_tokens: List[int] = field(default_factory=list)
    cache: List[Any] = field(default_factory=list)
    samplers: List[Any] = field(default_factory=list)
    logits_processors: List[Any] = field(default_factory=list)
    logprobs: List[Any] = field(default_factory=list)


class MockBatchGenerator:
    def __init__(self, batch=None):
        self.active_batch = batch
        self.stop_tokens: set[int] = set()
        self._next_result = []

    def next(self):
        return self._next_result


class MockModel:
    def __init__(self, logits: mx.array):
        self._logits = logits

    def __call__(self, x, cache=None):
        return self._logits


class MockProposer(BaseProposer):
    def __init__(self, proposal=None):
        self._proposal = proposal

    def propose(self, sequences, k):
        return self._proposal

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False


def _make_seq(token_ids, output_tokens=None):
    return MockSequenceState(
        token_ids=token_ids,
        output_tokens=output_tokens if output_tokens is not None else list(token_ids),
    )


def _make_det_logits(batch_size, seq_len, vocab_size, argmax_tokens):
    logits = mx.full((batch_size, seq_len, vocab_size), -10.0)
    for b in range(batch_size):
        for pos in range(seq_len):
            if pos < len(argmax_tokens[b]):
                tok = argmax_tokens[b][pos]
                logits[b, pos, tok] = 10.0
    return logits


def _make_det_probs(batch_size, k_plus_one, vocab_size, argmax_tokens,
                    top_prob=0.9, base_prob=0.01):
    probs = mx.full((batch_size, k_plus_one, vocab_size), base_prob)
    for b in range(batch_size):
        for pos in range(k_plus_one):
            tok = argmax_tokens[b][pos]
            probs[b, pos, tok] = top_prob
    return probs


def _make_engine(*, batch=None, model_logits=None, proposal=None,
                 config=None, stop_tokens=None):
    cfg = config or SpecDecodeConfig(mode="ngram", num_speculative_tokens=3)
    controller = DynamicSpecController(cfg)
    verifier = NGramVerifier(mode="greedy")
    proposer = MockProposer(proposal)
    model = MockModel(
        model_logits if model_logits is not None else mx.zeros((1, 1, 100))
    )
    bg = MockBatchGenerator(batch)
    if stop_tokens is not None:
        bg.stop_tokens = stop_tokens
    engine = SpecDecodeEngine(
        model=model, batch_generator=bg, proposer=proposer,
        verifier=verifier, config=cfg, controller=controller,
    )
    return engine, bg


# ===========================================================================
# 1. NGramProposer Adversarial Tests
# ===========================================================================

class TestNGramProposerAdversarial:
    """Adversarial edge cases for NGramProposer."""

    # --- Empty and degenerate inputs ---

    def test_empty_sequence_list(self):
        """Proposer with an empty batch should return None, not crash."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        result = proposer.propose([], k=3)
        # Empty list: no sequences -> no proposals
        assert result is None

    def test_k_equals_zero(self):
        """k=0 means no draft tokens requested. Should return None or empty."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq = _make_seq([1, 2, 3, 1, 2, 3])
        result = proposer.propose([seq], k=0)
        # With k=0, the continuation from a match is capped to 0 tokens
        # so proposals should be empty
        assert result is None

    def test_all_same_token(self):
        """All same tokens: [5,5,5,5,5,5,5] -- every n-gram matches."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        ctx = [5] * 20
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=5)
        assert result is not None
        # All proposals should be token 5
        tokens = result.draft_tokens.tolist()[0]
        assert all(t == 5 for t in tokens[:int(result.proposal_lens[0])])

    def test_very_long_sequence(self):
        """1500-token context with a pattern at the end -- should work."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        # 1490 unique tokens, then a repeated 10-token pattern
        unique = list(range(100, 1590))
        pattern = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ctx = unique + pattern + [7, 8, 9, 10]  # ends with partial repeat
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=5)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        plen = int(result.proposal_lens[0])
        # continuation from pattern match: [11, 12, 13, 14, 15]
        assert tokens[:plen] == [11, 12, 13, 14, 15]

    def test_two_tokens_with_unigram(self):
        """Context [A, B] with ngram_min=1: key=(B,), search for (B) earlier.
        Only pos 0 has A, no earlier B exists, so no match."""
        proposer = NGramProposer(ngram_max=1, ngram_min=1)
        seq = _make_seq([100, 200])
        result = proposer.propose([seq], k=3)
        # (200,) never appears before current position -> no match
        assert result is None

    def test_two_tokens_with_unigram_repeat(self):
        """Context [A, A] with ngram_min=1: key=(A,) at pos 0, continuation=[]
        since pos 0 + 1 = 1 which is the current position. No valid continuation."""
        proposer = NGramProposer(ngram_max=1, ngram_min=1)
        seq = _make_seq([100, 100])
        # key=(100,), search_end=1, range(0, -1, -1) = [0]
        # match at i=0, start=1, end=min(1+3, 2)=2, context[1:2]=[100]
        # that's 1 proposal token
        result = proposer.propose([seq], k=3)
        # The match IS at pos 0, continuation is context[1:min(4,2)] = [100]
        assert result is not None
        assert int(result.proposal_lens[0]) == 1
        assert result.draft_tokens.tolist()[0][0] == 100

    def test_ngram_max_equals_ngram_min(self):
        """ngram_max == ngram_min -- only one n-gram size tried."""
        proposer = NGramProposer(ngram_max=3, ngram_min=3)
        # Context needs a 3-gram match
        ctx = [1, 2, 3, 4, 5, 1, 2, 3]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=3)
        assert result is not None
        # 3-gram key=(1,2,3) at pos 0, continuation=[4,5,1]
        tokens = result.draft_tokens.tolist()[0]
        assert tokens[:3] == [4, 5, 1]

    def test_very_large_k(self):
        """k much larger than available continuation tokens."""
        proposer = NGramProposer(ngram_max=2, ngram_min=1)
        ctx = [1, 2, 3, 1, 2]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=100)
        assert result is not None
        plen = int(result.proposal_lens[0])
        # Continuation from match at pos 0: [3, 1, 2] = 3 tokens
        assert plen == 3

    def test_match_at_very_end_of_context(self):
        """Pattern match where continuation is only 1 token (match near end)."""
        proposer = NGramProposer(ngram_max=2, ngram_min=1)
        # Context: [1, 2, 99, 1, 2]
        # 2-gram key=(1,2), match at pos 0, continuation=[99, 1, 2]
        # But also at pos 3? No, search_end = 5-2=3, so range(2,-1,-1)
        # i=2: context[2:4]=[99,1] != (1,2)
        # i=1: context[1:3]=[2,99] != (1,2)
        # i=0: context[0:2]=[1,2] == (1,2), continuation = context[2:2+5]=[99,1,2]
        ctx = [1, 2, 99, 1, 2]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=5)
        assert result is not None
        plen = int(result.proposal_lens[0])
        assert plen == 3
        tokens = result.draft_tokens.tolist()[0]
        assert tokens[:plen] == [99, 1, 2]

    def test_multiple_patterns_most_recent_wins(self):
        """Multiple n-gram matches -- most recent (rightmost) should win."""
        proposer = NGramProposer(ngram_max=2, ngram_min=2)
        # [A, B, old_continuation..., A, B, new_continuation..., A, B]
        ctx = [10, 20, 77, 78, 10, 20, 88, 89, 10, 20]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=2)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        plen = int(result.proposal_lens[0])
        # Most recent match is pos 4 (before current pos 8)
        # Wait: search_end = 10-2=8, range(7, -1, -1)
        # i=7: context[7:9]=[89, 10] != (10, 20)
        # i=6: context[6:8]=[88, 89] != (10, 20)
        # i=5: context[5:7]=[20, 88] != (10, 20)
        # i=4: context[4:6]=[10, 20] == (10, 20), continuation=context[6:8]=[88,89]
        assert tokens[:plen] == [88, 89]

    # --- Suffix index adversarial ---

    def test_suffix_index_empty_tokens(self):
        """Building suffix index on empty list."""
        index = NGramProposer.build_suffix_index([], ngram_max=4)
        assert index == {}

    def test_suffix_index_single_token(self):
        """Building suffix index on single token -- no n-grams possible."""
        index = NGramProposer.build_suffix_index([42], ngram_max=4)
        assert index == {}

    def test_suffix_index_all_same_tokens(self):
        """Suffix index with all identical tokens."""
        tokens = [5] * 10
        index = NGramProposer.build_suffix_index(tokens, ngram_max=3)
        # All n-grams should map to multiple positions
        assert (5,) in index
        assert len(index[(5,)]) == 9  # positions 0-8 (not 9, since last needs 1 more)

    def test_suffix_index_incremental_consistency(self):
        """Incremental update should match fresh build for any pattern."""
        tokens = [1, 2, 3, 4, 5]
        index = NGramProposer.build_suffix_index(tokens[:3], ngram_max=3)
        NGramProposer.update_suffix_index(index, tokens, new_token_count=2, ngram_max=3)

        fresh = NGramProposer.build_suffix_index(tokens, ngram_max=3)
        proposer = NGramProposer(ngram_max=3, ngram_min=1)
        # Both should yield same search results
        assert proposer._index_search(index, tokens, k=5) == \
            proposer._index_search(fresh, tokens, k=5)

    def test_suffix_index_with_zero_new_tokens(self):
        """update_suffix_index with new_token_count=0 should be a no-op."""
        tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        index = NGramProposer.build_suffix_index(tokens, ngram_max=4)
        original_keys = set(index.keys())
        NGramProposer.update_suffix_index(index, tokens, new_token_count=0, ngram_max=4)
        # Keys should not change
        assert set(index.keys()) == original_keys

    # --- prompt_lookup boundary ---

    def test_prompt_lookup_false_empty_output(self):
        """prompt_lookup=False with empty output_tokens."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1, prompt_lookup=False)
        seq = MockSequenceState(
            token_ids=[1, 2, 3, 1, 2, 3],
            output_tokens=[],
        )
        result = proposer.propose([seq], k=3)
        assert result is None

    def test_prompt_lookup_false_single_output_token(self):
        """prompt_lookup=False with only 1 output token -- too short."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1, prompt_lookup=False)
        seq = MockSequenceState(
            token_ids=[1, 2, 3, 99],
            output_tokens=[99],
        )
        result = proposer.propose([seq], k=3)
        assert result is None


# ===========================================================================
# 2. NGramVerifier Adversarial Tests
# ===========================================================================

class TestNGramVerifierAdversarial:
    """Adversarial edge cases for NGramVerifier."""

    def test_proposal_lens_all_zero(self):
        """All sequences have proposal_lens=0 (no proposals)."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3
        batch_size = 3

        argmax_tokens = [[5, 6, 7, 8]] * batch_size
        target_probs = _make_det_probs(batch_size, k + 1, vocab, argmax_tokens)
        draft_tokens = mx.zeros((batch_size, k), dtype=mx.int32)
        proposal_lens = mx.zeros((batch_size,), dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert accepted.shape == (batch_size, k + 1)
        # All num_accepted should be 0
        for b in range(batch_size):
            assert int(num_acc[b]) == 0
            # Position 0 should have the correction token (target argmax at pos 0)
            assert int(accepted[b, 0]) == 5

    def test_single_draft_token_accepted(self):
        """k=1 with match. Output should have 2 positions (accepted + bonus)."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 1

        argmax_tokens = [[5, 9]]
        target_probs = _make_det_probs(1, k + 1, vocab, argmax_tokens)
        draft_tokens = mx.array([[5]], dtype=mx.int32)
        proposal_lens = mx.array([1], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert accepted.shape == (1, 2)
        assert int(num_acc[0]) == 1
        assert int(accepted[0, 0]) == 5
        assert int(accepted[0, 1]) == 9  # bonus

    def test_single_draft_token_rejected(self):
        """k=1 with no match. Should get correction token."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 1

        argmax_tokens = [[5, 9]]
        target_probs = _make_det_probs(1, k + 1, vocab, argmax_tokens)
        draft_tokens = mx.array([[7]], dtype=mx.int32)  # wrong
        proposal_lens = mx.array([1], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 0
        assert int(accepted[0, 0]) == 5  # correction

    def test_threshold_exactly_at_boundary(self):
        """Draft token has probability EXACTLY at threshold."""
        v = NGramVerifier(mode="threshold", threshold=0.3)
        vocab = 10
        k = 2

        probs = mx.full((1, k + 1, vocab), 0.01)
        # Position 0: token 4 has prob exactly 0.3
        probs[0, 0, 4] = 0.3
        # Position 1: token 4 has prob 0.29 (just below)
        probs[0, 1, 4] = 0.29
        # Bonus position
        probs[0, 2, 7] = 0.9

        draft_tokens = mx.array([[4, 4]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs=probs,
                                      draft_tokens=draft_tokens,
                                      proposal_lens=proposal_lens)

        # Position 0: 0.3 >= 0.3 -> accepted
        # Position 1: 0.29 < 0.3 -> rejected (cumprod kills it)
        assert int(num_acc[0]) == 1
        assert int(accepted[0, 0]) == 4

    def test_threshold_zero_accepts_everything(self):
        """threshold=0.0 should accept all tokens (even near-zero probability)."""
        v = NGramVerifier(mode="threshold", threshold=0.0)
        vocab = 10
        k = 3

        probs = mx.full((1, k + 1, vocab), 0.001)
        # Even very low probs should be >= 0.0
        draft_tokens = mx.array([[3, 4, 5]], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)

        accepted, num_acc = v.verify(probs, draft_tokens, proposal_lens)

        # All should be accepted since any prob >= 0.0
        assert int(num_acc[0]) == 3

    def test_threshold_one_rejects_almost_everything(self):
        """threshold=1.0 should reject tokens unless prob is exactly 1.0."""
        v = NGramVerifier(mode="threshold", threshold=1.0)
        vocab = 10
        k = 2

        probs = mx.full((1, k + 1, vocab), 0.001)
        probs[0, 0, 5] = 0.99  # high but not 1.0
        probs[0, 1, 6] = 0.99
        probs[0, 2, 7] = 0.9

        draft_tokens = mx.array([[5, 6]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(probs, draft_tokens, proposal_lens)

        # 0.99 < 1.0 -> rejected
        assert int(num_acc[0]) == 0

    def test_mixed_modes_with_partial_proposal_lens(self):
        """Mixed modes where some sequences have proposal_lens=0."""
        v = NGramVerifier(mode="greedy", threshold=0.1)
        vocab = 10
        k = 3

        # Seq 0: greedy, plen=3, all match
        # Seq 1: threshold, plen=0 (no proposals)
        argmax_tokens = [[5, 6, 7, 8], [1, 2, 3, 4]]
        target_probs = _make_det_probs(2, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([
            [5, 6, 7],
            [0, 0, 0],
        ], dtype=mx.int32)
        proposal_lens = mx.array([3, 0], dtype=mx.int32)

        accepted, num_acc = v.verify(
            target_probs, draft_tokens, proposal_lens,
            modes=["greedy", "threshold"],
        )

        assert int(num_acc[0]) == 3
        assert int(num_acc[1]) == 0
        # Seq 1 should get target argmax at pos 0 = 1
        assert int(accepted[1, 0]) == 1

    def test_large_batch_verification(self):
        """Batch of 16 sequences, varying acceptance patterns."""
        v = NGramVerifier(mode="greedy")
        batch_size = 16
        vocab = 50
        k = 5

        # Each sequence has different number of matching positions
        argmax_tokens = []
        draft_data = []
        for b in range(batch_size):
            # All sequences have same argmax sequence
            argmax_tokens.append([10, 11, 12, 13, 14, 15])
            # First b tokens match, rest don't
            row = []
            for j in range(k):
                if j < (b % (k + 1)):
                    row.append(10 + j)  # match
                else:
                    row.append(99)  # mismatch
            draft_data.append(row)

        target_probs = _make_det_probs(batch_size, k + 1, vocab, argmax_tokens)
        draft_tokens = mx.array(draft_data, dtype=mx.int32)
        proposal_lens = mx.array([k] * batch_size, dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert accepted.shape == (batch_size, k + 1)
        assert num_acc.shape == (batch_size,)

        # Verify each sequence has expected acceptance count
        for b in range(batch_size):
            expected = b % (k + 1)
            assert int(num_acc[b]) == expected, \
                f"Seq {b}: expected {expected} accepted, got {int(num_acc[b])}"

    def test_draft_token_zero_is_valid(self):
        """Token ID 0 is a valid token, not padding. Verify it works correctly."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 2

        # Target argmax is token 0 at all positions
        argmax_tokens = [[0, 0, 0]]
        target_probs = _make_det_probs(1, k + 1, vocab, argmax_tokens)

        # Draft tokens are all 0 (should match)
        draft_tokens = mx.array([[0, 0]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 2
        assert int(accepted[0, 0]) == 0
        assert int(accepted[0, 1]) == 0
        assert int(accepted[0, 2]) == 0  # bonus also 0

    def test_placeholder_token_id_not_confused_with_valid_token(self):
        """Ensure PLACEHOLDER_TOKEN_ID (-1) doesn't accidentally match valid tokens."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 2

        # draft_token is -1 (PLACEHOLDER), target argmax is -1
        # This is a degenerate case but should not crash
        # Note: -1 as token ID is unusual but we should handle it
        argmax_tokens = [[0, 0, 0]]  # normal argmax
        target_probs = _make_det_probs(1, k + 1, vocab, argmax_tokens)

        # Draft with PLACEHOLDER_TOKEN_ID values -- these won't match argmax=0
        draft_tokens = mx.array([[PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
                                 dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        # PLACEHOLDER_TOKEN_ID (-1) != 0, so none accepted
        assert int(num_acc[0]) == 0


# ===========================================================================
# 3. DynamicSpecController Adversarial Tests
# ===========================================================================

class TestControllerAdversarial:
    """Adversarial tests for DynamicSpecController."""

    def test_rapid_oscillation(self):
        """Rapidly alternating between 100% and 0% acceptance.
        EMA should converge to ~0.5 over time."""
        config = SpecDecodeConfig(
            mode="ngram",
            acceptance_rate_ema_alpha=0.1,
            disable_by_batch_size=0,
        )
        ctrl = DynamicSpecController(config)

        for i in range(200):
            if i % 2 == 0:
                ctrl.update(num_proposed=10, num_accepted=10)
            else:
                ctrl.update(num_proposed=10, num_accepted=0)

        # EMA should be near 0.5 (within tolerance due to finite samples)
        assert abs(ctrl.acceptance_rate_ema - 0.5) < 0.15

    def test_ema_stability_single_outlier(self):
        """Steady 80% acceptance, then one 0% outlier, then back to 80%.
        EMA should recover quickly."""
        config = SpecDecodeConfig(
            mode="ngram",
            acceptance_rate_ema_alpha=0.1,
        )
        ctrl = DynamicSpecController(config)

        # Establish baseline at 0.8
        for _ in range(100):
            ctrl.update(num_proposed=10, num_accepted=8)
        ema_before = ctrl.acceptance_rate_ema

        # Single catastrophic step
        ctrl.update(num_proposed=10, num_accepted=0)
        ema_after_outlier = ctrl.acceptance_rate_ema

        # Recover
        for _ in range(20):
            ctrl.update(num_proposed=10, num_accepted=8)
        ema_recovered = ctrl.acceptance_rate_ema

        # After outlier, EMA should drop but not catastrophically
        assert ema_after_outlier < ema_before
        assert ema_after_outlier > 0.5  # Single outlier shouldn't tank it

        # After recovery, should be close to 0.8 again
        assert abs(ema_recovered - 0.8) < 0.1

    def test_get_k_at_exact_ema_boundaries(self):
        """Test get_k at exact EMA boundary values."""
        config = SpecDecodeConfig(
            mode="ngram",
            num_speculative_tokens=5,
            adaptive_k=True,
            disable_by_batch_size=0,
        )
        ctrl = DynamicSpecController(config)

        # At exactly 0.8 -> moderate band (0.5 < 0.8 but NOT > 0.8)
        ctrl.acceptance_rate_ema = 0.8
        assert ctrl.get_k(1) == 3  # moderate: max(1, 5-2)=3

        # At exactly 0.5 -> conservative band (0.3 < 0.5 but NOT > 0.5)
        ctrl.acceptance_rate_ema = 0.5
        assert ctrl.get_k(1) == 1  # conservative

        # At exactly 0.3 -> should still speculate (>= threshold)
        # but adaptive says 0.3 is NOT > 0.3, so returns 0
        ctrl.acceptance_rate_ema = 0.3
        k = ctrl.get_k(1)
        # 0.3 >= 0.3 (threshold) -> should_speculate=True
        # 0.3 not > 0.8, not > 0.5, not > 0.3 -> returns 0
        assert k == 0

    def test_get_verification_mode_boundary(self):
        """Test get_verification_mode at temp=0.0 vs epsilon above."""
        config = SpecDecodeConfig(mode="ngram")
        ctrl = DynamicSpecController(config)

        assert ctrl.get_verification_mode(0.0) == "greedy"
        assert ctrl.get_verification_mode(1e-10) == "threshold"
        assert ctrl.get_verification_mode(float("inf")) == "threshold"

    def test_update_with_very_large_numbers(self):
        """Large num_proposed/accepted values should not overflow."""
        config = SpecDecodeConfig(mode="ngram")
        ctrl = DynamicSpecController(config)

        ctrl.update(num_proposed=10_000_000, num_accepted=9_999_999)
        assert ctrl.stats.total_proposed == 10_000_000
        assert ctrl.stats.total_accepted == 9_999_999
        assert ctrl.stats.acceptance_rate == pytest.approx(9_999_999 / 10_000_000)

    def test_should_speculate_batch_size_exactly_at_threshold(self):
        """batch_size exactly == disable_by_batch_size -> should return False."""
        config = SpecDecodeConfig(mode="ngram", disable_by_batch_size=8)
        ctrl = DynamicSpecController(config)
        assert ctrl.should_speculate(8) is False
        assert ctrl.should_speculate(7) is True

    def test_metrics_after_many_updates(self):
        """get_metrics should work correctly after many updates."""
        config = SpecDecodeConfig(mode="ngram")
        ctrl = DynamicSpecController(config)

        for i in range(1000):
            ctrl.update(num_proposed=5, num_accepted=i % 6, num_bonus=(1 if i % 6 == 5 else 0))
            if i % 3 == 0:
                ctrl.record_fallback()

        metrics = ctrl.get_metrics()
        assert metrics["total_steps"] == 1000
        assert metrics["total_proposed"] == 5000
        assert isinstance(metrics["acceptance_rate_ema"], float)
        assert not math.isnan(metrics["acceptance_rate_ema"])
        assert not math.isinf(metrics["acceptance_rate_ema"])

    def test_acceptance_rate_ema_alpha_edge_one(self):
        """EMA alpha=1.0 means no memory -- EMA equals last step rate."""
        config = SpecDecodeConfig(
            mode="ngram",
            acceptance_rate_ema_alpha=1.0,
        )
        ctrl = DynamicSpecController(config)

        ctrl.update(num_proposed=10, num_accepted=3)
        assert ctrl.acceptance_rate_ema == pytest.approx(0.3)

        ctrl.update(num_proposed=10, num_accepted=8)
        assert ctrl.acceptance_rate_ema == pytest.approx(0.8)


# ===========================================================================
# 4. SpecDecodeEngine Adversarial Tests
# ===========================================================================

class TestEngineAdversarial:
    """Adversarial tests for SpecDecodeEngine integration."""

    def test_speculative_step_empty_sequences(self):
        """Calling speculative_step with empty sequences list."""
        batch = MockBatch(
            uids=[],
            y=mx.array([]),
            tokens=[],
            num_tokens=[],
            max_tokens=[],
            cache=[],
            samplers=[],
        )
        engine, bg = _make_engine(batch=batch)
        # Empty batch -- proposer returns None -> fallback
        bg._next_result = []
        result = engine.speculative_step([])
        # Should fallback gracefully
        assert result == []

    def test_stop_token_in_first_accepted_token(self):
        """Stop token is the very first accepted token."""
        vocab_size = 200

        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        logits = _make_det_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=[MockCacheLayer()],
            samplers=[MockSampler(0.0)],
        )

        # Stop token is 10 (first accepted token)
        engine, bg = _make_engine(
            batch=batch, model_logits=logits, proposal=proposal,
            stop_tokens={10},
        )

        seqs = [MockSequenceState("r0", token_ids=[1, 2, 3, 4, 5])]
        responses = engine.speculative_step(seqs)

        assert len(responses) == 1
        assert responses[0].finish_reason == "stop"

    def test_max_tokens_reached_during_spec(self):
        """max_tokens is reached partway through accepted tokens."""
        vocab_size = 200

        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        logits = _make_det_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[6],  # 6 existing + 4 spec = 10 >= max_tokens=8
            max_tokens=[8],
            cache=[MockCacheLayer()],
            samplers=[MockSampler(0.0)],
        )

        engine, bg = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        seqs = [MockSequenceState("r0")]
        responses = engine.speculative_step(seqs)

        assert len(responses) == 1
        assert responses[0].finish_reason == "length"

    def test_all_sequences_rejected_cache_rollback(self):
        """All sequences reject all drafts -> maximum cache rollback."""
        vocab_size = 200

        # Draft tokens don't match target
        draft_tokens = mx.array([[90, 91, 92]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        # Target: [10, 11, 12, 99] -- none match draft
        logits = _make_det_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        cache_layers = [MockCacheLayer(_offset=20)]
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler(0.0)],
        )

        engine, bg = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        seqs = [MockSequenceState("r0")]
        responses = engine.speculative_step(seqs)

        # All rejected: num_accepted=0
        # Result A: per-sequence trim, keep=1, trim=4-1=3
        assert len(cache_layers[0].trim_per_seq_calls) == 1
        assert int(cache_layers[0].trim_per_seq_calls[0][0]) == 3
        assert responses[0].num_accepted == 0
        # Should still get 1 correction token
        assert len(responses[0].tokens) == 1
        assert responses[0].tokens[0] == 10  # target argmax at pos 0

    def test_controller_records_fallback_when_no_proposals(self):
        """When proposer returns None, controller should not update stats,
        but the engine falls back. Verify stats remain unchanged."""
        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=[MockCacheLayer()],
            samplers=[MockSampler()],
        )
        engine, bg = _make_engine(batch=batch, proposal=None)
        bg._next_result = []

        initial_steps = engine.controller.stats.total_steps
        engine.speculative_step([MockSequenceState("r0")])
        # No controller.update() called since proposal was None
        assert engine.controller.stats.total_steps == initial_steps

    def test_batch_y_consistency_after_step(self):
        """After speculative_step, batch.y should be an mx.array
        with correct values for all sequences."""
        vocab_size = 200

        draft_tokens = mx.array([[10, 11], [20, 21]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([2, 2], dtype=mx.int32),
        )
        logits = _make_det_logits(2, 3, vocab_size, [
            [10, 11, 99],
            [20, 21, 88],
        ])

        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=[MockCacheLayer()],
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )

        engine, bg = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        seqs = [MockSequenceState("r0"), MockSequenceState("r1")]
        engine.speculative_step(seqs)

        # batch.y should be mx.array, not a Python list
        assert isinstance(batch.y, mx.array)
        assert batch.y.shape[0] == 2


# ===========================================================================
# 5. Config Validation Adversarial Tests
# ===========================================================================

class TestConfigValidationAdversarial:
    """Exhaustive config boundary and validation tests."""

    def test_num_speculative_tokens_negative(self):
        """Negative num_speculative_tokens should fail."""
        config = SpecDecodeConfig(num_speculative_tokens=-5)
        with pytest.raises(ValueError, match="num_speculative_tokens must be >= 1"):
            config.validate()

    def test_num_speculative_tokens_boundary_one(self):
        """num_speculative_tokens=1 is the minimum valid value."""
        config = SpecDecodeConfig(num_speculative_tokens=1)
        config.validate()  # should not raise

    def test_num_speculative_tokens_boundary_twenty(self):
        """num_speculative_tokens=20 is the maximum valid value."""
        config = SpecDecodeConfig(num_speculative_tokens=20)
        config.validate()  # should not raise

    def test_ngram_min_zero(self):
        """ngram_min=0 should fail (min is 1)."""
        config = SpecDecodeConfig(ngram_min=0)
        with pytest.raises(ValueError, match="ngram_min must be >= 1"):
            config.validate()

    def test_ngram_min_negative(self):
        """ngram_min=-1 should fail."""
        config = SpecDecodeConfig(ngram_min=-1)
        with pytest.raises(ValueError, match="ngram_min must be >= 1"):
            config.validate()

    def test_ngram_max_equals_min_valid(self):
        """ngram_max == ngram_min should pass."""
        config = SpecDecodeConfig(ngram_max=3, ngram_min=3)
        config.validate()  # should not raise

    def test_acceptance_rate_threshold_zero(self):
        """threshold=0.0 is valid (always speculate)."""
        config = SpecDecodeConfig(acceptance_rate_threshold=0.0)
        config.validate()

    def test_acceptance_rate_threshold_one(self):
        """threshold=1.0 is valid (only speculate at 100% acceptance)."""
        config = SpecDecodeConfig(acceptance_rate_threshold=1.0)
        config.validate()

    def test_ema_alpha_one_valid(self):
        """alpha=1.0 is valid (no memory, last step only)."""
        config = SpecDecodeConfig(acceptance_rate_ema_alpha=1.0)
        config.validate()

    def test_ema_alpha_near_zero_valid(self):
        """alpha=0.001 is valid (very smooth)."""
        config = SpecDecodeConfig(acceptance_rate_ema_alpha=0.001)
        config.validate()

    def test_disable_by_batch_size_zero_valid(self):
        """disable_by_batch_size=0 means never disable."""
        config = SpecDecodeConfig(disable_by_batch_size=0)
        config.validate()

    def test_config_mode_literal_enforcement(self):
        """Mode must be one of 'none', 'ngram', 'draft'.
        Setting to invalid via object.__setattr__ should fail at create_proposer."""
        config = SpecDecodeConfig()
        object.__setattr__(config, "mode", "invalid_mode")
        with pytest.raises(ValueError, match="Unknown spec decode mode"):
            create_proposer(config)


# ===========================================================================
# 6. Cache Utils Adversarial Tests
# ===========================================================================

class TestCacheUtilsAdversarial:
    """Adversarial tests for cache utility functions."""

    def test_uniform_trim_zero(self):
        """uniform_trim with amount=0 should be a no-op."""
        layer = MockCacheLayer(_offset=10)
        uniform_trim([layer], 0)
        assert layer.trim_calls == []

    def test_uniform_trim_negative(self):
        """uniform_trim with negative amount should be a no-op."""
        layer = MockCacheLayer(_offset=10)
        uniform_trim([layer], -5)
        assert layer.trim_calls == []

    def test_uniform_trim_empty_cache_list(self):
        """uniform_trim with no cache layers should not crash."""
        uniform_trim([], 5)

    def test_uniform_trim_multiple_layers(self):
        """uniform_trim should trim all layers by the same amount."""
        layers = [MockCacheLayer(_offset=20) for _ in range(4)]
        uniform_trim(layers, 3)
        for layer in layers:
            assert layer.trim_calls == [3]

    def test_batch_variable_trim_all_zero(self):
        """batch_variable_trim with all zeros -- no-op."""
        layer = MockCacheLayer(_offset=10, _batch_size=3)
        trim_amounts = mx.array([0, 0, 0], dtype=mx.int32)
        batch_variable_trim([layer], trim_amounts)
        assert layer.trim_per_seq_calls == []  # early return, no call


# ===========================================================================
# 7. SpecDecodeStats Adversarial Tests
# ===========================================================================

class TestStatsAdversarial:
    """Adversarial tests for SpecDecodeStats."""

    def test_acceptance_rate_zero_denominator(self):
        """total_proposed=0 should return 0.0, not NaN or error."""
        stats = SpecDecodeStats()
        assert stats.acceptance_rate == 0.0

    def test_avg_tokens_per_step_zero_steps(self):
        """total_steps=0 should return 1.0, not error."""
        stats = SpecDecodeStats()
        assert stats.avg_tokens_per_step == 1.0

    def test_stats_very_large_values(self):
        """Large stat values shouldn't overflow or lose precision."""
        stats = SpecDecodeStats(
            total_proposed=10**9,
            total_accepted=9 * 10**8,
            total_steps=10**6,
            total_bonus_tokens=10**5,
        )
        assert stats.acceptance_rate == pytest.approx(0.9)
        avg = stats.avg_tokens_per_step
        assert not math.isnan(avg)
        assert not math.isinf(avg)

    def test_accepted_greater_than_proposed(self):
        """Technically invalid but should not crash."""
        stats = SpecDecodeStats(total_proposed=5, total_accepted=10)
        # acceptance_rate will be 2.0 -- unusual but no crash
        assert stats.acceptance_rate == pytest.approx(2.0)


# ===========================================================================
# 8. Integration: Engine + Real NGramProposer
# ===========================================================================

class TestEngineWithRealProposer:
    """Test SpecDecodeEngine with actual NGramProposer (no mock proposer)."""

    def test_engine_with_ngram_proposer_no_match(self):
        """Real NGramProposer finds no match -> engine falls back."""
        vocab_size = 100
        proposer = NGramProposer(ngram_max=4, ngram_min=1)

        logits = mx.zeros((1, 1, vocab_size))
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=[MockCacheLayer()],
            samplers=[MockSampler(0.0)],
        )

        cfg = SpecDecodeConfig(mode="ngram", num_speculative_tokens=3)
        controller = DynamicSpecController(cfg)
        verifier = NGramVerifier(mode="greedy")
        model = MockModel(logits)
        bg = MockBatchGenerator(batch)
        bg._next_result = ["fallback_sentinel"]

        engine = SpecDecodeEngine(
            model=model, batch_generator=bg, proposer=proposer,
            verifier=verifier, config=cfg, controller=controller,
        )

        # All unique tokens -> no n-gram match -> fallback
        seqs = [MockSequenceState(
            "r0",
            token_ids=[10, 20, 30, 40, 50],
            output_tokens=[10, 20, 30, 40, 50],
        )]
        result = engine.speculative_step(seqs)
        assert result == ["fallback_sentinel"]

    def test_engine_with_ngram_proposer_match_found(self):
        """Real NGramProposer finds match -> engine processes spec step.

        Note: adaptive_k must be disabled so that k equals the full
        num_speculative_tokens. With adaptive_k=True and default EMA=0.7,
        the controller would reduce k to max(1, 3-2)=1 (moderate band).
        """
        vocab_size = 200
        proposer = NGramProposer(ngram_max=4, ngram_min=1)

        # Sequence with a repeated pattern
        token_ids = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        # NGramProposer should find [1, 2, 3] as continuation

        # Target model should see verify_input of shape [1, 4]
        # (last_token=5, draft_0=1, draft_1=2, draft_2=3)
        # Target argmax: [1, 2, 3, 99] -> all drafts accepted + bonus
        logits = _make_det_logits(1, 4, vocab_size, [[1, 2, 3, 99]])

        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array(token_ids)],
            num_tokens=[10],
            max_tokens=[100],
            cache=[MockCacheLayer(_offset=15)],
            samplers=[MockSampler(0.0)],
        )

        # Disable adaptive_k to get full k=3
        cfg = SpecDecodeConfig(
            mode="ngram",
            num_speculative_tokens=3,
            adaptive_k=False,
            disable_by_batch_size=0,
        )
        controller = DynamicSpecController(cfg)
        verifier = NGramVerifier(mode="greedy")
        model = MockModel(logits)
        bg = MockBatchGenerator(batch)

        engine = SpecDecodeEngine(
            model=model, batch_generator=bg, proposer=proposer,
            verifier=verifier, config=cfg, controller=controller,
        )

        seqs = [MockSequenceState(
            "r0",
            token_ids=token_ids,
            output_tokens=token_ids,
        )]
        responses = engine.speculative_step(seqs)

        assert len(responses) == 1
        resp = responses[0]
        assert isinstance(resp, SpecResponse)
        # All 3 drafts should match + bonus
        assert resp.tokens == [1, 2, 3, 99]
        assert resp.num_accepted == 3
        assert resp.num_drafted == 3


# ===========================================================================
# 9. Verifier + Proposer Combined Edge Cases
# ===========================================================================

class TestVerifierProposerCombined:
    """Cross-component edge cases."""

    def test_proposal_result_padding_with_verify(self):
        """Proposer pads with 0, verifier should handle padded positions correctly
        by using proposal_lens to mask invalid positions."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3

        # Seq 0: plen=1, Seq 1: plen=3
        # Seq 0's positions 1,2 are padding (0) -- should not be verified
        argmax_tokens = [[5, 0, 0, 9], [5, 6, 7, 9]]
        target_probs = _make_det_probs(2, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([
            [5, 0, 0],  # only pos 0 is real
            [5, 6, 7],  # all real
        ], dtype=mx.int32)
        proposal_lens = mx.array([1, 3], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        # Seq 0: plen=1, draft[0]=5 matches argmax[0]=5 -> 1 accepted
        assert int(num_acc[0]) == 1
        # Seq 1: all match -> 3 accepted
        assert int(num_acc[1]) == 3

    def test_large_vocab_correctness(self):
        """Verify with vocab_size=128000 (LLM-scale) doesn't crash."""
        v = NGramVerifier(mode="greedy")
        vocab = 128000
        k = 5

        argmax_tokens = [[100, 200, 300, 400, 500, 600]]
        target_probs = _make_det_probs(1, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([[100, 200, 300, 400, 500]], dtype=mx.int32)
        proposal_lens = mx.array([5], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 5
        assert int(accepted[0, 5]) == 600  # bonus token


# ===========================================================================
# 10. Per-Sequence Trim Adversarial Tests (Result A)
# ===========================================================================

class TestPerSequenceTrimAdversarial:
    """Adversarial tests for Result A per-sequence trim behavior."""

    def test_trim_exceeds_available_offset(self):
        """trim_amounts > offset should be clamped by trim_per_sequence."""
        layer = MockCacheLayer(_offset=5, _batch_size=2)
        # Try to trim 10, but offset is only 5
        trim_amounts = mx.array([10, 3], dtype=mx.int32)
        batch_variable_trim([layer], trim_amounts)
        # Should clamp: min(10, 0+5)=5, min(3, 0+5)=3
        # offset = [5-5, 5-3] = [0, 2]
        assert int(layer.offset[0]) == 0
        assert int(layer.offset[1]) == 2

    def test_single_sequence_trim_no_regression(self):
        """Single sequence with per-seq trim should behave like uniform trim."""
        layer_a = MockCacheLayer(_offset=10, _batch_size=1)
        layer_b = MockCacheLayer(_offset=10, _batch_size=1)

        # Per-sequence trim
        batch_variable_trim([layer_a], mx.array([3], dtype=mx.int32))
        # Uniform trim
        uniform_trim([layer_b], 3)

        assert int(layer_a.offset[0]) == int(layer_b.offset[0])

    def test_all_sequences_same_trim_equals_uniform(self):
        """When all sequences trim the same amount, result matches uniform trim."""
        layer_a = MockCacheLayer(_offset=10, _batch_size=3)
        layer_b = MockCacheLayer(_offset=10, _batch_size=3)

        batch_variable_trim([layer_a], mx.array([4, 4, 4], dtype=mx.int32))
        uniform_trim([layer_b], 4)

        for i in range(3):
            assert int(layer_a.offset[i]) == int(layer_b.offset[i])

    def test_mixed_zero_and_nonzero_trims(self):
        """Some sequences trim 0, some trim more."""
        layer = MockCacheLayer(_offset=10, _batch_size=4)
        trim_amounts = mx.array([0, 5, 0, 10], dtype=mx.int32)
        batch_variable_trim([layer], trim_amounts)
        # offset = [10-0, 10-5, 10-0, 10-10] = [10, 5, 10, 0]
        expected = [10, 5, 10, 0]
        for i, exp in enumerate(expected):
            assert int(layer.offset[i]) == exp

    def test_large_batch_variable_trim(self):
        """Variable trim with 32 sequences -- no crash, correct offsets."""
        batch_size = 32
        layer = MockCacheLayer(_offset=50, _batch_size=batch_size)
        trims = [i % 10 for i in range(batch_size)]
        batch_variable_trim([layer], mx.array(trims, dtype=mx.int32))

        for i in range(batch_size):
            expected = 50 - trims[i]
            assert int(layer.offset[i]) == expected

    def test_trim_to_zero_all_sequences(self):
        """Trim all sequences to offset=0."""
        layer = MockCacheLayer(_offset=7, _batch_size=3)
        batch_variable_trim([layer], mx.array([7, 7, 7], dtype=mx.int32))
        for i in range(3):
            assert int(layer.offset[i]) == 0

    def test_result_a_b_identical_when_uniform_acceptance(self):
        """When all sequences accept the same number of tokens,
        Result A and Result B should produce identical outputs."""
        vocab_size = 200

        # Both sequences accept all 3 drafts
        draft_tokens = mx.array([[10, 11, 12], [20, 21, 22]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3, 3], dtype=mx.int32),
        )
        logits = _make_det_logits(2, 4, vocab_size, [
            [10, 11, 12, 99],
            [20, 21, 22, 88],
        ])

        # Run with Result A
        cache_a = [MockCacheLayer(_offset=20, _batch_size=2)]
        batch_a = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=cache_a,
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )
        engine_a, _ = _make_engine(batch=batch_a, model_logits=logits, proposal=proposal)
        engine_a._per_seq_trim = True
        seqs_a = [MockSequenceState("r0"), MockSequenceState("r1")]
        resp_a = engine_a.speculative_step(seqs_a)

        # Run with Result B
        cache_b = [MockCacheLayer(_offset=20, _batch_size=2)]
        proposal_b = ProposalResult(
            draft_tokens=mx.array([[10, 11, 12], [20, 21, 22]], dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.array([3, 3], dtype=mx.int32),
        )
        batch_b = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=cache_b,
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )
        engine_b, _ = _make_engine(batch=batch_b, model_logits=logits, proposal=proposal_b)
        engine_b._per_seq_trim = False
        seqs_b = [MockSequenceState("r0"), MockSequenceState("r1")]
        resp_b = engine_b.speculative_step(seqs_b)

        # When acceptance is uniform, both should produce same tokens
        for i in range(2):
            assert resp_a[i].tokens == resp_b[i].tokens
            assert resp_a[i].num_accepted == resp_b[i].num_accepted

    def test_max_divergent_acceptance(self):
        """One seq accepts all k, another accepts 0.
        Result A: seq0 gets k+1 tokens, seq1 gets 1."""
        vocab_size = 200

        draft_tokens = mx.array([
            [10, 11, 12],
            [90, 91, 92],
        ], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3, 3], dtype=mx.int32),
        )
        # Seq 0: all match, Seq 1: none match
        logits = _make_det_logits(2, 4, vocab_size, [
            [10, 11, 12, 99],
            [60, 61, 62, 63],
        ])

        cache_layers = [MockCacheLayer(_offset=20, _batch_size=2)]
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=cache_layers,
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )

        engine, _ = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        engine._per_seq_trim = True
        seqs = [MockSequenceState("r0"), MockSequenceState("r1")]
        responses = engine.speculative_step(seqs)

        # Result A: seq 0 keeps all 4 (3 accepted + bonus)
        assert responses[0].tokens == [10, 11, 12, 99]
        assert responses[0].num_accepted == 3
        # Seq 1 gets 1 correction token
        assert responses[1].tokens == [60]
        assert responses[1].num_accepted == 0

    def test_single_sequence_result_a_matches_b(self):
        """With batch_size=1, Result A and B produce identical output."""
        vocab_size = 200

        draft_tokens = mx.array([[10, 77, 78]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        logits = _make_det_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        # Result A
        cache_a = [MockCacheLayer(_offset=20)]
        batch_a = MockBatch(
            uids=[0], y=mx.array([5]),
            tokens=[mx.array([1, 2, 3])], num_tokens=[3],
            max_tokens=[100], cache=cache_a, samplers=[MockSampler(0.0)],
        )
        engine_a, _ = _make_engine(batch=batch_a, model_logits=logits, proposal=proposal)
        engine_a._per_seq_trim = True
        resp_a = engine_a.speculative_step([MockSequenceState("r0")])

        # Result B
        proposal_b = ProposalResult(
            draft_tokens=mx.array([[10, 77, 78]], dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        cache_b = [MockCacheLayer(_offset=20)]
        batch_b = MockBatch(
            uids=[0], y=mx.array([5]),
            tokens=[mx.array([1, 2, 3])], num_tokens=[3],
            max_tokens=[100], cache=cache_b, samplers=[MockSampler(0.0)],
        )
        engine_b, _ = _make_engine(batch=batch_b, model_logits=logits, proposal=proposal_b)
        engine_b._per_seq_trim = False
        resp_b = engine_b.speculative_step([MockSequenceState("r0")])

        # Single sequence: Result A and B identical
        assert resp_a[0].tokens == resp_b[0].tokens
        assert resp_a[0].num_accepted == resp_b[0].num_accepted

    def test_zero_acceptance_all_sequences(self):
        """All sequences accept 0 tokens — each gets 1 correction token."""
        vocab_size = 200

        # All draft tokens mismatch
        draft_tokens = mx.array([
            [90, 91, 92],
            [93, 94, 95],
            [96, 97, 98],
        ], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3, 3, 3], dtype=mx.int32),
        )
        logits = _make_det_logits(3, 4, vocab_size, [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ])

        cache = [MockCacheLayer(_offset=20, _batch_size=3)]
        batch = MockBatch(
            uids=[0, 1, 2],
            y=mx.array([5, 6, 7]),
            tokens=[mx.array([1]), mx.array([2]), mx.array([3])],
            num_tokens=[1, 1, 1],
            max_tokens=[100, 100, 100],
            cache=cache,
            samplers=[MockSampler(0.0), MockSampler(0.0), MockSampler(0.0)],
        )

        engine, _ = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        engine._per_seq_trim = True
        seqs = [MockSequenceState("r0"), MockSequenceState("r1"), MockSequenceState("r2")]
        responses = engine.speculative_step(seqs)

        # All get exactly 1 correction token
        for i, expected_tok in enumerate([10, 20, 30]):
            assert len(responses[i].tokens) == 1
            assert responses[i].tokens[0] == expected_tok
            assert responses[i].num_accepted == 0

    def test_all_k_accepted(self):
        """All sequences accept all k tokens — no trim needed."""
        vocab_size = 200

        draft_tokens = mx.array([
            [10, 11, 12],
            [20, 21, 22],
        ], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3, 3], dtype=mx.int32),
        )
        logits = _make_det_logits(2, 4, vocab_size, [
            [10, 11, 12, 99],
            [20, 21, 22, 88],
        ])

        cache = [MockCacheLayer(_offset=20, _batch_size=2)]
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=cache,
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )

        engine, _ = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        engine._per_seq_trim = True
        seqs = [MockSequenceState("r0"), MockSequenceState("r1")]
        responses = engine.speculative_step(seqs)

        # All accepted + bonus: 4 tokens each
        assert responses[0].tokens == [10, 11, 12, 99]
        assert responses[1].tokens == [20, 21, 22, 88]
        # No trim needed: keep=4, trim=0
        assert cache[0].trim_per_seq_calls == []

    def test_proposal_lens_zero_with_result_a(self):
        """Sequence with proposal_lens=0 still gets 1 token via fallback."""
        vocab_size = 200

        # Seq 0 has proposals, Seq 1 has none
        draft_tokens = mx.array([
            [10, 11, 12],
            [0, 0, 0],
        ], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3, 0], dtype=mx.int32),
        )
        logits = _make_det_logits(2, 4, vocab_size, [
            [10, 11, 12, 99],
            [50, 51, 52, 53],
        ])

        cache = [MockCacheLayer(_offset=20, _batch_size=2)]
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=cache,
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )

        engine, _ = _make_engine(batch=batch, model_logits=logits, proposal=proposal)
        engine._per_seq_trim = True
        seqs = [MockSequenceState("r0"), MockSequenceState("r1")]
        responses = engine.speculative_step(seqs)

        # Seq 0: all 3 accepted + bonus = 4 tokens
        assert responses[0].tokens == [10, 11, 12, 99]
        assert responses[0].num_accepted == 3
        # Seq 1: proposal_lens=0, gets 1 token (correction from target)
        assert len(responses[1].tokens) == 1
        assert responses[1].num_accepted == 0
