"""Tests for NGramVerifier — greedy and threshold verification strategies."""

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.verifier import (
    PLACEHOLDER_TOKEN_ID,
    NGramVerifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_deterministic_probs(
    batch_size: int,
    k_plus_one: int,
    vocab_size: int,
    argmax_tokens: list[list[int]],
    *,
    base_prob: float = 0.01,
    top_prob: float = 0.9,
) -> mx.array:
    """Create target_probs with known argmax at each position.

    argmax_tokens: [batch][k+1] — the token that should be argmax at each pos.
    Returns [batch, k+1, vocab_size] with top_prob at the argmax token.
    """
    probs = mx.full((batch_size, k_plus_one, vocab_size), base_prob)
    for b in range(batch_size):
        for pos in range(k_plus_one):
            tok = argmax_tokens[b][pos]
            probs[b, pos, tok] = top_prob
    return probs


def make_probs_with_specific_values(
    batch_size: int,
    k_plus_one: int,
    vocab_size: int,
    token_probs: list[list[tuple[int, float]]],
) -> mx.array:
    """Create target_probs with specific probabilities for specific tokens.

    token_probs: [batch][position] -> (token_id, probability)
    Remaining probability distributed uniformly.
    """
    probs = mx.full((batch_size, k_plus_one, vocab_size), 0.001)
    for b in range(batch_size):
        for pos in range(k_plus_one):
            tok, p = token_probs[b][pos]
            probs[b, pos, tok] = p
    return probs


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestNGramVerifierInit:
    def test_default_mode(self):
        v = NGramVerifier()
        assert v.mode == "greedy"
        assert v.threshold == 0.1

    def test_threshold_mode(self):
        v = NGramVerifier(mode="threshold", threshold=0.3)
        assert v.mode == "threshold"
        assert v.threshold == 0.3

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown verification mode"):
            NGramVerifier(mode="sampling")


# ---------------------------------------------------------------------------
# Greedy mode tests
# ---------------------------------------------------------------------------

class TestGreedyVerification:
    def test_greedy_all_accepted(self):
        """All draft tokens match target argmax → k accepted + 1 bonus token."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3

        # Target argmax at positions 0,1,2 = tokens 5,6,7; bonus at pos 3 = token 8
        argmax_tokens = [[5, 6, 7, 8]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([[5, 6, 7]], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert accepted.shape == (1, k + 1)
        assert num_acc.shape == (1,)
        assert int(num_acc[0]) == 3
        # All 3 drafts accepted + bonus token 8
        assert int(accepted[0, 0]) == 5
        assert int(accepted[0, 1]) == 6
        assert int(accepted[0, 2]) == 7
        assert int(accepted[0, 3]) == 8  # bonus

    def test_greedy_none_accepted(self):
        """First draft mismatches → 0 accepted + 1 correction token."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3

        # Target argmax at positions 0,1,2 = tokens 5,6,7
        argmax_tokens = [[5, 6, 7, 8]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        # Draft token 0 is wrong (9 != 5)
        draft_tokens = mx.array([[9, 6, 7]], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 0
        # Correction token at position 0 is target argmax = 5
        assert int(accepted[0, 0]) == 5
        # Remaining positions are placeholders
        assert int(accepted[0, 1]) == PLACEHOLDER_TOKEN_ID
        assert int(accepted[0, 2]) == PLACEHOLDER_TOKEN_ID
        assert int(accepted[0, 3]) == PLACEHOLDER_TOKEN_ID

    def test_greedy_partial_accept(self):
        """First 2 of 4 match, then mismatch → 2 accepted + 1 correction."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 4

        # Target argmax: 5,6,7,8; bonus pos: 9
        argmax_tokens = [[5, 6, 7, 8, 9]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        # First 2 match (5,6), position 2 wrong (0 != 7)
        draft_tokens = mx.array([[5, 6, 0, 8]], dtype=mx.int32)
        proposal_lens = mx.array([4], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 2
        assert int(accepted[0, 0]) == 5  # accepted
        assert int(accepted[0, 1]) == 6  # accepted
        assert int(accepted[0, 2]) == 7  # correction (target argmax at pos 2)
        assert int(accepted[0, 3]) == PLACEHOLDER_TOKEN_ID
        assert int(accepted[0, 4]) == PLACEHOLDER_TOKEN_ID

    def test_greedy_no_proposal(self):
        """proposal_lens=0 → uses target argmax at position 0."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3

        argmax_tokens = [[5, 6, 7, 8]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([[0, 0, 0]], dtype=mx.int32)  # irrelevant
        proposal_lens = mx.array([0], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 0
        assert int(accepted[0, 0]) == 5  # target argmax at pos 0

    def test_cumprod_masking(self):
        """Verify [T,T,F,T] becomes [T,T,F,F] via cumprod trick."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 4

        # Target argmax: 1,2,3,4; bonus: 5
        argmax_tokens = [[1, 2, 3, 4, 5]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        # Match at 0,1; mismatch at 2; match at 3 (should still be rejected)
        draft_tokens = mx.array([[1, 2, 0, 4]], dtype=mx.int32)
        proposal_lens = mx.array([4], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 2
        assert int(accepted[0, 0]) == 1  # accepted
        assert int(accepted[0, 1]) == 2  # accepted
        assert int(accepted[0, 2]) == 3  # correction (target argmax)
        assert int(accepted[0, 3]) == PLACEHOLDER_TOKEN_ID  # position 3 rejected despite matching
        assert int(accepted[0, 4]) == PLACEHOLDER_TOKEN_ID


# ---------------------------------------------------------------------------
# Threshold mode tests
# ---------------------------------------------------------------------------

class TestThresholdVerification:
    def test_threshold_accept_high_prob(self):
        """Draft token has high prob in target → accepted."""
        v = NGramVerifier(mode="threshold", threshold=0.1)
        vocab = 10
        k = 2

        # draft token 3 has high prob at both positions
        token_probs = [
            [(3, 0.5), (3, 0.4), (7, 0.9)],
        ]
        target_probs = make_probs_with_specific_values(1, k + 1, vocab, token_probs)

        draft_tokens = mx.array([[3, 3]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 2
        assert int(accepted[0, 0]) == 3
        assert int(accepted[0, 1]) == 3

    def test_threshold_reject_low_prob(self):
        """Draft token has low prob in target → rejected."""
        v = NGramVerifier(mode="threshold", threshold=0.1)
        vocab = 10
        k = 2

        # draft token 3 has low prob at position 0
        token_probs = [
            [(3, 0.05), (3, 0.5), (7, 0.9)],
        ]
        target_probs = make_probs_with_specific_values(1, k + 1, vocab, token_probs)

        # Make sure token 7 is argmax at pos 0 for correction
        target_probs[0, 0, 7] = 0.9

        draft_tokens = mx.array([[3, 3]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 0
        # Correction at pos 0 is argmax of target_probs[0, 0, :] = 7
        assert int(accepted[0, 0]) == 7

    def test_threshold_partial(self):
        """First position accepted, second rejected by threshold."""
        v = NGramVerifier(mode="threshold", threshold=0.2)
        vocab = 10
        k = 3

        # Position 0: token 5 has prob 0.6 (accept)
        # Position 1: token 6 has prob 0.05 (reject, < 0.2)
        # Position 2: token 7 has prob 0.8 (would accept but cumprod kills it)
        token_probs = [
            [(5, 0.6), (6, 0.05), (7, 0.8), (9, 0.9)],
        ]
        target_probs = make_probs_with_specific_values(1, k + 1, vocab, token_probs)
        # Set argmax at position 1 to token 8
        target_probs[0, 1, 8] = 0.9

        draft_tokens = mx.array([[5, 6, 7]], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 1
        assert int(accepted[0, 0]) == 5  # accepted
        # Correction at pos 1 is argmax of target_probs[0, 1, :] = 8
        assert int(accepted[0, 1]) == 8
        assert int(accepted[0, 2]) == PLACEHOLDER_TOKEN_ID


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------

class TestBatchVerification:
    def test_batch_mixed_acceptance(self):
        """Batch of 3 sequences with different acceptance counts."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3

        # Seq 0: all match → 3 accepted
        # Seq 1: first mismatch → 0 accepted
        # Seq 2: first 2 match → 2 accepted
        argmax_tokens = [
            [5, 6, 7, 8],  # seq 0
            [5, 6, 7, 8],  # seq 1
            [5, 6, 7, 8],  # seq 2
        ]
        target_probs = make_deterministic_probs(3, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([
            [5, 6, 7],  # all match
            [0, 6, 7],  # first mismatch
            [5, 6, 0],  # 2 match, 3rd mismatch
        ], dtype=mx.int32)
        proposal_lens = mx.array([3, 3, 3], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert accepted.shape == (3, k + 1)
        assert int(num_acc[0]) == 3
        assert int(num_acc[1]) == 0
        assert int(num_acc[2]) == 2

        # Seq 0: all accepted + bonus token 8
        assert int(accepted[0, 3]) == 8

        # Seq 1: correction at pos 0 = target argmax 5
        assert int(accepted[1, 0]) == 5
        assert int(accepted[1, 1]) == PLACEHOLDER_TOKEN_ID

        # Seq 2: 2 accepted, correction at pos 2 = 7
        assert int(accepted[2, 0]) == 5
        assert int(accepted[2, 1]) == 6
        assert int(accepted[2, 2]) == 7  # correction


# ---------------------------------------------------------------------------
# Output shape and placeholder tests
# ---------------------------------------------------------------------------

class TestOutputProperties:
    def test_output_shape(self):
        """Output is [B, k+1], num_accepted is [B]."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 5
        batch_size = 4

        argmax_tokens = [[0] * (k + 1)] * batch_size
        target_probs = make_deterministic_probs(batch_size, k + 1, vocab, argmax_tokens)
        draft_tokens = mx.zeros((batch_size, k), dtype=mx.int32)
        proposal_lens = mx.array([k] * batch_size, dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert accepted.shape == (batch_size, k + 1)
        assert num_acc.shape == (batch_size,)

    def test_placeholder_in_rejected_positions(self):
        """Rejected positions contain PLACEHOLDER_TOKEN_ID."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 4

        argmax_tokens = [[1, 2, 3, 4, 5]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        # Only first matches
        draft_tokens = mx.array([[1, 0, 0, 0]], dtype=mx.int32)
        proposal_lens = mx.array([4], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 1
        assert int(accepted[0, 0]) == 1   # accepted
        assert int(accepted[0, 1]) == 2   # correction
        assert int(accepted[0, 2]) == PLACEHOLDER_TOKEN_ID
        assert int(accepted[0, 3]) == PLACEHOLDER_TOKEN_ID
        assert int(accepted[0, 4]) == PLACEHOLDER_TOKEN_ID

    def test_bonus_token_on_full_acceptance(self):
        """When all k drafted tokens accepted, bonus = argmax(target_probs[:, k, :])."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 2

        # Bonus token at position k=2 should be token 9
        argmax_tokens = [[3, 4, 9]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([[3, 4]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 2
        assert int(accepted[0, 2]) == 9  # bonus token

    def test_correction_token_is_target_argmax(self):
        """At rejection point, correction token is target model's argmax."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 3

        argmax_tokens = [[1, 2, 3, 4]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)

        # Mismatch at position 1 (draft=9, target argmax=2)
        draft_tokens = mx.array([[1, 9, 3]], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)

        accepted, num_acc = v.verify(target_probs, draft_tokens, proposal_lens)

        assert int(num_acc[0]) == 1
        # Correction at position 1 = target argmax at pos 1 = 2
        assert int(accepted[0, 1]) == 2


# ---------------------------------------------------------------------------
# modes parameter tests (per-sequence verification mode)
# ---------------------------------------------------------------------------

class TestModesParameter:
    def test_modes_parameter_greedy_only(self):
        """modes=["greedy","greedy"] behaves same as self.mode="greedy"."""
        v = NGramVerifier(mode="threshold", threshold=0.1)  # default is threshold
        vocab = 10
        k = 2

        argmax_tokens = [[5, 6, 7], [5, 6, 7]]
        target_probs = make_deterministic_probs(2, k + 1, vocab, argmax_tokens)

        draft_tokens = mx.array([[5, 6], [5, 0]], dtype=mx.int32)
        proposal_lens = mx.array([2, 2], dtype=mx.int32)

        # Force greedy via modes
        accepted, num_acc = v.verify(
            target_probs, draft_tokens, proposal_lens,
            modes=["greedy", "greedy"],
        )

        # Seq 0: all match → 2 accepted
        assert int(num_acc[0]) == 2
        # Seq 1: first match, second mismatch → 1 accepted
        assert int(num_acc[1]) == 1

    def test_modes_parameter_threshold_only(self):
        """modes=["threshold","threshold"] behaves same as self.mode="threshold"."""
        v = NGramVerifier(mode="greedy", threshold=0.1)  # default is greedy

        vocab = 10
        k = 2

        # Token 3 has prob 0.5 (above threshold) at all positions
        token_probs = [
            [(3, 0.5), (3, 0.5), (7, 0.9)],
            [(3, 0.5), (3, 0.5), (7, 0.9)],
        ]
        target_probs = make_probs_with_specific_values(2, k + 1, vocab, token_probs)

        draft_tokens = mx.array([[3, 3], [3, 3]], dtype=mx.int32)
        proposal_lens = mx.array([2, 2], dtype=mx.int32)

        # Force threshold via modes
        accepted, num_acc = v.verify(
            target_probs, draft_tokens, proposal_lens,
            modes=["threshold", "threshold"],
        )

        # Both seqs: token 3 has 0.5 prob >= 0.1 threshold → all accepted
        assert int(num_acc[0]) == 2
        assert int(num_acc[1]) == 2

    def test_modes_parameter_mixed(self):
        """modes=["greedy","threshold"] → seq 0 greedy, seq 1 threshold."""
        v = NGramVerifier(mode="greedy", threshold=0.1)
        vocab = 10
        k = 2

        # Seq 0: argmax at pos 0 is token 5, pos 1 is token 6, pos 2 is token 7
        # Seq 1: token 3 has prob 0.3 at all positions (not argmax but above threshold)
        #        argmax at all positions is token 8
        probs = mx.full((2, k + 1, vocab), 0.01)
        # Seq 0: deterministic argmax
        probs[0, 0, 5] = 0.9
        probs[0, 1, 6] = 0.9
        probs[0, 2, 7] = 0.9
        # Seq 1: token 3 has 0.3 prob (above 0.1 threshold), but argmax is token 8
        probs[1, 0, 3] = 0.3
        probs[1, 0, 8] = 0.5  # argmax
        probs[1, 1, 3] = 0.3
        probs[1, 1, 8] = 0.5  # argmax
        probs[1, 2, 7] = 0.9

        # Seq 0 drafts match argmax (greedy will accept)
        # Seq 1 drafts are token 3 (not argmax, but above threshold)
        draft_tokens = mx.array([[5, 6], [3, 3]], dtype=mx.int32)
        proposal_lens = mx.array([2, 2], dtype=mx.int32)

        accepted, num_acc = v.verify(
            probs, draft_tokens, proposal_lens,
            modes=["greedy", "threshold"],
        )

        # Seq 0 (greedy): both match argmax → 2 accepted
        assert int(num_acc[0]) == 2
        # Seq 1 (threshold): token 3 has prob 0.3 >= 0.1 → 2 accepted
        assert int(num_acc[1]) == 2

        # Now verify greedy alone would reject seq 1
        accepted_g, num_acc_g = v._greedy(probs, draft_tokens, proposal_lens)
        assert int(num_acc_g[1]) == 0  # seq 1: token 3 != argmax 8

    def test_modes_none_uses_default(self):
        """modes=None falls back to self.mode."""
        v = NGramVerifier(mode="greedy")
        vocab = 10
        k = 2

        argmax_tokens = [[5, 6, 7]]
        target_probs = make_deterministic_probs(1, k + 1, vocab, argmax_tokens)
        draft_tokens = mx.array([[5, 6]], dtype=mx.int32)
        proposal_lens = mx.array([2], dtype=mx.int32)

        # modes=None should use self.mode="greedy"
        accepted, num_acc = v.verify(
            target_probs, draft_tokens, proposal_lens, modes=None
        )

        assert int(num_acc[0]) == 2
        assert int(accepted[0, 0]) == 5
        assert int(accepted[0, 1]) == 6
        assert int(accepted[0, 2]) == 7  # bonus
