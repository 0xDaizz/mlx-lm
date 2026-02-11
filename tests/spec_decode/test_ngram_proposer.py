"""Tests for NGramProposer — speculative decoding n-gram proposer.

Covers linear search, suffix index, batched propose(), and edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from mlx_lm_server.spec_decode.proposer.ngram import NGramProposer


# ---------------------------------------------------------------------------
# Lightweight mock — only the fields NGramProposer actually reads
# ---------------------------------------------------------------------------

@dataclass
class MockSequenceState:
    request_id: str = "test"
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


def _make_seq(
    token_ids: list[int],
    output_tokens: list[int] | None = None,
) -> MockSequenceState:
    """Helper to build a MockSequenceState.

    If output_tokens is None, treats all token_ids as output_tokens.
    """
    return MockSequenceState(
        token_ids=token_ids,
        output_tokens=output_tokens if output_tokens is not None else list(token_ids),
    )


# ===== Single-sequence tests (via propose() with batch of 1) =====


class TestExactRepeatPattern:
    """Context [1,2,3,4,5,1,2,3,4,5], last 2 tokens [4,5] match pos 3."""

    def test_exact_repeat_pattern(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        # Context: [1,2,3,4,5,1,2,3,4,5]
        # With n=4: key=(2,3,4,5), found at pos 1, continuation = context[5:8] = [1,2,3]
        seq = _make_seq([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result = proposer.propose([seq], k=3)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        lens = result.proposal_lens.tolist()
        assert lens[0] == 3
        assert tokens[:3] == [1, 2, 3]


class TestNoMatchUniqueTokens:
    """All unique tokens — no n-gram repeats, expect empty proposal."""

    def test_no_match_unique_tokens(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq = _make_seq([10, 20, 30, 40, 50, 60, 70, 80])
        result = proposer.propose([seq], k=3)
        assert result is None


class TestLargestNgramPreferred:
    """4-gram match exists alongside 2-gram match; 4-gram should win."""

    def test_largest_ngram_preferred(self):
        # Build context where a 4-gram match exists early and a 2-gram match
        # exists at a later position. The 4-gram should be preferred because
        # the algorithm tries largest n first.
        #
        # Context: [1,2,3,4, 9,9, 3,4, 1,2,3,4]
        #   n=4: key=(1,2,3,4) found at pos 0 → continuation: [9,9,3,4] (first 3: [9,9,3])
        #   n=2: key=(3,4) found at pos 6 → continuation: [1,2,3] (first 3: [1,2,3])
        # 4-gram wins → result is [9,9,3]
        proposer = NGramProposer(ngram_max=4, ngram_min=2)
        ctx = [1, 2, 3, 4, 9, 9, 3, 4, 1, 2, 3, 4]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=3)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        lens = result.proposal_lens.tolist()
        assert lens[0] == 3
        assert tokens[:3] == [9, 9, 3]


class TestMostRecentMatchPreferred:
    """Same n-gram at two positions; most recent (right-most) should be used."""

    def test_most_recent_match_preferred(self):
        # Context: [5,6, 7,8,9, 5,6, 1,2,3, 5,6]
        #   n=2: key=(5,6) — found at pos 5 (most recent before current pos 10)
        #   continuation from pos 5: context[7:7+3] = [1,2,3]
        #   Also exists at pos 0, continuation from pos 0: context[2:5] = [7,8,9]
        # Most recent match (pos 5) wins → [1,2,3]
        proposer = NGramProposer(ngram_max=2, ngram_min=2)
        ctx = [5, 6, 7, 8, 9, 5, 6, 1, 2, 3, 5, 6]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=3)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        lens = result.proposal_lens.tolist()
        assert lens[0] == 3
        assert tokens[:3] == [1, 2, 3]


class TestProposalLengthCappedAtK:
    """Match has >k continuation tokens but only k should be returned."""

    def test_proposal_length_capped_at_k(self):
        # Context: [1,2, 10,20,30,40,50,60,70,80,90,100, 1,2]
        # n=2: key=(1,2) at pos 0. Continuation: context[2:2+3]=[10,20,30]
        # Even though 10 continuation tokens exist, k=3 caps it.
        proposer = NGramProposer(ngram_max=2, ngram_min=1)
        ctx = [1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1, 2]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=3)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        lens = result.proposal_lens.tolist()
        assert lens[0] == 3
        assert tokens[:3] == [10, 20, 30]


class TestProposalShorterThanK:
    """Match near end of context — fewer continuation tokens than k."""

    def test_proposal_shorter_than_k(self):
        # Context: [1,2,3, 1,2,3, 7,8]
        # n=3: key=(3,7,8) — no match
        # n=2: key=(7,8) — no match
        # n=1: key=(8,) — no match
        # Hmm, let me design more carefully.
        #
        # Context: [1,2, 9, 1,2]
        #   n=2: key=(1,2) at pos 0. continuation: context[2:2+5]=[9,1,2]
        #   only 3 tokens available, k=5 → proposal len = 3
        proposer = NGramProposer(ngram_max=2, ngram_min=1)
        ctx = [1, 2, 9, 1, 2]
        seq = _make_seq(ctx)
        result = proposer.propose([seq], k=5)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        lens = result.proposal_lens.tolist()
        assert lens[0] == 3
        assert tokens[:3] == [9, 1, 2]


class TestContextTooShort:
    """Context has fewer tokens than ngram_min + 1 → empty."""

    def test_context_too_short(self):
        # ngram_min=2, context length must be >= 3 for any match
        proposer = NGramProposer(ngram_max=4, ngram_min=2)
        seq = _make_seq([1, 2])  # len=2 < 2+1=3
        result = proposer.propose([seq], k=3)
        assert result is None


class TestEmptyContext:
    """Empty token list → empty proposal."""

    def test_empty_context(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq = _make_seq([])
        result = proposer.propose([seq], k=3)
        assert result is None


class TestSingleTokenContext:
    """Only 1 token — no n-gram possible (need at least ngram_min + 1)."""

    def test_single_token_context(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq = _make_seq([42])
        result = proposer.propose([seq], k=3)
        assert result is None


# ===== Batch tests =====


class TestBatchMixedProposals:
    """Batch of 3 sequences: one with match, one without, one with short match."""

    def test_batch_mixed_proposals(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)

        # Seq 0: has a match — [1,2,3,4,5,1,2,3,4,5] → proposes [1,2,3]
        seq0 = _make_seq([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        # Seq 1: no match — all unique
        seq1 = _make_seq([10, 20, 30, 40, 50, 60, 70, 80])
        # Seq 2: short match — [1,2,9, 1,2] → proposes [9] (only 1 token after key)
        # Actually continuation from pos 0, n=2, key=(1,2), context[2:2+3]=[9,1,2] → 3 tokens
        # Let me use a tighter context: [1,2, 1,2] → key=(1,2) at pos 0, continuation=[1,2] but
        # those overlap with current position. Let me think again.
        # Context: [8,9, 8,9] → n=2: key=(8,9) at pos 0, continuation = context[2:2+3]=[8,9]
        # len=2 < k=3. That gives a short match of 2.
        seq2 = _make_seq([8, 9, 8, 9])

        result = proposer.propose([seq0, seq1, seq2], k=3)
        assert result is not None

        draft = result.draft_tokens.tolist()
        lens = result.proposal_lens.tolist()

        # Seq 0: 3 proposals
        assert lens[0] == 3
        assert draft[0][:3] == [1, 2, 3]

        # Seq 1: no proposals
        assert lens[1] == 0

        # Seq 2: 2 proposals (only 2 continuation tokens available)
        assert lens[2] == 2
        assert draft[2][:2] == [8, 9]

        # Padded shape should be [3, 3] (max proposal len = 3)
        assert result.draft_tokens.shape == (3, 3)

        # draft_probs should be None for ngram
        assert result.draft_probs is None


class TestBatchAllNoMatch:
    """All sequences have no match → propose() returns None."""

    def test_batch_all_no_match(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq0 = _make_seq([10, 20, 30, 40, 50])
        seq1 = _make_seq([60, 70, 80, 90, 100])
        result = proposer.propose([seq0, seq1], k=3)
        assert result is None


# ===== prompt_lookup=False test =====


class TestPromptLookupFalse:
    """With prompt_lookup=False, only searches output_tokens (generated)."""

    def test_prompt_lookup_false(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1, prompt_lookup=False)

        # Prompt has the pattern, but output_tokens do NOT
        # token_ids = prompt + output = [1,2,3,1,2,3] + [10,20,30,40,50]
        # output_tokens = [10,20,30,40,50] — all unique, no match
        prompt = [1, 2, 3, 1, 2, 3]
        output = [10, 20, 30, 40, 50]
        seq = _make_seq(
            token_ids=prompt + output,
            output_tokens=output,
        )
        result = proposer.propose([seq], k=3)
        assert result is None

    def test_prompt_lookup_false_with_output_match(self):
        """Output tokens have a pattern, should find match even without prompt."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1, prompt_lookup=False)

        prompt = [100, 200, 300]
        output = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        seq = _make_seq(
            token_ids=prompt + output,
            output_tokens=output,
        )
        result = proposer.propose([seq], k=3)
        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        lens = result.proposal_lens.tolist()
        assert lens[0] == 3
        assert tokens[:3] == [1, 2, 3]


# ===== Suffix index tests =====


class TestSuffixIndexMatchesLinear:
    """Verify suffix index produces same results as linear search."""

    def test_suffix_index_matches_linear(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)

        contexts = [
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9, 5, 6, 1, 2, 3, 5, 6],
            [1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1, 2],
            [1, 2, 9, 1, 2],
            [10, 20, 30, 40, 50],  # no match
        ]

        for ctx in contexts:
            linear_result = proposer._linear_search(ctx, k=5)
            index = NGramProposer.build_suffix_index(ctx, ngram_max=4)
            index_result = proposer._index_search(index, ctx, k=5)
            assert linear_result == index_result, (
                f"Mismatch for context {ctx}: "
                f"linear={linear_result}, index={index_result}"
            )


class TestSuffixIndexIncrementalUpdate:
    """Add tokens, update index incrementally, verify correctness."""

    def test_suffix_index_incremental_update(self):
        # Start with initial context
        initial = [1, 2, 3, 4, 5]
        index = NGramProposer.build_suffix_index(initial, ngram_max=4)

        # Add new tokens to create a pattern match
        full = initial + [1, 2, 3, 4, 5]
        NGramProposer.update_suffix_index(
            index, full, new_token_count=5, ngram_max=4
        )

        # Build fresh index for comparison
        fresh_index = NGramProposer.build_suffix_index(full, ngram_max=4)

        # Both indices should yield the same search result
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        incremental_result = proposer._index_search(index, full, k=3)
        fresh_result = proposer._index_search(fresh_index, full, k=3)
        assert incremental_result == fresh_result

        # Also check consistency with linear search
        linear_result = proposer._linear_search(full, k=3)
        assert incremental_result == linear_result

    def test_suffix_index_multiple_incremental_updates(self):
        """Multiple incremental updates accumulate correctly."""
        tokens = [1, 2, 3]
        index = NGramProposer.build_suffix_index(tokens, ngram_max=3)

        # Add 2 tokens
        tokens = tokens + [4, 5]
        NGramProposer.update_suffix_index(index, tokens, new_token_count=2, ngram_max=3)

        # Add 3 more tokens that create a repeat
        tokens = tokens + [1, 2, 3]
        NGramProposer.update_suffix_index(index, tokens, new_token_count=3, ngram_max=3)

        proposer = NGramProposer(ngram_max=3, ngram_min=1)
        idx_result = proposer._index_search(index, tokens, k=5)
        lin_result = proposer._linear_search(tokens, k=5)
        assert idx_result == lin_result


# ===== Properties and misc =====


class TestProposerProperties:
    """Verify BaseProposer interface properties."""

    def test_needs_draft_probs_false(self):
        proposer = NGramProposer()
        assert proposer.needs_draft_probs is False

    def test_requires_gpu_false(self):
        proposer = NGramProposer()
        assert proposer.requires_gpu is False

    def test_default_parameters(self):
        proposer = NGramProposer()
        assert proposer.ngram_max == 4
        assert proposer.ngram_min == 1
        assert proposer.prompt_lookup is True

    def test_custom_parameters(self):
        proposer = NGramProposer(ngram_max=6, ngram_min=3, prompt_lookup=False)
        assert proposer.ngram_max == 6
        assert proposer.ngram_min == 3
        assert proposer.prompt_lookup is False


class TestProposalResultShape:
    """Verify ProposalResult structure."""

    def test_result_dtypes(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq = _make_seq([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result = proposer.propose([seq], k=3)
        assert result is not None
        assert result.draft_tokens.dtype == mx.int32
        assert result.proposal_lens.dtype == mx.int32
        assert result.draft_probs is None

    def test_result_shape_single(self):
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        seq = _make_seq([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result = proposer.propose([seq], k=3)
        assert result is not None
        assert result.draft_tokens.shape == (1, 3)
        assert result.proposal_lens.shape == (1,)

    def test_padding_with_zeros(self):
        """Sequences with no proposals get zero-padded."""
        proposer = NGramProposer(ngram_max=4, ngram_min=1)
        # seq0 has match, seq1 does not
        seq0 = _make_seq([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        seq1 = _make_seq([10, 20, 30, 40, 50])
        result = proposer.propose([seq0, seq1], k=3)
        assert result is not None
        draft = result.draft_tokens.tolist()
        # seq1's row should be all zeros (padded)
        assert draft[1] == [0, 0, 0]
        assert result.proposal_lens.tolist()[1] == 0
