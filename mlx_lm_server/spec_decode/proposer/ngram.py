"""N-gram proposer for speculative decoding.

Searches the existing context (prompt + generated tokens) for n-gram
pattern matches and proposes continuation tokens. This is a CPU-only
operation with zero GPU overhead.

Effective for:
- Code generation (repetitive patterns, boilerplate)
- Translation (repeated phrases)
- Templated output (JSON, XML, structured formats)
- Summarization (copying from source)

Less effective for:
- Creative writing (unique token sequences)
- Reasoning (novel logical chains)
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import mlx.core as mx

from mlx_lm_server.spec_decode.proposer.base import BaseProposer, ProposalResult

if TYPE_CHECKING:
    from mlx_lm_server.types import SequenceState


class NGramProposer(BaseProposer):
    """N-gram context-matching proposer.

    Attributes:
        ngram_max: Maximum n-gram size to try (inclusive). Default: 4.
        ngram_min: Minimum n-gram size to try (inclusive). Default: 1.
        prompt_lookup: Whether to include prompt tokens in the search
            context. If False, only searches generated tokens.
    """

    def __init__(
        self,
        ngram_max: int = 4,
        ngram_min: int = 1,
        prompt_lookup: bool = True,
    ) -> None:
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.prompt_lookup = prompt_lookup

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False

    def propose(
        self,
        sequences: List[SequenceState],
        k: int,
    ) -> Optional[ProposalResult]:
        """Generate n-gram proposals for all sequences.

        For each sequence, searches its context for the longest n-gram
        match at the current position and proposes up to k continuation
        tokens.

        Args:
            sequences: List of SequenceState with token_ids and
                output_tokens attributes.
            k: Maximum number of draft tokens per sequence.

        Returns:
            ProposalResult with padded draft_tokens [B, max_proposal_len],
            draft_probs=None, and proposal_lens [B].
            Returns None if no sequence produced any proposals.
        """
        batch_proposals: List[List[int]] = []
        proposal_lens: List[int] = []
        any_found = False

        for seq in sequences:
            tokens = self._propose_single(seq, k)
            batch_proposals.append(tokens)
            proposal_lens.append(len(tokens))
            if tokens:
                any_found = True

        if not any_found:
            return None

        # Pad all proposals to the maximum length
        max_len = max(proposal_lens)
        if max_len == 0:
            return None

        padded = []
        for p in batch_proposals:
            padded.append(p + [0] * (max_len - len(p)))

        return ProposalResult(
            draft_tokens=mx.array(padded, dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.array(proposal_lens, dtype=mx.int32),
        )

    def _propose_single(self, seq: SequenceState, k: int) -> List[int]:
        """Generate n-gram proposal for a single sequence.

        Builds the context from prompt + generated tokens (or generated
        only if prompt_lookup is False). Searches from largest n-gram
        down to smallest, most recent match first.

        Args:
            seq: SequenceState with token_ids (full context) and
                output_tokens (generated tokens only).
            k: Maximum number of draft tokens.

        Returns:
            List of proposed token IDs (may be empty).
        """
        # Build search context
        # seq.token_ids = prompt_tokens + output_tokens (maintained by scheduler)
        if self.prompt_lookup:
            context = seq.token_ids
        else:
            context = seq.output_tokens

        if len(context) < self.ngram_min + 1:
            return []

        return self._linear_search(context, k)

    def _linear_search(self, context: List[int], k: int) -> List[int]:
        """Linear scan for n-gram match in context.

        Tries largest n-gram first. For each n-gram size, scans the
        context right-to-left (most recent match preferred).

        Time complexity: O(ngram_max * len(context))
        Acceptable for contexts up to ~2000 tokens. For longer contexts,
        use the suffix index (build_suffix_index + _index_search).

        Args:
            context: Full token ID list (prompt + generated).
            k: Maximum number of tokens to propose.

        Returns:
            List of proposed token IDs (may be shorter than k).
        """
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(context) < n + 1:
                continue

            key = tuple(context[-n:])
            search_end = len(context) - n  # Exclude current position

            # Reverse scan: most recent match is most relevant
            for i in range(search_end - 1, -1, -1):
                if tuple(context[i : i + n]) == key:
                    start = i + n
                    end = min(start + k, len(context))
                    proposals = list(context[start:end])
                    if len(proposals) >= 1:
                        return proposals[:k]

        return []

    @staticmethod
    def build_suffix_index(
        tokens: List[int], ngram_max: int = 4
    ) -> Dict[tuple, List[int]]:
        """Build an n-gram suffix index for O(1) lookup.

        Creates a dictionary mapping n-gram tuples to lists of positions
        where they occur. Called once after prefill, then updated
        incrementally as tokens are generated.

        Args:
            tokens: Full token ID list.
            ngram_max: Maximum n-gram size to index.

        Returns:
            Dict mapping (token_0, ..., token_{n-1}) -> [pos_0, pos_1, ...]
        """
        index: Dict[tuple, List[int]] = {}
        for n in range(1, ngram_max + 1):
            for i in range(len(tokens) - n):
                key = tuple(tokens[i : i + n])
                if key not in index:
                    index[key] = []
                index[key].append(i)
        return index

    @staticmethod
    def update_suffix_index(
        index: Dict[tuple, List[int]],
        tokens: List[int],
        new_token_count: int,
        ngram_max: int = 4,
    ) -> None:
        """Incrementally update suffix index with newly generated tokens.

        Instead of rebuilding the entire index, only processes the
        region affected by new tokens.

        Args:
            index: Existing suffix index to update in-place.
            tokens: Full token list (including new tokens).
            new_token_count: Number of tokens added since last update.
            ngram_max: Maximum n-gram size.
        """
        start = max(0, len(tokens) - new_token_count - ngram_max)
        for n in range(1, ngram_max + 1):
            for i in range(start, len(tokens) - n):
                key = tuple(tokens[i : i + n])
                if key not in index:
                    index[key] = []
                if not index[key] or index[key][-1] != i:
                    index[key].append(i)

    def _index_search(
        self, index: Dict[tuple, List[int]], context: List[int], k: int
    ) -> List[int]:
        """Search for n-gram match using pre-built suffix index.

        O(1) lookup per n-gram size instead of O(len(context)).

        Args:
            index: Pre-built suffix index from build_suffix_index().
            context: Full token ID list.
            k: Maximum number of tokens to propose.

        Returns:
            List of proposed token IDs (may be shorter than k).
        """
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(context) < n + 1:
                continue

            key = tuple(context[-n:])
            if key not in index:
                continue

            positions = index[key]
            # Reverse: most recent position first
            for pos in reversed(positions):
                if pos + n >= len(context):
                    continue  # Skip current position itself
                start = pos + n
                end = min(start + k, len(context))
                proposals = list(context[start:end])
                if len(proposals) >= 1:
                    return proposals[:k]

        return []
