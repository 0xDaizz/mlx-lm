"""Base proposer interface and factory function.

All proposer implementations (n-gram, draft model) must inherit
from BaseProposer and implement the propose() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_lm_server.spec_decode.config import SpecDecodeConfig
    from mlx_lm_server.types import SequenceState


@dataclass
class ProposalResult:
    """Output of a proposer's propose() call.

    Attributes:
        draft_tokens: [batch, k] int32 array of proposed token IDs.
            Padded with 0 for sequences with fewer than k proposals.
        draft_probs: [batch, k, vocab] float32 array of draft token
            probabilities. None for n-gram proposer (no draft probs).
            Required for rejection sampling (draft model).
        proposal_lens: [batch] int32 array of actual proposal length
            per sequence. Sequences with proposal_lens[i] == 0 had no
            viable proposals and should use normal decode for that step.
    """

    draft_tokens: mx.array
    draft_probs: Optional[mx.array]
    proposal_lens: mx.array


@dataclass
class SpecResponse:
    """Response from one speculative decoding step for a single sequence.

    Unlike BatchGenerator.Response which contains a single token,
    SpecResponse may contain multiple accepted tokens.

    Attributes:
        uid: Sequence UID (matches BatchGenerator UIDs)
        tokens: List of accepted token IDs (1 to k+1 tokens)
        logprobs: Log probabilities for each accepted token
        finish_reason: "stop", "length", or None if not finished
        prompt_cache: Callable to extract prompt cache (same as Response)
        num_drafted: Number of draft tokens that were proposed
        num_accepted: Number of draft tokens that were accepted
            (not counting the correction/bonus token)
    """

    uid: int
    tokens: List[int]
    logprobs: List[mx.array]
    finish_reason: Optional[str]
    prompt_cache: object
    num_drafted: int
    num_accepted: int


class BaseProposer(ABC):
    """Abstract base class for all speculative decoding proposers.

    Subclasses must implement:
    - propose(): Generate draft tokens for a batch of sequences
    - needs_draft_probs: Whether the proposer provides draft probabilities
    - requires_gpu: Whether the proposer uses Metal GPU
    """

    @abstractmethod
    def propose(
        self,
        sequences: List[SequenceState],
        k: int,
    ) -> Optional[ProposalResult]:
        """Generate draft tokens for all sequences in the batch.

        Args:
            sequences: List of SequenceState objects in decode phase.
                Each has token_ids (prompt + generated), output_tokens,
                and request_id attributes.
            k: Number of draft tokens to generate per sequence.

        Returns:
            ProposalResult with draft tokens and metadata, or None if
            no proposals could be generated for ANY sequence in the batch.
            Individual sequences with no proposals have proposal_lens[i] == 0.
        """
        ...

    @property
    @abstractmethod
    def needs_draft_probs(self) -> bool:
        """True if proposer provides draft_probs for rejection sampling.

        N-gram: False (no probabilities, uses greedy verification)
        Draft model: True (provides draft_probs for rejection sampling)
        """
        ...

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """True if proposer uses Metal GPU (affects resource planning).

        N-gram: False (pure CPU, Python list operations)
        Draft model: True (runs a separate model on GPU)
        """
        ...


def create_proposer(
    config: SpecDecodeConfig,
    target_model: object = None,
    target_tokenizer: object = None,
) -> Optional[BaseProposer]:
    """Factory function to create a proposer based on config.

    Args:
        config: SpecDecodeConfig with mode and proposer-specific settings.
        target_model: The target model (reserved for future use).
        target_tokenizer: The target tokenizer (used for draft model
            vocab compatibility validation).

    Returns:
        A BaseProposer instance, or None if mode is "none".

    Raises:
        ValueError: If the requested mode is not available.
    """
    if config.mode == "none":
        return None
    elif config.mode == "ngram":
        from mlx_lm_server.spec_decode.proposer.ngram import NGramProposer
        return NGramProposer(
            ngram_max=config.ngram_max,
            ngram_min=config.ngram_min,
            prompt_lookup=config.ngram_prompt_lookup,
        )
    elif config.mode == "draft":
        from mlx_lm_server.spec_decode.proposer.draft_model import DraftModelProposer
        return DraftModelProposer(
            model_path=config.draft_model_path,
            context_len=config.draft_context_len,
        )
    else:
        raise ValueError(f"Unknown spec decode mode: {config.mode}")
