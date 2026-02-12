"""MTP (Multi-Token Prediction) proposer for speculative decoding.

Uses model-internal MTP layers to generate draft tokens. Unlike the
draft model proposer, MTP reuses the target model's weights and only
needs the hidden states from the target model's last forward pass.

Phase 3: greedy argmax, no draft_probs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx

from mlx_lm_server.spec_decode.proposer.base import BaseProposer, ProposalResult

if TYPE_CHECKING:
    from mlx_lm_server.spec_decode.mtp_module import MTPModule
    from mlx_lm_server.types import SequenceState

logger = logging.getLogger(__name__)


class MTPProposer(BaseProposer):
    """Proposer using model-internal MTP layers.

    The engine calls set_hidden_states() after each target forward pass,
    providing the hidden states needed for the next propose() call.
    On the first step (no hidden states yet), propose() returns None
    and the engine falls back to normal decode.
    """

    def __init__(
        self,
        mtp_module: MTPModule,
        num_mtp_layers: int,
    ):
        self._mtp = mtp_module
        self._num_mtp_layers = num_mtp_layers
        self._cached_hidden: Optional[mx.array] = None  # [B, 1, D]

    def set_hidden_states(self, hidden: mx.array) -> None:
        """Store hidden states from target forward for next propose().

        Called by SpecDecodeEngine after target_forward and verification.

        Args:
            hidden: [B, 1, D] hidden states at the accepted position
                    for each sequence in the batch.
        """
        self._cached_hidden = hidden

    def invalidate_sequence(self, seq_idx: int) -> None:
        """Invalidate cached hidden when batch composition changes.

        When sequences are added/removed from the batch, the cached
        hidden states become invalid since they no longer correspond
        to the current batch.
        """
        # Simple approach: invalidate everything
        # A more sophisticated version could slice out the specific index
        self._cached_hidden = None

    def propose(
        self,
        sequences: List[SequenceState],
        k: int,
    ) -> Optional[ProposalResult]:
        """Generate draft tokens using MTP layers.

        Args:
            sequences: Active sequences in decode phase
            k: Requested number of draft tokens

        Returns:
            ProposalResult or None if no hidden states available.
        """
        if self._cached_hidden is None:
            return None  # First step or after invalidation

        if not sequences or k <= 0:
            return None

        # Clamp k to available MTP layers
        k = min(k, self._num_mtp_layers)

        batch_size = len(sequences)
        hidden = self._cached_hidden  # [B, 1, D]

        # Safety: verify cached hidden matches current batch size
        if self._cached_hidden.shape[0] != len(sequences):
            self._cached_hidden = None
            return None  # Batch size changed, need re-bootstrap

        draft_tokens = []
        next_token: mx.array | None = None

        for depth in range(k):
            # Get the previous token for embedding
            if depth == 0 or next_token is None:
                # Use last token from each sequence
                prev_token_ids = mx.array(
                    [[s.token_ids[-1]] for s in sequences]
                )
            else:
                prev_token_ids = next_token.reshape(batch_size, 1)

            # Get embedding of previous token
            token_embed = self._mtp.get_embed(prev_token_ids)  # [B, 1, D]

            # MTP forward: hidden + token_embed -> new_hidden + logits
            hidden, logits = self._mtp.predict(
                depth, hidden, token_embed
            )

            # Greedy argmax
            next_token = mx.argmax(logits[:, -1, :], axis=-1)  # [B]
            draft_tokens.append(next_token)

        # Evaluate all draft tokens
        mx.eval(*draft_tokens)

        # Stack into [B, k]
        draft_tokens_arr = mx.stack(draft_tokens, axis=1)  # [B, k]

        return ProposalResult(
            draft_tokens=draft_tokens_arr,
            draft_probs=None,  # v1: greedy verification
            proposal_lens=mx.full((batch_size,), k, dtype=mx.int32),
        )

    @property
    def needs_draft_probs(self) -> bool:
        return False  # v1: greedy, rejection sampling in future

    @property
    def requires_gpu(self) -> bool:
        return True
