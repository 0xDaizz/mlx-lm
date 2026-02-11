"""Draft model proposer for speculative decoding (Phase 2).

Uses a small draft model to generate candidate tokens via greedy
argmax decoding. Draft cache is created fresh per propose() call
and discarded after (D10: no persistence across steps).

Phase 2 scope: greedy draft tokens only, no draft_probs.
Rejection sampling deferred to Phase 3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx

from mlx_lm_server.spec_decode.proposer.base import BaseProposer, ProposalResult

if TYPE_CHECKING:
    from mlx_lm_server.types import SequenceState

logger = logging.getLogger(__name__)


class DraftModelProposer(BaseProposer):
    """Proposer that uses a small draft model for token generation.

    Phase 2: greedy argmax decoding, no draft_probs.
    """

    def __init__(self, model_path: str, context_len: int = 128):
        self.model_path = model_path
        self.context_len = min(context_len, 512)  # D9: clamp at 512
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self, target_tokenizer=None):
        """Load the draft model and optionally validate tokenizer compatibility.

        Args:
            target_tokenizer: If provided, validate vocab_size match (D7).
        """
        from mlx_lm.utils import load as mlx_load

        self.model, self.tokenizer = mlx_load(self.model_path)
        self._loaded = True

        if target_tokenizer is not None:
            if self.tokenizer.vocab_size != target_tokenizer.vocab_size:
                raise ValueError(
                    f"Draft vocab {self.tokenizer.vocab_size} != "
                    f"target vocab {target_tokenizer.vocab_size}"
                )
            draft_eos = getattr(self.tokenizer, "eos_token_id", None)
            target_eos = getattr(target_tokenizer, "eos_token_id", None)
            if draft_eos != target_eos:
                logger.warning(
                    f"Draft eos_token_id={draft_eos} != target eos_token_id={target_eos}"
                )

    def propose(
        self, sequences: List[SequenceState], k: int
    ) -> Optional[ProposalResult]:
        """Generate k draft tokens for each sequence using greedy argmax.

        D10: Fresh cache per step -- create and discard.
        D5: Greedy argmax (not categorical sampling).
        D9: Prefill min(context_len, len(seq.token_ids)) tokens.
        """
        if not self._loaded or self.model is None:
            return None
        if not sequences or k <= 0:
            return None

        all_draft_tokens = []
        all_proposal_lens = []

        for seq in sequences:
            tokens = seq.token_ids
            if not tokens:
                all_draft_tokens.append([0] * k)
                all_proposal_lens.append(0)
                continue

            # D9: use last context_len tokens
            context_len = min(self.context_len, len(tokens))
            context = tokens[-context_len:]

            # Create fresh draft cache (D10)
            from mlx_lm.models.cache import make_prompt_cache

            draft_cache = make_prompt_cache(self.model)

            # Prefill context
            context_mx = mx.array([context])
            if len(context) > 1:
                self.model(context_mx[:, :-1], cache=draft_cache)
                mx.eval([c.state for c in draft_cache])

            # Get logits for last context token
            logits = self.model(context_mx[:, -1:], cache=draft_cache)
            mx.eval(logits)

            # Greedy decode k tokens
            draft_tokens = []
            for step in range(k):
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                draft_tokens.append(next_token)

                if step < k - 1:
                    next_input = mx.array([[next_token]])
                    logits = self.model(next_input, cache=draft_cache)
                    mx.eval(logits)

            all_draft_tokens.append(draft_tokens)
            all_proposal_lens.append(k)
            # Draft cache is discarded here (goes out of scope)

        # Pad to uniform k
        max_k = k
        padded = []
        for dt in all_draft_tokens:
            padded.append(dt + [0] * (max_k - len(dt)))

        return ProposalResult(
            draft_tokens=mx.array(padded, dtype=mx.int32),
            draft_probs=None,  # Phase 2: no draft probs
            proposal_lens=mx.array(all_proposal_lens, dtype=mx.int32),
        )

    @property
    def needs_draft_probs(self) -> bool:
        return False  # Phase 2: greedy, no draft_probs needed

    @property
    def requires_gpu(self) -> bool:
        return True
