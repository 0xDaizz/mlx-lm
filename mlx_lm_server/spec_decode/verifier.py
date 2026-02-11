"""Verification strategies for speculative decoding.

NGramVerifier: Greedy or threshold-based verification for n-gram proposals.
Used when the proposer does not provide draft probabilities.
"""

from __future__ import annotations

import mlx.core as mx

PLACEHOLDER_TOKEN_ID = -1


class NGramVerifier:
    """Verifier for n-gram proposals (no draft probabilities).

    Two modes:
    - "greedy": Accept draft token if it matches target model's argmax.
      This is lossless -- the output is identical to non-speculative
      greedy decoding.
    - "threshold": Accept draft token if target model assigns it
      probability >= threshold. This is lossy but more permissive.

    Attributes:
        mode: "greedy" or "threshold"
        threshold: Minimum target probability for acceptance in
            threshold mode. Ignored in greedy mode.
    """

    def __init__(self, mode: str = "greedy", threshold: float = 0.1) -> None:
        if mode not in ("greedy", "threshold"):
            raise ValueError(f"Unknown verification mode: {mode}")
        self.mode = mode
        self.threshold = threshold

    def verify(
        self,
        target_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
        modes: list[str] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Verify draft tokens against target model probabilities.

        Args:
            target_probs: [batch, max_k+1, vocab_size] probability
                distribution from target model. Position i contains the
                distribution for predicting position i+1 given the context
                up to position i.
            draft_tokens: [batch, k] proposed token IDs from proposer.
            proposal_lens: [batch] actual proposal length per sequence.
            modes: Optional per-sequence verification mode list. When
                provided, applies "greedy" or "threshold" per sequence,
                enabling mixed-mode batches (e.g., temp=0 seqs use greedy,
                temp>0 seqs use threshold). When None, uses self.mode for all.

        Returns:
            Tuple of:
            - accepted_tokens: [batch, k+1] int32 array. Accepted positions
              contain token IDs, rejected positions contain PLACEHOLDER_TOKEN_ID.
              The k+1-th position is the "bonus" or "correction" token.
            - num_accepted: [batch] int32 array. Number of accepted draft
              tokens per sequence (not counting the bonus/correction token).
        """
        if modes is None:
            # Uniform mode for all sequences
            if self.mode == "greedy":
                return self._greedy(target_probs, draft_tokens, proposal_lens)
            else:
                return self._threshold(target_probs, draft_tokens, proposal_lens)

        # Per-sequence mixed mode: split into greedy and threshold groups,
        # apply each, then merge results. If all same mode, fast path.
        unique_modes = set(modes)
        if unique_modes == {"greedy"}:
            return self._greedy(target_probs, draft_tokens, proposal_lens)
        elif unique_modes == {"threshold"}:
            return self._threshold(target_probs, draft_tokens, proposal_lens)
        else:
            # Mixed batch: run both, merge per-sequence
            greedy_out, greedy_n = self._greedy(target_probs, draft_tokens, proposal_lens)
            thresh_out, thresh_n = self._threshold(target_probs, draft_tokens, proposal_lens)
            for i, m in enumerate(modes):
                if m == "threshold":
                    greedy_out[i] = thresh_out[i]
                    greedy_n[i] = thresh_n[i]
            return greedy_out, greedy_n

    def _greedy(
        self,
        target_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Greedy verification: accept iff draft == target argmax.

        This produces IDENTICAL output to non-speculative greedy decoding.
        There is no quality loss.

        Algorithm:
        1. Compute target_argmax = argmax(target_probs[:, :k, :])
        2. match = (draft_tokens == target_argmax) & valid_mask
        3. match_cumulative = cumprod(match) -- reject after first mismatch
        4. At first rejection: use target_argmax as correction token
        5. If all accepted: take argmax(target_probs[:, k, :]) as bonus token
        """
        batch_size, k = draft_tokens.shape

        output = mx.full(
            (batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32
        )

        # Target model's top-1 prediction at each draft position
        target_argmax = mx.argmax(target_probs[:, :k, :], axis=-1)  # [B, k]

        # Match: draft token equals target argmax
        match = draft_tokens == target_argmax  # [B, k]

        # Valid mask: only check positions within proposal_lens
        pos = mx.arange(k)[None, :]                    # [1, k]
        valid = pos < proposal_lens[:, None]            # [B, k]
        match = match & valid

        # Left-to-right: reject everything after first mismatch
        # cumprod trick: [T, T, F, T] -> [T, T, F, F]
        match_cum = mx.cumprod(match.astype(mx.float32), axis=1)
        match_mask = match_cum.astype(mx.bool_)  # [B, k]

        # Fill accepted positions with draft tokens
        output[:, :k] = mx.where(match_mask, draft_tokens, PLACEHOLDER_TOKEN_ID)

        # Count accepted tokens per sequence
        num_accepted = match_mask.astype(mx.int32).sum(axis=1)  # [B]

        # Fill correction/bonus token at position num_accepted[i]
        # For sequences with partial acceptance: correction = target_argmax
        # For sequences with full acceptance: bonus = argmax(target_probs[:, k, :])
        # For sequences with no proposals: normal decode token
        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])

            if plen == 0:
                # No proposal -- use target model's top-1 as the one token
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                # Partial acceptance -- correction token at rejection position
                output[b, n] = int(target_argmax[b, n])
            else:
                # Full acceptance -- bonus token from position after last draft
                output[b, plen] = mx.argmax(target_probs[b, plen, :])

        return output, num_accepted

    def _threshold(
        self,
        target_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Threshold verification: accept if target assigns >= threshold prob.

        More permissive than greedy but introduces quality loss.
        Useful when temperature > 0 and exact argmax match is too strict.

        Algorithm:
        1. Extract target_prob[draft_token] for each position
        2. accepted = (target_prob >= threshold) & valid_mask
        3. accepted_cumulative = cumprod(accepted)
        4. Correction/bonus same as greedy
        """
        batch_size, k = draft_tokens.shape

        output = mx.full(
            (batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32
        )

        # Extract target model's probability for each draft token
        target_p = mx.take_along_axis(
            target_probs[:, :k, :], draft_tokens[:, :, None], axis=2
        ).squeeze(-1)  # [B, k]

        accepted = target_p >= self.threshold

        pos = mx.arange(k)[None, :]
        valid = pos < proposal_lens[:, None]
        accepted = accepted & valid

        accepted_cum = mx.cumprod(accepted.astype(mx.float32), axis=1)
        accepted_mask = accepted_cum.astype(mx.bool_)

        output[:, :k] = mx.where(accepted_mask, draft_tokens, PLACEHOLDER_TOKEN_ID)
        num_accepted = accepted_mask.astype(mx.int32).sum(axis=1)

        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])

            if plen == 0:
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                output[b, n] = mx.argmax(target_probs[b, n, :])
            else:
                output[b, plen] = mx.argmax(target_probs[b, plen, :])

        return output, num_accepted
