"""SpecDecodeEngine — orchestrates speculative decoding steps.

Works alongside BatchGenerator:
- BatchGenerator handles prefill (prompt processing)
- SpecDecodeEngine handles decode (multi-token generation)
- Both share the same Batch object and its KV cache

The engine does NOT own the model or cache — it borrows them
from BatchGenerator.active_batch for each step.

GATE Result B: per-sequence offset manipulation is UNSAFE.
This engine uses uniform_trim() for cache rollback. When sequences
accept different numbers of tokens, the cache is trimmed to the
minimum accepted count. Sequences that accepted more tokens lose
some cache entries (conservative approach acceptable for Phase 1
n-gram where k is small).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import mlx.core as mx

from mlx_lm_server.spec_decode.cache_utils import uniform_trim
from mlx_lm_server.spec_decode.proposer.base import SpecResponse
from mlx_lm_server.spec_decode.verifier import PLACEHOLDER_TOKEN_ID

if TYPE_CHECKING:
    from mlx_lm_server.spec_decode.config import SpecDecodeConfig
    from mlx_lm_server.spec_decode.controller import DynamicSpecController
    from mlx_lm_server.spec_decode.proposer.base import BaseProposer
    from mlx_lm_server.types import SequenceState


class SpecDecodeEngine:
    """Orchestrates speculative decoding steps.

    Attributes:
        model: Reference to the target model (from BatchGenerator).
        batch_generator: The BatchGenerator instance (for state access).
        proposer: The proposer (n-gram, draft model).
        verifier: The verifier (greedy, threshold, rejection sampler).
        config: SpecDecodeConfig.
        controller: DynamicSpecController for adaptive behavior.
    """

    def __init__(
        self,
        model,
        batch_generator,
        proposer: BaseProposer,
        verifier,
        config: SpecDecodeConfig,
        controller: DynamicSpecController,
    ) -> None:
        self.model = model
        self.batch_generator = batch_generator
        self.proposer = proposer
        self.verifier = verifier
        self.config = config
        self.controller = controller

    def should_speculate(self, batch_size: int) -> bool:
        """Check if spec decode should be used for this step.

        Delegates to DynamicSpecController. Returns False if:
        - Spec decode is disabled (mode="none")
        - Batch size exceeds threshold
        - Acceptance rate EMA is below threshold
        - No active batch in BatchGenerator
        """
        if self.batch_generator.active_batch is None:
            return False
        return self.controller.should_speculate(batch_size)

    def speculative_step(
        self,
        sequences: List[SequenceState],
    ) -> List[SpecResponse]:
        """Execute one speculative decoding step.

        This replaces BatchGenerator.next() for the decode phase.
        The calling code (scheduler) should call this instead of
        batch_generator.next() when spec decode is active.

        Args:
            sequences: List of active SequenceState objects. Must match
                the sequences currently in BatchGenerator.active_batch
                (same UIDs, same order).

        Returns:
            List of SpecResponse, one per sequence. Each contains
            1 to k+1 accepted tokens.
        """
        batch = self.batch_generator.active_batch
        if batch is None:
            return []

        k = self.controller.get_k(len(sequences))
        if k == 0:
            return self._fallback_normal_decode()

        # --- Step 1: PROPOSE ---
        proposal = self.proposer.propose(sequences, k)
        if proposal is None:
            return self._fallback_normal_decode()

        # --- Step 2: BUILD VERIFICATION INPUT ---
        verify_input, verify_lens = self._build_verify_input(
            batch, proposal.draft_tokens, proposal.proposal_lens
        )

        # --- Step 3: TARGET MODEL FORWARD ---
        target_probs = self._target_forward(batch, verify_input)

        # --- Step 4: VERIFY ---
        per_seq_modes = []
        for i in range(len(sequences)):
            temp = getattr(batch.samplers[i], 'temperature', 0.0)
            per_seq_modes.append(self.controller.get_verification_mode(temp))

        accepted_tokens, num_accepted = self.verifier.verify(
            target_probs, proposal.draft_tokens, proposal.proposal_lens,
            modes=per_seq_modes,
        )

        # --- Step 5: CACHE ROLLBACK (Result B: uniform_trim) ---
        max_input_len = verify_input.shape[1]
        self._rollback_cache(batch, num_accepted, max_input_len)

        # --- Step 6: UPDATE BATCH STATE ---
        responses = self._update_batch_state(
            batch, sequences, accepted_tokens, num_accepted,
            proposal.proposal_lens, target_probs,
        )

        # --- Step 7: UPDATE CONTROLLER ---
        total_proposed = int(proposal.proposal_lens.sum())
        total_accepted = int(num_accepted.sum())
        total_bonus = sum(
            1 for i in range(len(sequences))
            if int(num_accepted[i]) >= int(proposal.proposal_lens[i])
            and int(proposal.proposal_lens[i]) > 0
        )
        self.controller.update(total_proposed, total_accepted, total_bonus)

        return responses

    def _build_verify_input(
        self,
        batch,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Build padded verification input for target model.

        For each sequence, the input is:
            [last_token, draft_token_0, ..., draft_token_{plen-1}]

        Sequences with proposal_lens[i] == 0 get just [last_token].
        All sequences are right-padded to the maximum input length.

        Returns:
            Tuple of (verify_input [B, max_input_len], verify_lens [B]).
        """
        batch_size = len(batch.uids)
        last_tokens = batch.y  # [B]

        inputs = []
        input_lens = []
        for i in range(batch_size):
            plen = int(proposal_lens[i])
            if plen == 0:
                inp = [int(last_tokens[i])]
            else:
                inp = [int(last_tokens[i])] + [
                    int(draft_tokens[i, j]) for j in range(plen)
                ]
            inputs.append(inp)
            input_lens.append(len(inp))

        max_len = max(input_lens)

        # Right-pad to max_len
        padded = []
        for inp in inputs:
            padded.append(inp + [0] * (max_len - len(inp)))

        verify_input = mx.array(padded, dtype=mx.int32)
        verify_lens = mx.array(input_lens, dtype=mx.int32)

        return verify_input, verify_lens

    def _target_forward(
        self,
        batch,
        verify_input: mx.array,
    ) -> mx.array:
        """Run target model forward pass with multi-token input.

        Applies per-sequence temperature scaling before softmax.
        top_p and logits_processors are NOT applied in Phase 1.

        Returns:
            target_probs: [B, max_input_len, vocab_size] softmax probs.
        """
        logits = self.model(verify_input, cache=batch.cache)

        # Per-sequence temperature scaling
        batch_size = logits.shape[0]
        for i in range(batch_size):
            temp = getattr(batch.samplers[i], 'temperature', 0.0)
            if temp > 0:
                logits[i] = logits[i] / temp

        target_probs = mx.softmax(logits, axis=-1)
        mx.eval(target_probs)

        return target_probs

    def _rollback_cache(
        self,
        batch,
        num_accepted: mx.array,
        max_input_len: int,
    ) -> None:
        """Roll back KV cache to account for rejected tokens.

        GATE Result B: uses uniform_trim to the minimum accepted count.
        All sequences are trimmed to the same level. Sequences that
        accepted more than the minimum lose some valid cache entries,
        which will be recomputed on the next forward pass.
        """
        min_accepted = int(num_accepted.min())
        keep = min_accepted + 1  # accepted drafts + correction/bonus token
        trim_amount = max_input_len - keep

        if trim_amount > 0:
            uniform_trim(batch.cache, trim_amount)

    def _update_batch_state(
        self,
        batch,
        sequences: List[SequenceState],
        accepted_tokens: mx.array,
        num_accepted: mx.array,
        proposal_lens: mx.array,
        target_probs: mx.array,
    ) -> List[SpecResponse]:
        """Update BatchGenerator's Batch state with accepted tokens.

        After spec step, syncs batch.y, batch.tokens, batch.num_tokens.
        Also checks finish conditions (stop tokens, max_tokens).

        Returns:
            List of SpecResponse, one per sequence.
        """
        responses = []
        min_accepted = int(num_accepted.min())
        # Result B: clamp emitted tokens to min_accepted + 1 since
        # cache was uniformly trimmed to that level.
        max_valid = min_accepted + 1

        for i, seq in enumerate(sequences):
            uid = batch.uids[i]
            n_accepted = int(num_accepted[i])
            plen = int(proposal_lens[i])

            # Extract valid tokens (not PLACEHOLDER)
            row = accepted_tokens[i]  # [k+1]
            valid_tokens = []
            for j in range(row.shape[0]):
                t = int(row[j])
                if t == PLACEHOLDER_TOKEN_ID:
                    break
                valid_tokens.append(t)

            if not valid_tokens:
                valid_tokens = [int(mx.argmax(target_probs[i, 0, :]))]

            # Clamp to what the cache supports after uniform trim
            valid_tokens = valid_tokens[:max_valid]

            # Update batch state — single bulk concatenation per sequence
            valid_arr = mx.array(valid_tokens)
            batch.tokens[i] = mx.concatenate([batch.tokens[i], valid_arr])
            batch.num_tokens[i] += len(valid_tokens)
            batch.y[i] = valid_tokens[-1]

            # Extract logprobs for each accepted token
            token_logprobs = []
            for j, t in enumerate(valid_tokens):
                lp = mx.log(target_probs[i, j, :] + 1e-10)
                token_logprobs.append(lp)

            # Check finish conditions
            finish_reason = None
            for t in valid_tokens:
                if t in self.batch_generator.stop_tokens:
                    finish_reason = "stop"
                    break
            if finish_reason is None and batch.num_tokens[i] >= batch.max_tokens[i]:
                finish_reason = "length"

            responses.append(SpecResponse(
                uid=uid,
                tokens=valid_tokens,
                logprobs=token_logprobs,
                finish_reason=finish_reason,
                prompt_cache=None,
                num_drafted=plen,
                num_accepted=min(n_accepted, min_accepted),
            ))

        # Update batch.y as mx.array
        new_y = mx.array([int(batch.y[i]) for i in range(len(batch.uids))])
        batch.y = new_y

        return responses

    def _fallback_normal_decode(self) -> list:
        """Fall back to BatchGenerator.next() for normal decode.

        Used when proposer returns no proposals or controller disables
        spec decode.
        """
        return self.batch_generator.next()
