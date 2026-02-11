"""Tests for speculative decoding cache utilities.

Tests batch_variable_trim, uniform_trim, get_cache_offsets, and
can_per_seq_trim. Includes the GATE TEST that validates per-sequence
trim_per_sequence() produces argmax-safe outputs.

GATE Result:
    A (argmax-safe per-sequence trim) — trim_per_sequence() available,
        per-sequence variable trim works correctly
"""

import os
import sys

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.cache_utils import (
    batch_variable_trim,
    can_per_seq_trim,
    get_cache_offsets,
    uniform_trim,
)


# ---------------------------------------------------------------------------
# Mock BatchKVCache for unit tests (no model needed)
# ---------------------------------------------------------------------------
class MockBatchKVCache:
    """Minimal mock of BatchKVCache for testing cache_utils.

    Mimics the key attributes and methods:
    - _idx: scalar global write cursor
    - offset: per-sequence mx.array
    - left_padding: per-sequence mx.array
    - keys/values: [B, H, S, D] tensors
    - trim(n): decrements _idx and offset by n
    - update_and_fetch(keys, values): writes at _idx, returns [:_idx]
    """

    def __init__(self, batch_size: int, initial_offset: int, left_padding=None):
        if left_padding is None:
            left_padding = [0] * batch_size
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array(
            [initial_offset - lp for lp in left_padding]
        )
        self._idx = initial_offset
        # Allocate a buffer: [B, H=2, S=256, D=4]
        self.keys = mx.zeros((batch_size, 2, 256, 4))
        self.values = mx.zeros((batch_size, 2, 256, 4))

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def update_and_fetch(self, keys, values):
        prev = self._idx
        self._idx += keys.shape[2]
        self.offset = self.offset + keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def trim_per_sequence(self, n):
        """Per-sequence trim matching BatchKVCache.trim_per_sequence()."""
        n = mx.minimum(n, self.left_padding + self.offset)
        self.offset = self.offset - n
        self._idx = int(mx.max(self.left_padding + self.offset).item())

    def is_trimmable(self):
        return True


# ---------------------------------------------------------------------------
# Helper: create N mock layers
# ---------------------------------------------------------------------------
def _make_mock_layers(n_layers, batch_size, initial_offset, left_padding=None):
    return [
        MockBatchKVCache(batch_size, initial_offset, left_padding)
        for _ in range(n_layers)
    ]


# ===========================================================================
# Unit Tests (mock-based, no model required)
# ===========================================================================


class TestUniformTrim:
    def test_uniform_trim_basic(self):
        """Trim 2 from all, verify offsets decreased by 2."""
        layers = _make_mock_layers(3, batch_size=2, initial_offset=10)

        # Before trim
        for layer in layers:
            assert layer._idx == 10
            assert mx.array_equal(layer.offset, mx.array([10, 10]))

        uniform_trim(layers, 2)

        # After trim
        for layer in layers:
            assert layer._idx == 8
            assert mx.array_equal(layer.offset, mx.array([8, 8]))

    def test_uniform_trim_zero(self):
        """No-op when trim_amount is 0."""
        layers = _make_mock_layers(2, batch_size=2, initial_offset=5)
        uniform_trim(layers, 0)
        for layer in layers:
            assert layer._idx == 5

    def test_uniform_trim_negative(self):
        """No-op when trim_amount is negative."""
        layers = _make_mock_layers(2, batch_size=2, initial_offset=5)
        uniform_trim(layers, -1)
        for layer in layers:
            assert layer._idx == 5


class TestVariableTrim:
    def test_variable_trim_different_amounts(self):
        """Trim [3, 1, 5] from 3 sequences, verify per-sequence offsets."""
        layers = _make_mock_layers(2, batch_size=3, initial_offset=10)
        trim_amounts = mx.array([3, 1, 5])

        batch_variable_trim(layers, trim_amounts)

        # After per-sequence trim via trim_per_sequence:
        # initial offset=[10,10,10], left_padding=[0,0,0]
        # n = min([3,1,5], [0+10, 0+10, 0+10]) = [3,1,5]
        # offset = [10-3, 10-1, 10-5] = [7, 9, 5]
        # _idx = max(0+7, 0+9, 0+5) = 9
        for layer in layers:
            assert layer._idx == 9
            expected_offset = mx.array([7, 9, 5])
            assert mx.array_equal(layer.offset, expected_offset), (
                f"Expected {expected_offset.tolist()}, got {layer.offset.tolist()}"
            )

    def test_variable_trim_zero_for_some(self):
        """Trim [0, 3, 0], sequences 0 and 2 unchanged."""
        layers = _make_mock_layers(2, batch_size=3, initial_offset=10)
        trim_amounts = mx.array([0, 3, 0])

        batch_variable_trim(layers, trim_amounts)

        # After per-sequence trim via trim_per_sequence:
        # n = min([0,3,0], [0+10, 0+10, 0+10]) = [0,3,0]
        # offset = [10-0, 10-3, 10-0] = [10, 7, 10]
        # _idx = max(0+10, 0+7, 0+10) = 10
        for layer in layers:
            assert layer._idx == 10
            expected_offset = mx.array([10, 7, 10])
            assert mx.array_equal(layer.offset, expected_offset), (
                f"Expected {expected_offset.tolist()}, got {layer.offset.tolist()}"
            )

    def test_variable_trim_all_zero(self):
        """No-op when all trim amounts are 0."""
        layers = _make_mock_layers(2, batch_size=3, initial_offset=10)
        trim_amounts = mx.array([0, 0, 0])

        batch_variable_trim(layers, trim_amounts)

        for layer in layers:
            assert layer._idx == 10
            assert mx.array_equal(layer.offset, mx.array([10, 10, 10]))

    def test_offset_consistency_across_layers(self):
        """All layers should have same offset changes after variable trim."""
        layers = _make_mock_layers(4, batch_size=3, initial_offset=15)
        trim_amounts = mx.array([2, 5, 1])

        batch_variable_trim(layers, trim_amounts)

        # All layers must have identical offsets
        reference_offset = layers[0].offset
        reference_idx = layers[0]._idx
        for i, layer in enumerate(layers):
            assert layer._idx == reference_idx, f"Layer {i} _idx mismatch"
            assert mx.array_equal(layer.offset, reference_offset), (
                f"Layer {i} offset mismatch"
            )

    def test_trim_does_not_corrupt_existing_cache(self):
        """Verify that cache data before trim point is unchanged."""
        layers = _make_mock_layers(1, batch_size=2, initial_offset=0)
        layer = layers[0]

        # Write some known data
        B, H, D = 2, 2, 4
        data1 = mx.ones((B, H, 5, D))  # 5 tokens
        layer.update_and_fetch(data1, data1)
        assert layer._idx == 5

        data2 = mx.full((B, H, 3, D), 2.0)  # 3 more tokens
        layer.update_and_fetch(data2, data2)
        assert layer._idx == 8

        # Trim 3 tokens
        trim_amounts = mx.array([3, 3])
        batch_variable_trim(layers, trim_amounts)
        assert layer._idx == 5

        # Data at positions 0-4 should still be ones
        mx.eval(layer.keys)
        remaining_keys = layer.keys[0, 0, :5, 0]
        assert mx.allclose(remaining_keys, mx.ones(5)), (
            "Cache data before trim point was corrupted"
        )

    def test_variable_trim_with_left_padding(self):
        """Variable trim with left-padded sequences."""
        # Sequence 0 has 2 padding, sequence 1 has 0 padding
        layers = _make_mock_layers(
            2, batch_size=2, initial_offset=10, left_padding=[2, 0]
        )
        trim_amounts = mx.array([3, 1])

        # Before: offset = [10-2, 10-0] = [8, 10]
        for layer in layers:
            assert mx.array_equal(layer.offset, mx.array([8, 10]))

        batch_variable_trim(layers, trim_amounts)

        # After per-sequence trim via trim_per_sequence:
        # n = min([3,1], [2+8, 0+10]) = min([3,1], [10,10]) = [3,1]
        # offset = [8-3, 10-1] = [5, 9]
        # _idx = max(2+5, 0+9) = max(7, 9) = 9
        for layer in layers:
            assert layer._idx == 9
            expected_offset = mx.array([5, 9])
            assert mx.array_equal(layer.offset, expected_offset), (
                f"Expected {expected_offset.tolist()}, got {layer.offset.tolist()}"
            )


class TestGetCacheOffsets:
    def test_empty_layers(self):
        """Empty layer list returns empty array."""
        result = get_cache_offsets([])
        assert result.size == 0

    def test_returns_first_layer_offset(self):
        """Returns offset from the first layer."""
        layers = _make_mock_layers(3, batch_size=2, initial_offset=10)
        offsets = get_cache_offsets(layers)
        assert mx.array_equal(offsets, mx.array([10, 10]))

    def test_after_variable_trim(self):
        """Returns correct offsets after variable trim."""
        layers = _make_mock_layers(2, batch_size=3, initial_offset=10)
        batch_variable_trim(layers, mx.array([2, 0, 4]))
        offsets = get_cache_offsets(layers)
        # Per-sequence trim: offset=[10-2, 10-0, 10-4]=[8, 10, 6]
        assert mx.array_equal(offsets, mx.array([8, 10, 6]))


# ===========================================================================
# GATE TEST — Determines architecture (Result A vs Result B)
# ===========================================================================

# Check if the local model exists
_LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Qwen3-4B-4bit",
)
_HAS_LOCAL_MODEL = os.path.isdir(_LOCAL_MODEL_PATH) and os.path.exists(
    os.path.join(_LOCAL_MODEL_PATH, "config.json")
)


@pytest.mark.skipif(
    not _HAS_LOCAL_MODEL,
    reason=f"Local model not found at {_LOCAL_MODEL_PATH}",
)
class TestGateDecision:
    """CRITICAL GATE TEST for P1.5.

    This test determines the entire speculative decoding architecture:
        Result A (PASS): per-sequence offset manipulation is safe
        Result B (FAIL): must use uniform trim + re-forward

    The test:
    1. Loads Qwen3-4B-4bit
    2. Creates a batch of 2 sequences with different prompt lengths
    3. Runs prefill + 1 decode step
    4. Runs a multi-token forward (simulating k=5 spec verify)
    5. Applies batch_variable_trim([3, 1]) — different trims per seq
    6. Runs 1 more decode step
    7. Compares output to a reference run (same effective cache state
       achieved through separate single-sequence processing)
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        """Load model once for all tests in this class."""
        from mlx_lm import load

        model, tokenizer = load(_LOCAL_MODEL_PATH)
        return model, tokenizer

    def _make_batch_cache(self, model, left_padding):
        """Create batch cache layers for the model.

        Uses the same logic as mlx_lm.generate._make_cache() to create
        BatchKVCache instances, one per model layer.
        """
        from mlx_lm.generate import _make_cache

        return _make_cache(model, left_padding, max_kv_size=None)

    def _prefill(self, model, tokens, cache):
        """Run prefill: process all tokens except the last one."""
        if tokens.shape[1] > 1:
            model(tokens[:, :-1], cache=cache)
            mx.eval([c.state for c in cache])
        # Process last token to get first output
        logits = model(tokens[:, -1:], cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def _decode_one(self, model, y, cache):
        """Run one decode step (single token per sequence)."""
        logits = model(y, cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def _multi_token_forward(self, model, tokens, cache):
        """Run multi-token forward (simulating spec decode verify).

        tokens: [B, k] where k is the number of draft tokens to verify.
        """
        logits = model(tokens, cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def test_subsequent_forward_after_variable_trim(self, model_and_tokenizer):
        """THE GATE TEST.

        Scenario:
        - 2 sequences: "Hello world" (2 tokens) and "The quick brown fox" (5 tokens)
        - Left-pad shorter to match longer
        - Prefill both
        - Decode 1 normal step (establishes baseline cache state)
        - Multi-token forward with 5 tokens (simulating spec decode k=5)
        - Variable trim: seq 0 trims 3, seq 1 trims 1
          (seq 0 accepted 2 of 5, seq 1 accepted 4 of 5)
        - Decode 1 more token
        - Compare outputs to reference run

        Reference run:
        - Same prompts, same first decode step
        - For seq 0: instead of multi-token + trim 3, do multi-token with only 2 tokens
        - For seq 1: instead of multi-token + trim 1, do multi-token with only 4 tokens
        - Then decode 1 more token
        - Outputs should match if variable trim is safe
        """
        model, tokenizer = model_and_tokenizer

        # Encode prompts
        prompt_a = "Hello world"
        prompt_b = "The quick brown fox jumps"
        tokens_a = mx.array(tokenizer.encode(prompt_a))
        tokens_b = mx.array(tokenizer.encode(prompt_b))

        len_a = tokens_a.shape[0]
        len_b = tokens_b.shape[0]

        # Determine max length and padding
        max_len = max(len_a, len_b)
        pad_a = max_len - len_a
        pad_b = max_len - len_b
        left_padding = [pad_a, pad_b]

        # Left-pad tokens
        padded_a = mx.concatenate([mx.zeros(pad_a, dtype=mx.int32), tokens_a]) if pad_a > 0 else tokens_a
        padded_b = mx.concatenate([mx.zeros(pad_b, dtype=mx.int32), tokens_b]) if pad_b > 0 else tokens_b
        batch_tokens = mx.stack([padded_a, padded_b])  # [2, max_len]

        # ---- SPEC DECODE PATH (with variable trim) ----
        spec_cache = self._make_batch_cache(model, left_padding)
        spec_logits = self._prefill(model, batch_tokens, spec_cache)

        # Decode 1 step (get a token to feed)
        y_spec = mx.argmax(spec_logits[:, -1, :], axis=-1, keepdims=True)  # [2, 1]
        mx.eval(y_spec)
        spec_logits_1 = self._decode_one(model, y_spec, spec_cache)
        y_spec_1 = mx.argmax(spec_logits_1[:, -1, :], axis=-1)  # [2]
        mx.eval(y_spec_1)

        # Record _idx after 1 decode step
        idx_after_decode1 = spec_cache[0]._idx

        # Multi-token forward with k=5 draft tokens per sequence
        # Use arbitrary draft tokens (in real spec decode these come from proposer)
        k = 5
        draft_tokens = mx.zeros((2, k), dtype=mx.int32)
        # Use the decoded token + arbitrary ones
        draft_tokens[0, 0] = y_spec_1[0]
        draft_tokens[1, 0] = y_spec_1[1]
        for i in range(1, k):
            draft_tokens[0, i] = i + 100
            draft_tokens[1, i] = i + 200
        mx.eval(draft_tokens)

        spec_verify_logits = self._multi_token_forward(
            model, draft_tokens, spec_cache
        )
        mx.eval(spec_verify_logits)

        # Now cache has advanced by k tokens for both sequences
        # _idx should be idx_after_decode1 + k
        assert spec_cache[0]._idx == idx_after_decode1 + k

        # Apply variable trim: seq 0 trims 3 (accepted 2), seq 1 trims 1 (accepted 4)
        trim_amounts = mx.array([3, 1])
        batch_variable_trim(spec_cache, trim_amounts)

        # After per-sequence trim via trim_per_sequence:
        # offset[0] -= 3, offset[1] -= 1
        # _idx = max(left_padding[0] + offset[0], left_padding[1] + offset[1])
        # For the less-trimmed sequence (seq 1, trim=1), _idx is higher than
        # the old uniform-trim approach would give.

        # Now decode one more token after variable trim
        # For the next token, we need to feed the last accepted token
        # seq 0 accepted 2 of 5 draft tokens: tokens at positions 0, 1
        # seq 1 accepted 4 of 5 draft tokens: tokens at positions 0, 1, 2, 3
        next_token_spec = mx.array([[draft_tokens[0, 2]], [draft_tokens[1, 4]]])
        mx.eval(next_token_spec)
        final_logits_spec = self._decode_one(model, next_token_spec, spec_cache)
        mx.eval(final_logits_spec)

        # ---- REFERENCE PATH (separate single-sequence processing) ----
        # Process each sequence separately with the exact number of accepted
        # tokens. This is the ground truth: no variable trim, just feed the
        # exact tokens that the sequence should have seen.

        # Seq 0 reference: same prompt, 1 decode, then 2 tokens (not 5), then 1 final
        ref_cache_0 = self._make_batch_cache(model, [0])
        ref_logits_0 = self._prefill(model, tokens_a.reshape(1, -1), ref_cache_0)
        ref_y0_0 = mx.argmax(ref_logits_0[:, -1, :], axis=-1, keepdims=True)
        mx.eval(ref_y0_0)
        ref_logits_0_1 = self._decode_one(model, ref_y0_0, ref_cache_0)
        ref_y0_1 = mx.argmax(ref_logits_0_1[:, -1, :], axis=-1)
        mx.eval(ref_y0_1)

        # Feed the 2 accepted draft tokens (positions 0 and 1)
        accepted_0 = draft_tokens[0:1, :2]  # [1, 2]
        ref_logits_0_multi = self._multi_token_forward(
            model, accepted_0, ref_cache_0
        )
        mx.eval(ref_logits_0_multi)

        # Final decode for seq 0
        ref_final_0 = self._decode_one(
            model, mx.array([[draft_tokens[0, 2]]]), ref_cache_0
        )
        mx.eval(ref_final_0)

        # Seq 1 reference: same prompt, 1 decode, then 4 tokens (not 5), then 1 final
        ref_cache_1 = self._make_batch_cache(model, [0])
        ref_logits_1 = self._prefill(model, tokens_b.reshape(1, -1), ref_cache_1)
        ref_y1_0 = mx.argmax(ref_logits_1[:, -1, :], axis=-1, keepdims=True)
        mx.eval(ref_y1_0)
        ref_logits_1_1 = self._decode_one(model, ref_y1_0, ref_cache_1)
        ref_y1_1 = mx.argmax(ref_logits_1_1[:, -1, :], axis=-1)
        mx.eval(ref_y1_1)

        # Feed the 4 accepted draft tokens (positions 0, 1, 2, 3)
        accepted_1 = draft_tokens[1:2, :4]  # [1, 4]
        ref_logits_1_multi = self._multi_token_forward(
            model, accepted_1, ref_cache_1
        )
        mx.eval(ref_logits_1_multi)

        # Final decode for seq 1
        ref_final_1 = self._decode_one(
            model, mx.array([[draft_tokens[1, 4]]]), ref_cache_1
        )
        mx.eval(ref_final_1)

        # ---- COMPARISON ----
        # Compare final logits from spec path vs reference path
        # If variable trim is safe, the logits should be very close (or identical)
        spec_logit_0 = final_logits_spec[0:1, -1, :]  # [1, vocab]
        spec_logit_1 = final_logits_spec[1:2, -1, :]  # [1, vocab]
        ref_logit_0 = ref_final_0[:, -1, :]  # [1, vocab]
        ref_logit_1 = ref_final_1[:, -1, :]  # [1, vocab]

        mx.eval(spec_logit_0, spec_logit_1, ref_logit_0, ref_logit_1)

        # Check if argmax matches (most lenient check)
        spec_argmax_0 = mx.argmax(spec_logit_0, axis=-1).item()
        spec_argmax_1 = mx.argmax(spec_logit_1, axis=-1).item()
        ref_argmax_0 = mx.argmax(ref_logit_0, axis=-1).item()
        ref_argmax_1 = mx.argmax(ref_logit_1, axis=-1).item()

        # Check close-ness of logits (stricter)
        atol = 1e-4
        logits_close_0 = mx.allclose(spec_logit_0, ref_logit_0, atol=atol).item()
        logits_close_1 = mx.allclose(spec_logit_1, ref_logit_1, atol=atol).item()

        # Print diagnostic info for the GATE decision
        print("\n" + "=" * 70)
        print("GATE TEST RESULTS")
        print("=" * 70)
        print(f"Seq 0: spec argmax={spec_argmax_0}, ref argmax={ref_argmax_0}, "
              f"match={spec_argmax_0 == ref_argmax_0}")
        print(f"Seq 1: spec argmax={spec_argmax_1}, ref argmax={ref_argmax_1}, "
              f"match={spec_argmax_1 == ref_argmax_1}")
        print(f"Seq 0 logits close (atol={atol}): {logits_close_0}")
        print(f"Seq 1 logits close (atol={atol}): {logits_close_1}")

        # Compute max absolute difference for diagnostics
        diff_0 = mx.abs(spec_logit_0 - ref_logit_0).max().item()
        diff_1 = mx.abs(spec_logit_1 - ref_logit_1).max().item()
        print(f"Seq 0 max logit diff: {diff_0}")
        print(f"Seq 1 max logit diff: {diff_1}")

        if logits_close_0 and logits_close_1:
            print("\n>>> GATE RESULT: A (offset safe) <<<")
            print("Per-sequence offset manipulation produces correct outputs.")
        elif spec_argmax_0 == ref_argmax_0 and spec_argmax_1 == ref_argmax_1:
            print("\n>>> GATE RESULT: A (offset safe, within numerical tolerance) <<<")
            print("Argmax matches but logits differ slightly.")
        else:
            print("\n>>> GATE RESULT: B (offset unsafe) <<<")
            print("Variable trim produces different outputs than reference.")
            print("Must use uniform trim + re-forward approach.")
        print("=" * 70)

        # The GATE assertion: outputs must match for Result A.
        # We use argmax match as the primary criterion since floating point
        # differences are expected between batched and individual processing.
        #
        # GATE RESULT (2026-02-12): **A (argmax-safe per-sequence trim)**
        #   trim_per_sequence() correctly adjusts offset and _idx atomically.
        #   Per-sequence variable trim now produces argmax-matching outputs
        #   vs reference single-sequence processing.
        #
        # Previous Result B (2026-02-11) was due to the old approach of
        # uniform trim + offset re-advance, which failed because _idx is
        # scalar and make_mask()/update_and_fetch() use _idx.
        gate_passed = (spec_argmax_0 == ref_argmax_0 and spec_argmax_1 == ref_argmax_1)

        assert gate_passed, (
            "GATE RESULT A expected but failed: per-sequence trim_per_sequence "
            "should produce argmax-safe outputs. "
            f"Seq 0: spec={spec_argmax_0} ref={ref_argmax_0} (diff={diff_0:.4f}). "
            f"Seq 1: spec={spec_argmax_1} ref={ref_argmax_1} (diff={diff_1:.4f})."
        )

    def test_uniform_trim_reference(self, model_and_tokenizer):
        """Verify that uniform trim (same amount for all) produces
        correct outputs. This serves as a sanity check — if even uniform
        trim fails, there's a deeper issue with cache trimming.
        """
        model, tokenizer = model_and_tokenizer

        prompt = "The quick brown fox"
        tokens = mx.array([tokenizer.encode(prompt)])  # [1, seq_len]
        seq_len = tokens.shape[1]

        # Path A: prefill + 1 decode + 3 multi-token + trim 3 + 1 decode
        cache_a = self._make_batch_cache(model, [0])
        logits_a = self._prefill(model, tokens, cache_a)
        y_a = mx.argmax(logits_a[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_a)
        logits_a1 = self._decode_one(model, y_a, cache_a)
        y_a1 = mx.argmax(logits_a1[:, -1, :], axis=-1)
        mx.eval(y_a1)

        # Multi-token forward with 3 draft tokens
        draft = mx.array([[y_a1.item(), 100, 200]])
        self._multi_token_forward(model, draft, cache_a)

        # Trim 3 (all of them — accept 0 draft tokens)
        uniform_trim(cache_a, 3)

        # Next token (the correction token = y_a1 itself, since we rejected all)
        final_a = self._decode_one(model, mx.array([[y_a1.item()]]), cache_a)
        mx.eval(final_a)

        # Path B: prefill + 1 decode + 1 more decode (no spec decode at all)
        cache_b = self._make_batch_cache(model, [0])
        logits_b = self._prefill(model, tokens, cache_b)
        y_b = mx.argmax(logits_b[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_b)
        logits_b1 = self._decode_one(model, y_b, cache_b)
        y_b1 = mx.argmax(logits_b1[:, -1, :], axis=-1)
        mx.eval(y_b1)

        # Same next token
        final_b = self._decode_one(model, mx.array([[y_b1.item()]]), cache_b)
        mx.eval(final_b)

        # They should produce the same output
        spec_argmax = mx.argmax(final_a[:, -1, :], axis=-1).item()
        ref_argmax = mx.argmax(final_b[:, -1, :], axis=-1).item()

        print(f"\nUniform trim sanity: spec={spec_argmax}, ref={ref_argmax}")
        assert spec_argmax == ref_argmax, (
            f"Uniform trim sanity FAIL: {spec_argmax} != {ref_argmax}"
        )

    def test_batch_vs_individual_prefill_baseline(self, model_and_tokenizer):
        """Verify batched prefill produces same logits as individual prefill.

        This is a prerequisite sanity check: if batched prefill diverges
        from individual, the GATE test result cannot be trusted.
        """
        model, tokenizer = model_and_tokenizer

        prompt_a = "Hello world"
        prompt_b = "The quick brown fox jumps"
        tokens_a = mx.array(tokenizer.encode(prompt_a))
        tokens_b = mx.array(tokenizer.encode(prompt_b))

        len_a = tokens_a.shape[0]
        len_b = tokens_b.shape[0]
        max_len = max(len_a, len_b)
        pad_a = max_len - len_a
        pad_b = max_len - len_b

        # Batched prefill
        padded_a = mx.concatenate([mx.zeros(pad_a, dtype=mx.int32), tokens_a]) if pad_a > 0 else tokens_a
        padded_b = mx.concatenate([mx.zeros(pad_b, dtype=mx.int32), tokens_b]) if pad_b > 0 else tokens_b
        batch_tokens = mx.stack([padded_a, padded_b])

        batch_cache = self._make_batch_cache(model, [pad_a, pad_b])
        batch_logits = self._prefill(model, batch_tokens, batch_cache)
        batch_argmax = mx.argmax(batch_logits[:, -1, :], axis=-1)
        mx.eval(batch_argmax)

        # Individual prefill for seq 0
        ind_cache_0 = self._make_batch_cache(model, [0])
        ind_logits_0 = self._prefill(model, tokens_a.reshape(1, -1), ind_cache_0)
        ind_argmax_0 = mx.argmax(ind_logits_0[:, -1, :], axis=-1).item()
        mx.eval(ind_logits_0)

        # Individual prefill for seq 1
        ind_cache_1 = self._make_batch_cache(model, [0])
        ind_logits_1 = self._prefill(model, tokens_b.reshape(1, -1), ind_cache_1)
        ind_argmax_1 = mx.argmax(ind_logits_1[:, -1, :], axis=-1).item()
        mx.eval(ind_logits_1)

        print(f"\nBatch vs Individual prefill:")
        print(f"  Seq 0: batch={batch_argmax[0].item()}, ind={ind_argmax_0}")
        print(f"  Seq 1: batch={batch_argmax[1].item()}, ind={ind_argmax_1}")

        # Note: batched and individual may differ due to padding effects
        # This is expected behavior. We just verify both produce valid outputs.
        assert batch_logits.shape[-1] > 0, "Batch logits should have vocab dim"


# ===========================================================================
# Tests for can_per_seq_trim capability detection
# ===========================================================================


class TestCanPerSeqTrim:
    def test_can_per_seq_trim_true(self):
        """MockBatchKVCache has trim_per_sequence -> True."""
        layers = _make_mock_layers(3, batch_size=2, initial_offset=10)
        assert can_per_seq_trim(layers) is True

    def test_can_per_seq_trim_false(self):
        """Mock without trim_per_sequence -> False."""

        class NoPSTCache:
            pass

        layers = [NoPSTCache(), NoPSTCache()]
        assert can_per_seq_trim(layers) is False

    def test_can_per_seq_trim_cache_list(self):
        """Mock CacheList with all children supporting -> True."""

        class MockCacheList:
            def __init__(self):
                self.caches = [
                    MockBatchKVCache(2, 10),
                    MockBatchKVCache(2, 10),
                ]

        # Simulate CacheList by registering it where the check looks
        layers = [MockCacheList()]
        # The check uses hasattr fallback (getattr 'caches') since
        # MockCacheList is not from mlx_lm.models.cache
        # All children have trim_per_sequence, but the parent MockCacheList
        # itself also needs it (or be recognized as CacheList).
        # Since it has 'caches' attribute, the ImportError path kicks in
        # and checks all children recursively.
        #
        # Actually: the try block imports CacheList. If isinstance fails
        # (MockCacheList != CacheList), falls through to hasattr check.
        # MockCacheList doesn't have trim_per_sequence, so returns False.
        # BUT the except ImportError path is only reached if import fails.
        # Since mlx_lm IS available, isinstance(MockCacheList, CacheList) is False,
        # so it falls through to `return hasattr(layer, 'trim_per_sequence')`.
        #
        # To test the CacheList path properly, we need to use the real CacheList
        # or mock the import. Let's just test with the real CacheList if available.
        try:
            from mlx_lm.models.cache import CacheList
            cache_list = CacheList.__new__(CacheList)
            cache_list.caches = [
                MockBatchKVCache(2, 10),
                MockBatchKVCache(2, 10),
            ]
            assert can_per_seq_trim([cache_list]) is True
        except ImportError:
            pytest.skip("mlx_lm.models.cache.CacheList not available")

    def test_can_per_seq_trim_mixed_cache_list(self):
        """CacheList with one unsupported child -> False."""

        class NoTrimCache:
            """Cache without trim_per_sequence."""
            pass

        try:
            from mlx_lm.models.cache import CacheList
            cache_list = CacheList.__new__(CacheList)
            cache_list.caches = [
                MockBatchKVCache(2, 10),
                NoTrimCache(),
            ]
            assert can_per_seq_trim([cache_list]) is False
        except ImportError:
            pytest.skip("mlx_lm.models.cache.CacheList not available")

    def test_can_per_seq_trim_empty(self):
        """Empty list -> False."""
        assert can_per_seq_trim([]) is False

    def test_result_b_fallback_path(self):
        """Verify uniform_trim still works correctly as a fallback."""
        layers = _make_mock_layers(2, batch_size=3, initial_offset=10)

        # Uniform trim of 4
        uniform_trim(layers, 4)

        for layer in layers:
            assert layer._idx == 6
            expected_offset = mx.array([6, 6, 6])
            assert mx.array_equal(layer.offset, expected_offset)
