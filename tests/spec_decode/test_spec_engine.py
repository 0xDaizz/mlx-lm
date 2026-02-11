"""Tests for SpecDecodeEngine — speculative decoding orchestrator.

Group A: Logic tests with mocked BatchGenerator/Batch (no real model).
Group B: Integration tests with real model (skipped if model not available).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import MagicMock

import mlx.core as mx

from mlx_lm_server.spec_decode.config import SpecDecodeConfig
from mlx_lm_server.spec_decode.controller import DynamicSpecController
from mlx_lm_server.spec_decode.engine import SpecDecodeEngine
from mlx_lm_server.spec_decode.proposer.base import (
    BaseProposer,
    ProposalResult,
    SpecResponse,
)
from mlx_lm_server.spec_decode.verifier import NGramVerifier, PLACEHOLDER_TOKEN_ID


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockSampler:
    """Sampler mock with configurable temperature."""

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature


class MockCacheLayer:
    """Minimal cache layer mock that tracks trim calls.

    Supports both uniform trim() and per-sequence trim_per_sequence().
    """

    def __init__(self, offset: int = 10, batch_size: int = 1):
        self._offset = offset
        self._batch_size = batch_size
        self.left_padding = mx.zeros((batch_size,), dtype=mx.int32)
        self.offset = mx.array([offset] * batch_size, dtype=mx.int32)
        self.trim_calls: list[int] = []
        self.trim_per_seq_calls: list = []

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
    """Minimal Batch mock matching mlx_lm.generate.Batch interface."""

    uids: List[int]
    y: mx.array
    tokens: List[mx.array]
    num_tokens: List[int]
    max_tokens: List[int]
    cache: List[Any]
    samplers: List[Any]
    logits_processors: List[Any] = field(default_factory=list)
    logprobs: List[Any] = field(default_factory=list)


class MockBatchGenerator:
    """Minimal BatchGenerator mock."""

    def __init__(self, batch: Optional[MockBatch] = None):
        self.active_batch = batch
        self.stop_tokens: set[int] = set()
        self._next_result = []

    def next(self):
        return self._next_result


class MockProposer(BaseProposer):
    """Proposer that returns pre-configured proposals."""

    def __init__(self, proposal: Optional[ProposalResult] = None):
        self._proposal = proposal

    def propose(self, sequences, k):
        return self._proposal

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False


class MockModel:
    """Model that returns pre-configured logits."""

    def __init__(self, logits: mx.array):
        self._logits = logits

    def __call__(self, x, cache=None):
        return self._logits


@dataclass
class MockSequenceState:
    """Minimal SequenceState mock."""

    request_id: str
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine(
    *,
    batch: Optional[MockBatch] = None,
    model_logits: Optional[mx.array] = None,
    proposal: Optional[ProposalResult] = None,
    config: Optional[SpecDecodeConfig] = None,
    stop_tokens: Optional[set[int]] = None,
) -> tuple[SpecDecodeEngine, MockBatchGenerator]:
    """Create a SpecDecodeEngine with mocked dependencies."""
    cfg = config or SpecDecodeConfig(mode="ngram", num_speculative_tokens=3)
    controller = DynamicSpecController(cfg)
    verifier = NGramVerifier(mode="greedy")
    proposer = MockProposer(proposal)
    model = MockModel(model_logits if model_logits is not None else mx.zeros((1, 1, 100)))
    bg = MockBatchGenerator(batch)
    if stop_tokens is not None:
        bg.stop_tokens = stop_tokens

    engine = SpecDecodeEngine(
        model=model,
        batch_generator=bg,
        proposer=proposer,
        verifier=verifier,
        config=cfg,
        controller=controller,
    )
    return engine, bg


def make_deterministic_logits(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    argmax_tokens: list[list[int]],
) -> mx.array:
    """Create logits with known argmax at each position."""
    logits = mx.full((batch_size, seq_len, vocab_size), -10.0)
    for b in range(batch_size):
        for pos in range(seq_len):
            if pos < len(argmax_tokens[b]):
                tok = argmax_tokens[b][pos]
                logits[b, pos, tok] = 10.0
    return logits


# ---------------------------------------------------------------------------
# Group A: Logic tests (no real model)
# ---------------------------------------------------------------------------


class TestBuildVerifyInput:
    """Test _build_verify_input produces correct padded input."""

    def test_single_sequence_with_proposals(self):
        batch = MockBatch(
            uids=[0],
            y=mx.array([42]),
            tokens=[mx.array([1, 2, 3])],
            num_tokens=[3],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, _ = make_engine(batch=batch)

        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)

        verify_input, verify_lens = engine._build_verify_input(
            batch, draft_tokens, proposal_lens
        )

        assert verify_input.shape == (1, 4)  # last_token + 3 drafts
        assert int(verify_input[0, 0]) == 42  # last token
        assert int(verify_input[0, 1]) == 10
        assert int(verify_input[0, 2]) == 11
        assert int(verify_input[0, 3]) == 12
        assert int(verify_lens[0]) == 4

    def test_single_sequence_no_proposals(self):
        batch = MockBatch(
            uids=[0],
            y=mx.array([42]),
            tokens=[mx.array([1, 2, 3])],
            num_tokens=[3],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, _ = make_engine(batch=batch)

        draft_tokens = mx.array([[0, 0, 0]], dtype=mx.int32)
        proposal_lens = mx.array([0], dtype=mx.int32)

        verify_input, verify_lens = engine._build_verify_input(
            batch, draft_tokens, proposal_lens
        )

        assert verify_input.shape == (1, 1)  # just last_token
        assert int(verify_input[0, 0]) == 42
        assert int(verify_lens[0]) == 1

    def test_batch_mixed_proposal_lens(self):
        batch = MockBatch(
            uids=[0, 1, 2],
            y=mx.array([10, 20, 30]),
            tokens=[mx.array([1]), mx.array([2]), mx.array([3])],
            num_tokens=[1, 1, 1],
            max_tokens=[100, 100, 100],
            cache=[],
            samplers=[MockSampler(), MockSampler(), MockSampler()],
        )
        engine, _ = make_engine(batch=batch)

        draft_tokens = mx.array([
            [100, 101, 102],
            [200, 0, 0],
            [300, 301, 0],
        ], dtype=mx.int32)
        proposal_lens = mx.array([3, 1, 2], dtype=mx.int32)

        verify_input, verify_lens = engine._build_verify_input(
            batch, draft_tokens, proposal_lens
        )

        assert verify_input.shape == (3, 4)  # padded to max(3+1, 1+1, 2+1) = 4
        # Sequence 0: [10, 100, 101, 102]
        assert int(verify_input[0, 0]) == 10
        assert int(verify_input[0, 3]) == 102
        # Sequence 1: [20, 200, 0, 0] (padded)
        assert int(verify_input[1, 0]) == 20
        assert int(verify_input[1, 1]) == 200
        assert int(verify_input[1, 2]) == 0  # padding
        # Sequence 2: [30, 300, 301, 0] (padded)
        assert int(verify_input[2, 0]) == 30
        assert int(verify_input[2, 2]) == 301
        assert int(verify_input[2, 3]) == 0  # padding

        assert list(verify_lens.tolist()) == [4, 2, 3]


class TestUpdateBatchState:
    """Test _update_batch_state correctly updates batch fields."""

    def test_single_sequence_all_accepted(self):
        """All 3 drafts accepted + bonus = 4 tokens emitted."""
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch)

        # accepted_tokens: [1, k+1=4] — all valid
        accepted = mx.array([[10, 11, 12, 13]], dtype=mx.int32)
        num_accepted = mx.array([3], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)
        # target_probs: [1, 4, vocab=100]
        target_probs = mx.softmax(
            make_deterministic_logits(1, 4, 100, [[10, 11, 12, 13]]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0", token_ids=[1, 2, 3, 4, 5])]

        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        assert len(responses) == 1
        resp = responses[0]
        assert resp.uid == 0
        # Result B: min_accepted=3, max_valid=4 => all 4 tokens
        assert resp.tokens == [10, 11, 12, 13]
        assert resp.num_drafted == 3
        assert resp.num_accepted == 3
        assert resp.finish_reason is None

        # batch.y should be last accepted token
        assert int(batch.y[0]) == 13
        # batch.num_tokens incremented by 4
        assert batch.num_tokens[0] == 9
        # batch.tokens extended
        assert batch.tokens[0].shape[0] == 9

    def test_single_sequence_partial_acceptance(self):
        """2 of 3 drafts accepted + correction = 3 tokens."""
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch)

        # accepted_tokens: 2 valid + correction at idx 2, rest PLACEHOLDER
        accepted = mx.array([[10, 11, 99, PLACEHOLDER_TOKEN_ID]], dtype=mx.int32)
        num_accepted = mx.array([2], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(1, 4, 100, [[10, 11, 99, 0]]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0")]

        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        assert len(responses) == 1
        resp = responses[0]
        # min_accepted=2, max_valid=3 => [10, 11, 99]
        assert resp.tokens == [10, 11, 99]
        assert resp.num_accepted == 2
        assert int(batch.y[0]) == 99
        assert batch.num_tokens[0] == 8

    def test_stop_token_detection(self):
        """Stop token in accepted tokens triggers finish_reason='stop'."""
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3])],
            num_tokens=[3],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch, stop_tokens={11})

        accepted = mx.array([[10, 11, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]], dtype=mx.int32)
        num_accepted = mx.array([1], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(1, 4, 100, [[10, 11, 0, 0]]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0")]
        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        assert responses[0].finish_reason == "stop"

    def test_max_tokens_detection(self):
        """Reaching max_tokens triggers finish_reason='length'."""
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3])],
            num_tokens=[8],  # close to limit
            max_tokens=[10],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch)

        accepted = mx.array([[10, 11, 12, PLACEHOLDER_TOKEN_ID]], dtype=mx.int32)
        num_accepted = mx.array([2], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(1, 4, 100, [[10, 11, 12, 0]]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0")]
        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        # 8 + 3 = 11 >= 10 → "length"
        assert responses[0].finish_reason == "length"

    def test_batch_per_sequence_output(self):
        """Result A: each sequence keeps its own accepted tokens."""
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=[MockCacheLayer(offset=20, batch_size=2)],
            samplers=[MockSampler(), MockSampler()],
        )
        engine, bg = make_engine(batch=batch)
        # Force Result A detection
        engine._per_seq_trim = True

        # Seq 0: 3 accepted (full), Seq 1: 1 accepted
        accepted = mx.array([
            [10, 11, 12, 13],
            [20, 99, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ], dtype=mx.int32)
        num_accepted = mx.array([3, 1], dtype=mx.int32)
        proposal_lens = mx.array([3, 3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(2, 4, 100, [
                [10, 11, 12, 13],
                [20, 99, 0, 0],
            ]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0"), MockSequenceState(request_id="r1")]
        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        # Result A: seq 0 keeps all 4 tokens (3 accepted + bonus)
        assert responses[0].tokens == [10, 11, 12, 13]
        assert responses[0].num_accepted == 3
        # Seq 1 gets 2 tokens (1 accepted + correction)
        assert responses[1].tokens == [20, 99]
        assert responses[1].num_accepted == 1

    def test_batch_result_b_clamping(self):
        """Result B fallback: tokens clamped to min_accepted + 1 across batch."""
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            num_tokens=[3, 3],
            max_tokens=[100, 100],
            cache=[],
            samplers=[MockSampler(), MockSampler()],
        )
        engine, bg = make_engine(batch=batch)
        # Force Result B
        engine._per_seq_trim = False

        # Seq 0: 3 accepted (full), Seq 1: 1 accepted
        # min_accepted = 1, max_valid = 2
        accepted = mx.array([
            [10, 11, 12, 13],
            [20, 99, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ], dtype=mx.int32)
        num_accepted = mx.array([3, 1], dtype=mx.int32)
        proposal_lens = mx.array([3, 3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(2, 4, 100, [
                [10, 11, 12, 13],
                [20, 99, 0, 0],
            ]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0"), MockSequenceState(request_id="r1")]
        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        # Seq 0 clamped to 2 tokens despite 4 valid tokens
        assert responses[0].tokens == [10, 11]
        assert responses[0].num_accepted == 1  # clamped to min_accepted
        # Seq 1 gets 2 tokens (1 accepted + correction)
        assert responses[1].tokens == [20, 99]
        assert responses[1].num_accepted == 1


class TestRollbackCache:
    """Test _rollback_cache for Result A (variable trim) and Result B (uniform trim)."""

    def test_variable_trim_called(self):
        """Result A: per-sequence trim via trim_per_sequence."""
        cache_layers = [MockCacheLayer(offset=20, batch_size=2)]
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([1, 2]),
            tokens=[mx.array([1]), mx.array([2])],
            num_tokens=[1, 1],
            max_tokens=[100, 100],
            cache=cache_layers,
            samplers=[MockSampler(), MockSampler()],
        )
        engine, _ = make_engine(batch=batch)
        engine._per_seq_trim = True

        # Seq 0: 3 accepted, Seq 1: 1 accepted
        # keep = [4, 2], trim = [4-4, 4-2] = [0, 2]
        num_accepted = mx.array([3, 1], dtype=mx.int32)
        engine._rollback_cache(batch, num_accepted, max_input_len=4)

        for layer in cache_layers:
            assert len(layer.trim_per_seq_calls) == 1
            trim_arr = layer.trim_per_seq_calls[0]
            assert int(trim_arr[0]) == 0  # seq 0: no trim needed
            assert int(trim_arr[1]) == 2  # seq 1: trim 2

    def test_uniform_trim_called_result_b(self):
        """Result B fallback: uniform trim to min_accepted."""
        cache_layers = [MockCacheLayer(offset=20, batch_size=2)]
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([1, 2]),
            tokens=[mx.array([1]), mx.array([2])],
            num_tokens=[1, 1],
            max_tokens=[100, 100],
            cache=cache_layers,
            samplers=[MockSampler(), MockSampler()],
        )
        engine, _ = make_engine(batch=batch)
        engine._per_seq_trim = False

        # Seq 0: 3 accepted, Seq 1: 1 accepted
        # min_accepted = 1, keep = 2, max_input_len = 4, trim = 2
        num_accepted = mx.array([3, 1], dtype=mx.int32)
        engine._rollback_cache(batch, num_accepted, max_input_len=4)

        for layer in cache_layers:
            assert layer.trim_calls == [2]  # trim_amount = 4 - (1+1) = 2

    def test_no_trim_when_all_accepted(self):
        cache_layers = [MockCacheLayer(offset=20)]
        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler()],
        )
        engine, _ = make_engine(batch=batch)
        engine._per_seq_trim = True

        # 3 accepted, max_input_len = 4 (3 drafts + 1 last_token)
        # keep = 3 + 1 = 4, trim = 4 - 4 = 0
        num_accepted = mx.array([3], dtype=mx.int32)
        engine._rollback_cache(batch, num_accepted, max_input_len=4)

        assert cache_layers[0].trim_per_seq_calls == []  # max trim is 0

    def test_trim_amount_calculation(self):
        cache_layers = [MockCacheLayer(offset=20)]
        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler()],
        )
        engine, _ = make_engine(batch=batch)
        engine._per_seq_trim = True

        # 0 accepted from k=5, max_input_len=6
        # keep = 0 + 1 = 1, trim = 6 - 1 = 5
        num_accepted = mx.array([0], dtype=mx.int32)
        engine._rollback_cache(batch, num_accepted, max_input_len=6)

        assert len(cache_layers[0].trim_per_seq_calls) == 1
        assert int(cache_layers[0].trim_per_seq_calls[0][0]) == 5


class TestShouldSpeculate:
    """Test should_speculate delegates to controller."""

    def test_no_active_batch_returns_false(self):
        engine, bg = make_engine()
        bg.active_batch = None
        assert engine.should_speculate(1) is False

    def test_with_active_batch_delegates_to_controller(self):
        batch = MockBatch(
            uids=[0], y=mx.array([1]), tokens=[mx.array([1])],
            num_tokens=[1], max_tokens=[100], cache=[], samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch)
        # Default config: mode="ngram", dynamic_enabled=True, ema starts at 0.7
        assert engine.should_speculate(1) is True

    def test_mode_none_returns_false(self):
        batch = MockBatch(
            uids=[0], y=mx.array([1]), tokens=[mx.array([1])],
            num_tokens=[1], max_tokens=[100], cache=[], samplers=[MockSampler()],
        )
        cfg = SpecDecodeConfig(mode="none")
        engine, bg = make_engine(batch=batch, config=cfg)
        assert engine.should_speculate(1) is False

    def test_large_batch_size_returns_false(self):
        batch = MockBatch(
            uids=[0], y=mx.array([1]), tokens=[mx.array([1])],
            num_tokens=[1], max_tokens=[100], cache=[], samplers=[MockSampler()],
        )
        cfg = SpecDecodeConfig(mode="ngram", disable_by_batch_size=4)
        engine, bg = make_engine(batch=batch, config=cfg)
        assert engine.should_speculate(4) is False
        assert engine.should_speculate(3) is True


class TestFallbackNormalDecode:
    """Test _fallback_normal_decode calls batch_generator.next()."""

    def test_returns_batch_generator_next(self):
        engine, bg = make_engine()
        sentinel = [MagicMock()]
        bg._next_result = sentinel

        result = engine._fallback_normal_decode()
        assert result is sentinel


class TestPerSequenceModes:
    """Test that per-sequence verification modes are computed from temperatures."""

    def test_mixed_temperatures(self):
        """temp=0 -> greedy, temp>0 -> threshold."""
        batch = MockBatch(
            uids=[0, 1, 2],
            y=mx.array([1, 2, 3]),
            tokens=[mx.array([1]), mx.array([2]), mx.array([3])],
            num_tokens=[1, 1, 1],
            max_tokens=[100, 100, 100],
            cache=[],
            samplers=[MockSampler(0.0), MockSampler(0.8), MockSampler(0.0)],
        )
        engine, _ = make_engine(batch=batch)

        modes = []
        for i in range(3):
            temp = getattr(batch.samplers[i], 'temperature', 0.0)
            modes.append(engine.controller.get_verification_mode(temp))

        assert modes == ["greedy", "threshold", "greedy"]


class TestTargetForward:
    """Test _target_forward applies temperature scaling."""

    def test_temperature_zero_no_scaling(self):
        """temp=0: logits passed through without division."""
        logits = mx.array([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler(0.0)],
        )
        engine, _ = make_engine(batch=batch, model_logits=logits)
        probs = engine._target_forward(batch, mx.array([[1]], dtype=mx.int32))

        # With no temperature scaling, argmax should be at position 2
        assert int(mx.argmax(probs[0, 0])) == 2

    def test_temperature_scaling_applied(self):
        """temp>0: logits are divided by temperature before softmax."""
        logits = mx.array([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler(0.5)],
        )
        engine, _ = make_engine(batch=batch, model_logits=logits)
        probs = engine._target_forward(batch, mx.array([[1]], dtype=mx.int32))

        # With temperature scaling, distribution should be sharper
        # softmax(3/0.5) = softmax(6) should have higher peak than softmax(3)
        assert int(mx.argmax(probs[0, 0])) == 2
        # Higher temperature means more peaked distribution
        assert float(probs[0, 0, 2]) > 0.5


class TestSpeculativeStep:
    """Test full speculative_step orchestration."""

    def test_no_active_batch_returns_empty(self):
        engine, bg = make_engine()
        bg.active_batch = None
        result = engine.speculative_step([])
        assert result == []

    def test_k_zero_falls_back(self):
        """When controller returns k=0, fallback to normal decode."""
        batch = MockBatch(
            uids=[0], y=mx.array([1]), tokens=[mx.array([1])],
            num_tokens=[1], max_tokens=[100], cache=[], samplers=[MockSampler()],
        )
        cfg = SpecDecodeConfig(mode="none")  # mode=none -> k=0
        engine, bg = make_engine(batch=batch, config=cfg)
        sentinel = [MagicMock()]
        bg._next_result = sentinel

        result = engine.speculative_step([MockSequenceState("r0")])
        assert result is sentinel

    def test_no_proposals_falls_back(self):
        """When proposer returns None, fallback to normal decode."""
        batch = MockBatch(
            uids=[0], y=mx.array([1]), tokens=[mx.array([1])],
            num_tokens=[1], max_tokens=[100], cache=[MockCacheLayer()],
            samplers=[MockSampler()],
        )
        # Proposer returns None (no proposals)
        engine, bg = make_engine(batch=batch, proposal=None)
        sentinel = [MagicMock()]
        bg._next_result = sentinel

        result = engine.speculative_step([MockSequenceState("r0")])
        assert result is sentinel

    def test_full_step_single_sequence_greedy(self):
        """Full speculative step with 1 sequence, all drafts accepted."""
        vocab_size = 200

        # Draft tokens that match the target model argmax
        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )

        # Target model returns logits where argmax matches draft tokens
        # verify_input will be [last_token, 10, 11, 12] -> 4 positions
        # target_probs at pos 0 -> argmax=10, pos 1 -> 11, pos 2 -> 12, pos 3 -> bonus
        logits = make_deterministic_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        cache_layers = [MockCacheLayer(offset=20)]
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler(0.0)],
        )

        engine, bg = make_engine(
            batch=batch,
            model_logits=logits,
            proposal=proposal,
        )

        seqs = [MockSequenceState("r0", token_ids=[1, 2, 3, 4, 5])]
        responses = engine.speculative_step(seqs)

        assert len(responses) == 1
        resp = responses[0]
        assert isinstance(resp, SpecResponse)
        assert resp.uid == 0
        # All 3 drafts accepted + bonus = 4 tokens
        assert resp.tokens == [10, 11, 12, 99]
        assert resp.num_drafted == 3
        assert resp.num_accepted == 3
        assert resp.finish_reason is None

        # Cache should not be trimmed (all accepted)
        assert cache_layers[0].trim_calls == []

        # batch.y updated to last token
        assert int(batch.y[0]) == 99
        # batch.num_tokens incremented
        assert batch.num_tokens[0] == 9  # 5 + 4

    def test_full_step_partial_acceptance(self):
        """Full step with partial acceptance: 1 of 3 drafts accepted."""
        vocab_size = 200

        # Draft tokens: only first matches target argmax
        draft_tokens = mx.array([[10, 77, 78]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )

        # Target argmax: [10, 11, 12, 99]
        # Draft[0]=10 matches, Draft[1]=77 != 11 -> rejection at pos 1
        logits = make_deterministic_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        cache_layers = [MockCacheLayer(offset=20)]
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler(0.0)],
        )

        engine, bg = make_engine(
            batch=batch,
            model_logits=logits,
            proposal=proposal,
        )

        seqs = [MockSequenceState("r0")]
        responses = engine.speculative_step(seqs)

        assert len(responses) == 1
        resp = responses[0]
        # 1 accepted + correction token = 2 tokens
        assert resp.tokens == [10, 11]
        assert resp.num_accepted == 1

        # Cache trim: Result A, per-sequence: keep=2, trim=4-2=2
        assert len(cache_layers[0].trim_per_seq_calls) == 1
        assert int(cache_layers[0].trim_per_seq_calls[0][0]) == 2

    def test_controller_updated_after_step(self):
        """Controller stats are updated after a successful spec step."""
        vocab_size = 200
        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        logits = make_deterministic_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=[MockCacheLayer()],
            samplers=[MockSampler(0.0)],
        )

        engine, bg = make_engine(batch=batch, model_logits=logits, proposal=proposal)
        seqs = [MockSequenceState("r0")]
        engine.speculative_step(seqs)

        stats = engine.controller.stats
        assert stats.total_proposed == 3
        assert stats.total_accepted == 3
        assert stats.total_steps == 1
        assert stats.total_bonus_tokens == 1  # all accepted = bonus

    def test_batch_two_sequences_different_acceptance(self):
        """Batch of 2: seq0 accepts all 3, seq1 accepts 0.

        Result A: each sequence keeps its own accepted tokens.
        """
        vocab_size = 200

        # Seq 0: all match, Seq 1: first mismatch
        draft_tokens = mx.array([
            [10, 11, 12],
            [50, 51, 52],
        ], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3, 3], dtype=mx.int32),
        )

        # Seq 0 target: [10, 11, 12, 99] (all match)
        # Seq 1 target: [60, 61, 62, 63] (none match)
        logits = make_deterministic_logits(2, 4, vocab_size, [
            [10, 11, 12, 99],
            [60, 61, 62, 63],
        ])

        cache_layers = [MockCacheLayer(offset=20, batch_size=2)]
        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([5, 6]),
            tokens=[mx.array([1, 2, 3, 4, 5]), mx.array([4, 5, 6])],
            num_tokens=[5, 3],
            max_tokens=[100, 100],
            cache=cache_layers,
            samplers=[MockSampler(0.0), MockSampler(0.0)],
        )

        engine, bg = make_engine(batch=batch, model_logits=logits, proposal=proposal)
        seqs = [MockSequenceState("r0"), MockSequenceState("r1")]
        responses = engine.speculative_step(seqs)

        assert len(responses) == 2

        # Result A: seq 0 keeps all 4 tokens (3 accepted + bonus)
        assert responses[0].tokens == [10, 11, 12, 99]
        assert responses[0].num_accepted == 3
        # Seq 1 gets 1 correction token
        assert responses[1].tokens == [60]
        assert responses[1].num_accepted == 0

        # Cache trim: Result A, per-sequence trim
        # seq 0: keep=4, trim=0; seq 1: keep=1, trim=3
        assert len(cache_layers[0].trim_per_seq_calls) == 1
        trim_arr = cache_layers[0].trim_per_seq_calls[0]
        assert int(trim_arr[0]) == 0  # seq 0
        assert int(trim_arr[1]) == 3  # seq 1


class TestResponseFields:
    """Test SpecResponse fields are correctly populated."""

    def test_logprobs_present(self):
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3])],
            num_tokens=[3],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch)

        accepted = mx.array([[10, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]], dtype=mx.int32)
        num_accepted = mx.array([0], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(1, 4, 100, [[10, 11, 12, 13]]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0")]
        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        resp = responses[0]
        assert len(resp.logprobs) == 1  # 1 token
        assert resp.logprobs[0].shape == (100,)  # vocab-sized logprob array
        assert resp.prompt_cache is None

    def test_uid_matches_batch_uid(self):
        batch = MockBatch(
            uids=[42],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3])],
            num_tokens=[3],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler()],
        )
        engine, bg = make_engine(batch=batch)

        accepted = mx.array([[10, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]], dtype=mx.int32)
        num_accepted = mx.array([0], dtype=mx.int32)
        proposal_lens = mx.array([3], dtype=mx.int32)
        target_probs = mx.softmax(
            make_deterministic_logits(1, 4, 100, [[10, 11, 12, 13]]),
            axis=-1,
        )

        seqs = [MockSequenceState(request_id="r0")]
        responses = engine._update_batch_state(
            batch, seqs, accepted, num_accepted, proposal_lens, target_probs
        )

        assert responses[0].uid == 42


class TestLazyDetection:
    """Test that _per_seq_trim is detected lazily and only once."""

    def test_lazy_detection_triggers_once(self):
        """Engine detects per_seq_trim on first speculative_step, not before."""
        vocab_size = 200
        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        logits = make_deterministic_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        cache_layers = [MockCacheLayer(offset=20)]
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler(0.0)],
        )

        engine, bg = make_engine(batch=batch, model_logits=logits, proposal=proposal)

        # Before first step, detection has not occurred
        assert engine._per_seq_trim is None

        seqs = [MockSequenceState("r0", token_ids=[1, 2, 3, 4, 5])]
        engine.speculative_step(seqs)

        # After first step, detection has occurred (MockCacheLayer has trim_per_sequence)
        assert engine._per_seq_trim is True

        # Second call doesn't re-detect (stays True)
        # Reset proposal for second step
        engine.proposer._proposal = ProposalResult(
            draft_tokens=mx.array([[10, 11, 12]], dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        engine.speculative_step(seqs)
        assert engine._per_seq_trim is True

    def test_detection_false_without_trim_per_sequence(self):
        """Engine detects Result B when cache lacks trim_per_sequence."""

        class BasicCacheLayer:
            """Cache without trim_per_sequence."""
            def __init__(self):
                self._offset = 20
                self.offset = mx.array([20], dtype=mx.int32)
            def trim(self, n):
                self._offset -= n
                self.offset = self.offset - n

        vocab_size = 200
        draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)
        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.array([3], dtype=mx.int32),
        )
        logits = make_deterministic_logits(1, 4, vocab_size, [[10, 11, 12, 99]])

        cache_layers = [BasicCacheLayer()]
        batch = MockBatch(
            uids=[0],
            y=mx.array([5]),
            tokens=[mx.array([1, 2, 3, 4, 5])],
            num_tokens=[5],
            max_tokens=[100],
            cache=cache_layers,
            samplers=[MockSampler(0.0)],
        )

        engine, bg = make_engine(batch=batch, model_logits=logits, proposal=proposal)
        seqs = [MockSequenceState("r0", token_ids=[1, 2, 3, 4, 5])]
        engine.speculative_step(seqs)

        assert engine._per_seq_trim is False
