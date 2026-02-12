"""Tests for MTP-specific behavior in SpecDecodeEngine and MTPProposer interaction.

Covers:
- MTP mode detection based on proposer capabilities
- Null proposal bootstrap (first step with no hidden states)
- Hidden state flow from target forward through to proposer
- Batch size guard in MTPProposer
- Temperature scaling (functional, no in-place mutation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import patch

import mlx.core as mx

from mlx_lm_server.spec_decode.config import SpecDecodeConfig
from mlx_lm_server.spec_decode.controller import DynamicSpecController
from mlx_lm_server.spec_decode.engine import SpecDecodeEngine
from mlx_lm_server.spec_decode.proposer.base import (
    BaseProposer,
    ProposalResult,
    SpecResponse,
)
from mlx_lm_server.spec_decode.verifier import NGramVerifier


# ---------------------------------------------------------------------------
# Mock objects (reused patterns from test_spec_engine.py)
# ---------------------------------------------------------------------------


class MockSampler:
    """Sampler mock with configurable temperature."""

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature


class MockCacheLayer:
    """Minimal cache layer mock that tracks trim calls."""

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
        self._next_result: list = []

    def next(self):
        return self._next_result


class MockProposer(BaseProposer):
    """Proposer that returns pre-configured proposals (no set_hidden_states)."""

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
# MTP-specific mock objects
# ---------------------------------------------------------------------------


class MTPLikeProposer(BaseProposer):
    """Proposer that has set_hidden_states, mimicking MTPProposer.

    Triggers _mtp_mode=True in SpecDecodeEngine because it has
    the set_hidden_states attribute.
    """

    def __init__(self, proposal: Optional[ProposalResult] = None):
        self._proposal = proposal
        self._cached_hidden: Optional[mx.array] = None
        self._set_hidden_calls: list[mx.array] = []

    def set_hidden_states(self, hidden: mx.array) -> None:
        self._cached_hidden = hidden
        self._set_hidden_calls.append(hidden)

    def propose(self, sequences, k):
        return self._proposal

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return True


class NGramLikeProposer(BaseProposer):
    """Proposer without set_hidden_states, mimicking NGramProposer."""

    def propose(self, sequences, k):
        return None

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False


class MockModelWithHidden:
    """Model mock whose .model() returns hidden states and has lm_head.

    Simulates the pattern used by forward_with_hidden:
      hidden = model.model(inputs, cache=cache)
      logits = model.lm_head(hidden)
    """

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100):
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._backbone = _MockBackbone(hidden_size)
        self._lm_head = _MockLMHead(vocab_size)

    @property
    def model(self):
        return self._backbone

    @property
    def lm_head(self):
        return self._lm_head

    def __call__(self, x, cache=None):
        hidden = self._backbone(x, cache=cache)
        return self._lm_head(hidden)


class _MockBackbone:
    """Backbone that returns deterministic hidden states.

    Supports the layer-iteration interface used by forward_with_hidden:
    - embed_tokens(inputs) -> [B, S, D]
    - layers: list of callable decoder layers
    - norm(h) -> identity
    """

    def __init__(self, hidden_size: int):
        self._hidden_size = hidden_size
        self.embed_tokens = _MockEmbedTokens(hidden_size)
        self.layers = [_MockDecoderLayer(hidden_size)]
        self.norm = _MockNorm()

    def __call__(self, x, cache=None):
        # Keep backward compatibility for any direct calls
        B, S = x.shape
        return mx.ones((B, S, self._hidden_size)) * 0.5


class _MockEmbedTokens:
    """Mock embedding that returns constant embeddings."""

    def __init__(self, hidden_size: int):
        self._hidden_size = hidden_size

    def __call__(self, inputs):
        B, S = inputs.shape
        return mx.ones((B, S, self._hidden_size)) * 0.5

    def as_linear(self, hidden):
        """For tied weights fallback."""
        B, S, _ = hidden.shape
        return mx.zeros((B, S, 100))


class _MockDecoderLayer:
    """Mock decoder layer that acts as identity."""

    def __init__(self, hidden_size: int):
        self._hidden_size = hidden_size

    def __call__(self, h, mask=None, cache=None):
        return h


class _MockNorm:
    """Mock norm that acts as identity."""

    def __call__(self, h):
        return h


class _MockLMHead:
    """LM head that returns deterministic logits from hidden states."""

    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    def __call__(self, hidden):
        # hidden: [B, S, D] -> logits: [B, S, vocab_size]
        B, S, _ = hidden.shape
        # Token 0 gets highest logit by default
        logits = mx.zeros((B, S, self._vocab_size))
        logits = logits.at[:, :, 0].add(10.0)
        return logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def make_engine(
    *,
    batch: Optional[MockBatch] = None,
    model: Optional[Any] = None,
    model_logits: Optional[mx.array] = None,
    proposer: Optional[BaseProposer] = None,
    config: Optional[SpecDecodeConfig] = None,
    stop_tokens: Optional[set[int]] = None,
) -> tuple[SpecDecodeEngine, MockBatchGenerator]:
    """Create a SpecDecodeEngine with mocked dependencies.

    Accepts either an explicit model or model_logits (not both).
    If proposer is not given, uses MockProposer (no MTP).
    """
    cfg = config or SpecDecodeConfig(mode="ngram", num_speculative_tokens=3)
    controller = DynamicSpecController(cfg)
    verifier = NGramVerifier(mode="greedy")
    if proposer is None:
        proposer = MockProposer(None)
    if model is None:
        model = MockModel(
            model_logits if model_logits is not None else mx.zeros((1, 1, 100))
        )
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


# ---------------------------------------------------------------------------
# TestMTPModeDetection
# ---------------------------------------------------------------------------


class TestMTPModeDetection:
    """Test that _mtp_mode is set based on proposer capabilities."""

    def test_mtp_mode_true_for_mtp_proposer(self):
        """Proposer with set_hidden_states -> _mtp_mode=True."""
        proposer = MTPLikeProposer()
        engine, _ = make_engine(proposer=proposer)
        assert engine._mtp_mode is True

    def test_mtp_mode_false_for_ngram_proposer(self):
        """NGram-like proposer without set_hidden_states -> _mtp_mode=False."""
        proposer = NGramLikeProposer()
        engine, _ = make_engine(proposer=proposer)
        assert engine._mtp_mode is False

    def test_mtp_mode_false_for_mock_proposer(self):
        """MockProposer (base) without set_hidden_states -> _mtp_mode=False."""
        proposer = MockProposer()
        engine, _ = make_engine(proposer=proposer)
        assert engine._mtp_mode is False


# ---------------------------------------------------------------------------
# TestMTPNullProposalBootstrap
# ---------------------------------------------------------------------------


@patch("mlx_lm.models.base.create_attention_mask", return_value=None)
class TestMTPNullProposalBootstrap:
    """Test the bootstrap path when MTP proposer returns None (first step)."""

    def _make_mtp_engine(
        self,
        batch_size: int = 1,
        vocab_size: int = 100,
        hidden_size: int = 64,
    ):
        """Helper to create an engine with MTP-like proposer and model with hidden."""
        # MTP-like proposer that returns None (first step, no cached hidden)
        proposer = MTPLikeProposer(proposal=None)

        model = MockModelWithHidden(
            hidden_size=hidden_size, vocab_size=vocab_size
        )

        cache_layers = [MockCacheLayer(offset=20, batch_size=batch_size)]
        samplers = [MockSampler(0.0) for _ in range(batch_size)]

        batch = MockBatch(
            uids=list(range(batch_size)),
            y=mx.array(list(range(1, batch_size + 1))),
            tokens=[mx.array([1, 2, 3]) for _ in range(batch_size)],
            num_tokens=[3] * batch_size,
            max_tokens=[100] * batch_size,
            cache=cache_layers,
            samplers=samplers,
        )

        seqs = [
            MockSequenceState(
                request_id=f"r{i}", token_ids=[1, 2, 3]
            )
            for i in range(batch_size)
        ]

        engine, bg = make_engine(
            batch=batch,
            model=model,
            proposer=proposer,
        )

        return engine, bg, batch, seqs, proposer

    def test_first_step_creates_null_proposal(self, mock_mask):
        """When propose() returns None in MTP mode, engine creates a null ProposalResult."""
        engine, bg, batch, seqs, proposer = self._make_mtp_engine()

        # The proposer has no cached hidden, so propose() returns None.
        # In MTP mode, the engine creates a null proposal instead of falling back.
        responses = engine.speculative_step(seqs)

        # Should produce responses (not fallback empty list)
        assert len(responses) == 1
        assert isinstance(responses[0], SpecResponse)

    def test_null_proposal_produces_one_token(self, mock_mask):
        """Bootstrap step with null proposal outputs at least 1 token."""
        engine, bg, batch, seqs, proposer = self._make_mtp_engine()

        responses = engine.speculative_step(seqs)

        assert len(responses) == 1
        assert len(responses[0].tokens) >= 1

    def test_null_proposal_captures_hidden(self, mock_mask):
        """After target_forward in MTP mode, hidden states are captured then passed to proposer."""
        engine, bg, batch, seqs, proposer = self._make_mtp_engine()

        # Before step, _last_hidden is None
        assert engine._last_hidden is None

        engine.speculative_step(seqs)

        # After the full step, _last_hidden should be cleared (passed to proposer)
        assert engine._last_hidden is None

        # The proposer should have received hidden states via set_hidden_states
        assert len(proposer._set_hidden_calls) == 1
        hidden = proposer._set_hidden_calls[0]
        assert hidden.ndim == 3  # [B, 1, D]
        assert hidden.shape[0] == 1  # batch_size=1
        assert hidden.shape[1] == 1  # single position

    def test_null_proposal_verify_input_shape(self, mock_mask):
        """Null proposal produces verify_input of shape [B, 1]."""
        engine, bg, batch, seqs, proposer = self._make_mtp_engine()

        # The null proposal has draft_tokens of shape [B, 1] with zeros
        # and proposal_lens of [0], so verify_input = [last_token] -> [B, 1]
        null_proposal = ProposalResult(
            draft_tokens=mx.zeros((1, 1), dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.zeros((1,), dtype=mx.int32),
        )
        verify_input, verify_lens = engine._build_verify_input(
            batch, null_proposal.draft_tokens, null_proposal.proposal_lens
        )

        assert verify_input.shape == (1, 1)
        assert int(verify_lens[0]) == 1


# ---------------------------------------------------------------------------
# TestMTPHiddenStateFlow
# ---------------------------------------------------------------------------


@patch("mlx_lm.models.base.create_attention_mask", return_value=None)
class TestMTPHiddenStateFlow:
    """Test hidden state flow from target forward to proposer."""

    def _make_engine_with_proposal(
        self,
        batch_size: int = 1,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_accepted_list: Optional[list[int]] = None,
    ):
        """Create an MTP engine where the proposer returns a real proposal.

        The model returns deterministic logits so we control acceptance.
        """
        if num_accepted_list is None:
            num_accepted_list = [0] * batch_size

        # Build draft tokens that will all be accepted or not based on logits
        # For simplicity, draft tokens match target argmax for accepted ones
        argmax_per_seq = []
        draft_per_seq = []
        for i in range(batch_size):
            n = num_accepted_list[i]
            # Target argmax at each position
            target_tokens = [10 + j for j in range(4)]  # [10,11,12,13]
            argmax_per_seq.append(target_tokens)
            # Draft tokens: match for first n, mismatch after
            draft_row = []
            for j in range(3):
                if j < n:
                    draft_row.append(target_tokens[j])
                else:
                    draft_row.append(99)  # mismatch
            draft_per_seq.append(draft_row)

        draft_tokens = mx.array(draft_per_seq, dtype=mx.int32)
        logits = make_deterministic_logits(batch_size, 4, vocab_size, argmax_per_seq)

        proposal = ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=None,
            proposal_lens=mx.full((batch_size,), 3, dtype=mx.int32),
        )

        proposer = MTPLikeProposer(proposal=proposal)
        model = MockModelWithHidden(hidden_size=hidden_size, vocab_size=vocab_size)

        # Override lm_head to produce our deterministic logits
        model._lm_head = _DeterministicLMHead(logits)

        cache_layers = [MockCacheLayer(offset=20, batch_size=batch_size)]
        samplers = [MockSampler(0.0) for _ in range(batch_size)]

        batch = MockBatch(
            uids=list(range(batch_size)),
            y=mx.array(list(range(1, batch_size + 1))),
            tokens=[mx.array([1, 2, 3]) for _ in range(batch_size)],
            num_tokens=[3] * batch_size,
            max_tokens=[100] * batch_size,
            cache=cache_layers,
            samplers=samplers,
        )

        seqs = [
            MockSequenceState(request_id=f"r{i}", token_ids=[1, 2, 3])
            for i in range(batch_size)
        ]

        engine, bg = make_engine(
            batch=batch,
            model=model,
            proposer=proposer,
        )

        return engine, bg, batch, seqs, proposer

    def test_hidden_state_passed_to_proposer(self, mock_mask):
        """set_hidden_states() is called on the proposer after target forward."""
        engine, bg, batch, seqs, proposer = self._make_engine_with_proposal(
            batch_size=1, num_accepted_list=[0]
        )

        engine.speculative_step(seqs)

        assert len(proposer._set_hidden_calls) == 1
        hidden = proposer._set_hidden_calls[0]
        assert hidden.shape[0] == 1  # batch_size
        assert hidden.shape[1] == 1  # single position slice

    def test_hidden_state_correct_indexing_0_accepted(self, mock_mask):
        """num_accepted=0 -> hidden[i, 0:1, :] is passed."""
        engine, bg, batch, seqs, proposer = self._make_engine_with_proposal(
            batch_size=1, hidden_size=64, num_accepted_list=[0]
        )

        engine.speculative_step(seqs)

        hidden = proposer._set_hidden_calls[0]
        # With 0 accepted, the engine slices hidden[i, 0:1, :]
        assert hidden.shape == (1, 1, 64)

    def test_hidden_state_correct_indexing_k_accepted(self, mock_mask):
        """num_accepted=k -> hidden[i, k:k+1, :] is passed."""
        engine, bg, batch, seqs, proposer = self._make_engine_with_proposal(
            batch_size=1, hidden_size=64, num_accepted_list=[2]
        )

        engine.speculative_step(seqs)

        hidden = proposer._set_hidden_calls[0]
        # With 2 accepted, the engine slices hidden[i, 2:3, :]
        assert hidden.shape == (1, 1, 64)
        # The hidden value should come from position 2 of the backbone output
        # MockBackbone returns all 0.5, so value is consistent
        assert float(hidden[0, 0, 0]) == 0.5

    def test_hidden_cleared_after_step(self, mock_mask):
        """_last_hidden is set to None after the step completes."""
        engine, bg, batch, seqs, proposer = self._make_engine_with_proposal(
            batch_size=1, num_accepted_list=[0]
        )

        engine.speculative_step(seqs)

        # _last_hidden should be cleared after being passed to proposer
        assert engine._last_hidden is None


class _DeterministicLMHead:
    """LM head that returns pre-set logits regardless of input hidden states."""

    def __init__(self, logits: mx.array):
        self._logits = logits

    def __call__(self, hidden):
        return self._logits


# ---------------------------------------------------------------------------
# TestMTPBatchSizeGuard
# ---------------------------------------------------------------------------


class TestMTPBatchSizeGuard:
    """Test MTPProposer's batch size guard in propose()."""

    def test_propose_returns_none_on_batch_mismatch(self):
        """cached_hidden.shape[0] != len(sequences) -> returns None."""
        from mlx_lm_server.spec_decode.proposer.mtp import MTPProposer

        # Create a minimal MTP module mock
        mtp_module = _MockMTPModule(hidden_size=64, vocab_size=100)
        proposer = MTPProposer(mtp_module=mtp_module, num_mtp_layers=3)

        # Set cached hidden for batch_size=2
        proposer.set_hidden_states(mx.ones((2, 1, 64)))

        # But propose with 3 sequences -> mismatch
        seqs = [
            MockSequenceState(request_id="r0", token_ids=[1, 2]),
            MockSequenceState(request_id="r1", token_ids=[3, 4]),
            MockSequenceState(request_id="r2", token_ids=[5, 6]),
        ]
        result = proposer.propose(seqs, k=3)
        assert result is None

        # Cached hidden should be invalidated
        assert proposer._cached_hidden is None

    def test_propose_succeeds_on_batch_match(self):
        """Matching batch size -> returns ProposalResult."""
        from mlx_lm_server.spec_decode.proposer.mtp import MTPProposer

        mtp_module = _MockMTPModule(hidden_size=64, vocab_size=100)
        proposer = MTPProposer(mtp_module=mtp_module, num_mtp_layers=3)

        # Set cached hidden for batch_size=2
        proposer.set_hidden_states(mx.ones((2, 1, 64)))

        # Propose with 2 sequences -> match
        seqs = [
            MockSequenceState(request_id="r0", token_ids=[1, 2]),
            MockSequenceState(request_id="r1", token_ids=[3, 4]),
        ]
        result = proposer.propose(seqs, k=2)
        assert result is not None
        assert isinstance(result, ProposalResult)
        assert result.draft_tokens.shape == (2, 2)  # [B, k]
        assert result.proposal_lens.shape == (2,)


class _MockMTPModule:
    """Minimal MTPModule mock for testing MTPProposer directly."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100):
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

    def get_embed(self, token_ids: mx.array) -> mx.array:
        """Return dummy embeddings: [B, 1, D]."""
        B = token_ids.shape[0]
        return mx.ones((B, 1, self._hidden_size)) * 0.1

    def predict(
        self, depth: int, hidden: mx.array, token_embed: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Return (new_hidden, logits).

        new_hidden: [B, 1, D]
        logits: [B, 1, vocab_size] with argmax at token 42
        """
        B = hidden.shape[0]
        new_hidden = hidden + token_embed  # simple combination
        logits = mx.zeros((B, 1, self._vocab_size))
        # Make token 42 the argmax
        logits = logits.at[:, :, 42].add(10.0)
        return new_hidden, logits


# ---------------------------------------------------------------------------
# TestMTPTemperatureScaling
# ---------------------------------------------------------------------------


class TestMTPTemperatureScaling:
    """Test temperature scaling in _target_forward is functional (no in-place mutation)."""

    def test_functional_temperature_no_mutation(self):
        """Logits tensor is not modified in place by temperature scaling."""
        original_logits = mx.array(
            [[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        )
        # Copy for comparison
        original_copy = mx.array(original_logits)

        model = MockModel(original_logits)
        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler(0.5)],
        )
        engine, _ = make_engine(batch=batch, model=model)

        _ = engine._target_forward(batch, mx.array([[1]], dtype=mx.int32))

        # Original logits should be unchanged (functional, not in-place)
        mx.eval(original_logits, original_copy)
        assert mx.allclose(original_logits, original_copy).item()

    def test_temperature_zero_no_scaling(self):
        """temp=0 -> effective temp=1.0 (no-op division), argmax preserved."""
        logits = mx.array(
            [[[1.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        )

        batch = MockBatch(
            uids=[0],
            y=mx.array([1]),
            tokens=[mx.array([1])],
            num_tokens=[1],
            max_tokens=[100],
            cache=[],
            samplers=[MockSampler(0.0)],  # temp=0 -> 1.0 no-op
        )
        engine, _ = make_engine(batch=batch, model_logits=logits)

        probs = engine._target_forward(batch, mx.array([[1]], dtype=mx.int32))

        # softmax(logits / 1.0) = softmax(logits)
        expected = mx.softmax(logits, axis=-1)
        mx.eval(probs, expected)
        assert mx.allclose(probs, expected, atol=1e-5).item()

    def test_temperature_broadcast(self):
        """Different temps per batch item are correctly broadcast."""
        logits = mx.array([
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ])  # [2, 1, 10]

        batch = MockBatch(
            uids=[0, 1],
            y=mx.array([1, 2]),
            tokens=[mx.array([1]), mx.array([2])],
            num_tokens=[1, 1],
            max_tokens=[100, 100],
            cache=[],
            samplers=[MockSampler(0.5), MockSampler(2.0)],
        )
        engine, _ = make_engine(batch=batch, model_logits=logits)

        probs = engine._target_forward(batch, mx.array([[1], [2]], dtype=mx.int32))

        # Seq 0: temp=0.5 -> sharper distribution (higher peak)
        # Seq 1: temp=2.0 -> flatter distribution (lower peak)
        peak_0 = float(mx.max(probs[0, 0]))
        peak_1 = float(mx.max(probs[1, 0]))
        assert peak_0 > peak_1, (
            f"Lower temperature should produce sharper distribution: "
            f"peak_0={peak_0:.4f} should be > peak_1={peak_1:.4f}"
        )

        # Verify argmax is at position 2 for both (highest logit)
        assert int(mx.argmax(probs[0, 0])) == 2
        assert int(mx.argmax(probs[1, 0])) == 2

        # Verify the actual scaling: seq 0 logits/0.5, seq 1 logits/2.0
        expected_0 = mx.softmax(logits[0:1] / 0.5, axis=-1)
        expected_1 = mx.softmax(logits[1:2] / 2.0, axis=-1)
        mx.eval(expected_0, expected_1)
        assert mx.allclose(probs[0:1], expected_0, atol=1e-5).item()
        assert mx.allclose(probs[1:2], expected_1, atol=1e-5).item()
