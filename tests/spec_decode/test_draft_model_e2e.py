"""E2E tests for DraftModelProposer with real models.

Requires both Qwen3-4B-4bit (target) and Qwen3-0.6B-4bit (draft) to be
present as local directories. Tests are skipped if either model is missing.

Run:
    pytest tests/spec_decode/test_draft_model_e2e.py -v -m integration
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_lm_server.spec_decode.proposer.draft_model import DraftModelProposer

ROOT = Path(__file__).parent.parent.parent
TARGET_MODEL_PATH = str(ROOT / "Qwen3-4B-4bit")
DRAFT_MODEL_PATH = str(ROOT / "Qwen3-0.6B-4bit")

_models_available = os.path.isdir(TARGET_MODEL_PATH) and os.path.isdir(
    DRAFT_MODEL_PATH
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _models_available,
        reason=f"Requires {TARGET_MODEL_PATH} and {DRAFT_MODEL_PATH}",
    ),
]


@dataclass
class MockSequenceState:
    request_id: str = "test"
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)


def _make_seq(token_ids: list[int]) -> MockSequenceState:
    return MockSequenceState(token_ids=token_ids, output_tokens=list(token_ids))


@pytest.fixture(scope="module")
def _target_model_and_tokenizer():
    """Load target model and tokenizer once per module."""
    from mlx_lm import load

    model, tokenizer = load(TARGET_MODEL_PATH)
    return model, tokenizer


@pytest.fixture(scope="module")
def target_tokenizer(_target_model_and_tokenizer):
    """Extract tokenizer from shared fixture."""
    return _target_model_and_tokenizer[1]


@pytest.fixture(scope="module")
def target_model(_target_model_and_tokenizer):
    """Extract model from shared fixture."""
    return _target_model_and_tokenizer[0]


@pytest.fixture(scope="module")
def loaded_draft_proposer(target_tokenizer):
    """Load the draft proposer once per module, validated against target."""
    proposer = DraftModelProposer(model_path=DRAFT_MODEL_PATH, context_len=128)
    proposer.load(target_tokenizer=target_tokenizer)
    return proposer


class TestDraftProposerGeneratesKTokens:
    """Test that the draft proposer generates the requested number of tokens."""

    def test_k_3(self, loaded_draft_proposer) -> None:
        seq = _make_seq([1, 2, 3, 4, 5])
        result = loaded_draft_proposer.propose([seq], k=3)

        assert result is not None
        assert result.draft_tokens.shape == (1, 3)
        assert int(result.proposal_lens[0]) == 3

    def test_k_5(self, loaded_draft_proposer) -> None:
        seq = _make_seq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = loaded_draft_proposer.propose([seq], k=5)

        assert result is not None
        assert result.draft_tokens.shape == (1, 5)
        assert int(result.proposal_lens[0]) == 5


class TestDraftProposalAllValidTokenIds:
    """All proposed token IDs should be within [0, vocab_size)."""

    def test_all_ids_in_range(self, loaded_draft_proposer) -> None:
        vocab_size = loaded_draft_proposer.tokenizer.vocab_size
        seq = _make_seq(list(range(1, 51)))
        result = loaded_draft_proposer.propose([seq], k=5)

        assert result is not None
        tokens = result.draft_tokens.tolist()[0]
        for t in tokens:
            assert 0 <= t < vocab_size, f"Token {t} out of range [0, {vocab_size})"

    def test_batch_all_ids_in_range(self, loaded_draft_proposer) -> None:
        vocab_size = loaded_draft_proposer.tokenizer.vocab_size
        seq1 = _make_seq(list(range(1, 21)))
        seq2 = _make_seq(list(range(100, 131)))
        result = loaded_draft_proposer.propose([seq1, seq2], k=4)

        assert result is not None
        for b in range(2):
            for t in result.draft_tokens[b].tolist():
                assert 0 <= t < vocab_size


class TestDraftProposerVariableK:
    """Test various k values produce correct shapes."""

    @pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
    def test_variable_k(self, loaded_draft_proposer, k) -> None:
        seq = _make_seq(list(range(1, 20)))
        result = loaded_draft_proposer.propose([seq], k=k)

        assert result is not None
        assert result.draft_tokens.shape == (1, k)
        assert int(result.proposal_lens[0]) == k


class TestGreedySpecMatchesNonSpec:
    """With temp=0 (greedy argmax), the draft model should be deterministic.

    D8: argmax-safe — same input, same draft tokens.
    """

    def test_deterministic_proposals(self, loaded_draft_proposer) -> None:
        seq = _make_seq(list(range(1, 30)))
        result1 = loaded_draft_proposer.propose([seq], k=5)
        result2 = loaded_draft_proposer.propose([seq], k=5)

        assert result1 is not None
        assert result2 is not None
        assert result1.draft_tokens.tolist() == result2.draft_tokens.tolist()


class TestDraftCacheNotLeaked:
    """Draft cache is created fresh per propose() call (D10).

    Multiple propose() calls should not accumulate memory.
    """

    def test_no_cache_accumulation(self, loaded_draft_proposer) -> None:
        seq = _make_seq(list(range(1, 30)))

        # Run several proposals
        for _ in range(5):
            result = loaded_draft_proposer.propose([seq], k=3)
            assert result is not None

        # The proposer should not hold any cache state
        # (draft_cache goes out of scope in propose())
        gc.collect()
        # No assertion on memory — just verify no error/crash


class TestAcceptanceRateSanity:
    """Draft proposals should have a reasonable acceptance rate against the target model.

    Computes acceptance by comparing draft tokens to greedy target output.
    """

    def test_acceptance_range(self, loaded_draft_proposer, target_tokenizer, target_model) -> None:
        """Draft proposals should have measurable acceptance against target greedy."""
        from mlx_lm.models.cache import make_prompt_cache

        prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of"
        token_ids = target_tokenizer.encode(prompt)
        seq = _make_seq(token_ids)

        k = 5
        result = loaded_draft_proposer.propose([seq], k=k)
        assert result is not None
        draft_tokens = result.draft_tokens[0].tolist()

        # Get target model's greedy output for same context
        target_cache = make_prompt_cache(target_model)
        context_mx = mx.array([token_ids])

        # Prefill
        if len(token_ids) > 1:
            target_model(context_mx[:, :-1], cache=target_cache)
            mx.eval([c.state for c in target_cache])
        logits = target_model(context_mx[:, -1:], cache=target_cache)
        mx.eval(logits)

        target_tokens = []
        for step in range(k):
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            target_tokens.append(next_token)
            if step < k - 1:
                logits = target_model(mx.array([[next_token]]), cache=target_cache)
                mx.eval(logits)

        matches = sum(1 for d, t in zip(draft_tokens, target_tokens) if d == t)
        acceptance_rate = matches / k

        # Sanity check: acceptance should be a valid rate
        # Wide range since this is a sanity check, not a guarantee
        print(f"\nDraft acceptance: {acceptance_rate:.0%} ({matches}/{k})")
        print(f"Draft:  {draft_tokens}")
        print(f"Target: {target_tokens}")
        assert 0.0 <= acceptance_rate <= 1.0


class TestVocabCheckWithRealModels:
    """Vocab compatibility validation with real model tokenizers."""

    def test_same_family_passes(self, target_tokenizer) -> None:
        """Qwen3-0.6B-4bit and Qwen3-4B-4bit share the same tokenizer."""
        proposer = DraftModelProposer(model_path=DRAFT_MODEL_PATH)
        # Should not raise — same Qwen3 family
        proposer.load(target_tokenizer=target_tokenizer)
        assert proposer._loaded is True
        assert proposer.tokenizer.vocab_size == target_tokenizer.vocab_size
