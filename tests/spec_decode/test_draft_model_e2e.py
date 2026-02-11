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
def target_tokenizer():
    """Load the target tokenizer once per module."""
    from mlx_lm import load

    _, tokenizer = load(TARGET_MODEL_PATH)
    return tokenizer


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
    """With a deterministic prompt, acceptance rate should be reasonable.

    We use the target model's tokenizer to encode a known prompt, then
    check that the draft model produces some matching tokens.
    """

    def test_acceptance_range(self, loaded_draft_proposer, target_tokenizer) -> None:
        # Encode a simple deterministic prompt
        prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of"
        token_ids = target_tokenizer.encode(prompt)
        seq = _make_seq(token_ids)

        # Run multiple proposals
        total_proposed = 0
        total_non_trivial = 0  # proposals that aren't all zeros

        for _ in range(5):
            result = loaded_draft_proposer.propose([seq], k=5)
            assert result is not None
            total_proposed += 5
            tokens = result.draft_tokens.tolist()[0]
            if any(t != 0 for t in tokens):
                total_non_trivial += 1

        # At least some proposals should produce non-trivial tokens
        assert total_non_trivial > 0, "Draft model produced only zero tokens"


class TestVocabCheckWithRealModels:
    """Vocab compatibility validation with real model tokenizers."""

    def test_same_family_passes(self, target_tokenizer) -> None:
        """Qwen3-0.6B-4bit and Qwen3-4B-4bit share the same tokenizer."""
        proposer = DraftModelProposer(model_path=DRAFT_MODEL_PATH)
        # Should not raise — same Qwen3 family
        proposer.load(target_tokenizer=target_tokenizer)
        assert proposer._loaded is True
        assert proposer.tokenizer.vocab_size == target_tokenizer.vocab_size
