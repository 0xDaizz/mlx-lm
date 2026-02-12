"""Shared test fixtures and mocks for spec_decode tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.nn as nn


@dataclass
class MockArgs:
    """Minimal config object with fields needed by MTPLayer / decoder layer."""

    hidden_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5
    head_dim: int = 16
    rope_theta: float = 10000.0
    vocab_size: int = 100


class MockDecoderLayer(nn.Module):
    """Minimal decoder layer mock that acts as identity."""

    def __init__(self, args):
        super().__init__()
        self._dim = args.hidden_size

    def __call__(self, x, mask=None, cache=None):
        return x


def mock_decoder_factory(args):
    """Factory that returns a MockDecoderLayer (1-arg signature)."""
    return MockDecoderLayer(args)


@dataclass
class MockSequenceState:
    """Minimal SequenceState mock."""

    request_id: str
    token_ids: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
