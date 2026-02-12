"""Tests for MTP utility functions.

Covers forward_with_hidden, get_embed_tokens, get_lm_head, and get_final_norm.
Uses MagicMock and simple mock classes with mlx.core arrays.
No real model downloads required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_lm_server.spec_decode.mtp_utils import (
    forward_with_hidden,
    get_embed_tokens,
    get_final_norm,
    get_lm_head,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityLayer:
    """Mock decoder layer that acts as identity."""

    def __call__(self, h, mask=None, cache=None):
        return h


def _make_model(
    *,
    hidden_size: int = 64,
    vocab_size: int = 100,
    num_layers: int = 2,
    has_lm_head: bool = True,
    norm_scale: float = 1.0,
) -> MagicMock:
    """Build a mock model matching the backbone iteration interface."""
    mock = MagicMock()

    # embed_tokens
    def embed_fn(inputs):
        B, S = inputs.shape
        return mx.ones((B, S, hidden_size))

    mock.model.embed_tokens = MagicMock(side_effect=embed_fn)

    # layers list
    mock.model.layers = [_IdentityLayer() for _ in range(num_layers)]

    # norm - multiply by norm_scale so we can test pre vs post norm
    def norm_fn(h):
        return h * norm_scale

    mock.model.norm = MagicMock(side_effect=norm_fn)

    if has_lm_head:
        # lm_head: hidden [B, S, D] -> logits [B, S, V]
        def lm_head_fn(hidden):
            B, S, _ = hidden.shape
            return mx.random.normal((B, S, vocab_size))

        mock.lm_head = MagicMock(side_effect=lm_head_fn)
    else:
        # Remove lm_head attribute so hasattr returns False
        del mock.lm_head

        # Tied embeddings fallback: embed_tokens.as_linear -> logits [B, S, V]
        def as_linear_fn(hidden):
            B, S, _ = hidden.shape
            return mx.random.normal((B, S, vocab_size))

        mock.model.embed_tokens.as_linear = MagicMock(side_effect=as_linear_fn)

    return mock


# ===================================================================
# TestForwardWithHidden
# ===================================================================


class TestForwardWithHidden:
    """Tests for forward_with_hidden(model, inputs, cache)."""

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_returns_logits_and_hidden_shapes(self, mock_mask):
        """Output is (logits [B, S, V], hidden [B, S, D])."""
        model = _make_model(hidden_size=64, vocab_size=100)
        inputs = mx.array([[1, 2, 3]])  # [1, 3]

        logits, hidden = forward_with_hidden(model, inputs)
        mx.eval(logits, hidden)

        assert logits.shape == (1, 3, 100)
        assert hidden.shape == (1, 3, 64)

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_with_lm_head(self, mock_mask):
        """When model.lm_head exists, it is used for the projection."""
        model = _make_model(has_lm_head=True)
        inputs = mx.array([[5, 10]])

        logits, hidden = forward_with_hidden(model, inputs)
        mx.eval(logits, hidden)

        model.lm_head.assert_called_once()

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_tied_embeddings_fallback(self, mock_mask):
        """When model has no lm_head, embed_tokens.as_linear is used."""
        model = _make_model(has_lm_head=False)
        inputs = mx.array([[5, 10]])

        logits, hidden = forward_with_hidden(model, inputs)
        mx.eval(logits, hidden)

        model.model.embed_tokens.as_linear.assert_called_once()

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_cache_forwarded(self, mock_mask):
        """The cache entries are passed to each layer via zip iteration."""
        model = _make_model(num_layers=2)
        inputs = mx.array([[1]])
        fake_cache = [MagicMock(name="cache_0"), MagicMock(name="cache_1")]

        # Replace identity layers with MagicMock layers to track calls
        mock_layer_0 = MagicMock(side_effect=lambda h, mask, cache=None: h)
        mock_layer_1 = MagicMock(side_effect=lambda h, mask, cache=None: h)
        model.model.layers = [mock_layer_0, mock_layer_1]

        forward_with_hidden(model, inputs, cache=fake_cache)

        # Each layer should have been called with its respective cache entry
        mock_layer_0.assert_called_once()
        mock_layer_1.assert_called_once()
        # Verify cache entries were passed correctly
        _, kwargs_0 = mock_layer_0.call_args
        _, kwargs_1 = mock_layer_1.call_args
        assert kwargs_0["cache"] is fake_cache[0]
        assert kwargs_1["cache"] is fake_cache[1]


# ===================================================================
# TestForwardWithHiddenPreNorm
# ===================================================================


class TestForwardWithHiddenPreNorm:
    """Tests verifying pre-norm hidden state behavior."""

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_returns_prenorm_hidden(self, mock_mask):
        """Returned hidden is PRE-NORM: backbone.norm(returned_h) != returned_h."""
        model = _make_model(hidden_size=64, norm_scale=2.0)
        inputs = mx.array([[1, 2, 3]])
        logits, hidden = forward_with_hidden(model, inputs)
        mx.eval(logits, hidden)
        # hidden should be pre-norm (scale=1.0 since layers are identity)
        # If we apply norm ourselves, it should differ
        normed = model.model.norm(hidden)
        mx.eval(normed)
        # normed = hidden * 2.0, so they should differ
        assert not mx.array_equal(hidden, normed)

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_logits_use_normed_hidden(self, mock_mask):
        """Logits are computed from norm(hidden), not raw hidden."""
        model = _make_model(hidden_size=64, has_lm_head=True)
        inputs = mx.array([[1, 2]])
        logits, hidden = forward_with_hidden(model, inputs)
        mx.eval(logits, hidden)
        # lm_head should have been called with norm output
        model.lm_head.assert_called_once()
        # The argument to lm_head is model.model.norm(h), verify norm was called
        model.model.norm.assert_called_once()

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_cache_creates_default_when_none(self, mock_mask):
        """When cache=None, a list of [None]*len(layers) is used."""
        model = _make_model(num_layers=3)
        inputs = mx.array([[1]])
        logits, hidden = forward_with_hidden(model, inputs, cache=None)
        mx.eval(logits, hidden)
        # All 3 layers should have been called (identity layers)
        assert hidden.shape == (1, 1, 64)

    @patch("mlx_lm.models.base.create_attention_mask", return_value=None)
    def test_prenorm_shape_matches(self, mock_mask):
        """Pre-norm hidden has shape [B, S, D]."""
        model = _make_model(hidden_size=32)
        inputs = mx.array([[1, 2, 3, 4]])
        logits, hidden = forward_with_hidden(model, inputs)
        mx.eval(logits, hidden)
        assert hidden.shape == (1, 4, 32)


# ===================================================================
# TestGetEmbedTokens
# ===================================================================


class TestGetEmbedTokens:
    """Tests for get_embed_tokens(model)."""

    def test_returns_embed_tokens(self):
        """Returns model.model.embed_tokens."""
        model = _make_model()
        result = get_embed_tokens(model)
        assert result is model.model.embed_tokens


# ===================================================================
# TestGetLmHead
# ===================================================================


class TestGetLmHead:
    """Tests for get_lm_head(model)."""

    def test_returns_lm_head(self):
        """When model.lm_head exists, return it directly."""
        model = _make_model(has_lm_head=True)
        result = get_lm_head(model)
        assert result is model.lm_head

    def test_tied_returns_as_linear(self):
        """When model has no lm_head, return embed_tokens.as_linear."""
        model = _make_model(has_lm_head=False)
        result = get_lm_head(model)
        assert result is model.model.embed_tokens.as_linear


# ===================================================================
# TestGetFinalNorm
# ===================================================================


class TestGetFinalNorm:
    """Tests for get_final_norm(model)."""

    def test_returns_norm(self):
        """Returns model.model.norm."""
        model = _make_model()
        result = get_final_norm(model)
        assert result is model.model.norm
