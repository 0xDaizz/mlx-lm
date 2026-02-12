"""Tests for MTP (Multi-Token Prediction) module.

Covers MTPLayer forward pass, MTPModule construction, shared references,
predict/embed calls, and shape correctness. Uses mock objects throughout
(no real model downloads required).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm_server.spec_decode.mtp_module import MTPLayer, MTPModule


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


@dataclass
class MockArgs:
    """Minimal config object with fields needed by MTPLayer / decoder layer."""

    hidden_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5
    head_dim: int = 16  # hidden_size // num_attention_heads
    rope_theta: float = 10000.0
    vocab_size: int = 100


class MockDecoderLayer(nn.Module):
    """Minimal decoder layer mock that acts as an identity function."""

    def __init__(self, args):
        super().__init__()
        self._dim = args.hidden_size

    def __call__(self, x, mask=None, cache=None):
        return x


def _mock_decoder_factory(args):
    """Factory that returns a MockDecoderLayer."""
    return MockDecoderLayer(args)


class MockSharedNorm(nn.Module):
    """Mock norm that acts as identity."""

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x


class MockLMHead(nn.Module):
    """Mock lm_head that projects hidden_size -> vocab_size."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, x):
        return self.linear(x)


class MockEmbedTokens(nn.Module):
    """Mock embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def __call__(self, x):
        return self.embed(x)


# ---------------------------------------------------------------------------
# MTPLayer tests
# ---------------------------------------------------------------------------


class TestMTPLayer:
    """Tests for individual MTPLayer."""

    def test_init_creates_components(self):
        """Verify enorm, hnorm, eh_proj, and block are created."""
        args = MockArgs()
        layer = MTPLayer(args, _mock_decoder_factory)

        assert isinstance(layer.enorm, nn.RMSNorm)
        assert isinstance(layer.hnorm, nn.RMSNorm)
        assert isinstance(layer.eh_proj, nn.Linear)
        assert isinstance(layer.block, MockDecoderLayer)

    def test_forward_shape(self):
        """Forward pass with [1, 1, 64] inputs produces [1, 1, 64] output."""
        args = MockArgs(hidden_size=64)
        layer = MTPLayer(args, _mock_decoder_factory)
        mx.eval(layer.parameters())

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))
        mx.eval(hidden, token_embed)

        out = layer(hidden, token_embed)
        mx.eval(out)

        assert out.shape == (1, 1, 64)

    def test_forward_batch(self):
        """With batch_size=2, output shape is [2, 1, 64]."""
        args = MockArgs(hidden_size=64)
        layer = MTPLayer(args, _mock_decoder_factory)
        mx.eval(layer.parameters())

        hidden = mx.random.normal((2, 1, 64))
        token_embed = mx.random.normal((2, 1, 64))
        mx.eval(hidden, token_embed)

        out = layer(hidden, token_embed)
        mx.eval(out)

        assert out.shape == (2, 1, 64)


# ---------------------------------------------------------------------------
# MTPModule tests
# ---------------------------------------------------------------------------


class TestMTPModule:
    """Tests for the MTPModule orchestration layer."""

    def _make_module(
        self, num_layers: int = 2, hidden_size: int = 64, vocab_size: int = 100
    ) -> MTPModule:
        """Helper to construct a fully wired MTPModule."""
        args = MockArgs(hidden_size=hidden_size, vocab_size=vocab_size)
        mod = MTPModule(args, _mock_decoder_factory, num_layers=num_layers)

        norm = MockSharedNorm()
        lm_head = MockLMHead(hidden_size, vocab_size)
        embed_tokens = MockEmbedTokens(vocab_size, hidden_size)
        mx.eval(lm_head.parameters())
        mx.eval(embed_tokens.parameters())

        mod.set_shared_refs(norm, lm_head, embed_tokens)
        return mod

    def test_init_layer_count(self):
        """num_layers=2 creates exactly 2 MTPLayer instances."""
        args = MockArgs()
        mod = MTPModule(args, _mock_decoder_factory, num_layers=2)
        assert len(mod.layers) == 2
        for layer in mod.layers:
            assert isinstance(layer, MTPLayer)

    def test_num_layers_property(self):
        """.num_layers returns the correct count."""
        args = MockArgs()
        mod1 = MTPModule(args, _mock_decoder_factory, num_layers=1)
        mod3 = MTPModule(args, _mock_decoder_factory, num_layers=3)
        assert mod1.num_layers == 1
        assert mod3.num_layers == 3

    def test_set_shared_refs(self):
        """After calling set_shared_refs, attributes are correctly set."""
        args = MockArgs()
        mod = MTPModule(args, _mock_decoder_factory, num_layers=1)

        # Before set_shared_refs
        assert mod.shared_norm is None
        assert mod.shared_lm_head is None
        assert mod._embed_tokens is None

        norm = MockSharedNorm()
        lm_head = MockLMHead(64, 100)
        embed = MockEmbedTokens(100, 64)

        mod.set_shared_refs(norm, lm_head, embed)

        assert mod.shared_norm is norm
        assert mod.shared_lm_head is lm_head
        assert mod._embed_tokens is embed

    def test_predict_shape(self):
        """predict() returns (hidden, logits) with correct shapes."""
        mod = self._make_module(num_layers=2, hidden_size=64, vocab_size=100)
        mx.eval(mod.parameters())

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))
        mx.eval(hidden, token_embed)

        out_hidden, logits = mod.predict(
            depth=0, hidden=hidden, token_embed=token_embed
        )
        mx.eval(out_hidden, logits)

        assert out_hidden.shape == (1, 1, 64)
        assert logits.shape == (1, 1, 100)

    def test_predict_uses_shared_head(self):
        """Verify shared_norm and shared_lm_head are called during predict."""
        args = MockArgs(hidden_size=64, vocab_size=100)
        mod = MTPModule(args, _mock_decoder_factory, num_layers=1)
        mx.eval(mod.parameters())

        # Use MagicMock for the shared components so we can track calls
        mock_norm = MagicMock(side_effect=lambda x: x)
        mock_lm_head = MagicMock(
            side_effect=lambda x: mx.zeros((x.shape[0], x.shape[1], 100))
        )
        mock_embed = MagicMock()

        mod.set_shared_refs(mock_norm, mock_lm_head, mock_embed)

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))
        mx.eval(hidden, token_embed)

        out_hidden, logits = mod.predict(
            depth=0, hidden=hidden, token_embed=token_embed
        )
        mx.eval(out_hidden, logits)

        mock_norm.assert_called_once()
        mock_lm_head.assert_called_once()

    def test_get_embed(self):
        """get_embed() calls _embed_tokens and returns correct shape."""
        mod = self._make_module(num_layers=1, hidden_size=64, vocab_size=100)

        token_ids = mx.array([[5, 10, 15]])
        mx.eval(token_ids)

        embed = mod.get_embed(token_ids)
        mx.eval(embed)

        assert embed.shape == (1, 3, 64)

    def test_get_embed_uses_embed_tokens(self):
        """Verify get_embed delegates to _embed_tokens."""
        args = MockArgs(hidden_size=64, vocab_size=100)
        mod = MTPModule(args, _mock_decoder_factory, num_layers=1)

        mock_embed = MagicMock(
            side_effect=lambda x: mx.zeros((1, x.shape[-1], 64))
        )
        mod._embed_tokens = mock_embed

        token_ids = mx.array([[1, 2, 3]])
        mod.get_embed(token_ids)

        mock_embed.assert_called_once()

    def test_predict_different_depths(self):
        """predict() at depth=0 and depth=1 uses different MTP layers."""
        mod = self._make_module(num_layers=2, hidden_size=64, vocab_size=100)
        mx.eval(mod.parameters())

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))
        mx.eval(hidden, token_embed)

        h0, logits0 = mod.predict(depth=0, hidden=hidden, token_embed=token_embed)
        h1, logits1 = mod.predict(depth=1, hidden=hidden, token_embed=token_embed)
        mx.eval(h0, logits0, h1, logits1)

        # Both should have valid shapes
        assert h0.shape == (1, 1, 64)
        assert h1.shape == (1, 1, 64)
        assert logits0.shape == (1, 1, 100)
        assert logits1.shape == (1, 1, 100)


class TestMTPLayerConstructor:
    """Tests for layer_idx try/except in MTPLayer.__init__."""

    def test_layer_idx_passed_to_factory(self):
        """Factory that accepts (args, layer_idx) should work."""
        args = MockArgs()
        call_log = []

        class FactoryWithIdx(nn.Module):
            def __init__(self, args, layer_idx=None):
                super().__init__()
                call_log.append(('with_idx', layer_idx))
                self._dim = args.hidden_size
            def __call__(self, x, mask=None, cache=None):
                return x

        _layer = MTPLayer(args, FactoryWithIdx)
        assert len(call_log) == 1
        assert call_log[0] == ('with_idx', 0)

    def test_layer_idx_fallback_no_idx(self):
        """Factory that only accepts (args) should work via fallback."""
        args = MockArgs()
        # _mock_decoder_factory only takes (args), not (args, layer_idx)
        layer = MTPLayer(args, _mock_decoder_factory)
        assert isinstance(layer.block, MockDecoderLayer)


class TestMTPLayerMoEIdx:
    """Tests for H3: layer_idx parameter in MTPLayer/MTPModule."""

    def test_moe_layer_idx_passed(self):
        """When layer_idx > 0, it's passed to the factory."""
        args = MockArgs()
        call_log = []

        class FactoryWithIdx(nn.Module):
            def __init__(self, args, layer_idx=None):
                super().__init__()
                call_log.append(('with_idx', layer_idx))
                self._dim = args.hidden_size

            def __call__(self, x, mask=None, cache=None):
                return x

        _layer = MTPLayer(args, FactoryWithIdx, layer_idx=42)
        assert len(call_log) == 1
        assert call_log[0] == ('with_idx', 42)

    def test_dense_layer_idx_zero(self):
        """Default layer_idx=0 for dense MLP."""
        args = MockArgs()
        call_log = []

        class FactoryWithIdx(nn.Module):
            def __init__(self, args, layer_idx=None):
                super().__init__()
                call_log.append(('with_idx', layer_idx))
                self._dim = args.hidden_size

            def __call__(self, x, mask=None, cache=None):
                return x

        _layer = MTPLayer(args, FactoryWithIdx)  # default layer_idx=0
        assert call_log[0] == ('with_idx', 0)

    def test_inspect_fallback(self):
        """When inspect.signature fails, try/except fallback works."""
        args = MockArgs()

        # _mock_decoder_factory only takes (args), so inspect sees 1 param
        # and calls with just (args)
        layer = MTPLayer(args, _mock_decoder_factory, layer_idx=5)
        assert isinstance(layer.block, MockDecoderLayer)

    def test_module_passes_layer_idx(self):
        """MTPModule passes layer_idx to all MTPLayer instances."""
        args = MockArgs()
        call_log = []

        class FactoryWithIdx(nn.Module):
            def __init__(self, args, layer_idx=None):
                super().__init__()
                call_log.append(layer_idx)
                self._dim = args.hidden_size

            def __call__(self, x, mask=None, cache=None):
                return x

        _mod = MTPModule(args, FactoryWithIdx, num_layers=3, layer_idx=10)
        assert len(call_log) == 3
        assert all(idx == 10 for idx in call_log)


class TestMTPModuleGuards:
    """Tests for None guards and depth bounds in MTPModule."""

    def test_predict_raises_without_shared_refs(self):
        """predict() raises RuntimeError if set_shared_refs() not called."""
        args = MockArgs()
        mod = MTPModule(args, _mock_decoder_factory, num_layers=1)
        mx.eval(mod.parameters())

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))

        import pytest
        with pytest.raises(RuntimeError, match="set_shared_refs"):
            mod.predict(depth=0, hidden=hidden, token_embed=token_embed)

    def test_get_embed_raises_without_shared_refs(self):
        """get_embed() raises RuntimeError if _embed_tokens is None."""
        args = MockArgs()
        mod = MTPModule(args, _mock_decoder_factory, num_layers=1)

        import pytest
        with pytest.raises(RuntimeError, match="_embed_tokens is None"):
            mod.get_embed(mx.array([[1, 2, 3]]))

    def test_predict_depth_out_of_range_positive(self):
        """predict() with depth >= num_layers raises IndexError."""
        args = MockArgs()
        mod = MTPModule(args, _mock_decoder_factory, num_layers=2)
        norm = MockSharedNorm()
        lm_head = MockLMHead(64, 100)
        embed = MockEmbedTokens(100, 64)
        mx.eval(lm_head.parameters(), embed.parameters())
        mod.set_shared_refs(norm, lm_head, embed)
        mx.eval(mod.parameters())

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))

        import pytest
        with pytest.raises(IndexError, match="out of range"):
            mod.predict(depth=2, hidden=hidden, token_embed=token_embed)

    def test_predict_depth_negative(self):
        """predict() with negative depth raises IndexError."""
        args = MockArgs()
        mod = MTPModule(args, _mock_decoder_factory, num_layers=2)
        norm = MockSharedNorm()
        lm_head = MockLMHead(64, 100)
        embed = MockEmbedTokens(100, 64)
        mx.eval(lm_head.parameters(), embed.parameters())
        mod.set_shared_refs(norm, lm_head, embed)
        mx.eval(mod.parameters())

        hidden = mx.random.normal((1, 1, 64))
        token_embed = mx.random.normal((1, 1, 64))

        import pytest
        with pytest.raises(IndexError, match="out of range"):
            mod.predict(depth=-1, hidden=hidden, token_embed=token_embed)
