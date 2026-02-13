"""Tests for memory-safe chunked parameter evaluation."""

import gc

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.utils import _chunked_eval_params, _extract_layer_index


# ---------------------------------------------------------------------------
# _extract_layer_index tests
# ---------------------------------------------------------------------------

class TestExtractLayerIndex:
    def test_standard_layers_format(self):
        assert _extract_layer_index("model.layers.0.self_attn.q_proj.weight") == 0

    def test_deep_layer_index(self):
        assert _extract_layer_index("model.layers.42.mlp.gate_proj.weight") == 42

    def test_h_format(self):
        assert _extract_layer_index("transformer.h.10.attn.weight") == 10

    def test_blocks_format(self):
        assert _extract_layer_index("blocks.5.norm.weight") == 5

    def test_no_layer_structure(self):
        assert _extract_layer_index("model.embed_tokens.weight") is None

    def test_non_numeric_after_layers(self):
        assert _extract_layer_index("model.layers.final.weight") is None

    def test_lm_head(self):
        assert _extract_layer_index("lm_head.weight") is None

    def test_bare_layers(self):
        assert _extract_layer_index("layers.99.weight") == 99


# ---------------------------------------------------------------------------
# Helper model classes
# ---------------------------------------------------------------------------

class SimpleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(32, 32)
        self.mlp = nn.Linear(32, 32)


class SimpleInner(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)
        self.layers = [SimpleLayer() for _ in range(num_layers)]
        self.norm = nn.RMSNorm(32)


class SimpleModel(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.model = SimpleInner(num_layers)
        self.lm_head = nn.Linear(32, 100, bias=False)


# ---------------------------------------------------------------------------
# _chunked_eval_params tests
# ---------------------------------------------------------------------------

class TestChunkedEvalParams:
    def test_basic_materialisation(self):
        """All parameters should be materialised after chunked eval."""
        model = SimpleModel(num_layers=4)
        _chunked_eval_params(model)

        from mlx.utils import tree_flatten
        for name, param in tree_flatten(model.parameters()):
            # Accessing .shape should work; accessing values should not raise
            assert param.shape is not None
            # Force a read to confirm it's materialised
            _ = param.tolist()

    def test_equivalent_to_mx_eval(self):
        """Chunked eval should produce identical values to mx.eval."""
        mx.random.seed(42)
        model_a = SimpleModel(num_layers=3)

        mx.random.seed(42)
        model_b = SimpleModel(num_layers=3)

        # Method A: standard bulk eval
        mx.eval(model_a.parameters())

        # Method B: chunked eval
        _chunked_eval_params(model_b)

        from mlx.utils import tree_flatten
        params_a = dict(tree_flatten(model_a.parameters()))
        params_b = dict(tree_flatten(model_b.parameters()))

        assert set(params_a.keys()) == set(params_b.keys())
        for key in params_a:
            assert mx.array_equal(params_a[key], params_b[key]), f"Mismatch at {key}"

    def test_no_layers_fallback(self):
        """Model without layer structure should fall back gracefully."""
        model = nn.Linear(16, 16)
        _chunked_eval_params(model)

        from mlx.utils import tree_flatten
        for _, param in tree_flatten(model.parameters()):
            _ = param.tolist()

    def test_single_layer(self):
        """Should work with a single-layer model."""
        model = SimpleModel(num_layers=1)
        _chunked_eval_params(model)

        from mlx.utils import tree_flatten
        for _, param in tree_flatten(model.parameters()):
            _ = param.tolist()

    def test_many_layers(self):
        """Should handle many layers without issue."""
        model = SimpleModel(num_layers=32)
        _chunked_eval_params(model)

        from mlx.utils import tree_flatten
        count = sum(1 for _ in tree_flatten(model.parameters()))
        # 32 layers * (2 linear weights + 2 biases) + embed + norm + lm_head
        assert count > 32 * 2
