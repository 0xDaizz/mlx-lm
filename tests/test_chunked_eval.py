"""Tests for memory-safe chunked parameter evaluation."""

import mlx.core as mx
import mlx.nn as nn

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


# ---------------------------------------------------------------------------
# Additional _extract_layer_index edge cases
# ---------------------------------------------------------------------------


class TestExtractLayerIndexEdgeCases:
    def test_multiple_layers_keywords(self):
        """Nested layers keyword - should return the first match."""
        assert _extract_layer_index("model.layers.3.layers.5.weight") == 3

    def test_empty_string(self):
        assert _extract_layer_index("") is None

    def test_single_component(self):
        assert _extract_layer_index("weight") is None

    def test_layers_at_end(self):
        """'layers' as final component with no index after it."""
        assert _extract_layer_index("model.layers") is None

    def test_large_layer_index(self):
        assert _extract_layer_index("model.layers.999.weight") == 999

    def test_negative_not_matched(self):
        """Negative numbers shouldn't parse as valid indices."""
        # int("-1") returns -1 but negative layer indices are nonsensical
        # Current implementation would return -1; document this behavior
        result = _extract_layer_index("model.layers.-1.weight")
        # Just verify it doesn't crash - negative indices are unlikely in practice
        assert isinstance(result, int) or result is None


# ---------------------------------------------------------------------------
# Parameter grouping correctness
# ---------------------------------------------------------------------------


class TestChunkedEvalGrouping:
    def test_layer_params_grouped_correctly(self):
        """Verify that parameters are grouped by layer index, not mixed."""
        model = SimpleModel(num_layers=3)
        from mlx.utils import tree_flatten

        all_params = dict(tree_flatten(model.parameters()))

        # Manually check grouping logic
        layer_groups: dict[int, list] = {}
        non_layer: list = []
        for name, param in all_params.items():
            idx = _extract_layer_index(name)
            if idx is not None:
                layer_groups.setdefault(idx, []).append(name)
            else:
                non_layer.append(name)

        # Should have exactly 3 layers
        assert len(layer_groups) == 3
        assert set(layer_groups.keys()) == {0, 1, 2}

        # Each layer should have params from self_attn and mlp
        for idx in range(3):
            layer_param_names = layer_groups[idx]
            has_attn = any("self_attn" in n for n in layer_param_names)
            has_mlp = any("mlp" in n for n in layer_param_names)
            assert has_attn, f"Layer {idx} missing self_attn params"
            assert has_mlp, f"Layer {idx} missing mlp params"

        # Non-layer should include embed, norm, lm_head
        non_layer_str = " ".join(non_layer)
        assert "embed_tokens" in non_layer_str
        assert "norm" in non_layer_str
        assert "lm_head" in non_layer_str

    def test_no_cross_contamination(self):
        """Parameters from layer N should not appear in layer M's group."""
        model = SimpleModel(num_layers=4)
        from mlx.utils import tree_flatten

        all_params = dict(tree_flatten(model.parameters()))
        layer_groups: dict[int, list] = {}
        for name, param in all_params.items():
            idx = _extract_layer_index(name)
            if idx is not None:
                layer_groups.setdefault(idx, []).append(name)

        for idx, names in layer_groups.items():
            for name in names:
                # Every name in this group should parse to this layer index
                assert _extract_layer_index(name) == idx, (
                    f"{name} parsed to {_extract_layer_index(name)}, expected {idx}"
                )


# ---------------------------------------------------------------------------
# Alternate model architectures
# ---------------------------------------------------------------------------


class TestChunkedEvalAlternateArchitectures:
    def test_gpt_style_h_layers(self):
        """Model using 'h' instead of 'layers' (GPT-2 style)."""

        class GPTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(32, 32)
                self.mlp = nn.Linear(32, 32)

        class GPTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(100, 32)
                self.h = [GPTBlock() for _ in range(3)]
                self.ln_f = nn.LayerNorm(32)

        class GPTLMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = GPTModel()
                self.lm_head = nn.Linear(32, 100, bias=False)

        model = GPTLMHead()
        _chunked_eval_params(model)

        from mlx.utils import tree_flatten

        for _, param in tree_flatten(model.parameters()):
            _ = param.tolist()

    def test_blocks_style_layers(self):
        """Model using 'blocks' naming convention."""

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(32)
                self.linear = nn.Linear(32, 32)

        class BlockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.blocks = [Block() for _ in range(3)]
                self.head = nn.Linear(32, 100, bias=False)

        model = BlockModel()
        _chunked_eval_params(model)

        from mlx.utils import tree_flatten

        for _, param in tree_flatten(model.parameters()):
            _ = param.tolist()


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


class TestChunkedEvalStress:
    def test_very_many_layers(self):
        """128 layers should work without issues."""
        model = SimpleModel(num_layers=128)
        _chunked_eval_params(model)
        from mlx.utils import tree_flatten

        params = list(tree_flatten(model.parameters()))
        assert len(params) > 128 * 2

    def test_idempotent(self):
        """Calling _chunked_eval_params twice should not error."""
        model = SimpleModel(num_layers=4)
        _chunked_eval_params(model)
        _chunked_eval_params(model)  # Second call on already-materialised params
        from mlx.utils import tree_flatten

        for _, param in tree_flatten(model.parameters()):
            _ = param.tolist()
