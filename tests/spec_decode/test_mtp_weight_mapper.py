"""Tests for MTP weight name mapper.

Uses fabricated weight key lists — no real model downloads needed.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_lm_server.spec_decode.mtp_weight_mapper import (
    _remap_sublayer,
    count_mtp_layers_from_weights,
    detect_pattern,
    extract_and_remap,
    get_num_mtp_layers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER = mx.zeros((4,))


def _make_weights(keys: list[str]) -> dict[str, mx.array]:
    """Build a weight dict from a list of keys, all mapped to a placeholder."""
    return {k: _PLACEHOLDER for k in keys}


# ===================================================================
# TestRemapSublayer
# ===================================================================


class TestRemapSublayer:
    """Direct tests for _remap_sublayer helper."""

    def test_enorm_passthrough(self):
        """enorm.weight passes through as layers.{depth}.enorm.weight."""
        assert _remap_sublayer(0, "enorm.weight") == "layers.0.enorm.weight"

    def test_hnorm_passthrough(self):
        assert _remap_sublayer(1, "hnorm.weight") == "layers.1.hnorm.weight"

    def test_eh_proj_passthrough(self):
        assert _remap_sublayer(0, "eh_proj.weight") == "layers.0.eh_proj.weight"

    def test_self_attn_becomes_block(self):
        """self_attn.* -> block.self_attn.*"""
        result = _remap_sublayer(0, "self_attn.q_proj.weight")
        assert result == "layers.0.block.self_attn.q_proj.weight"

    def test_mlp_becomes_block(self):
        result = _remap_sublayer(0, "mlp.gate_proj.weight")
        assert result == "layers.0.block.mlp.gate_proj.weight"

    def test_block_prefix_passthrough(self):
        """Remainder starting with 'block.' keeps as-is under layers.{depth}."""
        result = _remap_sublayer(0, "block.self_attn.q_proj.weight")
        assert result == "layers.0.block.self_attn.q_proj.weight"

    def test_unknown_sublayer_fallback_to_block(self):
        """Unknown remainder goes to block.* with a warning."""
        result = _remap_sublayer(0, "some_unknown.weight")
        assert result == "layers.0.block.some_unknown.weight"

    def test_input_layernorm_becomes_block(self):
        result = _remap_sublayer(0, "input_layernorm.weight")
        assert result == "layers.0.block.input_layernorm.weight"

    def test_post_attention_layernorm_becomes_block(self):
        result = _remap_sublayer(0, "post_attention_layernorm.weight")
        assert result == "layers.0.block.post_attention_layernorm.weight"


# ===================================================================
# TestGetNumMtpLayers
# ===================================================================


class TestGetNumMtpLayers:
    def test_num_nextn_predict_layers(self):
        config = {"num_nextn_predict_layers": 1}
        assert get_num_mtp_layers(config) == 1

    def test_num_mtp_layers(self):
        config = {"num_mtp_layers": 2}
        assert get_num_mtp_layers(config) == 2

    def test_no_config_key(self):
        assert get_num_mtp_layers({}) == 0

    def test_zero_value(self):
        config = {"num_nextn_predict_layers": 0}
        assert get_num_mtp_layers(config) == 0

    def test_priority_order(self):
        """First matching key wins — num_nextn_predict_layers over num_mtp_layers."""
        config = {"num_nextn_predict_layers": 3, "num_mtp_layers": 5}
        assert get_num_mtp_layers(config) == 3

    def test_n_predict_key(self):
        """n_predict key is supported."""
        config = {"n_predict": 3}
        assert get_num_mtp_layers(config) == 3

    def test_num_predict_layers_key(self):
        """num_predict_layers key is supported."""
        config = {"num_predict_layers": 2}
        assert get_num_mtp_layers(config) == 2


# ===================================================================
# TestDetectPattern
# ===================================================================


class TestDetectPattern:
    def test_detect_layer_index(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.27.mlp.down_proj.weight",
            "model.layers.28.enorm.weight",  # MTP layer (28 >= 28)
            "model.layers.28.hnorm.weight",
        ]
        result = detect_pattern(keys, model_type="deepseek_v3", num_hidden_layers=28)
        assert result == "layer_index"

    def test_detect_model_mtp(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.mtp.0.enorm.weight",
            "model.mtp.0.block.self_attn.q_proj.weight",
        ]
        result = detect_pattern(keys, model_type="mimo", num_hidden_layers=32)
        assert result == "model_mtp"

    def test_detect_model_mtp_layers(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.mtp_layers.0.enorm.weight",
            "model.mtp_layers.0.block.self_attn.q_proj.weight",
        ]
        result = detect_pattern(keys, model_type="mimo", num_hidden_layers=32)
        assert result == "model_mtp"

    def test_detect_ernie(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "mtp_hidden_norm.0.weight",
            "mtp_block.0.self_attn.q_proj.weight",
        ]
        result = detect_pattern(keys, model_type="ernie", num_hidden_layers=48)
        assert result == "ernie"

    def test_detect_flat_mtp(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "mtp.0.enorm.weight",
            "mtp.0.block.self_attn.q_proj.weight",
        ]
        result = detect_pattern(keys, model_type="qwen3_next", num_hidden_layers=32)
        assert result == "flat_mtp"

    def test_no_mtp_weights(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        result = detect_pattern(keys, model_type="llama", num_hidden_layers=32)
        assert result is None

    def test_layer_index_not_triggered_for_normal_layers(self):
        """model.layers.10 with num_hidden_layers=28 should NOT trigger layer_index."""
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.10.mlp.down_proj.weight",
            "model.layers.27.self_attn.o_proj.weight",
        ]
        result = detect_pattern(keys, model_type="deepseek_v3", num_hidden_layers=28)
        assert result is None


# ===================================================================
# TestExtractAndRemap
# ===================================================================


class TestExtractAndRemap:
    def test_layer_index_remap(self):
        """DeepSeek-style: model.layers.{N} where N >= num_hidden_layers."""
        weights = _make_weights([
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.27.mlp.down_proj.weight",
            "model.layers.28.enorm.weight",
            "model.layers.28.hnorm.weight",
            "model.layers.28.eh_proj.weight",
            "model.layers.28.self_attn.q_proj.weight",
        ])
        result = extract_and_remap(weights, "layer_index", num_hidden_layers=28)

        assert "layers.0.enorm.weight" in result
        assert "layers.0.hnorm.weight" in result
        assert "layers.0.eh_proj.weight" in result
        assert "layers.0.block.self_attn.q_proj.weight" in result
        # Normal layers should NOT appear in the result
        assert "layers.0.block.self_attn.q_proj.weight" in result
        assert all(not k.startswith("model.layers.0.") for k in result)

    def test_model_mtp_remap(self):
        """MiMo-style: model.mtp.{k}.xxx."""
        weights = _make_weights([
            "model.layers.0.self_attn.q_proj.weight",
            "model.mtp.0.enorm.weight",
            "model.mtp.0.hnorm.weight",
            "model.mtp.0.self_attn.q_proj.weight",
            "model.mtp.1.enorm.weight",
        ])
        result = extract_and_remap(weights, "model_mtp", num_hidden_layers=32)

        assert "layers.0.enorm.weight" in result
        assert "layers.0.hnorm.weight" in result
        assert "layers.0.block.self_attn.q_proj.weight" in result
        assert "layers.1.enorm.weight" in result
        assert len(result) == 4

    def test_ernie_remap(self):
        """Ernie-style: mtp_hidden_norm, mtp_embed_norm, mtp_block, etc."""
        weights = _make_weights([
            "model.layers.0.self_attn.q_proj.weight",
            "mtp_hidden_norm.0.weight",
            "mtp_embed_norm.0.weight",
            "mtp_proj.0.weight",
            "mtp_block.0.self_attn.q_proj.weight",
        ])
        result = extract_and_remap(weights, "ernie", num_hidden_layers=48)

        assert "layers.0.hnorm.weight" in result
        assert "layers.0.enorm.weight" in result
        assert "layers.0.eh_proj.weight" in result
        assert "layers.0.block.self_attn.q_proj.weight" in result
        assert len(result) == 4

    def test_flat_mtp_remap(self):
        """Qwen3-Next-style: mtp.{k}.xxx."""
        weights = _make_weights([
            "model.layers.0.self_attn.q_proj.weight",
            "mtp.0.enorm.weight",
            "mtp.0.hnorm.weight",
            "mtp.0.mlp.gate_proj.weight",
        ])
        result = extract_and_remap(weights, "flat_mtp", num_hidden_layers=32)

        assert "layers.0.enorm.weight" in result
        assert "layers.0.hnorm.weight" in result
        assert "layers.0.block.mlp.gate_proj.weight" in result
        assert len(result) == 3

    def test_block_sublayers_remapped(self):
        """self_attn, mlp, etc. should go under the block prefix."""
        weights = _make_weights([
            "mtp.0.self_attn.q_proj.weight",
            "mtp.0.self_attn.k_proj.weight",
            "mtp.0.mlp.gate_proj.weight",
            "mtp.0.mlp.up_proj.weight",
            "mtp.0.input_layernorm.weight",
            "mtp.0.post_attention_layernorm.weight",
        ])
        result = extract_and_remap(weights, "flat_mtp", num_hidden_layers=32)

        assert "layers.0.block.self_attn.q_proj.weight" in result
        assert "layers.0.block.self_attn.k_proj.weight" in result
        assert "layers.0.block.mlp.gate_proj.weight" in result
        assert "layers.0.block.mlp.up_proj.weight" in result
        assert "layers.0.block.input_layernorm.weight" in result
        assert "layers.0.block.post_attention_layernorm.weight" in result
        assert len(result) == 6


# ===================================================================
# TestCountMtpLayersFromWeights
# ===================================================================


class TestCountMtpLayersFromWeights:
    def test_count_single_layer(self):
        keys = [
            "model.layers.28.enorm.weight",
            "model.layers.28.hnorm.weight",
            "model.layers.28.self_attn.q_proj.weight",
        ]
        assert count_mtp_layers_from_weights(keys, "layer_index", num_hidden_layers=28) == 1

    def test_count_multiple_layers(self):
        keys = [
            "mtp.0.enorm.weight",
            "mtp.0.hnorm.weight",
            "mtp.1.enorm.weight",
            "mtp.1.hnorm.weight",
        ]
        assert count_mtp_layers_from_weights(keys, "flat_mtp", num_hidden_layers=32) == 2

    def test_count_empty(self):
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
        ]
        assert count_mtp_layers_from_weights(keys, "flat_mtp", num_hidden_layers=32) == 0

    def test_count_model_mtp_pattern(self):
        """model.mtp.{k}.* pattern is correctly counted."""
        keys = [
            "model.mtp.0.enorm.weight",
            "model.mtp.0.hnorm.weight",
            "model.mtp.1.enorm.weight",
        ]
        assert count_mtp_layers_from_weights(keys, "model_mtp", num_hidden_layers=32) == 2

    def test_count_model_mtp_layers_pattern(self):
        """model.mtp_layers.{k}.* pattern is correctly counted."""
        keys = [
            "model.mtp_layers.0.enorm.weight",
            "model.mtp_layers.0.hnorm.weight",
        ]
        assert count_mtp_layers_from_weights(keys, "model_mtp", num_hidden_layers=32) == 1


class TestErniePatternFix:
    """Tests for Ernie regex fixes (mtp_emb_norm, mtp_linear_proj)."""

    def test_ernie_emb_norm_detected(self):
        """mtp_emb_norm should trigger ernie pattern detection."""
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "mtp_emb_norm.0.weight",
            "mtp_block.0.self_attn.q_proj.weight",
        ]
        result = detect_pattern(keys, model_type="ernie", num_hidden_layers=48)
        assert result == "ernie"

    def test_ernie_linear_proj_detected(self):
        """mtp_linear_proj should be remapped to eh_proj."""
        weights = _make_weights([
            "mtp_hidden_norm.0.weight",
            "mtp_emb_norm.0.weight",
            "mtp_linear_proj.0.weight",
            "mtp_block.0.self_attn.q_proj.weight",
        ])
        result = extract_and_remap(weights, "ernie", num_hidden_layers=48)
        assert "layers.0.hnorm.weight" in result
        assert "layers.0.enorm.weight" in result
        assert "layers.0.eh_proj.weight" in result
        assert "layers.0.block.self_attn.q_proj.weight" in result

    def test_ernie_count_with_emb_norm(self):
        """count_mtp_layers_from_weights should count mtp_emb_norm keys."""
        keys = [
            "mtp_emb_norm.0.weight",
            "mtp_emb_norm.1.weight",
            "mtp_block.0.self_attn.q_proj.weight",
        ]
        count = count_mtp_layers_from_weights(keys, "ernie", num_hidden_layers=48)
        assert count == 2
