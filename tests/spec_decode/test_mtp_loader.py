"""Tests for MTP weight loader functions.

Covers _get_weight_files, _extract_model_args, load_mtp_weights,
build_mtp_module, and _dequant_fp8_weights.
Uses tmp_path for filesystem tests, MagicMock for model objects,
and unittest.mock.patch for mocking mx.load / filesystem ops.
No real model downloads required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_lm_server.spec_decode.mtp_loader import (
    _dequant_fp8_weights,
    _extract_model_args,
    _get_weight_files,
    _identify_mtp_shards,
    build_mtp_module,
    load_mtp_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER = mx.zeros((4,))


# ===================================================================
# TestGetWeightFiles
# ===================================================================


class TestGetWeightFiles:
    """Tests for _get_weight_files(path)."""

    def test_index_file(self, tmp_path):
        """When model.safetensors.index.json exists, return listed shards."""
        # Create shard files
        (tmp_path / "model-00001-of-00002.safetensors").touch()
        (tmp_path / "model-00002-of-00002.safetensors").touch()

        # Create index file
        index = {
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00001-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        result = _get_weight_files(tmp_path)

        assert len(result) == 2
        filenames = sorted(p.name for p in result)
        assert filenames == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def test_single_file(self, tmp_path):
        """When model.safetensors exists (no index), return it as single-element list."""
        (tmp_path / "model.safetensors").touch()

        result = _get_weight_files(tmp_path)

        assert len(result) == 1
        assert result[0].name == "model.safetensors"

    def test_glob_fallback(self, tmp_path):
        """When neither index nor model.safetensors exists, glob *.safetensors."""
        (tmp_path / "weights_part1.safetensors").touch()
        (tmp_path / "weights_part2.safetensors").touch()
        # Non-safetensors files should be excluded
        (tmp_path / "config.json").touch()

        result = _get_weight_files(tmp_path)

        assert len(result) == 2
        filenames = sorted(p.name for p in result)
        assert filenames == [
            "weights_part1.safetensors",
            "weights_part2.safetensors",
        ]

    def test_no_files(self, tmp_path):
        """Empty directory returns empty list."""
        result = _get_weight_files(tmp_path)
        assert result == []


# ===================================================================
# TestExtractModelArgs
# ===================================================================


class TestExtractModelArgs:
    """Tests for _extract_model_args(model, config)."""

    def test_from_model_args(self):
        """When model.args exists, return it directly."""
        model = MagicMock()
        model.args = SimpleNamespace(hidden_size=128)

        result = _extract_model_args(model, {})

        assert result is model.args

    def test_from_model_config(self):
        """When model.config exists (but no model.args), return model.config."""
        model = MagicMock(spec=[])
        model.config = SimpleNamespace(hidden_size=128)

        result = _extract_model_args(model, {})

        assert result is model.config

    def test_from_model_model_args(self):
        """When model.model.args exists (but no model.args or model.config), return it."""
        model = MagicMock(spec=[])
        model.model = MagicMock(spec=[])
        model.model.args = SimpleNamespace(hidden_size=128)

        result = _extract_model_args(model, {})

        assert result is model.model.args

    def test_fallback_simple_namespace(self):
        """When model has no args/config attrs, construct from config dict."""
        model = MagicMock(spec=[])

        config = {
            "hidden_size": 256,
            "num_attention_heads": 8,
            "rms_norm_eps": 1e-6,
            "head_dim": 32,
        }

        result = _extract_model_args(model, config)

        assert isinstance(result, SimpleNamespace)
        assert result.hidden_size == 256
        assert result.num_attention_heads == 8
        assert result.rms_norm_eps == 1e-6
        assert result.head_dim == 32

    def test_fallback_computes_head_dim(self):
        """When head_dim is not in config, it is auto-computed from hidden_size / num_attention_heads."""
        model = MagicMock(spec=[])

        config = {
            "hidden_size": 512,
            "num_attention_heads": 8,
        }

        result = _extract_model_args(model, config)

        assert isinstance(result, SimpleNamespace)
        assert result.head_dim == 64  # 512 // 8

    def test_fallback_hidden_size_zero_returns_none(self):
        """When hidden_size is 0 (or missing), return None."""
        model = MagicMock(spec=[])

        config = {"num_attention_heads": 4}  # no hidden_size

        result = _extract_model_args(model, config)

        assert result is None


# ===================================================================
# TestLoadMtpWeights
# ===================================================================


class TestLoadMtpWeights:
    """Tests for load_mtp_weights(model_path)."""

    def test_returns_none_no_config(self, tmp_path):
        """If config.json is missing, return None."""
        result = load_mtp_weights(str(tmp_path))
        assert result is None

    def test_returns_none_no_safetensors(self, tmp_path):
        """If no safetensors files exist, return None."""
        config = {"model_type": "test", "num_hidden_layers": 4}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = load_mtp_weights(str(tmp_path))
        assert result is None

    @patch("mlx_lm_server.spec_decode.mtp_loader.mx.load")
    def test_returns_none_no_pattern(self, mock_mx_load, tmp_path):
        """If no MTP pattern is detected, return None."""
        config = {"model_type": "llama", "num_hidden_layers": 32}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").touch()

        # Weights with no MTP keys
        mock_mx_load.return_value = {
            "model.layers.0.self_attn.q_proj.weight": _PLACEHOLDER,
            "model.layers.1.mlp.down_proj.weight": _PLACEHOLDER,
            "lm_head.weight": _PLACEHOLDER,
        }

        result = load_mtp_weights(str(tmp_path))
        assert result is None

    @patch("mlx_lm_server.spec_decode.mtp_loader.mx.load")
    def test_loads_and_remaps_flat_mtp(self, mock_mx_load, tmp_path):
        """flat_mtp pattern weights are correctly loaded and remapped."""
        config = {
            "model_type": "qwen3_next",
            "num_hidden_layers": 32,
            "num_mtp_layers": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").touch()

        mock_mx_load.return_value = {
            "model.layers.0.self_attn.q_proj.weight": _PLACEHOLDER,
            "mtp.0.enorm.weight": _PLACEHOLDER,
            "mtp.0.hnorm.weight": _PLACEHOLDER,
            "mtp.0.self_attn.q_proj.weight": _PLACEHOLDER,
        }

        result = load_mtp_weights(str(tmp_path))

        assert result is not None
        assert "layers.0.enorm.weight" in result
        assert "layers.0.hnorm.weight" in result
        assert "layers.0.block.self_attn.q_proj.weight" in result

    @patch("mlx_lm_server.spec_decode.mtp_loader.mx.load")
    def test_single_pass_loading(self, mock_mx_load, tmp_path):
        """mx.load is called only for MTP shards when index exists (selective loading)."""
        config = {
            "model_type": "qwen3_next",
            "num_hidden_layers": 32,
            "num_mtp_layers": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Create two shard files via index
        (tmp_path / "shard-00001.safetensors").touch()
        (tmp_path / "shard-00002.safetensors").touch()
        index = {
            "weight_map": {
                "model.layers.0.weight": "shard-00001.safetensors",
                "mtp.0.enorm.weight": "shard-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        mock_mx_load.return_value = {
            "mtp.0.enorm.weight": _PLACEHOLDER,
            "mtp.0.hnorm.weight": _PLACEHOLDER,
        }

        result = load_mtp_weights(str(tmp_path))

        # With selective loading, only shard-00002 (containing MTP keys) is loaded
        assert mock_mx_load.call_count == 1
        loaded_path = mock_mx_load.call_args[0][0]
        assert "shard-00002" in loaded_path
        assert result is not None


# ===================================================================
# TestBuildMtpModule
# ===================================================================


class TestBuildMtpModule:
    """Tests for build_mtp_module(model, model_path)."""

    @patch(
        "mlx_lm_server.spec_decode.mtp_loader._load_mtp_weights_with_meta",
        return_value=None,
    )
    def test_returns_none_when_no_weights(self, mock_load):
        """If _load_mtp_weights_with_meta returns None, build_mtp_module returns None."""
        model = MagicMock()
        result = build_mtp_module(model, "/fake/path")
        assert result is None
        mock_load.assert_called_once_with("/fake/path")


# ===================================================================
# TestDequantFp8
# ===================================================================


class TestDequantFp8:
    """Tests for _dequant_fp8_weights."""

    def test_dequant_with_scale_inv(self):
        """FP8 weight + scale_inv pair is dequantized to bfloat16."""
        # Create a small weight that aligns with block size 128
        # Shape: (128, 128) so no padding needed, scale shape: (1, 1)
        # mx.from_fp8 requires uint8 input (FP8 representation)
        weight = mx.ones((128, 128), dtype=mx.uint8)
        scale_inv = mx.array([[2.0]], dtype=mx.bfloat16)

        weights = {
            "layers.0.block.self_attn.q_proj.weight": weight,
            "layers.0.block.self_attn.q_proj.weight_scale_inv": scale_inv,
        }

        result = _dequant_fp8_weights(weights)

        # Scale key should be removed
        assert "layers.0.block.self_attn.q_proj.weight_scale_inv" not in result
        # Weight should still exist
        assert "layers.0.block.self_attn.q_proj.weight" in result

        dequantized = result["layers.0.block.self_attn.q_proj.weight"]
        mx.eval(dequantized)

        assert dequantized.dtype == mx.bfloat16
        assert dequantized.shape == (128, 128)

    def test_no_scale_keys_passthrough(self):
        """When no _scale_inv keys are present, weights pass through unchanged."""
        w1 = mx.ones((4, 4))
        w2 = mx.zeros((4, 4))

        weights = {
            "layers.0.enorm.weight": w1,
            "layers.0.hnorm.weight": w2,
        }

        result = _dequant_fp8_weights(weights)

        assert len(result) == 2
        assert "layers.0.enorm.weight" in result
        assert "layers.0.hnorm.weight" in result
        # Values should be the same arrays
        assert mx.array_equal(result["layers.0.enorm.weight"], w1)
        assert mx.array_equal(result["layers.0.hnorm.weight"], w2)


# ===================================================================
# Wave 3B: TestDequantFp8Fixed (C2 fix tests)
# ===================================================================


class TestDequantFp8Fixed:
    """Tests for C2 fix: from_fp8 instead of astype."""

    @patch("mlx_lm_server.spec_decode.mtp_loader.mx.from_fp8")
    def test_from_fp8_used_not_astype(self, mock_from_fp8):
        """_dequant should call mx.from_fp8, not .astype."""
        from mlx_lm_server.spec_decode.mtp_loader import _dequant

        weight = mx.ones((128, 128), dtype=mx.bfloat16)
        scale_inv = mx.array([[1.0]], dtype=mx.bfloat16)
        mock_from_fp8.return_value = weight  # identity for simplicity
        _dequant(weight, scale_inv)
        mock_from_fp8.assert_called_once()

    def test_dequant_non_aligned_shapes(self):
        """Non block-aligned shapes (130, 200) should still work with padding."""
        from mlx_lm_server.spec_decode.mtp_loader import _dequant

        # 130 rounds up to 256 (2*128), 200 rounds up to 256 (2*128)
        # scale shape: (ceil(130/128), ceil(200/128)) = (2, 2)
        # mx.from_fp8 requires uint8 input
        weight = mx.ones((130, 200), dtype=mx.uint8)
        scale_inv = mx.array([[1.0, 1.0], [1.0, 1.0]], dtype=mx.bfloat16)
        result = _dequant(weight, scale_inv)
        mx.eval(result)
        assert result.shape == (130, 200)
        assert result.dtype == mx.bfloat16


# ===================================================================
# Wave 3D: TestSelectiveShard (H1 selective shard tests)
# ===================================================================


class TestSelectiveShard:
    """Tests for H1: selective shard loading."""

    def test_identify_mtp_shards_layer_index(self, tmp_path):
        """Only shards with MTP keys (layer_idx >= num_hidden) are returned."""
        index = {
            "weight_map": {
                "model.layers.0.weight": "shard-001.safetensors",
                "model.layers.27.weight": "shard-002.safetensors",
                "model.layers.28.enorm.weight": "shard-003.safetensors",
                "model.layers.28.hnorm.weight": "shard-003.safetensors",
                "lm_head.weight": "shard-001.safetensors",
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))
        result = _identify_mtp_shards(index_path, num_hidden_layers=28)
        assert result == ["shard-003.safetensors"]

    def test_identify_mtp_shards_model_mtp(self, tmp_path):
        """model.mtp.* pattern shards are identified."""
        index = {
            "weight_map": {
                "model.layers.0.weight": "shard-001.safetensors",
                "model.mtp.0.enorm.weight": "shard-002.safetensors",
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))
        result = _identify_mtp_shards(index_path, num_hidden_layers=32)
        assert result == ["shard-002.safetensors"]

    @patch("mlx_lm_server.spec_decode.mtp_loader.mx.load")
    def test_only_mtp_shards_loaded(self, mock_mx_load, tmp_path):
        """When index exists, only MTP-containing shards are loaded."""
        config = {
            "model_type": "deepseek_v3",
            "num_hidden_layers": 28,
            "num_nextn_predict_layers": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "shard-001.safetensors").touch()
        (tmp_path / "shard-002.safetensors").touch()
        (tmp_path / "shard-003.safetensors").touch()
        index = {
            "weight_map": {
                "model.layers.0.weight": "shard-001.safetensors",
                "model.layers.27.weight": "shard-002.safetensors",
                "model.layers.28.enorm.weight": "shard-003.safetensors",
                "model.layers.28.hnorm.weight": "shard-003.safetensors",
                "model.layers.28.self_attn.q_proj.weight": "shard-003.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        mock_mx_load.return_value = {
            "model.layers.28.enorm.weight": mx.zeros((4,)),
            "model.layers.28.hnorm.weight": mx.zeros((4,)),
            "model.layers.28.self_attn.q_proj.weight": mx.zeros((4,)),
        }
        load_mtp_weights(str(tmp_path))
        # Only shard-003 should be loaded
        assert mock_mx_load.call_count == 1
        loaded_path = mock_mx_load.call_args[0][0]
        assert "shard-003" in loaded_path

    @patch("mlx_lm_server.spec_decode.mtp_loader.mx.load")
    def test_fallback_no_index(self, mock_mx_load, tmp_path):
        """Without index file, all safetensors are loaded (fallback)."""
        config = {"model_type": "test", "num_hidden_layers": 4}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").touch()
        mock_mx_load.return_value = {
            "model.layers.0.weight": mx.zeros((4,)),
        }
        load_mtp_weights(str(tmp_path))
        # Should still attempt loading
        mock_mx_load.assert_called_once()


# ===================================================================
# Wave 3E: TestBuildMtpModuleNoDoubleRead (H2 config double-read tests)
# ===================================================================


class TestBuildMtpModuleNoDoubleRead:
    """Tests for H2: config.json read only once."""

    @patch(
        "mlx_lm_server.spec_decode.mtp_loader._load_mtp_weights_with_meta",
        return_value=None,
    )
    def test_returns_none_when_no_weights(self, mock_load):
        """build_mtp_module returns None when internal loader returns None."""
        model = MagicMock()
        result = build_mtp_module(model, "/fake/path")
        assert result is None
        mock_load.assert_called_once_with("/fake/path")

    @patch("builtins.open", create=True)
    @patch(
        "mlx_lm_server.spec_decode.mtp_loader._load_mtp_weights_with_meta",
    )
    def test_config_not_reread_in_build(self, mock_meta, mock_open):
        """build_mtp_module should NOT re-read config.json -- it uses _load_mtp_weights_with_meta."""
        # If _load_mtp_weights_with_meta returns None, build_mtp_module
        # should NOT open any files
        mock_meta.return_value = None
        model = MagicMock()
        build_mtp_module(model, "/fake/path")
        mock_open.assert_not_called()
