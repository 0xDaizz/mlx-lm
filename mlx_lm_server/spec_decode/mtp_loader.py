"""MTP weight loader -- loads MTP weights from safetensors and builds MTPModule.

Strategy: Load MTP weights separately from the main model loading pipeline.
The main model's sanitize() strips MTP weights, so we re-read them from
safetensors files and remap to canonical MTPModule names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _load_mtp_weights_with_meta(model_path: str):
    """Load MTP weights and return (weights, config, resolved_path, num_mtp) or None."""
    from mlx_lm_server.spec_decode.mtp_weight_mapper import (
        count_mtp_layers_from_weights,
        detect_pattern,
        extract_and_remap,
        get_num_mtp_layers,
    )

    path = Path(model_path)
    if not path.exists():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model_path))
        except Exception as e:
            logger.warning("Cannot resolve model path %s: %s", model_path, e)
            return None

    config_path = path / "config.json"
    if not config_path.exists():
        logger.warning("No config.json found at %s", path)
        return None

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    num_hidden_layers = config.get("num_hidden_layers", 0)

    # Determine which files to load (H1: selective shard loading)
    index_path = path / "model.safetensors.index.json"
    all_keys_for_detection = None
    if index_path.exists():
        mtp_shard_names = _identify_mtp_shards(index_path, num_hidden_layers)
        if mtp_shard_names:
            weight_files = [path / s for s in mtp_shard_names if (path / s).exists()]
            # Use index's full key list for pattern detection
            with open(index_path) as f:
                all_keys_for_detection = list(
                    json.load(f).get("weight_map", {}).keys()
                )
        else:
            weight_files = _get_weight_files(path)
    else:
        weight_files = _get_weight_files(path)

    if not weight_files:
        logger.warning("No safetensors files found at %s", path)
        return None

    all_weights: dict[str, mx.array] = {}
    for wf in weight_files:
        loaded = mx.load(str(wf))
        if isinstance(loaded, dict):
            all_weights.update(loaded)  # type: ignore[arg-type]
    all_keys = list(all_weights.keys())

    if all_keys_for_detection is None:
        all_keys_for_detection = all_keys

    pattern = detect_pattern(all_keys_for_detection, model_type, num_hidden_layers)
    if pattern is None:
        logger.debug("No MTP weight pattern detected for %s", model_path)
        return None

    logger.info("Detected MTP pattern '%s' for model type '%s'", pattern, model_type)

    mtp_weights = extract_and_remap(all_weights, pattern, num_hidden_layers)
    mtp_weights = _dequant_fp8_weights(mtp_weights)

    if not mtp_weights:
        logger.warning("MTP pattern '%s' detected but no weights extracted", pattern)
        return None

    num_mtp = get_num_mtp_layers(config)
    if num_mtp == 0:
        num_mtp = count_mtp_layers_from_weights(
            all_keys_for_detection, pattern, num_hidden_layers
        )

    logger.info(
        "Loaded %d MTP weights across %d MTP layer(s)", len(mtp_weights), num_mtp
    )

    return mtp_weights, config, path, num_mtp


def load_mtp_weights(model_path: str) -> Optional[dict[str, mx.array]]:
    """Load MTP weights from safetensors files.

    Reads model config to detect MTP pattern, then loads only the
    MTP-related weights from safetensors shards.

    Args:
        model_path: HuggingFace model path or local directory

    Returns:
        Dict of canonical MTP weights, or None if no MTP weights found.
    """
    result = _load_mtp_weights_with_meta(model_path)
    return result[0] if result else None


def build_mtp_module(model: Any, model_path: str) -> Optional[Any]:
    """Build MTPModule, load weights, and set shared references.

    Args:
        model: The loaded target model
        model_path: Path to model directory (for loading MTP weights)

    Returns:
        Initialized MTPModule, or None if model has no MTP weights.
    """
    from mlx_lm_server.spec_decode.mtp_module import MTPModule
    from mlx_lm_server.spec_decode.mtp_utils import (
        get_embed_tokens,
        get_final_norm,
        get_lm_head,
    )

    result = _load_mtp_weights_with_meta(model_path)
    if result is None:
        return None

    mtp_weights, config, resolved_path, num_mtp = result

    if num_mtp == 0:
        import re as _re

        depths = set()
        for k in mtp_weights:
            m_match = _re.match(r"layers\.(\d+)\.", k)
            if m_match:
                depths.add(int(m_match.group(1)))
        num_mtp = len(depths)

    if num_mtp == 0:
        logger.warning("Could not determine number of MTP layers")
        return None

    decoder_layer_class = type(model.model.layers[0])
    model_args = _extract_model_args(model, config)
    if model_args is None:
        logger.warning("Could not extract model args for MTP module construction")
        return None

    # H3: Detect MoE in MTP weights for correct layer_idx
    has_moe_weights = any(
        "block.mlp.experts" in k or "block.mlp.gate" in k for k in mtp_weights
    )
    if has_moe_weights:
        fkdr = getattr(model_args, "first_k_dense_replace", None)
        mtp_layer_idx = (
            fkdr if fkdr is not None else config.get("num_hidden_layers", 0)
        )
    else:
        mtp_layer_idx = 0

    mtp_module = MTPModule(
        args=model_args,
        decoder_layer_factory=decoder_layer_class,
        num_layers=num_mtp,
        layer_idx=mtp_layer_idx,
    )

    mtp_module.set_shared_refs(
        norm=get_final_norm(model),
        lm_head=get_lm_head(model),
        embed_tokens=get_embed_tokens(model),
    )

    mtp_module.load_weights(list(mtp_weights.items()))
    mx.eval(mtp_module.parameters())

    logger.info(
        "Built MTPModule with %d layer(s), decoder_class=%s",
        num_mtp,
        decoder_layer_class.__name__,
    )

    return mtp_module


def _dequant_fp8_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Dequantize FP8 weights (DeepSeek V3/V32 pattern).

    FP8 weights have companion `*_scale_inv` keys. For each pair,
    apply block-wise dequantization.
    """
    new_weights = dict(weights)
    scale_keys = [k for k in weights if k.endswith("_scale_inv")]
    for sk in scale_keys:
        wk = sk.replace("_scale_inv", "")
        if wk in weights:
            new_weights[wk] = _dequant(weights[wk], weights[sk])
            del new_weights[sk]
    return new_weights


def _dequant(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Block-wise FP8 dequantization (mirrors deepseek_v3.py:381-395)."""
    weight = mx.from_fp8(weight, dtype=mx.bfloat16)
    bs = 128
    m, n = weight.shape
    pad_bottom = (-m) % bs
    pad_side = (-n) % bs
    if pad_bottom or pad_side:
        weight = mx.pad(weight, [(0, pad_bottom), (0, pad_side)])
    weight = weight.reshape(
        ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
    )
    weight = (weight * scale_inv[:, None, :, None]).reshape(
        m + pad_bottom, n + pad_side
    )
    return weight[:m, :n].astype(mx.bfloat16)


def _get_weight_files(path: Path) -> list[Path]:
    """Get safetensors weight file paths from model directory.

    Checks for index file first, falls back to glob.
    """
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        # Get unique shard filenames
        shards = sorted(set(index.get("weight_map", {}).values()))
        return [path / s for s in shards if (path / s).exists()]

    # Single file
    single = path / "model.safetensors"
    if single.exists():
        return [single]

    # Glob fallback
    return sorted(path.glob("*.safetensors"))


def _identify_mtp_shards(index_path: Path, num_hidden_layers: int) -> list[str]:
    """From safetensors index, find shards containing MTP weights."""
    import re as _re

    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map", {})

    mtp_shards: set[str] = set()
    for key, shard in weight_map.items():
        # layer_index pattern
        m = _re.match(r"model\.layers\.(\d+)\.", key)
        if m and int(m.group(1)) >= num_hidden_layers:
            mtp_shards.add(shard)
        # model_mtp / flat_mtp / ernie patterns
        if _re.match(r"(model\.mtp|mtp[_.])", key):
            mtp_shards.add(shard)

    return sorted(mtp_shards)


def _extract_model_args(model: Any, config: dict) -> Optional[Any]:
    """Extract model args object needed for MTP layer construction.

    Tries to find the args/config object from the model, falling back
    to constructing one from the config dict.
    """
    # Try to get args from the model directly
    if hasattr(model, "args"):
        return model.args
    if hasattr(model, "config"):
        return model.config
    if hasattr(model, "model") and hasattr(model.model, "args"):
        return model.model.args

    # Fallback: construct from full config dict
    from types import SimpleNamespace

    args = SimpleNamespace(**config)

    # Ensure required keys exist with fallbacks
    if not hasattr(args, 'hidden_size'):
        args.hidden_size = getattr(args, 'd_model', 0)
    if not hasattr(args, 'num_attention_heads'):
        args.num_attention_heads = getattr(args, 'n_head', 0)
    if not hasattr(args, 'rms_norm_eps'):
        args.rms_norm_eps = getattr(args, 'layer_norm_epsilon', 1e-5)

    # Compute head_dim if not specified
    head_dim = getattr(args, 'head_dim', 0)
    if head_dim == 0 and args.hidden_size > 0 and args.num_attention_heads > 0:
        args.head_dim = args.hidden_size // args.num_attention_heads

    if getattr(args, 'hidden_size', 0) == 0:
        return None

    return args
