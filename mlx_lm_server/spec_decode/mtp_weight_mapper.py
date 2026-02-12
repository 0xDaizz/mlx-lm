"""MTP weight name mapper for various model checkpoint formats.

Maps checkpoint weight keys to MTPModule's canonical naming scheme.
Supports 4 naming patterns across 11+ model families.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def get_num_mtp_layers(config: dict) -> int:
    """Extract number of MTP layers from model config.

    Checks common config keys used by different model families.
    Falls back to weight-based detection if config doesn't specify.

    Args:
        config: Model config dict (from config.json)

    Returns:
        Number of MTP layers, or 0 if not found.
    """
    # Direct config keys (ordered by frequency)
    for key in [
        "num_nextn_predict_layers",  # DeepSeek V3, GLM4
        "num_mtp_layers",            # MiMo
        "n_predict",                 # Some models
        "num_predict_layers",        # Variants
    ]:
        val = config.get(key)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    return 0


def detect_pattern(
    weight_keys: list[str],
    model_type: str,
    num_hidden_layers: int,
) -> Optional[str]:
    """Detect which MTP weight naming pattern the checkpoint uses.

    Args:
        weight_keys: List of all weight key names from checkpoint
        model_type: Model type string from config.json (e.g., "deepseek_v3")
        num_hidden_layers: Number of main model hidden layers

    Returns:
        Pattern name ("layer_index", "model_mtp", "ernie", "flat_mtp")
        or None if no MTP weights detected.
    """
    # Check for layer_index pattern: model.layers.{N} where N >= num_hidden_layers
    # These are MTP layers stored as extra layers beyond the main model layers
    for key in weight_keys:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        if m and int(m.group(1)) >= num_hidden_layers:
            return "layer_index"

    # Check for model_mtp pattern: model.mtp.{k} or model.mtp_layers.{k}
    for key in weight_keys:
        if re.match(r"model\.mtp[\._]", key):
            return "model_mtp"

    # Check for ernie pattern: mtp_hidden_norm.{k} or mtp_block.{k}
    for key in weight_keys:
        if re.match(r"mtp_(hidden_norm|block|embed_norm|emb_norm)\.", key):
            return "ernie"

    # Check for flat_mtp pattern: mtp.{k}
    for key in weight_keys:
        if re.match(r"mtp\.\d+\.", key):
            return "flat_mtp"

    return None


def extract_and_remap(
    weights: dict[str, mx.array],
    pattern: str,
    num_hidden_layers: int,
) -> dict[str, mx.array]:
    """Extract MTP weights and remap to canonical names.

    Takes the full weight dict, extracts only MTP-related weights,
    and remaps their keys to the canonical MTPModule format:
      layers.{depth}.enorm.weight
      layers.{depth}.hnorm.weight
      layers.{depth}.eh_proj.weight
      layers.{depth}.block.{path}

    Args:
        weights: Full checkpoint weight dict
        pattern: One of "layer_index", "model_mtp", "ernie", "flat_mtp"
        num_hidden_layers: Number of main model hidden layers

    Returns:
        Dict of remapped MTP weights with canonical names.
    """
    remapped: dict[str, mx.array] = {}

    if pattern == "layer_index":
        # model.layers.{N}.xxx -> layers.{N - num_hidden_layers}.xxx
        # where N >= num_hidden_layers
        for key, value in weights.items():
            m = re.match(r"model\.layers\.(\d+)\.(.*)", key)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx >= num_hidden_layers:
                    depth = layer_idx - num_hidden_layers
                    remainder = m.group(2)
                    canonical = _remap_sublayer(depth, remainder)
                    if canonical:
                        remapped[canonical] = value

    elif pattern == "model_mtp":
        # model.mtp.{k}.xxx or model.mtp_layers.{k}.xxx -> layers.{k}.xxx
        for key, value in weights.items():
            m = re.match(r"model\.mtp(?:_layers)?\.(\d+)\.(.*)", key)
            if m:
                depth = int(m.group(1))
                remainder = m.group(2)
                canonical = _remap_sublayer(depth, remainder)
                if canonical:
                    remapped[canonical] = value

    elif pattern == "ernie":
        # Ernie has a different structure:
        # mtp_hidden_norm.{k}.weight -> layers.{k}.hnorm.weight
        # mtp_embed_norm.{k}.weight -> layers.{k}.enorm.weight
        # mtp_eh_proj.{k}.weight -> layers.{k}.eh_proj.weight
        # mtp_block.{k}.xxx -> layers.{k}.block.xxx
        for key, value in weights.items():
            # mtp_hidden_norm.{k}.weight
            m = re.match(r"mtp_hidden_norm\.(\d+)\.(.*)", key)
            if m:
                depth = int(m.group(1))
                remapped[f"layers.{depth}.hnorm.{m.group(2)}"] = value
                continue
            # mtp_embed_norm.{k}.weight
            m = re.match(r"mtp_(?:embed|emb)_norm\.(\d+)\.(.*)", key)
            if m:
                depth = int(m.group(1))
                remapped[f"layers.{depth}.enorm.{m.group(2)}"] = value
                continue
            # mtp_eh_proj.{k}.weight (or mtp_proj)
            m = re.match(r"mtp_(?:eh_|linear_)?proj\.(\d+)\.(.*)", key)
            if m:
                depth = int(m.group(1))
                remapped[f"layers.{depth}.eh_proj.{m.group(2)}"] = value
                continue
            # mtp_block.{k}.xxx
            m = re.match(r"mtp_block\.(\d+)\.(.*)", key)
            if m:
                depth = int(m.group(1))
                remapped[f"layers.{depth}.block.{m.group(2)}"] = value
                continue

    elif pattern == "flat_mtp":
        # mtp.{k}.xxx -> layers.{k}.xxx
        for key, value in weights.items():
            m = re.match(r"mtp\.(\d+)\.(.*)", key)
            if m:
                depth = int(m.group(1))
                remainder = m.group(2)
                canonical = _remap_sublayer(depth, remainder)
                if canonical:
                    remapped[canonical] = value

    return remapped


def _remap_sublayer(depth: int, remainder: str) -> Optional[str]:
    """Map a sub-layer path to canonical format.

    Args:
        depth: MTP layer depth index
        remainder: The part after the layer prefix (e.g., "enorm.weight")

    Returns:
        Canonical key like "layers.{depth}.enorm.weight", or None if not MTP-related
    """
    # Direct MTP component names (these are already canonical)
    if remainder.startswith(("enorm.", "hnorm.", "eh_proj.")):
        return f"layers.{depth}.{remainder}"

    # Block/transformer components - need to be nested under "block"
    # Common patterns: self_attn.*, mlp.*, input_layernorm.*, post_attention_layernorm.*
    # These are the transformer decoder layer internals
    block_prefixes = (
        "self_attn.", "mlp.", "input_layernorm.", "post_attention_layernorm.",
        "attention.", "feed_forward.", "ffn_norm.", "attention_norm.",
        "pre_feedforward_layernorm.", "post_feedforward_layernorm.",
        "pre_attention_layernorm.",
    )
    if remainder.startswith(block_prefixes):
        return f"layers.{depth}.block.{remainder}"

    # Also handle "block.xxx" directly (some models already prefix with block)
    if remainder.startswith("block."):
        return f"layers.{depth}.{remainder}"

    # Unknown sub-layer -- still include it under block as a fallback
    # This handles model-specific layer naming
    logger.warning("Unknown MTP sub-layer for depth %d: %s (mapping to block.*)", depth, remainder)
    return f"layers.{depth}.block.{remainder}"


def count_mtp_layers_from_weights(
    weight_keys: list[str],
    pattern: str,
    num_hidden_layers: int,
) -> int:
    """Count number of MTP layers by analyzing weight keys.

    Used when config doesn't specify num_mtp_layers.

    Args:
        weight_keys: List of weight key names
        pattern: Detected pattern name
        num_hidden_layers: Number of main model hidden layers

    Returns:
        Number of MTP layers detected from weights.
    """
    depths: set[int] = set()

    if pattern == "layer_index":
        for key in weight_keys:
            m = re.match(r"model\.layers\.(\d+)\.", key)
            if m:
                idx = int(m.group(1))
                if idx >= num_hidden_layers:
                    depths.add(idx - num_hidden_layers)
    elif pattern == "model_mtp":
        for key in weight_keys:
            m = re.match(r"model\.mtp(?:_layers)?\.(\d+)\.", key)
            if m:
                depths.add(int(m.group(1)))
    elif pattern == "ernie":
        for key in weight_keys:
            m = re.match(r"mtp_(?:hidden_norm|block|embed_norm|emb_norm)\.(\d+)\.", key)
            if m:
                depths.add(int(m.group(1)))
    elif pattern == "flat_mtp":
        for key in weight_keys:
            m = re.match(r"mtp\.(\d+)\.", key)
            if m:
                depths.add(int(m.group(1)))

    return len(depths)
