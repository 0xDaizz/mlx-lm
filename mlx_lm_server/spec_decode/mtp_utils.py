"""Utility functions for MTP speculative decoding.

Provides hidden state extraction and embedding layer access
without requiring modifications to individual model files.
The forward_with_hidden function manually iterates backbone layers
to capture PRE-NORM hidden states for MTP consumption.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx


def forward_with_hidden(
    model: Any,
    inputs: mx.array,
    cache: Optional[Any] = None,
) -> tuple[mx.array, mx.array]:
    """Run model forward, returning logits and PRE-NORM hidden states.

    Manually iterates backbone layers to capture hidden states before
    the final norm. MTP layers' hnorm weights were trained on pre-norm
    input — passing post-norm hidden causes double normalization.

    Args:
        model: The target model (e.g., loaded via mlx_lm.utils.load)
        inputs: [B, S] input token IDs
        cache: Optional KV cache

    Returns:
        Tuple of:
        - logits: [B, S, vocab_size]
        - hidden: [B, S, hidden_size] PRE-NORM last hidden state
    """
    from mlx_lm.models.base import create_attention_mask

    backbone = model.model
    h = backbone.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(backbone.layers)

    mask = create_attention_mask(h, cache[0])

    for layer, c in zip(backbone.layers, cache):
        h = layer(h, mask, cache=c)

    # h is PRE-NORM — what MTP layers expect
    normed = backbone.norm(h)

    if hasattr(model, "lm_head"):
        logits = model.lm_head(normed)
    else:
        logits = backbone.embed_tokens.as_linear(normed)

    return logits, h  # Pre-norm hidden for MTP


def get_embed_tokens(model: Any) -> Any:
    """Get the model's embedding layer reference.

    Args:
        model: The target model

    Returns:
        The embedding layer (nn.Embedding) that can be called with token IDs
    """
    return model.model.embed_tokens


def get_lm_head(model: Any) -> Any:
    """Get the model's lm_head layer or tied embedding equivalent.

    Args:
        model: The target model

    Returns:
        Callable that maps hidden states to logits
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    return model.model.embed_tokens.as_linear


def get_final_norm(model: Any) -> Any:
    """Get the model's final layer norm (before lm_head).

    Args:
        model: The target model

    Returns:
        The final normalization layer (e.g., RMSNorm)
    """
    return model.model.norm
