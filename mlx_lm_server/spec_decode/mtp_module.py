"""MTP (Multi-Token Prediction) module for speculative decoding.

Implements generic MTPLayer and MTPModule that work with any model
architecture that has MTP weights (DeepSeek V3, GLM-5, MiMo, etc.).
MTP layers use the main model's embedding table and lm_head (shared).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class MTPLayer(nn.Module):
    """Single MTP depth: enorm + hnorm + eh_proj + transformer block.

    Each MTP layer takes:
    - hidden: previous depth's hidden state (or main model's last hidden)
    - token_embed: embedding of the predicted token from previous depth

    And produces a new hidden state via:
    1. RMSNorm both inputs independently
    2. Concatenate and project back to hidden_size
    3. Pass through a full transformer decoder layer
    """

    def __init__(self, args: Any, decoder_layer_factory, layer_idx: int = 0):
        """
        Args:
            args: Model config args (must have hidden_size, and whatever
                  the decoder layer needs).
            decoder_layer_factory: Callable that creates a decoder layer.
                  Typically ``type(model.model.layers[0])``.
            layer_idx: Layer index passed to decoder_layer_factory.
                  Controls Dense vs MoE layer creation in mixed-expert
                  architectures (e.g., GLM-5 uses MoE for MTP layers).
        """
        super().__init__()
        hidden_size = args.hidden_size
        self.enorm = nn.RMSNorm(hidden_size)
        self.hnorm = nn.RMSNorm(hidden_size)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        # Use inspect to determine factory signature, avoiding broad TypeError catch (H4)
        import inspect

        try:
            sig = inspect.signature(decoder_layer_factory)
            params = list(sig.parameters.keys())
            if len(params) >= 2:
                self.block = decoder_layer_factory(args, layer_idx)
            else:
                self.block = decoder_layer_factory(args)
        except (ValueError, TypeError):
            # Fallback for C extensions or unusual factories
            try:
                self.block = decoder_layer_factory(args, layer_idx)
            except TypeError:
                self.block = decoder_layer_factory(args)

    def __call__(
        self,
        hidden: mx.array,
        token_embed: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """Forward pass for one MTP depth.

        Args:
            hidden: [B, S, D] hidden state from previous depth
            token_embed: [B, S, D] embedding of predicted token
            mask: Optional attention mask
            cache: Optional KV cache for the transformer block

        Returns:
            [B, S, D] new hidden state
        """
        h = self.eh_proj(
            mx.concatenate(
                [self.hnorm(hidden), self.enorm(token_embed)], axis=-1
            )
        )
        # Transformer decoder layers accept (h, mask, cache)
        h = self.block(h, mask=mask, cache=cache)
        return h


class MTPModule(nn.Module):
    """Manages N MTP layers + shared head references.

    The shared_norm and shared_lm_head are NOT owned by this module;
    they are references to the main model's norm and lm_head layers.
    This means no additional memory is used for those components.
    """

    def __init__(
        self,
        args: Any,
        decoder_layer_factory,
        num_layers: int,
        layer_idx: int = 0,
    ):
        """
        Args:
            args: Model config args
            decoder_layer_factory: Factory for creating decoder layers
            num_layers: Number of MTP depths (usually 1, sometimes 2)
            layer_idx: Layer index for decoder layer factory (controls
                       Dense vs MoE in mixed-expert architectures)
        """
        super().__init__()
        self.layers = [
            MTPLayer(args, decoder_layer_factory, layer_idx=layer_idx)
            for _ in range(num_layers)
        ]
        # These will be set after construction to reference main model components
        self.shared_norm: Optional[nn.Module] = None  # -> model.model.norm
        self.shared_lm_head: Optional[nn.Module] = None  # -> model.lm_head
        self._embed_tokens: Optional[nn.Module] = None  # -> model.model.embed_tokens

    def set_shared_refs(
        self,
        norm: nn.Module,
        lm_head: Any,
        embed_tokens: nn.Module,
    ) -> None:
        """Set shared references to main model components.

        Args:
            norm: The main model's final layer norm (model.model.norm)
            lm_head: The main model's lm_head layer, or a callable
                     (like embed_tokens.as_linear for tied weights)
            embed_tokens: The main model's embedding layer
        """
        self.shared_norm = norm
        self.shared_lm_head = lm_head
        self._embed_tokens = embed_tokens

    def predict(
        self,
        depth: int,
        hidden: mx.array,
        token_embed: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass for one MTP depth -> (output_hidden, logits).

        Args:
            depth: MTP layer index (0 to num_layers-1)
            hidden: [B, S, D] hidden state from previous depth
            token_embed: [B, S, D] embedding of predicted token
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            Tuple of:
            - output_hidden: [B, S, D] hidden state after MTP layer
            - logits: [B, S, vocab_size] logits from shared head
        """
        if self.shared_norm is None or self.shared_lm_head is None:
            raise RuntimeError(
                "MTPModule.set_shared_refs() must be called before predict(). "
                "shared_norm or shared_lm_head is None."
            )
        if depth < 0 or depth >= self.num_layers:
            raise IndexError(
                f"MTP depth {depth} out of range [0, {self.num_layers})"
            )
        h = self.layers[depth](hidden, token_embed, mask=mask, cache=cache)
        # Apply shared norm + lm_head
        normed = self.shared_norm(h)
        logits = self.shared_lm_head(normed)
        return h, logits

    def get_embed(self, token_ids: mx.array) -> mx.array:
        """Get token embeddings using the shared embedding table.

        Args:
            token_ids: [B, S] token IDs

        Returns:
            [B, S, D] token embeddings
        """
        if self._embed_tokens is None:
            raise RuntimeError(
                "MTPModule.set_shared_refs() must be called before get_embed(). "
                "_embed_tokens is None."
            )
        return self._embed_tokens(token_ids)

    @property
    def num_layers(self) -> int:
        """Number of MTP depth layers."""
        return len(self.layers)
