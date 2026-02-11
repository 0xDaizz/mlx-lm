"""Speculative decoding module for mlx-lm-server.

Provides proposer-verifier framework for multi-token generation:
- N-gram proposer (CPU-only, zero overhead)
- Draft model proposer (small model generates candidates)
- Dynamic controller (adaptive speculation depth)
"""

from mlx_lm_server.spec_decode.config import SpecDecodeConfig

__all__ = ["SpecDecodeConfig"]
