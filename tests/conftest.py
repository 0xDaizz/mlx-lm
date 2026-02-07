"""Shared fixtures for mlx-lm-server tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mlx_lm_server.config import ServerConfig


@pytest.fixture
def test_config(tmp_path: Path) -> ServerConfig:
    """Server config suitable for testing (small pool, temp SSD dir)."""
    return ServerConfig(
        model="mlx-community/Qwen3-4B-4bit",
        block_size=4,  # Small blocks for faster tests
        num_blocks=64,
        ssd_cache_dir=tmp_path / "ssd-cache",
        ssd_ttl_days=1,
        max_batch_size=2,
        max_queue_size=8,
    )


@pytest.fixture
def tmp_ssd_dir(tmp_path: Path) -> Path:
    """Temporary directory for SSD cache tests."""
    d = tmp_path / "ssd-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
