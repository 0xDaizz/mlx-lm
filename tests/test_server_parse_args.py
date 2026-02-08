"""Tests for CLI argument parsing (parse_args)."""

from __future__ import annotations

import pytest
from mlx_lm_server.server import parse_args


class TestParseArgsSSD:
    """Verify new SSD CLI arguments are correctly wired to ServerConfig."""

    def test_default_ssd_policy(self):
        config = parse_args([])
        assert config.ssd_policy == "evict_only"

    def test_ssd_policy_write_through(self):
        config = parse_args(["--ssd-policy", "write_through"])
        assert config.ssd_policy == "write_through"

    def test_ssd_policy_invalid_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--ssd-policy", "invalid"])

    def test_default_ssd_durability(self):
        config = parse_args([])
        assert config.ssd_durability == "best_effort"

    def test_ssd_durability_persistent(self):
        config = parse_args(["--ssd-durability", "persistent"])
        assert config.ssd_durability == "persistent"

    def test_default_ssd_async_writes(self):
        config = parse_args([])
        assert config.ssd_async_writes is True

    def test_no_ssd_async_writes(self):
        config = parse_args(["--no-ssd-async-writes"])
        assert config.ssd_async_writes is False

    def test_ssd_async_writes_explicit(self):
        config = parse_args(["--ssd-async-writes"])
        assert config.ssd_async_writes is True

    def test_default_writer_queue_size(self):
        config = parse_args([])
        assert config.ssd_writer_queue_size == 512

    def test_writer_queue_size_custom(self):
        config = parse_args(["--ssd-writer-queue-size", "1024"])
        assert config.ssd_writer_queue_size == 1024

    def test_writer_queue_size_invalid(self):
        with pytest.raises(SystemExit):
            parse_args(["--ssd-writer-queue-size", "0"])

    def test_default_persistent_max_retries(self):
        config = parse_args([])
        assert config.ssd_persistent_max_retries == 3

    def test_persistent_max_retries_custom(self):
        config = parse_args(["--ssd-persistent-max-retries", "5"])
        assert config.ssd_persistent_max_retries == 5

    def test_persistent_max_retries_zero_allowed(self):
        config = parse_args(["--ssd-persistent-max-retries", "0"])
        assert config.ssd_persistent_max_retries == 0

    def test_persistent_max_retries_negative_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--ssd-persistent-max-retries", "-1"])

    def test_default_flush_interval(self):
        config = parse_args([])
        assert config.ssd_flush_interval_s == 1.0

    def test_flush_interval_custom(self):
        config = parse_args(["--ssd-flush-interval-s", "2.5"])
        assert config.ssd_flush_interval_s == 2.5

    def test_flush_interval_zero_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--ssd-flush-interval-s", "0"])

    def test_flush_interval_negative_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--ssd-flush-interval-s", "-1.0"])

    def test_combined_ssd_args(self):
        """All SSD args together should work."""
        config = parse_args([
            "--ssd-policy", "write_through",
            "--ssd-durability", "persistent",
            "--no-ssd-async-writes",
            "--ssd-writer-queue-size", "256",
            "--ssd-persistent-max-retries", "5",
            "--ssd-flush-interval-s", "0.5",
        ])
        assert config.ssd_policy == "write_through"
        assert config.ssd_durability == "persistent"
        assert config.ssd_async_writes is False
        assert config.ssd_writer_queue_size == 256
        assert config.ssd_persistent_max_retries == 5
        assert config.ssd_flush_interval_s == 0.5
