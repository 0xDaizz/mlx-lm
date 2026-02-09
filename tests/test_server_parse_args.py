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

    def test_boolean_optional_action_fallback(self, monkeypatch):
        """parse_args should work even without BooleanOptionalAction (Python 3.8)."""
        import argparse as _argparse
        # Temporarily remove BooleanOptionalAction
        original = getattr(_argparse, "BooleanOptionalAction", None)
        if original is not None:
            monkeypatch.delattr(_argparse, "BooleanOptionalAction")

        # Re-import to get fresh parse_args that checks hasattr
        # parse_args checks at call-time, so we just call it
        config = parse_args(["--no-ssd-async-writes"])
        assert config.ssd_async_writes is False

        config2 = parse_args(["--ssd-async-writes"])
        assert config2.ssd_async_writes is True

        config3 = parse_args([])
        assert config3.ssd_async_writes is True

        # Restore (monkeypatch handles this automatically)

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


class TestParseArgsGenerationTokens:
    """F1: --max-generation-tokens CLI argument."""

    def test_default_max_generation_tokens(self):
        config = parse_args([])
        assert config.max_generation_tokens == 32768

    def test_max_generation_tokens_cli_arg(self):
        config = parse_args(["--max-generation-tokens", "4096"])
        assert config.max_generation_tokens == 4096


class TestParseArgsRangeValidation:
    """U12: Core numeric params must be > 0."""

    def test_parse_args_rejects_zero_block_size(self):
        with pytest.raises(SystemExit):
            parse_args(["--block-size", "0"])

    def test_parse_args_rejects_negative_block_size(self):
        with pytest.raises(SystemExit):
            parse_args(["--block-size", "-1"])

    def test_parse_args_rejects_zero_num_blocks(self):
        with pytest.raises(SystemExit):
            parse_args(["--num-blocks", "0"])

    def test_parse_args_rejects_zero_max_batch_size(self):
        with pytest.raises(SystemExit):
            parse_args(["--max-batch-size", "0"])

    def test_parse_args_rejects_zero_max_queue_size(self):
        with pytest.raises(SystemExit):
            parse_args(["--max-queue-size", "0"])

    def test_parse_args_accepts_valid_values(self):
        config = parse_args([
            "--block-size", "32",
            "--num-blocks", "1024",
            "--max-batch-size", "4",
            "--max-queue-size", "64",
        ])
        assert config.block_size == 32
        assert config.num_blocks == 1024
        assert config.max_batch_size == 4
        assert config.max_queue_size == 64


class TestParseArgsPrefillBatchSize:
    """F7: prefill_batch_size > max_batch_size should be rejected."""

    def test_prefill_exceeds_max_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--prefill-batch-size", "16", "--max-batch-size", "8"])

    def test_prefill_equals_max_accepted(self):
        config = parse_args(["--prefill-batch-size", "8", "--max-batch-size", "8"])
        assert config.prefill_batch_size == 8
        assert config.max_batch_size == 8

    def test_prefill_less_than_max_accepted(self):
        config = parse_args(["--prefill-batch-size", "4", "--max-batch-size", "8"])
        assert config.prefill_batch_size == 4


class TestParseArgsDistributedHardening:
    """U10: Distributed CLI path existence checks."""

    def test_hostfile_not_exists_rejected(self, tmp_path):
        """Nonexistent --distributed-hostfile should be rejected."""
        with pytest.raises(SystemExit):
            parse_args([
                "--distributed-mode", "ring",
                "--distributed-hostfile", str(tmp_path / "nonexistent-hosts.json"),
            ])

    def test_ibv_devices_not_exists_rejected(self, tmp_path):
        """Nonexistent --distributed-ibv-devices should be rejected."""
        with pytest.raises(SystemExit):
            parse_args([
                "--distributed-mode", "jaccl",
                "--distributed-ibv-devices", str(tmp_path / "nonexistent-ibv.json"),
                "--distributed-jaccl-coordinator", "10.0.0.1:55000",
            ])

    def test_hostfile_exists_accepted(self, tmp_path):
        """Existing --distributed-hostfile should be accepted."""
        hostfile = tmp_path / "hosts.json"
        hostfile.write_text("{}")
        config = parse_args([
            "--distributed-mode", "ring",
            "--distributed-hostfile", str(hostfile),
        ])
        assert config.distributed_hostfile == str(hostfile)

    def test_ibv_devices_exists_accepted(self, tmp_path):
        """Existing --distributed-ibv-devices should be accepted."""
        ibv_file = tmp_path / "ibv.json"
        ibv_file.write_text("{}")
        config = parse_args([
            "--distributed-mode", "jaccl",
            "--distributed-ibv-devices", str(ibv_file),
            "--distributed-jaccl-coordinator", "10.0.0.1:55000",
        ])
        assert config.distributed_ibv_devices == str(ibv_file)


class TestParseArgsEnvFallback:
    """U11: Environment variable fallback for distributed config."""

    def test_env_fallback_for_distributed_hostfile(self, monkeypatch, tmp_path):
        """MLX_HOSTFILE env should be used as default for --distributed-hostfile."""
        hostfile = tmp_path / "env-hosts.json"
        hostfile.write_text("{}")
        monkeypatch.setenv("MLX_HOSTFILE", str(hostfile))
        config = parse_args(["--distributed-mode", "ring"])
        assert config.distributed_hostfile == str(hostfile)

    def test_env_fallback_for_ibv_and_coordinator(self, monkeypatch, tmp_path):
        """MLX_IBV_DEVICES and MLX_JACCL_COORDINATOR env should be used as defaults."""
        ibv_file = tmp_path / "env-ibv.json"
        ibv_file.write_text("{}")
        monkeypatch.setenv("MLX_IBV_DEVICES", str(ibv_file))
        monkeypatch.setenv("MLX_JACCL_COORDINATOR", "10.0.0.2:55000")
        config = parse_args(["--distributed-mode", "jaccl"])
        assert config.distributed_ibv_devices == str(ibv_file)
        assert config.distributed_jaccl_coordinator == "10.0.0.2:55000"

    def test_cli_overrides_env(self, monkeypatch, tmp_path):
        """CLI args should override environment variables."""
        env_file = tmp_path / "env-hosts.json"
        env_file.write_text("{}")
        cli_file = tmp_path / "cli-hosts.json"
        cli_file.write_text("{}")
        monkeypatch.setenv("MLX_HOSTFILE", str(env_file))
        config = parse_args([
            "--distributed-mode", "ring",
            "--distributed-hostfile", str(cli_file),
        ])
        assert config.distributed_hostfile == str(cli_file)
