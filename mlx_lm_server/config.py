"""Server configuration for mlx-lm-server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    """Configuration for the mlx-lm-server."""

    # Model
    model: str = "mlx-community/Qwen3-4B-4bit"
    adapter_path: str | None = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # KV Cache
    block_size: int = 16  # Tokens per KV block
    num_blocks: int = 2048  # Total blocks in pool
    kv_bits: int = 8  # KV cache quantization bits
    kv_group_size: int = 64  # Elements per quantization scale

    # SSD Tier
    ssd_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "mlx-lm-server" / "kv-cache")
    ssd_ttl_days: int = 7  # Days before SSD blocks are pruned
    ssd_enabled: bool = True
    ssd_policy: str = "evict_only"  # "evict_only" | "write_through"
    ssd_durability: str = "best_effort"  # "best_effort" | "persistent"
    ssd_async_writes: bool = True
    ssd_writer_queue_size: int = 512
    ssd_flush_interval_s: float = 1.0
    ssd_persistent_max_retries: int = 3

    # Scheduler
    max_batch_size: int = 8  # Max concurrent decode sequences
    prefill_batch_size: int = 4  # Max concurrent prefill sequences
    prefill_step_size: int = 2048  # Chunk size for prompt processing
    max_queue_size: int = 128  # Max pending requests

    # Model generation
    completion_batch_size: int = 32  # Max concurrent decode sequences in BatchGenerator
    max_kv_size: int | None = None  # Max KV cache size (None = unlimited)
    sequence_cache_size: int = 50  # Max cached sequence-level KV states

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 1.0
    default_top_p: float = 1.0

    # Request limits
    max_prompt_tokens: int = 32768  # Max prompt length in tokens
    max_generation_tokens: int = 32768  # Upper bound for max_tokens
    request_timeout_s: float = 120.0  # Timeout for inference requests
    first_token_timeout_s: float = 300.0  # Timeout for first token (prefill included)

    # Distributed
    use_distributed: bool = False  # DEPRECATED: use distributed_mode instead

    # Distributed / Tensor Parallel
    distributed_mode: str = "off"                          # off | ring | jaccl
    distributed_sharding: str = "tensor"                   # tensor | pipeline
    distributed_strict: bool = True
    distributed_hostfile: str | None = None
    distributed_ibv_devices: str | None = None
    distributed_jaccl_coordinator: str | None = None
