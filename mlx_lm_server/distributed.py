"""Distributed context initialization for Tensor Parallel / RDMA serving."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    """Holds distributed runtime state after initialization."""

    enabled: bool = False
    group: Any = None  # mx.distributed.Group or None
    rank: int = 0
    world_size: int = 1
    pipeline_group: Any = None
    tensor_group: Any = None
    backend: str = "off"  # off | ring | jaccl

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def init_distributed(config) -> DistributedContext:
    """Initialize distributed context based on ServerConfig.

    For mode 'off', returns a no-op context (single-machine).
    For 'ring' or 'jaccl', sets environment variables and calls mx.distributed.init().

    Args:
        config: ServerConfig with distributed_* fields.

    Returns:
        DistributedContext with group, rank, world_size, and shard groups.
    """
    mode = config.distributed_mode
    if mode == "off":
        logger.info("Distributed mode: off (single-machine)")
        return DistributedContext()

    import mlx.core as mx

    # Set environment variables based on backend
    if mode == "ring":
        if config.distributed_hostfile:
            os.environ["MLX_HOSTFILE"] = config.distributed_hostfile
        logger.info("Initializing distributed: backend=ring, strict=%s", config.distributed_strict)
        group = mx.distributed.init(backend="ring", strict=config.distributed_strict)

    elif mode == "jaccl":
        if config.distributed_ibv_devices:
            os.environ["MLX_IBV_DEVICES"] = config.distributed_ibv_devices
        if config.distributed_jaccl_coordinator:
            os.environ["MLX_JACCL_COORDINATOR"] = config.distributed_jaccl_coordinator
        logger.info("Initializing distributed: backend=jaccl (RDMA), strict=%s", config.distributed_strict)
        group = mx.distributed.init(backend="jaccl", strict=config.distributed_strict)

    else:
        raise ValueError(f"Unknown distributed_mode: {mode!r}")

    rank = group.rank()
    world_size = group.size()

    if world_size == 1:
        raise RuntimeError(
            f"distributed_mode={mode!r} but world_size=1. "
            "This indicates a misconfigured launch. "
            "Aborting to prevent port conflict and silent single-node operation."
        )

    # Determine shard groups based on sharding strategy
    sharding = config.distributed_sharding
    if sharding == "pipeline":
        raise ValueError(
            "Pipeline sharding is not supported in v1. Use distributed_sharding='tensor'."
        )
    tensor_group = group if sharding == "tensor" and world_size > 1 else None

    logger.info(
        "Distributed initialized: rank=%d/%d, sharding=%s, backend=%s",
        rank, world_size, sharding, mode,
    )

    return DistributedContext(
        enabled=True,
        group=group,
        rank=rank,
        world_size=world_size,
        pipeline_group=None,
        tensor_group=tensor_group,
        backend=mode,
    )


def finalize_distributed(ctx: DistributedContext) -> None:
    """Clean up distributed context. Currently a no-op (MLX handles cleanup)."""
    if ctx.enabled:
        logger.info("Distributed context finalized (rank=%d)", ctx.rank)
