"""Entry point for: python -m mlx_lm_server"""

from __future__ import annotations

import logging
import sys

import uvicorn

from mlx_lm_server.distributed import DistributedContext, finalize_distributed, init_distributed
from mlx_lm_server.server import create_app, parse_args

logger = logging.getLogger(__name__)


def main() -> None:
    config = parse_args()

    # --- Distributed initialization ---
    dist_ctx = DistributedContext()  # default disabled
    scheduler = None

    try:
        dist_ctx = init_distributed(config)

        # --- Load model and tokenizer ---
        if dist_ctx.enabled and dist_ctx.world_size > 1:
            from mlx_lm.utils import sharded_load

            logger.info(
                "Rank %d: loading model with sharded_load (TP)", dist_ctx.rank
            )
            model, tokenizer = sharded_load(
                config.model,
                pipeline_group=dist_ctx.pipeline_group,
                tensor_group=dist_ctx.tensor_group,
            )
        else:
            from mlx_lm import load

            model, tokenizer = load(
                config.model, adapter_path=config.adapter_path
            )

        # --- SSD cache dir (rank namespace for TP) ---
        ssd_cache_dir = config.ssd_cache_dir
        if dist_ctx.enabled and dist_ctx.world_size > 1:
            ssd_cache_dir = ssd_cache_dir / f"rank_{dist_ctx.rank}"

        # --- Create KV cache manager (optional) ---
        kv_cache_manager = None
        ssd_cache = None
        tiered_cache = None

        ssd_writer = None
        try:
            from mlx_lm_server.kv_cache_manager import (
                KVCacheManager,
                TieredKVCache,
                compute_model_fingerprint,
            )
            from mlx_lm_server.ssd_cache import SSDCache

            ssd_for_manager = None
            if config.ssd_enabled:
                fingerprint = compute_model_fingerprint(
                    config.model,
                    model,
                    config.kv_bits,
                    config.kv_group_size,
                    adapter_path=config.adapter_path,
                )
                ssd_dir = ssd_cache_dir / fingerprint
                ssd_cache = SSDCache(
                    ssd_dir,
                    config.ssd_ttl_days,
                    flush_interval_s=config.ssd_flush_interval_s,
                )
                ssd_for_manager = ssd_cache
            kv_cache_manager = KVCacheManager(config, ssd=ssd_for_manager)
            if config.ssd_enabled:
                # Create async writer if write-through is enabled
                if config.ssd_policy == "write_through" and config.ssd_async_writes:
                    from mlx_lm_server.ssd_writer import SSDWriterThread

                    ssd_writer = SSDWriterThread(
                        ssd=ssd_cache,
                        queue_size=config.ssd_writer_queue_size,
                        durability=config.ssd_durability,
                        max_retries=config.ssd_persistent_max_retries,
                    )
                tiered_cache = TieredKVCache(
                    kv_cache_manager,
                    ssd_cache,
                    writer=ssd_writer,
                    durability=config.ssd_durability,
                    max_retries=config.ssd_persistent_max_retries,
                )
        except ImportError:
            pass

        # --- Control bus (for future scheduler integration) ---
        control_bus = None
        if dist_ctx.enabled and dist_ctx.world_size > 1:
            from mlx_lm_server.distributed_bus import DistributedControlBus

            control_bus = DistributedControlBus(dist_ctx)
            logger.info(
                "Rank %d: control bus created (world_size=%d)",
                dist_ctx.rank,
                dist_ctx.world_size,
            )

        # --- Create scheduler ---
        from mlx_lm_server.scheduler import Scheduler

        scheduler = Scheduler(
            config=config,
            model=model,
            tokenizer=tokenizer,
            kv_cache_manager=kv_cache_manager,
            tiered_cache=tiered_cache,
            ssd_writer=ssd_writer,
            dist_ctx=dist_ctx,
            control_bus=control_bus,
        )
        scheduler.run_inference_loop()

        # --- Rank-based execution ---
        if not dist_ctx.enabled or dist_ctx.is_rank0:
            # Rank 0 (or single-machine): run HTTP server
            app = create_app(
                config=config, scheduler=scheduler, tokenizer=tokenizer
            )
            uvicorn.run(app, host=config.host, port=config.port)
        else:
            # Rank > 0: no HTTP server, wait for inference loop to finish
            logger.info(
                "Rank %d: waiting for inference loop (no HTTP server)",
                dist_ctx.rank,
            )
            scheduler.join_worker_loop(timeout=300.0)
            if scheduler.worker_timed_out:
                logger.critical(
                    "Rank %d: worker loop timed out — force exiting",
                    dist_ctx.rank,
                )
                import os
                os._exit(1)

    except RuntimeError as e:
        logger.critical("Fatal error: %s", e)
        sys.exit(1)
    finally:
        if scheduler is not None:
            try:
                scheduler.stop()
            except Exception:
                logger.warning("Error during scheduler shutdown", exc_info=True)
        finalize_distributed(dist_ctx)


if __name__ == "__main__":
    main()
