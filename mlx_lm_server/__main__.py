"""Entry point for: python -m mlx_lm_server"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import threading

import uvicorn

from mlx_lm_server.distributed import DistributedContext, finalize_distributed, init_distributed
from mlx_lm_server.server import create_app, parse_args

logger = logging.getLogger(__name__)


def _maybe_relaunch_under_mlx_launch() -> None:
    """Auto-relaunch the server under ``mlx.launch`` when distributed mode is requested.

    This performs a quick pre-parse of ``sys.argv`` to detect ``--distributed-mode``
    and, if it is ``ring`` or ``jaccl``, replaces the current process with an
    ``mlx.launch`` invocation that will re-enter ``main()`` with ``MLX_RANK`` set.
    """
    # Already running under mlx.launch — nothing to do.
    if os.environ.get("MLX_RANK") is not None:
        return

    # Minimal pre-parse — only the flags we need to decide whether to relaunch.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--distributed-mode", default="off")
    pre.add_argument("--distributed-hostfile", default=os.environ.get("MLX_HOSTFILE"))
    pre.add_argument("--distributed-ibv-devices", default=os.environ.get("MLX_IBV_DEVICES"))
    pre.add_argument("--distributed-jaccl-coordinator", default=os.environ.get("MLX_JACCL_COORDINATOR"))
    pre.add_argument("--num-local-ranks", type=int, default=None)
    args, _ = pre.parse_known_args()

    mode = args.distributed_mode
    if mode == "off":
        return

    if mode == "ring" and args.distributed_hostfile is None and args.num_local_ranks is None:
        print(
            "ERROR: --distributed-mode=ring requires --distributed-hostfile or --num-local-ranks.",
            file=sys.stderr,
        )
        sys.exit(2)
    if mode == "jaccl":
        if args.distributed_ibv_devices is None or args.distributed_jaccl_coordinator is None:
            print(
                "ERROR: --distributed-mode=jaccl requires both "
                "--distributed-ibv-devices and --distributed-jaccl-coordinator.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Ensure mlx.launch is on PATH.
    if shutil.which("mlx.launch") is None:
        print(
            "ERROR: --distributed-mode requires 'mlx.launch' but it was not found on PATH.\n"
            "Install mlx with distributed support and ensure mlx.launch is available.",
            file=sys.stderr,
        )
        sys.exit(1)

    # num_local_ranks takes priority over env-sourced hostfile
    if args.num_local_ranks is not None:
        args.distributed_hostfile = None  # Ignore env-sourced hostfile

    # Build the mlx.launch command.
    cmd: list[str] = ["mlx.launch", "--backend", mode]

    if mode == "ring":
        if args.distributed_hostfile:
            cmd += ["--hostfile", args.distributed_hostfile]
        elif args.num_local_ranks is not None:
            cmd += ["--hosts", "localhost", "-n", str(args.num_local_ranks)]
    elif mode == "jaccl":
        if args.distributed_hostfile:
            cmd += ["--hostfile", args.distributed_hostfile]
        if args.distributed_ibv_devices:
            os.environ["MLX_IBV_DEVICES"] = args.distributed_ibv_devices
        if args.distributed_jaccl_coordinator:
            os.environ["MLX_JACCL_COORDINATOR"] = args.distributed_jaccl_coordinator

    cmd += ["--", sys.executable, "-m", "mlx_lm_server"] + sys.argv[1:]

    logger.info("Re-launching under mlx.launch: %s", " ".join(cmd))
    os.execvp(cmd[0], cmd)


def main() -> None:
    _maybe_relaunch_under_mlx_launch()

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
        except ImportError as e:
            logger.warning("KV cache modules not available: %s", e)

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
                config=config, scheduler=scheduler, tokenizer=tokenizer,
                dist_ctx=dist_ctx,
            )
            uvicorn.run(
                app,
                host=config.host,
                port=config.port,
            )
        else:
            # Rank > 0: no HTTP server, wait for inference loop to finish
            logger.info(
                "Rank %d: waiting for inference loop (no HTTP server)",
                dist_ctx.rank,
            )
            scheduler.join_worker_loop(timeout=300.0)
            if scheduler.worker_timed_out:
                raise RuntimeError(
                    f"Rank {dist_ctx.rank}: worker loop timed out"
                )

    except RuntimeError as e:
        logger.critical("Fatal error: %s", e)
        sys.exit(1)
    finally:
        # Guard cleanup with a 10-second timeout. If scheduler.stop() or
        # finalize_distributed() hangs (stuck collective, stuck thread join),
        # the daemon timer forces process exit.
        cleanup_timer = threading.Timer(10.0, lambda: os._exit(1))
        cleanup_timer.daemon = True
        cleanup_timer.start()
        try:
            if scheduler is not None:
                try:
                    scheduler.stop()
                except Exception:
                    logger.warning("Error during scheduler shutdown", exc_info=True)
            if dist_ctx is not None:
                finalize_distributed(dist_ctx)
        finally:
            cleanup_timer.cancel()


if __name__ == "__main__":
    main()
