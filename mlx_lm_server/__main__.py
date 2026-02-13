"""Entry point for: python -m mlx_lm_server"""

from __future__ import annotations

import argparse
import atexit
import gc
import logging
import os
import shutil
import signal
import sys
import threading

import uvicorn

from mlx_lm_server.distributed import (
    DistributedContext,
    finalize_distributed,
    init_distributed,
)
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
    pre.add_argument(
        "--distributed-ibv-devices", default=os.environ.get("MLX_IBV_DEVICES")
    )
    pre.add_argument(
        "--distributed-jaccl-coordinator",
        default=os.environ.get("MLX_JACCL_COORDINATOR"),
    )
    pre.add_argument("--num-local-ranks", type=int, default=None)
    args, _ = pre.parse_known_args()

    mode = args.distributed_mode
    if mode == "off":
        return

    if (
        mode == "ring"
        and args.distributed_hostfile is None
        and args.num_local_ranks is None
    ):
        print(
            "ERROR: --distributed-mode=ring requires --distributed-hostfile or --num-local-ranks.",
            file=sys.stderr,
        )
        sys.exit(2)
    if mode == "jaccl":
        if (
            args.distributed_ibv_devices is None
            or args.distributed_jaccl_coordinator is None
        ):
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


def _setup_metal_memory() -> None:
    """Configure Metal memory limits (exo-style)."""
    try:
        import mlx.core as mx
    except ImportError:
        return

    try:
        if not mx.metal.is_available():
            return
        device_info = mx.device_info()
        max_rec_size = int(device_info.get("max_recommended_working_set_size", 0))
        mem_size = int(device_info.get("memory_size", 0))
        if max_rec_size > 0:
            mx.set_wired_limit(max_rec_size)
            logger.info(
                "Metal wired limit set to %d MB (total: %d MB)",
                max_rec_size // (1024 * 1024),
                mem_size // (1024 * 1024),
            )
        else:
            logger.info("max_recommended_working_set_size not available, skipping wired limit")
    except (AttributeError, Exception) as e:
        logger.debug(f"Could not configure Metal memory: {e}")


def _cleanup_metal() -> None:
    """Clean up Metal GPU resources.

    Critical sequence:
    1. gc.collect() — drop Python references to mx.array objects
    2. mx.set_wired_limit(0) — unwire buffers from Metal residency set
       (without this, OS CANNOT reclaim wired pages even after process exit;
       see llama.cpp PR #11427 and MLX PR #1510)
    3. mx.set_cache_limit(0) — prevent re-caching during cleanup
    4. mx.clear_cache() — release cached Metal buffers
    5. gc.collect() + mx.clear_cache() — second pass for circular refs
    """
    try:
        import mlx.core as mx

        gc.collect()
        mx.set_wired_limit(0)
        mx.set_cache_limit(0)
        mx.clear_cache()
        gc.collect()
        mx.clear_cache()
    except (AttributeError, Exception):
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
    gc.collect()
    logger.info("Metal cleanup completed")


def _emergency_exit() -> None:
    """Best-effort Metal unwire before forced exit.

    Called by cleanup_timer when scheduler.stop() hangs.
    Similar to eval_with_timeout() in mlx_lm/utils.py.
    """
    try:
        import mlx.core as mx
        mx.set_wired_limit(0)
        mx.set_cache_limit(0)
        mx.clear_cache()
    except Exception:
        pass
    os._exit(1)


def main() -> None:
    _maybe_relaunch_under_mlx_launch()

    _setup_metal_memory()
    atexit.register(_cleanup_metal)
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda s, f: sys.exit(0))

    config = parse_args()

    # --- Distributed initialization ---
    dist_ctx = DistributedContext()  # default disabled
    scheduler = None
    model = None
    tokenizer = None
    app = None
    kv_cache_manager = None
    ssd_cache = None
    tiered_cache = None
    ssd_writer = None
    control_bus = None

    try:
        dist_ctx = init_distributed(config)

        # --- Load model and tokenizer ---
        if dist_ctx.enabled and dist_ctx.world_size > 1:
            import mlx.core as mx
            from mlx_lm.utils import sharded_load

            logger.info("Rank %d: loading model with sharded_load (TP)", dist_ctx.rank)
            try:
                mx.reset_peak_memory()
                pre_mem = mx.get_active_memory()
                logger.info(
                    "Rank %d: pre-load active memory: %.1f GB",
                    dist_ctx.rank,
                    pre_mem / (1024**3),
                )
            except AttributeError:
                pass

            model, tokenizer = sharded_load(
                config.model,
                pipeline_group=dist_ctx.pipeline_group,
                tensor_group=dist_ctx.tensor_group,
            )

            try:
                post_mem = mx.get_active_memory()
                peak_mem = mx.get_peak_memory()
                logger.info(
                    "Rank %d: post-load memory: active=%.1f GB, peak=%.1f GB",
                    dist_ctx.rank,
                    post_mem / (1024**3),
                    peak_mem / (1024**3),
                )
            except AttributeError:
                pass
        else:
            from mlx_lm import load

            model, tokenizer = load(config.model, adapter_path=config.adapter_path)

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
                max_size_bytes = (
                    int(config.ssd_max_size_gb * (1024**3))
                    if config.ssd_max_size_gb > 0
                    else 0
                )
                ssd_cache = SSDCache(
                    ssd_dir,
                    config.ssd_ttl_days,
                    flush_interval_s=config.ssd_flush_interval_s,
                    max_size_bytes=max_size_bytes,
                )
                ssd_for_manager = ssd_cache
            kv_cache_manager = KVCacheManager(config, ssd=ssd_for_manager)
            if config.ssd_enabled:
                # Create async writer if write-through is enabled
                if config.ssd_policy == "write_through" and config.ssd_async_writes:
                    from mlx_lm_server.ssd_writer import SSDWriterThread

                    assert (
                        ssd_cache is not None
                    )  # guaranteed by ssd_enabled check above
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
                config=config,
                scheduler=scheduler,
                tokenizer=tokenizer,
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
            scheduler.join_worker_loop(timeout=None)

    except RuntimeError as e:
        logger.critical("Fatal error: %s", e)
        sys.exit(1)
    finally:
        # Guard cleanup with a 30-second timeout. If scheduler.stop() or
        # finalize_distributed() hangs (stuck collective, stuck thread join),
        # the daemon timer forces process exit.
        cleanup_timer = threading.Timer(30.0, _emergency_exit)
        cleanup_timer.daemon = True
        cleanup_timer.start()
        try:
            if scheduler is not None:
                try:
                    scheduler.stop()
                except Exception:
                    logger.warning("Error during scheduler shutdown", exc_info=True)
            # Release all heavy references to allow GC to free Metal memory.
            # These locals hold mx.array objects (model weights, KV caches, etc.)
            scheduler = None
            model = None
            tokenizer = None
            app = None
            kv_cache_manager = None
            tiered_cache = None
            ssd_cache = None
            ssd_writer = None
            control_bus = None
            gc.collect()  # Collect model arrays before clearing Metal cache
            _cleanup_metal()
            if dist_ctx is not None:
                finalize_distributed(dist_ctx)
        finally:
            cleanup_timer.cancel()


if __name__ == "__main__":
    main()
