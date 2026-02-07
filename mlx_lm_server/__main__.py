"""Entry point for: python -m mlx_lm_server"""

from __future__ import annotations

import uvicorn

from mlx_lm_server.server import create_app, parse_args


def main() -> None:
    config = parse_args()

    # --- Load model and tokenizer ---
    from mlx_lm import load

    model, tokenizer = load(config.model, adapter_path=config.adapter_path)

    # --- Create KV cache manager (optional) ---
    kv_cache_manager = None
    ssd_cache = None
    tiered_cache = None

    try:
        from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
        from mlx_lm_server.ssd_cache import SSDCache

        kv_cache_manager = KVCacheManager(config)
        if config.ssd_enabled:
            ssd_cache = SSDCache(config.ssd_cache_dir, config.ssd_ttl_days)
            tiered_cache = TieredKVCache(kv_cache_manager, ssd_cache)
    except ImportError:
        pass

    # --- Create scheduler ---
    from mlx_lm_server.scheduler import Scheduler

    scheduler = Scheduler(
        config=config,
        model=model,
        tokenizer=tokenizer,
        kv_cache_manager=kv_cache_manager,
    )
    # Wire tiered cache into scheduler for SSD eviction/pruning
    if tiered_cache is not None:
        scheduler._tiered_cache = tiered_cache
    scheduler.run_inference_loop()

    # --- Build and run ---
    app = create_app(config=config, scheduler=scheduler, tokenizer=tokenizer)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
