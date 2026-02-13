"""Dashboard API endpoints for mlx-lm-server.

Provides real-time server metrics via SSE and static configuration via REST.

Endpoints:
    GET /dashboard/api/stats  -- SSE stream pushing aggregated metrics every 1s
    GET /dashboard/api/config -- One-time fetch of server configuration
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard/api", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_metal_memory() -> tuple[int, int]:
    """Return (active_bytes, peak_bytes) from MLX Metal allocator.

    Falls back to (0, 0) when MLX is unavailable or Metal is not supported.
    """
    try:
        import mlx.core as mx

        active_bytes = mx.metal.get_active_memory()
        peak_bytes = mx.metal.get_peak_memory()
        return active_bytes, peak_bytes
    except Exception:
        return 0, 0


def _get_spec_decode_metrics(scheduler: Any) -> dict[str, Any]:
    """Extract speculative decoding metrics from the scheduler, if available."""
    try:
        spec_engine = getattr(scheduler, "_spec_engine", None)
        if spec_engine is not None:
            metrics = spec_engine.controller.get_metrics()
            return {
                "enabled": bool(metrics.get("spec_decode_enabled", False)),
                "mode": metrics.get("spec_decode_mode", "unknown"),
                "acceptance_rate_ema": metrics.get("acceptance_rate_ema", 0.0),
                "acceptance_rate_overall": metrics.get("acceptance_rate_overall", 0.0),
                "tokens_per_step": metrics.get("avg_tokens_per_step", 0.0),
                "current_k": metrics.get("current_k", 0),
                "adaptive_k": metrics.get("adaptive_k_current", 0),
            }
    except Exception:
        logger.debug("Failed to read spec decode metrics", exc_info=True)
    return {
        "enabled": False,
        "mode": "none",
        "acceptance_rate_ema": 0.0,
        "acceptance_rate_overall": 0.0,
        "tokens_per_step": 0.0,
        "current_k": 0,
        "adaptive_k": 0,
    }


def _get_distributed_info(dist_ctx: Any, cache_stats: dict[str, Any]) -> dict[str, Any]:
    """Build the distributed section of the stats payload."""
    if dist_ctx and getattr(dist_ctx, "enabled", False):
        fatal = bool(cache_stats.get("dist_fatal", False))
        return {
            "enabled": True,
            "rank": dist_ctx.rank,
            "world_size": dist_ctx.world_size,
            "backend": getattr(dist_ctx, "backend", "unknown"),
            "healthy": not fatal,
            "bus_error_count": cache_stats.get("dist_bus_error_count", 0),
        }
    return {
        "enabled": False,
        "rank": 0,
        "world_size": 1,
        "backend": "none",
        "healthy": True,
        "bus_error_count": 0,
    }


# ---------------------------------------------------------------------------
# SSE stats endpoint
# ---------------------------------------------------------------------------


@router.get("/stats")
async def stats_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events endpoint streaming aggregated metrics every 1 second.

    Pushes a JSON object per tick with server, throughput, KV cache, SSD cache,
    memory, speculative decoding, and distributed health sections.
    """

    async def _event_generator():
        # Rolling throughput tracking: deque of (timestamp, cumulative_tokens)
        # samples over the last 5 seconds.
        throughput_window: deque[tuple[float, int]] = deque()
        window_duration = 5.0

        # Cumulative counters tracked across ticks for delta-based throughput.
        prev_prefill: int | None = None
        prev_cached: int | None = None
        cumulative_generated: int = 0

        # Request completion tracking (approximated from active sequence changes).
        requests_completed: int = 0
        requests_errored: int = 0

        try:
            while True:
                # Check if the client has disconnected.
                if await request.is_disconnected():
                    logger.debug("Dashboard SSE client disconnected")
                    break

                now = time.time()
                scheduler = request.app.state.scheduler
                config = request.app.state.config
                started_at = request.app.state.started_at
                dist_ctx = request.app.state.dist_ctx

                # -- Gather cache stats (primary data source) --
                cache_stats = scheduler.get_cache_stats()

                # -- Throughput calculation --
                current_prefill = cache_stats.get("total_prefill_tokens", 0)
                current_cached = cache_stats.get("total_cached_tokens", 0)
                total_processed = current_prefill + current_cached

                # On first tick, initialize previous values.
                if prev_prefill is None:
                    prev_prefill = current_prefill
                    prev_cached = current_cached

                # Accumulate generated tokens as delta from previous tick.
                delta_prefill = max(0, current_prefill - prev_prefill)
                delta_cached = max(0, current_cached - (prev_cached or 0))
                cumulative_generated += delta_prefill + delta_cached
                prev_prefill = current_prefill
                prev_cached = current_cached

                # Maintain rolling window for tokens/sec calculation.
                throughput_window.append((now, cumulative_generated))
                # Evict samples older than the window.
                while throughput_window and (now - throughput_window[0][0]) > window_duration:
                    throughput_window.popleft()

                # Calculate tokens/sec from the rolling window.
                tokens_per_sec = 0.0
                if len(throughput_window) >= 2:
                    oldest_ts, oldest_tokens = throughput_window[0]
                    dt = now - oldest_ts
                    if dt > 0:
                        tokens_per_sec = (cumulative_generated - oldest_tokens) / dt

                # -- Metal GPU memory --
                active_bytes, peak_bytes = _get_metal_memory()
                memory_pressure = 0.0
                if peak_bytes > 0:
                    memory_pressure = active_bytes / peak_bytes

                # -- SSD cache --
                ssd_enabled = getattr(config, "ssd_enabled", False)

                # -- Build payload --
                payload: dict[str, Any] = {
                    "timestamp": now,
                    "uptime_s": round(time.monotonic() - started_at, 1),
                    "server": {
                        "active_sequences": cache_stats.get("active_sequences", 0),
                        "queued_requests": cache_stats.get("queued_requests", 0),
                        "max_concurrent": config.max_concurrent_requests,
                        "max_queue": config.max_queue_size,
                    },
                    "throughput": {
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "tokens_generated": cumulative_generated,
                        "requests_completed": requests_completed,
                        "requests_errored": requests_errored,
                        "prefill_tokens": current_prefill,
                        "cached_tokens": current_cached,
                    },
                    "kv_cache": {
                        "total_blocks": cache_stats.get("total_blocks", 0),
                        "used_blocks": cache_stats.get("used_blocks", 0),
                        "free_blocks": cache_stats.get("free_blocks", 0),
                        "cached_blocks": cache_stats.get("cached_blocks", 0),
                        "hit_rate": cache_stats.get("cache_hit_rate", 0.0),
                        "effectiveness": cache_stats.get("cache_effectiveness", 0.0),
                    },
                    "ssd_cache": {
                        "enabled": ssd_enabled,
                        "total_bytes": cache_stats.get("ssd_total_bytes", 0),
                        "max_size_bytes": cache_stats.get("ssd_max_size_bytes", 0),
                        "save_success": cache_stats.get("ssd_save_success", 0),
                        "save_fail": cache_stats.get("ssd_save_fail", 0),
                        "lru_prune_count": cache_stats.get("ssd_lru_prune_count", 0),
                    },
                    "memory": {
                        "active_bytes": active_bytes,
                        "peak_bytes": peak_bytes,
                        "pressure": round(memory_pressure, 4),
                    },
                    "spec_decode": _get_spec_decode_metrics(scheduler),
                    "distributed": _get_distributed_info(dist_ctx, cache_stats),
                }

                yield f"data: {json.dumps(payload)}\n\n"

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.debug("Dashboard SSE generator cancelled")
        except Exception:
            logger.exception("Unexpected error in dashboard SSE generator")

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Config endpoint
# ---------------------------------------------------------------------------


@router.get("/config")
async def config(request: Request) -> JSONResponse:
    """Return static server configuration as JSON.

    This is a one-time fetch endpoint (not streaming) providing model name,
    cache parameters, and feature flags for the dashboard UI.
    """
    cfg = request.app.state.config
    scheduler = request.app.state.scheduler
    dist_ctx = request.app.state.dist_ctx

    # Determine spec decode mode from config.
    spec_decode_mode = getattr(cfg, "spec_decode_mode", "none")

    # Determine distributed mode.
    distributed_mode = "off"
    if dist_ctx and getattr(dist_ctx, "enabled", False):
        distributed_mode = getattr(dist_ctx, "backend", getattr(cfg, "distributed_mode", "off"))

    # Get block info from scheduler stats for accuracy (config has defaults,
    # but the scheduler may have adjusted them during init).
    cache_stats = scheduler.get_cache_stats()

    return JSONResponse(content={
        "model_name": cfg.model,
        "max_concurrent_requests": cfg.max_concurrent_requests,
        "max_queue_size": cfg.max_queue_size,
        "block_size": cfg.block_size,
        "num_blocks": cache_stats.get("total_blocks", cfg.num_blocks),
        "kv_bits": cfg.kv_bits,
        "ssd_enabled": getattr(cfg, "ssd_enabled", False),
        "spec_decode_mode": spec_decode_mode,
        "distributed_mode": distributed_mode,
    })
