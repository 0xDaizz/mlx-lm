#!/Users/hw/mlx-lm-server/.venv/bin/python
"""System monitor for mlx-lm-server benchmarks.

Polls /health and /metrics endpoints at configurable intervals, writes CSV
output, and prints a summary of key metrics at exit.

Usage:
    python monitor.py --output bench.csv --duration 300 --interval 2
"""

from __future__ import annotations

import argparse
import csv
import io
import signal
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import json


# -----------------------------------------------------------------------
# Endpoint polling
# -----------------------------------------------------------------------

def fetch_health(base_url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    """Fetch JSON from /health. Returns None on failure."""
    try:
        with urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def fetch_metrics(base_url: str, timeout: float = 5.0) -> dict[str, float] | None:
    """Fetch Prometheus-style /metrics and parse into dict. Returns None on failure."""
    try:
        with urlopen(f"{base_url}/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return _parse_prometheus(text)
    except (URLError, OSError, ValueError):
        return None


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition format lines into {metric_name: value}."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                result[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return result


# -----------------------------------------------------------------------
# Snapshot extraction
# -----------------------------------------------------------------------

# CSV columns — order matters for the header row
CSV_COLUMNS = [
    "timestamp",
    "elapsed_s",
    "status",
    "utilization",
    # Block pool
    "total_blocks",
    "used_blocks",
    "free_blocks",
    "cached_blocks",
    # Sequences / requests
    "active_sequences",
    "queued_requests",
    # Cache effectiveness
    "cache_hit_rate",
    "cache_effectiveness",
    "cache_hits_block",
    "cache_hits_sequence",
    "cache_misses",
    # Token counters
    "tokens_generated",
    "total_prefill_tokens",
    "total_cached_tokens",
    "requests_completed",
    "total_requests",
    # SSD stats
    "ssd_save_success",
    "ssd_save_fail",
    "ssd_lru_prune_count",
    # Speculative decoding
    "spec_tokens_drafted",
    "spec_tokens_accepted",
    # Distributed
    "dist_enabled",
    "dist_rank",
    "dist_world_size",
    "dist_fatal",
    "dist_bus_error_count",
    # Prometheus metrics
    "prom_active_sequences",
    "prom_queued_requests",
    "prom_used_blocks",
    "prom_free_blocks",
    "prom_cache_hit_rate",
    "prom_dist_fatal",
]


def build_row(
    health: dict[str, Any] | None,
    metrics: dict[str, float] | None,
    start_time: float,
) -> dict[str, Any]:
    """Build a flat CSV row from health + metrics responses."""
    now = time.time()
    row: dict[str, Any] = {
        "timestamp": f"{now:.3f}",
        "elapsed_s": f"{now - start_time:.1f}",
    }

    if health is not None:
        cs = health.get("cache_stats", {})
        dist = health.get("distributed", {})
        row["status"] = health.get("status", "unknown")
        row["utilization"] = health.get("utilization", "")
        # Block pool
        row["total_blocks"] = cs.get("total_blocks", "")
        row["used_blocks"] = cs.get("used_blocks", "")
        row["free_blocks"] = cs.get("free_blocks", "")
        row["cached_blocks"] = cs.get("cached_blocks", "")
        # Sequences / requests
        row["active_sequences"] = cs.get("active_sequences", "")
        row["queued_requests"] = cs.get("queued_requests", "")
        # Cache
        row["cache_hit_rate"] = cs.get("cache_hit_rate", "")
        row["cache_effectiveness"] = cs.get("cache_effectiveness", "")
        row["cache_hits_block"] = cs.get("cache_hits_block", "")
        row["cache_hits_sequence"] = cs.get("cache_hits_sequence", "")
        row["cache_misses"] = cs.get("cache_misses", "")
        # Tokens
        row["tokens_generated"] = cs.get("tokens_generated", "")
        row["total_prefill_tokens"] = cs.get("total_prefill_tokens", "")
        row["total_cached_tokens"] = cs.get("total_cached_tokens", "")
        row["requests_completed"] = cs.get("requests_completed", "")
        row["total_requests"] = cs.get("total_requests", "")
        # SSD
        row["ssd_save_success"] = cs.get("ssd_save_success", "")
        row["ssd_save_fail"] = cs.get("ssd_save_fail", "")
        row["ssd_lru_prune_count"] = cs.get("ssd_lru_prune_count", "")
        # Speculative
        row["spec_tokens_drafted"] = cs.get("spec_tokens_drafted", "")
        row["spec_tokens_accepted"] = cs.get("spec_tokens_accepted", "")
        # Distributed
        row["dist_enabled"] = dist.get("enabled", "")
        row["dist_rank"] = dist.get("rank", "")
        row["dist_world_size"] = dist.get("world_size", "")
        row["dist_fatal"] = dist.get("fatal", "")
        row["dist_bus_error_count"] = cs.get("dist_bus_error_count", "")
    else:
        row["status"] = "unreachable"

    if metrics is not None:
        row["prom_active_sequences"] = metrics.get("mlx_lm_server_active_sequences", "")
        row["prom_queued_requests"] = metrics.get("mlx_lm_server_queued_requests", "")
        row["prom_used_blocks"] = metrics.get("mlx_lm_server_used_blocks", "")
        row["prom_free_blocks"] = metrics.get("mlx_lm_server_free_blocks", "")
        row["prom_cache_hit_rate"] = metrics.get("mlx_lm_server_cache_hit_rate", "")
        row["prom_dist_fatal"] = metrics.get("mlx_lm_server_dist_fatal", "")

    # Fill in missing columns with empty string
    for col in CSV_COLUMNS:
        row.setdefault(col, "")

    return row


# -----------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------

SUMMARY_KEYS = [
    "utilization",
    "used_blocks",
    "free_blocks",
    "cached_blocks",
    "active_sequences",
    "queued_requests",
    "cache_hit_rate",
    "cache_effectiveness",
    "tokens_generated",
    "requests_completed",
    "spec_tokens_drafted",
    "spec_tokens_accepted",
]


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print min/max/avg/last for key numeric metrics."""
    if not rows:
        print("\n[monitor] No data collected.")
        return

    print(f"\n{'=' * 72}")
    print(f"  Monitor Summary  ({len(rows)} samples)")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<28s} {'Min':>10s} {'Max':>10s} {'Avg':>10s} {'Last':>10s}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for key in SUMMARY_KEYS:
        values: list[float] = []
        for r in rows:
            raw = r.get(key, "")
            if raw == "" or raw is None:
                continue
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                continue
        if not values:
            continue
        mn = min(values)
        mx = max(values)
        avg = sum(values) / len(values)
        last = values[-1]
        print(f"  {key:<28s} {mn:>10.2f} {mx:>10.2f} {avg:>10.2f} {last:>10.2f}")

    # Derived: speculative acceptance rate
    last_row = rows[-1]
    drafted = _to_float(last_row.get("spec_tokens_drafted", ""))
    accepted = _to_float(last_row.get("spec_tokens_accepted", ""))
    if drafted is not None and drafted > 0 and accepted is not None:
        rate = accepted / drafted * 100
        print(f"  {'spec_acceptance_rate_%':<28s} {'':>10s} {'':>10s} {'':>10s} {rate:>10.1f}")

    # Duration
    first_elapsed = _to_float(rows[0].get("elapsed_s", ""))
    last_elapsed = _to_float(rows[-1].get("elapsed_s", ""))
    if first_elapsed is not None and last_elapsed is not None:
        print(f"\n  Duration: {last_elapsed - first_elapsed:.1f}s")

    print(f"{'=' * 72}")


def _to_float(val: Any) -> float | None:
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-lm-server benchmark monitor")
    parser.add_argument(
        "--url", default="http://localhost:8080",
        help="Server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="CSV output file path (default: stdout)",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=0,
        help="Duration in seconds (0 = run until Ctrl-C)",
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=2.0,
        help="Polling interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-sample console output",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    interval = max(0.5, args.interval)
    duration = args.duration

    # Graceful shutdown
    stop = False

    def _signal_handler(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Open output
    out_file = None
    if args.output:
        out_file = open(args.output, "w", newline="")
        writer = csv.DictWriter(out_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

    rows: list[dict[str, Any]] = []
    start_time = time.time()
    sample = 0

    print(f"[monitor] Polling {base_url} every {interval}s", file=sys.stderr)
    if duration > 0:
        print(f"[monitor] Will run for {duration}s", file=sys.stderr)
    print(f"[monitor] Output: {args.output or 'stdout'}", file=sys.stderr)

    try:
        while not stop:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break

            health = fetch_health(base_url)
            metrics = fetch_metrics(base_url)
            row = build_row(health, metrics, start_time)
            rows.append(row)

            if args.output and out_file:
                writer.writerow(row)
                out_file.flush()
            else:
                writer.writerow(row)
                sys.stdout.flush()

            sample += 1
            if not args.quiet:
                status = row.get("status", "?")
                used = row.get("used_blocks", "?")
                total = row.get("total_blocks", "?")
                active = row.get("active_sequences", "?")
                tgen = row.get("tokens_generated", "?")
                print(
                    f"[monitor] #{sample:>4d} | {elapsed:>7.1f}s | "
                    f"status={status} | blocks={used}/{total} | "
                    f"active={active} | tokens={tgen}",
                    file=sys.stderr,
                )

            # Sleep in short increments for responsive shutdown
            sleep_end = time.time() + interval
            while time.time() < sleep_end and not stop:
                time.sleep(min(0.25, sleep_end - time.time()))

    finally:
        if out_file:
            out_file.close()
        print_summary(rows)


if __name__ == "__main__":
    main()
