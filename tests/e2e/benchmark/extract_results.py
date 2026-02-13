#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Benchmark result extraction and cross-phase comparison.

Reads all JSON result files from a results directory, extracts the 5 required
metrics from each benchmark, groups by phase, and computes deltas vs baseline.

Outputs a summary table to stdout and optionally writes a markdown report.

Usage:
    python extract_results.py [--results-dir /tmp/kimi-bench-results]
    python extract_results.py --results-dir ./benchmark-results/20260214
    python extract_results.py --output report.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase detection from filename and JSON content
# ---------------------------------------------------------------------------

# Filename patterns map to phases.  Files are named like:
#   bench_single_20260214T120000.json          -> inferred from label/config
#   bench_memory_baseline_20260214T120000.json  -> label in filename
#   bench_ngram_ngram4_k5_20260214T120000.json  -> label in filename
#
# We also inspect the JSON "label" field when present.

PHASE_KEYWORDS = {
    "baseline": "baseline",
    "fp8": "fp8",
    "kv_fp8": "fp8",
    "ngram": "ngram",
    "ngram4": "ngram",
    "ssd": "ssd",
    "all_features": "all_features",
    "all": "all_features",
}

PHASE_DISPLAY = {
    "baseline": "Phase 1: Baseline",
    "fp8": "Phase 2: KV FP8",
    "ngram": "Phase 3: N-gram Spec",
    "ssd": "Phase 4: SSD Cache",
    "all_features": "Phase 5: All Features",
}

PHASE_ORDER = ["baseline", "fp8", "ngram", "ssd", "all_features"]


def detect_phase(filepath: Path, data: dict) -> str:
    """Detect which phase a result file belongs to."""
    # Check JSON label field first
    label = data.get("label", "").lower()
    for kw, phase in PHASE_KEYWORDS.items():
        if kw in label:
            return phase

    # Check filename
    stem = filepath.stem.lower()
    for kw, phase in PHASE_KEYWORDS.items():
        if kw in stem:
            return phase

    # Default: assume baseline if no label is found
    return "baseline"


# ---------------------------------------------------------------------------
# Metric extraction per benchmark type
# ---------------------------------------------------------------------------

# The 5 required metrics:
#   prefill_time_s   - time to process prompt (first token latency)
#   prefill_tok_s    - prompt tokens / prefill time
#   decode_time_s    - time from first generated token to last
#   decode_tok_s     - generated tokens / decode time
#   throughput_tok_s  - total tokens / total time

def _derive_metrics(r: dict) -> dict | None:
    """Derive the 5 required metrics from a single result entry.

    Works for bench_single, bench_cache, bench_memory, bench_ngram results
    that have ttft_s, generation_tok_s, total_time_s, prompt_tokens,
    completion_tokens.  Also works for bench_batch per-request results that
    already have the explicit fields.
    """
    if r.get("error"):
        return None

    # If the result already has the explicit 5 metrics (bench_batch), use them
    if all(k in r for k in ("prefill_time_s", "prefill_tok_s", "decode_time_s",
                            "decode_tok_s", "throughput_tok_s")):
        return {
            "prefill_time_s": r["prefill_time_s"],
            "prefill_tok_s": r["prefill_tok_s"],
            "decode_time_s": r["decode_time_s"],
            "decode_tok_s": r["decode_tok_s"],
            "throughput_tok_s": r["throughput_tok_s"],
        }

    # Derive from ttft_s / generation_tok_s style results
    ttft = r.get("ttft_s")
    total_time = r.get("total_time_s", 0)
    prompt_tokens = r.get("prompt_tokens", 0)
    completion_tokens = r.get("completion_tokens", 0)
    gen_tok_s = r.get("generation_tok_s", 0)

    if total_time <= 0:
        return None

    prefill_time_s = ttft if ttft is not None and ttft > 0 else total_time
    decode_time_s = (total_time - prefill_time_s) if ttft is not None else 0.0
    prefill_tok_s = (
        round(prompt_tokens / prefill_time_s, 1)
        if prefill_time_s > 0 and prompt_tokens > 0 else 0.0
    )
    decode_tok_s = gen_tok_s if gen_tok_s else (
        round(completion_tokens / decode_time_s, 2)
        if decode_time_s > 0 and completion_tokens > 0 else 0.0
    )
    throughput_tok_s = round(
        (prompt_tokens + completion_tokens) / total_time, 2
    ) if total_time > 0 else 0.0

    return {
        "prefill_time_s": round(prefill_time_s, 4),
        "prefill_tok_s": prefill_tok_s,
        "decode_time_s": round(decode_time_s, 4),
        "decode_tok_s": decode_tok_s,
        "throughput_tok_s": throughput_tok_s,
    }


def extract_bench_single(data: dict) -> list[dict]:
    """Extract metrics from bench_single JSON."""
    rows = []
    for r in data.get("results", []):
        m = _derive_metrics(r)
        if m:
            m["test_name"] = r.get("test_name", "unknown")
            rows.append(m)
    return rows


def extract_bench_batch(data: dict) -> list[dict]:
    """Extract metrics from bench_batch JSON.

    We report the aggregate throughput per batch size and the average
    per-request metrics.
    """
    rows = []
    for batch_entry in data.get("results", []):
        k = batch_entry.get("batch_size", "?")
        # Collect per-request metrics across all runs
        per_request_metrics = []
        for run in batch_entry.get("runs", []):
            for req in run.get("requests", []):
                m = _derive_metrics(req)
                if m:
                    per_request_metrics.append(m)

        if per_request_metrics:
            avg = {}
            for key in ("prefill_time_s", "prefill_tok_s", "decode_time_s",
                        "decode_tok_s", "throughput_tok_s"):
                vals = [m[key] for m in per_request_metrics if m[key]]
                avg[key] = round(statistics.mean(vals), 2) if vals else 0.0

            avg["test_name"] = f"batch_k{k}"
            avg["aggregate_throughput_tok_s"] = batch_entry.get(
                "avg_aggregate_throughput_tok_s", 0.0
            )
            rows.append(avg)
    return rows


def extract_bench_concurrent(data: dict) -> list[dict]:
    """Extract metrics from bench_concurrent JSON."""
    rows = []
    for level_result in data.get("results", []):
        agg = level_result.get("aggregate", {})
        conc = agg.get("concurrency", "?")

        # Per-request metrics
        per_request_metrics = []
        for req in level_result.get("per_request", []):
            m = _derive_metrics(req)
            if m:
                per_request_metrics.append(m)

        if per_request_metrics:
            avg = {}
            for key in ("prefill_time_s", "prefill_tok_s", "decode_time_s",
                        "decode_tok_s", "throughput_tok_s"):
                vals = [m[key] for m in per_request_metrics if m[key]]
                avg[key] = round(statistics.mean(vals), 2) if vals else 0.0

            avg["test_name"] = f"concurrent_{conc}"
            avg["aggregate_tok_s"] = agg.get("aggregate_tok_s", 0.0)
            rows.append(avg)
    return rows


def extract_bench_cache(data: dict) -> list[dict]:
    """Extract metrics from bench_cache JSON."""
    rows = []
    for r in data.get("results", []):
        m = _derive_metrics(r)
        if m:
            m["test_name"] = r.get("test_name", "unknown")
            rows.append(m)
    return rows


def extract_bench_memory(data: dict) -> list[dict]:
    """Extract metrics from bench_memory JSON."""
    rows = []
    for r in data.get("results", []):
        m = _derive_metrics(r)
        if m:
            m["test_name"] = r.get("test_name", "unknown")
            # Include memory-specific stats
            ha = r.get("health_after", {}) or {}
            cache = ha.get("cache_stats", {})
            m["used_blocks"] = cache.get("used_blocks")
            m["utilization"] = ha.get("utilization")
            rows.append(m)
    return rows


def extract_bench_ngram(data: dict) -> list[dict]:
    """Extract metrics from bench_ngram JSON."""
    rows = []
    for r in data.get("results", []):
        m = _derive_metrics(r)
        if m:
            m["test_name"] = r.get("category", "unknown")
            rows.append(m)

    # Include overall spec decode stats
    spec = data.get("spec_metrics_after") or {}
    return rows


EXTRACTORS = {
    "bench_single": extract_bench_single,
    "bench_batch": extract_bench_batch,
    "batch_inference": extract_bench_batch,
    "bench_concurrent": extract_bench_concurrent,
    "bench_cache": extract_bench_cache,
    "bench_memory": extract_bench_memory,
    "bench_ngram": extract_bench_ngram,
}


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Load all JSON files and group extracted metrics by phase and benchmark.

    Returns: {phase: {benchmark_name: [metric_rows]}}
    """
    grouped: dict[str, dict[str, list[dict]]] = {}

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}", file=sys.stderr)
        return grouped

    for fp in json_files:
        try:
            data = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {fp.name}: {e}", file=sys.stderr)
            continue

        bench_name = data.get("benchmark", "unknown")
        phase = detect_phase(fp, data)

        extractor = EXTRACTORS.get(bench_name)
        if extractor is None:
            print(f"  SKIP {fp.name}: unknown benchmark type '{bench_name}'",
                  file=sys.stderr)
            continue

        rows = extractor(data)
        if not rows:
            print(f"  SKIP {fp.name}: no valid metrics extracted",
                  file=sys.stderr)
            continue

        grouped.setdefault(phase, {}).setdefault(bench_name, []).extend(rows)
        print(f"  OK   {fp.name} -> {PHASE_DISPLAY.get(phase, phase)}"
              f" / {bench_name} ({len(rows)} rows)")

    return grouped


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_phase_summary(bench_rows: dict[str, list[dict]]) -> dict:
    """Compute average 5-metric summary for a phase across all benchmarks.

    Returns dict with per-benchmark averages and an overall average.
    """
    summary = {}
    all_metrics = []

    for bench_name, rows in bench_rows.items():
        metrics_keys = ("prefill_time_s", "prefill_tok_s", "decode_time_s",
                        "decode_tok_s", "throughput_tok_s")
        avgs = {}
        for key in metrics_keys:
            vals = [r[key] for r in rows if r.get(key) is not None and r[key] > 0]
            avgs[key] = round(statistics.mean(vals), 3) if vals else 0.0

        summary[bench_name] = {
            "count": len(rows),
            **avgs,
        }
        all_metrics.extend(rows)

    # Overall average across all benchmarks
    if all_metrics:
        overall = {}
        for key in ("prefill_time_s", "prefill_tok_s", "decode_time_s",
                     "decode_tok_s", "throughput_tok_s"):
            vals = [r[key] for r in all_metrics
                    if r.get(key) is not None and r[key] > 0]
            overall[key] = round(statistics.mean(vals), 3) if vals else 0.0
        summary["_overall"] = overall

    return summary


def compute_delta(baseline_val: float, test_val: float) -> str:
    """Compute percentage delta string like '+15.2%' or '-3.1%'."""
    if baseline_val == 0 or test_val == 0:
        return "N/A"
    delta_pct = ((test_val - baseline_val) / baseline_val) * 100
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.1f}%"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "prefill_time_s": "Prefill Time (s)",
    "prefill_tok_s": "Prefill (tok/s)",
    "decode_time_s": "Decode Time (s)",
    "decode_tok_s": "Decode (tok/s)",
    "throughput_tok_s": "Throughput (tok/s)",
}

# For time metrics, lower is better; for tok/s metrics, higher is better.
LOWER_IS_BETTER = {"prefill_time_s", "decode_time_s"}


def format_table(grouped: dict[str, dict[str, list[dict]]]) -> str:
    """Format a comparison table as a string."""
    lines = []

    # Compute summaries per phase
    summaries: dict[str, dict] = {}
    for phase in PHASE_ORDER:
        if phase in grouped:
            summaries[phase] = compute_phase_summary(grouped[phase])

    if not summaries:
        return "No results to display.\n"

    baseline_overall = summaries.get("baseline", {}).get("_overall", {})

    # --- Overall comparison table ---
    lines.append("=" * 90)
    lines.append("CROSS-PHASE COMPARISON (Overall Averages)")
    lines.append("=" * 90)
    lines.append("")

    # Header
    metric_keys = ["prefill_time_s", "prefill_tok_s", "decode_time_s",
                   "decode_tok_s", "throughput_tok_s"]
    header = f"{'Phase':<25}"
    for mk in metric_keys:
        header += f" | {METRIC_LABELS[mk]:>18}"
    lines.append(header)
    lines.append("-" * len(header))

    for phase in PHASE_ORDER:
        if phase not in summaries:
            continue
        overall = summaries[phase].get("_overall", {})
        row = f"{PHASE_DISPLAY.get(phase, phase):<25}"
        for mk in metric_keys:
            val = overall.get(mk, 0.0)
            row += f" | {val:>18.3f}"
        lines.append(row)

    lines.append("")

    # --- Delta vs baseline ---
    if baseline_overall:
        lines.append("=" * 90)
        lines.append("DELTA vs BASELINE")
        lines.append("=" * 90)
        lines.append("")

        header = f"{'Phase':<25}"
        for mk in metric_keys:
            header += f" | {METRIC_LABELS[mk]:>18}"
        lines.append(header)
        lines.append("-" * len(header))

        for phase in PHASE_ORDER:
            if phase == "baseline" or phase not in summaries:
                continue
            overall = summaries[phase].get("_overall", {})
            row = f"{PHASE_DISPLAY.get(phase, phase):<25}"
            for mk in metric_keys:
                bv = baseline_overall.get(mk, 0.0)
                tv = overall.get(mk, 0.0)
                delta = compute_delta(bv, tv)
                row += f" | {delta:>18}"
            lines.append(row)

        lines.append("")
        lines.append("Note: For time metrics (Prefill Time, Decode Time), "
                     "negative delta = improvement.")
        lines.append("      For tok/s metrics, positive delta = improvement.")

    # --- Per-benchmark detail ---
    lines.append("")
    lines.append("=" * 90)
    lines.append("PER-BENCHMARK DETAIL")
    lines.append("=" * 90)

    for phase in PHASE_ORDER:
        if phase not in summaries:
            continue
        lines.append("")
        lines.append(f"--- {PHASE_DISPLAY.get(phase, phase)} ---")

        for bench_name, bench_summary in summaries[phase].items():
            if bench_name == "_overall":
                continue
            count = bench_summary.get("count", 0)
            lines.append(f"  {bench_name} ({count} tests):")
            for mk in metric_keys:
                val = bench_summary.get(mk, 0.0)
                label = METRIC_LABELS[mk]
                lines.append(f"    {label:<22}: {val:.3f}")

    return "\n".join(lines) + "\n"


def format_markdown(grouped: dict[str, dict[str, list[dict]]]) -> str:
    """Format a comparison report as markdown."""
    lines = []

    summaries: dict[str, dict] = {}
    for phase in PHASE_ORDER:
        if phase in grouped:
            summaries[phase] = compute_phase_summary(grouped[phase])

    if not summaries:
        return "# Benchmark Report\n\nNo results found.\n"

    baseline_overall = summaries.get("baseline", {}).get("_overall", {})
    metric_keys = ["prefill_time_s", "prefill_tok_s", "decode_time_s",
                   "decode_tok_s", "throughput_tok_s"]

    lines.append("# Benchmark Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # Overall comparison table
    lines.append("## Cross-Phase Comparison (Overall Averages)")
    lines.append("")
    header = "| Phase |"
    sep = "| --- |"
    for mk in metric_keys:
        header += f" {METRIC_LABELS[mk]} |"
        sep += " ---: |"
    lines.append(header)
    lines.append(sep)

    for phase in PHASE_ORDER:
        if phase not in summaries:
            continue
        overall = summaries[phase].get("_overall", {})
        row = f"| {PHASE_DISPLAY.get(phase, phase)} |"
        for mk in metric_keys:
            val = overall.get(mk, 0.0)
            row += f" {val:.3f} |"
        lines.append(row)

    lines.append("")

    # Delta vs baseline
    if baseline_overall:
        lines.append("## Delta vs Baseline")
        lines.append("")
        header = "| Phase |"
        sep = "| --- |"
        for mk in metric_keys:
            header += f" {METRIC_LABELS[mk]} |"
            sep += " ---: |"
        lines.append(header)
        lines.append(sep)

        for phase in PHASE_ORDER:
            if phase == "baseline" or phase not in summaries:
                continue
            overall = summaries[phase].get("_overall", {})
            row = f"| {PHASE_DISPLAY.get(phase, phase)} |"
            for mk in metric_keys:
                bv = baseline_overall.get(mk, 0.0)
                tv = overall.get(mk, 0.0)
                delta = compute_delta(bv, tv)
                row += f" {delta} |"
            lines.append(row)

        lines.append("")
        lines.append("> For time metrics (Prefill Time, Decode Time), "
                     "negative delta = improvement.")
        lines.append("> For tok/s metrics, positive delta = improvement.")
        lines.append("")

    # Per-benchmark detail
    lines.append("## Per-Benchmark Detail")
    lines.append("")

    for phase in PHASE_ORDER:
        if phase not in summaries:
            continue
        lines.append(f"### {PHASE_DISPLAY.get(phase, phase)}")
        lines.append("")

        for bench_name, bench_summary in summaries[phase].items():
            if bench_name == "_overall":
                continue
            count = bench_summary.get("count", 0)
            lines.append(f"**{bench_name}** ({count} tests)")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            for mk in metric_keys:
                val = bench_summary.get(mk, 0.0)
                lines.append(f"| {METRIC_LABELS[mk]} | {val:.3f} |")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract and compare benchmark results across phases"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/tmp/kimi-bench-results"),
        help="Directory containing benchmark JSON files "
             "(default: /tmp/kimi-bench-results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this file (optional)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: Results directory not found: {results_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    print()

    grouped = load_results(results_dir)
    if not grouped:
        print("\nNo valid results found.", file=sys.stderr)
        sys.exit(1)

    print()

    # Print summary table to stdout
    table = format_table(grouped)
    print(table)

    # Optionally write markdown
    if args.output:
        md = format_markdown(grouped)
        args.output.write_text(md)
        print(f"Markdown report written to: {args.output}")


if __name__ == "__main__":
    main()
