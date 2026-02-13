#!/usr/bin/env bash
# monitor_system.sh — System-level monitoring for mlx-lm-server benchmarks
#
# Runs vm_stat, memory_pressure, and optional powermetrics in parallel.
# All background processes are cleaned up on exit.
#
# Usage:
#   ./monitor_system.sh [--duration SECONDS] [--interval SECONDS] [--output-dir DIR]
#
# Output files (in output dir):
#   vmstat.log          — vm_stat snapshots
#   memory_pressure.log — memory_pressure snapshots
#   power.log           — powermetrics GPU power (requires sudo, optional)
#   system_summary.txt  — final summary

set -euo pipefail

# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------
DURATION=0          # 0 = run until Ctrl-C
INTERVAL=2          # seconds
OUTPUT_DIR="/tmp/kimi-bench"

# -----------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)   DURATION="$2";    shift 2 ;;
        --interval)   INTERVAL="$2";    shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--duration SECONDS] [--interval SECONDS] [--output-dir DIR]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

VMSTAT_LOG="$OUTPUT_DIR/vmstat.log"
MEMORY_LOG="$OUTPUT_DIR/memory_pressure.log"
POWER_LOG="$OUTPUT_DIR/power.log"
SUMMARY_FILE="$OUTPUT_DIR/system_summary.txt"

# Track child PIDs for cleanup
PIDS=()

cleanup() {
    echo "[monitor_system] Shutting down..." >&2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait briefly for children
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    write_summary
    echo "[monitor_system] Logs saved to $OUTPUT_DIR" >&2
}

trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------
# vm_stat monitor
# -----------------------------------------------------------------------
start_vmstat() {
    local interval_int="${INTERVAL%.*}"
    [[ "$interval_int" -lt 1 ]] && interval_int=1

    echo "[monitor_system] Starting vm_stat (interval=${interval_int}s) -> $VMSTAT_LOG" >&2
    {
        echo "# vm_stat started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        vm_stat "$interval_int"
    } > "$VMSTAT_LOG" 2>&1 &
    PIDS+=($!)
}

# -----------------------------------------------------------------------
# memory_pressure monitor (periodic snapshots)
# -----------------------------------------------------------------------
start_memory_monitor() {
    echo "[monitor_system] Starting memory pressure monitor (interval=${INTERVAL}s) -> $MEMORY_LOG" >&2
    {
        echo "# memory_pressure started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        local count=0
        while true; do
            echo "--- sample $count at $(date -u '+%H:%M:%S') ---"
            # memory_pressure gives a quick system pressure level
            if command -v memory_pressure &>/dev/null; then
                memory_pressure -S 2>/dev/null || echo "(memory_pressure unavailable)"
            else
                # Fallback: parse vm_stat for page counts
                vm_stat | head -20
            fi
            echo ""
            count=$((count + 1))
            sleep "$INTERVAL"
        done
    } > "$MEMORY_LOG" 2>&1 &
    PIDS+=($!)
}

# -----------------------------------------------------------------------
# powermetrics (GPU power, optional — requires sudo)
# -----------------------------------------------------------------------
start_powermetrics() {
    if ! command -v powermetrics &>/dev/null; then
        echo "[monitor_system] powermetrics not found, skipping GPU power monitoring" >&2
        return
    fi

    # Convert interval to milliseconds
    local interval_ms
    interval_ms=$(python3 -c "print(int(${INTERVAL} * 1000))")
    [[ "$interval_ms" -lt 1000 ]] && interval_ms=1000

    echo "[monitor_system] Starting powermetrics (interval=${interval_ms}ms) -> $POWER_LOG" >&2
    echo "[monitor_system] NOTE: powermetrics requires sudo. Will skip if not authorized." >&2
    {
        echo "# powermetrics started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        # Try sudo non-interactively; skip if no auth
        sudo -n powermetrics \
            --samplers gpu_power,ane_power \
            -i "$interval_ms" \
            --format text 2>&1 || {
            echo "# powermetrics: sudo not available (non-interactive), skipping"
        }
    } > "$POWER_LOG" 2>&1 &
    PIDS+=($!)
}

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
write_summary() {
    {
        echo "========================================"
        echo "  System Monitor Summary"
        echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "========================================"
        echo ""

        echo "--- Hardware ---"
        sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "(unknown CPU)"
        echo "Physical Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1073741824}')"
        echo "CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
        echo ""

        echo "--- vm_stat (last snapshot) ---"
        if [[ -f "$VMSTAT_LOG" ]]; then
            tail -5 "$VMSTAT_LOG"
        else
            echo "(no data)"
        fi
        echo ""

        echo "--- Memory Pressure (last snapshot) ---"
        if [[ -f "$MEMORY_LOG" ]]; then
            # Get the last sample block
            tail -10 "$MEMORY_LOG"
        else
            echo "(no data)"
        fi
        echo ""

        echo "--- Power (last snapshot) ---"
        if [[ -f "$POWER_LOG" ]] && [[ -s "$POWER_LOG" ]]; then
            tail -10 "$POWER_LOG"
        else
            echo "(no data or powermetrics not available)"
        fi
        echo ""

        echo "--- Log Files ---"
        echo "  vmstat:     $VMSTAT_LOG"
        echo "  memory:     $MEMORY_LOG"
        echo "  power:      $POWER_LOG"
        echo "========================================"
    } > "$SUMMARY_FILE"

    # Also print to stderr
    cat "$SUMMARY_FILE" >&2
}

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
echo "[monitor_system] Output dir: $OUTPUT_DIR" >&2
echo "[monitor_system] Interval: ${INTERVAL}s, Duration: ${DURATION}s (0=unlimited)" >&2

start_vmstat
start_memory_monitor
start_powermetrics

echo "[monitor_system] All monitors started. PIDs: ${PIDS[*]}" >&2

if [[ "$DURATION" -gt 0 ]]; then
    echo "[monitor_system] Running for ${DURATION}s..." >&2
    sleep "$DURATION"
else
    echo "[monitor_system] Running until Ctrl-C..." >&2
    # Wait for any child to exit, or until interrupted
    while true; do
        sleep 1
    done
fi
