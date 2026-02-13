#!/usr/bin/env bash
# run_all.sh -- Master script to run all Kimi K2.5 benchmarks in order.
#
# This script runs benchmarks against an already-running server.
# It does NOT start or restart the server. Different server configs
# (spec decode, kv-bits, ssd-policy) require manual server restarts.
#
# Usage:
#   ./run_all.sh [--server-url http://localhost:8080] [--max-tokens 256]
#
# Prerequisites:
#   - Server running at the specified URL
#   - Python venv with httpx installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/hw/mlx-lm-server/.venv/bin/python"
RESULTS_DIR="/tmp/kimi-bench-results"

# Default args
SERVER_URL="http://localhost:8080"
MAX_TOKENS=256

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 [--server-url URL] [--max-tokens N]"
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Kimi K2.5 Benchmark Suite"
echo "============================================"
echo "  Server:     $SERVER_URL"
echo "  Max tokens: $MAX_TOKENS"
echo "  Results:    $RESULTS_DIR"
echo "  Python:     $PYTHON"
echo "============================================"
echo ""

# Check server health
echo "Checking server health..."
if ! curl -sf "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Server not reachable at $SERVER_URL"
    echo "Start the server first, then re-run this script."
    exit 1
fi
echo "Server is up."
echo ""

# Phase 1: Single request baseline
echo "==============================="
echo " Phase 1: Single Request Baseline"
echo "==============================="
"$PYTHON" "$SCRIPT_DIR/bench_single.py" \
    --server-url "$SERVER_URL" \
    --max-tokens "$MAX_TOKENS"
echo ""

# Phase 2: Memory stats
echo "==============================="
echo " Phase 2: Memory Stats"
echo "==============================="
"$PYTHON" "$SCRIPT_DIR/bench_memory.py" \
    --server-url "$SERVER_URL" \
    --max-tokens "$MAX_TOKENS" \
    --label "current"
echo ""

# Phase 3: Cache behavior
echo "==============================="
echo " Phase 3: Cache Hit/Miss"
echo "==============================="
"$PYTHON" "$SCRIPT_DIR/bench_cache.py" \
    --server-url "$SERVER_URL" \
    --max-tokens 128
echo ""

# Phase 4: N-gram spec decode
echo "==============================="
echo " Phase 4: N-gram Spec Decode"
echo "==============================="
"$PYTHON" "$SCRIPT_DIR/bench_ngram.py" \
    --server-url "$SERVER_URL" \
    --max-tokens "$MAX_TOKENS" \
    --label "current"
echo ""

# Phase 5: Concurrent load
echo "==============================="
echo " Phase 5: Concurrent Load"
echo "==============================="
"$PYTHON" "$SCRIPT_DIR/bench_concurrent.py" \
    --server-url "$SERVER_URL" \
    --max-tokens 128
echo ""

echo "============================================"
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS_DIR"
echo "============================================"
echo ""
echo "To compare different server configs, restart the server with"
echo "different flags and re-run individual benchmarks:"
echo ""
echo "  # Baseline (no spec decode)"
echo "  $PYTHON $SCRIPT_DIR/bench_ngram.py --label baseline"
echo ""
echo "  # N-gram spec decode (ngram-max=4, k=5)"
echo "  # Server: --spec-decode ngram --ngram-max 4 --num-speculative-tokens 5"
echo "  $PYTHON $SCRIPT_DIR/bench_ngram.py --label ngram4_k5"
echo ""
echo "  # FP8 KV cache"
echo "  # Server: --kv-bits 8"
echo "  $PYTHON $SCRIPT_DIR/bench_memory.py --label fp8"
echo ""
echo "Results files in $RESULTS_DIR can be compared with:"
echo "  ls -la $RESULTS_DIR/"
