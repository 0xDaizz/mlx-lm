#!/usr/bin/env bash
# launch_all.sh — JACCL TP + all features: KV FP8 + SSD write-through + N-gram spec decode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODEL="/Users/hw/mlx-lm-server/models/Kimi-K2.5"
HOSTFILE="/Users/hw/mlx-lm-server/examples/distributed/hosts-hwstudio.json"
MLX_LAUNCH="/Users/hw/mlx-lm-server/.venv/bin/mlx.launch"
PYTHON="/Users/hw/mlx-lm-server/.venv/bin/python"
PORT=8080
LOGFILE="/tmp/kimi-bench-all.log"
CONFIG_NAME="all"

export PYTHONPATH="$PROJECT_ROOT"

echo "[$CONFIG_NAME] Launching server on port $PORT ..."
echo "[$CONFIG_NAME] Log: $LOGFILE"

"$MLX_LAUNCH" --backend jaccl --hostfile "$HOSTFILE" --python "$PYTHON" -- \
    "$SCRIPT_DIR/server_wrapper.py" \
    --distributed-mode jaccl \
    --model "$MODEL" \
    --port "$PORT" \
    --kv-bits 8 \
    --kv-group-size 64 \
    --block-size 16 \
    --num-blocks 2048 \
    --ssd-policy write_through \
    --ssd-async-writes \
    --ssd-max-size-gb 50 \
    --spec-decode ngram \
    --ngram-max 4 \
    --num-speculative-tokens 5 \
    --first-token-timeout-s 600 \
    > "$LOGFILE" 2>&1 &

SERVER_PID=$!
echo "[$CONFIG_NAME] Server PID: $SERVER_PID (backgrounded)"

# Wait for health check
MAX_WAIT=600
ELAPSED=0
echo "[$CONFIG_NAME] Waiting for server health (max ${MAX_WAIT}s) ..."
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[$CONFIG_NAME] Server is healthy after ${ELAPSED}s"
        echo "$SERVER_PID" > /tmp/kimi-bench-server.pid
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$CONFIG_NAME] ERROR: Server process died. Check $LOGFILE"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo "[$CONFIG_NAME] ERROR: Server did not become healthy within ${MAX_WAIT}s"
echo "[$CONFIG_NAME] Check $LOGFILE for details"
kill "$SERVER_PID" 2>/dev/null || true
exit 1
