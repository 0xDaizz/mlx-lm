#!/usr/bin/env bash
# stop_server.sh — Gracefully stop the benchmark server
set -euo pipefail

PIDFILE="/tmp/kimi-bench-server.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "[stop] No PID file found at $PIDFILE"
    # Try to find and kill any mlx_lm_server processes on port 8080
    PIDS=$(lsof -ti :8080 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "[stop] Found processes on port 8080: $PIDS"
        echo "[stop] Sending SIGTERM ..."
        echo "$PIDS" | xargs kill 2>/dev/null || true
        sleep 3
        # Force kill if still alive
        REMAINING=$(lsof -ti :8080 2>/dev/null || true)
        if [ -n "$REMAINING" ]; then
            echo "[stop] Force killing remaining processes: $REMAINING"
            echo "$REMAINING" | xargs kill -9 2>/dev/null || true
        fi
        echo "[stop] Done."
    else
        echo "[stop] No server processes found on port 8080."
    fi
    exit 0
fi

SERVER_PID=$(cat "$PIDFILE")
echo "[stop] Stopping server PID: $SERVER_PID"

if kill -0 "$SERVER_PID" 2>/dev/null; then
    # Send SIGTERM for graceful shutdown
    kill "$SERVER_PID"
    echo "[stop] Sent SIGTERM, waiting for shutdown ..."

    # Wait up to 30s for graceful shutdown
    ELAPSED=0
    while [ $ELAPSED -lt 30 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[stop] Server stopped gracefully after ${ELAPSED}s"
            rm -f "$PIDFILE"
            exit 0
        fi
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    # Force kill if still alive
    echo "[stop] Server did not stop gracefully, sending SIGKILL ..."
    kill -9 "$SERVER_PID" 2>/dev/null || true
    sleep 1
    echo "[stop] Server force-killed."
else
    echo "[stop] Server PID $SERVER_PID is not running."
fi

rm -f "$PIDFILE"

# Also clean up any remaining mlx_lm_server on port 8080
REMAINING=$(lsof -ti :8080 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "[stop] Cleaning up remaining processes on port 8080: $REMAINING"
    echo "$REMAINING" | xargs kill -9 2>/dev/null || true
fi

echo "[stop] Done."
