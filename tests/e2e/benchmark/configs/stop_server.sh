#!/usr/bin/env bash
# stop_server.sh — Gracefully stop the benchmark server (2-node JACCL TP)
#
# Process topology:
#   mlx.launch (PID in pidfile)
#   ├── python rank 0 (local, port 8080, holds ~310GB wired Metal memory)
#   └── python rank 1 (hwStudio2 via SSH, holds ~310GB wired Metal memory)
#
# mlx.launch has NO SIGTERM handler — it dies instantly without propagating
# to children. So we must explicitly kill all server processes ourselves.
#
# Strategy:
#   1. Send SIGTERM to all server processes (local + remote)
#   2. Wait up to 40s for graceful Metal cleanup (_cleanup_metal takes ~30s max)
#   3. SIGKILL only as last resort
#
# Usage:
#   stop_server.sh [--remote-host HOST] [--hostfile PATH]
set -euo pipefail

PIDFILE="/tmp/kimi-bench-server.pid"
REMOTE_HOST=""
HOSTFILE=""
GRACE_PERIOD=40   # seconds — must exceed server's internal cleanup_timer (30s)
PROCESS_PATTERN="server_wrapper"  # match actual process name, not module

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-host) REMOTE_HOST="$2"; shift 2 ;;
        --hostfile) HOSTFILE="$2"; shift 2 ;;
        *) echo "[stop] Unknown argument: $1"; shift ;;
    esac
done

# ── Resolve remote host ──
if [ -z "$REMOTE_HOST" ]; then
    # Try hostfile (default path if not specified)
    if [ -z "$HOSTFILE" ]; then
        HOSTFILE="/Users/hw/mlx-lm-server/examples/distributed/hosts-hwstudio.json"
    fi
    if [ -f "$HOSTFILE" ]; then
        # Try SSH hostname first, then fall back to IP
        REMOTE_HOST=$(python3 -c "
import json, sys
try:
    with open('$HOSTFILE') as f:
        hf = json.load(f)
    hosts = hf.get('hosts', [])
    if len(hosts) >= 2:
        print(hosts[1].get('ssh', '') or hosts[1].get('ips', [''])[0])
except Exception:
    pass
" 2>/dev/null)
    fi
fi

# Last resort: try common names
if [ -z "$REMOTE_HOST" ]; then
    for candidate in hwstudio2 hwStudio2.local 192.168.0.107; do
        if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no "$candidate" "true" 2>/dev/null; then
            REMOTE_HOST="$candidate"
            break
        fi
    done
fi

if [ -z "$REMOTE_HOST" ]; then
    echo "[stop] WARNING: Could not determine remote host. Only stopping local processes."
fi

echo "[stop] Remote host: ${REMOTE_HOST:-<none>}"

# ── Helper: kill local server processes on port 8080 ──
kill_local_server() {
    local sig="${1:-TERM}"
    local pids
    pids=$(lsof -ti :8080 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[stop] Local port 8080 processes: $pids (SIG$sig)"
        echo "$pids" | xargs kill -"$sig" 2>/dev/null || true
        return 0
    fi
    return 1
}

# ── Helper: kill remote server processes ──
kill_remote_server() {
    if [ -z "$REMOTE_HOST" ]; then return 0; fi
    local sig="${1:-TERM}"
    echo "[stop] Remote ($REMOTE_HOST): sending SIG$sig to $PROCESS_PATTERN processes ..."
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$REMOTE_HOST" \
        "pgrep -f '$PROCESS_PATTERN' | xargs kill -$sig 2>/dev/null" \
        2>/dev/null || true
}

# ── 1. Kill mlx.launch parent (if pidfile exists) ──
if [ -f "$PIDFILE" ]; then
    SERVER_PID=$(cat "$PIDFILE")
    echo "[stop] Killing mlx.launch parent PID: $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    rm -f "$PIDFILE"
fi

# ── 2. Send SIGTERM to actual server processes (local + remote) ──
echo "[stop] Sending SIGTERM to all server processes ..."
kill_local_server "TERM" || true
kill_remote_server "TERM"

# ── 3. Wait for graceful shutdown (Metal cleanup needs up to 30s) ──
echo "[stop] Waiting up to ${GRACE_PERIOD}s for graceful Metal cleanup ..."
ELAPSED=0
while [ $ELAPSED -lt $GRACE_PERIOD ]; do
    LOCAL_ALIVE=$(lsof -ti :8080 2>/dev/null || true)
    REMOTE_ALIVE=""
    REMOTE_REACHABLE=true
    if [ -n "$REMOTE_HOST" ]; then
        REMOTE_ALIVE=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$REMOTE_HOST" \
            "pgrep -f '$PROCESS_PATTERN'" 2>/dev/null) || REMOTE_REACHABLE=false
    fi

    if [ -z "$LOCAL_ALIVE" ] && [ "$REMOTE_REACHABLE" = true ] && [ -z "$REMOTE_ALIVE" ]; then
        echo "[stop] All server processes exited gracefully after ${ELAPSED}s"
        echo "[stop] Done."
        exit 0
    fi

    if [ "$REMOTE_REACHABLE" = false ] && [ $((ELAPSED % 10)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
        echo "[stop] WARNING: Cannot reach $REMOTE_HOST — remote status unknown"
    fi

    # Progress every 10s
    if [ $((ELAPSED % 10)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
        echo "[stop] Still waiting ... (${ELAPSED}s elapsed, local: ${LOCAL_ALIVE:-none}, remote: ${REMOTE_ALIVE:-none})"
    fi

    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# ── 4. SIGKILL as last resort ──
echo "[stop] Grace period expired. Force-killing remaining processes ..."
kill_local_server "KILL" || true
kill_remote_server "KILL"
sleep 2

# ── 5. Final verification ──
REMAINING_LOCAL=$(lsof -ti :8080 2>/dev/null || true)
if [ -n "$REMAINING_LOCAL" ]; then
    echo "[stop] WARNING: Local processes still alive after SIGKILL: $REMAINING_LOCAL"
else
    echo "[stop] Local: clean"
fi

if [ -n "$REMOTE_HOST" ]; then
    REMAINING_REMOTE=""
    VERIFY_REACHABLE=true
    REMAINING_REMOTE=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$REMOTE_HOST" \
        "pgrep -f '$PROCESS_PATTERN'" 2>/dev/null) || VERIFY_REACHABLE=false
    if [ "$VERIFY_REACHABLE" = false ]; then
        echo "[stop] WARNING: Cannot reach $REMOTE_HOST — remote status unknown (may need manual check)"
    elif [ -n "$REMAINING_REMOTE" ]; then
        echo "[stop] WARNING: Remote processes still alive after SIGKILL: $REMAINING_REMOTE"
    else
        echo "[stop] Remote: clean"
    fi
fi

echo "[stop] Done."
