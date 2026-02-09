#!/usr/bin/env bash
# launch_distributed.sh — Convenience wrapper for distributed mlx-lm-server
#
# Usage:
#   ./scripts/launch_distributed.sh ring hosts.json [server args...]
#   ./scripts/launch_distributed.sh jaccl hosts.json --ibv-devices /dev/ibv0 --coordinator host:port [server args...]
#
# Examples:
#   ./scripts/launch_distributed.sh ring examples/distributed/hosts-localhost.json --model mlx-community/Qwen3-4B-4bit
#   ./scripts/launch_distributed.sh jaccl examples/distributed/hosts-2node.json --ibv-devices /dev/ibv0 --coordinator 192.168.1.1:9000 --model mlx-community/Qwen3-4B-4bit

set -euo pipefail

usage() {
    echo "Usage: $0 <ring|jaccl> <hostfile> [--ibv-devices PATH] [--coordinator HOST:PORT] [server args...]"
    echo ""
    echo "Backends:"
    echo "  ring   — TCP ring backend (requires hostfile)"
    echo "  jaccl  — RDMA/JACCL backend (requires hostfile, --ibv-devices, --coordinator)"
    echo ""
    echo "Examples:"
    echo "  $0 ring examples/distributed/hosts-localhost.json --model mlx-community/Qwen3-4B-4bit"
    echo "  $0 jaccl hosts.json --ibv-devices /dev/ibv0 --coordinator 192.168.1.1:9000 --model X"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

BACKEND="$1"
HOSTFILE="$2"
shift 2

if [[ "$BACKEND" != "ring" && "$BACKEND" != "jaccl" ]]; then
    echo "Error: backend must be 'ring' or 'jaccl', got '$BACKEND'"
    exit 1
fi

if [[ ! -f "$HOSTFILE" ]]; then
    echo "Error: hostfile not found: $HOSTFILE"
    exit 1
fi

if ! command -v mlx.launch &>/dev/null; then
    echo "Error: mlx.launch not found. Install mlx: pip install mlx"
    exit 1
fi

# Parse optional jaccl flags
IBV_DEVICES=""
COORDINATOR=""
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ibv-devices)
            IBV_DEVICES="$2"
            shift 2
            ;;
        --coordinator)
            COORDINATOR="$2"
            shift 2
            ;;
        *)
            SERVER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$BACKEND" == "jaccl" ]]; then
    if [[ -z "$IBV_DEVICES" || -z "$COORDINATOR" ]]; then
        echo "Error: jaccl backend requires --ibv-devices and --coordinator"
        exit 1
    fi
    export MLX_IBV_DEVICES="$IBV_DEVICES"
    export MLX_JACCL_COORDINATOR="$COORDINATOR"
fi

echo "Launching distributed mlx-lm-server:"
echo "  Backend:  $BACKEND"
echo "  Hostfile: $HOSTFILE"
if [[ "$BACKEND" == "jaccl" ]]; then
    echo "  IBV:      $IBV_DEVICES"
    echo "  Coord:    $COORDINATOR"
fi
echo "  Args:     ${SERVER_ARGS[*]:-<none>}"
echo ""

exec mlx.launch --backend "$BACKEND" --hostfile "$HOSTFILE" -- \
    python -m mlx_lm_server \
    --distributed-mode "$BACKEND" \
    --distributed-hostfile "$HOSTFILE" \
    "${SERVER_ARGS[@]}"
