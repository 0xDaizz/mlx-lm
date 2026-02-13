#!/usr/bin/env bash
# post_reboot_setup.sh — LOCAL-only post-reboot setup for a single node
# Auto-detects hwStudio1 or hwStudio2 by hostname
# Usage: sudo bash /Users/hw/mlx-lm-server/post_reboot_setup.sh

set -o pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; ERRORS=1; }

ERRORS=0

# --- Root check ---
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}ERROR: must run as root (sudo bash $0)${NC}"
    exit 1
fi

# --- Detect node ---
HOSTNAME=$(hostname -s)
echo "Hostname: ${HOSTNAME}"

case "$HOSTNAME" in
    hwStudio1)
        EN3_IP="10.10.0.1"
        EN5_IP="10.10.1.1"
        PING_EN3="10.10.0.2"
        PING_EN5="10.10.1.2"
        ;;
    hwStudio2)
        EN3_IP="10.10.0.2"
        EN5_IP="10.10.1.2"
        PING_EN3="10.10.0.1"
        PING_EN5="10.10.1.1"
        ;;
    *)
        echo -e "${RED}ERROR: unknown hostname '${HOSTNAME}' (expected hwStudio1 or hwStudio2)${NC}"
        exit 1
        ;;
esac
echo "Role: en3=${EN3_IP}/30, en5=${EN5_IP}/30"
echo ""

# --- 1. Wired memory ---
echo "1. Wired memory"
WIRED_PAGES=$(vm_stat | awk '/Pages wired down/ {gsub(/\./,"",$NF); print $NF}')
WIRED_MB=$(( WIRED_PAGES * 16384 / 1048576 ))
echo "  ${WIRED_MB} MB (${WIRED_PAGES} pages)"

# --- 2. iogpu wired limits ---
echo "2. Setting iogpu wired limits"
if bash /Users/hw/mlx-lm-server/configure_mlx.sh 2>&1 | sed 's/^/  /'; then
    ok "configure_mlx.sh"
else
    fail "configure_mlx.sh"
fi

# --- 3. Tear down bridge0 ---
echo "3. Tearing down bridge0"
ifconfig bridge0 down 2>/dev/null
ifconfig bridge0 deletem en3 2>/dev/null
ifconfig bridge0 deletem en5 2>/dev/null
ok "bridge0 down + deletem en3/en5"

# --- 4. Assign Thunderbolt IPs ---
echo "4. Assigning Thunderbolt IPs"
if ifconfig en3 ${EN3_IP}/30 up 2>&1; then
    ok "en3 = ${EN3_IP}/30"
else
    fail "en3 = ${EN3_IP}/30"
fi
if ifconfig en5 ${EN5_IP}/30 up 2>&1; then
    ok "en5 = ${EN5_IP}/30"
else
    fail "en5 = ${EN5_IP}/30"
fi

# --- 5. Ping other node ---
echo "5. Pinging other node (2s wait)"
sleep 2
if ping -c 1 -W 2 ${PING_EN3} >/dev/null 2>&1; then
    ok "ping ${PING_EN3} (en3)"
else
    fail "ping ${PING_EN3} (en3) — run this script on the other node first"
fi
if ping -c 1 -W 2 ${PING_EN5} >/dev/null 2>&1; then
    ok "ping ${PING_EN5} (en5)"
else
    fail "ping ${PING_EN5} (en5) — run this script on the other node first"
fi

# --- Summary ---
echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}Done — all steps passed.${NC}"
else
    echo -e "${RED}Done — some steps failed (see above).${NC}"
fi
