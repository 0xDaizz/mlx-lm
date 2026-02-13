#!/usr/bin/env bash
# Get total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))
# wired_limit_mb = max(80% RAM, RAM-5GB)
EIGHTY_PERCENT=$(($TOTAL_MEM_MB * 80 / 100))
MINUS_5GB=$(($TOTAL_MEM_MB - 5120))
if [ $EIGHTY_PERCENT -gt $MINUS_5GB ]; then
  WIRED_LIMIT_MB=$EIGHTY_PERCENT
else
  WIRED_LIMIT_MB=$MINUS_5GB
fi
# wired_lwm_mb = max(70% RAM, RAM-8GB)
SEVENTY_PERCENT=$(($TOTAL_MEM_MB * 70 / 100))
MINUS_8GB=$(($TOTAL_MEM_MB - 8192))
if [ $SEVENTY_PERCENT -gt $MINUS_8GB ]; then
  WIRED_LWM_MB=$SEVENTY_PERCENT
else
  WIRED_LWM_MB=$MINUS_8GB
fi
echo "Total memory: $TOTAL_MEM_MB MB"
echo "Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
echo "Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB"
if [ "$EUID" -eq 0 ]; then
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
else
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
fi
