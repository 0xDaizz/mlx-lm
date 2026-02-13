"use client";

import type { ServerMetrics } from "@/lib/types";
import { formatPercent, formatNumber } from "@/lib/format";
import GaugeRing from "./GaugeRing";

interface Props {
  metrics: ServerMetrics | null;
}

export default function KVCachePanel({ metrics }: Props) {
  const kv = metrics?.kv_cache;
  const utilization = kv ? kv.used_blocks / (kv.total_blocks || 1) : 0;
  const gaugeColor = utilization > 0.9 ? "#ef4444" : utilization > 0.7 ? "#f59e0b" : "#06b6d4";

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">KV Cache</h3>
      <div className="flex items-center gap-4">
        <GaugeRing value={utilization} size={100} color={gaugeColor}>
          <span className="font-mono-metric text-lg font-semibold" style={{ color: gaugeColor }}>
            {formatPercent(utilization)}
          </span>
        </GaugeRing>
        <div className="flex-1 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-zinc-500">Used</span>
            <span className="font-mono-metric text-zinc-300">{kv ? formatNumber(kv.used_blocks) : "--"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Free</span>
            <span className="font-mono-metric text-zinc-300">{kv ? formatNumber(kv.free_blocks) : "--"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Cached</span>
            <span className="font-mono-metric text-zinc-300">{kv ? formatNumber(kv.cached_blocks) : "--"}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Hit Rate</span>
            <span className="font-mono-metric text-amber-400">{kv ? formatPercent(kv.hit_rate) : "--"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
