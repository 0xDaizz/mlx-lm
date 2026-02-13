"use client";

import type { ServerMetrics } from "@/lib/types";
import { formatBytes, formatNumber } from "@/lib/format";

interface Props {
  metrics: ServerMetrics | null;
}

export default function SSDCachePanel({ metrics }: Props) {
  const ssd = metrics?.ssd_cache;
  if (!ssd?.enabled) {
    return (
      <div className="card glow-cyan p-4 opacity-50">
        <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">SSD Cache</h3>
        <p className="text-sm text-zinc-600">Disabled</p>
      </div>
    );
  }

  const usage = ssd.max_size_bytes > 0 ? ssd.total_bytes / ssd.max_size_bytes : 0;

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">SSD Cache</h3>
      <div className="space-y-2 text-sm">
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-zinc-500">Disk Usage</span>
            <span className="font-mono-metric text-zinc-300">
              {formatBytes(ssd.total_bytes)}
              {ssd.max_size_bytes > 0 ? ` / ${formatBytes(ssd.max_size_bytes)}` : ""}
            </span>
          </div>
          {ssd.max_size_bytes > 0 && (
            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(usage * 100, 100)}%` }}
              />
            </div>
          )}
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-500">Saves</span>
          <span className="font-mono-metric text-emerald-400">{formatNumber(ssd.save_success)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-500">Failures</span>
          <span className={`font-mono-metric ${ssd.save_fail > 0 ? "text-red-400" : "text-zinc-600"}`}>
            {ssd.save_fail}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-500">LRU Pruned</span>
          <span className="font-mono-metric text-zinc-300">{formatNumber(ssd.lru_prune_count)}</span>
        </div>
      </div>
    </div>
  );
}
