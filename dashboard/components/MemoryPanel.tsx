"use client";

import type { ServerMetrics } from "@/lib/types";
import { formatBytes, formatPercent } from "@/lib/format";

interface Props {
  metrics: ServerMetrics | null;
}

export default function MemoryPanel({ metrics }: Props) {
  const mem = metrics?.memory;
  const pressure = mem?.pressure ?? 0;
  const barColor = pressure > 0.9 ? "bg-red-500" : pressure > 0.7 ? "bg-amber-500" : "bg-cyan-500";

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">Metal GPU Memory</h3>
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-zinc-500">Active</span>
            <span className="font-mono-metric text-zinc-300">{mem ? formatBytes(mem.active_bytes) : "--"}</span>
          </div>
          <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${barColor} rounded-full transition-all duration-500`}
              style={{ width: `${pressure * 100}%` }}
            />
          </div>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-zinc-500">Peak</span>
          <span className="font-mono-metric text-zinc-300">{mem ? formatBytes(mem.peak_bytes) : "--"}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-zinc-500">Pressure</span>
          <span
            className={`font-mono-metric font-medium ${
              pressure > 0.9 ? "text-red-400" : pressure > 0.7 ? "text-amber-400" : "text-cyan-400"
            }`}
          >
            {mem ? formatPercent(pressure) : "--"}
          </span>
        </div>
      </div>
    </div>
  );
}
