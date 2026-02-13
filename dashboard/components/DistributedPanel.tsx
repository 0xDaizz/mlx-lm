"use client";

import type { ServerMetrics } from "@/lib/types";

interface Props {
  metrics: ServerMetrics | null;
}

export default function DistributedPanel({ metrics }: Props) {
  const dist = metrics?.distributed;
  if (!dist?.enabled) return null;

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">Distributed Cluster</h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
        <div>
          <p className="text-zinc-500">Rank</p>
          <p className="font-mono-metric text-lg text-zinc-200">
            {dist.rank} <span className="text-zinc-600">/ {dist.world_size}</span>
          </p>
        </div>
        <div>
          <p className="text-zinc-500">Backend</p>
          <p className="font-mono-metric text-lg text-zinc-200 uppercase">{dist.backend}</p>
        </div>
        <div>
          <p className="text-zinc-500">Health</p>
          <p className={`text-lg font-medium ${dist.healthy ? "text-emerald-400" : "text-red-400"}`}>
            {dist.healthy ? "Healthy" : "Unhealthy"}
          </p>
        </div>
        <div>
          <p className="text-zinc-500">Bus Errors</p>
          <p className={`font-mono-metric text-lg ${dist.bus_error_count > 0 ? "text-red-400" : "text-zinc-200"}`}>
            {dist.bus_error_count}
          </p>
        </div>
      </div>
    </div>
  );
}
