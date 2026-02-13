"use client";

import type { ServerMetrics, ServerConfig } from "@/lib/types";
import { formatDuration } from "@/lib/format";

interface Props {
  metrics: ServerMetrics | null;
  config: ServerConfig | null;
  connected: boolean;
}

export default function ServerHeader({ metrics, config, connected }: Props) {
  return (
    <header className="card glow-cyan p-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        {/* Status indicator */}
        <div className="relative">
          <div
            className={`w-3 h-3 rounded-full ${connected ? "bg-emerald-400 animate-pulse-glow" : "bg-zinc-600"}`}
          />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-zinc-100 tracking-tight">
            {config?.model_name || "Loading..."}
          </h1>
          <p className="text-xs text-zinc-500">
            mlx-lm-server {config?.distributed_mode !== "off" ? `| ${config?.distributed_mode?.toUpperCase()}` : ""}
            {config?.spec_decode_mode !== "none" ? ` | Spec: ${config?.spec_decode_mode}` : ""}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-6">
        <div className="text-right">
          <p className="text-xs text-zinc-500">Uptime</p>
          <p className="font-mono-metric text-sm text-zinc-300">
            {metrics ? formatDuration(metrics.uptime_s) : "--"}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-zinc-500">Status</p>
          <p className={`text-sm font-medium ${connected ? "text-emerald-400" : "text-zinc-600"}`}>
            {connected ? "Connected" : "Disconnected"}
          </p>
        </div>
      </div>
    </header>
  );
}
