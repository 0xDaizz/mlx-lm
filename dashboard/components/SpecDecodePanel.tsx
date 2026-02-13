"use client";

import type { ServerMetrics } from "@/lib/types";
import { formatPercent } from "@/lib/format";
import GaugeRing from "./GaugeRing";

interface Props {
  metrics: ServerMetrics | null;
}

export default function SpecDecodePanel({ metrics }: Props) {
  const spec = metrics?.spec_decode;
  if (!spec?.enabled) {
    return (
      <div className="card glow-cyan p-4 opacity-50">
        <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">Speculative Decoding</h3>
        <p className="text-sm text-zinc-600">Disabled</p>
      </div>
    );
  }

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">Speculative Decoding</h3>
      <div className="flex items-center gap-4">
        <GaugeRing value={spec.acceptance_rate_ema} size={90} color="#10b981">
          <span className="font-mono-metric text-base font-semibold text-emerald-400">
            {formatPercent(spec.acceptance_rate_ema)}
          </span>
        </GaugeRing>
        <div className="flex-1 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-zinc-500">Mode</span>
            <span className="font-mono-metric text-zinc-300 uppercase">{spec.mode}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Accept (EMA)</span>
            <span className="font-mono-metric text-emerald-400">{formatPercent(spec.acceptance_rate_ema)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Tok/Step</span>
            <span className="font-mono-metric text-zinc-300">{spec.tokens_per_step.toFixed(1)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">k</span>
            <span className="font-mono-metric text-zinc-300">
              {spec.current_k} {spec.adaptive_k !== spec.current_k ? `\u2192 ${spec.adaptive_k}` : ""}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
