"use client";

import type { ServerMetrics } from "@/lib/types";
import type { TimeSeriesPoint } from "@/lib/types";
import { formatNumber, formatPercent } from "@/lib/format";
import Sparkline from "./Sparkline";

interface Props {
  metrics: ServerMetrics | null;
  throughputHistory: TimeSeriesPoint[];
}

interface MetricCardProps {
  label: string;
  value: string;
  subValue?: string;
  color: string;
  sparkData?: TimeSeriesPoint[];
  sparkColor?: string;
}

function MetricCard({ label, value, subValue, color, sparkData, sparkColor }: MetricCardProps) {
  return (
    <div className="card glow-cyan p-4 flex flex-col justify-between min-h-[100px]">
      <p className="text-xs text-zinc-500 uppercase tracking-wider">{label}</p>
      <div className="flex items-end justify-between mt-2">
        <div>
          <p className={`font-mono-metric text-2xl font-semibold`} style={{ color }}>
            {value}
          </p>
          {subValue && <p className="text-xs text-zinc-500 mt-0.5">{subValue}</p>}
        </div>
        {sparkData && sparkData.length > 1 && (
          <Sparkline data={sparkData} width={80} height={28} color={sparkColor || color} />
        )}
      </div>
    </div>
  );
}

export default function MetricsGrid({ metrics, throughputHistory }: Props) {
  const m = metrics;
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <MetricCard
        label="Active Requests"
        value={m ? String(m.server.active_sequences) : "--"}
        subValue={m ? `/ ${m.server.max_concurrent || "\u221e"}` : undefined}
        color="#06b6d4"
      />
      <MetricCard
        label="Queued"
        value={m ? String(m.server.queued_requests) : "--"}
        subValue={m ? `/ ${m.server.max_queue}` : undefined}
        color="#8b5cf6"
      />
      <MetricCard
        label="Tokens/sec"
        value={m ? m.throughput.tokens_per_sec.toFixed(1) : "--"}
        subValue={m ? `${formatNumber(m.throughput.prefill_tokens)} prefill` : undefined}
        color="#10b981"
        sparkData={throughputHistory}
        sparkColor="#10b981"
      />
      <MetricCard
        label="Cache Hit Rate"
        value={m ? formatPercent(m.kv_cache.hit_rate) : "--"}
        subValue={m ? `Eff: ${formatPercent(m.kv_cache.effectiveness)}` : undefined}
        color="#f59e0b"
      />
    </div>
  );
}
