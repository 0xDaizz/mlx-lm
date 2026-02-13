"use client";

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { TimeSeriesPoint } from "@/lib/types";

interface Props {
  data: TimeSeriesPoint[];
}

export default function RequestChart({ data }: Props) {
  const chartData = data.map((d) => ({
    time: Math.floor(d.time),
    value: d.value,
  }));

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">Active Requests</h3>
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(63, 63, 70, 0.5)" />
            <XAxis dataKey="time" stroke="#71717a" tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}s`} />
            <YAxis stroke="#71717a" tick={{ fontSize: 10 }} width={30} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid rgba(139, 92, 246, 0.2)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value) => [`${value}`, "Active"]}
              labelFormatter={(label) => `${label}s`}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#8b5cf6"
              fill="#8b5cf6"
              fillOpacity={0.15}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
