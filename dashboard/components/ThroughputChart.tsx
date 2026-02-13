"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { TimeSeriesPoint } from "@/lib/types";

interface Props {
  data: TimeSeriesPoint[];
  title: string;
  color?: string;
  unit?: string;
}

export default function ThroughputChart({ data, title, color = "#06b6d4", unit = "tok/s" }: Props) {
  const chartData = data.map((d) => ({
    time: Math.floor(d.time),
    value: Number(d.value.toFixed(1)),
  }));

  return (
    <div className="card glow-cyan p-4">
      <h3 className="text-xs text-zinc-500 uppercase tracking-wider mb-3">{title}</h3>
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(63, 63, 70, 0.5)" />
            <XAxis
              dataKey="time"
              stroke="#71717a"
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => `${v}s`}
            />
            <YAxis
              stroke="#71717a"
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => `${v}`}
              width={40}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid rgba(6, 182, 212, 0.2)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value) => [`${value} ${unit}`, title]}
              labelFormatter={(label) => `${label}s`}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: color }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
