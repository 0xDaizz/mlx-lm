"use client";

import type { TimeSeriesPoint } from "@/lib/types";

interface SparklineProps {
  data: TimeSeriesPoint[];
  width?: number;
  height?: number;
  color?: string;
  fillOpacity?: number;
}

export default function Sparkline({
  data,
  width = 100,
  height = 32,
  color = "#06b6d4",
  fillOpacity = 0.15,
}: SparklineProps) {
  if (data.length < 2) return <div style={{ width, height }} />;

  const values = data.map((d) => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((d.value - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(" ");

  const fillPoints = `0,${height} ${points} ${width},${height}`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polygon points={fillPoints} fill={color} opacity={fillOpacity} />
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}
