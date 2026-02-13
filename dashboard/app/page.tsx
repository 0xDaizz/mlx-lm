"use client";

import { useMetricsStream } from "@/hooks/useMetricsStream";
import ServerHeader from "@/components/ServerHeader";
import MetricsGrid from "@/components/MetricsGrid";
import ThroughputChart from "@/components/ThroughputChart";
import RequestChart from "@/components/RequestChart";
import KVCachePanel from "@/components/KVCachePanel";
import MemoryPanel from "@/components/MemoryPanel";
import SSDCachePanel from "@/components/SSDCachePanel";
import SpecDecodePanel from "@/components/SpecDecodePanel";
import DistributedPanel from "@/components/DistributedPanel";

export default function DashboardPage() {
  const { metrics, config, connected, throughputHistory, requestHistory } = useMetricsStream();

  return (
    <main className="max-w-7xl mx-auto px-4 py-6 space-y-4">
      <ServerHeader metrics={metrics} config={config} connected={connected} />
      <MetricsGrid metrics={metrics} throughputHistory={throughputHistory} />

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ThroughputChart data={throughputHistory} title="Token Throughput" color="#10b981" unit="tok/s" />
        <RequestChart data={requestHistory} />
      </div>

      {/* Panels grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <KVCachePanel metrics={metrics} />
        <MemoryPanel metrics={metrics} />
        <SSDCachePanel metrics={metrics} />
        <SpecDecodePanel metrics={metrics} />
      </div>

      {/* Distributed (conditional) */}
      <DistributedPanel metrics={metrics} />
    </main>
  );
}
