"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { ServerMetrics, ServerConfig, TimeSeriesPoint } from "@/lib/types";

const MAX_HISTORY = 60; // 60 seconds of history

export function useMetricsStream() {
  const [metrics, setMetrics] = useState<ServerMetrics | null>(null);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [connected, setConnected] = useState(false);
  const [throughputHistory, setThroughputHistory] = useState<TimeSeriesPoint[]>([]);
  const [requestHistory, setRequestHistory] = useState<TimeSeriesPoint[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);
  const startTimeRef = useRef<number>(Date.now());

  // Fetch config once
  useEffect(() => {
    const base = window.location.origin;
    fetch(`${base}/dashboard/api/config`)
      .then((res) => res.json())
      .then(setConfig)
      .catch(console.error);
  }, []);

  // SSE connection
  useEffect(() => {
    const base = window.location.origin;
    const es = new EventSource(`${base}/dashboard/api/stats`);
    eventSourceRef.current = es;

    es.onopen = () => setConnected(true);

    es.onmessage = (event) => {
      try {
        const data: ServerMetrics = JSON.parse(event.data);
        setMetrics(data);

        const elapsed = (Date.now() - startTimeRef.current) / 1000;

        setThroughputHistory((prev) => {
          const next = [...prev, { time: elapsed, value: data.throughput.tokens_per_sec }];
          return next.slice(-MAX_HISTORY);
        });

        setRequestHistory((prev) => {
          const next = [...prev, { time: elapsed, value: data.server.active_sequences }];
          return next.slice(-MAX_HISTORY);
        });
      } catch (e) {
        console.error("Failed to parse SSE data:", e);
      }
    };

    es.onerror = () => {
      setConnected(false);
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, []);

  return { metrics, config, connected, throughputHistory, requestHistory };
}
