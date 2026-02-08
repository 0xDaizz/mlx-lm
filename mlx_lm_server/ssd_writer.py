"""Async SSD writer thread with inflight counter + polling shutdown.

Provides non-blocking SSD writes from the inference thread via a bounded queue.
Three-level backpressure: put_nowait → put(50ms) → sync fallback.
Deterministic shutdown via inflight counter + polling-based drain.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_lm_server.ssd_cache import SSDCache

logger = logging.getLogger(__name__)


class SSDWriterThread:
    """Background thread for async SSD writes with graceful shutdown.

    Concurrency protection:
    - _life_lock: enqueue/stop lifecycle synchronization
    - _pending_lock: producer↔worker _pending set synchronization
    - _stats_lock: producer↔worker _stats dict synchronization

    Shutdown protocol:
    1. stop() sets _closing=True under _life_lock
    2. Waits for all in-flight enqueues to complete (inflight counter)
    3. Best-effort sentinel enqueue (fast exit hint)
    4. Worker polls queue with timeout; exits on _closing + drained OR sentinel
    5. stop() joins the thread
    """

    def __init__(
        self,
        ssd: SSDCache,
        queue_size: int = 512,
        durability: str = "best_effort",
        max_retries: int = 3,
    ) -> None:
        self._ssd = ssd
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._durability = durability
        self._max_retries = max_retries

        # Lifecycle
        self._sentinel = object()
        self._closing = False
        self._inflight_enqueues = 0
        self._life_lock = threading.Lock()
        self._no_inflight = threading.Condition(self._life_lock)
        # Stats
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "writer_enqueue_total": 0,
            "writer_enqueue_dedup_skip": 0,
            "writer_save_success": 0,
            "writer_save_fail": 0,
            "writer_queue_full_sync_fallback": 0,
            "writer_retry_attempts": 0,
            "writer_retry_final_fail": 0,
        }

        self._thread = threading.Thread(
            target=self._run, daemon=False, name="ssd-writer"
        )
        self._thread.start()

    def _inc(self, key: str, n: int = 1) -> None:
        """Thread-safe stat increment."""
        with self._stats_lock:
            self._stats[key] += n

    def _save_with_durability(self, block_hash: str, kv_data, num_tokens) -> str:
        """Save with optional persistent retry. Returns: "saved"|"dedup"|"collision"|"error"."""
        result = self._ssd.save_block(block_hash, kv_data, num_tokens)
        if result != "error":
            return result
        if self._durability != "persistent":
            return "error"
        for _ in range(self._max_retries):
            self._inc("writer_retry_attempts")
            retry_result = self._ssd.save_block(block_hash, kv_data, num_tokens)
            if retry_result != "error":
                return retry_result
        self._inc("writer_retry_final_fail")
        return "error"

    def _record_save_result(self, result: str, block_hash: str) -> None:
        """Record stats for a save result."""
        if result == "saved":
            self._inc("writer_save_success")
        elif result == "error":
            self._inc("writer_save_fail")
            logger.error("SSD writer save failed: %s", block_hash)
        # "dedup" and "collision" are not failures — no action needed

    def _run(self) -> None:
        """Worker loop: process queue items until closing + drained."""
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._life_lock:
                    if self._closing and self._inflight_enqueues == 0 and self._queue.empty():
                        break
                continue
            if item is self._sentinel:
                break
            block_hash, kv_data, num_tokens = item
            try:
                result = self._save_with_durability(block_hash, kv_data, num_tokens)
                self._record_save_result(result, block_hash)
            except Exception:
                self._inc("writer_save_fail")
                logger.exception("SSD writer unexpected error: %s", block_hash)
            finally:
                with self._pending_lock:
                    self._pending.discard(block_hash)

    def enqueue(
        self, block_hash: str, kv_data, num_tokens: int | None = None
    ) -> str:
        """Enqueue a block for async SSD write.

        Returns:
            "queued" if successfully enqueued or sync-fallback completed,
            "dedup" if block is already pending,
            "closing" if writer is shutting down.
        """
        with self._life_lock:
            if self._closing:
                return "closing"
            self._inflight_enqueues += 1
        try:
            with self._pending_lock:
                if block_hash in self._pending:
                    self._inc("writer_enqueue_dedup_skip")
                    return "dedup"
                self._pending.add(block_hash)
            self._inc("writer_enqueue_total")

            # Level 0: non-blocking
            try:
                self._queue.put_nowait((block_hash, kv_data, num_tokens))
                return "queued"
            except queue.Full:
                pass
            # Level 1: short wait (50ms)
            try:
                self._queue.put((block_hash, kv_data, num_tokens), timeout=0.05)
                return "queued"
            except queue.Full:
                pass
            # Level 2: sync fallback
            with self._pending_lock:
                self._pending.discard(block_hash)
            self._inc("writer_queue_full_sync_fallback")
            try:
                result = self._save_with_durability(block_hash, kv_data, num_tokens)
                self._record_save_result(result, block_hash)
            except Exception:
                self._inc("writer_save_fail")
            return "queued"
        finally:
            with self._life_lock:
                self._inflight_enqueues -= 1
                if self._closing and self._inflight_enqueues == 0:
                    self._no_inflight.notify_all()

    def stop(self, drain_timeout: float = 5.0) -> bool:
        """Graceful shutdown. Sets closing flag, waits for drain, joins thread.

        The worker loop detects _closing + empty queue and exits without
        needing a sentinel. Sentinel is still attempted as a fast-exit hint.

        Returns True if shutdown completed cleanly.
        """
        deadline = time.monotonic() + drain_timeout
        with self._life_lock:
            self._closing = True
            while self._inflight_enqueues > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error("Timed out waiting for in-flight enqueues")
                    break  # Don't return False — still try to join
                self._no_inflight.wait(timeout=remaining)

        # Best-effort sentinel (fast exit hint for worker)
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            pass  # Worker will exit via polling check

        remaining = max(0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)
        ok = not self._thread.is_alive()
        if not ok:
            logger.error(
                "SSD writer thread did not exit within %ss, ~%d items may be lost",
                drain_timeout, self._queue.qsize(),
            )
        return ok

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of writer statistics."""
        with self._stats_lock:
            return dict(self._stats)

    @property
    def pending_count(self) -> int:
        """Number of blocks currently pending write."""
        with self._pending_lock:
            return len(self._pending)

    @property
    def queue_size(self) -> int:
        """Current number of items in the write queue."""
        return self._queue.qsize()
