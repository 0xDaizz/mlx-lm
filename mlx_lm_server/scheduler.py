"""Continuous Batching Scheduler for mlx-lm-server.

Implements iteration-level scheduling with:
- Thread-safe request queue (FIFO)
- Sequence lifecycle management (init, prefill, decode, finish)
- Streaming token delivery via per-request queues
- Stop sequence / EOS / max_tokens detection
- Request cancellation
- Background inference loop

The scheduler is the bridge between the FastAPI server (Phase 3) and the
mlx-lm engine. It accepts InferenceRequests, manages their lifecycle,
and delivers TokenEvents back to callers.

Design: The scheduler works without a real model for testing.
Pass model=None and tokenizer=None for mock-based tests.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any
from mlx_lm_server.config import ServerConfig
from mlx_lm_server.types import (
    InferenceRequest,
    SchedulerOutputs,
    SequenceState,
    TokenEvent,
)

import copy

from mlx_lm_server.kv_cache_manager import (
    decompose_cache_to_blocks,
    reconstruct_cache_from_blocks,
)
from mlx_lm_server.sequence_cache import SequenceCacheStore

try:
    from mlx_lm.generate import BatchGenerator, stream_generate
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    BatchGenerator = None
    stream_generate = None
    make_prompt_cache = None
    trim_prompt_cache = None
    can_trim_prompt_cache = None
    make_sampler = None

logger = logging.getLogger(__name__)

BUS_ERROR_THRESHOLD = 10
BUS_OUTBOX_MAXSIZE = 1024
BUS_OUTBOX_CONTROL_RESERVE = 64
BUS_SHUTDOWN_ENQUEUE_TIMEOUT_S = 2.0


class RequestQueue:
    """Thread-safe FIFO queue for incoming inference requests.

    Uses a threading.Lock and collections.deque for O(1) append/popleft.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._lock = threading.Lock()
        self._queue: deque[InferenceRequest] = deque()
        self._max_size = max_size

    def add(self, request: InferenceRequest) -> None:
        """Add a request to the queue.

        Raises:
            RuntimeError: If the queue is full.
        """
        with self._lock:
            if len(self._queue) >= self._max_size:
                raise RuntimeError(
                    f"Request queue is full (max_size={self._max_size})"
                )
            self._queue.append(request)

    def pop_batch(self, max_size: int) -> list[InferenceRequest]:
        """Pop up to max_size requests from the front of the queue (FIFO)."""
        with self._lock:
            batch: list[InferenceRequest] = []
            for _ in range(min(max_size, len(self._queue))):
                batch.append(self._queue.popleft())
            return batch

    def cancel(self, request_id: str) -> bool:
        """Remove a request from the queue by request_id.

        Returns True if the request was found and removed, False otherwise.
        """
        with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    del self._queue[i]
                    return True
            return False

    @property
    def size(self) -> int:
        """Current number of requests in the queue."""
        with self._lock:
            return len(self._queue)


class Scheduler:
    """Continuous batching scheduler for LLM inference.

    Manages the full lifecycle of inference requests:
    1. Receives requests via submit_request()
    2. Initializes sequences (tokenize, check prefix cache)
    3. Runs prefill for uncached tokens
    4. Runs decode steps to generate tokens
    5. Delivers tokens via streaming queues or blocking get

    The inference loop runs in a background thread, started by
    run_inference_loop(). The server calls submit_request() and
    register_stream() to interact with the scheduler.

    Args:
        config: Server configuration.
        model: The mlx-lm model (or None for testing).
        tokenizer: The tokenizer (or None for testing).
        kv_cache_manager: Optional KVCacheManager for prefix caching.
        ssd_writer: Optional SSDWriterThread for async write-through.
    """

    def __init__(
        self,
        config: ServerConfig,
        model=None,
        tokenizer=None,
        kv_cache_manager=None,
        tiered_cache=None,
        ssd_writer=None,
        dist_ctx=None,
        control_bus=None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache_manager = kv_cache_manager

        # Request queue
        self.request_queue = RequestQueue(max_size=config.max_queue_size)

        # Active sequences keyed by request_id
        self._active_sequences: dict[str, SequenceState] = {}
        self._active_lock = threading.Lock()

        # Token streams: request_id -> Queue[TokenEvent]
        self._streams: dict[str, queue.Queue[TokenEvent]] = {}
        self._streams_lock = threading.Lock()

        # Results for non-streaming requests: request_id -> list[TokenEvent]
        self._results: dict[str, list[TokenEvent]] = {}
        self._results_ready: dict[str, threading.Event] = {}
        self._results_lock = threading.Lock()

        # Cancelled request IDs
        self._cancelled: set[str] = set()
        self._cancelled_lock = threading.Lock()

        # Inference loop control
        self._running = False
        self._new_request_event = threading.Event()
        self._inference_thread: threading.Thread | None = None

        # Mock generation callback for testing (when model is None).
        # Signature: (request_id, prompt_tokens, step) -> (token_id, token_text, finish_reason|None)
        self._mock_generate: Callable[
            [str, list[int], int], tuple[int, str, str | None]
        ] | None = None

        # BatchGenerator state (real model path)
        self._batch_generator: Any = None
        # === Inference-thread-owned mappings ===
        # _request_id_to_uid and _uid_to_request_id are ONLY modified by the
        # inference thread (never from HTTP handlers). They do NOT need lock
        # protection. HTTP handlers access _active_sequences (protected by
        # _active_lock) for cross-thread operations.
        self._uid_to_request_id: dict[int, str] = {}
        self._request_id_to_uid: dict[str, int] = {}
        self._sequence_cache: SequenceCacheStore | None = None

        # Tiered cache (RAM + SSD) — injected via constructor or tests
        self._tiered_cache: Any = tiered_cache
        self._ssd_writer = ssd_writer

        # Distributed context and control bus (tensor parallel)
        self._dist_ctx = dist_ctx
        self._control_bus = control_bus
        # Local queue for events to broadcast (rank0 only, used by inference loop)
        self._bus_outbox: queue.Queue = queue.Queue(
            maxsize=BUS_OUTBOX_MAXSIZE if control_bus is not None else 0
        )

        # Bus error tracking for resilience (U3)
        self._bus_error_count: int = 0
        self._bus_unpack_error_count: int = 0
        self._dist_fatal: bool = False
        self._dist_fatal_reason: str = ""
        # Bus retry events for publish failure ordering (U7)
        self._bus_retry_events: list = []

        # SSD pruning counter (prune every N inference steps)
        self._ssd_prune_interval: int = 1000
        self._ssd_prune_counter: int = 0
        self._ssd_last_prune_time: float = time.time()
        self._ssd_prune_time_interval: float = 3600.0  # Prune at most every 1 hour

        # G2 Prefill Cache Salvage: tracks UIDs with cache misses during insert.
        # When these UIDs are cancelled or timed out, their prefill KV caches
        # are extracted via BatchGenerator.remove(return_prompt_caches=True)
        # and saved to sequence cache + block-level cache, so the prefill
        # computation is not wasted and can be reused by future requests.
        self._pending_cache_saves: set[int] = set()

        # Cache effectiveness counters
        self._stats: dict[str, int] = {
            "cache_hits_sequence": 0,
            "cache_hits_block": 0,
            "cache_misses": 0,
            "requests_completed": 0,
            "requests_errored": 0,
            "tokens_generated": 0,
            "total_prefill_tokens": 0,
            "total_cached_tokens": 0,
            "total_requests": 0,
            "submitted_requests": 0,
            "accepted_requests": 0,
        }
        self._stats_lock = threading.Lock()

        # Shutdown health fields
        self._shutdown_clean: bool = True
        self._shutdown_partial_flush: bool = False
        self.worker_timed_out: bool = False
        self.shutdown_status: str = "clean"
        self._stopped: bool = False

        # Crash recovery: validate SSD index on startup
        if (self._tiered_cache is not None
            and hasattr(self._tiered_cache, 'ssd')
            and self._tiered_cache.ssd is not None
            and hasattr(self._tiered_cache.ssd, 'validate_index')):
            try:
                validation = self._tiered_cache.ssd.validate_index()
                if validation.get("orphans_cleaned", 0) > 0 or validation.get("missing_cleaned", 0) > 0:
                    logger.info(
                        "SSD cache validation: %d orphans cleaned, %d missing entries removed",
                        validation.get("orphans_cleaned", 0),
                        validation.get("missing_cleaned", 0),
                    )
            except Exception as e:
                logger.warning("SSD cache validation failed on startup: %s", e)

        # Initialize BatchGenerator if model is available
        if self.model is not None and BatchGenerator is not None:
            self._create_batch_generator()
            self._sequence_cache = SequenceCacheStore(
                max_entries=config.sequence_cache_size
            )

        if self.model is not None and BatchGenerator is None:
            raise ImportError(
                "model was provided but BatchGenerator failed to import. "
                "Check mlx_lm installation."
            )

    def _inc_stat(self, key: str, value: int = 1) -> None:
        """Thread-safe increment of a stats counter."""
        with self._stats_lock:
            self._stats[key] += value

    def _create_batch_generator(self) -> None:
        """Create or recreate the BatchGenerator instance."""
        if self.model is None or BatchGenerator is None:
            return

        stop_tokens = set()
        if self.tokenizer is not None:
            eos_ids = getattr(self.tokenizer, "eos_token_ids", set())
            if isinstance(eos_ids, (set, frozenset)):
                stop_tokens = set(eos_ids)
            elif isinstance(eos_ids, int):
                stop_tokens = {eos_ids}

        self._batch_generator = BatchGenerator(
            self.model,
            stop_tokens=stop_tokens,
            completion_batch_size=self.config.completion_batch_size,
            prefill_batch_size=self.config.prefill_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            max_kv_size=self.config.max_kv_size,
        )

    # --- Public API (called by server / tests) ---

    def submit_request(self, request: InferenceRequest) -> None:
        """Submit a new inference request to the scheduler.

        In distributed mode (rank0), this method ONLY enqueues a submit event
        to the bus outbox. The inference thread's _drain_bus_outbox() will
        publish the event to all ranks and then do the rank0 local apply
        (request_queue.add + result buffers). This ensures all ranks see
        the same events in the same order, preventing rank divergence.

        In non-distributed mode, the request is added directly to the queue
        and result buffers are created immediately.
        """
        if self._dist_fatal:
            raise RuntimeError("Distributed control plane degraded")

        is_streaming = getattr(request, "stream", False)

        if self._control_bus is not None and self._dist_ctx is not None and self._dist_ctx.is_rank0:
            # ===== DISTRIBUTED MODE: outbox enqueue ONLY =====
            # Do NOT call request_queue.add() here.
            # The inference thread's _drain_bus_outbox() will:
            #   1. publish to all ranks
            #   2. rank0 local apply (add + buffers)
            from mlx_lm_server.distributed_bus import ControlEvent
            from mlx_lm_server.types import BusOutboxFullError

            # Pre-register result buffers for non-streaming requests so get_result()
            # can wait immediately without racing on KeyError before local apply.
            if not is_streaming:
                with self._results_lock:
                    self._results.setdefault(request.request_id, [])
                    self._results_ready.setdefault(request.request_id, threading.Event())

            if self._bus_outbox.qsize() >= BUS_OUTBOX_MAXSIZE - BUS_OUTBOX_CONTROL_RESERVE:
                if not is_streaming:
                    self._cleanup_result_buffers(request.request_id)
                raise BusOutboxFullError(request.request_id)
            try:
                self._bus_outbox.put_nowait(ControlEvent.submit(request))
            except queue.Full:
                if not is_streaming:
                    self._cleanup_result_buffers(request.request_id)
                raise BusOutboxFullError(request.request_id)
            self._inc_stat("submitted_requests")
            self._new_request_event.set()
        else:
            # ===== NON-DISTRIBUTED MODE: direct add =====
            if not is_streaming:
                with self._results_lock:
                    self._results[request.request_id] = []
                    self._results_ready[request.request_id] = threading.Event()
            try:
                self.request_queue.add(request)
            except Exception:
                if not is_streaming:
                    with self._results_lock:
                        self._results.pop(request.request_id, None)
                        self._results_ready.pop(request.request_id, None)
                raise
            self._inc_stat("total_requests")
            self._inc_stat("accepted_requests")
            self._inc_stat("submitted_requests")
            self._new_request_event.set()

        logger.debug("Submitted request %s", request.request_id)

    def register_stream(self, request_id: str) -> queue.Queue[TokenEvent]:
        """Register a streaming token queue for a request.

        Returns a Queue that will receive TokenEvent objects as tokens
        are generated. A TokenEvent with finish_reason != None signals
        the end of generation.
        """
        q: queue.Queue[TokenEvent] = queue.Queue(maxsize=256)
        with self._streams_lock:
            self._streams[request_id] = q
        return q

    def unregister_stream(self, request_id: str) -> None:
        """Remove a previously registered stream queue, if present."""
        with self._streams_lock:
            self._streams.pop(request_id, None)

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        """Wait for and return generation results.

        API Contract:
        - Normal: blocks until finished, returns list[TokenEvent]
        - Timeout: raises TimeoutError — caller MUST call cancel_request() to free resources
        - After cancel: may raise KeyError (already cleaned up) or return [cancelled_event]
        - Recommended pattern: on TimeoutError → cancel_request() → do NOT call get_result() again
        """
        with self._results_lock:
            event = self._results_ready.get(request_id)
        if event is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Request {request_id} timed out after {timeout}s")

        with self._results_lock:
            tokens = self._results.pop(request_id, [])
            self._results_ready.pop(request_id, None)
        return tokens

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request.

        In distributed mode (rank0), this method ONLY enqueues a cancel event
        to the bus outbox. The inference thread's _drain_bus_outbox() will
        publish the event to all ranks and then do the rank0 local apply
        (queue cancel / _cancelled set). This ensures rank-consistent ordering.

        In non-distributed mode, the cancel is applied immediately:
        - If the request is still in the queue, it is removed.
        - If it is already active, it is marked as cancelled.

        Returns True if the request was found (or queued for distributed cancel).

        Lock ordering: _active_lock FIRST, then _cancelled_lock — matches
        schedule_step() to prevent circular-wait deadlock.
        """
        if self._control_bus is not None and self._dist_ctx is not None and self._dist_ctx.is_rank0:
            # ===== DISTRIBUTED MODE: outbox enqueue ONLY =====
            # Do NOT call request_queue.cancel() here.
            # Do NOT manipulate _cancelled directly.
            # The inference thread's _drain_bus_outbox() will do local apply.
            from mlx_lm_server.distributed_bus import ControlEvent
            try:
                self._bus_outbox.put_nowait(ControlEvent.cancel(request_id))
            except queue.Full:
                logger.critical("Bus outbox full even for cancel event — distributed state is degraded")
                self._dist_fatal = True
                self._dist_fatal_reason = "bus_outbox_full_cancel"
                self._running = False
                self._new_request_event.set()
                return False  # Cancel was NOT enqueued
            self._new_request_event.set()
            return True

        # ===== NON-DISTRIBUTED MODE: immediate cancel =====
        # Try removing from queue first (no lock needed for this)
        if self.request_queue.cancel(request_id):
            self._signal_finish(request_id, finish_reason="cancelled")
            return True

        # Use consistent lock ordering: _active_lock FIRST, then _cancelled_lock
        with self._active_lock:
            with self._cancelled_lock:
                self._cancelled.add(request_id)
            if request_id in self._active_sequences:
                return True

        # Request not found in queue or active — clean up any stale buffers
        self._cleanup_result_buffers(request_id)
        with self._streams_lock:
            self._streams.pop(request_id, None)
        return False

    # --- Inference Loop ---

    def run_inference_loop(self, blocking: bool = False) -> None:
        """Start the inference loop.

        If blocking=True, runs in the current thread (for testing).
        Otherwise, starts a background daemon thread.
        """
        self._running = True
        if blocking:
            self._inference_loop()
        else:
            self._inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True,
                name="scheduler-inference-loop",
            )
            self._inference_thread.start()

    def join_worker_loop(self, timeout: float = 300.0) -> None:
        """Block until the inference loop finishes. Used by non-rank0 workers.

        Args:
            timeout: Maximum seconds to wait. If exceeded, sets worker_timed_out=True.
        """
        self.worker_timed_out = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=timeout)
            if self._inference_thread.is_alive():
                logger.critical("Worker inference loop did not exit within %ss.", timeout)
                self.worker_timed_out = True
        else:
            logger.info("Worker loop finished")

    def stop(self) -> None:
        """Stop the inference loop and gracefully shut down all subsystems."""
        if self._stopped:
            return
        self._stopped = True

        # 1. Enqueue shutdown event FIRST (while inference loop is still alive to drain)
        if self._control_bus is not None and self._dist_ctx is not None and self._dist_ctx.is_rank0:
            self._enqueue_shutdown_event()

        # 2. NOW stop the inference loop
        self._running = False
        self._new_request_event.set()

        # 3. Signal active sequences and drain queue
        # NOTE: _signal_finish is safe to call multiple times for the same
        # request_id — it appends events to the result buffer. Callers
        # may receive duplicate finish events during shutdown, which is
        # benign (get_result returns all accumulated events).
        with self._active_lock:
            for rid, seq in list(self._active_sequences.items()):
                if not seq.is_finished:
                    self._signal_finish(rid, finish_reason="error")

        while True:
            reqs = self.request_queue.pop_batch(128)
            if not reqs:
                break
            for req in reqs:
                self._signal_finish(req.request_id, finish_reason="error")

        # 4. Join inference thread
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
            if self._inference_thread.is_alive():
                logger.warning(
                    "Inference thread did not stop within 5s — may be stuck in model inference"
                )
            else:
                self._inference_thread = None

        # Stop async writer thread (must happen BEFORE ssd.flush)
        if self._ssd_writer is not None:
            writer_ok = self._ssd_writer.stop(drain_timeout=5.0)
            if not writer_ok:
                self._shutdown_clean = False
                self.shutdown_status = "writer_failed"
                logger.critical(
                    "SSD writer did not stop cleanly. Partial durability state."
                )

        if self._batch_generator is not None:
            try:
                self._batch_generator.close()
            except Exception:
                pass

        # Flush SSD cache index
        if self._tiered_cache and hasattr(self._tiered_cache, 'ssd') and self._tiered_cache.ssd is not None:
            try:
                self._tiered_cache.ssd.flush()
                if not self._shutdown_clean:
                    self._shutdown_partial_flush = True
                    self.shutdown_status = "partial_flush"
                    logger.error(
                        "Partial SSD flush after incomplete writer shutdown. "
                        "validate_index() will reconcile on next startup."
                    )
            except Exception:
                logger.warning("Failed to flush SSD cache index on shutdown")

    def _enqueue_shutdown_event(self) -> None:
        """Enqueue a shutdown event to the bus outbox for distributed broadcast."""
        try:
            from mlx_lm_server.distributed_bus import ControlEvent

            deadline = time.monotonic() + BUS_SHUTDOWN_ENQUEUE_TIMEOUT_S
            enqueued = False
            while time.monotonic() < deadline:
                try:
                    self._bus_outbox.put_nowait(ControlEvent.shutdown())
                    enqueued = True
                    break
                except queue.Full:
                    # Let the inference loop make progress and retry without
                    # discarding queued submit/cancel events.
                    self._new_request_event.set()
                    time.sleep(0.01)
            if not enqueued:
                logger.critical(
                    "Cannot enqueue shutdown event within timeout; marking distributed fatal"
                )
                self._dist_fatal = True
                self._dist_fatal_reason = "bus_outbox_full_shutdown"
                self._shutdown_clean = False
                self.shutdown_status = "shutdown_event_enqueue_failed"
        except Exception:
            logger.warning("Failed to queue shutdown event", exc_info=True)

    def _inference_loop(self) -> None:
        """Main inference loop: schedule -> prefill -> decode -> emit tokens."""
        logger.info("Inference loop started")
        while self._running:
            try:
                if self._batch_generator is not None:
                    self._batch_inference_step()
                else:
                    self._mock_inference_step()
            except Exception as e:
                logger.error(
                    "Exception in inference loop iteration: %s", e, exc_info=True
                )
                if self._batch_generator is not None:
                    self._handle_batch_error(e)
                else:
                    self._handle_mock_error()

        # Final bus drain: broadcast shutdown to non-rank0 workers
        if self._control_bus is not None and self._dist_ctx is not None and self._dist_ctx.is_rank0:
            try:
                self._drain_bus_outbox()
            except Exception:
                logger.warning("Failed to drain bus outbox during shutdown", exc_info=True)

        logger.info("Inference loop stopped")

    def _mock_inference_step(self) -> None:
        """One iteration of the mock inference loop (model=None)."""
        # Distributed bus synchronization (TP mode) — same as _batch_inference_step
        if self._control_bus is not None and self._dist_ctx is not None:
            if self._dist_ctx.is_rank0:
                self._drain_bus_outbox()
            else:
                if not self._apply_bus_events():
                    return  # shutdown received

        # Wait for new requests if nothing is active
        with self._active_lock:
            has_active = len(self._active_sequences) > 0
        if not has_active and self.request_queue.size == 0:
            self._new_request_event.wait(timeout=0.1)
            self._new_request_event.clear()
            if not self._running:
                return
            return

        # Run one schedule step
        outputs = self.schedule_step()

        # Run prefill for new sequences
        if outputs.prefill_sequences:
            self._run_prefill(outputs.prefill_sequences)

        # Run one decode step for active sequences
        with self._active_lock:
            decode_seqs = [
                s for s in self._active_sequences.values()
                if not s.is_finished
            ]
        if decode_seqs:
            token_events = self._run_decode_step(decode_seqs)
            self._emit_tokens(token_events)

        # Clean up finished sequences
        self._cleanup_finished()

    def _apply_bus_events(self) -> bool:
        """Receive and apply control events from the distributed bus.

        Called by non-rank0 workers at the start of each batch inference step.
        Receives a compound event containing all pending events for this step.
        Returns False if shutdown was received.
        """
        if self._control_bus is None:
            return True

        from mlx_lm_server.distributed_bus import ControlEvent

        try:
            raw_event = self._control_bus.recv()
            self._bus_error_count = 0
        except Exception:
            logger.error("Failed to receive bus event", exc_info=True)
            self._bus_error_count += 1
            if self._bus_error_count >= BUS_ERROR_THRESHOLD:
                logger.critical(
                    "Bus error threshold (%d) reached — shutting down distributed",
                    BUS_ERROR_THRESHOLD,
                )
                self._dist_fatal = True
                self._dist_fatal_reason = "bus_error_threshold"
                self._running = False
                self._new_request_event.set()
                return False
            return True  # Continue despite error

        # Unpack compound event (list of sub-events)
        if raw_event.typ == "batch":
            try:
                events = raw_event.unpack_batch()
                self._bus_unpack_error_count = 0
            except Exception:
                logger.error("FATAL: rank divergence imminent — failed to deserialize compound bus event", exc_info=True)
                self._bus_unpack_error_count += 1
                self._dist_fatal = True
                self._dist_fatal_reason = "deserialization_failure"
                self._running = False
                self._new_request_event.set()
                return False
        else:
            # Single event (backward compat or noop fallback)
            events = [raw_event]

        for event in events:
            if event.typ == "submit":
                request = event.unpack_request()
                if request is not None:
                    is_streaming = getattr(request, "stream", False)
                    try:
                        self.request_queue.add(request)
                    except Exception:
                        logger.warning(
                            "Failed to add request %s from bus event to queue",
                            request.request_id, exc_info=True,
                        )
                        continue
                    if not is_streaming:
                        is_worker = self._dist_ctx is not None and not self._dist_ctx.is_rank0
                        if not is_worker:
                            with self._results_lock:
                                self._results[request.request_id] = []
                                self._results_ready[request.request_id] = threading.Event()
                    self._inc_stat("total_requests")
                    self._new_request_event.set()

            elif event.typ == "cancel":
                request_id = event.unpack_request_id()
                if request_id is not None:
                    # Try removing from queue first (matching rank0 behavior)
                    if self.request_queue.cancel(request_id):
                        # Clean up result buffers for queued request (U6 fix)
                        self._cleanup_result_buffers(request_id)
                    else:
                        with self._cancelled_lock:
                            self._cancelled.add(request_id)

            elif event.typ == "shutdown":
                self._running = False
                self._new_request_event.set()
                return False

            # "noop" — do nothing

        return True

    def _drain_bus_outbox(self) -> None:
        """Rank0: broadcast all queued events to all ranks via the control bus.

        Uses retry-first pattern: if previous events failed to publish,
        retry those first before draining new events from the outbox.
        This preserves event ordering across retries.

        After successful publish, performs rank0 local apply: adds requests
        to the local queue, creates result buffers, and handles cancels.
        This ensures rank0 only applies events that were successfully
        broadcast to all ranks, preventing rank divergence.
        """
        if self._control_bus is None:
            return

        from mlx_lm_server.distributed_bus import ControlEvent

        # Retry-first: if we have pending retries, use those instead of draining new events
        if self._bus_retry_events:
            events = self._bus_retry_events
            self._bus_retry_events = []
        else:
            # Drain all pending events from the outbox
            events = []
            while True:
                try:
                    events.append(self._bus_outbox.get_nowait())
                except queue.Empty:
                    break

        if not events:
            events = [ControlEvent.noop()]

        # Broadcast the list of events as a single compound event
        compound = ControlEvent.batch(events)
        try:
            self._control_bus.publish(compound)
            self._bus_error_count = 0
        except Exception:
            logger.error("Failed to broadcast %d bus events", len(events), exc_info=True)
            self._bus_error_count += 1
            if self._bus_error_count >= BUS_ERROR_THRESHOLD:
                logger.critical(
                    "Bus error threshold (%d) reached — shutting down distributed",
                    BUS_ERROR_THRESHOLD,
                )
                # Best-effort shutdown broadcast
                try:
                    from mlx_lm_server.distributed_bus import ControlEvent as _CE
                    self._control_bus.publish(_CE.batch([_CE.shutdown()]))
                except Exception:
                    pass
                self._dist_fatal = True
                self._dist_fatal_reason = "bus_error_threshold"
                self._running = False
                self._new_request_event.set()
            else:
                # Preserve events for retry (skip noops)
                real_events = [ev for ev in events if ev.typ != "noop"]
                if real_events:
                    self._bus_retry_events = real_events
                    # Signal quick retry
                    self._new_request_event.set()
            # CRITICAL: Do NOT do local apply if publish failed — that would
            # diverge rank0 from workers who never received these events.
            return

        # ===== Rank0 local apply (after successful publish) =====
        for ev in events:
            if ev.typ == "submit":
                request = ev.unpack_request()
                if request is not None:
                    is_streaming = getattr(request, "stream", False)
                    try:
                        self.request_queue.add(request)
                        if not is_streaming:
                            with self._results_lock:
                                # Keep pre-registered buffers/event if already present.
                                self._results.setdefault(request.request_id, [])
                                self._results_ready.setdefault(request.request_id, threading.Event())
                        self._inc_stat("total_requests")
                        self._inc_stat("accepted_requests")
                    except Exception:
                        logger.exception("Rank0 local add failed for %s — cancelling immediately", request.request_id)
                        # Create result buffers so _signal_finish can notify waiters
                        if not is_streaming:
                            with self._results_lock:
                                self._results.setdefault(request.request_id, [])
                                self._results_ready.setdefault(request.request_id, threading.Event())
                        # Mark cancelled so schedule_step removes it before collective ops
                        with self._cancelled_lock:
                            self._cancelled.add(request.request_id)
                        self._signal_finish(request_id=request.request_id, finish_reason="error")
            elif ev.typ == "cancel":
                request_id = ev.unpack_request_id()
                if request_id is not None:
                    if not self.request_queue.cancel(request_id):
                        with self._active_lock:
                            is_active = request_id in self._active_sequences
                        if is_active:
                            with self._cancelled_lock:
                                self._cancelled.add(request_id)
                        else:
                            # Unknown request on rank0 local apply: clean up any
                            # pre-registered buffers/streams to avoid leaks.
                            self._cleanup_result_buffers(request_id)
                            self.unregister_stream(request_id)
                    else:
                        # Request was in queue — signal finish (get_result pops buffers)
                        self._signal_finish(request_id, finish_reason="cancelled")
            # "shutdown" and "noop" — no local apply needed
            # (shutdown is handled in _apply_bus_events path or directly)

    def _batch_inference_step(self) -> None:
        """One step of the batch inference loop using BatchGenerator."""
        # Distributed bus synchronization (TP mode)
        if self._control_bus is not None and self._dist_ctx is not None:
            if self._dist_ctx.is_rank0:
                # rank0: broadcast all queued events (or noop) atomically
                self._drain_bus_outbox()
            else:
                # rank>0: receive and apply compound event
                if not self._apply_bus_events():
                    return  # shutdown received

        # Wait for work if idle
        with self._active_lock:
            has_active = bool(self._uid_to_request_id)
        if not has_active and self.request_queue.size == 0:
            self._new_request_event.wait(timeout=0.1)
            self._new_request_event.clear()
            if not self._running:
                return
            return

        # 1. Process cancellations
        self._process_cancellations_batch()

        # 2. Insert new requests
        self._insert_new_requests_batch()

        # 3. Run one batch step if there are active sequences
        with self._active_lock:
            has_active = bool(self._uid_to_request_id)
        if not has_active:
            return

        # Call next() to get responses for all active sequences
        try:
            responses = self._batch_generator.next()
        except Exception as e:
            logger.error("BatchGenerator.next() failed: %s", e, exc_info=True)
            raise

        # 4. Process responses
        events, uids_to_remove, finished_caches = self._process_batch_responses(responses)

        # 5. Emit token events
        self._emit_tokens(events)

        # 6. Remove early-stopped UIDs from BatchGenerator
        if uids_to_remove:
            try:
                caches = self._batch_generator.remove(
                    uids_to_remove, return_prompt_caches=True
                )
                if caches:
                    finished_caches.update(caches)
            except Exception as e:
                logger.warning("Failed to remove UIDs from BatchGenerator: %s", e)

        # 7. Store finished caches
        self._store_finished_caches(finished_caches)

        # 8. Periodic SSD pruning
        self._prune_ssd_if_needed()

        # 9. Clean up finished sequences
        self._cleanup_finished_batch()

    def _process_batch_responses(self, responses) -> tuple[list[TokenEvent], list[int], dict[int, list]]:
        """Process batch responses: detokenize, check stops, create events.

        Returns:
            Tuple of (events, uids_to_remove, finished_caches).
        """
        uids_to_remove: list[int] = []
        events: list[TokenEvent] = []
        finished_caches: dict[int, list] = {}

        for r in responses:
            request_id = self._uid_to_request_id.get(r.uid)
            if request_id is None:
                continue

            with self._active_lock:
                seq = self._active_sequences.get(request_id)
            if seq is None:
                continue

            # Detokenize
            detokenizer = getattr(seq, "_detokenizer", None)
            if detokenizer is not None:
                detokenizer.add_token(r.token)
                token_text = detokenizer.last_segment
            else:
                token_text = str(r.token)

            seq.output_tokens.append(r.token)
            seq.token_ids.append(r.token)
            seq.output_text += token_text

            finish_reason = r.finish_reason

            # Check our custom stop conditions if BatchGenerator hasn't stopped
            if finish_reason is None:
                request = getattr(seq, "_request", None)
                if request is not None:
                    finish_reason = self._check_stop_conditions(seq, request)
                    if finish_reason is not None:
                        uids_to_remove.append(r.uid)

            if finish_reason is not None:
                seq.is_finished = True
                seq.finish_reason = finish_reason
                self._inc_stat("requests_completed")
                # Save prompt cache for sequence cache store and block decomposition
                if r.prompt_cache is not None:
                    try:
                        prompt_cache = r.prompt_cache
                        finished_caches[r.uid] = prompt_cache
                        # Also store on seq for block decomposition in cleanup (P7.2)
                        seq._prompt_cache = prompt_cache  # type: ignore[attr-defined]
                    except Exception:
                        pass

            event = TokenEvent(
                request_id=request_id,
                token_id=r.token,
                token_text=token_text,
                finish_reason=finish_reason,
            )
            events.append(event)

        return events, uids_to_remove, finished_caches

    def _store_finished_caches(self, finished_caches: dict[int, list]) -> None:
        """Store finished sequence caches and decompose to block-level cache."""
        for uid, prompt_cache in finished_caches.items():
            rid = self._uid_to_request_id.get(uid)
            if rid is None:
                continue
            with self._active_lock:
                seq = self._active_sequences.get(rid)
            if seq is None:
                continue

            # Compute prompt tokens (token_ids minus output_tokens)
            prompt_tokens = seq.token_ids[:len(seq.token_ids) - len(seq.output_tokens)]

            # Store in sequence-level cache
            # A2: prompt_cache includes generated tokens' KV — trim before storing
            if self._sequence_cache is not None:
                can_store = True
                num_generated = len(seq.output_tokens)
                if num_generated > 0 and prompt_cache is not None:
                    if can_trim_prompt_cache is not None and can_trim_prompt_cache(prompt_cache):
                        trim_prompt_cache(prompt_cache, num_generated)
                    else:
                        logger.warning(
                            "Skip sequence-cache store: non-trimmable cache type for %s",
                            rid,
                        )
                        can_store = False
                if can_store:
                    self._sequence_cache.store(prompt_tokens, prompt_cache)

            # Decompose into block-level cache for finer-grained sharing
            if self.kv_cache_manager is not None:
                try:
                    block_dicts = decompose_cache_to_blocks(
                        prompt_cache, prompt_tokens,
                        self.kv_cache_manager.block_size,
                    )
                    protected_ids: set[int] = set()
                    for bd in block_dicts:
                        block_id = self.kv_cache_manager.cache_block(
                            block_hash=bd['block_hash'],
                            token_ids=bd['token_ids'],
                            kv_data=bd['kv_data_per_layer'],
                            tiered_cache=self._tiered_cache,
                            exclude_ids=protected_ids,
                            ssd_policy=self.config.ssd_policy,
                        )
                        if block_id is not None:
                            protected_ids.add(block_id)
                            # Release the protection ref — block is now in hash_table
                            # and will be pushed to eviction heap by free_blocks()
                            self.kv_cache_manager.free_blocks([block_id])
                except Exception as e:
                    logger.warning("Failed to decompose cache to blocks: %s", e)

    def _prune_ssd_if_needed(self) -> None:
        """Periodically prune expired SSD cache blocks (step-based or time-based)."""
        self._ssd_prune_counter += 1
        now = time.time()
        should_prune = (
            self._ssd_prune_counter >= self._ssd_prune_interval
            or (now - self._ssd_last_prune_time) >= self._ssd_prune_time_interval
        )
        if should_prune:
            self._ssd_prune_counter = 0
            self._ssd_last_prune_time = now
            if self._tiered_cache is not None and self._tiered_cache.ssd is not None:
                try:
                    pruned = self._tiered_cache.ssd.prune_expired()
                    if pruned > 0:
                        logger.info("Pruned %d expired SSD blocks", pruned)
                except Exception as e:
                    logger.warning("SSD pruning failed: %s", e)

    def _process_cancellations_batch(self) -> None:
        """Remove cancelled requests from BatchGenerator.

        G2: When removing cancelled UIDs, extract prompt caches via
        return_prompt_caches=True. For UIDs that had cache misses during
        prefill (_pending_cache_saves), store the extracted caches to the
        sequence cache and block-level cache. This preserves prefill
        computation so future requests with overlapping prefixes can reuse it.

        All cache extraction happens on the inference thread (this method is
        called from _batch_inference_step), which is safe because
        BatchGenerator.remove() is not thread-safe.
        """
        with self._cancelled_lock:
            cancelled = set(self._cancelled)

        uids_to_remove = []
        for rid in cancelled:
            uid = self._request_id_to_uid.get(rid)
            if uid is not None:
                uids_to_remove.append(uid)

        # G2: Extract prompt caches before removing from batch.
        # Only request caches if there are UIDs worth saving (had cache misses).
        salvageable_uids = self._pending_cache_saves & set(uids_to_remove)
        returned_caches: dict[int, list] = {}

        if uids_to_remove:
            try:
                if salvageable_uids:
                    caches = self._batch_generator.remove(
                        uids_to_remove, return_prompt_caches=True
                    )
                    if caches:
                        # Only keep caches for UIDs that had cache misses
                        returned_caches = {
                            uid: cache
                            for uid, cache in caches.items()
                            if uid in salvageable_uids
                        }
                else:
                    self._batch_generator.remove(uids_to_remove)
            except Exception as e:
                logger.warning("Failed to remove cancelled UIDs: %s", e)

        # G2: Store salvaged caches BEFORE cleaning up UID mappings and
        # active sequences, because _store_finished_caches reads both
        # _uid_to_request_id and _active_sequences.
        if returned_caches:
            try:
                self._store_finished_caches(returned_caches)
                logger.debug(
                    "G2: Salvaged %d prefill caches from cancelled requests",
                    len(returned_caches),
                )
            except Exception as e:
                logger.warning(
                    "G2: Failed to store salvaged caches from cancelled requests: %s", e
                )

        # Clean up mappings and signal finish
        for rid in cancelled:
            uid = self._request_id_to_uid.pop(rid, None)
            if uid is not None:
                self._uid_to_request_id.pop(uid, None)
                self._pending_cache_saves.discard(uid)
            with self._active_lock:
                seq = self._active_sequences.pop(rid, None)
            if seq is not None:
                seq.is_finished = True
                seq.finish_reason = "cancelled"
                # Free KV cache blocks to prevent memory leak (F4 fix)
                if self.kv_cache_manager is not None and seq.block_ids:
                    self.kv_cache_manager.free_blocks(seq.block_ids)
                self._signal_finish(rid, finish_reason="cancelled")
            with self._cancelled_lock:
                self._cancelled.discard(rid)

    def _insert_new_requests_batch(self) -> None:
        """Pop requests from queue and insert into BatchGenerator."""
        with self._active_lock:
            available = self.config.max_batch_size - len(self._active_sequences)
        if available <= 0:
            return

        new_requests = self.request_queue.pop_batch(available)
        if not new_requests:
            return

        for req in new_requests:
            seq = None
            try:
                seq = self._init_sequence(req)

                # Set up detokenizer
                if self.tokenizer is not None:
                    detok = getattr(self.tokenizer, "detokenizer", None)
                    if detok is not None:
                        seq._detokenizer = copy.copy(detok)
                        seq._detokenizer.reset()

                # NOTE: Do NOT add to _active_sequences here. It must happen
                # AFTER insert() succeeds, so a failed insert() doesn't leave
                # a stuck sequence with no UID (Bug 3 / Issue 4 fix).

                # Look up caches: block-level first (finer granularity),
                # then sequence-level (faster, no reconstruction overhead)
                cache = None
                remaining_tokens = seq.token_ids

                # Block-level cache lookup via KVCacheManager
                if cache is None and self.kv_cache_manager is not None and self.model is not None:
                    num_cached = self.kv_cache_manager.find_cached_prefix(seq.token_ids)
                    if num_cached > 0:
                        # Allocate blocks to track the reference
                        block_ids = self.kv_cache_manager.allocate_blocks(
                            seq.token_ids[:num_cached]
                        )
                        seq.block_ids = block_ids
                        # D1+G1: Validate all blocks before reconstruction
                        try:
                            block_data = []
                            all_blocks_valid = True
                            for bid in block_ids:
                                block = self.kv_cache_manager.get_block(bid)
                                if block.kv_data is not None:
                                    block_data.append(block.kv_data)
                                else:
                                    # G1: Try SSD promote if tiered cache available
                                    promoted = False
                                    if block.block_hash and self._tiered_cache:
                                        if hasattr(self._tiered_cache, 'ssd') and self._tiered_cache.ssd is not None:
                                            try:
                                                raw_data = self._tiered_cache.ssd.load_block(block.block_hash)
                                                if raw_data is not None:
                                                    # Normalize: load_block may return list[dict] or dict
                                                    if isinstance(raw_data, dict):
                                                        raw_data = [raw_data]
                                                    block.kv_data = raw_data
                                                    block.last_accessed = time.time()
                                                    block_data.append(raw_data)
                                                    promoted = True
                                            except Exception as e:
                                                logger.debug("SSD promote failed for block %s: %s", block.block_hash, e)
                                    if not promoted:
                                        all_blocks_valid = False
                                        break

                            if all_blocks_valid and block_data:
                                cache = reconstruct_cache_from_blocks(
                                    [{'kv_data_per_layer': d}
                                     for d in block_data],
                                    self.model,
                                )
                                logger.debug(
                                    "Block cache hit for %s: %d tokens cached",
                                    req.request_id, num_cached,
                                )
                                self._inc_stat("cache_hits_block")
                                self._inc_stat("total_cached_tokens", num_cached)
                                remaining_tokens = seq.token_ids[num_cached:]
                            else:
                                # Partial/incomplete blocks — fall back to uncached path
                                logger.warning(
                                    "Partial block data for %s — falling back to uncached",
                                    req.request_id,
                                )
                                self.kv_cache_manager.free_blocks(block_ids)
                                seq.block_ids = []
                                cache = None
                                remaining_tokens = seq.token_ids
                        except Exception as e:
                            logger.warning(
                                "Failed to reconstruct cache from blocks for %s: %s",
                                req.request_id, e,
                            )
                            self.kv_cache_manager.free_blocks(block_ids)
                            seq.block_ids = []
                            cache = None
                            remaining_tokens = seq.token_ids

                # Sequence-level cache lookup (fallback)
                if cache is None and self._sequence_cache is not None:
                    cached, remaining = self._sequence_cache.find_longest_prefix(seq.token_ids)
                    if cached is not None:
                        cache = cached
                        remaining_tokens = remaining
                        num_seq_cached = len(seq.token_ids) - len(remaining)
                        self._inc_stat("cache_hits_sequence")
                        self._inc_stat("total_cached_tokens", num_seq_cached)
                        logger.debug(
                            "Sequence cache hit for %s: %d tokens cached",
                            req.request_id,
                            num_seq_cached,
                        )

                # Track cache miss (neither block nor sequence cache hit)
                if cache is None:
                    self._inc_stat("cache_misses")

                # Create sampler
                sampler = None
                if make_sampler is not None:
                    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

                # Handle max_tokens=0 gracefully
                if req.max_tokens <= 0:
                    # Free any blocks allocated during cache lookup
                    if self.kv_cache_manager is not None and seq.block_ids:
                        self.kv_cache_manager.free_blocks(seq.block_ids)
                        seq.block_ids = []
                    seq.is_finished = True
                    seq.finish_reason = "length"
                    self._signal_finish(req.request_id, finish_reason="length")
                    continue

                # Track prefill tokens (tokens that need actual computation)
                self._inc_stat("total_prefill_tokens", len(remaining_tokens))

                # A3: Handle full cache hit — trim cache by 1 to avoid last-token duplication
                if not remaining_tokens:
                    remaining_tokens = [seq.token_ids[-1]]
                    if cache is not None:
                        if can_trim_prompt_cache is not None and can_trim_prompt_cache(cache):
                            trim_prompt_cache(cache, 1)
                        else:
                            # Non-trimmable cache — fall back to uncached path
                            logger.debug(
                                "Full cache hit but non-trimmable — falling back to uncached path"
                            )
                            if self.kv_cache_manager is not None and seq.block_ids:
                                self.kv_cache_manager.free_blocks(seq.block_ids)
                            seq.block_ids = []
                            cache = None
                            remaining_tokens = seq.token_ids

                # Insert into BatchGenerator
                try:
                    insert_kwargs = {
                        "prompts": [remaining_tokens],
                        "max_tokens": [req.max_tokens],
                    }
                    if cache is not None:
                        insert_kwargs["caches"] = [cache]
                    if sampler is not None:
                        insert_kwargs["samplers"] = [sampler]

                    uids = self._batch_generator.insert(**insert_kwargs)
                    uid = uids[0]
                    self._uid_to_request_id[uid] = req.request_id
                    self._request_id_to_uid[req.request_id] = uid
                    seq._batch_uid = uid
                    # G2: Track cache-miss inserts for early prefill save
                    if cache is None:
                        self._pending_cache_saves.add(uid)
                    # Register in _active_sequences AFTER insert() succeeds
                    # and UID is assigned (Bug 3 / Issue 4 fix).
                    with self._active_lock:
                        self._active_sequences[req.request_id] = seq
                    logger.debug(
                        "Inserted request %s as UID %d", req.request_id, uid
                    )
                except Exception as e:
                    logger.error("Failed to insert request %s: %s", req.request_id, e)
                    seq.is_finished = True
                    seq.finish_reason = "error"
                    # Free KV cache blocks allocated during cache lookup
                    if self.kv_cache_manager is not None and seq.block_ids:
                        self.kv_cache_manager.free_blocks(seq.block_ids)
                        seq.block_ids = []
                    self._signal_finish(req.request_id, finish_reason="error")
            except Exception as e:
                # Request was popped from queue but failed during setup —
                # signal error to caller so they don't hang forever
                request_id = req.request_id
                logger.error("Failed to insert request %s: %s", request_id, e)
                # Free KV blocks allocated during cache lookup to prevent leak
                if seq is not None and self.kv_cache_manager is not None and seq.block_ids:
                    self.kv_cache_manager.free_blocks(seq.block_ids)
                    seq.block_ids = []
                self._signal_finish(request_id, finish_reason="error")
                # Clean up if sequence was partially added to active set
                with self._active_lock:
                    self._active_sequences.pop(request_id, None)

    def _cleanup_finished_batch(self) -> None:
        """Clean up finished sequences from the batch path."""
        with self._active_lock:
            finished_ids = [
                rid for rid, seq in self._active_sequences.items()
                if seq.is_finished
            ]
            for rid in finished_ids:
                seq = self._active_sequences.pop(rid)
                # Clean up UID mappings
                uid = self._request_id_to_uid.pop(rid, None)
                if uid is not None:
                    self._uid_to_request_id.pop(uid, None)
                    self._pending_cache_saves.discard(uid)
                # Free KV cache blocks if manager is available
                if self.kv_cache_manager is not None and seq.block_ids:
                    self.kv_cache_manager.free_blocks(seq.block_ids)
                logger.debug(
                    "Cleaned up batch sequence %s (reason=%s)",
                    rid,
                    seq.finish_reason,
                )

    def _handle_batch_error(self, error: Exception) -> None:
        """Handle error in batch inference path."""
        with self._active_lock:
            for seq in self._active_sequences.values():
                if not seq.is_finished:
                    seq.is_finished = True
                    seq.finish_reason = "error"
                    self._inc_stat("requests_errored")
                    self._signal_finish(seq.request_id, finish_reason="error")
        self._cleanup_finished_batch()

        # Clear stale cancelled IDs (they referred to now-dead sequences)
        with self._cancelled_lock:
            self._cancelled.clear()

        # Reset BatchGenerator
        try:
            self._batch_generator.close()
        except Exception:
            pass
        self._create_batch_generator()
        self._uid_to_request_id.clear()
        self._request_id_to_uid.clear()
        self._pending_cache_saves.clear()

    def _handle_mock_error(self) -> None:
        """Handle error in mock inference path (existing behavior)."""
        with self._active_lock:
            for seq in self._active_sequences.values():
                if not seq.is_finished:
                    seq.is_finished = True
                    seq.finish_reason = "error"
                    self._inc_stat("requests_errored")
                    self._signal_finish(seq.request_id, finish_reason="error")
        self._cleanup_finished()

    # --- Scheduling ---

    def schedule_step(self) -> SchedulerOutputs:
        """One scheduling iteration.

        1. Remove finished/cancelled sequences from active set.
        2. Pop new requests from queue to fill available slots.
        3. Classify sequences as prefill vs decode.
        """
        prefill_sequences: list[SequenceState] = []
        decode_sequences: list[SequenceState] = []
        preempted: list[SequenceState] = []

        # 1. Remove finished and cancelled sequences
        with self._active_lock:
            finished_ids = [
                rid for rid, seq in self._active_sequences.items()
                if seq.is_finished
            ]
            for rid in finished_ids:
                del self._active_sequences[rid]

            # Handle cancellations
            with self._cancelled_lock:
                cancelled = set(self._cancelled)
            for rid in list(self._active_sequences):
                if rid in cancelled:
                    seq = self._active_sequences.pop(rid)
                    seq.is_finished = True
                    seq.finish_reason = "cancelled"
                    # Free KV cache blocks to prevent memory leak (F4 fix)
                    if self.kv_cache_manager is not None and seq.block_ids:
                        self.kv_cache_manager.free_blocks(seq.block_ids)
                    self._signal_finish(rid, finish_reason="cancelled")
                    with self._cancelled_lock:
                        self._cancelled.discard(rid)

            # 2. Fill available slots
            available = self.config.max_batch_size - len(self._active_sequences)

        if available > 0:
            new_requests = self.request_queue.pop_batch(available)
            for req in new_requests:
                try:
                    seq = self._init_sequence(req)
                except Exception as e:
                    logger.error("Failed to init sequence for %s: %s", req.request_id, e)
                    self._signal_finish(req.request_id, finish_reason="error")
                    continue
                with self._active_lock:
                    self._active_sequences[req.request_id] = seq
                if seq.num_computed_tokens < len(seq.token_ids):
                    prefill_sequences.append(seq)
                else:
                    decode_sequences.append(seq)

        # 3. Classify existing active sequences
        with self._active_lock:
            for seq in self._active_sequences.values():
                if seq.is_finished:
                    continue
                if seq not in prefill_sequences and seq not in decode_sequences:
                    decode_sequences.append(seq)

        return SchedulerOutputs(
            prefill_sequences=prefill_sequences,
            decode_sequences=decode_sequences,
            preempted_sequences=preempted,
        )

    def _init_sequence(self, request: InferenceRequest) -> SequenceState:
        """Initialize a SequenceState from an InferenceRequest.

        - Uses prompt_tokens directly (already tokenized).
        - Checks prefix cache for cached tokens.
        """
        token_ids = list(request.prompt_tokens)
        num_cached = 0

        # Check prefix cache if available (mock path only — batch path
        # does its own block-level lookup in _insert_new_requests_batch)
        if self.kv_cache_manager is not None and self.model is None:
            num_cached = self.kv_cache_manager.find_cached_prefix(token_ids)
            logger.debug(
                "Request %s: %d/%d tokens cached",
                request.request_id,
                num_cached,
                len(token_ids),
            )

        seq = SequenceState(
            request_id=request.request_id,
            token_ids=token_ids,
            num_computed_tokens=num_cached,
        )
        # Store request metadata on the sequence for later use
        seq._request = request  # type: ignore[attr-defined]
        return seq

    def _run_prefill(self, sequences: list[SequenceState]) -> None:
        """Process uncached prompt tokens for the given sequences.

        In mock mode (model=None), marks all prompt tokens as computed.
        Real model prefill is handled by BatchGenerator in _batch_inference_step.
        """
        for seq in sequences:
            # Track prefill vs cached tokens (mock path)
            num_cached = seq.num_computed_tokens
            num_prefill = len(seq.token_ids) - num_cached
            self._inc_stat("total_prefill_tokens", num_prefill)
            self._inc_stat("total_cached_tokens", num_cached)
            # Mock mode: mark all prompt tokens as computed (prefill complete)
            seq.num_computed_tokens = len(seq.token_ids)
            logger.debug(
                "Prefill complete for %s (%d tokens, %d cached, %d prefilled)",
                seq.request_id,
                seq.num_computed_tokens,
                num_cached,
                num_prefill,
            )

    def _run_decode_step(self, sequences: list[SequenceState]) -> list[TokenEvent]:
        """Run one decode iteration on active sequences.

        In mock mode, uses _mock_generate callback.
        Real model decode is handled by BatchGenerator in _batch_inference_step.

        Returns a list of TokenEvents, one per sequence that produced a token.
        """
        events: list[TokenEvent] = []

        for seq in sequences:
            if seq.is_finished:
                continue

            request = getattr(seq, "_request", None)
            if request is None:
                continue

            # Calculate decode step number (tokens generated so far)
            step = len(seq.output_tokens)

            # Check max_tokens before generating
            if step >= request.max_tokens:
                seq.is_finished = True
                seq.finish_reason = "length"
                event = TokenEvent(
                    request_id=seq.request_id,
                    token_id=-1,
                    token_text="",
                    finish_reason="length",
                )
                events.append(event)
                continue

            # Generate token
            if self._mock_generate is not None:
                token_id, token_text, finish_reason = self._mock_generate(
                    seq.request_id, seq.token_ids, step
                )
            else:
                # No model and no mock: generate placeholder tokens
                token_id = step + 1
                token_text = f"tok{token_id}"
                finish_reason = None

            # Update sequence state
            seq.output_tokens.append(token_id)
            seq.token_ids.append(token_id)
            seq.output_text += token_text

            # Check stop conditions
            if finish_reason is None:
                finish_reason = self._check_stop_conditions(seq, request)

            if finish_reason is not None:
                seq.is_finished = True
                seq.finish_reason = finish_reason

            event = TokenEvent(
                request_id=seq.request_id,
                token_id=token_id,
                token_text=token_text,
                finish_reason=finish_reason,
            )
            events.append(event)

        return events

    def _check_stop_conditions(
        self, seq: SequenceState, request: InferenceRequest
    ) -> str | None:
        """Check if a sequence should stop generating.

        Checks:
        1. Max tokens reached
        2. EOS token detected (token_id == 0 by convention, or via tokenizer)
        3. Stop sequence found in output text
        """
        # Max tokens
        if len(seq.output_tokens) >= request.max_tokens:
            return "length"

        # EOS token detection
        if self.tokenizer is not None:
            eos_ids = getattr(self.tokenizer, "eos_token_ids", set())
            if seq.output_tokens and seq.output_tokens[-1] in eos_ids:
                return "stop"

        # Stop sequence detection
        if request.stop_sequences and seq.output_text:
            for stop_seq in request.stop_sequences:
                if stop_seq in seq.output_text:
                    idx = seq.output_text.index(stop_seq)
                    seq.output_text = seq.output_text[:idx]
                    return "stop"

        return None

    # --- Token Delivery ---

    def _put_event_to_stream(
        self, stream: queue.Queue, event: TokenEvent, request_id: str, is_finish: bool = False
    ) -> None:
        """Put a token event onto a stream queue with backpressure handling.

        For finish events (is_finish=True), uses non-blocking put_nowait with
        drop-oldest retry (max 3 attempts) to guarantee delivery without blocking
        the inference loop. For regular token events, uses put_nowait with
        backpressure overflow cancellation.
        """
        if is_finish:
            for attempt in range(3):
                try:
                    stream.put_nowait(event)
                    break
                except queue.Full:
                    try:
                        stream.get_nowait()  # drop oldest token
                    except queue.Empty:
                        pass
            else:
                logger.error("Stream queue full for %s after 3 retries, finish event dropped", request_id)
        else:
            try:
                stream.put_nowait(event)
            except queue.Full:
                # Backpressure overflow: client is consuming tokens too slowly.
                # Instead of silently dropping tokens (which causes data corruption),
                # signal an error and cancel the request (Bug 4 / Issue 6 fix).
                logger.warning(
                    "Stream backpressure overflow for %s: client consuming tokens too slowly, "
                    "cancelling request",
                    request_id,
                )
                # Add to cancelled set so scheduler stops generating for this request
                with self._cancelled_lock:
                    self._cancelled.add(request_id)
                # Drain the queue to make room for the finish event
                while not stream.empty():
                    try:
                        stream.get_nowait()
                    except queue.Empty:
                        break
                # Deliver the error finish event to the client
                error_event = TokenEvent(
                    request_id=request_id,
                    token_id=-1,
                    token_text="",
                    finish_reason="error",
                )
                try:
                    stream.put_nowait(error_event)
                except queue.Full:
                    logger.error(
                        "Failed to deliver backpressure error event for %s", request_id
                    )

    def _emit_tokens(self, events: list[TokenEvent]) -> None:
        """Deliver token events to streams and result buffers."""
        for event in events:
            self._inc_stat("tokens_generated")
            rid = event.request_id

            # Deliver to stream if registered
            with self._streams_lock:
                stream = self._streams.get(rid)
            if stream is not None:
                self._put_event_to_stream(stream, event, rid, is_finish=(event.finish_reason is not None))
                if event.finish_reason is not None:
                    with self._streams_lock:
                        self._streams.pop(rid, None)

            # Deliver to result buffer
            with self._results_lock:
                if rid in self._results:
                    self._results[rid].append(event)
                    if event.finish_reason is not None:
                        ready = self._results_ready.get(rid)
                        if ready is not None:
                            ready.set()

    def _cleanup_result_buffers(self, request_id: str) -> None:
        """Remove result buffers for a request to prevent memory leaks.

        Called after a request is cancelled or finishes abnormally and
        no caller will retrieve the results via get_result().
        """
        with self._results_lock:
            self._results.pop(request_id, None)
            self._results_ready.pop(request_id, None)

    def _signal_finish(self, request_id: str, finish_reason: str) -> None:
        """Signal that a request has finished (for queue-cancelled requests)."""
        event = TokenEvent(
            request_id=request_id,
            token_id=-1,
            token_text="",
            finish_reason=finish_reason,
        )

        with self._streams_lock:
            stream = self._streams.pop(request_id, None)
        if stream is not None:
            self._put_event_to_stream(stream, event, request_id, is_finish=True)

        with self._results_lock:
            if request_id in self._results:
                self._results[request_id].append(event)
                ready = self._results_ready.get(request_id)
                if ready is not None:
                    ready.set()

    def _cleanup_finished(self) -> None:
        """Remove finished sequences from the active set and free resources."""
        with self._active_lock:
            finished_ids = [
                rid for rid, seq in self._active_sequences.items()
                if seq.is_finished
            ]
            for rid in finished_ids:
                seq = self._active_sequences.pop(rid)
                # Track completed (non-error) requests
                if seq.finish_reason != "error":
                    self._inc_stat("requests_completed")
                # Free KV cache blocks if manager is available
                if self.kv_cache_manager is not None and seq.block_ids:
                    self.kv_cache_manager.free_blocks(seq.block_ids)
                logger.debug(
                    "Cleaned up sequence %s (reason=%s)",
                    rid,
                    seq.finish_reason,
                )

    # --- Introspection ---

    @property
    def num_active_sequences(self) -> int:
        """Number of currently active (in-flight) sequences."""
        with self._active_lock:
            return len(self._active_sequences)

    @property
    def num_queued_requests(self) -> int:
        """Number of requests waiting in the queue."""
        return self.request_queue.size

    def get_cache_stats(self) -> dict[str, Any]:
        """Return a snapshot of cache and scheduler statistics for the /health endpoint.

        The snapshot includes block pool stats, effectiveness counters, SSD tier stats,
        async writer stats, and shutdown health fields. All counters are read under lock
        for consistency.
        """
        stats: dict[str, Any] = {
            "active_sequences": self.num_active_sequences,
            "queued_requests": self.num_queued_requests,
        }
        if self.kv_cache_manager is not None:
            stats["total_blocks"] = self.kv_cache_manager.pool.num_blocks
            stats["used_blocks"] = self.kv_cache_manager.num_used_blocks
            stats["free_blocks"] = self.kv_cache_manager.num_free_blocks
            stats["cached_blocks"] = self.kv_cache_manager.num_cached_blocks
        # Add effectiveness counters (snapshot under lock for consistency)
        with self._stats_lock:
            stats.update(self._stats)
            total_lookups = (
                self._stats["cache_hits_block"]
                + self._stats["cache_hits_sequence"]
                + self._stats["cache_misses"]
            )
            hits = self._stats["cache_hits_block"] + self._stats["cache_hits_sequence"]
            cached = self._stats["total_cached_tokens"]
            prefill = self._stats["total_prefill_tokens"]

        # Merge KV cache manager stats (kv_ prefix)
        if self.kv_cache_manager is not None and hasattr(self.kv_cache_manager, 'get_stats'):
            stats.update(self.kv_cache_manager.get_stats())

        # Merge SSD cache stats (ssd_ prefix)
        if (self._tiered_cache is not None
                and hasattr(self._tiered_cache, 'ssd')
                and self._tiered_cache.ssd is not None
                and hasattr(self._tiered_cache.ssd, 'get_stats')):
            stats.update(self._tiered_cache.ssd.get_stats())

        # Merge async writer stats (writer_ prefix)
        if (self._tiered_cache is not None
                and hasattr(self._tiered_cache, 'get_writer_stats')):
            stats.update(self._tiered_cache.get_writer_stats())

        # Merge sync durability stats (tiered_sync_ prefix)
        if (self._tiered_cache is not None
                and hasattr(self._tiered_cache, 'get_sync_stats')):
            stats.update(self._tiered_cache.get_sync_stats())

        stats["cache_hit_rate"] = (
            hits / total_lookups if total_lookups > 0 else 0.0
        )
        stats["cache_effectiveness"] = (
            cached / (cached + prefill) if (cached + prefill) > 0 else 0.0
        )
        # Distributed control-plane health
        stats["dist_fatal"] = self._dist_fatal
        stats["dist_fatal_reason"] = self._dist_fatal_reason
        stats["dist_bus_error_count"] = self._bus_error_count
        stats["dist_bus_unpack_error_count"] = self._bus_unpack_error_count
        stats["dist_outbox_size"] = self._bus_outbox.qsize()
        # Shutdown health fields
        stats["shutdown_clean"] = self._shutdown_clean
        stats["shutdown_partial_flush"] = self._shutdown_partial_flush
        stats["shutdown_status"] = self.shutdown_status
        return stats

    def shutdown(self) -> None:
        """Graceful shutdown — stops the inference loop."""
        self.stop()

    @property
    def is_running(self) -> bool:
        """Whether the inference loop is running."""
        return self._running
