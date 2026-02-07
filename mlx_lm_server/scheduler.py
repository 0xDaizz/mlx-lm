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

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        config: ServerConfig,
        model=None,
        tokenizer=None,
        kv_cache_manager=None,
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

    # --- Public API (called by server / tests) ---

    def submit_request(self, request: InferenceRequest) -> None:
        """Submit a new inference request to the scheduler.

        The request is added to the queue and the inference loop is notified.
        """
        self.request_queue.add(request)

        # Set up result storage for non-streaming requests
        with self._results_lock:
            self._results[request.request_id] = []
            self._results_ready[request.request_id] = threading.Event()

        self._new_request_event.set()
        logger.debug("Submitted request %s", request.request_id)

    def register_stream(self, request_id: str) -> queue.Queue[TokenEvent]:
        """Register a streaming token queue for a request.

        Returns a Queue that will receive TokenEvent objects as tokens
        are generated. A TokenEvent with finish_reason != None signals
        the end of generation.
        """
        q: queue.Queue[TokenEvent] = queue.Queue()
        with self._streams_lock:
            self._streams[request_id] = q
        return q

    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]:
        """Blocking get for non-streaming request results.

        Waits until the request finishes, then returns all TokenEvents.
        """
        with self._results_lock:
            event = self._results_ready.get(request_id)
        if event is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        event.wait(timeout=timeout)

        with self._results_lock:
            tokens = self._results.pop(request_id, [])
            self._results_ready.pop(request_id, None)
        return tokens

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request.

        If the request is still in the queue, it is removed.
        If it is already active, it is marked as cancelled and will
        be cleaned up at the next schedule step.

        Returns True if the request was found.
        """
        # Try removing from queue first
        if self.request_queue.cancel(request_id):
            self._signal_finish(request_id, finish_reason="cancelled")
            return True

        # Mark as cancelled if active
        with self._cancelled_lock:
            self._cancelled.add(request_id)

        with self._active_lock:
            if request_id in self._active_sequences:
                return True

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

    def stop(self) -> None:
        """Stop the inference loop."""
        self._running = False
        self._new_request_event.set()  # Wake up the loop
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
            self._inference_thread = None

    def _inference_loop(self) -> None:
        """Main inference loop: schedule -> prefill -> decode -> emit tokens."""
        logger.info("Inference loop started")
        while self._running:
            try:
                # Wait for new requests if nothing is active
                with self._active_lock:
                    has_active = len(self._active_sequences) > 0
                if not has_active and self.request_queue.size == 0:
                    self._new_request_event.wait(timeout=0.1)
                    self._new_request_event.clear()
                    if not self._running:
                        break
                    continue

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

            except Exception as e:
                logger.error(
                    "Exception in inference loop iteration: %s", e, exc_info=True
                )
                # Mark all active sequences as failed to prevent infinite loops
                # on persistent errors. The loop continues to serve new requests.
                with self._active_lock:
                    for seq in self._active_sequences.values():
                        if not seq.is_finished:
                            seq.is_finished = True
                            seq.finish_reason = "error"
                            self._signal_finish(seq.request_id, finish_reason="error")
                self._cleanup_finished()

        logger.info("Inference loop stopped")

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
                    self._signal_finish(rid, finish_reason="cancelled")
                    with self._cancelled_lock:
                        self._cancelled.discard(rid)

            # 2. Fill available slots
            available = self.config.max_batch_size - len(self._active_sequences)

        if available > 0:
            new_requests = self.request_queue.pop_batch(available)
            for req in new_requests:
                seq = self._init_sequence(req)
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
                if seq not in prefill_sequences:
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

        # Check prefix cache if available
        if self.kv_cache_manager is not None:
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

        In mock mode (model=None), simply marks all prompt tokens as computed.
        With a real model, would run the model forward pass on uncached tokens.
        """
        for seq in sequences:
            # Mark all prompt tokens as computed (prefill complete)
            seq.num_computed_tokens = len(seq.token_ids)
            logger.debug(
                "Prefill complete for %s (%d tokens)",
                seq.request_id,
                seq.num_computed_tokens,
            )

    def _run_decode_step(self, sequences: list[SequenceState]) -> list[TokenEvent]:
        """Run one decode iteration on active sequences.

        In mock mode, uses _mock_generate callback.
        With a real model, would call BatchGenerator.next().

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
            elif self.model is not None:
                # Real model path (to be integrated with BatchGenerator)
                token_id, token_text, finish_reason = self._generate_with_model(
                    seq, step
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

    def _generate_with_model(self, seq: SequenceState, step: int) -> tuple[int, str, str | None]:
        """Generate a token using the real model (placeholder for Phase 3 integration)."""
        # This would use BatchGenerator. For now, a stub.
        raise NotImplementedError(
            "Real model generation requires BatchGenerator integration"
        )

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
                    return "stop"

        return None

    # --- Token Delivery ---

    def _emit_tokens(self, events: list[TokenEvent]) -> None:
        """Deliver token events to streams and result buffers."""
        for event in events:
            rid = event.request_id

            # Deliver to stream if registered
            with self._streams_lock:
                stream = self._streams.get(rid)
            if stream is not None:
                stream.put(event)
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
            stream.put(event)

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
        """Return cache statistics for the /health endpoint."""
        stats: dict[str, Any] = {
            "active_sequences": self.num_active_sequences,
            "queued_requests": self.num_queued_requests,
        }
        if self.kv_cache_manager is not None:
            stats["total_blocks"] = self.kv_cache_manager.pool.num_blocks
            stats["used_blocks"] = self.kv_cache_manager.num_used_blocks
            stats["free_blocks"] = self.kv_cache_manager.num_free_blocks
            stats["cached_blocks"] = self.kv_cache_manager.num_cached_blocks
        return stats

    def shutdown(self) -> None:
        """Graceful shutdown — stops the inference loop."""
        self.stop()

    @property
    def is_running(self) -> bool:
        """Whether the inference loop is running."""
        return self._running
