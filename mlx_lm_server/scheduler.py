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

        # BatchGenerator state (real model path)
        self._batch_generator: Any = None
        self._uid_to_request_id: dict[int, str] = {}
        self._request_id_to_uid: dict[str, int] = {}
        self._sequence_cache: SequenceCacheStore | None = None

        # Tiered cache (RAM + SSD) — set by __main__.py or tests
        self._tiered_cache: Any = None

        # SSD pruning counter (prune every N inference steps)
        self._ssd_prune_interval: int = 1000
        self._ssd_prune_counter: int = 0

        # Initialize BatchGenerator if model is available
        if self.model is not None and BatchGenerator is not None:
            self._create_batch_generator()
            self._sequence_cache = SequenceCacheStore(
                max_entries=config.sequence_cache_size
            )

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

        The request is added to the queue and the inference loop is notified.
        Only creates _results/_results_ready entries for non-streaming requests
        to prevent resource leaks (streaming requests use register_stream instead).
        """
        self.request_queue.add(request)

        # Set up result storage only for non-streaming requests
        if not getattr(request, "stream", False):
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
        if self._batch_generator is not None:
            try:
                self._batch_generator.close()
            except Exception:
                pass

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

        logger.info("Inference loop stopped")

    def _mock_inference_step(self) -> None:
        """One iteration of the mock inference loop (model=None)."""
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

    def _batch_inference_step(self) -> None:
        """One step of the batch inference loop using BatchGenerator."""
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

        # 7. Store finished caches in SequenceCacheStore and decompose to blocks
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
            if self._sequence_cache is not None:
                self._sequence_cache.store(prompt_tokens, prompt_cache)

            # Decompose into block-level cache for finer-grained sharing
            if self.kv_cache_manager is not None:
                try:
                    block_dicts = decompose_cache_to_blocks(
                        prompt_cache, prompt_tokens,
                        self.kv_cache_manager.block_size,
                    )
                    for bd in block_dicts:
                        bh = bd['block_hash']
                        with self.kv_cache_manager.lock:
                            if bh not in self.kv_cache_manager.hash_table:
                                try:
                                    block = self.kv_cache_manager.pool.get_free_block()
                                except Exception:
                                    if self._tiered_cache is not None:
                                        self._tiered_cache.evict_to_ssd(num_blocks=1)
                                    else:
                                        self.kv_cache_manager._evict_lru_locked(num_blocks=1)
                                    try:
                                        block = self.kv_cache_manager.pool.get_free_block()
                                    except Exception:
                                        continue
                                block.block_hash = bh
                                block.token_ids = bd['token_ids']
                                block.ref_count = 0
                                block.last_accessed = time.time()
                                block.kv_data = list(bd['kv_data_per_layer'])
                                self.kv_cache_manager.hash_table[bh] = block.block_id
                except Exception as e:
                    logger.debug("Failed to decompose cache to blocks: %s", e)

        # 8. Periodic SSD pruning (P7.4)
        self._ssd_prune_counter += 1
        if self._ssd_prune_counter >= self._ssd_prune_interval:
            self._ssd_prune_counter = 0
            if self._tiered_cache is not None and self._tiered_cache.ssd is not None:
                try:
                    pruned = self._tiered_cache.ssd.prune_expired()
                    if pruned > 0:
                        logger.info("Pruned %d expired SSD blocks", pruned)
                except Exception as e:
                    logger.warning("SSD pruning failed: %s", e)

        # 9. Clean up finished sequences
        self._cleanup_finished_batch()

    def _process_cancellations_batch(self) -> None:
        """Remove cancelled requests from BatchGenerator."""
        with self._cancelled_lock:
            cancelled = set(self._cancelled)

        uids_to_remove = []
        for rid in cancelled:
            uid = self._request_id_to_uid.get(rid)
            if uid is not None:
                uids_to_remove.append(uid)

        if uids_to_remove:
            try:
                self._batch_generator.remove(uids_to_remove)
            except Exception as e:
                logger.warning("Failed to remove cancelled UIDs: %s", e)

        # Clean up mappings and signal finish
        for rid in cancelled:
            uid = self._request_id_to_uid.pop(rid, None)
            if uid is not None:
                self._uid_to_request_id.pop(uid, None)
            with self._active_lock:
                seq = self._active_sequences.pop(rid, None)
            if seq is not None:
                seq.is_finished = True
                seq.finish_reason = "cancelled"
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
            seq = self._init_sequence(req)

            # Set up detokenizer
            if self.tokenizer is not None:
                detok = getattr(self.tokenizer, "detokenizer", None)
                if detok is not None:
                    seq._detokenizer = copy.copy(detok)
                    seq._detokenizer.reset()

            with self._active_lock:
                self._active_sequences[req.request_id] = seq

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
                    # Try to reconstruct cache from blocks
                    try:
                        block_data = []
                        block_size = self.kv_cache_manager.block_size
                        for i in range(len(block_ids)):
                            block = self.kv_cache_manager.pool.blocks[block_ids[i]]
                            if block.kv_data is not None:
                                block_data.append(block.kv_data)
                        if block_data:
                            cache = reconstruct_cache_from_blocks(
                                [{'kv_data_per_layer': [d] if not isinstance(d, list) else d}
                                 for d in block_data],
                                self.model,
                            )
                            logger.debug(
                                "Block cache hit for %s: %d tokens cached",
                                req.request_id, num_cached,
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to reconstruct cache from blocks for %s: %s",
                            req.request_id, e,
                        )
                        cache = None

            # Sequence-level cache lookup (fallback)
            if cache is None and self._sequence_cache is not None:
                cached, remaining = self._sequence_cache.find_longest_prefix(seq.token_ids)
                if cached is not None:
                    cache = cached
                    remaining_tokens = remaining if remaining else seq.token_ids
                    logger.debug(
                        "Sequence cache hit for %s: %d tokens cached",
                        req.request_id,
                        len(seq.token_ids) - len(remaining),
                    )

            # Create sampler
            sampler = None
            if make_sampler is not None:
                sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

            # Handle max_tokens=0 gracefully
            if req.max_tokens <= 0:
                seq.is_finished = True
                seq.finish_reason = "length"
                self._signal_finish(req.request_id, finish_reason="length")
                continue

            # Insert into BatchGenerator
            try:
                insert_kwargs = {
                    "prompts": [seq.token_ids],
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
                logger.debug(
                    "Inserted request %s as UID %d", req.request_id, uid
                )
            except Exception as e:
                logger.error("Failed to insert request %s: %s", req.request_id, e)
                seq.is_finished = True
                seq.finish_reason = "error"
                self._signal_finish(req.request_id, finish_reason="error")

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

    def _handle_mock_error(self) -> None:
        """Handle error in mock inference path (existing behavior)."""
        with self._active_lock:
            for seq in self._active_sequences.values():
                if not seq.is_finished:
                    seq.is_finished = True
                    seq.finish_reason = "error"
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

        In mock mode (model=None), marks all prompt tokens as computed.
        Real model prefill is handled by BatchGenerator in _batch_inference_step.
        """
        for seq in sequences:
            # Mock mode: mark all prompt tokens as computed (prefill complete)
            seq.num_computed_tokens = len(seq.token_ids)
            logger.debug(
                "Prefill complete for %s (%d tokens)",
                seq.request_id,
                seq.num_computed_tokens,
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
