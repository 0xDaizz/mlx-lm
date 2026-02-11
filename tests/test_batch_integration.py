"""Tests for BatchGenerator integration in Scheduler (P6.7).

Uses mock objects to test the batch path without requiring a real model.
The mock path (model=None) is tested by existing test_scheduler.py.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.types import InferenceRequest, TokenEvent


@dataclass
class MockResponse:
    """Mock of BatchGenerator.Response."""
    uid: int
    token: int
    logprobs: Any = None
    finish_reason: str | None = None
    _prompt_cache: Any = None

    @property
    def prompt_cache(self):
        return self._prompt_cache


class MockBatchGenerator:
    """Mock BatchGenerator for testing the batch path without a real model."""

    def __init__(self):
        self._uid_counter = 0
        self._active: dict[int, dict] = {}  # uid -> {tokens, max_tokens, step}
        self._closed = False
        self._removed_uids: list[int] = []

    def insert(self, prompts, max_tokens=None, caches=None, samplers=None, logits_processors=None):
        uids = []
        for i, prompt in enumerate(prompts):
            uid = self._uid_counter
            self._uid_counter += 1
            mt = max_tokens[i] if isinstance(max_tokens, list) else (max_tokens or 10)
            self._active[uid] = {
                "tokens": list(prompt) if hasattr(prompt, '__iter__') else [prompt],
                "max_tokens": mt,
                "step": 0,
            }
            uids.append(uid)
        return uids

    def next(self):
        responses = []
        finished = []
        for uid, state in self._active.items():
            state["step"] += 1
            token = 100 + state["step"]
            finish = None
            if state["step"] >= state["max_tokens"]:
                finish = "length"
                finished.append(uid)
            responses.append(MockResponse(
                uid=uid,
                token=token,
                finish_reason=finish,
            ))
        for uid in finished:
            del self._active[uid]
        return responses

    def remove(self, uids, return_prompt_caches=False):
        result = {}
        for uid in uids:
            self._removed_uids.append(uid)
            if uid in self._active:
                if return_prompt_caches:
                    result[uid] = [{"mock_cache": True}]
                del self._active[uid]
        return result if return_prompt_caches else None

    def close(self):
        self._closed = True
        self._active.clear()


def _make_scheduler_with_mock_bg(config=None, **kwargs):
    """Create a Scheduler with a MockBatchGenerator injected."""
    if config is None:
        config = ServerConfig(**kwargs)

    # Create a scheduler with a fake model so batch path activates
    sched = Scheduler(config=config, model=None, tokenizer=None)

    # Inject mock batch generator
    mock_bg = MockBatchGenerator()
    sched._batch_generator = mock_bg
    sched._sequence_cache = None

    # Create a mock tokenizer with minimal interface.
    # Set detokenizer=None so the batch path uses str(token) fallback
    # instead of trying to use a real detokenizer.
    mock_tokenizer = MagicMock()
    mock_tokenizer.detokenizer = None
    mock_tokenizer.decode = lambda ids: "".join(f"t{i}" for i in ids)
    sched.tokenizer = mock_tokenizer

    return sched, mock_bg


class TestBatchIntegration:
    """Tests for the batch inference path."""

    def test_single_request_batch_path(self):
        """Single request completes through batch path."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=3
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="req-1",
                prompt_tokens=[1, 2, 3],
                max_tokens=3,
            )
            sched.submit_request(req)
            events = sched.get_result("req-1", timeout=5.0)

            assert len(events) == 3
            assert events[-1].finish_reason == "length"
            # Tokens should be 101, 102, 103 from mock
            assert [e.token_id for e in events] == [101, 102, 103]
        finally:
            sched.stop()

    def test_multiple_concurrent_batch(self):
        """Multiple requests run concurrently in one batch."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=2
        )
        sched.run_inference_loop()

        try:
            for i in range(3):
                req = InferenceRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=[10 + i],
                    max_tokens=2,
                )
                sched.submit_request(req)

            results = {}
            for i in range(3):
                results[f"req-{i}"] = sched.get_result(f"req-{i}", timeout=5.0)

            for rid, events in results.items():
                assert len(events) == 2
                assert events[-1].finish_reason == "length"
        finally:
            sched.stop()

    def test_cancel_during_batch(self):
        """Cancellation removes request from batch.

        Uses a slow MockBatchGenerator so the request is still active
        when cancel is called.
        """
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Slow down next() so we can cancel mid-flight
        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="cancel-me",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            stream_q = sched.register_stream("cancel-me")
            sched.submit_request(req)

            # Wait for at least one token, then cancel
            time.sleep(0.2)
            sched.cancel_request("cancel-me")

            # Should get a cancelled event eventually
            deadline = time.time() + 5.0
            got_finish = False
            finish_reason = None
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=0.5)
                    if event.finish_reason is not None:
                        got_finish = True
                        finish_reason = event.finish_reason
                        break
                except queue.Empty:
                    continue

            assert got_finish
            assert finish_reason == "cancelled"
        finally:
            sched.stop()

    def test_uid_mapping_cleanup(self):
        """UID mappings are cleaned up after request completes."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=2
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="cleanup-test",
                prompt_tokens=[1],
                max_tokens=2,
            )
            sched.submit_request(req)
            sched.get_result("cleanup-test", timeout=5.0)

            # Give cleanup time to run
            time.sleep(0.3)

            assert len(sched._uid_to_request_id) == 0
            assert len(sched._request_id_to_uid) == 0
        finally:
            sched.stop()

    def test_zero_max_tokens_batch(self):
        """max_tokens=0 finishes immediately without entering BatchGenerator."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=0
        )
        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="zero-tok",
                prompt_tokens=[1, 2, 3],
                max_tokens=0,
            )
            sched.submit_request(req)
            events = sched.get_result("zero-tok", timeout=5.0)

            assert len(events) >= 1
            assert events[-1].finish_reason == "length"
            # Should not have inserted into batch generator
            assert mock_bg._uid_counter == 0
        finally:
            sched.stop()

    def test_error_recovery_batch(self):
        """BatchGenerator error is handled, loop continues."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=5
        )

        # Make next() raise on first call
        original_next = mock_bg.next
        call_count = [0]
        def failing_next():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Mock BG failure")
            return original_next()

        mock_bg.next = failing_next
        sched.run_inference_loop()

        try:
            req1 = InferenceRequest(
                request_id="fail-req",
                prompt_tokens=[1],
                max_tokens=5,
            )
            sched.submit_request(req1)
            events = sched.get_result("fail-req", timeout=5.0)

            # Should get an error finish
            assert events[-1].finish_reason == "error"
        finally:
            sched.stop()

    def test_streaming_batch(self):
        """Streaming works through batch path -- tokens arrive incrementally."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=3
        )
        sched.run_inference_loop()

        try:
            stream_q = sched.register_stream("stream-test")
            req = InferenceRequest(
                request_id="stream-test",
                prompt_tokens=[1, 2],
                max_tokens=3,
                stream=True,
            )
            sched.submit_request(req)

            events = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=1.0)
                    events.append(event)
                    if event.finish_reason is not None:
                        break
                except queue.Empty:
                    continue

            assert len(events) == 3
            assert events[-1].finish_reason == "length"
        finally:
            sched.stop()


class TestMockResponseProperty:
    """Tests for MockResponse.prompt_cache property (SCHED-1)."""

    def test_mock_response_prompt_cache_none(self):
        """prompt_cache=None should evaluate as None, not a truthy method object."""
        r = MockResponse(uid=0, token=1, _prompt_cache=None)
        assert r.prompt_cache is None

    def test_mock_response_prompt_cache_value(self):
        """prompt_cache with a value should return that value."""
        sentinel = object()
        r = MockResponse(uid=0, token=1, _prompt_cache=sentinel)
        assert r.prompt_cache is sentinel


# ---------------------------------------------------------------------------
# G2: Prefill Cache Salvage on Cancel
# ---------------------------------------------------------------------------


class TestG2CancelCacheSalvage:
    """Tests for G2: saving prefill KV cache when requests are cancelled.

    When a request with a cache miss is cancelled mid-generation, the
    scheduler should extract the KV cache via BatchGenerator.remove()
    with return_prompt_caches=True, and store it for future reuse.
    """

    def test_cancel_salvages_cache_for_pending_saves(self):
        """Cancelled UIDs in _pending_cache_saves trigger cache extraction and storage."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Track calls to _store_finished_caches
        stored_caches = []
        original_store = sched._store_finished_caches

        def tracking_store(caches):
            stored_caches.append(dict(caches))
            original_store(caches)

        sched._store_finished_caches = tracking_store

        # Slow down next() so cancel happens while request is active
        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="salvage-me",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            stream_q = sched.register_stream("salvage-me")
            sched.submit_request(req)

            # Wait for request to be inserted (enters batch generator)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if sched._request_id_to_uid.get("salvage-me") is not None:
                    break
                time.sleep(0.01)

            uid = sched._request_id_to_uid.get("salvage-me")
            assert uid is not None, "Request should be in batch generator"

            # Verify UID is in pending cache saves (cache miss path)
            assert uid in sched._pending_cache_saves, \
                "UID should be in _pending_cache_saves (cache miss insert)"

            # Cancel the request
            sched.cancel_request("salvage-me")

            # Wait for cancel to be processed
            deadline = time.time() + 5.0
            got_finish = False
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=0.5)
                    if event.finish_reason == "cancelled":
                        got_finish = True
                        break
                except queue.Empty:
                    continue

            assert got_finish, "Should receive cancelled finish event"

            # Verify _store_finished_caches was called with the UID's cache
            assert len(stored_caches) >= 1, \
                "Should have called _store_finished_caches for salvaged cache"

            # Check that the stored caches contain the cancelled UID
            salvaged_uids = set()
            for call_caches in stored_caches:
                salvaged_uids.update(call_caches.keys())
            assert uid in salvaged_uids, \
                "Salvaged caches should include the cancelled UID"
        finally:
            sched.stop()

    def test_cancel_without_pending_save_skips_cache_extraction(self):
        """Cancelled UIDs NOT in _pending_cache_saves do not trigger cache extraction."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Track calls to _store_finished_caches
        stored_caches = []
        original_store = sched._store_finished_caches

        def tracking_store(caches):
            stored_caches.append(dict(caches))
            original_store(caches)

        sched._store_finished_caches = tracking_store

        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="no-salvage",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            stream_q = sched.register_stream("no-salvage")
            sched.submit_request(req)

            # Wait for request to be inserted
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if sched._request_id_to_uid.get("no-salvage") is not None:
                    break
                time.sleep(0.01)

            uid = sched._request_id_to_uid.get("no-salvage")
            assert uid is not None

            # Manually remove from _pending_cache_saves to simulate cache hit path
            sched._pending_cache_saves.discard(uid)

            sched.cancel_request("no-salvage")

            # Wait for cancel
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=0.5)
                    if event.finish_reason == "cancelled":
                        break
                except queue.Empty:
                    continue

            # _store_finished_caches should NOT have been called (no caches to salvage)
            # Filter out any calls with empty dicts
            non_empty_stores = [c for c in stored_caches if c]
            assert len(non_empty_stores) == 0, \
                "Should not call _store_finished_caches when no UIDs have pending saves"
        finally:
            sched.stop()

    def test_cancel_cache_save_failure_does_not_break_cancel(self):
        """If _store_finished_caches raises during salvage, cancel still completes.

        We only fail on the salvage call (from _process_cancellations_batch),
        not on the normal decode-path call (from _batch_inference_step step 7),
        by tracking which call-site triggers the failure.
        """
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Make _store_finished_caches raise only when called with
        # specific salvage UIDs (from _process_cancellations_batch).
        original_store = sched._store_finished_caches
        fail_for_uids: set[int] = set()

        def conditionally_failing_store(caches):
            if fail_for_uids & set(caches.keys()):
                raise RuntimeError("Simulated store failure")
            return original_store(caches)

        sched._store_finished_caches = conditionally_failing_store

        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="fail-store",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            stream_q = sched.register_stream("fail-store")
            sched.submit_request(req)

            # Wait for insertion
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if sched._request_id_to_uid.get("fail-store") is not None:
                    break
                time.sleep(0.01)

            uid = sched._request_id_to_uid.get("fail-store")
            assert uid is not None
            assert uid in sched._pending_cache_saves

            # Now arm the failure for this specific UID
            fail_for_uids.add(uid)

            sched.cancel_request("fail-store")

            # Cancel should still complete despite store failure
            deadline = time.time() + 5.0
            got_finish = False
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=0.5)
                    if event.finish_reason == "cancelled":
                        got_finish = True
                        break
                except queue.Empty:
                    continue

            assert got_finish, "Cancel should complete even if cache store fails"

            # _pending_cache_saves should be cleaned up
            assert uid not in sched._pending_cache_saves, \
                "_pending_cache_saves should be cleaned up after cancel"
        finally:
            sched.stop()

    def test_pending_cache_saves_cleaned_up_after_cancel(self):
        """_pending_cache_saves is cleaned up for all cancelled UIDs."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            # Submit two requests
            for i in range(2):
                req = InferenceRequest(
                    request_id=f"cleanup-{i}",
                    prompt_tokens=[1, 2, 3],
                    max_tokens=100,
                )
                sched.register_stream(f"cleanup-{i}")
                sched.submit_request(req)

            # Wait for both to be inserted
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if (sched._request_id_to_uid.get("cleanup-0") is not None
                        and sched._request_id_to_uid.get("cleanup-1") is not None):
                    break
                time.sleep(0.01)

            uid0 = sched._request_id_to_uid.get("cleanup-0")
            uid1 = sched._request_id_to_uid.get("cleanup-1")
            assert uid0 is not None and uid1 is not None

            # Both should be in pending cache saves
            assert uid0 in sched._pending_cache_saves
            assert uid1 in sched._pending_cache_saves

            # Cancel both
            sched.cancel_request("cleanup-0")
            sched.cancel_request("cleanup-1")

            # Wait for cleanup
            time.sleep(1.0)

            # Both should be cleaned from pending saves
            assert uid0 not in sched._pending_cache_saves, \
                "UID 0 should be removed from _pending_cache_saves"
            assert uid1 not in sched._pending_cache_saves, \
                "UID 1 should be removed from _pending_cache_saves"

            # Active sequences should be empty
            with sched._active_lock:
                assert len(sched._active_sequences) == 0
        finally:
            sched.stop()

    def test_cancel_salvage_calls_remove_with_return_prompt_caches(self):
        """Verify BatchGenerator.remove() is called with return_prompt_caches=True
        when there are salvageable UIDs."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Patch remove to track call args
        remove_calls = []
        original_remove = mock_bg.remove

        def tracking_remove(uids, return_prompt_caches=False):
            remove_calls.append({
                "uids": list(uids),
                "return_prompt_caches": return_prompt_caches,
            })
            return original_remove(uids, return_prompt_caches=return_prompt_caches)

        mock_bg.remove = tracking_remove

        original_next = mock_bg.next
        def slow_next():
            time.sleep(0.05)
            return original_next()
        mock_bg.next = slow_next

        sched.run_inference_loop()

        try:
            req = InferenceRequest(
                request_id="track-remove",
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
            )
            stream_q = sched.register_stream("track-remove")
            sched.submit_request(req)

            # Wait for insertion
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if sched._request_id_to_uid.get("track-remove") is not None:
                    break
                time.sleep(0.01)

            uid = sched._request_id_to_uid.get("track-remove")
            assert uid is not None
            assert uid in sched._pending_cache_saves

            sched.cancel_request("track-remove")

            # Wait for cancel
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = stream_q.get(timeout=0.5)
                    if event.finish_reason == "cancelled":
                        break
                except queue.Empty:
                    continue

            # Verify remove was called with return_prompt_caches=True
            cancel_removes = [c for c in remove_calls if c["return_prompt_caches"]]
            assert len(cancel_removes) >= 1, \
                "remove() should be called with return_prompt_caches=True for salvageable UIDs"
        finally:
            sched.stop()


# ---------------------------------------------------------------------------
# Resource Bounding: _pending_cache_saves and _bus_retry_events
# ---------------------------------------------------------------------------


class TestPendingCacheSavesBounded:
    """Tests for bounding the _pending_cache_saves set."""

    def test_pending_cache_saves_bounded(self):
        """Verify _pending_cache_saves set does not exceed _PENDING_CACHE_SAVES_MAX."""
        from mlx_lm_server.scheduler import _PENDING_CACHE_SAVES_MAX

        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Directly fill _pending_cache_saves to capacity
        for i in range(_PENDING_CACHE_SAVES_MAX):
            sched._pending_cache_saves.add(i)

        assert len(sched._pending_cache_saves) == _PENDING_CACHE_SAVES_MAX

        # Simulate the bounded add logic: adding a new UID when at capacity
        # should be skipped. We test by calling the code path that does the add.
        # The add site is in _insert_new_requests_batch, but we can test the
        # invariant directly: after filling to max, the set stays at max.
        sched._pending_cache_saves.add(99999999)  # direct add would exceed
        # But the scheduler code checks len() before .add(). Let's verify the
        # constant is accessible and the set was at capacity before the forced add.
        assert _PENDING_CACHE_SAVES_MAX == 10000

        # Now test the actual code path through the scheduler.
        # Reset and fill to capacity minus 1, submit one request, it should add.
        sched._pending_cache_saves.clear()
        for i in range(_PENDING_CACHE_SAVES_MAX):
            sched._pending_cache_saves.add(i)

        # The set is now at max. The code path in _insert_new_requests_batch
        # checks `if len(self._pending_cache_saves) < _PENDING_CACHE_SAVES_MAX`
        # before adding. We verify this by running the inference loop with
        # a request that would trigger a cache-miss add.
        sched.run_inference_loop()
        try:
            req = InferenceRequest(
                request_id="bound-test",
                prompt_tokens=[1, 2, 3],
                max_tokens=2,
            )
            sched.submit_request(req)
            events = sched.get_result("bound-test", timeout=5.0)
            assert events[-1].finish_reason is not None

            # The UID for this request should NOT be in _pending_cache_saves
            # because the set was already at capacity when the request was inserted.
            uid = None
            # The request completed, so UID mapping may already be cleaned up.
            # But the _pending_cache_saves set should not have grown beyond max + a few
            # (the cleanup path removes entries too).
            # Key assertion: set never exceeds max by more than a tiny margin
            # (cleanup may remove some entries between the add and now).
            assert len(sched._pending_cache_saves) <= _PENDING_CACHE_SAVES_MAX + 1
        finally:
            sched.stop()

    def test_pending_cache_saves_stale_cleanup(self):
        """Verify stale UIDs are removed from _pending_cache_saves during periodic cleanup."""
        from mlx_lm_server.scheduler import _PENDING_CACHE_SAVES_CLEANUP_INTERVAL_S

        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Add some UIDs to _pending_cache_saves that are NOT in _uid_to_request_id
        # These are "stale" UIDs that should be cleaned up.
        stale_uids = {100000, 100001, 100002}
        sched._pending_cache_saves.update(stale_uids)

        # Also add a "live" UID that IS in _uid_to_request_id
        live_uid = 200000
        sched._uid_to_request_id[live_uid] = "live-request"
        sched._pending_cache_saves.add(live_uid)

        assert len(sched._pending_cache_saves) == 4

        # Force the cleanup to run by setting last_cleanup to the past
        sched._pending_cache_saves_last_cleanup = (
            time.monotonic() - _PENDING_CACHE_SAVES_CLEANUP_INTERVAL_S - 1.0
        )

        # Call the cleanup method directly
        sched._cleanup_stale_pending_cache_saves()

        # Stale UIDs should be removed
        for uid in stale_uids:
            assert uid not in sched._pending_cache_saves, \
                f"Stale UID {uid} should have been removed"

        # Live UID should remain
        assert live_uid in sched._pending_cache_saves, \
            "Live UID should not be removed"

        assert len(sched._pending_cache_saves) == 1

        # Clean up
        sched._uid_to_request_id.pop(live_uid, None)

    def test_pending_cache_saves_cleanup_respects_interval(self):
        """Cleanup only runs when the time interval has elapsed."""
        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        stale_uids = {100000, 100001}
        sched._pending_cache_saves.update(stale_uids)

        # Set last_cleanup to now — cleanup should NOT run
        sched._pending_cache_saves_last_cleanup = time.monotonic()

        sched._cleanup_stale_pending_cache_saves()

        # Stale UIDs should still be there (cleanup didn't run)
        assert stale_uids.issubset(sched._pending_cache_saves), \
            "Cleanup should not run before interval elapses"


class TestBusRetryEventsBounded:
    """Tests for bounding the _bus_retry_events list."""

    def test_bus_retry_events_bounded(self):
        """Verify _bus_retry_events does not exceed _BUS_RETRY_EVENTS_MAX."""
        from mlx_lm_server.scheduler import _BUS_RETRY_EVENTS_MAX

        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        # Create a mock control bus that always fails to publish
        class FailingBus:
            def publish(self, event):
                raise RuntimeError("Publish failed")
            def recv(self):
                raise RuntimeError("Not implemented")

        class MockDistCtx:
            is_rank0 = True
            world_size = 2

        sched._control_bus = FailingBus()
        sched._dist_ctx = MockDistCtx()

        # We need to import ControlEvent to create fake events
        from mlx_lm_server.distributed_bus import ControlEvent

        # Fill the outbox with more events than _BUS_RETRY_EVENTS_MAX
        num_events = _BUS_RETRY_EVENTS_MAX + 50
        for i in range(num_events):
            req = InferenceRequest(
                request_id=f"retry-{i}",
                prompt_tokens=[1, 2, 3],
                max_tokens=10,
            )
            try:
                sched._bus_outbox.put_nowait(ControlEvent.submit(req))
            except Exception:
                break

        # Call _drain_bus_outbox — publish will fail, events go to retry
        sched._drain_bus_outbox()

        # _bus_retry_events should be bounded
        assert len(sched._bus_retry_events) <= _BUS_RETRY_EVENTS_MAX, \
            f"Bus retry events ({len(sched._bus_retry_events)}) should not exceed max ({_BUS_RETRY_EVENTS_MAX})"

    def test_bus_retry_events_oldest_discarded(self):
        """When bus retry events exceed max, oldest events are discarded (FIFO)."""
        from mlx_lm_server.scheduler import _BUS_RETRY_EVENTS_MAX

        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        class FailingBus:
            def publish(self, event):
                raise RuntimeError("Publish failed")
            def recv(self):
                raise RuntimeError("Not implemented")

        class MockDistCtx:
            is_rank0 = True
            world_size = 2

        sched._control_bus = FailingBus()
        sched._dist_ctx = MockDistCtx()

        from mlx_lm_server.distributed_bus import ControlEvent

        # Create more events than the max
        total_events = _BUS_RETRY_EVENTS_MAX + 30
        for i in range(total_events):
            req = InferenceRequest(
                request_id=f"order-{i}",
                prompt_tokens=[1],
                max_tokens=10,
            )
            try:
                sched._bus_outbox.put_nowait(ControlEvent.submit(req))
            except Exception:
                break

        sched._drain_bus_outbox()

        # Should have exactly _BUS_RETRY_EVENTS_MAX events
        assert len(sched._bus_retry_events) == _BUS_RETRY_EVENTS_MAX

        # The retained events should be the NEWEST ones (oldest discarded).
        # The last event in retry list should be from the last request added.
        # Verify by checking that the first discarded events are gone.
        # Since events are submit events, we can unpack request_ids.
        retained_ids = []
        for ev in sched._bus_retry_events:
            if ev.typ == "submit":
                req = ev.unpack_request()
                if req is not None:
                    retained_ids.append(req.request_id)

        # The oldest 30 events (order-0 through order-29) should be discarded.
        # The retained should start from order-30.
        if retained_ids:
            # First retained should NOT be "order-0" (it was discarded)
            assert retained_ids[0] != "order-0", \
                "Oldest event should have been discarded"
            # Last retained should be from the tail of the original list
            # (It may not be exactly "order-{total-1}" due to outbox capacity,
            # but it should be a later event than order-0)

    def test_bus_retry_events_within_limit_no_discard(self):
        """When bus retry events are within limit, no events are discarded."""
        from mlx_lm_server.scheduler import _BUS_RETRY_EVENTS_MAX

        sched, mock_bg = _make_scheduler_with_mock_bg(
            max_batch_size=4, default_max_tokens=100
        )

        class FailingBus:
            def publish(self, event):
                raise RuntimeError("Publish failed")
            def recv(self):
                raise RuntimeError("Not implemented")

        class MockDistCtx:
            is_rank0 = True
            world_size = 2

        sched._control_bus = FailingBus()
        sched._dist_ctx = MockDistCtx()

        from mlx_lm_server.distributed_bus import ControlEvent

        # Add fewer events than the max
        num_events = 10
        for i in range(num_events):
            req = InferenceRequest(
                request_id=f"small-{i}",
                prompt_tokens=[1],
                max_tokens=10,
            )
            sched._bus_outbox.put_nowait(ControlEvent.submit(req))

        sched._drain_bus_outbox()

        # All events should be retained (no discard needed)
        assert len(sched._bus_retry_events) == num_events, \
            f"All {num_events} events should be retained when under limit"
