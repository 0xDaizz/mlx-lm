"""Integration tests (Phase 4).

End-to-end tests that exercise the full stack:
- P4.2: Basic scheduler E2E (request -> generate -> result)
- P4.3: Prefix cache E2E (shared prefix detection)
- P4.4: SSD tier E2E (allocate -> evict -> SSD lookup)
- P4.5: Concurrent requests E2E (multiple requests in parallel)

All tests use mocks -- no real model loading required.
Run: .venv/bin/python -m pytest tests/test_integration.py -v --tb=short
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.kv_cache_manager import KVCacheManager, TieredKVCache
from mlx_lm_server.scheduler import Scheduler
from mlx_lm_server.ssd_cache import SSDCache
from mlx_lm_server.types import InferenceRequest, TokenEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str = "req-1",
    prompt_tokens: list[int] | None = None,
    max_tokens: int = 5,
    stream: bool = False,
    stop_sequences: list[str] | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens or [1, 2, 3, 4],
        max_tokens=max_tokens,
        stop_sequences=stop_sequences or [],
        stream=stream,
    )


def _make_config(tmp_path: Path | None = None, **overrides) -> ServerConfig:
    defaults = dict(
        block_size=4,
        num_blocks=64,
        max_batch_size=8,
        max_queue_size=32,
        prefill_batch_size=4,
    )
    if tmp_path is not None:
        defaults["ssd_cache_dir"] = tmp_path / "ssd-cache"
    defaults.update(overrides)
    return ServerConfig(**defaults)


# ---------------------------------------------------------------------------
# P4.2 — E2E Basic Test
# ---------------------------------------------------------------------------


class TestE2EBasic:
    """Submit a request to the scheduler, get result back.

    Verifies the complete flow: request -> scheduler -> generate tokens -> result.
    """

    def test_e2e_basic(self):
        """Full E2E: submit request -> scheduler processes -> tokens returned."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Mock generator: produce 5 tokens then stop
        def mock_gen(request_id, token_ids, step):
            if step >= 4:
                return (step + 10, f"word{step}", "stop")
            return (step + 10, f"word{step}", None)

        scheduler._mock_generate = mock_gen

        # Submit a request
        req = _make_request("e2e-basic", prompt_tokens=[100, 200, 300, 400], max_tokens=10)
        scheduler.submit_request(req)

        # Run the inference loop in background
        scheduler.run_inference_loop(blocking=False)
        try:
            # Wait for result
            tokens = scheduler.get_result("e2e-basic", timeout=5.0)

            # Verify we got 5 tokens (steps 0-4)
            assert len(tokens) == 5, f"Expected 5 tokens, got {len(tokens)}"

            # Verify token content
            assert tokens[0].token_id == 10
            assert tokens[0].token_text == "word0"
            assert tokens[0].finish_reason is None

            assert tokens[4].token_id == 14
            assert tokens[4].token_text == "word4"
            assert tokens[4].finish_reason == "stop"

            # Verify all tokens belong to the right request
            for t in tokens:
                assert t.request_id == "e2e-basic"

        finally:
            scheduler.stop()

    def test_e2e_basic_max_tokens(self):
        """E2E: request respects max_tokens and stops with 'length' reason."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Never return a finish reason -- let max_tokens handle it
        scheduler._mock_generate = lambda rid, tids, step: (step + 1, f"t{step}", None)

        req = _make_request("e2e-maxlen", max_tokens=3)
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            tokens = scheduler.get_result("e2e-maxlen", timeout=5.0)
            assert len(tokens) == 3
            assert tokens[-1].finish_reason == "length"
        finally:
            scheduler.stop()

    def test_e2e_basic_streaming(self):
        """E2E: streaming delivers tokens incrementally via queue."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 2:
                return (step + 50, f"s{step}", "stop")
            return (step + 50, f"s{step}", None)

        scheduler._mock_generate = mock_gen

        req = _make_request("e2e-stream", max_tokens=10, stream=True)
        stream_q = scheduler.register_stream("e2e-stream")
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            events: list[TokenEvent] = []
            while True:
                ev = stream_q.get(timeout=5.0)
                events.append(ev)
                if ev.finish_reason is not None:
                    break

            assert len(events) == 3
            assert events[0].token_text == "s0"
            assert events[-1].finish_reason == "stop"
        finally:
            scheduler.stop()


# ---------------------------------------------------------------------------
# P4.3 — E2E Prefix Cache Test
# ---------------------------------------------------------------------------


class TestE2EPrefix:
    """Send two requests with shared prefix tokens.

    Verifies that find_cached_prefix() finds the shared prefix on the
    second request.
    """

    def test_e2e_prefix_cache_hit(self):
        """Two requests sharing a prefix: second request finds cached blocks."""
        config = _make_config(block_size=4, num_blocks=32)
        kv_mgr = KVCacheManager(config)

        # Shared prefix: tokens [1,2,3,4,5,6,7,8] (2 blocks of 4)
        shared_prefix = [1, 2, 3, 4, 5, 6, 7, 8]

        # Request A tokens: shared prefix + unique suffix
        tokens_a = shared_prefix + [10, 11, 12, 13]
        # Request B tokens: shared prefix + different suffix
        tokens_b = shared_prefix + [20, 21, 22, 23]

        # Before any allocation, no prefix should be cached
        assert kv_mgr.find_cached_prefix(tokens_a) == 0
        assert kv_mgr.find_cached_prefix(tokens_b) == 0

        # Allocate blocks for request A
        ids_a = kv_mgr.allocate_blocks(tokens_a)
        assert len(ids_a) == 3  # 12 tokens / 4 block_size = 3 blocks

        # Now check prefix cache for request B (shares first 2 blocks)
        cached_b = kv_mgr.find_cached_prefix(tokens_b)
        assert cached_b == 8, f"Expected 8 cached tokens, got {cached_b}"

        # Allocate ALL blocks for B (no num_existing_blocks skip).
        # This way, the allocator sees the shared blocks in the hash table
        # and increments their ref_count (cache hit path).
        ids_b = kv_mgr.allocate_blocks(tokens_b)
        assert len(ids_b) == 3  # 2 shared (cache hit) + 1 unique (new alloc)

        # The shared blocks should be the same block_ids (cache hit reuse)
        assert ids_a[0] == ids_b[0]
        assert ids_a[1] == ids_b[1]

        # The shared blocks should have ref_count=2 (allocated by both A and B)
        shared_block_0 = kv_mgr.pool.blocks[ids_a[0]]
        assert shared_block_0.ref_count == 2

    def test_e2e_prefix_no_shared(self):
        """Two requests with no shared prefix: cache miss on second."""
        config = _make_config(block_size=4, num_blocks=32)
        kv_mgr = KVCacheManager(config)

        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]

        kv_mgr.allocate_blocks(tokens_a)

        # Completely different tokens - no prefix match
        cached_b = kv_mgr.find_cached_prefix(tokens_b)
        assert cached_b == 0

    def test_e2e_prefix_with_scheduler(self):
        """Scheduler uses KVCacheManager for prefix caching during init_sequence."""
        config = _make_config(block_size=4, num_blocks=32)
        kv_mgr = KVCacheManager(config)

        scheduler = Scheduler(
            config=config,
            model=None,
            tokenizer=None,
            kv_cache_manager=kv_mgr,
        )

        # Pre-populate cache by allocating blocks for a prefix
        prefix_tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        kv_mgr.allocate_blocks(prefix_tokens)

        # Now init a sequence with the same prefix + extra tokens
        req = _make_request(
            "prefix-req",
            prompt_tokens=prefix_tokens + [90, 91, 92, 93],
            max_tokens=5,
        )
        seq = scheduler._init_sequence(req)

        # The scheduler should have detected 8 cached tokens
        assert seq.num_computed_tokens == 8, (
            f"Expected 8 cached tokens, got {seq.num_computed_tokens}"
        )


# ---------------------------------------------------------------------------
# P4.4 — E2E SSD Tier Test
# ---------------------------------------------------------------------------


class TestE2ESSD:
    """Allocate blocks, attach KV data, evict to SSD, verify SSD lookup."""

    def test_e2e_ssd_full_flow(self, tmp_path):
        """Full tiered flow: allocate -> attach KV -> evict to SSD -> SSD lookup."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-cache", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # 1. Allocate blocks for a token sequence
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        ids = kv_mgr.allocate_blocks(tokens)
        assert len(ids) == 2

        # 2. Attach KV data to each block
        kv_data_0 = {
            "keys": mx.ones((1, 2, 4, 8)),
            "values": mx.zeros((1, 2, 4, 8)),
        }
        kv_data_1 = {
            "keys": mx.zeros((1, 2, 4, 8)),
            "values": mx.ones((1, 2, 4, 8)),
        }
        block_0 = kv_mgr.pool.blocks[ids[0]]
        block_1 = kv_mgr.pool.blocks[ids[1]]
        block_0.kv_data = kv_data_0
        block_1.kv_data = kv_data_1
        hash_0 = block_0.block_hash
        hash_1 = block_1.block_hash

        # 3. Verify RAM lookup works
        result_ram = tiered.lookup(hash_0)
        assert result_ram is not None
        assert mx.allclose(result_ram["keys"], kv_data_0["keys"]).item()

        # 4. Free blocks so they are eligible for eviction
        kv_mgr.free_blocks(ids)

        # 5. Evict to SSD
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2

        # 6. Verify RAM no longer has them
        assert hash_0 not in kv_mgr.hash_table
        assert hash_1 not in kv_mgr.hash_table

        # 7. Verify SSD has them
        assert ssd.num_blocks == 2
        assert hash_0 in ssd.index
        assert hash_1 in ssd.index

        # 8. Lookup via tiered (should fall through to SSD)
        result_0 = tiered.lookup(hash_0)
        assert result_0 is not None
        assert mx.allclose(result_0["keys"], kv_data_0["keys"]).item()
        assert mx.allclose(result_0["values"], kv_data_0["values"]).item()

        result_1 = tiered.lookup(hash_1)
        assert result_1 is not None
        assert mx.allclose(result_1["keys"], kv_data_1["keys"]).item()
        assert mx.allclose(result_1["values"], kv_data_1["values"]).item()

        # 9. Verify blocks returned to free pool
        assert kv_mgr.num_free_blocks == 8

    def test_e2e_ssd_miss(self, tmp_path):
        """Lookup misses when block is not in RAM or SSD."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-cache", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        result = tiered.lookup(999999)
        assert result is None

    def test_e2e_ssd_evict_without_kv_data(self, tmp_path):
        """Blocks without kv_data are evicted from RAM but not saved to SSD."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-cache", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        tokens = [1, 2, 3, 4]
        ids = kv_mgr.allocate_blocks(tokens)
        # Don't attach kv_data
        kv_mgr.free_blocks(ids)

        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1

        # SSD should be empty (no kv_data to save)
        assert ssd.num_blocks == 0

        # Block should be back in free pool
        assert kv_mgr.num_free_blocks == 8


# ---------------------------------------------------------------------------
# P4.5 — E2E Concurrent Test
# ---------------------------------------------------------------------------


class TestE2EConcurrent:
    """Submit multiple requests concurrently to the scheduler.

    All should complete within timeout, each with the expected number of tokens.
    """

    def test_e2e_concurrent_4(self):
        """4 concurrent requests all complete successfully."""
        config = _make_config(max_batch_size=8, max_queue_size=32)
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Each request generates exactly 3 tokens then stops
        def mock_gen(request_id, token_ids, step):
            if step >= 2:
                return (step + 100, f"t{step}", "stop")
            return (step + 100, f"t{step}", None)

        scheduler._mock_generate = mock_gen

        # Submit 4 requests
        for i in range(4):
            req = _make_request(
                f"concurrent-{i}",
                prompt_tokens=[i * 10 + j for j in range(4)],
                max_tokens=10,
            )
            scheduler.submit_request(req)

        # Run inference loop
        scheduler.run_inference_loop(blocking=False)
        try:
            # Collect results for all 4 requests
            results = {}
            for i in range(4):
                rid = f"concurrent-{i}"
                results[rid] = scheduler.get_result(rid, timeout=10.0)

            # Verify each request got 3 tokens
            for rid, tokens in results.items():
                assert len(tokens) == 3, f"{rid}: expected 3 tokens, got {len(tokens)}"
                assert tokens[-1].finish_reason == "stop"
                assert tokens[0].token_id == 100
                assert tokens[1].token_id == 101
                assert tokens[2].token_id == 102

        finally:
            scheduler.stop()

    def test_e2e_concurrent_varied_lengths(self):
        """Concurrent requests with different max_tokens all complete correctly."""
        config = _make_config(max_batch_size=8, max_queue_size=32)
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Generate tokens indefinitely; rely on max_tokens to stop
        scheduler._mock_generate = lambda rid, tids, step: (step + 1, f"t{step}", None)

        # Submit requests with varying max_tokens
        lengths = [2, 5, 3, 7]
        for i, max_tok in enumerate(lengths):
            req = _make_request(
                f"varied-{i}",
                prompt_tokens=[i * 100 + j for j in range(4)],
                max_tokens=max_tok,
            )
            scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            for i, max_tok in enumerate(lengths):
                rid = f"varied-{i}"
                tokens = scheduler.get_result(rid, timeout=10.0)
                assert len(tokens) == max_tok, (
                    f"{rid}: expected {max_tok} tokens, got {len(tokens)}"
                )
                assert tokens[-1].finish_reason == "length"
        finally:
            scheduler.stop()

    def test_e2e_concurrent_with_threads(self):
        """Submit requests from multiple threads simultaneously."""
        config = _make_config(max_batch_size=8, max_queue_size=32)
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 1:
                return (step + 200, f"t{step}", "stop")
            return (step + 200, f"t{step}", None)

        scheduler._mock_generate = mock_gen
        scheduler.run_inference_loop(blocking=False)

        results: dict[str, list[TokenEvent]] = {}
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def submit_and_collect(idx: int):
            try:
                barrier.wait(timeout=5.0)
                rid = f"thread-{idx}"
                req = _make_request(
                    rid,
                    prompt_tokens=[idx * 10 + j for j in range(4)],
                    max_tokens=5,
                )
                scheduler.submit_request(req)
                tokens = scheduler.get_result(rid, timeout=10.0)
                results[rid] = tokens
            except Exception as e:
                errors.append(e)

        try:
            threads = [
                threading.Thread(target=submit_and_collect, args=(i,))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15.0)

            assert not errors, f"Thread errors: {errors}"
            assert len(results) == 4
            for rid, tokens in results.items():
                assert len(tokens) == 2  # 2 tokens (step 0 + step 1 with stop)
                assert tokens[-1].finish_reason == "stop"
        finally:
            scheduler.stop()
