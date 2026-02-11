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

from conftest import make_test_request as _make_request
from conftest import make_test_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path | None = None, **overrides) -> ServerConfig:
    overrides.setdefault("max_batch_size", 8)
    overrides.setdefault("prefill_batch_size", 4)
    return make_test_config(tmp_path=tmp_path, **overrides)


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

        result = tiered.lookup("nonexistent_hash")
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


# ---------------------------------------------------------------------------
# D4 — Full SSD Round-Trip Integration Tests
# ---------------------------------------------------------------------------


class TestSSDRoundTrip:
    """Full SSD round-trip integration tests (D4).

    Tests multi-layer KV data save/load through the tiered cache,
    verifying data integrity after SSD round-trip.
    """

    def test_ssd_multilayer_roundtrip(self, tmp_path):
        """Multi-layer KV data survives SSD save/load round-trip intact."""
        ssd = SSDCache(cache_dir=tmp_path / "ssd-roundtrip", ttl_days=7)

        # Create realistic multi-layer KV data (4 layers)
        num_layers = 4
        kv_data = []
        for i in range(num_layers):
            scale = float(i + 1)
            kv_data.append({
                "keys": mx.ones((1, 8, 4, 16)) * scale,
                "values": mx.ones((1, 8, 4, 16)) * (scale + 0.5),
            })

        # Save to SSD
        ssd.save_block("test_hash_abc123", kv_data)

        # Load from SSD
        loaded = ssd.load_block("test_hash_abc123")

        # Verify loaded is not None
        assert loaded is not None, "load_block returned None"

        # Verify loaded is a list with the correct number of layers
        assert isinstance(loaded, list), f"Expected list, got {type(loaded)}"
        assert len(loaded) == num_layers, (
            f"Expected {num_layers} layers, got {len(loaded)}"
        )

        # For each layer, verify keys and values match exactly
        for i in range(num_layers):
            scale = float(i + 1)
            expected_keys = mx.ones((1, 8, 4, 16)) * scale
            expected_values = mx.ones((1, 8, 4, 16)) * (scale + 0.5)

            assert mx.allclose(loaded[i]["keys"], expected_keys).item(), (
                f"Layer {i} keys mismatch"
            )
            assert mx.allclose(loaded[i]["values"], expected_values).item(), (
                f"Layer {i} values mismatch"
            )

            # Verify shapes match
            assert loaded[i]["keys"].shape == (1, 8, 4, 16), (
                f"Layer {i} keys shape mismatch: {loaded[i]['keys'].shape}"
            )
            assert loaded[i]["values"].shape == (1, 8, 4, 16), (
                f"Layer {i} values shape mismatch: {loaded[i]['values'].shape}"
            )

            # Verify dtypes match
            assert loaded[i]["keys"].dtype == expected_keys.dtype, (
                f"Layer {i} keys dtype mismatch: {loaded[i]['keys'].dtype}"
            )
            assert loaded[i]["values"].dtype == expected_values.dtype, (
                f"Layer {i} values dtype mismatch: {loaded[i]['values'].dtype}"
            )

    def test_ssd_tiered_evict_reload_multilayer(self, tmp_path):
        """Tiered cache evict-to-SSD and reload preserves multi-layer KV data."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-tiered", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # Allocate blocks for tokens [1,2,3,4,5,6,7,8] (2 blocks of 4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = kv_mgr.allocate_blocks(tokens)
        assert len(ids) == 2

        # Create multi-layer KV data for each block (2 layers)
        num_layers = 2
        original_data = {}
        for block_idx, block_id in enumerate(ids):
            block = kv_mgr.pool.blocks[block_id]
            kv_data = []
            for layer in range(num_layers):
                scale = float((block_idx + 1) * 10 + layer)
                kv_data.append({
                    "keys": mx.ones((1, 4, 4, 8)) * scale,
                    "values": mx.ones((1, 4, 4, 8)) * (scale + 0.1),
                })
            block.kv_data = kv_data
            original_data[block.block_hash] = kv_data

        # Save the block hashes before eviction
        block_0 = kv_mgr.pool.blocks[ids[0]]
        block_1 = kv_mgr.pool.blocks[ids[1]]
        hash_0 = block_0.block_hash
        hash_1 = block_1.block_hash
        assert hash_0 is not None
        assert hash_1 is not None

        # Free blocks so they are eligible for eviction
        kv_mgr.free_blocks(ids)

        # Evict to SSD
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2

        # Verify SSD has 2 blocks
        assert ssd.num_blocks == 2

        # Verify hashes are no longer in RAM hash_table
        assert hash_0 not in kv_mgr.hash_table
        assert hash_1 not in kv_mgr.hash_table

        # Lookup via tiered cache — should find in SSD
        result_0 = tiered.lookup(hash_0)
        assert result_0 is not None, "Tiered lookup for hash_0 returned None"
        assert isinstance(result_0, list), f"Expected list, got {type(result_0)}"
        assert len(result_0) == num_layers

        # Verify the loaded data matches original
        for layer in range(num_layers):
            assert mx.allclose(
                result_0[layer]["keys"],
                original_data[hash_0][layer]["keys"],
            ).item(), f"hash_0 layer {layer} keys mismatch after SSD round-trip"
            assert mx.allclose(
                result_0[layer]["values"],
                original_data[hash_0][layer]["values"],
            ).item(), f"hash_0 layer {layer} values mismatch after SSD round-trip"

        # Also verify block 1
        result_1 = tiered.lookup(hash_1)
        assert result_1 is not None, "Tiered lookup for hash_1 returned None"
        assert isinstance(result_1, list)
        assert len(result_1) == num_layers

        for layer in range(num_layers):
            assert mx.allclose(
                result_1[layer]["keys"],
                original_data[hash_1][layer]["keys"],
            ).item(), f"hash_1 layer {layer} keys mismatch after SSD round-trip"
            assert mx.allclose(
                result_1[layer]["values"],
                original_data[hash_1][layer]["values"],
            ).item(), f"hash_1 layer {layer} values mismatch after SSD round-trip"

    def test_ssd_roundtrip_preserves_dtype(self, tmp_path):
        """SSD round-trip preserves float16 dtype."""
        ssd = SSDCache(cache_dir=tmp_path / "ssd-dtype", ttl_days=7)

        kv_data = [{
            "keys": mx.ones((1, 4, 4, 8), dtype=mx.float16),
            "values": mx.zeros((1, 4, 4, 8), dtype=mx.float16),
        }]

        ssd.save_block("dtype_test_hash", kv_data)
        loaded = ssd.load_block("dtype_test_hash")

        assert loaded is not None, "load_block returned None"
        assert isinstance(loaded, list)
        assert len(loaded) == 1

        assert loaded[0]["keys"].dtype == mx.float16, (
            f"Expected float16 keys, got {loaded[0]['keys'].dtype}"
        )
        assert loaded[0]["values"].dtype == mx.float16, (
            f"Expected float16 values, got {loaded[0]['values'].dtype}"
        )

        # Verify data values are correct
        assert mx.allclose(
            loaded[0]["keys"],
            mx.ones((1, 4, 4, 8), dtype=mx.float16),
        ).item(), "float16 keys data mismatch after round-trip"
        assert mx.allclose(
            loaded[0]["values"],
            mx.zeros((1, 4, 4, 8), dtype=mx.float16),
        ).item(), "float16 values data mismatch after round-trip"

    def test_ssd_prefix_after_reload(self, tmp_path):
        """find_cached_prefix only checks RAM hash_table, not SSD.

        Documents current behavior: after eviction to SSD, blocks are no
        longer discoverable via find_cached_prefix (which only checks
        the RAM hash_table). The SSD tier is only consulted via
        tiered.lookup() with an explicit block hash.
        """
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-prefix", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # Allocate blocks for tokens [10,20,30,40] (1 block of 4)
        tokens = [10, 20, 30, 40]
        ids = kv_mgr.allocate_blocks(tokens)
        assert len(ids) == 1

        # Attach KV data to the block
        block = kv_mgr.pool.blocks[ids[0]]
        block.kv_data = [{
            "keys": mx.ones((1, 4, 4, 8)),
            "values": mx.ones((1, 4, 4, 8)),
        }]

        # Save the block hash
        block_hash = block.block_hash
        assert block_hash is not None
        assert block_hash in kv_mgr.hash_table

        # Verify find_cached_prefix finds the first block
        query = [10, 20, 30, 40, 50, 60, 70, 80]
        cached = kv_mgr.find_cached_prefix(query)
        assert cached == 4, f"Expected 4 cached tokens before eviction, got {cached}"

        # Free and evict to SSD
        kv_mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1

        # Hash is removed from RAM hash_table
        assert block_hash not in kv_mgr.hash_table

        # find_cached_prefix returns 0 — SSD is not consulted
        cached_after = kv_mgr.find_cached_prefix(query)
        assert cached_after == 0, (
            f"Expected 0 cached tokens after SSD eviction, got {cached_after}"
        )

        # But tiered.lookup() still finds the data in SSD
        ssd_result = tiered.lookup(block_hash)
        assert ssd_result is not None, "Tiered lookup should find block in SSD"

    def test_ssd_cache_roundtrip_with_data_integrity(self, tmp_path):
        """Full SSD round-trip: create KV data -> allocate -> cache -> evict -> SSD -> reload -> verify.

        This is a comprehensive integration test that exercises the entire tiered
        cache flow end-to-end, using realistic block hashes from compute_block_hash()
        and random KV data to verify data integrity after an SSD round-trip.

        Steps verified:
        1. Create KVCacheManager, SSDCache, and TieredKVCache
        2. Create random multi-layer KV data with mx.random.uniform
        3. Allocate blocks using compute_block_hash for realistic hashes
        4. Verify blocks exist in RAM
        5. Evict blocks from RAM via TieredKVCache (saves to SSD)
        6. Verify blocks are no longer in RAM
        7. Verify SSD index contains the block hash entries
        8. Lookup via TieredKVCache (falls through to SSD)
        9. Load block data from SSD
        10. Verify loaded data matches original (shape, dtype, values)
        """
        from mlx_lm_server.kv_cache_manager import compute_block_hash

        # --- Step 1: Create manager, SSD cache, and tiered wrapper ---
        block_size = 4
        num_layers = 3
        n_heads = 4
        head_dim = 16
        config = _make_config(
            tmp_path=tmp_path, block_size=block_size, num_blocks=8,
        )
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-integrity", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # --- Step 2: Create random multi-layer KV data ---
        # Use a fixed seed for reproducibility within the test
        mx.random.seed(42)
        token_sequence = [100, 200, 300, 400, 500, 600, 700, 800]  # 2 blocks of 4

        # Compute expected block hashes using compute_block_hash
        block_0_tokens = token_sequence[0:block_size]
        block_1_tokens = token_sequence[block_size:2 * block_size]
        expected_hash_0 = compute_block_hash([], block_0_tokens)
        expected_hash_1 = compute_block_hash(
            token_sequence[:block_size], block_1_tokens,
        )

        # Create per-block, per-layer KV data using random arrays
        original_kv = {}  # hash -> list[dict] for verification later
        for block_idx, expected_hash in enumerate([expected_hash_0, expected_hash_1]):
            layers = []
            for layer_idx in range(num_layers):
                keys = mx.random.uniform(
                    shape=(1, n_heads, block_size, head_dim),
                )
                values = mx.random.uniform(
                    shape=(1, n_heads, block_size, head_dim),
                )
                # Force evaluation so arrays are materialized
                mx.eval(keys)
                mx.eval(values)
                layers.append({"keys": keys, "values": values})
            original_kv[expected_hash] = layers

        # --- Step 3: Allocate blocks and store KV data ---
        block_ids = kv_mgr.allocate_blocks(token_sequence)
        assert len(block_ids) == 2, f"Expected 2 blocks, got {len(block_ids)}"

        # Attach KV data to each block
        for i, block_id in enumerate(block_ids):
            block = kv_mgr.pool.blocks[block_id]
            block_hash = block.block_hash
            assert block_hash is not None, f"Block {block_id} has no hash"
            block.kv_data = original_kv[block_hash]

        # Verify block hashes match expected
        block_0 = kv_mgr.pool.blocks[block_ids[0]]
        block_1 = kv_mgr.pool.blocks[block_ids[1]]
        assert block_0.block_hash == expected_hash_0, (
            f"Block 0 hash mismatch: {block_0.block_hash} != {expected_hash_0}"
        )
        assert block_1.block_hash == expected_hash_1, (
            f"Block 1 hash mismatch: {block_1.block_hash} != {expected_hash_1}"
        )

        # --- Step 4: Verify blocks exist in RAM ---
        assert expected_hash_0 in kv_mgr.hash_table, "Block 0 not in RAM hash_table"
        assert expected_hash_1 in kv_mgr.hash_table, "Block 1 not in RAM hash_table"

        # Verify RAM lookup via tiered cache returns data
        ram_result_0 = tiered.lookup(expected_hash_0)
        assert ram_result_0 is not None, "RAM lookup for block 0 returned None"
        assert isinstance(ram_result_0, list), f"Expected list, got {type(ram_result_0)}"
        assert len(ram_result_0) == num_layers

        # --- Step 5: Free blocks and evict to SSD ---
        kv_mgr.free_blocks(block_ids)
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2, f"Expected 2 evicted blocks, got {len(evicted)}"

        # --- Step 6: Verify blocks are no longer in RAM ---
        assert expected_hash_0 not in kv_mgr.hash_table, (
            "Block 0 still in RAM after eviction"
        )
        assert expected_hash_1 not in kv_mgr.hash_table, (
            "Block 1 still in RAM after eviction"
        )
        assert kv_mgr.num_free_blocks == 8, (
            f"Expected all 8 blocks free, got {kv_mgr.num_free_blocks}"
        )

        # --- Step 7: Verify SSD index contains block hash entries ---
        assert ssd.num_blocks == 2, f"Expected 2 SSD blocks, got {ssd.num_blocks}"
        assert expected_hash_0 in ssd.index, "Block 0 hash not in SSD index"
        assert expected_hash_1 in ssd.index, "Block 1 hash not in SSD index"

        # Verify the SSD index file on disk contains the entries
        index_path = ssd.cache_dir / "index.json"
        assert index_path.exists(), "SSD index.json file does not exist"
        import json
        index_data = json.loads(index_path.read_text())
        # New format: blocks nested under "blocks" key
        blocks_data = index_data.get("blocks", index_data)
        assert expected_hash_0 in blocks_data, (
            "Block 0 hash not in persisted index.json"
        )
        assert expected_hash_1 in blocks_data, (
            "Block 1 hash not in persisted index.json"
        )

        # Verify safetensors files exist on disk
        assert (ssd.cache_dir / f"block_{expected_hash_0}.safetensors").exists(), (
            "Block 0 safetensors file missing"
        )
        assert (ssd.cache_dir / f"block_{expected_hash_1}.safetensors").exists(), (
            "Block 1 safetensors file missing"
        )

        # --- Steps 8-9: Lookup via TieredKVCache (falls through to SSD) ---
        loaded_0 = tiered.lookup(expected_hash_0)
        loaded_1 = tiered.lookup(expected_hash_1)

        assert loaded_0 is not None, "Tiered lookup for block 0 returned None"
        assert loaded_1 is not None, "Tiered lookup for block 1 returned None"

        # --- Step 10: Verify loaded data matches original ---
        for block_hash, loaded_data in [
            (expected_hash_0, loaded_0),
            (expected_hash_1, loaded_1),
        ]:
            original = original_kv[block_hash]
            assert isinstance(loaded_data, list), (
                f"Expected list for {block_hash}, got {type(loaded_data)}"
            )
            assert len(loaded_data) == num_layers, (
                f"Expected {num_layers} layers for {block_hash}, "
                f"got {len(loaded_data)}"
            )

            for layer_idx in range(num_layers):
                orig_keys = original[layer_idx]["keys"]
                orig_values = original[layer_idx]["values"]
                load_keys = loaded_data[layer_idx]["keys"]
                load_values = loaded_data[layer_idx]["values"]

                # Shape preservation
                assert load_keys.shape == orig_keys.shape, (
                    f"{block_hash} layer {layer_idx} keys shape mismatch: "
                    f"{load_keys.shape} != {orig_keys.shape}"
                )
                assert load_values.shape == orig_values.shape, (
                    f"{block_hash} layer {layer_idx} values shape mismatch: "
                    f"{load_values.shape} != {orig_values.shape}"
                )

                # Dtype preservation
                assert load_keys.dtype == orig_keys.dtype, (
                    f"{block_hash} layer {layer_idx} keys dtype mismatch: "
                    f"{load_keys.dtype} != {orig_keys.dtype}"
                )
                assert load_values.dtype == orig_values.dtype, (
                    f"{block_hash} layer {layer_idx} values dtype mismatch: "
                    f"{load_values.dtype} != {orig_values.dtype}"
                )

                # Data integrity (values match exactly)
                assert mx.allclose(load_keys, orig_keys).item(), (
                    f"{block_hash} layer {layer_idx} keys data mismatch "
                    f"after SSD round-trip"
                )
                assert mx.allclose(load_values, orig_values).item(), (
                    f"{block_hash} layer {layer_idx} values data mismatch "
                    f"after SSD round-trip"
                )


# ---------------------------------------------------------------------------
# Cycle 2 — Integration Test Coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gap 1: Concurrent cache_block + evict_to_ssd
# ---------------------------------------------------------------------------


class TestConcurrentCacheBlockEvict:
    """Test that concurrent cache_block() and evict_to_ssd() don't corrupt state."""

    def test_concurrent_cache_and_evict_no_corruption(self, tmp_path):
        """Concurrent cache_block and evict_to_ssd maintain consistent state."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=16)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-concurrent", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        # Pre-fill the pool: allocate and free some blocks to make them evictable
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        ids_a = kv_mgr.allocate_blocks(tokens_a)
        for bid in ids_a:
            block = kv_mgr.pool.blocks[bid]
            block.kv_data = [{"keys": mx.ones((1, 2, 4, 8)), "values": mx.zeros((1, 2, 4, 8))}]
        kv_mgr.free_blocks(ids_a)

        def cache_worker():
            try:
                barrier.wait(timeout=5.0)
                for i in range(10):
                    block_hash = f"cache_worker_{i}"
                    token_ids = [100 + i * 4 + j for j in range(4)]
                    kv_data = [{"keys": mx.ones((1, 2, 4, 8)) * float(i), "values": mx.zeros((1, 2, 4, 8))}]
                    kv_mgr.cache_block(
                        block_hash=block_hash,
                        token_ids=token_ids,
                        kv_data=kv_data,
                        tiered_cache=tiered,
                    )
            except Exception as e:
                errors.append(e)

        def evict_worker():
            try:
                barrier.wait(timeout=5.0)
                for _ in range(10):
                    tiered.evict_to_ssd(num_blocks=1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=cache_worker)
        t2 = threading.Thread(target=evict_worker)
        t1.start()
        t2.start()
        t1.join(timeout=15.0)
        t2.join(timeout=15.0)

        assert not errors, f"Concurrent errors: {errors}"

        # State consistency: all blocks in hash_table have valid block_ids
        for block_hash, block_id in kv_mgr.hash_table.items():
            block = kv_mgr.pool.blocks[block_id]
            assert block.block_hash == block_hash, (
                f"Block {block_id} hash mismatch: {block.block_hash} != {block_hash}"
            )

        # No double-allocated blocks
        allocated_ids = set(kv_mgr.hash_table.values())
        free_ids = set(kv_mgr.pool.free_queue)
        assert not (allocated_ids & free_ids), (
            f"Blocks both allocated and free: {allocated_ids & free_ids}"
        )


# ---------------------------------------------------------------------------
# Gap 2: SSD promote during block-level cache lookup
# ---------------------------------------------------------------------------


class TestSSDPromoteDuringLookup:
    """Test SSD promotion when block has been evicted (kv_data=None in hash_table)."""

    def test_ssd_promote_after_eviction(self, tmp_path):
        """Blocks evicted to SSD can be promoted back via tiered lookup."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-promote", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # 1. Allocate and populate blocks
        tokens = [10, 20, 30, 40]
        ids = kv_mgr.allocate_blocks(tokens)
        original_kv = [{"keys": mx.ones((1, 2, 4, 8)) * 3.0, "values": mx.ones((1, 2, 4, 8)) * 7.0}]
        block = kv_mgr.pool.blocks[ids[0]]
        block.kv_data = original_kv
        block_hash = block.block_hash

        # 2. Free and evict to SSD
        kv_mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=1)
        assert len(evicted) == 1
        assert block_hash not in kv_mgr.hash_table

        # 3. Promote via tiered lookup
        promoted = tiered.lookup(block_hash)
        assert promoted is not None
        assert isinstance(promoted, list)
        assert len(promoted) == 1
        assert mx.allclose(promoted[0]["keys"], mx.ones((1, 2, 4, 8)) * 3.0).item()
        assert mx.allclose(promoted[0]["values"], mx.ones((1, 2, 4, 8)) * 7.0).item()

    def test_ssd_promote_data_matches_original(self, tmp_path):
        """Promoted SSD data exactly matches the original pre-eviction data."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=8)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-match", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # Create multi-layer data
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = kv_mgr.allocate_blocks(tokens)

        originals = {}
        for block_id in ids:
            block = kv_mgr.pool.blocks[block_id]
            mx.random.seed(block_id * 42)
            kv = [
                {"keys": mx.random.uniform(shape=(1, 4, 4, 8)), "values": mx.random.uniform(shape=(1, 4, 4, 8))},
                {"keys": mx.random.uniform(shape=(1, 4, 4, 8)), "values": mx.random.uniform(shape=(1, 4, 4, 8))},
            ]
            for layer in kv:
                mx.eval(layer["keys"])
                mx.eval(layer["values"])
            block.kv_data = kv
            originals[block.block_hash] = kv

        kv_mgr.free_blocks(ids)
        tiered.evict_to_ssd(num_blocks=2)

        for bh, original_kv in originals.items():
            promoted = tiered.lookup(bh)
            assert promoted is not None, f"Block {bh} not found after promote"
            assert len(promoted) == 2
            for i in range(2):
                assert mx.allclose(promoted[i]["keys"], original_kv[i]["keys"]).item()
                assert mx.allclose(promoted[i]["values"], original_kv[i]["values"]).item()


# ---------------------------------------------------------------------------
# Gap 3: Mock error recovery continuation
# ---------------------------------------------------------------------------


class TestMockErrorRecoveryContinuation:
    """Test that after a mock generate error, scheduler can accept new requests."""

    def test_error_then_new_request_succeeds(self):
        """After a mock_generate error, scheduler recovers and processes new requests."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        call_count = 0

        def failing_then_ok(rid, tids, step):
            nonlocal call_count
            call_count += 1
            if rid == "fail-req":
                raise RuntimeError("Simulated generation failure")
            if step >= 1:
                return (step + 50, f"ok{step}", "stop")
            return (step + 50, f"ok{step}", None)

        scheduler._mock_generate = failing_then_ok

        # Submit a request that will fail
        req_fail = _make_request("fail-req", max_tokens=5)
        scheduler.submit_request(req_fail)

        scheduler.run_inference_loop(blocking=False)
        try:
            # Wait for error result
            tokens_fail = scheduler.get_result("fail-req", timeout=5.0)
            assert any(t.finish_reason == "error" for t in tokens_fail)

            # Now submit a request that should succeed
            req_ok = _make_request("ok-req", max_tokens=5)
            scheduler.submit_request(req_ok)

            tokens_ok = scheduler.get_result("ok-req", timeout=5.0)
            assert len(tokens_ok) == 2
            assert tokens_ok[-1].finish_reason == "stop"
            assert tokens_ok[0].token_text == "ok0"
        finally:
            scheduler.stop()


# ---------------------------------------------------------------------------
# Gap 4: Full request lifecycle end-to-end
# ---------------------------------------------------------------------------


class TestFullRequestLifecycle:
    """Test the complete request lifecycle with mock scheduler."""

    def test_complete_lifecycle_tokens_and_stats(self):
        """submit -> init -> prefill -> decode -> finish -> stats updated."""
        config = _make_config(block_size=4, num_blocks=32)
        kv_mgr = KVCacheManager(config)
        scheduler = Scheduler(
            config=config, model=None, tokenizer=None, kv_cache_manager=kv_mgr,
        )

        def mock_gen(rid, tids, step):
            if step >= 2:
                return (step + 10, f"t{step}", "stop")
            return (step + 10, f"t{step}", None)

        scheduler._mock_generate = mock_gen

        req = _make_request("lifecycle-1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], max_tokens=10)
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            tokens = scheduler.get_result("lifecycle-1", timeout=5.0)

            # Verify token events received
            assert len(tokens) == 3
            assert tokens[0].token_id == 10
            assert tokens[0].token_text == "t0"
            assert tokens[0].finish_reason is None
            assert tokens[2].finish_reason == "stop"

            # Verify stats updated
            stats = scheduler.get_cache_stats()
            assert stats["requests_completed"] >= 1
            assert stats["tokens_generated"] >= 3
        finally:
            scheduler.stop()

    def test_lifecycle_blocks_freed_after_completion(self):
        """Blocks allocated for a request are freed after it completes."""
        config = _make_config(block_size=4, num_blocks=16)
        kv_mgr = KVCacheManager(config)
        scheduler = Scheduler(
            config=config, model=None, tokenizer=None, kv_cache_manager=kv_mgr,
        )

        initial_free = kv_mgr.num_free_blocks

        scheduler._mock_generate = lambda rid, tids, step: (1, "x", "stop")

        req = _make_request("free-test", prompt_tokens=[1, 2, 3, 4], max_tokens=5)
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            scheduler.get_result("free-test", timeout=5.0)
            # Give cleanup time to run
            time.sleep(0.2)
            # Blocks should be freed back
            assert kv_mgr.num_free_blocks == initial_free
        finally:
            scheduler.stop()

    def test_lifecycle_streaming_token_events(self):
        """Streaming request delivers tokens via stream queue and stats are updated."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        def mock_gen(rid, tids, step):
            if step >= 3:
                return (step + 20, f"s{step}", "stop")
            return (step + 20, f"s{step}", None)

        scheduler._mock_generate = mock_gen

        req = _make_request("stream-lifecycle", max_tokens=10, stream=True)
        stream_q = scheduler.register_stream("stream-lifecycle")
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            events: list[TokenEvent] = []
            while True:
                ev = stream_q.get(timeout=5.0)
                events.append(ev)
                if ev.finish_reason is not None:
                    break

            assert len(events) == 4
            assert events[0].token_text == "s0"
            assert events[-1].finish_reason == "stop"

            stats = scheduler.get_cache_stats()
            assert stats["tokens_generated"] >= 4
        finally:
            scheduler.stop()


# ---------------------------------------------------------------------------
# Gap 5: Streaming cancel cleanup
# ---------------------------------------------------------------------------


class TestStreamingCancelCleanup:
    """Test that cancelling a streaming request cleans up all resources."""

    def test_cancel_streaming_frees_stream_queue(self):
        """Cancelling a streaming request removes the stream queue."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Use a slow generator to keep the request active
        def slow_gen(rid, tids, step):
            time.sleep(0.05)
            return (step + 1, f"t{step}", None)

        scheduler._mock_generate = slow_gen

        req = _make_request("cancel-stream", max_tokens=100, stream=True)
        stream_q = scheduler.register_stream("cancel-stream")
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            # Wait for at least one token
            ev = stream_q.get(timeout=5.0)
            assert ev.token_text == "t0"

            # Cancel the request
            cancelled = scheduler.cancel_request("cancel-stream")
            assert cancelled is True

            # Wait for cleanup
            time.sleep(0.3)

            # Stream should receive a finish event (error or cancelled)
            finish_received = False
            while not stream_q.empty():
                ev = stream_q.get_nowait()
                if ev.finish_reason is not None:
                    finish_received = True

            # After cancel, stream queue should be removed from scheduler
            with scheduler._streams_lock:
                assert "cancel-stream" not in scheduler._streams

            # Active sequence should be cleaned up
            with scheduler._active_lock:
                assert "cancel-stream" not in scheduler._active_sequences
        finally:
            scheduler.stop()

    def test_cancel_queued_streaming_request(self):
        """Cancelling a queued (not yet active) streaming request cleans up properly."""
        config = _make_config(max_batch_size=1, max_queue_size=8)
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Block the scheduler with a long request
        def slow_gen(rid, tids, step):
            if rid == "blocker":
                time.sleep(0.05)
                if step >= 5:
                    return (step, "x", "stop")
                return (step, "x", None)
            return (step, "y", "stop")

        scheduler._mock_generate = slow_gen

        # Submit blocker
        blocker = _make_request("blocker", max_tokens=10)
        scheduler.submit_request(blocker)

        # Submit streaming request that will be queued
        req = _make_request("queued-cancel", max_tokens=5, stream=True)
        stream_q = scheduler.register_stream("queued-cancel")
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            time.sleep(0.1)
            # Cancel the queued request before it gets processed
            cancelled = scheduler.cancel_request("queued-cancel")
            assert cancelled is True

            # Wait for cleanup
            time.sleep(0.5)

            with scheduler._streams_lock:
                assert "queued-cancel" not in scheduler._streams
        finally:
            scheduler.stop()


# ---------------------------------------------------------------------------
# Gap 6: Eviction under memory pressure
# ---------------------------------------------------------------------------


class TestEvictionUnderMemoryPressure:
    """Test LRU eviction when all blocks are allocated."""

    def test_new_allocation_triggers_eviction(self, tmp_path):
        """When all blocks are used, cache_block triggers eviction of ref_count=0 blocks."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=4)
        kv_mgr = KVCacheManager(config)

        # Allocate all blocks
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ids = kv_mgr.allocate_blocks(tokens_a)
        assert len(ids) == 4
        assert kv_mgr.num_free_blocks == 0

        # Free first 2 blocks so they become evictable
        kv_mgr.free_blocks(ids[:2])

        # cache_block should evict a free block and allocate
        new_kv = [{"keys": mx.ones((1, 2, 4, 8)), "values": mx.zeros((1, 2, 4, 8))}]
        new_id = kv_mgr.cache_block(
            block_hash="new_block_hash",
            token_ids=[100, 200, 300, 400],
            kv_data=new_kv,
        )
        assert new_id is not None

    def test_evicted_blocks_have_zero_ref_count(self, tmp_path):
        """Only blocks with ref_count == 0 are eligible for eviction."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=4)
        kv_mgr = KVCacheManager(config)

        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ids = kv_mgr.allocate_blocks(tokens)

        # Free only 1 block
        kv_mgr.free_blocks([ids[0]])

        # Evict — should only evict the freed block
        evicted = kv_mgr.evict_lru(num_blocks=4)
        assert len(evicted) == 1  # Only 1 block has ref_count=0
        assert evicted[0] == ids[0]

    def test_evicted_blocks_recoverable_from_ssd(self, tmp_path):
        """Evicted blocks with kv_data can be recovered from SSD."""
        config = _make_config(tmp_path=tmp_path, block_size=4, num_blocks=4)
        kv_mgr = KVCacheManager(config)
        ssd = SSDCache(cache_dir=tmp_path / "ssd-pressure", ttl_days=7)
        tiered = TieredKVCache(ram=kv_mgr, ssd=ssd)

        # Allocate and populate all blocks
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = kv_mgr.allocate_blocks(tokens)
        for bid in ids:
            block = kv_mgr.pool.blocks[bid]
            block.kv_data = [{"keys": mx.ones((1, 2, 4, 8)) * float(bid), "values": mx.zeros((1, 2, 4, 8))}]
        hashes = [kv_mgr.pool.blocks[bid].block_hash for bid in ids]

        # Free and evict to SSD
        kv_mgr.free_blocks(ids)
        evicted = tiered.evict_to_ssd(num_blocks=2)
        assert len(evicted) == 2

        # Verify recovery from SSD
        for bh in hashes:
            recovered = tiered.lookup(bh)
            assert recovered is not None, f"Block {bh} not recoverable from SSD"


# ---------------------------------------------------------------------------
# Gap 7: Sequence cache trie prefix matching accuracy
# ---------------------------------------------------------------------------


class TestSequenceCacheTrieAccuracy:
    """Test edge cases in SequenceCacheStore prefix matching."""

    def test_exact_match_returns_full_cache(self):
        """Exact token match returns cache with empty remaining."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2, 3, 4, 5], [{"exact": True}])

        result, remaining = store.find_longest_prefix([1, 2, 3, 4, 5])
        assert result is not None
        assert result == [{"exact": True}]
        assert remaining == []

    def test_partial_prefix_match(self):
        """Partial prefix returns cache and remaining tokens."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2, 3], [{"prefix": True}])

        result, remaining = store.find_longest_prefix([1, 2, 3, 4, 5, 6])
        assert result is not None
        assert result == [{"prefix": True}]
        assert remaining == [4, 5, 6]

    def test_no_match_returns_none(self):
        """Completely different tokens return None."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2, 3], [{"data": True}])

        result, remaining = store.find_longest_prefix([10, 20, 30])
        assert result is None
        assert remaining == [10, 20, 30]

    def test_multiple_entries_longest_prefix_wins(self):
        """With multiple matching prefixes, the longest one is returned."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore
        store = SequenceCacheStore(max_entries=10)

        store.store([1, 2], [{"len": 2}])
        store.store([1, 2, 3], [{"len": 3}])
        store.store([1, 2, 3, 4, 5], [{"len": 5}])

        result, remaining = store.find_longest_prefix([1, 2, 3, 4, 5, 6, 7])
        assert result is not None
        assert result == [{"len": 5}]
        assert remaining == [6, 7]

    def test_eviction_under_capacity(self):
        """Oldest entries are evicted when capacity is exceeded."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore
        store = SequenceCacheStore(max_entries=3)

        store.store([1], [{"first": True}])
        store.store([2], [{"second": True}])
        store.store([3], [{"third": True}])

        # Should evict [1] (oldest)
        store.store([4], [{"fourth": True}])
        assert store.size == 3

        result, _ = store.find_longest_prefix([1])
        assert result is None  # Evicted

        result, _ = store.find_longest_prefix([4])
        assert result is not None

    def test_single_token_match(self):
        """Single token prefix matching works correctly."""
        from mlx_lm_server.sequence_cache import SequenceCacheStore
        store = SequenceCacheStore(max_entries=10)
        store.store([42], [{"single": True}])

        result, remaining = store.find_longest_prefix([42, 99])
        assert result is not None
        assert result == [{"single": True}]
        assert remaining == [99]


# ---------------------------------------------------------------------------
# Gap 8: SSD validate_index recovery
# ---------------------------------------------------------------------------


class TestSSDValidateIndexRecovery:
    """Test crash recovery via validate_index()."""

    def test_validate_cleans_orphan_files(self, tmp_path):
        """validate_index removes orphan safetensors files not in index."""
        ssd = SSDCache(cache_dir=tmp_path / "ssd-orphan", ttl_days=7)

        # Save a real block
        kv_data = [{"keys": mx.ones((1, 2, 4, 8)), "values": mx.zeros((1, 2, 4, 8))}]
        ssd.save_block("real_hash", kv_data)
        assert ssd.num_blocks == 1

        # Create an orphan file manually
        orphan_path = ssd.cache_dir / "block_orphan_hash.safetensors"
        orphan_path.write_bytes(b"fake data")
        assert orphan_path.exists()

        # Validate should clean the orphan
        result = ssd.validate_index()
        assert result["orphans_cleaned"] == 1
        assert result["missing_cleaned"] == 0
        assert not orphan_path.exists()
        assert ssd.num_blocks == 1  # Real block still there

    def test_validate_cleans_missing_entries(self, tmp_path):
        """validate_index removes index entries whose files are missing."""
        ssd = SSDCache(cache_dir=tmp_path / "ssd-missing", ttl_days=7)

        # Save two blocks
        kv1 = [{"keys": mx.ones((1, 2, 4, 8)), "values": mx.zeros((1, 2, 4, 8))}]
        kv2 = [{"keys": mx.zeros((1, 2, 4, 8)), "values": mx.ones((1, 2, 4, 8))}]
        ssd.save_block("hash_1", kv1)
        ssd.save_block("hash_2", kv2)
        assert ssd.num_blocks == 2

        # Delete one block's file manually (simulating crash/corruption)
        file_path = ssd.index["hash_1"].filepath
        Path(file_path).unlink()

        # Validate should clean the missing entry
        result = ssd.validate_index()
        assert result["missing_cleaned"] == 1
        assert result["orphans_cleaned"] == 0
        assert ssd.num_blocks == 1
        assert "hash_1" not in ssd.index
        assert "hash_2" in ssd.index

    def test_validate_both_orphans_and_missing(self, tmp_path):
        """validate_index handles both orphans and missing entries simultaneously."""
        ssd = SSDCache(cache_dir=tmp_path / "ssd-both", ttl_days=7)

        # Save a block then corrupt the state
        kv = [{"keys": mx.ones((1, 2, 4, 8)), "values": mx.zeros((1, 2, 4, 8))}]
        ssd.save_block("good_hash", kv)
        ssd.save_block("missing_hash", kv)

        # Create orphan file
        orphan_path = ssd.cache_dir / "block_orphan.safetensors"
        orphan_path.write_bytes(b"fake")

        # Remove the file for missing_hash
        file_path = ssd.index["missing_hash"].filepath
        Path(file_path).unlink()

        result = ssd.validate_index()
        assert result["orphans_cleaned"] >= 1
        assert result["missing_cleaned"] == 1
        assert "good_hash" in ssd.index
        assert "missing_hash" not in ssd.index
        assert not orphan_path.exists()


# ---------------------------------------------------------------------------
# Gap 9: Non-blocking finish events
# ---------------------------------------------------------------------------


class TestNonBlockingFinishEvents:
    """Test that finish events are delivered even when stream queue is full."""

    def test_finish_event_delivered_when_queue_full(self):
        """Finish event arrives via put_nowait with drop-oldest retry."""
        import queue as qmod

        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Register a stream with a very small queue
        small_q: qmod.Queue[TokenEvent] = qmod.Queue(maxsize=2)
        with scheduler._streams_lock:
            scheduler._streams["full-test"] = small_q

        # Fill the queue manually
        small_q.put(TokenEvent(request_id="full-test", token_id=1, token_text="a", finish_reason=None))
        small_q.put(TokenEvent(request_id="full-test", token_id=2, token_text="b", finish_reason=None))
        assert small_q.full()

        # Deliver a finish event using the scheduler's method
        finish_event = TokenEvent(
            request_id="full-test", token_id=-1, token_text="", finish_reason="stop"
        )
        scheduler._put_event_to_stream(small_q, finish_event, "full-test", is_finish=True)

        # The finish event should be in the queue (one token was dropped)
        events = []
        while not small_q.empty():
            events.append(small_q.get_nowait())

        # At least one event should be the finish event
        finish_events = [e for e in events if e.finish_reason == "stop"]
        assert len(finish_events) == 1, f"Expected 1 finish event, got {len(finish_events)}"

    def test_slow_consumer_still_gets_finish(self):
        """A slow consumer with a full queue still gets the finish event delivered."""
        config = _make_config()
        scheduler = Scheduler(config=config, model=None, tokenizer=None)

        # Use a generator that finishes quickly
        def fast_gen(rid, tids, step):
            if step >= 0:
                return (step + 1, "done", "stop")
            return (step + 1, "tok", None)

        scheduler._mock_generate = fast_gen

        req = _make_request("slow-consumer", max_tokens=5, stream=True)
        stream_q = scheduler.register_stream("slow-consumer")
        scheduler.submit_request(req)

        scheduler.run_inference_loop(blocking=False)
        try:
            # Collect all events
            events = []
            while True:
                ev = stream_q.get(timeout=5.0)
                events.append(ev)
                if ev.finish_reason is not None:
                    break

            assert any(e.finish_reason == "stop" for e in events)
        finally:
            scheduler.stop()


# ---------------------------------------------------------------------------
# Gap 10: Health endpoint state transitions
# ---------------------------------------------------------------------------


class TestHealthEndpointStateTransitions:
    """Test /health returns correct status strings for different states."""

    @pytest.mark.anyio
    async def test_health_ok_status(self, tmp_path):
        """Normal operation returns status 'ok'."""
        from conftest import MockSchedulerForApp, SimpleTokenizer
        from httpx import ASGITransport, AsyncClient
        from mlx_lm_server.server import create_app

        scheduler = MockSchedulerForApp()
        cfg = ServerConfig(
            model="test-model", block_size=4, num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd", max_batch_size=2, max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=scheduler, tokenizer=SimpleTokenizer())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"

    @pytest.mark.anyio
    async def test_health_degraded_status(self, tmp_path):
        """High utilization (>=72% of threshold) returns status 'degraded'."""
        from conftest import MockSchedulerForApp, SimpleTokenizer
        from httpx import ASGITransport, AsyncClient
        from mlx_lm_server.server import create_app

        class DegradedScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {"total_blocks": 100, "used_blocks": 75, "free_blocks": 25, "hit_rate": 0.5}

        scheduler = DegradedScheduler()
        cfg = ServerConfig(
            model="test-model", block_size=4, num_blocks=100,
            ssd_cache_dir=tmp_path / "ssd", max_batch_size=2, max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=scheduler, tokenizer=SimpleTokenizer())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"

    @pytest.mark.anyio
    async def test_health_overloaded_status(self, tmp_path):
        """Utilization at or above threshold returns status 'overloaded' with 503."""
        from conftest import MockSchedulerForApp, SimpleTokenizer
        from httpx import ASGITransport, AsyncClient
        from mlx_lm_server.server import create_app

        class OverloadedScheduler(MockSchedulerForApp):
            def get_cache_stats(self):
                return {"total_blocks": 100, "used_blocks": 95, "free_blocks": 5, "hit_rate": 0.5}

        scheduler = OverloadedScheduler()
        cfg = ServerConfig(
            model="test-model", block_size=4, num_blocks=100,
            ssd_cache_dir=tmp_path / "ssd", max_batch_size=2, max_queue_size=8,
            memory_pressure_threshold=0.9,
        )
        app = create_app(config=cfg, scheduler=scheduler, tokenizer=SimpleTokenizer())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "overloaded"

    @pytest.mark.anyio
    async def test_health_shutting_down_status(self, tmp_path):
        """After shutdown initiated, returns status 'shutting_down' with 503."""
        from conftest import MockSchedulerForApp, SimpleTokenizer
        from httpx import ASGITransport, AsyncClient
        from mlx_lm_server.server import create_app

        scheduler = MockSchedulerForApp()
        cfg = ServerConfig(
            model="test-model", block_size=4, num_blocks=64,
            ssd_cache_dir=tmp_path / "ssd", max_batch_size=2, max_queue_size=8,
        )
        app = create_app(config=cfg, scheduler=scheduler, tokenizer=SimpleTokenizer())
        # Set shutting_down state
        app.state.shutting_down = True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "shutting_down"
            assert data["ready"] is False
