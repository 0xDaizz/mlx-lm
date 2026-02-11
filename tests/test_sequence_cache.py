"""Tests for SequenceCacheStore (P6.7)."""

import copy
import threading

import pytest

from mlx_lm_server.sequence_cache import SequenceCacheStore, _clone_cache_list


class TestSequenceCacheStore:
    """Unit tests for SequenceCacheStore."""

    def test_store_and_retrieve_exact(self):
        """Exact token match returns cached value."""
        store = SequenceCacheStore(max_entries=10)
        tokens = [1, 2, 3, 4, 5]
        fake_cache = [{"layer0": "data0"}, {"layer1": "data1"}]

        store.store(tokens, fake_cache)
        assert store.size == 1

        result, remaining = store.find_longest_prefix(tokens)
        assert result is not None
        assert remaining == []
        assert result == fake_cache

    def test_prefix_match(self):
        """Shorter cached prefix is found when query is longer."""
        store = SequenceCacheStore(max_entries=10)
        cached_tokens = [1, 2, 3]
        query_tokens = [1, 2, 3, 4, 5]
        fake_cache = [{"layer": "data"}]

        store.store(cached_tokens, fake_cache)

        result, remaining = store.find_longest_prefix(query_tokens)
        assert result is not None
        assert remaining == [4, 5]
        assert result == fake_cache

    def test_longest_prefix_wins(self):
        """When multiple prefixes match, longest one is returned."""
        store = SequenceCacheStore(max_entries=10)

        store.store([1, 2], [{"short": True}])
        store.store([1, 2, 3, 4], [{"long": True}])

        result, remaining = store.find_longest_prefix([1, 2, 3, 4, 5, 6])
        assert result is not None
        assert result == [{"long": True}]
        assert remaining == [5, 6]

    def test_miss_returns_none(self):
        """Unknown tokens return None and full token list."""
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2, 3], [{"data": True}])

        result, remaining = store.find_longest_prefix([10, 20, 30])
        assert result is None
        assert remaining == [10, 20, 30]

    def test_lru_eviction(self):
        """Oldest entry is evicted when max_entries is exceeded."""
        store = SequenceCacheStore(max_entries=2)

        store.store([1], [{"first": True}])
        store.store([2], [{"second": True}])
        assert store.size == 2

        store.store([3], [{"third": True}])
        assert store.size == 2

        # First entry should be evicted
        result, remaining = store.find_longest_prefix([1])
        assert result is None
        assert remaining == [1]

        # Second should still be there
        result, remaining = store.find_longest_prefix([2])
        assert result is not None

    def test_lru_access_refreshes(self):
        """Accessing an entry refreshes its LRU position."""
        store = SequenceCacheStore(max_entries=2)

        store.store([1], [{"first": True}])
        store.store([2], [{"second": True}])

        # Access [1] to refresh it
        store.find_longest_prefix([1])

        # Insert new entry — should evict [2] (oldest now)
        store.store([3], [{"third": True}])

        # [1] should still exist
        result, _ = store.find_longest_prefix([1])
        assert result is not None

        # [2] should be evicted
        result, _ = store.find_longest_prefix([2])
        assert result is None

    def test_deepcopy_isolation(self):
        """Stored cache is independent from the original (deepcopy)."""
        store = SequenceCacheStore(max_entries=10)
        original_cache = [{"key": [1, 2, 3]}]

        store.store([1, 2], original_cache)

        # Mutate original
        original_cache[0]["key"].append(999)

        result, _ = store.find_longest_prefix([1, 2])
        assert result is not None
        assert result[0]["key"] == [1, 2, 3]  # Not mutated

    def test_retrieve_isolation(self):
        """Retrieved cache is independent from stored (deepcopy on retrieval)."""
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2], [{"key": [1, 2, 3]}])

        result1, _ = store.find_longest_prefix([1, 2])
        result1[0]["key"].append(999)

        result2, _ = store.find_longest_prefix([1, 2])
        assert result2[0]["key"] == [1, 2, 3]  # Not mutated

    def test_clear(self):
        """clear() removes all entries."""
        store = SequenceCacheStore(max_entries=10)
        store.store([1], [{"a": True}])
        store.store([2], [{"b": True}])
        assert store.size == 2

        store.clear()
        assert store.size == 0

    def test_update_existing(self):
        """Storing with same tokens updates the cache."""
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2], [{"old": True}])
        store.store([1, 2], [{"new": True}])
        assert store.size == 1

        result, _ = store.find_longest_prefix([1, 2])
        assert result == [{"new": True}]

    def test_thread_safety(self):
        """Concurrent access doesn't crash or corrupt state."""
        store = SequenceCacheStore(max_entries=100)
        errors = []

        def writer(tid):
            try:
                for i in range(50):
                    store.store([tid, i], [{"tid": tid, "i": i}])
            except Exception as e:
                errors.append(e)

        def reader(tid):
            try:
                for i in range(50):
                    store.find_longest_prefix([tid, i])
            except Exception as e:
                errors.append(e)

        threads = []
        for tid in range(4):
            threads.append(threading.Thread(target=writer, args=(tid,)))
            threads.append(threading.Thread(target=reader, args=(tid,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_empty_tokens(self):
        """Empty token list can be stored and retrieved."""
        store = SequenceCacheStore(max_entries=10)
        store.store([], [{"empty": True}])

        result, remaining = store.find_longest_prefix([])
        assert result is not None
        assert remaining == []


# ---------------------------------------------------------------------------
# D2 — _clone_cache_list (fast clone for plain KVCache)
# ---------------------------------------------------------------------------


class _FakePlainKVCache:
    """Mock plain KVCache with keys/values/offset (no group_size)."""

    def __init__(self, keys, values, offset, step=None):
        self.keys = keys
        self.values = values
        self.offset = offset
        if step is not None:
            self.step = step


class _FakeQuantizedKVCache:
    """Mock QuantizedKVCache with group_size attribute."""

    def __init__(self, data):
        self.keys = data
        self.values = data
        self.offset = 4
        self.group_size = 64


class TestCloneCacheList:
    """Tests for _clone_cache_list (D2)."""

    def test_plain_kvcache_fast_path(self):
        """Plain KVCache objects are cloned via slice, not deepcopy."""
        import mlx.core as mx
        keys = mx.zeros((1, 2, 8, 4))
        values = mx.ones((1, 2, 8, 4))
        obj = _FakePlainKVCache(keys, values, offset=6)

        cloned = _clone_cache_list([obj])
        assert len(cloned) == 1
        c = cloned[0]
        # Should be sliced to offset
        assert c.keys.shape == (1, 2, 6, 4)
        assert c.values.shape == (1, 2, 6, 4)
        assert c.offset == 6

    def test_plain_kvcache_step_copied(self):
        """Step attribute is copied for plain KVCache."""
        import mlx.core as mx
        keys = mx.zeros((1, 2, 8, 4))
        values = mx.ones((1, 2, 8, 4))
        obj = _FakePlainKVCache(keys, values, offset=4, step=2)

        cloned = _clone_cache_list([obj])
        assert hasattr(cloned[0], 'step')
        assert cloned[0].step == 2

    def test_quantized_kvcache_deepcopy_fallback(self):
        """QuantizedKVCache falls back to deepcopy."""
        obj = _FakeQuantizedKVCache(data="test_data")
        cloned = _clone_cache_list([obj])
        assert len(cloned) == 1
        assert cloned[0] is not obj
        assert hasattr(cloned[0], 'group_size')

    def test_dict_deepcopy_fallback(self):
        """Dict objects fall back to deepcopy."""
        obj = {"keys": [1, 2, 3], "values": [4, 5, 6]}
        cloned = _clone_cache_list([obj])
        assert cloned[0] is not obj
        assert cloned[0] == obj
        # Mutation isolation
        cloned[0]["keys"].append(99)
        assert obj["keys"] == [1, 2, 3]

    def test_mixed_list(self):
        """Mixed list of plain KVCache and dicts is handled correctly."""
        import mlx.core as mx
        plain = _FakePlainKVCache(
            mx.zeros((1, 2, 8, 4)), mx.ones((1, 2, 8, 4)), offset=4
        )
        dict_obj = {"keys": [1, 2], "values": [3, 4]}

        cloned = _clone_cache_list([plain, dict_obj])
        assert len(cloned) == 2
        assert cloned[0].keys.shape == (1, 2, 4, 4)  # Sliced
        assert cloned[1] is not dict_obj  # Deepcopied

    def test_clone_independence(self):
        """Cloned plain KVCache is independent from original."""
        import mlx.core as mx
        keys = mx.zeros((1, 2, 8, 4))
        values = mx.ones((1, 2, 8, 4))
        obj = _FakePlainKVCache(keys, values, offset=4)

        cloned = _clone_cache_list([obj])
        # Original and clone should be different objects
        assert cloned[0] is not obj
        assert cloned[0].keys is not obj.keys


# ---------------------------------------------------------------------------
# F3 — Dead code annotation verification
# ---------------------------------------------------------------------------


class TestDeadCodeBranch:
    """Verify the unreachable branch in find_longest_prefix (F3)."""

    def test_longer_cache_branch_unreachable(self):
        """Stored cache at [A,B,C,D], query [A,B] — should be a miss,
        not a partial match, because trie stores cache_value only on leaf."""
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2, 3, 4], [{"full": True}])

        # Query shorter than stored — trie only has leaf at depth 4
        result, remaining = store.find_longest_prefix([1, 2])
        assert result is None  # No partial match possible
        assert remaining == [1, 2]

    def test_exact_match_does_not_trigger_longer_branch(self):
        """Exact match: best_key_len == len(tokens), not > len(tokens)."""
        store = SequenceCacheStore(max_entries=10)
        store.store([1, 2, 3], [{"exact": True}])

        result, remaining = store.find_longest_prefix([1, 2, 3])
        assert result is not None
        assert remaining == []


# ---------------------------------------------------------------------------
# QuantizedKVCache fast-path clone tests
# ---------------------------------------------------------------------------


class _FakeQuantizedKVCacheFull:
    """Mock QuantizedKVCache with tuple-of-3 keys/values (data, scales, biases).

    Uses realistic dimensions: D=128 so that D//el_per_int and D//group_size
    produce non-zero values matching real QuantizedKVCache layout.
    """

    def __init__(self, offset, group_size=64, bits=8):
        import mlx.core as mx
        B, H, S, D = 1, 2, 16, 128
        el_per_int = 8 * 4 // bits  # mx.uint32 size = 4 bytes
        gs = group_size
        self.offset = offset
        self.group_size = group_size
        self.bits = bits
        # keys = (quantized_data, scales, biases)
        self.keys = (
            mx.zeros((B, H, S, D // el_per_int), dtype=mx.uint32),
            mx.ones((B, H, S, D // gs), dtype=mx.float16),
            mx.full((B, H, S, D // gs), 0.5, dtype=mx.float16),
        )
        self.values = (
            mx.zeros((B, H, S, D // el_per_int), dtype=mx.uint32),
            mx.ones((B, H, S, D // gs), dtype=mx.float16),
            mx.full((B, H, S, D // gs), 0.5, dtype=mx.float16),
        )


class TestCloneQuantizedCacheFastPath:
    """Tests for QuantizedKVCache fast-path in _clone_cache_list."""

    def test_clone_quantized_cache_fast_path(self):
        """QuantizedKVCache-like objects are cloned via slice, not deepcopy."""
        import mlx.core as mx

        obj = _FakeQuantizedKVCacheFull(offset=10, group_size=64, bits=8)
        original_keys_shapes = [obj.keys[i].shape for i in range(3)]

        cloned = _clone_cache_list([obj])
        assert len(cloned) == 1
        c = cloned[0]

        # Clone should be a different object
        assert c is not obj

        # keys/values tuples should be new tuples
        assert c.keys is not obj.keys
        assert c.values is not obj.values

        # Each component should be sliced to offset=10
        for i in range(3):
            assert c.keys[i].shape[2] == 10, f"keys[{i}] seq dim should be 10"
            assert c.values[i].shape[2] == 10, f"values[{i}] seq dim should be 10"

        # Original should be unchanged (each component has its own shape)
        for i in range(3):
            assert obj.keys[i].shape == original_keys_shapes[i]

        # Attributes preserved
        assert c.offset == 10
        assert c.group_size == 64
        assert c.bits == 8

    def test_clone_quantized_cache_independence(self):
        """Modifying original does not affect cloned QuantizedKVCache."""
        import mlx.core as mx

        obj = _FakeQuantizedKVCacheFull(offset=8, group_size=64, bits=8)
        cloned = _clone_cache_list([obj])
        c = cloned[0]

        # Mutate original offset
        obj.offset = 999

        # Clone should be unaffected (copy.copy copies scalar attrs)
        # Note: copy.copy shares scalars by value for int, so offset
        # won't be affected by reassignment on the original
        assert c.offset == 8

        # The sliced arrays should be independent views
        assert c.keys[0].shape[2] == 8
        assert obj.keys[0].shape[2] == 16  # original untouched

    def test_clone_quantized_cache_fallback(self):
        """Objects with group_size but unexpected keys structure fall back to deepcopy."""

        class _BrokenQuantized:
            def __init__(self):
                self.keys = "not_a_tuple"  # Unexpected structure
                self.values = "not_a_tuple"
                self.offset = 4
                self.group_size = 64
                self.bits = 8

        obj = _BrokenQuantized()
        # Should not raise — falls back to deepcopy
        cloned = _clone_cache_list([obj])
        assert len(cloned) == 1
        assert cloned[0] is not obj
        assert cloned[0].group_size == 64

    def test_clone_plain_cache_still_works(self):
        """Regression: plain KVCache clone still works after QuantizedKVCache changes."""
        import mlx.core as mx

        keys = mx.zeros((1, 2, 8, 4))
        values = mx.ones((1, 2, 8, 4))
        obj = _FakePlainKVCache(keys, values, offset=6)

        cloned = _clone_cache_list([obj])
        assert len(cloned) == 1
        c = cloned[0]

        assert c is not obj
        assert c.keys.shape == (1, 2, 6, 4)
        assert c.values.shape == (1, 2, 6, 4)
        assert c.offset == 6

    def test_clone_mixed_plain_and_quantized(self):
        """Mixed list of plain and quantized caches handled correctly."""
        import mlx.core as mx

        plain = _FakePlainKVCache(
            mx.zeros((1, 2, 8, 4)), mx.ones((1, 2, 8, 4)), offset=4
        )
        quant = _FakeQuantizedKVCacheFull(offset=6, group_size=64, bits=8)

        cloned = _clone_cache_list([plain, quant])
        assert len(cloned) == 2

        # Plain should be sliced
        assert cloned[0].keys.shape == (1, 2, 4, 4)
        assert not hasattr(cloned[0], 'group_size')

        # Quantized should be sliced
        for i in range(3):
            assert cloned[1].keys[i].shape[2] == 6
        assert cloned[1].group_size == 64
