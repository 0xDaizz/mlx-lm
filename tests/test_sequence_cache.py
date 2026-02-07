"""Tests for SequenceCacheStore (P6.7)."""

import threading

import pytest

from mlx_lm_server.sequence_cache import SequenceCacheStore


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
