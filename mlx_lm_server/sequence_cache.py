"""Sequence-level KV cache store with LRU eviction.

Thread-safe LRU cache mapping token sequences to List[KVCache] objects,
the format BatchGenerator.insert(caches=...) expects.

Modeled on upstream LRUPromptCache (mlx_lm/server.py).
"""

from __future__ import annotations

import copy
import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

try:
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
except ImportError:
    can_trim_prompt_cache = None
    trim_prompt_cache = None


class SequenceCacheStore:
    """Thread-safe LRU cache for sequence-level KV caches.

    Maps token sequences (as tuples) to List[KVCache] objects.
    On lookup, finds the longest matching prefix. If the cached
    sequence is longer than the query, trims the cache copy.

    Args:
        max_entries: Maximum number of cached entries before LRU eviction.
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple[int, ...], list] = OrderedDict()
        self._lock = threading.Lock()

    def find_longest_prefix(
        self, tokens: list[int]
    ) -> tuple[list | None, list[int]]:
        """Find the longest cached prefix matching the given tokens.

        Searches all cached entries and finds the one with the longest
        overlap with the given tokens.

        Args:
            tokens: Token sequence to look up.

        Returns:
            Tuple of (cache_copy, remaining_tokens):
            - cache_copy: Deep copy of cached List[KVCache], or None on miss
            - remaining_tokens: Tokens not covered by the cache
        """
        token_tuple = tuple(tokens)

        with self._lock:
            # Try exact match first (O(1))
            if token_tuple in self._cache:
                self._cache.move_to_end(token_tuple)
                cache_copy = copy.deepcopy(self._cache[token_tuple])
                return cache_copy, []

            # Search for longest matching prefix among all cached keys
            best_key: tuple[int, ...] | None = None
            best_match_len = 0

            for cached_key in self._cache:
                # Token-by-token prefix matching
                common = 0
                for a, b in zip(cached_key, token_tuple):
                    if a != b:
                        break
                    common += 1
                if common > best_match_len:
                    best_key = cached_key
                    best_match_len = common

            if best_key is None:
                return None, list(tokens)

            self._cache.move_to_end(best_key)
            cache_copy = copy.deepcopy(self._cache[best_key])

            best_key_len = len(best_key)

            if best_key_len > len(token_tuple):
                # Cached sequence is longer — trim excess
                excess = best_key_len - len(token_tuple)
                if can_trim_prompt_cache is not None and trim_prompt_cache is not None:
                    if can_trim_prompt_cache(cache_copy):
                        trim_prompt_cache(cache_copy, excess)
                return cache_copy, []
            elif best_key_len == len(token_tuple):
                # Exact match (shouldn't reach here but handle it)
                return cache_copy, []
            else:
                # Cached sequence is shorter — return remaining tokens
                remaining = list(tokens[best_key_len:])
                return cache_copy, remaining

    def store(self, tokens: list[int], prompt_cache: list) -> None:
        """Store a sequence-level KV cache.

        Deep-copies the cache before storing to prevent mutation.
        Evicts oldest entry if over max_entries.

        Args:
            tokens: The full token sequence this cache covers.
            prompt_cache: List[KVCache] from BatchGenerator.
        """
        token_tuple = tuple(tokens)
        # Shallow list copy for real KV cache objects (already independent
        # after batch.extract_cache()), with deepcopy fallback for safety.
        # A full deepcopy of large KV tensors is extremely expensive but
        # needed for correctness when the stored objects are mutable dicts.
        try:
            # Check if this looks like a real KV cache list (has .state attr)
            if prompt_cache and hasattr(prompt_cache[0], "state"):
                cache_copy = list(prompt_cache)
            else:
                cache_copy = copy.deepcopy(prompt_cache)
        except Exception:
            cache_copy = copy.deepcopy(prompt_cache)

        with self._lock:
            if token_tuple in self._cache:
                self._cache[token_tuple] = cache_copy
                self._cache.move_to_end(token_tuple)
            else:
                self._cache[token_tuple] = cache_copy
                while len(self._cache) > self._max_entries:
                    self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        """Number of cached entries."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._cache.clear()
