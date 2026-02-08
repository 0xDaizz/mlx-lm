"""Sequence-level KV cache store with LRU eviction.

Thread-safe LRU cache mapping token sequences to List[KVCache] objects,
the format BatchGenerator.insert(caches=...) expects.

Uses a trie for O(M) prefix lookup (M = query length) instead of O(N*M)
linear scan. LRU eviction is maintained via OrderedDict side-index.

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


class _TrieNode:
    """Internal trie node for token-level prefix indexing."""

    __slots__ = ("children", "cache_value", "token_key")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.cache_value: list | None = None  # Non-None = this node stores a cached entry
        self.token_key: tuple[int, ...] | None = None  # Full key for LRU tracking


class SequenceCacheStore:
    """Thread-safe LRU cache for sequence-level KV caches.

    Maps token sequences (as tuples) to List[KVCache] objects.
    On lookup, finds the longest matching prefix using a trie
    for O(M) lookup instead of scanning all entries.

    Args:
        max_entries: Maximum number of cached entries before LRU eviction.
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._max_entries = max_entries
        self._root = _TrieNode()
        self._lock = threading.Lock()
        self._size = 0
        # LRU tracking: maps token_key -> trie node (for eviction)
        self._lru_order: OrderedDict[tuple[int, ...], _TrieNode] = OrderedDict()

    def find_longest_prefix(
        self, tokens: list[int]
    ) -> tuple[list | None, list[int]]:
        """Find the longest cached prefix matching the given tokens.

        Walks the trie following token IDs from root. At each node,
        if cache_value is not None, records it as the best match so far.
        Stops when no child exists for the next token or end of tokens.

        Args:
            tokens: Token sequence to look up.

        Returns:
            Tuple of (cache_copy, remaining_tokens):
            - cache_copy: Deep copy of cached List[KVCache], or None on miss
            - remaining_tokens: Tokens not covered by the cache
        """
        # Phase 1: under lock — find best match, grab reference
        cache_ref = None
        best_key_len = 0
        with self._lock:
            node = self._root
            best_node: _TrieNode | None = None
            best_depth = 0

            # Check root node (empty token key)
            if node.cache_value is not None:
                best_node = node
                best_depth = 0

            # Walk the trie
            for i, token in enumerate(tokens):
                child = node.children.get(token)
                if child is None:
                    break
                node = child
                if node.cache_value is not None:
                    best_node = node
                    best_depth = i + 1

            if best_node is None:
                return None, list(tokens)

            # Refresh LRU order
            assert best_node.token_key is not None
            self._lru_order.move_to_end(best_node.token_key)
            cache_ref = best_node.cache_value  # Just grab reference, don't copy yet
            best_key_len = len(best_node.token_key)

        # Phase 2: outside lock — expensive deepcopy
        cache_copy = copy.deepcopy(cache_ref)

        if best_key_len > len(tokens):
            # Cached sequence is longer than query -- trim excess
            excess = best_key_len - len(tokens)
            if can_trim_prompt_cache is not None and trim_prompt_cache is not None:
                if can_trim_prompt_cache(cache_copy):
                    trim_prompt_cache(cache_copy, excess)
            return cache_copy, []
        elif best_key_len == len(tokens):
            return cache_copy, []
        else:
            # Cached sequence is shorter -- return remaining tokens
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
            # Walk/create trie nodes for each token
            node = self._root
            for token in token_tuple:
                child = node.children.get(token)
                if child is None:
                    child = _TrieNode()
                    node.children[token] = child
                node = child

            # 'node' is now the leaf for this token sequence
            is_update = node.cache_value is not None

            node.cache_value = cache_copy
            node.token_key = token_tuple

            if is_update:
                # Update existing entry -- refresh LRU, don't change size
                self._lru_order[token_tuple] = node
                self._lru_order.move_to_end(token_tuple)
            else:
                # New entry
                self._lru_order[token_tuple] = node
                self._size += 1

                # Evict oldest entries if over capacity
                while self._size > self._max_entries:
                    self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry. Must be called with _lock held."""
        if not self._lru_order:
            return

        evicted_key, evicted_node = self._lru_order.popitem(last=False)
        evicted_node.cache_value = None
        evicted_node.token_key = None
        self._size -= 1

        # Prune empty trie branches: walk from root to the evicted leaf
        # and remove nodes that have no children and no cache_value
        self._prune_path(evicted_key)

    def _prune_path(self, token_key: tuple[int, ...]) -> None:
        """Remove empty trie nodes along the path for token_key.

        Walks from root to the leaf, then prunes upward removing nodes
        that have no children and no cache_value. Must be called with _lock held.
        """
        if not token_key:
            return  # Root node is never pruned

        # Collect the path: list of (parent_node, token, child_node)
        path: list[tuple[_TrieNode, int, _TrieNode]] = []
        node = self._root
        for token in token_key:
            child = node.children.get(token)
            if child is None:
                return  # Path already pruned
            path.append((node, token, child))
            node = child

        # Walk backwards pruning empty nodes
        for parent, token, child in reversed(path):
            if child.cache_value is None and not child.children:
                del parent.children[token]
            else:
                break  # Stop pruning once we hit a node that's still needed

    @property
    def size(self) -> int:
        """Number of cached entries."""
        with self._lock:
            return self._size

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._root = _TrieNode()
            self._lru_order.clear()
            self._size = 0
