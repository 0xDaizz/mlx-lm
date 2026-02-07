"""Hash-based KV Cache Manager with automatic prefix caching.

Implements a vLLM-style block-level KV cache with:
- Pre-allocated block pool with free queue
- Hash-based block identification for automatic prefix caching
- Thread-safe ref_count management
- LRU eviction for blocks with ref_count == 0

Design references:
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
- mlx-lm issues #499 (server batching), #548 (persistent batch cache)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.types import KVCacheBlock

logger = logging.getLogger(__name__)


class BlockPoolExhaustedError(Exception):
    """Raised when no free blocks are available in the pool."""


class BlockPool:
    """Pre-allocated pool of KVCacheBlock objects with a free queue.

    All blocks are created at init time. The free queue tracks which
    block_ids are available for allocation.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.num_blocks = num_blocks
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_blocks)
        ]
        self.free_queue: deque[int] = deque(range(num_blocks))

    def get_free_block(self) -> KVCacheBlock:
        """Get a free block from the pool.

        Returns:
            A KVCacheBlock that was previously unused or returned.

        Raises:
            BlockPoolExhaustedError: If no free blocks are available.
        """
        if not self.free_queue:
            raise BlockPoolExhaustedError(
                f"Block pool exhausted: all {self.num_blocks} blocks are in use"
            )
        block_id = self.free_queue.popleft()
        return self.blocks[block_id]

    def return_block(self, block_id: int) -> None:
        """Return a block to the free pool.

        The block's metadata is cleared before it is made available again.

        Args:
            block_id: The ID of the block to return.
        """
        block = self.blocks[block_id]
        block.block_hash = None
        block.token_ids = []
        block.ref_count = 0
        block.last_accessed = 0.0
        block.kv_data = None
        self.free_queue.append(block_id)

    @property
    def num_free(self) -> int:
        """Number of free blocks available."""
        return len(self.free_queue)


def compute_block_hash(prefix_tokens: list[int], block_tokens: list[int]) -> int:
    """Compute a deterministic hash for a KV cache block.

    The hash is derived from the full token context: all tokens that came
    before this block (prefix) concatenated with the tokens in this block.
    This ensures that the same token sequence always maps to the same hash,
    enabling automatic prefix caching.

    Args:
        prefix_tokens: All tokens preceding this block.
        block_tokens: The tokens contained in this block.

    Returns:
        Integer hash value uniquely identifying this block's content in context.
    """
    return hash(tuple(prefix_tokens) + tuple(block_tokens))


class KVCacheManager:
    """Hash-based KV cache manager with automatic prefix caching.

    Manages a pool of KV cache blocks identified by content hashes.
    When a new request arrives, it walks the token sequence block-by-block,
    checking if each block's hash already exists in the hash table.
    Matching blocks are reused (ref_count incremented), avoiding redundant
    prefill computation.

    Thread safety: All mutations to ref_count and hash_table are protected
    by a threading.Lock.

    Attributes:
        config: Server configuration (block_size, num_blocks, etc.).
        pool: Pre-allocated block pool.
        hash_table: Maps block_hash -> block_id for cache lookups.
        lock: Threading lock for thread-safe mutations.
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.block_size = config.block_size
        self.pool = BlockPool(config.num_blocks)
        self.hash_table: dict[int, int] = {}  # block_hash -> block_id
        self.lock = threading.Lock()

    def compute_block_hash(
        self, prefix_tokens: list[int], block_tokens: list[int]
    ) -> int:
        """Compute hash for a block given its prefix and content tokens.

        Delegates to the module-level compute_block_hash function.
        """
        return compute_block_hash(prefix_tokens, block_tokens)

    def find_cached_prefix(self, token_ids: list[int]) -> int:
        """Find how many leading tokens are already cached (block-aligned).

        Walks the token sequence block-by-block from the start. For each
        block-sized chunk, computes its hash and checks the hash table.
        If a matching block is found AND its stored token_ids match
        (collision verification), the prefix extends. Stops at the first
        miss.

        Args:
            token_ids: The full token sequence to check.

        Returns:
            Number of cached tokens (always a multiple of block_size).
            Returns 0 if nothing is cached.
        """
        num_tokens = len(token_ids)
        num_full_blocks = num_tokens // self.block_size
        cached_tokens = 0

        with self.lock:
            for i in range(num_full_blocks):
                start = i * self.block_size
                end = start + self.block_size
                prefix = token_ids[:start]
                block_tokens = token_ids[start:end]

                block_hash = compute_block_hash(prefix, block_tokens)

                if block_hash not in self.hash_table:
                    break

                block_id = self.hash_table[block_hash]
                block = self.pool.blocks[block_id]

                # Collision verification: ensure stored tokens match
                if block.token_ids != block_tokens:
                    logger.warning(
                        "Hash collision detected for hash %d: "
                        "stored tokens %s != requested tokens %s",
                        block_hash,
                        block.token_ids[:8],  # Log first few tokens
                        block_tokens[:8],
                    )
                    break

                cached_tokens = end

        return cached_tokens

    def allocate_blocks(
        self, token_ids: list[int], num_existing_blocks: int = 0
    ) -> list[int]:
        """Allocate blocks for a token sequence, reusing cached blocks.

        For each block-sized chunk of tokens (starting after existing blocks):
        1. Compute the block hash
        2. If hash exists in hash_table and tokens match -> reuse (increment ref_count)
        3. Otherwise -> allocate from free pool, register in hash_table

        If the free pool is exhausted during allocation, attempts LRU eviction
        to reclaim blocks. Raises BlockPoolExhaustedError if eviction also fails.

        Args:
            token_ids: The full token sequence to allocate blocks for.
            num_existing_blocks: Number of blocks already allocated for this
                sequence (skip these).

        Returns:
            List of block_ids allocated (in order), starting from
            num_existing_blocks onward.
        """
        num_tokens = len(token_ids)
        num_full_blocks = num_tokens // self.block_size
        allocated_block_ids: list[int] = []

        with self.lock:
            for i in range(num_existing_blocks, num_full_blocks):
                start = i * self.block_size
                end = start + self.block_size
                prefix = token_ids[:start]
                block_tokens = token_ids[start:end]

                block_hash = compute_block_hash(prefix, block_tokens)

                # Check if this block is already cached
                if block_hash in self.hash_table:
                    block_id = self.hash_table[block_hash]
                    block = self.pool.blocks[block_id]

                    # Collision verification
                    if block.token_ids == block_tokens:
                        block.ref_count += 1
                        block.last_accessed = time.time()
                        allocated_block_ids.append(block_id)
                        logger.debug(
                            "Cache hit for block %d (hash=%d, ref_count=%d)",
                            block_id,
                            block_hash,
                            block.ref_count,
                        )
                        continue
                    else:
                        # Hash collision with different tokens — treat as miss
                        logger.warning(
                            "Hash collision during allocation for hash %d",
                            block_hash,
                        )

                # Cache miss — allocate a new block
                try:
                    block = self.pool.get_free_block()
                except BlockPoolExhaustedError:
                    # Try eviction before giving up (release lock temporarily
                    # is not needed since _evict_lru_locked works under the
                    # same lock)
                    evicted = self._evict_lru_locked(num_blocks=1)
                    if not evicted:
                        # Roll back allocations made in this call
                        for bid in allocated_block_ids:
                            b = self.pool.blocks[bid]
                            b.ref_count -= 1
                        raise BlockPoolExhaustedError(
                            "Block pool exhausted and no blocks eligible for eviction"
                        )
                    block = self.pool.get_free_block()

                block.block_hash = block_hash
                block.token_ids = list(block_tokens)  # Store a copy
                block.ref_count = 1
                block.last_accessed = time.time()
                self.hash_table[block_hash] = block.block_id
                allocated_block_ids.append(block.block_id)
                logger.debug(
                    "Allocated new block %d (hash=%d)",
                    block.block_id,
                    block_hash,
                )

        return allocated_block_ids

    def free_blocks(self, block_ids: list[int]) -> None:
        """Decrement ref_count for the given blocks.

        When ref_count reaches 0 the block stays in the hash_table so it
        can be reused by future requests with matching prefixes. It will
        only be reclaimed when evict_lru() runs and selects it.

        Args:
            block_ids: List of block IDs to release.
        """
        with self.lock:
            for block_id in block_ids:
                block = self.pool.blocks[block_id]
                if block.ref_count <= 0:
                    logger.warning(
                        "Attempted to free block %d with ref_count=%d",
                        block_id,
                        block.ref_count,
                    )
                    continue
                block.ref_count -= 1
                logger.debug(
                    "Freed block %d (ref_count now %d)", block_id, block.ref_count
                )

    def evict_lru(self, num_blocks: int = 1) -> list[int]:
        """Evict least-recently-used blocks with ref_count == 0.

        Finds blocks that are no longer referenced by any active sequence,
        sorts them by last_accessed (oldest first), evicts up to num_blocks,
        removes them from hash_table, clears their data, and returns them
        to the free pool.

        Args:
            num_blocks: Maximum number of blocks to evict.

        Returns:
            List of evicted block_ids. May be shorter than num_blocks if
            fewer eligible blocks exist.
        """
        with self.lock:
            return self._evict_lru_locked(num_blocks)

    def _evict_lru_locked(self, num_blocks: int = 1) -> list[int]:
        """Internal eviction logic — caller must hold self.lock.

        Args:
            num_blocks: Maximum number of blocks to evict.

        Returns:
            List of evicted block_ids.
        """
        # Find all blocks with ref_count == 0 that are in the hash_table
        candidates: list[KVCacheBlock] = []
        for block_hash, block_id in self.hash_table.items():
            block = self.pool.blocks[block_id]
            if block.ref_count == 0:
                candidates.append(block)

        # Sort by last_accessed ascending (oldest first = LRU)
        candidates.sort(key=lambda b: b.last_accessed)

        evicted_ids: list[int] = []
        for block in candidates[:num_blocks]:
            # Remove from hash table
            if block.block_hash is not None and block.block_hash in self.hash_table:
                del self.hash_table[block.block_hash]
            # Return to free pool (clears block metadata)
            self.pool.return_block(block.block_id)
            evicted_ids.append(block.block_id)
            logger.debug("Evicted block %d (hash=%d)", block.block_id, block.block_hash)

        return evicted_ids

    # --- Convenience / introspection methods ---

    @property
    def num_free_blocks(self) -> int:
        """Number of blocks in the free pool (not allocated)."""
        return self.pool.num_free

    @property
    def num_cached_blocks(self) -> int:
        """Number of blocks registered in the hash table."""
        return len(self.hash_table)

    @property
    def num_used_blocks(self) -> int:
        """Number of blocks currently allocated (not free)."""
        return self.pool.num_blocks - self.pool.num_free
