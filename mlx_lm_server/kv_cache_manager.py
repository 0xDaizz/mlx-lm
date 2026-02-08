"""Hash-based KV Cache Manager with automatic prefix caching.

Implements a vLLM-style block-level KV cache with:
- Pre-allocated block pool with free queue
- Hash-based block identification for automatic prefix caching
- Thread-safe ref_count management
- LRU eviction for blocks with ref_count == 0
- MLX KV cache adapter (extract/inject blocks from mlx-lm cache layers)
- Tiered lookup (RAM -> SSD -> miss)

Design references:
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
- mlx-lm issues #499 (server batching), #548 (persistent batch cache)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.types import KVCacheBlock

if TYPE_CHECKING:
    from mlx_lm_server.ssd_cache import SSDCache

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
        # Track which blocks were freshly allocated (vs cache hits)
        # so we can properly rollback on failure
        freshly_allocated: list[int] = []

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
                        # Roll back allocations made in this call:
                        # - For cache hits: just decrement ref_count
                        # - For freshly allocated: remove from hash_table
                        #   and return to free pool
                        for bid in allocated_block_ids:
                            b = self.pool.blocks[bid]
                            if bid in freshly_allocated:
                                # Fully undo the fresh allocation
                                if b.block_hash is not None and b.block_hash in self.hash_table:
                                    del self.hash_table[b.block_hash]
                                self.pool.return_block(bid)
                            else:
                                # Cache hit — just decrement ref_count
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
                freshly_allocated.append(block.block_id)
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

    def evict_lru(self, num_blocks: int = 1, exclude_ids: set[int] | None = None) -> list[int]:
        """Evict least-recently-used blocks with ref_count == 0.

        Finds blocks that are no longer referenced by any active sequence,
        sorts them by last_accessed (oldest first), evicts up to num_blocks,
        removes them from hash_table, clears their data, and returns them
        to the free pool.

        Args:
            num_blocks: Maximum number of blocks to evict.
            exclude_ids: Block IDs to exclude from eviction (e.g. freshly stored blocks).

        Returns:
            List of evicted block_ids. May be shorter than num_blocks if
            fewer eligible blocks exist.
        """
        with self.lock:
            return self._evict_lru_locked(num_blocks, exclude_ids=exclude_ids)

    def _evict_lru_locked(self, num_blocks: int = 1, exclude_ids: set[int] | None = None) -> list[int]:
        """Internal eviction logic — caller must hold self.lock.

        Args:
            num_blocks: Maximum number of blocks to evict.
            exclude_ids: Block IDs to exclude from eviction (e.g. freshly stored blocks).

        Returns:
            List of evicted block_ids.
        """
        # Find all blocks with ref_count == 0 that are in the hash_table
        candidates: list[KVCacheBlock] = []
        for block_hash, block_id in self.hash_table.items():
            block = self.pool.blocks[block_id]
            if block.ref_count == 0 and (exclude_ids is None or block.block_id not in exclude_ids):
                candidates.append(block)

        # Sort by last_accessed ascending (oldest first = LRU)
        candidates.sort(key=lambda b: b.last_accessed)

        evicted_ids: list[int] = []
        for block in candidates[:num_blocks]:
            # Save hash before return_block clears it
            saved_hash = block.block_hash
            # Remove from hash table
            if saved_hash is not None and saved_hash in self.hash_table:
                del self.hash_table[saved_hash]
            # Return to free pool (clears block metadata)
            self.pool.return_block(block.block_id)
            evicted_ids.append(block.block_id)
            logger.debug("Evicted block %d (hash=%s)", block.block_id, saved_hash)

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


# ---------------------------------------------------------------------------
# MLX KV Cache Adapter (P1.7 - P1.10)
# ---------------------------------------------------------------------------


def _is_quantized_kv(kv) -> bool:
    """Check if KV data is in quantized tuple format (data, scales, biases)."""
    return isinstance(kv, tuple) and len(kv) == 3


def _dequantize_kv(kv_tuple, group_size: int, bits: int) -> mx.array:
    """Dequantize a (data, scales, biases) tuple back to float array."""
    return mx.dequantize(*kv_tuple, group_size=group_size, bits=bits)


def extract_block(
    keys: mx.array,
    values: mx.array,
    start_pos: int,
    block_size: int,
) -> dict[str, mx.array]:
    """Extract a block of K/V data from an mlx-lm cache layer's state.

    The mlx-lm KVCache stores keys and values with shape
    (B, n_kv_heads, seq_len, head_dim). This function slices out
    a block_size window along the seq_len dimension.

    Args:
        keys: Key tensor from cache.state, shape (B, H, S, D).
        values: Value tensor from cache.state, shape (B, H, S, Dv).
        start_pos: Starting position along the seq_len axis.
        block_size: Number of positions to extract.

    Returns:
        Dict with 'keys' and 'values', each of shape (B, H, block_size, D).
    """
    k_block = keys[:, :, start_pos : start_pos + block_size, :]
    v_block = values[:, :, start_pos : start_pos + block_size, :]
    return {"keys": k_block, "values": v_block}


def inject_blocks(
    blocks: list[dict[str, mx.array]],
) -> dict[str, mx.array]:
    """Reconstruct K/V tensors by concatenating blocks along the seq_len axis.

    Takes a list of block dicts (each from extract_block) and concatenates
    them to form the full K/V tensors.

    Args:
        blocks: List of dicts, each containing 'keys' and 'values' arrays
            with shape (B, H, block_size, D).

    Returns:
        Dict with 'keys' and 'values', each of shape
        (B, H, total_seq_len, D) where total_seq_len = sum of block sizes.
    """
    if not blocks:
        raise ValueError("blocks list must not be empty")
    all_keys = mx.concatenate([b["keys"] for b in blocks], axis=2)
    all_values = mx.concatenate([b["values"] for b in blocks], axis=2)
    return {"keys": all_keys, "values": all_values}


# ---------------------------------------------------------------------------
# Tiered KV Cache (P1.16 - P1.18)
# ---------------------------------------------------------------------------


class TieredKVCache:
    """Two-tier KV cache: RAM (KVCacheManager) + SSD (SSDCache).

    Lookup order: RAM hash_table -> SSD cache -> miss (None).
    On RAM eviction, blocks are demoted to SSD before being freed.

    Attributes:
        ram: The in-memory KVCacheManager.
        ssd: The SSD-backed cache (optional; None disables SSD tier).
    """

    def __init__(self, ram: KVCacheManager, ssd: SSDCache | None = None) -> None:
        self.ram = ram
        self.ssd = ssd

    def lookup(self, block_hash: int) -> list[dict[str, mx.array]] | dict[str, mx.array] | None:
        """Look up KV data for a block hash, checking RAM then SSD.

        Args:
            block_hash: The block hash to look up.

        Returns:
            KV data (list[dict] from RAM/multi-layer SSD, or dict from legacy SSD),
            or None on miss.
        """
        # Check RAM first
        with self.ram.lock:
            if block_hash in self.ram.hash_table:
                block_id = self.ram.hash_table[block_hash]
                block = self.ram.pool.blocks[block_id]
                if block.kv_data is not None:
                    block.last_accessed = time.time()
                    return block.kv_data
                return None

        # Check SSD
        if self.ssd is not None:
            kv_data = self.ssd.load_block(block_hash)
            if kv_data is not None:
                return kv_data

        return None

    def evict_to_ssd(self, num_blocks: int = 1, exclude_ids: set[int] | None = None) -> list[int]:
        """Evict LRU blocks from RAM, saving them to SSD first.

        For each block being evicted that has kv_data, the data is
        persisted to SSD before the block is freed from RAM.

        Args:
            num_blocks: Number of blocks to evict.
            exclude_ids: Block IDs to exclude from eviction (e.g. freshly stored blocks).

        Returns:
            List of evicted block_ids.
        """
        with self.ram.lock:
            # Find eviction candidates
            candidates: list[KVCacheBlock] = []
            for block_hash, block_id in self.ram.hash_table.items():
                block = self.ram.pool.blocks[block_id]
                if block.ref_count == 0 and (exclude_ids is None or block.block_id not in exclude_ids):
                    candidates.append(block)

            candidates.sort(key=lambda b: b.last_accessed)

            evicted_ids: list[int] = []
            for block in candidates[:num_blocks]:
                # Save to SSD before evicting
                if (
                    self.ssd is not None
                    and block.kv_data is not None
                    and block.block_hash is not None
                ):
                    self.ssd.save_block(block.block_hash, block.kv_data)

                # Remove from hash table
                if (
                    block.block_hash is not None
                    and block.block_hash in self.ram.hash_table
                ):
                    del self.ram.hash_table[block.block_hash]
                # Return to free pool
                self.ram.pool.return_block(block.block_id)
                evicted_ids.append(block.block_id)

        return evicted_ids


# ---------------------------------------------------------------------------
# Cache Format Bridge (P7.1)
# ---------------------------------------------------------------------------


def decompose_cache_to_blocks(
    prompt_cache: list,
    token_ids: list[int],
    block_size: int,
) -> list[dict]:
    """Extract block-level KV data from a sequence-level cache.

    Takes a List[KVCache] (one per model layer) and a token sequence,
    decomposes the KV data into block_size chunks aligned with token blocks.

    For each block-sized chunk of token_ids, extracts the corresponding
    K/V slices from each layer's cache using extract_block().

    Args:
        prompt_cache: List of KVCache objects (one per layer).
        token_ids: The token sequence these caches correspond to.
        block_size: Number of tokens per block.

    Returns:
        List of dicts, each containing:
        - 'block_hash': int — hash for this block
        - 'token_ids': list[int] — tokens in this block
        - 'kv_data_per_layer': list[dict] — per-layer K/V data
          Each dict has 'keys' and 'values' mx.arrays
    """
    num_tokens = len(token_ids)
    num_full_blocks = num_tokens // block_size
    blocks = []

    for i in range(num_full_blocks):
        start = i * block_size
        end = start + block_size
        prefix = token_ids[:start]
        block_tokens = token_ids[start:end]

        block_hash = compute_block_hash(prefix, block_tokens)

        # Extract KV data from each layer
        kv_data_per_layer = []
        for cache_layer in prompt_cache:
            # Skip RotatingKVCache — positional slicing would be wrong
            if hasattr(cache_layer, "offset") and hasattr(cache_layer, "max_size"):
                return []
            state = cache_layer.state
            if state is None or len(state) < 2:
                continue
            keys, values = state[0], state[1]
            # Handle QuantizedKVCache: state returns tuple-of-tuples
            if _is_quantized_kv(keys):
                group_size = getattr(cache_layer, 'group_size', 64)
                bits = getattr(cache_layer, 'bits', 8)
                keys = _dequantize_kv(keys, group_size, bits)
                values = _dequantize_kv(values, group_size, bits)
            block_data = extract_block(keys, values, start, block_size)
            kv_data_per_layer.append(block_data)

        blocks.append({
            'block_hash': block_hash,
            'token_ids': list(block_tokens),
            'kv_data_per_layer': kv_data_per_layer,
        })

    return blocks


def reconstruct_cache_from_blocks(
    blocks: list[dict],
    model=None,
) -> list:
    """Reconstruct a List[KVCache] from block-level data.

    Creates plain KVCache objects per layer (not QuantizedKVCache), injects
    the block data by concatenating K/V tensors from all blocks for each layer.

    Plain KVCache is required because QuantizedKVCache lacks merge() which
    is needed by BatchGenerator's _merge_caches().

    Args:
        blocks: List of dicts from decompose_cache_to_blocks().
            Each dict has 'kv_data_per_layer' (list of per-layer K/V dicts).
        model: Unused (kept for backwards compatibility). Plain KVCache objects
            are created directly without needing the model.

    Returns:
        List of KVCache objects ready for BatchGenerator.insert(caches=...).
    """
    if not blocks:
        return []

    from mlx_lm.models.cache import KVCache as PlainKVCache

    # Determine number of layers from first block
    num_layers = len(blocks[0]['kv_data_per_layer']) if blocks else 0

    # Create plain KVCache objects (not QuantizedKVCache)
    cache = [PlainKVCache() for _ in range(num_layers)]

    # For each layer, collect K/V blocks and concatenate
    for layer_idx in range(num_layers):
        layer_blocks = []
        for block in blocks:
            if layer_idx < len(block['kv_data_per_layer']):
                layer_blocks.append(block['kv_data_per_layer'][layer_idx])

        if not layer_blocks:
            continue

        # Concatenate all blocks for this layer
        reconstructed = inject_blocks(layer_blocks)

        # Use .state setter which sets keys, values, AND offset automatically
        cache[layer_idx].state = (reconstructed['keys'], reconstructed['values'])

    return cache
