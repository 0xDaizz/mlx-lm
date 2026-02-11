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

import heapq
import hashlib
import logging
import struct
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.types import KVCacheBlock

if TYPE_CHECKING:
    from mlx_lm_server.ssd_cache import SSDCache
    from mlx_lm_server.ssd_writer import SSDWriterThread

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
        self._free_set: set[int] = set(range(num_blocks))

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
        self._free_set.discard(block_id)
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
        if block_id in self._free_set:
            logger.warning("BlockPool: double-return of block %d ignored", block_id)
            return
        self._free_set.add(block_id)
        self.free_queue.append(block_id)

    @property
    def num_free(self) -> int:
        """Number of free blocks available."""
        return len(self.free_queue)


def _compute_chain_hash(block_tokens: list[int], prev_hash: str | None = None) -> str:
    """Compute a chain hash for a single block, given the previous block's hash.

    O(block_size) per call. When chained across N blocks, total cost is O(N *
    block_size) = O(total_tokens), compared to the old O(N^2 * block_size)
    approach that re-hashed the full prefix for every block.

    Args:
        block_tokens: Token IDs in this block.
        prev_hash: Hex digest of the preceding block (None for the first block).

    Returns:
        Hex digest string identifying this block in its chain context.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update((prev_hash or "").encode('ascii'))
    for tok in block_tokens:
        h.update(struct.pack("<i", tok))
    return h.hexdigest()


def compute_block_hash(prefix_tokens: list[int], block_tokens: list[int]) -> str:
    """Compute a deterministic hash for a KV cache block.

    Legacy wrapper that builds the chain hash incrementally over prefix
    blocks, then computes the final hash for block_tokens. Existing callers
    and tests continue to work without changes.

    Args:
        prefix_tokens: All tokens preceding this block.
        block_tokens: The tokens contained in this block.

    Returns:
        Hex digest string uniquely identifying this block's content in context.
    """
    if not block_tokens:
        logger.warning(
            "compute_block_hash called with empty block_tokens — returning sentinel hash"
        )
        return _compute_chain_hash([], None)

    block_size = len(block_tokens)
    prev_hash = None
    for i in range(0, len(prefix_tokens), block_size):
        chunk = prefix_tokens[i:i + block_size]
        if len(chunk) == block_size:
            prev_hash = _compute_chain_hash(chunk, prev_hash)
    return _compute_chain_hash(block_tokens, prev_hash)


def compute_model_fingerprint(
    model_name: str,
    model,
    kv_bits: int,
    kv_group_size: int,
    adapter_path: str | None = None,
) -> str:
    """Compute a fingerprint for a model+quantization+adapter combination.

    Used to namespace SSD cache directories so that different models,
    quantization settings, or LoRA adapters do not share cached KV blocks.

    Args:
        model_name: Model identifier string (e.g. "mlx-community/Qwen3-4B-4bit").
        model: The loaded model object (used to extract config dimensions).
        kv_bits: KV cache quantization bit-width.
        kv_group_size: KV cache quantization group size.
        adapter_path: Optional path to a LoRA adapter. Different adapters
            produce different KV activations and must not share cache.

    Returns:
        Hex digest string (32 chars) uniquely identifying this configuration.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(model_name.encode("utf-8"))
    h.update(struct.pack("<ii", kv_bits, kv_group_size))
    h.update((adapter_path or "").encode("utf-8"))
    if hasattr(model, "config"):
        cfg = model.config
        h.update(struct.pack(
            "<iii",
            getattr(cfg, "num_hidden_layers", 0),
            getattr(cfg, "num_key_value_heads", 0),
            getattr(cfg, "hidden_size", 0),
        ))
    return h.hexdigest()


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

    def __init__(self, config: ServerConfig, ssd: SSDCache | None = None) -> None:
        self.config = config
        self.block_size = config.block_size
        self.pool = BlockPool(config.num_blocks)
        self.hash_table: dict[str, int] = {}  # block_hash -> block_id
        self.lock = threading.Lock()
        self._eviction_heap: list[tuple[float, int]] = []  # (last_accessed, block_id)
        self._heap_pushes: int = 0  # Total pushes since last rebuild
        self.ssd = ssd  # Optional SSD tier for fallback lookups and promote
        # SSD-related stats counters
        self._stats: dict[str, int] = {
            "kv_promote_hits": 0,
            "kv_promote_fail": 0,
            "kv_lookup_hits": 0,
            "kv_lookup_miss": 0,
        }

    def _rebuild_eviction_heap(self) -> None:
        """Rebuild eviction heap, removing stale entries. Must hold self.lock."""
        new_heap = []
        for block_hash, block_id in self.hash_table.items():
            block = self.pool.blocks[block_id]
            if block.ref_count == 0:
                new_heap.append((block.last_accessed, block_id))
        heapq.heapify(new_heap)
        self._eviction_heap = new_heap
        self._heap_pushes = 0

    def compute_block_hash(
        self, prefix_tokens: list[int], block_tokens: list[int]
    ) -> str:
        """Compute hash for a block given its prefix and content tokens.

        Delegates to the module-level compute_block_hash function.
        """
        return compute_block_hash(prefix_tokens, block_tokens)

    def get_block(self, block_id: int) -> KVCacheBlock:
        """Get a block by its ID. The block must be allocated (ref_count > 0)."""
        return self.pool.blocks[block_id]

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
            prev_hash: str | None = None
            for i in range(num_full_blocks):
                start = i * self.block_size
                end = start + self.block_size
                block_tokens = token_ids[start:end]

                block_hash = _compute_chain_hash(block_tokens, prev_hash)

                if block_hash not in self.hash_table:
                    # SSD fallback: check if block exists on disk (index-only, no I/O)
                    if self.ssd is not None and self.ssd.has_block(block_hash):
                        # Block was evicted to SSD — count as cached.
                        # allocate_blocks() will handle the actual promote later.
                        prev_hash = block_hash
                        cached_tokens = end
                        continue
                    break

                block_id = self.hash_table[block_hash]
                block = self.pool.blocks[block_id]

                # Collision verification: ensure stored tokens match
                if block.token_ids != block_tokens:
                    logger.warning(
                        "Hash collision detected for hash %s: "
                        "stored tokens %s != requested tokens %s",
                        block_hash,
                        block.token_ids[:8],  # Log first few tokens
                        block_tokens[:8],
                    )
                    break

                prev_hash = block_hash
                cached_tokens = end

        return cached_tokens

    def _rollback_allocations_locked(
        self, allocated_block_ids: list[int], freshly_allocated: list[int]
    ) -> None:
        """Rollback all allocations made so far. Caller must hold self.lock.

        For freshly allocated blocks (new blocks pulled from free pool),
        removes their hash_table entry and returns them to the free pool.
        For reused blocks (cache hits with incremented ref_count),
        decrements their ref_count back.

        Args:
            allocated_block_ids: All block IDs allocated in this call so far.
            freshly_allocated: Subset of allocated_block_ids that were freshly
                pulled from the free pool (not cache-hit reuses).
        """
        for bid in allocated_block_ids:
            b = self.pool.blocks[bid]
            if bid in freshly_allocated:
                if b.block_hash is not None and self.hash_table.get(b.block_hash) == bid:
                    del self.hash_table[b.block_hash]
                self.pool.return_block(bid)
            else:
                b.ref_count -= 1

    def allocate_blocks(
        self, token_ids: list[int], num_existing_blocks: int = 0
    ) -> list[int]:
        """Allocate blocks for a token sequence, reusing cached blocks.

        Uses 3-pass pattern to avoid holding lock during SSD I/O:
        Pass 1 (locked): classify blocks as RAM-HIT / SSD-PROMOTE / MISS
        Pass 2 (unlocked): load SSD data for promote candidates
        Pass 3 (locked): allocate blocks with loaded data

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

        # Pre-compute all block hashes (CPU-only, no lock needed)
        block_info: list[tuple[list[int], str]] = []  # (block_tokens, block_hash)
        prev_hash: str | None = None
        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            if end > num_tokens:
                break
            block_tokens = token_ids[start:end]
            block_hash = _compute_chain_hash(block_tokens, prev_hash)
            block_info.append((list(block_tokens), block_hash))
            prev_hash = block_hash

        # Classification constants
        RAM_HIT = "ram_hit"
        SSD_PROMOTE = "ssd_promote"
        MISS = "miss"
        COLLISION = "collision"

        # --- Pass 1 (locked): classify blocks ---
        classifications: list[tuple[int, str, int | None]] = []
        # (block_info_index, classification, block_id_for_ram_hits)
        ssd_hashes_to_load: list[tuple[int, str]] = []  # (index, block_hash)
        allocated_block_ids: list[int] = []
        freshly_allocated: list[int] = []

        with self.lock:
            for idx in range(num_existing_blocks, len(block_info)):
                block_tokens, block_hash = block_info[idx]

                # Check RAM cache
                if block_hash in self.hash_table:
                    block_id = self.hash_table[block_hash]
                    block = self.pool.blocks[block_id]

                    # Collision verification
                    if block.token_ids == block_tokens:
                        # RAM hit — process immediately
                        block.ref_count += 1
                        block.last_accessed = time.time()
                        allocated_block_ids.append(block_id)
                        self._stats["kv_lookup_hits"] += 1
                        logger.debug(
                            "Cache hit for block %d (hash=%s, ref_count=%d)",
                            block_id, block_hash, block.ref_count,
                        )
                        classifications.append((idx, RAM_HIT, block_id))
                        continue
                    else:
                        # Hash collision
                        logger.warning(
                            "Hash collision during allocation for hash %s",
                            block_hash,
                        )
                        classifications.append((idx, COLLISION, None))
                        continue

                # Check SSD index (index-only, fast — no I/O)
                if self.ssd is not None and self.ssd.has_block(block_hash):
                    ssd_hashes_to_load.append((idx, block_hash))
                    classifications.append((idx, SSD_PROMOTE, None))
                    continue

                # Cache miss
                classifications.append((idx, MISS, None))

        # --- Pass 2 (no lock): load SSD data ---
        ssd_loaded: dict[int, list | dict | None] = {}  # idx -> raw_data
        for idx, block_hash in ssd_hashes_to_load:
            raw_data = self.ssd.load_block(block_hash) if self.ssd is not None else None
            ssd_loaded[idx] = raw_data

        # --- Pass 3 (locked): allocate blocks with loaded data ---
        with self.lock:
            for idx, classification, ram_block_id in classifications:
                block_tokens, block_hash = block_info[idx]

                if classification == RAM_HIT:
                    # Already processed in Pass 1
                    continue

                if classification == SSD_PROMOTE:
                    # Re-check: another thread may have added this block to RAM
                    if block_hash in self.hash_table:
                        block_id = self.hash_table[block_hash]
                        block = self.pool.blocks[block_id]
                        if block.token_ids == block_tokens:
                            block.ref_count += 1
                            block.last_accessed = time.time()
                            allocated_block_ids.append(block_id)
                            self._stats["kv_lookup_hits"] += 1
                            continue
                        # else: collision — fall through to MISS handling

                    raw_data = ssd_loaded.get(idx)
                    if raw_data is not None:
                        # Normalize format: ensure list[dict]
                        if isinstance(raw_data, dict):
                            raw_data = [raw_data]
                        try:
                            block = self.pool.get_free_block()
                        except BlockPoolExhaustedError:
                            evicted = self._evict_lru_locked(num_blocks=1)
                            if not evicted:
                                self._stats["kv_promote_fail"] += 1
                                self._rollback_allocations_locked(allocated_block_ids, freshly_allocated)
                                raise BlockPoolExhaustedError(
                                    "Block pool exhausted and no blocks eligible for eviction"
                                )
                            block = self.pool.get_free_block()

                        block.block_hash = block_hash
                        block.token_ids = block_tokens
                        block.kv_data = raw_data
                        block.ref_count = 1
                        block.last_accessed = time.time()
                        self.hash_table[block_hash] = block.block_id
                        allocated_block_ids.append(block.block_id)
                        freshly_allocated.append(block.block_id)
                        self._stats["kv_promote_hits"] += 1
                        logger.debug(
                            "SSD promote for block %d (hash=%s)",
                            block.block_id, block_hash,
                        )
                        continue

                    # SSD load returned None (stale index) — reclassify as
                    # COLLISION so the MISS handler allocates a block WITHOUT
                    # registering block_hash in hash_table.  This prevents
                    # "phantom" hash entries that block self-healing via
                    # cache_block().
                    self._stats["kv_promote_fail"] += 1
                    classification = COLLISION

                # MISS or COLLISION or failed SSD promote
                is_collision = (classification == COLLISION)
                try:
                    block = self.pool.get_free_block()
                except BlockPoolExhaustedError:
                    evicted = self._evict_lru_locked(num_blocks=1)
                    if not evicted:
                        self._rollback_allocations_locked(allocated_block_ids, freshly_allocated)
                        raise BlockPoolExhaustedError(
                            "Block pool exhausted and no blocks eligible for eviction"
                        )
                    block = self.pool.get_free_block()

                block.token_ids = block_tokens
                block.ref_count = 1
                block.last_accessed = time.time()
                allocated_block_ids.append(block.block_id)
                freshly_allocated.append(block.block_id)
                self._stats["kv_lookup_miss"] += 1

                if is_collision:
                    block.block_hash = None
                    logger.debug(
                        "Allocated collision block %d (hash not registered)",
                        block.block_id,
                    )
                else:
                    block.block_hash = block_hash
                    self.hash_table[block_hash] = block.block_id
                    logger.debug(
                        "Allocated new block %d (hash=%s)",
                        block.block_id, block_hash,
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
                if block.ref_count == 0:
                    if block.block_hash is None:
                        # Collision blocks have no cache value (not in hash_table),
                        # so return them directly to the free pool instead of
                        # pushing to the eviction heap where they'd be stuck forever
                        # (_evict_lru_locked skips block_hash=None entries).
                        self.pool.return_block(block_id)
                        logger.debug(
                            "Returned collision block %d directly to free pool",
                            block_id,
                        )
                    else:
                        heapq.heappush(self._eviction_heap, (block.last_accessed, block_id))
                        self._heap_pushes += 1
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

        Uses a min-heap for O(log N) per eviction instead of O(N log N) sort.
        Stale heap entries (reused blocks, non-zero ref_count) are lazily skipped.

        Args:
            num_blocks: Maximum number of blocks to evict.
            exclude_ids: Block IDs to exclude from eviction (e.g. freshly stored blocks).

        Returns:
            List of evicted block_ids.
        """
        evicted_ids: list[int] = []

        # Rebuild heap if stale entries likely dominate
        if self._heap_pushes > max(len(self.hash_table) * 2, 100):
            self._rebuild_eviction_heap()

        while len(evicted_ids) < num_blocks and self._eviction_heap:
            ts, block_id = heapq.heappop(self._eviction_heap)
            block = self.pool.blocks[block_id]

            # Skip stale entries: block was reused, freed, or ref_count changed
            if block.ref_count != 0:
                continue
            if block.block_hash is None:
                continue  # Already returned to free pool
            if block.last_accessed != ts:
                continue  # Timestamp changed (block was accessed since being pushed)
            if exclude_ids is not None and block_id in exclude_ids:
                # Re-push excluded blocks so they can be evicted later
                heapq.heappush(self._eviction_heap, (ts, block_id))
                self._heap_pushes += 1
                continue

            # Evict this block
            saved_hash = block.block_hash
            if saved_hash is not None and saved_hash in self.hash_table:
                del self.hash_table[saved_hash]
            self.pool.return_block(block_id)
            evicted_ids.append(block_id)
            logger.debug("Evicted block %d (hash=%s)", block_id, saved_hash)

        # Fallback: if heap was all stale but there are still evictable blocks,
        # do a linear scan (rare case, O(N))
        if len(evicted_ids) < num_blocks:
            candidates: list[KVCacheBlock] = []
            for bh, bid in self.hash_table.items():
                block = self.pool.blocks[bid]
                if block.ref_count == 0 and (exclude_ids is None or bid not in exclude_ids):
                    if bid not in evicted_ids:
                        candidates.append(block)
            candidates.sort(key=lambda b: b.last_accessed)

            for block in candidates[:num_blocks - len(evicted_ids)]:
                saved_hash = block.block_hash
                if saved_hash is not None and saved_hash in self.hash_table:
                    del self.hash_table[saved_hash]
                self.pool.return_block(block.block_id)
                evicted_ids.append(block.block_id)
                logger.debug("Evicted block %d (hash=%s) [fallback]", block.block_id, saved_hash)

        return evicted_ids

    def cache_block(
        self,
        block_hash: str,
        token_ids: list[int],
        kv_data: list,
        tiered_cache=None,
        exclude_ids: set[int] | None = None,
        ssd_policy: str = "evict_only",
    ) -> int | None:
        """Atomically cache a block: allocate or reuse, with eviction fallback.

        If the block_hash already exists in the hash table, this is a no-op
        (the block is already cached). Otherwise, allocates a free block,
        populates it, and registers it in the hash table.

        Uses a 2-phase pattern:
        - Phase 1 (inside lock): RAM allocation and hash_table registration.
        - Phase 2 (outside lock): If ssd_policy == "write_through", persist
          block to SSD via tiered_cache.write_through().

        Args:
            block_hash: Hash identifying this block.
            token_ids: Token IDs stored in this block (for collision verification).
            kv_data: Per-layer KV data for this block.
            tiered_cache: Optional TieredKVCache for SSD eviction fallback
                and write-through.
            exclude_ids: Block IDs to exclude from eviction.
            ssd_policy: "evict_only" (default) or "write_through". When
                "write_through", the block is also persisted to SSD after
                RAM allocation completes.

        Returns:
            The block_id of the cached block, or None if caching failed.
        """
        result_id = None
        need_two_phase_eviction = False

        # --- Phase 1: RAM alloc under lock ---
        with self.lock:
            # Already cached — nothing to do
            if block_hash in self.hash_table:
                return None  # Already cached — no SSD write needed

            # Try to allocate a free block
            try:
                block = self.pool.get_free_block()
            except BlockPoolExhaustedError:
                if tiered_cache is not None:
                    # Two-phase eviction: release lock before calling
                    # evict_to_ssd(), which acquires self.ram.lock (same
                    # object). Python's threading.Lock is NOT reentrant,
                    # so holding it here would deadlock.
                    need_two_phase_eviction = True
                else:
                    self._evict_lru_locked(num_blocks=1, exclude_ids=exclude_ids)
                    try:
                        block = self.pool.get_free_block()
                    except BlockPoolExhaustedError:
                        return None
                    # Successfully allocated after RAM-only eviction
                    block.block_hash = block_hash
                    block.token_ids = list(token_ids)
                    block.ref_count = 1
                    block.last_accessed = time.time()
                    block.kv_data = list(kv_data)
                    self.hash_table[block_hash] = block.block_id
                    result_id = block.block_id
            else:
                # Successfully allocated on first try (no eviction needed)
                block.block_hash = block_hash
                block.token_ids = list(token_ids)
                block.ref_count = 1
                block.last_accessed = time.time()
                block.kv_data = list(kv_data)
                self.hash_table[block_hash] = block.block_id
                result_id = block.block_id

        # --- Two-phase eviction (tiered_cache path) ---
        if need_two_phase_eviction:
            # Lock has been released. evict_to_ssd() will acquire it internally.
            tiered_cache.evict_to_ssd(num_blocks=1, exclude_ids=exclude_ids)

            # Re-acquire lock and retry allocation
            with self.lock:
                # Another thread may have cached this block while we released the lock
                if block_hash in self.hash_table:
                    return None  # Another thread beat us — no-op

                try:
                    block = self.pool.get_free_block()
                except BlockPoolExhaustedError:
                    return None

                block.block_hash = block_hash
                block.token_ids = list(token_ids)
                block.ref_count = 1
                block.last_accessed = time.time()
                block.kv_data = list(kv_data)
                self.hash_table[block_hash] = block.block_id
                result_id = block.block_id

        # --- Phase 2: SSD write-through (outside lock) ---
        if result_id is not None and ssd_policy == "write_through" and tiered_cache is not None:
            try:
                tiered_cache.write_through(block_hash, kv_data, len(token_ids))
            except Exception as e:
                logger.warning("SSD write-through failed for %s: %s", block_hash, e)

        return result_id

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

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of KV cache manager statistics."""
        with self.lock:
            return dict(self._stats)


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

    def __init__(
        self,
        ram: KVCacheManager,
        ssd: SSDCache | None = None,
        writer: SSDWriterThread | None = None,
        durability: str = "best_effort",
        max_retries: int = 3,
    ) -> None:
        self.ram = ram
        self.ssd = ssd
        self._writer = writer  # Optional SSDWriterThread for async writes
        self._durability = durability
        self._max_retries = max_retries
        # Sync path stats
        self._sync_stats_lock = threading.Lock()
        self._sync_stats: dict[str, int] = {
            "tiered_sync_save_attempts": 0,
            "tiered_sync_save_success": 0,
            "tiered_sync_retry_attempts": 0,
            "tiered_sync_save_fail": 0,
            "tiered_sync_save_collision": 0,
        }

    def _save_to_ssd_with_durability(self, block_hash: str, kv_data, num_tokens: int | None = None) -> str:
        """Save to SSD with optional persistent retry.

        Returns: "saved" | "dedup" | "collision" | "error"
        """
        if self.ssd is None:
            return "error"

        with self._sync_stats_lock:
            self._sync_stats["tiered_sync_save_attempts"] += 1
        result = self.ssd.save_block(block_hash, kv_data, num_tokens)
        if result == "saved" or result == "dedup":
            with self._sync_stats_lock:
                self._sync_stats["tiered_sync_save_success"] += 1
            return result
        if result == "collision":
            with self._sync_stats_lock:
                self._sync_stats["tiered_sync_save_collision"] += 1
            return result

        # result == "error"
        if self._durability != "persistent":
            with self._sync_stats_lock:
                self._sync_stats["tiered_sync_save_fail"] += 1
            return "error"

        for _ in range(self._max_retries):
            with self._sync_stats_lock:
                self._sync_stats["tiered_sync_retry_attempts"] += 1
            retry_result = self.ssd.save_block(block_hash, kv_data, num_tokens)
            if retry_result == "saved" or retry_result == "dedup":
                with self._sync_stats_lock:
                    self._sync_stats["tiered_sync_save_success"] += 1
                return retry_result
            if retry_result == "collision":
                with self._sync_stats_lock:
                    self._sync_stats["tiered_sync_save_collision"] += 1
                return retry_result

        with self._sync_stats_lock:
            self._sync_stats["tiered_sync_save_fail"] += 1
        logger.warning("Sync SSD persistent retry exhausted for %s", block_hash)
        return "error"

    def lookup(self, block_hash: str) -> list[dict[str, mx.array]] | dict[str, mx.array] | None:
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

    def write_through(self, block_hash: str, kv_data, num_tokens: int | None = None) -> None:
        """Write block to SSD tier (async if writer available, else sync).

        When an SSDWriterThread is attached, the write is enqueued for
        background processing. Otherwise, falls back to synchronous
        SSDCache.save_block().

        Args:
            block_hash: Hash identifying the block.
            kv_data: Per-layer KV data to persist.
            num_tokens: Number of tokens in the block (optional, for SSD metadata).
        """
        if self._writer is not None:
            status = self._writer.enqueue(block_hash, kv_data, num_tokens)
            if status == "closing":
                logger.debug("Writer closing for %s, attempting sync fallback", block_hash)
                self._save_to_ssd_with_durability(block_hash, kv_data, num_tokens)
            elif status == "dedup":
                logger.debug("Writer dedup skip for %s", block_hash)
            # "queued" — nothing to do
            return
        if self.ssd is not None:
            self._save_to_ssd_with_durability(block_hash, kv_data, num_tokens)

    def get_writer_stats(self) -> dict[str, int]:
        """Return async writer stats (empty dict if no writer)."""
        if self._writer is not None:
            return self._writer.get_stats()
        return {}

    def get_sync_stats(self) -> dict[str, int]:
        """Return a snapshot of sync durability statistics."""
        with self._sync_stats_lock:
            return dict(self._sync_stats)

    def evict_to_ssd(self, num_blocks: int = 1, exclude_ids: set[int] | None = None) -> list[int]:
        """Evict LRU blocks from RAM, saving them to SSD first.

        Uses 2-phase pattern to avoid holding ram.lock during SSD I/O:
        Phase 1 (under lock): select candidates via eviction heap
        Phase 2 (no lock): save to SSD
        Phase 3 (under lock): re-check and evict confirmed saves from RAM

        Args:
            num_blocks: Number of blocks to evict.
            exclude_ids: Block IDs to exclude from eviction (e.g. freshly stored blocks).

        Returns:
            List of evicted block_ids.
        """
        # --- Phase 1: select candidates under lock using heap ---
        candidates_to_save: list[tuple[KVCacheBlock, str, list]] = []
        direct_evict: list[KVCacheBlock] = []

        with self.ram.lock:
            # Rebuild heap if stale entries likely dominate
            if self.ram._heap_pushes > max(len(self.ram.hash_table) * 2, 100):
                self.ram._rebuild_eviction_heap()

            selected: list[KVCacheBlock] = []
            skipped: list[tuple[float, int]] = []

            while len(selected) < num_blocks and self.ram._eviction_heap:
                ts, block_id = heapq.heappop(self.ram._eviction_heap)
                block = self.ram.pool.blocks[block_id]

                # Skip stale entries (same logic as _evict_lru_locked)
                if block.ref_count != 0:
                    continue
                if block.block_hash is None:
                    continue
                if block.last_accessed != ts:
                    continue
                if exclude_ids is not None and block_id in exclude_ids:
                    skipped.append((ts, block_id))
                    continue

                selected.append(block)

            # Re-push skipped (excluded) blocks back onto the heap
            for entry in skipped:
                heapq.heappush(self.ram._eviction_heap, entry)
                self.ram._heap_pushes += 1

            # Fallback: if heap was all stale but there are still evictable blocks
            if len(selected) < num_blocks:
                selected_ids = {b.block_id for b in selected}
                fallback_candidates: list[KVCacheBlock] = []
                for bh, bid in self.ram.hash_table.items():
                    block = self.ram.pool.blocks[bid]
                    if (block.ref_count == 0
                            and (exclude_ids is None or bid not in exclude_ids)
                            and bid not in selected_ids):
                        fallback_candidates.append(block)
                fallback_candidates.sort(key=lambda b: b.last_accessed)
                selected.extend(fallback_candidates[:num_blocks - len(selected)])

            for block in selected[:num_blocks]:
                if (
                    self.ssd is not None
                    and block.kv_data is not None
                    and block.block_hash is not None
                ):
                    candidates_to_save.append((block, block.block_hash, block.kv_data))
                else:
                    direct_evict.append(block)

        # --- Phase 2: save to SSD WITHOUT holding ram.lock ---
        save_results: dict[int, str] = {}
        for block, block_hash, kv_data in candidates_to_save:
            try:
                result = self._save_to_ssd_with_durability(block_hash, kv_data)
            except Exception as e:
                logger.warning("SSD save failed for block %d: %s", block.block_id, e)
                result = "error"
            save_results[block.block_id] = result

        # --- Phase 3: evict confirmed saves under lock ---
        evicted_ids: list[int] = []
        with self.ram.lock:
            for block in direct_evict:
                # Guard: block may have been evicted by another thread between phases
                if block.block_hash is None:
                    continue  # Already returned to free pool
                if block.ref_count != 0:
                    continue
                if block.block_hash in self.ram.hash_table:
                    del self.ram.hash_table[block.block_hash]
                self.ram.pool.return_block(block.block_id)
                evicted_ids.append(block.block_id)

            for block, orig_block_hash, kv_data in candidates_to_save:
                result = save_results.get(block.block_id, "error")
                if result not in ("saved", "dedup"):
                    logger.warning(
                        "SSD persist not confirmed (%s) for block %d; skip eviction",
                        result, block.block_id,
                    )
                    if block.block_hash is not None:
                        block.last_accessed = time.time()
                    continue
                # Guard: block may have been recycled by another thread
                if block.block_hash is None:
                    continue  # Already returned to free pool
                if block.block_hash != orig_block_hash:
                    continue  # Block recycled for different content
                if block.ref_count != 0:
                    continue
                if block.block_hash in self.ram.hash_table:
                    del self.ram.hash_table[block.block_hash]
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

    prev_hash: str | None = None
    for i in range(num_full_blocks):
        start = i * block_size
        end = start + block_size
        block_tokens = token_ids[start:end]

        block_hash = _compute_chain_hash(block_tokens, prev_hash)

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
        prev_hash = block_hash

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
