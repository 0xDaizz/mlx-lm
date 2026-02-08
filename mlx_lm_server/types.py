"""Shared data types for mlx-lm-server.

All components (KV cache manager, scheduler, server) import from here
to ensure consistent interfaces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class InferenceRequest:
    """A single inference request from a client."""

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class KVCacheBlock:
    """A single block in the hash-based KV cache.

    Each block stores a fixed number of tokens' KV data.
    The block_hash is computed from prefix_tokens + block_tokens,
    and token_ids are stored for hash collision verification.
    """

    block_id: int
    block_hash: str | None = None
    token_ids: list[int] = field(default_factory=list)
    ref_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    kv_data: list[dict] | None = None  # Per-layer K/V data: list of {'keys': mx.array, 'values': mx.array}


@dataclass
class SequenceState:
    """Tracks the state of an in-flight generation sequence."""

    request_id: str
    token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    is_finished: bool = False
    finish_reason: str | None = None
    output_text: str = ""
    output_tokens: list[int] = field(default_factory=list)


@dataclass
class SchedulerOutputs:
    """Output of a single scheduler step."""

    prefill_sequences: list[SequenceState] = field(default_factory=list)
    decode_sequences: list[SequenceState] = field(default_factory=list)
    preempted_sequences: list[SequenceState] = field(default_factory=list)


@dataclass
class TokenEvent:
    """A single token generated for a request (used for streaming)."""

    request_id: str
    token_id: int
    token_text: str
    finish_reason: str | None = None


@dataclass
class SSDBlockMeta:
    """Metadata for a KV cache block persisted to SSD."""

    block_hash: str
    filepath: Path
    last_accessed: datetime = field(default_factory=datetime.now)
    num_tokens: int = 0
