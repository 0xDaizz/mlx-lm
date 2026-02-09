"""Distributed control bus for cross-rank event synchronization in TP mode."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mlx.core as mx

if TYPE_CHECKING:
    from .distributed import DistributedContext

logger = logging.getLogger(__name__)

ControlEventType = Literal["submit", "cancel", "shutdown", "noop"]


@dataclass
class ControlEvent:
    """Event broadcast from rank0 to all ranks."""
    typ: ControlEventType
    payload: bytes | None = None  # pickled InferenceRequest or request_id

    @classmethod
    def submit(cls, request) -> ControlEvent:
        return cls(typ="submit", payload=pickle.dumps(request))

    @classmethod
    def cancel(cls, request_id: str) -> ControlEvent:
        return cls(typ="cancel", payload=pickle.dumps(request_id))

    @classmethod
    def shutdown(cls) -> ControlEvent:
        return cls(typ="shutdown")

    @classmethod
    def noop(cls) -> ControlEvent:
        return cls(typ="noop")

    def unpack_request(self):
        """Deserialize payload as InferenceRequest."""
        if self.payload is None:
            return None
        return pickle.loads(self.payload)

    def unpack_request_id(self) -> str | None:
        """Deserialize payload as request_id string."""
        if self.payload is None:
            return None
        return pickle.loads(self.payload)


class DistributedControlBus:
    """Broadcasts control events from rank0 to all ranks via mx.distributed.all_sum.

    Uses the same pickle + all_sum pattern as upstream mlx_lm/server.py _share_object().
    Every inference step, all ranks must call recv() exactly once to stay synchronized.
    When rank0 has no events, it sends a noop to prevent collective deadlock.
    """

    def __init__(self, dist_ctx: DistributedContext) -> None:
        self.group = dist_ctx.group
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self._stream = mx.new_stream(mx.default_device())

    def publish(self, event: ControlEvent) -> None:
        """Rank0 broadcasts an event to all ranks. Must only be called on rank0."""
        if self.rank != 0:
            raise RuntimeError("publish() must only be called on rank 0")
        self._broadcast_object(event)

    def recv(self) -> ControlEvent:
        """All ranks receive the next event. Blocking collective call.

        rank0: call publish() first, then recv() returns the same event.
        rank>0: blocks until rank0 publishes.
        """
        return self._receive_object()

    def publish_and_recv(self, event: ControlEvent | None = None) -> ControlEvent:
        """Convenience: rank0 publishes event, all ranks receive it.

        On rank0, event must be provided. On rank>0, event is ignored.
        """
        if self.rank == 0:
            if event is None:
                event = ControlEvent.noop()
            return self._broadcast_object(event)
        else:
            return self._receive_object()

    def _broadcast_object(self, obj: ControlEvent) -> ControlEvent:
        """Rank0 sends object via all_sum (pickle + size broadcast)."""
        with mx.stream(self._stream):
            data = mx.array(list(pickle.dumps(obj)), dtype=mx.uint8)
            size_arr = mx.array([data.size], dtype=mx.int32)
            size_arr = mx.distributed.all_sum(size_arr, group=self.group)
            mx.eval(size_arr)
            data = mx.distributed.all_sum(data, group=self.group)
            mx.eval(data)
            return obj

    def _receive_object(self) -> ControlEvent:
        """Non-rank0 receives object via all_sum."""
        with mx.stream(self._stream):
            size_arr = mx.array([0], dtype=mx.int32)
            size_arr = mx.distributed.all_sum(size_arr, group=self.group)
            mx.eval(size_arr)
            size = size_arr.item()
            if size == 0:
                return ControlEvent.noop()
            buf = mx.zeros(size, dtype=mx.uint8)
            buf = mx.distributed.all_sum(buf, group=self.group)
            mx.eval(buf)
            return pickle.loads(bytes(buf.tolist()))
