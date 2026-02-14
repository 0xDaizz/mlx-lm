"""Distributed control bus for cross-rank event synchronization in TP mode."""

from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .distributed import DistributedContext

logger = logging.getLogger(__name__)

MAX_BUS_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16 MB

_ALLOWED_MODULES = {
    "mlx_lm_server.distributed_bus": {"ControlEvent"},
    "mlx_lm_server.types": {"InferenceRequest"},
    "builtins": {"list", "dict", "tuple", "set", "frozenset", "str", "int", "float", "bool", "bytes", "type", "NoneType"},
    "collections": {"OrderedDict"},
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        allowed = _ALLOWED_MODULES.get(module)
        if allowed is not None and name in allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Restricted unpickle: {module}.{name} not in whitelist"
        )


def restricted_loads(data: bytes):
    return RestrictedUnpickler(io.BytesIO(data)).load()


ControlEventType = Literal["submit", "cancel", "shutdown", "noop", "batch"]


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

    @classmethod
    def batch(cls, events: list["ControlEvent"]) -> "ControlEvent":
        """Create a compound batch event containing multiple sub-events."""
        return cls(typ="batch", payload=pickle.dumps(events))

    def unpack_batch(self) -> list["ControlEvent"]:
        """Deserialize payload as a list of ControlEvents."""
        if self.payload is None:
            return []
        return restricted_loads(self.payload)

    def unpack_request(self):
        """Deserialize payload as InferenceRequest."""
        if self.payload is None:
            return None
        return restricted_loads(self.payload)

    def unpack_request_id(self) -> str | None:
        """Deserialize payload as request_id string."""
        if self.payload is None:
            return None
        return restricted_loads(self.payload)


class DistributedControlBus:
    """Broadcasts control events from rank0 to all ranks via mx.distributed.all_sum.

    Uses the same pickle + all_sum pattern as upstream mlx_lm/server.py _share_object().
    Every inference step, all ranks must call recv() exactly once to stay synchronized.
    When rank0 has no events, it sends a noop to prevent collective deadlock.
    """

    def __init__(self, dist_ctx: DistributedContext, stream=None) -> None:
        import mlx.core as mx

        self.group = dist_ctx.group
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        if stream is None:
            from mlx_lm.generate import generation_stream
            stream = generation_stream
        self._stream = stream  # Use generation_stream to serialize with model all_sum
        self._mx = mx

    def publish(self, event: ControlEvent) -> None:
        """Rank0 broadcasts an event to all ranks. Must only be called on rank0."""
        if self.rank != 0:
            raise RuntimeError("publish() must only be called on rank 0")
        self._broadcast_object(event)

    def recv(self) -> ControlEvent:
        """Receive the next event from rank0. Must only be called on rank>0.

        Blocks until rank0 calls publish(). The publish() and recv() calls
        form two halves of the same collective all_sum operations.
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
        assert self.rank == 0, "_broadcast_object must only be called from rank 0"
        raw = pickle.dumps(obj)
        if len(raw) > MAX_BUS_PAYLOAD_BYTES:
            raise ValueError(
                f"Bus payload too large: {len(raw)} bytes "
                f"(max {MAX_BUS_PAYLOAD_BYTES})"
            )
        mx = self._mx
        with mx.stream(self._stream):
            data = mx.array(raw, dtype=mx.uint8)
            size_arr = mx.array([data.size], dtype=mx.int32)
            size_arr = mx.distributed.all_sum(size_arr, group=self.group)
            mx.eval(size_arr)
            data = mx.distributed.all_sum(data, group=self.group)
            mx.eval(data)
            return obj

    def _receive_object(self) -> ControlEvent:
        """Non-rank0 receives object via all_sum."""
        mx = self._mx
        with mx.stream(self._stream):
            size_arr = mx.array([0], dtype=mx.int32)
            size_arr = mx.distributed.all_sum(size_arr, group=self.group)
            mx.eval(size_arr)
            size = size_arr.item()
            if size == 0:
                # Defensive: should not happen since broadcast always sends pickled events
                return ControlEvent.noop()
            buf = mx.zeros(size, dtype=mx.uint8)
            buf = mx.distributed.all_sum(buf, group=self.group)
            mx.eval(buf)
            try:
                return restricted_loads(bytes(buf))
            except Exception as e:
                logger.error("Failed to deserialize control event (%d bytes)", size, exc_info=True)
                raise RuntimeError(
                    f"Failed to deserialize control event ({size} bytes)"
                ) from e

    def share_object(self, obj):
        """Synchronize an object from rank0 to all ranks (blocking collective).

        Rank 0 sends obj, rank >0 ignores local obj and receives rank 0's version.
        Always call on ALL ranks (1-2 all_sum calls). Even with empty list,
        must be called unconditionally to avoid collective count divergence.
        """
        mx = self._mx
        with mx.stream(self._stream):
            if self.rank == 0:
                if not obj:  # None, empty list, etc.
                    size_arr = mx.array([0], dtype=mx.int32)
                    mx.eval(mx.distributed.all_sum(size_arr, group=self.group))
                    return obj if obj is not None else []
                else:
                    data = mx.array(list(pickle.dumps(obj)), dtype=mx.uint8)
                    size_arr = mx.array([data.size], dtype=mx.int32)
                    mx.eval(mx.distributed.all_sum(size_arr, group=self.group))
                    mx.eval(mx.distributed.all_sum(data, group=self.group))
                    return obj
            else:
                size_arr = mx.array([0], dtype=mx.int32)
                size_arr = mx.distributed.all_sum(size_arr, group=self.group)
                mx.eval(size_arr)
                size = size_arr.item()
                if size == 0:
                    return []
                data = mx.zeros(size, dtype=mx.uint8)
                data = mx.distributed.all_sum(data, group=self.group)
                mx.eval(data)
                return restricted_loads(bytes(data))
