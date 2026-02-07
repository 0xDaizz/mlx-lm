"""FastAPI server with OpenAI-compatible API for mlx-lm-server.

Provides /v1/chat/completions, /v1/completions, /v1/models, and /health endpoints.
Integrates with the Scheduler for continuous batching inference.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from queue import Queue
from typing import Any, AsyncIterator, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.types import InferenceRequest, TokenEvent


# ---------------------------------------------------------------------------
# Scheduler protocol — anything providing these methods can serve as scheduler
# ---------------------------------------------------------------------------


class SchedulerProtocol(Protocol):
    """Interface the server expects from the scheduler."""

    def submit_request(self, request: InferenceRequest) -> None: ...
    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]: ...
    def get_result(self, request_id: str) -> list[TokenEvent]: ...
    def get_cache_stats(self) -> dict[str, Any]: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Pydantic models — OpenAI-compatible request / response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | str | None = None


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "mlx-lm-server"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_stop(stop: list[str] | str | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return stop


def _make_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _make_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    config: ServerConfig,
    scheduler: SchedulerProtocol,
    tokenizer: Any = None,
) -> FastAPI:
    """Create a FastAPI application wired to the given scheduler.

    Parameters
    ----------
    config:
        Server configuration.
    scheduler:
        An object implementing SchedulerProtocol.
    tokenizer:
        A tokenizer with ``encode`` and ``decode`` methods.
        Used to convert chat messages to token ids.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Shutdown: flush and stop
        app.state.shutting_down = True
        scheduler.shutdown()

    app = FastAPI(title="mlx-lm-server", version="0.1.0", lifespan=lifespan)

    # Set state eagerly so it's available immediately (not just inside lifespan)
    app.state.config = config
    app.state.scheduler = scheduler
    app.state.tokenizer = tokenizer
    app.state.model_name = config.model
    app.state.shutting_down = False

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(exc.detail),
                    type="invalid_request_error" if exc.status_code < 500 else "server_error",
                    code=str(exc.status_code),
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="Internal server error",
                    type="server_error",
                    code="500",
                )
            ).model_dump(),
        )

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        tok = app.state.tokenizer
        if tok is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")

        # Build a prompt string from messages using chat template if available
        prompt_text = _format_chat_messages(body.messages, tok)
        prompt_tokens: list[int] = tok.encode(prompt_text)

        request_id = _make_request_id()
        stop_seqs = _normalize_stop(body.stop)

        inf_req = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop_sequences=stop_seqs,
            stream=body.stream,
        )

        sched = app.state.scheduler

        if body.stream:
            return _stream_chat_response(
                sched, inf_req, app.state.model_name, request_id, len(prompt_tokens)
            )

        # Non-streaming: submit and collect all tokens
        sched.submit_request(inf_req)
        events = await asyncio.get_event_loop().run_in_executor(
            None, sched.get_result, request_id
        )

        completion_text = "".join(e.token_text for e in events)
        finish_reason = events[-1].finish_reason if events else "stop"
        completion_tokens = len(events)

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=app.state.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_tokens) + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # POST /v1/completions
    # ------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        tok = app.state.tokenizer
        if tok is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")

        prompt_tokens: list[int] = tok.encode(body.prompt)

        request_id = _make_completion_id()
        stop_seqs = _normalize_stop(body.stop)

        inf_req = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop_sequences=stop_seqs,
            stream=body.stream,
        )

        sched = app.state.scheduler

        if body.stream:
            return _stream_completion_response(
                sched, inf_req, app.state.model_name, request_id, len(prompt_tokens)
            )

        sched.submit_request(inf_req)
        events = await asyncio.get_event_loop().run_in_executor(
            None, sched.get_result, request_id
        )

        completion_text = "".join(e.token_text for e in events)
        finish_reason = events[-1].finish_reason if events else "stop"
        completion_tokens = len(events)

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=app.state.model_name,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text=completion_text,
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_tokens) + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(
            data=[ModelInfo(id=app.state.model_name)]
        )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        sched = app.state.scheduler
        cache_stats = sched.get_cache_stats()
        return {"status": "ok", "cache_stats": cache_stats}

    return app


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------


def _format_chat_messages(messages: list[ChatMessage], tokenizer: Any) -> str:
    """Convert chat messages to a prompt string.

    Tries to use the tokenizer's ``apply_chat_template`` if available,
    otherwise falls back to a simple concatenation.
    """
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback: simple formatting
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    parts.append("assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _stream_chat_response(
    scheduler: SchedulerProtocol,
    inf_req: InferenceRequest,
    model_name: str,
    request_id: str,
    prompt_tokens: int,
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        token_queue = scheduler.register_stream(inf_req.request_id)
        scheduler.submit_request(inf_req)

        completion_tokens = 0
        loop = asyncio.get_event_loop()

        while True:
            event: TokenEvent | None = await loop.run_in_executor(
                None, lambda: token_queue.get(timeout=30)
            )
            if event is None:
                break

            completion_tokens += 1

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event.token_text},
                        "finish_reason": event.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if event.finish_reason is not None:
                break

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _stream_completion_response(
    scheduler: SchedulerProtocol,
    inf_req: InferenceRequest,
    model_name: str,
    request_id: str,
    prompt_tokens: int,
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        token_queue = scheduler.register_stream(inf_req.request_id)
        scheduler.submit_request(inf_req)

        loop = asyncio.get_event_loop()

        while True:
            event: TokenEvent | None = await loop.run_in_executor(
                None, lambda: token_queue.get(timeout=30)
            )
            if event is None:
                break

            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": event.token_text,
                        "finish_reason": event.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if event.finish_reason is not None:
                break

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args(args: list[str] | None = None) -> ServerConfig:
    """Parse CLI arguments into a ServerConfig."""
    import argparse

    parser = argparse.ArgumentParser(description="mlx-lm-server")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--kv-bits", type=int, default=8)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--ssd-cache-dir", type=str, default=None)
    parser.add_argument("--ssd-ttl-days", type=int, default=7)
    parser.add_argument("--no-ssd", action="store_true", default=False)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--prefill-batch-size", type=int, default=4)
    parser.add_argument("--max-queue-size", type=int, default=128)
    parser.add_argument("--default-max-tokens", type=int, default=512)

    parsed = parser.parse_args(args)

    kwargs: dict[str, Any] = {
        "model": parsed.model,
        "host": parsed.host,
        "port": parsed.port,
        "block_size": parsed.block_size,
        "num_blocks": parsed.num_blocks,
        "kv_bits": parsed.kv_bits,
        "kv_group_size": parsed.kv_group_size,
        "ssd_ttl_days": parsed.ssd_ttl_days,
        "ssd_enabled": not parsed.no_ssd,
        "max_batch_size": parsed.max_batch_size,
        "prefill_batch_size": parsed.prefill_batch_size,
        "max_queue_size": parsed.max_queue_size,
        "default_max_tokens": parsed.default_max_tokens,
    }

    if parsed.adapter_path is not None:
        kwargs["adapter_path"] = parsed.adapter_path
    if parsed.ssd_cache_dir is not None:
        kwargs["ssd_cache_dir"] = parsed.ssd_cache_dir

    return ServerConfig(**kwargs)
