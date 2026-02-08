"""FastAPI server with OpenAI-compatible API for mlx-lm-server.

Provides /v1/chat/completions, /v1/completions, /v1/models, and /health endpoints.
Integrates with the Scheduler for continuous batching inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _safe_encode(tokenizer: Any, text: str) -> list[int]:
    """Encode text, suppressing special tokens if the tokenizer supports it."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _get_eos_token_ids(tokenizer: Any) -> set[int]:
    """Safely extract EOS token ID set from a tokenizer."""
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids, (set, frozenset, list)):
        return set(eos_ids)
    if isinstance(eos_ids, int):
        return {eos_ids}
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return {eos_id}
    return set()


# ---------------------------------------------------------------------------
# Scheduler protocol — anything providing these methods can serve as scheduler
# ---------------------------------------------------------------------------


class SchedulerProtocol(Protocol):
    """Interface the server expects from the scheduler."""

    def submit_request(self, request: InferenceRequest) -> None: ...
    def register_stream(self, request_id: str) -> Queue[TokenEvent | None]: ...
    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]: ...
    def cancel_request(self, request_id: str) -> bool: ...
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


class ValidatedParams:
    """Holds validated and clamped request parameters."""

    __slots__ = ("temperature", "top_p", "max_tokens", "stop_sequences")

    def __init__(self, temperature: float, top_p: float, max_tokens: int, stop_sequences: list[str]):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences


def _validate_and_prepare_request(
    body: ChatCompletionRequest | CompletionRequest,
    config: ServerConfig,
    prompt_tokens: list[int],
    shutting_down: bool,
    model_name: str,
) -> ValidatedParams:
    """Validate common request fields and return clamped parameters.

    Raises HTTPException on validation failure.
    """
    # 1. Server shutting down
    if shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")

    # 2. Model mismatch (skip if client didn't specify a model)
    if body.model and body.model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Model mismatch: requested '{body.model}' but server loaded '{model_name}'",
        )

    # 3. Empty prompt
    if len(prompt_tokens) == 0:
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    # 4. Prompt length
    if len(prompt_tokens) > config.max_prompt_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long: {len(prompt_tokens)} tokens exceeds maximum of {config.max_prompt_tokens}",
        )

    # 5-7. Clamp generation parameters
    temperature = max(0.0, min(2.0, body.temperature))
    top_p = max(0.0, min(1.0, body.top_p))
    max_tokens = max(1, body.max_tokens)

    # 8. Normalize stop sequences
    stop_sequences = _normalize_stop(body.stop)

    return ValidatedParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
    )


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
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
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
    # Common non-streaming inference helper
    # ------------------------------------------------------------------

    async def _do_inference(
        prompt_tokens: list[int],
        request_id: str,
        inf_req: InferenceRequest,
        format_response,
    ):
        """Common inference logic for non-streaming requests.

        Args:
            format_response: Callable(request_id, model_name, completion_text,
                finish_reason, prompt_token_count, completion_token_count) -> response model
        """
        sched = app.state.scheduler
        tok = app.state.tokenizer
        timeout = app.state.config.request_timeout_s

        try:
            sched.submit_request(inf_req)
        except RuntimeError as e:
            raise HTTPException(status_code=429, detail=str(e))
        try:
            events = await asyncio.get_running_loop().run_in_executor(
                None, lambda: sched.get_result(request_id, timeout=timeout)
            )
        except TimeoutError:
            sched.cancel_request(request_id)
            raise HTTPException(status_code=504, detail="Request timed out")

        if not events:
            sched.cancel_request(request_id)
            raise HTTPException(status_code=504, detail="Request timed out waiting for inference")

        # Exclude EOS token from output if present (by token_id, not text)
        filtered_events = events
        eos_ids = _get_eos_token_ids(tok) if tok else set()
        if events and eos_ids and events[-1].token_id in eos_ids:
            filtered_events = events[:-1]
        completion_text = "".join(e.token_text for e in filtered_events)
        finish_reason = events[-1].finish_reason if events else "stop"

        # Truncate stop-sequence text from output (the scheduler truncates
        # seq.output_text, but individual TokenEvent.token_text values still
        # carry the raw text, so the joined string may contain the stop sequence).
        if finish_reason == "stop" and inf_req.stop_sequences:
            for stop_seq in inf_req.stop_sequences:
                idx = completion_text.find(stop_seq)
                if idx != -1:
                    completion_text = completion_text[:idx]
                    break

        completion_tokens = len(filtered_events)

        return format_response(
            request_id, app.state.model_name, completion_text,
            finish_reason, len(prompt_tokens), completion_tokens,
        )

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        tok = app.state.tokenizer
        if tok is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")

        result = _format_chat_messages(body.messages, tok, tokenize=True)
        prompt_tokens: list[int] = result if isinstance(result, list) else _safe_encode(tok, result)
        request_id = _make_request_id()

        params = _validate_and_prepare_request(
            body, app.state.config, prompt_tokens,
            app.state.shutting_down, app.state.model_name,
        )

        inf_req = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop_sequences=params.stop_sequences,
            stream=body.stream,
        )

        if body.stream:
            return _stream_response(
                app.state.scheduler, inf_req, app.state.model_name,
                request_id, len(prompt_tokens), tok,
                format_chunk=_format_chat_chunk,
                request_timeout_s=app.state.config.request_timeout_s,
                first_token_timeout_s=app.state.config.first_token_timeout_s,
            )

        return await _do_inference(prompt_tokens, request_id, inf_req, _format_chat_response)

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

        params = _validate_and_prepare_request(
            body, app.state.config, prompt_tokens,
            app.state.shutting_down, app.state.model_name,
        )

        inf_req = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop_sequences=params.stop_sequences,
            stream=body.stream,
        )

        if body.stream:
            return _stream_response(
                app.state.scheduler, inf_req, app.state.model_name,
                request_id, len(prompt_tokens), tok,
                format_chunk=_format_completion_chunk,
                request_timeout_s=app.state.config.request_timeout_s,
                first_token_timeout_s=app.state.config.first_token_timeout_s,
            )

        return await _do_inference(prompt_tokens, request_id, inf_req, _format_completion_response)

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


def _format_chat_messages(
    messages: list[ChatMessage], tokenizer: Any, tokenize: bool = False
) -> str | list[int]:
    """Convert chat messages to a prompt string or token list.

    When *tokenize* is True, tries to get a ``list[int]`` directly from the
    tokenizer's ``apply_chat_template``.  If the tokenizer returns a ``str``
    instead (e.g. SimpleTokenizer), the caller should fall back to
    ``_safe_encode``.

    Returns either ``str`` or ``list[int]``.
    """
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msg_dicts, tokenize=tokenize, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning("apply_chat_template failed: %s, using fallback format", e)

    # Fallback: simple formatting
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    parts.append("assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chunk / response formatters
# ---------------------------------------------------------------------------


def _format_chat_chunk(request_id: str, model_name: str, token_text: str, finish_reason: str | None) -> dict:
    """Format a chat completion SSE chunk."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": finish_reason}],
    }


def _format_completion_chunk(request_id: str, model_name: str, token_text: str, finish_reason: str | None) -> dict:
    """Format a text completion SSE chunk."""
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "text": token_text, "finish_reason": finish_reason}],
    }


def _format_chat_response(request_id, model_name, text, finish_reason, prompt_tokens, completion_tokens):
    """Format a non-streaming chat completion response."""
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
            finish_reason=finish_reason,
        )],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _format_completion_response(request_id, model_name, text, finish_reason, prompt_tokens, completion_tokens):
    """Format a non-streaming text completion response."""
    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[CompletionResponseChoice(
            index=0,
            text=text,
            finish_reason=finish_reason,
        )],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def _stream_response(
    scheduler: SchedulerProtocol,
    inf_req: InferenceRequest,
    model_name: str,
    request_id: str,
    prompt_tokens: int,
    tokenizer: Any = None,
    format_chunk=None,
    request_timeout_s: float = 120.0,
    first_token_timeout_s: float | None = None,
) -> StreamingResponse:
    """Unified SSE streaming for both chat and completion endpoints.

    Args:
        format_chunk: Callable(request_id, model_name, token_text, finish_reason) -> dict
            Creates the SSE chunk payload. Differs between chat and completion formats.
        request_timeout_s: Per-token timeout in seconds for the stream queue.
        first_token_timeout_s: Timeout for the first token (prefill). If None,
            uses *request_timeout_s* for all tokens.
    """
    async def event_generator() -> AsyncIterator[str]:
        token_queue = scheduler.register_stream(inf_req.request_id)
        try:
            try:
                scheduler.submit_request(inf_req)
            except RuntimeError as e:
                error_data = {"error": {"message": str(e), "type": "rate_limit_error"}}
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            loop = asyncio.get_running_loop()
            eos_ids = _get_eos_token_ids(tokenizer) if tokenizer else set()
            stop_sequences = inf_req.stop_sequences or []
            max_stop_len = max((len(s) for s in stop_sequences), default=0)
            text_buffer = ""
            first_token = True

            while True:
                # C3: use longer timeout for first token (prefill)
                timeout = (
                    first_token_timeout_s if first_token and first_token_timeout_s is not None
                    else request_timeout_s
                )
                try:
                    event: TokenEvent | None = await loop.run_in_executor(
                        None, lambda t=timeout: token_queue.get(timeout=t)
                    )
                except queue.Empty:
                    error_data = {"error": {"message": f"Stream timeout: no tokens received for {timeout} seconds", "type": "server_error"}}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
                if event is None:
                    break
                first_token = False

                # C2: Filter EOS token by token_id instead of text comparison
                token_text = event.token_text
                if eos_ids and event.token_id in eos_ids:
                    token_text = ""

                # C1: Stop sequence buffering
                if max_stop_len > 0:
                    # Buffer mode: accumulate text and check for stop sequences
                    text_buffer += token_text

                    # Check for stop sequence match in buffer
                    stop_found = False
                    for stop_seq in stop_sequences:
                        idx = text_buffer.find(stop_seq)
                        if idx != -1:
                            safe_text = text_buffer[:idx]
                            if safe_text:
                                yield f"data: {json.dumps(format_chunk(request_id, model_name, safe_text, None))}\n\n"
                            yield f"data: {json.dumps(format_chunk(request_id, model_name, '', 'stop'))}\n\n"
                            stop_found = True
                            scheduler.cancel_request(request_id)
                            break

                    if stop_found:
                        break

                    if event.finish_reason is not None:
                        # Natural end: flush remaining buffer with final stop check
                        final_text = text_buffer
                        for stop_seq in stop_sequences:
                            idx = final_text.find(stop_seq)
                            if idx != -1:
                                final_text = final_text[:idx]
                                break
                        if final_text:
                            yield f"data: {json.dumps(format_chunk(request_id, model_name, final_text, None))}\n\n"
                        yield f"data: {json.dumps(format_chunk(request_id, model_name, '', event.finish_reason))}\n\n"
                        break

                    # Flush safe prefix (keep max_stop_len - 1 chars buffered)
                    if len(text_buffer) > max_stop_len - 1:
                        safe_len = len(text_buffer) - (max_stop_len - 1)
                        safe_text = text_buffer[:safe_len]
                        text_buffer = text_buffer[safe_len:]
                        yield f"data: {json.dumps(format_chunk(request_id, model_name, safe_text, None))}\n\n"
                else:
                    # No stop sequences: original behavior (zero overhead)
                    chunk = format_chunk(request_id, model_name, token_text, event.finish_reason)
                    yield f"data: {json.dumps(chunk)}\n\n"

                    if event.finish_reason is not None:
                        break

            yield "data: [DONE]\n\n"
        finally:
            # Always cancel to clean up scheduler resources (stream queue, etc.)
            # Safe to call even if request already finished — cancel is idempotent.
            scheduler.cancel_request(inf_req.request_id)

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
    parser.add_argument("--ssd-policy", type=str,
                        choices=["evict_only", "write_through"],
                        default="evict_only")
    parser.add_argument("--ssd-durability", type=str,
                        choices=["best_effort", "persistent"],
                        default="best_effort")
    parser.add_argument("--ssd-async-writes",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--ssd-writer-queue-size", type=int, default=512)
    parser.add_argument("--ssd-persistent-max-retries", type=int, default=3)
    parser.add_argument("--ssd-flush-interval-s", type=float, default=1.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--prefill-batch-size", type=int, default=4)
    parser.add_argument("--max-queue-size", type=int, default=128)
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--completion-batch-size", type=int, default=32)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--sequence-cache-size", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--first-token-timeout-s", type=float, default=300.0)

    parsed = parser.parse_args(args)

    if parsed.ssd_writer_queue_size < 1:
        parser.error("--ssd-writer-queue-size must be >= 1")
    if parsed.ssd_persistent_max_retries < 0:
        parser.error("--ssd-persistent-max-retries must be >= 0")
    if parsed.ssd_flush_interval_s <= 0:
        parser.error("--ssd-flush-interval-s must be > 0")

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
        "ssd_policy": parsed.ssd_policy,
        "ssd_durability": parsed.ssd_durability,
        "ssd_async_writes": parsed.ssd_async_writes,
        "ssd_writer_queue_size": parsed.ssd_writer_queue_size,
        "ssd_persistent_max_retries": parsed.ssd_persistent_max_retries,
        "ssd_flush_interval_s": parsed.ssd_flush_interval_s,
        "max_batch_size": parsed.max_batch_size,
        "prefill_batch_size": parsed.prefill_batch_size,
        "max_queue_size": parsed.max_queue_size,
        "default_max_tokens": parsed.default_max_tokens,
        "completion_batch_size": parsed.completion_batch_size,
        "sequence_cache_size": parsed.sequence_cache_size,
        "max_prompt_tokens": parsed.max_prompt_tokens,
        "request_timeout_s": parsed.request_timeout_s,
        "first_token_timeout_s": parsed.first_token_timeout_s,
    }

    if parsed.adapter_path is not None:
        kwargs["adapter_path"] = parsed.adapter_path
    if parsed.ssd_cache_dir is not None:
        kwargs["ssd_cache_dir"] = parsed.ssd_cache_dir
    if parsed.max_kv_size is not None:
        kwargs["max_kv_size"] = parsed.max_kv_size

    return ServerConfig(**kwargs)
