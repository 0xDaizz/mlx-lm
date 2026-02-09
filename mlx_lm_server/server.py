"""FastAPI server with OpenAI-compatible API for mlx-lm-server.

Provides /v1/chat/completions, /v1/completions, /v1/models, and /health endpoints.
Integrates with the Scheduler for continuous batching inference.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from typing import Any, AsyncIterator, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator

from mlx_lm_server.config import ServerConfig
from mlx_lm_server.types import InferenceRequest, TokenEvent, BusOutboxFullError

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
    def unregister_stream(self, request_id: str) -> None: ...
    def get_result(self, request_id: str, timeout: float | None = None) -> list[TokenEvent]: ...
    def cancel_request(self, request_id: str) -> bool: ...
    def get_cache_stats(self) -> dict[str, Any]: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Pydantic models — OpenAI-compatible request / response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | str | None = None

    @field_validator('messages')
    @classmethod
    def messages_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError('messages must not be empty')
        return v


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

    # 5. Reject non-positive max_tokens
    if body.max_tokens is not None and body.max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )

    # 6-8. Clamp generation parameters
    temperature = max(0.0, min(2.0, body.temperature))
    top_p = max(0.0, min(1.0, body.top_p))
    max_tokens = min(max(1, body.max_tokens), config.max_generation_tokens)

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
    dist_ctx: Any = None,
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
    dist_ctx:
        Optional DistributedContext for tensor parallel status reporting.
    """

    # ------------------------------------------------------------------
    # Dedicated thread pool for inference polling
    # ------------------------------------------------------------------
    # Prevents executor starvation under concurrent load.
    # Formula: max(max_concurrent_requests, 32) + 16 headroom.
    # If max_concurrent_requests=0 (unlimited), use 128 as a sane default.
    _mcr = config.max_concurrent_requests
    _pool_size = min((max(_mcr, 32) + 16) if _mcr > 0 else 128, 256)
    _inference_executor = ThreadPoolExecutor(
        max_workers=_pool_size,
        thread_name_prefix="inference-poll",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Shutdown: flush and stop
        app.state.shutting_down = True
        _inference_executor.shutdown(wait=False)
        scheduler.shutdown()

    app = FastAPI(title="mlx-lm-server", version="0.1.0", lifespan=lifespan)

    # Set state eagerly so it's available immediately (not just inside lifespan)
    app.state.config = config
    app.state.scheduler = scheduler
    app.state.tokenizer = tokenizer
    app.state.model_name = config.model
    app.state.shutting_down = False
    app.state.dist_ctx = dist_ctx
    app.state.started_at = time.monotonic()

    # ------------------------------------------------------------------
    # Admission / security middleware
    # ------------------------------------------------------------------
    # FastAPI middleware is LIFO: last registered runs first (outermost).
    # We register concurrency_limiter FIRST, then api_key_guard, then
    # request_size_guard so execution order is:
    #   request_size_guard (outermost) -> api_key_guard -> concurrency_limiter (innermost)
    # This ensures auth rejects BEFORE semaphore acquisition, and size
    # rejects BEFORE auth processing.
    # ------------------------------------------------------------------

    health_bypass_paths = {"/health", "/livez", "/readyz", "/metrics"}
    concurrency_bypass_paths = health_bypass_paths | {"/v1/models"}
    _concurrency_sem: asyncio.Semaphore | None = None
    if config.max_concurrent_requests > 0:
        _concurrency_sem = asyncio.Semaphore(config.max_concurrent_requests)

    def _try_acquire_semaphore(sem: asyncio.Semaphore) -> bool:
        """Atomically try to acquire a semaphore without awaiting.

        Returns True if acquired, False if no permits available.
        This avoids the TOCTOU gap between locked() and await acquire().
        """
        if sem._value <= 0:
            return False
        sem._value -= 1
        return True

    @app.middleware("http")
    async def concurrency_limiter(request: Request, call_next):
        if _concurrency_sem is None:
            return await call_next(request)
        # Bypass for health and non-inference endpoints
        if request.url.path in concurrency_bypass_paths:
            return await call_next(request)
        # Atomic non-blocking acquire: no TOCTOU gap between check and acquire
        if not _try_acquire_semaphore(_concurrency_sem):
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="Server is at capacity, please retry later",
                        type="rate_limit_error",
                        code="rate_limit_exceeded",
                    )
                ).model_dump(),
            )
        try:
            return await call_next(request)
        finally:
            _concurrency_sem.release()

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):
        api_key = app.state.config.api_key
        if not api_key:
            return await call_next(request)
        if request.url.path in health_bypass_paths:
            return await call_next(request)
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        auth = request.headers.get("authorization")
        if auth is None or not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="Missing or invalid Authorization header",
                        type="authentication_error",
                        code="invalid_api_key",
                    )
                ).model_dump(),
            )

        provided = auth[len("Bearer "):].strip()
        if not hmac.compare_digest(provided, api_key):
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="Invalid API key",
                        type="authentication_error",
                        code="invalid_api_key",
                    )
                ).model_dump(),
            )
        return await call_next(request)

    @app.middleware("http")
    async def request_size_guard(request: Request, call_next):
        """Enforce request body size limit at the ASGI layer.

        Two-phase enforcement:
        1. Fast-path: reject immediately if Content-Length header exceeds limit.
        2. Slow-path: wrap the ASGI receive callable to count actual bytes
           received, catching chunked transfers and missing/forged
           Content-Length headers.  Uses a flag instead of raising inside
           the receive wrapper (Starlette's BaseHTTPMiddleware does not
           propagate exceptions from receive wrappers correctly).
        """
        if request.method in {"POST", "PUT", "PATCH"} and request.url.path.startswith("/v1/"):
            max_bytes = app.state.config.max_request_bytes
            # Fast-path: reject based on Content-Length header if present
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        return JSONResponse(
                            status_code=413,
                            content=ErrorResponse(
                                error=ErrorDetail(
                                    message=f"Request body too large (max {max_bytes} bytes)",
                                    type="invalid_request_error",
                                    code="request_too_large",
                                )
                            ).model_dump(),
                        )
                except ValueError:
                    pass
            # Slow-path: wrap the ASGI receive to count actual bytes.
            # When the limit is exceeded we set a flag and return an empty
            # body with more_body=False to terminate the stream early,
            # then replace the handler response with a 413 after call_next.
            received = 0
            oversized = False
            original_receive = request._receive

            async def counting_receive():
                nonlocal received, oversized
                if oversized:
                    # Already detected oversized — return empty to stop reads
                    return {"type": "http.request", "body": b"", "more_body": False}
                message = await original_receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"")
                    received += len(body)
                    if received > max_bytes:
                        oversized = True
                        return {"type": "http.request", "body": b"", "more_body": False}
                return message

            request._receive = counting_receive
            response = await call_next(request)
            if oversized:
                return JSONResponse(
                    status_code=413,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            message=f"Request body too large (max {max_bytes} bytes)",
                            type="invalid_request_error",
                            code="request_too_large",
                        )
                    ).model_dump(),
                )
            return response
        return await call_next(request)

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

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Format validation errors without leaking internal file paths
        # (str(exc) includes absolute paths from Pydantic internals)
        errors = exc.errors()
        parts = []
        for e in errors:
            loc = " -> ".join(str(l) for l in e.get("loc", []))
            msg = e.get("msg", "validation error")
            parts.append(f"{loc}: {msg}" if loc else msg)
        message = "; ".join(parts) if parts else "Validation error"
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=message,
                    type="invalid_request_error",
                    code="validation_error",
                )
            ).model_dump(),
        )

    # ------------------------------------------------------------------
    # T3.3: Memory pressure admission control
    # ------------------------------------------------------------------

    def _check_memory_pressure() -> bool:
        """Return True if memory pressure exceeds threshold."""
        stats = scheduler.get_cache_stats()
        total = stats.get("total_blocks", 0)
        if total == 0:
            return False
        used = stats.get("used_blocks", 0)
        return used / total >= config.memory_pressure_threshold

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
        except BusOutboxFullError:
            raise HTTPException(status_code=503, detail="Server overloaded — distributed bus full")
        except RuntimeError as e:
            if "Distributed control plane degraded" in str(e):
                raise HTTPException(status_code=503, detail=str(e))
            raise HTTPException(status_code=429, detail=str(e))

        # Track whether we successfully built the response.  If not,
        # the finally block cancels the request (handles client disconnect
        # via asyncio.CancelledError and any other unexpected failure).
        completed = False
        try:
            # Poll with short intervals to detect client disconnect.
            # Between polls, asyncio.CancelledError can propagate if client disconnects.
            loop = asyncio.get_running_loop()
            poll_interval = 2.0
            elapsed = 0.0
            events = None
            while elapsed < timeout:
                remaining = timeout - elapsed
                wait = min(poll_interval, remaining) if remaining > 0 else poll_interval
                try:
                    events = await loop.run_in_executor(
                        _inference_executor, lambda t=wait: sched.get_result(request_id, timeout=t)
                    )
                    break  # Got result
                except TimeoutError:
                    elapsed += wait
                    continue
                except KeyError as e:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Request state unavailable for {request_id}",
                    ) from e

            if not events:
                raise HTTPException(status_code=504, detail="Request timed out")

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

            result = format_response(
                request_id, app.state.model_name, completion_text,
                finish_reason, len(prompt_tokens), completion_tokens,
            )
            completed = True
            return result
        finally:
            if not completed:
                try:
                    sched.cancel_request(request_id)
                except Exception:
                    logger.warning("Best-effort cancel failed for %s", request_id)

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        tok = app.state.tokenizer
        if tok is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")

        result = _format_chat_messages(body.messages, tok, tokenize=True)
        # Handle transformers >= 5.0 BatchEncoding (dict-like with 'input_ids')
        if isinstance(result, list):
            prompt_tokens: list[int] = result
        elif hasattr(result, 'input_ids'):
            prompt_tokens = result['input_ids']
            if not isinstance(prompt_tokens, list):
                prompt_tokens = list(prompt_tokens)
        else:
            prompt_tokens: list[int] = _safe_encode(tok, result)
        request_id = _make_request_id()

        params = _validate_and_prepare_request(
            body, app.state.config, prompt_tokens,
            app.state.shutting_down, app.state.model_name,
        )

        if _check_memory_pressure():
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="Server under memory pressure, please retry later",
                        type="server_error",
                        code="memory_pressure",
                    )
                ).model_dump(),
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
                is_chat=True,
                executor=_inference_executor,
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

        if _check_memory_pressure():
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message="Server under memory pressure, please retry later",
                        type="server_error",
                        code="memory_pressure",
                    )
                ).model_dump(),
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
                is_chat=False,
                executor=_inference_executor,
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

    def _distributed_health(cache_stats: dict[str, Any]) -> dict[str, Any]:
        dist = app.state.dist_ctx
        return {
            "enabled": dist.enabled if dist else False,
            "rank": dist.rank if dist and dist.enabled else None,
            "world_size": dist.world_size if dist and dist.enabled else None,
            "fatal": bool(cache_stats.get("dist_fatal", False)),
            "fatal_reason": cache_stats.get("dist_fatal_reason") or None,
        }

    # ------------------------------------------------------------------
    # GET /livez
    # ------------------------------------------------------------------

    @app.get("/livez")
    async def livez():
        return {
            "status": "alive",
            "uptime_s": max(0.0, time.monotonic() - app.state.started_at),
        }

    # ------------------------------------------------------------------
    # GET /readyz
    # ------------------------------------------------------------------

    @app.get("/readyz")
    async def readyz():
        cache_stats = app.state.scheduler.get_cache_stats()
        dist = _distributed_health(cache_stats)
        reasons: list[str] = []
        if app.state.shutting_down:
            reasons.append("shutting_down")
        if dist["fatal"]:
            reasons.append(dist["fatal_reason"] or "distributed_fatal")
        if _check_memory_pressure():
            reasons.append("memory_pressure")
        ready = len(reasons) == 0
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "ready": ready,
                "reasons": reasons,
                "distributed": dist,
            },
        )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        cache_stats = app.state.scheduler.get_cache_stats()
        dist = _distributed_health(cache_stats)

        # T3.4: Determine load-aware status
        if app.state.shutting_down:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "shutting_down",
                    "ready": False,
                    "cache_stats": cache_stats,
                    "distributed": dist,
                },
            )

        total_blocks = cache_stats.get("total_blocks", 0)
        used_blocks = cache_stats.get("used_blocks", 0)
        utilization = used_blocks / total_blocks if total_blocks > 0 else 0.0

        threshold = config.memory_pressure_threshold
        if dist["fatal"]:
            status = "distributed_fatal"
            status_code = 503
        elif utilization >= threshold:
            status = "overloaded"
            status_code = 503
        elif utilization >= threshold * 0.8:
            status = "degraded"
            status_code = 200
        else:
            status = "ok"
            status_code = 200

        return JSONResponse(
            status_code=status_code,
            content={
                "status": status,
                "ready": status_code == 200,
                "utilization": utilization,
                "cache_stats": cache_stats,
                "distributed": dist,
            },
        )

    # ------------------------------------------------------------------
    # GET /metrics
    # ------------------------------------------------------------------

    @app.get("/metrics")
    async def metrics():
        stats = app.state.scheduler.get_cache_stats()

        def _metric_value(value: Any) -> float:
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return float(value)
            return 0.0

        lines = [
            "# HELP mlx_lm_server_active_sequences Number of active in-flight sequences.",
            "# TYPE mlx_lm_server_active_sequences gauge",
            f"mlx_lm_server_active_sequences {_metric_value(stats.get('active_sequences', 0))}",
            "# HELP mlx_lm_server_queued_requests Number of queued inference requests.",
            "# TYPE mlx_lm_server_queued_requests gauge",
            f"mlx_lm_server_queued_requests {_metric_value(stats.get('queued_requests', 0))}",
            "# HELP mlx_lm_server_used_blocks Number of used KV cache blocks.",
            "# TYPE mlx_lm_server_used_blocks gauge",
            f"mlx_lm_server_used_blocks {_metric_value(stats.get('used_blocks', 0))}",
            "# HELP mlx_lm_server_free_blocks Number of free KV cache blocks.",
            "# TYPE mlx_lm_server_free_blocks gauge",
            f"mlx_lm_server_free_blocks {_metric_value(stats.get('free_blocks', 0))}",
            "# HELP mlx_lm_server_cache_hit_rate Cache hit ratio.",
            "# TYPE mlx_lm_server_cache_hit_rate gauge",
            f"mlx_lm_server_cache_hit_rate {_metric_value(stats.get('cache_hit_rate', 0.0))}",
            "# HELP mlx_lm_server_dist_fatal Distributed control plane fatal flag.",
            "# TYPE mlx_lm_server_dist_fatal gauge",
            f"mlx_lm_server_dist_fatal {_metric_value(stats.get('dist_fatal', False))}",
            "# HELP mlx_lm_server_shutdown_clean Shutdown completed cleanly.",
            "# TYPE mlx_lm_server_shutdown_clean gauge",
            f"mlx_lm_server_shutdown_clean {_metric_value(stats.get('shutdown_clean', True))}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

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
    msg_dicts = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            if isinstance(m.content, list):
                # Multi-part content: extract text parts
                text_parts = []
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                d["content"] = "".join(text_parts)
            else:
                d["content"] = m.content
        else:
            d["content"] = ""
        if m.name is not None:
            d["name"] = m.name
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        msg_dicts.append(d)

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # return_dict=False ensures transformers >= 5.0 returns list[int]
            # instead of BatchEncoding when tokenize=True.
            return tokenizer.apply_chat_template(
                msg_dicts, tokenize=tokenize, add_generation_prompt=True,
                return_dict=False,
            )
        except TypeError:
            # Tokenizer doesn't accept return_dict — retry without it
            try:
                return tokenizer.apply_chat_template(
                    msg_dicts, tokenize=tokenize, add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning("apply_chat_template failed: %s, using fallback format", e)
        except Exception as e:
            logger.warning("apply_chat_template failed: %s, using fallback format", e)

    # Fallback: simple formatting
    parts: list[str] = []
    for m in msg_dicts:
        parts.append(f"{m['role']}: {m.get('content', '')}")
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
    is_chat: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> StreamingResponse:
    """Unified SSE streaming for both chat and completion endpoints.

    Args:
        format_chunk: Callable(request_id, model_name, token_text, finish_reason) -> dict
            Creates the SSE chunk payload. Differs between chat and completion formats.
        request_timeout_s: Per-token timeout in seconds for the stream queue.
        first_token_timeout_s: Timeout for the first token (prefill). If None,
            uses *request_timeout_s* for all tokens.
        executor: ThreadPoolExecutor for polling. If None, uses the default executor.
    """
    # Register the stream and submit BEFORE creating the generator.
    # This allows us to raise HTTPException (proper 503) for distributed errors
    # instead of yielding an SSE error payload. Raising HTTPException inside
    # an async generator doesn't work with Starlette — it bypasses the
    # exception handler and crashes the stream.
    token_queue = scheduler.register_stream(inf_req.request_id)
    try:
        scheduler.submit_request(inf_req)
    except BusOutboxFullError:
        scheduler.unregister_stream(inf_req.request_id)
        raise HTTPException(status_code=503, detail="Server overloaded — distributed bus full")
    except RuntimeError as e:
        scheduler.unregister_stream(inf_req.request_id)
        if "Distributed control plane degraded" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=429, detail=str(e))
    except Exception:
        scheduler.unregister_stream(inf_req.request_id)
        raise

    async def event_generator() -> AsyncIterator[str]:
        try:
            # Emit role delta as first chunk for chat completions (OpenAI spec)
            if is_chat:
                role_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

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
                        executor, lambda t=timeout: token_queue.get(timeout=t)
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
            try:
                scheduler.cancel_request(inf_req.request_id)
            except Exception:
                logger.warning("Best-effort stream cancel failed for %s", inf_req.request_id)

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
    parser.add_argument("--api-key", type=str, default=os.environ.get("MLX_LM_SERVER_API_KEY"))
    parser.add_argument("--api-key-file", type=str, default=os.environ.get("MLX_LM_SERVER_API_KEY_FILE"))
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=1_048_576,
        help="Maximum accepted request body size in bytes",
    )
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
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--ssd-async-writes",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    else:
        parser.add_argument(
            "--ssd-async-writes",
            dest="ssd_async_writes",
            action="store_true",
        )
        parser.add_argument(
            "--no-ssd-async-writes",
            dest="ssd_async_writes",
            action="store_false",
        )
        parser.set_defaults(ssd_async_writes=True)
    parser.add_argument("--ssd-writer-queue-size", type=int, default=512)
    parser.add_argument("--ssd-persistent-max-retries", type=int, default=3)
    parser.add_argument("--ssd-flush-interval-s", type=float, default=1.0)
    parser.add_argument("--ssd-max-size-gb", type=float, default=50.0,
                        help="Maximum SSD cache size in GB (0 = unlimited)")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--prefill-batch-size", type=int, default=4)
    parser.add_argument("--max-queue-size", type=int, default=128)
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--completion-batch-size", type=int, default=32)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--sequence-cache-size", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--max-generation-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--first-token-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-concurrent-requests", type=int, default=64,
                        help="Max concurrent inference requests (0 = unlimited)")
    parser.add_argument("--memory-pressure-threshold", type=float, default=0.9,
                        help="Reject requests when block utilization >= this (0.0-1.0)")

    # Distributed / Tensor Parallel
    parser.add_argument("--distributed-mode", type=str, default="off",
                        choices=["off", "ring", "jaccl"],
                        help="Distributed backend: off (single-machine), ring, or jaccl (RDMA)")
    parser.add_argument("--distributed-sharding", type=str, default="tensor",
                        choices=["tensor", "pipeline"],
                        help="Sharding strategy: tensor or pipeline")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--distributed-strict",
                            action=argparse.BooleanOptionalAction, default=True,
                            help="Strict distributed init (default: True)")
    else:
        parser.add_argument("--distributed-strict",
                            dest="distributed_strict", action="store_true")
        parser.add_argument("--no-distributed-strict",
                            dest="distributed_strict", action="store_false")
        parser.set_defaults(distributed_strict=True)
    parser.add_argument("--distributed-hostfile", type=str, default=os.environ.get("MLX_HOSTFILE"),
                        help="Hostfile path for ring backend")
    parser.add_argument("--distributed-ibv-devices", type=str, default=os.environ.get("MLX_IBV_DEVICES"),
                        help="IBV devices path for jaccl backend")
    parser.add_argument("--distributed-jaccl-coordinator", type=str, default=os.environ.get("MLX_JACCL_COORDINATOR"),
                        help="JACCL coordinator address (host:port)")
    parser.add_argument("--num-local-ranks", type=int, default=None,
                        help="Number of local ranks for single-machine TP (used by auto-relaunch)")

    parsed = parser.parse_args(args)

    if parsed.ssd_writer_queue_size < 1:
        parser.error("--ssd-writer-queue-size must be >= 1")
    if parsed.ssd_persistent_max_retries < 0:
        parser.error("--ssd-persistent-max-retries must be >= 0")
    if parsed.ssd_flush_interval_s <= 0:
        parser.error("--ssd-flush-interval-s must be > 0")
    if parsed.ssd_max_size_gb < 0:
        parser.error("--ssd-max-size-gb must be >= 0")
    if parsed.max_request_bytes <= 0:
        parser.error("--max-request-bytes must be > 0")

    # Core numeric parameter validation
    if parsed.block_size <= 0:
        parser.error("--block-size must be > 0")
    if parsed.num_blocks <= 0:
        parser.error("--num-blocks must be > 0")
    if parsed.max_batch_size <= 0:
        parser.error("--max-batch-size must be > 0")
    if parsed.max_queue_size <= 0:
        parser.error("--max-queue-size must be > 0")
    if parsed.prefill_batch_size > parsed.max_batch_size:
        parser.error("--prefill-batch-size cannot exceed --max-batch-size")
    if parsed.max_concurrent_requests < 0:
        parser.error("--max-concurrent-requests must be >= 0")
    if not (0.0 <= parsed.memory_pressure_threshold <= 1.0):
        parser.error("--memory-pressure-threshold must be between 0.0 and 1.0")
    if parsed.default_max_tokens <= 0:
        parser.error("--default-max-tokens must be > 0")
    if parsed.completion_batch_size <= 0:
        parser.error("--completion-batch-size must be > 0")
    if parsed.max_prompt_tokens <= 0:
        parser.error("--max-prompt-tokens must be > 0")
    if parsed.max_generation_tokens <= 0:
        parser.error("--max-generation-tokens must be > 0")
    if parsed.request_timeout_s <= 0:
        parser.error("--request-timeout-s must be > 0")
    if parsed.first_token_timeout_s <= 0:
        parser.error("--first-token-timeout-s must be > 0")
    if parsed.num_local_ranks is not None and parsed.num_local_ranks <= 0:
        parser.error("--num-local-ranks must be > 0")

    if parsed.api_key is not None and parsed.api_key_file is not None:
        parser.error("Specify only one of --api-key or --api-key-file")

    api_key = parsed.api_key
    if parsed.api_key_file is not None:
        if not os.path.exists(parsed.api_key_file):
            parser.error(f"--api-key-file path does not exist: {parsed.api_key_file}")
        try:
            with open(parsed.api_key_file, encoding="utf-8") as f:
                api_key = f.read().strip()
        except OSError as e:
            parser.error(f"Failed to read --api-key-file: {e}")
        if not api_key:
            parser.error("--api-key-file is empty")
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            parser.error("--api-key must not be empty")

    # If num_local_ranks is set, hostfile is irrelevant — force clear
    if parsed.num_local_ranks is not None and parsed.distributed_hostfile is not None:
        logging.getLogger(__name__).warning(
            "Both --num-local-ranks and --distributed-hostfile/MLX_HOSTFILE set; "
            "ignoring hostfile in favor of local ranks"
        )
        parsed.distributed_hostfile = None

    # Distributed validation
    already_launched = os.environ.get("MLX_RANK") is not None
    if (
        parsed.distributed_mode == "ring"
        and parsed.distributed_hostfile is None
        and parsed.num_local_ranks is None
        and not already_launched
    ):
        parser.error(
            "--distributed-hostfile or --num-local-ranks is required "
            "when --distributed-mode=ring"
        )
    if parsed.distributed_mode == "jaccl" and not already_launched:
        if parsed.distributed_ibv_devices is None or parsed.distributed_jaccl_coordinator is None:
            parser.error(
                "--distributed-ibv-devices and --distributed-jaccl-coordinator "
                "are required when --distributed-mode=jaccl"
            )
    if parsed.distributed_mode != "off" and parsed.adapter_path is not None:
        parser.error("Distributed mode does not support adapter_path")
    if parsed.distributed_sharding == "pipeline":
        parser.error(
            "Pipeline sharding is not supported in v1. "
            "Use --distributed-sharding tensor"
        )
    if parsed.distributed_mode == "off" and any(
        v is not None for v in (
            parsed.distributed_hostfile,
            parsed.distributed_ibv_devices,
            parsed.distributed_jaccl_coordinator,
        )
    ):
        logging.getLogger(__name__).warning(
            "Distributed flags (--distributed-hostfile, --distributed-ibv-devices, "
            "--distributed-jaccl-coordinator) are ignored when --distributed-mode=off"
        )

    if (
        parsed.distributed_mode == "ring"
        and parsed.distributed_hostfile is not None
        and parsed.num_local_ranks is None
        and not already_launched
        and not os.path.exists(parsed.distributed_hostfile)
    ):
        parser.error(f"--distributed-hostfile path does not exist: {parsed.distributed_hostfile}")
    if parsed.distributed_mode == "jaccl" and parsed.distributed_ibv_devices is not None and not already_launched and not os.path.exists(parsed.distributed_ibv_devices):
        parser.error(f"--distributed-ibv-devices path does not exist: {parsed.distributed_ibv_devices}")

    kwargs: dict[str, Any] = {
        "model": parsed.model,
        "host": parsed.host,
        "port": parsed.port,
        "max_request_bytes": parsed.max_request_bytes,
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
        "ssd_max_size_gb": parsed.ssd_max_size_gb,
        "max_batch_size": parsed.max_batch_size,
        "prefill_batch_size": parsed.prefill_batch_size,
        "max_queue_size": parsed.max_queue_size,
        "default_max_tokens": parsed.default_max_tokens,
        "completion_batch_size": parsed.completion_batch_size,
        "sequence_cache_size": parsed.sequence_cache_size,
        "max_prompt_tokens": parsed.max_prompt_tokens,
        "max_generation_tokens": parsed.max_generation_tokens,
        "request_timeout_s": parsed.request_timeout_s,
        "first_token_timeout_s": parsed.first_token_timeout_s,
        "max_concurrent_requests": parsed.max_concurrent_requests,
        "memory_pressure_threshold": parsed.memory_pressure_threshold,
        "distributed_mode": parsed.distributed_mode,
        "distributed_sharding": parsed.distributed_sharding,
        "distributed_strict": parsed.distributed_strict,
    }

    if api_key is not None:
        kwargs["api_key"] = api_key

    if parsed.distributed_hostfile is not None:
        kwargs["distributed_hostfile"] = parsed.distributed_hostfile
    if parsed.distributed_ibv_devices is not None:
        kwargs["distributed_ibv_devices"] = parsed.distributed_ibv_devices
    if parsed.distributed_jaccl_coordinator is not None:
        kwargs["distributed_jaccl_coordinator"] = parsed.distributed_jaccl_coordinator

    if parsed.adapter_path is not None:
        kwargs["adapter_path"] = parsed.adapter_path
    if parsed.ssd_cache_dir is not None:
        kwargs["ssd_cache_dir"] = Path(parsed.ssd_cache_dir)
    if parsed.max_kv_size is not None:
        kwargs["max_kv_size"] = parsed.max_kv_size

    return ServerConfig(**kwargs)
