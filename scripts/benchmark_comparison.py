#!/usr/bin/env python3
"""A/B benchmark: mlx_lm_server (ours) vs mlx_lm.server (baseline).

Compares our enhanced server (continuous batching, prefix caching, paged KV
cache) against the original sequential mlx_lm server across four scenarios:

  1. Concurrent Request Throughput   -- N parallel requests
  2. Multi-Agent Prefix Sharing      -- shared system prompt + different user messages
  3. Repeated Context (Chat History)  -- growing multi-turn conversation
  4. Single Request Latency           -- overhead sanity check

For the baseline we call ``stream_generate`` directly (simulating sequential
processing, which is what mlx_lm.server does under the hood). For ours we
use the Scheduler with concurrent threading.

Usage:
    python scripts/benchmark_comparison.py --repeat 2
    python scripts/benchmark_comparison.py --scenario all --repeat 1
"""

from __future__ import annotations

import argparse
import copy
import queue
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_MODEL_PATH = str(PROJECT_ROOT / "Qwen3-4B-4bit")
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Timing metrics for a single request."""

    ttft_ms: float = 0.0
    total_time_s: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    error: str | None = None

    @property
    def throughput_tps(self) -> float:
        if self.total_time_s <= 0 or self.tokens_generated <= 0:
            return 0.0
        return self.tokens_generated / self.total_time_s


@dataclass
class ScenarioResult:
    """Aggregated results for one side (baseline or ours) of a scenario."""

    label: str = ""
    wall_time_s: float = 0.0
    per_request: list[RequestResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.per_request)

    @property
    def agg_throughput_tps(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_tokens / self.wall_time_s

    @property
    def avg_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self.per_request if r.ttft_ms > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_total_s(self) -> float:
        vals = [r.total_time_s for r in self.per_request if r.total_time_s > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def num_errors(self) -> int:
        return sum(1 for r in self.per_request if r.error is not None)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[Any, Any]] = {}


def load_model(model_path: str) -> tuple[Any, Any]:
    if model_path in _model_cache:
        return _model_cache[model_path]
    resolved = model_path
    if Path(LOCAL_MODEL_PATH).is_dir() and model_path == DEFAULT_MODEL:
        resolved = LOCAL_MODEL_PATH
    elif Path(model_path).is_dir():
        resolved = model_path
    print(f"  Loading model from {resolved} ...")
    t0 = time.time()
    from mlx_lm import load
    model, tokenizer = load(resolved)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    _model_cache[model_path] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Baseline helpers  (sequential, simulating mlx_lm.server)
# ---------------------------------------------------------------------------


def baseline_generate_one(
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float = 0.0,
) -> RequestResult:
    """Run one request through the baseline pipeline (stream_generate).

    This simulates what the original mlx_lm.server does for a single request:
    sequential generate, no batching, per-request fresh prompt cache.
    """
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache

    result = RequestResult(prompt_tokens=len(prompt_tokens))
    sampler = make_sampler(temp=temperature)
    prompt_cache = make_prompt_cache(model)

    try:
        t_start = time.perf_counter()
        first_token_time = None
        n_tokens = 0

        for gen in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            n_tokens += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()
                result.ttft_ms = (first_token_time - t_start) * 1000

        t_end = time.perf_counter()
        result.total_time_s = t_end - t_start
        result.tokens_generated = n_tokens
    except Exception as e:
        result.error = str(e)

    return result


def baseline_sequential(
    model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    max_tokens: int,
    temperature: float = 0.0,
) -> ScenarioResult:
    """Run multiple prompts sequentially (simulating baseline under concurrency).

    The baseline server processes one request at a time, so concurrent requests
    queue up and total wall time = sum of individual times.
    """
    sr = ScenarioResult(label="baseline")
    t_wall_start = time.perf_counter()
    for prompt_tokens in prompts:
        r = baseline_generate_one(model, tokenizer, prompt_tokens, max_tokens, temperature)
        sr.per_request.append(r)
    sr.wall_time_s = time.perf_counter() - t_wall_start
    return sr


# ---------------------------------------------------------------------------
# Our server helpers  (Scheduler with continuous batching)
# ---------------------------------------------------------------------------


def _make_scheduler(
    model: Any,
    tokenizer: Any,
    model_path: str,
    max_batch_size: int = 8,
    num_blocks: int = 256,
    block_size: int = 16,
) -> Any:
    """Create a Scheduler + KVCacheManager with real model."""
    from mlx_lm_server.config import ServerConfig
    from mlx_lm_server.kv_cache_manager import KVCacheManager
    from mlx_lm_server.scheduler import Scheduler

    config = ServerConfig(
        model=model_path,
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_queue_size=64,
        ssd_enabled=False,
    )
    kv_mgr = KVCacheManager(config)
    sched = Scheduler(
        config=config,
        model=model,
        tokenizer=tokenizer,
        kv_cache_manager=kv_mgr,
    )
    sched.run_inference_loop()
    return sched


def ours_generate_one_streaming(
    sched: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float = 0.0,
) -> RequestResult:
    """Submit one request to our scheduler and collect streaming results."""
    from mlx_lm_server.types import InferenceRequest

    rid = f"bench-{uuid.uuid4().hex[:10]}"
    req = InferenceRequest(
        request_id=rid,
        prompt_tokens=list(prompt_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    result = RequestResult(prompt_tokens=len(prompt_tokens))
    stream_q = sched.register_stream(rid)

    t_start = time.perf_counter()
    sched.submit_request(req)

    first_token_time = None
    n_tokens = 0

    deadline = time.time() + 120.0
    while time.time() < deadline:
        try:
            ev = stream_q.get(timeout=5.0)
        except queue.Empty:
            continue
        n_tokens += 1
        if first_token_time is None:
            first_token_time = time.perf_counter()
            result.ttft_ms = (first_token_time - t_start) * 1000
        if ev.finish_reason is not None:
            break

    t_end = time.perf_counter()
    result.total_time_s = t_end - t_start
    result.tokens_generated = n_tokens
    return result


def ours_concurrent(
    sched: Any,
    prompts: list[list[int]],
    max_tokens: int,
    temperature: float = 0.0,
) -> ScenarioResult:
    """Submit all prompts concurrently to our scheduler and measure wall time."""
    sr = ScenarioResult(label="ours")
    results: list[RequestResult] = []
    lock = threading.Lock()

    def worker(prompt_tokens: list[int]) -> None:
        r = ours_generate_one_streaming(sched, prompt_tokens, max_tokens, temperature)
        with lock:
            results.append(r)

    threads = []
    t_wall_start = time.perf_counter()
    for pt in prompts:
        t = threading.Thread(target=worker, args=(pt,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=180.0)
    sr.wall_time_s = time.perf_counter() - t_wall_start
    sr.per_request = results
    return sr


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an advanced AI research assistant specializing in computer science "
    "and machine learning. You provide detailed, accurate, and well-structured "
    "responses. You follow instructions carefully and format your output clearly. "
    "When analyzing code, you look for correctness, performance, and security issues. "
    "You cite sources when making factual claims. You break down complex problems "
    "into manageable steps and explain your reasoning. You are helpful, harmless, "
    "and honest. Today's date is February 2026. You are running on Apple Silicon "
    "hardware using the MLX framework for inference. Your responses should be "
    "concise but thorough, typically 2-3 paragraphs unless asked otherwise."
)

AGENT_MESSAGES = [
    "What are the key differences between transformer and mamba architectures?",
    "Explain how prefix caching works in vLLM and why it matters for multi-agent systems.",
    "Describe the advantages of continuous batching over static batching for LLM serving.",
    "How does Apple's MLX framework compare to PyTorch for on-device inference?",
    "What are the main challenges in distributed inference across multiple Mac Studios?",
    "Explain KV cache quantization: what are the tradeoffs of INT8 vs FP16?",
    "How does paged attention differ from standard multi-head attention?",
    "What role does speculative decoding play in reducing inference latency?",
]

CHAT_TURNS = [
    "Hi, I need help understanding KV caches in transformer models. What are they and why do they matter for inference?",
    "Thanks! Now explain how paged attention improves memory efficiency compared to the naive approach.",
    "Interesting. Can you also cover how hash-based prefix caching works? I want to understand the block-level caching approach.",
    "One more question: how does continuous batching allow higher throughput than static batching?",
]


def build_chat_prompt(tokenizer: Any, system: str, messages: list[dict]) -> list[int]:
    """Build tokenized prompt from chat messages using the tokenizer's template."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return tokenizer.encode(text)
        except Exception:
            pass
    # Fallback
    parts = []
    for m in messages:
        parts.append(f"{m['role']}: {m['content']}")
    parts.append("assistant:")
    return tokenizer.encode("\n".join(parts))


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _speedup_str(base_val: float, ours_val: float) -> str:
    """Format speedup: '2.5x' or '-' if not meaningful."""
    if base_val <= 0 or ours_val <= 0:
        return "-"
    ratio = base_val / ours_val
    if ratio >= 1.0:
        return f"{ratio:.1f}x"
    else:
        return f"{1/ratio:.1f}x slower"


def print_scenario(
    title: str,
    base: ScenarioResult,
    ours: ScenarioResult,
    extra_rows: list[tuple[str, str, str, str]] | None = None,
) -> None:
    """Print a comparison table for one scenario."""
    w = 64
    print()
    print(f"  {title}")
    print(f"  {'=' * w}")
    print(f"  {'':24s} {'Baseline':>14s} {'Ours':>14s} {'Speedup':>10s}")
    print(f"  {'-' * w}")

    # Wall time
    print(
        f"  {'Wall time:':24s} {base.wall_time_s:>13.2f}s {ours.wall_time_s:>13.2f}s "
        f"{_speedup_str(base.wall_time_s, ours.wall_time_s):>10s}"
    )

    # Aggregate throughput
    print(
        f"  {'Agg tok/s:':24s} {base.agg_throughput_tps:>13.1f}  {ours.agg_throughput_tps:>13.1f}  "
        f"{_speedup_str(ours.agg_throughput_tps, base.agg_throughput_tps):>10s}"
    )

    # Avg TTFT
    print(
        f"  {'Avg TTFT:':24s} {base.avg_ttft_ms:>11.0f}ms {ours.avg_ttft_ms:>11.0f}ms "
        f"{_speedup_str(base.avg_ttft_ms, ours.avg_ttft_ms):>10s}"
    )

    # Total tokens
    print(
        f"  {'Total tokens:':24s} {base.total_tokens:>14d} {ours.total_tokens:>14d} {'':>10s}"
    )

    # Errors
    if base.num_errors > 0 or ours.num_errors > 0:
        print(
            f"  {'Errors:':24s} {base.num_errors:>14d} {ours.num_errors:>14d} {'':>10s}"
        )

    if extra_rows:
        print(f"  {'-' * w}")
        for label, base_val, ours_val, speedup in extra_rows:
            print(f"  {label:24s} {base_val:>14s} {ours_val:>14s} {speedup:>10s}")

    print(f"  {'=' * w}")


# ---------------------------------------------------------------------------
# Scenario 1: Concurrent Request Throughput
# ---------------------------------------------------------------------------


def scenario_concurrent_throughput(
    model: Any,
    tokenizer: Any,
    model_path: str,
    repeat: int,
    warmup: int,
) -> None:
    print("\n" + "#" * 70)
    print("  Scenario 1: Concurrent Request Throughput")
    print("#" * 70)

    prompt_text = "Explain the concept of gradient descent in machine learning."
    prompt_tokens = tokenizer.encode(prompt_text)
    max_tokens = 32

    concurrency_levels = [1, 2, 4, 8]

    for N in concurrency_levels:
        prompts = [list(prompt_tokens) for _ in range(N)]
        tag = f"N={N}, max_tokens={max_tokens}"

        # --- Warmup ---
        print(f"\n  [{tag}] Warmup ({warmup} rounds) ...")
        sched = _make_scheduler(model, tokenizer, model_path, max_batch_size=max(N, 4))
        try:
            for _ in range(warmup):
                baseline_generate_one(model, tokenizer, prompt_tokens, max_tokens)
                ours_generate_one_streaming(sched, prompt_tokens, max_tokens)
        except Exception:
            pass

        # --- Measured runs ---
        base_agg = ScenarioResult(label="baseline")
        ours_agg = ScenarioResult(label="ours")

        for run_i in range(repeat):
            print(f"  [{tag}] Run {run_i + 1}/{repeat} ...")

            # Baseline: sequential
            b = baseline_sequential(model, tokenizer, prompts, max_tokens)

            # Ours: concurrent
            o = ours_concurrent(sched, prompts, max_tokens)

            # Aggregate across repeats (use best wall time)
            if base_agg.wall_time_s == 0 or b.wall_time_s < base_agg.wall_time_s:
                base_agg = b
            if ours_agg.wall_time_s == 0 or o.wall_time_s < ours_agg.wall_time_s:
                ours_agg = o

        sched.stop()
        print_scenario(f"Concurrent Throughput (N={N}, {max_tokens} max_tokens)", base_agg, ours_agg)


# ---------------------------------------------------------------------------
# Scenario 2: Multi-Agent Prefix Sharing
# ---------------------------------------------------------------------------


def scenario_prefix_sharing(
    model: Any,
    tokenizer: Any,
    model_path: str,
    repeat: int,
    warmup: int,
    num_agents: int = 4,
) -> None:
    print("\n" + "#" * 70)
    print("  Scenario 2: Multi-Agent Prefix Sharing")
    print("#" * 70)

    max_tokens = 32

    # Build prompts: shared system prompt + different agent messages
    chat_prompts: list[list[int]] = []
    for i in range(num_agents):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": AGENT_MESSAGES[i % len(AGENT_MESSAGES)]},
        ]
        tokens = build_chat_prompt(tokenizer, SYSTEM_PROMPT, messages)
        chat_prompts.append(tokens)

    system_tokens = tokenizer.encode(SYSTEM_PROMPT)
    print(f"  Shared system prompt: ~{len(system_tokens)} tokens")
    print(f"  Number of agents: {num_agents}")
    print(f"  Max tokens per response: {max_tokens}")
    print()
    print("  Strategy: sequential requests to demonstrate prefix cache warming.")
    print("  Baseline: each request recomputes full prefill from scratch.")
    print("  Ours: first request populates cache; subsequent requests reuse it.")

    # --- Warmup (warm up the model, not the cache) ---
    print(f"\n  Warmup ({warmup} rounds) ...")
    for _ in range(warmup):
        baseline_generate_one(model, tokenizer, chat_prompts[0], max_tokens)

    # --- Measured runs: pick best ---
    best_base: ScenarioResult | None = None
    best_ours: ScenarioResult | None = None

    for run_i in range(repeat):
        print(f"  Run {run_i + 1}/{repeat} ...")

        # BASELINE: sequential processing, each request starts with fresh cache
        base = ScenarioResult(label="baseline")
        t0 = time.perf_counter()
        for i, tp in enumerate(chat_prompts):
            r = baseline_generate_one(model, tokenizer, tp, max_tokens)
            base.per_request.append(r)
        base.wall_time_s = time.perf_counter() - t0

        # OURS: sequential requests through same scheduler -- prefix cache warms up
        # Fresh scheduler so first request is truly cold
        sched_fresh = _make_scheduler(model, tokenizer, model_path, max_batch_size=4)
        ours = ScenarioResult(label="ours")
        sleep_overhead = 0.0
        t0 = time.perf_counter()
        for i, tp in enumerate(chat_prompts):
            r = ours_generate_one_streaming(sched_fresh, tp, max_tokens)
            ours.per_request.append(r)
            # Brief pause to allow block decomposition after each request
            t_sleep_start = time.perf_counter()
            time.sleep(0.15)
            sleep_overhead += time.perf_counter() - t_sleep_start
        ours.wall_time_s = time.perf_counter() - t0 - sleep_overhead
        sched_fresh.stop()

        if best_base is None or base.wall_time_s < best_base.wall_time_s:
            best_base = base
        if best_ours is None or ours.wall_time_s < best_ours.wall_time_s:
            best_ours = ours

    # Per-request TTFT breakdown
    extra_rows: list[tuple[str, str, str, str]] = []
    if best_base and best_ours:
        for i in range(len(chat_prompts)):
            if i < len(best_base.per_request) and i < len(best_ours.per_request):
                br = best_base.per_request[i]
                or_ = best_ours.per_request[i]
                label = f"Agent {i + 1} TTFT:" if i > 0 else "Agent 1 (cold) TTFT:"
                extra_rows.append((
                    label,
                    f"{br.ttft_ms:.0f}ms",
                    f"{or_.ttft_ms:.0f}ms",
                    _speedup_str(br.ttft_ms, or_.ttft_ms),
                ))
        # Averages for 2nd+ agents
        if len(best_base.per_request) > 1 and len(best_ours.per_request) > 1:
            rest_base = statistics.mean([r.ttft_ms for r in best_base.per_request[1:] if r.ttft_ms > 0])
            rest_ours = statistics.mean([r.ttft_ms for r in best_ours.per_request[1:] if r.ttft_ms > 0])
            extra_rows.append((
                "2nd+ avg TTFT:",
                f"{rest_base:.0f}ms",
                f"{rest_ours:.0f}ms",
                _speedup_str(rest_base, rest_ours),
            ))

    print_scenario(
        f"Scenario 2a: Prefix Sharing, Sequential ({num_agents} agents)",
        best_base or ScenarioResult(),
        best_ours or ScenarioResult(),
        extra_rows=extra_rows,
    )

    # --- 2b: Concurrent prefix sharing ---
    # This is the real multi-agent scenario: all agents send requests at once.
    # Baseline processes them sequentially; ours batches them.
    print(f"\n  Now running concurrent variant (all {num_agents} agents at once) ...")

    best_base_conc: ScenarioResult | None = None
    best_ours_conc: ScenarioResult | None = None

    for run_i in range(repeat):
        print(f"  Concurrent run {run_i + 1}/{repeat} ...")

        # Baseline: still sequential (it cannot batch)
        base_c = baseline_sequential(model, tokenizer, chat_prompts, max_tokens)

        # Ours: all requests fired concurrently
        sched_conc = _make_scheduler(model, tokenizer, model_path, max_batch_size=max(num_agents, 4))
        ours_c = ours_concurrent(sched_conc, chat_prompts, max_tokens)
        sched_conc.stop()

        if best_base_conc is None or base_c.wall_time_s < best_base_conc.wall_time_s:
            best_base_conc = base_c
        if best_ours_conc is None or ours_c.wall_time_s < best_ours_conc.wall_time_s:
            best_ours_conc = ours_c

    print_scenario(
        f"Scenario 2b: Prefix Sharing, Concurrent ({num_agents} agents at once)",
        best_base_conc or ScenarioResult(),
        best_ours_conc or ScenarioResult(),
    )


# ---------------------------------------------------------------------------
# Scenario 3: Repeated Context (Chat History Reuse)
# ---------------------------------------------------------------------------


def scenario_chat_history(
    model: Any,
    tokenizer: Any,
    model_path: str,
    repeat: int,
    warmup: int,
) -> None:
    print("\n" + "#" * 70)
    print("  Scenario 3: Repeated Context (Chat History Reuse)")
    print("#" * 70)

    max_tokens = 32

    # Build multi-turn prompts: each turn includes all previous turns
    turn_prompts: list[list[int]] = []
    messages_so_far: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for i, user_msg in enumerate(CHAT_TURNS):
        messages_so_far.append({"role": "user", "content": user_msg})
        tokens = build_chat_prompt(tokenizer, SYSTEM_PROMPT, list(messages_so_far))
        turn_prompts.append(tokens)
        # Add a fake assistant response for the next turn's context
        if i < len(CHAT_TURNS) - 1:
            messages_so_far.append({
                "role": "assistant",
                "content": f"Here is my response to turn {i + 1}. "
                           "I provide a thorough explanation covering the key concepts."
            })

    print(f"  Number of turns: {len(turn_prompts)}")
    for i, tp in enumerate(turn_prompts):
        print(f"    Turn {i + 1}: {len(tp)} prompt tokens")

    # --- Warmup ---
    print(f"\n  Warmup ({warmup} rounds) ...")
    for _ in range(warmup):
        baseline_generate_one(model, tokenizer, turn_prompts[0], max_tokens)

    # --- Measured runs ---
    best_base: ScenarioResult | None = None
    best_ours: ScenarioResult | None = None

    for run_i in range(repeat):
        print(f"  Run {run_i + 1}/{repeat} ...")

        # BASELINE: each turn is independent, full prefill every time
        base = ScenarioResult(label="baseline")
        t0 = time.perf_counter()
        for tp in turn_prompts:
            r = baseline_generate_one(model, tokenizer, tp, max_tokens)
            base.per_request.append(r)
        base.wall_time_s = time.perf_counter() - t0

        # OURS: sequential turns through same scheduler (prefix cache grows)
        # Fresh scheduler so turn 1 is cold, subsequent turns benefit from cache
        sched_chat = _make_scheduler(model, tokenizer, model_path, max_batch_size=4)
        ours = ScenarioResult(label="ours")
        sleep_overhead = 0.0
        t0 = time.perf_counter()
        for tp in turn_prompts:
            r = ours_generate_one_streaming(sched_chat, tp, max_tokens)
            ours.per_request.append(r)
            t_sleep_start = time.perf_counter()
            time.sleep(0.15)  # Brief pause for cache decomposition
            sleep_overhead += time.perf_counter() - t_sleep_start
        ours.wall_time_s = time.perf_counter() - t0 - sleep_overhead
        sched_chat.stop()

        if best_base is None or base.wall_time_s < best_base.wall_time_s:
            best_base = base
        if best_ours is None or ours.wall_time_s < best_ours.wall_time_s:
            best_ours = ours

    # Per-turn TTFT breakdown
    extra_rows: list[tuple[str, str, str, str]] = []
    if best_base and best_ours:
        for i in range(len(turn_prompts)):
            if i < len(best_base.per_request) and i < len(best_ours.per_request):
                br = best_base.per_request[i]
                or_ = best_ours.per_request[i]
                extra_rows.append((
                    f"Turn {i + 1} TTFT:",
                    f"{br.ttft_ms:.0f}ms",
                    f"{or_.ttft_ms:.0f}ms",
                    _speedup_str(br.ttft_ms, or_.ttft_ms),
                ))

    print_scenario(
        f"Chat History Reuse ({len(turn_prompts)} turns)",
        best_base or ScenarioResult(),
        best_ours or ScenarioResult(),
        extra_rows=extra_rows,
    )


# ---------------------------------------------------------------------------
# Scenario 4: Single Request Latency (Sanity Check)
# ---------------------------------------------------------------------------


def scenario_single_latency(
    model: Any,
    tokenizer: Any,
    model_path: str,
    repeat: int,
    warmup: int,
) -> None:
    print("\n" + "#" * 70)
    print("  Scenario 4: Single Request Latency (Sanity Check)")
    print("#" * 70)

    prompt_text = "What is 2+2? Answer in one word."
    prompt_tokens = tokenizer.encode(prompt_text)
    max_tokens = 16

    print(f"  Prompt: {len(prompt_tokens)} tokens")
    print(f"  Max tokens: {max_tokens}")

    sched = _make_scheduler(model, tokenizer, model_path, max_batch_size=4)

    # --- Warmup ---
    print(f"\n  Warmup ({warmup} rounds) ...")
    for _ in range(warmup):
        baseline_generate_one(model, tokenizer, prompt_tokens, max_tokens)
        ours_generate_one_streaming(sched, prompt_tokens, max_tokens)

    # --- Measured runs ---
    base_results: list[RequestResult] = []
    ours_results: list[RequestResult] = []

    for run_i in range(repeat):
        print(f"  Run {run_i + 1}/{repeat} ...")
        br = baseline_generate_one(model, tokenizer, prompt_tokens, max_tokens)
        base_results.append(br)
        or_ = ours_generate_one_streaming(sched, prompt_tokens, max_tokens)
        ours_results.append(or_)

    sched.stop()

    # Build scenario results from best run
    best_base_idx = min(range(len(base_results)), key=lambda i: base_results[i].total_time_s)
    best_ours_idx = min(range(len(ours_results)), key=lambda i: ours_results[i].total_time_s)

    base = ScenarioResult(label="baseline", wall_time_s=base_results[best_base_idx].total_time_s)
    base.per_request = [base_results[best_base_idx]]
    ours = ScenarioResult(label="ours", wall_time_s=ours_results[best_ours_idx].total_time_s)
    ours.per_request = [ours_results[best_ours_idx]]

    # Extra: show all runs
    extra_rows: list[tuple[str, str, str, str]] = []
    for i in range(len(base_results)):
        extra_rows.append((
            f"Run {i + 1} total:",
            f"{base_results[i].total_time_s:.3f}s",
            f"{ours_results[i].total_time_s:.3f}s",
            _speedup_str(base_results[i].total_time_s, ours_results[i].total_time_s),
        ))

    # Averages
    if len(base_results) > 1:
        avg_base = statistics.mean(r.total_time_s for r in base_results)
        avg_ours = statistics.mean(r.total_time_s for r in ours_results)
        extra_rows.append((
            "Average total:",
            f"{avg_base:.3f}s",
            f"{avg_ours:.3f}s",
            _speedup_str(avg_base, avg_ours),
        ))

    print_scenario(
        f"Single Request Latency ({max_tokens} max_tokens)",
        base, ours,
        extra_rows=extra_rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B benchmark: mlx_lm_server vs baseline mlx_lm.server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Model name or local path",
    )
    parser.add_argument(
        "--scenario",
        choices=["concurrent", "prefix", "chat", "single", "all"],
        default="all",
        help="Which scenario to run",
    )
    parser.add_argument(
        "--repeat", type=int, default=2,
        help="Number of measured runs per scenario (best is reported)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup rounds before measuring",
    )
    parser.add_argument(
        "--num-agents", type=int, default=4,
        help="Number of agents for prefix sharing scenario",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  mlx-lm-server vs baseline: Comparative Benchmark")
    resolved_model = args.model
    if Path(LOCAL_MODEL_PATH).is_dir() and args.model == DEFAULT_MODEL:
        resolved_model = LOCAL_MODEL_PATH
    print(f"  Model:   {resolved_model}")
    print(f"  Repeat:  {args.repeat}")
    print(f"  Warmup:  {args.warmup}")
    print("=" * 70)

    model, tokenizer = load_model(args.model)

    t0 = time.time()

    scenarios = (
        [args.scenario]
        if args.scenario != "all"
        else ["single", "concurrent", "prefix", "chat"]
    )

    for scenario in scenarios:
        if scenario == "concurrent":
            scenario_concurrent_throughput(model, tokenizer, args.model, args.repeat, args.warmup)
        elif scenario == "prefix":
            scenario_prefix_sharing(model, tokenizer, args.model, args.repeat, args.warmup, args.num_agents)
        elif scenario == "chat":
            scenario_chat_history(model, tokenizer, args.model, args.repeat, args.warmup)
        elif scenario == "single":
            scenario_single_latency(model, tokenizer, args.model, args.repeat, args.warmup)

    elapsed = time.time() - t0
    print(f"\n  Total benchmark time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
