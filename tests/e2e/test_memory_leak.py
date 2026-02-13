#!/usr/bin/env python3
"""E2E test: server lifecycle harness with Metal memory leak detection.

Manages the full server lifecycle on a 2-node JACCL TP setup:
  1. Record baseline wired memory on both nodes
  2. Launch server (loads ~310GB per node)
  3. Verify server is functional (smoke test)
  4. Run external test suites (optional, via --run-tests)
  5. Gracefully shut down server
  6. Verify wired memory returns to baseline

Can also be used as a pure lifecycle harness (--no-memory-check) to
wrap other test suites with automatic server start/stop.

Usage:
    # Standalone memory leak test
    .venv/bin/python tests/e2e/test_memory_leak.py

    # Wrap API tests in server lifecycle + memory check
    .venv/bin/python tests/e2e/test_memory_leak.py \\
        --run-tests "tests/e2e/test_api_comprehensive.py"

    # Multiple test suites
    .venv/bin/python tests/e2e/test_memory_leak.py \\
        --run-tests "tests/e2e/test_api_comprehensive.py" \\
                    "tests/e2e/benchmark/run_all.sh"

    # Pure lifecycle harness (no memory checks)
    .venv/bin/python tests/e2e/test_memory_leak.py \\
        --run-tests "tests/e2e/test_api_comprehensive.py" \\
        --no-memory-check

    # Against already-running server
    .venv/bin/python tests/e2e/test_memory_leak.py --skip-launch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_SCRIPT = PROJECT_ROOT / "tests" / "e2e" / "benchmark" / "configs" / "launch_baseline.sh"
STOP_SCRIPT = PROJECT_ROOT / "tests" / "e2e" / "benchmark" / "configs" / "stop_server.sh"

PAGE_SIZE_BYTES = 16384  # Apple Silicon page size (16 KB)

DEFAULT_SERVER_URL = "http://localhost:8080"
DEFAULT_REMOTE_HOST = "hwStudio2.local"
DEFAULT_LEAK_THRESHOLD_GB = 4.0
MIN_LOADED_INCREASE_GB = 50.0  # Kimi K2.5 TP-2 should add at least 50GB per node


# ---------------------------------------------------------------------------
# Result tracking (matches test_api_comprehensive.py)
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    detail: str = ""
    error: str = ""


@dataclass
class TestSuite:
    name: str = "Test Suite"
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult) -> None:
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name} ({result.duration_s:.1f}s)")
        if result.detail:
            for line in result.detail.split("\n"):
                print(f"         {line}")
        if result.error:
            print(f"         ERROR: {result.error}")

    def summary(self) -> bool:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print(f"\n{'=' * 60}")
        print(f"  {self.name}: {passed}/{total} passed")
        print(f"{'=' * 60}")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
        if failed > 0:
            print(f"\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error or r.detail}")
        print()
        return failed == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wired_pages(host: str | None = None) -> int:
    """Get wired memory page count from vm_stat. host=None means local."""
    if host:
        cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
               host, "vm_stat"]
    else:
        cmd = ["vm_stat"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(
            f"vm_stat failed on {host or 'localhost'}: {result.stderr.strip()}"
        )
    for line in result.stdout.splitlines():
        if "Pages wired" in line:
            # Format: "Pages wired down:                 12345."
            return int(line.split(":")[1].strip().rstrip("."))
    raise RuntimeError(f"Could not parse wired pages from vm_stat on {host or 'localhost'}")


def pages_to_gb(pages: int) -> float:
    """Convert vm_stat page count to gigabytes."""
    return (pages * PAGE_SIZE_BYTES) / (1024 ** 3)


def wait_for_health(url: str, timeout: int = 600, interval: int = 5) -> bool:
    """Poll health endpoint until ready or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def check_port_free(port: int = 8080) -> bool:
    """Check if port is free (no server already running)."""
    result = subprocess.run(["lsof", "-ti", f":{port}"],
                            capture_output=True, text=True)
    return result.stdout.strip() == ""


def timestamp() -> str:
    """Return current time as HH:MM:SS string."""
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Test: Baseline memory
# ---------------------------------------------------------------------------

def record_baseline_memory(
    nodes: dict[str, str | None],
) -> dict[str, int]:
    """Record wired memory on all nodes. Returns {node_name: pages}."""
    baseline: dict[str, int] = {}
    for name, host in nodes.items():
        pages = get_wired_pages(host)
        baseline[name] = pages
        print(f"  Baseline {name}: {pages_to_gb(pages):.2f} GB wired")
    return baseline


# ---------------------------------------------------------------------------
# Test: Server launch
# ---------------------------------------------------------------------------

def test_server_launch(server_url: str) -> TestResult:
    """Launch server via launch_baseline.sh and verify health."""
    t0 = time.monotonic()
    print(f"  [{timestamp()}] Running {LAUNCH_SCRIPT.name} ...")
    try:
        proc = subprocess.run(
            ["bash", str(LAUNCH_SCRIPT)],
            capture_output=True, text=True, timeout=660,
        )
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-500:] if proc.stderr else "(no stderr)"
            return TestResult(
                name="server_launch",
                passed=False,
                duration_s=elapsed,
                error=f"exit code {proc.returncode}\n{stderr_tail}",
            )
        return TestResult(
            name="server_launch",
            passed=True,
            duration_s=elapsed,
            detail=f"Server healthy after {elapsed:.0f}s",
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            name="server_launch",
            passed=False,
            duration_s=660,
            error="launch_baseline.sh timed out after 660s",
        )


# ---------------------------------------------------------------------------
# Test: Loaded memory
# ---------------------------------------------------------------------------

def test_loaded_memory(
    nodes: dict[str, str | None],
    baseline: dict[str, int],
) -> TestResult:
    """Verify wired memory increased significantly on both nodes after model load."""
    t0 = time.monotonic()
    details: list[str] = []
    all_ok = True

    for name, host in nodes.items():
        pages = get_wired_pages(host)
        increase_gb = pages_to_gb(pages - baseline[name])
        details.append(
            f"{name}: {pages_to_gb(pages):.1f} GB wired (+{increase_gb:.1f} GB from baseline)"
        )
        if increase_gb < MIN_LOADED_INCREASE_GB:
            all_ok = False

    elapsed = time.monotonic() - t0
    return TestResult(
        name="loaded_memory",
        passed=all_ok,
        duration_s=elapsed,
        detail="\n".join(details),
        error="" if all_ok else f"Expected >{MIN_LOADED_INCREASE_GB:.0f} GB increase per node",
    )


# ---------------------------------------------------------------------------
# Test: Smoke request
# ---------------------------------------------------------------------------

def test_smoke_request(server_url: str) -> TestResult:
    """Make one simple non-streaming chat completion request."""
    t0 = time.monotonic()
    try:
        # Fetch model name
        model_resp = httpx.get(f"{server_url}/v1/models", timeout=10)
        model_resp.raise_for_status()
        model_name = model_resp.json()["data"][0]["id"]

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
        }
        resp = httpx.post(
            f"{server_url}/v1/chat/completions", json=payload, timeout=120,
        )
        elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            return TestResult(
                name="smoke_request",
                passed=False,
                duration_s=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        content = resp.json()["choices"][0]["message"]["content"]
        return TestResult(
            name="smoke_request",
            passed=True,
            duration_s=elapsed,
            detail=f"Model: {model_name}\nResponse: {content[:80]}",
        )
    except Exception as e:
        return TestResult(
            name="smoke_request",
            passed=False,
            duration_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Test: Shutdown
# ---------------------------------------------------------------------------

def test_shutdown() -> TestResult:
    """Run stop_server.sh and verify it completes."""
    t0 = time.monotonic()
    print(f"  [{timestamp()}] Running {STOP_SCRIPT.name} ...")
    try:
        proc = subprocess.run(
            ["bash", str(STOP_SCRIPT)],
            capture_output=True, text=True, timeout=120,
        )
        elapsed = time.monotonic() - t0
        stdout_tail = proc.stdout.strip()[-300:] if proc.stdout else ""
        stderr_tail = proc.stderr.strip()[-300:] if proc.stderr else ""

        return TestResult(
            name="shutdown",
            passed=proc.returncode == 0,
            duration_s=elapsed,
            detail=stdout_tail,
            error=stderr_tail if proc.returncode != 0 else "",
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            name="shutdown",
            passed=False,
            duration_s=120,
            error="stop_server.sh timed out after 120s",
        )


# ---------------------------------------------------------------------------
# Test: Memory released (THE KEY TEST)
# ---------------------------------------------------------------------------

def test_memory_released(
    nodes: dict[str, str | None],
    baseline: dict[str, int],
    threshold_gb: float,
) -> TestResult:
    """Verify wired memory on both nodes is within threshold of baseline."""
    t0 = time.monotonic()
    details: list[str] = []
    all_clean = True

    for name, host in nodes.items():
        try:
            current = get_wired_pages(host)
            leaked_gb = pages_to_gb(current - baseline[name])
            status = "ok" if leaked_gb <= threshold_gb else "LEAKED"
            details.append(
                f"{name}: {pages_to_gb(baseline[name]):.2f} GB -> "
                f"{pages_to_gb(current):.2f} GB "
                f"(delta {leaked_gb:+.2f} GB) [{status}]"
            )
            if leaked_gb > threshold_gb:
                all_clean = False
        except Exception as e:
            details.append(f"{name}: ERROR -- {e}")
            all_clean = False

    elapsed = time.monotonic() - t0
    return TestResult(
        name="memory_released",
        passed=all_clean,
        duration_s=elapsed,
        detail="\n".join(details),
        error="" if all_clean else f"Wired memory leak > {threshold_gb:.1f} GB detected",
    )


# ---------------------------------------------------------------------------
# Test: No residual processes
# ---------------------------------------------------------------------------

def test_no_residual_processes(remote_host: str) -> TestResult:
    """Verify no mlx_lm_server processes remain on either node."""
    t0 = time.monotonic()
    details: list[str] = []
    all_clean = True

    # Local: check port 8080
    local_pids = subprocess.run(
        ["lsof", "-ti", ":8080"], capture_output=True, text=True,
    ).stdout.strip()
    if local_pids:
        details.append(f"local port 8080 PIDs: {local_pids}")
        all_clean = False
    else:
        details.append("local port 8080: clean")

    # Remote: check for mlx_lm_server processes
    try:
        remote_result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             remote_host, "pgrep -f mlx_lm_server"],
            capture_output=True, text=True, timeout=10,
        )
        remote_pids = remote_result.stdout.strip()
        if remote_pids:
            details.append(f"{remote_host} PIDs: {remote_pids}")
            all_clean = False
        else:
            details.append(f"{remote_host}: clean")
    except Exception as e:
        details.append(f"{remote_host}: ERROR -- {e}")
        all_clean = False

    elapsed = time.monotonic() - t0
    return TestResult(
        name="no_residual_processes",
        passed=all_clean,
        duration_s=elapsed,
        detail="\n".join(details),
    )


# ---------------------------------------------------------------------------
# External test runner
# ---------------------------------------------------------------------------

def run_external_tests(commands: list[str], server_url: str) -> list[TestResult]:
    """Run external test scripts while server is up."""
    results = []
    for cmd in commands:
        name = Path(cmd.split()[0]).stem  # e.g. "test_api_comprehensive"
        t0 = time.monotonic()
        print(f"  [{timestamp()}] Running: {cmd}")
        try:
            # Determine how to run: .sh via bash, .py via .venv/bin/python
            if cmd.strip().endswith(".sh"):
                full_cmd = ["bash", cmd]
            elif cmd.strip().endswith(".py"):
                python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
                full_cmd = [python] + cmd.split()
            else:
                full_cmd = ["bash", "-c", cmd]

            # Pass server URL as env var for tests that need it
            env = {**os.environ, "SERVER_URL": server_url}
            proc = subprocess.run(
                full_cmd, capture_output=True, text=True,
                timeout=1800, env=env,  # 30 min timeout per test
                cwd=str(PROJECT_ROOT),
            )
            elapsed = time.monotonic() - t0
            stdout_tail = proc.stdout.strip()[-500:] if proc.stdout else ""
            stderr_tail = proc.stderr.strip()[-300:] if proc.stderr else ""

            results.append(TestResult(
                name=f"external:{name}",
                passed=proc.returncode == 0,
                duration_s=elapsed,
                detail=stdout_tail,
                error=stderr_tail if proc.returncode != 0 else "",
            ))
        except subprocess.TimeoutExpired:
            results.append(TestResult(
                name=f"external:{name}",
                passed=False,
                duration_s=1800,
                error=f"Timed out after 1800s: {cmd}",
            ))
        except Exception as e:
            results.append(TestResult(
                name=f"external:{name}",
                passed=False,
                duration_s=time.monotonic() - t0,
                error=f"{type(e).__name__}: {e}",
            ))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E memory leak test for 2-node JACCL TP setup",
    )
    parser.add_argument(
        "--server-url", default=DEFAULT_SERVER_URL,
        help=f"Server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--remote-host", default=DEFAULT_REMOTE_HOST,
        help=f"Remote node hostname (default: {DEFAULT_REMOTE_HOST})",
    )
    parser.add_argument(
        "--leak-threshold-gb", type=float, default=DEFAULT_LEAK_THRESHOLD_GB,
        help=f"Max acceptable wired memory increase after shutdown in GB "
             f"(default: {DEFAULT_LEAK_THRESHOLD_GB})",
    )
    parser.add_argument(
        "--skip-launch", action="store_true",
        help="Skip server launch/shutdown (assume server is already running). "
             "Only runs smoke test and current memory snapshot.",
    )
    parser.add_argument(
        "--run-tests", nargs="*", default=None, metavar="CMD",
        help="Test scripts to run while server is up. "
             "Each CMD is executed via 'bash -c CMD' (for .sh) or "
             "'.venv/bin/python CMD' (for .py). Runs after smoke test, before shutdown.",
    )
    parser.add_argument(
        "--no-memory-check", action="store_true",
        help="Skip memory baseline/release checks (use as pure lifecycle harness).",
    )
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")
    remote_host: str = args.remote_host
    threshold_gb: float = args.leak_threshold_gb
    skip_launch: bool = args.skip_launch
    no_memory_check: bool = args.no_memory_check
    run_tests: list[str] | None = args.run_tests

    nodes: dict[str, str | None] = {"local": None, remote_host: remote_host}
    suite = TestSuite(name="Memory Leak Test (2-Node JACCL TP)")

    # Dynamic step counter
    step = 0

    def next_step(label: str) -> None:
        nonlocal step
        step += 1
        print(f"[Step {step}] {label}")

    # ── Banner ──
    print(f"\n{'=' * 60}")
    print(f"  Memory Leak Test")
    print(f"  Server:    {server_url}")
    print(f"  Remote:    {remote_host}")
    if not no_memory_check:
        print(f"  Threshold: {threshold_gb:.1f} GB")
    if run_tests:
        print(f"  External:  {len(run_tests)} test command(s)")
    if no_memory_check:
        print(f"  Mode:      lifecycle harness (no memory checks)")
    print(f"  Time:      {timestamp()}")
    print(f"{'=' * 60}\n")

    # ── Pre-flight checks ──
    if not no_memory_check:
        print("[Pre-flight] Checking remote node reachability ...")
        try:
            get_wired_pages(remote_host)
            print(f"  {remote_host}: reachable")
        except Exception as e:
            print(f"ABORT: Cannot reach {remote_host}: {e}")
            sys.exit(2)

    if not skip_launch and not check_port_free():
        print("ABORT: Port 8080 already in use. "
              "Stop existing server first or use --skip-launch.")
        sys.exit(2)
    print()

    # ── Baseline memory ──
    baseline: dict[str, int] = {}
    if not no_memory_check:
        next_step("Recording baseline wired memory ...")
        baseline = record_baseline_memory(nodes)
        print()

    # ── Launch server ──
    if not skip_launch:
        next_step("Launching server ...")
        result = test_server_launch(server_url)
        suite.add(result)
        print()
        if not result.passed:
            # Cannot continue without a healthy server
            suite.summary()
            sys.exit(1)
    else:
        next_step("Skipping launch (--skip-launch) ...")
        if not wait_for_health(server_url, timeout=10):
            print("ABORT: --skip-launch specified but server is not healthy")
            sys.exit(2)
        suite.add(TestResult(
            name="server_already_running",
            passed=True,
            duration_s=0.0,
            detail="--skip-launch: server is healthy",
        ))
        print()

    # ── Loaded memory ──
    if not no_memory_check:
        next_step("Checking loaded memory ...")
        suite.add(test_loaded_memory(nodes, baseline))
        print()

    # ── Smoke test ──
    next_step("Smoke test (one chat completion) ...")
    suite.add(test_smoke_request(server_url))
    print()

    # ── External tests ──
    if run_tests:
        next_step(f"Running {len(run_tests)} external test(s) ...")
        ext_results = run_external_tests(run_tests, server_url)
        for r in ext_results:
            suite.add(r)
        print()

    if skip_launch:
        # With --skip-launch we do not shut down or check memory release
        print("  Skipping shutdown / memory checks (--skip-launch)")
        print()
        all_passed = suite.summary()
        sys.exit(0 if all_passed else 1)

    # ── Shutdown ──
    next_step("Shutting down server ...")
    result = test_shutdown()
    suite.add(result)
    print()

    # ── Memory released -- THE KEY TEST ──
    if not no_memory_check:
        next_step("Waiting 5s for memory settle ...")
        time.sleep(5)

        print(f"[Step {step}] Checking memory release ...")
        suite.add(test_memory_released(nodes, baseline, threshold_gb))
        print()

    # ── No residual processes ──
    next_step("Checking for residual processes ...")
    suite.add(test_no_residual_processes(remote_host))
    print()

    # ── Summary ──
    all_passed = suite.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
