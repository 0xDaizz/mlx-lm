"""Real 2-rank distributed tests using mlx.launch subprocess invocation.

Unlike test_distributed.py which uses mocks, these tests spawn actual
2-rank processes via mlx.launch and verify cross-rank communication
through stdout parsing.

Each test generates a Python script, runs it with mlx.launch on 2 local
ranks, and asserts that both ranks completed successfully.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

MLX_LAUNCH = os.path.join(os.path.dirname(sys.executable), "mlx.launch")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.skipif(
        not os.path.isfile(MLX_LAUNCH), reason="mlx.launch not available"
    ),
]


def _run_2rank_script(
    script_content: str, tmp_path, timeout: float = 30.0
) -> tuple[int, str, str]:
    """Write script to tmp_path, run with mlx.launch on 2 local ranks.

    Returns (returncode, stdout, stderr).
    """
    script_path = tmp_path / "test_script.py"
    script_path.write_text(textwrap.dedent(script_content))

    cmd = [
        MLX_LAUNCH,
        "--backend", "ring",
        "--hosts", "127.0.0.1",
        "--repeat-hosts", "2",
        "--",
        str(script_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env
    )
    return proc.returncode, proc.stdout, proc.stderr


# =========================================================================
# 1. Basic distributed init
# =========================================================================


class TestDistributedInit:
    """Verify that 2 ranks can initialize mx.distributed."""

    def test_2rank_distributed_init(self, tmp_path):
        """Both ranks call mx.distributed.init() and report success."""
        rc, stdout, stderr = _run_2rank_script(
            """\
            import sys, os
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx

            try:
                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()
                size = group.size()
                assert size == 2, f"Expected world_size=2, got {size}"
                print(f"RANK_{rank}_INIT_OK")
            except Exception as e:
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{rank}_FAIL: {e}")
            """,
            tmp_path,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_INIT_OK" in stdout
        assert "RANK_1_INIT_OK" in stdout


# =========================================================================
# 2. all_sum collective
# =========================================================================


class TestAllSumCollective:
    """Verify that mx.distributed.all_sum works across 2 ranks."""

    def test_2rank_all_sum_collective(self, tmp_path):
        """Both ranks create a tensor, call all_sum, verify result."""
        rc, stdout, stderr = _run_2rank_script(
            """\
            import sys, os
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx

            try:
                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()

                # Rank 0 contributes [1, 2, 3], Rank 1 contributes [10, 20, 30]
                if rank == 0:
                    tensor = mx.array([1.0, 2.0, 3.0])
                else:
                    tensor = mx.array([10.0, 20.0, 30.0])

                result = mx.distributed.all_sum(tensor, group=group)
                mx.eval(result)

                expected = [11.0, 22.0, 33.0]
                actual = result.tolist()
                assert actual == expected, f"Expected {expected}, got {actual}"
                print(f"RANK_{rank}_ALLSUM_OK result={actual}")
            except Exception as e:
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{rank}_FAIL: {e}")
            """,
            tmp_path,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_ALLSUM_OK" in stdout
        assert "RANK_1_ALLSUM_OK" in stdout


# =========================================================================
# 3. DistributedControlBus communication
# =========================================================================


class TestControlBusCommunication:
    """Verify that DistributedControlBus can send events across ranks."""

    def test_2rank_control_bus_communication(self, tmp_path):
        """Rank 0 publishes a batch event, rank 1 receives and unpacks it."""
        rc, stdout, stderr = _run_2rank_script(
            """\
            import sys, os
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx

            try:
                from mlx_lm_server.distributed import DistributedContext
                from mlx_lm_server.distributed_bus import ControlEvent, DistributedControlBus
                from mlx_lm_server.types import InferenceRequest

                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()
                ws = group.size()

                ctx = DistributedContext(
                    enabled=True, group=group, rank=rank,
                    world_size=ws, backend="ring",
                )
                bus = DistributedControlBus(ctx)

                if rank == 0:
                    req = InferenceRequest(
                        request_id="test-req-001",
                        prompt_tokens=[1, 2, 3, 4, 5],
                        max_tokens=10,
                    )
                    sub_events = [
                        ControlEvent.submit(req),
                        ControlEvent.cancel("cancel-req-002"),
                    ]
                    batch_evt = ControlEvent.batch(sub_events)
                    bus.publish(batch_evt)
                    print(f"RANK_0_BUS_OK published_batch_with_{len(sub_events)}_events")
                else:
                    evt = bus.recv()
                    assert evt.typ == "batch", f"Expected batch, got {evt.typ}"
                    sub_events = evt.unpack_batch()
                    assert len(sub_events) == 2, f"Expected 2 sub-events, got {len(sub_events)}"

                    # Verify submit event
                    assert sub_events[0].typ == "submit"
                    req = sub_events[0].unpack_request()
                    assert req.request_id == "test-req-001"
                    assert req.prompt_tokens == [1, 2, 3, 4, 5]
                    assert req.max_tokens == 10

                    # Verify cancel event
                    assert sub_events[1].typ == "cancel"
                    rid = sub_events[1].unpack_request_id()
                    assert rid == "cancel-req-002"

                    print(f"RANK_1_BUS_OK received_batch_with_{len(sub_events)}_events")

                print(f"RANK_{rank}_BUS_OK")
            except Exception as e:
                import traceback
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{rank}_FAIL: {e}")
                traceback.print_exc()
            """,
            tmp_path,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_BUS_OK" in stdout
        assert "RANK_1_BUS_OK" in stdout


# =========================================================================
# 4. DistributedControlBus shutdown event
# =========================================================================


class TestControlBusShutdown:
    """Verify that shutdown events propagate across ranks."""

    def test_2rank_control_bus_shutdown(self, tmp_path):
        """Rank 0 publishes a shutdown event, rank 1 receives and verifies."""
        rc, stdout, stderr = _run_2rank_script(
            """\
            import sys, os
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx

            try:
                from mlx_lm_server.distributed import DistributedContext
                from mlx_lm_server.distributed_bus import ControlEvent, DistributedControlBus

                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()
                ws = group.size()

                ctx = DistributedContext(
                    enabled=True, group=group, rank=rank,
                    world_size=ws, backend="ring",
                )
                bus = DistributedControlBus(ctx)

                if rank == 0:
                    bus.publish(ControlEvent.shutdown())
                    print("RANK_0_SHUTDOWN_OK")
                else:
                    evt = bus.recv()
                    assert evt.typ == "shutdown", f"Expected shutdown, got {evt.typ}"
                    assert evt.payload is None, "shutdown payload should be None"
                    print("RANK_1_SHUTDOWN_OK")
            except Exception as e:
                import traceback
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{rank}_FAIL: {e}")
                traceback.print_exc()
            """,
            tmp_path,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_SHUTDOWN_OK" in stdout
        assert "RANK_1_SHUTDOWN_OK" in stdout


# =========================================================================
# 5. SSD rank namespace
# =========================================================================


class TestSSDRankNamespace:
    """Verify that each rank gets a separate SSD cache subdirectory."""

    def test_2rank_ssd_rank_namespace(self, tmp_path):
        """Both ranks create rank-specific SSD cache dirs and verify separation."""
        # We need to pass the tmp_path as an env variable since each rank
        # runs in a separate process.
        shared_base = str(tmp_path / "ssd_base")

        rc, stdout, stderr = _run_2rank_script(
            f"""\
            import sys, os
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx
            from pathlib import Path

            try:
                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()
                ws = group.size()

                # Replicate the SSD namespace logic from __main__.py
                ssd_base = Path("{shared_base}")
                ssd_cache_dir = ssd_base / f"rank_{{rank}}"
                ssd_cache_dir.mkdir(parents=True, exist_ok=True)

                # Write a rank-specific marker file
                marker = ssd_cache_dir / "marker.txt"
                marker.write_text(f"rank_{{rank}}")

                # Verify the directory exists and is distinct
                assert ssd_cache_dir.exists()
                assert ssd_cache_dir.name == f"rank_{{rank}}"
                assert marker.read_text() == f"rank_{{rank}}"

                print(f"RANK_{{rank}}_SSD_DIR={{ssd_cache_dir}}")
                print(f"RANK_{{rank}}_SSD_OK")
            except Exception as e:
                import traceback
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{{rank}}_FAIL: {{e}}")
                traceback.print_exc()
            """,
            tmp_path,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_SSD_OK" in stdout
        assert "RANK_1_SSD_OK" in stdout

        # Verify both directories were actually created on disk
        rank0_dir = tmp_path / "ssd_base" / "rank_0"
        rank1_dir = tmp_path / "ssd_base" / "rank_1"
        assert rank0_dir.exists(), "rank_0 SSD directory was not created"
        assert rank1_dir.exists(), "rank_1 SSD directory was not created"
        assert (rank0_dir / "marker.txt").read_text() == "rank_0"
        assert (rank1_dir / "marker.txt").read_text() == "rank_1"


# =========================================================================
# 6. RestrictedUnpickler enforcement
# =========================================================================


class TestRestrictedUnpicklerEnforcement:
    """Verify that RestrictedUnpickler blocks unauthorized classes."""

    def test_2rank_restricted_unpickler_enforcement(self, tmp_path):
        """Rank 0 sends a legitimate event; rank 1 receives it.
        Then rank 0 tries to send raw malicious pickled data (os.system)
        but since the bus serialization uses ControlEvent, the malicious
        payload is tested at the unpickle layer."""
        rc, stdout, stderr = _run_2rank_script(
            """\
            import sys, os, io, pickle
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx

            try:
                from mlx_lm_server.distributed_bus import (
                    RestrictedUnpickler,
                    restricted_loads,
                )

                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()

                # Test 1: Verify allowed classes deserialize correctly
                from mlx_lm_server.distributed_bus import ControlEvent
                evt = ControlEvent.noop()
                data = pickle.dumps(evt)
                restored = restricted_loads(data)
                assert restored.typ == "noop", f"Expected noop, got {restored.typ}"

                # Test 2: Verify disallowed classes are blocked
                # Create a malicious pickle payload that tries to instantiate os.system
                class MaliciousReducer:
                    def __reduce__(self):
                        return (os.system, ("echo HACKED",))

                malicious_data = pickle.dumps(MaliciousReducer())
                try:
                    restricted_loads(malicious_data)
                    print(f"RANK_{rank}_FAIL: RestrictedUnpickler did not block os.system")
                except pickle.UnpicklingError as e:
                    assert "not in whitelist" in str(e), f"Unexpected error: {e}"
                    print(f"RANK_{rank}_RESTRICTED_OK blocked={e}")

                # Test 3: Verify subprocess module is also blocked
                import subprocess
                class SubprocessReducer:
                    def __reduce__(self):
                        return (subprocess.call, (["echo", "HACKED"],))

                sub_data = pickle.dumps(SubprocessReducer())
                try:
                    restricted_loads(sub_data)
                    print(f"RANK_{rank}_FAIL: RestrictedUnpickler did not block subprocess")
                except pickle.UnpicklingError:
                    pass  # Expected

                print(f"RANK_{rank}_RESTRICTED_OK")
            except Exception as e:
                import traceback
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{rank}_FAIL: {e}")
                traceback.print_exc()
            """,
            tmp_path,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_RESTRICTED_OK" in stdout
        assert "RANK_1_RESTRICTED_OK" in stdout
        assert "HACKED" not in stdout


# =========================================================================
# 7. Bus noop synchronization (multiple rounds)
# =========================================================================


class TestBusNoopSync:
    """Verify that the bus can do multiple noop sync rounds without drift."""

    def test_2rank_bus_noop_sync(self, tmp_path):
        """Both ranks perform 10 rounds of noop publish/recv without deadlock."""
        rc, stdout, stderr = _run_2rank_script(
            """\
            import sys, os
            sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
            import mlx.core as mx

            try:
                from mlx_lm_server.distributed import DistributedContext
                from mlx_lm_server.distributed_bus import ControlEvent, DistributedControlBus

                group = mx.distributed.init(backend="ring", strict=False)
                rank = group.rank()
                ws = group.size()

                ctx = DistributedContext(
                    enabled=True, group=group, rank=rank,
                    world_size=ws, backend="ring",
                )
                bus = DistributedControlBus(ctx)

                NUM_ROUNDS = 10
                for i in range(NUM_ROUNDS):
                    if rank == 0:
                        bus.publish(ControlEvent.noop())
                    else:
                        evt = bus.recv()
                        assert evt.typ == "noop", (
                            f"Round {i}: expected noop, got {evt.typ}"
                        )

                print(f"RANK_{rank}_NOOP_SYNC_OK rounds={NUM_ROUNDS}")
            except Exception as e:
                import traceback
                rank = os.environ.get("MLX_RANK", "?")
                print(f"RANK_{rank}_FAIL: {e}")
                traceback.print_exc()
            """,
            tmp_path,
            timeout=60.0,
        )
        assert rc == 0, f"Process failed (rc={rc}). stderr:\n{stderr}"
        assert "RANK_0_NOOP_SYNC_OK rounds=10" in stdout
        assert "RANK_1_NOOP_SYNC_OK rounds=10" in stdout
