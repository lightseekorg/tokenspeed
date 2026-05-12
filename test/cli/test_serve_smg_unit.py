# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unit tests for the smg orchestrator lifecycle.

We mock ``_proc`` so no real subprocesses are spawned. The goal is to
pin shutdown ordering, exit-code propagation, and timeout escalation.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokenspeed.cli._argsplit import OrchestratorOpts
from tokenspeed.cli.serve_smg import run_smg


def _make_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = asyncio.StreamReader()
    proc.stderr = asyncio.StreamReader()
    proc.stdout.feed_eof()
    proc.stderr.feed_eof()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode if returncode is not None else 0)
    return proc


@pytest.mark.asyncio
async def test_engine_start_timeout_kills_engine_and_exits_nonzero():
    engine = _make_proc()
    opts = OrchestratorOpts(engine_startup_timeout=0)
    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving",
        AsyncMock(side_effect=TimeoutError("engine never reached SERVING")),
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ) as tk:
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc != 0
    tk.assert_awaited_with(engine, drain_timeout=opts.drain_timeout)


@pytest.mark.asyncio
async def test_gateway_first_then_engine_on_clean_shutdown():
    """Gateway is SIGTERMed before the engine (front before back)."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    call_order: list[str] = []

    async def tracked_term(proc, *, drain_timeout):
        if proc is gateway:
            call_order.append("gateway")
        elif proc is engine:
            call_order.append("engine")

    # During startup, both child procs are alive — wait() must hang so the
    # fail-fast-on-child-exit path doesn't trigger. Probes resolve first,
    # then in the post-ready phase gateway exits cleanly to drive shutdown.
    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 0

    async def gateway_wait():
        # Block during startup; once startup is done, resolve immediately to
        # simulate "gateway exited cleanly" and drive the shutdown path.
        await startup_done.wait()
        return 0

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    async def probe_then_schedule_release(*args, **kwargs):
        # The second probe (wait_http_ready) is the last thing before the
        # main lifecycle await. Schedule the gate release for the next
        # event-loop tick so probe success is recorded by _probe_or_stop
        # *before* gateway.wait() resolves (which would otherwise trip the
        # new fail-fast-on-child-exit path).
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, startup_done.set)

    opts = OrchestratorOpts(engine_startup_timeout=10, gateway_startup_timeout=10)
    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", AsyncMock(return_value=gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready",
        side_effect=probe_then_schedule_release,
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", side_effect=tracked_term
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert call_order == ["gateway", "engine"]
    assert rc == 0


@pytest.mark.asyncio
async def test_signal_handlers_installed_before_spawning_engine():
    """Signal handlers must be live before any subprocess.

    Otherwise a Ctrl-C during the readiness probe (up to
    engine_startup_timeout = 600s by default) propagates KeyboardInterrupt
    out of asyncio.run, skipping terminate_then_kill and leaking the
    engine subprocess.

    We assert the order by patching ``loop.add_signal_handler`` on the
    running loop and recording the order of calls relative to
    ``spawn_engine``.
    """
    call_order: list[str] = []

    real_loop = asyncio.get_running_loop()
    real_add_signal_handler = real_loop.add_signal_handler

    def tracking_add_signal_handler(sig, callback, *args):
        call_order.append(f"add_signal_handler:{sig}")
        return real_add_signal_handler(sig, callback, *args)

    async def tracking_spawn_engine(*args, **kwargs):
        call_order.append("spawn_engine")
        # Raise immediately so we don't have to mock the rest of the
        # lifecycle. The orchestrator's finally block will run; that's
        # what we care about (didn't crash without cleanup).
        raise RuntimeError("simulated spawn failure")

    opts = OrchestratorOpts()
    with patch.object(
        real_loop, "add_signal_handler", side_effect=tracking_add_signal_handler
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_engine", side_effect=tracking_spawn_engine
    ):
        with pytest.raises(RuntimeError, match="simulated spawn failure"):
            await run_smg(
                engine_args=[],
                gateway_args=[],
                opts=opts,
                user_host="127.0.0.1",
                user_port=8000,
            )

    # Both signals must have their handlers registered BEFORE spawn_engine.
    assert "spawn_engine" in call_order
    spawn_idx = call_order.index("spawn_engine")
    sigterm_idx = call_order.index(f"add_signal_handler:{signal.SIGTERM}")
    sigint_idx = call_order.index(f"add_signal_handler:{signal.SIGINT}")
    assert sigterm_idx < spawn_idx, (
        f"SIGTERM handler installed after spawn_engine " f"(order: {call_order})"
    )
    assert sigint_idx < spawn_idx, (
        f"SIGINT handler installed after spawn_engine " f"(order: {call_order})"
    )


@pytest.mark.asyncio
async def test_stop_during_engine_probe_exits_zero():
    """SIGTERM during engine readiness probe is a clean exit (rc=0), not a failure.

    Race scenario: external /health probe (e.g. integration test polling
    at 500ms) sees ready and SIGTERMs the orchestrator while the
    orchestrator's own internal probe is still polling. The orchestrator
    must treat this as a clean shutdown, not a startup failure.
    """
    engine = _make_proc(returncode=0)
    opts = OrchestratorOpts(engine_startup_timeout=10)

    async def slow_probe(*args, **kwargs):
        # Simulate a probe that's still polling when stop fires.
        await asyncio.sleep(60)

    # During startup the engine is alive — its wait() must hang so the
    # fail-fast-on-child-exit path doesn't trigger.
    async def hung_wait():
        await asyncio.sleep(60)

    engine.wait = AsyncMock(side_effect=hung_wait)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # Trip the stop event after a tick so the probe loses the race.
    loop.call_later(0.05, stop.set)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", side_effect=slow_probe
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
            _stop_event=stop,
        )
    assert rc == 0


@pytest.mark.asyncio
async def test_first_nonzero_child_exit_propagates():
    engine = _make_proc(returncode=42)
    gateway = _make_proc(returncode=0)
    opts = OrchestratorOpts()

    # During startup probes the engine must be "alive" (wait hangs). After
    # both probes complete the gateway probe releases the gate, and engine
    # then exits with rc=42 to drive the post-ready exit path.
    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 42

    async def gateway_wait():
        # Gateway hasn't exited; wait forever until cancelled.
        await asyncio.Event().wait()

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    async def gateway_probe_then_release(*args, **kwargs):
        # Schedule the gate release for a later tick so probe success is
        # recorded by _probe_or_stop *before* engine.wait() resolves —
        # otherwise the new fail-fast-on-child-exit path trips during
        # startup instead of the post-ready exit path we want to test.
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, startup_done.set)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", AsyncMock(return_value=gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready",
        side_effect=gateway_probe_then_release,
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc == 42


@pytest.mark.asyncio
async def test_engine_exit_during_probe_fails_fast():
    """If the engine subprocess exits before gRPC SERVING, return rc=1 immediately
    instead of waiting for engine_startup_timeout. Bug surfaced when bad
    --tool-call-parser/--reasoning-parser values made smg's clap exit at startup
    and the orchestrator waited the full timeout."""
    engine = _make_proc(returncode=2)  # already-exited child
    opts = OrchestratorOpts(engine_startup_timeout=600)

    async def hung_probe(*args, **kwargs):
        # Probe never resolves; we should NOT reach this completing.
        await asyncio.sleep(60)

    # Make engine.wait() resolve quickly with the non-zero rc.
    engine.wait = AsyncMock(return_value=2)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", side_effect=hung_probe
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc == 1


@pytest.mark.asyncio
async def test_gateway_exit_during_probe_fails_fast():
    """Same as above but for the gateway probe path. Engine probe succeeds first,
    then gateway exits during wait_http_ready."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=2)
    opts = OrchestratorOpts(engine_startup_timeout=10, gateway_startup_timeout=600)

    async def hung_http(*args, **kwargs):
        await asyncio.sleep(60)

    gateway.wait = AsyncMock(return_value=2)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", AsyncMock(return_value=gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready", side_effect=hung_http
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc == 1


def test_run_smg_from_args_sets_process_title(monkeypatch):
    """Bug-2 regression: orchestrator should set proc title to 'ts-serve'
    so pgrep -f ts-serve finds exactly one process.
    """
    captured = {}

    def fake_run(*a, **kw):
        return 0

    monkeypatch.setattr("tokenspeed.cli.serve_smg.asyncio.run", fake_run)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._check_serve_extra_installed", lambda: None
    )
    fake_setproctitle = type(
        "M", (), {"setproctitle": lambda title: captured.setdefault("title", title)}
    )
    monkeypatch.setitem(sys.modules, "setproctitle", fake_setproctitle)

    from argparse import Namespace

    from tokenspeed.cli.serve_smg import run_smg_from_args

    try:
        run_smg_from_args(Namespace(), ["--model", "/tmp/x"])
    except SystemExit:
        pass
    assert captured.get("title") == "ts-serve"
