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

"""Unit tests for the smg orchestrator lifecycle."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli._argsplit import OrchestratorOpts
from tokenspeed.cli.serve_smg import (
    _DEFAULT_SMG_DISABLE_FLAGS,
    _gateway_args_with_default_log_level,
    _gateway_args_with_default_port,
    _gateway_args_with_default_reasoning_parser,
    _gateway_args_with_defaults,
    _gateway_args_with_smg_disable_defaults,
    _user_host_port_from_gateway_args,
    run_smg,
)


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


def test_gateway_args_default_port_is_8000():
    gateway_args = _gateway_args_with_default_port(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--port", "8000"]
    assert _user_host_port_from_gateway_args(gateway_args) == ("0.0.0.0", 8000)


def test_gateway_args_preserve_user_port():
    gateway_args = _gateway_args_with_default_port(["--port", "8413"])

    assert gateway_args == ["--port", "8413"]
    assert _user_host_port_from_gateway_args(gateway_args) == ("0.0.0.0", 8413)


def test_gateway_args_default_reasoning_parser_is_none():
    gateway_args = _gateway_args_with_default_reasoning_parser(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--reasoning-parser", "none"]


def test_gateway_args_preserve_user_reasoning_parser():
    gateway_args = _gateway_args_with_default_reasoning_parser(
        ["--reasoning-parser", "qwen3"]
    )

    assert gateway_args == ["--reasoning-parser", "qwen3"]


def test_gateway_args_defaults_include_port_and_reasoning_parser():
    gateway_args = _gateway_args_with_defaults(["--model", "/tmp/x"])

    assert gateway_args == [
        "--model",
        "/tmp/x",
        "--port",
        "8000",
        "--reasoning-parser",
        "none",
        "--disable-circuit-breaker",
        "--disable-retries",
        "--log-level",
        "warn",
    ]


def test_gateway_args_default_log_level_is_warn():
    gateway_args = _gateway_args_with_default_log_level(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--log-level", "warn"]


def test_gateway_args_preserve_user_log_level():
    gateway_args = _gateway_args_with_default_log_level(["--log-level", "debug"])

    assert gateway_args == ["--log-level", "debug"]


def test_smg_disable_flags_appended_when_absent():
    gateway_args = _gateway_args_with_smg_disable_defaults(["--model", "/tmp/x"])

    assert gateway_args == [
        "--model",
        "/tmp/x",
        "--disable-circuit-breaker",
        "--disable-retries",
    ]


def test_smg_disable_flags_not_duplicated():
    """Idempotent: passing the flag explicitly must not double it up for smg's argparse."""
    gateway_args = _gateway_args_with_smg_disable_defaults(
        ["--disable-circuit-breaker"]
    )

    assert gateway_args == [
        "--disable-circuit-breaker",
        "--disable-retries",
    ]


def test_smg_disable_flag_set_covers_both():
    assert _DEFAULT_SMG_DISABLE_FLAGS == (
        "--disable-circuit-breaker",
        "--disable-retries",
    )


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

    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 0

    async def gateway_wait():
        await startup_done.wait()
        return 0

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    async def probe_then_schedule_release(*args, **kwargs):
        # Schedule release for the next tick so probe success is recorded
        # before gateway.wait() resolves.
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
    """Signal handlers must be live before any subprocess."""
    call_order: list[str] = []

    real_loop = asyncio.get_running_loop()
    real_add_signal_handler = real_loop.add_signal_handler

    def tracking_add_signal_handler(sig, callback, *args):
        call_order.append(f"add_signal_handler:{sig}")
        return real_add_signal_handler(sig, callback, *args)

    async def tracking_spawn_engine(*args, **kwargs):
        call_order.append("spawn_engine")
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
    """SIGTERM during engine readiness probe is a clean exit (rc=0)."""
    engine = _make_proc(returncode=0)
    opts = OrchestratorOpts(engine_startup_timeout=10)

    async def slow_probe(*args, **kwargs):
        await asyncio.sleep(60)

    async def hung_wait():
        await asyncio.sleep(60)

    engine.wait = AsyncMock(side_effect=hung_wait)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
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

    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 42

    async def gateway_wait():
        await asyncio.Event().wait()

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    async def gateway_probe_then_release(*args, **kwargs):
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
    """If the engine exits before gRPC SERVING, return rc=1 immediately."""
    engine = _make_proc(returncode=2)
    opts = OrchestratorOpts(engine_startup_timeout=600)

    async def hung_probe(*args, **kwargs):
        await asyncio.sleep(60)

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
    """If the gateway exits during wait_http_ready, return rc=1 immediately."""
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
    """Orchestrator sets proc title to 'ts-serve' so pgrep -f ts-serve finds it."""
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
