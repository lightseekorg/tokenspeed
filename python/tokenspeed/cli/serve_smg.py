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

"""``ts serve`` orchestrator: spawn smg gateway + gRPC engine, tag logs, probe
readiness, and tear down gateway-first on shutdown."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from tokenspeed.cli._argsplit import OrchestratorOpts, split_argv
from tokenspeed.cli._logprefix import ENGINE_TAG, GATEWAY_TAG, tag_stream
from tokenspeed.cli._proc import (
    spawn_engine,
    spawn_gateway,
    terminate_then_kill,
    wait_grpc_serving,
    wait_http_ready,
)
from tokenspeed.runtime.utils.network import get_free_port
from tokenspeed.runtime.utils.process import kill_process_tree

logger = logging.getLogger(__name__)


def _check_serve_extra_installed() -> None:
    import importlib.util

    missing: list[str] = []
    if importlib.util.find_spec("smg") is None:
        missing.append("smg")
    if importlib.util.find_spec("smg_grpc_servicer.tokenspeed.server") is None:
        missing.append("smg-grpc-servicer")
    if missing:
        sys.stderr.write(
            "ts serve requires the [serve] extra:\n\n"
            "    pip install 'tokenspeed[serve]' \\\n"
            "        --extra-index-url https://lightseek.org/whl/cu130/\n\n"
            "Swap the index for other variants:\n"
            "    https://lightseek.org/whl/cu129/      (CUDA 12.9)\n"
            "    https://lightseek.org/whl/rocm7.2/    (ROCm 7.2)\n\n"
            f"Missing: {', '.join(missing)}\n"
        )
        sys.exit(1)


def _user_host_port_from_gateway_args(gateway_args: list[str]) -> tuple[str, int]:
    """Pull --host / --port out of the gateway-bound argv.

    Defaults match smg's clap (host=0.0.0.0, port=30000). The argv MUST
    be in canonical ``[--flag, value, ...]`` form as produced by
    ``split_argv``; equals-form (``--port=8000``) is not handled here.
    """
    host = "0.0.0.0"
    port = 30000
    it = iter(gateway_args)
    for token in it:
        if token == "--host":
            host = next(it)
        elif token == "--port":
            port = int(next(it))
    return host, port


async def _stream_to(proc, tag: str) -> None:
    await asyncio.gather(
        tag_stream(proc.stdout, tag, sys.stdout),
        tag_stream(proc.stderr, tag, sys.stderr),
    )


async def _drain_log(task: asyncio.Task, timeout: float = 2.0) -> None:
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()


class _ShutdownDuringStartup(Exception):
    pass


class _ChildExitedDuringStartup(Exception):
    pass


async def _probe_or_stop(
    probe_coro, stop_event: asyncio.Event, *, proc=None, label: str = ""
):
    """Race a readiness probe against the stop event and (optionally) the
    subprocess's own exit.

    - probe success → return result
    - stop event   → raise ``_ShutdownDuringStartup``
    - proc exits   → raise ``_ChildExitedDuringStartup`` with returncode + label
    """
    probe_task = asyncio.create_task(probe_coro)
    stop_task = asyncio.create_task(stop_event.wait())
    tasks = [probe_task, stop_task]
    proc_task = None
    if proc is not None:
        proc_task = asyncio.create_task(proc.wait())
        tasks.append(proc_task)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    if proc_task is not None and proc_task in done:
        rc = proc_task.result()
        raise _ChildExitedDuringStartup(
            f"{label} subprocess exited with rc={rc} during startup; "
            f"see [{label}] log lines above for the cause"
        )
    if stop_task in done:
        raise _ShutdownDuringStartup()
    return probe_task.result()


async def run_smg(
    *,
    engine_args: list[str],
    gateway_args: list[str],
    opts: OrchestratorOpts,
    user_host: str,
    user_port: int,
    _stop_event: asyncio.Event | None = None,
) -> int:
    """Lifecycle loop. Returns the orchestrator's exit code."""
    engine = None
    gateway = None
    engine_log: asyncio.Task | None = None
    gateway_log: asyncio.Task | None = None

    # Install signal handlers before spawning any subprocess so a Ctrl-C
    # during the readiness probe doesn't skip terminate_then_kill.
    stop = _stop_event if _stop_event is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows: signal handlers via asyncio aren't supported. Out of scope.

    try:
        engine_port = get_free_port()

        engine = await spawn_engine(engine_args, host="127.0.0.1", port=engine_port)
        engine_log = asyncio.create_task(_stream_to(engine, ENGINE_TAG))

        await _probe_or_stop(
            wait_grpc_serving(
                f"127.0.0.1:{engine_port}",
                timeout=float(opts.engine_startup_timeout),
            ),
            stop,
            proc=engine,
            label=ENGINE_TAG,
        )

        gateway = await spawn_gateway(
            gateway_args, engine_host="127.0.0.1", engine_port=engine_port
        )
        gateway_log = asyncio.create_task(_stream_to(gateway, GATEWAY_TAG))

        await _probe_or_stop(
            wait_http_ready(
                f"http://{user_host}:{user_port}/health",
                timeout=float(opts.gateway_startup_timeout),
            ),
            stop,
            proc=gateway,
            label=GATEWAY_TAG,
        )

        sys.stdout.write(f"ts serve ready on http://{user_host}:{user_port}\n")
        sys.stdout.flush()

        engine_wait = asyncio.create_task(engine.wait())
        gateway_wait = asyncio.create_task(gateway.wait())
        stop_wait = asyncio.create_task(stop.wait())

        done, pending = await asyncio.wait(
            [engine_wait, gateway_wait, stop_wait],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        rc_engine = engine.returncode if engine.returncode is not None else 0
        rc_gateway = gateway.returncode if gateway.returncode is not None else 0
        if rc_engine != 0:
            return rc_engine
        if rc_gateway != 0:
            return rc_gateway
        return 0

    except _ChildExitedDuringStartup as exc:
        logger.error("startup failed: %s", exc)
        return 1
    except _ShutdownDuringStartup:
        logger.info("shutdown signal received during startup; exiting cleanly")
        return 0
    except TimeoutError as exc:
        logger.error("startup failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("interrupted; exiting cleanly")
        return 0
    finally:
        # Shutdown order: gateway first, then engine.
        if gateway is not None:
            await terminate_then_kill(gateway, drain_timeout=opts.drain_timeout)
        if engine is not None:
            await terminate_then_kill(engine, drain_timeout=opts.drain_timeout)

        drain_tasks = [
            _drain_log(t) for t in (engine_log, gateway_log) if t is not None
        ]
        if drain_tasks:
            await asyncio.gather(*drain_tasks, return_exceptions=True)

        # Final reap: walk only the children of our two known subprocesses —
        # never os.getpid(), which under pytest would walk the test runner's
        # children and SIGKILL unrelated test fixtures.
        for proc in (engine, gateway):
            if proc is not None:
                try:
                    kill_process_tree(proc.pid, include_parent=False)
                except Exception:  # noqa: BLE001
                    pass


def run_smg_from_args(args: argparse.Namespace, raw_argv: list[str]) -> None:
    """Entry point called from cli/__main__.py for ``ts serve``."""
    try:
        import setproctitle

        setproctitle.setproctitle("ts-serve")
    except ImportError:
        pass

    _check_serve_extra_installed()
    split = split_argv(raw_argv)
    user_host, user_port = _user_host_port_from_gateway_args(split.gateway)
    rc = asyncio.run(
        run_smg(
            engine_args=split.engine,
            gateway_args=split.gateway,
            opts=split.opts,
            user_host=user_host,
            user_port=user_port,
        )
    )
    sys.exit(rc)
