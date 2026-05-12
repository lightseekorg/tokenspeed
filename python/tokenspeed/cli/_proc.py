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

"""Subprocess supervision helpers for ``ts serve``.

Responsibilities:

* Pick a free internal port for the engine subprocess.
* Spawn engine (``python -m smg_grpc_servicer.tokenspeed``) and gateway
  (``python -m smg launch``) with stdout/stderr piped to the orchestrator
  for tagging.
* Probe readiness: ``grpc.health.v1.Health.Check`` for the engine,
  HTTP ``GET /health`` for the gateway.
* Graceful shutdown: SIGTERM, drain, SIGKILL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import sys
import time

logger = logging.getLogger(__name__)


def pick_free_port() -> int:
    """Bind ``127.0.0.1:0`` and return the kernel-assigned port.

    The socket is closed before returning, so the port is free at the
    moment of return — there is a small TOCTOU window between this call
    and the engine's ``bind()``. In practice the engine binds within
    milliseconds; collision would require another process to grab the
    same ephemeral port in the gap, which the kernel actively avoids.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def spawn_engine(
    args: list[str],
    *,
    host: str,
    port: int,
) -> asyncio.subprocess.Process:
    """Spawn the engine subprocess with PIPE stdio.

    The default command is ``python -m smg_grpc_servicer.tokenspeed``.
    For testing without CUDA, set ``TS_SERVE_ENGINE_MODULE`` to point at
    a fake engine module that implements the same ``--host``/``--port``
    CLI and the gRPC health + minimal TokenSpeedScheduler service.
    """
    module = os.environ.get("TS_SERVE_ENGINE_MODULE", "smg_grpc_servicer.tokenspeed")
    cmd = [
        sys.executable,
        "-m",
        module,
        "--host",
        host,
        "--port",
        str(port),
        *args,
    ]
    logger.info("spawn engine: %s", " ".join(cmd))
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def spawn_gateway(
    args: list[str],
    *,
    engine_host: str,
    engine_port: int,
) -> asyncio.subprocess.Process:
    """Spawn ``python -m smg launch`` with PIPE stdio.

    Single-worker topology — retries and circuit-breaker are disabled
    by default (no other worker to fail over to).
    """
    cmd = [
        sys.executable,
        "-m",
        "smg",
        "launch",
        "--worker-urls",
        f"grpc://{engine_host}:{engine_port}",
        "--disable-retries",
        "--disable-circuit-breaker",
        *args,
    ]
    logger.info("spawn gateway: %s", " ".join(cmd))
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def wait_grpc_serving(
    target: str,
    *,
    timeout: float,
    poll_interval: float = 1.0,
) -> None:
    """Poll ``grpc.health.v1.Health.Check`` on ``target`` until SERVING.

    Raises ``TimeoutError`` if the deadline elapses without SERVING.
    """
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = health_pb2_grpc.HealthStub(channel)
                resp = await stub.Check(
                    health_pb2.HealthCheckRequest(service=""),
                    timeout=2.0,
                )
                if resp.status == health_pb2.HealthCheckResponse.SERVING:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        await asyncio.sleep(poll_interval)
    detail = f" (last error: {last_err!r})" if last_err is not None else ""
    raise TimeoutError(
        f"engine never reached SERVING on {target} within {timeout:.0f}s{detail}"
    )


async def wait_http_ready(
    url: str,
    *,
    timeout: float,
    poll_interval: float = 1.0,
) -> None:
    """Poll HTTP ``GET <url>`` until 200, or raise ``TimeoutError``."""
    import aiohttp

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2.0)
                ) as resp:
                    if resp.status == 200:
                        return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        await asyncio.sleep(poll_interval)
    detail = f" (last error: {last_err!r})" if last_err is not None else ""
    raise TimeoutError(
        f"gateway never reached 200 on {url} within {timeout:.0f}s{detail}"
    )


async def terminate_then_kill(proc, *, drain_timeout: float) -> None:
    """SIGTERM, wait up to ``drain_timeout``, then SIGKILL if still alive."""
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=drain_timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
