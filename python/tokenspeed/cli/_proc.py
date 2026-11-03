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

``ts serve`` runs SMG's gateway in front of a headless engine on SMG's direct
ZMQ lane: the gateway binds the engine's sockets and the engine ranks dial
in. The gateway derives every socket from the worker's URL: for an ``ipc://``
URL the data plane under its path and the TCP handshake on a loopback port
hashed from it (``zmq_handshake_port``); for a ``tcp://`` URL, which serves
ranks on other nodes, the handshake at the URL itself. Either way the engine
only needs the handshake's address.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

_GATEWAY_MODULE = "smg"
_ENGINE_MODULE_DEFAULT = "tokenspeed.runtime.entrypoints.engine"
_PIPE_LINE_LIMIT = 64 * 1024 * 1024
# The loopback host of SMG's direct lane: its handshake listens there.
ZMQ_LOOPBACK_HOST = "127.0.0.1"


def zmq_socket_dir() -> str:
    """The directory of the workers' ``ipc://`` sockets: SMG's own default,
    per user (``SMG_ZMQ_SOCKET_DIR`` overrides it, for SMG as for us)."""
    return os.environ.get("SMG_ZMQ_SOCKET_DIR", f"/tmp/smg-zmq-{os.getuid()}")


def zmq_handshake_port(worker_url: str) -> int:
    """The TCP handshake port SMG derives from an ``ipc://`` worker URL.

    Mirrors SMG's ``derive_handshake_port``: FNV-1a over the path after
    ``ipc://``, folded into 20000..29999.
    """
    path = worker_url.removeprefix("ipc://")
    digest = 0xCBF29CE484222325
    for byte in path.encode():
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return 20000 + digest % 10000


def engine_argv(
    args: list[str], *, host: str, port: int, handshake: tuple[str, int] | None
) -> list[str]:
    """Build the command line that runs the headless engine.

    Args:
        args: Engine-side arguments to forward verbatim.
        host: The engine's ``--host``.
        port: The engine's ``--port``, which seeds its derived ports.
        handshake: The gateway's handshake address ``(host, port)``; None on
            a follower node none of whose ranks registers with the gateway.

    Returns:
        An ``argv`` list whose first element is the Python interpreter.
    """
    module = os.environ.get("TS_SERVE_ENGINE_MODULE", _ENGINE_MODULE_DEFAULT)
    argv = [sys.executable, "-m", module, "--host", host, "--port", str(port), *args]
    if handshake is not None:
        handshake_host, handshake_port = handshake
        # Last, so they win over anything in ``args``: the gateway owns the
        # handshake, and its engine identities count from 0.
        argv += [
            "--data-parallel-address",
            handshake_host,
            "--data-parallel-rpc-port",
            str(handshake_port),
            "--zmq-engine-index",
            "0",
        ]
    return argv


async def spawn_engine(
    args: list[str],
    *,
    host: str,
    port: int,
    handshake: tuple[str, int],
) -> asyncio.subprocess.Process:
    """Spawn the headless engine with PIPE stdio, its ranks dialing the
    gateway's ``handshake`` ``(host, port)``."""
    cmd = engine_argv(args, host=host, port=port, handshake=handshake)
    logger.info(f"spawn engine: {' '.join(cmd)}")
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_PIPE_LINE_LIMIT,
    )


async def spawn_gateway(
    args: list[str],
    *,
    worker_url: str,
    engine_count: int,
) -> asyncio.subprocess.Process:
    """Spawn ``python -m smg launch`` with PIPE stdio, fronting one TokenSpeed
    engine at ``worker_url`` with ``engine_count`` attention-DP ranks."""
    cmd = [
        sys.executable,
        "-m",
        _GATEWAY_MODULE,
        "launch",
        "--worker-urls",
        worker_url,
        # The handshake is shared across runtimes: the gateway cannot probe
        # which wire the engine speaks.
        "--backend",
        "tokenspeed",
        "--disable-retries",
        "--disable-circuit-breaker",
    ]
    if engine_count > 1:
        # Every attention-DP rank registers on the worker's one socket set.
        cmd += ["--zmq-engine-count", str(engine_count)]
    cmd += args
    logger.info(f"spawn gateway: {' '.join(cmd)}")
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_PIPE_LINE_LIMIT,
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
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2.0)
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            await asyncio.sleep(poll_interval)
    detail = f" (last error: {last_err!r})" if last_err is not None else ""
    raise TimeoutError(f"{url} never answered 200 within {timeout:.0f}s{detail}")


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
