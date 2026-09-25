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

"""Control-endpoint helpers for the in-engine RL control app (the HTTP surface slime drives).

The engine advertises where its control app listens and what it can do so a
fronting gateway (SMG, ``crates/rl``) can drive it without guessing. The
label keys are SMG's ``rl.*`` capability-override keys; SMG's discovery turns
every key here into a worker label. The one engine import is the scheduler's
``SUPPORTED_WEIGHT_UPDATE_SOURCES``, so the advertisement cannot drift from what
the dispatcher implements; the module still needs no model to import.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.engine.io_struct import SUPPORTED_WEIGHT_UPDATE_SOURCES

# Values SMG's static capability table cannot know about this build. Update when
# a route becomes end to end. ``rl.update_from`` is absent here on purpose:
# :func:`capabilities` derives it from the scheduler's supported-source set.
_CAPABILITIES: dict[str, str] = {
    "rl.pause_modes": "wait,abort,keep",
    "rl.abort": "true",
    "rl.flush_cache": "true",
    "rl.sleep_wake": "true",
    "rl.reports_weight_version": "true",
}


def control_bind_host(server_args: Any) -> str:
    """Host the control app binds: ``--rl-control-host``, else ``--host``."""
    return getattr(server_args, "rl_control_host", None) or server_args.host


def control_url(server_args: Any) -> str | None:
    """Base URL of the control app, or ``None`` when ``--rl-control-port`` is unset."""
    port = getattr(server_args, "rl_control_port", None)
    if not port:
        return None
    host = control_bind_host(server_args)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


def capabilities() -> dict[str, str]:
    """What this build implements, as SMG ``rl.*`` labels.

    ``rl.update_from`` follows the scheduler's dispatcher: a source it cannot
    handle is never advertised, because the control app refuses it with 501.
    """
    caps = dict(_CAPABILITIES)
    caps["rl.update_from"] = ",".join(sorted(SUPPORTED_WEIGHT_UPDATE_SOURCES))
    return caps


def advertisement(server_args: Any) -> dict[str, str]:
    """Everything the gRPC servicer merges into ``GetServerInfo.server_args``."""
    out = capabilities()
    url = control_url(server_args)
    if url:
        out["rl.control_url"] = url
    return out


def install_bearer_auth(app: FastAPI, api_key: str) -> None:
    """Require ``Authorization: Bearer <api_key>`` on every route of ``app``.

    Constant-time comparison; a miss answers 401 in the same JSON shape the
    control routes use for every other failure.
    """
    expected = f"Bearer {api_key}"

    @app.middleware("http")
    async def _require_bearer(request: Request, call_next):
        presented = request.headers.get("authorization", "")
        if not hmac.compare_digest(presented, expected):
            return JSONResponse(
                {"success": False, "message": "unauthorized"}, status_code=401
            )
        return await call_next(request)
