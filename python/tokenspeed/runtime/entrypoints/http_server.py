"""Control-plane HTTP server that runs alongside the smg gateway.

Runs automatically on ``main_port + 1`` when ``tokenspeed serve`` starts.
Override the port with ``--control-port PORT``.

Architecture::

    Client (generation)  ──►  smg gateway  :8000  ──►  gRPC engine
    Client (control)     ──►  http_server   :8001

Currently exposes only health/readiness probes. Additional control endpoints
(pause/continue generation, weight updates, cache flush) will be added once
the engine-side gRPC methods are available.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

app = FastAPI()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/readiness")
async def readiness():
    return JSONResponse({"status": "ready"})


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start(
    *,
    gateway_url: str,
    host: str = "127.0.0.1",
    port: int = 8081,
) -> None:
    """Start the control HTTP server (blocking).

    Args:
        gateway_url: Base URL of the smg gateway (reserved for future use).
        host: Bind address for the control server.
        port: Bind port for the control server.
    """
    logger.info(
        "Starting TokenSpeed control HTTP server on %s:%d",
        host,
        port,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
