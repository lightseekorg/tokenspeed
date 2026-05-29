"""HTTP server sidecar that runs alongside the smg gateway.

Runs automatically on ``main_port + 1`` when ``tokenspeed serve`` starts.
Override the port with ``--control-port PORT``.

Architecture::

    Client  ──►  http_server  :8001  ──►  smg gateway  :8000  ──►  gRPC engine

Generation endpoints (/generate, /v1/chat/completions, /v1/completions, etc.)
are proxied transparently to smg, including streaming responses.
"""

from __future__ import annotations

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

app = FastAPI()

# Set by start() before uvicorn.run() is called.
_gateway_url: str = ""

_STREAM_CHUNK_SIZE = 8192


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Control — stubs (direct engine wiring to be added)
# ---------------------------------------------------------------------------


@app.post("/flush_cache")
async def flush_cache(request: Request):
    return await _proxy_request(request)


@app.get("/get_server_info")
async def get_server_info():
    return JSONResponse({})


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile():
    return JSONResponse({"status": "ok"})


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile():
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Passthrough proxy
# ---------------------------------------------------------------------------


async def _proxy_request(request: Request) -> StreamingResponse | JSONResponse:
    """Forward any request to the smg gateway, preserving streaming."""
    url = f"{_gateway_url.rstrip('/')}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async def _iter_response(resp):
        async for chunk in resp.content.iter_chunked(_STREAM_CHUNK_SIZE):
            yield chunk

    session = aiohttp.ClientSession()
    resp = await session.request(
        method=request.method,
        url=url,
        headers=headers,
        data=body,
        timeout=aiohttp.ClientTimeout(total=600),
    )

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return StreamingResponse(
            _iter_response(resp),
            status_code=resp.status,
            media_type="text/event-stream",
            background=None,
        )

    data = await resp.read()
    await session.close()
    return JSONResponse(
        content=None,
        status_code=resp.status,
        headers=dict(resp.headers),
    ).__class__(
        content=data,
        status_code=resp.status,
        media_type=content_type or "application/json",
    )


@app.api_route("/generate", methods=["GET", "POST"])
async def generate(request: Request):
    return await _proxy_request(request)


@app.api_route("/v1/completions", methods=["POST"])
async def completions(request: Request):
    return await _proxy_request(request)


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    return await _proxy_request(request)


@app.api_route("/v1/models", methods=["GET"])
async def models(request: Request):
    return await _proxy_request(request)


@app.api_route("/v1/messages", methods=["POST"])
async def messages(request: Request):
    return await _proxy_request(request)


@app.api_route("/v1/responses", methods=["POST"])
async def responses(request: Request):
    return await _proxy_request(request)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start(
    *,
    gateway_url: str,
    host: str = "127.0.0.1",
    port: int = 8081,
) -> None:
    """Start the HTTP server (blocking).

    Args:
        gateway_url: Base URL of the smg gateway, e.g. ``http://127.0.0.1:8000``.
        host: Bind address.
        port: Bind port.
    """
    global _gateway_url
    _gateway_url = gateway_url
    logger.info(
        "Starting TokenSpeed HTTP server on %s:%d (gateway: %s)",
        host,
        port,
        gateway_url,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
