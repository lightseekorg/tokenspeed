"""HTTP server sidecar that runs alongside the smg gateway.

Runs automatically on ``main_port + 1`` when ``tokenspeed serve`` starts.
Override the port with ``--control-port PORT``.

Architecture::

    Client  ──►  http_server  :8001
                    ├─ /health, /get_server_info, /get_model_info,
                    │  /health_check, /abort  ──►  gRPC engine  (direct)
                    └─ /generate, /v1/*, /flush_cache
                         ──►  smg gateway  :8000  ──►  gRPC engine
"""

from __future__ import annotations

import aiohttp
import grpc
import grpc.aio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from google.protobuf.json_format import MessageToDict
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2 as pb
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2_grpc as pb_grpc

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

app = FastAPI()

# Set by start() before uvicorn.run().
_gateway_url: str = ""
_engine_grpc_addr: str = ""

_STREAM_CHUNK_SIZE = 8192


def _stub() -> pb_grpc.TokenSpeedSchedulerStub:
    channel = grpc.aio.insecure_channel(_engine_grpc_addr)
    return pb_grpc.TokenSpeedSchedulerStub(channel)


# ---------------------------------------------------------------------------
# Health (local)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# gRPC direct
# ---------------------------------------------------------------------------


@app.get("/get_server_info")
async def get_server_info():
    resp = await _stub().GetServerInfo(pb.GetServerInfoRequest())
    return JSONResponse(MessageToDict(resp, preserving_proto_field_name=True))


@app.get("/get_model_info")
async def get_model_info():
    resp = await _stub().GetModelInfo(pb.GetModelInfoRequest())
    return JSONResponse(MessageToDict(resp, preserving_proto_field_name=True))


@app.get("/health_check")
async def health_check():
    resp = await _stub().HealthCheck(pb.HealthCheckRequest())
    return JSONResponse(MessageToDict(resp, preserving_proto_field_name=True))


@app.post("/abort")
async def abort(request: Request):
    body = await request.json()
    resp = await _stub().Abort(
        pb.AbortRequest(
            request_id=body.get("request_id", ""),
            reason=body.get("reason", ""),
        )
    )
    return JSONResponse(MessageToDict(resp, preserving_proto_field_name=True))


# ---------------------------------------------------------------------------
# smg passthrough — generation + cache
# ---------------------------------------------------------------------------


async def _proxy_request(request: Request) -> StreamingResponse | JSONResponse:
    url = f"{_gateway_url.rstrip('/')}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async def _iter(resp):
        async for chunk in resp.content.iter_chunked(_STREAM_CHUNK_SIZE):
            yield chunk

    async with aiohttp.ClientSession() as session:
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
                _iter(resp),
                status_code=resp.status,
                media_type="text/event-stream",
            )
        data = await resp.read()
        return Response(
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


@app.post("/flush_cache")
async def flush_cache(request: Request):
    return await _proxy_request(request)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile(request: Request):
    return await _proxy_request(request)


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile(request: Request):
    return await _proxy_request(request)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start(
    *,
    gateway_url: str,
    engine_grpc_addr: str,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> None:
    """Start the HTTP server (blocking).

    Args:
        gateway_url: Base URL of the smg gateway for generation passthrough.
        engine_grpc_addr: ``host:port`` of the gRPC engine for direct calls.
        host: Bind address.
        port: Bind port.
    """
    global _gateway_url, _engine_grpc_addr
    _gateway_url = gateway_url
    _engine_grpc_addr = engine_grpc_addr
    logger.info(
        "Starting TokenSpeed HTTP server on %s:%d " "(gateway: %s, engine gRPC: %s)",
        host,
        port,
        gateway_url,
        engine_grpc_addr,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
