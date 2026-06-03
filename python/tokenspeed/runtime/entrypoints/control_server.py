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

"""Control HTTP server (sidecar) that runs alongside the smg gateway.

Runs automatically on ``main_port + 1`` when ``tokenspeed serve`` starts.
Override the port with ``--control-port PORT``.

Architecture::

    Client  ──►  control_server  :8001
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
# Base URL of the in-engine RL weight-sync control plane. The orchestrator
# always sets this for `ts serve`; if it is ever empty the weight routes return
# 503 (the public surface still advertises the paths).
_rl_control_url: str = ""
_grpc_channel: grpc.aio.Channel | None = None
_grpc_stub: pb_grpc.TokenSpeedSchedulerStub | None = None

_STREAM_CHUNK_SIZE = 8192

# Proxy timeout: no `total` cap — streaming responses run as long as the model
# keeps generating, and a wall-clock total would abort long generations
# mid-stream. `sock_connect` bounds connecting to smg; `sock_read` bounds
# inactivity (a genuinely hung/stalled upstream) without killing a legitimately
# long but actively-streaming request.
_PROXY_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)


def _stub() -> pb_grpc.TokenSpeedSchedulerStub:
    # Lazily create a single shared channel/stub. gRPC channels are expensive
    # (HTTP/2 connection + background threads) and must be reused, not created
    # per request, to avoid leaking sockets and file descriptors.
    global _grpc_channel, _grpc_stub
    if _grpc_stub is None:
        _grpc_channel = grpc.aio.insecure_channel(_engine_grpc_addr)
        _grpc_stub = pb_grpc.TokenSpeedSchedulerStub(_grpc_channel)
    return _grpc_stub


# ---------------------------------------------------------------------------
# Health (proxied to smg → engine)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(request: Request):
    return await _proxy_request(request)


# ---------------------------------------------------------------------------
# gRPC direct
# ---------------------------------------------------------------------------


async def _grpc_call(coro) -> JSONResponse:
    """Await a gRPC unary call and serialize the response, mapping engine
    errors to a clean 503 instead of an unhandled 500 + stack trace."""
    try:
        resp = await coro
    except grpc.aio.AioRpcError as e:
        return JSONResponse(
            {"error": "engine unavailable", "detail": e.details()},
            status_code=503,
        )
    return JSONResponse(MessageToDict(resp, preserving_proto_field_name=True))


@app.get("/get_server_info")
async def get_server_info():
    return await _grpc_call(_stub().GetServerInfo(pb.GetServerInfoRequest()))


@app.get("/get_model_info")
async def get_model_info():
    return await _grpc_call(_stub().GetModelInfo(pb.GetModelInfoRequest()))


@app.get("/health_check")
async def health_check():
    return await _grpc_call(_stub().HealthCheck(pb.HealthCheckRequest()))


@app.post("/abort")
async def abort(request: Request):
    body = await request.json()
    return await _grpc_call(
        _stub().Abort(
            pb.AbortRequest(
                request_id=body.get("request_id", ""),
                reason=body.get("reason", ""),
            )
        )
    )


# ---------------------------------------------------------------------------
# smg passthrough — generation + cache
# ---------------------------------------------------------------------------


async def _proxy_request(
    request: Request, base_url: str | None = None
) -> StreamingResponse | Response:
    base_url = base_url if base_url is not None else _gateway_url
    url = f"{base_url.rstrip('/')}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    # NOTE: the session must outlive the response. For streaming, FastAPI
    # consumes the body iterator *after* this function returns, so we cannot
    # close the session in an `async with` block here — it would close the
    # upstream connection mid-stream. Instead the session is closed in the
    # generator's `finally` (streaming) or after `read()` (non-streaming).
    session = aiohttp.ClientSession()
    try:
        resp = await session.request(
            method=request.method,
            url=url,
            headers=headers,
            data=body,
            timeout=_PROXY_TIMEOUT,
        )
    except Exception:
        await session.close()
        raise

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:

        async def _iter():
            try:
                async for chunk in resp.content.iter_chunked(_STREAM_CHUNK_SIZE):
                    yield chunk
            finally:
                resp.release()
                await session.close()

        return StreamingResponse(
            _iter(),
            status_code=resp.status,
            media_type="text/event-stream",
        )

    try:
        data = await resp.read()
        return Response(
            content=data,
            status_code=resp.status,
            media_type=content_type or "application/json",
        )
    finally:
        await session.close()


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
# RL weight transfer — proxied to the in-engine control plane
#
# The weight-update HTTP API that RL trainers drive. The heavy weight payloads
# travel out-of-band (NCCL / CUDA-IPC); only metadata flows through here. The
# control plane runs inside the engine process next to AsyncLLM (see
# runtime/entrypoints/weight_transfer_http.py); the sidecar proxies to it on
# _rl_control_url.
# ---------------------------------------------------------------------------


async def _proxy_to_rl_control(request: Request) -> StreamingResponse | Response:
    if not _rl_control_url:
        return JSONResponse(
            {"error": "weight-sync control plane is unavailable on this server"},
            status_code=503,
        )
    return await _proxy_request(request, base_url=_rl_control_url)


@app.post("/init_weight_transfer_engine")
async def init_weight_transfer_engine(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/start_weight_update")
async def start_weight_update(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/update_weights")
async def update_weights(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/finish_weight_update")
async def finish_weight_update(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/pause")
async def pause(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/resume")
async def resume(request: Request):
    return await _proxy_to_rl_control(request)


@app.get("/get_world_size")
async def get_world_size(request: Request):
    return await _proxy_to_rl_control(request)


@app.get("/is_paused")
async def is_paused(request: Request):
    return await _proxy_to_rl_control(request)


# ---------------------------------------------------------------------------
# RL weight transfer — SGLang dialect, proxied to the same in-engine control app
#
# These routes are mounted on the same in-engine RL control app as the
# vLLM-native endpoints (runtime/entrypoints/sglang_compat_http.py), so they
# proxy to the same _rl_control_url. Endpoint names/fields match SGLang so
# slime/miles and verl's SGLang rollout drive tokenspeed unchanged.
# ---------------------------------------------------------------------------


@app.post("/init_weights_update_group")
async def init_weights_update_group(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/destroy_weights_update_group")
async def destroy_weights_update_group(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/pause_generation")
async def pause_generation(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/continue_generation")
async def continue_generation(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/release_memory_occupation")
async def release_memory_occupation(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request):
    return await _proxy_to_rl_control(request)


@app.post("/abort_request")
async def abort_request(request: Request):
    return await _proxy_to_rl_control(request)


# GET /flush_cache is SGLang's verb; POST /flush_cache (above) proxies to the
# gateway. Both coexist on distinct methods.
@app.get("/flush_cache")
async def flush_cache_get(request: Request):
    return await _proxy_to_rl_control(request)


@app.get("/health_generate")
async def health_generate(request: Request):
    return await _proxy_to_rl_control(request)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def build_control_server(
    *,
    gateway_url: str,
    engine_grpc_addr: str,
    rl_control_url: str = "",
    host: str = "127.0.0.1",
    port: int = 8001,
) -> uvicorn.Server:
    """Configure the proxy targets and return an unstarted ``uvicorn.Server``.

    The caller runs ``server.run()`` (blocking) and may poll ``server.started``
    to detect when the socket is bound and accepting connections.

    Args:
        gateway_url: Base URL of the smg gateway for generation passthrough.
        engine_grpc_addr: ``host:port`` of the gRPC engine for direct calls.
        rl_control_url: Base URL of the in-engine RL control plane (vLLM-native
            + SGLang-compatible weight sync). Empty disables those routes (they
            return 503).
        host: Bind address.
        port: Bind port.
    """
    global _gateway_url, _engine_grpc_addr, _rl_control_url
    _gateway_url = gateway_url
    _engine_grpc_addr = engine_grpc_addr
    _rl_control_url = rl_control_url
    logger.info(
        "Starting TokenSpeed HTTP server on %s:%d "
        "(gateway: %s, engine gRPC: %s, weight transfer: %s)",
        host,
        port,
        gateway_url,
        engine_grpc_addr,
        rl_control_url or "disabled",
    )
    return uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
