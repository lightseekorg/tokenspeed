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

"""Lightweight FastAPI HTTP server exposing tokenspeed's AsyncLLM.

Provides an sglang-compatible HTTP surface for control-plane operations such as
memory occupation management (POST /release_memory_occupation,
POST /resume_memory_occupation) and cache flushing (POST /flush_cache).

This server is intended for direct use in tests, PD-disaggregated node
deployments, and any scenario that requires an HTTP endpoint backed by a
running AsyncLLM instance.  Production deployments use the SMG gRPC servicer
instead; see ``tokenspeed.cli.serve_smg``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def create_app(async_llm: AsyncLLM, server_args: ServerArgs) -> FastAPI:
    """Build the FastAPI application for a tokenspeed HTTP server node."""

    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        if async_llm.is_server_starting():
            return Response(status_code=503)
        return Response(status_code=200)

    @app.post("/flush_cache")
    async def flush_cache():
        await async_llm.flush_cache()
        return Response(status_code=200)

    @app.post("/release_memory_occupation")
    async def release_memory_occupation(request: dict | None = None):
        """Offload model weights, KV cache, and CUDA graphs to CPU via torch_memory_saver.

        Requires the server to have been started with ``--enable-memory-saver``.

        Optional JSON body: ``{"tags": ["weights", "kv_cache"]}``
        """
        from tokenspeed.runtime.engine.io_struct import ReleaseMemoryOccupationReqInput

        tags = (request or {}).get("tags") if request else None
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        await async_llm.release_memory_occupation(obj)
        return ORJSONResponse({"success": True})

    @app.post("/resume_memory_occupation")
    async def resume_memory_occupation(request: dict | None = None):
        """Restore previously offloaded memory regions to GPU.

        Requires the server to have been started with ``--enable-memory-saver``.

        Optional JSON body: ``{"tags": ["weights", "kv_cache"]}``
        """
        from tokenspeed.runtime.engine.io_struct import ResumeMemoryOccupationReqInput

        tags = (request or {}).get("tags") if request else None
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        await async_llm.resume_memory_occupation(obj)
        return ORJSONResponse({"success": True})

    return app


async def run_http_server(
    async_llm: AsyncLLM,
    server_args: ServerArgs,
    host: str,
    port: int,
) -> None:
    """Launch the uvicorn server on ``host:port``."""

    app = create_app(async_llm, server_args)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        loop="auto",
        log_config=None,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    logger.info("Starting tokenspeed HTTP server at %s:%s", host, port)
    await server.serve()
