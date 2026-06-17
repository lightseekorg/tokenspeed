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

"""TokenSpeed gRPC server entrypoint used by ``ts serve``.

This mirrors the bundled ``smg_grpc_servicer.tokenspeed`` entrypoint while
keeping TokenSpeed-specific transport policy in this repository. In
particular, long non-streaming generation requests can leave the gRPC
server-streaming response idle for minutes; the SMG client sends HTTP/2
keepalive pings during that interval, so the server must allow them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from typing import TYPE_CHECKING

import grpc

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)

_GRPC_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_GRPC_HARD_CEILING_BYTES = (1 << 31) - 1


def _grpc_max_message_bytes() -> int:
    """Return the configured gRPC message ceiling."""
    raw = os.getenv("TOKENSPEED_GRPC_MAX_MESSAGE_BYTES")
    if not raw:
        return _GRPC_DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=%r is not an int; falling back to %d",
            raw,
            _GRPC_DEFAULT_MAX_BYTES,
        )
        return _GRPC_DEFAULT_MAX_BYTES
    if value <= 0:
        logger.warning(
            "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=%d must be positive; falling back to %d",
            value,
            _GRPC_DEFAULT_MAX_BYTES,
        )
        return _GRPC_DEFAULT_MAX_BYTES
    if value > _GRPC_HARD_CEILING_BYTES:
        logger.warning(
            "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=%d exceeds gRPC ceiling %d; clamping",
            value,
            _GRPC_HARD_CEILING_BYTES,
        )
        return _GRPC_HARD_CEILING_BYTES
    return value


def _grpc_server_options(max_message_bytes: int) -> list[tuple[str, int | bool]]:
    """Build gRPC server options for long idle server-streaming requests."""
    return [
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        # Allow repeated client keepalive pings while a long non-streaming
        # generation request is still running but has not yielded a DATA frame.
        ("grpc.http2.max_pings_without_data", 0),
        # Disable strike-based GOAWAY enforcement for the same long-idle stream.
        ("grpc.http2.max_ping_strikes", 0),
        ("grpc.keepalive_permit_without_calls", True),
    ]


async def serve_grpc(server_args: ServerArgs) -> None:
    """Run the TokenSpeed gRPC server until a shutdown signal is received."""
    from grpc_health.v1 import health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc
    from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
    from smg_grpc_servicer.tokenspeed.health_servicer import (
        TokenSpeedHealthServicer,
    )
    from smg_grpc_servicer.tokenspeed.scheduler_launcher import launch_engine
    from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

    logger.info("Launching TokenSpeed scheduler + AsyncLLM...")
    async_llm, scheduler_info = launch_engine(server_args)

    max_message_bytes = _grpc_max_message_bytes()
    grpc_options = _grpc_server_options(max_message_bytes)
    logger.info("TokenSpeed gRPC server options: %s", grpc_options)
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=grpc_options,
    )

    health_servicer = TokenSpeedHealthServicer(
        async_llm=async_llm,
        scheduler_info=scheduler_info,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    servicer = TokenSpeedSchedulerServicer(
        async_llm=async_llm,
        server_args=server_args,
        scheduler_info=scheduler_info,
        health_servicer=health_servicer,
    )
    tokenspeed_scheduler_pb2_grpc.add_TokenSpeedSchedulerServicer_to_server(
        servicer, server
    )

    service_names = (
        tokenspeed_scheduler_pb2.DESCRIPTOR.services_by_name[
            "TokenSpeedScheduler"
        ].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{server_args.host}:{server_args.port}"
    server.add_insecure_port(listen_addr)
    logger.info("TokenSpeed gRPC server listening on %s", listen_addr)

    await server.start()

    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(server_args, health_servicer),
        daemon=True,
    )
    warmup_thread.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down TokenSpeed gRPC server")
        try:
            await servicer.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("servicer.shutdown() raised")
        await server.stop(5.0)
        if warmup_thread.is_alive():
            warmup_thread.join(timeout=5.0)


def _wait_and_warmup(server_args: ServerArgs, health_servicer) -> None:
    """Probe the gRPC server until it can generate one token, then set SERVING."""
    from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc
    from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

    if os.getenv("TOKENSPEED_SKIP_GRPC_WARMUP", "0").lower() in ("1", "true", "yes"):
        logger.info("TOKENSPEED_SKIP_GRPC_WARMUP=1; skipping warmup")
        health_servicer.set_serving()
        return

    warmup_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(
        server_args.host, server_args.host
    )
    grpc_url = f"{warmup_host}:{server_args.port}"
    max_message_bytes = _grpc_max_message_bytes()
    channel = grpc.insecure_channel(
        grpc_url,
        options=[
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ],
    )
    stub = tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel)

    deadline = time.time() + 180
    connected = False
    while time.time() < deadline:
        try:
            stub.GetModelInfo(
                tokenspeed_scheduler_pb2.GetModelInfoRequest(),
                timeout=5,
            )
            connected = True
            break
        except Exception as e:  # noqa: BLE001
            logger.debug("Warmup: GetModelInfo not ready yet: %s", e)
            time.sleep(1)

    if not connected:
        logger.error("TokenSpeed gRPC warmup failed: GetModelInfo never succeeded")
        channel.close()
        return

    warmup_ok = False
    try:
        warmup = tokenspeed_scheduler_pb2.GenerateRequest(
            request_id=f"WARMUP_{time.time()}",
            tokenized=tokenspeed_scheduler_pb2.TokenizedInput(
                input_ids=[0],
                original_text="warmup",
            ),
            sampling_params=tokenspeed_scheduler_pb2.SamplingParams(
                temperature=0.0,
                max_new_tokens=1,
            ),
            stream=False,
        )
        final = None
        for resp in stub.Generate(warmup, timeout=600):
            final = resp
        if final is None or not final.HasField("complete"):
            logger.warning("Warmup Generate returned no Complete frame (last=%r)", final)
        else:
            logger.info("Warmup generation succeeded")
            warmup_ok = True
    except Exception as e:  # noqa: BLE001
        logger.warning("TokenSpeed warmup failed: %s", e)
    finally:
        channel.close()

    if warmup_ok:
        health_servicer.set_serving()
        logger.info("TokenSpeed gRPC server is ready to serve")
    else:
        logger.error("TokenSpeed gRPC warmup did not produce a complete frame")


def main(argv: list[str] | None = None) -> None:
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    server_args = prepare_server_args(argv)

    try:
        import uvloop
    except ImportError:
        uvloop = None
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    asyncio.run(serve_grpc(server_args))


if __name__ == "__main__":
    main()
