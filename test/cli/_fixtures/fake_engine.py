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

"""CPU engine fixture with native control RPCs and the SMG ZMQ handshake."""

from __future__ import annotations

import argparse
import asyncio
import signal
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


class _FakeHealth(health_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

    def Watch(self, request, context):
        yield health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )


def _add_tokenspeed_servicer(server) -> None:
    """Register a no-op TokenSpeedScheduler servicer for readiness + shutdown tests."""
    from smg_grpc_proto import tokenspeed_scheduler_pb2 as pb
    from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc as pbg
    from smg_grpc_proto.generated import common_pb2

    class _Servicer(pbg.TokenSpeedSchedulerServicer):
        def GetModelInfo(self, request, context):
            return pb.GetModelInfoResponse(
                model_path="/fake",
                tokenizer_path="/fake",
                served_model_name="fake-model",
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                max_context_length=2048,
                max_req_input_len=2048,
                vocab_size=32000,
                eos_token_ids=[2],
                pad_token_id=0,
                bos_token_id=1,
                weight_version="fake-v0",
                default_sampling_params_json="",
            )

        def HealthCheck(self, request, context):
            return pb.HealthCheckResponse(healthy=True, message="fake engine ok")

        def GetServerInfo(self, request, context):
            return pb.GetServerInfoResponse()

        def GetLoads(self, request, context):
            return pb.GetLoadsResponse()

        def FlushCache(self, request, context):
            return common_pb2.FlushCacheResponse(success=True, message="flushed")

    pbg.add_TokenSpeedSchedulerServicer_to_server(_Servicer(), server)


async def _connect_zmq(address: str):
    import msgspec
    import zmq
    import zmq.asyncio

    context = zmq.asyncio.Context()
    handshake = context.socket(zmq.DEALER)
    incoming = context.socket(zmq.DEALER)
    outgoing = context.socket(zmq.PUSH)
    sockets = (handshake, incoming, outgoing)
    try:
        handshake.setsockopt(zmq.IDENTITY, b"\x00\x00")
        handshake.connect(address)
        await handshake.send(
            msgspec.msgpack.encode({"status": "HELLO", "local": True, "headless": True})
        )
        init = msgspec.msgpack.decode(await handshake.recv())
        await handshake.send(
            msgspec.msgpack.encode({"status": "READY", "local": True, "headless": True})
        )
        incoming.setsockopt(zmq.IDENTITY, b"\x00\x00")
        incoming.connect(init["addresses"]["inputs"][0])
        outgoing.connect(init["addresses"]["outputs"][0])
        await incoming.send(
            msgspec.msgpack.encode(
                {
                    "max_model_len": 2048,
                    "num_gpu_blocks": 16,
                    "prefix_granularity": 1,
                    "dtype": "bfloat16",
                    "world_size": 1,
                    "data_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "decode_context_parallel_size": 1,
                    "data_parallel_rank": 0,
                    "max_num_seqs": 16,
                    "max_num_batched_tokens": 2048,
                    "instance_id": "fake-zmq",
                    "vllm_version": "",
                }
            )
        )
        return context, sockets
    except BaseException:
        for sock in sockets:
            sock.close(linger=0)
        context.term()
        raise


async def _serve(host: str, port: int, handshake_address: str | None) -> None:
    transport = await _connect_zmq(handshake_address) if handshake_address else None
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=4))
    health_pb2_grpc.add_HealthServicer_to_server(_FakeHealth(), server)
    _add_tokenspeed_servicer(server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await server.stop(grace=2.0)
        if transport is not None:
            context, sockets = transport
            for sock in sockets:
                sock.close(linger=0)
            context.term()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--zmq-msgpack", action="store_true")
    parser.add_argument("--data-parallel-rpc-port", type=int)
    args, _ignored = parser.parse_known_args(argv)
    handshake_address = (
        f"tcp://127.0.0.1:{args.data_parallel_rpc_port}" if args.zmq_msgpack else None
    )
    asyncio.run(_serve(args.host, args.port, handshake_address))


if __name__ == "__main__":
    main()
