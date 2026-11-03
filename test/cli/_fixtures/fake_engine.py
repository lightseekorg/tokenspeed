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

"""A CPU fake of the headless engine on SMG's direct ZMQ lane.

Run as ``ts serve``'s engine module (``TS_SERVE_ENGINE_MODULE``), it takes
the engine's command line, registers with the gateway's handshake as one
engine rank does (``ts-host``'s ``EngineTransport::connect``: HELLO, INIT,
READY, then the ready response on the input socket) and answers every
generate request in one finished output, so ``ts serve`` orchestration and
the gateway's request handling run without a GPU or model weights.
"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Callable

import msgspec
import zmq

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    BatchTokenIDOutSlim,
    TokenizedGenerateReqInput,
)

# The ids every request gets back unless the caller supplies a responder.
FAKE_OUTPUT_IDS = [11, 12, 13]


class _ReadyMessage(msgspec.Struct):
    """The engine's handshake status (``ts-wire``'s ``ReadyMessage``, a map)."""

    status: str
    local: bool = True
    headless: bool = True
    parallel_config_hash: str | None = None


def _register(context: zmq.Context, handshake: str, engine_index: int, ready):
    """Register as engine ``engine_index``; return its input and output
    sockets."""
    identity = engine_index.to_bytes(2, "little")
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, identity)
    dealer.connect(handshake)
    dealer.send(msgspec.msgpack.encode(_ReadyMessage(status="HELLO")))
    init = msgspec.msgpack.decode(dealer.recv_multipart()[-1])
    dealer.send(msgspec.msgpack.encode(_ReadyMessage(status="READY")))
    dealer.close()
    inputs = context.socket(zmq.DEALER)
    inputs.setsockopt(zmq.IDENTITY, identity)
    inputs.connect(init["addresses"]["inputs"][0])
    inputs.send(zmq_wire.encode(ready))
    outputs = context.socket(zmq.PUSH)
    outputs.connect(init["addresses"]["outputs"][0])
    return inputs, outputs


def serve(
    argv: list[str] | None = None,
    respond: Callable[[TokenizedGenerateReqInput], list[int]] | None = None,
    max_model_len: int = 2048,
) -> None:
    """Register with the gateway named by the engine command line ``argv``
    and answer requests until SIGTERM: each generate request finishes with
    ``respond(request)``'s ids (``FAKE_OUTPUT_IDS`` by default)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parallel-address", required=True)
    parser.add_argument("--data-parallel-rpc-port", type=int, required=True)
    parser.add_argument("--zmq-engine-index", type=int, required=True)
    parser.add_argument("--model", default="/fake")
    args, _ignored = parser.parse_known_args(argv)

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    context = zmq.Context()
    ready = zmq_wire.WireEngineCoreReadyResponse(
        max_model_len=max_model_len,
        num_gpu_blocks=64,
        prefix_granularity=16,
        max_num_seqs=8,
        instance_id=args.model,
    )
    inputs, outputs = _register(
        context,
        f"tcp://{args.data_parallel_address}:{args.data_parallel_rpc_port}",
        args.zmq_engine_index,
        ready,
    )
    poller = zmq.Poller()
    poller.register(inputs, zmq.POLLIN)
    while not stop:
        if not poller.poll(timeout=100):
            continue
        for request in zmq_wire.decode_request_frames(inputs.recv_multipart()):
            if not isinstance(request, TokenizedGenerateReqInput):
                continue
            output_ids = (respond or (lambda _: FAKE_OUTPUT_IDS))(request)
            batch = BatchTokenIDOutSlim(
                rids=[request.rid],
                output_ids=[output_ids],
                finished_reasons=["stop"],
                prompt_tokens=[len(request.input_ids)],
                completion_tokens=[len(output_ids)],
                cached_tokens=[0],
                output_token_logprobs_val=[[]],
                output_token_logprobs_idx=[[]],
                engine_index=args.zmq_engine_index,
                finish_reason_details=[{"type": "stop"}],
            )
            outputs.send(msgspec.msgpack.encode(batch))
    context.destroy(linger=0)


if __name__ == "__main__":
    serve()
