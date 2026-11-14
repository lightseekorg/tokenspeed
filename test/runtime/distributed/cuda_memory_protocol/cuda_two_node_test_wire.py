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

"""Drive individual engines through the production wire across two hosts."""

import itertools
import json
import time
import uuid

import msgpack
import zmq


class EngineExited(RuntimeError):
    pass


class Wire:
    def __init__(self, host, *, base_port=28368):
        self.host = host
        self.base_port = base_port
        self.context = zmq.Context()
        self.counter = itertools.count(1)
        self.sockets = []
        for offset, kind in enumerate((zmq.ROUTER, zmq.ROUTER, zmq.PULL)):
            socket = self.context.socket(kind)
            socket.linger = 0
            if kind == zmq.ROUTER:
                socket.router_mandatory = 1
            socket.bind(f"tcp://{host}:{base_port + offset}")
            self.sockets.append(socket)
        self.handshake, self.input, self.output = self.sockets
        self.engines = {}

    def close(self):
        for socket in self.sockets:
            socket.close()
        self.context.term()

    @staticmethod
    def recv(socket, deadline):
        wait_ms = max(0, int((deadline - time.monotonic()) * 1000))
        if not socket.poll(wait_ms):
            raise TimeoutError("TokenSpeed wire receive timed out")
        return socket.recv_multipart()

    def handshake_init(self):
        return msgpack.packb(
            {
                "addresses": {
                    "inputs": [f"tcp://{self.host}:{self.base_port + 1}"],
                    "outputs": [f"tcp://{self.host}:{self.base_port + 2}"],
                }
            },
            use_bin_type=True,
        )

    @staticmethod
    def record(event, **fields):
        print(
            json.dumps({"time": time.monotonic(), "event": event, **fields}), flush=True
        )

    def send_control(self, engine, method, args):
        call = next(self.counter)
        identity = engine.to_bytes(2, "little")
        self.input.send_multipart(
            [identity, b"\x02", msgpack.packb([call, method, args])]
        )
        self.record("sent", call=call, engine=engine, method=method, args=args)
        return call

    def message(self, deadline):
        payload, *extra = self.recv(self.output, deadline)
        received_ns = time.monotonic_ns()
        if payload == b"ENGINE_CORE_DEAD":
            self.record("engine_dead")
            raise EngineExited("engine exited")
        assert not extra, extra
        return msgpack.unpackb(payload, raw=False), received_ns

    def check_reply(self, output, call, engine, *, success=True, message=None):
        kind, actual_call, ok, detail, actual_engine = output
        assert kind == "ControlReplySlim", output
        assert (actual_call, actual_engine) == (call, engine), output
        self.record("reply", call=call, engine=engine, success=ok, detail=detail)
        assert ok is success, output
        if message:
            assert message in detail, output

    def control(
        self, plan, *, success=True, delay=0, message=None, timeout=45, check_reply=None
    ):
        pending = {}
        start = time.monotonic()
        for engine, method, args in plan:
            if pending and delay:
                until = time.monotonic() + delay
                while time.monotonic() < until:
                    try:
                        output, _ = self.message(until)
                    except TimeoutError:
                        break
                    if output[0] == "ControlReplySlim":
                        raise AssertionError(f"Premature reply {output}")
            pending[self.send_control(engine, method, args)] = engine

        first_reply = None
        while pending:
            output, received_ns = self.message(start + timeout)
            if output[0] != "ControlReplySlim":
                continue
            call = output[1]
            if first_reply is None:
                first_reply = time.monotonic() - start
            self.check_reply(
                output, call, pending.pop(call), success=success, message=message
            )
            if check_reply is not None:
                check_reply(received_ns)
        return first_reply

    def all(self, method, args=(), **kwargs):
        return self.control(
            [(engine, method, list(args)) for engine in self.engines], **kwargs
        )

    def send_generate(self, engine, rid, prompt, max_tokens, **fields):
        from tokenspeed.runtime.engine.io_struct import (
            MsgpackEncoder,
            TokenizedGenerateReqInput,
        )
        from tokenspeed.runtime.sampling.sampling_params import SamplingParams

        request = TokenizedGenerateReqInput(
            rid=rid,
            input_ids=prompt,
            sampling_params=SamplingParams(
                max_new_tokens=max_tokens, temperature=0, ignore_eos=True, seed=17
            ),
            **fields,
        )
        frames = [bytes(frame) for frame in MsgpackEncoder().encode(request)]
        self.input.send_multipart([engine.to_bytes(2, "little"), b"\x00", *frames])

    def collect_generation(self, requests, max_tokens, deadline, *, on_control=None):
        outputs = {
            rid: (
                engine,
                [],
                self.engines[engine]["positional_layouts"]["BatchTokenIDOutSlim"],
            )
            for rid, engine in requests.items()
        }
        result = {}
        while outputs:
            frame, received_ns = self.message(deadline)
            if frame[0] == "ControlReplySlim" and on_control is not None:
                on_control(frame, received_ns)
            if frame[0] != "BatchTokenIDOutSlim" or not frame[1]:
                continue
            for rid in frame[1]:
                if rid not in outputs:
                    continue
                engine, tokens, columns = outputs[rid]
                batch = dict(zip(columns, frame[1:]))
                i = batch["rids"].index(rid)
                tokens.extend(batch["output_ids"][i])
                finish = batch["finished_reasons"][i]
                if finish:
                    self.record(
                        "generated",
                        base_port=self.base_port,
                        engine=engine,
                        rid=rid,
                        tokens=tokens,
                        finish=finish,
                        detail=batch["finish_reason_details"][i],
                    )
                    assert finish == "length" and len(tokens) == max_tokens, batch
                    result[engine] = tokens
                    del outputs[rid]
        return result

    def generate(self, prompt, max_tokens):
        requests = {uuid.uuid4().hex: engine for engine in self.engines}
        for rid, engine in requests.items():
            self.send_generate(engine, rid, prompt, max_tokens)
        return self.collect_generation(requests, max_tokens, time.monotonic() + 150)


def register_wires(*wire_counts):
    # Serve INIT for every role as soon as it dials. They can load at
    # different speeds, and each has a bounded wait for the frontend.
    poller = zmq.Poller()
    handshakes = {}
    inputs = {}
    for wire, count in wire_counts:
        poller.register(wire.handshake, zmq.POLLIN)
        poller.register(wire.input, zmq.POLLIN)
        handshakes[wire.handshake] = wire, set(), count
        inputs[wire.input] = wire, count

    deadline = time.monotonic() + 360
    while handshakes or inputs:
        ready = poller.poll(max(0, int((deadline - time.monotonic()) * 1000)))
        if not ready:
            raise TimeoutError("TokenSpeed wire registration timed out")
        for sock, _ in ready:
            identity, payload = sock.recv_multipart()
            message = msgpack.unpackb(payload, raw=False)
            if sock in handshakes:
                wire, identities, count = handshakes[sock]
                if message["status"] == "HELLO":
                    sock.send_multipart([identity, wire.handshake_init()])
                elif message["status"] == "READY":
                    identities.add(identity)
                    if len(identities) == count:
                        poller.unregister(sock)
                        del handshakes[sock]
                else:
                    raise ValueError(message)
            else:
                wire, count = inputs[sock]
                engine = int.from_bytes(identity, "little")
                wire.engines[engine] = message
                if len(wire.engines) == count:
                    wire.record(
                        "registered", base_port=wire.base_port, engines=wire.engines
                    )
                    poller.unregister(sock)
                    del inputs[sock]
