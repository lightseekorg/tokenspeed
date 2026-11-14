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

"""Read-only Python call-boundary evidence, distinct from native wire ACKs."""

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
import uuid

INSTRUMENTED = set()
INSTALLED = False
TARGETS = {
    "tokenspeed.runtime.engine.rust_host",
    "tokenspeed.runtime.entrypoints.wire_driver",
}


def record(event, **fields):
    path = os.environ.get("TS_VALIDATION_TIMELINE")
    if not path:
        return
    entry = {
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, (json.dumps(entry, default=repr) + "\n").encode())
    finally:
        os.close(fd)


class FrontendProbe:
    def __init__(self, frontend):
        self.frontend = frontend
        self.sequence = 0
        self.ingress_id = uuid.uuid4().hex
        self.seen_tokens = set()

    def __getattr__(self, name):
        return getattr(self.frontend, name)

    async def control(self, requests):
        from tokenspeed.runtime.engine import zmq_wire

        self.sequence += 1
        sequence = f"{self.ingress_id}:{self.sequence}"
        decoded = [
            zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_CONTROL, bytes(x)])[0]
            for x in requests
        ]
        fields = [
            {
                "type": type(x).__name__,
                "rid": getattr(x, "rid", None),
                "tags": getattr(x, "tags", None),
            }
            for x in decoded
        ]
        record(
            "public_request",
            sequence=sequence,
            requests=fields,
            expected_ingress_replies=1,
        )
        try:
            replies = await self.frontend.control(requests)
        except BaseException as error:
            record("public_error", sequence=sequence, error=repr(error))
            raise
        outputs = [zmq_wire.decode_control_output(x[0]) for x in replies]
        record(
            "public_reply",
            sequence=sequence,
            requests=fields,
            count=len(replies),
            outputs=[
                {
                    "type": type(x).__name__,
                    "success": getattr(x, "success", None),
                    "message": getattr(x, "message", None),
                }
                for x in outputs
            ],
        )
        return replies

    async def next_outputs(self):
        chunks = await self.frontend.next_outputs()
        for rid, chunk in chunks:
            if (
                isinstance(chunk, dict)
                and chunk.get("output_ids")
                and rid not in self.seen_tokens
            ):
                self.seen_tokens.add(rid)
                record(
                    "first_token_observed",
                    request_id=rid,
                    first_chunk_tokens=len(chunk["output_ids"]),
                )
        return chunks


def instrument(module):
    if module.__name__ in INSTRUMENTED:
        return
    INSTRUMENTED.add(module.__name__)
    if module.__name__.endswith("wire_driver"):
        init = module.WireDriver.__init__

        @functools.wraps(init)
        def wrapped(self, *args, **kwargs):
            init(self, *args, **kwargs)
            from pathlib import Path

            starttime = (
                Path(f"/proc/{self.process.pid}/stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19]
            )
            record(
                "launcher",
                launcher_pid=self.process.pid,
                launcher_starttime=starttime,
                ingress_count=len(self.engines()),
            )
            self.frontend = FrontendProbe(self.frontend)

        module.WireDriver.__init__ = wrapped
        return
    plane = module.HostDataPlane
    for name in (
        "check_memory_transition",
        "release_memory",
        "resume_memory",
        "check_weight_update",
        "update_weights",
        "commit_weight_update",
    ):
        original = getattr(plane, name)

        def wrap(method, operation):
            @functools.wraps(method)
            def observed(self, *args, **kwargs):
                rank = module.torch.distributed.get_rank()
                fields = {"rank": rank, "operation": operation}
                if operation in (
                    "release_memory",
                    "resume_memory",
                    "check_memory_transition",
                ):
                    fields["tags"] = args[0]
                if operation == "check_memory_transition":
                    fields["release"] = args[1] if len(args) > 1 else kwargs["release"]
                if operation == "commit_weight_update":
                    fields["commit_success"] = args[0]
                record("rank_apply_begin", **fields)
                try:
                    result = method(self, *args, **kwargs)
                except BaseException as error:
                    record("rank_apply_error", **fields, error=repr(error))
                    raise
                record(
                    "rank_apply_complete",
                    **fields,
                    result=result,
                    evidence_scope="Python return immediately before native completion reply; not wire ACK capture",
                )
                return result

            return observed

        setattr(plane, name, wrap(original, name))


class Loader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        instrument(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


def install():
    global INSTALLED
    if not INSTALLED:
        sys.meta_path.insert(0, Finder())
        INSTALLED = True
    for name in TARGETS:
        if name in sys.modules:
            instrument(sys.modules[name])
