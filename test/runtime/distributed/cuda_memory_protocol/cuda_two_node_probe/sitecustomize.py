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

"""Trace the actual startup and memory-operation path of each test rank."""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path

TRACE = os.environ.get("TS_TEST_TOPOLOGY_TRACE")
HOST = os.environ.get("TOKENSPEED_HOST_BINARY")

if TRACE and HOST and Path("/proc/self/exe").resolve() == Path(HOST).resolve():
    rank = None

    def record(event, **fields):
        row = dict(
            event=event, rank=rank, pid=os.getpid(), ns=time.monotonic_ns(), **fields
        )
        fd = os.open(TRACE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(row) + "\n").encode())
        finally:
            os.close(fd)

    def attach(module):
        bootstrap = module.HostBootstrap
        original_init = bootstrap.__init__

        @wraps(original_init)
        def init(self, server_args, *args, **kwargs):
            global rank
            mapping = server_args.mapping
            rank = mapping.rank
            visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            record(
                "mapping",
                node=server_args.node_rank,
                world=list(mapping.world_group),
                visible=visible,
                local_rank=mapping.local_rank,
                tp=list(mapping.attn.tp_group),
                dp=list(mapping.attn.dp_group),
                ep=list(mapping.moe.ep_group),
                pp=list(mapping.pp_group),
            )
            return original_init(self, server_args, *args, **kwargs)

        bootstrap.__init__ = init

        original_prepare = module.prepare_bootstrap

        @wraps(original_prepare)
        def prepare(*args, **kwargs):
            capable, start = original_prepare(*args, **kwargs)

            @wraps(start)
            def traced_start(outbox, protocol, *args, **kwargs):
                result = start(outbox, protocol, *args, **kwargs)
                record("negotiated", protocol=protocol, capability=capable)
                return result

            return capable, traced_start

        module.prepare_bootstrap = prepare

        gather = bootstrap.collective_addresses

        @wraps(gather)
        def exchange(self, port):
            try:
                addresses = gather(self, port)
            except RuntimeError as error:
                record("rendezvous_refused", error=str(error))
                raise
            record("rendezvous_ok", addresses=addresses)
            return addresses

        bootstrap.collective_addresses = exchange
        original_ready = bootstrap.report_ready

        @wraps(original_ready)
        def ready(self):
            result = original_ready(self)
            record("service_ready")
            return result

        bootstrap.report_ready = ready

        def trace_apply(original, release):
            @wraps(original)
            def wrapped(self, tags):
                record("apply_begin", release=release, tags=tags)
                delay_rank = os.environ.get("TS_TEST_TOPOLOGY_DELAY_RANK")
                if release and delay_rank is not None and rank == int(delay_rank):
                    time.sleep(0.5)
                result = original(self, tags)
                record("apply_complete", release=release, tags=tags)
                return result

            return wrapped

        plane = module.HostDataPlane
        for name, release in (("release_memory", True), ("resume_memory", False)):
            setattr(plane, name, trace_apply(getattr(plane, name), release))

    class ProbeLoader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            return self.wrapped.create_module(spec)

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            attach(module)

    class ProbeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "tokenspeed.runtime.engine.rust_host":
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = ProbeLoader(spec.loader)
                return spec
            return None

    sys.meta_path.insert(0, ProbeFinder())
