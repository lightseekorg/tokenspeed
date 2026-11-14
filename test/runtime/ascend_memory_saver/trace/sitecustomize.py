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

"""Lightweight rank-only graph and VMM lifecycle evidence for the benchmark."""

import ctypes
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path

PHASE = "before"
FIRST_REPLAYS = set()
GRAPH_POOL_POLICY = os.environ.get("TOKENSPEED_TEST_ACL_GRAPH_POLICY")
POISON_GRAPH_POOL = GRAPH_POOL_POLICY == "discard"
CYCLE = 0
RESUMED_STATS = None
CYCLE_UPDATES = set()


def record(event, **fields):
    row = {"event": event, "pid": os.getpid(), "phase": PHASE, **fields}
    line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    fd = os.open(
        os.environ["GSM8K_GRAPH_TRACE"], os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644
    )
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def trace_graph(name, method):
    @wraps(method)
    def wrapped(graph, *args, **kwargs):
        if name == "update":
            if RESUMED_STATS is not None and id(graph) not in CYCLE_UPDATES:
                if CYCLE_UPDATES:
                    # Another bucket already used the shared pool. Poison it
                    # again to test this bucket's first replay independently.
                    import torch

                    torch.npu.synchronize()
                    poison_graph_pool(RESUMED_STATS, poison_byte(), graph=id(graph))
                CYCLE_UPDATES.add(id(graph))
            return method(graph, *args, **kwargs)
        result = method(graph, *args, **kwargs)
        if name == "replay":
            key = PHASE, id(graph)
            if key not in FIRST_REPLAYS:
                FIRST_REPLAYS.add(key)
                record("first_replay", graph=id(graph), pool=list(graph.pool()))
        else:
            fields = {"graph": id(graph)}
            if name == "capture_end":
                fields["pool"] = list(graph.pool())
                if GRAPH_POOL_POLICY:
                    fields["fia_tasks"] = len(
                        graph.graph_dispatch_mode.graph_dispatch_records
                    )
            record(name, **fields)
        return result

    return wrapped


def poison_byte():
    # Include non-finite BF16 values and out-of-range indices.
    return (0xFF, 0x7F, 0xA5)[(CYCLE - 1) % 3]


def poison_graph_pool(stats, byte, *, graph=None):
    memset = ctypes.CDLL("libascendcl.so").aclrtMemset
    memset.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t)
    memset.restype = ctypes.c_int
    blocks = [block for block in stats["blocks"] if block["tag"] == "acl_graph"]
    if not blocks:
        raise RuntimeError("No managed ACL Graph allocations")
    for block in blocks:
        if block["policy"] != "discard" or not block["mapped"]:
            raise RuntimeError(f"Expected a remapped discard allocation: {block}")
        size = block["aligned_bytes"]
        status = memset(int(block["ptr"], 16), size, byte, size)
        if status:
            raise RuntimeError(f"aclrtMemset failed: {status}")
    record("graph_pool_poison", byte=byte, blocks=len(blocks), graph=graph)


def trace_transition(name, method):
    @wraps(method)
    def wrapped(adapter, tag=None):
        global PHASE, CYCLE, RESUMED_STATS
        before = adapter.stats()
        started = time.monotonic()
        result = method(adapter, tag)
        elapsed = time.monotonic() - started
        after = adapter.stats()
        record(name, tag=tag, before=before, after=after, seconds=elapsed)
        if name == "resume" and tag == "acl_graph":
            PHASE = "after"
            if GRAPH_POOL_POLICY:
                CYCLE += 1
                PHASE = f"cycle-{CYCLE}"
            if POISON_GRAPH_POOL:
                CYCLE_UPDATES.clear()
                RESUMED_STATS = after
                poison_graph_pool(after, poison_byte())
        return result

    return wrapped


class TracingLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        if module.__name__ == "torch_npu.npu.graphs":
            graph = module.NPUGraph
            names = ("capture_begin", "capture_end", "reset", "replay")
            if POISON_GRAPH_POOL:
                names += ("update",)
            for name in names:
                setattr(graph, name, trace_graph(name, getattr(graph, name)))
            record("instrumented", module=module.__name__)
        else:
            adapter = module.AscendMemorySaverAdapter
            for name in ("pause", "resume"):
                setattr(adapter, name, trace_transition(name, getattr(adapter, name)))


class TracingFinder(importlib.abc.MetaPathFinder):
    targets = {
        "torch_npu.npu.graphs",
        "tokenspeed.runtime.utils.ascend_memory_saver_adapter",
    }

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = TracingLoader(spec.loader)
        return spec


if os.environ.get("GSM8K_GRAPH_TRACE") and str(
    Path("/proc/self/exe").resolve()
) == os.environ.get("GSM8K_TRACE_HOST_EXE"):
    sys.meta_path.insert(0, TracingFinder())
