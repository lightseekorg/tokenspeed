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

"""Benchmark-only partition of an actual captured MTP decode graph.

Both scheduling choices execute identical nodes from the serving graph. The
original Torch graph stays alive to own its private allocation pool. Only the
GDN commit may run beside drafting; PLE/indexer commit and runtime bookkeeping
join the draft first. This is an experiment adapter, not a serving default.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from weakref import WeakKeyDictionary

import torch
from cuda.bindings import driver as cu


def checked(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA graph operation failed: {result}")
    return result[1:]


def graph_nodes(graph):
    _, count = checked(cu.cuGraphGetNodes(graph, 0))
    nodes, _ = checked(cu.cuGraphGetNodes(graph, count))
    return list(nodes)


class GraphPair:
    def __init__(
        self, graph: torch.cuda.CUDAGraph, markers: dict[int, torch.cuda.Event]
    ):
        self.owner = graph
        self.prefix = None
        self.suffix = None
        self.prefix_exec = None
        self.suffix_exec = None
        original = cu.CUgraph(graph.raw_cuda_graph())
        nodes = graph_nodes(original)
        handles = {int(node): node for node in nodes}
        cuts = []
        kinds = Counter()
        for node in nodes:
            (kind,) = checked(cu.cuGraphNodeGetType(node))
            kinds[str(kind)] += 1
            if kind == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EVENT_RECORD:
                (event,) = checked(cu.cuGraphEventRecordNodeGetEvent(node))
                if int(event) in markers:
                    cuts.append(int(node))
        if len(cuts) != 1:
            raise RuntimeError(
                f"Expected one MTP marker in decode graph, found {len(cuts)}"
            )
        _, _, _, count = checked(cu.cuGraphGetEdges(original, 0))
        sources, targets, _, _ = checked(cu.cuGraphGetEdges(original, count))
        parents = {int(node): [] for node in nodes}
        children = {int(node): [] for node in nodes}
        for source, target in zip(sources, targets):
            parents[int(target)].append(int(source))
            children[int(source)].append(int(target))
        before = {cuts[0]}
        stack = list(parents[cuts[0]])
        while stack:
            node = stack.pop()
            if node not in before:
                before.add(node)
                stack.extend(parents[node])
        after = set()
        stack = list(children[cuts[0]])
        while stack:
            node = stack.pop()
            if node not in after:
                after.add(node)
                stack.extend(children[node])
        independent = set(handles) - before - after
        # Assign branches unordered with the marker to the prefix. Since the
        # suffix is the complete descendant closure, no suffix can point back
        # into this prefix. All original nodes therefore execute exactly once.
        before |= independent
        if any(
            int(source) in after and int(target) in before
            for source, target in zip(sources, targets)
        ):
            raise RuntimeError("MTP graph partition has a backward dependency")
        if not after:
            raise RuntimeError("MTP suffix has no nodes")
        (self.prefix,) = checked(cu.cuGraphClone(original))
        (self.suffix,) = checked(cu.cuGraphClone(original))
        # Clone mappings are collected before removing nodes and their edges.
        prefix_remove = [
            checked(cu.cuGraphNodeFindInClone(handles[n], self.prefix))[0]
            for n in after
        ]
        suffix_remove = [
            checked(cu.cuGraphNodeFindInClone(handles[n], self.suffix))[0]
            for n in before
        ]
        for node in prefix_remove:
            checked(cu.cuGraphDestroyNode(node))
        for node in suffix_remove:
            checked(cu.cuGraphDestroyNode(node))
        (self.prefix_exec,) = checked(cu.cuGraphInstantiate(self.prefix, 0))
        (self.suffix_exec,) = checked(cu.cuGraphInstantiate(self.suffix, 0))
        self.metadata = {
            "total_nodes": len(nodes),
            "target_nodes": len(before),
            "draft_nodes": len(after),
            "node_kinds": dict(kinds),
            "independent_prefix_nodes": len(independent),
        }

    def launch_target(self, stream: torch.cuda.Stream):
        checked(cu.cuGraphLaunch(self.prefix_exec, cu.CUstream(stream.cuda_stream)))

    def launch_draft(self, stream: torch.cuda.Stream):
        checked(cu.cuGraphLaunch(self.suffix_exec, cu.CUstream(stream.cuda_stream)))

    def close(self):
        if self.prefix_exec is not None:
            checked(cu.cuGraphExecDestroy(self.prefix_exec))
            self.prefix_exec = None
        if self.suffix_exec is not None:
            checked(cu.cuGraphExecDestroy(self.suffix_exec))
            self.suffix_exec = None
        if self.prefix is not None:
            checked(cu.cuGraphDestroy(self.prefix))
            self.prefix = None
        if self.suffix is not None:
            checked(cu.cuGraphDestroy(self.suffix))
            self.suffix = None


class DraftState:
    def __init__(self, *, scheduling: str):
        if scheduling not in {"serial", "overlap"}:
            raise ValueError(f"Unknown MTP commit scheduling: {scheduling}")
        self.scheduling = scheduling
        self.markers: dict[int, torch.cuda.Event] = {}
        self.graphs: dict[int, GraphPair] = {}
        self.stream = torch.cuda.Stream()
        self.target_done = torch.cuda.Event()
        self.pending = False
        self.launch_stream: torch.cuda.Stream | None = None

    def mark(self):
        event = torch.cuda.Event(external=True)
        event.record()
        self.markers[event.cuda_event] = event

    def replay(self, graph, *, batch: int, graph_batch: int):
        if self.pending:
            raise RuntimeError(
                "MTP replay started before the preceding commit joined drafting"
            )
        key = id(graph)
        if key not in self.graphs:
            pair = GraphPair(graph, self.markers)
            self.graphs[key] = pair
            print(
                f"GDN_MTP_GRAPH_PARTITION {json.dumps(dict(batch=graph_batch,scheduling=self.scheduling,**pair.metadata))}",
                flush=True,
            )
        pair = self.graphs[key]
        stream = torch.cuda.current_stream()
        with torch.cuda.nvtx.range(f"mtp_target_graph bs={batch}"):
            pair.launch_target(stream)
        if self.scheduling == "overlap" and batch:
            self.target_done.record(stream)
            self.stream.wait_event(self.target_done)
            with torch.cuda.nvtx.range(f"mtp_draft_graph bs={batch}"):
                pair.launch_draft(self.stream)
            self.launch_stream = stream
            self.pending = True
        else:
            with torch.cuda.nvtx.range(f"mtp_draft_graph bs={batch}"):
                pair.launch_draft(stream)

    def join(self):
        if self.pending:
            self.launch_stream.wait_stream(self.stream)
            self.pending = False
            self.launch_stream = None


_STATES: WeakKeyDictionary = WeakKeyDictionary()
_INSTALLED = False


def install(*, scheduling: str):
    global _INSTALLED
    if _INSTALLED:
        return
    if scheduling not in {"serial", "overlap"}:
        raise ValueError("The benchmark requires explicit serial or overlap scheduling")
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaAttnBackend,
    )

    original_graph_init = torch.cuda.CUDAGraph.__init__
    original_init = Eagle.__init__
    original_run = Eagle.run
    original_commit = MambaAttnBackend.commit_state_after_acceptance

    @wraps(original_graph_init)
    def graph_init(self, *args, **kwargs):
        if not args:
            kwargs.setdefault("keep_graph", True)
        return original_graph_init(self, *args, **kwargs)

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _STATES[self] = DraftState(scheduling=scheduling)

    @wraps(original_run)
    def run(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            _STATES[self].mark()
        return original_run(self, *args, **kwargs)

    def replay(self, graph, *, bs, graph_bs):
        state = _STATES[self]
        if self.tree_spec is not None:
            raise RuntimeError("This overlap benchmark is for ordinary chain MTP")
        state.replay(graph, batch=bs, graph_batch=graph_bs)

    @wraps(original_commit)
    def commit(self, *args, **kwargs):
        original_commit(self, *args, **kwargs)
        # GDN has now been queued on the forward stream. Join before returning
        # to the remaining attention-state commits and before runtime frontier updates.
        for state in list(_STATES.values()):
            state.join()

    torch.cuda.CUDAGraph.__init__ = graph_init
    Eagle.__init__ = init
    Eagle.run = run
    Eagle.replay_decode_graph = replay
    MambaAttnBackend.commit_state_after_acceptance = commit
    _INSTALLED = True
