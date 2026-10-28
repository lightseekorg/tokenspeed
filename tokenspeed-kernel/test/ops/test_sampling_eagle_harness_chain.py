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

"""EAGLE-Harness chained rounds: from an ordinary first graph, graphs captured
by torch and instantiated for the chain run the depths the go flags select
and then the finalize the sampled flag selects, and the launching stream
orders the work it queues after ``eagle_harness_chain_join`` after the whole
chain."""

import pytest
import torch
from tokenspeed_kernel.ops.sampling.eagle_harness import (
    eagle_harness_chain_begin,
    eagle_harness_chain_destroy,
    eagle_harness_chain_instantiate,
    eagle_harness_chain_join,
    eagle_harness_chain_load,
    eagle_harness_chain_next,
    eagle_harness_device_chain_available,
)

# CUDA chains by device-side graph launch, HIP (gfx950) by its launcher thread.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not eagle_harness_device_chain_available(),
    reason="the platform has no eagle_harness device-chain ops",
)

_LAST_DEPTH = 4
_FINALIZE = 100
# Long enough that the next graph is chosen while the posting graph still runs.
_SLEEP_CYCLES = 100_000


class _Rounds:
    """A chain of depth graphs whose depth ``d`` writes its decision
    ``go[d + 1] = plan[d + 1]`` on a side branch, as a policy does at the
    depth's start, chains on it, and then logs ``d``; finalize graphs log
    ``_FINALIZE`` (+ 1 when sampled)."""

    def __init__(self) -> None:
        device = torch.device("cuda")
        eagle_harness_chain_load()
        self.log = torch.zeros(256, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.plan = torch.zeros(_LAST_DEPTH + 1, dtype=torch.int32, device=device)
        self.go = torch.zeros(_LAST_DEPTH + 1, dtype=torch.int32, device=device)
        self.sampled = torch.zeros(1, dtype=torch.bool, device=device)
        self.execs = torch.zeros(_LAST_DEPTH + 3, dtype=torch.int64, device=device)
        side = torch.cuda.Stream()
        stream = torch.cuda.Stream()
        self.graphs: list[torch.cuda.CUDAGraph] = []
        with torch.cuda.stream(stream):
            for depth in range(1, _LAST_DEPTH + 1):
                # Depth 1 is an ordinary graph; the chain launches the rest.
                graph = torch.cuda.CUDAGraph(keep_graph=depth > 1)
                with torch.cuda.graph(graph, stream=stream):
                    side.wait_stream(stream)
                    with torch.cuda.stream(side):
                        if depth < _LAST_DEPTH:
                            self.go[depth + 1 : depth + 2].copy_(
                                self.plan[depth + 1 : depth + 2]
                            )
                        eagle_harness_chain_next(
                            self.execs, self.go, self.sampled, depth, _LAST_DEPTH
                        )
                    torch.cuda._sleep(_SLEEP_CYCLES)
                    self.note(depth)
                    stream.wait_stream(side)
                self.graphs.append(graph)
            for finalize in (_FINALIZE, _FINALIZE + 1):
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream):
                    self.note(finalize)
                self.graphs.append(graph)
        self.handles = [
            eagle_harness_chain_instantiate(graph.raw_cuda_graph(), stream.cuda_stream)
            for graph in self.graphs[1:]
        ]
        self.execs[2:].copy_(torch.tensor(self.handles))
        # Uploaded on the capture stream; the rounds launch from the current one.
        stream.synchronize()

    def note(self, tag: int) -> None:
        self.log.index_fill_(0, self.count, tag)
        self.count.add_(1)

    def launch(self, plan: list[int], draw: bool) -> None:
        """Queue one round on the current stream and begin it (no join)."""
        self.plan.copy_(torch.tensor([0, 0, *plan], dtype=torch.int32))
        self.sampled.fill_(draw)
        self.graphs[0].replay()
        eagle_harness_chain_begin(torch.cuda.current_stream().cuda_stream)

    def logged(self) -> list[int]:
        torch.cuda.synchronize()
        return self.log[: int(self.count)].tolist()

    def release(self) -> None:
        torch.cuda.synchronize()
        for handle in self.handles:
            eagle_harness_chain_destroy(handle)


def test_chain_follows_the_decisions_to_the_selected_finalize() -> None:
    rounds = _Rounds()
    stream = torch.cuda.current_stream().cuda_stream

    def run(plan: list[int], draw: bool) -> list[int]:
        rounds.count.zero_()
        rounds.launch(plan, draw)
        eagle_harness_chain_join(stream)
        # Ordered after the whole chain.
        rounds.note(-1)
        return rounds.logged()

    # plan[d + 1] (from depth 2) is depth d's decision; the last depth always finalizes.
    assert run([1, 1, 1], False) == [1, 2, 3, 4, _FINALIZE, -1]
    assert run([0, 1, 1], True) == [1, _FINALIZE + 1, -1]
    assert run([1, 0, 1], False) == [1, 2, _FINALIZE, -1]
    assert run([1, 1, 0], True) == [1, 2, 3, _FINALIZE + 1, -1]
    rounds.release()


def test_rounds_queue_back_to_back_without_a_host_sync() -> None:
    """A joined round's stream queues the next round behind it while it
    still runs: the rounds run in order."""
    rounds = _Rounds()
    stream = torch.cuda.current_stream().cuda_stream
    rounds.count.zero_()
    rounds.launch([1, 1, 1], False)
    eagle_harness_chain_join(stream)
    rounds.launch([1, 0, 1], True)
    eagle_harness_chain_join(stream)
    rounds.note(-1)
    assert rounds.logged() == [1, 2, 3, 4, _FINALIZE, 1, 2, _FINALIZE + 1, -1]
    rounds.release()
