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

"""EAGLE-Harness device-chained rounds: from an ordinary first graph, graphs
captured by torch and instantiated for device launch run the depths the go
flags select and then the finalize the sampled flag selects, and the
launching stream orders its later work after the whole chain."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.eagle_harness_chain import (
    eagle_harness_chain_destroy,
    eagle_harness_chain_instantiate,
    eagle_harness_chain_next,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="eagle_harness_chain CUDA kernel is NVIDIA-only",
)

_LAST_DEPTH = 4
_FINALIZE = 100


def test_chain_follows_the_go_flags_to_the_selected_finalize() -> None:
    device = torch.device("cuda")
    log = torch.zeros(64, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int64, device=device)
    go = torch.zeros(_LAST_DEPTH + 1, dtype=torch.int32, device=device)
    sampled = torch.zeros(1, dtype=torch.bool, device=device)
    execs = torch.zeros(_LAST_DEPTH + 3, dtype=torch.int64, device=device)
    side = torch.cuda.Stream()

    def note(tag: int) -> None:
        log.index_fill_(0, count, tag)
        count.add_(1)

    stream = torch.cuda.Stream()
    graphs = []
    with torch.cuda.stream(stream):
        for depth in range(1, _LAST_DEPTH + 1):
            # Depth 1 is an ordinary graph; the rest launch on the device.
            graph = torch.cuda.CUDAGraph(keep_graph=depth > 1)
            with torch.cuda.graph(graph, stream=stream):
                # The launch is enqueued on a side branch, as after a depth's decision.
                side.wait_stream(stream)
                with torch.cuda.stream(side):
                    eagle_harness_chain_next(execs, go, sampled, depth, _LAST_DEPTH)
                torch.cuda._sleep(100_000)
                note(depth)
                stream.wait_stream(side)
            graphs.append(graph)
        for finalize in (_FINALIZE, _FINALIZE + 1):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                note(finalize)
            graphs.append(graph)
    handles = [
        eagle_harness_chain_instantiate(graph.raw_cuda_graph(), stream.cuda_stream)
        for graph in graphs[1:]
    ]
    execs[2:].copy_(torch.tensor(handles))
    # Uploaded on the capture stream; the rounds launch from the default one.
    stream.synchronize()

    def run(go_flags: list[int], draw: bool) -> list[int]:
        go.copy_(torch.tensor([0, *go_flags], dtype=torch.int32))
        sampled.fill_(draw)
        count.zero_()
        graphs[0].replay()
        # Ordered after the whole chain.
        note(-1)
        torch.cuda.synchronize()
        return log[: int(count)].tolist()

    # go[d + 1] is read once depth d's graph ends; the last depth always finalizes.
    assert run([1, 1, 1, 1], False) == [1, 2, 3, 4, _FINALIZE, -1]
    assert run([1, 0, 1, 1], True) == [1, _FINALIZE + 1, -1]
    assert run([1, 1, 0, 1], False) == [1, 2, _FINALIZE, -1]
    torch.cuda.synchronize()
    for handle in handles:
        eagle_harness_chain_destroy(handle)
