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

"""EAGLE-Harness device-chained rounds: a round's captured graphs launch one
another on the GPU, so the host launches depth 1 and never waits on the
round's decisions.

Each depth's graph holds a kernel, right after the depth's decision, that
tail-launches the round's next graph of its padded batch: depth ``d + 1``
while ``go[d + 1]`` is set, else the finalize graph, sampled or not. A tail
launch starts once the launching graph has completed, and the stream that
launched depth 1 runs its later work after the whole round. Depth 1 is an
ordinary graph the host launches; the later depths and the finalize are
instantiated for device launch, which allows kernel, memcpy and memset nodes
and fork/join edges, but no event nodes. A device launch skips torch's
``CUDAGraph.replay``, which advances the philox offsets of the torch
generators a graph captured: the round graphs draw from their own counters
(``eagle_harness_sample_children``) and must capture no torch random op. A
round reaching a graph its batch did not capture traps on the device (a sticky
CUDA error); the drafter refuses such a round on the host before launching it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.cuda.eagle_harness_chain import (
    eagle_harness_chain_destroy,
    eagle_harness_chain_instantiate,
    eagle_harness_chain_load,
    eagle_harness_chain_next,
)

__all__ = ["DeviceChain"]


class DeviceChain:
    """The device-launchable round graphs of each padded batch.

    Args:
        max_bs: The largest padded batch.
        last_depth: The deepest depth a round drafts (``tree_depth - 1``).
        device: The CUDA device the rounds run on.
    """

    def __init__(self, max_bs: int, last_depth: int, device: torch.device) -> None:
        self.last_depth = last_depth
        # Column d >= 2: depth d's exec; then finalize execs (unsampled, sampled); 0: not captured.
        self._execs = torch.zeros(
            (max_bs + 1, last_depth + 3), dtype=torch.int64, device=device
        )
        # Per padded batch: the exec handles it owns.
        self._handles: dict[int, list[int]] = {}
        # Loaded here: the first launch may be inside a graph capture.
        eagle_harness_chain_load()

    def end_depth(
        self, rows: int, depth: int, go: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        """Enqueue, inside depth ``depth``'s graph at padded batch ``rows``,
        the launch of the round's next graph, which starts once that graph
        has ended."""
        eagle_harness_chain_next(self._execs[rows], go, sampled, depth, self.last_depth)

    def instantiate(
        self,
        rows: int,
        depths: list[torch.cuda.CUDAGraph],
        finalize: list[torch.cuda.CUDAGraph],
        stream: torch.cuda.Stream,
    ) -> None:
        """Instantiate the graphs of padded batch ``rows`` (``keep_graph``
        captures) for device launch: ``depths`` from depth 2, ``finalize``
        not sampled then sampled (when captured), and publish their handles
        to its chain."""
        handles = [
            eagle_harness_chain_instantiate(graph.raw_cuda_graph(), stream.cuda_stream)
            for graph in (*depths, *finalize)
        ]
        row = [0] * (self.last_depth + 3)
        row[2 : 2 + len(depths)] = handles[: len(depths)]
        finalize_col = self.last_depth + 1
        row[finalize_col : finalize_col + len(finalize)] = handles[len(depths) :]
        self._execs[rows].copy_(torch.tensor(row, dtype=torch.int64))
        self._handles[rows] = handles

    def release(self) -> None:
        """Free every exec handle, once no round runs."""
        torch.cuda.synchronize()
        for handles in self._handles.values():
            for handle in handles:
                eagle_harness_chain_destroy(handle)
        self._handles.clear()
        self._execs.zero_()
