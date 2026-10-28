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

"""EAGLE-Harness device-chained rounds: a round's captured graphs choose one
another on the GPU, so the host launches depth 1 and never waits on the
round's decisions.

Each depth's graph holds a kernel, right after the depth's decision, that
chooses the round's next graph of its padded batch: depth ``d + 1`` while
``go[d + 1]`` is set, else the finalize graph, sampled or not. On CUDA the
kernel tail-launches it (device-side graph launch). HIP has no device-side
graph launch: the kernel posts its choice to host memory, where a native
launcher thread (per device, outside the GIL) launches it on the round's
stream. Either way the next graph starts once the choosing graph has
completed, and since a depth's decision runs at its start, the next graph is
chosen while the GPU still drafts the depth before it.

A round is begun (``begin``) once the host has queued its depth 1 on the
round's stream; nothing else is queued there until ``join`` returns (at once
on CUDA; on HIP once the finalize is queued, at most until the last decision
lands), and the stream then runs its later work after the whole round. Each
round is joined before the next one's work is queued there. Depth 1 is an ordinary graph the host launches; the later depths and
the finalize are instantiated for the chain (on CUDA for device launch, which
allows kernel, memcpy and memset nodes and fork/join edges, but no event
nodes). A chain launch skips torch's ``CUDAGraph.replay``, which advances the
philox offsets of the torch generators a graph captured: the round graphs draw
from their own counters (``eagle_harness_sample_children``) and must capture
no torch random op. A round reaching a graph its batch did not capture traps
on the device; the drafter refuses such a round on the host before launching
it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling.eagle_harness import (
    eagle_harness_chain_begin,
    eagle_harness_chain_destroy,
    eagle_harness_chain_instantiate,
    eagle_harness_chain_join,
    eagle_harness_chain_load,
    eagle_harness_chain_next,
)

__all__ = ["DeviceChain"]


class DeviceChain:
    """The chained round graphs of each padded batch.

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
        # The stream of the round begun and not yet joined (None: none).
        self._stream: torch.cuda.Stream | None = None
        # Loaded here: the first launch may be inside a graph capture.
        with torch.cuda.device(device):
            eagle_harness_chain_load()

    def begin(self, stream: torch.cuda.Stream) -> None:
        """Begin the round whose depth 1 the host has just queued on
        ``stream``: the chain launches the rest behind it."""
        eagle_harness_chain_begin(stream.cuda_stream)
        self._stream = stream

    def join(self) -> None:
        """Return once the begun round has queued its last graph on its
        stream, which then runs work queued from now on after the round."""
        stream, self._stream = self._stream, None
        if stream is not None:
            eagle_harness_chain_join(stream.cuda_stream)

    def end_depth(
        self, rows: int, depth: int, go: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        """Enqueue, inside depth ``depth``'s graph at padded batch ``rows``,
        the choice of the round's next graph, which starts once that graph
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
        captures) for the chain: ``depths`` from depth 2, ``finalize``
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
        """Free every exec handle, once no round is being launched."""
        self.join()
        torch.cuda.synchronize()
        for handles in self._handles.values():
            for handle in handles:
                eagle_harness_chain_destroy(handle)
        self._handles.clear()
        self._execs.zero_()
