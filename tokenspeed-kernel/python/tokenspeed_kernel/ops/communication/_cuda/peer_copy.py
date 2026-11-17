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

"""Copy-engine fetch of a fixed list of byte ranges, such as one layer's peer weights.

When every GPU pulls from all peers at once, copy-engine bandwidth depends on
how reads spread over the sources. Whole-shard copies reach about 420 GB/s per
GB300 at 16 ranks; one batch of small chunks, each round taking one chunk from
every peer in a fixed per-rank random order, reaches about 700 GB/s beside
compute. The chunk is ``SPREAD_BYTES`` divided by the number of ranks. A worker thread submits the
batch, since the call blocks while the copy stream's queue is full; GPU-side
sequence flags order fetches against readers. Enqueue a flag wait only after
its write is enqueued (``CopyEngineWorker.wait_submitted``): a wait ahead of its
producer blocks the GPU, and lazy module loading on any thread then deadlocks.
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

import torch

SPREAD_BYTES = 8 << 20


def build_copy_schedule(
    copies: dict[int, list[tuple[int, int, int]]], *, ranks: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interleave per-peer copies into one chunked batch order.

    Args:
        copies: Peer rank -> (destination, source, bytes) copies from that peer.
        ranks: Number of ranks pulling at once; sets the chunk size.
        rank: This rank; seeds its fixed peer order.

    Returns:
        CPU int64 tensors (destinations, sources, sizes), one entry per chunk.
    """
    chunk = SPREAD_BYTES // ranks
    queues = {
        peer: [
            (dst + offset, src + offset, min(chunk, size - offset))
            for dst, src, size in items
            for offset in range(0, size, chunk)
        ]
        for peer, items in copies.items()
    }
    order = sorted(queues)
    random.Random(rank).shuffle(order)
    rounds = max((len(q) for q in queues.values()), default=0)
    schedule = [
        queues[peer][r]
        for r in range(rounds)
        for peer in order
        if r < len(queues[peer])
    ]
    columns = list(zip(*schedule)) if schedule else [(), (), ()]
    return tuple(torch.tensor(column, dtype=torch.int64) for column in columns)


@cache
def load_peer_copy_module():
    """Compile the worker and flag operations; call at startup, before serving."""
    from tokenspeed_kernel.thirdparty.flashinfer.jit import build_cuda_module

    return build_cuda_module(
        "tokenspeed_peer_copy_v2", [Path(__file__).with_name("peer_copy.cu")]
    )


class SequenceFlags:
    """Monotonic 64-bit counters in device memory, waited on and written by streams."""

    def __init__(self, count: int, device: torch.device) -> None:
        self._module = load_peer_copy_module()
        self.values = torch.zeros(count, dtype=torch.int64, device=device)

    def wait(self, index: int, value: int, stream: torch.cuda.Stream) -> None:
        """Make ``stream`` wait until counter ``index`` reaches ``value``."""
        self._module.wait_flag(self.values, index, value, stream.cuda_stream)

    def write(self, index: int, value: int, stream: torch.cuda.Stream) -> None:
        """Set counter ``index`` to ``value`` after prior work on ``stream``."""
        self._module.write_flag(self.values, index, value, stream.cuda_stream)


class CopyEngineWorker:
    """Background submitter of per-layer copy batches on one copy stream."""

    def __init__(self, device: torch.device, stream: torch.cuda.Stream) -> None:
        self._module = load_peer_copy_module()
        self._handle: int | None = self._module.create_worker(
            device.index, stream.cuda_stream
        )

    def set_layer(
        self, layer: int, schedule: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        self._module.set_layer(self._handle, layer, *schedule)

    def fetch(
        self,
        layer: int,
        flags: SequenceFlags,
        *,
        wait_index: int,
        wait_value: int,
        ready_index: int,
        ready_value: int,
    ) -> int:
        """Queue: wait for the flag, copy the layer, then write the ready flag.

        The wait flag's write must already be enqueued. Returns the job number
        for ``wait_submitted``.
        """
        return self._module.fetch(
            self._handle,
            layer,
            flags.values,
            wait_index,
            wait_value,
            ready_index,
            ready_value,
        )

    def wait_submitted(self, job: int) -> None:
        """Block, without the GIL, until ``job``'s ready-flag write is enqueued."""
        self._module.wait_submitted(self._handle, job)

    def close(self) -> None:
        """Submit the queued fetches and stop the worker thread."""
        if self._handle is not None:
            self._module.destroy_worker(self._handle)
            self._handle = None
