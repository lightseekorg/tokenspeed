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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.communication._cuda.peer_copy import (
    SPREAD_BYTES,
    CopyEngineWorker,
    SequenceFlags,
    build_copy_schedule,
)
from tokenspeed_kernel.platform import current_platform

needs_gpu = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="copy-engine worker needs an NVIDIA GPU"
)


def _copies():
    # Peer -> (destination, source, bytes): uneven sizes so peers run out of
    # chunks in different rounds.
    return {
        3: [(1_000_000, 50_000_000, 700_000), (9_000_000, 60_000_000, 5)],
        5: [(20_000_000, 70_000_000, 2_100_000)],
        6: [(30_000_000, 80_000_000, 524_288)],
    }


def test_schedule_covers_every_range_in_chunked_peer_rounds():
    ranks = 16
    chunk = SPREAD_BYTES // ranks
    dst, src, size = build_copy_schedule(_copies(), ranks=ranks, rank=2)
    entries = list(zip(dst.tolist(), src.tolist(), size.tolist()))
    assert all(0 < n <= chunk for _, _, n in entries)
    owner = {}
    for peer, items in _copies().items():
        for d, s, n in items:
            owner.update({(d + o, s + o): peer for o in range(0, n, chunk)})
    covered = {}
    for d, s, n in entries:
        covered.setdefault(owner[d, s], []).append((d, s, n))
    for peer, items in _copies().items():
        expected = sorted(
            (d + o, s + o, min(chunk, n - o))
            for d, s, n in items
            for o in range(0, n, chunk)
        )
        assert sorted(covered[peer]) == expected
    # Each round takes one chunk from every peer that still has one, in one fixed order.
    peers = [owner[d, s] for d, s, _ in entries]
    first_round = peers[:3]
    assert sorted(first_round) == [3, 5, 6]
    later = [p for p in peers[3:] if p != 6]
    assert later == [p for p in first_round if p != 6] * 2 + [5, 5]


def test_schedule_order_is_fixed_per_rank_and_differs_across_ranks():
    copies = {
        peer: [(peer << 32, (peer << 32) + (1 << 31), 1 << 20)] for peer in range(1, 16)
    }

    def order(rank):
        dst, _, _ = build_copy_schedule(copies, ranks=16, rank=rank)
        return [d >> 32 for d in dst.tolist()[:15]]

    assert order(0) == order(0)
    assert sorted(order(0)) == list(range(1, 16))
    assert len({tuple(order(rank)) for rank in range(16)}) > 1


@needs_gpu
def test_worker_copies_after_release_and_before_ready_readers():
    device = torch.device("cuda", torch.cuda.current_device())
    copy_stream, writer, reader = (torch.cuda.Stream(device=device) for _ in range(3))
    src = torch.zeros(3 << 20, dtype=torch.uint8, device=device)
    dst = torch.zeros_like(src)
    seen = torch.empty_like(src)
    flags = SequenceFlags(2, device)
    worker = CopyEngineWorker(device, copy_stream)
    try:
        schedule = build_copy_schedule(
            {1: [(dst.data_ptr(), src.data_ptr(), src.numel())]}, ranks=4, rank=0
        )
        worker.set_layer(7, schedule)
        torch.cuda.synchronize()
        # Every flag wait is enqueued after its write, as the store does.
        with torch.cuda.stream(writer):
            torch.cuda._sleep(200_000_000)
            src.fill_(42)
            flags.write(1, 1, writer)
        job = worker.fetch(
            7, flags, wait_index=1, wait_value=1, ready_index=0, ready_value=1
        )
        worker.wait_submitted(job)
        with torch.cuda.stream(reader):
            flags.wait(0, 1, reader)
            seen.copy_(dst)
        torch.cuda.synchronize()
        torch.testing.assert_close(seen, torch.full_like(seen, 42), rtol=0, atol=0)
    finally:
        worker.close()
