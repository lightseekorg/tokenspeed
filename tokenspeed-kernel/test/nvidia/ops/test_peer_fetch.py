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
from tokenspeed_kernel.ops.communication._cuda.peer_fetch import (
    build_fetch_plan,
    launch_peer_fetch,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="peer fetch needs an NVIDIA GPU"
)


def _fetch(copies, src, dst, *, tile_bytes, ctas):
    base = torch.empty(2, dtype=torch.int64, device="cuda")
    plan = build_fetch_plan(
        copies, base=base.data_ptr(), tile_bytes=tile_bytes, device=dst.device
    )
    # The store copies leftovers with copy engines; plain byte copies here.
    for d, s, size in plan.leftovers:
        a, b = d - dst.data_ptr(), s - src.data_ptr()
        dst[a : a + size].copy_(src[b : b + size])
    launch_peer_fetch(
        plan,
        base=base,
        counter=torch.zeros(1, dtype=torch.int32, device="cuda"),
        ctas=ctas,
    )
    return plan


@pytest.mark.parametrize("ctas", [8, 24, 64])
def test_fetch_copies_every_range_exactly(ctas):
    torch.manual_seed(0)
    src = torch.randint(0, 256, (8 << 20,), dtype=torch.uint8, device="cuda")
    dst = torch.zeros(8 << 20, dtype=torch.uint8, device="cuda")
    # (destination offset, source offset, bytes): co-aligned multi-tile ranges,
    # unaligned heads and tails, sub-16-byte ranges, and a range whose source
    # and destination differ in 16-byte alignment.
    ranges = [
        (0, 1 << 20, 3 << 20),
        (3 << 20, 4 << 20, 70_000 + 9),
        ((3 << 20) + 70_016 + 3, 19, 5),
        ((4 << 20) + 5, (5 << 20) + 5, 1_000_003),
        ((6 << 20) + 1, (6 << 20) + 8, 300_001),
        ((7 << 20), 0, 16),
    ]
    copies = [(dst.data_ptr() + d, src.data_ptr() + s, n) for d, s, n in ranges]
    plan = _fetch(copies, src, dst, tile_bytes=64 << 10, ctas=ctas)
    expected = torch.zeros_like(dst)
    for d, s, n in ranges:
        expected[d : d + n] = src[s : s + n]
    torch.cuda.synchronize()
    assert torch.equal(dst, expected)
    assert plan.tiles.shape[0] > 48
    assert sum(n for _, _, n in plan.leftovers) < 400_000


def test_tiles_interleave_ranges():
    base = torch.empty(2, dtype=torch.int64, device="cuda")
    start = base.data_ptr()
    copies = [(start + 4096 * i, start + (1 << 30) + 4096 * i, 1024) for i in range(3)]
    plan = build_fetch_plan(copies, base=start, tile_bytes=512, device=base.device)
    sources = plan.tiles[:, 1].tolist()
    # Two tiles per range, alternating between the three ranges.
    assert [(s * 8 - (1 << 30)) // 4096 for s in sources] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("ctas", [0, 12])
def test_ctas_must_fill_whole_clusters(ctas):
    src = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    with pytest.raises(ValueError):
        _fetch(
            [(dst.data_ptr(), src.data_ptr(), 1 << 20)],
            src,
            dst,
            tile_bytes=64 << 10,
            ctas=ctas,
        )
