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


"""Exact LCM integer metadata with invalid pages and live native graph inputs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_window_metadata_native import (
    compressor_metadata,
    decode_window,
)


def slots(p, r, table, rows, pages):
    logical = p // rows
    if not table.numel():
        return torch.full_like(p, -1)
    page = table[
        r.clamp(0, table.shape[0] - 1), logical.clamp(0, table.shape[1] - 1)
    ].long()
    valid = (
        (p >= 0)
        & (r >= 0)
        & (r < table.shape[0])
        & (logical < table.shape[1])
        & (page >= 1)
        & (page < pages)
    )
    return torch.where(valid, page * rows + p % rows, -1)


def expected(p, r, t, pages):
    lengths = (p.cpu() + 1).clamp(0, 128).int()
    p, r, t = p.cpu().long(), r.cpu().long(), t.cpu()
    active = (p >= 0) & (p % 2 == 1)
    pp = torch.where(active, p - 1, -1)
    pr = torch.where(active, r, -1)
    prev = torch.full_like(p, -1)
    prev[1:] = torch.where(
        active[1:] & (r[1:] >= 0) & (r[:-1] == r[1:]) & (p[:-1] == pp[1:]),
        torch.arange(max(0, p.numel() - 1)),
        -1,
    )
    pair = [active, pp, pr, prev, slots(pp, pr, t, 2, pages), slots(p, r, t, 2, pages)]
    want = (p - 127).clamp_min(0)[:, None] + torch.arange(128)
    want = torch.where(want > p[:, None], -1, want)
    window = [
        slots(p, r, t, 64, pages),
        slots(want, r[:, None].expand_as(want), t, 64, pages).int(),
        lengths,
    ]
    return pair, window


@pytest.mark.parametrize(
    "n,width,dtype",
    [
        (0, 53, torch.int64),
        (1, 0, torch.int64),
        (8, 53, torch.int64),
        (40, 15937, torch.int64),
        (48, 15937, torch.int32),
        (49, 53, torch.int64),
        (192, 53, torch.int64),
        (257, 53, torch.int64),
        (512, 53, torch.int64),
    ],
)
def test_exact_native_replay(n, width, dtype):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(410)
    backing = torch.empty(n * 2, device="npu", dtype=dtype)
    p = backing[::2]
    r = torch.empty_like(backing)[::2]
    storage = torch.randint(-2, 11000, (8, width * 2), device="npu", dtype=dtype)
    table = storage[:, ::2]
    pages = 10003
    pair = [torch.empty(n, device="npu", dtype=torch.bool)] + [
        torch.empty(n, device="npu", dtype=torch.int64) for _ in range(5)
    ]
    window = [
        torch.empty(n, device="npu", dtype=torch.int64),
        torch.empty(n, 128, device="npu", dtype=torch.int32),
        torch.empty(n, device="npu", dtype=torch.int32),
    ]

    def run():
        assert compressor_metadata(p, r, table, pages, *pair)
        assert decode_window(p, r, *window, table, pages)

    p.zero_()
    r.zero_()
    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        run()
    for step in range(10):
        p.copy_(
            torch.arange(n, device="npu", dtype=dtype)
            + [0, 50000, 1019900, 2**30, 0, 0, 0, 0, 0, 0][step]
        )
        r.copy_(torch.arange(n, device="npu", dtype=dtype) // max(1, n // 8))
        if step == 4:
            p.fill_(-1)
        elif step == 5:
            r[::3] = -1
            p[::5] = -128
        elif step == 6:
            r[::2] = 10000
            if dtype == torch.int64:
                p[::4] = 2**40
                r[1::4] = 2**40
        elif step == 7:
            storage.random_(-2, 11000)
            if dtype == torch.int64 and width:
                table[:, ::7] = 2**40
        elif step == 8:
            p.fill_(torch.iinfo(dtype).max)
        elif step == 9:
            p.fill_(torch.iinfo(dtype).min)
        graph.replay()
        torch.npu.synchronize()
        refs = expected(p, r, table, pages)
        for outputs, reference in zip((pair, window), refs):
            for actual, ref in zip(outputs, reference):
                torch.testing.assert_close(actual.cpu(), ref, atol=0, rtol=0)
