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


"""Exact metadata mapping and live graph replay, independently checked on CPU."""

import os

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel_npu.ops.dsv41_metadata import global_slots


def setup_module():
    torch.npu.set_device(int(os.environ.get("V41_TEST_DEVICE", "0")))
    torch.set_num_threads(4)


def oracle(rows, positions, requests, table, ratio, pages):
    r, p, q, t = [x.cpu().long() for x in (rows, positions, requests, table)]
    col = r // 64
    valid = (r >= 0) & (r < ((p[:, None] + 1).clamp_min(0) // ratio))
    valid &= (q[:, None] >= 0) & (q[:, None] < t.shape[0]) & (col < t.shape[1])
    if not t.numel():
        return torch.full(r.shape, -1, dtype=torch.int32)
    page = t[q.clamp(0, t.shape[0] - 1)[:, None], col.clamp(0, t.shape[1] - 1)]
    valid &= (page > 0) & (page < pages)
    return torch.where(valid, page * 64 + r % 64, -1).int()


@pytest.mark.parametrize(
    "n,k,ratio,strided,dtype",
    [
        (0, 512, 4, False, torch.int32),
        (3, 0, 128, False, torch.int64),
        (1, 17, 4, True, torch.int64),
        (32, 512, 4, False, torch.int32),
        (192, 512, 128, True, torch.int64),
        (192, 16384, 4, False, torch.int32),
        (160, 513, 4, True, torch.int64),
    ],
)
def test_mapping_and_replay(n, k, ratio, strided, dtype):
    torch.manual_seed(123)
    rows = torch.randint(
        -100, 14000, (n, k * (2 if strided else 1)), device="npu", dtype=dtype
    )
    if strided:
        rows = rows[:, ::2]
    pos = torch.randint(-10, 52000, (n,), device="npu", dtype=dtype)
    req = torch.randint(-2, 34, (n,), device="npu", dtype=dtype)
    table = torch.randint(-1, 25000, (32, 405 * 2), device="npu", dtype=torch.int32)
    if strided:
        table = table[:, ::2]

    def check(actual):
        torch.npu.synchronize()
        assert torch.equal(actual.cpu(), oracle(rows, pos, req, table, ratio, 24000))

    check(global_slots(rows, pos, req, table, ratio, 24000))
    check(dsv41.global_slots(rows, pos, req, table, ratio, 24000))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = global_slots(rows, pos, req, table, ratio, 24000)
    for _ in range(3):
        rows.copy_(torch.randint(-100, 14000, rows.shape, device="npu", dtype=dtype))
        table.copy_(
            torch.randint(-2, 25000, table.shape, device="npu", dtype=torch.int32)
        )
        pos.copy_(torch.randint(-10, 52000, (n,), device="npu", dtype=dtype))
        req.copy_(torch.randint(-2, 34, (n,), device="npu", dtype=dtype))
        graph.replay()
        check(out)


@pytest.mark.parametrize("shape", [(0, 10), (32, 0)])
def test_empty_table(shape):
    rows = torch.tensor([[0, 1, -1, 2**40]], dtype=torch.int64, device="npu")
    p = torch.tensor([50000], device="npu")
    r = torch.tensor([0], device="npu")
    table = torch.empty(shape, dtype=torch.int32, device="npu")
    assert torch.equal(
        global_slots(rows, p, r, table, 4, 100).cpu(),
        torch.full((1, 4), -1, dtype=torch.int32),
    )


def test_large_invalid_ids_and_pages():
    rows = torch.tensor(
        [[0, 63, 64, 127, 2**40, -(2**40)]], device="npu", dtype=torch.int64
    )
    pos = torch.tensor([50000], device="npu", dtype=torch.int64)
    req = torch.tensor([0], device="npu", dtype=torch.int64)
    table = torch.tensor([[3, 100, -1]], device="npu", dtype=torch.int32)
    out = global_slots(rows, pos, req, table, 4, 100)
    assert torch.equal(out.cpu(), oracle(rows, pos, req, table, 4, 100))
