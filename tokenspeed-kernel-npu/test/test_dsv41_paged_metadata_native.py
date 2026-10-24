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


"""Integer-address correctness, stable order, boundaries and graph replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops import dsv41_paged_metadata_native as native
from tokenspeed_kernel_npu.ops.dsv41_paged_metadata_native import prepare_indices


@pytest.fixture(autouse=True)
def device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    if native._library() is None:
        pytest.skip("Build dsv41_paged_metadata first")


@pytest.mark.parametrize(
    "n,w0,w1",
    [
        (1, 128, 0),
        (8, 128, 512),
        (192, 128, 512),
        (192, 128, 2048),
        (33, 64, 128),
        (192, 0, 512),
    ],
)
def test_indices_and_graph(n, w0, w1):
    width = w0 + w1
    g0 = (64, 128, 64000)
    g1 = (1000000000, 256, 128000)
    ast = torch.empty((n, w0 + 32), device="npu", dtype=torch.int32)
    bst = torch.empty((n, w1 + 32), device="npu", dtype=torch.int32)
    a = ast[:, :w0]
    b = bst[:, :w1]
    lst = torch.empty(n, 4, device="npu", dtype=torch.int32)
    la = lst[:, 0]
    lb = lst[:, 2]

    def fill(seed):
        torch.manual_seed(seed)
        ast.random_(-256, 65000)
        bst.random_(-256, 130000)
        la.random_(-8, w0 + 8)
        lb.random_(-8, w1 + 8)
        if n > 1:
            la[0] = 0
            lb[0] = 0
            la[1] = w0
            lb[1] = w1
        if w0 >= 8:
            a[:, 0] = -1
            a[:, 1] = 0
            a[:, 2] = 2**31 - 1
            a[:, 3] = 64000
            a[:, 4] = 63999
            a[:, 5] = 0
            a[:, 6] = -(2**31)
        if w1 >= 8:
            b[:, 0] = -1
            b[:, 1] = 0
            b[:, 2] = 2**31 - 1
            b[:, 3] = 128000
            b[:, 4] = 127999
            b[:, 5] = 0
            b[:, 6] = -(2**31)

    def check(result):
        assert result is not None
        out, counts = result
        torch.npu.synchronize()
        ref = torch.full((n, width), -1, dtype=torch.int32)
        cc = torch.zeros(n, dtype=torch.int32)
        ac, bc, lc0, lc1 = (
            a.cpu().long(),
            b.cpu().long(),
            la.cpu().long(),
            lb.cpu().long(),
        )
        for row in range(n):
            vals = []
            for src, length, g in [(ac[row], lc0[row], g0), (bc[row], lc1[row], g1)]:
                src = src[: max(0, min(len(src), int(length)))]
                ids = src[(src >= 0) & (src < g[2])]
                vals.extend((g[0] + ids // 64 * g[1] + ids % 64).tolist())
            ref[row, : len(vals)] = torch.tensor(vals, dtype=torch.int32)
            cc[row] = len(vals)
        assert torch.equal(out.cpu(), ref), (
            n,
            width,
            (out.cpu() != ref).nonzero()[:10],
            out.cpu()[0, :30],
            ref[0, :30],
        )
        assert torch.equal(counts.cpu(), cc), (counts.cpu(), cc)

    def run():
        return prepare_indices(a, la, b, lb, g0, g1, w1, 1001000000)

    fill(711)
    check(run())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    for seed in (133, 149, 277):
        fill(seed)
        graph.replay()
        check(result)


def test_exact_capacity_boundary_and_fallback():
    cap = 2**24
    slots = torch.zeros(2, 64, device="npu", dtype=torch.int32)
    slots[:, 0] = cap - 1
    slots[:, 1] = cap
    slots[:, 2] = -1
    lengths = torch.full((2,), 3, device="npu", dtype=torch.int32)
    geometry = (0, 64, cap)
    actual = prepare_indices(slots, lengths, slots, lengths, geometry, geometry, 0, cap)
    assert actual is not None
    ids, count = actual
    assert ids[:, 0].cpu().tolist() == [cap - 1, cap - 1]
    assert bool((ids[:, 1:] == -1).all())
    assert count.cpu().tolist() == [1, 1]
    assert (
        prepare_indices(
            slots.long(), lengths, slots, lengths, geometry, geometry, 0, cap
        )
        is None
    )
    assert (
        prepare_indices(
            slots[:, ::2], lengths, slots, lengths, geometry, geometry, 0, cap
        )
        is None
    )
    assert (
        prepare_indices(slots, lengths, slots, lengths, geometry, geometry, 1, cap)
        is None
    )
    assert (
        prepare_indices(
            slots, lengths, slots, lengths, (0, 64, cap + 64), geometry, 0, cap + 64
        )
        is None
    )
