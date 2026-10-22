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


import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops import dsv41_index_metadata_native as native
from tokenspeed_kernel_npu.ops.dsv41_index_native import native_index


@pytest.fixture(autouse=True)
def npu_device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    if native._library() is None:
        pytest.skip("Build optional dsv41_index_metadata kernel")


def candidate(table, visible, candidates, pages):
    n, width = table.shape
    count = width if candidates is None else candidates.shape[1]
    stride = (count + 15) // 16 * 16
    m = torch.full((n, stride), -1, device=table.device, dtype=torch.int32)
    nt = torch.zeros_like(m)
    counts = torch.empty((n, 16), device=table.device, dtype=torch.int32)
    assert native.compact_index_metadata(
        table, visible, candidates, pages, m, nt, counts
    )
    return m, nt, counts


def oracle(table, visible, candidates, pages):
    table = table.cpu().long()
    visible = visible.cpu().long()
    candidates = None if candidates is None else candidates.cpu().long()
    n, width = table.shape
    count = width if candidates is None else candidates.shape[1]
    stride = (count + 15) // 16 * 16
    mapping = torch.full((n, stride), -1, dtype=torch.int32)
    nt = torch.zeros_like(mapping)
    counts = torch.empty(n, 16, dtype=torch.int32)
    for i in range(n):
        v = max(0, min(int(visible[i]), width * 64))
        unit = 64 if candidates is None else 8
        ids = torch.arange((v + 63) // 64) if candidates is None else candidates[i]
        valid = (ids >= 0) & (ids * unit < v)
        pages_i = table[i, (ids * unit // 64).clamp(0, width - 1)]
        valid &= (pages_i > 0) & (pages_i < pages)
        selected = ids[valid]
        mapping[i, : len(selected)] = selected.int()
        if candidates is None:
            nt[i, : len(selected)] = pages_i[valid].int()
        counts[i] = int((v - selected * unit).clamp(0, unit).sum())
    return mapping, nt, counts


def check(actual, expected):
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "n,width,count",
    [
        (1, 64, -1),
        (32, 1024, -1),
        (192, 15936, -1),
        (32, 15936, 2048),
        (192, 15936, 2048),
        (3, 256, 128),
        (3, 16384, -1),
        (48, 1293, -1),
        (48, 1293, 2048),
        (7, 17, 63),
        (1, 1, 1),
        (3, 1294, 2049),
    ],
)
def test_metadata_integer_reference_and_mutable_graph(n, width, count):
    torch.manual_seed(918)
    pages = 20000
    t = torch.randint(-2, pages + 2, (n, width + 8), dtype=torch.int32, device="npu")[
        :, :width
    ]
    v = torch.randint(-4, width * 64, (n * 2,), dtype=torch.int32, device="npu")[::2]
    c = (
        None
        if count < 0
        else torch.randint(
            -3, min(width * 8, 7000), (n, count + 8), dtype=torch.int32, device="npu"
        )[:, :count]
    )
    if c is not None:
        c[:, ::13] = -1
        c[:, 0] = -(2**31)
        if count > 1:
            c[:, 1] = 2**31 - 1
    check(candidate(t, v, c, pages), oracle(t, v, c, pages))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = candidate(t, v, c, pages)
    for step in range(3):
        if step == 0:
            v.zero_()
        elif step == 1:
            v.fill_(width * 64 - 5)
        else:
            t[:, ::3] = 0
            v.fill_(min(51198, width * 64))
        graph.replay()
        check(out, oracle(t, v, c, pages))


@pytest.mark.parametrize("with_candidates", [False, True])
@pytest.mark.parametrize("width", [64, 65, 1293])
def test_native_index_selection_matches_existing_metadata(
    monkeypatch, with_candidates, width
):
    torch.manual_seed(144)
    n, pages = 6, width + 1
    q = torch.randn(n, 32, 128, device="npu", dtype=torch.bfloat16)
    w = torch.randn(n, 32, device="npu", dtype=torch.bfloat16)
    cache = torch.randn(pages, 64, 128, device="npu", dtype=torch.bfloat16)
    table = torch.arange(1, width + 1, device="npu", dtype=torch.int32).repeat(n, 1)
    visible = torch.full((n,), 4093, device="npu", dtype=torch.int32)
    cb = (
        torch.arange(512, device="npu", dtype=torch.int32).repeat(n, 1)
        if with_candidates
        else None
    )
    run = lambda: native_index(q, w, cache, table, visible, cb)
    actual = run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    with monkeypatch.context() as patch:
        patch.setattr(native, "compact_index_metadata", lambda *args: False)
        expected = run()
        check(actual, tuple(x.cpu() for x in expected))
        for step in range(3):
            q.copy_(torch.randn_like(q))
            visible.fill_(4093 - step * 65)
            graph.replay()
            expected = run()
            check(captured, tuple(x.cpu() for x in expected))


def test_unaligned_row_and_base_addresses():
    torch.manual_seed(114)
    n, width, count, pages = 7, 1293, 2047, 2000
    table = torch.randint(
        -2, pages + 2, (n, width + 3), device="npu", dtype=torch.int32
    )[:, 1 : width + 1]
    visible = torch.full((n,), 51203, device="npu", dtype=torch.int32)
    candidates = torch.randint(
        -2, 7000, (n, count + 4), device="npu", dtype=torch.int32
    )[:, 1 : count + 1]
    for cb in (None, candidates):
        check(candidate(table, visible, cb, pages), oracle(table, visible, cb, pages))
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = candidate(table, visible, cb, pages)
        for step in range(3):
            table[:, ::5] = step
            visible.fill_(51203 - step * 17)
            graph.replay()
            check(out, oracle(table, visible, cb, pages))


def test_single_partial_candidate_is_not_padded_with_valid_zero_ids():
    table = torch.ones((1, 1), device="npu", dtype=torch.int32)
    visible = torch.tensor([7], device="npu", dtype=torch.int32)
    ids = torch.zeros((1, 1), device="npu", dtype=torch.int32)
    check(candidate(table, visible, ids, 2), oracle(table, visible, ids, 2))
