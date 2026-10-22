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
from tokenspeed_kernel_npu.ops.dsv41_index_scan import _index_score_scan_kernel
from tokenspeed_kernel_npu.ops.dsv41_index_scan_native import scan_scores
from tokenspeed_kernel_npu.ops.dsv41_index_select import select_full_scores


def inputs(n, width):
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(421)
    q = (torch.randint(-8, 9, (n, 32, 128), dtype=torch.int32).float() / 16).bfloat16()
    w = (torch.randint(-8, 9, (n, 32), dtype=torch.int32).float() / 64).bfloat16()
    cache = (
        torch.randint(-8, 9, (65, 64, 128), dtype=torch.int32).float() / 16
    ).bfloat16()
    table = (
        torch.randint(1, 65, ((n + 5) // 6, width // 64), dtype=torch.int32)
        .repeat_interleave(6, 0)[:n]
        .clone()
    )
    return tuple(t.npu() for t in (q, w, cache, table))


def run(q, w, cache, table, visible, out, shards, new):
    n = q.shape[0]
    width = out.shape[1]
    if new:
        scan_scores(q, w, cache, table, visible, out, parts=4, shards=shards)
    else:
        _index_score_scan_kernel[n, 16](
            q,
            w,
            cache,
            table,
            visible,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            w.stride(0),
            w.stride(1),
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            table.stride(0),
            table.stride(1),
            visible.stride(0),
            width,
            cache.shape[0],
            table.shape[1],
            32,
            32 // shards,
            shards,
            32,
            16,
            64,
            64,
            enable_fp_fusion=False,
            multibuffer=False,
        )


def reference(q, w, cache, table, visible, shards):
    q, w, cache, table, visible = [t.cpu() for t in (q, w, cache, table, visible)]
    n = q.shape[0]
    width = table.shape[1] * 64
    out = torch.full((n, width), -torch.inf)
    for row in range(n):
        count = min(max(int(visible[row]), 0), width)
        cols = torch.arange(count)
        pages = table[row, cols // 64].long()
        valid = (pages > 0) & (pages < cache.shape[0])
        keys = cache[pages.clamp(0, cache.shape[0] - 1), cols % 64]
        dots = (q[row].float() @ keys.float().T).bfloat16().float()
        products = (dots.clamp_min(0) * w[row, :, None].float()).bfloat16().float()
        value = torch.zeros(count)
        for part in products.chunk(shards, dim=0):
            value += part.sum(0).bfloat16().float()
        out[row, :count] = value.bfloat16().float().masked_fill(~valid, -torch.inf)
    return out


@pytest.mark.parametrize("n,shards", [(1, 1), (7, 1), (12, 2), (12, 16)])
def test_exact_and_replay(n, shards):
    width = 1088
    q, w, cache, table = inputs(n, width)
    visible = torch.tensor(
        ([1027, 1029, 0, -1, 65, 1028] * ((n + 5) // 6))[:n],
        device="npu",
        dtype=torch.int32,
    )
    table[:, 2] = 0
    table[:, 3] = -1
    table[:, 4] = cache.shape[0]
    out = torch.full((n, width), -torch.inf, device="npu")
    old = out.clone()

    def check():
        run(q, w, cache, table, visible, old, shards, False)
        torch.testing.assert_close(out, old, atol=0, rtol=0)
        torch.testing.assert_close(
            out.cpu(), reference(q, w, cache, table, visible, shards), atol=0, rtol=0
        )

    run(q, w, cache, table, visible, out, shards, True)
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run(q, w, cache, table, visible, out, shards, True)
    # Change both common-page and divergent-page branches, including odd tail.
    for phase in range(3):
        if n > 1:
            table[1, 6] = table[0, 6] if phase % 2 else (table[0, 6] % 64) + 1
        visible.fill_(width - 3 - phase)
        q.mul_(-1)
        out.fill_(-torch.inf)
        old.fill_(-torch.inf)
        graph.replay()
        check()


def test_row_and_block_selection():
    n, width = 12, 16640
    q, w, cache, table = inputs(n, width)
    visible = torch.arange(width - n, width, device="npu", dtype=torch.int32)
    table[:, 20] = -1
    old = torch.full((n, width), -torch.inf, device="npu")
    out = old.clone()
    run(q, w, cache, table, visible, old, 16, False)
    run(q, w, cache, table, visible, out, 16, True)
    torch.testing.assert_close(out, old, atol=0, rtol=0)
    for a, b in zip(
        select_full_scores(out, visible, 512, 2048, logical_capacity=width),
        select_full_scores(old, visible, 512, 2048, logical_capacity=width),
    ):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.mark.parametrize("parts", [3, 6, 12, 24])
def test_balanced_partitions_preserve_scores_on_replay(parts):
    q, w, cache, table = inputs(7, 2112)
    visible = torch.tensor(
        [2101, 2049, 65, 0, -1, 2109, 2112], device="npu", dtype=torch.int32
    )
    table[:, 2] = -1
    out = torch.full((7, 2112), -torch.inf, device="npu")
    scan_scores(q, w, cache, table, visible, out, shards=2, parts=parts)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scan_scores(q, w, cache, table, visible, out, shards=2, parts=parts)
    for step in range(3):
        out.fill_(-torch.inf)
        visible.add_(1)
        q.neg_()
        table[1] = table[0] if step % 2 else table[0].roll(1)
        graph.replay()
        torch.testing.assert_close(
            out.cpu(), reference(q, w, cache, table, visible, 2), atol=0, rtol=0
        )


def test_long_context_replay_stability():
    n, width = 192, 65536
    q, w, cache, table = inputs(n, width)
    # Ordinary random BF16 input exercises rounding, cancellation and signed scores.
    q.normal_()
    w.normal_()
    cache.normal_()
    visible = torch.full((n,), 51200, device="npu", dtype=torch.int32)
    table[1::12] = table[1::12].roll(1, 1)
    table[:, 5] = 0
    table[:, 8] = -1
    out = torch.full((n, width), -torch.inf, device="npu")
    old = out.clone()
    run(q, w, cache, table, visible, out, 1, True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run(q, w, cache, table, visible, out, 1, True)
    for step in range(12):
        visible.fill_(51200 + step)
        if step % 3 == 0:
            q.neg_()
        if step % 4 == 0:
            table[1::12] = table[::12]
        elif step % 4 == 1:
            table[1::12] = table[::12].roll(1, 1)
        out.fill_(-torch.inf)
        old.fill_(-torch.inf)
        graph.replay()
        run(q, w, cache, table, visible, old, 1, False)
        torch.testing.assert_close(out, old, atol=0, rtol=0)


@pytest.mark.parametrize("n", [1, 6])
def test_candidate_publication_small_graph(n):
    from tokenspeed_kernel_npu.ops.dsv41_index_device import topk_tile

    q, w, cache, table = inputs(n, 16384)
    visible = torch.full((n,), 16381, dtype=torch.int32, device="npu")
    out = tuple(
        torch.empty(shape, device="npu", dtype=torch.int32)
        for shape in ((n, 512), (n,), (n, 2048), (n,))
    )

    def launch():
        return topk_tile(q, w, cache, table, visible, None, 512, 2048, 1024, 8, out)

    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for width in (16381, 157):
        visible.fill_(width)
        table[:, 2] = 0
        graph.replay()
        torch.npu.synchronize()
        scores = reference(q, w, cache, table, visible, 8)
        for row in range(n):
            for ids, lengths, values in (
                (out[0], out[1], scores),
                (out[2], out[3], scores.reshape(n, -1, 8).amax(-1)),
            ):
                source = values[row].clone()
                if ids.shape[1] == 2048:
                    valid = torch.isfinite(source).nonzero().flatten()
                    if valid.numel():
                        source[valid[-1]] = torch.inf
                k = ids.shape[1]
                expected = source.topk(k).values
                count = int((expected > -torch.inf).sum())
                actual_ids = ids[row].cpu().long()
                assert int(lengths[row]) == count
                assert (actual_ids[count:] == -1).all()
                chosen = actual_ids[:count]
                assert chosen.unique().numel() == count
                torch.testing.assert_close(
                    source[chosen].sort().values,
                    expected[:count].sort().values,
                    atol=0,
                    rtol=0,
                )
