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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa._cute_dsl import deep_select
from tokenspeed_kernel.ops.attention.dsv41._cute_dsl.selection import select_full

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 ragged initial-window geometry",
)


def _check_topk(scores, ends, indices, values, topk):
    for row, end in enumerate(ends.tolist()):
        end = max(0, min(end, scores.shape[1]))
        reference = scores[row, :end].masked_fill(
            torch.isnan(scores[row, :end]), -torch.inf
        )
        expected = reference.topk(min(topk, end)).values
        expected = expected[expected > -torch.inf]
        live = values[row] > -torch.inf
        ids = indices[row, live].long()
        got = values[row, live]
        assert ids.numel() == expected.numel()
        assert ids.unique().numel() == ids.numel()
        assert bool(((ids >= 0) & (ids < end)).all())
        assert torch.equal(got, scores[row, ids])
        assert torch.equal(got.sort().values, expected.sort().values)


@pytest.mark.parametrize(
    "topk,end",
    [(2048, n) for n in (2049, 3069, 4097, 6138, 8191, 12287)]
    + [(512, n) for n in (513, 2049, 4097)],
)
@pytest.mark.parametrize(
    "distribution", ["constant", "increasing", "tail_dominant", "random"]
)
def test_partial_initial_window_fills_topk(topk, end, distribution):
    width = (end + 255) // 256 * 256
    scores = torch.full((1, width), torch.nan, device="cuda")
    if distribution == "constant":
        scores[0, :end] = 1
    elif distribution == "increasing":
        scores[0, :end] = torch.arange(end, device="cuda", dtype=torch.float32)
    elif distribution == "tail_dominant":
        scores[0, :end] = 0
        tail = (end - 1) // 2048 * 2048
        scores[0, tail:end] = 100
    else:
        generator = torch.Generator(device="cuda").manual_seed(113)
        scores[0, :end] = torch.randn(end, device="cuda", generator=generator)
    ends = torch.tensor([end], device="cuda", dtype=torch.int32)
    ids, values = deep_select.deepselect_topk(
        scores, ends, topk, capacity=deep_select.capacity_for(topk), cluster_size=1
    )
    torch.cuda.synchronize()
    _check_topk(scores, ends, ids, values, topk)


def test_ragged_prefix_graph_ignores_large_unwritten_tail():
    lengths = [0, 17, 2048, 2049, 3069, 6138, 8191]
    width, topk = 524288, 2048
    scores = torch.full((len(lengths), width), torch.nan, device="cuda")
    for row, end in enumerate(lengths):
        scores[row, :end] = torch.arange(end, device="cuda", dtype=torch.float32)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    ids = torch.empty((len(lengths), topk), device="cuda", dtype=torch.int32)
    values = torch.empty((len(lengths), topk), device="cuda", dtype=torch.float32)

    def run():
        deep_select.deepselect_topk(
            scores,
            ends,
            topk,
            capacity=2048,
            cluster_size=1,
            indices=ids,
            values=values,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for current in (lengths, list(reversed(lengths))):
        scores.fill_(torch.nan)
        for row, end in enumerate(current):
            scores[row, :end] = torch.arange(end, device="cuda", dtype=torch.float32)
        ends.copy_(torch.tensor(current, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        _check_topk(scores, ends, ids, values, topk)


@pytest.mark.parametrize("visible", [24550, 49101, 65523])
def test_full_pipeline_keeps_2048_ragged_block_candidates(visible):
    width = 1048576
    scores = torch.full((1, width), torch.nan, device="cuda")
    scores[0, :visible] = torch.arange(visible, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([visible], device="cuda", dtype=torch.int32)
    table = torch.arange(width // 64, device="cuda", dtype=torch.int32)[None, :]
    out = (
        torch.empty((1, 512), device="cuda", dtype=torch.int32),
        torch.empty((1,), device="cuda", dtype=torch.int32),
        torch.empty((1, 2048), device="cuda", dtype=torch.int32),
        torch.empty((1,), device="cuda", dtype=torch.int32),
    )
    select_full(scores, table, lengths, width // 64, 512, 2048, out)
    torch.cuda.synchronize()
    row_ids = out[0][0, : out[1][0]].long()
    block_ids = out[2][0, : out[3][0]].long()
    assert row_ids.numel() == row_ids.unique().numel() == 512
    assert block_ids.numel() == block_ids.unique().numel() == 2048
    assert torch.equal(
        scores[0, row_ids].sort().values,
        scores[0, :visible].topk(512).values.sort().values,
    )
    block_count = (visible + 7) // 8
    padded = torch.full((block_count * 8,), -torch.inf, device="cuda")
    padded[:visible] = scores[0, :visible]
    maxima = padded.reshape(-1, 8).amax(1)
    maxima[-1] = torch.inf
    assert bool(((block_ids >= 0) & (block_ids < block_count)).all())
    assert torch.equal(
        maxima[block_ids].sort().values, maxima.topk(2048).values.sort().values
    )
