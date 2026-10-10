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

"""GFX950 CSA2 row selection matches the torch top-k selection."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv41._gluon.indexer import (  # noqa: E402
    select_rows_torch,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41 import indexer  # noqa: E402

DEVICE = "cuda"


def _logits(queries, width, generator, levels=None):
    logits = torch.randn(queries, width, device=DEVICE, generator=generator)
    if levels is not None:
        # Few distinct values: many equal-score ties at the boundary.
        logits = (logits * levels).round()
    visible = torch.randint(
        0, width + 1, (queries,), device=DEVICE, generator=generator
    )
    visible[0] = width
    visible[-1] = 0
    columns = torch.arange(width, device=DEVICE)
    return logits.masked_fill(columns[None] >= visible[:, None], -torch.inf)


def _candidates(queries, width, ordered, generator):
    blocks = torch.stack(
        [
            torch.randperm(1 << 20, device=DEVICE, generator=generator)[: width // 8]
            for _ in range(queries)
        ]
    )
    if ordered:
        blocks = blocks.sort(dim=1).values
    return blocks.to(torch.int32)


def _select(fn, logits, candidates, topk):
    rows = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=DEVICE)
    lens = torch.empty(logits.shape[0], dtype=torch.int32, device=DEVICE)
    fn(logits, candidates, topk, rows, lens)
    return rows, lens


@pytest.mark.parametrize(
    ("queries", "width", "topk", "candidates"),
    [
        (192, 2048, 512, None),
        (192, 4608, 512, "ordered"),
        (64, 16384, 512, "shuffled"),
        (8, 100, 512, None),
        (16, 3000, 300, "ordered"),
        (16, 777, 1, None),
    ],
)
def test_select_matches_torch(queries, width, topk, candidates):
    generator = torch.Generator(device=DEVICE).manual_seed(width)
    logits = _logits(queries, width, generator)
    blocks = (
        None
        if candidates is None
        else _candidates(queries, width, candidates == "ordered", generator)
    )
    rows, lens = _select(
        indexer.launch_gluon_dsv41_index_topk_select_gfx950, logits, blocks, topk
    )
    ref_rows, ref_lens = _select(select_rows_torch, logits, blocks, topk)
    assert torch.equal(lens, ref_lens)
    assert torch.equal(rows, ref_rows)


def test_select_boundary_ties_pick_an_equivalent_subset():
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    queries, width, topk = 32, 4096, 512
    logits = _logits(queries, width, generator, levels=4)
    rows, lens = _select(
        indexer.launch_gluon_dsv41_index_topk_select_gfx950, logits, None, topk
    )
    ref_rows, ref_lens = _select(select_rows_torch, logits, None, topk)
    assert torch.equal(lens, ref_lens)
    for q in range(queries):
        n = int(lens[q])
        picked = rows[q, :n].long()
        assert torch.all(picked[1:] > picked[:-1])
        assert torch.all(rows[q, n:] == -1)
        # Same multiset of selected scores as torch.topk.
        assert torch.equal(
            logits[q, picked].sort().values,
            logits[q, ref_rows[q, :n].long()].sort().values,
        )


def test_select_batch_shapes_share_one_binary():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    select = indexer.launch_gluon_dsv41_index_topk_select_gfx950
    _select(select, _logits(3, 1000, generator), None, 512)
    with assert_no_triton_compile(indexer.gluon_dsv41_index_topk_select_gfx950):
        for queries, width in ((2, 64), (32, 2048), (192, 4608), (7, 16384)):
            _select(select, _logits(queries, width, generator), None, 512)
