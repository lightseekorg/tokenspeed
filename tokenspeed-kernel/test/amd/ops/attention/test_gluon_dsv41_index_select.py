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

"""GFX950/GFX1250 CSA2 row selection matches the torch top-k selection."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4, is_cdna5

if not (is_cdna4() or is_cdna5()):
    pytest.skip("AMD CDNA4 or CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv41._gluon.indexer import (  # noqa: E402
    select_rows_torch,
)

if is_cdna4():
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.indexer import (
        gluon_dsv41_index_topk_select_gfx950 as select_kernel,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.indexer import (
        launch_gluon_dsv41_index_topk_select_gfx950 as launch_select,
    )
else:
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41.topk import (
        gluon_dsv41_index_topk_select_gfx1250 as select_kernel,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41.topk import (
        launch_gluon_dsv41_index_topk_select_gfx1250 as launch_select,
    )

DEVICE = "cuda"
# Largest selection capacity each launcher accepts.
MAX_TOPK = 2048 if is_cdna4() else 1024


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


def _select(fn, logits, candidates, topk, live_widths=None):
    rows = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=DEVICE)
    lens = torch.empty(logits.shape[0], dtype=torch.int32, device=DEVICE)
    fn(logits, candidates, live_widths, topk, rows, lens)
    return rows, lens


@pytest.mark.parametrize(
    ("queries", "width", "topk", "candidates"),
    [
        (192, 2048, 512, None),
        (192, 4608, 512, "ordered"),
        (64, 16384, 512, "shuffled"),
        (16, 32768, MAX_TOPK, "shuffled"),
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
    rows, lens = _select(launch_select, logits, blocks, topk)
    ref_rows, ref_lens = _select(select_rows_torch, logits, blocks, topk)
    assert torch.equal(lens, ref_lens)
    assert torch.equal(rows, ref_rows)


def test_select_live_widths_skip_the_dead_tail_of_capacity_rows():
    # Full selection scores a page-table-capacity row but only the visible
    # prefix is finite; bounding the scan must not change the picks.
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    queries, width, topk = 48, 65536, 512
    logits = torch.full((queries, width), -torch.inf, device=DEVICE)
    visible = torch.randint(0, 4097, (queries,), device=DEVICE, generator=generator)
    visible[0], visible[1] = 0, 4096
    for q in range(queries):
        n = int(visible[q])
        logits[q, :n] = torch.randn(n, device=DEVICE, generator=generator)
    rows, lens = _select(
        launch_select,
        logits,
        None,
        topk,
        live_widths=visible.to(torch.int32),
    )
    ref_rows, ref_lens = _select(select_rows_torch, logits, None, topk)
    assert torch.equal(lens, ref_lens)
    assert torch.equal(rows, ref_rows)


def test_select_boundary_ties_pick_an_equivalent_subset():
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    queries, width, topk = 32, 4096, 512
    logits = _logits(queries, width, generator, levels=4)
    rows, lens = _select(launch_select, logits, None, topk)
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


def test_select_signed_zero_boundary_ties_keep_lowest_columns():
    if not is_cdna5():
        pytest.skip("the gfx950 selector orders -0.0 below +0.0")
    queries, width, topk = 2, 64, 8
    logits = torch.zeros(queries, width, device=DEVICE)
    logits[:, :4] = 1.0
    logits[:, 4::2] = -0.0
    rows, lens = _select(launch_select, logits, None, topk)
    assert torch.all(lens == topk)
    expected = torch.arange(topk, dtype=torch.int32, device=DEVICE)
    assert torch.equal(rows, expected.expand(queries, topk))


def test_select_batch_shapes_share_one_binary():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    select = launch_select

    def limits(queries, width):
        return torch.full((queries,), width // 2, dtype=torch.int32, device=DEVICE)

    _select(select, _logits(3, 1000, generator), None, 512)
    _select(select, _logits(3, 1000, generator), None, 512, limits(3, 1000))
    with assert_no_triton_compile(select_kernel):
        for queries, width in ((2, 64), (32, 2048), (192, 4608), (7, 16384)):
            logits = _logits(queries, width, generator)
            _select(select, logits, None, 512)
            _select(select, logits, None, 512, limits(queries, width))


def test_select_rejects_layouts_the_kernel_would_misaddress():
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    queries, width, topk = 4, 64, 16
    logits = _logits(queries, width, generator)
    candidates = _candidates(queries, 2 * width, True, generator)
    rows = torch.empty(queries, topk, dtype=torch.int32, device=DEVICE)
    lens = torch.empty(queries, dtype=torch.int32, device=DEVICE)
    launch = launch_select
    for bad in (
        # Strided candidates would map picks to the wrong block ids.
        dict(candidates=candidates[:, ::2]),
        # Too few candidate blocks for the logits width reads past each row.
        dict(candidates=candidates[:, : width // 8 - 1]),
        dict(row_out=rows[:, : topk - 1]),
        dict(row_lens=torch.empty(2 * queries, dtype=torch.int32, device=DEVICE)[::2]),
        dict(live_widths=torch.empty(queries + 1, dtype=torch.int32, device=DEVICE)),
    ):
        args = dict(
            logits=logits,
            candidates=candidates[:, : width // 8],
            live_widths=None,
            topk=topk,
            row_out=rows,
            row_lens=lens,
        )
        args.update(bad)
        with pytest.raises(ValueError):
            launch(**args)
