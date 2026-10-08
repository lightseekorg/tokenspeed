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

"""Row-wise FP32 radix top-k coverage for GFX1250 CSA2 selection."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

import tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41 as dsv41_gfx1250  # noqa: E402
from tokenspeed_kernel.ops.attention.dsv41._gluon import (  # noqa: E402
    indexer as gluon_indexer,
)
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41 import (  # noqa: E402
    launch_triton_dsv41_index_topk_select_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41 import (  # noqa: E402
    topk as topk_mod,
)


def _assert_valid_topk(
    scores: torch.Tensor, k: int, values: torch.Tensor, columns: torch.Tensor
) -> None:
    """Check ``(values, columns)`` is a top-k of ``scores`` in column order with
    threshold ties taken lowest column first."""
    width = scores.shape[1]
    assert values.shape == columns.shape == (scores.shape[0], k)
    assert columns.dtype == torch.int64
    assert bool(((columns >= 0) & (columns < width)).all())
    assert bool((columns[:, 1:] > columns[:, :-1]).all())
    torch.testing.assert_close(scores.gather(1, columns), values, atol=0, rtol=0)
    expected = scores.topk(k, dim=1).values
    torch.testing.assert_close(
        values.sort(dim=1, descending=True).values, expected, atol=0, rtol=0
    )
    threshold = expected[:, -1:]
    above = (scores > threshold).sum(dim=1, keepdim=True)
    tie_columns = torch.where(
        scores == threshold,
        torch.arange(width, device=scores.device),
        width,
    )
    first_ties = tie_columns.sort(dim=1).values
    taken = (values == threshold).sum(dim=1)
    for row in range(scores.shape[0]):
        count = int(taken[row])
        assert count == k - int(above[row])
        picked = columns[row][values[row] == threshold[row]]
        torch.testing.assert_close(picked, first_ties[row, :count], atol=0, rtol=0)


@pytest.mark.parametrize(
    ("width", "k"),
    [
        (8, 8),
        (64, 64),
        (512, 512),
        (513, 512),
        (4096, 512),
        (6250, 2048),
        (12544, 512),
        (25000, 512),
        (50000, 512),
    ],
)
def test_topk_matches_torch_on_random_rows(width: int, k: int) -> None:
    torch.manual_seed(width + k)
    scores = torch.randn((9, width), device="cuda") * 30
    scores[:, ::5] = -float("inf")
    values, columns = launch_triton_dsv41_index_topk_select_gfx1250(scores, k)
    _assert_valid_topk(scores, k, values, columns)


def test_topk_takes_threshold_ties_in_column_order() -> None:
    torch.manual_seed(1)
    scores = torch.randint(-3, 4, (6, 9000), device="cuda").float()
    scores[0] = 0.0
    scores[1, 4000:] = -float("inf")
    scores[2] = -float("inf")
    scores[3, :100] = -0.0
    for k in (512, 2048):
        values, columns = launch_triton_dsv41_index_topk_select_gfx1250(scores, k)
        _assert_valid_topk(scores, k, values, columns)


def test_topk_reuses_kernel_across_widths_and_k() -> None:
    launch_triton_dsv41_index_topk_select_gfx1250(
        torch.randn((128, 4096), device="cuda"), 512
    )
    with assert_no_triton_compile(topk_mod.triton_dsv41_index_topk_select_gfx1250):
        for rows, width, k in (
            (128, 64, 64),
            (200, 16384, 512),
            (256, 50000, 512),
            (256, 6250, 2048),
        ):
            launch_triton_dsv41_index_topk_select_gfx1250(
                torch.randn((rows, width), device="cuda"), k
            )


@pytest.mark.parametrize(("rows", "radix"), [(96, False), (128, True)])
def test_csa2_selector_uses_radix_topk_for_full_score_tiles(
    rows: int, radix: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def counted(scores: torch.Tensor, k: int):
        calls.append(scores.shape)
        return launch_triton_dsv41_index_topk_select_gfx1250(scores, k)

    monkeypatch.setattr(
        dsv41_gfx1250, "launch_triton_dsv41_index_topk_select_gfx1250", counted
    )
    scores = torch.randn((rows, 9000), device="cuda")
    values, columns = gluon_indexer.launch_gfx1250_topk(scores, 512)
    assert bool(calls) == radix
    torch.testing.assert_close(scores.gather(1, columns), values, atol=0, rtol=0)
    torch.testing.assert_close(
        values.sort(dim=1, descending=True).values,
        scores.topk(512, dim=1).values,
        atol=0,
        rtol=0,
    )
