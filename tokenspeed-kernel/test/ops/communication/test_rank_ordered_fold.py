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

"""The rank-ordered fold matches a left-to-right fp32 fold bit for bit."""

import pytest
import torch
from tokenspeed_kernel.ops.communication.triton import (
    _rank_ordered_fold_kernel,
    rank_ordered_fold,
)
from utils import assert_no_triton_compile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fold_matches_left_to_right_fp32_fold_on_a_gathered_layout(world, dtype):
    generator = torch.Generator().manual_seed(world)
    rows, cols = 37, 4608
    # A last-dim all-gather: [rows, world * cols], rank-major along the width.
    gathered = (torch.randn((rows, world * cols), generator=generator) * 1e3).to(
        dtype=dtype, device="cuda"
    )
    parts = gathered.view(rows, world, cols).transpose(0, 1)
    expected = parts[0].float()
    for rank in range(1, world):
        expected = expected + parts[rank].float()
    out = torch.empty((rows, cols), dtype=dtype, device="cuda")
    rank_ordered_fold(parts, out)
    assert torch.equal(out, expected.to(dtype))
    # A row folds to the same bits inside a larger payload.
    wide = torch.cat((torch.randn((rows, world * cols)).to(gathered), gathered[5:6]))
    wide_out = torch.empty((rows + 1, cols), dtype=dtype, device="cuda")
    rank_ordered_fold(wide.view(rows + 1, world, cols).transpose(0, 1), wide_out)
    assert torch.equal(wide_out[rows], out[5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_fold_reuses_its_kernel_for_variable_rows_and_widths():
    def fold(rows, cols):
        gathered = torch.randn((rows, 4 * cols), dtype=torch.bfloat16, device="cuda")
        out = torch.empty((rows, cols), dtype=torch.bfloat16, device="cuda")
        rank_ordered_fold(gathered.view(rows, 4, cols).transpose(0, 1), out)

    fold(16, 4096)
    with assert_no_triton_compile(_rank_ordered_fold_kernel):
        for rows, cols in ((1, 4096), (37, 4096), (300, 7168), (5, 4600)):
            fold(rows, cols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_zero_sums_are_stored_as_positive_zero():
    # The lamport all-reduce kernels send -0.0 as +0.0; the fold matches them.
    parts = torch.tensor([[[-0.0, -0.0, 1.0]], [[-0.0, 0.0, -1.0]]], device="cuda")
    out = torch.empty((1, 3), device="cuda")
    rank_ordered_fold(parts, out)
    assert out.view(torch.int32).tolist() == [[0, 0, 0]]
