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

"""The fused page-table expansion must match the torch spelling exactly.

These are cache page ids: a wrong column sends a kernel reading another
request's KV, so the comparison is integer equality.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.triton.page_table import (  # noqa: E402
    expand_page_table,
)


def _reference(page_table, out, ratio, max_kernel_pages):
    """The zero/arange/clamp/mul/add/copy chain the kernel replaces."""
    rows, logical_columns = page_table.shape
    out[:rows].zero_()
    result = out[:rows, :max_kernel_pages]
    offsets = torch.arange(ratio, dtype=page_table.dtype, device=page_table.device)
    expanded = (page_table.clamp_min(0).unsqueeze(-1) * ratio + offsets).reshape(
        rows, logical_columns * ratio
    )
    copy_columns = min(max_kernel_pages, expanded.shape[1])
    result[:, :copy_columns].copy_(expanded[:, :copy_columns])
    return result


@pytest.mark.parametrize(
    ("ratio", "rows", "logical_cols"),
    [(2, 1, 1), (2, 3, 32), (4, 1, 5), (4, 16, 32), (8, 3, 5), (8, 16, 1)],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_matches_the_torch_chain(ratio, rows, logical_cols, dtype):
    torch.manual_seed(rows * 100 + logical_cols)
    table = torch.randint(0, 500, (rows, logical_cols), device="cuda", dtype=dtype)
    max_kernel_pages = logical_cols * ratio
    got_buf = torch.full((rows, max_kernel_pages), -7, device="cuda", dtype=dtype)
    want_buf = torch.full((rows, max_kernel_pages), -7, device="cuda", dtype=dtype)
    got = expand_page_table(
        table, got_buf, ratio=ratio, max_kernel_pages=max_kernel_pages
    )
    want = _reference(table, want_buf, ratio, max_kernel_pages)
    assert torch.equal(got, want), (ratio, rows, logical_cols, dtype)


@pytest.mark.parametrize("ratio", [2, 4])
def test_holes_clamp_to_the_null_page(ratio):
    table = torch.randint(0, 500, (4, 8), device="cuda", dtype=torch.int32)
    table[:, ::2] = -1
    cols = 8 * ratio
    got = expand_page_table(
        table,
        torch.empty((4, cols), device="cuda", dtype=torch.int32),
        ratio=ratio,
        max_kernel_pages=cols,
    )
    want = _reference(
        table, torch.empty((4, cols), device="cuda", dtype=torch.int32), ratio, cols
    )
    assert torch.equal(got, want)


def test_narrow_request_leaves_the_tail_zero():
    """Asking for fewer kernel pages than the table holds must zero the rest,
    not leave whatever the buffer held before."""
    table = torch.randint(1, 500, (3, 8), device="cuda", dtype=torch.int32)
    buf = torch.full((3, 64), 99, device="cuda", dtype=torch.int32)
    got = expand_page_table(table, buf, ratio=4, max_kernel_pages=10)
    assert got.shape == (3, 10)
    want = _reference(
        table, torch.full((3, 64), 99, device="cuda", dtype=torch.int32), 4, 10
    )
    assert torch.equal(got, want)
    # The columns the caller will not read must not keep the stale 99s either.
    assert torch.equal(buf[:, 10:], torch.zeros_like(buf[:, 10:]))


def test_writes_in_place_for_graph_replay():
    """Replay needs the recorded buffer, so the result must be that buffer
    with the expansion actually in it."""
    table = torch.randint(1, 500, (2, 4), device="cuda", dtype=torch.int32)
    buf = torch.empty((2, 16), device="cuda", dtype=torch.int32)
    ptr = buf.data_ptr()
    got = expand_page_table(table, buf, ratio=4, max_kernel_pages=16)
    assert got.data_ptr() == ptr
    want = _reference(
        table, torch.empty((2, 16), device="cuda", dtype=torch.int32), 4, 16
    )
    assert torch.equal(got, want)


def test_strided_inputs_are_rejected_not_misread():
    """A strided column view would make the kernel read the wrong pages."""
    wide = torch.randint(1, 500, (2, 8), device="cuda", dtype=torch.int32)
    out = torch.empty((2, 8), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="unit-stride"):
        expand_page_table(wide[:, ::2], out, ratio=2, max_kernel_pages=8)
    wide_out = torch.empty((2, 16), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="unit-stride"):
        expand_page_table(wide[:, :4], wide_out[:, ::2], ratio=2, max_kernel_pages=8)


def test_output_aliasing_the_table_is_rejected():
    table = torch.randint(1, 500, (2, 8), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="alias"):
        expand_page_table(table, table, ratio=2, max_kernel_pages=8)


def test_more_columns_than_one_program_block():
    """A fixed cache table can exceed the block width; the second column
    program must cover its own range."""
    table = torch.randint(1, 500, (3, 400), device="cuda", dtype=torch.int32)
    cols = 400 * 4
    got = expand_page_table(
        table,
        torch.empty((3, cols), device="cuda", dtype=torch.int32),
        ratio=4,
        max_kernel_pages=cols,
    )
    want = _reference(
        table, torch.empty((3, cols), device="cuda", dtype=torch.int32), 4, cols
    )
    assert torch.equal(got, want)


def test_a_pitched_row_view_is_followed():
    """Grouped replay hands rows of a wider stack, so stride(0) > shape[1]."""
    stack = torch.randint(1, 500, (4, 64), device="cuda", dtype=torch.int32)
    table = stack[:2, :5]
    assert table.stride(0) > table.shape[1]
    out_stack = torch.empty((4, 64), device="cuda", dtype=torch.int32)
    out = out_stack[:2, :20]
    got = expand_page_table(table, out, ratio=4, max_kernel_pages=20)
    want = _reference(
        table.contiguous(),
        torch.empty((2, 20), device="cuda", dtype=torch.int32),
        4,
        20,
    )
    assert torch.equal(got, want)
