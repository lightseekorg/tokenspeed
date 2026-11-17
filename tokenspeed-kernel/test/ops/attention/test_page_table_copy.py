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

"""``copy_page_table`` must match a torch oracle: ``out[:live_rows,
:source_cols] == source[:live_rows]`` and every other entry of ``out`` is 0,
even when ``out`` starts out full of stale page IDs. ``source`` itself must
never change."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mla._triton import page_table
from utils import assert_no_triton_compile

# A page ID that never legitimately appears (page IDs are non-negative), so a
# leftover sentinel in ``out`` is unmistakable.
_STALE = 0x5A5A5A5A


def _oracle(
    source: torch.Tensor, out_shape: tuple[int, int], live_rows: int
) -> torch.Tensor:
    out = torch.zeros(out_shape, dtype=torch.int32)
    out[:live_rows, : source.shape[1]] = source[:live_rows].cpu()
    return out


def _run_and_check(source: torch.Tensor, out: torch.Tensor, live_rows: int):
    source_before = source.clone()
    expected = _oracle(source, tuple(out.shape), live_rows).to(out.device)
    page_table.copy_page_table(source, out, live_rows)
    if out.is_cuda:
        torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(source, source_before)


@pytest.mark.parametrize(
    ("out_shape", "source_shape", "live_rows"),
    [
        ((64, 524288), (64, 66560), 64),
        ((64, 524288), (64, 66560), 8),
        ((64, 524288), (64, 66560), 1),
        ((64, 524288), (64, 66560), 0),
        ((4, 16), (4, 16), 4),  # source_cols == out_cols
        ((6, 16), (4, 16), 4),  # out has more rows than source
        ((1, 1), (1, 1), 1),
        ((1, 1), (1, 1), 0),
        ((3, 1025), (3, 1), 2),  # out spans more than one 1024 block
    ],
)
def test_cpu_oracle_parity(out_shape, source_shape, live_rows):
    torch.manual_seed(0)
    source = torch.randint(1, 1000, source_shape, dtype=torch.int32)
    out = torch.full(out_shape, _STALE, dtype=torch.int32)
    _run_and_check(source, out, live_rows)


def test_cpu_strided_source_view():
    torch.manual_seed(0)
    # A source whose row stride is larger than its column count (a view into
    # a wider buffer), still contiguous per-column (stride(1) == 1).
    backing = torch.randint(1, 1000, (4, 40), dtype=torch.int32)
    source = backing[:, :16]
    assert source.stride(0) == 40 and source.stride(1) == 1
    out = torch.full((4, 16), _STALE, dtype=torch.int32)
    _run_and_check(source, out, live_rows=3)


if not torch.cuda.is_available():
    pytest.skip("CUDA required for the remaining cases", allow_module_level=True)


@pytest.mark.parametrize(
    ("out_shape", "source_shape", "live_rows"),
    [
        ((64, 524288), (64, 66560), 64),
        ((64, 524288), (64, 66560), 8),
        ((64, 524288), (64, 66560), 1),
        ((64, 524288), (64, 66560), 0),
        ((4, 16), (4, 16), 4),
        ((6, 16), (4, 16), 4),
        ((1, 1), (1, 1), 1),
        ((1, 1), (1, 1), 0),
        ((3, 1025), (3, 1), 2),
    ],
)
def test_cuda_kernel_oracle_parity(out_shape, source_shape, live_rows):
    torch.manual_seed(0)
    source = torch.randint(1, 1000, source_shape, dtype=torch.int32, device="cuda")
    out = torch.full(out_shape, _STALE, dtype=torch.int32, device="cuda")
    _run_and_check(source, out, live_rows)


def test_cuda_strided_source_view():
    torch.manual_seed(0)
    backing = torch.randint(1, 1000, (4, 40), dtype=torch.int32, device="cuda")
    source = backing[:, :16]
    assert source.stride(0) == 40 and source.stride(1) == 1
    out = torch.full((4, 16), _STALE, dtype=torch.int32, device="cuda")
    _run_and_check(source, out, live_rows=3)


def test_cuda_small_shapes():
    for out_shape, source_shape, live_rows in [
        ((1, 1), (1, 1), 0),
        ((1, 1), (1, 1), 1),
        ((2, 3), (1, 3), 1),
        ((2, 3), (1, 3), 0),
    ]:
        torch.manual_seed(1)
        source = torch.randint(1, 1000, source_shape, dtype=torch.int32, device="cuda")
        out = torch.full(out_shape, _STALE, dtype=torch.int32, device="cuda")
        _run_and_check(source, out, live_rows)


def test_no_recompile_across_rows_live_rows_and_cols():
    # Warm up before asserting: the DSV4.1 BS64 shape first.
    source = torch.randint(1, 1000, (64, 66560), dtype=torch.int32, device="cuda")
    out = torch.full((64, 524288), _STALE, dtype=torch.int32, device="cuda")
    page_table.copy_page_table(source, out, 64)
    torch.cuda.synchronize()

    with assert_no_triton_compile(page_table._copy_page_table_kernel):
        for rows, cols, source_cols, live_rows in (
            (1, 1024, 1024, 1),
            (1, 1024, 1024, 0),
            (2, 2048, 1024, 1),
            (17, 4096, 2048, 9),
            (64, 524288, 66560, 64),
            (64, 524288, 66560, 0),
            (256, 1023, 511, 128),
            (256, 1025, 513, 128),
        ):
            source = torch.randint(
                1, 1000, (rows, source_cols), dtype=torch.int32, device="cuda"
            )
            out = torch.full((rows, cols), _STALE, dtype=torch.int32, device="cuda")
            page_table.copy_page_table(source, out, live_rows)
        torch.cuda.synchronize()


class TestValidation:
    def _valid_args(self, device="cuda"):
        source = torch.ones((2, 4), dtype=torch.int32, device=device)
        out = torch.zeros((2, 4), dtype=torch.int32, device=device)
        return source, out

    def test_wrong_dtype_raises(self):
        source, out = self._valid_args()
        with pytest.raises(ValueError, match="int32"):
            page_table.copy_page_table(source.long(), out, 1)
        with pytest.raises(ValueError, match="int32"):
            page_table.copy_page_table(source, out.long(), 1)

    def test_non_unit_column_stride_raises(self):
        source, out = self._valid_args()
        backing = torch.zeros((2, 8), dtype=torch.int32, device=source.device)
        strided = backing[:, ::2]
        with pytest.raises(ValueError, match="contiguous"):
            page_table.copy_page_table(strided, out, 1)
        with pytest.raises(ValueError, match="contiguous"):
            page_table.copy_page_table(source, backing[:, ::2], 1)

    def test_live_rows_exceeds_capacity_raises(self):
        source, out = self._valid_args()
        with pytest.raises(ValueError, match="capacity"):
            page_table.copy_page_table(source, out, 3)  # > min(source rows, out rows)
        with pytest.raises(ValueError, match="capacity"):
            page_table.copy_page_table(source, out, -1)

    def test_source_wider_than_out_raises(self):
        source = torch.ones((2, 8), dtype=torch.int32, device="cuda")
        out = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="capacity"):
            page_table.copy_page_table(source, out, 2)
