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

"""Parity tests: ``expand_block_table_rows`` must equal the per-row torch
reference, on both the CUDA kernel and the non-CUDA torch fallback, and must
overwrite every entry of ``out`` exactly once even when it is pre-filled with
stale data."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.metadata import torch_expand_block_table_rows
from tokenspeed_kernel.ops.metadata import expand_block_table_rows
from tokenspeed_kernel.ops.metadata.block_table_rows import (
    _expand_block_table_rows_kernel,
)
from utils import assert_no_triton_compile

# A stale sentinel that is not 0, -1 or any value this suite writes: every
# entry of ``out`` must be overwritten, or it survives the comparison.
_SENTINEL = 0x5A5A5A5A

# (width, start, end) per row. 0 <= start <= end <= width.
_ROW_SHAPES = [
    [(1, 0, 0)],
    [(1, 0, 1)],
    [(5, 0, 0)],  # all-hole row
    [(5, 5, 5)],  # all-padding row
    [(5, 2, 2)],  # empty span in the middle
    [(5, 0, 5)],  # full span
    [(1023, 0, 1023)],
    [(1024, 0, 1024)],
    [(1025, 0, 1025)],
    [(1023, 1000, 1023)],
    [(1024, 1000, 1024)],
    [(1025, 1000, 1025)],
    [(200000, 199997, 200000), (3, 1, 2)],
    # DSV4.1 BS64 shape: 64 rows cycling through 4 widths, with a short live
    # span of 1-3 pages near each row's end (as a sliding-window group
    # uploads after _row_windows).
    [
        (width, width - span, width)
        for width, span in ([(2080, 1), (1040, 2), (2080, 3), (66560, 1)] * 16)
    ],
    [(3, 0, 0), (3, 3, 3), (3, 1, 2)],  # empty values tensor case driver
]


def _build_rows(shapes: list[tuple[int, int, int]]):
    """rows/out_numel/total_values for one case; row sources are 0-based
    into a tightly packed values tensor."""
    rows = []
    offset = 0
    source = 0
    for width, start, end in shapes:
        rows.append([offset, width, start, end, source])
        offset += width
        source += end - start
    return rows, offset, source


def _reference(
    rows_t: torch.Tensor, values_t: torch.Tensor, numel: int
) -> torch.Tensor:
    out = torch.full((numel,), _SENTINEL, dtype=torch.int32)
    torch_expand_block_table_rows(rows_t.cpu(), values_t.cpu(), out)
    return out


def _values_for(total_values: int, device) -> torch.Tensor:
    # Values hold 1.. (never 0/-1), so they stay distinguishable from holes
    # and padding in a failure message.
    return torch.arange(1, total_values + 1, dtype=torch.int32, device=device)


def _aligned_values(total_values: int, byte_offset: int, device) -> torch.Tensor:
    """``total_values`` entries whose data pointer is shifted by
    ``byte_offset`` int32 elements, to vary the pointer-alignment class
    ``do_not_specialize_on_alignment=["Values"]`` keys on. The row ``source``
    fields index this tensor from 0, as ``_build_rows`` produces."""
    pool = torch.arange(
        1, byte_offset + total_values + 1, dtype=torch.int32, device=device
    )
    return pool[byte_offset:]


@pytest.mark.parametrize("shapes", _ROW_SHAPES)
def test_cpu_torch_path_matches_reference(shapes):
    rows, numel, total_values = _build_rows(shapes)
    rows_t = torch.tensor(rows, dtype=torch.int32)
    values_t = _values_for(total_values, "cpu")
    out = torch.full((numel,), _SENTINEL, dtype=torch.int32)
    expand_block_table_rows(rows_t, values_t, out)
    assert torch.equal(out, _reference(rows_t, values_t, numel))
    assert not (out == _SENTINEL).any()


def test_cpu_torch_path_empty_values_tensor():
    shapes = [(3, 0, 0), (3, 3, 3), (3, 1, 1)]  # every row's span is empty
    rows, numel, total_values = _build_rows(shapes)
    assert total_values == 0
    rows_t = torch.tensor(rows, dtype=torch.int32)
    values_t = torch.empty(0, dtype=torch.int32)
    out = torch.full((numel,), _SENTINEL, dtype=torch.int32)
    expand_block_table_rows(rows_t, values_t, out)
    assert torch.equal(out, _reference(rows_t, values_t, numel))


@pytest.mark.parametrize("seed", range(10))
def test_cpu_torch_path_matches_reference_randomized(seed):
    g = torch.Generator().manual_seed(seed)
    num_rows = int(torch.randint(1, 12, (1,), generator=g))
    shapes = []
    for _ in range(num_rows):
        width = int(torch.randint(0, 40, (1,), generator=g))
        start = int(torch.randint(0, width + 1, (1,), generator=g))
        end = int(torch.randint(start, width + 1, (1,), generator=g))
        shapes.append((width, start, end))
    rows, numel, total_values = _build_rows(shapes)
    rows_t = torch.tensor(rows, dtype=torch.int32)
    values_t = _values_for(total_values, "cpu")
    out = torch.full((numel,), _SENTINEL, dtype=torch.int32)
    expand_block_table_rows(rows_t, values_t, out)
    assert torch.equal(out, _reference(rows_t, values_t, numel))


if not torch.cuda.is_available():
    pytest.skip("CUDA required for the remaining cases", allow_module_level=True)


@pytest.mark.parametrize("shapes", _ROW_SHAPES)
def test_cuda_kernel_matches_reference(shapes):
    rows, numel, total_values = _build_rows(shapes)
    rows_t = torch.tensor(rows, dtype=torch.int32, device="cuda")
    values_t = _values_for(total_values, "cuda")
    out = torch.full((numel,), _SENTINEL, dtype=torch.int32, device="cuda")
    expand_block_table_rows(rows_t, values_t, out)
    torch.cuda.synchronize()
    expected = _reference(rows_t, values_t, numel).to("cuda")
    assert torch.equal(out, expected)
    assert not (out == _SENTINEL).any()


def test_cuda_kernel_empty_values_tensor():
    shapes = [(3, 0, 0), (3, 3, 3), (3, 1, 1)]
    rows, numel, total_values = _build_rows(shapes)
    assert total_values == 0
    rows_t = torch.tensor(rows, dtype=torch.int32, device="cuda")
    values_t = torch.empty(0, dtype=torch.int32, device="cuda")
    out = torch.full((numel,), _SENTINEL, dtype=torch.int32, device="cuda")
    expand_block_table_rows(rows_t, values_t, out)
    torch.cuda.synchronize()
    assert torch.equal(out, _reference(rows_t, values_t, numel).to("cuda"))


def test_no_recompile_across_row_counts_widths_and_value_alignment():
    # Warm the kernel once on every alignment class the sweep below hits so
    # the context below asserts zero *new* specializations, matching the
    # pattern `assert_no_triton_compile` documents.
    for value_offset in range(4):
        rows, numel, total_values = _build_rows([(7, 2, 5)])
        rows_t = torch.tensor(rows, dtype=torch.int32, device="cuda")
        values_t = _aligned_values(total_values, value_offset, "cuda")
        out = torch.empty(numel, dtype=torch.int32, device="cuda")
        expand_block_table_rows(rows_t, values_t, out)
    torch.cuda.synchronize()

    with assert_no_triton_compile(_expand_block_table_rows_kernel):
        for num_rows in (1, 2, 17, 256):
            for width in (1, 17, 1024, 2048):
                for value_offset in range(4):
                    shapes = [(width, width // 3, width) for _ in range(num_rows)]
                    rows, numel, total_values = _build_rows(shapes)
                    rows_t = torch.tensor(rows, dtype=torch.int32, device="cuda")
                    values_t = _aligned_values(total_values, value_offset, "cuda")
                    out = torch.empty(numel, dtype=torch.int32, device="cuda")
                    expand_block_table_rows(rows_t, values_t, out)
        torch.cuda.synchronize()


class TestValidation:
    def _valid_args(self, device="cuda"):
        rows = torch.tensor([[0, 2, 0, 2, 0]], dtype=torch.int32, device=device)
        values = torch.tensor([1, 2], dtype=torch.int32, device=device)
        out = torch.empty(2, dtype=torch.int32, device=device)
        return rows, values, out

    def test_rows_wrong_dtype_raises(self):
        rows, values, out = self._valid_args()
        with pytest.raises(ValueError, match="rows"):
            expand_block_table_rows(rows.long(), values, out)

    def test_rows_wrong_width_raises(self):
        rows, values, out = self._valid_args()
        bad_rows = torch.tensor([[0, 2, 0, 2, 0, 0]], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="rows"):
            expand_block_table_rows(bad_rows, values, out)

    def test_rows_non_contiguous_raises(self):
        rows, values, out = self._valid_args()
        wide = torch.zeros((1, 10), dtype=torch.int32, device="cuda")
        wide[0, ::2] = torch.tensor([0, 2, 0, 2, 0])
        non_contig = wide[:, ::2]
        assert not non_contig.is_contiguous()
        with pytest.raises(ValueError, match="rows"):
            expand_block_table_rows(non_contig, values, out)

    def test_values_wrong_ndim_raises(self):
        rows, values, out = self._valid_args()
        with pytest.raises(ValueError, match="values"):
            expand_block_table_rows(rows, values.view(1, 2), out)

    def test_out_non_contiguous_raises(self):
        rows, values, out = self._valid_args()
        wide = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="out"):
            expand_block_table_rows(rows, values, wide[:, 0])

    def test_device_mismatch_raises(self):
        rows, values, out = self._valid_args(device="cuda")
        with pytest.raises(ValueError, match="device"):
            expand_block_table_rows(rows, values, out.cpu())
        with pytest.raises(ValueError, match="device"):
            expand_block_table_rows(rows, values.cpu(), out)
