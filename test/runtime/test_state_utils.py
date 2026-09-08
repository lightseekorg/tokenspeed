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

from tokenspeed.runtime.layers.attention.backends.state.utils import row_stride_i32


@pytest.mark.parametrize(
    ("dtype", "dense_stride", "interleaved_stride"),
    [(torch.bfloat16, 4, 16), (torch.float32, 8, 32), (torch.int64, 16, 64)],
)
def test_row_stride_i32_preserves_padding_between_state_rows(
    dtype: torch.dtype, dense_stride: int, interleaved_stride: int
) -> None:
    dense = torch.empty((3, 8), dtype=dtype)
    interleaved = torch.empty((3, 4, 8), dtype=dtype)[:, 2, :]

    assert dense.shape == interleaved.shape
    assert not interleaved.is_contiguous()
    assert row_stride_i32(dense) == dense_stride
    assert row_stride_i32(interleaved) == interleaved_stride


def test_row_stride_i32_allows_empty_row_payloads() -> None:
    rows = torch.empty_strided((3, 0), (8, 1), dtype=torch.int64)

    assert row_stride_i32(rows) == 16


def test_row_stride_i32_rejects_noncontiguous_payloads() -> None:
    rows = torch.empty((3, 2, 4), dtype=torch.float32).transpose(1, 2)

    with pytest.raises(RuntimeError, match="contiguous row payloads"):
        row_stride_i32(rows)


def test_row_stride_i32_rejects_unaligned_row_starts() -> None:
    rows = torch.empty_strided((3, 2), (3, 1), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="4-byte aligned"):
        row_stride_i32(rows)
