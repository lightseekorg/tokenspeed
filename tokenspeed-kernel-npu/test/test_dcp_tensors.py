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

"""CPU tensor regressions for collective DCP owner selection and empty shards."""

import pytest
import torch
from tokenspeed_kernel_npu.ops.dcp_tensors import (
    merge_normalized_partials,
    select_local_topk,
)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_local_topk_preserves_selected_order_and_ignores_foreign_and_invalid_ids(
    rank, dtype
):
    width = 256
    ids = torch.arange(3 * width, dtype=torch.int32).reshape(3, 1, width)
    ids[0, 0, ::7] = -1
    positions = torch.arange(width, dtype=dtype).repeat(3, 1)
    positions[0] = positions[0].flip(0) + rank * width
    positions[1] += ((rank + 1) % 8) * width
    positions[2, ::2] += rank * width
    positions[2, 1::2] += ((rank + 1) % 8) * width
    original_ids, original_positions = ids.clone(), positions.clone()
    result, chunks = select_local_topk(ids, positions, rank=rank)
    for row in range(3):
        expected = [
            int(ids[row, 0, p % width])
            for p in positions[row].tolist()
            if p >= 0 and p // width == rank and ids[row, 0, p % width] >= 0
        ]
        expected += [-1] * (width - len(expected))
        assert result[row, 0].tolist() == expected
        assert chunks[row] == (sum(x >= 0 for x in expected) + 127) // 128
    assert result.dtype == torch.int32 and chunks.dtype == torch.int32
    torch.testing.assert_close(ids, original_ids, atol=0, rtol=0)
    torch.testing.assert_close(positions, original_positions, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_normalized_merge_masks_nan_empty_shards_and_preserves_inputs(dtype):
    output = torch.tensor(
        [
            [[float("nan"), float("nan")], [1, 3], [float("nan"), 9]],
            [[float("nan"), 7], [5, 7], [6, 8]],
            [[float("nan"), 0], [0, 0], [10, 12]],
        ],
        dtype=dtype,
    )
    lse = torch.tensor([[0, 1, 0], [0, 3, 1], [0, 0, 1]]).float().log()
    original_output, original_lse = output.clone(), lse.clone()
    actual = merge_normalized_partials(output, lse)
    expected = torch.tensor([[0, 0], [4, 6], [8, 10]], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(output, original_output, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(lse, original_lse, atol=0, rtol=0)


def test_normalized_merge_uses_exponentially_weighted_finite_lse():
    output = torch.tensor([[[2.0]], [[10.0]]], dtype=torch.float32)
    lse = torch.tensor([[100.0], [100.0 + torch.log(torch.tensor(3.0))]])
    actual = merge_normalized_partials(output, lse)
    torch.testing.assert_close(
        actual, torch.tensor([[8.0]], dtype=torch.bfloat16), atol=0, rtol=0
    )
