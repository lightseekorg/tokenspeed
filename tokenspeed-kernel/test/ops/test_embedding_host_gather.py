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
from tokenspeed_kernel.ops.embedding import host_gather


def test_cpu_uint8_row_gather_masks_invalid_ids():
    table = torch.arange(20 * 8, dtype=torch.uint8).reshape(20, 8)
    indices = torch.tensor([[0, 3, -1, 19, 20]])
    out = host_gather.uint8_row_gather(table, indices, None)
    expected = table[indices.clamp(0, 19)]
    expected[indices < 0] = 0
    expected[indices >= 20] = 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_host_uint8_row_gather_matches_index():
    table = torch.arange(32 * 64, dtype=torch.uint8).reshape(32, 64)
    if torch.cuda.is_available():
        table = table.pin_memory()
    indices = torch.tensor([0, 7, 31, -1, 32], device="cuda:0")
    out = host_gather.uint8_row_gather(table, indices, None)
    expected = torch.zeros(5, 64, dtype=torch.uint8, device="cuda:0")
    expected[0] = table[0]
    expected[1] = table[7]
    expected[2] = table[31]
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)
