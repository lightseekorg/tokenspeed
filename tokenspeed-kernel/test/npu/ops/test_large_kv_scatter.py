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

"""Large paged MHA scatters stay within the Ascend launch limit."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.kvcache.triton import store_kv_cache

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("rows", [32767, 32768, 65536])
def test_large_paged_mha_scatter(rows):
    page_size = 128
    width = 128
    pages = (rows + page_size - 1) // page_size
    source = torch.arange(rows * width * 2, dtype=torch.int32)
    source = (source % 113).view(rows, 2, 1, width).to(torch.bfloat16)
    locations = torch.arange(rows - 1, -1, -1, dtype=torch.int64)

    # A sentinel row between each pair of pages detects incorrect page strides.
    shape = (pages, page_size + 1, 1, width)
    expected_k = torch.full(shape, -1, dtype=torch.bfloat16)
    expected_v = torch.full_like(expected_k, -1)
    address = locations // page_size, locations % page_size
    expected_k[address] = source[:, 0]
    expected_v[address] = source[:, 1]

    source = source.npu()
    k_dst = torch.full(shape, -1, dtype=torch.bfloat16, device="npu")
    v_dst = torch.full_like(k_dst, -1)
    store_kv_cache(
        source[:, 0],
        source[:, 1],
        k_dst[:, :page_size],
        v_dst[:, :page_size],
        locations.npu(),
        enable_pdl=False,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(k_dst.cpu(), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v_dst.cpu(), expected_v, rtol=0, atol=0)
