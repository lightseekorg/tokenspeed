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

"""Paged history indices preserve physical slots at large table row offsets."""

from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch

from tokenspeed.runtime.layers.attention.chunk import (
    create_chunked_cache_kv_indices_paged,
)

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_paged_history_table_row_offsets_exceed_int32():
    # Only three table entries are touched. The third row's element offset
    # exceeds 2**31 although the runtime stride itself still fits int32.
    stride = (1 << 30) + 1
    table = torch.empty_strided((3, 1), (stride, 1), dtype=torch.int32, device="cuda")
    table[:, 0] = torch.tensor([17, 19, 23], dtype=torch.int32, device="cuda")
    starts = torch.zeros(3, dtype=torch.int32, device="cuda")
    lengths = torch.ones(3, dtype=torch.int32, device="cuda")
    cumulative = torch.arange(4, dtype=torch.int32, device="cuda")
    output = torch.empty(3, dtype=torch.int32, device="cuda")
    create_chunked_cache_kv_indices_paged[(3,)](
        table,
        starts,
        lengths,
        cumulative,
        output,
        page_table_ptr_stride=table.stride(0),
        PAGE_SIZE=128,
    )
    torch.testing.assert_close(
        output.cpu(),
        torch.tensor([17, 19, 23], dtype=torch.int32) * 128,
        rtol=0,
        atol=0,
    )
