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
from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 3, 512, 2048])
@pytest.mark.parametrize("sanitize", [False, True])
def test_masked_mla_write_preserves_foreign_rows_and_graph_replay(rows, sanitize):
    cache = torch.full((rows + 1, 1, 576), 7, dtype=torch.bfloat16, device="cuda")
    values = torch.randn((rows, 1, 576), dtype=torch.bfloat16, device="cuda")
    if sanitize:
        values[:, :, 0] = float("nan")
        values[:, :, 512] = float("inf")
    slots = torch.arange(1, rows + 1, dtype=torch.int64, device="cuda")
    owned = torch.arange(rows, device="cuda") % 2 == 0
    slots.masked_fill_(~owned, 0)

    def write():
        set_mla_kv_buffer_triton(
            cache,
            slots,
            values[..., :512],
            values[..., 512:],
            write_mask=owned,
            enable_pdl=False,
            sanitize=sanitize,
        )

    def check():
        expected = torch.full_like(cache, 7)
        expected[slots[owned]] = (
            torch.nan_to_num(values[owned]) if sanitize else values[owned]
        )
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)
        assert torch.all(cache[0] == 7)

    write()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    cache.fill_(7)
    owned.logical_not_()
    slots.copy_(torch.arange(1, rows + 1, device="cuda"))
    slots.masked_fill_(~owned, 0)
    graph.replay()
    check()
    cache.fill_(7)
    owned.fill_(False)
    slots.zero_()
    graph.replay()
    assert torch.all(cache == 7)
