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

"""Paged MLA scatter/gather and block-split FP8 index storage."""

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    get_mla_kv_buffer_triton,
    index_k_block_split_scatter,
    set_mla_kv_buffer_triton,
)


@pytest.mark.parametrize("rank,page_size", [(512, 64), (1024, 32)])
@pytest.mark.parametrize("rows", [9, 513])
def test_mla_paged_scatter_gather_preserves_other_fields(rank, page_size, rows):
    pages = (rows + page_size - 1) // page_size + 2
    storage = torch.full((pages, page_size + 3, 1, rank + 64), 7, device="cuda").to(
        torch.float8_e4m3fn
    )
    cache = storage[:, :page_size]
    src = torch.randn(rows, 1, rank + 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm((pages - 1) * page_size, device="cuda")[:rows] + page_size
    mask = torch.arange(rows, device="cuda") % 3 != 0
    set_mla_kv_buffer_triton(
        cache,
        slots,
        src[..., :rank],
        src[..., rank:],
        enable_pdl=False,
        write_mask=mask,
    )
    nope = torch.empty_like(src[..., :rank])
    rope = torch.empty_like(src[..., rank:])
    get_mla_kv_buffer_triton(cache, slots, nope, rope, enable_pdl=False)
    expected = src.to(torch.float8_e4m3fn).to(torch.bfloat16)
    expected[~mask] = 7
    torch.testing.assert_close(
        torch.cat((nope, rope), dim=-1), expected, rtol=0, atol=0
    )
    assert torch.all(storage[:, page_size:].float() == 7)
    assert torch.all(cache[0].float() == 7)


def test_index_scatter_uses_physical_page_stride():
    page_size, head_dim, pages = 64, 128, 4
    width = head_dim + 4
    storage = torch.full(
        (pages, page_size + 8, width), 17, device="cuda", dtype=torch.uint8
    )
    cache = storage[:, :page_size]
    slots = torch.tensor([63, 64, 127, 129, 250], device="cuda")
    keys = torch.randn(5, head_dim, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.arange(1, 6, device="cuda", dtype=torch.float32).unsqueeze(-1)
    mask = torch.tensor([1, 1, 0, 1, 1], device="cuda").bool()
    index_k_block_split_scatter(
        cache,
        keys,
        scales,
        slots,
        page_size=page_size,
        head_dim=head_dim,
        group_size=128,
        write_mask=mask,
    )
    raw_pages = cache.view(pages, page_size * width)
    key_view = raw_pages[:, : page_size * head_dim].view(pages, page_size, head_dim)
    scale_view = (
        raw_pages[:, page_size * head_dim :]
        .view(torch.float32)
        .view(pages, page_size, 1)
    )
    page, offset = slots // page_size, slots % page_size
    assert torch.equal(key_view[page[mask], offset[mask]], keys.view(torch.uint8)[mask])
    torch.testing.assert_close(scale_view[page[mask], offset[mask]], scales[mask])
    assert torch.all(key_view[page[~mask], offset[~mask]] == 17)
    assert torch.all(storage[:, page_size:] == 17)
