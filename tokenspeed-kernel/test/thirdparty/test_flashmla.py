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

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)


@pytest.mark.skipif(not platform.is_hopper, reason="Requires Hopper GPU")
@pytest.mark.parametrize(
    "dtype,num_q_heads,head_dim_v,qk_rope_head_dim",
    [(torch.bfloat16, 16, 512, 64)],
)
def test_mla_decode_with_paged_kvcache(
    device: str,
    dtype: torch.dtype,
    num_q_heads: int,
    head_dim_v: int,
    qk_rope_head_dim: int,
) -> None:
    batch_size = 4
    q_len_per_req = 1
    page_size = 64
    max_seq_len = 1024
    kv_cache_dim = head_dim_v + qk_rope_head_dim
    cache_seqlens = torch.tensor([424, 531, 851, 987], device=device, dtype=torch.int32)
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_seq_len + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q = torch.randn(
        batch_size,
        q_len_per_req,
        num_q_heads,
        kv_cache_dim,
        device=device,
        dtype=dtype,
    )

    block_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        block_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        1,
        kv_cache_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(block_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            k_cache[physical_block, :tokens_in_block] = torch.randn(
                tokens_in_block,
                1,
                kv_cache_dim,
                device=device,
                dtype=dtype,
            )

    tile_scheduler_metadata, _ = get_mla_metadata()

    out, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        softmax_scale=1.0 / math.sqrt(kv_cache_dim),
        causal=True,
    )

    assert out.shape == (batch_size, q_len_per_req, num_q_heads, head_dim_v)
    assert lse.shape == (batch_size, num_q_heads, q_len_per_req)
