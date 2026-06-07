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
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
    trtllm_batch_decode_with_kv_cache_mla,
    trtllm_ragged_attention_deepseek,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

pytestmark = pytest.mark.skipif(
    not (platform.is_blackwell),
    reason="FlashInfer TRTLLM tests require Blackwell GPU.",
)


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        (torch.bfloat16, 128, 8, 8),
        (torch.bfloat16, 128, 16, 2),
    ],
)
def test_mha_prefill(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 3
    seqlens = torch.tensor([834, 278, 768], device=device, dtype=torch.int32)
    total_len = int(seqlens.sum().item())
    max_len = int(seqlens.max().item())
    workspace_buffer = torch.empty(150 * 1024 * 1024, device=device, dtype=torch.uint8)

    query = torch.randn(total_len, num_q_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(total_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(total_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    cum_seq_lens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
    cum_seq_lens = torch.nn.functional.pad(cum_seq_lens, (1, 0))

    out = trtllm_ragged_attention_deepseek(
        query=query,
        key=key,
        value=value,
        workspace_buffer=workspace_buffer,
        seq_lens=seqlens,
        max_q_len=max_len,
        max_kv_len=max_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        o_sf_scale=-1.0,
        batch_size=batch_size,
        window_left=-1,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        enable_pdl=False,
        is_causal=True,
        return_lse=False,
    )

    assert out.shape == query.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        (torch.bfloat16, 128, 8, 8),
        (torch.bfloat16, 128, 16, 2),
    ],
)
def test_mha_prefill_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 3
    page_size = 64
    max_kv_len = 1024
    workspace_buffer = torch.empty(512 * 1024 * 1024, device=device, dtype=torch.uint8)
    seq_lens = torch.tensor([834, 278, 768], device=device, dtype=torch.int32)
    total_q = int(seq_lens.sum().item())
    max_q_len = int(seq_lens.max().item())
    num_blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    query = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=dtype)
    cum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    cum_seq_lens = torch.nn.functional.pad(cum_seq_lens, (1, 0))

    block_tables = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        block_tables[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(seq_lens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(block_tables[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            k_cache[physical_block, :, :tokens_in_block] = torch.randn(
                num_kv_heads,
                tokens_in_block,
                head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache[physical_block, :, :tokens_in_block] = torch.randn(
                num_kv_heads,
                tokens_in_block,
                head_dim,
                device=device,
                dtype=dtype,
            )

    out = trtllm_batch_context_with_kv_cache(
        query=query,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        out_dtype=dtype,
    )

    assert out.shape == query.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 8), (torch.bfloat16, 128, 16, 2)],
)
def test_mha_decode_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    page_size = 64
    max_seq_len = 1024
    workspace_buffer = torch.empty(512 * 1024 * 1024, device=device, dtype=torch.uint8)
    seq_lens = torch.tensor([424, 531, 851, 987], device=device, dtype=torch.int32)
    num_blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_seq_len + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    query = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)

    block_tables = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        block_tables[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(seq_lens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(block_tables[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            k_cache[physical_block, :, :tokens_in_block] = torch.randn(
                num_kv_heads,
                tokens_in_block,
                head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache[physical_block, :, :tokens_in_block] = torch.randn(
                num_kv_heads,
                tokens_in_block,
                head_dim,
                device=device,
                dtype=dtype,
            )

    out = trtllm_batch_decode_with_kv_cache(
        query=query,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        out_dtype=dtype,
    )

    assert out.shape == query.shape


@pytest.mark.parametrize(
    "dtype,num_q_heads,qk_head_dim,kv_lora_rank",
    [(torch.bfloat16, 16, 64, 256)],
)
def test_mla_decode_with_kvcache(
    device: str,
    dtype: torch.dtype,
    num_q_heads: int,
    qk_head_dim: int,
    kv_lora_rank: int,
) -> None:
    batch_size = 4
    q_len_per_req = 1
    page_size = 64
    max_seq_len = 1024
    qk_nope_head_dim = qk_head_dim
    qk_rope_head_dim = qk_head_dim
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim
    query_head_dim = kv_lora_rank + qk_rope_head_dim
    output_head_dim = kv_lora_rank
    workspace_buffer = torch.empty(150 * 1024 * 1024, device=device, dtype=torch.uint8)
    seq_lens = torch.tensor([424, 531, 851, 987], device=device, dtype=torch.int32)
    num_blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_seq_len + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    query = torch.randn(
        batch_size,
        q_len_per_req,
        num_q_heads,
        query_head_dim,
        device=device,
        dtype=dtype,
    )

    block_tables = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        block_tables[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    kv_cache = torch.zeros(
        total_num_blocks,
        1,
        page_size,
        kv_cache_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(seq_lens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(block_tables[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            kv_cache[physical_block, 0, :tokens_in_block] = torch.randn(
                tokens_in_block,
                kv_cache_dim,
                device=device,
                dtype=dtype,
            )

    out = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / math.sqrt(query_head_dim),
    )

    assert out.shape == (batch_size, q_len_per_req, num_q_heads, output_head_dim)
