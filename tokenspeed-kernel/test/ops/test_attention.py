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
from tokenspeed_kernel import (
    attn_merge_state,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Unified attention tests require an NVIDIA or AMD GPU.",
)


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_prefill(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    seqlens_list = [17, 9, 12]
    max_seqlen = max(seqlens_list)
    cu_seqlens_cpu = [0]
    for seqlen in seqlens_list:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    seqlens = torch.tensor(seqlens_list, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32)
    total_tokens = int(seqlens.sum().item())

    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    out = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
    )

    assert out.shape == q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_extend_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens_list = [63, 48, 17, 80]
    query_seqlens_list = [3, 1, 2, 4]
    max_query_seqlen = max(query_seqlens_list)
    max_cache_seqlen_used = max(
        prefix_len + query_len
        for prefix_len, query_len in zip(prefix_seqlens_list, query_seqlens_list)
    )
    prefix_seqlens = torch.tensor(prefix_seqlens_list, device=device, dtype=torch.int32)
    query_seqlens = torch.tensor(query_seqlens_list, device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + query_seqlens
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())
    total_q = int(query_seqlens.sum().item())

    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.cumsum(query_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0))

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            if tokens_in_block > 0:
                k_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )

    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_query_seqlen,
        max_seqlen_k=max_cache_seqlen_used,
    )

    assert out.shape == q.shape

    triton_out, triton_lse = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=prefix_seqlens,
        max_seqlen_q=max_query_seqlen,
        max_seqlen_k=int(prefix_seqlens.max().item()),
        return_lse=True,
        solution="triton",
    )

    assert triton_out.shape == q.shape
    assert triton_lse.shape == (q.shape[0], q.shape[1])


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
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
    max_cache_seqlen = 256
    prefix_seqlens = torch.tensor([63, 129, 17, 191], device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + 1
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            if tokens_in_block > 0:
                k_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )

    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_cache_seqlen,
    )

    assert out.shape == q.shape


@pytest.mark.parametrize(
    "dtype,num_heads,qk_head_dim,v_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 192, 128, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 192, 128, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_mla_prefill(
    device: str,
    solution: str,
    is_causal: bool,
    dtype: torch.dtype,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
) -> None:
    q_lens = [853, 1045]
    kv_lens = q_lens
    cu_seqlens_q = torch.tensor([0, 853, 1898], device=device, dtype=torch.int32)
    cu_seqlens_kv = cu_seqlens_q
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    q = torch.randn(
        sum(q_lens), num_heads, qk_head_dim, device=device, dtype=init_dtype
    )
    k = torch.randn(
        sum(kv_lens), num_heads, qk_head_dim, device=device, dtype=init_dtype
    )
    v = torch.randn(
        sum(kv_lens), num_heads, v_head_dim, device=device, dtype=init_dtype
    )
    if dtype != init_dtype:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    out, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        return_lse=True,
        solution=solution,
    )

    refs = []
    ref_lses = []
    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        q_i = q[q_offset : q_offset + q_len].float()
        k_i = k[kv_offset : kv_offset + kv_len].float()
        v_i = v[kv_offset : kv_offset + kv_len].float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * softmax_scale
        if is_causal:
            q_pos = torch.arange(q_len, device=device) + max(kv_len - q_len, 0)
            k_pos = torch.arange(kv_len, device=device)
            mask = q_pos[:, None] >= k_pos[None, :]
            scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.einsum("hqk,khd->qhd", probs, v_i))
        ref_lses.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
        q_offset += q_len
        kv_offset += kv_len
    out_ref = torch.cat(refs, dim=0)
    lse_ref = torch.cat(ref_lses, dim=0)

    assert out.shape == (q.shape[0], q.shape[1], v.shape[-1])
    assert lse.shape == (q.shape[0], q.shape[1])
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,num_heads,kv_lora_rank,qk_rope_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 512, 64, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 512, 64, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
def test_mla_decode_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> None:
    batch_size = 2
    q_len = 1
    page_size = 4
    max_seqlen_k = 7
    num_pages = 4
    qk_nope_head_dim = 128
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    q = torch.randn(
        batch_size,
        q_len,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=init_dtype,
    )
    kv_cache = torch.randn(
        num_pages,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=init_dtype,
    )
    if dtype != init_dtype:
        q = q.to(dtype)
        kv_cache = kv_cache.to(dtype)
    page_table = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([5, 7], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    out, lse = mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        return_lse=True,
        solution=solution,
    )

    refs = []
    ref_lses = []
    for batch_idx in range(batch_size):
        kv_rows = []
        for pos in range(int(cache_seqlens[batch_idx].item())):
            page = page_table[batch_idx, pos // page_size]
            kv_rows.append(kv_cache[page, pos % page_size, 0])
        kv = torch.stack(kv_rows).float()
        scores = torch.einsum("hd,kd->hk", q[batch_idx, 0].float(), kv)
        scores = scores * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.matmul(probs, kv[:, :kv_lora_rank]).unsqueeze(0))
        ref_lses.append(torch.logsumexp(scores, dim=-1).unsqueeze(0))
    out_ref = torch.stack(refs, dim=0)
    lse_ref = torch.stack(ref_lses, dim=0)

    assert out.shape == (batch_size, q_len, num_heads, kv_lora_rank)
    assert lse.shape == (batch_size, q_len, num_heads)
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,head_dim,num_heads",
    [(torch.bfloat16, 64, 8)],
)
@pytest.mark.parametrize(
    "solution",
    [None, "triton", "cuda"],
    ids=["auto", "triton", "cuda"],
)
def test_attn_merge_state(
    device: str,
    solution: str | None,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
) -> None:
    if solution == "cuda" and not (platform.is_nvidia and platform.is_hopper_plus):
        pytest.skip("CUDA merge-state kernel is NVIDIA Hopper+-only")

    total_q = 31
    out_a = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    out_b = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    lse_a = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)
    lse_b = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)

    out, lse = attn_merge_state(
        out_a,
        lse_a,
        out_b,
        lse_b,
        solution=solution,
    )

    lse_ref = torch.maximum(lse_a, lse_b)
    weight_a = torch.exp(lse_a - lse_ref)
    weight_b = torch.exp(lse_b - lse_ref)
    denom = weight_a + weight_b
    out_ref = (
        out_a.float() * weight_a[..., None] + out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse_ref = lse_ref + torch.log(denom)

    assert out.shape == out_a.shape
    assert lse.shape == lse_a.shape
    torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)
