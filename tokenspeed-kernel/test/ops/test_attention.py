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
)

torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


def _randn(shape: tuple[int, ...], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    tensor = torch.randn(shape, device=device, dtype=init_dtype)
    if dtype != init_dtype:
        tensor = tensor.to(dtype)
    return tensor


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16-d64"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "gluon"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("is_sliding", [False, True], ids=["full", "sliding"])
def test_mha_prefill(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    has_sink: bool,
    is_sliding: bool,
    require,
) -> None:
    require("attention", "mha_prefill", solution, dtype, "q")
    if solution == "fa4" and (has_sink or is_sliding):
        pytest.skip("FA4 MHA prefill does not support sinks or sliding window")

    seqlens_list = [851, 914, 1053]
    max_seqlen = max(seqlens_list)
    cu_seqlens_cpu = [0]
    for seqlen in seqlens_list:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    seqlens = torch.tensor(seqlens_list, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32)
    total_tokens = int(seqlens.sum().item())

    q = _randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    k = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    sinks = _randn((num_q_heads,), device=device, dtype=q.dtype) if has_sink else None
    window_left = 127 if is_sliding else -1

    out = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        sinks=sinks,
        solution=solution,
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16-d64"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize("window_left", [-1, 127], ids=["full", "sliding"])
def test_mha_prefill_lse(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    window_left: int,
    require,
) -> None:
    require("attention", "mha_prefill", solution, dtype, "q")

    seqlens_list = [851, 914, 1053]
    max_seqlen = max(seqlens_list)
    cu_seqlens_cpu = [0]
    for seqlen in seqlens_list:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    cu_seqlens = torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32)
    total_tokens = cu_seqlens_cpu[-1]

    q = _randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    k = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(head_dim)
    group = num_q_heads // num_kv_heads

    out, lse = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        return_lse=True,
        solution=solution,
    )

    assert out.shape == q.shape
    assert lse.shape == (total_tokens, num_q_heads)

    # Reference: natural-log log-sum-exp over a causal MHA prefill.
    ref_outs = []
    ref_lses = []
    for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        q_i = q[start:end].float()
        k_i = k[start:end].float()
        k_exp = k_i.repeat_interleave(group, dim=1)
        v_exp = v[start:end].float().repeat_interleave(group, dim=1)
        seq_len = end - start
        scores = torch.einsum("qhd,khd->hqk", q_i, k_exp) * sm_scale
        pos = torch.arange(seq_len, device=device)
        mask = pos[:, None] >= pos[None, :]
        if window_left >= 0:
            mask &= (pos[:, None] - pos[None, :]) <= window_left
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref_outs.append(torch.einsum("hqk,khd->qhd", probs, v_exp))
        ref_lses.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
    out_ref = torch.cat(ref_outs, dim=0)
    lse_ref = torch.cat(ref_lses, dim=0)

    torch.testing.assert_close(out.float(), out_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
        pytest.param(torch.float8_e4m3fn, 128, 8, 2, id="fp8-d128"),
        pytest.param(torch.float8_e5m2, 64, 8, 2, id="fp8-e5m2"),
        pytest.param(torch.float8_e5m2, 128, 8, 2, id="fp8-e5m2-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer", "gluon"])
def test_mha_extend_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_extend_with_kvcache", solution, dtype, "q")

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

    q = _randn((total_q, num_q_heads, head_dim), device=device, dtype=dtype)
    cu_seqlens_q = torch.cumsum(query_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0))
    cu_seqlens_kv = torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_kv = torch.nn.functional.pad(cu_seqlens_kv, (1, 0))

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
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_query_seqlen,
        max_seqlen_k=max_cache_seqlen_used,
        solution=solution,
    )

    assert out.shape == q.shape

    if solution in ("triton", "gluon"):
        lse_out, lse = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=prefix_seqlens,
            max_seqlen_q=max_query_seqlen,
            max_seqlen_k=int(prefix_seqlens.max().item()),
            return_lse=True,
            solution=solution,
        )

        assert lse_out.shape == q.shape
        assert lse.shape == (q.shape[0], q.shape[1])


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.bfloat16, 128, 8, 2, id="bf16-d128"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
        pytest.param(torch.float8_e4m3fn, 128, 8, 2, id="fp8-d128"),
        pytest.param(torch.float8_e5m2, 64, 8, 2, id="fp8-e5m2"),
        pytest.param(torch.float8_e5m2, 128, 8, 2, id="fp8-e5m2-d128"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer", "gluon"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_mha_decode_with_kvcache(
    device: str,
    solution: str,
    seqlen_q: int,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")

    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens = torch.tensor([63, 129, 17, 191], device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + seqlen_q
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q = _randn(
        (batch_size * seqlen_q, num_q_heads, head_dim),
        device=device,
        dtype=dtype,
    )

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
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_cache_seqlen,
        max_seqlen_q=seqlen_q,
        solution=solution,
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,head_dim,num_heads",
    [(torch.bfloat16, 64, 8)],
)
@pytest.mark.parametrize(
    "solution",
    ["triton", "cuda"],
)
@pytest.mark.parametrize("inplace", [False, True], ids=["out-of-place", "in-place"])
def test_attn_merge_state(
    device: str,
    solution: str,
    inplace: bool,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    require,
) -> None:
    require("attention", "attn_merge_state", solution, dtype, "out_a")

    total_q = 31
    out_a = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    out_b = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    lse_a = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)
    lse_b = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)
    out_a_ref_input = out_a.clone()
    lse_a_ref_input = lse_a.clone()

    out, lse = attn_merge_state(
        out_a,
        lse_a,
        out_b,
        lse_b,
        inplace=inplace,
        solution=solution,
    )

    lse_ref = torch.maximum(lse_a_ref_input, lse_b)
    weight_a = torch.exp(lse_a_ref_input - lse_ref)
    weight_b = torch.exp(lse_b - lse_ref)
    denom = weight_a + weight_b
    out_ref = (
        out_a_ref_input.float() * weight_a[..., None]
        + out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse_ref = lse_ref + torch.log(denom)

    if inplace:
        assert out.data_ptr() == out_a.data_ptr()
        assert lse.data_ptr() == lse_a.data_ptr()
    else:
        torch.testing.assert_close(out_a, out_a_ref_input)
        torch.testing.assert_close(lse_a, lse_a_ref_input)

    assert out.shape == out_a.shape
    assert lse.shape == lse_a.shape
    torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)
