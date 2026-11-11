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
from tokenspeed_kernel.ops.attention import attn_merge_state
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.platform import current_platform

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
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="e4m3-d64"),
        pytest.param(torch.float8_e4m3fn, 128, 8, 2, id="e4m3-d128"),
        pytest.param(torch.float8_e5m2, 64, 8, 2, id="e5m2-d64"),
        pytest.param(torch.float8_e5m2, 128, 8, 2, id="e5m2-d128"),
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
    "is_causal,use_custom_mask",
    [(True, False), (False, False), (True, True)],
    ids=["causal", "noncausal", "custom-future"],
)
def test_mha_prefill_triton_window_bounds(
    device: str, require, is_causal: bool, use_custom_mask: bool
) -> None:
    from tokenspeed_kernel.ops.attention.mha._triton.prefill import (
        prefill_attention_fwd,
    )

    require("attention", "mha_extend_with_kvcache", "triton", torch.bfloat16, "q")
    # A cached suffix with unaligned window boundaries and a partial query tile.
    query_len, prefix_len, window_left, page_size = 257, 577, 511, 64
    kv_len = prefix_len + query_len
    num_pages = (kv_len + page_size - 1) // page_size
    q = _randn((query_len, 4, 64), device=device, dtype=torch.bfloat16)
    k = _randn((num_pages * page_size, 2, 64), device=device, dtype=q.dtype)
    v = _randn(k.shape, device=device, dtype=q.dtype)
    out = torch.empty_like(q)
    page_table = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32)
    q_pos = prefix_len + torch.arange(query_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    mask = q_pos[:, None] - k_pos[None, :] <= window_left
    if use_custom_mask:
        # The custom mask replaces causality and selects a future key tile.
        mask &= k_pos[None, :] == kv_len - 1
    elif is_causal:
        mask &= q_pos[:, None] >= k_pos[None, :]

    # Call the shared kernel entry point because the public API has no custom mask.
    prefill_attention_fwd(
        q_extend=q,
        k_extend=k,
        v_extend=v,
        o_extend=out,
        k_buffer=k.view(num_pages, page_size, 2, 64).flip(0).flatten(0, 1),
        v_buffer=v.view(num_pages, page_size, 2, 64).flip(0).flatten(0, 1),
        cu_seqlens_q=torch.tensor([0, query_len], device=device, dtype=torch.int32),
        cache_seqlens=torch.tensor([kv_len], device=device, dtype=torch.int32),
        custom_mask=mask if use_custom_mask else None,
        is_causal=is_causal,
        max_len_extend=query_len,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=0.0,
        skip_prefix_custom_mask=False,
        sliding_window_size=window_left,
        sinks=None,
        page_table=page_table,
        page_table_stride_b=num_pages,
        page_size=page_size,
        has_kv_cache=True,
        lse_extend=None,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(0, 1),
        k[:kv_len].float().repeat_interleave(2, dim=1).transpose(0, 1),
        v[:kv_len].float().repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=mask,
    ).transpose(0, 1)
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


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
    if solution == "gluon" and current_platform().is_cdna5 and seqlen_q != 1:
        pytest.skip("GFX1250 Gluon decode currently supports one query token")

    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens = torch.tensor([63, 129, 17, 191], dtype=torch.int32)
    cache_seqlens = prefix_seqlens + seqlen_q
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    # Build inputs on CPU for the SDPA reference below.
    q = _randn(
        (batch_size * seqlen_q, num_q_heads, head_dim),
        device="cpu",
        dtype=dtype,
    )

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
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
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    expected_out = None
    if seqlen_q == 1:
        group_size = num_q_heads // num_kv_heads
        expected = []
        for batch_idx, cache_len in enumerate(cache_seqlens.tolist()):
            num_blocks = int(num_blocks_per_seq[batch_idx].item())
            physical_blocks = page_table[batch_idx, :num_blocks].long()
            k_i = k_cache[physical_blocks].reshape(-1, num_kv_heads, head_dim)
            v_i = v_cache[physical_blocks].reshape(-1, num_kv_heads, head_dim)
            k_i = k_i[:cache_len].repeat_interleave(group_size, dim=1)
            v_i = v_i[:cache_len].repeat_interleave(group_size, dim=1)
            expected.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[batch_idx : batch_idx + 1].float().unsqueeze(2),
                    k_i.float().permute(1, 0, 2).unsqueeze(0),
                    v_i.float().permute(1, 0, 2).unsqueeze(0),
                ).squeeze(2)
            )
        expected_out = torch.cat(expected, dim=0)

    q = q.to(device)
    k_cache = k_cache.to(device)
    v_cache = v_cache.to(device)
    page_table = page_table.to(device)
    cache_seqlens = cache_seqlens.to(device)

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
    expected_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    assert out.dtype == expected_dtype
    if expected_out is not None:
        tol = 3e-1 if dtype in _FP8_DTYPES else 3e-2
        torch.testing.assert_close(out.float().cpu(), expected_out, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "cache_seqlen,adversarial_inputs",
    [
        pytest.param(9 * 64, False, id="uneven-peeled-split"),
        pytest.param(64, True, id="inactive-peeled-tiles"),
    ],
)
def test_mha_decode_with_kvcache_gluon_peeled_split(
    device: str,
    cache_seqlen: int,
    adversarial_inputs: bool,
    require,
) -> None:
    require(
        "attention",
        "mha_decode_with_kvcache",
        "gluon",
        torch.bfloat16,
        "q",
    )

    batch_size = 1
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    page_size = 64
    num_pages = 9
    max_seqlen_k = num_pages * page_size

    if adversarial_inputs:
        q = torch.full(
            (batch_size, num_q_heads, head_dim),
            16.0,
            dtype=torch.bfloat16,
        )
        k_cache = torch.full(
            (num_pages, page_size, num_kv_heads, head_dim),
            -16.0,
            dtype=torch.bfloat16,
        )
    else:
        q = torch.randn(
            batch_size,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
        )
        k_cache = torch.randn(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
        )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(num_pages, dtype=torch.int32).reshape(1, num_pages)
    cache_seqlens = torch.full((batch_size,), cache_seqlen, dtype=torch.int32)

    group_size = num_q_heads // num_kv_heads
    k_ref = k_cache.reshape(max_seqlen_k, num_kv_heads, head_dim)[:cache_seqlen]
    v_ref = v_cache.reshape(max_seqlen_k, num_kv_heads, head_dim)[:cache_seqlen]
    k_ref = k_ref.repeat_interleave(group_size, dim=1)
    v_ref = v_ref.repeat_interleave(group_size, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2),
        k_ref.permute(1, 0, 2).unsqueeze(0),
        v_ref.permute(1, 0, 2).unsqueeze(0),
    ).squeeze(2)

    out = mha_decode_with_kvcache(
        q=q.to(device),
        k_cache=k_cache.to(device),
        v_cache=v_cache.to(device),
        page_table=page_table.to(device),
        cache_seqlens=cache_seqlens.to(device),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=1,
        solution="gluon",
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.cpu(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("op", ["decode", "extend"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_fa4_mha_fp8_kvcache_matches_bf16_on_dequant_inputs(
    device: str,
    op: str,
    seqlen_q: int,
    require,
) -> None:
    """Dense-FP8 fa4 path vs the BF16 kernel on the same representable values.

    q/k/v are quantized to e4m3 and the BF16 reference runs on the exact
    dequantized values (e4m3 -> bf16 is lossless), so the comparison isolates
    the kernel's in-fp8 compute error from input quantization error.
    """
    mode = f"mha_{op}_with_kvcache"
    require("attention", mode, "fa4", torch.float8_e4m3fn, "q")

    torch.manual_seed(20260722)
    batch_size = 2
    page_size = 128
    num_q_heads, num_kv_heads, head_dim = 16, 16, 128  # MHA draft shape
    prefix_seqlens = torch.tensor([2305, 2205], device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + seqlen_q
    max_seqlen_k = int(cache_seqlens.max().item())
    blocks_per_seq = (max_seqlen_k + page_size - 1) // page_size
    page_table = torch.arange(
        batch_size * blocks_per_seq, device=device, dtype=torch.int32
    ).view(batch_size, blocks_per_seq)

    q8 = _randn(
        (batch_size * seqlen_q, num_q_heads, head_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k8 = _randn(
        (batch_size * blocks_per_seq, page_size, num_kv_heads, head_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    v8 = _randn(
        (batch_size * blocks_per_seq, page_size, num_kv_heads, head_dim),
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    kwargs = dict(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=max_seqlen_k,
        solution="fa4",
    )
    if op == "extend":
        query_seqlens = torch.full(
            (batch_size,), seqlen_q, device=device, dtype=torch.int32
        )
        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(query_seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        kwargs.update(
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, is_causal=True
        )
        run = mha_extend_with_kvcache
    else:
        run = mha_decode_with_kvcache

    out_fp8 = run(q=q8, k_cache=k8, v_cache=v8, **kwargs)
    out_ref = run(
        q=q8.to(torch.bfloat16),
        k_cache=k8.to(torch.bfloat16),
        v_cache=v8.to(torch.bfloat16),
        **kwargs,
    )

    assert out_fp8.dtype == torch.bfloat16
    torch.testing.assert_close(out_fp8.float(), out_ref.float(), atol=1e-1, rtol=1e-1)


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


def _paged_decode_problem(
    seq_lens: list[int],
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    device: str,
):
    """Random queries and a paged cache whose pages are shuffled across requests."""
    pages = [max(1, -(-seq_len // page_size)) for seq_len in seq_lens]
    page_ids = torch.randperm(sum(pages), device=device, dtype=torch.int64).int()
    page_table = torch.zeros(
        len(seq_lens), max(pages), device=device, dtype=torch.int32
    )
    start = 0
    for req, count in enumerate(pages):
        page_table[req, :count] = page_ids[start : start + count]
        start += count
    shape = (sum(pages), page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(shape, device=device, dtype=torch.bfloat16)
    q = torch.randn(
        len(seq_lens) * rows, num_q_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    cache_seqlens = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return q, k_cache, v_cache, page_table, cache_seqlens


def _causal_decode_reference(
    q, k_cache, v_cache, page_table, cache_seqlens, rows, window_left, sinks, scale
):
    """fp32 ``(out, natural-log lse)``: row ``t`` of ``rows`` sees the first
    ``seq_len - (rows - 1 - t)`` keys (the last ``window_left + 1`` of them)."""
    _, page_size, num_kv_heads, head_dim = k_cache.shape
    group = q.shape[1] // num_kv_heads
    outs, lses = [], []
    for req, seq_len in enumerate(cache_seqlens.tolist()):
        pages = page_table[req, : max(1, -(-seq_len // page_size))].long()
        keys = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        values = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        keys = keys.float().repeat_interleave(group, dim=1)
        values = values.float().repeat_interleave(group, dim=1)
        scores = (
            torch.einsum("rhd,khd->hrk", q[req * rows : (req + 1) * rows].float(), keys)
            * scale
        )
        pos = torch.arange(seq_len, device=q.device)
        end = seq_len - (rows - 1 - torch.arange(rows, device=q.device))
        visible = pos[None, :] < end[:, None]
        if window_left >= 0:
            visible &= pos[None, :] >= end[:, None] - (window_left + 1)
        scores = scores.masked_fill(~visible[None], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        if sinks is not None:
            scores = torch.cat(
                [scores, sinks.float()[:, None, None].expand(-1, rows, 1)], dim=-1
            )
            values = torch.cat([values, values.new_zeros(1, *values.shape[1:])])
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        outs.append(torch.einsum("hrk,khd->rhd", probs, values))
        lses.append(lse.transpose(0, 1))
    return torch.cat(outs), torch.cat(lses)


@pytest.mark.parametrize(
    "rows,num_q_heads,num_kv_heads,head_dim",
    [
        pytest.param(8, 24, 4, 256, id="r8-gqa6-d256"),
        pytest.param(16, 24, 4, 256, id="r16-gqa6-d256"),
        pytest.param(4, 8, 8, 128, id="r4-mha-d128"),
        pytest.param(1, 8, 2, 128, id="r1-gqa4-d128"),
    ],
)
@pytest.mark.parametrize("window_left", [-1, 100], ids=["full", "sliding"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
def test_triton_mha_decode_folds_query_rows(
    device: str,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_left: int,
    has_sink: bool,
    require,
) -> None:
    """Every query row of a request decoded in one pass over its keys matches
    per-row causal attention, including rows whose prefix is empty."""
    require("attention", "mha_decode_with_kvcache", "triton", torch.bfloat16, "q")
    # Shorter than the rows (empty leading rows), page-aligned, and long ones.
    seq_lens = [3, rows, 64, 333, 2049]
    q, k_cache, v_cache, page_table, cache_seqlens = _paged_decode_problem(
        seq_lens, rows, num_q_heads, num_kv_heads, head_dim, 64, device
    )
    sinks = torch.randn(num_q_heads, device=device) if has_sink else None
    scale = head_dim**-0.5
    expected_out, expected_lse = _causal_decode_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, rows, window_left, sinks, scale
    )
    result = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max(seq_lens),
        max_seqlen_q=rows,
        window_left=window_left,
        sinks=sinks,
        return_lse=not has_sink,
        softmax_scale=scale,
        solution="triton",
    )
    out, lse = result if not has_sink else (result, None)
    torch.testing.assert_close(out.float(), expected_out, rtol=2e-2, atol=2e-2)
    if lse is not None:
        # Rows with no key in reach report -inf, so a later merge drops them.
        torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-4)


def test_triton_mha_decode_folded_no_recompile_across_batches(
    device: str, require
) -> None:
    """The folded decode's split count follows the batch size; it is a runtime
    value, so serving's batch sizes share the compiled kernels."""
    from tokenspeed_kernel.ops.attention.mha._triton.decode import (
        _fwd_folded_kernel_stage1,
        _fwd_folded_kernel_stage2,
    )
    from utils import assert_no_triton_compile

    require("attention", "mha_decode_with_kvcache", "triton", torch.bfloat16, "q")
    # One KV pool for every batch, as in serving (the pool size is part of the
    # AMD pointer specialization; the batch must not be).
    max_bs = 256
    q_all, k_cache, v_cache, page_table, cache_seqlens = _paged_decode_problem(
        [17 + 29 * i % 700 for i in range(max_bs)], 16, 24, 4, 256, 64, device
    )

    def run(bs: int, rows: int) -> None:
        mha_decode_with_kvcache(
            q=q_all[: bs * rows],
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table[:bs],
            cache_seqlens=cache_seqlens[:bs],
            max_seqlen_k=page_table.shape[1] * 64,
            max_seqlen_q=rows,
            return_lse=True,
            solution="triton",
        )

    for rows in (8, 16):
        run(1, rows)
    with assert_no_triton_compile(_fwd_folded_kernel_stage1, _fwd_folded_kernel_stage2):
        for bs in (2, 3, 8, 17, 32, 64, 256):
            for rows in (8, 16):
                run(bs, rows)


def _paged_cache(
    lens: list[int], page_size: int, num_kv_heads: int, head_dim: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffled pages for requests of ``lens`` keys: k/v caches and the table."""
    pages = [max(1, -(-n // page_size)) for n in lens]
    order = torch.randperm(sum(pages)).tolist()
    table = torch.zeros(len(lens), max(pages), dtype=torch.int32)
    start = 0
    for i, n in enumerate(pages):
        table[i, :n] = torch.tensor(order[start : start + n], dtype=torch.int32)
        start += n
    shape = (sum(pages), page_size, num_kv_heads, head_dim)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)
    return k.to(device), v.to(device), table.to(device)


def _decode_reference(q, k_cache, v_cache, table, lens, window_left, scale):
    """FP32 decode of one query per request over its last keys, with the LSE."""
    page_size, num_kv_heads, head_dim = k_cache.shape[1:]
    group = q.shape[1] // num_kv_heads
    outs, lses = [], []
    for i, n in enumerate(lens):
        start = max(0, n - window_left - 1) if window_left >= 0 else 0
        rows = table[i].long()
        k = k_cache[rows].reshape(-1, num_kv_heads, head_dim)[start:n].float()
        v = v_cache[rows].reshape(-1, num_kv_heads, head_dim)[start:n].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        logits = torch.einsum("hd,khd->hk", q[i].float(), k) * scale
        if n == start:
            outs.append(torch.zeros_like(q[i], dtype=torch.float32))
            lses.append(torch.full((q.shape[1],), -math.inf, device=q.device))
            continue
        lses.append(torch.logsumexp(logits, dim=1))
        outs.append(torch.einsum("hk,khd->hd", logits.softmax(dim=1), v))
    return torch.stack(outs), torch.stack(lses)


@pytest.mark.parametrize("num_q_heads", [4, 16], ids=["mha", "gqa"])
@pytest.mark.parametrize("window_left", [-1, 300], ids=["full", "window"])
def test_triton_mha_decode_returns_lse_at_every_batch_size(
    device: str, num_q_heads: int, window_left: int, require
) -> None:
    """The Triton decode matches an FP32 reference at every batch size, its
    output alone (the single-row kernels, which split a small batch's keys
    across programs) and with the natural-log LSE (the folded kernel); an
    empty cache yields a zero output and an LSE of -inf, and a new batch size
    or key split reuses the compiled kernels."""
    from tokenspeed_kernel.ops.attention.mha._triton import decode
    from utils import assert_no_triton_compile

    require("attention", "mha_decode_with_kvcache", "triton", torch.bfloat16, "q")
    num_kv_heads, head_dim, page_size = 4, 128, 64
    scale = head_dim**-0.5

    def check(lens: list[int]) -> None:
        k_cache, v_cache, table = _paged_cache(
            lens, page_size, num_kv_heads, head_dim, device
        )
        q = torch.randn(
            len(lens), num_q_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        out_ref, lse_ref = _decode_reference(
            q, k_cache, v_cache, table, lens, window_left, scale
        )
        for return_lse in (False, True):
            result = mha_decode_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=table,
                cache_seqlens=torch.tensor(lens, dtype=torch.int32, device=device),
                max_seqlen_k=max(lens),
                max_seqlen_q=1,
                window_left=window_left,
                return_lse=return_lse,
                softmax_scale=scale,
                solution="triton",
            )
            out = result[0] if return_lse else result
            torch.testing.assert_close(out.float(), out_ref, rtol=2e-2, atol=2e-2)
            if return_lse:
                torch.testing.assert_close(result[1], lse_ref, rtol=1e-4, atol=1e-4)

    # Warm both alignment classes of the page-table stride (16 and 2 pages).
    check([1000, 1])
    check([100])
    kernels = (
        decode._fwd_kernel_stage1,
        decode._fwd_grouped_kernel_stage1,
        decode._fwd_kernel_stage2,
        decode._fwd_folded_kernel_stage1,
        decode._fwd_folded_kernel_stage2,
    )
    with assert_no_triton_compile(*kernels):
        # One request: its keys split across programs.
        check([2049])
        # Long and short requests, and an empty cache.
        check([777, 0, 64, 1500, 3])
        # Enough requests to fill the GPU without splitting.
        check([int(n) for n in torch.randint(1, 900, (512,))])
