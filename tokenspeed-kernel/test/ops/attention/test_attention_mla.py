from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel import (
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_prefill,
)

torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


@pytest.mark.parametrize(
    "dtype,num_heads,qk_head_dim,v_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 192, 128, id="bf16"),
        pytest.param(torch.float8_e4m3fn, 128, 192, 128, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_mla_prefill(
    device: str,
    solution: str,
    is_causal: bool,
    dtype: torch.dtype,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    require,
) -> None:
    require("attention", "mla_prefill", solution, dtype, "q")

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
    "solution,q_dtype,kv_dtype,num_heads,kv_lora_rank,qk_rope_head_dim,batch_size,page_size",
    [
        pytest.param(
            "triton",
            torch.bfloat16,
            torch.bfloat16,
            128,
            512,
            64,
            2,
            4,
            id="triton-bf16",
        ),
        pytest.param(
            "triton",
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            128,
            512,
            64,
            2,
            4,
            id="triton-fp8",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.bfloat16,
            16,
            512,
            64,
            4,
            64,
            id="gluon-bh16bn64",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            1,
            64,
            id="gluon-fp8-bh16bn128-k3",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            8,
            64,
            id="gluon-bf16q-fp8kv-bh16bn128-k3-batch8",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            32,
            64,
            id="gluon-bf16q-fp8kv-bh16bn128-k3-batch32",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            64,
            64,
            id="gluon-bf16q-fp8kv-bh16bn128-k3-batch64",
        ),
        pytest.param(
            "gluon",
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            1,
            64,
            id="gluon-native-fp8q-fp8kv-bh16bn128-k3",
        ),
        pytest.param(
            "gluon",
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            8,
            64,
            id="gluon-native-fp8q-fp8kv-bh16bn128-k3-batch8",
        ),
        pytest.param(
            "gluon",
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            32,
            64,
            id="gluon-native-fp8q-fp8kv-bh16bn128-k3-batch32",
        ),
        pytest.param(
            "gluon",
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            12,
            512,
            64,
            64,
            64,
            id="gluon-native-fp8q-fp8kv-bh16bn128-k3-batch64",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.float8_e5m2,
            12,
            512,
            64,
            1,
            64,
            id="gluon-fp8-e5m2-bh16bn128-k3",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.bfloat16,
            128,
            512,
            64,
            64,
            64,
            id="gluon-bh64",
        ),
    ],
)
def test_mla_decode_with_kvcache(
    device: str,
    solution: str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    batch_size: int,
    page_size: int,
    require,
) -> None:
    require("attention", "mla_decode_with_kvcache", solution, q_dtype, "q")

    q_len = 1
    qk_nope_head_dim = 128
    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    # Runtime seqlens cycled across the batch, spanning sub-page to multi-page
    # relative to page_size (this also leaves some trailing split-K tiles empty).
    seqlen_cycle = [page_size + 1, page_size, 2 * page_size + 1, 1]
    cache_seqlens_list = [
        seqlen_cycle[i % len(seqlen_cycle)] for i in range(batch_size)
    ]
    visible_max_seqlen_k = max(cache_seqlens_list)
    max_seqlen_k = visible_max_seqlen_k
    if solution == "gluon" and kv_dtype in _FP8_DTYPES:
        # K3 reserves a 300K context even when the visible cache is short. This
        # selects 256 split-K workgroups and exercises the empty-split
        # sanitization used by production long-context decode.
        max_seqlen_k = 300_000
    max_pages = (visible_max_seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * max_pages

    q_init_dtype = torch.bfloat16 if q_dtype in _FP8_DTYPES else q_dtype
    kv_init_dtype = torch.bfloat16 if kv_dtype in _FP8_DTYPES else kv_dtype
    q = torch.randn(
        batch_size,
        q_len,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=q_init_dtype,
    )
    kv_cache = torch.randn(
        num_pages,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=kv_init_dtype,
    )
    if q_dtype != q_init_dtype:
        q = q.to(q_dtype)
    if kv_dtype != kv_init_dtype:
        kv_cache = kv_cache.to(kv_dtype)

    cache_seqlens = torch.tensor(cache_seqlens_list, device=device, dtype=torch.int32)
    page_table = torch.arange(num_pages, device=device, dtype=torch.int32).reshape(
        batch_size, max_pages
    )
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
    if q_dtype in _FP8_DTYPES:
        assert out.dtype == torch.bfloat16
    assert lse.shape == (batch_size, q_len, num_heads)
    out_tol = 1e-1 if q_dtype in _FP8_DTYPES or kv_dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


def test_mla_extend_with_kvcache_bf16(device: str, require) -> None:
    require(
        "attention",
        "mla_extend_with_kvcache",
        "gluon",
        torch.bfloat16,
        "q",
    )

    torch.manual_seed(1)
    num_heads = 24
    kv_lora_rank = 512
    rope_dim = 64
    qk_dim = kv_lora_rank + rope_dim
    page_size = 64
    query_lens = [3, 2]
    prefix_lens = [0, 5]
    cache_lens = [
        q_len + prefix for q_len, prefix in zip(query_lens, prefix_lens, strict=True)
    ]
    total_q = sum(query_lens)

    q = torch.randn(
        total_q,
        num_heads,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        len(query_lens),
        page_size,
        1,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_table = torch.arange(
        len(query_lens), device=device, dtype=torch.int32
    ).unsqueeze(1)
    cache_seqlens = torch.tensor(cache_lens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 3, 10], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(128 + rope_dim)

    out = mla_extend_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(cache_lens),
        qk_nope_head_dim=128,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        softmax_scale=softmax_scale,
        is_causal=True,
        solution="gluon",
    )

    refs = []
    q_start = 0
    for batch_idx, (q_len, prefix_len) in enumerate(
        zip(query_lens, prefix_lens, strict=True)
    ):
        kv = kv_cache[batch_idx, : cache_lens[batch_idx], 0].float()
        for query_idx in range(q_len):
            visible_kv = kv[: prefix_len + query_idx + 1]
            scores = torch.einsum(
                "hd,kd->hk", q[q_start + query_idx].float(), visible_kv
            )
            scores *= softmax_scale
            probs = torch.softmax(scores, dim=-1)
            refs.append(torch.matmul(probs, visible_kv[:, :kv_lora_rank]))
        q_start += q_len

    torch.testing.assert_close(out.float(), torch.stack(refs), rtol=8e-2, atol=8e-2)
