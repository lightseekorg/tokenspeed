from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel import (
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


@pytest.mark.parametrize(
    "dtype,num_heads,qk_head_dim,v_head_dim",
    [
        pytest.param(torch.float16, 128, 192, 128, id="fp16"),
        pytest.param(torch.bfloat16, 128, 192, 128, id="bf16"),
        pytest.param(torch.float8_e4m3fn, 128, 192, 128, id="fp8-e4m3"),
        pytest.param(torch.float8_e5m2, 128, 192, 128, id="fp8-e5m2"),
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
    out_tol = 2e-1 if dtype in _FP8_DTYPES else 8e-2
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
            torch.float16,
            torch.float16,
            16,
            512,
            64,
            4,
            64,
            id="gluon-gfx1250-fp16",
        ),
        pytest.param(
            "gluon",
            torch.float8_e5m2,
            torch.float8_e5m2,
            16,
            512,
            64,
            4,
            64,
            id="gluon-gfx1250-fp8-e5m2",
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
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.bfloat16,
            64,
            512,
            64,
            1,
            64,
            id="gluon-h64-small-b1",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.bfloat16,
            64,
            512,
            64,
            2,
            64,
            id="gluon-h64-small-b2",
        ),
        pytest.param(
            "gluon",
            torch.bfloat16,
            torch.bfloat16,
            64,
            512,
            64,
            4,
            64,
            id="gluon-h64-small-b4",
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
    if q_dtype != kv_dtype and not platform.is_cdna4:
        pytest.skip("only test mixed query/cache dtypes on gfx950")

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
    elif (
        solution == "gluon"
        and q_dtype == torch.bfloat16
        and kv_dtype == torch.bfloat16
        and num_heads == 64
        and batch_size in (1, 2, 4)
    ):
        max_seqlen_k = 80_000
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


def test_mla_decode_noncausal_block_sliding_window_matches_reference_and_captures(
    device: str,
) -> None:
    torch.manual_seed(91)
    block_size = 8
    context_len = 177
    cache_len = context_len + block_size
    window_left = 129
    page_size = 64
    num_heads = 2
    kv_lora_rank = 8
    qk_rope_head_dim = 4
    qk_nope_head_dim = 4
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    max_pages = math.ceil(cache_len / page_size)

    q = torch.randn(
        block_size,
        1,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        max_pages,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_table = torch.arange(max_pages, device=device, dtype=torch.int32).repeat(
        block_size, 1
    )
    cache_seqlens = torch.full(
        (block_size,), cache_len, device=device, dtype=torch.int32
    )
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    def run(query: torch.Tensor) -> torch.Tensor:
        return mla_decode_with_kvcache(
            q=query,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=cache_len,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            window_left=window_left,
            noncausal_block_size=block_size,
        )

    output = run(q)
    dense_kv = kv_cache.reshape(-1, qk_head_dim)[:cache_len].float()
    expected = []
    for block_position in range(block_size):
        start = max(
            0,
            context_len - window_left + block_position,
        )
        visible = dense_kv[start:cache_len]
        scores = torch.einsum("hd,kd->hk", q[block_position, 0].float(), visible)
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        expected.append(torch.matmul(probs, visible[:, :kv_lora_rank]))
    expected_output = torch.stack(expected).unsqueeze(1)
    torch.testing.assert_close(output.float(), expected_output, rtol=8e-2, atol=8e-2)

    graph_output = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output.copy_(run(q))
    eager_output = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float8_e4m3fn, id="fp8-e4m3"),
        pytest.param(torch.float8_e5m2, id="fp8-e5m2"),
    ],
)
def test_mla_extend_with_kvcache(device: str, dtype: torch.dtype, require) -> None:
    require(
        "attention",
        "mla_extend_with_kvcache",
        "gluon",
        dtype,
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
    ).to(dtype)
    kv_cache = torch.randn(
        len(query_lens),
        page_size,
        1,
        qk_dim,
        device=device,
        dtype=torch.bfloat16,
    ).to(dtype)
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
    assert out.dtype == torch.bfloat16

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

    tol = 1.5e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), torch.stack(refs), rtol=tol, atol=tol)


def _run_fixed_bf16_mla_decode_case(
    *,
    device: str,
    override: str,
    cache_seqlens_list: list[int],
    return_lse: bool,
    comparison_override: str | None = None,
    use_out: bool = False,
) -> None:
    batch_size = len(cache_seqlens_list)
    num_heads = 64
    page_size = 64
    max_seqlen_k = 80_000
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    live_max_seqlen_k = max(cache_seqlens_list)
    live_max_pages = (live_max_seqlen_k + page_size - 1) // page_size

    q = torch.randn(
        batch_size,
        1,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        batch_size * live_max_pages,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_seqlens = torch.tensor(
        cache_seqlens_list,
        device=device,
        dtype=torch.int32,
    )
    page_table = torch.zeros(
        batch_size,
        (max_seqlen_k + page_size - 1) // page_size,
        device=device,
        dtype=torch.int32,
    )
    for batch_idx, seqlen in enumerate(cache_seqlens_list):
        page_count = (seqlen + page_size - 1) // page_size
        page_start = batch_idx * live_max_pages
        page_table[batch_idx, :page_count] = torch.arange(
            page_start,
            page_start + page_count,
            device=device,
            dtype=torch.int32,
        )
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    out_buffer = None
    if use_out:
        out_buffer = torch.full(
            (batch_size, 1, num_heads, kv_lora_rank),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        )

    result = mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        out=out_buffer,
        override=override,
    )
    if return_lse:
        out, lse = result
    else:
        assert isinstance(result, torch.Tensor)
        out = result
        lse = None
    if out_buffer is not None:
        assert out is out_buffer

    if comparison_override is not None:
        comparison = mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
            override=comparison_override,
        )
        if return_lse:
            comparison_out, comparison_lse = comparison
        else:
            assert isinstance(comparison, torch.Tensor)
            comparison_out = comparison
            comparison_lse = None
        torch.testing.assert_close(
            comparison_out.float(),
            out.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        if return_lse:
            assert lse is not None
            assert comparison_lse is not None
            torch.testing.assert_close(
                comparison_lse,
                lse,
                rtol=2e-2,
                atol=2e-2,
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

    assert out.shape == (batch_size, 1, num_heads, kv_lora_rank)
    torch.testing.assert_close(out.float(), out_ref, rtol=8e-2, atol=8e-2)
    if return_lse:
        assert lse is not None
        lse_ref = torch.stack(ref_lses, dim=0)
        assert lse.shape == (batch_size, 1, num_heads)
        torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "override",
    [
        pytest.param(
            "gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
            id="bh16-multiblock",
        ),
        pytest.param(
            "gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
            id="bh64-small",
        ),
    ],
)
@pytest.mark.parametrize(
    "cache_seqlens_list",
    [
        pytest.param([64], id="b1"),
        pytest.param([63, 65], id="b2"),
        pytest.param([1, 64, 65, 129], id="b4"),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True], ids=["output", "lse"])
def test_mla_decode_small_batch_fixed_entrypoints_match_reference(
    device: str,
    override: str,
    cache_seqlens_list: list[int],
    return_lse: bool,
    require,
) -> None:
    if not platform.is_cdna4:
        pytest.skip("fixed gfx950 MLA decode entrypoints require CDNA4")
    require(
        "attention",
        "mla_decode_with_kvcache",
        "gluon",
        torch.bfloat16,
        "q",
    )
    _run_fixed_bf16_mla_decode_case(
        device=device,
        override=override,
        cache_seqlens_list=cache_seqlens_list,
        return_lse=return_lse,
    )


@pytest.mark.parametrize(
    "cache_seqlens_list",
    [
        pytest.param([64], id="b1"),
        pytest.param([63, 65], id="b2"),
        pytest.param([1, 64, 65, 129], id="b4"),
    ],
)
def test_mla_decode_small_batch_fixed_entrypoints_match_each_other(
    device: str,
    cache_seqlens_list: list[int],
    require,
) -> None:
    if not platform.is_cdna4:
        pytest.skip("fixed gfx950 MLA decode entrypoints require CDNA4")
    require(
        "attention",
        "mla_decode_with_kvcache",
        "gluon",
        torch.bfloat16,
        "q",
    )
    _run_fixed_bf16_mla_decode_case(
        device=device,
        override="gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
        comparison_override="gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
        cache_seqlens_list=cache_seqlens_list,
        return_lse=True,
    )


@pytest.mark.parametrize(
    "override",
    [
        pytest.param(
            "gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
            id="bh16-multiblock",
        ),
        pytest.param(
            "gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
            id="bh64-small",
        ),
    ],
)
def test_mla_decode_small_batch_fixed_entrypoints_use_out(
    device: str,
    override: str,
    require,
) -> None:
    if not platform.is_cdna4:
        pytest.skip("fixed gfx950 MLA decode entrypoints require CDNA4")
    require(
        "attention",
        "mla_decode_with_kvcache",
        "gluon",
        torch.bfloat16,
        "q",
    )
    _run_fixed_bf16_mla_decode_case(
        device=device,
        override=override,
        cache_seqlens_list=[63, 65],
        return_lse=True,
        use_out=True,
    )


def test_mla_decode_with_kvcache_projected_value_matches_split_and_captures(
    device: str,
) -> None:
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("K3 fused MLA epilogue requires CDNA4 or CDNA5")
    from tokenspeed_kernel import mla_project_value

    torch.manual_seed(67)
    q = torch.randn(1, 1, 12, 576, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    kv_cache = torch.randn(64, 64, 1, 576, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    page_table = torch.arange(64, device=device, dtype=torch.int32).view(1, 64)
    cache_seqlens = torch.tensor([4096], device=device, dtype=torch.int32)
    weight = torch.randn(12, 512, 128, device=device, dtype=torch.bfloat16)
    gate = torch.randn(1, 1536, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(gate)
    projected_override = (
        "gluon_mla_decode_projected_value_gfx1250" if platform.is_cdna5 else None
    )
    attention = mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=8192,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0 / math.sqrt(192),
        solution="gluon",
    )
    expected = mla_project_value(
        attention.reshape(1, 12, 512),
        weight,
        gate=gate,
    )

    mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=8192,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0 / math.sqrt(192),
        value_weight=weight,
        gate=gate,
        out=output,
        override=projected_override,
    )
    eager_output = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=8192,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=1.0 / math.sqrt(192),
            value_weight=weight,
            gate=gate,
            out=output,
            override=projected_override,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, eager_output, atol=0, rtol=0)
    # Split scheduling and reducer order can differ from the generic
    # composition. Validate within the established FP8 reduction envelope.
    torch.testing.assert_close(output, expected, atol=0.125, rtol=0.05)


@pytest.mark.parametrize("use_gate", [False, True])
def test_mla_decode_with_kvcache_composes_projected_value_fallback(
    monkeypatch: pytest.MonkeyPatch,
    use_gate: bool,
) -> None:
    import tokenspeed_kernel.ops.attention as attention_ops
    from tokenspeed_kernel.selection import NoKernelFoundError

    latent = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4) / 16
    q = torch.empty(2, 1, 3, 6)
    kv_cache = torch.empty(1, 8, 1, 6)
    page_table = torch.zeros(2, 1, dtype=torch.int32)
    cache_seqlens = torch.ones(2, dtype=torch.int32)
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 32
    raw_gate = torch.linspace(-1, 1, 12).reshape(2, 6)
    gate = raw_gate if use_gate else None
    output = torch.empty_like(raw_gate)

    def select_decode_kernel(*args, **kwargs):
        if args[1] == "mla_project_value":
            raise NoKernelFoundError
        if args[1] == "mla_decode_projected_value":
            assert kwargs["traits"]["support_logit_cap"] is True
            raise NoKernelFoundError
        assert args[1] == "mla_decode_with_kvcache"
        split_decode.name = "split_decode"
        return split_decode

    def split_decode(**kwargs):
        assert kwargs["kv_lora_rank"] == 4
        assert kwargs["qk_rope_head_dim"] == 2
        assert kwargs["logit_cap"] == 2.5
        return latent

    monkeypatch.setattr(attention_ops, "select_kernel", select_decode_kernel)

    returned = attention_ops.mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=8,
        qk_nope_head_dim=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        softmax_scale=1.0,
        value_weight=weight,
        gate=gate,
        out=output,
        logit_cap=2.5,
    )
    expected = torch.bmm(
        latent.reshape(2, 3, 4).transpose(0, 1).contiguous(),
        weight,
    )
    expected = expected.transpose(0, 1).reshape_as(output)
    if gate is not None:
        expected = expected * torch.sigmoid(gate)
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, expected)


def test_mla_decode_with_kvcache_rejects_invalid_projected_out() -> None:
    q = torch.empty(2, 1, 3, 6)
    kv_cache = torch.empty(1, 8, 1, 6)
    page_table = torch.zeros(2, 1, dtype=torch.int32)
    cache_seqlens = torch.ones(2, dtype=torch.int32)
    weight = torch.empty(3, 4, 2)

    def call(out: torch.Tensor) -> None:
        mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=8,
            qk_nope_head_dim=2,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            softmax_scale=1.0,
            value_weight=weight,
            out=out,
        )

    with pytest.raises(ValueError, match="contiguous and colocated"):
        call(torch.empty(2, 12)[:, ::2])
    with pytest.raises(ValueError, match="contiguous and colocated"):
        call(torch.empty(2, 6, device="meta"))

    with pytest.raises(ValueError, match="gate requires value_weight"):
        mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=8,
            qk_nope_head_dim=2,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            softmax_scale=1.0,
            gate=torch.empty(2, 6),
        )
    with pytest.raises(ValueError, match="does not support return_lse"):
        mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=8,
            qk_nope_head_dim=2,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            softmax_scale=1.0,
            return_lse=True,
            value_weight=weight,
            out=torch.empty(2, 6),
        )


def test_mla_project_value_fallback_preserves_fp32_gate_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed_kernel.ops.attention as attention_ops
    from tokenspeed_kernel.selection import NoKernelFoundError

    def no_kernel(*args, **kwargs):
        raise NoKernelFoundError

    monkeypatch.setattr(attention_ops, "select_kernel", no_kernel)
    attention = torch.tensor([[[-3.109375]]], dtype=torch.bfloat16)
    weight = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    gate = torch.tensor([[6.21875]], dtype=torch.bfloat16)

    output = attention_ops.mla_project_value(attention, weight, gate=gate)
    expected = (attention.float().reshape(1, 1) * gate.float().sigmoid()).to(
        torch.bfloat16
    )

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert not torch.equal(output, attention.reshape(1, 1) * gate.sigmoid())


def test_mla_project_value_nvidia_fallback_writes_projection_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed_kernel.ops.attention as attention_ops
    from tokenspeed_kernel.selection import NoKernelFoundError

    def no_kernel(*args, **kwargs):
        raise NoKernelFoundError

    original_bmm = torch.bmm
    bmm_out = None

    def record_bmm(
        input: torch.Tensor,
        mat2: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal bmm_out
        bmm_out = out
        return original_bmm(input, mat2, out=out)

    monkeypatch.setattr(attention_ops, "select_kernel", no_kernel)
    monkeypatch.setattr(
        attention_ops,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    monkeypatch.setattr(torch, "bmm", record_bmm)
    attention = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4, 5)
    expected = torch.einsum("bhl,hlv->bhv", attention, weight).reshape(2, 15)

    output = attention_ops.mla_project_value(attention, weight)

    assert bmm_out is not None
    torch.testing.assert_close(output, expected)


def test_mla_normalize_project_query_cuda_fallback(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed_kernel.ops.attention as attention_ops
    from tokenspeed_kernel.selection import NoKernelFoundError

    def no_kernel(*args, **kwargs):
        raise NoKernelFoundError

    monkeypatch.setattr(attention_ops, "select_kernel", no_kernel)
    query = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
    kv = torch.randn(2, 32, device=device, dtype=torch.bfloat16)
    query_weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn(32, device=device, dtype=torch.bfloat16)
    projection_weight = torch.randn(48, 64, device=device, dtype=torch.bfloat16)
    eps = 1e-6
    query_fp32 = query.float()
    expected_query = (
        query_fp32
        * torch.rsqrt(query_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * query_weight.float()
    ).to(torch.bfloat16)
    kv_fp32 = kv.float()
    expected_kv = (
        kv_fp32
        * torch.rsqrt(kv_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * kv_weight.float()
    ).to(torch.bfloat16)
    expected_output = expected_query @ projection_weight.t()

    returned_query, absorbed_query = attention_ops.mla_normalize_project_query(
        query,
        kv,
        query_weight,
        kv_weight,
        projection_weight,
        eps=eps,
        prepare_absorbed_query=True,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
    )

    assert absorbed_query is None
    torch.testing.assert_close(kv, expected_kv, atol=0, rtol=0)
    torch.testing.assert_close(returned_query, expected_output, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("heads", [12, 16])
def test_mla_normalize_project_query_split_output(
    device: str,
    require,
    heads: int,
) -> None:
    import tokenspeed_kernel.ops.attention as attention_ops

    require(
        "attention",
        "mla_normalize_project_query",
        "gluon",
        torch.bfloat16,
        "query",
    )
    query = torch.randn(1, 1536, device=device, dtype=torch.bfloat16)
    kv = torch.randn(1, 512, device=device, dtype=torch.bfloat16)
    query_weight = torch.randn(1536, device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device=device, dtype=torch.bfloat16)
    projection_weight = torch.randn(
        heads * 192, 1536, device=device, dtype=torch.bfloat16
    )
    expected_kv = kv.clone()
    eps = 1e-6

    returned_query, absorbed_query = attention_ops.mla_normalize_project_query(
        query,
        kv,
        query_weight,
        kv_weight,
        projection_weight,
        eps=eps,
        prepare_absorbed_query=True,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        solution="gluon",
    )

    query_fp32 = query.float()
    normalized_query = (
        query_fp32
        * torch.rsqrt(query_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * query_weight.float()
    ).to(torch.bfloat16)
    projected = (normalized_query.float() @ projection_weight.float().t()).to(
        torch.bfloat16
    )
    projected = projected.view(1, heads, 192)
    kv_fp32 = expected_kv.float()
    expected_kv = (
        kv_fp32
        * torch.rsqrt(kv_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * kv_weight.float()
    ).to(torch.bfloat16)

    assert absorbed_query is not None
    torch.testing.assert_close(
        returned_query, projected[..., :128], atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        absorbed_query[..., 512:],
        projected[..., 128:],
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(kv, expected_kv, atol=0, rtol=0)


def test_gfx950_k3_decode_split_policy() -> None:
    """Bound K3's 16K-capacity BF16-Q decode without changing long contexts."""
    if not current_platform().is_cdna4:
        pytest.skip("K3 FP8 MLA split policy targets CDNA4")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        _select_num_kv_splits_bh16bn128,
    )

    assert (
        _select_num_kv_splits_bh16bn128(
            batch=1,
            max_seqlen_k=16_384,
            block_n=128,
        )
        == 16
    )
    assert (
        _select_num_kv_splits_bh16bn128(
            batch=1,
            max_seqlen_k=300_000,
            block_n=128,
        )
        == 128
    )
