# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for Gluon MLA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (  # noqa: E402
    gluon_mla_decode_fp8xfp8_gfx950,
)

_HEADS = 12
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_QK_DIM = _KV_LORA_RANK + _ROPE_DIM
_PAGE_SIZE = 64
_SOFTMAX_SCALE = 192**-0.5


def _make_inputs(seqlen: int, batch_size: int = 1):
    pages_per_batch = (seqlen + _PAGE_SIZE - 1) // _PAGE_SIZE
    pages = batch_size * pages_per_batch
    q = (
        torch.randn(
            batch_size,
            1,
            _HEADS,
            _QK_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    kv_cache = (
        torch.randn(
            pages,
            _PAGE_SIZE,
            1,
            _QK_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(
        batch_size, pages_per_batch
    )
    cache_seqlens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    return q, kv_cache, page_table, cache_seqlens


def _reference(q: torch.Tensor, kv_cache: torch.Tensor, seqlen: int):
    batch_size = q.shape[0]
    kv = kv_cache[:, :, 0].reshape(batch_size, -1, _QK_DIM)[:, :seqlen].float()
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].float(), kv) * _SOFTMAX_SCALE
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhk,bkd->bhd", probs, kv[:, :, :_KV_LORA_RANK]).unsqueeze(1)
    lse = torch.logsumexp(scores, dim=-1).unsqueeze(1)
    return out, lse


def _run(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    return_lse: bool = True,
    out: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
):
    return gluon_mla_decode_fp8xfp8_gfx950(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=(
            int(cache_seqlens.max().item()) if max_seqlen_k is None else max_seqlen_k
        ),
        qk_nope_head_dim=128,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_SOFTMAX_SCALE,
        return_lse=return_lse,
        out=out,
    )


@pytest.mark.parametrize("seqlen", [63, 65, 4096])
def test_native_fp8_mla_matches_fp32_reference(seqlen: int) -> None:
    q, kv_cache, page_table, cache_seqlens = _make_inputs(seqlen)
    out, lse = _run(q, kv_cache, page_table, cache_seqlens, return_lse=True)
    ref_out, ref_lse = _reference(q, kv_cache, seqlen)

    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    torch.testing.assert_close(out.float(), ref_out, rtol=0.12, atol=0.12)
    torch.testing.assert_close(lse, ref_lse, rtol=0.08, atol=0.08)


@pytest.mark.parametrize("batch_size", [2, 7, 8, 32, 64, 65])
def test_native_fp8_mla_supported_batches(batch_size: int) -> None:
    seqlen = 2 * _PAGE_SIZE + 1
    q, kv_cache, page_table, cache_seqlens = _make_inputs(seqlen, batch_size)
    out, lse = _run(q, kv_cache, page_table, cache_seqlens, return_lse=True)
    ref_out, ref_lse = _reference(q, kv_cache, seqlen)

    torch.testing.assert_close(out.float(), ref_out, rtol=0.12, atol=0.12)
    torch.testing.assert_close(lse, ref_lse, rtol=0.08, atol=0.08)


def test_native_fp8_mla_ignores_recycled_tail_nan() -> None:
    seqlen = _PAGE_SIZE + 1
    q, kv_cache, page_table, cache_seqlens = _make_inputs(seqlen)
    clean = _run(q, kv_cache, page_table, cache_seqlens, return_lse=False)

    dirty = kv_cache.clone()
    dirty[-1, 1:] = torch.full(
        dirty[-1, 1:].shape,
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    got = _run(q, dirty, page_table, cache_seqlens, return_lse=False)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, clean, rtol=0, atol=0)


def test_native_fp8_mla_large_pool_uses_safe_64_bit_addresses() -> None:
    batch_size = 32
    first_seqlen = 2049
    last_seqlen = 2112
    max_seqlen_k = 8192
    q, compact_cache, _, cache_seqlens = _make_inputs(last_seqlen, batch_size)

    bytes_per_page = _PAGE_SIZE * _QK_DIM * compact_cache.element_size()
    pool_pages = 0x80000000 // bytes_per_page + 1
    large_cache = torch.empty(
        (pool_pages, _PAGE_SIZE, 1, _QK_DIM),
        dtype=compact_cache.dtype,
        device="cuda",
    )
    active_pages = compact_cache.shape[0]
    large_cache[:active_pages].copy_(compact_cache)
    large_cache[-active_pages:].copy_(compact_cache)

    pages_per_batch = compact_cache.shape[0] // batch_size
    table_pages = max_seqlen_k // _PAGE_SIZE
    page_table = torch.full(
        (batch_size, table_pages),
        pool_pages + 1,
        dtype=torch.int32,
        device="cuda",
    )
    low_pages = torch.arange(active_pages, device="cuda", dtype=torch.int32).view(
        batch_size, pages_per_batch
    )
    high_pages = low_pages + (pool_pages - active_pages)

    got = None
    for active_mapping in (low_pages, high_pages):
        page_table[:, :pages_per_batch].copy_(active_mapping)
        for seqlen in range(first_seqlen, last_seqlen + 1):
            cache_seqlens.fill_(seqlen)
            got = _run(
                q,
                large_cache,
                page_table,
                cache_seqlens,
                return_lse=False,
                max_seqlen_k=max_seqlen_k,
            )
    torch.cuda.synchronize()

    assert got is not None
    ref_out, _ = _reference(q, compact_cache, last_seqlen)
    torch.testing.assert_close(got.float(), ref_out, rtol=0.12, atol=0.12)


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_native_fp8_mla_single_split_cuda_graph_replay(batch_size: int) -> None:
    q, kv_cache, page_table, cache_seqlens = _make_inputs(_PAGE_SIZE, batch_size)
    out = torch.empty(
        (batch_size, 1, _HEADS, _KV_LORA_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    _run(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        return_lse=False,
        out=out,
        max_seqlen_k=_PAGE_SIZE,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(
            q,
            kv_cache,
            page_table,
            cache_seqlens,
            return_lse=False,
            out=out,
            max_seqlen_k=_PAGE_SIZE,
        )
    graph.replay()
    torch.cuda.synchronize()

    ref_out, _ = _reference(q, kv_cache, _PAGE_SIZE)
    assert captured.data_ptr() == out.data_ptr()
    torch.testing.assert_close(captured.float(), ref_out, rtol=0.12, atol=0.12)
