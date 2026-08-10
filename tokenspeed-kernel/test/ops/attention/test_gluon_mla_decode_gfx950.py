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
