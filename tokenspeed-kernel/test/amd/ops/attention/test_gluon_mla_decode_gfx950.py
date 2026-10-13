# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import math

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

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


def _query_block_reference(q, cache, table, lengths, softmax_scale=_SOFTMAX_SCALE):
    output = torch.zeros((*q.shape[:-1], _KV_LORA_RANK), device=q.device)
    lse = torch.full(q.shape[:-1], -float("inf"), device=q.device)
    for request, length in enumerate(lengths.tolist()):
        pages = table[request, : math.ceil(length / cache.shape[1])].long()
        kv = cache[pages].reshape(-1, _QK_DIM)
        for position in range(q.shape[1]):
            visible = max(0, length - q.shape[1] + position + 1)
            if visible:
                values = kv[:visible].float()
                scores = q[request, position].float() @ values.T * softmax_scale
                output[request, position] = scores.softmax(-1) @ values[:, :512]
                lse[request, position] = torch.logsumexp(scores, -1)
    return output, lse


def _query_block_decode(
    q, cache, table, lengths, *, softmax_scale=_SOFTMAX_SCALE, **kwargs
):
    from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache

    return mla_decode_with_kvcache(
        q=q,
        kv_cache=cache,
        page_table=table,
        cache_seqlens=lengths,
        qk_nope_head_dim=128,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=softmax_scale,
        solution="gluon",
        **kwargs,
    )


def _make_query_block_inputs(cache_lengths, heads, page, width):
    gen = torch.Generator(device="cuda").manual_seed(129)
    pages_per_request = [math.ceil(length / page) for length in cache_lengths]
    num_pages = sum(pages_per_request)
    cache = (
        torch.randn(num_pages + 1, page, 1, _QK_DIM, generator=gen, device="cuda")
        * 0.25
    ).to(torch.float8_e4m3fn)
    cache[-1] = float("nan")
    permutation = torch.randperm(num_pages, generator=gen, device="cuda")
    table = torch.full(
        (len(cache_lengths), max(pages_per_request)),
        num_pages,
        device="cuda",
        dtype=torch.int32,
    )
    offset = 0
    for request, (length, pages) in enumerate(zip(cache_lengths, pages_per_request)):
        table[request, :pages] = permutation[offset : offset + pages]
        offset += pages
        if length % page:
            cache[table[request, pages - 1], length % page :] = float("nan")
    q = (
        torch.randn(
            len(cache_lengths), width, heads, _QK_DIM, generator=gen, device="cuda"
        )
        * 0.25
    ).to(cache.dtype)
    lengths = torch.tensor(cache_lengths, device="cuda", dtype=torch.int32)
    return q, cache, table, lengths


@pytest.mark.parametrize(
    "heads,page,width,cache_lengths,small_weights",
    [
        (12, 64, 4, [0, 3, 1025, 4095, 4096, 16385], False),
        (12, 64, 4, [0, 1, 3, 63, 64, 65, 1025, 4095], False),
        (8, 64, 8, [0, 7, 4095, 4096, 16385], False),
        (12, 64, 5, [0, 4, 4095, 4096, 16385], False),
        (24, 64, 5, [0] * 12 + [4095, 4096, 37888, 37889], False),
        (16, 64, 15, [0, 14, 1025], False),
        (24, 128, 6, [0, 5, 1025], False),
        (128, 256, 4, [1025, 3] + [0] * 15, False),
        (12, 64, 6, [0, 3, 4095, 4096, 51200, 51201, 65536, 65537], False),
        (24, 64, 4, [0] * 12 + [4095, 4096, 31744, 31745], False),
        pytest.param(12, 64, 4, [8192], True, id="small-weights"),
        pytest.param(12, 64, 4, [16384], True, id="reuse-small-weights"),
        pytest.param(12, 64, 6, [60000] + [0] * 7, True, id="wide-small-weights"),
    ],
)
def test_query_block_mla_matches_reference(
    heads, page, width, cache_lengths, small_weights
):
    q, cache, table, lengths = _make_query_block_inputs(
        cache_lengths, heads, page, width
    )
    scale = _SOFTMAX_SCALE
    if small_weights:
        # Constant values make the exact average one, even when a logit spike
        # pushes the other weights below the FP8 representable range.
        q.zero_()
        q[..., _KV_LORA_RANK] = 1
        cache.zero_()
        cache[..., 0] = 1
        cache[table[0, 0], 0, 0, _KV_LORA_RANK] = 8
        scale = 1.0
    output = torch.empty(
        *q.shape[:-1], _KV_LORA_RANK, device="cuda", dtype=torch.bfloat16
    )
    result, lse = _query_block_decode(
        q,
        cache,
        table,
        lengths,
        max_seqlen_k=max(cache_lengths),
        softmax_scale=scale,
        return_lse=True,
        out=output,
    )
    ref, ref_lse = _query_block_reference(q, cache, table, lengths, scale)
    assert result.data_ptr() == output.data_ptr()
    assert (output.float() - ref).norm() / ref.norm() < 0.05
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)
    assert torch.count_nonzero(output[lengths == 0]) == 0
    if small_weights:
        torch.testing.assert_close(
            output[lengths > 0, ..., 0],
            torch.ones_like(output[lengths > 0, ..., 0]),
            atol=1e-2,
            rtol=0,
        )


def test_query_block_mla_projected_graph_refreshes_page_and_causal_lengths():
    batch_size, width = 8, 6
    torch.manual_seed(0)
    q, cache, table, lengths = _make_query_block_inputs(
        [65537] * batch_size, _HEADS, _PAGE_SIZE, width
    )
    weights = torch.randn(
        _HEADS, _KV_LORA_RANK, 128, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(_KV_LORA_RANK)
    gate = torch.randn(batch_size * width, 3648, device="cuda", dtype=torch.bfloat16)[
        :, -_HEADS * 128 :
    ]
    output = torch.empty(
        batch_size * width, _HEADS * 128, device="cuda", dtype=torch.bfloat16
    )

    def run():
        return _query_block_decode(
            q,
            cache,
            table,
            lengths,
            max_seqlen_k=1_048_576,
            value_weight=weights,
            gate=gate,
            out=output,
        )

    lengths.fill_(65)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    # Zero lengths are padded graph requests: their rows must be zero.
    for short, long in (
        (4095, 4096),
        (51200, 51201),
        (51201, 51200),
        (65537, 0),
        (0, 0),
    ):
        history = [short if row % 2 == 0 else long for row in range(batch_size)]
        lengths.copy_(torch.tensor(history, device="cuda", dtype=torch.int32))
        table.copy_(table.roll(1, 0))
        output.fill_(float("nan"))
        graph.replay()
        reference, _ = _query_block_reference(q, cache, table, lengths)
        projected = torch.einsum(
            "bqhd,hdv->bqhv", reference, weights.float()
        ).reshape_as(output)
        expected = projected * torch.sigmoid(gate.float())
        assert result.data_ptr() == output.data_ptr()
        assert torch.isfinite(output).all()
        assert (output.float() - expected).norm() / expected.norm().clamp_min(
            1e-20
        ) < 0.05


def test_query_block_mla_verify_reuses_split_bucket_specialization():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla import decode

    kernels = (
        decode.gluon_mla_decode_fp8_query_blocks_gfx950,
        decode.gluon_mla_decode_fp8_query_blocks_reduce_gfx950,
    )

    def check(batch_size, table_columns, softmax_scale):
        q, cache, table, lengths = _make_query_block_inputs(
            [4096] + [i + 4 for i in range(batch_size - 1)], _HEADS, _PAGE_SIZE, 6
        )
        table = torch.nn.functional.pad(table, (0, table_columns - table.shape[1]))
        output = _query_block_decode(
            q,
            cache,
            table,
            lengths,
            max_seqlen_k=1_048_576,
            softmax_scale=softmax_scale,
        )
        reference, _ = _query_block_reference(q, cache, table, lengths, softmax_scale)
        assert (output.float() - reference).norm() / reference.norm() < 0.05

    # Keep both split buckets while changing counts (5/11 to 8/16) and
    # crossing divisibility classes for the page-table stride (64 to 72).
    check(22, 64, _SOFTMAX_SCALE)
    with assert_no_triton_compile(*kernels):
        check(16, 72, _SOFTMAX_SCALE * 0.75)
