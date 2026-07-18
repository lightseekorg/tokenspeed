# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MXFP8 block-scaled paged attention vs the bf16 kernel.

Builds a paged fp8 KV cache + interleaved UE8M0 scales (store_sf_interleaved),
quantizes q per token, and compares mha_decode_with_kvcache /
mha_extend_with_kvcache outputs on the block-scaled path against the same op
running on the unquantized bf16 cache. The only error source is e4m3+e8m0
quantization of q/k/v, so agreement is tight but not bit-exact.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform

platform = current_platform()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(
        not platform.is_blackwell,
        reason="MXFP8 attention kernels require SM10x Blackwell",
    ),
]

requires_sm100 = pytest.mark.skipif(
    platform.arch_version != ArchVersion(10, 0),
    reason="plain FA4 MXFP8 kernels require SM100 Blackwell",
)

PAGE = 128
HEAD_DIM = 128
SF_DIM = HEAD_DIM // 32
HEADS_Q = 16
HEADS_KV = 4


def _require_blockscaled_fa4():
    flash_attn = pytest.importorskip("tokenspeed_kernel.ops.attention.flash_attn")
    if not getattr(flash_attn, "_FA4_HAS_BLOCKSCALED", False):
        pytest.skip("installed FA4 build has no blockscaled (sfq) support")


def _quantize(x: torch.Tensor):
    """[T, H, D] bf16 -> (fp8 [T, H, D], e8m0 [T, H, SF_DIM])."""
    from tokenspeed_kernel import quantize_mxfp8

    t, h, d = x.shape
    q, sf = quantize_mxfp8(x.reshape(t * h, d))
    return q.reshape(t, h, d), sf.view(torch.float8_e8m0fnu).reshape(t, h, SF_DIM)


def _build_paged_cache(seq_lens: list[int], seed: int):
    """Per-request contiguous pages; returns bf16/fp8 caches, scales, page table."""
    from tokenspeed_kernel.ops.kvcache.triton import store_sf_interleaved

    torch.manual_seed(seed)
    pages_per_seq = [math.ceil(s / PAGE) for s in seq_lens]
    num_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)

    k_bf16 = torch.randn(
        num_pages * PAGE, HEADS_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v_bf16 = torch.randn_like(k_bf16)

    k_fp8, k_sf_tok = _quantize(k_bf16)
    v_fp8, v_sf_tok = _quantize(v_bf16)

    k_scale = torch.zeros(
        num_pages,
        HEADS_KV,
        32,
        SF_DIM,
        SF_DIM,
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    v_scale = torch.zeros_like(k_scale)
    loc = torch.arange(num_pages * PAGE, device="cuda", dtype=torch.int64)
    store_sf_interleaved(k_sf_tok, k_scale, loc)
    store_sf_interleaved(v_sf_tok, v_scale, loc)

    page_table = torch.zeros(len(seq_lens), max_pages, dtype=torch.int32, device="cuda")
    next_page = 0
    for b, n in enumerate(pages_per_seq):
        page_table[b, :n] = torch.arange(next_page, next_page + n, device="cuda")
        next_page += n

    def paged(t):
        return t.reshape(num_pages, PAGE, HEADS_KV, HEAD_DIM)

    # Dequantized-bf16 reference caches: isolates the kernel under test from
    # k/v quantization error (the fp8 path sees exactly these values).
    k_dq = _dequant(k_fp8, k_sf_tok)
    v_dq = _dequant(v_fp8, v_sf_tok)
    return {
        "k_bf16": paged(k_dq),
        "v_bf16": paged(v_dq),
        "k_fp8": paged(k_fp8),
        "v_fp8": paged(v_fp8),
        "k_scale": k_scale,
        "v_scale": v_scale,
        "page_table": page_table,
    }


def _dequant(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    scale = sf.to(torch.float32).repeat_interleave(32, dim=-1)
    return (q.to(torch.float32) * scale).to(torch.bfloat16)


def _check(out: torch.Tensor, ref: torch.Tensor):
    out, ref = out.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(1), ref.flatten(1), dim=-1
    ).min()
    rel = (out - ref).abs().max() / ref.abs().max()
    assert cos > 0.99, f"cosine {cos:.5f}"
    assert rel < 0.08, f"max rel err {rel:.4f}"


@pytest.mark.parametrize("window_left", [-1, 511])
@requires_sm100
def test_decode_mxfp8_matches_bf16(window_left: int):
    _require_blockscaled_fa4()
    from tokenspeed_kernel.ops.attention import mha_decode_with_kvcache

    seq_lens = [900, 300, 1533]
    cache = _build_paged_cache(seq_lens, seed=7)
    cache_seqlens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    torch.manual_seed(11)
    q_bf16 = torch.randn(
        len(seq_lens), HEADS_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    q_fp8, q_sf = _quantize(q_bf16)

    ref = mha_decode_with_kvcache(
        q=_dequant(q_fp8, q_sf),
        k_cache=cache["k_bf16"],
        v_cache=cache["v_bf16"],
        page_table=cache["page_table"],
        cache_seqlens=cache_seqlens,
        max_seqlen_k=2048,
        max_seqlen_q=1,
        window_left=window_left,
        solution="fa4",
    )
    out = mha_decode_with_kvcache(
        q=q_fp8,
        k_cache=cache["k_fp8"],
        v_cache=cache["v_fp8"],
        page_table=cache["page_table"],
        cache_seqlens=cache_seqlens,
        max_seqlen_k=2048,
        max_seqlen_q=1,
        window_left=window_left,
        q_scale=q_sf,
        k_scale=cache["k_scale"],
        v_scale=cache["v_scale"],
        solution="fa4",
    )
    _check(out, ref)


@pytest.mark.parametrize("window_left", [-1, 511])
@requires_sm100
def test_extend_mxfp8_matches_bf16(window_left: int):
    _require_blockscaled_fa4()
    from tokenspeed_kernel.ops.attention import mha_extend_with_kvcache

    seq_lens = [700, 1200]
    extend_lens = [64, 96]
    cache = _build_paged_cache(seq_lens, seed=23)
    cache_seqlens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    cu_q = torch.tensor(
        [0, extend_lens[0], sum(extend_lens)], device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0, seq_lens[0], sum(seq_lens)], device="cuda", dtype=torch.int32
    )

    torch.manual_seed(29)
    q_bf16 = torch.randn(
        sum(extend_lens), HEADS_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    q_fp8, q_sf = _quantize(q_bf16)

    common = dict(
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        page_table=cache["page_table"],
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max(extend_lens),
        max_seqlen_k=2048,
        is_causal=True,
        window_left=window_left,
        solution="fa4",
    )
    ref = mha_extend_with_kvcache(
        q=_dequant(q_fp8, q_sf),
        k_cache=cache["k_bf16"],
        v_cache=cache["v_bf16"],
        **common,
    )
    out = mha_extend_with_kvcache(
        q=q_fp8,
        k_cache=cache["k_fp8"],
        v_cache=cache["v_fp8"],
        q_scale=q_sf,
        k_scale=cache["k_scale"],
        v_scale=cache["v_scale"],
        **common,
    )
    _check(out, ref)


@pytest.mark.parametrize("window_left", [-1, 511])
def test_rel_decode_oob_pool_bytes_have_zero_effect(window_left: int):
    """OOB contract: unwritten rows of the boundary page must not affect the
    output even when their bytes decode to fp8 NaN (recycled pages hold
    arbitrary bytes; P@V would compute 0 * NaN = NaN without the kernel's
    seqused V-zeroing, and the S-side masks only cover the K side)."""
    import tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode as rel_decode
    from tokenspeed_kernel.ops.kvcache.triton import store_sf_interleaved

    torch.manual_seed(3)
    seqlen = 1281  # decode lands on row 0 of a fresh page (1280 % 128 == 0)
    num_pages = seqlen // PAGE + 2
    k8 = torch.zeros(
        num_pages * PAGE,
        HEADS_KV,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    v8 = torch.zeros_like(k8)
    ksf = torch.zeros(
        num_pages,
        HEADS_KV,
        1,
        32,
        SF_DIM,
        SF_DIM,
        device="cuda",
        dtype=torch.float8_e8m0fnu,
    )
    vsf = torch.zeros_like(ksf)

    kv = torch.randn(seqlen, HEADS_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_q, k_sf = _quantize(kv)
    v_q, v_sf = _quantize(kv * 0.5)
    loc = torch.arange(seqlen, device="cuda", dtype=torch.int64)
    k8[:seqlen] = k_q
    v8[:seqlen] = v_q
    store_sf_interleaved(k_sf, ksf.view(num_pages, HEADS_KV, 32, SF_DIM, SF_DIM), loc)
    store_sf_interleaved(v_sf, vsf.view(num_pages, HEADS_KV, 32, SF_DIM, SF_DIM), loc)

    q16 = torch.randn(1, HEADS_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q8, q_sf = _quantize(q16)
    extent = 512 if window_left > 0 else 2048
    rel = 0.5 * torch.randn(1, HEADS_Q, extent, device="cuda", dtype=torch.bfloat16)
    pt = torch.arange(num_pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    seq = torch.tensor([seqlen], device="cuda", dtype=torch.int32)

    def run():
        return rel_decode.rel_mha_decode_tsmha(
            q=q8,
            k_cache=k8.view(num_pages, PAGE, HEADS_KV, HEAD_DIM),
            v_cache=v8.view(num_pages, PAGE, HEADS_KV, HEAD_DIM),
            page_table=pt,
            cache_seqlens=seq,
            rel_logits=rel,
            cu_seqlens_q=None,
            window_left=window_left,
            q_scale=q_sf,
            k_scale=ksf,
            v_scale=vsf,
        )

    ref = run()
    assert torch.isfinite(ref.float()).all()

    # Poison every unwritten row of the boundary page with all-NaN bytes
    # (data 0x7F = e4m3 NaN, SF 0xFF = e8m0 NaN).
    page = (seqlen - 1) // PAGE
    row_in_page = (seqlen - 1) % PAGE
    for cache in (k8, v8):
        cache.view(num_pages, PAGE, HEADS_KV, HEAD_DIM)[page, row_in_page + 1 :].view(
            torch.uint8
        ).fill_(0x7F)

    out = run()
    assert torch.isfinite(out.float()).all(), "OOB NaN bytes leaked into output"
    assert torch.equal(out, ref), "OOB bytes changed the output"
