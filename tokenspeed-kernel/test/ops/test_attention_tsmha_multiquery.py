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

"""Multi-query (uniform q_len > 1, MTP-verify shape) rel decode coverage.

Answered against the torch reference:

1. The route MTP verify actually takes — v2's NATIVE prediction dimension
   (unexpanded [B] seqlens / [B, W] table) — through the public dispatch
   (``test_dispatch_native_multiq``) and the kernel entry point
   (``test_v2_native_prediction_vs_reference``).
2. v1 batch-mode native prediction incl. the local-window mask fix
   (``test_v1_native_prediction_*``).
"""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(29)

DTYPE = torch.bfloat16
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
TOL = 2e-2
SCALE = 1.0 / HEAD_DIM


def _skip_unless_supported() -> None:
    pytest.importorskip("tokenspeed_kernel.ops.attention.cute_dsl.rel_mha")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tokenspeed-mha decode kernel is SM100-only")


def _ref_row(q_row, k, v, rel_row, rel_extent, window_left, pos):
    """One query row at absolute position ``pos``. q_row [H,D]; k/v [Sk,KV,D];
    rel_row [H,E]."""
    Sk = k.shape[0]
    H = q_row.shape[0]
    kx = k.repeat_interleave(H // NUM_KV_HEADS, dim=1)
    vx = v.repeat_interleave(H // NUM_KV_HEADS, dim=1)
    logits = torch.einsum("hd,khd->hk", q_row.float(), kx.float()) * SCALE
    dist = pos - torch.arange(Sk, device=q_row.device)
    in_range = (dist >= 0) & (dist < rel_extent)
    idx = dist.clamp(0, rel_extent - 1)
    bias = rel_row.float().gather(-1, idx.unsqueeze(0).expand(H, Sk))
    logits = logits + bias.masked_fill(~in_range[None], 0.0)
    masked = dist < 0
    if window_left >= 0:
        masked = masked | (dist > window_left)
    logits = logits.masked_fill(masked[None], -torch.inf)
    return torch.einsum("hk,khd->hd", torch.softmax(logits, dim=-1), vx.float())


def _build_paged(kv_lens, page, device):
    pages_per = [(L + page - 1) // page for L in kv_lens]
    k_cache = torch.zeros(
        sum(pages_per), page, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    v_cache = torch.zeros_like(k_cache)
    table = torch.zeros(len(kv_lens), max(pages_per), device=device, dtype=torch.int32)
    ks, vs, nxt = [], [], 0
    for i, L in enumerate(kv_lens):
        ki = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
        vi = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
        ks.append(ki)
        vs.append(vi)
        for p in range(pages_per[i]):
            table[i, p] = nxt
            rows = min(page, L - p * page)
            k_cache[nxt, :rows] = ki[p * page : p * page + rows]
            v_cache[nxt, :rows] = vi[p * page : p * page + rows]
            nxt += 1
    return k_cache, v_cache, table, ks, vs


CASES = [
    pytest.param(128, 512, 511, [300, 641], 4, id="swa-p128-k4"),
    pytest.param(128, 1024, -1, [300, 641], 4, id="full-p128-k4"),
    pytest.param(128, 512, 511, [137, 260], 3, id="swa-p128-k3"),
    # Clamped graph-padded row (seq == k, the replay clamp floor) next to
    # real rows: the padded row's spans are 1..k and must not poison
    # neighbors (the churn-corruption regression).
    pytest.param(128, 512, 511, [4, 641, 300], 4, id="swa-p128-k4-minrow"),
]


def _reference(qs, ks, vs, rels, kv_lens, k_new, rel_extent, window_left):
    outs = []
    for i, L in enumerate(kv_lens):
        for t in range(k_new):
            pos = L - k_new + t
            outs.append(
                _ref_row(
                    qs[i * k_new + t],
                    ks[i][: pos + 1],
                    vs[i][: pos + 1],
                    rels[i * k_new + t],
                    rel_extent,
                    window_left,
                    pos,
                )
            )
    return torch.stack(outs)


@pytest.mark.parametrize("page,rel_extent,window_left,kv_lens,k_new", CASES)
def test_dispatch_native_multiq(page, rel_extent, window_left, kv_lens, k_new) -> None:
    """Uniform msq>1 dispatch takes v2's NATIVE prediction dimension — the
    UNEXPANDED [B, W] table with prediction=k_new — on SWA and full
    attention alike (gate-validated route; full-attn prediction
    causality fixed 2026-07-14)."""
    _skip_unless_supported()
    import tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode_v2 as v2mod
    from tokenspeed_kernel import rel_mha_decode_with_kvcache

    device = torch.device("cuda")
    B = len(kv_lens)
    k_cache, v_cache, table, ks, vs = _build_paged(kv_lens, page, device)
    q = torch.randn(B * k_new, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = (
        torch.randn(B * k_new, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE)
        * 0.2
    )
    cu_q = torch.arange(0, (B + 1) * k_new, k_new, device=device, dtype=torch.int32)
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    calls = []
    orig = v2mod.rel_mha_decode_tsmha_v2

    def spy(*args, **kwargs):
        calls.append(
            (
                kwargs.get("page_table", args[3] if len(args) > 3 else None),
                kwargs.get("prediction"),
            )
        )
        return orig(*args, **kwargs)

    v2mod.rel_mha_decode_tsmha_v2 = spy
    try:
        out = rel_mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=table,
            cache_seqlens=seqlens,
            max_seqlen_k=max(kv_lens),
            rel_logits=rel,
            cu_seqlens_q=cu_q,
            max_seqlen_q=k_new,
            window_left=window_left,
            softmax_scale=SCALE,
        ).view(B * k_new, NUM_Q_HEADS, HEAD_DIM)
    finally:
        v2mod.rel_mha_decode_tsmha_v2 = orig

    assert len(calls) == 1, "uniform multi-query decode must take v2"
    pt, pred = calls[0]
    assert pt is table, "native route must receive the UNEXPANDED table"
    assert pred == k_new, "native route must pass prediction=k_new"
    ref = _reference(q, ks, vs, rel, kv_lens, k_new, rel_extent, window_left)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)


# Regression fingerprint for the full-attn prediction causality bug (fixed
# 2026-07-14): the tail block masked
# only key >= seqlen, so prediction rows attended their successors' keys —
# error ~1/seq at every length (0.02-0.12 for seq <= 128, growing with P;
# P=1 and SWA were exact because their mask paths carry the causal term).
# Short full-attn seqlens make any regression unmissable here.
SHORT_FULL_ATTN_CASES = [
    pytest.param(
        128,
        1024,
        -1,
        kv_lens,
        k_new,
        id=f"full-shortseq-{'x'.join(map(str, kv_lens))}-k{k_new}",
    )
    for kv_lens in ([37], [37, 900], [64, 900])
    for k_new in (2, 4, 6)
]


# Direct-entry coverage: the dispatch test above already checks CASES
# numerics through the public op (enable_pdl=False kernels), so here only
# one SWA case rides along for the enable_pdl=True default — the rest of
# the direct sweep is the short-seq causality fingerprint.
@pytest.mark.parametrize(
    "page,rel_extent,window_left,kv_lens,k_new",
    [CASES[3]] + SHORT_FULL_ATTN_CASES,
)
def test_v2_native_prediction_vs_reference(
    page, rel_extent, window_left, kv_lens, k_new
) -> None:
    """v2 with the kernel's NATIVE prediction dimension (no metadata
    expansion): unexpanded [B] seqlens / [B, W] table, q token-major
    [B*k, H, D], in-kernel per-row causality."""
    _skip_unless_supported()
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode_v2 import (
        rel_mha_decode_tsmha_v2,
    )

    device = torch.device("cuda")
    B = len(kv_lens)
    k_cache, v_cache, table, ks, vs = _build_paged(kv_lens, page, device)
    q = torch.randn(B * k_new, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = (
        torch.randn(B * k_new, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE)
        * 0.2
    )
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    out = rel_mha_decode_tsmha_v2(
        q,
        k_cache,
        v_cache,
        table,
        seqlens,
        rel,
        window_left,
        SCALE,
        prediction=k_new,
    ).view(B * k_new, NUM_Q_HEADS, HEAD_DIM)

    ref = _reference(q, ks, vs, rel, kv_lens, k_new, rel_extent, window_left)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("page,rel_extent,window_left,kv_lens,k_new", CASES)
def test_v1_native_prediction_vs_reference(
    page, rel_extent, window_left, kv_lens, k_new
) -> None:
    """v1 (batch-mode shear + fwd) with native prediction: unexpanded [B]
    seqlens / [B, W] table, per-row sheared bias tile, in-kernel
    bottom-right causal masking."""
    _skip_unless_supported()
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode import (
        rel_mha_decode_tsmha,
    )

    device = torch.device("cuda")
    B = len(kv_lens)
    k_cache, v_cache, table, ks, vs = _build_paged(kv_lens, page, device)
    q = torch.randn(B * k_new, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = (
        torch.randn(B * k_new, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE)
        * 0.2
    )
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    out = rel_mha_decode_tsmha(
        q,
        k_cache,
        v_cache,
        table,
        seqlens,
        rel,
        None,  # cu_seqlens_q: unused (batch mode)
        window_left,
        softmax_scale=SCALE,
        prediction=k_new,
    ).view(B * k_new, NUM_Q_HEADS, HEAD_DIM)

    ref = _reference(q, ks, vs, rel, kv_lens, k_new, rel_extent, window_left)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)


def test_v1_native_prediction_tile_boundary() -> None:
    """P rows straddling a 128-tile boundary (positions 4094..4097): the
    shear's negative anchored phases must keep every row exact."""
    _skip_unless_supported()
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode import (
        rel_mha_decode_tsmha,
    )

    device = torch.device("cuda")
    page, rel_extent, window_left, kv_lens, k_new = 128, 512, 511, [4098, 300], 4
    B = len(kv_lens)
    k_cache, v_cache, table, ks, vs = _build_paged(kv_lens, page, device)
    q = torch.randn(B * k_new, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = (
        torch.randn(B * k_new, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE)
        * 0.2
    )
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    out = rel_mha_decode_tsmha(
        q,
        k_cache,
        v_cache,
        table,
        seqlens,
        rel,
        None,
        window_left,
        softmax_scale=SCALE,
        prediction=k_new,
    ).view(B * k_new, NUM_Q_HEADS, HEAD_DIM)
    ref = _reference(q, ks, vs, rel, kv_lens, k_new, rel_extent, window_left)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("window_left", [511, -1])
@pytest.mark.xfail(
    reason="v1 mxfp8 native-prediction pack path: the Q scale-factor tensor "
    "(mSFQ) is built rank-4 for the batch-mode (prediction>1) Q view, but the "
    "fwd kernel expects the rank-6 packed-SF layout — 'Mismatched Tensor on "
    "argument #8 ... expected ndim=6'. Distinct from the local-window mask bug "
    "(now fixed); this is the remaining blocker for v1 mxfp8 multi-q. NOTE: the "
    "flashinfer quantize_mxfp8 JIT also needs an nvcc whose version matches the "
    "CUDA runtime headers (else CCCL's CTK compat guard errors at build).",
    strict=False,
)
def test_v1_native_prediction_mxfp8(window_left) -> None:
    """MXFP8 blockscaled v1 with native prediction matches the bf16 v1 run
    on the dequantized cache (fp8 quantization is the only error source)."""
    _skip_unless_supported()
    import math as _math

    from tokenspeed_kernel import quantize_mxfp8
    from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode import (
        rel_mha_decode_tsmha,
    )
    from tokenspeed_kernel.ops.kvcache.triton import store_sf_interleaved

    device = torch.device("cuda")
    page, extent, k_new = 128, 512, 4
    kv_lens = [700, 391]
    B = len(kv_lens)
    SF = HEAD_DIM // 32

    def _quant(x):
        t, h, d = x.shape
        qd, sf = quantize_mxfp8(x.reshape(t * h, d))
        return qd.reshape(t, h, d), sf.view(torch.float8_e8m0fnu).reshape(t, h, SF)

    def _dequant(qd, sf):
        scale = sf.to(torch.float32).repeat_interleave(32, dim=-1)
        return (qd.to(torch.float32) * scale).to(torch.bfloat16)

    pages_per = [_math.ceil(L / page) for L in kv_lens]
    num_pages = sum(pages_per)
    k16 = torch.randn(
        num_pages * page, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    v16 = torch.randn_like(k16)
    k8, ksf = _quant(k16)
    v8, vsf = _quant(v16)
    k_scale = torch.zeros(
        num_pages, NUM_KV_HEADS, 32, SF, SF, dtype=torch.float8_e8m0fnu, device=device
    )
    v_scale = torch.zeros_like(k_scale)
    loc = torch.arange(num_pages * page, device=device, dtype=torch.int64)
    store_sf_interleaved(ksf, k_scale, loc)
    store_sf_interleaved(vsf, v_scale, loc)
    table = torch.zeros(B, max(pages_per), dtype=torch.int32, device=device)
    nxt = 0
    for b, n in enumerate(pages_per):
        table[b, :n] = torch.arange(nxt, nxt + n, device=device)
        nxt += n
    paged = lambda t: t.reshape(num_pages, page, NUM_KV_HEADS, HEAD_DIM)

    q16 = (
        torch.randn(B * k_new, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    )
    q8, qsf = _quant(q16)
    rel = torch.randn(B * k_new, NUM_Q_HEADS, extent, device=device, dtype=DTYPE) * 0.2
    seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    out8 = rel_mha_decode_tsmha(
        q8,
        paged(k8),
        paged(v8),
        table,
        seqlens,
        rel,
        None,
        window_left,
        softmax_scale=SCALE,
        q_scale=qsf,
        k_scale=k_scale,
        v_scale=v_scale,
        prediction=k_new,
    ).view(B * k_new, NUM_Q_HEADS, HEAD_DIM)
    out16 = rel_mha_decode_tsmha(
        _dequant(q8, qsf),
        paged(_dequant(k8, ksf)),
        paged(_dequant(v8, vsf)),
        table,
        seqlens,
        rel,
        None,
        window_left,
        softmax_scale=SCALE,
        prediction=k_new,
    ).view(B * k_new, NUM_Q_HEADS, HEAD_DIM)

    cos = torch.nn.functional.cosine_similarity(
        out8.float().flatten(1), out16.float().flatten(1), dim=-1
    ).min()
    assert cos > 0.99, f"cosine {cos:.5f}"
