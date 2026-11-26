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

"""MXFP4 paged KV attention (tokenspeed_kernel.ops.attention.mxfp4_kv) on
gfx950: decode, extend and page dequantization against attention over the
torch codec's dequantized cache."""

from __future__ import annotations

import math

import pytest
import torch
from mxfp4_kv_reference import mxfp4_dequantize, mxfp4_quantize
from tokenspeed_kernel.ops.attention.mxfp4_kv import (
    _decode_splits,
    mxfp4_cache_planes,
    mxfp4_decode,
    mxfp4_dequant_pages,
    mxfp4_extend,
    mxfp4_row_bytes,
)
from utils import is_cdna4

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="needs AMD gfx950")

# Muse Spark at TP8, per rank.
HQ, H, D, P = 16, 2, 64, 64
BF16 = torch.bfloat16


class _Cache:
    """Random BF16 pages quantized into MXFP4 cache fields."""

    def __init__(self, bs: int, ctx: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        pages_per = math.ceil(ctx / P)
        npages = bs * pages_per + 4
        pt = torch.randperm(npages, device="cuda", generator=g)[: bs * pages_per]
        self.page_table = pt.view(bs, pages_per).int()
        self.planes, self.dequant = [], []
        for outlier in (4.0, 1.0):
            x = torch.randn(npages, P, H, D, device="cuda", generator=g, dtype=BF16)
            x[..., 5] *= outlier
            field = torch.empty(
                npages * P, H, mxfp4_row_bytes(D), dtype=torch.uint8, device="cuda"
            )
            data, scales = mxfp4_cache_planes(field, P, D)
            data[:], scales[:] = mxfp4_quantize(x)
            self.planes.append((data, scales))
            self.dequant.append(mxfp4_dequantize(data, scales))


def _attend(q, cache, seqlens, qlens, window_left, scale):
    """fp32 attention over the dequantized pages."""
    kf, vf = cache.dequant
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    row = 0
    for b, qlen in enumerate(qlens):
        n = int(seqlens[b])
        pages = cache.page_table[b, : math.ceil(n / P)].long()
        kk = kf[pages].reshape(-1, H, D)[:n].repeat_interleave(HQ // H, dim=1)
        vv = vf[pages].reshape(-1, H, D)[:n].repeat_interleave(HQ // H, dim=1)
        for t in range(qlen):
            end = n - (qlen - 1 - t)
            lo = max(0, end - (window_left + 1)) if window_left >= 0 else 0
            s = torch.einsum("hd,nhd->hn", q[row].float(), kk[lo:end]) * scale
            out[row] = torch.einsum("hn,nhd->hd", s.softmax(-1), vv[lo:end])
            row += 1
    return out


def _gated(out, gate, eps):
    """The weightless-RMSNorm sigmoid gate applied to rounded output rows."""
    v = out.to(BF16).float()
    var = v.square().mean(-1, keepdim=True)
    g = gate.float().view(v.shape)
    return v * torch.rsqrt(var + eps) * torch.sigmoid(g)


def _rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


@pytest.mark.parametrize(
    "bs,ctx,qlen,window_left",
    [
        (3, 40, 1, -1),  # one split
        (3, 1000, 1, -1),
        (3, 3000, 1, 2047),
        (64, 4200, 1, 2047),
        (2, 700, 4, -1),
        (2, 5000, 3, 2047),
    ],
)
@pytest.mark.parametrize("gated", [False, True])
def test_decode(bs, ctx, qlen, window_left, gated):
    cache = _Cache(bs, ctx, seed=bs * ctx + qlen)
    seqlens = torch.randint(
        max(qlen, ctx // 2), ctx + 1, (bs,), device="cuda", dtype=torch.int32
    )
    eps = 1e-6
    # Twice: the split-K arrival counters re-arm after each launch.
    for call in range(2):
        q = torch.randn(bs * qlen, HQ, D, device="cuda", dtype=BF16)
        gate = (
            torch.randn(bs * qlen, HQ * D + 32, device="cuda", dtype=BF16)[:, : HQ * D]
            if gated
            else None
        )
        out = mxfp4_decode(
            q,
            *cache.planes,
            cache.page_table,
            seqlens,
            window_left=window_left,
            qlen=qlen,
            softmax_scale=D**-0.5,
            gate=gate,
            gate_eps=eps,
        )
        ref = _attend(q, cache, seqlens, [qlen] * bs, window_left, D**-0.5)
        if gated:
            ref = _gated(ref, gate, eps)
        assert _rel(out, ref) < 1e-2, f"call {call}"


@pytest.mark.parametrize(
    "bs,ctx,window_left",
    [(2, 70, -1), (2, 700, -1), (2, 5000, 2047), (16, 3000, -1), (16, 5000, 2047)],
)
def test_decode_over_a_wide_page_table(bs, ctx, window_left):
    """A CUDA-graph page table is as wide as the longest sequence; a short
    sequence uses fewer splits than the launch."""
    qlen = 2
    cache = _Cache(bs, ctx, seed=ctx)
    page_table = torch.zeros(bs, 2080, dtype=torch.int32, device="cuda")
    page_table[:, : cache.page_table.shape[1]] = cache.page_table
    seqlens = torch.linspace(ctx // 2 + 1, ctx, bs, device="cuda").to(torch.int32)
    for _ in range(2):
        q = torch.randn(bs * qlen, HQ, D, device="cuda", dtype=BF16)
        out = mxfp4_decode(
            q,
            *cache.planes,
            page_table,
            seqlens,
            window_left=window_left,
            qlen=qlen,
            softmax_scale=D**-0.5,
            gate=None,
            gate_eps=1e-6,
        )
        ref = _attend(q, cache, seqlens, [qlen] * bs, window_left, D**-0.5)
        assert _rel(out, ref) < 1e-2


def test_long_decode_splits_the_keys():
    assert _decode_splits(2 * H, 32, torch.device("cuda")) > 1


@pytest.mark.parametrize(
    "prefix,qlens,window_left",
    [(0, (150, 37), -1), (200, (150, 37), -1), (2100, (300,), 2047)],
)
def test_extend(prefix, qlens, window_left):
    bs = len(qlens)
    cache = _Cache(bs, prefix + max(qlens), seed=prefix + 1)
    seqlens = torch.tensor(
        [prefix + n for n in qlens], device="cuda", dtype=torch.int32
    )
    cu = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(qlens), HQ, D, device="cuda", dtype=BF16)
    out = mxfp4_extend(
        q,
        *cache.planes,
        cache.page_table,
        seqlens,
        cu,
        max(qlens),
        window_left=window_left,
        softmax_scale=D**-0.5,
    )
    ref = _attend(q, cache, seqlens, qlens, window_left, D**-0.5)
    assert _rel(out, ref) < 1e-2


def test_dequant_pages_matches_the_codec():
    cache = _Cache(1, 20 * P, seed=7)
    ids = cache.page_table[0, ::3].contiguous()
    k_out = torch.empty(ids.shape[0], P, H, D, dtype=BF16, device="cuda")
    v_out = torch.empty_like(k_out)
    mxfp4_dequant_pages(*cache.planes, ids, k_out, v_out)
    assert torch.equal(k_out, cache.dequant[0][ids.long()].to(BF16))
    assert torch.equal(v_out, cache.dequant[1][ids.long()].to(BF16))
