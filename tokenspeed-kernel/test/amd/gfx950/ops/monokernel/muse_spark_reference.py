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

"""Torch references of the Muse Spark decode math (the regular path's
rounding points) for the monokernel tests, one TP8 rank at a time."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.benchmark.muse_spark_monokernel import (
    raw_experts,
    rope_rows,
    rope_table,
)
from tokenspeed_kernel.benchmark.muse_spark_monokernel_gluon import (
    preprocessed_experts,
)
from tokenspeed_kernel.numerics.torch.attention.mha import (
    torch_dense_attention,
    torch_merge_attention_states,
)
from tokenspeed_kernel.numerics.torch.attention.prologue import torch_head_norm_rope
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)

HEAD, HEADS, KV_HEADS, GROUP, PAGE = 64, 16, 2, 8, 64
HIDDEN, POOLED, INTER, TOP_K, OWNED = 8192, 4096, 512, 8, 1024
K_ROW, V_ROW, GATE_ROW = 1024, 1152, 1280


def bf16_gemm(x: torch.Tensor, w: torch.Tensor):
    """``x @ w.T`` in FP64, and a bound on a kernel's difference from it after
    FP32 sums (in any order) and one rounding to BF16.

    The products of BF16 values are exact in FP32; each of the ``K`` FP32
    additions rounds by at most 2^-24 of ``|x| @ |w|.T``, and independent
    roundings stay within 6 standard deviations (2^-24 / sqrt(3) each). The
    BF16 rounding then adds its unit roundoff 2^-8.

    Returns ``(exact, bound)``, both FP64 ``[rows, N]``.
    """
    exact = x.double() @ w.double().T
    sums = (6 / math.sqrt(3) * math.sqrt(x.shape[-1]) * 2**-24) * (
        x.double().abs() @ w.double().abs().T
    )
    return exact, sums + 2**-8 * (exact.abs() + sums)


def norm_rope(
    x: torch.Tensor, gain: float, cs: torch.Tensor, rope: bool, dtype: torch.dtype
) -> torch.Tensor:
    """Per-head RMS norm (one gain for every element), then interleaved RoPE
    (``cs``: one row of :func:`rope_table`), computed in FP64 by the shared
    reference (``torch_head_norm_rope``) and rounded once to ``dtype``."""
    heads = x.reshape(-1, HEAD)
    values, _ = torch_head_norm_rope(
        heads,
        HEAD,
        norm_weight=torch.full((HEAD,), gain, dtype=torch.float64, device=x.device),
        norm_weight_offset=0.0,
        norm_eps=1e-5,
        cos_sin_cache=cs.reshape(1, -1) if rope else None,
        positions=(
            torch.zeros(heads.shape[0], dtype=torch.long, device=x.device)
            if rope
            else None
        ),
        neox=False,
        mrope_section=None,
        mrope_interleaved=False,
    )
    return values.reshape(x.shape).to(dtype)


# The kernel rounds each softmax weight to BF16 before the PV product, a
# relative error of at most 2^-9 a weight; independent roundings give the
# output an error of standard deviation at most 2^-9 / sqrt(3) * sqrt(sum_j
# w_j^2 v_j^2). The bound below allows 6 of those, and 2^-16 * sum_j w_j |v_j|
# for the FP32 scores and sums.
_WEIGHT_STEP_SIGMAS = 6 * 2**-9 / math.sqrt(3)
_FP32_SLACK = 2**-16


def verify_attention(q, q_cached, keys, values, own_k, own_v, scale):
    """Attention of one row's q heads ``q [heads, 64]`` (of one KV head) over
    its visible cached keys ``keys`` / ``values`` ``[n, 64]`` and the launch's
    keys up to its own ``own_k`` / ``own_v`` ``[t + 1, 64]``, in FP64 by the
    shared references (``torch_dense_attention`` per part, merged by
    ``torch_merge_attention_states``).

    ``q_cached`` scores the cached keys (``q``, or q rounded once to an FP8
    cache's dtype where the kernel's long full-layer path multiplies in FP8);
    ``q`` scores the launch's keys. That path is not the regular path's math
    exactly: the regular path scores every key of an FP8 cache with the FP8
    q, and here the launch's own keys still take the BF16 q. ``scale`` is
    the logit scale (``head_dim ** -0.5``).

    Returns ``(out, error)``, both FP64 ``[heads, head_dim]``: ``error`` bounds the
    kernel's difference from ``out`` (its BF16 softmax weights and FP32 sums).
    """

    def part(query, k, v):
        return torch_dense_attention(
            query.double()[None],
            k.double()[:, None],
            v.double()[:, None],
            scale=scale,
            causal_offset=None,
            window_left=None,
            sinks=None,
        )

    out, lse = part(q, own_k, own_v)
    if keys.shape[0]:
        cached_out, cached_lse = part(q_cached, keys, values)
        out, _ = torch_merge_attention_states(
            cached_out, cached_lse, out, lse, math.log2(math.e)
        )
    scores = torch.cat(
        [q_cached.double() @ keys.double().T, q.double() @ own_k.double().T], 1
    )
    w = torch.softmax(scores * scale, dim=-1)
    v = torch.cat([values, own_v]).double()
    error = _WEIGHT_STEP_SIGMAS * (w.pow(2) @ v.pow(2)).sqrt()
    return out[0], error + _FP32_SLACK * (w @ v.abs())


def attention_z(qkvg, k_cache, v_cache, table, length, window, cs):
    """One rank's gated head-norm output ``z [rows, 1024]`` of the rows of one
    request (``qkvg [rows, 2304]``; row ``t`` at position ``length - 1 +
    t``), and the rows' K, V ``[rows, 2, 64]`` and cache slots.

    ``k_cache`` / ``v_cache``: ``[slots, 2, 64]`` before the rows' writes, BF16
    or FP8 e4m3 at unit scale: the rows' K and V round once to that dtype,
    and attention reads them (and the cache) exactly widened. ``cs``: FP32
    ``[rows, 64]``, the RoPE rows of the rows' positions (row ``t``: position
    ``length - 1 + t``).

    Returns ``(z, k, v, slots, z_exact, z_error)``. ``z`` has the regular
    path's BF16 roundings (the attention output and z). ``z_exact`` is z
    without them, and ``z_error`` bounds the kernel's difference from it, both
    FP32 ``[rows, 1024]``: the attention error of :func:`verify_attention`
    plus the attention output's BF16 rounding (unit roundoff 2^-8), carried
    through the head norm (``z = a / rms(a) * gate`` moves by ``(da + |z| *
    rms(da)) / rms(a) * gate``), then z's own BF16 rounding.
    """
    rope = window != 0
    kv_dtype = k_cache.dtype
    rows = qkvg.shape[0]
    L0 = length - 1
    k = torch.stack(
        [
            norm_rope(
                qkvg[t, K_ROW:V_ROW].view(KV_HEADS, HEAD),
                1.0,
                cs[t],
                rope,
                kv_dtype,
            )
            for t in range(rows)
        ]
    )
    v = qkvg[:, V_ROW:GATE_ROW].view(rows, KV_HEADS, HEAD).to(kv_dtype)
    positions = torch.arange(L0, L0 + rows)
    slots = table[positions // PAGE].long() * PAGE + positions % PAGE
    first = max(0, L0 - window + 1) if window else 0
    cached = torch.arange(first, L0)
    cached_slots = table[cached // PAGE].long() * PAGE + cached % PAGE
    keys, values = k_cache[cached_slots].float(), v_cache[cached_slots].float()
    z = torch.empty(rows, HEADS, HEAD)
    z_exact = torch.empty(rows, HEADS, HEAD)
    z_error = torch.empty(rows, HEADS, HEAD)
    # Full layers with 8 rows (64 MFMA rows) score LONG_KEYS cached keys of
    # an FP8 cache with q rounded once from FP32 to the cache's dtype (the
    # regular path's FP8 query).
    fp8_qk = window == 0 and rows == 8 and L0 >= 32768 and kv_dtype != torch.bfloat16
    for t in range(rows):
        q_args = (qkvg[t, :K_ROW].view(HEADS, HEAD), 5.473006248474121, cs[t])
        q = norm_rope(*q_args, rope, torch.bfloat16)
        q_cached = norm_rope(*q_args, rope, kv_dtype) if fp8_qk else q
        gate = torch.sigmoid(qkvg[t, GATE_ROW:].view(HEADS, HEAD).double())
        # A sliding row sees the cached keys from L0 + t - window + 1 on.
        seen = max(0, L0 + t - window + 1 - first) if window else 0
        for kv in range(KV_HEADS):
            heads = slice(kv * GROUP, (kv + 1) * GROUP)
            a, da = verify_attention(
                q[heads],
                q_cached[heads],
                keys[seen:, kv],
                values[seen:, kv],
                k[: t + 1, kv].float(),
                v[: t + 1, kv].float(),
                HEAD**-0.5,
            )
            exact = a / torch.sqrt(a.pow(2).mean(-1, keepdim=True) + 1e-5)
            z_exact[t, heads] = (exact * gate[heads]).float()
            # The kernel's BF16 attention output: its error, then one rounding.
            da = da + 2**-8 * (a.abs() + da)
            inv = torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-5)
            rms_da = da.pow(2).mean(-1, keepdim=True).sqrt()
            dz = (da + exact.abs() * rms_da) * inv * gate[heads]
            # Then z's own rounding.
            dz = dz + 2**-8 * (exact.abs() * gate[heads] + dz)
            z_error[t, heads] = dz.float()
            a = a.to(torch.bfloat16).double()
            inv = torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-5)
            z[t, heads] = (a * inv * gate[heads]).float()
    return (
        z.reshape(rows, -1).to(torch.bfloat16),
        k,
        v,
        slots,
        z_exact.reshape(rows, -1),
        z_error.reshape(rows, -1),
    )


def expert_partials(rank: int, experts: int, pooled, gain, ids) -> torch.Tensor:
    """Rank ``rank``'s BF16 route partials ``[rows, 8, 4096]`` of each row's
    picks ``ids [rows, 8]`` (``pooled [rows, 4096]``), on ``pooled``'s device.
    Each picked expert is dequantized once."""
    x = pooled.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain.to(x.device)
    xq = fake_quantize_mxfp4_even(x.to(torch.bfloat16).float())
    w13, s13, w2, s2 = (t.to(x.device) for t in raw_experts(rank, experts))
    out = torch.empty(ids.shape[0], TOP_K, POOLED, device=x.device)
    for e in ids.unique().tolist():
        up = dequantize_mxfp4(w13[e], s13[e]).float()
        down = dequantize_mxfp4(w2[e], s2[e]).float()
        for row, k in (ids == e).nonzero().tolist():
            acc = up @ xq[row]
            h = acc[0::2] / (1.0 + torch.exp(-acc[0::2])) * acc[1::2]
            hq = fake_quantize_mxfp4_even(h.to(torch.bfloat16).float().view(1, -1))
            out[row, k] = down @ hq.view(-1)
    return out.to(torch.bfloat16)


def _blocks(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(*t.shape[:-1], -1, 32)


def _mx_exponent(t: torch.Tensor) -> torch.Tensor:
    """``fake_quantize_mxfp4_even``'s scale exponent of each 32-value block."""
    amax = _blocks(t.float()).abs().amax(dim=-1, keepdim=True)
    return ((amax.view(torch.int32) + 0x200000) >> 23) - 2


def mxfp4_flip_bound(ref, lo, hi) -> torch.Tensor:
    """A bound on ``|q(v) - q(ref)|`` for every ``v`` with ``lo <= v <= hi``
    elementwise, ``q`` = ``fake_quantize_mxfp4_even`` (FP32, last dimension in
    32-value blocks).

    Within a block whose scale cannot change over the interval (the block's
    smallest and largest possible amax give ``ref``'s scale exponent), ``q``
    is monotone in each element, so the endpoints bound it exactly: elements
    away from a rounding boundary get 0. Elsewhere one code step of the
    larger scale is allowed (2 * scale covers the widest E2M1 step and the
    saturation above 6)."""
    qr, ql, qh = (fake_quantize_mxfp4_even(t) for t in (ref, lo, hi))
    exact = _blocks(torch.maximum((ql - qr).abs(), (qh - qr).abs()))
    big = torch.maximum(lo.abs(), hi.abs())
    small = torch.where(lo * hi > 0, torch.minimum(lo.abs(), hi.abs()), 0.0)
    e = torch.stack([_mx_exponent(t) for t in (ref, big, small)])
    same = (e[0] == e[1]) & (e[0] == e[2])
    scale = torch.exp2(e.amax(0).clamp(0, 254).float() - 127)
    loose = _blocks(torch.maximum(hi - ref, ref - lo)) + 2 * scale
    return torch.where(same, exact, torch.maximum(exact, loose)).reshape(ref.shape)


def _bf16_interval(x: torch.Tensor, dx: torch.Tensor):
    """BF16 of ``x`` and of its interval's ends, as FP32: round-to-nearest is
    monotone, so a value within ``dx`` of ``x`` rounds between them."""
    return tuple(
        t.to(torch.float32).to(torch.bfloat16).float() for t in (x, x - dx, x + dx)
    )


# FP32 sums of K exact products: 6 standard deviations of K independent
# roundings of at most 2^-24 of the sum of magnitudes.
def _sums(k: int) -> float:
    return 6 / math.sqrt(3) * math.sqrt(k) * 2**-24


def expert_partials_bounded(raw, pooled, gain, ids, cols):
    """One rank's expert route partials in FP64 and a bound on its kernel's
    BF16 ones, ``[rows, 8, cols]`` each, from the raw MXFP4 shards ``raw``
    (``raw_experts``' tuple), ``pooled [rows, 4096]`` and each row's picks
    ``ids [rows, 8]`` (output columns ``cols``, a slice), on their device.

    The kernel norms the pooled row in FP32 (within 2^-16 of it), rounds it to
    BF16 and quantizes to MXFP4, sums the up projection in FP32, applies
    SiLU (FP32 exp and divide: 2^-20), rounds to BF16, quantizes again, sums
    the down projection in FP32 and rounds to BF16. Each step's error carries
    into the next; a quantization can step one code where the error reaches
    a rounding boundary (:func:`mxfp4_flip_bound`).
    """
    w13, s13, w2, s2 = raw
    x = pooled.double()
    x = (
        x
        * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        * gain.double().to(x.device)
    )
    xb, xlo, xhi = _bf16_interval(x, 2**-16 * x.abs())
    xq = fake_quantize_mxfp4_even(xb).double()
    dxq = mxfp4_flip_bound(xb, xlo, xhi).double()
    rows = ids.shape[0]
    exact = torch.empty(
        rows, TOP_K, cols.stop - cols.start, dtype=torch.float64, device=x.device
    )
    bound = torch.empty_like(exact)
    for e in ids.unique().tolist():
        up = dequantize_mxfp4(w13[e], s13[e]).double()
        down = dequantize_mxfp4(w2[e][cols], s2[e][cols]).double()
        for row, k in (ids == e).nonzero().tolist():
            acc = up @ xq[row]
            dacc = up.abs() @ dxq[row] + _sums(up.shape[1]) * (up.abs() @ xq[row].abs())
            g, u, dg, du = acc[0::2], acc[1::2], dacc[0::2], dacc[1::2]
            silu = g / (1.0 + torch.exp(-g))
            h = silu * u
            # |SiLU'(g)| stays below 1.0998, so 1.1 bounds how far g's error moves it.
            dh = 1.1 * dg * (u.abs() + du) + silu.abs() * du + 2**-20 * h.abs()
            hb, hlo, hhi = _bf16_interval(h.view(1, -1), dh.view(1, -1))
            hq = fake_quantize_mxfp4_even(hb).double().view(-1)
            dhq = mxfp4_flip_bound(hb, hlo, hhi).double().view(-1)
            out = down @ hq
            dout = down.abs() @ dhq + _sums(down.shape[1]) * (down.abs() @ hq.abs())
            exact[row, k] = out
            bound[row, k] = dout + 2**-8 * (out.abs() + dout)
    return exact, bound
