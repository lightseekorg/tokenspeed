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
from types import SimpleNamespace

import torch
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)

HEAD, HEADS, KV_HEADS, GROUP, PAGE = 64, 16, 2, 8, 64
HIDDEN, POOLED, INTER, TOP_K, OWNED = 8192, 4096, 512, 8, 1024
K_ROW, V_ROW, GATE_ROW = 1024, 1152, 1280


def rope_table(positions: int) -> torch.Tensor:
    """FP32 ``[positions, 64]``: cos of the 32 pairs, then sin (theta 500000)."""
    return rope_rows(0, positions)


def rope_rows(first: int, count: int) -> torch.Tensor:
    """Rows ``first .. first + count`` of the RoPE table (:func:`rope_table`)."""
    inv = 1.0 / 500000.0 ** (torch.arange(0, HEAD, 2, dtype=torch.float32) / HEAD)
    t = torch.arange(first, first + count, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([t.cos(), t.sin()], dim=1)


def norm_rope(
    x: torch.Tensor, gain: float, cs: torch.Tensor, rope: bool, dtype: torch.dtype
) -> torch.Tensor:
    """Per-head RMS norm, then interleaved RoPE: FP32, one rounding to
    ``dtype``."""
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain
    if rope:
        c, s = cs[:32], cs[32:]
        lo, hi = y[..., 0::2], y[..., 1::2]
        y = torch.stack([lo * c - hi * s, hi * c + lo * s], dim=-1).flatten(-2)
    return y.to(dtype)


def _online(state, scores, values):
    """One chunk of the monokernel's online base-2 softmax for a batch of q
    heads (``scores [heads, keys]``, -inf where masked); its weights round
    to BF16 before the PV product."""
    m, l, acc = state
    m_new = torch.maximum(m, scores.amax(-1))
    m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
    alpha = torch.exp2(m - m_safe)
    p = torch.exp2(scores - m_safe[:, None])
    acc = acc * alpha[:, None] + p.to(torch.bfloat16).float() @ values
    return m_new, l * alpha + p.sum(-1), acc


def verify_attention(
    q,
    q_cached,
    keys,
    values,
    own_k,
    own_v,
    t,
    first,
    L0,
    row_first,
    splits,
    chunk,
    align,
):
    """The monokernel's split attention of row ``t``'s q heads ``q [heads,
    64]`` (of one KV head): the splits share the cached keys ``first .. L0 -
    1`` (``keys`` / ``values`` ``[L0 - first, 64]``) in chunks of ``chunk``, the
    keys before ``row_first`` masked; the last split then takes the launch's
    keys ``0 .. t`` (``own_k`` / ``own_v``) as one more chunk. Then the LSE
    merge. A split's keys are a multiple of ``align`` (whole pages when 64).
    ``q_cached`` scores the cached keys (``q``, or q rounded once to an FP8
    cache's dtype where the kernel's long full-layer path multiplies in FP8);
    ``q`` scores the launch's keys. That path is not the regular path's math
    exactly: the regular path scores every key of an FP8 cache with the FP8
    q, and here the launch's own keys still take the BF16 q, and P x V stays
    BF16. Returns ``[heads, 64]``."""
    scale = 0.125 * math.log2(math.e)
    heads = q.shape[0]
    n = L0 - first
    per = -(-n // splits)
    per = -(-per // align) * align
    pos = torch.arange(first, L0)
    scores = (q_cached.float() @ keys.T) * scale
    scores = torch.where(pos >= row_first, scores, float("-inf"))
    outs, lses = [], []
    for s in range(splits):
        lo, hi = s * per, min(s * per + per, n)
        state = (
            torch.full((heads,), float("-inf")),
            torch.zeros(heads),
            torch.zeros(heads, HEAD),
        )
        for c0 in range(lo, hi, chunk):
            c1 = min(c0 + chunk, hi)
            state = _online(state, scores[:, c0:c1], values[c0:c1])
        if s == splits - 1:
            own = (q.float() @ own_k[: t + 1].T) * scale
            state = _online(state, own, own_v[: t + 1])
        m, l, acc = state
        outs.append(acc / torch.where(l > 0, l, 1.0)[:, None])
        lses.append(
            torch.where(l > 0, m + torch.log2(torch.where(l > 0, l, 1.0)), -math.inf)
        )
    lse = torch.stack(lses)  # [splits, heads]
    w = torch.where(lse == -math.inf, 0.0, torch.exp2(lse - lse.amax(0)))
    return (torch.stack(outs) * w[:, :, None]).sum(0) / w.sum(0)[:, None]


def attention_z(qkvg, k_cache, v_cache, table, length, window, splits, cs):
    """One rank's gated head-norm output ``z [rows, 1024]`` of the rows of one
    request (``qkvg [rows, 2304]``; row ``t`` at position ``length - 1 +
    t``), and the rows' K, V ``[rows, 2, 64]`` and cache slots.

    ``k_cache`` / ``v_cache``: ``[slots, 2, 64]`` before the rows' writes, BF16
    or FP8 e4m3 at unit scale: the rows' K and V round once to that dtype,
    and attention reads them (and the cache) exactly widened. ``cs``: FP32
    ``[rows, 64]``, the RoPE rows of the rows' positions.
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
    # Full layers with 8 rows (64 MFMA rows) take attention.py's key groups
    # from LONG_KEYS cached keys: a BF16 cache splits on whole pages, an FP8
    # cache scores its keys with q rounded once from FP32 to the cache's
    # dtype (the regular path's FP8 query).
    groups = window == 0 and rows == 8 and L0 >= 32768
    fp8_qk = groups and kv_dtype != torch.bfloat16
    for t in range(rows):
        q_args = (qkvg[t, :K_ROW].view(HEADS, HEAD), 5.473006248474121, cs[t])
        q = norm_rope(*q_args, rope, torch.bfloat16)
        q_cached = norm_rope(*q_args, rope, kv_dtype) if fp8_qk else q
        gate = qkvg[t, GATE_ROW:].view(HEADS, HEAD).float()
        row_first = L0 + t - window + 1 if window else 0
        for kv in range(KV_HEADS):
            heads = slice(kv * GROUP, (kv + 1) * GROUP)
            a = verify_attention(
                q[heads],
                q_cached[heads],
                keys[:, kv],
                values[:, kv],
                k[:, kv].float(),
                v[:, kv].float(),
                t,
                first,
                L0,
                row_first,
                splits,
                128,
                64 if groups and not fp8_qk else 1,
            )
            a = a.to(torch.bfloat16).float()
            inv = torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-5)
            z[t, heads] = a * inv * torch.sigmoid(gate[heads])
    return z.reshape(rows, -1).to(torch.bfloat16), k, v, slots


def raw_experts(rank: int, experts: int):
    """Random MXFP4 expert shards of ``rank``: w13 rows interleave gate / up."""
    g = torch.Generator().manual_seed(500 + rank)
    w13 = torch.randint(
        0, 256, (experts, 2 * INTER, POOLED // 2), dtype=torch.uint8, generator=g
    )
    s13 = torch.randint(
        122, 127, (experts, 2 * INTER, POOLED // 32), dtype=torch.uint8, generator=g
    )
    w2 = torch.randint(
        0, 256, (experts, POOLED, INTER // 2), dtype=torch.uint8, generator=g
    )
    s2 = torch.randint(
        118, 123, (experts, POOLED, INTER // 32), dtype=torch.uint8, generator=g
    )
    return w13, s13, w2, s2


def preprocessed_experts(rank: int, experts: int, device: torch.device):
    """``raw_experts`` through the regular path's MoE weight preprocessing."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )

    w13, s13, w2, s2 = raw_experts(rank, experts)
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    for name, t in (
        ("w13_weight", w13),
        ("w13_weight_scale", s13),
        ("w2_weight", w2),
        ("w2_weight_scale", s2),
    ):
        setattr(module, name, torch.nn.Parameter(t.to(device), requires_grad=False))
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return (
        _extract_gluon_raw_w(module.w13_weight_triton_tensor),
        _extract_gluon_raw_s(module.w13_precision_config.b_mx_scale),
        _extract_gluon_raw_w(module.w2_weight_triton_tensor),
        _extract_gluon_raw_s(module.w2_precision_config.b_mx_scale),
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
