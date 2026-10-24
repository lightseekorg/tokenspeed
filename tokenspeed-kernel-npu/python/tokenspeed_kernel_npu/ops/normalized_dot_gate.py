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

"""Engram gate fusion retaining the existing CANN reductions and activations."""

from math import prod

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _prepare_gate(
    H, KV, QW, KW, HS, KS, PROD, N, D: tl.constexpr, HC: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    hc = tl.program_id(1)
    d = tl.arange(0, B)
    ok = d < D
    i = (row * HC + hc) * D + d
    h = tl.load(H + i, ok, 0).to(tl.float32)
    k = tl.load(KV + row * (HC + 1) * D + hc * D + d, ok, 0).to(tl.float32)
    qw = tl.load(QW + hc * D + d, ok, 0).to(tl.float32)
    kw = tl.load(KW + hc * D + d, ok, 0).to(tl.float32)
    tl.store(HS + i, h * h, ok)
    tl.store(KS + i, k * k, ok)
    tl.store(PROD + i, (h * (qw * kw)) * k, ok)


@triton.jit
def _apply_gate(H, KV, G, M, O, N, D: tl.constexpr, HC: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    hc = tl.program_id(1)
    d = tl.arange(0, B)
    ok = d < D
    i = (row * HC + hc) * D + d
    h = tl.load(H + i, ok, 0).to(tl.float32)
    v = tl.load(KV + row * (HC + 1) * D + HC * D + d, ok, 0).to(tl.float32)
    g = tl.load(G + row * HC + hc)
    g = tl.where(tl.load(M + row) != 0, g, 0.0)
    tl.store(O + i, h + g * v, ok)


def normalized_dot_gate(residual, key_value, query_weight, key_weight, mask, eps):
    """Preserve FP32 operation order; fuse only elementwise preparation and add.

    The common dispatcher validates contiguous colocated BF16 operands. CANN
    still owns means, sums, rsqrt, sqrt and sigmoid: fusing their reductions
    changes BF16 outputs close to rounding boundaries. All inputs stay live
    in native graph replay. Channel widths outside the bounded tile fall back
    to the original tensor expression.
    """
    hc, dim = residual.shape[-2:]
    n = prod(residual.shape[:-2])
    if not n:
        return torch.empty_like(residual)
    fast = dim % 32 == 0 and dim <= 8192
    if fast:
        h_squared = torch.empty_like(residual, dtype=torch.float32)
        key_squared = torch.empty_like(h_squared)
        products = torch.empty_like(h_squared)
        _prepare_gate[(n, hc)](
            residual,
            key_value,
            query_weight,
            key_weight,
            h_squared,
            key_squared,
            products,
            n,
            dim,
            hc,
            triton.next_power_of_2(dim),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    else:
        key = key_value[..., : hc * dim].float().unflatten(-1, (hc, dim))
        h = residual.float()
        weight = query_weight.float() * key_weight.float()
        h_squared, key_squared, products = h.square(), key.square(), h * weight * key
    rstd = torch.rsqrt(h_squared.mean(-1) + eps) * torch.rsqrt(
        key_squared.mean(-1) + eps
    )
    dot = products.sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    negative = dot.contiguous().view(torch.int32) < 0
    gate = torch.sigmoid(torch.where(negative, -magnitude, magnitude))
    if fast:
        output = torch.empty_like(residual)
        _apply_gate[(n, hc)](
            residual,
            key_value,
            gate,
            mask,
            output,
            n,
            dim,
            hc,
            triton.next_power_of_2(dim),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        return output
    gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    value = key_value[..., hc * dim :].float()
    return (residual.float() + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(
        residual.dtype
    )
