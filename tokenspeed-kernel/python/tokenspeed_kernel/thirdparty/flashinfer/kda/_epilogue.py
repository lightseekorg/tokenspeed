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

"""Headwise BF16 norm and prepared-FP8 epilogues; no FlashInfer import required."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import ArchVersion, current_platform


@triton.jit
def _gated_rmsnorm_bf16_kernel(
    x,
    gate,
    weight,
    out,
    EPS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    PDL: tl.constexpr,
    EARLY_SIGNAL: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, D)
    column = head * D + k
    w = tl.load(weight + k).to(tl.float32)
    g = tl.load(gate + row * GATE_STRIDE + column).to(tl.float32)
    if EARLY_SIGNAL:
        gate_sigmoid = tl.sigmoid(g)
    if PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_SIGNAL:
            tl.extra.cuda.gdc_launch_dependents()
    value = tl.load(x + row * H * D + column).to(tl.float32)
    variance = tl.sum(value * value, 0) / D
    if not EARLY_SIGNAL:
        gate_sigmoid = tl.sigmoid(g)
    normalized = value * tl.rsqrt(variance + EPS) * w * gate_sigmoid
    tl.store(out + row * H * D + column, normalized)
    if PDL and not EARLY_SIGNAL:
        tl.extra.cuda.gdc_launch_dependents()


def _gated_rmsnorm_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
    *,
    enable_pdl: bool,
) -> torch.Tensor:
    """Apply the BF16 decode epilogue with one warp per attention head."""
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    assert gate.shape == x.shape and gate.stride(-1) == 1 and head_dim == 128
    out = torch.empty_like(x)
    _gated_rmsnorm_bf16_kernel[(x.shape[0], num_heads)](
        x,
        gate,
        weight,
        out,
        EPS=eps,
        H=num_heads,
        D=head_dim,
        GATE_STRIDE=gate.stride(0),
        PDL=enable_pdl,
        EARLY_SIGNAL=(
            enable_pdl
            and num_heads == 12
            and current_platform().arch_version
            in (ArchVersion(10, 0), ArchVersion(10, 3))
        ),
        num_warps=1,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out


@triton.jit
def _gated_rmsnorm_fp8_kernel(
    x,
    gate,
    weight,
    values,
    scales,
    fp8_max,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    PADDED_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    EARLY_SIGNAL: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    column = head * DIM + tl.arange(0, DIM)
    w = tl.load(weight + tl.arange(0, DIM)).to(tl.float32)
    g = tl.load(gate + row * GATE_STRIDE + column, mask=row < ROWS, other=0).to(
        tl.float32
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_SIGNAL:
            tl.extra.cuda.gdc_launch_dependents()
    v = tl.load(x + row * HEADS * DIM + column, mask=row < ROWS, other=0).to(tl.float32)
    variance = tl.sum(v * v, 0) / DIM
    normalized = v * tl.rsqrt(variance + EPS) * w * tl.sigmoid(g)
    # Preserve the BF16 rounding of the original norm -> quantization chain.
    rounded = normalized.to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(rounded), 0)
    if ROWS % 4 == 0:
        # The unpadded baseline uses CUDA's round-to-nearest divisions.
        inverse = tl.where(amax == 0.0, 1.0, tl.div_rn(fp8_max, amax))
        scale = tl.div_rn(1.0, inverse)
    else:
        # The padded baseline uses the existing Triton quantizer.
        inverse = tl.where(amax == 0.0, 1.0, fp8_max / amax)
        scale = 1.0 / inverse
    quantized = tl.clamp(rounded * inverse, -fp8_max, fp8_max)
    tl.store(values + row * HEADS * DIM + column, quantized)
    tl.store(scales + head * PADDED_ROWS + row, scale)
    if ENABLE_PDL and not EARLY_SIGNAL:
        tl.extra.cuda.gdc_launch_dependents()


def gated_rmsnorm_fp8_prepacked(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded FP8 rows and contiguous [heads, padded_rows] FP32 scales."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.shape == gate.shape and gate.stride(-1) == 1
    assert head_dim == 128 and x.shape[1] == num_heads * head_dim
    rows = x.shape[0]
    padded_rows = triton.cdiv(rows, 4) * 4
    values = torch.empty(
        padded_rows, num_heads * head_dim, device=x.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(num_heads, padded_rows, device=x.device, dtype=torch.float32)
    _gated_rmsnorm_fp8_kernel[(padded_rows, num_heads)](
        x,
        gate,
        weight,
        values,
        scales,
        448.0,
        EPS=eps,
        ROWS=rows,
        PADDED_ROWS=padded_rows,
        HEADS=num_heads,
        DIM=head_dim,
        GATE_STRIDE=gate.stride(0),
        ENABLE_PDL=enable_pdl,
        EARLY_SIGNAL=(
            enable_pdl
            and num_heads == 12
            and current_platform().arch_version
            in (ArchVersion(10, 0), ArchVersion(10, 3))
        ),
        num_warps=1,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return values, scales
