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

"""Portable FP32 normalization and L2 clipping with explicit stores."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_IS_HIP = tl.constexpr(torch.version.hip is not None)


@triton.jit
def _torch_rsqrt(x):
    """FP32 ``x`` rounded as ``torch.rsqrt`` rounds it: the approximate
    instruction on CUDA, but correctly rounded on ROCm."""
    if _IS_HIP:
        return libdevice.rsqrt(x.to(tl.float64)).to(tl.float32)
    return tl.rsqrt(x)


@triton.jit
def _rms_normalized(value, WIDTH: tl.constexpr, EPS, MEAN_BY_RECIPROCAL: tl.constexpr):
    """``value`` over its RMS: FP32 mean of squares, a correctly rounded
    divide or with ``MEAN_BY_RECIPROCAL`` a multiply by FP32 ``1 / WIDTH``,
    plus EPS, then rsqrt."""
    total = tl.sum(value * value, 0)
    if MEAN_BY_RECIPROCAL:
        mean = total * (1.0 / WIDTH)
    else:
        mean = tl.div_rn(total, WIDTH)
    return value * _torch_rsqrt(mean + EPS)


@triton.jit
def _l2_clipped(value, threshold, eps):
    """``value`` scaled down to L2 norm ``threshold`` when it is longer. The
    norm is ``s * rsqrt(s)`` of the sum of squares ``s``, as the reference
    computes it."""
    sum_sq = tl.sum(value * value, 0)
    norm = tl.where(sum_sq > 0, sum_sq * _torch_rsqrt(sum_sq), 0.0)
    multiplier = tl.minimum(1.0, tl.div_rn(threshold, norm + eps))
    return value * multiplier


@triton.jit
def _rmsnorm_precision_kernel(
    X,
    Weight,
    Out,
    group,
    group_stride,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    MEAN_BY_RECIPROCAL: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        # Waits before it releases its successor, which may then prefetch.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    # Input rows come in groups of `group` adjacent rows, group_stride apart.
    source = (row // group) * group_stride + (row % group) * WIDTH
    value = tl.load(X + source + cols, cols < WIDTH, 0).to(tl.float32)
    value = _rms_normalized(value, WIDTH, EPS, MEAN_BY_RECIPROCAL)
    if HAS_WEIGHT:
        value = value * tl.load(Weight + cols, cols < WIDTH, 0).to(tl.float32)
    tl.store(Out + row * WIDTH + cols, value, cols < WIDTH)


@register_kernel(
    "layernorm",
    "rmsnorm_precision",
    name="triton_rmsnorm_precision",
    solution="triton",
    signatures=format_signatures(
        ("x",), "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_rmsnorm_precision(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    output_dtype: torch.dtype,
    *,
    mean_by_reciprocal: bool,
) -> torch.Tensor:
    """Normalize input rows into a contiguous output with one store per row.

    A ``[tokens, groups, width]`` view whose groups' rows are adjacent, such
    as per-head rows of a column slice, is read in place; other
    non-contiguous inputs are copied first.
    """
    width = x.shape[-1]
    if width > 65536:
        raise ValueError("RMSNorm width exceeds the portable Triton row limit")
    output = torch.empty(x.shape, device=x.device, dtype=output_dtype)
    rows = x.numel() // width
    group, group_stride = 1, width
    if not x.is_contiguous():
        if x.ndim == 3 and x.stride(2) == 1 and x.stride(1) == width:
            group, group_stride = x.shape[1], x.stride(0)
        else:
            x = x.contiguous()
    if rows:
        _rmsnorm_precision_kernel[(rows,)](
            x,
            weight,
            output,
            group,
            group_stride,
            width,
            eps,
            weight is not None,
            mean_by_reciprocal,
            triton.next_power_of_2(width),
            pdl_enabled(),
            num_warps=4 if width < 4096 else 8,
            enable_fp_fusion=False,
            **({"launch_pdl": True} if pdl_enabled() else {}),
        )
    return output


@triton.jit
def _add_rmsnorm_precision_kernel(
    X,
    Residual,
    Weight,
    Out,
    SumOut,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    MEAN_BY_RECIPROCAL: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """``_rmsnorm_precision_kernel`` over the FP32 sum of two rows, which it
    also stores once in ``SumOut``'s dtype."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    value = tl.load(X + row * WIDTH + cols, mask, 0).to(tl.float32) + tl.load(
        Residual + row * WIDTH + cols, mask, 0
    ).to(tl.float32)
    tl.store(SumOut + row * WIDTH + cols, value, mask)
    value = _rms_normalized(value, WIDTH, EPS, MEAN_BY_RECIPROCAL)
    if HAS_WEIGHT:
        value = value * tl.load(Weight + cols, mask, 0).to(tl.float32)
    tl.store(Out + row * WIDTH + cols, value, mask)


def triton_add_rmsnorm_precision(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    output_dtype: torch.dtype,
    sum_dtype: torch.dtype,
    *,
    mean_by_reciprocal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the FP32 sum of contiguous rows, also stored in ``sum_dtype``;
    the launch geometry of ``triton_rmsnorm_precision``, so the statistics
    reduce alike."""
    width = x.shape[-1]
    if width > 65536:
        raise ValueError("RMSNorm width exceeds the portable Triton row limit")
    output = torch.empty(x.shape, device=x.device, dtype=output_dtype)
    summed = torch.empty(x.shape, device=x.device, dtype=sum_dtype)
    rows = x.numel() // width
    if rows:
        _add_rmsnorm_precision_kernel[(rows,)](
            x,
            residual,
            weight,
            output,
            summed,
            width,
            eps,
            weight is not None,
            mean_by_reciprocal,
            triton.next_power_of_2(width),
            pdl_enabled(),
            num_warps=4 if width < 4096 else 8,
            enable_fp_fusion=False,
            **({"launch_pdl": True} if pdl_enabled() else {}),
        )
    return output, summed
