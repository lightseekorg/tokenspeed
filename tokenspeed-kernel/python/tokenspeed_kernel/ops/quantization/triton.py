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
from typing import Optional

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    M,
    x_row_stride,
    out_row_stride,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N

    # PDL: wait for the producer kernel (e.g., kv_b_proj GEMM) to drain before
    # we read its output. No-op when disabled.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    x_off = m_idx[:, None] * x_row_stride + n_idx[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)

    x = x.to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            scale = tl.load(scale)
        x = x * (1.0 / scale)
    x_fp8 = x.to(FP8_DTYPE)

    out_off = m_idx[:, None] * out_row_stride + n_idx[None, :]
    tl.store(out_ptr + out_off, x_fp8, mask=mask)

    # PDL: signal that dependents (e.g., FMHA) can begin their preamble.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _flatten_to_2d(x: torch.Tensor):
    """Flatten leading dims into a single M, returning (M, N, row_stride).

    Requires stride(-1) == 1 and that all leading dims pack contiguously
    onto the row stride — i.e. ``stride(d) == shape(d+1) * stride(d+1)`` for
    every ``d < ndim - 2``. This holds for fully-contiguous tensors and for
    last-dim slices like ``kv[..., qk_nope:]`` where the leading dims still
    pack onto a uniform row stride.
    """
    assert x.stride(-1) == 1, f"expected stride-1 inner dim, got stride={x.stride(-1)}"
    N = x.shape[-1]
    if x.ndim == 1:
        return 1, N, N
    M = x.numel() // N
    row_stride = x.stride(-2)
    # Validate that every leading dim packs onto the next.
    for d in range(x.ndim - 2):
        expected = x.shape[d + 1] * x.stride(d + 1)
        if x.stride(d) != expected:
            raise ValueError(
                f"cannot flatten dim {d}: stride={x.stride(d)} but expected "
                f"shape[{d+1}]*stride[{d+1}]={expected}. Tensor shape={tuple(x.shape)}, "
                f"stride={tuple(x.stride())}."
            )
    return M, N, row_stride


def fp8_quantize(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Cast a BF16/FP16 tensor to FP8 with an optional per-tensor scale.

    Computes ``out = saturate((x / scale) -> fp8)`` element-wise when scale is
    provided. When scale is omitted, this is a pure FP8 cast.

    Args:
        x: BF16 or FP16 tensor. Must have stride(-1) == 1; leading dims must
           pack uniformly onto the row stride (true for contiguous tensors and
           for last-dim slice views like ``kv[..., qk_nope:]``).
        scale: optional scalar divisor applied before the cast. Python values
           are passed as plain kernel args; scalar tensors are loaded on device.
        out: optional pre-allocated FP8 output. Same shape as ``x``. If not
           provided, allocated as contiguous.
        fp8_dtype: ``torch.float8_e4m3fn`` (default) or ``torch.float8_e5m2``.
        enable_pdl: opt into Programmatic Dependent Launch (Hopper+). Caller
           must also pass ``launch_pdl=True`` upstream / downstream as needed.

    Returns:
        FP8 tensor with the same shape as ``x``.
    """
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"fp8_quantize input must be bf16/fp16, got {x.dtype}"
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    has_scale = scale is not None
    has_scale_tensor = isinstance(scale, torch.Tensor)
    if has_scale_tensor:
        assert scale.numel() == 1, "scale tensor must be scalar"
        scale = scale.contiguous()

    M, N, x_row_stride = _flatten_to_2d(x)

    if out is None:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
    else:
        assert out.shape == x.shape and out.dtype == fp8_dtype
    out_M, _, out_row_stride = _flatten_to_2d(out)
    assert out_M == M

    fp8_dtype_const = tl.float8e4nv if fp8_dtype is torch.float8_e4m3fn else tl.float8e5
    block_n = triton.next_power_of_2(N)

    # Block-size heuristic — picked from per-shape best configs in an
    # nsys-driven sweep on B200 (kv_a [s,512] and v [s,h,128] for K2.5).
    # Pattern: num_warps=4, num_stages=2 win universally; BLOCK_M ramps with
    # M to amortize launch as the grid grows.
    # See tasks/k2.5_optimization/{tune_fp8_quantize_nsys,parse_tune_fp8_quantize_nsys}.py
    if M <= 2048:
        block_m = 4
    elif M <= 16384:
        block_m = 16
    else:
        block_m = 32
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, block_m),)

    # ``launch_pdl`` is a NVIDIA-only Triton runtime kwarg (Hopper+ Programmatic
    # Dependent Launch). The HIP backend rejects unknown kwargs, so only forward
    # it when PDL is actually requested.
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}

    _fp8_quantize_kernel[grid](
        x,
        out,
        1.0 if scale is None else scale,
        M,
        x_row_stride,
        out_row_stride,
        N=N,
        BLOCK_N=block_n,
        FP8_DTYPE=fp8_dtype_const,
        BLOCK_M=block_m,
        HAS_SCALE=has_scale,
        HAS_SCALE_TENSOR=has_scale_tensor,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kwargs,
    )
    return out


@register_kernel(
    "quantization",
    "fp8",
    name="triton_quantize_fp8",
    solution="triton",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={"has_scale": frozenset({True, False})},
    priority=Priority.PORTABLE,
)
def triton_quantize_fp8(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    return fp8_quantize(x, scale=scale, enable_pdl=enable_pdl)


__all__ = [
    "fp8_quantize",
]
