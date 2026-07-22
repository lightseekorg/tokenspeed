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
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    M,
    N,
    x_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
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

    # PDL: wait for the producer kernel (e.g., kv_b_proj GEMM) to drain before
    # we read its output. No-op when disabled.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    if EVEN_N:
        load_mask = m_mask[:, None]
    else:
        load_mask = m_mask[:, None] & (n_idx[None, :] < N)

    x_off = m_idx[:, None] * x_row_stride + n_idx[None, :]
    x = tl.load(x_ptr + x_off, mask=load_mask)

    x = x.to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            scale = tl.load(scale)
        x = x * (1.0 / scale)
    x_fp8 = x.to(FP8_DTYPE)

    out_off = m_idx[:, None] * out_row_stride + n_idx[None, :]
    tl.store(out_ptr + out_off, x_fp8, mask=load_mask)

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
        fp8_dtype: ``torch.float8_e4m3fn`` (default), ``torch.float8_e5m2`` or
           ``torch.float8_e4m3fnuz`` (the bias-8 e4m3 used on AMD CDNA3).
        enable_pdl: opt into Programmatic Dependent Launch (Hopper+). Caller
           must also pass ``launch_pdl=True`` upstream / downstream as needed.

    Returns:
        FP8 tensor with the same shape as ``x``.
    """
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"fp8_quantize input must be bf16/fp16, got {x.dtype}"
    assert fp8_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
    ), f"fp8_quantize unsupported fp8 dtype: {fp8_dtype}"
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

    if fp8_dtype is torch.float8_e4m3fn:
        fp8_dtype_const = tl.float8e4nv
    elif fp8_dtype is torch.float8_e5m2:
        fp8_dtype_const = tl.float8e5
    else:
        fp8_dtype_const = tl.float8e4b8

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

    block_n = max(1, triton.next_power_of_2(N))
    even_n = block_n == N

    # ``launch_pdl`` is a NVIDIA-only Triton runtime kwarg (Hopper+ Programmatic
    # Dependent Launch). The HIP backend rejects unknown kwargs, so only forward
    # it when PDL is actually requested.
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}

    _fp8_quantize_kernel[grid](
        x,
        out,
        1.0 if scale is None else scale,
        M,
        N,
        x_row_stride,
        out_row_stride,
        BLOCK_N=block_n,
        EVEN_N=even_n,
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


@triton.jit
def _fp8_token_group_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    group_size,
    eps,
    bit8_min,
    bit8_max,
    BLOCK: tl.constexpr,
):
    group_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < group_size
    offsets = group_id.to(tl.int64) * group_size + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0), eps) / bit8_max
    out = tl.clamp(x / scale, bit8_min, bit8_max).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(scale_ptr + group_id, scale)


def _fp8_token_group_quantize(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"the last dimension of x must be divisible by group_size, got "
            f"shape={tuple(x.shape)}, group_size={group_size}"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    out_dtype = platform.fp8e4m3fn.dtype
    out = torch.empty_like(x, device=x.device, dtype=out_dtype)
    scales = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    groups = x.numel() // group_size
    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    bit8_max = platform.fp8e4m3fn.max
    bit8_min = -bit8_max

    _fp8_token_group_quantize_kernel[(groups,)](
        x,
        out,
        scales,
        group_size,
        1e-10,
        bit8_min=bit8_min,
        bit8_max=bit8_max,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return out, scales


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


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="triton_quantize_fp8_with_scale",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "granularity": frozenset({"token_group_32", "token_group_128"}),
        "scale_encoding": frozenset({"float32"}),
    },
    priority=Priority.PORTABLE,
)
def triton_quantize_fp8_with_scale(
    x: torch.Tensor,
    granularity: str = "tensor",
    group_size: int | None = None,
    scale_encoding: str = "float32",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if granularity != "token_group" or group_size not in {32, 128}:
        raise ValueError(
            "triton FP8 dynamic quantization supports only "
            f"granularity='token_group' with group_size 32 or 128, got "
            f"granularity={granularity!r}, group_size={group_size}."
        )
    if scale_encoding != "float32":
        raise ValueError(
            "triton FP8 dynamic quantization currently requires "
            f"scale_encoding='float32', got {scale_encoding!r}."
        )
    return _fp8_token_group_quantize(x.contiguous(), group_size)


@triton.jit
def _mxfp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    x_row_stride,
    out_row_stride,
    scale_row_stride,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offs = group * 32 + tl.arange(0, 32)

    x = tl.load(x_ptr + row * x_row_stride + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    exp = tl.ceil(tl.log2(amax / 448.0))
    exp = tl.clamp(exp, -127.0, 127.0)
    exp = tl.where(amax > 0, exp, -127.0)
    q = x * tl.exp2(-exp)
    q8 = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    tl.store(out_ptr + row * out_row_stride + offs, q8)
    tl.store(scale_ptr + row * scale_row_stride + group, (exp + 127.0).to(tl.uint8))


@register_kernel(
    "quantization",
    "mxfp8",
    name="triton_quantize_mxfp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=Priority.PORTABLE,
)
def mxfp8_quantize(
    x: torch.Tensor,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 tensor to MXFP8.

    The last dimension is quantized in groups of 32 values. The returned data
    has the same shape as ``x`` and FP8-E4M3 storage; the returned scales have
    shape ``x.shape[:-1] + (x.shape[-1] // 32,)`` and E8M0 byte encoding.
    """
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"mxfp8_quantize input must be bf16/fp16, got {x.dtype}"
    M, N, x_row_stride = _flatten_to_2d(x)
    if N % 32 != 0:
        raise ValueError("mxfp8_quantize requires the last dimension divisible by 32")

    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
    scales = torch.empty((*x.shape[:-1], N // 32), dtype=scale_dtype, device=x.device)

    out_M, _, out_row_stride = _flatten_to_2d(out)
    scales_M, _, scale_row_stride = _flatten_to_2d(scales)
    assert out_M == M and scales_M == M
    # enable_pdl is accepted for launch-site parity with other backends and unused here.

    _mxfp8_quantize_kernel[(M, N // 32)](
        x,
        out.view(torch.uint8),
        scales.view(torch.uint8),
        x_row_stride,
        out_row_stride,
        scale_row_stride,
        num_warps=1,
    )
    return out, scales


@triton.jit
def _mxfp4_quantize_block(x):
    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1
    amax = tl.max(tl.abs(x), axis=0)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    scale_byte = scale_e8m0_unbiased.to(tl.uint8) + 127
    qx = x * tl.exp2(-scale_e8m0_unbiased)
    qx = qx.to(tl.uint32, bitcast=True)

    sign = qx & 0x80000000
    qx = qx ^ sign
    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_exp: tl.constexpr = (127 - 1) + (23 - 1) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << 23
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (23 - 1)) & 1
    normal_x += 0xC11FFFFF
    normal_x += mant_odd
    normal_x = normal_x >> (23 - 1)
    normal_x = normal_x.to(tl.uint8)

    e2m1 = tl.full(x.shape, 0x7, dtype=tl.uint8)
    e2m1 = tl.where(normal_mask, normal_x, e2m1)
    e2m1 = tl.where(denormal_mask, denormal_x, e2m1)
    sign_lp = sign >> (23 + 8 - 1 - 2)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1 = e2m1 | sign_lp
    e2m1 = tl.reshape(e2m1, [16, 2])
    evens, odds = tl.split(e2m1)
    return evens | (odds << 4), scale_byte


@triton.jit
def _mxfp4_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    M,
    x_row_stride,
    out_row_stride,
    scale_row_stride,
    N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_k = pid_g * 32 + tl.arange(0, 32)
    x = tl.load(x_ptr + pid_m * x_row_stride + offs_k).to(tl.float32)

    packed, scale_byte = _mxfp4_quantize_block(x)
    pack_idx = tl.arange(0, 16)
    tl.store(out_ptr + pid_m * out_row_stride + pid_g * 16 + pack_idx, packed)
    tl.store(scale_ptr + pid_m * scale_row_stride + pid_g, scale_byte)


def mxfp4_quantize(
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 tensor to packed MXFP4.

    The last dimension is quantized in groups of 32 values. The returned data
    packs two E2M1 values per byte and returns one uint8 E8M0 scale per group.
    """
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"mxfp4_quantize input must be bf16/fp16, got {x.dtype}"
    M, N, x_row_stride = _flatten_to_2d(x)
    if N % 32 != 0:
        raise ValueError("mxfp4_quantize requires the last dimension divisible by 32")

    out_shape = (*x.shape[:-1], N // 2)
    scale_shape = (*x.shape[:-1], N // 32)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.uint8, device=x.device)
    else:
        assert out.shape == out_shape and out.dtype == torch.uint8
    if scales is None:
        scales = torch.empty(scale_shape, dtype=torch.uint8, device=x.device)
    else:
        assert scales.shape == scale_shape and scales.dtype == torch.uint8

    out_M, _, out_row_stride = _flatten_to_2d(out)
    scales_M, _, scale_row_stride = _flatten_to_2d(scales)
    assert out_M == M and scales_M == M

    grid = (M, N // 32)
    _mxfp4_quantize_kernel[grid](
        x,
        out,
        scales,
        M,
        x_row_stride,
        out_row_stride,
        scale_row_stride,
        N=N,
        num_warps=1,
    )
    return out, scales


@register_kernel(
    "quantization",
    "mxfp4",
    name="triton_quantize_mxfp4",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "scale_size": frozenset({32}),
        "scale_layout": frozenset({"linear"}),
        "has_global_scale": frozenset({False}),
        "scale_encoding": frozenset({"ue8m0"}),
    },
    priority=Priority.PORTABLE,
)
def triton_quantize_mxfp4(
    x: torch.Tensor,
    global_scale: float | None = None,
    scale_size: int = 32,
    scale_layout: str = "linear",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if global_scale is not None:
        raise ValueError("triton MXFP4 quantization does not support global_scale")
    if scale_size != 32:
        raise ValueError(
            f"triton MXFP4 quantization requires scale_size=32, got {scale_size}"
        )
    if scale_layout != "linear":
        raise ValueError(
            "triton MXFP4 quantization requires scale_layout='linear', "
            f"got {scale_layout!r}"
        )
    return mxfp4_quantize(x)


__all__ = [
    "fp8_quantize",
    "mxfp4_quantize",
    "triton_quantize_mxfp4",
    "triton_quantize_fp8_with_scale",
]
