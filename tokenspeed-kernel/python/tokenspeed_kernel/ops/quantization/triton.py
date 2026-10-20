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
import math
from typing import Optional

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


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
                f"shape[{d + 1}]*stride[{d + 1}]={expected}. Tensor shape={tuple(x.shape)}, "
                f"stride={tuple(x.stride())}."
            )
    return M, N, row_stride


def fp8_quantize(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool | None = None,
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
        enable_pdl: opt into Programmatic Dependent Launch (Hopper+), defaulting
           to the platform setting. Caller must also enable PDL upstream /
           downstream as needed.

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
    else:
        fp8_dtype_const = tl.float8e5

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
    num_stages = 2

    block_n = max(1, triton.next_power_of_2(N))
    even_n = block_n == N
    # Wide rows spill registers at those tiles: cap elements per program, add warps.
    block_m = max(1, min(block_m, 16384 // block_n))
    num_warps = 8 if block_n >= 2048 else 4

    grid = (triton.cdiv(M, block_m),)

    # ``launch_pdl`` is a NVIDIA-only Triton runtime kwarg (Hopper+ Programmatic
    # Dependent Launch). The HIP backend rejects unknown kwargs, so only forward
    # it when PDL is actually requested.
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
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


# Dynamic per-token FP8 E4M3 activation quantization (compressed-tensors
# FP8_DYNAMIC), by TRT-LLM's rule (perTokenQuantization with FP8 min scaling,
# as its FP8 per-channel MoE quantizes its input): for a row with largest
# magnitude amax, the dequant scale is max(amax / 448, 1 / (448 * 512)) and the
# codes are RN(x * min(448 / amax, 448 * 512)), saturated to +-448; a zero row
# quantizes to zeros. Both divisions are IEEE-rounded (div_rn), so every kernel
# that inlines these helpers (the standalone quantizer below, the small-M FP8
# GEMM's prologue, the RMSNorm epilogue) writes the same scales and codes.
_FP8_E4M3_MAX = tl.constexpr(448.0)
_FP8_TOKEN_MIN_SCALE = tl.constexpr(1.0 / (448.0 * 512.0))
_FP8_TOKEN_MAX_MULTIPLIER = tl.constexpr(448.0 * 512.0)


@triton.jit
def fp8_token_scale(amax):
    """FP32 dequant scale of a row (or rows) whose largest magnitude is ``amax``."""
    return tl.maximum(tl.div_rn(amax, _FP8_E4M3_MAX), _FP8_TOKEN_MIN_SCALE)


@triton.jit
def fp8_token_multiplier(amax):
    """FP32 quantization multiplier of a row whose largest magnitude is ``amax``."""
    return tl.minimum(tl.div_rn(_FP8_E4M3_MAX, amax), _FP8_TOKEN_MAX_MULTIPLIER)


@triton.jit
def fp8_quantize_by_multiplier(x, multiplier):
    """FP8 E4M3 codes of FP32 ``x`` times its quantization ``multiplier``."""
    q = tl.clamp(x * multiplier, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return q.to(tl.float8e4nv, fp_downcast_rounding="rtne")


@triton.jit
def fp8_dequantize_unscaled(raw, token_scale, channel_scale):
    """The scaled FP8 GEMM's output from an unscaled one, widened to FP32.

    ``raw`` is a 16-bit load of an FP8 GEMM run with unit scales (the codes'
    product, rounded once to the 16-bit output dtype); ``token_scale`` and
    ``channel_scale`` are the FP32 per-token and per-channel dequant scales it
    did not apply. The result is ``(raw * token_scale) * channel_scale`` in
    FP32 (token first, as the scaled GEMMs' epilogues multiply), rounded to
    ``raw``'s dtype as their output is, then widened back to FP32.
    """
    value = (raw.to(tl.float32) * token_scale) * channel_scale
    return value.to(raw.dtype).to(tl.float32)


def dequantize_unscaled_fp8(
    raw: torch.Tensor, token_scales: torch.Tensor, channel_scales: torch.Tensor
) -> torch.Tensor:
    """``fp8_dequantize_unscaled`` in PyTorch, for a whole ``[M, N]`` output.

    Args:
        raw: ``[M, N]`` 16-bit output of an FP8 GEMM run with unit scales.
        token_scales: FP32 ``[M]`` / ``[M, 1]`` per-token scales.
        channel_scales: FP32 ``[N]`` / ``[N, 1]`` per-channel scales.

    Returns:
        ``[M, N]`` tensor in ``raw``'s dtype: ``(raw * token) * channel`` in
        FP32, rounded once.
    """
    value = raw.float() * token_scales.reshape(-1, 1)
    return (value * channel_scales.reshape(1, -1)).to(raw.dtype)


@triton.jit
def _fp8_token_quantize_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    K,
    x_row_stride,
    q_row_stride,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    codes = fp8_quantize_by_multiplier(x, fp8_token_multiplier(amax))
    tl.store(q_ptr + row * q_row_stride + cols, codes, mask=mask)
    tl.store(scale_ptr + row, fp8_token_scale(amax))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def fp8_quantize_per_token(
    x: torch.Tensor, enable_pdl: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the rows of ``x`` to FP8 E4M3 with one FP32 scale each, in one launch.

    Args:
        x: ``[M, K]`` BF16 or FP16 activations with a contiguous last axis.
        enable_pdl: Launch with Programmatic Dependent Launch (Hopper+); None
            takes the platform setting.

    Returns:
        ``(codes, scales)``: contiguous ``[M, K]`` FP8 E4M3 codes and FP32
        ``[M, 1]`` dequant scales, ``x ~= codes * scales``.
    """
    if x.ndim != 2 or x.stride(-1) != 1:
        raise ValueError(
            f"x must be [M, K] with a contiguous last axis, got shape "
            f"{tuple(x.shape)} stride {x.stride()}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"x must be BF16 or FP16, got {x.dtype}")
    m, k = x.shape
    codes = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m == 0:
        return codes, scales
    block = triton.next_power_of_2(k)
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    _fp8_token_quantize_kernel[(m,)](
        x,
        codes,
        scales,
        k,
        x.stride(0),
        codes.stride(0),
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        num_warps=min(max(block // 512, 1), 8),
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return codes, scales


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

    out_dtype = torch.float8_e4m3fn
    out = torch.empty_like(x, device=x.device, dtype=out_dtype)
    scales = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    groups = x.numel() // group_size
    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    fp8_finfo = torch.finfo(out_dtype)

    _fp8_token_group_quantize_kernel[(groups,)](
        x,
        out,
        scales,
        group_size,
        1e-10,
        bit8_min=fp8_finfo.min,
        bit8_max=fp8_finfo.max,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return out, scales


@triton.jit
def _fp8_block_quantize_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    n,
    k,
    x_stride_n,
    scale_stride_n,
    block_n,
    block_k,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_n_id = tl.program_id(0)
    block_k_id = tl.program_id(1)
    rows = tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_K)
    row_offsets = block_n_id * block_n + rows
    column_offsets = block_k_id * block_k + columns
    mask = (rows[:, None] < block_n) & (columns[None, :] < block_k)
    mask &= (row_offsets[:, None] < n) & (column_offsets[None, :] < k)
    offsets = row_offsets[:, None].to(tl.int64) * x_stride_n + column_offsets[None, :]

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values)), eps) / fp8_max
    quantized = tl.clamp(values / scale, fp8_min, fp8_max).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, quantized, mask=mask)
    tl.store(scale_ptr + block_n_id * scale_stride_n + block_k_id, scale)


def _fp8_block_quantize(
    x: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"block FP8 quantization requires a 2-D tensor, got {x.ndim}D")
    block_n, block_k = block_size
    x = x.contiguous()
    n, k = x.shape
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (triton.cdiv(n, block_n), triton.cdiv(k, block_k)),
        dtype=torch.float32,
        device=x.device,
    )
    fp8_finfo = torch.finfo(torch.float8_e4m3fn)
    _fp8_block_quantize_kernel[(scales.shape[0], scales.shape[1])](
        x,
        out,
        scales,
        n,
        k,
        x.stride(0),
        scales.stride(0),
        block_n,
        block_k,
        1e-10,
        fp8_finfo.min,
        fp8_finfo.max,
        BLOCK_N=triton.next_power_of_2(block_n),
        BLOCK_K=triton.next_power_of_2(block_k),
        num_warps=8,
    )
    return out, scales


@triton.jit
def _fp8_roundtrip_kernel(
    x_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    groups_per_row: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    row = group_id // groups_per_row
    column_group = group_id - row * groups_per_row
    columns = column_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    x = tl.load(x_ptr + row * x_row_stride + columns).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / 448.0)))
    quantized = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    dequantized = quantized.to(tl.float32) * scale
    tl.store(out_ptr + row * out_row_stride + columns, dequantized)


@register_kernel(
    "quantization",
    "fp8",
    name="triton_quantize_fp8_roundtrip",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    traits={
        "dequantize": frozenset({True}),
        "group_size": frozenset({64, 128}),
        "scale_encoding": frozenset({"ue8m0"}),
    },
    priority=Priority.PORTABLE,
)
def triton_quantize_fp8_roundtrip(
    x: torch.Tensor,
    group_size: int,
    scale_encoding: str,
) -> torch.Tensor:
    if group_size not in {64, 128}:
        raise ValueError(f"unsupported FP8 round-trip group_size: {group_size}")
    if scale_encoding != "ue8m0":
        raise ValueError(
            f"unsupported FP8 round-trip scale encoding: {scale_encoding!r}"
        )

    rows, width, x_row_stride = _flatten_to_2d(x)
    if width % group_size != 0:
        raise ValueError(
            f"last dimension {width} must be divisible by group_size {group_size}"
        )
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    out_rows, _, out_row_stride = _flatten_to_2d(out)
    assert out_rows == rows
    groups_per_row = width // group_size
    _fp8_roundtrip_kernel[(rows * groups_per_row,)](
        x,
        out,
        x_row_stride,
        out_row_stride,
        groups_per_row=groups_per_row,
        GROUP_SIZE=group_size,
        num_warps=1,
        num_stages=1,
    )
    return out


@register_kernel(
    "quantization",
    "fp8",
    name="triton_quantize_fp8",
    solution="triton",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "dequantize": frozenset({False}),
        "has_scale": frozenset({True, False}),
    },
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
    name="triton_quantize_fp8_block",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    traits={
        "granularity": frozenset({"block"}),
        "scale_encoding": frozenset({"float32"}),
    },
    priority=Priority.PORTABLE,
)
def triton_quantize_fp8_block(
    x: torch.Tensor,
    *,
    granularity: str,
    group_size: int | None,
    block_size: tuple[int, int],
    scale_encoding: str,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if granularity != "block":
        raise ValueError("block FP8 quantization requires block granularity")
    if enable_pdl:
        raise ValueError("block FP8 quantization does not support enable_pdl=True")
    if scale_encoding != "float32":
        raise ValueError("block FP8 quantization requires float32 scales")
    return _fp8_block_quantize(x, block_size)


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="triton_quantize_fp8_with_scale",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "granularity": frozenset({"token", "token_group_32", "token_group_128"}),
        "scale_encoding": frozenset({"float32"}),
    },
    priority=Priority.PORTABLE,
)
def triton_quantize_fp8_with_scale(
    x: torch.Tensor,
    granularity: str = "token",
    group_size: int | None = None,
    scale_encoding: str = "float32",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_encoding != "float32":
        raise ValueError(
            "triton FP8 dynamic quantization currently requires "
            f"scale_encoding='float32', got {scale_encoding!r}."
        )
    if granularity == "token":
        width = x.shape[-1]
        values, scales = _fp8_token_group_quantize(x.contiguous(), width)
        return values, scales
    if granularity == "token_group" and group_size in {32, 128}:
        return _fp8_token_group_quantize(x.contiguous(), group_size)
    raise ValueError(
        "triton FP8 dynamic quantization supports token or token_group "
        f"with group_size 32 or 128; got granularity={granularity!r}, "
        f"group_size={group_size}."
    )


# Up to this many rows, fp8_quantize_per_token takes about a third of the
# vendor token quantizer's time on GB200.
_FP8_PER_TOKEN_MAX_M = 64


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="triton_quantize_fp8_per_token",
    solution="triton",
    features={"token_fp8_min_scaling"},
    # SM100 and newer, where the per-token FP8 GEMMs that take its output run.
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "m_max": frozenset({_FP8_PER_TOKEN_MAX_M}),
        "granularity": frozenset({"token"}),
        "scale_encoding": frozenset({"float32"}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_quantize_fp8_per_token(
    x: torch.Tensor,
    *,
    granularity: str,
    group_size: int | None,
    scale_encoding: str,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``quantize_fp8(x, granularity="token")`` by ``fp8_quantize_per_token``.

    Selected for a 2-D ``x`` of at most 64 rows (the ``m`` trait).

    Args:
        x: ``[M, K]`` BF16 or FP16 activations with a contiguous last axis.
        granularity: Must be ``"token"``.
        group_size: Must be None; a token is one group.
        scale_encoding: Must be ``"float32"``.
        enable_pdl: Launch with Programmatic Dependent Launch.

    Returns:
        FP8 E4M3 ``[M, K]`` codes and FP32 ``[M, 1]`` dequant scales.
    """
    if granularity != "token" or group_size is not None or scale_encoding != "float32":
        raise ValueError(
            "the per-token FP8 quantizer takes token granularity with float32 "
            f"scales, got granularity={granularity!r}, group_size={group_size}, "
            f"scale_encoding={scale_encoding!r}"
        )
    return fp8_quantize_per_token(x, enable_pdl=enable_pdl)


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="triton_quantize_fp8_per_token_portable",
    solution="triton",
    features={"token_fp8_min_scaling"},
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "granularity": frozenset({"token"}),
        "scale_encoding": frozenset({"float32"}),
    },
    priority=Priority.PORTABLE + 1,
)
def triton_quantize_fp8_per_token_portable(
    x: torch.Tensor,
    *,
    granularity: str,
    group_size: int | None,
    scale_encoding: str,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable per-token E4M3 quantization with a minimum scale, for any row count.

    Args:
        x: BF16 or FP16 activations; the last dimension is one token.
        granularity: Must be ``"token"``.
        group_size: Must be None.
        scale_encoding: Must be ``"float32"``.
        enable_pdl: Launch with Programmatic Dependent Launch on Hopper+.

    Returns:
        FP8 E4M3 codes with the input shape and FP32 scales with a singleton
        last dimension. The scale
        floor and independently rounded reciprocal match the shared token
        quantizer, including zero and tiny rows.
    """
    if enable_pdl and not current_platform().is_hopper_plus:
        raise ValueError("this platform does not support enable_pdl")
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    values, scales = triton_quantize_fp8_per_token(
        rows,
        granularity=granularity,
        group_size=group_size,
        scale_encoding=scale_encoding,
        enable_pdl=enable_pdl,
    )
    return values.reshape(shape), scales.reshape(*shape[:-1], 1)


# The widest SwiGLU half (intermediate size) one program holds: a row per
# program, for the per-token amax of the FP8 output.
_SILU_AND_MUL_FP8_MAX_INTERMEDIATE_SIZE = 16384


@triton.jit
def _silu_and_mul_quantize_fp8_kernel(
    y_ptr,
    token_scale_ptr,
    channel_scale_ptr,
    codes_ptr,
    scale_ptr,
    width,
    y_row_stride,
    codes_row_stride,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    # The channel scales are parameters nothing in the decode graph writes.
    gate_scale = tl.load(channel_scale_ptr + cols, mask=mask, other=0.0)
    up_scale = tl.load(channel_scale_ptr + width + cols, mask=mask, other=0.0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    token_scale = tl.load(token_scale_ptr + row)
    y_row = y_ptr + row * y_row_stride
    gate = fp8_dequantize_unscaled(
        tl.load(y_row + cols, mask=mask, other=0.0), token_scale, gate_scale
    )
    up = fp8_dequantize_unscaled(
        tl.load(y_row + width + cols, mask=mask, other=0.0), token_scale, up_scale
    )
    # silu_and_mul's FP32 SwiGLU, rounded once to the 16-bit dtype it stores.
    act = (gate * tl.sigmoid(gate) * up).to(y_ptr.dtype.element_ty).to(tl.float32)
    act = tl.where(mask, act, 0.0)
    amax = tl.max(tl.abs(act), axis=0)
    codes = fp8_quantize_by_multiplier(act, fp8_token_multiplier(amax))
    tl.store(codes_ptr + row * codes_row_stride + cols, codes, mask=mask)
    tl.store(scale_ptr + row, fp8_token_scale(amax))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def triton_silu_and_mul_quantize_fp8(
    x: torch.Tensor,
    token_scales: torch.Tensor,
    channel_scales: torch.Tensor,
    *,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu_and_mul_quantize_fp8`` in one launch with one program per row.

    Each element is the scaled GEMM's 16-bit output
    (``fp8_dequantize_unscaled``), the SwiGLU ``silu(gate) * up`` is evaluated
    in FP32 and rounded once to ``x``'s dtype as ``silu_and_mul`` stores it,
    and that row is quantized as ``fp8_quantize_per_token`` would quantize it.

    Args:
        x: ``[M, 2 * I]`` BF16 or FP16 unscaled GEMM output (gate columns,
            then up columns) with a contiguous last axis.
        token_scales: FP32 ``[M, 1]`` or ``[M]`` per-token scales of the
            GEMM's FP8 input.
        channel_scales: FP32 ``[2 * I, 1]`` or ``[2 * I]`` per-channel weight
            scales of the GEMM.
        enable_pdl: Launch with Programmatic Dependent Launch.

    Returns:
        ``(codes, scales)``: contiguous FP8 E4M3 ``[M, I]`` codes and FP32
        ``[M, 1]`` scales of the activation.

    Raises:
        ValueError: A layout or dtype outside the above, or ``I`` above 16384.
    """
    if x.ndim != 2 or x.stride(-1) != 1 or x.shape[-1] % 2:
        raise ValueError(
            "x must be [M, 2 * I] with a contiguous last axis, got shape "
            f"{tuple(x.shape)} stride {x.stride()}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"x must be BF16 or FP16, got {x.dtype}")
    m, width = x.shape[0], x.shape[1] // 2
    if width > _SILU_AND_MUL_FP8_MAX_INTERMEDIATE_SIZE:
        raise ValueError(
            f"I={width} exceeds the one-row-per-program limit "
            f"{_SILU_AND_MUL_FP8_MAX_INTERMEDIATE_SIZE}"
        )
    for name, scales, count in (
        ("token_scales", token_scales, m),
        ("channel_scales", channel_scales, 2 * width),
    ):
        if (
            scales.dtype != torch.float32
            or scales.numel() != count
            or not scales.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be {count} contiguous FP32 values, got "
                f"{scales.dtype} {tuple(scales.shape)}"
            )
    codes = torch.empty((m, width), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m == 0:
        return codes, scales
    block = triton.next_power_of_2(width)
    _silu_and_mul_quantize_fp8_kernel[(m,)](
        x,
        token_scales,
        channel_scales,
        codes,
        scales,
        width,
        x.stride(0),
        codes.stride(0),
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        num_warps=min(max(block // 512, 1), 16),
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return codes, scales


def triton_rmsnorm_quantize_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None,
    round_residual_sum_bf16: bool,
    x_scale: float | None,
    residual_scale: float | None,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """``rmsnorm_quantize_fp8`` by the Triton RMSNorm, whose launch also writes
    the per-token FP8 codes and scales of its output (``token_fp8_out``).

    Arguments and returns are those of ``rmsnorm_quantize_fp8``.
    """
    # The Triton RMSNorm module imports this module's FP8 helpers.
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    rows = x.numel() // x.shape[-1]
    codes = torch.empty((rows, x.shape[-1]), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    normed = rmsnorm(
        x,
        weight,
        eps,
        residual,
        enable_pdl=enable_pdl,
        round_residual_sum_bf16=round_residual_sum_bf16,
        x_scale=x_scale,
        residual_scale=residual_scale,
        token_fp8_out=(codes, scales),
    )
    if residual is None:
        return normed, None, codes, scales
    out, residual_sum = normed
    return out, residual_sum, codes, scales


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
def triton_quantize_mxfp8(
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
    ), f"triton_quantize_mxfp8 input must be bf16/fp16, got {x.dtype}"
    M, N, x_row_stride = _flatten_to_2d(x)
    if N % 32 != 0:
        raise ValueError(
            "triton_quantize_mxfp8 requires the last dimension divisible by 32"
        )

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


NVFP4_BLOCK = 16


@triton.jit
def _rcp_approx(x):
    """flashinfer's reciprocal: the approximate, flush-to-zero PTX instruction."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mailbox_row_dirty(halves, sentinel_half: tl.constexpr):
    """Whether any BF16 element of the row still holds the sentinel's half."""
    return tl.max((halves == sentinel_half).to(tl.int32), axis=0) != 0


@triton.jit
def _rmsnorm_nvfp4_quant_kernel(
    x_ptr,
    weight_ptr,
    global_scale_ptr,
    out_ptr,
    scale_ptr,
    width: tl.constexpr,
    eps: tl.constexpr,
    weight_offset: tl.constexpr,
    sentinel_word: tl.constexpr,
    sentinel_half: tl.constexpr,
    MAILBOX: tl.constexpr,
    BLOCK: tl.constexpr,
    SF_BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # flashinfer: sf = e4m3(gs * amax * rcp(6)), y = e2m1(x * rcp(sf * rcp(gs))).
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    if MAILBOX:
        # One row-wide poll load keeps the plain path's variance sum order bitwise.
        half_ptr = x_ptr.to(tl.pointer_type(tl.int16)) + row * width + offsets
        halves = tl.load(half_ptr, mask=mask, other=0, volatile=True)
        while _mailbox_row_dirty(halves, sentinel_half):
            halves = tl.load(half_ptr, mask=mask, other=0, volatile=True)
        values = halves.to(tl.bfloat16, bitcast=True).to(tl.float32)
        word_ptr = x_ptr.to(tl.pointer_type(tl.int32)) + row * (width // 2)
        word_ptr += tl.arange(0, BLOCK // 2)
        tl.store(word_ptr, tl.full([BLOCK // 2], sentinel_word, tl.int32))
    else:
        values = tl.load(x_ptr + row * width + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    variance = tl.sum(values * values, axis=0) / width
    values *= tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values *= (weight + weight_offset).to(tl.bfloat16).to(tl.float32)
    values = values.to(tl.bfloat16).to(tl.float32)
    # flashinfer's quantizer is flush-to-zero: a subnormal input is a zero to it.
    values = tl.where(tl.abs(values) < 1.1754944e-38, 0.0, values)
    blocks = tl.reshape(values, [BLOCK // SF_BLOCK, SF_BLOCK])
    amax = tl.max(tl.abs(blocks), axis=1)
    global_scale = tl.load(global_scale_ptr) + tl.zeros_like(amax)
    rcp_six = _rcp_approx(tl.full(amax.shape, 6.0, tl.float32))
    sf_value = tl.where(amax != 0.0, global_scale * (amax * rcp_six), 0.0)
    sf_e4m3 = sf_value.to(tl.float8e4nv)
    sf_f32 = sf_e4m3.to(tl.float32)
    out_scale = _rcp_approx(sf_f32 * _rcp_approx(global_scale))
    out_scale = tl.where(amax != 0.0, out_scale, 0.0)
    scaled = tl.reshape(blocks * out_scale[:, None], [BLOCK // 2, 2])
    low, high = tl.split(scaled)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $2, $1; mov.b32 $0, {b, b, b, b}; }",
        "=r,f,f",
        [low, high],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    half = tl.arange(0, BLOCK // 2)
    tl.store(
        out_ptr + row * (width // 2) + half,
        (packed & 0xFF).to(tl.uint8),
        mask=half < width // 2,
    )
    sf_index = tl.arange(0, BLOCK // SF_BLOCK)
    tl.store(
        scale_ptr + row * (width // SF_BLOCK) + sf_index,
        sf_e4m3.to(tl.uint8, bitcast=True),
        mask=sf_index < width // SF_BLOCK,
    )


@register_kernel(
    "quantization",
    "rmsnorm_nvfp4",
    name="triton_rmsnorm_nvfp4_quant",
    solution="triton",
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def triton_rmsnorm_nvfp4_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    global_scale: torch.Tensor,
    mailbox_sentinel: int | None,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton kernel of :func:`tokenspeed_kernel.ops.quantization.rmsnorm_nvfp4_quant`."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("x must be a BF16 GPU tensor shaped [rows, width]")
    rows, width = x.shape
    if width % NVFP4_BLOCK:
        raise ValueError(f"width must be a multiple of {NVFP4_BLOCK}, got {width}")
    if weight.shape != (width,) or weight.device != x.device:
        raise ValueError("weight must be colocated and match x's final dimension")
    if global_scale.numel() != 1 or global_scale.dtype != torch.float32:
        raise ValueError("global_scale must be one FP32 value")
    if mailbox_sentinel is not None:
        if not 0 <= mailbox_sentinel <= 0xFFFFFFFF or (mailbox_sentinel & 0xFFFF) != (
            mailbox_sentinel >> 16
        ):
            raise ValueError("the mailbox sentinel is one word whose halves agree")
        if width & (width - 1) or not x.is_contiguous():
            raise ValueError("a mailbox needs contiguous rows of a power-of-two width")
    values = x.contiguous()
    packed = torch.empty((rows, width // 2), dtype=torch.uint8, device=x.device)
    # flashinfer's consumers may touch the round_up(rows, 16) tail; keep that storage.
    padded_rows = (rows + 15) // 16 * 16
    scales = torch.empty(
        (padded_rows, width // NVFP4_BLOCK), dtype=torch.uint8, device=x.device
    )[:rows]
    if rows == 0:
        return packed, scales
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise ValueError("the normalized dimension is too wide for one Triton block")
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    sentinel = 0 if mailbox_sentinel is None else mailbox_sentinel
    _rmsnorm_nvfp4_quant_kernel[(rows,)](
        values,
        weight,
        global_scale,
        packed,
        scales,
        width=width,
        eps=eps,
        weight_offset=weight_offset,
        # The kernel compares int16 halves and stores int32 words: signed images.
        sentinel_word=sentinel - (1 << 32) if sentinel >= 1 << 31 else sentinel,
        sentinel_half=(sentinel & 0xFFFF) - (1 << 16 if sentinel & 0x8000 else 0),
        MAILBOX=mailbox_sentinel is not None,
        BLOCK=block,
        SF_BLOCK=NVFP4_BLOCK,
        ENABLE_PDL=use_pdl,
        num_warps=8,
        **launch,
    )
    return packed, scales


__all__ = [
    "fp8_quantize",
    "mxfp4_quantize",
    "triton_quantize_fp8_roundtrip",
    "triton_quantize_mxfp4",
    "triton_rmsnorm_nvfp4_quant",
    "triton_quantize_fp8_with_scale",
]


@triton.jit
def _fp8_group32_ue8m0_quantize(X, Q, S, K: tl.constexpr, X0: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    row = group // (K // 32)
    column = (group % (K // 32)) * 32 + tl.arange(0, 32)
    values = tl.load(X + row * X0 + column).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-4)
    raw = amax * (1.0 / 448.0)
    bits = raw.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    inverse_scale = 1.0 / scale
    quantized = (values * inverse_scale).to(tl.float8e4nv)
    tl.store(Q + row * K + column, quantized)
    tl.store(S + group, exponent.to(tl.uint8))


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="triton_quantize_fp8_group32_ue8m0",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={
        "granularity": frozenset({"token_group_32"}),
        "scale_encoding": frozenset({"ue8m0"}),
    },
    priority=Priority.PORTABLE,
)
def triton_quantize_fp8_group32_ue8m0(
    x: torch.Tensor,
    granularity: str,
    group_size: int,
    scale_encoding: str,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize group-32 E4M3 data with UE8M0 scales, matching V4.1 act_quant.

    Args:
        x: [M,K] floating input with contiguous K divisible by 32.
        granularity: Must be ``token_group``.
        group_size: Must be 32.
        scale_encoding: Must be ``ue8m0``.
        enable_pdl: Accepted for the shared interface; this kernel has no PDL.

    Returns:
        FP8 [M,K] data and uint8 UE8M0 [M,K/32] scales.
    """
    if (
        x.ndim != 2
        or x.shape[1] % 32
        or x.stride(1) != 1
        or granularity != "token_group"
        or group_size != 32
        or scale_encoding != "ue8m0"
    ):
        raise ValueError(
            "FP8 UE8M0 quantization requires [M,K] with contiguous K divisible by 32"
        )
    quantized = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(
        (x.shape[0], x.shape[1] // 32), dtype=torch.uint8, device=x.device
    )
    if x.numel():
        _fp8_group32_ue8m0_quantize[(x.numel() // 32,)](
            x,
            quantized,
            scales,
            K=x.shape[1],
            X0=x.stride(0),
            num_warps=1,
            num_stages=1,
        )
    return quantized, scales


@triton.jit
def _gated_fp8_quantize_kernel(
    x_ptr,
    gate_ptr,
    norm_ptr,
    out_ptr,
    scale_ptr,
    x_stride,
    gate_stride,
    columns,
    eps,
    HEAD_DIM: tl.constexpr,
    HAS_NORM: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = columns // 128
    head = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    col = tl.arange(0, BLOCK_D)
    valid = (head[:, None] < columns // HEAD_DIM) & (col[None, :] < HEAD_DIM)
    offsets = head[:, None] * HEAD_DIM + col[None, :]
    gate = tl.load(gate_ptr + row * gate_stride + offsets, valid, 0).to(tl.float32)
    if HAS_NORM:
        weight = tl.load(norm_ptr + col, col < HEAD_DIM, 0).to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    value = tl.load(x_ptr + row * x_stride + offsets, valid, 0).to(tl.float32)
    if HAS_NORM:
        variance = tl.sum(value * value, axis=1) / HEAD_DIM
        value = value * tl.rsqrt(variance + eps)[:, None] * weight[None, :]
    value = (value * tl.sigmoid(gate)).to(x_ptr.dtype.element_ty).to(tl.float32)
    if HEAD_DIM != BLOCK_D:
        # Remove per-head padding before forming the 128-element FP8 groups.
        linear = tl.arange(0, BLOCK_H * BLOCK_D)
        source = linear // HEAD_DIM * BLOCK_D + linear % HEAD_DIM
        value = tl.gather(
            value.reshape(BLOCK_H * BLOCK_D),
            tl.minimum(source, BLOCK_H * BLOCK_D - 1),
            axis=0,
        )
    value = value.reshape(BLOCK_H * BLOCK_D // 128, 128)
    # Match the native quantizer's input rounding and reciprocal scale convention.
    floor = tl.full((), 1e-10, tl.float32).to(x_ptr.dtype.element_ty).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), axis=1), floor)
    quant_scale = tl.div_rn(448.0, amax)
    scale = tl.div_rn(1.0, quant_scale)
    quantized = tl.clamp(value * quant_scale[:, None], -448.0, 448.0)
    local_group = tl.arange(0, BLOCK_H * BLOCK_D // 128)
    group = tl.program_id(1) * (BLOCK_H * HEAD_DIM // 128) + local_group
    valid_group = (local_group < BLOCK_H * HEAD_DIM // 128) & (group < groups)
    offsets = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * columns + offsets, quantized, valid_group[:, None])
    tl.store(scale_ptr + row * groups + group, scale, valid_group)


@triton.jit(do_not_specialize=["X0"])
def _rmsnorm_fp8_group32_ue8m0(
    X,
    W,
    N,
    Q,
    S,
    X0,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * X0 + col, col < K, 0).to(tl.float32)
    w = tl.load(W + col, col < K, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
    normalized = (x * inv * w).to(N.dtype.element_ty)
    tl.store(N + row * K + col, normalized, col < K)
    values = tl.reshape(normalized.to(tl.float32), (BLOCK // 32, 32))
    raw = tl.maximum(tl.max(tl.abs(values), 1), 1e-4) * (1.0 / 448.0)
    bits = raw.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    quantized = tl.reshape(
        values * (1.0 / (exponent << 23).to(tl.float32, bitcast=True))[:, None],
        (BLOCK,),
    ).to(tl.float8e4nv)
    tl.store(Q + row * K + col, quantized, col < K)
    groups = tl.arange(0, BLOCK // 32)
    tl.store(S + row * (K // 32) + groups, exponent.to(tl.uint8), groups < K // 32)


@register_kernel(
    "quantization",
    "gated_fp8",
    name="triton_gated_fp8_quantize",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(("x", "gate"), "dense", {torch.bfloat16}),
    traits={},
    priority=Priority.SPECIALIZED,
)
def triton_gated_fp8_quantize(
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    eps: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm, sigmoid gating, and per-128 FP8 quantization."""
    if x.ndim != 2 or x.shape[-1] % 128 or x.stride(-1) != 1:
        raise ValueError(
            "Gated FP8 quantization requires rows of contiguous 128-element groups"
        )
    if (
        gate.shape != x.shape
        or gate.dtype != x.dtype
        or gate.device != x.device
        or gate.stride(-1) != 1
    ):
        raise ValueError("Gate must match input rows, dtype and device")
    if norm_weight is not None and (
        norm_weight.ndim != 1
        or norm_weight.numel() == 0
        or x.shape[1] % norm_weight.numel() != 0
        or norm_weight.device != x.device
        or norm_weight.stride(0) != 1
        or eps <= 0
    ):
        raise ValueError(
            "RMSNorm requires a contiguous weight whose length divides the row "
            "and positive epsilon"
        )
    rows, columns = x.shape
    head_dim = norm_weight.numel() if norm_weight is not None else 128
    groups = columns // 128
    # Tile boundaries must align with both whole heads and whole FP8 groups.
    block_h = triton.next_power_of_2(
        max(128 // math.gcd(head_dim, 128), triton.cdiv(1024, head_dim))
    )
    values = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, groups), device=x.device, dtype=torch.float32)
    if rows:
        _gated_fp8_quantize_kernel[(rows, triton.cdiv(columns, block_h * head_dim))](
            x,
            gate,
            norm_weight,
            values,
            scales,
            x.stride(0),
            gate.stride(0),
            columns,
            eps,
            HEAD_DIM=head_dim,
            HAS_NORM=norm_weight is not None,
            ENABLE_PDL=enable_pdl,
            BLOCK_H=block_h,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return values, scales


@register_kernel(
    "quantization",
    "rmsnorm_fp8_group32_ue8m0",
    name="triton_rmsnorm_fp8_group32_ue8m0",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={},
    priority=Priority.PORTABLE,
)
def triton_rmsnorm_fp8_group32_ue8m0(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 normalized rows, E4M3 codes and canonical group-32 scales."""
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.shape[1] <= 0
        or x.shape[1] % 32
        or x.stride(1) != 1
        or weight.shape != (x.shape[1],)
        or not weight.is_contiguous()
        or weight.dtype != x.dtype
        or weight.device != x.device
        or eps <= 0
    ):
        raise ValueError("Expected BF16 [M,K] input and [K] weight, K divisible by 32")
    m, k = x.shape
    normalized = torch.empty_like(x, memory_format=torch.contiguous_format)
    quantized = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
    if m:
        _rmsnorm_fp8_group32_ue8m0[(m,)](
            x,
            weight,
            normalized,
            quantized,
            scales,
            x.stride(0),
            k,
            eps,
            triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normalized, quantized, scales


@triton.jit(do_not_specialize=["X0"])
def _silu_and_mul_fp8_group32_ue8m0(
    X,
    Q,
    S,
    X0,
    I: tl.constexpr,
    LIMIT: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # silu(gate) * up rounded to BF16 exactly as _silu_and_mul_kernel, then
    # quantized like _fp8_group32_ue8m0_quantize.
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = column < I
    gate = tl.load(X + row * X0 + column, mask, 0.0).to(tl.float32)
    up = tl.load(X + row * X0 + I + column, mask, 0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, LIMIT)
        up = tl.clamp(up, -LIMIT, LIMIT)
    values = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(values, (BLOCK // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse_scale = 1.0 / (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.reshape(groups * inverse_scale[:, None], (BLOCK,))
    tl.store(Q + row * I + column, quantized.to(tl.float8e4nv), mask)
    group = tl.program_id(1) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(S + row * (I // 32) + group, exponent.to(tl.uint8), group < I // 32)


@register_kernel(
    "quantization",
    "silu_and_mul_fp8_group32_ue8m0",
    name="triton_silu_and_mul_fp8_group32_ue8m0",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={},
    priority=Priority.PORTABLE,
)
def triton_silu_and_mul_fp8_group32_ue8m0(
    x: torch.Tensor, limit: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 codes and UE8M0 group-32 scales of BF16 silu(gate) * up."""
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or x.shape[1] % 64
        or x.stride(1) != 1
        or (limit is not None and limit <= 0)
    ):
        raise ValueError(
            "Expected BF16 [M, 2I] gate/up rows with contiguous columns, I % 32 == 0"
        )
    m, inner = x.shape[0], x.shape[1] // 2
    quantized = torch.empty((m, inner), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, inner // 32), dtype=torch.uint8, device=x.device)
    block = min(1024, triton.next_power_of_2(inner))
    if m:
        _silu_and_mul_fp8_group32_ue8m0[(m, triton.cdiv(inner, block))](
            x,
            quantized,
            scales,
            x.stride(0),
            inner,
            0.0 if limit is None else limit,
            limit is not None,
            block,
            num_warps=4,
        )
    return quantized, scales


@triton.jit
def _fp8_swizzled_scale_offset(row, column, K_TILES: tl.constexpr):
    # Equivalent to [Mtiles,4,32,Ktiles,4].transpose(1,3).contiguous().
    return (
        (row // 128 * K_TILES + column // 4) * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + column % 4
    )


@triton.jit(
    do_not_specialize=["M", "X0"],
    do_not_specialize_on_alignment=["M", "X0"],
)
def _fp8_group32_ue8m0_quantize_swizzled(
    X,
    Q,
    S,
    M,
    X0,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    k_tiles = tl.cdiv(K, 128)
    if pid * GROUPS < M * (K // 32):
        group = pid * GROUPS + tl.arange(0, GROUPS)
        row = group // (K // 32)
        scale_column = group % (K // 32)
        column = scale_column[:, None] * 32 + tl.arange(0, 32)[None, :]
        values = tl.load(X + row[:, None] * X0 + column, row[:, None] < M, 0.0).to(
            tl.float32
        )
        # Preserve the original quantizer's exact arithmetic and rounding boundary.
        amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
        raw = amax * (1.0 / 448.0)
        bits = raw.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        quantized = tl.div_rn(values, scale[:, None]).to(tl.float8e4nv)
        tl.store(Q + row[:, None] * K + column, quantized, row[:, None] < M)
        scale_offset = _fp8_swizzled_scale_offset(row, scale_column, k_tiles)
        tl.store(S + scale_offset, exponent.to(tl.uint8), row < M)

    # Padding writes have disjoint addresses from all live scale stores. Each
    # program owns a fixed slice, so no memset/copy or cross-CTA barrier is needed.
    if M % 128 != 0 or K % 128 != 0:
        padded_columns = k_tiles * 4
        index = pid * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
        pad_row, pad_column = index // padded_columns, index % padded_columns
        padding = (index < tl.cdiv(M, 128) * 128 * padded_columns) & (
            (pad_row >= M) | (pad_column >= K // 32)
        )
        offset = _fp8_swizzled_scale_offset(pad_row, pad_column, k_tiles)
        tl.store(S + offset, 0, padding)


def _quantize_fp8_group32_swizzled(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact group-32 quantization in the prepared GEMM's F8_128x4 layout."""
    m, k = x.shape
    quantized = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale_count = triton.cdiv(m, 128) * 128 * triton.cdiv(k, 128) * 4
    scales = torch.empty((scale_count,), dtype=torch.uint8, device=x.device)
    if m:
        # Small inputs need enough independent CTAs; large inputs amortize
        # address/reduction overhead across more groups per CTA. These static
        # work sizes never depend on input values or request history.
        total_groups = m * (k // 32)
        if total_groups <= 1024:
            groups = 4
        elif total_groups <= 8192:
            groups = 16
        elif total_groups <= 131072:
            groups = 32
        else:
            groups = 64
        zero_block = 256
        programs = max(
            triton.cdiv(m * (k // 32), groups), triton.cdiv(scale_count, zero_block)
        )
        _fp8_group32_ue8m0_quantize_swizzled[(programs,)](
            x,
            quantized,
            scales,
            m,
            x.stride(0),
            k,
            groups,
            zero_block,
            num_warps=4,
            num_stages=1,
        )
    return quantized, scales
