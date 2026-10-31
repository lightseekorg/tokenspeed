from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.ops.layernorm._triton.gated_rmsnorm import (
    gated_rmsnorm_kernel,
)
from tokenspeed_kernel.ops.quantization.triton import (
    fp8_dequantize_unscaled,
    fp8_quantize_by_multiplier,
    fp8_token_multiplier,
    fp8_token_scale,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.warmup import kernel_launcher

_FP32_MIN_NORMAL = tl.constexpr(1.1754943508222875e-38)


@triton.jit
def _fp8_swizzled_scale_offset(row, column, K_TILES: tl.constexpr):
    # Equivalent to [Mtiles,4,32,Ktiles,4].transpose(1,3).contiguous().
    return (
        (row // 128 * K_TILES + column // 4) * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + column % 4
    )


@triton.jit
def _mul_rn_f32(a, b):
    # One IEEE round-to-nearest FP32 multiply that the compiler can neither
    # contract into an FMA nor flush to zero.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    x_scale,
    residual_scale,
    x_row_stride,
    out_row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    ROUND_RESIDUAL_SUM_BF16: tl.constexpr,
    SCALE_INPUTS_BF16: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    fp8_out_ptr,
    fp8_scale_ptr,
    TOKEN_FP8_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(
        x_ptr + row.to(tl.int64) * x_row_stride + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if SCALE_INPUTS_BF16:
            # Each product is rounded to BF16, as an eager BF16 tensor times a
            # Python float is.
            x = _mul_rn_f32(x, x_scale).to(tl.bfloat16).to(tl.float32)
            residual = (
                _mul_rn_f32(residual, residual_scale).to(tl.bfloat16).to(tl.float32)
            )
        x += residual
        if ROUND_RESIDUAL_SUM_BF16:
            # Round the sum once to BF16, as an eager BF16 add does, so that
            # the norm reads the stored residual.
            x = x.to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out_ptr + row_offsets, x, mask=mask)

    variance = tl.sum(x * x, axis=0) / n_cols
    x *= tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + row.to(tl.int64) * out_row_stride + offsets, x * weight, mask=mask
    )
    if TOKEN_FP8_OUT:
        # The stored row quantized per token (as fp8_quantize_per_token
        # quantizes it): FP8 E4M3 codes and one FP32 dequant scale.
        normed = (x * weight).to(out_ptr.dtype.element_ty).to(tl.float32)
        amax = tl.max(tl.abs(normed), axis=0)
        codes = fp8_quantize_by_multiplier(normed, fp8_token_multiplier(amax))
        # Store the codes in vectors of at most 16 bytes of the output (8
        # BF16 elements). A vector of 16 one-byte codes would have the
        # compiler give every thread 16 elements of the row, loads included,
        # and so sum the squares above in another order than the call
        # without token_fp8_out, which changes some output bits.
        code_offsets = tl.max_contiguous(
            tl.arange(0, BLOCK), 128 // out_ptr.dtype.element_ty.primitive_bitwidth
        )
        tl.store(fp8_out_ptr + row * n_cols + code_offsets, codes, mask=mask)
        tl.store(fp8_scale_ptr + row, fp8_token_scale(amax))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def reference_rmsnorm_row(x, weight, eps: tl.constexpr, N: tl.constexpr):
    """RMSNorm one FP32 row in the eager reference's arithmetic order.

    ``x`` and ``weight`` are FP32 vectors of ``N`` elements. The mean scales
    the sum by the FP32 reciprocal of ``N`` exactly as ``torch.mean`` does,
    the row is scaled before the weight multiplies it, and nothing is
    rounded in between; the caller casts the result once. Only the
    summation order can differ from the sequential reference.
    """
    scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / N) + eps)
    return weight * (x * scale)


@triton.jit
def _reference_rmsnorm_kernel(
    X,
    W,
    OUT,
    X0,
    O0,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    mask = d < N
    x = tl.load(X + row * X0 + d, mask, 0.0).to(tl.float32)
    weight = tl.load(W + d, mask, 0.0).to(tl.float32)
    tl.store(OUT + row * O0 + d, reference_rmsnorm_row(x, weight, EPS, N), mask)


def reference_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None
) -> torch.Tensor:
    """RMSNorm ``x`` with the cast order of the eager FP32 reference.

    Args:
        x: Floating ``[..., N]`` rows with a unit last stride.
        weight: ``[N]`` contiguous weight in any floating dtype.
        eps: Variance epsilon added to the FP32 mean of squares.
        out: Contiguous destination shaped like ``x`` in ``x``'s dtype, or
            None to allocate one.

    Returns:
        ``out``: ``(weight.float() * (x.float() * rsqrt(mean(x.float()**2) +
        eps))).to(x.dtype)`` row by row, rounded once at the store.
    """
    n = x.shape[-1]
    if x.ndim < 1 or x.stride(-1) != 1:
        raise ValueError("reference RMSNorm rows need a unit last stride")
    if weight.shape != (n,) or not weight.is_contiguous():
        raise ValueError(f"reference RMSNorm weight must be contiguous [{n}]")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("reference RMSNorm out must be contiguous and match x")
    rows = x.reshape(-1, n) if x.ndim != 2 else x
    if rows.numel():
        _reference_rmsnorm_kernel[(rows.shape[0],)](
            rows,
            weight,
            out.view(-1, n),
            rows.stride(0),
            n,
            N=n,
            EPS=eps,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8 if n >= 4096 else 4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _rmsnorm_fused_parallel_kernel(
    input1_ptr,
    weight1_ptr,
    output1_ptr,
    input2_ptr,
    weight2_ptr,
    output2_ptr,
    n_cols1: tl.constexpr,
    n_cols2: tl.constexpr,
    stride_input1: tl.constexpr,
    stride_output1: tl.constexpr,
    stride_input2: tl.constexpr,
    stride_output2: tl.constexpr,
    eps: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
):
    row = tl.program_id(0)

    offsets1 = tl.arange(0, BLOCK1)
    mask1 = offsets1 < n_cols1
    input1_offsets = row * stride_input1 + offsets1
    output1_offsets = row * stride_output1 + offsets1
    input1 = tl.load(input1_ptr + input1_offsets, mask=mask1, other=0.0).to(tl.float32)
    variance1 = tl.sum(input1 * input1, axis=0) / n_cols1
    weight1 = tl.load(weight1_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
    output1 = input1 * tl.rsqrt(variance1 + eps) * weight1
    tl.store(output1_ptr + output1_offsets, output1, mask=mask1)

    offsets2 = tl.arange(0, BLOCK2)
    mask2 = offsets2 < n_cols2
    input2_offsets = row * stride_input2 + offsets2
    output2_offsets = row * stride_output2 + offsets2
    input2 = tl.load(input2_ptr + input2_offsets, mask=mask2, other=0.0).to(tl.float32)
    variance2 = tl.sum(input2 * input2, axis=0) / n_cols2
    weight2 = tl.load(weight2_ptr + offsets2, mask=mask2, other=0.0).to(tl.float32)
    output2 = input2 * tl.rsqrt(variance2 + eps) * weight2
    tl.store(output2_ptr + output2_offsets, output2, mask=mask2)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool | None = False,
    round_residual_sum_bf16: bool = False,
    x_scale: float | None = None,
    residual_scale: float | None = None,
    token_fp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm over the last dimension with FP32 intermediates.

    Args:
        x: Input shaped ``[..., hidden_size]``.
        weight: ``[hidden_size]`` affine weight.
        eps: Epsilon added to the mean of squares.
        residual: Optional ``x``-shaped tensor added to ``x`` before the norm.
        out: Optional contiguous output.
        enable_pdl: Launch with programmatic dependent launch (NVIDIA Hopper
            or newer): the kernel waits for the previous kernel before it
            reads its inputs and releases dependent kernels after its last
            store. None follows ``pdl_enabled()``.
        round_residual_sum_bf16: Round the FP32 sum ``x + residual`` to BF16
            before it is stored and normalized, so that the call equals an
            eager BF16 ``x + residual`` followed by ``rmsnorm`` of the sum.
            Needs BF16 ``x`` and ``residual``. By default the norm reads the
            unrounded FP32 sum.
        x_scale, residual_scale: Optional finite float multipliers of ``x``
            and ``residual``; a missing one is 1.0. Each product is rounded to
            BF16 before the add, as an eager BF16 tensor times a Python float
            is. They are runtime arguments, so new values do not recompile.
            Need ``round_residual_sum_bf16=True`` and an NVIDIA GPU.
        token_fp8_out: Optional ``(codes, scales)`` buffers, contiguous FP8
            E4M3 ``[rows, hidden_size]`` and FP32 ``[rows, 1]``: the same
            launch also writes the output quantized per token, as
            ``ops/quantization/triton.py``'s ``fp8_quantize_per_token`` would
            quantize the stored output, for an FP8 GEMM that reads it.
            Callers reach it through ``ops.quantization.rmsnorm_quantize_fp8``.

    Returns:
        ``out``, or ``(out, residual_sum)`` when ``residual`` is given.
    """
    if round_residual_sum_bf16 and (
        residual is None
        or x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
    ):
        raise ValueError("round_residual_sum_bf16 needs BF16 x and residual")
    scale_inputs = x_scale is not None or residual_scale is not None
    x_scale = 1.0 if x_scale is None else x_scale
    residual_scale = 1.0 if residual_scale is None else residual_scale
    if scale_inputs:
        if not round_residual_sum_bf16:
            raise ValueError(
                "x_scale and residual_scale need round_residual_sum_bf16=True"
            )
        if not current_platform().is_nvidia:
            raise ValueError("x_scale and residual_scale need an NVIDIA GPU")
        for name, value in (("x_scale", x_scale), ("residual_scale", residual_scale)):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float, got {value!r}")
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    # Without a residual the kernel reads and writes strided rows (e.g. a
    # column slice of a fused projection) in place; other layouts are made
    # contiguous.
    strided_rows = residual is None and x.ndim == 2 and x.stride(-1) == 1
    if not x.is_contiguous() and not strided_rows:
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x if strided_rows else x.view(-1, hidden_size)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device) if out is None else out
    strided_out = residual is None and out.ndim == 2 and out.stride(-1) == 1
    if not out.is_contiguous() and not strided_out:
        raise ValueError("out must be contiguous or 2-D rows with unit column stride")
    out_2d = out if strided_out else out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    if token_fp8_out is not None:
        codes, scales = token_fp8_out
        if (
            codes.dtype != torch.float8_e4m3fn
            or tuple(codes.shape) != tuple(x_2d.shape)
            or not codes.is_contiguous()
            or scales.dtype != torch.float32
            or scales.numel() != x_2d.shape[0]
            or not scales.is_contiguous()
        ):
            raise ValueError(
                "token_fp8_out takes contiguous FP8 E4M3 codes shaped like the "
                f"{tuple(x_2d.shape)} rows and FP32 [rows, 1] scales, got "
                f"{codes.dtype} {tuple(codes.shape)} and {scales.dtype} "
                f"{tuple(scales.shape)}"
            )
    else:
        codes, scales = None, None
    block = triton.next_power_of_2(hidden_size)
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual,
        weight,
        out_2d,
        residual_out,
        x_scale,
        residual_scale,
        x_2d.stride(0),
        out_2d.stride(0),
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
        ROUND_RESIDUAL_SUM_BF16=round_residual_sum_bf16,
        SCALE_INPUTS_BF16=scale_inputs,
        ENABLE_PDL=enable_pdl,
        fp8_out_ptr=codes,
        fp8_scale_ptr=scales,
        TOKEN_FP8_OUT=token_fp8_out is not None,
        **launch_kwargs,
    )
    if residual is None:
        return out
    return out, residual_out


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    row_stride,
    out_row_stride,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < GROUP_SIZE
    group_offset = group * GROUP_SIZE
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(
        x_ptr + row * row_stride + group_offset + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / GROUP_SIZE
    weight = tl.load(weight_ptr + group_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    normalized = x * tl.rsqrt(variance + eps) * (1.0 + weight)
    tl.store(
        out_ptr + row * out_row_stride + group_offset + offsets,
        normalized,
        mask=mask,
    )


def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Grouped Gemma RMSNorm without the unused inverse-RMS output.

    ``weight`` is the checkpoint offset, so the kernel multiplies by
    ``1 + weight``. Variance is independent for each last-dimension group.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    width = int(x.shape[-1])
    if group_size <= 0 or width % group_size:
        raise ValueError(
            f"group_size must divide the last dimension ({width}), got {group_size}"
        )
    if weight.shape != (width,):
        raise ValueError(
            f"weight must have shape {(width,)}, got {tuple(weight.shape)}"
        )
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("weight must match x dtype and device")
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must match x shape, dtype, and device")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if x.numel() == 0:
        return out
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    rows = x.numel() // width
    groups = width // group_size
    x_2d = x.view(rows, width)
    out_2d = out.view(rows, width)
    block = triton.next_power_of_2(group_size)
    if block > 65536:
        raise ValueError("group_size is too large for the Triton reduction")
    enable_pdl = pdl_enabled()
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _grouped_gemma_rmsnorm_kernel[(rows, groups)](
        x_2d,
        weight,
        out_2d,
        x_2d.stride(0),
        out_2d.stride(0),
        eps=eps,
        GROUP_SIZE=group_size,
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        **launch_kwargs,
    )
    return out


@triton.jit
def _gated_residual_combine_norm_kernel(
    residual_ptr,
    block_ptr,
    inject_ptr,
    weight_ptr,
    residual_out_ptr,
    norm_out_ptr,
    weight_group_stride,
    eps: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    PRELOAD_RESIDUAL: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    GROUPS_FIRST: tl.constexpr,
):
    row = tl.program_id(1 if GROUPS_FIRST else 0)
    group = tl.program_id(0 if GROUPS_FIRST else 1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < GROUP_SIZE
    positions = row * WIDTH + group * GROUP_SIZE + offsets

    if ENABLE_PDL and not PRELOAD_RESIDUAL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    residual = tl.load(residual_ptr + positions, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(
        weight_ptr + group * weight_group_stride + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    if ENABLE_PDL and PRELOAD_RESIDUAL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    block_output = tl.load(
        block_ptr + row * GROUP_SIZE + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    logit = tl.load(inject_ptr + row * (WIDTH // GROUP_SIZE) + group).to(tl.float32)
    combined = residual + block_output * 2.0 * tl.sigmoid(logit)
    # Match a standalone combine store before the following norm reads it.
    combined = combined.to(residual_ptr.dtype.element_ty).to(tl.float32)
    tl.store(residual_out_ptr + positions, combined, mask=mask)
    combined = tl.where(mask, combined, 0.0)

    variance = tl.sum(combined * combined, axis=0) / GROUP_SIZE
    normalized = combined * tl.rsqrt(variance + eps) * (1.0 + weight)
    tl.store(norm_out_ptr + positions, normalized, mask=mask)


def gated_residual_combine_norm(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_logits: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
    preload_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine gated residual branches and apply grouped Gemma RMSNorm."""
    if residual.ndim < 1:
        raise ValueError("residual must have at least one dimension")
    width = int(residual.shape[-1])
    if group_size <= 0 or width % group_size:
        raise ValueError(
            f"group_size must divide the last dimension ({width}), got {group_size}"
        )
    groups = width // group_size
    for name, value, shape in (
        ("block_output", block_output, (*residual.shape[:-1], group_size)),
        ("inject_logits", inject_logits, (*residual.shape[:-1], groups)),
    ):
        if (
            value.shape != shape
            or value.dtype != residual.dtype
            or value.device != residual.device
        ):
            raise ValueError(
                f"{name} must have shape {tuple(shape)} and match residual "
                "dtype and device"
            )
    if weight.shape not in ((width,), (group_size,)):
        raise ValueError(
            f"weight must have shape {(width,)} or {(group_size,)}, "
            f"got {tuple(weight.shape)}"
        )
    if weight.dtype != residual.dtype or weight.device != residual.device:
        raise ValueError("weight must match residual dtype and device")

    combined = torch.empty_like(residual, memory_format=torch.contiguous_format)
    normalized = torch.empty_like(residual, memory_format=torch.contiguous_format)
    if residual.numel() == 0:
        return combined, normalized
    if not residual.is_contiguous():
        residual = residual.contiguous()
        preload_residual = False
    if not weight.is_contiguous():
        weight = weight.contiguous()
        preload_residual = False
    block_output = block_output.contiguous()
    inject_logits = inject_logits.contiguous()

    block = triton.next_power_of_2(group_size)
    if block > 65536:
        raise ValueError("group_size is too large for the Triton reduction")
    rows = residual.numel() // width
    groups_first = (
        4096 <= rows <= 65535
        and groups == 4
        and group_size == 2560
        and residual.dtype in (torch.bfloat16, torch.float16)
        and current_platform().is_nvidia
        and current_platform().arch_version == ArchVersion(10, 7)
    )
    # Reuse each row's block output without exceeding CUDA's grid-y limit.
    grid = (groups, rows) if groups_first else (rows, groups)
    # Both grid orders are fixed JIT configurations; warm each before serving.
    enable_pdl = pdl_enabled()
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _gated_residual_combine_norm_kernel[grid](
        residual,
        block_output,
        inject_logits,
        weight,
        combined,
        normalized,
        group_size if weight.numel() == width else 0,
        eps=eps,
        WIDTH=width,
        GROUP_SIZE=group_size,
        BLOCK=block,
        PRELOAD_RESIDUAL=preload_residual,
        ENABLE_PDL=enable_pdl,
        GROUPS_FIRST=groups_first,
        **launch_kwargs,
    )
    return combined, normalized


@triton.jit
def _grouped_rmsnorm_kernel(
    x_ptr,
    out_ptr,
    row_stride,
    out_row_stride,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < GROUP_SIZE
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(
        x_ptr + row * row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / GROUP_SIZE
    normalized = x * tl.rsqrt(variance + eps)
    tl.store(
        out_ptr + row * out_row_stride + offsets,
        normalized,
        mask=mask,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def grouped_rmsnorm(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Apply weight-free RMSNorm independently to last-dimension groups.

    Args:
        x: GPU input shaped ``[..., width]``.
        group_size: Number of contiguous last-dimension values sharing one RMS
            statistic. The last dimension must be divisible by this value.
        eps: Epsilon added before reciprocal square root.
        out: Optional contiguous output matching ``x``. ``out=x`` is supported
            for graph-safe in-place normalization.

    Returns:
        Normalized tensor matching ``x`` shape and dtype.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    width = int(x.shape[-1])
    if group_size <= 0 or width % group_size:
        raise ValueError(
            f"group_size must divide the last dimension ({width}), got {group_size}"
        )
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must match x shape, dtype, and device")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if x.numel() == 0:
        return out
    if not x.is_contiguous():
        x = x.contiguous()

    rows = x.numel() // group_size
    x_2d = x.view(rows, group_size)
    out_2d = out.view(rows, group_size)
    block = triton.next_power_of_2(group_size)
    if block > 65536:
        raise ValueError("group_size is too large for the Triton reduction")
    enable_pdl = pdl_enabled()
    launch_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _grouped_rmsnorm_kernel[(rows,)](
        x_2d,
        out_2d,
        x_2d.stride(0),
        out_2d.stride(0),
        eps=eps,
        GROUP_SIZE=group_size,
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        **launch_kwargs,
    )
    return out


@triton.jit
def _ssmax_scale_f32(position):
    """apply_ssmax_root10's FP32 scale for one position (offset 1, denominator 2048,
    exponent 0.1): exp(0.1 * log1p((p + 1) * 2^-11)) with the CUDA math library
    (libdevice); the multiply by 2^-11 is the exact form of the division by 2048
    (measured bit-equal to Torch on [0, 2^20] for int32 and int64)."""
    ratio = (position.to(tl.float32) + 1.0) * 0.00048828125
    return libdevice.exp(0.1 * libdevice.log1p(ratio))


@triton.jit
def _fused_qk_rmsnorm_kernel(
    q_in_ptr,
    k_in_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    q_in_token_stride,
    k_in_token_stride,
    q_out_token_stride,
    k_out_token_stride,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    ROUND_AFTER_AFFINE: tl.constexpr = False,
    SSMAX_INKERNEL: tl.constexpr = False,
    token_scale_ptr=None,
    q_scale_ptr=None,
    k_scale_ptr=None,
    v_in_ptr=None,
    v_scale_ptr=None,
    v_out_ptr=None,
    g_in_ptr=None,
    g_scale_ptr=None,
    g_out_ptr=None,
    v_in_token_stride=0,
    g_in_token_stride=0,
    v_out_token_stride=0,
    g_out_token_stride=0,
    DEQUANT: tl.constexpr = False,
):
    # 2D grid: (token, head). Heads in [0, num_q_heads) handle q rows;
    # heads in [num_q_heads, num_q_heads + num_kv_heads) handle k rows.
    # Inputs may be non-contiguous along the leading axis (e.g. views from a
    # qkv split) — we use the explicit token strides to compute addresses.
    # DEQUANT: q and k are an unscaled FP8 GEMM's output; each program first
    # applies the token and channel scales to its head (the scaled GEMM's
    # 16-bit output), and also writes the dequantized output gate head of the
    # same index (q programs) or V head (k programs).
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    offsets = tl.arange(0, BLOCK)
    mask = offsets < head_dim

    if is_k:
        in_addrs = (
            k_in_ptr + token * k_in_token_stride + local_head * head_dim + offsets
        )
        out_addrs = (
            k_out_ptr + token * k_out_token_stride + local_head * head_dim + offsets
        )
        w_addrs = k_weight_ptr + offsets
    else:
        in_addrs = (
            q_in_ptr + token * q_in_token_stride + local_head * head_dim + offsets
        )
        out_addrs = (
            q_out_ptr + token * q_out_token_stride + local_head * head_dim + offsets
        )
        w_addrs = q_weight_ptr + offsets

    # Weights are parameters nothing in the decode graph writes; safe to load before the PDL wait.
    w = tl.load(w_addrs, mask=mask, other=0.0).to(tl.float32)
    # Adding +0.0 would turn a -0.0 weight into +0.0.
    if WEIGHT_OFFSET != 0.0:
        w = w + WEIGHT_OFFSET

    if ENABLE_PDL:
        # Wait for the producer's stores before the first dependent load.
        tl.extra.cuda.gdc_wait()

    raw = tl.load(in_addrs, mask=mask, other=0.0)
    if DEQUANT:
        token_scale = tl.load(token_scale_ptr + token)
        head_cols = local_head * head_dim + offsets
        if is_k:
            scale = tl.load(k_scale_ptr + head_cols, mask=mask, other=0.0)
            pass_in = v_in_ptr + token * v_in_token_stride + head_cols
            pass_scale = tl.load(v_scale_ptr + head_cols, mask=mask, other=0.0)
            pass_out = v_out_ptr + token * v_out_token_stride + head_cols
        else:
            scale = tl.load(q_scale_ptr + head_cols, mask=mask, other=0.0)
            pass_in = g_in_ptr + token * g_in_token_stride + head_cols
            pass_scale = tl.load(g_scale_ptr + head_cols, mask=mask, other=0.0)
            pass_out = g_out_ptr + token * g_out_token_stride + head_cols
        passed = fp8_dequantize_unscaled(
            tl.load(pass_in, mask=mask, other=0.0), token_scale, pass_scale
        )
        tl.store(pass_out, passed, mask=mask)
        x = fp8_dequantize_unscaled(raw, token_scale, scale)
    else:
        x = raw.to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    x = x * tl.rsqrt(var + eps)
    output = x * w
    if ROUND_AFTER_AFFINE:
        # Beam's order: BeamRMSNorm stores bf16(x * inv_rms * w); the SSMax scale
        # multiplies that stored value.
        output = output.to(tl.bfloat16).to(tl.float32)
    if SSMAX_INKERNEL:
        # apply_ssmax_root10(q, positions, 1, 2048.0): FP32 scale from the token's
        # position with the CUDA math library (libdevice log1p / exp), then
        # bf16(fp32(q) * scale) once; IEEE .rn multiply, no FTZ. K rows are unchanged.
        position = tl.load(positions_ptr + token)
        scale = _ssmax_scale_f32(position)
        output = tl.where(is_k, output, _mul_rn_f32(output, scale))
    tl.store(out_addrs, output, mask=mask)
    if ENABLE_PDL:
        # All stores issued; let the dependent kernel begin its prologue.
        tl.extra.cuda.gdc_launch_dependents()


@dataclass(frozen=True)
class UnscaledFP8QKVGate:
    """The scales and the other columns of an unscaled FP8 Q/K/V/gate GEMM.

    For ``qk_rmsnorm`` / ``fused_qk_rmsnorm_rope`` with ``dequant=``: their
    ``q`` and ``k`` are the Q and K columns of an FP8 GEMM run with unit
    scales, and the launch applies the scales it left out (the scaled GEMM's
    16-bit output, ``fp8_dequantize_unscaled``) before the norm, and writes
    the dequantized V and output gate columns to ``v_out`` and ``gate_out``.

    Attributes:
        token_scales: FP32 ``[T]`` / ``[T, 1]`` per-token scales of the GEMM's
            FP8 input.
        q_scales, k_scales, v_scales, gate_scales: FP32 per-channel weight
            scales of the Q, K, V and gate columns (``[N]`` / ``[N, 1]``,
            contiguous).
        v, gate: The raw V ``[T, kv_heads * head_dim]`` and gate
            ``[T, q_heads * head_dim]`` columns (rows may be strided).
        v_out, gate_out: Destinations shaped like ``v`` and ``gate`` with a
            contiguous last axis.
    """

    token_scales: torch.Tensor
    q_scales: torch.Tensor
    k_scales: torch.Tensor
    v: torch.Tensor
    v_scales: torch.Tensor
    gate: torch.Tensor
    gate_scales: torch.Tensor
    v_out: torch.Tensor
    gate_out: torch.Tensor


def _dequant_launch_kwargs(
    dequant: UnscaledFP8QKVGate | None,
    q: torch.Tensor,
    k: torch.Tensor,
    n_tokens: int,
) -> dict[str, object]:
    """The DEQUANT kernel arguments of ``dequant``, checked; none without it."""
    if dequant is None:
        return {}
    q_width, kv_width = q.shape[-1], k.shape[-1]
    for name, tensor, count in (
        ("token_scales", dequant.token_scales, n_tokens),
        ("q_scales", dequant.q_scales, q_width),
        ("k_scales", dequant.k_scales, kv_width),
        ("v_scales", dequant.v_scales, kv_width),
        ("gate_scales", dequant.gate_scales, q_width),
    ):
        if (
            tensor.dtype != torch.float32
            or tensor.numel() != count
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"dequant.{name} must be {count} contiguous FP32 values, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    for name, tensor, width in (
        ("v", dequant.v, kv_width),
        ("v_out", dequant.v_out, kv_width),
        ("gate", dequant.gate, q_width),
        ("gate_out", dequant.gate_out, q_width),
    ):
        if (
            tensor.ndim != 2
            or tuple(tensor.shape) != (n_tokens, width)
            or tensor.stride(-1) != 1
            or tensor.dtype != q.dtype
        ):
            raise ValueError(
                f"dequant.{name} must be {q.dtype} [{n_tokens}, {width}] with a "
                f"contiguous last axis, got {tensor.dtype} {tuple(tensor.shape)} "
                f"stride {tensor.stride()}"
            )
    return {
        "token_scale_ptr": dequant.token_scales,
        "q_scale_ptr": dequant.q_scales,
        "k_scale_ptr": dequant.k_scales,
        "v_in_ptr": dequant.v,
        "v_scale_ptr": dequant.v_scales,
        "v_out_ptr": dequant.v_out,
        "g_in_ptr": dequant.gate,
        "g_scale_ptr": dequant.gate_scales,
        "g_out_ptr": dequant.gate_out,
        "v_in_token_stride": dequant.v.stride(0),
        "g_in_token_stride": dequant.gate.stride(0),
        "v_out_token_stride": dequant.v_out.stride(0),
        "g_out_token_stride": dequant.gate_out.stride(0),
        "DEQUANT": True,
    }


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float,
    enable_pdl: bool | None = None,
    ssmax_positions: torch.Tensor | None = None,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm of q and k in a single kernel launch; heads scale by
    ``weight_offset + weight``, formed in fp32.

    Reads from possibly non-contiguous q/k (e.g. views into a qkv-split tensor)
    and writes to fresh contiguous output tensors. The kernel uses the input
    leading-axis stride directly, so no ``.contiguous()`` copy is required
    on the inputs.

    Args:
        ssmax_positions: ``None`` (the default) leaves the kernel as before. A
            contiguous int32 / int64 ``[tokens]`` vector of absolute positions
            folds Beam's root10 SSMax into the launch: the affine output is
            rounded once to BF16 (``ROUND_AFTER_AFFINE``, the value
            ``BeamRMSNorm`` stores), the Q rows are multiplied in FP32 by
            ``exp(0.1 * log1p((p + 1) * 2^-11))`` computed with the CUDA math
            library and one IEEE round-to-nearest multiply (``SSMAX_INKERNEL``),
            and stored once as BF16; the K rows are stored unchanged. The
            constants (offset 1, denominator 2048, exponent 0.1) are Beam's and
            fixed in the kernel; ``p >= 0`` is the caller's contract, as it is
            for ``apply_ssmax_root10`` (not checked: no device read on this
            host).
        dequant: ``None`` (the default) leaves the kernel as before. Given, q
            and k are an unscaled FP8 GEMM's Q and K columns: each head is
            dequantized first (:class:`UnscaledFP8QKVGate`), and the same
            launch writes the dequantized V and output gate columns.
    """
    if q.shape[0] == 0:
        return torch.empty_like(q), torch.empty_like(k)
    head_dim = q_weight.shape[0]
    assert k_weight.shape[0] == head_dim, "q/k_weight must share head_dim"
    assert q.shape[-1] % head_dim == 0 and k.shape[-1] % head_dim == 0
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1
    ), "qk_rmsnorm requires the last dim to be contiguous"

    num_q_heads = q.shape[-1] // head_dim
    num_kv_heads = k.shape[-1] // head_dim
    n_tokens = q.numel() // q.shape[-1]
    block = triton.next_power_of_2(head_dim)

    q_in_stride = q.stride(0) if q.dim() > 1 else q.shape[-1]
    k_in_stride = k.stride(0) if k.dim() > 1 else k.shape[-1]
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    ssmax = ssmax_positions is not None
    if ssmax:
        if ssmax_positions.ndim != 1 or ssmax_positions.shape[0] != n_tokens:
            raise ValueError(
                f"ssmax_positions must be [{n_tokens}] (one per token), got "
                f"{tuple(ssmax_positions.shape)}"
            )
        if not ssmax_positions.is_contiguous():
            raise ValueError("ssmax_positions must be contiguous")
        if ssmax_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"ssmax_positions must be int32 or int64, got {ssmax_positions.dtype}"
            )
        if ssmax_positions.device != q.device:
            raise ValueError(
                f"ssmax_positions is on {ssmax_positions.device}, q is on {q.device}"
            )

    # Allocate fresh contiguous outputs so downstream RoPE/attention kernels
    # — which assume row-major layouts — work without further copies.
    q_out = torch.empty((n_tokens, q.shape[-1]), dtype=q.dtype, device=q.device)
    k_out = torch.empty((n_tokens, k.shape[-1]), dtype=k.dtype, device=k.device)

    kwargs = _dequant_launch_kwargs(dequant, q, k, n_tokens)
    if current_platform().is_nvidia:
        kwargs["launch_pdl"] = enable_pdl
    _fused_qk_rmsnorm_kernel[(n_tokens, num_q_heads + num_kv_heads)](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        ssmax_positions,
        q_in_stride,
        k_in_stride,
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps,
        WEIGHT_OFFSET=weight_offset,
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        ROUND_AFTER_AFFINE=ssmax,
        SSMAX_INKERNEL=ssmax,
        **kwargs,
    )
    return q_out, k_out


def rmsnorm_fused_parallel(
    input1: torch.Tensor,
    weight1: torch.Tensor,
    output1: torch.Tensor,
    input2: torch.Tensor,
    weight2: torch.Tensor,
    output2: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> None:
    del enable_pdl
    if input1.shape[0] == 0:
        return
    if input1.dim() != 2 or input2.dim() != 2:
        raise ValueError("rmsnorm_fused_parallel expects 2D inputs")
    if input1.shape[0] != input2.shape[0]:
        raise ValueError(f"input row mismatch: {input1.shape[0]} vs {input2.shape[0]}")
    if input1.shape != output1.shape:
        raise ValueError(
            f"output1 shape {tuple(output1.shape)} does not match input1 "
            f"shape {tuple(input1.shape)}"
        )
    if input2.shape != output2.shape:
        raise ValueError(
            f"output2 shape {tuple(output2.shape)} does not match input2 "
            f"shape {tuple(input2.shape)}"
        )
    if input1.shape[-1] != weight1.shape[0]:
        raise ValueError(
            f"weight1 shape {tuple(weight1.shape)} does not match hidden size "
            f"{input1.shape[-1]}"
        )
    if input2.shape[-1] != weight2.shape[0]:
        raise ValueError(
            f"weight2 shape {tuple(weight2.shape)} does not match hidden size "
            f"{input2.shape[-1]}"
        )
    tensors = (input1, weight1, output1, input2, weight2, output2)
    if any(t.stride(-1) != 1 for t in tensors):
        raise ValueError("rmsnorm_fused_parallel requires contiguous last dimension")

    n_cols1 = input1.shape[-1]
    n_cols2 = input2.shape[-1]
    block1 = triton.next_power_of_2(n_cols1)
    block2 = triton.next_power_of_2(n_cols2)
    _rmsnorm_fused_parallel_kernel[(input1.shape[0],)](
        input1,
        weight1,
        output1,
        input2,
        weight2,
        output2,
        n_cols1,
        n_cols2,
        input1.stride(0),
        output1.stride(0),
        input2.stride(0),
        output2.stride(0),
        eps,
        BLOCK1=block1,
        BLOCK2=block2,
    )


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_ftz(a, b):
    # fp4_quantize is built with fast math: its products flush subnormal inputs and results to zero.
    return tl.inline_asm_elementwise(
        "mul.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e2m1x2(lo, hi):
    # One byte, lo in the low nibble: cvt puts its first source in the upper half.
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_nvfp4(
    normed,
    row,
    c,
    fp4_ptr,
    sf_ptr,
    stride_fp4,
    sf_scale,
    rcp_sf_scale,
    rcp6,
    BLOCK,
    K_TILES,
):
    # fp4_quantize's E4M3-scale recipe, bit for bit: block amax / 6 scaled into E4M3, values into E2M1.
    blocks = tl.reshape(normed, [BLOCK // 16, 16])
    vec_max = tl.max(tl.abs(blocks), axis=1).to(tl.float32)
    # Its zero test flushes too: a subnormal block amax counts as zero.
    vec_max = tl.where(vec_max < _FP32_MIN_NORMAL, 0.0, vec_max)
    sf = _mul_ftz(sf_scale, _mul_ftz(vec_max, rcp6)).to(tl.float8e4nv)
    out_scale = tl.where(
        vec_max != 0,
        _rcp_approx_ftz(_mul_ftz(sf.to(tl.float32), rcp_sf_scale)),
        0.0,
    )
    scaled = _mul_ftz(blocks.to(tl.float32), out_scale[:, None])
    even, odd = tl.split(tl.reshape(scaled, [BLOCK // 16, 8, 2]))
    packed = tl.reshape(_e2m1x2(even, odd).to(tl.uint8), [BLOCK // 2])
    tl.store(
        fp4_ptr + row * stride_fp4 + c * (BLOCK // 2) + tl.arange(0, BLOCK // 2), packed
    )
    k = c * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    tl.store(
        sf_ptr + _fp8_swizzled_scale_offset(row, k, K_TILES),
        sf.to(tl.uint8, bitcast=True),
    )


@triton.jit(do_not_specialize=["n_rows"])
def _add_rmsnorm_kernel(
    x_ptr,
    x2_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    out_fp8_ptr,
    fp8_scale_ptr,
    out_fp4_ptr,
    out_sf_ptr,
    fp4_scale_ptr,
    stride_x,
    stride_x2,
    stride_residual,
    stride_out,
    stride_out_fp8,
    stride_out_fp4,
    n_rows,
    n_cols: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
    CHUNKS: tl.constexpr,
    HAS_X2: tl.constexpr,
    HAS_FP8: tl.constexpr,
    HAS_FP4: tl.constexpr,
    GEMMA: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    # fp4_quantize's 128x4 scale layout counts 4 scales (64 columns) per k tile.
    K_TILES: tl.constexpr = n_cols // 64
    if HAS_FP4:
        if row >= n_rows:
            # Rows past the batch in the last 128-row scale tile get zero scales, as fp4_quantize writes them.
            if ENABLE_PDL:
                tl.extra.cuda.gdc_wait()
            for c in tl.static_range(CHUNKS):
                k = c * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
                tl.store(
                    out_sf_ptr + _fp8_swizzled_scale_offset(row, k, K_TILES),
                    tl.zeros([BLOCK // 16], dtype=tl.uint8),
                )
            return
    # Weights and the quant scale are model constants, loaded before the wait.
    weights = ()
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        weight = tl.load(weight_ptr + offsets, mask=offsets < n_cols, other=0.0).to(
            tl.float32
        )
        if GEMMA:
            weight = 1.0 + weight
        weights = weights + (weight,)
    if HAS_FP8:
        inv_scale = 1.0 / tl.load(fp8_scale_ptr).to(tl.float32)
    if HAS_FP4:
        sf_scale = tl.load(fp4_scale_ptr).to(tl.float32)
        rcp_sf_scale = _rcp_approx_ftz(
            tl.full([BLOCK // 16], 0.0, tl.float32) + sf_scale
        )
        rcp6 = _rcp_approx_ftz(tl.full([BLOCK // 16], 6.0, tl.float32))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    # Every row load before the in-place residual store, which loads cannot move past.
    totals = ()
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        mask = offsets < n_cols
        addend = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0)
        if HAS_X2:
            # Sum the two addends in their own dtype, as an all-reduce input would be.
            addend += tl.load(x2_ptr + row * stride_x2 + offsets, mask=mask, other=0.0)
        total = addend.to(tl.float32) + tl.load(
            residual_ptr + row * stride_residual + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        totals = totals + (total,)
    squares = tl.zeros([BLOCK], dtype=tl.float32)
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        tl.store(
            residual_ptr + row * stride_residual + offsets,
            totals[c].to(residual_ptr.dtype.element_ty),
            mask=offsets < n_cols,
        )
        squares += totals[c] * totals[c]
    variance = tl.sum(squares, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    for c in tl.static_range(CHUNKS):
        offsets = c * BLOCK + cols
        mask = offsets < n_cols
        normed = (totals[c] * rstd * weights[c]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + offsets, normed, mask=mask)
        if HAS_FP8:
            # The cast saturates like quantize_fp8's, so NaN stays NaN as it does there.
            quant = normed.to(tl.float32) * inv_scale
            tl.store(
                out_fp8_ptr + row * stride_out_fp8 + offsets,
                quant.to(out_fp8_ptr.dtype.element_ty),
                mask=mask,
            )
        if HAS_FP4:
            _store_nvfp4(
                normed,
                row,
                c,
                out_fp4_ptr,
                out_sf_ptr,
                stride_out_fp4,
                sf_scale,
                rcp_sf_scale,
                rcp6,
                BLOCK,
                K_TILES,
            )


def _row_block(cols: int) -> int:
    # Whole power-of-two chunks spare the masked lanes of one rounded-up block (5120 = 5 x 1024).
    chunk = cols & -cols
    return chunk if chunk >= 1024 else triton.next_power_of_2(cols)


# fp4_quantize switches its recipe when one of these is exactly "1" at launch; the NVFP4 copy follows the default.
_FP4_RECIPE_ENV = (
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6",
)


def _nvfp4_copy_possible(n_cols: int) -> bool:
    # Blackwell's E2M1 conversions, and a row the NVFP4 stores, which are unmasked, cover in whole chunks.
    return (
        current_platform().is_blackwell
        and n_cols % 64 == 0
        and n_cols % _row_block(n_cols) == 0
    )


def nvfp4_copy_supported(n_cols: int) -> bool:
    """Whether ``add_rmsnorm``'s NVFP4 copy of ``n_cols``-wide rows equals ``fp4_quantize``'s.

    The copy needs Blackwell and a width that is a power of two of at least 64
    or a multiple of 1024 (walked in whole power-of-two chunks, 5120 = 5 x 1024),
    and reproduces ``fp4_quantize``'s default recipe bit for bit.
    """
    return _nvfp4_copy_possible(n_cols) and not any(
        os.environ.get(name) == "1" for name in _FP4_RECIPE_ENV
    )


def add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    x2: torch.Tensor | None,
    out: torch.Tensor,
    out_fp8: torch.Tensor | None,
    fp8_scale: torch.Tensor | None,
    out_fp4: tuple[torch.Tensor, torch.Tensor] | None,
    fp4_scale: torch.Tensor | None,
    gemma: bool,
) -> None:
    """``residual += x (+ x2)``, then RMSNorm, optionally also into static FP8 or NVFP4.

    Args:
        x: ``[M, N]`` addend, rows may be strided but columns dense.
        residual: ``[M, N]`` residual stream, updated in place.
        weight: ``[N]`` norm weight.
        eps: Norm epsilon.
        x2: Optional second ``[M, N]`` addend, summed with ``x`` in their own dtype.
        out: ``[M, N]`` normalized output; may alias ``x``.
        out_fp8: Optional ``[M, N]`` FP8 output quantized with ``fp8_scale``.
        fp8_scale: One-element FP32 dequant scale, given exactly with
            ``out_fp8``.
        out_fp4: Optional ``(values, scales)`` NVFP4 copy, as ``fp4_quantize``
            returns it with its default recipe: ``[M, N // 2]`` uint8 E2M1 pairs and ``[ceil(M / 128) * 128,
            N // 16]`` uint8 E4M3 block scales in its 128x4 swizzled layout.
        fp4_scale: One-element FP32 global scale the NVFP4 copy is quantized
            with (``fp4_quantize``'s ``global_scale``), given exactly with ``out_fp4``.
        gemma: Scale by ``1 + weight`` in FP32 (Gemma) instead of ``weight``.
    """
    if (out_fp8 is None) != (fp8_scale is None):
        raise ValueError("out_fp8 and fp8_scale are given together")
    if (out_fp4 is None) != (fp4_scale is None):
        raise ValueError("out_fp4 and fp4_scale are given together")
    if out_fp8 is not None and out_fp4 is not None:
        raise ValueError("add_rmsnorm writes one quantized copy, FP8 or NVFP4")
    tensors = [t for t in (x, x2, residual, out, out_fp8) if t is not None]
    if any(t.dim() != 2 or t.shape != x.shape or t.stride(1) != 1 for t in tensors):
        raise ValueError("add_rmsnorm operands must be [M, N] with dense columns")
    rows, cols = x.shape
    if rows == 0:
        return
    width = triton.next_power_of_2(cols)
    block = _row_block(cols)
    grid_rows = rows
    if out_fp4 is not None:
        values, scales = out_fp4
        grid_rows = triton.cdiv(rows, 128) * 128
        if not _nvfp4_copy_possible(cols):
            raise ValueError(f"no NVFP4 copy for N={cols} columns on this device")
        if (
            values.shape != (rows, cols // 2)
            or values.stride(1) != 1
            or scales.shape != (grid_rows, cols // 16)
            or not scales.is_contiguous()
        ):
            raise ValueError("out_fp4 must be fp4_quantize's (values, scales) for x")
    enable_pdl = pdl_enabled()
    _add_rmsnorm_kernel[(grid_rows,)](
        x,
        x if x2 is None else x2,
        residual,
        weight,
        out,
        x if out_fp8 is None else out_fp8,
        weight if fp8_scale is None else fp8_scale,
        x if out_fp4 is None else out_fp4[0],
        x if out_fp4 is None else out_fp4[1],
        weight if fp4_scale is None else fp4_scale,
        x.stride(0),
        0 if x2 is None else x2.stride(0),
        residual.stride(0),
        out.stride(0),
        0 if out_fp8 is None else out_fp8.stride(0),
        0 if out_fp4 is None else out_fp4[0].stride(0),
        rows,
        cols,
        eps,
        BLOCK=block,
        CHUNKS=triton.cdiv(cols, block),
        HAS_X2=x2 is not None,
        HAS_FP8=out_fp8 is not None,
        HAS_FP4=out_fp4 is not None,
        GEMMA=gemma,
        ENABLE_PDL=enable_pdl,
        num_warps=min(max(width // 256, 1), 8),
        **({"launch_pdl": True} if enable_pdl else {}),
    )


# Rows each gated RMSNorm program normalizes when one warp covers a row.
GATED_RMSNORM_ROWS_PER_PROGRAM = 4


@register_kernel(
    "layernorm",
    "gated_rmsnorm",
    name="triton_gated_rmsnorm",
    solution="triton",
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
    weights_independent: bool,
) -> torch.Tensor:
    """Gated RMSNorm over groups of ``group_size`` features; see
    ``tokenspeed_kernel.ops.layernorm.gated_rmsnorm``."""
    enable_pdl = pdl_enabled()
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weights_independent = weights_independent and weight.is_contiguous()
    weight = weight.contiguous()

    M, N = x.shape
    assert N % group_size == 0
    ngroups = N // group_size
    assert weight.shape in ((N,), (group_size,))
    out = torch.empty_like(
        x, dtype=x.dtype if fp8_scale is None else torch.float8_e4m3fn
    )
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    block_m = GATED_RMSNORM_ROWS_PER_PROGRAM if num_warps == 1 else 1
    grid = (triton.cdiv(M, block_m), ngroups)
    with torch.cuda.device(x.device.index):
        # Pass the heuristic explicitly so startup can retain and load each
        # compiled binary; the heuristic wrapper discards that return value.
        kernel_launcher(gated_rmsnorm_kernel.fn, grid)(
            x,
            out,
            weight,
            z,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            HAS_Z=z is not None,
            NORM_BEFORE_GATE=norm_before_gate,
            SIGMOID_GATE=sigmoid_gate,
            num_warps=block_m * num_warps,
            ENABLE_PDL=enable_pdl,
            WEIGHTS_INDEPENDENT=weights_independent,
            OUT_FP8=fp8_scale is not None,
            SHARED_WEIGHT=weight.shape[0] != N,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return out.reshape(x_shape_og)


@triton.jit
def _dequantize_head_half(in_ptrs, scale_ptrs, out_ptrs, token_scale, mask):
    """Store the dequantized (``fp8_dequantize_unscaled``) values of an unscaled
    FP8 GEMM output range."""
    raw = tl.load(in_ptrs, mask=mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, fp8_dequantize_unscaled(raw, token_scale, scale), mask=mask)


@triton.jit
def _fused_qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    token_scale_ptr=None,
    q_scale_ptr=None,
    k_scale_ptr=None,
    v_in_ptr=None,
    v_scale_ptr=None,
    v_out_ptr=None,
    g_in_ptr=None,
    g_scale_ptr=None,
    g_out_ptr=None,
    v_in_token_stride=0,
    g_in_token_stride=0,
    v_out_token_stride=0,
    g_out_token_stride=0,
    DEQUANT: tl.constexpr = False,
):
    """Per-head QK RMSNorm then full NeoX RoPE; one program per (token, head).

    DEQUANT: q and k are an unscaled FP8 GEMM's output, dequantized first (the
    scaled GEMM's 16-bit output); q programs also write the dequantized output
    gate head of their index, k programs the V head of theirs."""
    if ENABLE_PDL:
        # Q and K are the projections' stores, before the first load.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    heads = num_q_heads + num_kv_heads
    row = tl.program_id(0).to(tl.int64)
    token = row // heads
    head = row % heads
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    offs = tl.arange(0, HALF_BLOCK)
    mask = offs < half_dim
    q_mask = mask & ~is_k
    k_mask = mask & is_k
    offsets = local_head * head_dim + offs
    q_base = q_ptr + token * q_stride_t + offsets
    k_base = k_ptr + token * k_stride_t + offsets
    q1 = tl.load(q_base, mask=q_mask, other=0.0)
    q2 = tl.load(q_base + half_dim, mask=q_mask, other=0.0)
    k1 = tl.load(k_base, mask=k_mask, other=0.0)
    k2 = tl.load(k_base + half_dim, mask=k_mask, other=0.0)
    if DEQUANT:
        token_scale = tl.load(token_scale_ptr + token)
        s1 = tl.where(
            is_k,
            tl.load(k_scale_ptr + offsets, mask=k_mask, other=0.0),
            tl.load(q_scale_ptr + offsets, mask=q_mask, other=0.0),
        )
        s2 = tl.where(
            is_k,
            tl.load(k_scale_ptr + offsets + half_dim, mask=k_mask, other=0.0),
            tl.load(q_scale_ptr + offsets + half_dim, mask=q_mask, other=0.0),
        )
        x1 = fp8_dequantize_unscaled(tl.where(is_k, k1, q1), token_scale, s1)
        x2 = fp8_dequantize_unscaled(tl.where(is_k, k2, q2), token_scale, s2)
        _dequantize_head_half(
            g_in_ptr + token * g_in_token_stride + offsets,
            g_scale_ptr + offsets,
            g_out_ptr + token * g_out_token_stride + offsets,
            token_scale,
            q_mask,
        )
        _dequantize_head_half(
            g_in_ptr + token * g_in_token_stride + offsets + half_dim,
            g_scale_ptr + offsets + half_dim,
            g_out_ptr + token * g_out_token_stride + offsets + half_dim,
            token_scale,
            q_mask,
        )
        _dequantize_head_half(
            v_in_ptr + token * v_in_token_stride + offsets,
            v_scale_ptr + offsets,
            v_out_ptr + token * v_out_token_stride + offsets,
            token_scale,
            k_mask,
        )
        _dequantize_head_half(
            v_in_ptr + token * v_in_token_stride + offsets + half_dim,
            v_scale_ptr + offsets + half_dim,
            v_out_ptr + token * v_out_token_stride + offsets + half_dim,
            token_scale,
            k_mask,
        )
    else:
        x1 = tl.where(is_k, k1, q1).to(tl.float32)
        x2 = tl.where(is_k, k2, q2).to(tl.float32)
    qw1 = tl.load(q_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    qw2 = tl.load(q_weight_ptr + half_dim + offs, mask=mask, other=0.0).to(tl.float32)
    kw1 = tl.load(k_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    kw2 = tl.load(k_weight_ptr + half_dim + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.where(is_k, kw1, qw1)
    w2 = tl.where(is_k, kw2, qw2)
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    # The normalized head is rounded to the input dtype before RoPE.
    x1_norm = (x1 * inv_rms * w1).to(INPUT_DTYPE).to(tl.float32)
    x2_norm = (x2 * inv_rms * w2).to(INPUT_DTYPE).to(tl.float32)

    pos = tl.load(positions_ptr + token).to(tl.int64)
    cache_offset = pos * cache_stride_p
    cos = tl.load(cos_sin_cache_ptr + cache_offset + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_dim + offs, mask=mask, other=0.0
    ).to(tl.float32)
    o1 = x1_norm * cos - x2_norm * sin
    o2 = x2_norm * cos + x1_norm * sin
    q_out_base = q_out_ptr + token * q_out_stride_t + offsets
    k_out_base = k_out_ptr + token * k_out_stride_t + offsets
    tl.store(q_out_base, o1, mask=q_mask)
    tl.store(q_out_base + half_dim, o2, mask=q_mask)
    tl.store(k_out_base, o1, mask=k_mask)
    tl.store(k_out_base + half_dim, o2, mask=k_mask)


def fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head QK RMSNorm then full NeoX RoPE in one launch.

    Each head is normalized in FP32, scaled by its weight and rounded to the
    input dtype; RoPE rotates that value in FP32 and rounds once more. This
    is a different rounding than the attention prologue's fused kernel, which
    rotates the unrounded norm, so a model that needs this order calls the
    kernel (Beam's sliding layers call it directly) and
    gives its attention layer no norm or rotary step. Where
    ``pdl_enabled()`` it is a programmatic dependent launch that waits on the
    projection before its first load.

    Args:
        q: (n_tokens, num_q_heads * head_dim), rows may be strided
        k: (n_tokens, num_kv_heads * head_dim), rows may be strided
        q_weight: (head_dim,) RMSNorm weight of the query heads
        k_weight: (head_dim,) RMSNorm weight of the key heads
        cos_sin_cache: (max_pos, head_dim) packed [cos|sin], float32
        positions: (n_tokens,) int32 or int64
        eps: RMSNorm epsilon
        num_q_heads: number of Q heads (after TP split)
        num_kv_heads: number of KV heads (after TP split)
        head_dim: per-head dimension (must be even)
        dequant: ``None`` (the default) leaves the kernel as before. Given, q
            and k are an unscaled FP8 GEMM's Q and K columns: each head is
            dequantized first (:class:`UnscaledFP8QKVGate`), and the same
            launch writes the dequantized V and output gate columns.

    Returns:
        (q_out, k_out), both contiguous (n_tokens, heads * head_dim)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    n_tokens = q.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q.dtype, device=q.device
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    if n_tokens == 0:
        return q_out, k_out

    half_dim = head_dim // 2
    half_block = triton.next_power_of_2(half_dim)
    q_stride = q.stride(0) if q.dim() > 1 else q.shape[-1]
    k_stride = k.stride(0) if k.dim() > 1 else k.shape[-1]
    enable_pdl = pdl_enabled()
    dequant_kwargs = _dequant_launch_kwargs(dequant, q, k, n_tokens)
    _fused_qk_rmsnorm_rope_kernel[(n_tokens * (num_q_heads + num_kv_heads),)](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_stride,
        k_stride,
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        half_dim,
        eps,
        INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        HALF_BLOCK=half_block,
        ENABLE_PDL=enable_pdl,
        num_warps=max(1, half_block // 64),
        num_stages=2,
        **dequant_kwargs,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return q_out, k_out


__all__ = [
    "GATED_RMSNORM_ROWS_PER_PROGRAM",
    "UnscaledFP8QKVGate",
    "add_rmsnorm",
    "fused_qk_rmsnorm_rope",
    "grouped_gemma_rmsnorm",
    "nvfp4_copy_supported",
    "rmsnorm",
    "qk_rmsnorm",
    "rmsnorm_fused_parallel",
    "triton_gated_rmsnorm",
]
