"""Fused SwiGLU + FP8 UE8M0 quantization for DeepSeek V4 shared expert MLP.

Replaces the sequence:
    gate, up = gate_up.float().chunk(2, -1)
    gate = clamp(gate, max=limit)
    up   = clamp(up, -limit, limit)
    x    = (silu(gate) * up).to(bf16)
    # … then online FP8 quant inside down_proj GEMM

with a single Triton kernel that reads gate_up once and writes FP8 + UE8M0
packed block scale, suitable for feeding directly into deep_gemm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    create_per_token_group_quant_fp8_output_scale,
)


@triton.jit
def _fused_swiglu_fp8_ue8m0_kernel(
    gate_up_ptr,
    out_ptr,
    scale_ptr,
    M,
    N: tl.constexpr,
    gate_up_stride_row,
    out_stride_row,
    scale_col_stride,
    swiglu_limit,
    eps,
    bit8_min,
    bit8_max,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    groups_per_row = N // GROUP_SIZE
    row = pid // groups_per_row
    group_col = pid % groups_per_row

    gate_offset = (
        row.to(tl.int64) * gate_up_stride_row + group_col.to(tl.int64) * GROUP_SIZE
    )
    up_offset = gate_offset + N
    out_offset = row.to(tl.int64) * out_stride_row + group_col.to(tl.int64) * GROUP_SIZE

    cols = tl.arange(0, GROUP_SIZE)

    gate = tl.load(gate_up_ptr + gate_offset + cols).to(tl.float32)
    up = tl.load(gate_up_ptr + up_offset + cols).to(tl.float32)

    if swiglu_limit > 0.0:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)

    silu_gate = gate * tl.sigmoid(gate)
    y = silu_gate * up

    _absmax = tl.max(tl.abs(y))
    scale_raw = tl.maximum(_absmax / bit8_max, eps)
    exponent = tl.ceil(tl.log2(scale_raw))
    y_s = tl.exp2(exponent)
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + out_offset + cols, y_q)

    scale_pack_col = group_col // 4
    scale_pack_pos = group_col % 4
    scale_ptr_offset = scale_pack_col.to(tl.int64) * scale_col_stride + row.to(tl.int64)
    exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint32)
    packed_scale = exponent_biased << (scale_pack_pos * 8)
    tl.atomic_or(scale_ptr + scale_ptr_offset, packed_scale, sem="relaxed")


def fused_swiglu_fp8_ue8m0(
    gate_up: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU activation + FP8 UE8M0 block-scale quantization.

    Args:
        gate_up: [M, 2*N] tensor (gate in first half, up in second half).
                 Can be BF16 or FP8 (will be cast to float32 internally).
        swiglu_limit: Clamp bound. 0 or negative disables clamping.

    Returns:
        (fp8_out, scale): fp8_out is [M, N] float8_e4m3fn,
                          scale is UE8M0 packed int32 column-major TMA-aligned.
    """
    assert gate_up.ndim == 2, f"Expected 2D input, got {gate_up.ndim}D"
    M, two_N = gate_up.shape
    assert two_N % 2 == 0
    N = two_N // 2
    assert N % 128 == 0, f"N={N} must be multiple of 128 for UE8M0 group_size=128"

    GROUP_SIZE = 128
    dtype = torch.float8_e4m3fn
    info = torch.finfo(dtype)

    out = torch.empty((M, N), device=gate_up.device, dtype=dtype)
    scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=(M, N),
        device=gate_up.device,
        group_size=GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

    num_groups = M * (N // GROUP_SIZE)
    _fused_swiglu_fp8_ue8m0_kernel[(num_groups,)](
        gate_up,
        out,
        scale,
        M,
        N,
        gate_up.stride(0),
        out.stride(0),
        scale.stride(-1),
        swiglu_limit if swiglu_limit is not None and swiglu_limit > 0 else 0.0,
        1e-10,
        bit8_min=info.min,
        bit8_max=info.max,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=min(max(GROUP_SIZE // 256, 1), 8),
        num_stages=1,
    )

    return out, scale
