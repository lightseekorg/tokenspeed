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

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.triton import silu_and_mul
from tokenspeed_kernel.ops.gemm._triton.bf16x3 import (
    BF16X3_MIN_M,
    split_fp32_weight_bf16x3,
    triton_bf16x3_gemm_fp32,
)
from tokenspeed_kernel.ops.gemm._triton.fp8 import (
    triton_bmm_fp8_blockscale,
    triton_scaled_mm,
    w8a8_block_fp8_matmul_triton,
)
from tokenspeed_kernel.ops.gemm._triton.gemv import (
    grouped_bf16_projection_rowcta,
    triton_rowcta_gemm_fp32,
    triton_rowcta_gemv,
)
from tokenspeed_kernel.ops.gemm._triton.mxfp4 import triton_mm_mxfp4
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, Platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)

_fp8_dtype = torch.float8_e4m3fn


# ===-----------------------------------------------------------------------===#
# FP8 block-scale: Triton (AMD, NVIDIA)
# ===-----------------------------------------------------------------------===#


_MXFP8_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_MXFP8_UE8M0_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_FLOAT_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_FORMAT_SIGNATURES = (
    format_signatures(("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_BLOCK_SCALE)
    | format_signatures(("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_UE8M0_SCALE)
    | frozenset(
        {
            format_signature(
                a=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_FLOAT_1X32_SCALE),
                b=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_UE8M0_SCALE),
            )
        }
    )
)


@register_kernel(
    "gemm",
    "mm",
    name="triton_mm_fp8_blockscale",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd", "nvidia"}),
    ),
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PERFORMANT + 3,
)
def triton_mm_fp8_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert block_size is not None, "block_size is required for triton_mm_fp8_blockscale"
    assert (
        A_scales is not None
    ), "A_scales is required; online quantization should be done by the caller"
    if B_scales is None:
        raise ValueError("B_scales is required for triton MXFP8 GEMM")
    return w8a8_block_fp8_matmul_triton(
        A,
        B,
        A_scales,
        B_scales,
        block_size=block_size,
        output_dtype=out_dtype,
        out=out,
    )


register_kernel(
    "gemm",
    "bmm",
    name="triton_bmm_fp8_blockscale",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={
        "a_inner_stride_one": frozenset({True}),
        "out_inner_stride_one": frozenset({True}),
    },
    priority=Priority.PERFORMANT + 3,
)(triton_bmm_fp8_blockscale)


_TRITON_GROUPED_OUTPUT_SIGNATURES = frozenset(
    format_signature(
        attention=dense_tensor_format(input_dtype),
        weight=dense_tensor_format(_fp8_dtype),
    )
    for input_dtype in (torch.float16, torch.bfloat16)
)


@register_kernel(
    "gemm",
    "dsv4_grouped_output_projection",
    name="triton_dsv4_grouped_output_projection",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=_TRITON_GROUPED_OUTPUT_SIGNATURES,
    traits={},
    priority=Priority.PERFORMANT + 3,
)
def triton_dsv4_grouped_output_projection(
    *,
    attention: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    num_groups: int,
    heads_per_group: int,
    output_dim: int,
    nope_dim: int,
    rope_dim: int,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Run V4's grouped output projection with canonical scales and Triton BMM."""
    from tokenspeed_kernel.ops.attention.dsv4.triton import (
        dsv4_fused_inv_rope_fp8_quant,
    )

    values, scales = dsv4_fused_inv_rope_fp8_quant(
        attention,
        positions,
        cos_sin_cache,
        n_groups=num_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=block_size[1],
        tma_aligned_scales=False,
    )
    input_dim = heads_per_group * attention.shape[-1]
    grouped_weight = weight.view(num_groups, output_dim, input_dim)
    block_n, block_k = block_size
    grouped_scales = weight_scale.view(
        num_groups,
        output_dim // block_n,
        input_dim // block_k,
    )
    output = torch.empty(
        (attention.shape[0], num_groups, output_dim),
        dtype=torch.bfloat16,
        device=attention.device,
    )
    triton_bmm_fp8_blockscale(
        values.transpose(0, 1),
        grouped_weight,
        scales.transpose(0, 1),
        grouped_scales,
        output.dtype,
        block_size=list(block_size),
        out=output.transpose(0, 1),
    )
    return output


# ===-----------------------------------------------------------------------===#
# MXFP4: Triton (AMD)
# ===-----------------------------------------------------------------------===#


_MXFP4_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(32,),
)
_MXFP4_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp4", {torch.uint8}, scale=_MXFP4_SCALE
)


register_kernel(
    "gemm",
    "mm",
    name="triton_mm_mxfp4",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=_MXFP4_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE,
)(triton_mm_mxfp4)


# ===-----------------------------------------------------------------------===#
# MXFP4 linear with dynamic MXFP4 activations: Triton (AMD)
# ===-----------------------------------------------------------------------===#


# Weight stored in checkpoint order: packed E2M1 [N, K/2] plus UE8M0 [N, K/32].
MXFP4_LINEAR_LINEAR_LAYOUT = "linear"
MXFP4_LINEAR_SIGNATURE = format_signature(
    x=dense_tensor_format(torch.bfloat16),
    weight=tensor_format("mxfp4", torch.uint8, scale=_MXFP4_SCALE),
)
MXFP4_LINEAR_SIGNATURES = frozenset({MXFP4_LINEAR_SIGNATURE})


@register_kernel(
    "gemm",
    "mxfp4_linear",
    name="triton_mxfp4_linear",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=MXFP4_LINEAR_SIGNATURES,
    traits={"weight_layout": frozenset({MXFP4_LINEAR_LINEAR_LAYOUT})},
    priority=Priority.PORTABLE,
)
def triton_mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """``mxfp4(x) @ W.T``: quantize ``x`` to MXFP4, then the MXFP4 GEMM."""
    x_fp4, x_scale = mxfp4_quantize(x)
    return triton_mm_mxfp4(x_fp4, weight, x_scale, weight_scale, x.dtype, out=out)


@register_kernel(
    "gemm",
    "mxfp4_swiglu_linear",
    name="triton_mxfp4_swiglu_linear",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=MXFP4_LINEAR_SIGNATURES,
    traits={"weight_layout": frozenset({MXFP4_LINEAR_LINEAR_LAYOUT})},
    priority=Priority.PORTABLE,
)
def triton_mxfp4_swiglu_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """``silu(gate) * up`` of ``mxfp4(x) @ W.T`` with stacked [gate; up] rows."""
    gate_up = triton_mxfp4_linear(x, weight, weight_scale, out=None)
    if out is None:
        out = torch.empty(
            (x.shape[0], weight.shape[0] // 2), dtype=x.dtype, device=x.device
        )
    return silu_and_mul(gate_up, out)


# ===-----------------------------------------------------------------------===#
# Scaled FP8: Triton (NVIDIA SM100+, AMD gfx950+)
# ===-----------------------------------------------------------------------===#


_FP8_TENSOR_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="tensor",
)
_FP8_CHANNEL_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="channel",
)
_FP8_SCALED_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=_FP8_TENSOR_SCALE
) | format_signatures(("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=_FP8_CHANNEL_SCALE)


@register_kernel(
    "gemm",
    "mm",
    name="triton_mm_fp8_scaled",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia", "amd"}),
        vendor_min_arch_versions={
            "nvidia": ArchVersion(10, 0),
            "amd": ArchVersion(9, 5),
        },
    ),
    signatures=_FP8_SCALED_FORMAT_SIGNATURES,
    traits={
        "b_layout": frozenset({"KN"}),
    },
    priority=Priority.PERFORMANT + 2,
)
def triton_mm_fp8_scaled(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 E4M3 GEMM with per-token/per-tensor A and per-channel/per-tensor B scales.

    Args:
        A: ``[M, K]`` FP8 activations.
        B: ``[K, N]`` FP8 weights (a transposed ``[N, K]`` view is accepted).
        A_scales: FP32 ``[M, 1]`` per-token or ``[1, 1]`` per-tensor scales.
        B_scales: FP32 ``[N, 1]`` per-channel or ``[1, 1]`` per-tensor scales.
        out_dtype: Output dtype.
        alpha: Must be None; dequantization uses the operand scales.
        block_size: Must be None; scales are per token/channel or per tensor.
        bias: Optional ``[N]`` bias added in the epilogue.
        out: Optional ``[M, N]`` output buffer.

    Returns:
        ``[M, N]`` tensor ``(A * A_scales) @ (B * B_scales^T)`` in ``out_dtype``.
    """
    if alpha is not None or block_size is not None:
        raise ValueError("row-wise scaled FP8 GEMM takes no alpha or block_size")
    return triton_scaled_mm(
        A,
        B,
        A_scales,
        B_scales,
        out_dtype=out_dtype,
        bias=bias,
        out=out,
    )


# ===-----------------------------------------------------------------------===#
# MXFP8 skinny: Triton (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if Platform.get().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.skinny import (
        launch_triton_mm_mxfp8_skinny_gfx950 as _mm_mxfp8_skinny_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.skinny import (
        supports_mxfp8_skinny_shape as _supports_mxfp8_skinny_shape,
    )

    @register_kernel(
        "gemm",
        "mm",
        name="triton_mm_mxfp8_skinny_gfx950",
        solution="triton",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    a=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_UE8M0_SCALE),
                    b=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_UE8M0_SCALE),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "mnk_problem_filter": frozenset({_supports_mxfp8_skinny_shape}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "block_scale_layout": frozenset({"canonical"}),
            "out_dtype": frozenset({torch.bfloat16, torch.float16}),
            "out_inner_stride_one": frozenset({True}),
        },
    )
    def triton_mm_mxfp8_skinny_gfx950(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        alpha: torch.Tensor | None,
        block_size: list[int] | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if A_scales is None or B_scales is None or block_size is None:
            raise ValueError("gfx950 skinny MXFP8 GEMM requires scales and block_size")
        return _mm_mxfp8_skinny_impl(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            out=out,
        )

else:

    def triton_mm_mxfp8_skinny_gfx950(**kwargs):
        raise ImportError(
            "triton_mm_mxfp8_skinny_gfx950 requires tokenspeed-kernel-amd"
        )


# ===-----------------------------------------------------------------------===#
# FP32 decode GEMM: Triton BF16x3 (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


register_kernel(
    "gemm",
    "mv",
    name="triton_bf16x3_gemm_fp32",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={
        "m_min": frozenset({BF16X3_MIN_M}),
        "n_align": frozenset({128}),
        # In a GB200 sweep (N 128 to 1024, K 2048 to 8192, 17 to 2048 rows)
        # this kernel was up to 1.6x slower than the FP32 Torch product for
        # 128-row weights at most row counts up to 128, and at least 1.05x
        # faster at every point for weights of 256 rows and more. 128-row
        # weights stay on Torch.
        "n_min": frozenset({256}),
        "n_max": frozenset({1024}),
        "k_align": frozenset({64}),
        # K stays within that sweep's range. Up to 128 rows the smallest margin
        # for 256-row weights fell from 1.52x at K 6144 to 1.05x at K 2048.
        "k_min": frozenset({2048}),
        "k_max": frozenset({8192}),
        "weight_split": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
    weight_preprocessor=split_fp32_weight_bf16x3,
)(triton_bf16x3_gemm_fp32)


# ===-----------------------------------------------------------------------===#
# BF16 and FP32 decode GEMV: Triton row-CTA
# ===-----------------------------------------------------------------------===#


# Registry dispatch: rowcta owns BF16 M == 1 and the FP32 row-CTA kernel
# owns narrow FP32-weight projections with M <= 16 (FP32 or BF16 activations).
# From 17 rows, BF16 activations against an FP32 weight take
# gluon_simt_gemm_fp32 up to 96 rows and, with a weight split,
# triton_bf16x3_gemm_fp32, where their registrations match. Torch handles
# the shapes that no registered kernel takes.
_BF16_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    }
)


_FP32_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.float32),
            weight=dense_tensor_format(torch.float32),
        )
    }
)


# BF16 activations against an FP32 weight, with an FP32 result.
_BF16_FP32_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.float32),
        )
    }
)


register_kernel(
    "gemm",
    "mv",
    name="triton_rowcta_gemv",
    solution="triton",
    signatures=_BF16_SIG,
    traits={
        "m": frozenset({1}),
        "n_min": frozenset({128}),
        "k_min": frozenset({128}),
    },
    priority=Priority.SPECIALIZED,
)(triton_rowcta_gemv)


def _fp32_rowcta_k_fits(m: int, n: int, k: int) -> bool:
    # One row is reduced in a single block of up to 64K elements. More rows
    # unroll 512-wide blocks, which can spill registers past 15 blocks.
    return k <= (65536 if m == 1 else 7680)


# The FP32 kernel beat Torch at every measured shape inside these traits on
# GB200. Outside them Torch won some shapes: more than 1024 outputs (short
# rows, or 9 to 16 rows), rows that are not a multiple of four elements (no
# 16-byte loads), and K past the bounds of _fp32_rowcta_k_fits.
register_kernel(
    "gemm",
    "mv",
    name="triton_rowcta_gemm_fp32",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_FP32_SIG | _BF16_FP32_SIG,
    traits={
        "m": frozenset(range(1, 17)),
        "n_max": frozenset({1024}),
        "k_align": frozenset({4}),
        "mnk_problem_filter": frozenset({_fp32_rowcta_k_fits}),
    },
    priority=Priority.SPECIALIZED,
)(triton_rowcta_gemm_fp32)


# ===-----------------------------------------------------------------------===#
# BF16 grouped projection: Triton row-CTA (NVIDIA SM90+)
# ===-----------------------------------------------------------------------===#


register_kernel(
    "gemm",
    "grouped_bf16_projection",
    name="grouped_bf16_projection_rowcta",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
    ),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "batch": frozenset({2}),
        "m": frozenset({1}),
        "n": frozenset({1024}),
        "k": frozenset({4096}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({True}),
        "is_cuda": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)(grouped_bf16_projection_rowcta)
