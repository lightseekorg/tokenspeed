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
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
)

# ===-----------------------------------------------------------------------===#
# BF16 grouped projection: direct contiguous output (AMD gfx950)
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "gemm",
    "grouped_bf16_projection",
    name="torch_grouped_bf16_projection_gfx950",
    solution="torch",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures(("x", "weight"), "dense", {torch.bfloat16}),
    traits={
        "batch": frozenset({2}),
        "m_min": frozenset({2}),
        "m_max": frozenset({192}),
        "n": frozenset({1024}),
        "k": frozenset({4096}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({True}),
        "is_cuda": frozenset({True}),
    },
    # Prefer the specialized Gluon kernel regardless of registration order.
    priority=Priority.PERFORMANT,
)
def torch_grouped_bf16_projection_gfx950(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Project BF16 groups into their final contiguous token-major layout.

    Args:
        x: BF16 ``[tokens, groups, input_dim]`` with unit inner stride.
        weight: BF16 ``[groups, output_dim, input_dim]`` with unit inner stride.
        out: Optional contiguous BF16 ``[tokens, groups, output_dim]`` storage.

    Returns:
        BF16 output, or ``out`` when supplied. In inference the batched GEMM
        writes contiguous token-major storage through a group-major view with
        padded row strides, so flattening groups into features needs no copy.
        Gradient-enabled calls retain the canonical reference layout.
    """
    # PyTorch's out= GEMM does not support autograd; retain the reference
    # behavior for callers that explicitly request gradients.
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        from tokenspeed_kernel.numerics.torch.gemm import (
            torch_grouped_bf16_projection,
        )

        return torch_grouped_bf16_projection(x, weight, out)
    if out is None:
        out = x.new_empty((x.shape[0], weight.shape[0], weight.shape[1]))
    torch.bmm(x.permute(1, 0, 2), weight.transpose(1, 2), out=out.permute(1, 0, 2))
    return out


# ===-----------------------------------------------------------------------===#
# Scaled FP8: PyTorch row-wise scaled GEMM (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


_FP8_CHANNEL_SCALE = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
# Below this many rows the Triton kernel is as fast and skips the scale expansion.
TORCH_FP8_SCALED_MIN_M = 65


@register_kernel(
    "gemm",
    "mm",
    name="torch_mm_fp8_scaled",
    solution="torch",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(
        ("a", "b"), "scaled-fp8", {torch.float8_e4m3fn}, scale=_FP8_CHANNEL_SCALE
    ),
    # torch._scaled_mm reads row-major A and column-major B, both 16-aligned,
    # and its row-wise scaling writes BF16 only (an FP16 output is garbage on
    # SM100, not refused).
    traits={
        "m_min": frozenset({TORCH_FP8_SCALED_MIN_M}),
        "out_dtype": frozenset({torch.bfloat16}),
        "n_align": frozenset({16}),
        "k_align": frozenset({16}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({False}),
        "b_layout": frozenset({"KN"}),
    },
    priority=Priority.PERFORMANT + 3,
)
def torch_mm_fp8_scaled(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 GEMM with per-token/per-tensor A and per-channel/per-tensor B scales.

    Runs PyTorch's row-wise scaled CUTLASS GEMM, which applies both scales to
    the FP32 accumulator like ``triton_mm_fp8_scaled``.

    Args:
        A: ``[M, K]`` row-major FP8 activations.
        B: ``[K, N]`` column-major FP8 weights (a transposed ``[N, K]``).
        A_scales: FP32 ``[M, 1]`` per-token or one-element per-tensor scales.
        B_scales: FP32 ``[N, 1]`` per-channel or one-element per-tensor scales.
        out_dtype: BF16; the row-wise scaling writes nothing else.
        alpha: Must be None; the scales carry the dequant.
        block_size: Must be None; the scales are per token or channel.
        out: Optional ``[M, N]`` output buffer.

    Returns:
        ``[M, N]`` tensor ``(A * A_scales) @ (B * B_scales^T)``, ``out`` when given.
    """
    if alpha is not None or block_size is not None:
        raise ValueError("row-wise scaled FP8 GEMM takes no alpha or block_size")
    # The out_dtype trait keeps FP16 away in normal selection, but an override
    # skips trait matching; torch._scaled_mm writes garbage rather than refusing.
    if out_dtype != torch.bfloat16:
        raise ValueError(f"row-wise scaled FP8 GEMM writes BF16, not {out_dtype}")
    m, n = A.shape[0], B.shape[1]
    # The kernel takes only dense [M, 1] and [1, N] scales; per-tensor ones are expanded.
    scale_a = A_scales.reshape(-1, 1).expand(m, 1).contiguous()
    scale_b = B_scales.reshape(1, -1).expand(1, n).contiguous()
    return torch._scaled_mm(
        A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype, out=out
    )


# ===-----------------------------------------------------------------------===#
# Decode GEMV: PyTorch
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "gemm",
    "mv",
    name="torch_decode_gemv",
    solution="torch",
    signatures=frozenset(
        format_signature(
            x=dense_tensor_format(x_dtype),
            weight=dense_tensor_format(weight_dtype),
        )
        for x_dtype, weight_dtype in (
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.float32),
        )
    ),
    traits={},
    priority=Priority.PORTABLE,
)
def torch_decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.dtype == torch.bfloat16 and weight.dtype == torch.float32:
        x = x.float()
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    return x @ weight.t()
