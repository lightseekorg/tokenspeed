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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    Platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures
from tokenspeed_kernel.thirdparty.deepgemm import block_fp8_mm, prepare_block_fp8_weight


def prepare_deepgemm_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare standard block-FP8 weights for explicit DeepGEMM serving.

    Args:
        weight: E4M3 [N,K] weights, N%64=0 and K%128=0.
        scales: Canonical FP32 [ceil(N/128),K/128] block scales.

    Returns:
        Hopper's unchanged canonical tensors, or Blackwell's requantized
        weights and native packed UE8M0 scales. Call once after loading.
    """
    platform = Platform.get()
    if not (
        platform.is_hopper
        or platform.arch_version in (ArchVersion(10, 0), ArchVersion(10, 3))
    ):
        raise ValueError("DeepGEMM dense FP8 requires SM90, SM100 or SM103")
    n, k = weight.shape
    if n % 64 or k % 128:
        raise ValueError("DeepGEMM weights require N%64=0 and K%128=0")
    if platform.is_hopper:
        return weight, scales
    return prepare_block_fp8_weight(weight, scales)


@register_kernel(
    "gemm",
    "mm",
    name="deepgemm_mm_fp8_blockscale",
    solution="deepgemm",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=frozenset().union(
        *(
            format_signatures(
                ("a", "b"),
                "mxfp8",
                {torch.float8_e4m3fn},
                scale=ScaleFormat(
                    storage_dtype=dtype, granularity="block", block_shape=(128, 128)
                ),
            )
            for dtype in (torch.float32, torch.int32)
        )
    ),
    traits={
        "n_align": frozenset({64}),
        "k_align": frozenset({128}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({True}),
        "block_scale_layout": frozenset({"canonical", "canonical_blackwell"}),
        "out_dtype": frozenset({torch.bfloat16}),
        "out_inner_stride_one": frozenset({True}),
    },
    # Below the existing Triton block-FP8 leaf: auto keeps its current choice.
    priority=Priority.PERFORMANT,
)
def deepgemm_mm_fp8_blockscale(
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
    """Explicit E4M3 GEMM with Hopper FP32 or Blackwell packed UE8M0 scales.

    Activations use 1x128 scales and weights use 128x128 scales. Output is
    BF16; a supplied contiguous output buffer is written in place. Hopper
    converts scale layouts per call; Blackwell consumes prepared packed scales.
    Unsupported inputs are rejected even for a forced kernel override.
    """
    platform = Platform.get()
    packed = platform.arch_version in (ArchVersion(10, 0), ArchVersion(10, 3))
    if not (platform.is_hopper or packed):
        raise ValueError("DeepGEMM block FP8 requires SM90, SM100 or SM103")
    if alpha is not None or block_size is None or tuple(block_size) != (128, 128):
        raise ValueError("DeepGEMM requires 128x128 block scales and alpha=None")
    if (
        out_dtype != torch.bfloat16
        or A.dtype != torch.float8_e4m3fn
        or B.dtype != A.dtype
    ):
        raise ValueError("DeepGEMM requires E4M3 inputs and BF16 output")
    if A.ndim != 2 or B.ndim != 2 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("DeepGEMM requires contiguous 2D A[M,K] and B[N,K]")
    m, k = A.shape
    n, kb = B.shape
    if m < 0 or n < 1 or k < 1 or k != kb or n % 64 or k % 128:
        raise ValueError(
            "DeepGEMM requires positive dimensions, matching K, N%64=0 and K%128=0"
        )
    if not A.is_cuda or B.device != A.device:
        raise ValueError("DeepGEMM operands must share a CUDA device")
    for scale, rows, shape in (
        (A_scales, m, (m, (k // 128 + 3) // 4) if packed else (m, k // 128)),
        (
            B_scales,
            n,
            (n, (k // 128 + 3) // 4) if packed else ((n + 127) // 128, k // 128),
        ),
    ):
        if (
            scale is None
            or scale.dtype != (torch.int32 if packed else torch.float32)
            or tuple(scale.shape) != shape
            or scale.device != A.device
        ):
            raise ValueError(
                "DeepGEMM requires matching canonical FP32 or packed UE8M0 scales"
            )
        if (
            packed
            and m
            and (scale.stride(0) != 1 or scale.stride(1) != (rows + 3) // 4 * 4)
        ):
            raise ValueError(
                "DeepGEMM packed scales require TMA-aligned MN-major strides"
            )
    if not packed and not B_scales.is_contiguous():
        raise ValueError("DeepGEMM requires contiguous weight scales")
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "DeepGEMM out must be contiguous BF16 [M,N] on the operand device"
        )
    if not m:
        return out
    return block_fp8_mm(A, B, A_scales, B_scales, out, enable_pdl=pdl_enabled())
