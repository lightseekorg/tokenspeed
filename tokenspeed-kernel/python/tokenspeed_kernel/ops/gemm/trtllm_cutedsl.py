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

"""Opt-in TRT-LLM CuTe-DSL FP8 GEMM with unmodified FP32 block scales."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures


@register_kernel(
    "gemm",
    "mm",
    name="trtllm_cutedsl_mm_fp8_blockscale",
    solution="trtllm_cutedsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 9),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(
        ("a", "b"),
        "mxfp8",
        {torch.float8_e4m3fn},
        scale=ScaleFormat(
            storage_dtype=torch.float32, granularity="block", block_shape=(128, 128)
        ),
    ),
    traits={
        "n_align": frozenset({128}),
        "k_align": frozenset({128}),
        "block_scale_layout": frozenset({"canonical", "canonical_blackwell"}),
    },
    priority=Priority.REFERENCE,
)
def trtllm_cutedsl_mm_fp8_blockscale(
    A, B, A_scales, B_scales, out_dtype, *, alpha, block_size, **kwargs
):
    """Compute FP8 A @ B.T using canonical FP32 1x128 and 128x128 scales.

    Args:
        A: Contiguous E4M3 activation matrix [M,K].
        B: Contiguous E4M3 weight matrix [N,K].
        A_scales: Logical FP32 [M,K/128] activation scales.
        B_scales: FP32 [N/128,K/128] checkpoint scales.
        out_dtype: BF16 output type.
        alpha: Must be None; additional scaling is unsupported.
        block_size: Must be [128,128].
        kwargs: Optional preallocated contiguous BF16 output named out.
    Returns:
        BF16 result [M,N], using out when provided.
    """
    from tokenspeed_kernel.thirdparty.trtllm_blockwise import blockwise_gemm

    if (
        alpha is not None
        or tuple(block_size) != (128, 128)
        or out_dtype != torch.bfloat16
    ):
        raise ValueError(
            "TRT-LLM CuTe-DSL requires BF16 output, 128x128 blocks and alpha=None"
        )
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise ValueError("TRT-LLM CuTe-DSL requires E4M3 inputs")
    m, k = A.shape
    n, bk = B.shape
    if k != bk or n % 128 or k % 128 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("TRT-LLM CuTe-DSL requires contiguous aligned matrices")
    if (
        A_scales is None
        or B_scales is None
        or A_scales.dtype != torch.float32
        or B_scales.dtype != torch.float32
        or A_scales.shape != (m, k // 128)
        or B_scales.shape != (n // 128, k // 128)
        or any(t.device != A.device for t in (B, A_scales, B_scales))
    ):
        raise ValueError(
            "TRT-LLM CuTe-DSL requires canonical FP32 scales on the input device"
        )
    out = kwargs.get("out")
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=out_dtype)
    if (
        out.shape != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError("Invalid TRT-LLM CuTe-DSL output buffer")
    return blockwise_gemm(A, B, A_scales, B_scales, out)
