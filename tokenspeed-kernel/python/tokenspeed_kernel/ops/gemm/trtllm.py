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

"""TRT-LLM GEMM kernels exposed via the numerics registry.

- ``cublaslt_mm_nvfp4`` wraps the cuBLASLt NVFP4 GEMM runner (heuristic algo 0).
"""

from __future__ import annotations

import importlib.util

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signature,
    format_signatures,
    tensor_format,
)

_HAS_TRTLLM_KERNEL = (
    current_platform().is_nvidia
    and importlib.util.find_spec("trtllm_kernel") is not None
)
if _HAS_TRTLLM_KERNEL:
    # DeepEP must initialize before TRT-LLM's static CUDA runtime.
    import deep_ep  # noqa: F401
    import trtllm_kernel  # noqa: F401


# ===-----------------------------------------------------------------------===#
# DeepSeek V3 fused-A projection: TRT-LLM (NVIDIA)
# ===-----------------------------------------------------------------------===#


def dsv3_fused_a_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Run TRT-LLM's shape-specialized DeepSeek V3 fused-A projection."""
    if not _HAS_TRTLLM_KERNEL:
        raise RuntimeError("Fused-A GEMM requires a compatible trtllm_kernel package")
    return torch.ops.trtllm.dsv3_fused_a_gemm_op(mat_a, mat_b, None, None)


# ===-----------------------------------------------------------------------===#
# NVFP4: TRT-LLM cuBLASLt (NVIDIA SM100+)
# ===-----------------------------------------------------------------------===#


_fp4_dtypes: frozenset[torch.dtype] = frozenset({torch.uint8, torch.float4_e2m1fn_x2})
_NVFP4_SCALE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float32, torch.uint8, torch.float8_e4m3fn}
)
_NVFP4_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=a_scale_dtype, granularity="block", block_shape=(16,)
            ),
        ),
        b=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=b_scale_dtype, granularity="block", block_shape=(16,)
            ),
        ),
    )
    for storage_dtype in _fp4_dtypes
    for a_scale_dtype in _NVFP4_SCALE_DTYPES
    for b_scale_dtype in _NVFP4_SCALE_DTYPES
)

# One stateful torchbind instance per output dtype. Each holds its own
# per-shape algo cache inside C++.
_runner_cache: dict[torch.dtype, object] = {}
_CUBLASLT_HEURISTIC_ALGO = 0


def _get_runner(out_dtype: torch.dtype):
    runner = _runner_cache.get(out_dtype)
    if runner is None:
        runner = torch.classes.trtllm.CublasLtFP4GemmRunner(out_dtype)
        _runner_cache[out_dtype] = runner
    return runner


if _HAS_TRTLLM_KERNEL:

    @register_kernel(
        "gemm",
        "mm",
        name="cublaslt_mm_nvfp4",
        solution="cublaslt",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_NVFP4_FORMAT_SIGNATURES,
        traits={},
        priority=Priority.SPECIALIZED + 3,
    )
    def cublaslt_mm_nvfp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor,
        B_scales: torch.Tensor,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runner = _get_runner(out_dtype)
        output = runner.run_gemm(
            A,
            B.T,
            A_scales,
            B_scales.T,
            alpha,
            False,
            _CUBLASLT_HEURISTIC_ALGO,
        )
        if out is not None:
            out.copy_(output)
            return out
        return output


# ===-----------------------------------------------------------------------===#
# FP8 block-scale: TRT-LLM CuTe DSL (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


# TRT-LLM CuTe-DSL FP8 GEMM with unmodified FP32 block scales.

_BLOCK_FP8_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 9),
    vendors=frozenset({"nvidia"}),
)
_BLOCK_FP8_SIGNATURES = format_signatures(
    ("a", "b"),
    "mxfp8",
    {torch.float8_e4m3fn},
    scale=ScaleFormat(
        storage_dtype=torch.float32, granularity="block", block_shape=(128, 128)
    ),
)
_BLOCK_FP8_TRAITS = {
    "n_align": frozenset({128}),
    "k_align": frozenset({128}),
    "a_contiguous": frozenset({True}),
    "b_contiguous": frozenset({True}),
    "block_scale_layout": frozenset({"canonical", "canonical_blackwell"}),
    "has_alpha": frozenset({False}),
    "operands_aligned_16": frozenset({True}),
    "out_aligned_16": frozenset({True}),
    "out_contiguous": frozenset({True}),
    "out_dtype": frozenset({torch.bfloat16}),
    "scales_aligned_16": frozenset({True}),
}


# Both registrations use one implementation. Keep the low-priority entry for
# explicit solution selection below the automatic M=17 threshold.
@register_kernel(
    "gemm",
    "mm",
    name="trtllm_cutedsl_mm_fp8_blockscale_large_m",
    solution="trtllm_cutedsl",
    capability=_BLOCK_FP8_CAPABILITY,
    signatures=_BLOCK_FP8_SIGNATURES,
    traits={"m_min": frozenset({17}), **_BLOCK_FP8_TRAITS},
    priority=Priority.SPECIALIZED + 3,
)
@register_kernel(
    "gemm",
    "mm",
    name="trtllm_cutedsl_mm_fp8_blockscale",
    solution="trtllm_cutedsl",
    capability=_BLOCK_FP8_CAPABILITY,
    signatures=_BLOCK_FP8_SIGNATURES,
    traits=_BLOCK_FP8_TRAITS,
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
