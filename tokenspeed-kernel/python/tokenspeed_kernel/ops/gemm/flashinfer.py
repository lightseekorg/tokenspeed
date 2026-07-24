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
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signature,
    format_signatures,
    tensor_format,
)

platform = current_platform()
_fp8_dtype = Platform.get().fp8e4m3fn.dtype

_fp4_dtypes: frozenset[torch.dtype] = frozenset({torch.uint8, torch.float4_e2m1fn_x2})
_MXFP8_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_NVFP4_SCALE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float32, torch.uint8, torch.float8_e4m3fn}
)
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_SCALE
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

# ---- FlashInfer block-scaled FP8 ----------------------------------------

gemm_fp8_nt_groupwise = error_fn
tinygemm_bf16 = error_fn

if platform.is_hopper_plus:
    try:
        from flashinfer.gemm import (
            gemm_fp8_nt_groupwise,
            tinygemm_bf16,
        )
    except ImportError:
        pass

if gemm_fp8_nt_groupwise is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_fp8_blockscale",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_FORMAT_SIGNATURES,
        traits={
            "n_align_128": frozenset({True}),
            "k_align_128": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 3,
        tags={"throughput"},
    )
    def flashinfer_mm_fp8_blockscale(
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
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for FP8 blockscale GEMM"
        orig_m = A.shape[0]
        scale_m = A_scales.shape[0]
        if orig_m % 4 != 0 or scale_m != orig_m:
            padded_m = max(((orig_m + 3) // 4) * 4, scale_m)
            A_padded = A.new_zeros((padded_m, A.shape[1]))
            A_padded[:orig_m] = A

            if scale_m != padded_m:
                A_scales_padded = A_scales.new_ones((padded_m, A_scales.shape[1]))
                A_scales_padded[:scale_m] = A_scales
                A_scales = A_scales_padded
            A = A_padded

        output = gemm_fp8_nt_groupwise(
            A,
            B,
            A_scales.t().contiguous(),
            B_scales.t().contiguous(),
            scale_major_mode="MN",
            out_dtype=out_dtype,
        )
        output = output[:orig_m] if output.shape[0] != orig_m else output
        if out is not None:
            out.copy_(output)
            return out
        return output


# ---- FlashInfer MXFP8 (1,32) ue8m0, cute-dsl backend ---------------------

mm_mxfp8 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    try:
        from flashinfer.gemm import mm_mxfp8
    except ImportError:
        pass

_MXFP8_UE8M0_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_FLOAT_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_1X32_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format("mxfp8", _fp8_dtype, scale=a_scale),
        b=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_UE8M0_1X32_SCALE),
    )
    for a_scale in (_MXFP8_FLOAT_1X32_SCALE, _MXFP8_UE8M0_1X32_SCALE)
)


def has_flashinfer_mxfp8() -> bool:
    """Whether the flashinfer cute-dsl MXFP8 (1,32) GEMM is usable here.

    Returns:
        True when running on an NVIDIA Blackwell (SM10x) GPU with a
        flashinfer build that provides ``mm_mxfp8``.
    """
    return mm_mxfp8 is not error_fn


if mm_mxfp8 is not error_fn:
    from tokenspeed_kernel.ops.gemm.fp8_utils import swizzle_mxfp8_scale

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_mxfp8",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_1X32_FORMAT_SIGNATURES,
        traits={
            "k_align_32": frozenset({True}),
            "n_min_128": frozenset({True}),
            "k_min_128": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_mxfp8(
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
        """MXFP8 (1,32)-block ue8m0 GEMM via flashinfer's cute-dsl backend.

        Args:
            A: ``[M, K]`` float8_e4m3fn activations.
            B: ``[N, K]`` (or ``[K, N]`` column-major) float8_e4m3fn weight.
            A_scales: uint8 e8m0 activation scales, either 1D in the
                F8_128x4 swizzled layout or ``[M, K // 32]`` row-major
                (re-swizzled per call; prefer pre-swizzled).
            B_scales: uint8 e8m0 weight scales, same layout options with
                ``[N, K // 32]`` row-major.
            out_dtype: Output dtype (bf16/fp16).
            alpha: Unused.
            block_size: Must be ``[1, 32]``.
            out: Optional output buffer.

        Returns:
            ``[M, N]`` tensor of ``out_dtype``.
        """
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for MXFP8 GEMM"
        assert block_size == [1, 32], f"expected block_size [1, 32], got {block_size}"
        k = A.shape[1]
        # B follows the dispatch convention of a [N, K] weight (row-major,
        # like the Triton kernel assumes); mm_mxfp8 wants the [K, N]
        # column-major view. Shape alone cannot disambiguate square weights,
        # so decide by memory layout.
        if B.shape[0] == k and B.stride(0) == 1:
            b = B
        else:
            b = B.t()
        n = b.shape[1]
        if k < 128 or k % 32 != 0 or n < 128:
            raise ValueError(
                f"flashinfer_mm_mxfp8 requires K >= 128, K % 32 == 0 and "
                f"N >= 128, got K={k}, N={n}"
            )
        if A_scales.dtype != torch.uint8 or B_scales.dtype != torch.uint8:
            raise ValueError(
                "flashinfer_mm_mxfp8 requires uint8 e8m0 scales, got "
                f"A_scales={A_scales.dtype}, B_scales={B_scales.dtype}"
            )
        if A_scales.dim() != 1:
            A_scales = swizzle_mxfp8_scale(A_scales.contiguous(), A.shape[0], k)
        if B_scales.dim() != 1:
            B_scales = swizzle_mxfp8_scale(B_scales.contiguous(), n, k)
        output = mm_mxfp8(
            A,
            b,
            A_scales,
            B_scales,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )
        if out is not None:
            out.copy_(output)
            return out
        return output


# ---- FlashInfer FP4 -----------------------------------------------------

mm_fp4 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    try:
        from flashinfer import mm_fp4
    except ImportError:
        pass

if mm_fp4 is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_NVFP4_FORMAT_SIGNATURES,
        traits={},
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_nvfp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        enable_pdl: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # backend="cutlass" (not "auto") to skip flashinfer's cuDNN-graph plan compile.
        output = mm_fp4(
            A,
            B,
            A_scales,
            B_scales,
            alpha,
            out_dtype,
            backend="cutlass",
            enable_pdl=enable_pdl,
        )
        if out is not None:
            out.copy_(output)
            return out
        return output
