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

platform = current_platform()
_fp8_dtype = Platform.get().fp8e4m3fn.dtype

_fp4_dtypes: frozenset[torch.dtype] = frozenset({torch.uint8, torch.float4_e2m1fn_x2})

# ---- FlashInfer block-scaled FP8 ----------------------------------------

gemm_fp8_nt_groupwise = error_fn
fp8_blockscale_gemm_sm90 = error_fn
tinygemm_bf16 = error_fn

if platform.is_hopper_plus:
    try:
        from flashinfer.gemm import (
            fp8_blockscale_gemm_sm90,
            gemm_fp8_nt_groupwise,
            tinygemm_bf16,
        )
    except ImportError:
        pass


if fp8_blockscale_gemm_sm90 is not error_fn and platform.is_hopper:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_sm90_mm_fp8_blockscale",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={_fp8_dtype},
        traits={
            "quant": frozenset({"mxfp8"}),
            "n_align_64": frozenset({True}),
            "k_align_128": frozenset({True}),
        },
        # Keep throughput-oriented kernels ahead under the default objective.
        # FP8 Linear call sites opt into the latency objective after loading
        # stable K/N shapes, which lets selection choose this SM90 path without
        # a runtime token-count gate.
        priority=4,
        tags={"latency", "non_elementwise_numerics"},
    )
    def flashinfer_sm90_mm_fp8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
    ) -> torch.Tensor:
        assert A_scales is None, "FlashInfer sm90 GEMM quantizes BF16 activations"
        assert B_scales is not None, "B_scales is required for FP8 blockscale GEMM"
        return fp8_blockscale_gemm_sm90(
            A,
            B,
            weight_scale=B_scales,
            out_dtype=out_dtype,
        )


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
        dtypes={_fp8_dtype},
        traits={
            "quant": frozenset({"mxfp8"}),
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
        return output[:orig_m] if output.shape[0] != orig_m else output


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
        dtypes=_fp4_dtypes,
        traits={
            "quant": frozenset({"nvfp4"}),
        },
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
    ) -> torch.Tensor:
        # backend="cutlass" (not "auto") to skip flashinfer's cuDNN-graph plan compile.
        return mm_fp4(
            A,
            B,
            A_scales,
            B_scales,
            alpha,
            out_dtype,
            backend="cutlass",
            enable_pdl=enable_pdl,
        )
