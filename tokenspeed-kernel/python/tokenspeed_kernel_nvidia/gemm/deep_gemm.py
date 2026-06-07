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
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, Platform
from tokenspeed_kernel.signature import ScaleFormat, format_signatures
from tokenspeed_kernel_nvidia.registration import Priority, register_kernel

_fp8_dtype = Platform.get().fp8e4m3fn.dtype
_MXFP8_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_SCALE
)

try:
    from tokenspeed_kernel_nvidia.thirdparty.deep_gemm import (
        fp8_gemm_nt,
        get_mn_major_tma_aligned_tensor,
        get_num_sms,
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nt_masked,
        set_num_sms,
    )
except ImportError:
    fp8_gemm_nt = None  # type: ignore[assignment]
    get_mn_major_tma_aligned_tensor = None  # type: ignore[assignment]
    get_num_sms = None  # type: ignore[assignment]
    m_grouped_fp8_gemm_nt_contiguous = None  # type: ignore[assignment]
    m_grouped_fp8_gemm_nt_masked = None  # type: ignore[assignment]
    set_num_sms = None  # type: ignore[assignment]

if fp8_gemm_nt is not None:

    @register_kernel(
        "gemm",
        "mm",
        name="deep_gemm_mm_fp8_blockscale",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_FORMAT_SIGNATURES,
        traits={
            "n_align_64": frozenset({True}),
            "k_align_128": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
        tags={"throughput"},
    )
    def deep_gemm_mm_fp8_blockscale(
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
        if A_scales.dtype == torch.float32:
            A_scales = get_mn_major_tma_aligned_tensor(A_scales)
        N = B.shape[0]
        C = A.new_empty(A.shape[0], N, dtype=torch.bfloat16)
        fp8_gemm_nt((A, A_scales), (B, B_scales), C)
        return C.to(out_dtype)
