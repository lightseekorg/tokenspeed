# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Registration shim for AITER-derived GEMM kernels."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format
from tokenspeed_kernel.thirdparty.triton.aiter_fp8_gemm import (
    aiter_preshuffled_fp8_gemm,
    preshuffle_fp8_weight,
)

__all__ = ["preshuffle_fp8_weight"]


@register_kernel(
    "gemm",
    "mm",
    name="triton_aiter_mm_fp8_blockscale_preshuffle_gfx950",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                a=tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32,
                        granularity="block",
                        block_shape=(128, 128),
                    ),
                ),
                b=tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32,
                        granularity="block",
                        block_shape=(128, 128),
                    ),
                ),
            )
        }
    ),
    # The weight has a non-canonical physical layout that generic GEMM traits
    # cannot express. Runtime layers opt in after load-time preshuffling.
    priority=Priority.REFERENCE,
    traits={},
)
def triton_aiter_mm_fp8_blockscale_preshuffle_gfx950(
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
    if A_scales is None or B_scales is None:
        raise ValueError("preshuffled FP8 GEMM requires activation and weight scales")
    if block_size != [128, 128]:
        raise ValueError(f"preshuffled FP8 GEMM requires [128, 128], got {block_size}")
    if alpha is not None:
        raise ValueError("preshuffled FP8 GEMM does not accept alpha")
    return aiter_preshuffled_fp8_gemm(A, B, A_scales, B_scales, out_dtype, out=out)
