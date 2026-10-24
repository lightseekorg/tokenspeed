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

"""Dense router projection for LongCat 2.0.

The canonical BF16 shape is ``[M, 8192] x [896, 8192].T -> [M, 896]`` with
FP32 accumulation and output.  Measured on B300 with Torch 2.13/CUDA 13.0
(microseconds per call)::

    M                   8     32     64     80     96    112    128
    best CuTe warm    5.09   7.42  12.40  13.01  15.68  18.36  21.02
    cublasLt warm    15.16  15.08  15.82  14.99  15.07  15.07  15.17
    best CuTe cold   17.12  17.41  21.28  22.78  24.96  27.07  29.34
    cublasLt cold    27.84  27.97  29.18  27.97  28.38  28.35  28.32

Each layer owns a distinct 14 MiB router weight, so the production policy uses
the cold-L2 cutover: CuTe dot-product through M=4, CuTe split-K Tensor Core
through M=112, and cublasLt above it.  Other LongCat shapes retain the dsv3
router fallback.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel.ops.gemm.cuda import dsv3_router_gemm as _dsv3_router_gemm
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import LLBf16Router

LONGCAT_HIDDEN_SIZE = 8192
LONGCAT_ROUTER_SIZE = 896


# B300, BF16 x BF16 -> FP32, [M, 8192] x [896, 8192].T.  The split-K
# configurations are the winners at the upper end of each M range; above 112,
# cublasLt overtakes the cold-L2 split-K path.  Keep this policy separate from
# K3's K=7168 measurements, whose split-K cutover is M=32.
class _LongcatBf16Router(LLBf16Router):
    max_m = 112
    max_m_dotprod = 4

    @staticmethod
    def block_size_for(m: int) -> int:
        return 256 if m == 2 else 128

    @staticmethod
    def splitk_config_for(m: int) -> tuple[int, int]:
        if m <= 16:
            return 4, 4
        if m <= 32:
            return 2, 4
        if m <= 64:
            return 4, 2
        return 2, 4


_longcat_ll_bf16_router = _LongcatBf16Router()


@lru_cache(maxsize=1)
def _mm_out_dtype_supported() -> bool:
    try:
        a = torch.empty(1, 2, dtype=torch.bfloat16, device="cuda")
        return torch.mm(a, a.t(), out_dtype=torch.float32).dtype == torch.float32
    except (TypeError, RuntimeError):
        return False


def longcat_router_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Compute LongCat router logits with an FP32 result.

    Args:
        hidden_states: Activation tensor shaped ``[M, K]``.
        weight: Router weight tensor shaped ``[N, K]``.
        out: Optional FP32 destination shaped ``[M, N]``.
        solution: ``"auto"`` selects the measured CuTe or cublasLt path for
            LongCat 2.0 and falls back to Torch on unsupported inputs.
        enable_pdl: Enable PDL in compatible kernel implementations.

    Returns:
        FP32 router logits shaped ``[M, N]``.
    """
    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("LongCat router projection expects two rank-2 tensors")
    m, input_width = hidden_states.shape
    output_width, weight_width = weight.shape
    if input_width != weight_width:
        raise ValueError(
            "LongCat router projection input width does not match weight width"
        )
    if out is not None and (
        out.shape != (m, output_width)
        or out.dtype is not torch.float32
        or out.device != hidden_states.device
    ):
        raise ValueError(f"out must be a [{m}, {output_width}] float32 tensor")
    if solution not in {"auto", "ll_bf16", "cublas", "dsv3", "torch"}:
        raise ValueError(f"unknown LongCat router solution {solution!r}")

    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and m > 0
        and input_width == LONGCAT_HIDDEN_SIZE
        and output_width == LONGCAT_ROUTER_SIZE
    )
    dsv3_usable = (
        hidden_states.is_cuda
        and hidden_states.dtype is torch.bfloat16
        and weight.dtype in (torch.bfloat16, torch.float32)
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and hidden_states.device == weight.device
        and m > 0
    )
    if solution == "auto":
        platform = Platform.get()
        if (
            platform.is_hopper_plus
            and specialized
            and _longcat_ll_bf16_router.supports(hidden_states, weight, m)
        ):
            solution = "ll_bf16"
        elif platform.is_hopper_plus and specialized and _mm_out_dtype_supported():
            solution = "cublas"
        elif platform.is_hopper_plus and dsv3_usable:
            solution = "dsv3"
        else:
            solution = "torch"

    if solution == "ll_bf16":
        if not specialized or not _longcat_ll_bf16_router.supports(
            hidden_states, weight, m
        ):
            raise ValueError(
                "LongCat ll_bf16 requires contiguous CUDA BF16 "
                "[M, 8192] input and [896, 8192] weight with 1 <= M <= 112"
            )
        return _longcat_ll_bf16_router(
            hidden_states, weight, out, enable_pdl=enable_pdl
        )

    if solution == "cublas":
        if not specialized or not _mm_out_dtype_supported():
            raise ValueError(
                "LongCat cublas requires contiguous CUDA BF16 [M, 8192] "
                "input, [896, 8192] weight, and torch.mm out_dtype support"
            )
        if out is None:
            return torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        return torch.mm(
            hidden_states,
            weight.t(),
            out=out,
            out_dtype=torch.float32,
        )

    if solution == "dsv3":
        if not dsv3_usable:
            raise ValueError(
                "LongCat dsv3 requires contiguous CUDA BF16 activations and "
                "contiguous BF16 or FP32 weights"
            )
        logits = _dsv3_router_gemm(
            hidden_states,
            weight,
            out_dtype=torch.float32,
            enable_pdl=enable_pdl,
        )
        if out is None:
            return logits
        out.copy_(logits)
        return out

    logits = torch.nn.functional.linear(hidden_states.float(), weight.float())
    if out is None:
        return logits
    out.copy_(logits)
    return out


__all__ = [
    "LONGCAT_HIDDEN_SIZE",
    "LONGCAT_ROUTER_SIZE",
    "longcat_router_projection",
]
