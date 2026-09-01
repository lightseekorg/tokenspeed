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

"""Registry entry for the vendored CuTe low-latency BF16 router GEMM."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm.flashinfer import (
    flashinfer_cute_dsl_mm_bf16,
    has_flashinfer_cute_dsl_bf16,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import MAX_M, ll_bf16_router

_BF16_IN_FP32_OUT = {
    format_signature(
        hidden_states=dense_tensor_format(torch.bfloat16),
        weight=dense_tensor_format(torch.bfloat16),
    )
}


@register_kernel(
    "gemm",
    "router_projection",
    name="cute_dsl_ll_bf16_router",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        # Split-K reduces through DSMEM inside a thread block cluster.
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=_BF16_IN_FP32_OUT,
    priority=Priority.SPECIALIZED,
    traits={"out_dtype": frozenset({"float32"})},
    tags={"nvidia", "cute_dsl", "decode", "moe"},
)
def cute_dsl_ll_bf16_router(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Router logits in FP32 from BF16 operands, for decode-sized M.

    Args:
        hidden_states: ``[M, K]`` contiguous BF16 activation.
        weight: ``[N, K]`` contiguous BF16 router weight.
        out: Optional ``[M, N]`` FP32 destination; allocated when omitted.

    Returns:
        ``[M, N]`` FP32 router logits, ``out`` when it was given.
    """
    return ll_bf16_router(hidden_states, weight, out)


def ll_bf16_router_supported(
    hidden_states: torch.Tensor, weight: torch.Tensor, m: int
) -> bool:
    """Whether the vendored driver can serve this call.

    Args:
        hidden_states: ``[M, K]`` activation; weight: ``[N, K]`` weight.
        m: Token count, which selects the dot-product or split-K backend.

    Returns:
        True when a vendored kernel is compilable and applicable here.
    """
    return ll_bf16_router.supports(hidden_states, weight, m)


# The kernels issue 32-byte vector loads from the base of each operand.
_PTR_ALIGN = 32
# Both backends step K in multiples of 128.
_K_ALIGN = 128


def ll_bf16_mm_supported(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """Whether :func:`ll_bf16_mm` can serve these operands.

    Args:
        x: ``[..., K]`` BF16 activation; its leading dims flatten to ``M``.
        weight: ``[N, K]`` BF16 weight.
        bias: Optional ``[N]`` BF16 bias.

    Returns:
        True when every guard holds, so the caller can fall back otherwise.
    """
    if weight.ndim != 2 or x.ndim < 1:
        return False
    n, k = weight.shape
    # Checked on the originals: a reshape of a non-contiguous x would copy,
    # voiding the pointer-alignment check below.
    if x.shape[-1] != k or k % _K_ALIGN:
        return False
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if not x.is_contiguous() or not weight.is_contiguous():
        return False
    if x.device != weight.device:
        return False
    if x.data_ptr() % _PTR_ALIGN or weight.data_ptr() % _PTR_ALIGN:
        return False
    m = x.numel() // k
    if not 1 <= m <= MAX_M:
        return False
    if bias is not None and (
        bias.ndim != 1
        or bias.shape[0] != n
        or bias.dtype is not torch.bfloat16
        or not bias.is_contiguous()
        or bias.device != x.device
    ):
        return False
    # Only the vendored path carries further requirements -- its own toolchain,
    # and a cluster for the split-K reduce above MAX_M_DOTPROD.
    return has_flashinfer_cute_dsl_bf16() or ll_bf16_router.supports(
        x.view(m, k), weight, m
    )


def ll_bf16_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` in BF16, the dense-linear form of the router GEMM.

    Args:
        x: ``[..., K]`` contiguous BF16 activation; leading dims flatten to
            ``M``, which must not exceed :data:`MAX_M`.
        weight: ``[N, K]`` contiguous BF16 weight.
        bias: Optional contiguous ``[N]`` BF16 bias.
        out: Optional ``[..., N]`` BF16 destination; allocated when omitted.

    Returns:
        ``[..., N]`` BF16 tensor carrying ``x``'s leading dims, ``out`` when it
        was given.
    """
    k = weight.shape[1]
    m = x.numel() // k
    n = weight.shape[0]
    # view, not reshape: a reshaped non-contiguous out would take the writes.
    flat_out = None if out is None else out.view(m, n)
    if has_flashinfer_cute_dsl_bf16():
        result = flashinfer_cute_dsl_mm_bf16(x.view(m, k), weight, bias, flat_out)
    else:
        result = ll_bf16_router(
            x.view(m, k),
            weight,
            flat_out,
            bias=bias,
            out_dtype=torch.bfloat16,
        )
    return result.view(*x.shape[:-1], n)


__all__ = [
    "MAX_M",
    "cute_dsl_ll_bf16_router",
    "ll_bf16_mm",
    "ll_bf16_mm_supported",
    "ll_bf16_router_supported",
]
