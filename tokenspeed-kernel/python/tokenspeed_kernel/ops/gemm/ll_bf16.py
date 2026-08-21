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


__all__ = ["MAX_M", "cute_dsl_ll_bf16_router", "ll_bf16_router_supported"]
