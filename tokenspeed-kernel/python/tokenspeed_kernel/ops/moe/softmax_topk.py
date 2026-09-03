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

"""Softmax top-k routing entry point."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import Platform, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

_TRITON_MAX_EXPERTS = 1024
_TRITON_MAX_TOPK = 32
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _triton_eligible(router_logits: torch.Tensor, topk: int) -> bool:
    return (
        router_logits.is_cuda
        and router_logits.dtype in _SUPPORTED_DTYPES
        and router_logits.shape[0] > 0
        and router_logits.stride(1) == 1
        and router_logits.shape[1] <= _TRITON_MAX_EXPERTS
        and topk <= _TRITON_MAX_TOPK
        and Platform.get().is_nvidia
    )


def moe_softmax_topk(
    router_logits: torch.Tensor,
    topk: int,
    *,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using softmax routing in one fused launch when supported.

    Args:
        router_logits: Router logits shaped ``[tokens, experts]``. CUDA FP16,
            BF16, and FP32 inputs with a contiguous expert dimension use the
            fused NVIDIA Triton kernel; other inputs use the PyTorch fallback.
        topk: Number of experts selected per token.
        renormalize: Normalize the selected weights to sum to one. When false,
            return probabilities from the softmax over all experts.
        routed_scaling_factor: Scale applied to every selected route weight.
        solution: Optional implementation override, such as ``"triton"`` or
            ``"torch"``.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, topk]``. Weights are
        FP32 and ids are INT64, matching ``torch.topk`` and DeepEP's routing
        metadata contract.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    tokens, experts = router_logits.shape
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must have a floating-point dtype")
    if not 0 < topk <= experts:
        raise ValueError(f"topk must be in [1, {experts}], got {topk}")
    if tokens == 0:
        shape = (0, topk)
        return (
            torch.empty(shape, device=router_logits.device, dtype=torch.float32),
            torch.empty(shape, device=router_logits.device, dtype=torch.int64),
        )

    if solution is None and not _triton_eligible(router_logits, topk):
        solution = "torch"
    enable_pdl = pdl_enabled()
    if solution == "torch":
        return torch_softmax_topk(
            router_logits=router_logits,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            enable_pdl=enable_pdl,
        )

    kernel = select_kernel(
        "moe",
        "softmax_topk",
        format_signature(
            router_logits=dense_tensor_format(router_logits.dtype),
        ),
        solution=solution,
    )
    return kernel(
        router_logits=router_logits,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "moe",
    "softmax_topk",
    name="torch_softmax_topk",
    solution="torch",
    signatures=format_signatures("router_logits", "dense", set(_SUPPORTED_DTYPES)),
    priority=Priority.PORTABLE,
    tags={"portability", "reference"},
)
def torch_softmax_topk(
    *,
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for ordinary softmax top-k routing."""
    del enable_pdl
    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


__all__ = ["moe_softmax_topk", "torch_softmax_topk"]
