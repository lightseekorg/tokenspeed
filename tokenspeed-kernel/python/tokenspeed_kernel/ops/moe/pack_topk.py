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

"""Pack precomputed top-k routes into dense router logits."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


def pack_topk_router_logits(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Encode selected routes as dense FP32 log-probabilities.

    Args:
        topk_weights: FP32 route weights with shape ``[num_tokens, top_k]``.
        topk_ids: Expert indices with shape ``[num_tokens, top_k]`` matching
            ``topk_weights``.
        num_experts: Total number of experts represented by the dense output.

    Returns:
        An FP32 tensor with shape ``[num_tokens, num_experts]`` containing the
        log weight for each selected expert and a large negative value for
        every unselected expert.

    Eligible NVIDIA inputs use one Triton launch, replacing the fill, clamp,
    cast, log and scatter chain emitted by the PyTorch expression. Other
    platforms retain the PyTorch reference implementation.
    """
    if topk_weights.ndim != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError("topk weights and ids must have the same 2D shape")
    if topk_weights.device != topk_ids.device:
        raise ValueError("topk weights and ids must be on the same device")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk weights must be float32")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk ids must be int32 or int64")
    if not topk_weights.is_contiguous() or not topk_ids.is_contiguous():
        raise ValueError("topk weights and ids must be contiguous")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    tokens = topk_ids.shape[0]
    if tokens == 0:
        return torch.empty(
            (0, num_experts), dtype=torch.float32, device=topk_weights.device
        )
    kernel = select_kernel(
        "moe",
        "pack_topk_router_logits",
        format_signature(topk_weights=dense_tensor_format(topk_weights.dtype)),
        solution=None if topk_weights.is_cuda else "torch",
    )
    return kernel(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=num_experts,
    )


@register_kernel(
    "moe",
    "pack_topk_router_logits",
    name="torch_pack_topk_router_logits",
    solution="torch",
    signatures=format_signatures("topk_weights", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
    tags={"portability", "reference"},
)
def torch_pack_topk_router_logits(
    *,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """PyTorch reference for packing selected routes as dense logits."""
    tokens = topk_weights.shape[0]
    output = torch.full(
        (tokens, num_experts),
        -1.0e20,
        dtype=torch.float32,
        device=topk_weights.device,
    )
    output.scatter_(
        1,
        topk_ids.long(),
        topk_weights.clamp_min(torch.finfo(torch.float32).tiny).log(),
    )
    return output


__all__ = ["pack_topk_router_logits", "torch_pack_topk_router_logits"]
