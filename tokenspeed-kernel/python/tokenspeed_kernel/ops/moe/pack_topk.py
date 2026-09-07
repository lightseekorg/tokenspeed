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

"""Pack precomputed top-k routes into dense router logits in one launch."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _pack_topk_router_logits_kernel(
    topk_weights,
    topk_ids,
    output,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    block_experts: tl.constexpr,
):
    token = tl.program_id(0)
    expert_offsets = tl.arange(0, block_experts)
    logits = tl.full((block_experts,), -1.0e20, tl.float32)
    route_base = token * top_k
    for route in tl.static_range(0, top_k):
        expert = tl.load(topk_ids + route_base + route)
        weight = tl.load(topk_weights + route_base + route).to(tl.float32)
        log_weight = tl.log(tl.maximum(weight, 1.1754943508222875e-38))
        logits = tl.where(expert_offsets == expert, log_weight, logits)
    tl.store(
        output + token * num_experts + expert_offsets,
        logits,
        mask=expert_offsets < num_experts,
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

    CUDA inputs use one Triton launch, replacing the fill, clamp, cast, log and
    scatter chain emitted by the PyTorch expression. CPU remains a reference
    fallback for configuration tests.
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

    tokens, top_k = topk_ids.shape
    if tokens == 0:
        return torch.empty(
            (0, num_experts), dtype=torch.float32, device=topk_weights.device
        )
    if not topk_weights.is_cuda:
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

    block_experts = triton.next_power_of_2(num_experts)
    output = torch.empty(
        (tokens, num_experts), dtype=torch.float32, device=topk_weights.device
    )
    _pack_topk_router_logits_kernel[(tokens,)](
        topk_weights,
        topk_ids,
        output,
        num_experts=num_experts,
        top_k=top_k,
        block_experts=block_experts,
        num_warps=4,
    )
    return output


__all__ = ["pack_topk_router_logits"]
