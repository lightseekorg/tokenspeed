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

"""Single-launch top-k route packing for NVIDIA GPUs."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


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


@register_kernel(
    "moe",
    "pack_topk_router_logits",
    name="triton_pack_topk_router_logits",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures("topk_weights", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
    tags={"nvidia", "cuda_graph", "latency"},
)
def triton_pack_topk_router_logits(
    *,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Pack selected routes into dense FP32 logits with one Triton launch."""
    tokens, top_k = topk_ids.shape
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


__all__ = ["triton_pack_topk_router_logits"]
