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
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _routing_kernel(
    topk_ids_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_routes,
    BLOCK_ROUTES: tl.constexpr,
):
    expert_id = tl.program_id(0)
    count = 0
    num_blocks = tl.cdiv(num_routes, BLOCK_ROUTES)
    for block_id in range(num_blocks):
        route_ids = block_id * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
        route_mask = route_ids < num_routes
        selected_experts = tl.load(topk_ids_ptr + route_ids, mask=route_mask, other=-1)
        matches = route_mask & (selected_experts == expert_id)
        local_rank = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        tl.store(
            expert_route_ids_ptr + expert_id * num_routes + count + local_rank,
            route_ids,
            mask=matches,
        )
        count += tl.sum(matches.to(tl.int32), axis=0)
    tl.store(expert_counts_ptr + expert_id, count)


def _routing(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int32).contiguous()
    num_routes = topk_ids.numel()
    expert_route_ids = torch.empty(
        (num_experts, num_routes), device=topk_ids.device, dtype=torch.int32
    )
    expert_counts = torch.empty(num_experts, device=topk_ids.device, dtype=torch.int32)
    block_routes = 128 if num_routes <= 128 else 1024
    _routing_kernel[(num_experts,)](
        topk_ids,
        expert_route_ids,
        expert_counts,
        num_routes,
        BLOCK_ROUTES=block_routes,
        num_warps=4,
    )
    return expert_route_ids, expert_counts


@triton.jit
def _combine_kernel(
    route_output_ptr,
    topk_weights_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for slot in range(top_k):
        route_offsets = token_offsets * top_k + slot
        values = tl.load(
            route_output_ptr
            + route_offsets[:, None] * hidden_size
            + hidden_offsets[None, :],
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weights = tl.load(topk_weights_ptr + route_offsets, mask=token_mask, other=0.0)
        acc += values.to(tl.float32) * weights[:, None]
    tl.store(
        output_ptr + token_offsets[:, None] * hidden_size + hidden_offsets[None, :],
        acc,
        mask=token_mask[:, None] & hidden_mask[None, :],
    )


def _combine(
    route_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    num_tokens, top_k = topk_weights.shape
    hidden_size = output.shape[1]
    _combine_kernel[(triton.cdiv(num_tokens, 4), triton.cdiv(hidden_size, 256))](
        route_output,
        topk_weights.contiguous(),
        output,
        num_tokens,
        hidden_size,
        top_k=top_k,
        BLOCK_M=4,
        BLOCK_N=256,
        num_warps=4,
    )


def _prepare_routed_output(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_route_ids, expert_counts = _routing(topk_ids, num_experts)
    # Invalid expert ids are absent from routing; zero their canonical rows.
    route_output = torch.zeros(
        (topk_ids.numel(), x.shape[1]), device=x.device, dtype=x.dtype
    )
    return expert_route_ids, expert_counts, route_output, torch.empty_like(x)


def _num_programs(
    device: torch.device, route_count: int, size: int, block_n: int
) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return min(num_sms, route_count * triton.cdiv(size, block_n))


def _validate_launch(
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> None:
    if not do_finalize:
        raise ValueError("Triton MoE does not support deferred finalization")
    if int(getattr(w, "ep_size", 1)) != 1:
        raise ValueError("Triton MoE does not support expert parallelism")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Triton MoE requires precomputed topk weights and ids")


def _validate_topk(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> None:
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use torch.int32 or torch.int64")
    if not topk_weights.is_floating_point():
        raise TypeError("topk_weights must use a floating-point dtype")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and x must be on the same device")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k > 0]")


def _swiglu_params(w: torch.nn.Module) -> tuple[float, float | None, float]:
    swiglu_arg = getattr(w, "swiglu_arg", None)
    return (
        float(getattr(swiglu_arg, "alpha", 1.0) or 1.0),
        getattr(swiglu_arg, "limit", None),
        float(getattr(w, "swiglu_beta", 0.0) or 0.0),
    )


@triton.jit
def _swiglu(
    gate,
    up,
    alpha: tl.constexpr,
    limit: tl.constexpr,
    beta: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.clamp(up, -limit, limit)
    return gate * tl.sigmoid(alpha * gate) * (up + beta)
