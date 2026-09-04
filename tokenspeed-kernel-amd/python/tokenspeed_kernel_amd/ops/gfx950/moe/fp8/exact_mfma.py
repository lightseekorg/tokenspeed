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

"""Exact BF16-MFMA execution from compact block-scaled FP8 expert weights."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
    moe_align_block_size_device,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage1_kernel import invoke_stage1
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import invoke_stage2


def _select_route_block_size(num_tokens: int, topk: int, num_experts: int) -> int:
    """Choose the routed-row tile from the average rows per expert."""
    routed_rows = num_tokens * topk
    if routed_rows < 2 * num_experts:
        return 16
    if routed_rows < 8 * num_experts:
        return 32
    return 64


def gluon_fp8_block_exact_mfma_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_start: int = 0,
    expert_parallel: bool = False,
) -> torch.Tensor:
    """Apply block-FP8 experts with BF16-rounded weights and BF16 MFMA.

    Compact weights are dequantized into LDS, preserving the arithmetic of
    materialized BF16 experts while avoiding their extra HBM traffic.

    Args:
        hidden_states: BF16 token states ``[tokens, hidden]``.
        w13: Block-scaled FP8 gate/up expert weights.
        w2: Block-scaled FP8 down expert weights.
        w13_scale: FP32 inverse scales for 128x128 blocks of ``w13``.
        w2_scale: FP32 inverse scales for 128x128 blocks of ``w2``.
        topk_ids: Routed expert IDs ``[tokens, top_k]``.
        topk_weights: Route weights ``[tokens, top_k]``.
        expert_start: First global expert ID owned by this rank.
        expert_parallel: Whether routes span experts on multiple ranks.

    Returns:
        Finalized BF16 states ``[tokens, hidden]``.
    """
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert hidden_states.dtype == torch.bfloat16
    assert w13.dtype in fp8_dtypes and w2.dtype in fp8_dtypes
    assert w13_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32
    num_tokens, hidden_size = hidden_states.shape
    num_experts, twice_intermediate, weight_hidden = w13.shape
    intermediate_size = twice_intermediate // 2
    assert weight_hidden == hidden_size
    assert w2.shape == (num_experts, hidden_size, intermediate_size)
    assert w13_scale.shape == (
        num_experts,
        twice_intermediate // 128,
        hidden_size // 128,
    )
    assert w2_scale.shape == (
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
    )
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    block_m = (
        64
        if expert_parallel
        else _select_route_block_size(num_tokens, topk_ids.shape[1], num_experts)
    )
    sorted_ids, sorted_experts, sorted_weights, num_valid = moe_align_block_size_device(
        topk_ids,
        topk_weights,
        num_experts,
        block_m,
        expert_start=expert_start,
    )
    intermediate = torch.empty(
        (num_tokens * topk_ids.shape[1], intermediate_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    invoke_stage1(
        hidden_states,
        w13,
        sorted_ids,
        sorted_experts,
        num_valid,
        intermediate,
        topk_ids.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=64 if block_m == 16 else 128,
        num_xcds=1 if block_m == 16 else 8,
        group_size_m=4 if block_m == 16 else 8,
        split_k=1,
        w1_scale=w13_scale,
    )
    output_factory = torch.zeros if expert_parallel else torch.empty
    output = output_factory(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    invoke_stage2(
        intermediate,
        w2,
        sorted_ids,
        sorted_experts,
        sorted_weights,
        num_valid,
        output,
        topk_ids.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=64 if block_m == 16 else 128,
        BLOCK_K=128 if block_m == 16 else 64,
        expert_parallel=expert_parallel,
        w2_scale=w2_scale,
    )
    return output
