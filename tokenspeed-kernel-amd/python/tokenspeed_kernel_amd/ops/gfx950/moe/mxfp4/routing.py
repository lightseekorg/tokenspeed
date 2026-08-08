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


"""Fused routing exports used by gfx950 A4W4 decode and package prefill."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_kernels import (
    gluon_topk_route_supported,
    invoke_sigmoid_bias_topk_route_gluon,
    invoke_softmax_topk_route_gluon,
)

cdna4 = gl.amd.cdna4
_PREFILL_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_PREFILL_ROUTE_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


@gluon.jit
def _sigmoid_bias_topk_route_prefill_kernel(
    logits_ptr,
    bias_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    stride_lm,
    stride_le,
    stride_be,
    stride_tim,
    stride_tik,
    stride_twm,
    stride_twk,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    X_DTYPE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    topk_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    expert = gl.arange(0, EP, layout=expert_layout)
    expert_mask = expert < E

    logits = cdna4.buffer_load(
        logits_ptr,
        (token * stride_lm + expert * stride_le).to(gl.int32),
        mask=expert_mask,
        other=-float("inf"),
    ).to(gl.float32)
    scores = gl.fdiv(1.0, 1.0 + gl.exp(-logits)).to(X_DTYPE)
    bias = cdna4.buffer_load(
        bias_ptr,
        (expert * stride_be).to(gl.int32),
        mask=expert_mask,
        other=0.0,
    ).to(gl.float32)
    choice = gl.where(expert_mask, scores.to(gl.float32) + bias, -float("inf"))

    topk_lane = gl.arange(0, TKP, layout=topk_layout)
    selected_ids = gl.zeros([TKP], gl.int32, topk_layout)
    selected_weights = gl.zeros([TKP], gl.float32, topk_layout)
    sentinel = gl.full([EP], E, gl.int32, expert_layout)
    for rank in gl.static_range(TOPK):
        maximum = gl.max(choice, axis=0)
        selected = gl.min(
            gl.where((choice == maximum) & expert_mask, expert, sentinel), axis=0
        )
        weight = gl.sum(gl.where(expert == selected, scores, 0.0), axis=0)
        selected_ids = gl.where(topk_lane == rank, selected, selected_ids)
        selected_weights = gl.where(topk_lane == rank, weight, selected_weights)
        choice = gl.where(expert == selected, -float("inf"), choice)

    if NORMALIZE_TOPK_WEIGHTS:
        selected_weights = selected_weights.to(X_DTYPE)
        denominator = gl.sum(selected_weights, axis=0)
        denominator = gl.where(denominator != 0.0, denominator, 1.0)
        selected_weights = selected_weights.to(gl.float32) * (
            ROUTED_SCALING_FACTOR / denominator
        )

    topk_mask = topk_lane < TOPK
    cdna4.buffer_store(
        selected_ids,
        topk_ids_ptr,
        (token * stride_tim + topk_lane * stride_tik).to(gl.int32),
        mask=topk_mask,
    )
    cdna4.buffer_store(
        selected_weights,
        topk_weights_ptr,
        (token * stride_twm + topk_lane * stride_twk).to(gl.int32),
        mask=topk_mask,
    )


def invoke_sigmoid_bias_topk_route_prefill_gluon(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Large-M Kimi noaux_tc routing with one independent CTA per token."""
    if (
        router_logits.ndim != 2
        or router_logits.dtype not in _PREFILL_ROUTE_DTYPES
        or not router_logits.is_cuda
        or not 0 < topk <= 16
        or router_logits.shape[1] > 1024
    ):
        raise ValueError("unsupported gfx950 prefill sigmoid-route shape")
    if correction_bias.shape != (router_logits.shape[1],):
        raise ValueError("correction_bias must have shape [num_experts]")

    if not router_logits.is_contiguous():
        router_logits = router_logits.contiguous()
    if not correction_bias.is_contiguous():
        correction_bias = correction_bias.contiguous()
    tokens, experts = router_logits.shape
    topk_ids = torch.empty(
        (tokens, topk), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=router_logits.device
    )
    num_warps = 1
    _sigmoid_bias_topk_route_prefill_kernel[(tokens,)](
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
        router_logits.stride(0),
        router_logits.stride(1),
        correction_bias.stride(0),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        E=experts,
        TOPK=topk,
        EP=_next_pow2(experts),
        TKP=_next_pow2(topk),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        X_DTYPE=_PREFILL_ROUTE_GL_DTYPE[router_logits.dtype],
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return topk_ids, topk_weights


__all__ = [
    "gluon_topk_route_supported",
    "invoke_sigmoid_bias_topk_route_gluon",
    "invoke_sigmoid_bias_topk_route_prefill_gluon",
    "invoke_softmax_topk_route_gluon",
]
