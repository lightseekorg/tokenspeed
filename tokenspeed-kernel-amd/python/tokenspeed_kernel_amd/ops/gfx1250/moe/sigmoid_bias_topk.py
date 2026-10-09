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

"""GFX1250 biased-sigmoid top-k for expert counts outside the Kimi specialists."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna5 = gl.amd.cdna5

__all__ = [
    "gluon_sigmoid_bias_topk_route_gfx1250",
    "launch_gluon_sigmoid_bias_topk_route_gfx1250",
]

_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _route_launch_metadata(grid, kernel, args):
    """Report the logit and bias reads plus the id and weight writes."""
    tokens = grid[0]
    experts = args["E"]
    topk = args["TOPK"]
    return {
        "name": kernel.name,
        "bytes": tokens * experts * args["logits_ptr"].element_size()
        + experts * args["bias_ptr"].element_size()
        + tokens * topk * 8,
    }


def _next_pow2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


@gluon.jit(
    launch_metadata=_route_launch_metadata,
    do_not_specialize=["stride_lm"],
)
def gluon_sigmoid_bias_topk_route_gfx1250(
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
    NUM_WARPS: gl.constexpr,
):
    """Select ``TOPK`` experts for one token and write FP32 route weights.

    Selection scores are FP32 ``sigmoid(logit) + bias``. NaN scores are
    excluded, and equal scores take the lowest expert id. Weights are the
    FP32 sigmoid of the selected logits, optionally divided by their sum,
    then multiplied by ``ROUTED_SCALING_FACTOR``.
    """
    token = gl.program_id(0)
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    topk_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    expert = gl.arange(0, EP, layout=expert_layout)
    expert_mask = expert < E

    logits = cdna5.buffer_load(
        logits_ptr,
        (token * stride_lm + expert * stride_le).to(gl.int32),
        mask=expert_mask,
        other=-float("inf"),
    ).to(gl.float32)
    # Match torch.sigmoid, including large finite logits.
    abs_logits = gl.where(logits < 0, -logits, logits)
    exp_neg = gl.exp(-abs_logits)
    positive = logits >= 0
    scores = gl.where(
        positive, gl.fdiv(1.0, 1.0 + exp_neg), gl.fdiv(exp_neg, 1.0 + exp_neg)
    )
    bias = cdna5.buffer_load(
        bias_ptr,
        (expert * stride_be).to(gl.int32),
        mask=expert_mask,
        other=0.0,
    ).to(gl.float32)
    choice = scores + bias
    choice = gl.where(expert_mask & (choice == choice), choice, -float("inf"))

    topk_lane = gl.arange(0, TKP, layout=topk_layout)
    selected_ids = gl.zeros([TKP], gl.int32, topk_layout)
    selected_weights = gl.zeros([TKP], gl.float32, topk_layout)
    sentinel = gl.full([EP], E, gl.int32, expert_layout)
    for rank in gl.static_range(TOPK):
        maximum = gl.max(choice, axis=0)
        selected = gl.min(
            gl.where((choice == maximum) & expert_mask, expert, sentinel),
            axis=0,
        )
        weight = gl.sum(gl.where(expert == selected, scores, 0.0), axis=0)
        selected_ids = gl.where(topk_lane == rank, selected, selected_ids)
        selected_weights = gl.where(topk_lane == rank, weight, selected_weights)
        choice = gl.where(expert == selected, -float("inf"), choice)

    if NORMALIZE_TOPK_WEIGHTS:
        denominator = gl.sum(selected_weights, axis=0)
        denominator = gl.where(denominator != 0.0, denominator, 1.0)
        selected_weights = selected_weights / denominator
    selected_weights = selected_weights * ROUTED_SCALING_FACTOR

    topk_mask = topk_lane < TOPK
    cdna5.buffer_store(
        selected_ids,
        topk_ids_ptr,
        (token * stride_tim + topk_lane * stride_tik).to(gl.int32),
        mask=topk_mask,
    )
    cdna5.buffer_store(
        selected_weights,
        topk_weights_ptr,
        (token * stride_twm + topk_lane * stride_twk).to(gl.int32),
        mask=topk_mask,
    )


def launch_gluon_sigmoid_bias_topk_route_gfx1250(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse sigmoid scoring, biased top-k selection, and route-weight scaling.

    Args:
        router_logits: Router logits shaped ``[tokens, experts]`` with a
            contiguous expert dimension.
        correction_bias: Selection-only bias shaped ``[experts]``.
        topk: Number of selected experts, at most 16.
        routed_scaling_factor: Scale applied to selected route weights.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        weights_dtype: Output dtype for the route weights. Selection and
            normalization run in FP32.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, topk]`` with
        ``weights_dtype`` weights and INT32 ids.
    """
    if (
        router_logits.ndim != 2
        or not router_logits.is_cuda
        or router_logits.dtype not in _ROUTE_DTYPES
        or router_logits.stride(1) != 1
    ):
        raise ValueError(
            "gfx1250 sigmoid-bias top-k requires 2D CUDA FP16/BF16/FP32 logits "
            "with a contiguous expert dimension"
        )
    tokens, experts = router_logits.shape
    if experts > 1024 or not 0 < topk <= min(experts, 16):
        raise ValueError(
            "gfx1250 sigmoid-bias top-k supports at most 1024 experts and "
            f"topk in [1, min(experts, 16)], got experts={experts}, topk={topk}"
        )
    if (
        correction_bias.shape != (experts,)
        or correction_bias.device != router_logits.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError(
            "gfx1250 sigmoid-bias top-k requires a contiguous colocated "
            f"correction bias shaped [{experts}]"
        )

    ids = torch.empty((tokens, topk), dtype=torch.int32, device=router_logits.device)
    weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=router_logits.device
    )
    if tokens == 0:
        return weights.to(weights_dtype), ids

    # Four warps cover the 1024-expert reduction.
    num_warps = 4
    gluon_sigmoid_bias_topk_route_gfx1250[(tokens,)](
        router_logits,
        correction_bias,
        ids,
        weights,
        router_logits.stride(0),
        router_logits.stride(1),
        correction_bias.stride(0),
        ids.stride(0),
        ids.stride(1),
        weights.stride(0),
        weights.stride(1),
        E=experts,
        TOPK=topk,
        EP=_next_pow2(experts),
        TKP=_next_pow2(topk),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return weights.to(weights_dtype), ids
