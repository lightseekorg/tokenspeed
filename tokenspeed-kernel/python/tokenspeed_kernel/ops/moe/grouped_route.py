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

"""Beam's grouped sigmoid route in PyTorch.

The FP32 sigmoid of the router logits plus the correction bias, the sum of the
top ``group_topk`` scores in each of ``num_groups`` groups, the top
``topk_groups`` groups, the top ``topk`` experts of the kept groups, their
unbiased scores renormalized (sum plus ``1e-20``) and multiplied by the routed
scale. ``triton_router_route_lanepar``
(:mod:`tokenspeed_kernel.ops.moe._triton.router_select`) computes
:func:`torch_grouped_route` in one launch, byte for byte, for Beam's routing
shape; Beam runs it on NVIDIA SM100 to SM103 and these statements elsewhere.
"""

from __future__ import annotations

import torch

__all__ = ["torch_grouped_route", "torch_route_epilogue", "torch_router_select"]


def torch_router_select(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_groups: int,
    group_topk: int,
    topk_groups: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The route's scoring and grouped selection.

    Returns:
        ``(scores, group_ids, ids)``: the FP32 sigmoid ``[rows, experts]``, the
        kept groups ``[rows, topk_groups]`` and the selected experts
        ``[rows, topk]`` (int64, in descending order).
    """
    scores = gating_output.float().sigmoid()
    choices = scores + correction_bias
    rows, experts = choices.shape
    grouped = choices.reshape(rows, num_groups, experts // num_groups)
    group_scores = grouped.topk(group_topk, dim=-1).values.sum(dim=-1)
    group_ids = group_scores.topk(topk_groups, dim=-1, sorted=True).indices
    keep_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    keep_groups.scatter_(1, group_ids, True)
    keep = keep_groups.unsqueeze(-1).expand_as(grouped).reshape_as(choices)
    masked_choices = choices.masked_fill(~keep, float("-inf"))
    ids = masked_choices.topk(topk, dim=-1, sorted=True).indices
    return scores, group_ids, ids


def torch_route_epilogue(
    scores: torch.Tensor,
    ids: torch.Tensor,
    renormalize: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The selected scores, renormalized under ``renormalize``, then TopK's
    ``topk_weights *= routed_scaling_factor`` with a ``scale``; int32 ids."""
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    if scale is not None:
        weights *= scale
    return weights, ids.to(torch.int32)


def torch_grouped_route(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_groups: int,
    group_topk: int,
    topk_groups: int,
    topk: int,
    renormalize: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole route: FP32 ``[rows, topk]`` weights and int32 ids."""
    scores, _, ids = torch_router_select(
        gating_output, correction_bias, num_groups, group_topk, topk_groups, topk
    )
    return torch_route_epilogue(scores, ids, renormalize, scale)
