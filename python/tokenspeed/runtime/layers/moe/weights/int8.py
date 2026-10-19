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
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import make_weight_loader
from tokenspeed.runtime.utils import set_weight_attrs


def _load_channel_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    loaded_weight = loaded_weight.squeeze()
    target = param.data[local_expert_id]
    if shard_id in {"w1", "w3"}:
        shard_size = target.numel() // 2
        target = target.narrow(0, 0 if shard_id == "w1" else shard_size, shard_size)
    elif shard_id != "w2":
        raise ValueError(f"Unknown W8A8 scale shard: {shard_id}")
    if loaded_weight.numel() != target.numel():
        raise ValueError(
            f"W8A8 channel scale has {loaded_weight.numel()} values; "
            f"expected {target.numel()} for {shard_id}"
        )
    target.copy_(loaded_weight.reshape_as(target))


def _load_smooth_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    if shard_id not in {"w1", "w2", "w3"}:
        raise ValueError(f"Unknown W8A8 smooth-scale shard: {shard_id}")
    target = param.data[local_expert_id]
    loaded_weight = loaded_weight.squeeze()
    if loaded_weight.numel() != target.numel():
        raise ValueError(
            f"W8A8 smooth scale has {loaded_weight.numel()} values; "
            f"expected {target.numel()} for {shard_id}"
        )
    target.copy_(loaded_weight.reshape_as(target))


def create_int8_weight_pair(
    spec: MoELayerSpec,
    layer: nn.Module,
    *,
    smooth_quant: bool,
) -> int:
    """Create canonical per-channel INT8 MoE weights and scales."""
    ispp = spec.intermediate_size // spec.tp_size
    w13_weight = nn.Parameter(
        torch.empty(
            spec.num_local_experts, 2 * ispp, spec.hidden_size, dtype=torch.int8
        ),
        requires_grad=False,
    )
    w2_weight = nn.Parameter(
        torch.empty(spec.num_local_experts, spec.hidden_size, ispp, dtype=torch.int8),
        requires_grad=False,
    )
    w13_weight_scale = nn.Parameter(
        torch.empty(spec.num_local_experts, 2 * ispp, dtype=torch.float32),
        requires_grad=False,
    )
    w2_weight_scale = nn.Parameter(
        torch.empty(spec.num_local_experts, spec.hidden_size, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)
    layer.register_parameter("w2_weight", w2_weight)
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    weight_loader = make_weight_loader(spec)
    for weight in (w13_weight, w2_weight):
        set_weight_attrs(weight, {"weight_loader": weight_loader})
    for scale in (w13_weight_scale, w2_weight_scale):
        set_weight_attrs(scale, {"weight_loader": _load_channel_scale})

    if smooth_quant:
        w13_smooth_scale = nn.Parameter(
            torch.ones(spec.num_local_experts, spec.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        w2_smooth_scale = nn.Parameter(
            torch.ones(spec.num_local_experts, ispp, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_smooth_scale", w13_smooth_scale)
        layer.register_parameter("w2_smooth_scale", w2_smooth_scale)
        set_weight_attrs(w13_smooth_scale, {"weight_loader": _load_smooth_scale})
        set_weight_attrs(w2_smooth_scale, {"weight_loader": _load_smooth_scale})

    return ispp


__all__ = ["create_int8_weight_pair"]
