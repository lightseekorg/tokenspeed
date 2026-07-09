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
from tokenspeed.runtime.layers.moe.weights.loaders import (
    make_group_scale_loader,
    make_weight_loader,
    per_tensor_scale_loader,
)
from tokenspeed.runtime.utils import set_weight_attrs


def create_nvfp4_weight_pair(
    spec: MoELayerSpec,
    layer: nn.Module,
    *,
    group_size: int,
) -> None:
    ispp = spec.intermediate_size // spec.tp_size
    w13_weight = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            2 * ispp,
            spec.hidden_size // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    w2_weight = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            spec.hidden_size,
            ispp // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)
    layer.register_parameter("w2_weight", w2_weight)

    w13_weight_scale = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            2 * ispp,
            spec.hidden_size // group_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            spec.hidden_size,
            ispp // group_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    w13_weight_scale_2 = torch.nn.Parameter(
        torch.empty(spec.num_local_experts, 2, dtype=torch.float32),
        requires_grad=False,
    )
    w2_weight_scale_2 = torch.nn.Parameter(
        torch.empty(spec.num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    w13_input_scale = torch.nn.Parameter(
        torch.empty(spec.num_local_experts, 2, dtype=torch.float32),
        requires_grad=False,
    )
    w2_input_scale = torch.nn.Parameter(
        torch.empty(spec.num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
    layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
    layer.register_parameter("w13_input_scale", w13_input_scale)
    layer.register_parameter("w2_input_scale", w2_input_scale)

    weight_loader = make_weight_loader(spec)
    scale_loader = make_group_scale_loader(spec)
    per_tensor_loader = per_tensor_scale_loader()
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w13_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w13_weight_scale_2, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w2_weight_scale_2, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w13_input_scale, {"weight_loader": per_tensor_loader})
    set_weight_attrs(w2_input_scale, {"weight_loader": per_tensor_loader})


__all__ = ["create_nvfp4_weight_pair"]
