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


def create_dense_weight_pair(
    spec: MoELayerSpec,
    layer: nn.Module,
    *,
    params_dtype: torch.dtype,
    with_bias: bool = False,
) -> int:
    ispp = spec.intermediate_size // spec.tp_size
    w13_weight = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            2 * ispp,
            spec.hidden_size,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    w2_weight = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            spec.hidden_size,
            ispp,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)
    layer.register_parameter("w2_weight", w2_weight)

    weight_loader = make_weight_loader(spec)
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

    if with_bias:
        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(spec.num_local_experts, 2 * ispp, dtype=params_dtype),
            requires_grad=False,
        )
        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(spec.num_local_experts, spec.hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        bias_loader = make_weight_loader(spec, is_bias=True)
        set_weight_attrs(w13_weight_bias, {"weight_loader": bias_loader})
        set_weight_attrs(w2_weight_bias, {"weight_loader": bias_loader})

    return ispp


__all__ = ["create_dense_weight_pair"]
