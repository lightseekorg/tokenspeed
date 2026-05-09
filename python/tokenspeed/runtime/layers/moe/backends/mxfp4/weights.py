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

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend

MXFP4_BLOCK = 32


def create_mxfp4_weights(
    backend: MoEBackend,
    layer: nn.Module,
    num_local_experts: int,
    hidden_size_padded: int,
    ispp_padded: int,
    with_bias: bool = False,
) -> None:
    from tokenspeed.runtime.utils import set_weight_attrs

    # Fused gate_up_proj (column parallel)
    w13_weight = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            2 * ispp_padded,
            hidden_size_padded // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)

    w13_weight_scale = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            2 * ispp_padded,
            hidden_size_padded // MXFP4_BLOCK,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)

    # down_proj (row parallel)
    w2_weight = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            hidden_size_padded,
            ispp_padded // 2,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight", w2_weight)

    w2_weight_scale = torch.nn.Parameter(
        torch.zeros(
            num_local_experts,
            hidden_size_padded,
            ispp_padded // MXFP4_BLOCK,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    if with_bias:
        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(num_local_experts, 2 * ispp_padded, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(num_local_experts, hidden_size_padded, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)

    # Set up weight loader (no transpose for packed uint8 mxfp4)
    weight_loader = backend._make_weight_loader()
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})
    if with_bias:
        set_weight_attrs(w13_weight_bias, {"weight_loader": weight_loader})
        set_weight_attrs(w2_weight_bias, {"weight_loader": weight_loader})


__all__ = ["MXFP4_BLOCK", "create_mxfp4_weights"]
