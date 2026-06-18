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
    make_weight_loader,
    round_up,
)
from tokenspeed.runtime.utils import set_weight_attrs


def create_fp8_block_scale_inverses(
    spec: MoELayerSpec,
    layer: nn.Module,
    *,
    intermediate_size_per_partition: int,
    block_shape: tuple[int, int],
) -> None:
    block_n, block_k = block_shape
    w13_weight_scale = torch.nn.Parameter(
        torch.ones(
            spec.num_local_experts,
            2 * round_up(intermediate_size_per_partition, block_n) // block_n,
            round_up(spec.hidden_size, block_k) // block_k,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.ones(
            spec.num_local_experts,
            round_up(spec.hidden_size, block_n) // block_n,
            round_up(intermediate_size_per_partition, block_k) // block_k,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
    layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

    weight_loader = make_weight_loader(spec)
    set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})


__all__ = ["create_fp8_block_scale_inverses"]
