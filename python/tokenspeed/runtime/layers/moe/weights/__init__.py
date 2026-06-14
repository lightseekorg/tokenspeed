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

import torch

from tokenspeed.runtime.layers.moe.weights.fp8 import create_fp8_block_scale_inverses
from tokenspeed.runtime.layers.moe.weights.mxfp4 import (
    create_mxfp4_fp8_input_scales,
    create_mxfp4_weight_pair,
)
from tokenspeed.runtime.layers.moe.weights.nvfp4 import create_nvfp4_weight_pair
from tokenspeed.runtime.layers.moe.weights.unquant import create_dense_weight_pair


def create_layer_weights(
    spec,
    layer,
    quant_kind: str,
    quant_config,
    *,
    with_bias: bool = False,
    solution: str | None = None,
) -> None:
    if quant_kind == "unquant":
        create_dense_weight_pair(
            spec,
            layer,
            params_dtype=torch.get_default_dtype(),
            with_bias=with_bias,
        )
        return

    if quant_kind == "fp8":
        ispp = create_dense_weight_pair(
            spec,
            layer,
            params_dtype=torch.float8_e4m3fn,
            with_bias=with_bias,
        )
        create_fp8_block_scale_inverses(
            spec,
            layer,
            intermediate_size_per_partition=ispp,
            block_shape=quant_config.weight_block_size,
        )
        return

    if quant_kind == "nvfp4":
        create_nvfp4_weight_pair(
            spec,
            layer,
            group_size=quant_config.group_size,
        )
        return

    if quant_kind == "mxfp4":
        create_mxfp4_weight_pair(spec, layer, with_bias=with_bias, solution=solution)
        if quant_config.is_w4a8_fp8:
            create_mxfp4_fp8_input_scales(layer, spec.num_local_experts)
        return

    raise RuntimeError(f"Unsupported MoE quant kind: {quant_kind}")


__all__ = [
    "create_layer_weights",
]
