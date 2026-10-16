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
# all copies of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Symmetric channel INT8 weights with dynamic symmetric token activations."""

import torch
from tokenspeed_kernel.ops.gemm.int8 import prepare_int8_linear

from tokenspeed.runtime.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsScheme,
)


class CompressedTensorsW8A8Int8(CompressedTensorsScheme):
    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer,
        output_partition_sizes,
        input_size_per_partition,
        params_dtype,
        weight_loader,
        **kwargs,
    ):
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=torch.int8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ChannelQuantScaleParameter(
                data=torch.empty(sum(output_partition_sizes), 1, dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer):
        layer._int8_plan = prepare_int8_linear(
            layer.weight, layer.weight_scale, output_dtype=layer.params_dtype
        )

    def apply_weights(self, layer, x, bias=None):
        return layer._int8_plan(
            x,
            bias=bias,
            defer_dequant=getattr(layer, "defer_dequant", False),
        )
