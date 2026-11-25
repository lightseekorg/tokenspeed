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

"""Block FP8 storage with the ordinary BF16 GEMM numerical contract."""

import torch
from tokenspeed_kernel.ops.gemm import mm, plan_dense_gemm
from tokenspeed_kernel.ops.quantization import dequantize_fp8_block

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod


class W8A16Fp8BlockLinearMethod(UnquantizedLinearMethod):
    """Keep codes/scales resident; own no persistent BF16 weight copy."""

    def __init__(self, quant_config):
        super().__init__()
        self.quant_config = quant_config
        self._storage = Fp8LinearMethod(quant_config)

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("W8A16 block computation requires BF16 activations")
        self._storage.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs
        )
        self.kernel_plan = plan_dense_gemm(
            mode="mm",
            input_size=input_size_per_partition,
            output_size=sum(output_partition_sizes),
            dtype=torch.bfloat16,
            batch_size=None,
            batch_invariant=self.batch_invariant,
        )

    def process_weights_after_loading(self, layer):
        from tokenspeed.runtime.layers.quantization.w8a16_fp8_block import (
            validate_w8a16_weights,
        )

        validate_w8a16_weights(layer.weight, layer.weight_scale_inv)

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise ValueError("W8A16 block computation requires BF16 activations")
        weight = dequantize_fp8_block(layer.weight, layer.weight_scale_inv)
        return mm(x, weight, bias=bias, plan=self.kernel_plan)
