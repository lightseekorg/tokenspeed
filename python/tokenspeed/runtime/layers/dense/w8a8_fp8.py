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
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config


def w8a8_fp8_per_channel_mm(
    x: torch.Tensor,
    weight_kn: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FP8 GEMM with dynamic per-token activations and per-channel weights.

    Args:
        x: ``[M, K]`` activations; quantized to FP8 E4M3 per token.
        weight_kn: ``[K, N]`` FP8 E4M3 weight (a transposed ``[N, K]`` view).
        weight_scale: FP32 ``[N, 1]`` per-output-channel dequant scales.
        out_dtype: Output dtype.

    Returns:
        ``[M, N]`` output in ``out_dtype``.
    """
    # One quantization group spanning the row is per-token scaling.
    qinput, x_scale = per_token_group_quant_fp8(x, x.shape[-1])
    return kernel_mm(
        qinput,
        weight_kn,
        A_scales=x_scale,
        B_scales=weight_scale,
        out_dtype=out_dtype,
        quant="fp8",
    )


class W8A8Fp8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Fp8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = layer.weight_scale.detach()
            # If checkpoint offline quantized with w8a8_fp8, load the weight and weight_scale directly.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        else:
            # use per-channel quantization on weight
            qweight, weight_scale = per_token_group_quant_fp8(
                layer.weight, layer.weight.shape[-1]
            )
            weight_scale = weight_scale.t().contiguous()

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)
        else:
            layer.weight_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        input = x
        weight = layer.weight
        weight_scale = layer.weight_scale

        # View input as 2D matrix for fp8 methods
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]

        output = w8a8_fp8_per_channel_mm(input_2d, weight, weight_scale, input.dtype)
        if bias is not None:
            output = output + bias
        return output.view(*output_shape)
