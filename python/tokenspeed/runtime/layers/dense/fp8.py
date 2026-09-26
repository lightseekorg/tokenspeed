# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.quantization import quantize_fp8
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.utils import convert_to_channelwise
from tokenspeed.runtime.utils.env import global_server_args_dict


class Fp8LinearMethod(LinearMethodBase):
    """Load FP8 weights and execute linear layers through quantize_fp8 and mm."""

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant and self.quant_config.is_checkpoint_fp8_serialized:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if input_size > input_size_per_partition:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                output_size > output_size_per_partition
                or len(output_partition_sizes) > 1
            ):
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    if self.quant_config.activation_scheme != "dynamic":
                        raise ValueError(
                            "Block FP8 requires dynamic activation quantization."
                        )
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    if self.quant_config.linear_activation_scheme != "dynamic":
                        raise ValueError(
                            "Block FP8 requires dynamic linear activation quantization."
                        )
                scale_dtype = self.quant_config.weight_scale_dtype
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=scale_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                if scale_dtype == torch.uint8:
                    scale.zero_()
                else:
                    scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.block_quant:
            if not self.quant_config.is_checkpoint_fp8_serialized:
                qweight, weight_scale = quantize_fp8(
                    layer.weight.data,
                    granularity="block",
                    block_size=self.quant_config.weight_block_size,
                )
                layer.weight = Parameter(qweight, requires_grad=False)
                layer.register_parameter(
                    "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
                )
                layer.input_scale = None
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.quant_config.is_checkpoint_fp8_serialized:
                # apply per-channel quantization default as
                qweight, weight_scale = quantize_fp8(
                    layer.weight,
                    granularity="token",
                )
                weight_scale = weight_scale.t().contiguous()

                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )

                weight = layer.weight
                weight_scale = convert_to_channelwise(
                    layer.weight_scale, layer.logical_widths
                )

                # Update layer with new values.
                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        block_scale: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if self.block_quant:
            input_2d = x.view(-1, x.shape[-1])
            output_shape = [*x.shape[:-1], layer.weight.shape[0]]
            output_dtype = output_dtype or x.dtype
            if block_scale is None:
                scale_encoding = (
                    "ue8m0"
                    if layer.weight_scale_inv.dtype == torch.uint8
                    else "float32"
                )
                input_2d, block_scale = quantize_fp8(
                    input_2d,
                    granularity="token_group",
                    group_size=self.quant_config.weight_block_size[1],
                    scale_encoding=scale_encoding,
                )
            solution = None
            if global_server_args_dict[
                "dense_gemm_backend"
            ] == "trtllm_cutedsl" and tuple(self.quant_config.weight_block_size) == (
                128,
                128,
            ):
                solution = "trtllm_cutedsl"
            output = mm(
                input_2d,
                layer.weight,
                A_scales=block_scale,
                B_scales=layer.weight_scale_inv,
                bias=bias,
                out_dtype=output_dtype,
                quant="mxfp8",
                block_size=self.quant_config.weight_block_size,
                solution=solution,
            )
            return output.to(dtype=output_dtype).view(*output_shape)
        else:
            input = x
            weight = layer.weight
            weight_scale = layer.weight_scale
            input_scale = layer.input_scale

            # View input as 2D matrix for fp8 methods
            input_2d = input.view(-1, input.shape[-1])
            output_shape = [*input.shape[:-1], weight.shape[1]]

            if input_scale is not None:
                if input_scale.numel() != 1:
                    raise ValueError(
                        f"input_scale must contain exactly one value, got {input_scale.numel()}."
                    )
                qinput, x_scale = quantize_fp8(input_2d, scale=input_scale)
            else:
                qinput, x_scale = quantize_fp8(input_2d, granularity="token")

            qinput = qinput.view(-1, qinput.shape[-1])

            output = mm(
                qinput,
                weight,
                A_scales=x_scale,
                B_scales=weight_scale,
                out_dtype=input.dtype,
                quant="fp8",
            )
            if bias is not None:
                output = output + bias
            return output.view(*output_shape)
