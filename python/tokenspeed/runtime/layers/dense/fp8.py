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


import logging

import tokenspeed_kernel
import torch
from tokenspeed_kernel import fp8_linear, prepare_fp8_linear
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    per_block_quant_fp8,
    per_token_group_quant_fp8,
    per_token_quant_fp8,
    static_quant_fp8,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

from tokenspeed.runtime.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.utils import convert_to_channelwise


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

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
                qweight, weight_scale = per_block_quant_fp8(
                    layer.weight.data, self.quant_config.weight_block_size
                )
                layer.weight = Parameter(qweight, requires_grad=False)
                layer.register_parameter(
                    "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
                )
                layer.input_scale = None
            grouped_output_projection_plan = getattr(
                layer, "_dsv4_grouped_output_projection_plan", None
            )
            if grouped_output_projection_plan is not None:
                layer.weight_scale_inv.data = (
                    tokenspeed_kernel.dsv4_grouped_output_projection_process_weights(
                        grouped_output_projection_plan,
                        layer.weight.data,
                        layer.weight_scale_inv.data,
                    )
                )
                return
            layer._prepared_fp8_linear = prepare_fp8_linear(
                layer.weight.data,
                layer.weight_scale_inv.data,
                self.quant_config.weight_block_size,
                scale_format=getattr(self.quant_config, "scale_fmt", None),
            )
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.quant_config.is_checkpoint_fp8_serialized:
                # apply per-channel quantization default as
                qweight, weight_scale = per_token_group_quant_fp8(
                    layer.weight, layer.weight.shape[-1]
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
            if (
                input_2d.shape[0] <= 8
                and current_platform().is_nvidia
                and current_platform().arch_version < ArchVersion(9, 0)
            ):
                # SM80: W8A16 decode GEMV (see w8a16_gemv module docstring).
                from tokenspeed_kernel.ops.gemm.w8a16_gemv import (
                    w8a16_decode_gemv,
                )

                return w8a16_decode_gemv(
                    input_2d,
                    layer.weight,
                    layer.weight_scale_inv,
                    out_dtype=output_dtype,
                    bias=bias,
                ).view(*output_shape)
            plan = getattr(layer, "_prepared_fp8_linear", None)
            if plan is None:
                output = tokenspeed_kernel.mm(
                    input_2d,
                    layer.weight,
                    A_scales=block_scale,
                    B_scales=layer.weight_scale_inv,
                    bias=bias,
                    out_dtype=output_dtype,
                    quant="mxfp8",
                    block_size=self.quant_config.weight_block_size,
                )
            else:
                output = fp8_linear(
                    plan,
                    input_2d,
                    layer.weight,
                    layer.weight_scale_inv,
                    input_scales=block_scale,
                    bias=bias,
                    out_dtype=output_dtype,
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
                qinput, x_scale = static_quant_fp8(input_2d, input_scale)
            else:
                qinput, x_scale = per_token_quant_fp8(input_2d)

            qinput = qinput.view(-1, qinput.shape[-1])

            output = tokenspeed_kernel.mm(
                qinput,
                weight,
                A_scales=x_scale,
                B_scales=weight_scale,
                out_dtype=input.dtype,
            )
            if bias is not None:
                output = output + bias
            return output.view(*output_shape)

    def apply_with_activation(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: torch.nn.Module,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan = getattr(layer, "_prepared_fp8_linear", None)
        prepare_activation = getattr(activation, "prepare_for_fp8_linear", None)
        if self.block_quant and plan is not None and prepare_activation is not None:
            prepared = prepare_activation(x, plan)
            if prepared is not None:
                values, scales = prepared
                return self.apply(
                    layer,
                    values,
                    bias=bias,
                    block_scale=scales,
                    output_dtype=x.dtype,
                )
        return super().apply_with_activation(layer, x, activation, bias)

    def prepared_linear_plan(self, layer: torch.nn.Module) -> object | None:
        return getattr(layer, "_prepared_fp8_linear", None)
