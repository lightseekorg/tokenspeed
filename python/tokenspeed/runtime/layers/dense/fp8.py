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


import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_quant_fp8,
    static_quant_fp8,
)
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import SelectionObjective
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.dense.utils import normalize_e4m3fn_to_e4m3fnuz
from tokenspeed.runtime.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.utils import convert_to_channelwise

platform = Platform.get()


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

        if self.block_quant:
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
                    assert self.quant_config.activation_scheme == "dynamic"
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    assert self.quant_config.linear_activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
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
            # If ROCm, normalize the weights and scales to e4m3fnuz
            if platform.is_fp8e4m3fnuz:
                # activation_scheme: dynamic
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=None,
                )
                layer.input_scale = None
            else:
                weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data
            layer.weight.data = weight.data
            layer.weight_scale_inv.data = weight_scale.data
            layer._mm_kernel = tokenspeed_kernel.select_mm_kernel(
                A_dtype=platform.fp8e4m3fn.dtype,
                K=layer.input_size_per_partition,
                N=layer.output_size_per_partition,
                quant="mxfp8",
                objective=SelectionObjective.LATENCY,
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
            selected_kernel = getattr(layer, "_mm_kernel", None)

            output = tokenspeed_kernel.mm(
                input_2d,
                layer.weight,
                A_scales=block_scale,
                B_scales=layer.weight_scale_inv,
                bias=bias,
                out_dtype=output_dtype,
                quant="mxfp8",
                block_size=self.quant_config.weight_block_size,
                selected_kernel=selected_kernel,
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
                assert input_scale.numel() == 1
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
