# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MXFP4 dense linear bridge for checkpoint-serialized Kimi MLP weights."""

from __future__ import annotations

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.quantization.base_config import QuantizeMethodBase

MXFP4_BLOCK = 32


class Mxfp4LinearMethod(QuantizeMethodBase):
    """Packed MXFP4 dense weights.

    Kimi-K2.5 MXFP4 stores dense layer-0 MLP and MoE shared-expert MLP tensors
    in the same packed FP4/e8m0 format as routed experts. Runtime activations
    are quantized to packed MXFP4 before the dense GEMM so checkpoint weights
    can stay packed in VRAM.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.group_size = getattr(quant_config, "group_size", MXFP4_BLOCK)

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
        del input_size, output_size
        _validate_mxfp4_partition(input_size_per_partition)
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        scale_loader = _wrap_e8m0_scale_loader(weight_loader)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype
        layer.orig_dtype = params_dtype

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        weight.output_dim = 0
        weight.input_dim = 1
        if weight_loader is not None:
            weight.weight_loader = weight_loader
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        weight_scale.output_dim = 0
        weight_scale.input_dim = 1
        if scale_loader is not None:
            weight_scale.weight_loader = scale_loader
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_mxfp4_dense_processed", False):
            return
        layer.weight_triton_tensor = layer.weight.data
        layer.weight_scale_triton_tensor = layer.weight_scale.data
        layer._mxfp4_dense_processed = True

    def apply(self, layer, x, bias=None):
        if not getattr(layer, "_mxfp4_dense_processed", False):
            self.process_weights_after_loading(layer)
        input_2d = x.reshape(-1, x.shape[-1])
        output_shape = (*x.shape[:-1], layer.output_size_per_partition)
        input_quant, input_scale = mxfp4_quantize(input_2d)
        output = tokenspeed_kernel.mm(
            input_quant,
            layer.weight_triton_tensor,
            A_scales=input_scale,
            B_scales=layer.weight_scale_triton_tensor,
            bias=bias,
            out_dtype=x.dtype,
            quant="mxfp4",
            expected_kernel_name="triton_mm_mxfp4",
        )
        return output.reshape(*output_shape)


def _validate_mxfp4_partition(input_size_per_partition: int) -> None:
    if input_size_per_partition % 2 != 0:
        raise ValueError(
            f"MXFP4 input partition {input_size_per_partition} must be divisible by 2"
        )
    if input_size_per_partition % MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP4 input partition {input_size_per_partition} must be divisible by "
            f"{MXFP4_BLOCK}"
        )


def _wrap_e8m0_scale_loader(weight_loader):
    if weight_loader is None:
        return None

    def scale_loader(param, loaded_weight, *args, **kwargs):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if (
            e8m0_dtype is not None
            and param.dtype == torch.uint8
            and loaded_weight.dtype == e8m0_dtype
        ):
            loaded_weight = loaded_weight.view(torch.uint8)
        return weight_loader(param, loaded_weight, *args, **kwargs)

    return scale_loader


__all__ = [
    "Mxfp4LinearMethod",
]
