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

"""MXFP4 dense linear bridge for checkpoint-serialized MXFP4 MLP weights."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm import (
    mxfp4_linear,
    mxfp4_swiglu_linear,
    prepare_mxfp4_linear,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp4
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.quantization.base_config import QuantizeMethodBase

MXFP4_BLOCK = 32


class Mxfp4LinearMethod(QuantizeMethodBase):
    """Packed MXFP4 dense weights.

    Kimi-K2.5 and Quark MXFP4 checkpoints store dense MLP and MoE
    shared-expert MLP tensors in the same packed FP4/e8m0 format as routed
    experts. Runtime activations are quantized to the checkpoint's activation
    format, MXFP4 or MXFP8 (dynamic per-32 scales), by the ``mxfp4_linear``
    kernels so checkpoint weights stay packed in VRAM. After loading, the
    weight is reordered once into the layout the selected kernel reads.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.group_size = getattr(quant_config, "group_size", MXFP4_BLOCK)
        activation_format = quant_config.linear_activation_format
        if activation_format is None:
            raise ValueError(
                "Mxfp4LinearMethod needs a checkpoint with MXFP4 or MXFP8 "
                "dynamic activations"
            )
        self.activation_format: str = activation_format

    @classmethod
    def for_activation_format(cls, activation_format: str) -> Mxfp4LinearMethod:
        """A method for weights quantized at load time, not by a checkpoint."""
        method = cls.__new__(cls)
        method.quant_config = None
        method.group_size = MXFP4_BLOCK
        method.activation_format = activation_format
        return method

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
        # The gated MLP owning a gate_up projection sets this (see
        # enable_swiglu) before loading, so the weight is prepared for the
        # fused SwiGLU kernel.
        layer.mxfp4_swiglu = False
        layer.mxfp4_linear_weight = None

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

    def enable_swiglu(self, layer: torch.nn.Module) -> None:
        """Mark ``layer`` as a stacked [gate; up] projection run by apply_swiglu."""
        if layer.mxfp4_linear_weight is not None:
            raise RuntimeError("enable_swiglu must precede weight processing")
        layer.mxfp4_swiglu = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.mxfp4_linear_weight is not None:
            return
        prepared = prepare_mxfp4_linear(
            layer.weight.data,
            layer.weight_scale.data,
            swiglu=layer.mxfp4_swiglu,
            activation_format=self.activation_format,
        )
        # Same shape and dtype in the kernel's layout; replacing the data
        # releases the checkpoint-order copy.
        layer.weight.data = prepared.weight
        layer.mxfp4_linear_weight = prepared

    def apply(self, layer, x, bias=None):
        if layer.mxfp4_linear_weight is None:
            self.process_weights_after_loading(layer)
        input_2d = x.reshape(-1, x.shape[-1])
        output_shape = (*x.shape[:-1], layer.output_size_per_partition)
        output = mxfp4_linear(
            input_2d,
            layer.mxfp4_linear_weight,
            activation_format=self.activation_format,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(*output_shape)

    def apply_swiglu(self, layer, x: torch.Tensor) -> torch.Tensor:
        """``silu(gate) * up`` of a stacked [gate; up] projection, fused.

        Args:
            layer: Projection marked with ``enable_swiglu``.
            x: BF16 input ``[..., K]``.

        Returns:
            BF16 ``[..., N/2]``.
        """
        if not layer.mxfp4_swiglu:
            raise RuntimeError("apply_swiglu needs a layer marked by enable_swiglu")
        if layer.mxfp4_linear_weight is None:
            self.process_weights_after_loading(layer)
        input_2d = x.reshape(-1, x.shape[-1])
        output = mxfp4_swiglu_linear(
            input_2d,
            layer.mxfp4_linear_weight,
            activation_format=self.activation_format,
        )
        return output.reshape(*x.shape[:-1], layer.output_size_per_partition // 2)


# Rows of the LM head quantized per call, to bound the temporary memory.
_LM_HEAD_CHUNK_ROWS = 16384


def quantize_lm_head_mxfp4(lm_head: torch.nn.Module, activation_format: str) -> None:
    """Quantize a BF16 LM head in place to MXFP4 weights.

    Every 32 weights share one UE8M0 scale (OCP MX rounding, as
    ``quantize_mxfp4`` does). The head then runs ``mxfp4_linear``, which
    quantizes its input row to ``activation_format``; this reads a quarter of
    the BF16 head's bytes and changes the logits by the MXFP4 rounding.

    Args:
        lm_head: The LM head; its ``weight`` is ``[rows, hidden]`` and it must
            not share that tensor with the input embedding.
        activation_format: ``"mxfp4"`` or ``"mxfp8"``.
    """
    weight = lm_head.weight.data
    if weight.dim() != 2 or not weight.dtype.is_floating_point:
        raise ValueError(
            f"LM head weight must be a 2-D float tensor, got {weight.dtype}"
        )
    rows, hidden = weight.shape
    _validate_mxfp4_partition(hidden)
    packed = torch.empty(rows, hidden // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        rows, hidden // MXFP4_BLOCK, dtype=torch.uint8, device=weight.device
    )
    for start in range(0, rows, _LM_HEAD_CHUNK_ROWS):
        chunk = weight[start : start + _LM_HEAD_CHUNK_ROWS].to(torch.bfloat16)
        q, s = quantize_mxfp4(chunk, scale_layout="linear")
        packed[start : start + chunk.shape[0]] = q
        scales[start : start + chunk.shape[0]] = s.view(torch.uint8).view(
            chunk.shape[0], -1
        )
    method = Mxfp4LinearMethod.for_activation_format(activation_format)
    # The same fields Mxfp4LinearMethod.create_weights gives a linear layer.
    lm_head.weight = Parameter(packed, requires_grad=False)
    lm_head.weight_scale = Parameter(scales, requires_grad=False)
    lm_head.input_size_per_partition = hidden
    lm_head.output_size_per_partition = rows
    lm_head.mxfp4_swiglu = False
    lm_head.mxfp4_linear_weight = None
    lm_head.quant_method = method
    method.process_weights_after_loading(lm_head)


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
    "quantize_lm_head_mxfp4",
]
