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

"""Weight-only per-block FP8 linear method (ModelOpt ``FP8_PB_WO``).

Weights stay FP8 (e4m3) in device memory with float32 per-``[block_n,
block_k]`` dequant scales; activations are never quantized (bf16/fp16 pass
through) and the GEMM dequantizes weight blocks in-kernel (W8A16). This
preserves ModelOpt's weight-only export semantics exactly: the only deviation
from a bf16-dequantized reference GEMM is fp32-accumulator tiling order.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.gemm.triton import wo_block_fp8_matmul_triton
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8BlockWeightOnlyConfig
from tokenspeed.runtime.layers.quantization.utils import modelopt_block_scale_to_2d

_SCALE_UNLOADED_SENTINEL = torch.finfo(torch.float32).min


def _wrap_block_scale_loader(weight_loader: Callable | None) -> Callable | None:
    """Wrap a weight loader to normalize the scale layout before sharding."""
    if weight_loader is None:
        return None

    def _loader(param, loaded_weight: torch.Tensor, *args, **kwargs):
        return weight_loader(
            param, modelopt_block_scale_to_2d(loaded_weight), *args, **kwargs
        )

    return _loader


class Fp8BlockWeightOnlyLinearMethod(LinearMethodBase):
    """Linear method for weight-only per-block FP8 (ModelOpt ``FP8_PB_WO``).

    Registers:
        weight: FP8 (e4m3) ``[out_partition, in_partition]``.
        weight_scale: float32 ``[ceil(out/block_n), ceil(in/block_k)]`` dequant
            multipliers, named after the checkpoint tensor so ModelOpt exports
            load without renames (ModelOpt's 4-D scale layout is normalized in
            the loader).

    TP sharding narrows the FP8 weight and the block-scale grid together, so
    per-partition sizes must align to the block shape whenever a dimension is
    actually sharded (asserted in :meth:`create_weights`).
    """

    def __init__(self, quant_config: Fp8BlockWeightOnlyConfig):
        self.quant_config = quant_config
        self.weight_block_size = list(quant_config.weight_block_size)

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
        block_n, block_k = self.weight_block_size

        # Row-parallel shards narrow the K axis: every partition boundary must
        # land on a scale-block boundary or shards would split a scale cell.
        if input_size > input_size_per_partition:
            if input_size_per_partition % block_k != 0:
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )
        # Column-parallel or merged shards narrow the N axis likewise.
        if output_size > output_size_per_partition or len(output_partition_sizes) > 1:
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

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale = BlockQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=_wrap_block_scale_loader(weight_loader),
        )
        # Sentinel lets process_weights_after_loading fail loudly on scales
        # that never loaded (e.g. checkpoint/runtime name drift).
        scale[:] = _SCALE_UNLOADED_SENTINEL
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "FP8_PB_WO weight must stay float8_e4m3fn, got "
                f"{layer.weight.dtype}."
            )
        if bool((layer.weight_scale.data == _SCALE_UNLOADED_SENTINEL).any()):
            raise RuntimeError(
                "FP8_PB_WO weight_scale was never (fully) loaded; the "
                "checkpoint is missing the scale tensor or the name mapping "
                "dropped it."
            )
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        block_scale: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if block_scale is not None:
            # A non-None block_scale means the caller pre-quantized the
            # activations for a fused w8a8 path; that would silently change
            # FP8_PB_WO's weight-only semantics.
            raise ValueError(
                "FP8_PB_WO is weight-only: activations must stay bf16/fp16, "
                "but a pre-quantized activation block_scale was passed. "
                "Disable the fp8 activation-quant fusion for this layer."
            )
        input_2d = x.view(-1, x.shape[-1])
        out_dtype = output_dtype or x.dtype
        output = wo_block_fp8_matmul_triton(
            input_2d,
            layer.weight,
            layer.weight_scale,
            block_size=self.weight_block_size,
            output_dtype=out_dtype,
        )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], layer.weight.shape[0])
