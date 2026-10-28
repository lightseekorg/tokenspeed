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

"""ModelOpt MXFP8 linear weights kept quantized through native Marlin GEMM."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm import Mxfp8LinearPlan, prepare_mxfp8_linear
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.utils import set_weight_attrs


class ModelOptMxfp8LinearMethod(LinearMethodBase):
    """Load E4M3/E8M0 shards, then prepare the dense Marlin W8A16 operator."""

    def __init__(
        self, *, use_atomic_add: bool, use_fp32_reduce: bool, override: str | None
    ) -> None:
        self.use_atomic_add = use_atomic_add
        self.use_fp32_reduce = use_fp32_reduce
        self.override = override
        self.kernel: Mxfp8LinearPlan | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if params_dtype != torch.bfloat16:
            raise ValueError("ModelOpt MXFP8 dense Marlin requires BF16 activations")
        if input_size_per_partition <= 0 or input_size_per_partition % 32:
            raise ValueError(
                "ModelOpt MXFP8 TP input partitions must be divisible by 32"
            )
        if not output_partition_sizes or min(output_partition_sizes) <= 0:
            raise ValueError("ModelOpt MXFP8 output partitions must be positive")
        n, k = sum(output_partition_sizes), input_size_per_partition
        loader = extra_weight_attrs.get("weight_loader")

        def load_weight(param, loaded_weight, *args, **kwargs):
            if self.kernel is not None:
                raise RuntimeError(
                    "MXFP8 Marlin weight refits require a fresh model load"
                )
            if loaded_weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "Expected E4M3 MXFP8 weight; BF16 refits cannot be silently quantized"
                )
            if loader is None:
                if args or kwargs:
                    raise TypeError("No sharded loader was supplied")
                if param.shape != loaded_weight.shape:
                    raise ValueError("MXFP8 weight shape mismatch")
                param.data.copy_(loaded_weight)
            else:
                loader(param, loaded_weight, *args, **kwargs)

        def load_scale(param, loaded_weight, *args, **kwargs):
            if self.kernel is not None:
                raise RuntimeError(
                    "MXFP8 Marlin scale refits require a fresh model load"
                )
            if loaded_weight.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise TypeError("MXFP8 scales must be raw uint8/E8M0 bytes")
            raw = loaded_weight.view(torch.uint8)
            if loader is None:
                if args or kwargs:
                    raise TypeError("No sharded loader was supplied")
                if param.shape != raw.shape:
                    raise ValueError("MXFP8 scale shape mismatch")
                param.data.copy_(raw)
            else:
                loader(param, raw, *args, **kwargs)

        for name, shape, dtype, wrapped_loader in (
            ("weight", (n, k), torch.float8_e4m3fn, load_weight),
            ("weight_scale", (n, k // 32), torch.uint8, load_scale),
        ):
            param = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(
                param,
                {
                    **extra_weight_attrs,
                    "input_dim": 1,
                    "output_dim": 0,
                    "weight_loader": wrapped_loader,
                },
            )
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.kernel is not None:
            return
        self.kernel = prepare_mxfp8_linear(
            layer.weight.data,
            layer.weight_scale.data,
            input_dtype=torch.bfloat16,
            use_atomic_add=self.use_atomic_add,
            use_fp32_reduce=self.use_fp32_reduce,
            solution=None,
            override=self.override,
        )
        # Replace raw parameters rather than retaining an extra full weight copy.
        # Keep their loader contracts so an unsupported refit fails explicitly.
        for name, prepared in (
            ("weight", self.kernel.weight),
            ("weight_scale", self.kernel.weight_scale),
        ):
            old = layer.get_parameter(name)
            param = Parameter(prepared, requires_grad=False)
            set_weight_attrs(param, dict(old.__dict__))
            layer.register_parameter(name, param)

    def weight_for_absorption(self, layer: torch.nn.Module) -> torch.Tensor:
        """Return the BF16 tensor needed by absorbed MLA, not by dense GEMM."""
        if self.kernel is not None:
            return self.kernel.dequantize_for_absorption()
        from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import (
            dequantize_mxfp8_weight,
        )

        return dequantize_mxfp8_weight(layer.weight, layer.weight_scale)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.kernel is None:
            raise RuntimeError("Marlin MXFP8 weights must be prepared before forward")
        return self.kernel.apply(x, bias)
