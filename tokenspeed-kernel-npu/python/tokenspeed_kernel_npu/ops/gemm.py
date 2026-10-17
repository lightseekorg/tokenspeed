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
import torch_npu


def prepare_weight_nz(weight: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    if weight.device.type != "npu" or weight.ndim != 2:
        raise ValueError("Weight-NZ preparation requires a two-dimensional NPU tensor.")
    if transpose:
        weight = weight.transpose(0, 1).contiguous()

    config = torch.npu.config
    previous = getattr(config, "allow_internal_format", None)
    try:
        config.allow_internal_format = True
        prepared = torch_npu.npu_format_cast(weight, 29)
        actual_format = torch_npu.get_npu_format(prepared)
        if actual_format != 29:
            raise RuntimeError(
                f"Weight-NZ preparation requested format 29, got {actual_format}."
            )
        return prepared
    finally:
        config.allow_internal_format = previous


class PreparedInt8Linear(torch.nn.Module):
    """Post-load NZ weights for dynamic-token INT8 dense matmul."""

    def __init__(self, weight, weight_scale, *, output_dtype):
        super().__init__()
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("INT8 linear requires [N,K] INT8 weights")
        if weight_scale.numel() != weight.shape[0]:
            raise ValueError("INT8 linear requires one scale per output channel")
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("INT8 linear output must be BF16 or FP16")
        self.output_dtype = output_dtype
        self.register_buffer(
            "weight", prepare_weight_nz(weight, transpose=True), persistent=False
        )
        self.register_buffer(
            "weight_scale", weight_scale.flatten().float(), persistent=False
        )

    def forward(self, x, *, bias=None, defer_dequant=False):
        if defer_dequant and bias is not None:
            raise ValueError("Deferred INT8 dequantization requires a bias-free linear")
        if isinstance(x, tuple):
            values, scales = x
            if values.dtype != torch.int8 or scales.dtype != torch.float32:
                raise ValueError(
                    "Prequantized input requires INT8 values and FP32 scales"
                )
        else:
            if x.numel() == 0:
                values = x.to(torch.int8)
                scales = x.new_empty(x.shape[:-1], dtype=torch.float32)
            else:
                values, scales = torch_npu.npu_dynamic_quant(x)
        output_shape = values.shape[:-1] + (self.weight.shape[1],)
        if values.numel() == 0:
            result = values.new_empty(
                output_shape,
                dtype=torch.int32 if defer_dequant else self.output_dtype,
            )
        else:
            result = torch_npu.npu_quant_matmul(
                values.reshape(-1, values.shape[-1]),
                self.weight,
                self.weight_scale,
                pertoken_scale=None if defer_dequant else scales.reshape(-1),
                bias=bias,
                output_dtype=torch.int32 if defer_dequant else self.output_dtype,
            ).reshape(output_shape)
        return (result, scales) if defer_dequant else result
