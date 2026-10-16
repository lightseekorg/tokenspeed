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

"""Ascend fused activation operators."""

from functools import lru_cache

import torch
import torch_npu


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Apply SwiGLU to last-dimension [gate, up], optionally writing ``out``."""
    if x.numel() == 0:
        if out is not None:
            return out
        return x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,))
    result = torch_npu.npu_swiglu(x, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def dequant_silu_and_mul_quant(x, weight_scale, activation_scale):
    """Dequantize INT32 [gate,up], apply SwiGLU and dynamically quantize rows."""
    if x.numel() == 0:
        return (
            x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=torch.int8),
            x.new_empty(x.shape[:-1], dtype=torch.float32),
        )
    values, scales = torch_npu.npu_dequant_swiglu_quant(
        x.reshape(-1, x.shape[-1]),
        weight_scale=weight_scale,
        activation_scale=activation_scale.reshape(-1),
        activate_left=True,
        quant_mode=1,
    )
    return (
        values.reshape(x.shape[:-1] + (x.shape[-1] // 2,)),
        scales.reshape(x.shape[:-1]),
    )


@lru_cache(maxsize=None)
def _unit_smooth_scale(device: torch.device, width: int) -> torch.Tensor:
    # CANN 9.0's SwiGluQuant reads this optional input unconditionally.
    # Keep the identity tensor alive across graph capture/replay; do not create
    # a Fill kernel on every activation call.
    return torch.ones((1, width), device=device, dtype=torch.float32)


def silu_and_mul_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused floating SwiGLU + INT8 quantization with dequantization scales."""
    if x.numel() == 0:
        return (
            x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=torch.int8),
            x.new_empty(x.shape[:-1], dtype=torch.float32),
        )
    values, inverse_scale = torch_npu.npu_swiglu_quant(
        x,
        smooth_scales=_unit_smooth_scale(x.device, x.shape[-1] // 2),
        activate_left=True,
        quant_mode=1,
    )
    # Unlike DynamicQuant/DequantSwigluQuant this op returns 127/amax.
    return values, inverse_scale.reciprocal()
