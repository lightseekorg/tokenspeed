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

import math

import torch
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx1250 import (
    PrecisionConfig,
)

_MXFP_BLOCK_SIZE = 32
_GFX1250_SCALE_N_BLOCK = 128


def _release_parameter(module: torch.nn.Module, name: str) -> None:
    if hasattr(module, name):
        delattr(module, name)


def _interleave_gate_up_rows(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % tensor.ndim
    rows = int(tensor.shape[dim])
    if rows % 2 != 0:
        raise ValueError(f"W13 gate/up row dimension must be even, got {rows}")
    gate, up = tensor.split(rows // 2, dim=dim)
    shape = list(tensor.shape)
    return torch.stack((gate, up), dim=dim + 1).reshape(shape).contiguous()


def _make_k_packed_mxfp4_weight(quant_tensor: torch.Tensor) -> torch.Tensor:
    """Return packed MXFP4 W storage in ``(..., K_packed, N)`` layout."""
    if quant_tensor.ndim < 2:
        raise ValueError("MXFP4 weight tensor must have at least 2 dimensions")
    *leading_shape, n, k_packed = quant_tensor.shape
    out_shape = [*leading_shape, k_packed, n]
    out_strides = [0] * len(out_shape)
    out_strides[-2] = 1
    out_strides[-1] = k_packed
    running_stride = n * k_packed
    for dim in range(len(leading_shape) - 1, -1, -1):
        out_strides[dim] = running_stride
        running_stride *= out_shape[dim]
    out = torch.empty_strided(
        out_shape, out_strides, dtype=quant_tensor.dtype, device=quant_tensor.device
    )
    out.copy_(quant_tensor.transpose(-2, -1))
    return out


def _swizzle_gfx1250_mxfp4_scale(scale: torch.Tensor) -> torch.Tensor:
    """Match triton_kernels' GFX1250MXScaleLayout byte swizzle."""
    if scale.ndim < 2:
        raise ValueError("MXFP4 scale tensor must have at least 2 dimensions")

    scale = scale.transpose(-2, -1).contiguous()
    *leading_shape, k_scale, n = scale.shape
    leading = math.prod(leading_shape)
    align_k_scale = min(4, max(int(k_scale), 1))
    k_scale_pad = math.ceil(k_scale / align_k_scale) * align_k_scale
    n_pad = math.ceil(n / _GFX1250_SCALE_N_BLOCK) * _GFX1250_SCALE_N_BLOCK

    scale = torch.nn.functional.pad(
        scale,
        (0, n_pad - n, 0, k_scale_pad - k_scale),
    )
    scale = scale.transpose(-1, -2)
    scale = scale.view(
        leading,
        n_pad // _GFX1250_SCALE_N_BLOCK,
        4,
        _GFX1250_SCALE_N_BLOCK // 4,
        k_scale_pad // align_k_scale,
        align_k_scale,
    )
    scale = scale.permute(0, 1, 4, 3, 2, 5).contiguous()
    scale = scale.reshape(
        leading,
        n_pad // _GFX1250_SCALE_N_BLOCK,
        k_scale_pad * _GFX1250_SCALE_N_BLOCK,
    )
    scale = scale.transpose(-1, -2)
    assert scale.stride(-2) == 1
    return scale


def _swizzle_mxfp4(quant_tensor: torch.Tensor, scale: torch.Tensor):
    quant_tensor = _make_k_packed_mxfp4_weight(quant_tensor)
    scale = _swizzle_gfx1250_mxfp4_scale(scale)
    return quant_tensor, scale


def preprocess_gluon_mxfp4_gfx1250_moe_weights(
    plan: dict,
    w: torch.nn.Module,
) -> None:
    """Prepare MXFP4 MoE weights for the gfx1250 Gluon kernel.

    Args:
        plan: MoE execution plan. Currently unused; accepted for the common
            tokenspeed-kernel weight preprocessor signature.
        w: Module containing ``w13``/``w2`` packed MXFP4 weights, e8m0 scales,
            optional biases, and optional static FP8 activation scales. The
            module is mutated in-place with gfx1250-ready weight tensors and
            precision configs.
    """
    del plan
    w13_layout = getattr(w, "w13_input_layout", "concatenated")
    if w13_layout not in {"interleaved", "concatenated"}:
        raise ValueError(f"unknown w13_input_layout: {w13_layout!r}")

    w13_weight = w.w13_weight
    w13_weight_scale = w.w13_weight_scale

    if w13_layout == "concatenated":
        w13_weight = torch.nn.Parameter(
            _interleave_gate_up_rows(w13_weight.data, dim=-2),
            requires_grad=False,
        )
        w13_weight_scale = torch.nn.Parameter(
            _interleave_gate_up_rows(w13_weight_scale.data, dim=-2),
            requires_grad=False,
        )

    if hasattr(w, "w13_weight_bias"):
        w13_weight_bias = w.w13_weight_bias.to(torch.float32)
        if w13_layout == "concatenated":
            w13_weight_bias = _interleave_gate_up_rows(w13_weight_bias, dim=-1)
        w._gluon_w13_bias_is_zero = not bool(
            torch.count_nonzero(w13_weight_bias).item()
        )
        w.w13_weight_bias = torch.nn.Parameter(w13_weight_bias, requires_grad=False)
    if hasattr(w, "w2_weight_bias"):
        w2_weight_bias = w.w2_weight_bias.to(torch.float32)
        w._gluon_w2_bias_is_zero = not bool(torch.count_nonzero(w2_weight_bias).item())
        w.w2_weight_bias = torch.nn.Parameter(w2_weight_bias, requires_grad=False)

    w13_weight, w13_scale = _swizzle_mxfp4(
        w13_weight.data,
        w13_weight_scale.data,
    )
    w2_weight, w2_scale = _swizzle_mxfp4(
        w.w2_weight.data,
        w.w2_weight_scale.data,
    )

    quant_config = getattr(w, "quant_config", None)
    use_dynamic_mxfp4_activations = bool(
        getattr(quant_config, "use_dynamic_mxfp4_activations", False)
    )
    has_static_fp8_scales = (
        hasattr(w, "w13_input_scale")
        and hasattr(w, "w2_input_scale")
        and not use_dynamic_mxfp4_activations
    )
    w13_act_scale = None
    w2_act_scale = None
    if has_static_fp8_scales:
        w13_in_scale = (
            w.w13_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w13_input_scale.device)
            .contiguous()
        )
        w2_in_scale = (
            w.w2_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(w.w2_input_scale.device)
            .contiguous()
        )
        w.w13_act_scale = w13_in_scale
        w.w2_act_scale = w2_in_scale
        w13_act_scale = w13_in_scale
        w2_act_scale = w2_in_scale
    w13_weight.act_scale = w13_act_scale
    w2_weight.act_scale = w2_act_scale

    out_dtype = torch.bfloat16
    w.w13_precision_config = PrecisionConfig(
        b_mx_scale=w13_scale,
        out_dtype=out_dtype,
    )
    w.w2_precision_config = PrecisionConfig(
        b_mx_scale=w2_scale,
        out_dtype=out_dtype,
    )
    w.w13_precision_config.b_microblock_size = _MXFP_BLOCK_SIZE
    w.w2_precision_config.b_microblock_size = _MXFP_BLOCK_SIZE

    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    _release_parameter(w, "w13_weight")
    _release_parameter(w, "w2_weight")
    _release_parameter(w, "w13_weight_scale")
    _release_parameter(w, "w2_weight_scale")
    torch.cuda.empty_cache()


__all__ = [
    "preprocess_gluon_mxfp4_gfx1250_moe_weights",
    "_make_k_packed_mxfp4_weight",
]
