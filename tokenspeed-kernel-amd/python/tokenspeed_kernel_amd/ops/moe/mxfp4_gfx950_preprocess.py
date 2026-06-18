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

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel_amd.ops.moe.utils import FP4, wrap_torch_tensor

_MXFP_BLOCK_SIZE = 32
_GLUON_COMBINE_BLOCK_N = 128
_NON_K_PRESHUFFLE_BLOCK_SIZE = 32
_ALIGN_K_SCALE_SWIZZLE = 8
_ALIGN_N_SWIZZLE = 32


@dataclass
class GluonFlexData:
    dtype: torch.dtype | None = None
    scale: torch.Tensor | float | None = None


@dataclass
class GluonFlexCtx:
    lhs_data: GluonFlexData
    rhs_data: GluonFlexData


@dataclass
class GluonPrecisionConfig:
    flex_ctx: GluonFlexCtx
    b_mx_scale: torch.Tensor
    b_microblock_size: int
    out_dtype: torch.dtype | None
    a_mx_scale: torch.Tensor | None = None


def _is_scale_swizzled_cdna4(scale: torch.Tensor) -> bool:
    return scale.stride(-2) == 1 and scale.stride(-1) >= scale.shape[-2]


def _swizzle_scales_cdna4(scale: torch.Tensor) -> torch.Tensor:
    assert (
        scale.dtype == torch.uint8
    ), f"_swizzle_scales_cdna4: expected uint8 e8m0 scales, got {scale.dtype}"
    if _is_scale_swizzled_cdna4(scale):
        return scale

    scale = scale.transpose(-2, -1).contiguous()
    *leading_shape, k_scale, n = scale.shape
    batches = 1
    for dim in leading_shape:
        batches *= dim

    k_scale_padded = (
        (k_scale + _ALIGN_K_SCALE_SWIZZLE - 1)
        // _ALIGN_K_SCALE_SWIZZLE
        * _ALIGN_K_SCALE_SWIZZLE
    )
    n_padded = (n + _ALIGN_N_SWIZZLE - 1) // _ALIGN_N_SWIZZLE * _ALIGN_N_SWIZZLE

    scale = scale.mT.contiguous().mT
    scale = torch.nn.functional.pad(
        scale,
        (0, n_padded - n, 0, k_scale_padded - k_scale),
    )
    scale = scale.transpose(-1, -2)
    scale = scale.reshape(batches, n_padded, k_scale_padded)
    scale = scale.view(
        batches,
        n_padded // _NON_K_PRESHUFFLE_BLOCK_SIZE,
        2,
        16,
        k_scale_padded // 8,
        2,
        4,
        1,
    )
    scale = scale.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    scale = scale.reshape(
        batches,
        n_padded // _NON_K_PRESHUFFLE_BLOCK_SIZE,
        k_scale_padded * _NON_K_PRESHUFFLE_BLOCK_SIZE,
    )
    return scale.transpose(-1, -2)


def _swizzle_mxfp4_for_gfx950(
    quant_tensor: torch.Tensor,
    scale: torch.Tensor,
):
    quant_tensor = quant_tensor.transpose(-2, -1)
    quant_tensor = wrap_torch_tensor(quant_tensor, dtype=FP4)
    scale = _swizzle_scales_cdna4(scale)
    return quant_tensor, GluonFlexData(), scale


def _pad_w2_to_block_n(w: torch.nn.Module, block_n: int) -> None:
    original_n = int(w.w2_weight.shape[-2])
    w._w2_logical_n = original_n
    if original_n % block_n == 0:
        return

    n_padded = (original_n + block_n - 1) // block_n * block_n
    extra_n = n_padded - original_n
    w2_weight = w.w2_weight.data
    w2_scale = w.w2_weight_scale.data
    w.w2_weight = torch.nn.Parameter(
        torch.cat(
            [
                w2_weight,
                torch.zeros(
                    *w2_weight.shape[:-2],
                    extra_n,
                    w2_weight.shape[-1],
                    dtype=w2_weight.dtype,
                    device=w2_weight.device,
                ),
            ],
            dim=-2,
        ),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        torch.cat(
            [
                w2_scale,
                torch.full(
                    (*w2_scale.shape[:-2], extra_n, w2_scale.shape[-1]),
                    127,
                    dtype=w2_scale.dtype,
                    device=w2_scale.device,
                ),
            ],
            dim=-2,
        ),
        requires_grad=False,
    )

    if hasattr(w, "w2_weight_bias"):
        w2_bias = w.w2_weight_bias.data
        w.w2_weight_bias = torch.nn.Parameter(
            torch.cat(
                [
                    w2_bias,
                    torch.zeros(
                        *w2_bias.shape[:-1],
                        extra_n,
                        dtype=w2_bias.dtype,
                        device=w2_bias.device,
                    ),
                ],
                dim=-1,
            ),
            requires_grad=False,
        )


def _raw_storage(obj: Any) -> torch.Tensor:
    storage = getattr(obj, "storage", None)
    data = getattr(storage, "data", None)
    if isinstance(data, torch.Tensor):
        return data
    assert isinstance(obj, torch.Tensor)
    return obj


def _attach_gluon_bpreshuffle(w: torch.nn.Module) -> None:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
        shuffle_weight_for_gluon_dot_layout,
    )

    targets = (
        ("w13_weight_triton_tensor", None),
        ("w2_weight_triton_tensor", getattr(w, "_w2_logical_n", None)),
    )
    for attr, logical_n in targets:
        raw = _raw_storage(getattr(w, attr))
        try:
            shuffled = shuffle_weight_for_gluon_dot_layout(raw)
        except (AssertionError, ValueError):
            continue
        if logical_n is not None and logical_n != shuffled.shape[-1]:
            shuffled.original_n = int(logical_n)
            raw.original_n = int(logical_n)
        raw._gluon_shuffled = shuffled


def preprocess_gluon_mxfp4_gfx950_moe_weights(
    plan: dict,
    w: torch.nn.Module,
    *,
    preshuffle: bool = True,
) -> None:
    if preshuffle:
        _pad_w2_to_block_n(w, _GLUON_COMBINE_BLOCK_N)

    w13_weight_bias = w.w13_weight_bias.to(torch.float32)
    w2_weight_bias = w.w2_weight_bias.to(torch.float32)
    w.w13_weight_bias = torch.nn.Parameter(w13_weight_bias, requires_grad=False)
    w.w2_weight_bias = torch.nn.Parameter(w2_weight_bias, requires_grad=False)

    w13_weight, w13_flex, w13_scale = _swizzle_mxfp4_for_gfx950(
        w.w13_weight, w.w13_weight_scale
    )
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4_for_gfx950(
        w.w2_weight, w.w2_weight_scale
    )

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

    fp8_dtype = torch.float8_e4m3fn
    w13_lhs = GluonFlexData(dtype=fp8_dtype, scale=w13_in_scale)
    w2_lhs = GluonFlexData(dtype=fp8_dtype, scale=w2_in_scale)
    out_dtype = torch.bfloat16

    w.w13_precision_config = GluonPrecisionConfig(
        flex_ctx=GluonFlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
        b_mx_scale=w13_scale,
        b_microblock_size=_MXFP_BLOCK_SIZE,
        out_dtype=out_dtype,
    )
    w.w2_precision_config = GluonPrecisionConfig(
        flex_ctx=GluonFlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=_MXFP_BLOCK_SIZE,
        out_dtype=out_dtype,
    )

    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    del w.w13_weight
    del w.w2_weight

    if preshuffle:
        _attach_gluon_bpreshuffle(w)

    torch.cuda.empty_cache()
