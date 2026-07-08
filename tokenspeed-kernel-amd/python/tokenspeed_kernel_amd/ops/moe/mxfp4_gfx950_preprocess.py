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

_MXFP_BLOCK_SIZE = 32
_GLUON_COMBINE_BLOCK_N = 128
_CDNA4_SCALE_N_BLOCK = 32
_CDNA4_SCALE_K_BLOCK = 8


@dataclass
class InFlexData:
    dtype: torch.dtype | None = None
    scale: torch.Tensor | None = None


@dataclass
class FlexCtx:
    lhs_data: InFlexData | None = None
    rhs_data: InFlexData | None = None


@dataclass
class PrecisionConfig:
    flex_ctx: FlexCtx | None = None
    b_mx_scale: Any | None = None
    b_microblock_size: int | None = None
    out_dtype: torch.dtype | None = None
    a_mx_scale: Any | None = None
    a_microblock_size: int | None = None


def _make_k_packed_mxfp4_weight(quant_tensor: torch.Tensor) -> torch.Tensor:
    """Return W storage in the gfx950 apply kernel's canonical K-packed layout."""
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


def _swizzle_cdna4_mxfp4_scale(scale: torch.Tensor) -> torch.Tensor:
    """Match triton_kernels' CDNA4MXScaleLayout byte swizzle using PyTorch ops."""
    if scale.ndim < 2:
        raise ValueError("MXFP4 scale tensor must have at least 2 dimensions")
    scale = scale.transpose(-2, -1).contiguous()
    *leading_shape, k_scale, n = scale.shape
    leading = 1
    for dim in leading_shape:
        leading *= dim
    k_scale_pad = (k_scale + _CDNA4_SCALE_K_BLOCK - 1) // _CDNA4_SCALE_K_BLOCK
    k_scale_pad *= _CDNA4_SCALE_K_BLOCK
    n_pad = (n + _CDNA4_SCALE_N_BLOCK - 1) // _CDNA4_SCALE_N_BLOCK
    n_pad *= _CDNA4_SCALE_N_BLOCK

    scale = scale.mT.contiguous().mT
    scale = torch.nn.functional.pad(scale, (0, n_pad - n, 0, k_scale_pad - k_scale))
    scale = scale.transpose(-1, -2)
    scale = scale.reshape(leading, n_pad, k_scale_pad)
    scale = scale.view(leading, n_pad // 32, 2, 16, k_scale_pad // 8, 2, 4, 1)
    scale = scale.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    scale = scale.reshape(leading, n_pad // 32, k_scale_pad * 32)
    scale = scale.transpose(-1, -2)
    assert scale.stride(-2) == 1
    return scale


def _swizzle_mxfp4(quant_tensor: torch.Tensor, scale: torch.Tensor, num_warps: int):
    """Weight swizzle for mxfp4 MoE, used for OAI mxfp4 kernel."""
    del num_warps
    quant_tensor = _make_k_packed_mxfp4_weight(quant_tensor)
    scale = _swizzle_cdna4_mxfp4_scale(scale)
    return quant_tensor, InFlexData(), scale


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


def _attach_gluon_preshuffle(w: torch.nn.Module) -> None:
    from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950

    targets = (
        ("w13_weight_triton_tensor", None),
        ("w2_weight_triton_tensor", getattr(w, "_w2_logical_n", None)),
    )
    for attr, logical_n in targets:
        wrapped = getattr(w, attr, None)
        if wrapped is None:
            continue
        raw = fused_mxfp_gfx950._extract_gluon_raw_w(wrapped)
        try:
            shuffled = fused_mxfp_gfx950.shuffle_weight_for_gluon_dot_layout(raw)
        except (AssertionError, ValueError):
            continue
        if logical_n is not None and int(logical_n) != int(shuffled.shape[-1]):
            shuffled.original_n = int(logical_n)
            raw.original_n = int(logical_n)
            wrapped.original_n = int(logical_n)
        setattr(w, attr, shuffled)


def _attach_w2_logical_n(w: torch.nn.Module) -> None:
    from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950

    logical_n = getattr(w, "_w2_logical_n", None)
    wrapped = getattr(w, "w2_weight_triton_tensor", None)
    if logical_n is None or wrapped is None:
        return
    raw = fused_mxfp_gfx950._extract_gluon_raw_w_unshuffled(wrapped)
    if int(logical_n) != int(raw.shape[-1]):
        raw.original_n = int(logical_n)
        wrapped.original_n = int(logical_n)


def preprocess_gluon_mxfp4_gfx950_moe_weights(
    plan: dict,
    w: torch.nn.Module,
    *,
    preshuffle: bool = True,
) -> None:
    _pad_w2_to_block_n(w, _GLUON_COMBINE_BLOCK_N)

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
        _release_parameter(w, "w13_weight")
        w13_weight_scale = torch.nn.Parameter(
            _interleave_gate_up_rows(w13_weight_scale.data, dim=-2),
            requires_grad=False,
        )
        _release_parameter(w, "w13_weight_scale")

    if hasattr(w, "w13_weight_bias"):
        w13_weight_bias = w.w13_weight_bias.to(torch.float32)
        if w13_layout == "concatenated":
            w13_weight_bias = _interleave_gate_up_rows(w13_weight_bias, dim=-1)
        w.w13_weight_bias = torch.nn.Parameter(w13_weight_bias, requires_grad=False)
    if hasattr(w, "w2_weight_bias"):
        w2_weight_bias = w.w2_weight_bias.to(torch.float32)
        w.w2_weight_bias = torch.nn.Parameter(w2_weight_bias, requires_grad=False)

    num_warps = 8
    w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
        w13_weight, w13_weight_scale, num_warps
    )
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
        w.w2_weight, w.w2_weight_scale, num_warps
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

        fp8_dtype = torch.float8_e4m3fn
        w13_lhs = InFlexData(dtype=fp8_dtype, scale=w13_in_scale)
        w2_lhs = InFlexData(dtype=fp8_dtype, scale=w2_in_scale)
    else:
        w13_lhs = InFlexData()
        w2_lhs = InFlexData()
    out_dtype = torch.bfloat16

    w.w13_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
        b_mx_scale=w13_scale,
        b_microblock_size=_MXFP_BLOCK_SIZE,
        out_dtype=out_dtype,
    )
    w.w2_precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=_MXFP_BLOCK_SIZE,
        out_dtype=out_dtype,
    )

    w.w13_weight_triton_tensor = w13_weight
    w.w2_weight_triton_tensor = w2_weight
    _attach_w2_logical_n(w)
    _release_parameter(w, "w13_weight")
    _release_parameter(w, "w2_weight")
    _release_parameter(w, "w13_weight_scale")
    _release_parameter(w, "w2_weight_scale")

    if preshuffle:
        _attach_gluon_preshuffle(w)

    torch.cuda.empty_cache()
