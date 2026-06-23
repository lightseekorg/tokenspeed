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
from tokenspeed_kernel_amd._triton import redirect_triton_to_tokenspeed_triton

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.matmul_details  # noqa: F401
    import triton_kernels.matmul_details.opt_flags  # noqa: F401
    import triton_kernels.numerics  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401

import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout

_MXFP_BLOCK_SIZE = 32
_GLUON_COMBINE_BLOCK_N = 128


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


def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    """Weight swizzle for mxfp4 MoE, used for OAI mxfp4 kernel."""

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    # This backend preprocessor is gfx950-specific. Fix block_k=256 to support
    # scale swizzling, matching the upstream Gluon MXFP4 path on AMD.
    opt_flags.update_opt_flags_constraints({"block_k": 256})
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


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
        raw._gluon_shuffled = shuffled
        wrapped._gluon_shuffled = shuffled


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

    w13_weight_bias = w.w13_weight_bias.to(torch.float32)
    w2_weight_bias = w.w2_weight_bias.to(torch.float32)
    w.w13_weight_bias = torch.nn.Parameter(w13_weight_bias, requires_grad=False)
    w.w2_weight_bias = torch.nn.Parameter(w2_weight_bias, requires_grad=False)

    num_warps = 8
    w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
        w.w13_weight, w.w13_weight_scale, num_warps
    )
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
        w.w2_weight, w.w2_weight_scale, num_warps
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
    w13_lhs = InFlexData(dtype=fp8_dtype, scale=w13_in_scale)
    w2_lhs = InFlexData(dtype=fp8_dtype, scale=w2_in_scale)
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
    del w.w13_weight
    del w.w2_weight

    if preshuffle:
        _attach_gluon_preshuffle(w)

    torch.cuda.empty_cache()
