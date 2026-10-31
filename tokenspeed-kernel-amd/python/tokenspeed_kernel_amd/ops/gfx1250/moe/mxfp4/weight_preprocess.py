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
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (
    _PREFILL_DOWN_TILES,
    PrecisionConfig,
    decode_tile_config,
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


def kplane_mxfp4_weight_gfx1250(weight: torch.Tensor, plane_bytes: int) -> torch.Tensor:
    """Store a ``(E, K_packed, N)`` N-major MXFP4 weight one K step per plane.

    Each ``plane_bytes`` step of K becomes its own row-major ``[N,
    plane_bytes]`` slab, so a ``[block_n, plane_bytes]`` weight tile is one
    contiguous run. A short K (Kimi K3's 384-wide TP8 down projection) would
    otherwise stream as 64-byte pieces of 192-byte rows, which TDM reads at a
    fraction of the contiguous bandwidth.

    The returned tensor keeps the ``(E, K_packed, N)`` shape and the expert and
    N strides kernels read, but its K addressing is only meaningful to kernels
    honouring the ``kplane_rows``/``kplane_bytes`` attributes it carries.
    """
    experts, k_packed, n = weight.shape
    if k_packed % plane_bytes:
        raise ValueError(f"K_packed {k_packed} is not a multiple of {plane_bytes}")
    rows = weight.transpose(1, 2)
    if not rows.is_contiguous():
        raise ValueError("expected an N-major (E, K_packed, N) weight")
    planes = (
        rows.reshape(experts, n, k_packed // plane_bytes, plane_bytes)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    view = torch.as_strided(
        planes, (experts, k_packed, n), (k_packed * n, 1, plane_bytes)
    )
    view.act_scale = getattr(weight, "act_scale", None)
    view.kplane_rows = n
    view.kplane_bytes = plane_bytes
    return view


def preprocess_gluon_mxfp4_gfx1250_situ_moe_weights(
    plan: dict,
    w: torch.nn.Module,
) -> None:
    """:func:`preprocess_gluon_mxfp4_gfx1250_moe_weights` for the FP8 SiTU MoE.

    That MoE runs its down projection with the decode tiles and
    ``_PREFILL_DOWN_TILES``. When both step K by the same width and tile N
    exactly, the down weight is stored in K planes of that width.
    """
    preprocess_gluon_mxfp4_gfx1250_moe_weights(plan, w)
    w2 = w.w2_weight_triton_tensor
    k = int(w2.shape[1]) * 2
    n = int(w2.shape[2])
    decode = decode_tile_config(k)
    block_k = decode["block_k"]
    if (
        block_k == _PREFILL_DOWN_TILES["block_k"]
        and k % block_k == 0
        and n % decode["block_n"] == 0
        and n % _PREFILL_DOWN_TILES["block_n"] == 0
        and k // block_k > 1
    ):
        w.w2_weight_triton_tensor = kplane_mxfp4_weight_gfx1250(w2, block_k // 2)
        torch.cuda.empty_cache()


__all__ = [
    "kplane_mxfp4_weight_gfx1250",
    "preprocess_gluon_mxfp4_gfx1250_situ_moe_weights",
    "preprocess_gluon_mxfp4_gfx1250_moe_weights",
    "_make_k_packed_mxfp4_weight",
]
