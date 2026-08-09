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

# The gdot128 preshuffle below must mirror the MFMA geometry the fused
# pipelined kernel reads with (fused/pipelined_program preshuffled-W layouts).
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _GLUON_DOT_K_QUAD,
    _GLUON_DOT_K_WIDTH,
    _GLUON_DOT_N_LANE,
    _GLUON_DOT_SUB_TILE_K,
    _extract_gluon_raw_w,
    _extract_gluon_raw_w_unshuffled,
)

# CDNA4 MXFP4 scale layout is defined once in mxfp4_cdna4_scale_layout so the
# weight-scale (B) preshuffle here and the activation-scale (A) gather in
# mxfp4.scale stays in lock-step. Local aliases preserve the
# historical private names used throughout this module.
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    MXFP4_BLOCK as _MXFP_BLOCK_SIZE,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    swizzle_cdna4_mxfp4_scale as _swizzle_cdna4_mxfp4_scale,
)

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


def shuffle_weight_for_gluon_dot_layout(
    w: torch.Tensor,
    *,
    block_k_pk: int = 128,
    block_n: int = 128,
) -> torch.Tensor:
    K_pk, N = w.shape[-2], w.shape[-1]

    if block_k_pk <= 0 or block_k_pk % _GLUON_DOT_SUB_TILE_K != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires block_k_pk "
            f"to be a positive multiple of {_GLUON_DOT_SUB_TILE_K} "
            f"(MFMA SUB_TILE_K); got {block_k_pk}."
        )
    if block_n <= 0 or block_n % _GLUON_DOT_N_LANE != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires block_n to "
            f"be a positive multiple of {_GLUON_DOT_N_LANE} (MFMA "
            f"N_LANE); got {block_n}."
        )
    # The preshuffled W path uses tile-level predicates instead of a full
    # per-element N mask, so N must be block_n-aligned. The combine GEMM pads
    # W + W-scale at the backend before this helper sees the tensor; we still
    # assert here to catch unaligned callers.
    if N % block_n != 0:
        raise ValueError(
            f"shuffle_weight_for_gluon_dot_layout requires N "
            f"divisible by block_n={block_n} (got N={N}); the kernel's "
            f"preshuffled W path assumes block_n-aligned N. Pad the raw W "
            f"and its e8m0 W-scale at the backend layer (W with zeros, "
            f"scale with 127 = identity) BEFORE calling "
            f"``swizzle_mxfp4`` and this helper; trim the kernel "
            f"output back to the logical N in the high-level launcher."
        )
    k_tile_bytes = block_k_pk * block_n
    # Zero-pad K_pk to a multiple of block_k_pk (kernel's k_limit_w
    # masks the tail); supports gpt-oss-120b H=2880 etc.
    K_pk_padded = (K_pk + block_k_pk - 1) // block_k_pk * block_k_pk
    N_CTA_TILES = N // block_n

    # In-tile dims: (n_block, k_block, k_quad, n_in_sub, k_within).
    k_block_dim = block_k_pk // _GLUON_DOT_SUB_TILE_K
    stride_n_in_sub = _GLUON_DOT_K_WIDTH
    stride_k_quad = _GLUON_DOT_N_LANE * _GLUON_DOT_K_WIDTH
    stride_k_block = _GLUON_DOT_K_QUAD * stride_k_quad
    stride_n_block = k_block_dim * stride_k_block

    # (k, n) -> shuffled HBM byte offset within one CTA tile.
    k = torch.arange(block_k_pk, dtype=torch.int64).view(-1, 1)
    n = torch.arange(block_n, dtype=torch.int64).view(1, -1)
    k_within = k % _GLUON_DOT_K_WIDTH
    k_quad = (k // _GLUON_DOT_K_WIDTH) % _GLUON_DOT_K_QUAD
    k_block = k // _GLUON_DOT_SUB_TILE_K
    n_in_sub = n % _GLUON_DOT_N_LANE
    n_block_in_tile = n // _GLUON_DOT_N_LANE
    in_tile_offset = (
        n_block_in_tile * stride_n_block
        + k_block * stride_k_block
        + k_quad * stride_k_quad
        + n_in_sub * stride_n_in_sub
        + k_within
    )

    # Across CTA tiles: tile (kt, nt) -> byte (kt * N_CTA_TILES + nt) * k_tile_bytes.
    K_grid_full = torch.arange(K_pk_padded, dtype=torch.int64).view(-1, 1)
    N_grid_full = torch.arange(N, dtype=torch.int64).view(1, -1)
    kt = K_grid_full // block_k_pk
    nt = N_grid_full // block_n
    k_in_tile = K_grid_full % block_k_pk
    n_in_tile = N_grid_full % block_n
    in_tile_2d = in_tile_offset[k_in_tile, n_in_tile]  # [K_pk_padded, N]
    P = (kt * N_CTA_TILES + nt) * k_tile_bytes + in_tile_2d

    leading_shape = list(w.shape[:-2])
    leading = 1
    for s in leading_shape:
        leading *= s
    w_kn = w.reshape(leading, K_pk, N)
    # Zero-pad K_pk -> K_pk_padded; the kernel's k_limit_w masks the tail.
    if K_pk_padded != K_pk:
        pad = torch.zeros(
            leading, K_pk_padded - K_pk, N, dtype=w.dtype, device=w.device
        )
        w_kn_padded = torch.cat([w_kn, pad], dim=-2)
    else:
        w_kn_padded = w_kn
    # K-innermost flat source: src[e, n*K_pk_padded + k] = W[e, k, n].
    w_nk_contig = w_kn_padded.transpose(-1, -2).contiguous()
    src_flat = w_nk_contig.reshape(leading, K_pk_padded * N)

    K_grid = (
        torch.arange(K_pk_padded, dtype=torch.int64).view(-1, 1).expand(K_pk_padded, N)
    )
    N_grid = torch.arange(N, dtype=torch.int64).view(1, -1).expand(K_pk_padded, N)
    src_idx_kn = (N_grid * K_pk_padded + K_grid).flatten().to(w.device)
    P_flat = P.flatten().to(w.device)

    src_in_kn_order = src_flat.index_select(-1, src_idx_kn)
    out_flat = torch.empty(leading, K_pk_padded * N, dtype=w.dtype, device=w.device)
    out_flat.scatter_(
        -1,
        P_flat.unsqueeze(0).expand_as(out_flat),
        src_in_kn_order,
    )

    # Shape is (..., K_pk_padded, N); ``k_limit_w`` (= original K_pk)
    # masks the padded tail. Launcher must pass logical K from X.
    out = out_flat.view(*leading_shape, K_pk_padded, N)
    out.is_shuffled_for_gluon_dot = True
    out.original_k_pk = K_pk
    out.gluon_dot_block_k_pk = block_k_pk
    out.gluon_dot_block_n = block_n
    return out


def _attach_gluon_preshuffle(w: torch.nn.Module) -> None:
    targets = (
        ("w13_weight_triton_tensor", None),
        ("w2_weight_triton_tensor", getattr(w, "_w2_logical_n", None)),
    )
    for attr, logical_n in targets:
        wrapped = getattr(w, attr, None)
        if wrapped is None:
            continue
        raw = _extract_gluon_raw_w(wrapped)
        try:
            shuffled = shuffle_weight_for_gluon_dot_layout(raw)
        except (AssertionError, ValueError):
            continue
        if logical_n is not None and int(logical_n) != int(shuffled.shape[-1]):
            shuffled.original_n = int(logical_n)
            raw.original_n = int(logical_n)
            wrapped.original_n = int(logical_n)
        setattr(w, attr, shuffled)


def _attach_w2_logical_n(w: torch.nn.Module) -> None:
    logical_n = getattr(w, "_w2_logical_n", None)
    wrapped = getattr(w, "w2_weight_triton_tensor", None)
    if logical_n is None or wrapped is None:
        return
    raw = _extract_gluon_raw_w_unshuffled(wrapped)
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
        w._gluon_w13_bias_is_zero = not bool(
            torch.count_nonzero(w13_weight_bias).item()
        )
        w.w13_weight_bias = torch.nn.Parameter(w13_weight_bias, requires_grad=False)
    if hasattr(w, "w2_weight_bias"):
        w2_weight_bias = w.w2_weight_bias.to(torch.float32)
        w._gluon_w2_bias_is_zero = not bool(torch.count_nonzero(w2_weight_bias).item())
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

    # Attach the MXFP4 package-prefill aliases whenever preshuffle
    # produced the gdot128 layout. They are metadata-only views over that storage
    # (no second model-sized copy), so attaching them is free. When preshuffle is
    # disabled the package path cannot run anyway (the dynamic entry returns None
    # without these aliases and falls back to the reference path).
    if preshuffle:
        from tokenspeed_kernel_amd.ops.gfx950.moe import mxfp4

        mxfp4.attach_prefill_aliases(w, w13_scale, w2_scale)

    torch.cuda.empty_cache()
