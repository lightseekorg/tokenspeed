# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""Fused Triton kernel for DFlash KV materialization (decode-only fast path).

Adapted from sglang's `fused_kv_materialize.py`. Provides two variants:

1. `_fused_norm_rope_stacked` — writes to workspace tensors (original).
2. `_fused_norm_rope_stacked_scatter` — writes directly into per-layer KV pool
   buffers (avoids intermediate workspace + per-layer scatter launches).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Module-level cache for per-layer buffer pointer tensors.
# KV pool buffers are allocated once and never reallocated, so data_ptr()
# stays valid for the entire server lifetime.
# ---------------------------------------------------------------------------

_cached_kv_ptrs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_kv_buffer_ptrs(
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build or retrieve cached per-layer buffer pointer tensors."""
    cache_key = k_buffers[0].data_ptr()
    if cache_key in _cached_kv_ptrs:
        return _cached_kv_ptrs[cache_key]
    device = k_buffers[0].device
    k_ptrs = torch.tensor(
        [b.data_ptr() for b in k_buffers], dtype=torch.int64, device=device
    )
    v_ptrs = torch.tensor(
        [b.data_ptr() for b in v_buffers], dtype=torch.int64, device=device
    )
    _cached_kv_ptrs[cache_key] = (k_ptrs, v_ptrs)
    return k_ptrs, v_ptrs


# ---------------------------------------------------------------------------
# Kernel: fused RMSNorm + RoPE + direct scatter into KV pool
# ---------------------------------------------------------------------------


@triton.jit
def _fused_norm_rope_scatter_kernel(
    kv_ptr,  # [total_ctx, n_layers, kv_size * 2]
    k_norm_weight_ptr,  # [n_layers, head_dim]
    eps_ptr,  # [n_layers]
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [total_ctx]
    loc_ptr,  # [total_ctx] — scatter destination slot indices
    k_buf_ptrs_ptr,  # [n_layers] — data_ptr per layer
    v_buf_ptrs_ptr,  # [n_layers] — data_ptr per layer
    inv_k_scale_ptr,  # [n_layers] float32 or dummy when USE_PROVIDED_SCALE=False
    inv_v_scale_ptr,  # [n_layers] float32 or dummy when USE_PROVIDED_SCALE=False
    kv_stride_ctx,
    kv_stride_layer,
    k_norm_weight_stride_layer,
    cos_sin_stride_pos,
    dst_row_stride,  # k_buffer[layer].stride(0), same for all layers
    total_ctx,
    n_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    IS_FP8: tl.constexpr,
    USE_PROVIDED_SCALE: tl.constexpr,
):
    """Fused RMSNorm(K) + RoPE(K) + scatter to KV pool. Grid: (total_ctx, num_kv_heads, n_layers)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    layer_id = tl.program_id(2)
    if ctx_id >= total_ctx or layer_id >= n_layers:
        return

    position = tl.load(positions_ptr + ctx_id)
    eps = tl.load(eps_ptr + layer_id).to(tl.float32)
    dst_slot = tl.load(loc_ptr + ctx_id).to(tl.int64)

    kv_base = kv_ptr + ctx_id * kv_stride_ctx + layer_id * kv_stride_layer
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim

    if IS_FP8:
        k_buf_ptr = tl.load(k_buf_ptrs_ptr + layer_id).to(
            tl.pointer_type(tl.float8e4nv)
        )
        v_buf_ptr = tl.load(v_buf_ptrs_ptr + layer_id).to(
            tl.pointer_type(tl.float8e4nv)
        )
    else:
        k_buf_ptr = tl.load(k_buf_ptrs_ptr + layer_id).to(tl.pointer_type(tl.bfloat16))
        v_buf_ptr = tl.load(v_buf_ptrs_ptr + layer_id).to(tl.pointer_type(tl.bfloat16))
    k_write = k_buf_ptr + dst_slot * dst_row_stride + head_id * head_dim
    v_write = v_buf_ptr + dst_slot * dst_row_stride + head_id * head_dim

    if USE_PROVIDED_SCALE:
        inv_k_scale = tl.load(inv_k_scale_ptr + layer_id).to(tl.float32)
        inv_v_scale = tl.load(inv_v_scale_ptr + layer_id).to(tl.float32)
    else:
        inv_k_scale = 1.0
        inv_v_scale = 1.0

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(
        k_norm_weight_ptr + layer_id * k_norm_weight_stride_layer + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr
        + layer_id * k_norm_weight_stride_layer
        + half_rotary_dim
        + offs,
        mask=mask_half,
        other=1.0,
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    if IS_FP8:
        v_out = (v_raw.to(tl.float32) * inv_v_scale).to(tl.float8e4nv)
        k_first_out = (k_rot_first * inv_k_scale).to(tl.float8e4nv)
        k_second_out = (k_rot_second * inv_k_scale).to(tl.float8e4nv)
        k_pass_out = (k_normed * inv_k_scale).to(tl.float8e4nv)
    else:
        v_out = v_raw
        k_first_out = k_rot_first.to(v_raw.dtype)
        k_second_out = k_rot_second.to(v_raw.dtype)
        k_pass_out = k_normed.to(v_raw.dtype)

    tl.store(v_write + offs, v_out, mask=mask_hd)
    tl.store(k_write + offs, k_first_out, mask=mask_half)
    tl.store(k_write + half_rotary_dim + offs, k_second_out, mask=mask_half)
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_pass_out, mask=mask_pass)


def _fused_norm_rope_stacked_scatter(
    kv: torch.Tensor,  # [total_ctx, n_layers, kv_size*2]
    k_norm_weight: torch.Tensor,  # [n_layers, head_dim]
    eps: torch.Tensor,  # [n_layers]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [total_ctx]
    loc: torch.Tensor,  # [total_ctx] — scatter slot indices
    k_buffers: list[torch.Tensor],  # per-layer k_buffer from KV pool
    v_buffers: list[torch.Tensor],  # per-layer v_buffer from KV pool
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    inv_k_scales: torch.Tensor | None = None,
    inv_v_scales: torch.Tensor | None = None,
) -> None:
    """Fused RMSNorm + RoPE + scatter into KV pool for all layers in one launch.

    When ``k_buffers[0].dtype`` is ``torch.float8_e4m3fn`` the kernel quantizes
    K and V to FP8 before the scatter store. If ``inv_k_scales`` and
    ``inv_v_scales`` are provided (per-layer float32 tensors of shape [n_layers]),
    values are multiplied by the inverse scale prior to FP8 conversion; otherwise
    a scale of 1.0 is used.
    """
    if kv.ndim != 3:
        raise ValueError(
            f"Expected 3D kv [total_ctx, n_layers, kv_size*2], got {tuple(kv.shape)}."
        )
    total_ctx, n_layers, kv_dim = kv.shape
    if total_ctx == 0:
        return

    kv_size = num_kv_heads * head_dim
    half_rotary_dim = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)
    k_ptrs, v_ptrs = _get_kv_buffer_ptrs(k_buffers, v_buffers)
    dst_row_stride = k_buffers[0].stride(0)

    is_fp8 = k_buffers[0].dtype == torch.float8_e4m3fn
    use_provided_scale = inv_k_scales is not None and inv_v_scales is not None
    if use_provided_scale:
        inv_k_arg = inv_k_scales
        inv_v_arg = inv_v_scales
    else:
        # Dummy pointer; kernel does not dereference when USE_PROVIDED_SCALE=False.
        inv_k_arg = eps
        inv_v_arg = eps

    _fused_norm_rope_scatter_kernel[(total_ctx, num_kv_heads, n_layers)](
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        loc,
        k_ptrs,
        v_ptrs,
        inv_k_arg,
        inv_v_arg,
        kv.stride(0),
        kv.stride(1),
        k_norm_weight.stride(0),
        cos_sin_cache.stride(0),
        dst_row_stride,
        total_ctx,
        n_layers,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        half_rotary_dim,
        BLOCK_HD,
        is_fp8,
        use_provided_scale,
    )


# ---------------------------------------------------------------------------
# Original kernel (writes to workspace, kept as fallback)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_norm_rope_kernel_stacked(
    kv_ptr,
    k_norm_weight_ptr,
    eps_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    k_out_ptr,
    v_out_ptr,
    kv_stride_ctx,
    kv_stride_layer,
    k_norm_weight_stride_layer,
    cos_sin_stride_pos,
    k_out_stride_layer,
    k_out_stride_ctx,
    k_out_stride_head,
    v_out_stride_layer,
    v_out_stride_ctx,
    v_out_stride_head,
    total_ctx,
    n_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads, n_layers)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    layer_id = tl.program_id(2)
    if ctx_id >= total_ctx or layer_id >= n_layers:
        return

    position = tl.load(positions_ptr + ctx_id)
    eps = tl.load(eps_ptr + layer_id).to(tl.float32)
    kv_base = kv_ptr + ctx_id * kv_stride_ctx + layer_id * kv_stride_layer
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    k_write = (
        k_out_ptr
        + layer_id * k_out_stride_layer
        + ctx_id * k_out_stride_ctx
        + head_id * k_out_stride_head
    )
    v_write = (
        v_out_ptr
        + layer_id * v_out_stride_layer
        + ctx_id * v_out_stride_ctx
        + head_id * v_out_stride_head
    )

    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(
        k_norm_weight_ptr + layer_id * k_norm_weight_stride_layer + offs,
        mask=mask_hd,
        other=1.0,
    ).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr
        + layer_id * k_norm_weight_stride_layer
        + half_rotary_dim
        + offs,
        mask=mask_half,
        other=1.0,
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    tl.store(v_write + offs, v_raw, mask=mask_hd)
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


def _fused_norm_rope_stacked(
    kv: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + RoPE materialization for all layers (workspace variant)."""
    if kv.ndim != 3:
        raise ValueError(
            f"Expected 3D kv [total_ctx, n_layers, kv_size*2], got {tuple(kv.shape)}."
        )

    total_ctx, n_layers, kv_dim = kv.shape
    if total_ctx == 0:
        empty = torch.empty(
            (n_layers, 0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        return empty, empty

    kv_size = num_kv_heads * head_dim
    half_rotary_dim = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)
    expected_shape = (n_layers, total_ctx, num_kv_heads, head_dim)
    if k_out is None:
        k_out = torch.empty(expected_shape, dtype=kv.dtype, device=kv.device)
    if v_out is None:
        v_out = torch.empty_like(k_out)

    _fused_norm_rope_kernel_stacked[(total_ctx, num_kv_heads, n_layers)](
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        k_out,
        v_out,
        kv.stride(0),
        kv.stride(1),
        k_norm_weight.stride(0),
        cos_sin_cache.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        total_ctx,
        n_layers,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        half_rotary_dim,
        BLOCK_HD,
    )
    return k_out, v_out
