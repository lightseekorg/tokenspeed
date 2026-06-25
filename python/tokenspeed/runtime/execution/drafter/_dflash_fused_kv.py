"""Fused Triton kernel for DFlash KV materialization (decode-only fast path).

Adapted from sglang's `fused_kv_materialize.py`. Provides two variants:

1. `_fused_norm_rope_stacked` — writes to workspace tensors (original).
2. `_fused_norm_rope_stacked_scatter` — writes directly into per-layer KV pool
   buffers (avoids intermediate workspace + per-layer scatter launches).
"""

from __future__ import annotations

from typing import Optional

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

    k_buf_ptr = tl.load(k_buf_ptrs_ptr + layer_id).to(tl.pointer_type(tl.bfloat16))
    v_buf_ptr = tl.load(v_buf_ptrs_ptr + layer_id).to(tl.pointer_type(tl.bfloat16))
    k_write = k_buf_ptr + dst_slot * dst_row_stride + head_id * head_dim
    v_write = v_buf_ptr + dst_slot * dst_row_stride + head_id * head_dim

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
) -> None:
    """Fused RMSNorm + RoPE + scatter into KV pool for all layers in one launch."""
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

    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if loc.dtype != torch.int64:
        loc = loc.to(torch.int64)

    k_ptrs, v_ptrs = _get_kv_buffer_ptrs(k_buffers, v_buffers)
    dst_row_stride = k_buffers[0].stride(0)

    _fused_norm_rope_scatter_kernel[(total_ctx, num_kv_heads, n_layers)](
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        loc,
        k_ptrs,
        v_ptrs,
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
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
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

    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

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
