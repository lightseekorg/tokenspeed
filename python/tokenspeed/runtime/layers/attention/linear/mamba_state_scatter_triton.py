# Adapted from fla-org/flash-linear-attention
# This file has been modified for this repository.
# License: https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
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

"""Fused Triton kernels for Mamba state copy and zero operations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mamba_state_snapshot_kernel(
    pool_ptr,
    src_indices_ptr,  # [num_valid]
    dst_indices_ptr,  # [num_valid]
    cache_lengths_ptr,  # [num_valid] or nullptr (0 when page_size==0)
    page_size,  # 0 means no page filtering
    elem_per_entry: tl.constexpr,
    layer_stride,
    req_stride,
    pool_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    In-place copy kernel: pool[:, dst[i], :] = pool[:, src[i], :]
    Skips copy if page_size > 0 and cache_lengths[i] % page_size != 0.

    Grid: (num_valid, num_layers) — loops over elem_per_entry internally.
    Invalid entries early-return wasting only 1 block instead of
    ceil(elem_per_entry / BLOCK_SIZE) blocks.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)

    src_idx = tl.load(src_indices_ptr + pid_req).to(tl.int64)
    dst_idx = tl.load(dst_indices_ptr + pid_req).to(tl.int64)

    # Skip self-copy (no-op)
    if src_idx == dst_idx:
        return

    # Page-boundary filter: skip if not aligned
    if page_size > 0:
        cl = tl.load(cache_lengths_ptr + pid_req).to(tl.int64)
        if cl % page_size != 0:
            return

    # Bounds check
    if not (
        (src_idx >= 0) & (src_idx < pool_size) & (dst_idx >= 0) & (dst_idx < pool_size)
    ):
        return

    src_offset = pid_layer * layer_stride + src_idx * req_stride
    dst_offset = pid_layer * layer_stride + dst_idx * req_stride

    for start in tl.static_range(0, elem_per_entry, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elem_per_entry
        data = tl.load(pool_ptr + src_offset + offsets, mask=mask)
        tl.store(pool_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_copy(
    pool: torch.Tensor,  # [num_layers, pool_size, *state_shape] or [pool_size, *state_shape]
    src_indices: torch.Tensor,  # [num_valid]
    dst_indices: torch.Tensor,  # [num_valid]
    cache_lengths: torch.Tensor | None = None,  # [num_valid], for page filter
    page_size: int = 0,  # 0 means no page filtering
):
    """
    Copy mamba states: pool[:, dst_indices[i], :] = pool[:, src_indices[i], :]

    Handles both COW copy and checkpoint snapshot. Invalid indices (< 0 or
    >= pool_size) are skipped inside the kernel. When page_size > 0 and
    cache_lengths is provided, also skips entries where
    cache_lengths[i] % page_size != 0.

    Supports both 3D pool tensors ``[num_layers, pool_size, *state_shape]``
    and 2D per-layer slices ``[pool_size, *state_shape]`` (which may be
    non-contiguous views into a larger cache).  For 2D inputs the kernel
    is launched with ``num_layers=1``.

    Args:
        pool: State tensor, either 3D [num_layers, pool_size, *state_shape]
            or 2D [pool_size, *state_shape].
        src_indices: Source slot indices [num_valid], int32 or int64.
        dst_indices: Destination slot indices [num_valid], int32 or int64.
        cache_lengths: Per-entry cache lengths for page-boundary filtering.
        page_size: When > 0, only copy entries where cache_lengths[i] is
            aligned to page_size. Set to 0 to disable filtering (used by
            COW copy where all valid entries must be copied).
    """
    num_valid = src_indices.shape[0]
    if num_valid == 0:
        return

    if not pool.is_cuda:
        raise ValueError("fused_mamba_state_copy only supports CUDA tensors.")
    if pool.ndim < 2:
        raise ValueError(f"pool must be at least 2D, got {pool.ndim}D")
    if src_indices.shape[0] != dst_indices.shape[0]:
        raise ValueError(
            f"indices length mismatch: {src_indices.shape[0]} vs {dst_indices.shape[0]}"
        )

    if pool.ndim == 2:
        num_layers = 1
        pool_size = pool.shape[0]
        elem_per_entry = pool.numel() // pool_size
        layer_stride = 0  # unused when num_layers=1 (pid_layer is always 0)
        req_stride = pool.stride(0)
    else:
        num_layers = pool.shape[0]
        pool_size = pool.shape[1]
        elem_per_entry = pool.numel() // (num_layers * pool_size)
        layer_stride = pool.stride(0)
        req_stride = pool.stride(1)

    if not src_indices.is_contiguous():
        src_indices = src_indices.contiguous()
    if not dst_indices.is_contiguous():
        dst_indices = dst_indices.contiguous()
    src_indices = src_indices.to(torch.int32)
    dst_indices = dst_indices.to(torch.int32)

    if page_size > 0 and cache_lengths is not None:
        cache_lengths = cache_lengths.to(torch.int32)
    else:
        cache_lengths = src_indices  # unused; kernel skips when page_size==0
        page_size = 0

    BLOCK_SIZE = 8192
    grid = (num_valid, num_layers)

    _mamba_state_snapshot_kernel[grid](
        pool,
        src_indices,
        dst_indices,
        cache_lengths,
        page_size,
        elem_per_entry,
        layer_stride,
        req_stride,
        pool_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )


@triton.jit
def _mamba_state_zero_kernel(
    pool_ptr,
    indices_ptr,  # [bs] — indices to zero; negative values are skipped
    elem_per_entry: tl.constexpr,
    layer_stride,
    req_stride,
    pool_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Zero kernel: pool[:, indices[i], :] = 0
    Skips entries where indices[i] < 0 or indices[i] >= pool_size.

    Grid: (bs, num_layers) — loops over elem_per_entry internally.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)

    idx = tl.load(indices_ptr + pid_req).to(tl.int64)

    # Skip invalid entries (negative sentinel from torch.where)
    if (idx < 0) | (idx >= pool_size):
        return

    dst_offset = pid_layer * layer_stride + idx * req_stride

    zero_val = tl.zeros([BLOCK_SIZE], dtype=pool_ptr.dtype.element_ty)
    for start in tl.static_range(0, elem_per_entry, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elem_per_entry
        tl.store(pool_ptr + dst_offset + offsets, zero_val, mask=mask)


def fused_mamba_state_zero(
    pool: torch.Tensor,  # [num_layers, pool_size, *state_shape]
    indices: torch.Tensor,  # [bs] — slots to zero; negative values skipped inside kernel
):
    """
    Zero mamba states: pool[:, indices[i], :] = 0 for valid indices.

    Invalid indices (< 0) are skipped inside the kernel, avoiding any
    CPU-GPU synchronization from boolean indexing or .any() checks.

    Args:
        pool: State tensor [num_layers, pool_size, *state_shape], must be contiguous.
        indices: Slot indices [bs], int64. Negative values are treated as invalid
            and skipped (no-op).
    """
    bs = indices.shape[0]
    if bs == 0:
        return

    if not pool.is_cuda:
        raise ValueError("fused_mamba_state_zero only supports CUDA tensors.")
    if not pool.is_contiguous():
        raise ValueError("pool tensor must be contiguous")
    if pool.ndim < 2:
        raise ValueError(f"pool must be at least 2D, got {pool.ndim}D")

    num_layers = pool.shape[0]
    pool_size = pool.shape[1]
    elem_per_entry = pool.numel() // (num_layers * pool_size)

    layer_stride = pool.stride(0)
    req_stride = pool.stride(1)

    if not indices.is_contiguous():
        indices = indices.contiguous()

    BLOCK_SIZE = 8192
    grid = (bs, num_layers)

    _mamba_state_zero_kernel[grid](
        pool,
        indices,
        elem_per_entry,
        layer_stride,
        req_stride,
        pool_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
