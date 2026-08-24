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

"""Windowed compressed-state reduction for DeepSeek V4.

Each compressed token folds a window of paged state rows into one row: the
score half of each row gates the value half, softmax-normalized **along the
window** independently per dimension, then RMS-normalized. The reference walks
that as roughly twenty-five tensor ops -- page-table lookup, validity masks,
two gathers, a softmax, a weighted sum and the norm -- which at decode widths
costs far more in launches than in arithmetic.

The window is ``2 * compress_ratio`` wide with overlap, so 256 rows against a
512-wide head at the model's larger ratio: too much to hold at once. The
reduction therefore streams the window in tiles under a running max and sum,
the same way attention streams keys, which keeps the whole thing in one launch
regardless of ratio.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["deepseek_v4_compress_state_windows"]

_NEG = -1.0e30


@triton.jit(do_not_specialize=["num_tokens", "max_pages", "max_base"])
def _deepseek_v4_compress_windows_kernel(
    state_cache_ptr,
    block_table_ptr,
    base_offsets_ptr,
    positions_ptr,
    req_idx_ptr,
    slots_ptr,
    rms_weight_ptr,
    out_ptr,
    valid_token_ptr,
    num_tokens,
    max_pages,
    max_base,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_dim,
    block_table_stride,
    state_width,
    HEAD_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    OVERLAP: tl.constexpr,
    HAS_BASE: tl.constexpr,
    EPS: tl.constexpr,
    NEG: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    position = tl.load(positions_ptr + pid).to(tl.int64)
    slot = tl.load(slots_ptr + pid).to(tl.int64)
    req = tl.maximum(tl.load(req_idx_ptr + pid).to(tl.int64), 0)

    # A token only produces a compressed row on the ratio boundary.
    tl.store(
        valid_token_ptr + pid,
        (slot >= 0) & (((position + 1) % COMPRESS_RATIO) == 0),
    )

    base_page = tl.zeros((), tl.int64)
    if HAS_BASE:
        # The base table is sized by request count, which padded or stale slots
        # can index past; clamp it the way the tensor-op reference does.
        safe_req = tl.minimum(req, max_base - 1)
        base_page = tl.load(base_offsets_ptr + safe_req).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    run_max = tl.full((HEAD_DIM,), NEG, tl.float32)
    run_sum = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for w0 in range(0, WINDOW, BLOCK_W):
        offs_w = w0 + tl.arange(0, BLOCK_W)
        in_window = offs_w < WINDOW
        window_pos = position - WINDOW + 1 + offs_w

        table_idx = window_pos // CACHE_BLOCK - base_page
        live = (
            in_window
            & (window_pos >= 0)
            & (table_idx >= 0)
            & (table_idx < max_pages)
        )
        block_number = tl.load(
            block_table_ptr + req * block_table_stride + table_idx,
            mask=live,
            other=-1,
        ).to(tl.int64)
        live = live & (block_number >= 0)

        block_number = tl.where(live, block_number, 0)
        pos_in_block = tl.maximum(window_pos, 0) % CACHE_BLOCK
        row = (
            state_cache_ptr
            + block_number * cache_stride_block
            + pos_in_block * cache_stride_pos
        )

        # With overlap the second half of the window reads the next head's
        # columns, which is what the reference's head_offsets encodes.
        col = offs_d
        if OVERLAP:
            col = tl.where(offs_w >= COMPRESS_RATIO, HEAD_DIM, 0)[:, None] + offs_d[
                None, :
            ]
        else:
            col = tl.zeros((BLOCK_W, 1), tl.int32) + offs_d[None, :]

        alive = live[:, None]
        # The cache is a view into a shared arena, so the feature dimension is
        # not guaranteed to be unit-stride.
        values = tl.load(
            row[:, None] + col * cache_stride_dim, mask=alive, other=0.0
        ).to(tl.float32)
        scores = tl.load(
            row[:, None] + (state_width + col) * cache_stride_dim,
            mask=alive,
            other=NEG,
        ).to(tl.float32)
        scores = tl.where(alive, scores, NEG)

        # Softmax runs along the window, independently for every dimension.
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(run_max, tile_max)
        rescale = tl.exp(run_max - new_max)
        p = tl.exp(scores - new_max[None, :])
        p = tl.where(alive, p, 0.0)
        run_sum = run_sum * rescale + tl.sum(p, axis=0)
        acc = acc * rescale + tl.sum(p * values, axis=0)
        run_max = new_max

    compressed = acc / tl.where(run_sum > 0.0, run_sum, 1.0)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_DIM
    normed = compressed * tl.math.rsqrt(variance + EPS)
    normed = normed * tl.load(rms_weight_ptr + offs_d).to(tl.float32)
    tl.store(out_ptr + pid * HEAD_DIM + offs_d, normed)


def deepseek_v4_compress_state_windows(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold each token's state window into one RMS-normalized row.

    Args:
        state_cache: ``[num_blocks, block_size, 2 * state_width]`` paged state;
            the first half of each row holds values, the second half scores.
        token_to_req_indices: ``[num_tokens]`` owning request per token.
        positions: ``[num_tokens]`` absolute position per token.
        compressor_slot_mapping: ``[num_tokens]`` destination slot; negative
            marks a token that produces no compressed row.
        block_table: ``[num_reqs, max_pages]`` logical-to-physical page map.
        block_table_base_offsets: optional ``[num_reqs]`` first logical page.
        compressor_block_size: tokens per state-cache block.
        rms_norm_weight: ``[head_dim]`` normalization weight.
        rms_norm_eps: epsilon inside the reciprocal square root.
        compress_ratio: tokens folded per compressed row.
        head_dim: width of the compressed row.
        overlap: read the second window half from the next head's columns.

    Returns:
        ``(compressed, valid_token)`` with a float32 ``[num_tokens, head_dim]``
        result and a bool ``[num_tokens]`` mask of the rows that are real.
    """
    device = state_cache.device
    num_tokens = min(int(compressor_slot_mapping.numel()), int(positions.numel()))
    if num_tokens == 0:
        return (
            torch.empty((0, head_dim), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.bool),
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")

    window = (2 if overlap else 1) * compress_ratio
    out = torch.empty((num_tokens, head_dim), device=device, dtype=torch.float32)
    valid = torch.empty((num_tokens,), device=device, dtype=torch.bool)

    _deepseek_v4_compress_windows_kernel[(num_tokens,)](
        state_cache,
        block_table,
        block_table_base_offsets,
        positions[:num_tokens].contiguous(),
        token_to_req_indices[:num_tokens].contiguous(),
        compressor_slot_mapping[:num_tokens].contiguous(),
        rms_norm_weight,
        out,
        valid,
        num_tokens,
        int(block_table.shape[1]),
        (
            int(block_table_base_offsets.shape[0])
            if block_table_base_offsets is not None
            else 1
        ),
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        block_table.stride(0),
        state_cache.shape[-1] // 2,
        HEAD_DIM=head_dim,
        WINDOW=window,
        BLOCK_W=min(window, 32),
        COMPRESS_RATIO=compress_ratio,
        CACHE_BLOCK=compressor_block_size,
        OVERLAP=overlap,
        HAS_BASE=block_table_base_offsets is not None,
        EPS=rms_norm_eps,
        NEG=_NEG,
        num_warps=4,
        num_stages=1,
    )
    return out, valid
