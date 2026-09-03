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

"""Expand selected KPool rows into global FlatKV slots."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _kpool_expand_slots_kernel(
    out,
    out_stride,
    lens,
    pool_indices,
    pool_indices_stride,
    causal_lens,
    req_ids,
    block_table,
    block_table_stride,
    block_table_cols: tl.constexpr,
    block_size: tl.constexpr,
    pool_size: tl.constexpr,
    group_topk: tl.constexpr,
    out_cols: tl.constexpr,
    append_tail: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    req = tl.load(req_ids + token).to(tl.int32)
    seq_len = tl.maximum(tl.load(causal_lens + token).to(tl.int32), 0)
    num_pools = seq_len // pool_size
    tail_count = seq_len - num_pools * pool_size
    history_len = tl.minimum(num_pools * pool_size, group_topk * pool_size)
    select_all = num_pools <= group_topk

    count = tl.zeros((), dtype=tl.int32)
    for start in range(0, out_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_range = cols < out_cols
        pool_rank = cols // pool_size
        pool_slot = cols - pool_rank * pool_size

        is_history = in_range & (cols < history_len)
        selected_pool = tl.load(
            pool_indices + token * pool_indices_stride + pool_rank,
            mask=is_history & (pool_rank < group_topk),
            other=-1,
        ).to(tl.int32)
        pool = tl.where(select_all, pool_rank, selected_pool)
        valid = is_history & (pool >= 0) & (pool < num_pools)
        raw_slot = pool * pool_size + pool_slot

        if append_tail:
            is_tail = in_range & (cols >= history_len)
            is_tail &= cols < history_len + tail_count
            raw_slot = tl.where(
                is_tail, num_pools * pool_size + cols - history_len, raw_slot
            )
            valid |= is_tail

        valid &= (raw_slot >= 0) & (raw_slot < seq_len)
        page_idx = raw_slot // block_size
        valid &= page_idx < block_table_cols
        page = tl.load(
            block_table + req * block_table_stride + page_idx,
            mask=valid,
            other=0,
        ).to(tl.int32)
        global_slot = page * block_size + raw_slot - page_idx * block_size
        tl.store(
            out + token * out_stride + cols,
            tl.where(valid, global_slot, -1),
            mask=in_range,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(lens + token, count)


def expand_kpool_to_flat_kv(
    pool_indices: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    kv_page_size: int,
    append_tail: bool,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand request-local pool ids into global FlatKV slots.

    Args:
        pool_indices: Selected request-local pool ids.
        causal_lens: Visible raw-token lengths.
        req_ids: Request id for each query.
        kv_block_table: Raw-token cache page table.
        pool_size: Raw tokens represented by one pool.
        kv_page_size: Raw tokens in each cache page.
        append_tail: Include the visible incomplete pool.
        out: Optional caller-owned slot output.
        lens_out: Optional caller-owned valid-count output.

    Returns:
        Global FlatKV slots and valid counts.
    """
    num_tokens, topk_pools = pool_indices.shape
    width = topk_pools * pool_size + (pool_size - 1 if append_tail else 0)
    if kv_page_size <= 0:
        raise ValueError(f"kv_page_size must be positive, got {kv_page_size}")
    if kv_block_table.dim() != 2 or kv_block_table.shape[1] == 0:
        raise ValueError("kv_block_table must contain at least one page per request")

    if out is None:
        out = torch.empty(
            (num_tokens, width), dtype=torch.int32, device=pool_indices.device
        )
    elif (
        tuple(out.shape) != (num_tokens, width)
        or out.dtype != torch.int32
        or out.device != pool_indices.device
    ):
        raise ValueError(
            f"out must be int32 {(num_tokens, width)} on {pool_indices.device}"
        )
    if lens_out is None:
        lens_out = torch.empty(
            num_tokens, dtype=torch.int32, device=pool_indices.device
        )
    elif (
        tuple(lens_out.shape) != (num_tokens,)
        or lens_out.dtype != torch.int32
        or lens_out.device != pool_indices.device
    ):
        raise ValueError(
            f"lens_out must be int32 {(num_tokens,)} on {pool_indices.device}"
        )

    _kpool_expand_slots_kernel[(num_tokens,)](
        out,
        out.stride(0),
        lens_out,
        pool_indices,
        pool_indices.stride(0),
        causal_lens,
        req_ids,
        kv_block_table,
        kv_block_table.stride(0),
        kv_block_table.shape[1],
        block_size=kv_page_size,
        pool_size=pool_size,
        group_topk=topk_pools,
        out_cols=width,
        append_tail=bool(append_tail),
        BLOCK=min(triton.next_power_of_2(width), 1024),
        num_warps=8,
        num_stages=1,
    )
    return out, lens_out


__all__ = ["expand_kpool_to_flat_kv"]
