# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""DSV4 cache slot mappings, active-page validation, and decode metadata."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv4._triton.common import (
    _as_int32_block_table,
)


@triton.jit
def _dsv4_compressed_slot_mapping_kernel(
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    pad_id: tl.constexpr,
    candidate_block: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + req_idx).to(tl.int32)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int32)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    start_pos = seq_len - query_len

    for i in range(0, query_len, candidate_block):
        offsets = i + tl.arange(0, candidate_block)
        mask = offsets < query_len
        pos = start_pos + offsets
        valid = (pos + 1) % compress_ratio == 0
        compressed_pos = pos // compress_ratio
        block_ids = compressed_pos // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_ids,
            mask=mask & valid,
            other=0,
        ).to(tl.int64)
        slot_ids = block_numbers * block_size + compressed_pos % block_size
        values = tl.where(valid, slot_ids, pad_id)
        tl.store(slot_mapping_ptr + query_start + offsets, values, mask=mask)


def dsv4_compressed_slot_mapping(
    *,
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build compressed KV slot mapping for DeepSeek V4."""

    if out is None:
        out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
    out.fill_(-1)
    slot_mapping = out[:num_tokens]
    if num_tokens == 0:
        return slot_mapping

    _dsv4_compressed_slot_mapping_kernel[(block_table.shape[0],)](
        slot_mapping,
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        block_table.stride(0),
        block_size=block_size,
        compress_ratio=compress_ratio,
        pad_id=-1,
        candidate_block=1024,
    )
    return slot_mapping


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_reqs",
        "max_blocks_per_seq",
    ]
)
def _dsv4_compact_compressed_slot_mapping_kernel(
    slot_mapping_ptr,
    token_to_req_indices_ptr,
    token_to_req_indices_stride,
    query_start_loc_ptr,
    query_start_loc_stride,
    seq_lens_ptr,
    seq_lens_stride,
    is_valid_token_ptr,
    is_valid_token_stride,
    block_table_ptr,
    block_table_stride,
    block_table_base_offsets_ptr,
    block_table_base_offsets_stride,
    num_tokens,
    num_reqs,
    max_blocks_per_seq,
    has_valid_token: tl.constexpr,
    has_block_table_base_offsets: tl.constexpr,
    block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_is_valid = token_idx < num_tokens
    if has_valid_token:
        token_is_valid &= tl.load(
            is_valid_token_ptr + token_idx * is_valid_token_stride,
            mask=token_is_valid,
            other=False,
        )

    req_idx = tl.load(
        token_to_req_indices_ptr + token_idx * token_to_req_indices_stride,
        mask=token_is_valid,
        other=-1,
    ).to(tl.int32)
    req_is_valid = token_is_valid & (req_idx >= 0) & (req_idx < num_reqs)
    query_start = tl.load(
        query_start_loc_ptr + req_idx * query_start_loc_stride,
        mask=req_is_valid,
        other=0,
    ).to(tl.int64)
    query_end = tl.load(
        query_start_loc_ptr + (req_idx + 1) * query_start_loc_stride,
        mask=req_is_valid,
        other=0,
    ).to(tl.int64)
    seq_len = tl.load(
        seq_lens_ptr + req_idx * seq_lens_stride,
        mask=req_is_valid,
        other=0,
    ).to(tl.int64)
    position = seq_len - (query_end - query_start) + token_idx - query_start
    compressed_position = position // compress_ratio
    logical_page = compressed_position // block_size
    offset = compressed_position % block_size

    base_page = tl.zeros((), dtype=tl.int64)
    if has_block_table_base_offsets:
        base_page = tl.load(
            block_table_base_offsets_ptr + req_idx * block_table_base_offsets_stride,
            mask=req_is_valid,
            other=0,
        ).to(tl.int64)
    table_page = logical_page - base_page
    page_is_valid = (
        req_is_valid
        & (position >= 0)
        & ((position + 1) % compress_ratio == 0)
        & (table_page >= 0)
        & (table_page < max_blocks_per_seq)
    )
    page_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + table_page,
        mask=page_is_valid,
        other=-1,
    ).to(tl.int64)
    slot = page_id * block_size + offset
    tl.store(
        slot_mapping_ptr + token_idx,
        tl.where(page_is_valid & (page_id >= 0), slot, -1),
    )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_reqs",
        "max_blocks_per_req",
    ]
)
def _dsv4_group_slot_mapping_kernel(
    out_ptr,
    positions_ptr,
    positions_stride,
    req_indices_ptr,
    req_indices_stride,
    block_table_ptr,
    block_table_stride,
    base_offsets_ptr,
    base_offsets_stride,
    valid_token_ptr,
    valid_token_stride,
    num_tokens,
    num_reqs,
    max_blocks_per_req,
    req_repeat: tl.constexpr,
    valid_repeat: tl.constexpr,
    has_base_offsets: tl.constexpr,
    has_valid_token: tl.constexpr,
    rows_per_block: tl.constexpr,
    entry_stride_tokens: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_in_range = token_idx < num_tokens
    position = tl.load(
        positions_ptr + token_idx * positions_stride,
        mask=token_in_range,
        other=-1,
    ).to(tl.int64)
    req_value_idx = token_idx // req_repeat
    req_idx = tl.load(
        req_indices_ptr + req_value_idx * req_indices_stride,
        mask=token_in_range,
        other=-1,
    ).to(tl.int64)
    req_is_valid = token_in_range & (req_idx >= 0) & (req_idx < num_reqs)

    token_is_valid = req_is_valid
    if has_valid_token:
        valid_value_idx = token_idx // valid_repeat
        token_is_valid &= tl.load(
            valid_token_ptr + valid_value_idx * valid_token_stride,
            mask=token_in_range,
            other=False,
        )

    logical_row = position // entry_stride_tokens
    logical_block = logical_row // rows_per_block
    row_offset = logical_row % rows_per_block
    base_block = tl.zeros((), dtype=tl.int64)
    if has_base_offsets:
        base_block = tl.load(
            base_offsets_ptr + req_idx * base_offsets_stride,
            mask=req_is_valid,
            other=0,
        ).to(tl.int64)
    table_block = logical_block - base_block
    table_entry_is_valid = (
        token_is_valid
        & (position >= 0)
        & (table_block >= 0)
        & (table_block < max_blocks_per_req)
    )
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + table_block,
        mask=table_entry_is_valid,
        other=-1,
    ).to(tl.int64)
    slot = block_id * rows_per_block + row_offset
    tl.store(
        out_ptr + token_idx,
        tl.where(table_entry_is_valid & (block_id >= 0), slot, -1),
        mask=token_in_range,
    )


def dsv4_compact_compressed_slot_mapping(
    *,
    num_tokens: int,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    block_table_base_offsets: torch.Tensor | None,
    is_valid_token: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Build slots for a grouped DeepSeek V4 compressed-cache page table.

    The table may contain full logical rows or compact rows accompanied by a
    per-request base logical-page offset. Invalid and non-boundary tokens map
    to ``-1``.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if token_to_req_indices.numel() < num_tokens:
        raise ValueError(
            "token_to_req_indices must cover every token: "
            f"tokens={num_tokens}, request_indices={token_to_req_indices.numel()}"
        )
    if query_start_loc.dim() != 1 or seq_lens.dim() != 1:
        raise ValueError("query_start_loc and seq_lens must be 1-D")
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2-D, got {tuple(block_table.shape)}")
    if block_table_base_offsets is not None and block_table_base_offsets.dim() != 1:
        raise ValueError("block_table_base_offsets must be 1-D")
    if is_valid_token is not None:
        if is_valid_token.numel() != num_tokens:
            if is_valid_token.numel() <= 0 or num_tokens % is_valid_token.numel() != 0:
                raise ValueError(
                    "is_valid_token must cover or evenly expand across every token: "
                    f"tokens={num_tokens}, validity={is_valid_token.numel()}"
                )
            is_valid_token = is_valid_token.repeat_interleave(
                num_tokens // is_valid_token.numel()
            )
        is_valid_token = is_valid_token.to(device=seq_lens.device, dtype=torch.bool)
    if out is None:
        out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
    if out.dim() != 1 or out.dtype != torch.int64 or out.numel() < num_tokens:
        raise ValueError(
            "out must be a 1-D int64 tensor with at least num_tokens entries, got "
            f"shape={tuple(out.shape)} dtype={out.dtype} tokens={num_tokens}"
        )
    if out.stride(0) != 1:
        raise ValueError("out must be contiguous")

    slot_mapping = out[:num_tokens]
    if out.numel() == 0:
        return slot_mapping

    num_reqs = min(
        seq_lens.numel(),
        max(0, query_start_loc.numel() - 1),
        block_table.shape[0],
        (
            block_table_base_offsets.numel()
            if block_table_base_offsets is not None
            else seq_lens.numel()
        ),
    )
    if seq_lens.is_cuda:
        req_indices_i32 = token_to_req_indices.to(torch.int32)
        query_start_i32 = query_start_loc.to(torch.int32)
        seq_lens_i32 = seq_lens.to(torch.int32)
        block_table_i32 = _as_int32_block_table(block_table)
        validity_arg = seq_lens_i32 if is_valid_token is None else is_valid_token
        base_offsets_arg = (
            seq_lens_i32
            if block_table_base_offsets is None
            else block_table_base_offsets.to(torch.int32)
        )
        _dsv4_compact_compressed_slot_mapping_kernel[(out.numel(),)](
            out,
            req_indices_i32,
            req_indices_i32.stride(0),
            query_start_i32,
            query_start_i32.stride(0),
            seq_lens_i32,
            seq_lens_i32.stride(0),
            validity_arg,
            validity_arg.stride(0),
            block_table_i32,
            block_table_i32.stride(0),
            base_offsets_arg,
            base_offsets_arg.stride(0),
            num_tokens,
            num_reqs,
            block_table_i32.shape[1],
            has_valid_token=is_valid_token is not None,
            has_block_table_base_offsets=block_table_base_offsets is not None,
            block_size=block_size,
            compress_ratio=compress_ratio,
        )
        return slot_mapping

    out.fill_(-1)
    if num_tokens == 0 or num_reqs == 0:
        return slot_mapping
    req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
    valid_req = (req_idx >= 0) & (req_idx < num_reqs)
    safe_req = req_idx.clamp(0, num_reqs - 1)
    query_starts = query_start_loc[safe_req].to(torch.int64)
    query_lens = query_start_loc[safe_req + 1].to(torch.int64) - query_starts
    positions = (
        seq_lens[safe_req].to(torch.int64)
        - query_lens
        + torch.arange(num_tokens, dtype=torch.int64, device=seq_lens.device)
        - query_starts
    )
    compressed_positions = torch.div(positions, compress_ratio, rounding_mode="floor")
    table_pages = torch.div(compressed_positions, block_size, rounding_mode="floor")
    if block_table_base_offsets is not None:
        table_pages -= block_table_base_offsets[safe_req].to(torch.int64)
    valid_page = (
        valid_req
        & (positions >= 0)
        & (((positions + 1) % compress_ratio) == 0)
        & (table_pages >= 0)
        & (table_pages < block_table.shape[1])
    )
    safe_page = table_pages.clamp(0, max(0, block_table.shape[1] - 1))
    if block_table.shape[1] == 0:
        page_ids = torch.full_like(table_pages, -1)
    else:
        page_ids = block_table.to(torch.int64)[safe_req, safe_page]
    valid_page &= page_ids >= 0
    if is_valid_token is not None:
        valid_page &= is_valid_token[:num_tokens]
    slots = page_ids * block_size + compressed_positions % block_size
    slot_mapping.copy_(torch.where(valid_page, slots, torch.full_like(slots, -1)))
    return slot_mapping


def dsv4_group_slot_mapping(
    *,
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    rows_per_block: int,
    entry_stride_tokens: int,
    base_offsets: torch.Tensor | None,
    is_valid_token: torch.Tensor | None,
) -> torch.Tensor:
    """Map logical token positions to physical rows of a cache block table.

    Request indices and validity values may either have one entry per token or
    evenly expand across packed token groups. Invalid requests, table entries,
    positions, pages, and graph-padding tokens map to ``-1``.
    """

    if rows_per_block <= 0:
        raise ValueError(f"rows_per_block must be positive, got {rows_per_block}")
    if entry_stride_tokens <= 0:
        raise ValueError(
            f"entry_stride_tokens must be positive, got {entry_stride_tokens}"
        )
    if positions.dim() != 1 or req_indices.dim() != 1:
        raise ValueError("positions and req_indices must be 1-D")
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2-D, got {tuple(block_table.shape)}")
    num_tokens = positions.numel()
    if not positions.is_cuda:
        raise ValueError("dsv4_group_slot_mapping requires CUDA tensors")
    out = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    if num_tokens == 0:
        return out
    if req_indices.numel() <= 0 or num_tokens % req_indices.numel() != 0:
        raise ValueError(
            "req_indices must evenly expand across positions: "
            f"tokens={num_tokens}, request_indices={req_indices.numel()}"
        )
    if base_offsets is not None and base_offsets.dim() != 1:
        raise ValueError("base_offsets must be 1-D")
    if is_valid_token is not None:
        if is_valid_token.dim() != 1:
            raise ValueError("is_valid_token must be 1-D")
        if is_valid_token.numel() <= 0 or num_tokens % is_valid_token.numel() != 0:
            raise ValueError(
                "is_valid_token must evenly expand across positions: "
                f"tokens={num_tokens}, validity={is_valid_token.numel()}"
            )
    if block_table.shape[0] == 0 or block_table.shape[1] == 0:
        out.fill_(-1)
        return out
    dummy = positions
    base_arg = dummy if base_offsets is None else base_offsets
    valid_arg = dummy if is_valid_token is None else is_valid_token
    _dsv4_group_slot_mapping_kernel[(num_tokens,)](
        out,
        positions,
        positions.stride(0),
        req_indices,
        req_indices.stride(0),
        block_table,
        block_table.stride(0),
        base_arg,
        base_arg.stride(0),
        valid_arg,
        valid_arg.stride(0),
        num_tokens,
        block_table.shape[0],
        block_table.shape[1],
        req_repeat=num_tokens // req_indices.numel(),
        valid_repeat=(
            1 if is_valid_token is None else num_tokens // is_valid_token.numel()
        ),
        has_base_offsets=base_offsets is not None,
        has_valid_token=is_valid_token is not None,
        rows_per_block=rows_per_block,
        entry_stride_tokens=entry_stride_tokens,
    )
    return out


@triton.jit(do_not_specialize=["actual_bs"])
def _dsv4_validate_active_cache_pages_kernel(
    out_ptr,
    seq_lens_ptr,
    seq_lens_stride,
    block_table_ptr,
    block_table_row_stride,
    block_table_col_stride,
    actual_bs,
    table_width,
    raw_tokens_per_page,
    max_page_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.arange(0, BLOCK_SIZE)
    row_is_live = row < actual_bs
    seq_len = tl.load(
        seq_lens_ptr + row * seq_lens_stride,
        mask=row_is_live,
        other=0,
    ).to(tl.int64)
    has_tokens = row_is_live & (seq_len > 0)
    required_page = (tl.maximum(seq_len, 1) - 1) // raw_tokens_per_page
    page_is_in_bounds = required_page < table_width
    safe_page = tl.minimum(tl.maximum(required_page, 0), table_width - 1)
    page_id = tl.load(
        block_table_ptr
        + row * block_table_row_stride
        + safe_page * block_table_col_stride,
        mask=has_tokens & page_is_in_bounds,
        other=1,
    ).to(tl.int64)
    invalid = row_is_live & (
        (seq_len < 0)
        | (
            has_tokens
            & ((~page_is_in_bounds) | (page_id <= 0) | (page_id > max_page_id))
        )
    )
    any_invalid = tl.max(invalid.to(tl.int32), axis=0)
    tl.store(out_ptr, any_invalid == 0)


def dsv4_validate_active_cache_pages(
    *,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    actual_bs: int,
    raw_tokens_per_page: int,
    max_page_id: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Validate every live request's active cache page with one CUDA kernel."""
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2-D, got {tuple(block_table.shape)}")
    if not seq_lens.is_cuda or not block_table.is_cuda:
        raise ValueError("dsv4_validate_active_cache_pages requires CUDA tensors")
    if seq_lens.device != block_table.device:
        raise ValueError("seq_lens and block_table must use the same CUDA device")
    if actual_bs < 0 or actual_bs > min(seq_lens.numel(), block_table.shape[0]):
        raise ValueError(
            "actual_bs must fit seq_lens and block_table rows: "
            f"actual_bs={actual_bs}, seq_lens={seq_lens.numel()}, "
            f"table_rows={block_table.shape[0]}"
        )
    if block_table.shape[1] <= 0:
        raise ValueError("block_table must have at least one column")
    if raw_tokens_per_page <= 0:
        raise ValueError(
            f"raw_tokens_per_page must be positive, got {raw_tokens_per_page}"
        )
    if max_page_id <= 0:
        raise ValueError(f"max_page_id must be positive, got {max_page_id}")
    if (
        out.shape != (1,)
        or out.dtype != torch.bool
        or not out.is_cuda
        or out.device != seq_lens.device
    ):
        raise ValueError(
            "out must be a one-element CUDA bool tensor on the input device"
        )
    if actual_bs == 0:
        out.fill_(True)
        return out
    block_size = triton.next_power_of_2(max(1, int(seq_lens.numel())))
    _dsv4_validate_active_cache_pages_kernel[(1,)](
        out,
        seq_lens,
        seq_lens.stride(0),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        actual_bs,
        block_table.shape[1],
        raw_tokens_per_page,
        max_page_id,
        BLOCK_SIZE=block_size,
    )
    return out


@triton.jit
def _dsv4_indexer_decode_metadata_kernel(
    out_block_tables_ptr,
    out_block_tables_stride,
    out_context_lens_ptr,
    positions_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_table_base_offsets_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    compress_ratio: tl.constexpr,
    cache_block_size: tl.constexpr,
    max_blocks: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    req = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)
    req_valid = (req >= 0) & (req < rows)
    safe_req = tl.maximum(0, tl.minimum(req, rows - 1))
    base_logical_page = tl.zeros((), dtype=tl.int64)
    if block_table_base_offsets_ptr is not None:
        base_logical_page = tl.load(block_table_base_offsets_ptr + safe_req).to(
            tl.int64
        )
    compressed_lens = tl.maximum(
        ((pos + 1) // compress_ratio) - base_logical_page * cache_block_size,
        0,
    )
    num_valid_pages = tl.zeros((), dtype=tl.int64)
    for col_start in range(0, max_blocks, candidate_block):
        col_offsets = col_start + tl.arange(0, candidate_block)
        col_mask = col_offsets < max_blocks
        in_cols = col_offsets < cols
        safe_col = tl.where(in_cols, col_offsets, 0)
        bt_load_mask = col_mask & in_cols & req_valid
        bt_vals = tl.load(
            block_table_ptr + safe_req * block_table_stride + safe_col,
            mask=bt_load_mask,
            other=0,
        )
        page_valid = (bt_vals >= 0) & in_cols
        final_mask = page_valid & req_valid & col_mask
        masked_bt = tl.where(final_mask, bt_vals, 0)
        tl.store(
            out_block_tables_ptr + token_idx * out_block_tables_stride + col_offsets,
            masked_bt,
            mask=col_mask,
        )
        num_valid_pages += tl.sum(final_mask.to(tl.int64), axis=0)
    available_lens = num_valid_pages * cache_block_size
    context_len_val = tl.minimum(compressed_lens, available_lens)
    context_len_val = tl.where(req_valid, context_len_val, 0)
    tl.store(out_context_lens_ptr + token_idx, context_len_val.to(tl.int32))


def dsv4_indexer_decode_metadata_compute(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    max_blocks: int,
    out_context_lens: torch.Tensor,
    out_block_tables: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Build decode-indexer context lengths and block tables in one Triton pass."""
    num_tokens = int(positions.shape[0]) if positions.ndim >= 1 else 0
    if num_tokens == 0:
        return
    if out_context_lens.dtype != torch.int32 or out_block_tables.dtype != torch.int32:
        raise TypeError("output buffers must be int32")
    positions_i64 = positions.to(torch.int64)
    token_to_req_indices_i32 = token_to_req_indices.to(torch.int32)
    block_table_i32 = block_table.to(torch.int32)
    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    candidate_block = min(1024, max(16, triton.next_power_of_2(max_blocks)))
    _dsv4_indexer_decode_metadata_kernel[(num_tokens,)](
        out_block_tables,
        out_block_tables.stride(0),
        out_context_lens,
        positions_i64,
        token_to_req_indices_i32,
        block_table_i32,
        block_table_i32.stride(0),
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        rows=rows,
        cols=cols,
        compress_ratio=int(compress_ratio),
        cache_block_size=int(cache_block_size),
        max_blocks=int(max_blocks),
        candidate_block=candidate_block,
    )
