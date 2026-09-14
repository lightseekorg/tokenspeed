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

"""Selected DSV4 attention indices for SWA and compressed KV segments."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv4._triton.common import (
    DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    _as_int32_block_table,
)


@triton.jit
def _dsv4_compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    is_valid_token_ptr,
    has_valid_token: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if has_valid_token:
        is_valid_token = tl.load(is_valid_token_ptr + token_idx)
        if not is_valid_token:
            tl.store(topk_lens_ptr + token_idx, 0)
            return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    count = tl.zeros((), dtype=tl.int32)

    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        valid = local_idx >= 0
        block_indices = local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & valid,
            other=0,
        )
        block_offsets = local_idx % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, count)


def dsv4_compute_global_topk_indices_and_lens(
    *,
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local CSA top-k indices to global KV slots in one Triton kernel."""

    if topk_indices.dtype != torch.int32:
        raise TypeError(f"topk_indices must be int32, got {topk_indices.dtype}")
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be 2-D, got {tuple(topk_indices.shape)}")
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    if num_tokens == 0:
        return global_topk_indices, topk_lens
    if is_valid_token is not None:
        is_valid_token = is_valid_token[:num_tokens].to(
            device=topk_indices.device,
            dtype=torch.bool,
        )
    if not topk_indices.is_cuda:
        valid = topk_indices >= 0
        if is_valid_token is not None:
            valid = valid & is_valid_token[:, None]
        req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
        rows = int(block_table.shape[0]) if block_table.dim() >= 1 else 0
        cols = int(block_table.shape[1]) if block_table.dim() >= 2 else 0
        if rows <= 0 or cols <= 0:
            global_topk_indices.fill_(-1)
            topk_lens.zero_()
            return global_topk_indices, topk_lens
        safe_local = torch.where(valid, topk_indices, torch.zeros_like(topk_indices))
        block_indices = torch.div(safe_local, block_size, rounding_mode="floor")
        block_offsets = safe_local % block_size
        req_valid = (req_idx >= 0) & (req_idx < rows)
        block_valid = (block_indices >= 0) & (block_indices < cols)
        valid = valid & req_valid[:, None] & block_valid
        safe_req = req_idx.clamp(0, rows - 1)
        safe_block = block_indices.long().clamp(0, cols - 1)
        block_numbers = block_table[safe_req[:, None], safe_block]
        global_topk_indices.copy_(
            torch.where(
                valid,
                block_numbers.to(torch.int32) * block_size + block_offsets,
                torch.full_like(topk_indices, -1),
            )
        )
        topk_lens.copy_(valid.sum(dim=1, dtype=torch.int32))
        return global_topk_indices, topk_lens
    if is_valid_token is None:
        is_valid_token = torch.empty(0, dtype=torch.bool, device=topk_indices.device)

    _dsv4_compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        token_to_req_indices.to(torch.int32),
        block_table.to(torch.int32),
        block_table.stride(0),
        is_valid_token,
        is_valid_token.numel() != 0,
        block_size=block_size,
        topk=topk_indices.shape[-1],
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit(do_not_specialize=["block_table_stride", "max_blocks_per_seq"])
def _dsv4_decode_dense_compressed_indices_and_lens_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    positions_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    num_reqs,
    max_blocks_per_seq,
    has_valid_token: tl.constexpr,
    has_block_table_base_offsets: tl.constexpr,
    block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    width: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_is_valid = tl.full((), True, tl.int1)
    if has_valid_token:
        token_is_valid = tl.load(is_valid_token_ptr + token_idx)

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)
    req_is_valid = (req_idx >= 0) & (req_idx < num_reqs)
    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    compressed_len = tl.minimum(
        tl.maximum((position + 1) // compress_ratio, 0),
        width,
    ).to(tl.int32)
    compressed_len = tl.where(token_is_valid, compressed_len, 0)
    tl.store(lens_ptr + token_idx, compressed_len)

    base_page = tl.zeros((), dtype=tl.int32)
    if has_block_table_base_offsets:
        base_page = tl.load(
            block_table_base_offsets_ptr + req_idx,
            mask=req_is_valid,
            other=0,
        ).to(tl.int32)

    for start in range(0, width, candidate_block):
        offsets = start + tl.arange(0, candidate_block)
        store_mask = offsets < width
        entry_is_valid = store_mask & token_is_valid & (offsets < compressed_len)
        logical_page = offsets // block_size
        table_page = logical_page - base_page
        page_is_valid = (
            entry_is_valid
            & req_is_valid
            & (table_page >= 0)
            & (table_page < max_blocks_per_seq)
        )
        block_number = tl.load(
            block_table_ptr + req_idx * block_table_stride + table_page,
            mask=page_is_valid,
            other=-1,
        ).to(tl.int32)
        slot = block_number * block_size + offsets % block_size
        value = tl.where(page_is_valid & (block_number >= 0), slot, -1)
        tl.store(
            indices_ptr + token_idx * indices_stride + offsets,
            value,
            mask=store_mask,
        )


def dsv4_decode_dense_compressed_indices_and_lens(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    width: int,
    block_table_base_offsets: torch.Tensor | None,
    is_valid_token: torch.Tensor | None,
    out_indices: torch.Tensor | None,
    out_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense compressed decode KV slots and per-token lengths.

    The result matches the logical dense-prefix mapping previously expressed
    as a chain of PyTorch div/clamp/where/index operations.  On CUDA, one
    Triton launch constructs physical cache slots directly so a first HCA
    layer cannot permanently capture that elementwise chain into every graph
    replay.
    """

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    num_tokens = positions.numel()
    if token_to_req_indices.numel() < num_tokens:
        raise ValueError(
            "token_to_req_indices must cover every position: "
            f"positions={num_tokens}, request_indices={token_to_req_indices.numel()}"
        )
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2-D, got {tuple(block_table.shape)}")
    if out_indices is None:
        out_indices = torch.empty(
            (num_tokens, width),
            dtype=torch.int32,
            device=positions.device,
        )
    if out_lens is None:
        out_lens = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=positions.device,
        )
    if out_indices.shape != (num_tokens, width) or out_indices.dtype != torch.int32:
        raise ValueError(
            "out_indices must be int32 with shape "
            f"{(num_tokens, width)}, got {tuple(out_indices.shape)} {out_indices.dtype}"
        )
    if out_lens.shape != (num_tokens,) or out_lens.dtype != torch.int32:
        raise ValueError(
            "out_lens must be int32 with shape "
            f"{(num_tokens,)}, got {tuple(out_lens.shape)} {out_lens.dtype}"
        )
    if num_tokens == 0 or width == 0:
        return out_indices, out_lens

    if is_valid_token is not None:
        is_valid_token = is_valid_token[:num_tokens].to(
            device=positions.device,
            dtype=torch.bool,
        )
    if positions.is_cuda:
        if is_valid_token is None:
            is_valid_token = torch.empty(
                0,
                dtype=torch.bool,
                device=positions.device,
            )
        block_table_i32 = _as_int32_block_table(block_table)
        candidate_block = min(1024, triton.next_power_of_2(max(1, width)))
        _dsv4_decode_dense_compressed_indices_and_lens_kernel[(num_tokens,)](
            out_indices,
            out_indices.stride(0),
            out_lens,
            positions,
            token_to_req_indices.to(torch.int32),
            is_valid_token,
            block_table_i32,
            (
                block_table_base_offsets.to(torch.int32)
                if block_table_base_offsets is not None
                else None
            ),
            block_table_i32.stride(0),
            block_table_i32.shape[0],
            block_table_i32.shape[1],
            is_valid_token.numel() != 0,
            block_table_base_offsets is not None,
            block_size=block_size,
            compress_ratio=compress_ratio,
            width=width,
            candidate_block=candidate_block,
        )
        return out_indices, out_lens

    req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
    compressed_lens = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp(0, width)
    if is_valid_token is not None:
        compressed_lens = torch.where(
            is_valid_token,
            compressed_lens,
            torch.zeros_like(compressed_lens),
        )
    offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
    valid = offsets[None, :] < compressed_lens[:, None]
    safe_local = torch.where(
        valid, offsets[None, :], torch.zeros_like(offsets)[None, :]
    )
    pages = torch.div(safe_local, block_size, rounding_mode="floor")
    if block_table_base_offsets is not None:
        rows = int(block_table_base_offsets.shape[0])
        valid_req = (req_idx >= 0) & (req_idx < rows)
        safe_req = req_idx.clamp(0, max(0, rows - 1))
        if rows <= 0:
            pages = torch.full_like(pages, -1)
        else:
            base_pages = block_table_base_offsets.to(torch.int64)[safe_req]
            pages = torch.where(valid_req[:, None], pages - base_pages[:, None], -1)
    page_offsets = safe_local % block_size
    rows = int(block_table.shape[0])
    cols = int(block_table.shape[1])
    if rows <= 0 or cols <= 0:
        page_ids = torch.full_like(pages, -1)
    else:
        page_valid = (
            (req_idx[:, None] >= 0)
            & (req_idx[:, None] < rows)
            & (pages >= 0)
            & (pages < cols)
        )
        safe_req = req_idx.clamp(0, rows - 1)
        safe_page = pages.clamp(0, cols - 1)
        page_ids = torch.where(
            page_valid,
            block_table.to(torch.int64)[safe_req[:, None], safe_page],
            torch.full_like(pages, -1),
        )
    out_indices.copy_(
        torch.where(
            valid & (page_ids >= 0),
            page_ids * block_size + page_offsets,
            torch.full_like(page_ids, -1),
        ).to(torch.int32)
    )
    out_lens.copy_(compressed_lens.to(torch.int32))
    return out_indices, out_lens


@triton.jit
def _dsv4_combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_base_offsets_ptr,
    workspace_width,
    compressed_base,
    compressed_block_size,
    compressed_table_capacity,
    has_block_table_base_offsets: tl.constexpr,
    topk: tl.constexpr,
    compress_ratio: tl.constexpr,
    window_size: tl.constexpr,
    padded_topk: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        base_row = tl.zeros((), dtype=tl.int32)
        if has_block_table_base_offsets:
            base_row = (
                tl.load(block_table_base_offsets_ptr + batch_idx).to(tl.int32)
                * compressed_block_size
            )
        live_compressed_len = tl.maximum(
            tl.minimum(
                (pos + 1) // compress_ratio - base_row, compressed_table_capacity
            ),
            0,
        )
        topk_len = tl.minimum(live_compressed_len, topk)
        swa_len = tl.minimum(pos + 1, window_size)

        topk_offsets = tl.arange(0, padded_topk)
        topk_mask = topk_offsets < topk_len
        topk_values = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + topk_offsets,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = topk_mask & (topk_values >= 0)
        valid_topk_i32 = valid_topk.to(tl.int32)
        compact_topk_offsets = tl.cumsum(valid_topk_i32, 0) - 1
        compact_topk_len = tl.sum(valid_topk_i32, axis=0)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + compact_topk_offsets,
            topk_values + workspace_width * batch_idx,
            mask=valid_topk,
        )

        swa_offsets = tl.arange(0, window_size)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + compact_topk_len
            + swa_offsets,
            workspace_width * batch_idx
            + compressed_base
            + swa_offsets
            + pos
            - swa_len
            + 1
            - gather_start,
            mask=swa_offsets < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, compact_topk_len + swa_len)


def dsv4_combine_topk_swa_indices(
    *,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    workspace_width: int,
    compressed_base: int,
    block_table_base_offsets: torch.Tensor | None = None,
    compressed_block_size: int = 1,
    compressed_table_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FlashMLA sparse prefill indices from compressed prefix and SWA."""

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
        * DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        -1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0 or num_reqs == 0:
        return combined_indices, combined_lens
    if compressed_block_size <= 0:
        raise ValueError("compressed_block_size must be positive")
    if compressed_table_capacity is None:
        compressed_table_capacity = compressed_base

    _dsv4_combine_topk_swa_indices_kernel[(num_reqs, 128)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        gather_lens.to(torch.int32),
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else seq_lens
        ),
        workspace_width,
        compressed_base,
        compressed_block_size,
        compressed_table_capacity,
        has_block_table_base_offsets=block_table_base_offsets is not None,
        topk=topk,
        compress_ratio=compress_ratio,
        window_size=window_size,
        padded_topk=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _dsv4_build_dense_prefill_local_compressed_indices_kernel(
    out_ptr,
    out_stride,
    positions_ptr,
    token_to_req_indices_ptr,
    block_table_base_offsets_ptr,
    compressed_block_size,
    compressed_table_capacity,
    has_block_table_base_offsets: tl.constexpr,
    width: tl.constexpr,
    compress_ratio: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    base_row = tl.zeros((), dtype=tl.int64)
    if has_block_table_base_offsets:
        req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)
        base_row = (
            tl.load(block_table_base_offsets_ptr + req_idx).to(tl.int64)
            * compressed_block_size
        )
    compressed_len = tl.minimum(
        tl.maximum((position + 1) // compress_ratio - base_row, 0),
        tl.minimum(width, compressed_table_capacity),
    )
    for start in range(0, width, block):
        offsets = start + tl.arange(0, block)
        mask = offsets < width
        values = tl.where(offsets < compressed_len, base_row + offsets, -1)
        tl.store(out_ptr + token_idx * out_stride + offsets, values, mask=mask)


def dsv4_build_dense_prefill_local_compressed_indices(
    *,
    positions: torch.Tensor,
    compress_ratio: int,
    width: int,
    out: torch.Tensor,
    token_to_req_indices: torch.Tensor | None = None,
    block_table_base_offsets: torch.Tensor | None = None,
    compressed_block_size: int = 1,
    compressed_table_capacity: int | None = None,
) -> torch.Tensor:
    """Build C128A/HCA prefill-local compressed prefix indices into `out`."""

    result = out[: positions.numel(), :width]
    if positions.numel() == 0 or width <= 0:
        return result
    if result.stride(1) != 1:
        raise ValueError(
            "dense prefill compressed indices output must be contiguous in the last dim"
        )
    if block_table_base_offsets is not None and token_to_req_indices is None:
        raise ValueError(
            "token_to_req_indices is required with block_table_base_offsets"
        )
    if compressed_table_capacity is None:
        compressed_table_capacity = width
    metadata_arg = positions if token_to_req_indices is None else token_to_req_indices
    base_offsets_arg = (
        positions if block_table_base_offsets is None else block_table_base_offsets
    )
    if positions.is_cuda:
        _dsv4_build_dense_prefill_local_compressed_indices_kernel[(positions.numel(),)](
            result,
            result.stride(0),
            positions,
            metadata_arg,
            base_offsets_arg,
            compressed_block_size,
            compressed_table_capacity,
            has_block_table_base_offsets=block_table_base_offsets is not None,
            width=width,
            compress_ratio=compress_ratio,
            block=1024,
        )
        return result

    compressed_ends = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    )
    if block_table_base_offsets is None:
        base_rows = torch.zeros_like(compressed_ends)
    else:
        base_rows = block_table_base_offsets.to(torch.int64)[
            token_to_req_indices.to(torch.int64)
        ] * int(compressed_block_size)
    compressed_lens = (compressed_ends - base_rows).clamp(
        0, min(width, int(compressed_table_capacity))
    )
    offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
    local = base_rows[:, None] + offsets[None, :]
    valid = offsets[None, :] < compressed_lens[:, None]
    result.copy_(torch.where(valid, local, torch.full_like(local, -1)).to(torch.int32))
    return result


@triton.jit
def _dsv4_combine_dense_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    positions_ptr,
    token_to_req_indices_ptr,
    seq_lens_ptr,
    compressed_lens_ptr,
    gather_lens_ptr,
    workspace_width,
    compressed_base,
    combined_topk: tl.constexpr,
    compress_ratio: tl.constexpr,
    window_size: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * candidate_block + tl.arange(0, candidate_block)
    mask = offsets < combined_topk

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)
    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    gather_len = tl.load(gather_lens_ptr + req_idx).to(tl.int32)
    gather_start = seq_len - gather_len
    if compress_ratio > 1:
        compressed_len = tl.minimum(
            (pos + 1) // compress_ratio,
            tl.load(compressed_lens_ptr + req_idx).to(tl.int32),
        )
    else:
        compressed_len = tl.full((), 0, tl.int32)
    swa_len = tl.minimum(pos + 1, window_size)
    total_len = compressed_len + swa_len

    request_base = workspace_width * req_idx
    values = tl.full((candidate_block,), -1, tl.int32)
    is_compressed = offsets < compressed_len
    values = tl.where(is_compressed, request_base + offsets, values)

    swa_offsets = offsets - compressed_len
    is_swa = (offsets >= compressed_len) & (offsets < total_len)
    swa_values = (
        request_base + compressed_base + swa_offsets + pos - swa_len + 1 - gather_start
    )
    values = tl.where(is_swa, swa_values, values)

    tl.store(
        combined_indices_ptr + token_idx * combined_indices_stride + offsets,
        values,
        mask=mask,
    )
    tl.store(combined_lens_ptr + token_idx, total_len, mask=block_idx == 0)


def dsv4_combine_dense_swa_indices(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    compressed_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    workspace_width: int,
    compressed_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense-compressed plus SWA sparse prefill indices."""

    num_tokens = positions.numel()
    combined_topk = (
        (
            max(compressed_base + window_size, 1)
            + DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
            - 1
        )
        // DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
        * DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=positions.device)
    if num_tokens == 0:
        return combined_indices, combined_lens

    candidate_block = 128
    _dsv4_combine_dense_swa_indices_kernel[
        (num_tokens, triton.cdiv(combined_topk, candidate_block))
    ](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        positions,
        token_to_req_indices.to(torch.int32),
        seq_lens.to(torch.int32),
        compressed_lens.to(torch.int32),
        gather_lens.to(torch.int32),
        workspace_width,
        compressed_base,
        combined_topk=combined_topk,
        compress_ratio=compress_ratio,
        window_size=window_size,
        candidate_block=candidate_block,
    )
    return combined_indices, combined_lens


@triton.jit(do_not_specialize=["block_table_stride", "max_blocks_per_seq"])
def _dsv4_decode_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    max_blocks_per_seq,
    has_valid_token: tl.constexpr,
    window_size: tl.constexpr,
    block_size: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if has_valid_token:
        is_valid = tl.load(is_valid_token_ptr + token_idx)
        if not is_valid:
            tl.store(swa_lens_ptr + token_idx, 0)
            return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)

    query_start = tl.load(query_start_loc_ptr + req_idx).to(tl.int32)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int32)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    prefix_len = seq_len - query_len
    pos = prefix_len + token_idx - query_start

    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1
    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + token_idx, swa_len)

    for i in range(0, window_size, candidate_block):
        offsets = i + tl.arange(0, candidate_block)
        mask = offsets < window_size
        pos_offsets = start_pos + offsets
        valid = offsets < swa_len
        block_indices = pos_offsets // block_size
        if block_table_base_offsets_ptr is not None:
            block_indices -= tl.load(block_table_base_offsets_ptr + req_idx)
        valid = valid & (block_indices >= 0) & (block_indices < max_blocks_per_seq)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=valid,
            other=-1,
        )
        block_offsets = pos_offsets % block_size
        slot_ids = block_numbers * block_size + block_offsets
        values = tl.where(valid & (block_numbers >= 0), slot_ids, -1)
        tl.store(
            swa_indices_ptr + token_idx * swa_indices_stride + offsets,
            values,
            mask=mask,
        )


def dsv4_decode_swa_indices_and_lens(
    *,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    window_size: int,
    block_size: int,
    block_table_base_offsets: torch.Tensor | None = None,
    is_valid_token: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    out_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DeepSeek V4 decode SWA KV slot indices once per metadata step."""

    num_tokens = token_to_req_indices.shape[0]
    if out_indices is None:
        out_indices = torch.empty(
            (num_tokens, window_size),
            dtype=torch.int32,
            device=seq_lens.device,
        )
    if out_lens is None:
        out_lens = torch.empty(num_tokens, dtype=torch.int32, device=seq_lens.device)
    if num_tokens == 0:
        return out_indices, out_lens
    if is_valid_token is None:
        is_valid_token = torch.empty(0, dtype=torch.bool, device=seq_lens.device)
    else:
        is_valid_token = is_valid_token[:num_tokens].to(
            device=seq_lens.device,
            dtype=torch.bool,
        )

    candidate_block = min(1024, triton.next_power_of_2(window_size))
    block_table_i32 = _as_int32_block_table(block_table)
    _dsv4_decode_swa_indices_and_lens_kernel[(num_tokens,)](
        out_indices,
        out_indices.stride(0),
        out_lens,
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        token_to_req_indices.to(torch.int32),
        is_valid_token,
        block_table_i32,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table_i32.stride(0),
        block_table_i32.shape[-1],
        is_valid_token.numel() != 0,
        window_size=window_size,
        block_size=block_size,
        candidate_block=candidate_block,
    )
    return out_indices, out_lens
