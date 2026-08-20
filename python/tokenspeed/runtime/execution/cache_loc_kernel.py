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

"""
Triton kernels for computing cache locations and updating page tables.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@triton.jit
def compute_out_cache_loc_kernel(
    # Input pointers
    input_lengths_ptr,  # [batch_size] or None for uniform mode
    cache_start_ptr,  # [batch_size]
    page_table_ptr,  # [batch_size, max_pages], batch-ordered
    cumsum_lengths_ptr,  # [batch_size] or None for uniform mode
    # Output pointer
    out_cache_loc_ptr,  # [total_tokens]
    # Scalars
    uniform_input_length,  # used when input_lengths_ptr is None
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    window_pages: tl.constexpr,  # 0 = full history; >0 = sliding ring width
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unified kernel to compute out_cache_loc for both prefill and decode.

    For each token in each request, compute:
        position = cache_start[req_idx] + token_offset_in_seq
        page_idx = position // page_size
        offset_in_page = position % page_size
        page_id = page_table[req_idx, page_idx]
        out_cache_loc = page_id * page_size + offset_in_page

    The page table is batch-ordered: row i holds the pages of the request at
    batch position i.

    For decode, input_lengths are all 1.
    For prefill, input_lengths vary.

    When all requests share the same input_length (the multi-step drafter
    case), callers pass ``input_lengths_ptr=None`` (and ``cumsum_lengths_ptr=None``)
    together with ``uniform_input_length`` set to the shared length. Triton
    specializes the kernel on the None-ness of the pointers at JIT time and
    dead-code-eliminates the corresponding GMEM reads.

    ``window_pages`` selects the column mapping at JIT time. Full history
    (0) is the degenerate ring: columns grow with position and overflow
    clamps to the reserved slot 0. A sliding group (>0) retains only the
    last ``window_pages`` pages per request, so absolute position ``p``
    lives in ring column ``(p // page_size) % window_pages`` — the modulo
    IS the construction that makes addressing a slid-out column
    impossible. Columns the scheduler punched to the null page read as
    page 0 and route to slot 0 either way.
    """
    # Program ID represents which request we're processing
    req_idx = tl.program_id(0)

    valid_cache_len = tl.load(cache_start_ptr + req_idx)

    if input_lengths_ptr is not None:
        input_length = tl.load(input_lengths_ptr + req_idx)
        # Always load from cumsum, use 0 index for first request to ensure type consistency
        offset_idx = tl.where(req_idx > 0, req_idx - 1, 0)
        output_offset = tl.load(cumsum_lengths_ptr + offset_idx)
        # Zero out offset for first request
        output_offset = tl.where(req_idx > 0, output_offset, 0)
    else:
        input_length = uniform_input_length
        output_offset = req_idx * uniform_input_length

    # Process tokens in blocks
    num_blocks = tl.cdiv(input_length, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE

        # Compute token offsets within this block
        token_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = token_offsets < input_length

        # Compute logical positions
        positions = valid_cache_len + token_offsets

        # Compute page indices and offsets
        page_indices = positions // page_size
        if window_pages > 0:
            # Ring mapping: overflow is impossible by construction.
            page_indices = page_indices % window_pages
            overflow = page_indices < 0  # constant-false, same dtype as below
        else:
            overflow = page_indices >= max_pages
            # Clamp to last valid page to avoid OOB GMEM read.
            page_indices = tl.minimum(page_indices, max_pages - 1)
        offsets_in_page = positions % page_size

        # Load page IDs from the batch-ordered page table.
        page_ptrs = page_table_ptr + req_idx * max_pages + page_indices
        page_ids = tl.load(page_ptrs, mask=mask, other=0)

        # Compute physical cache locations
        cache_locs = page_ids * page_size + offsets_in_page
        # For overflow tokens, route to slot 0 (a fixed safe dummy target that
        # never aliases a real request's KV data). This avoids using a dynamic
        # page_table[0][0] load whose value can change at runtime and corrupt
        # other requests' KV cache or trigger IndexKernel out-of-bounds.
        # Null/hole pages (id <= 0) route to slot 0 as well.
        cache_locs = tl.where(overflow | (page_ids <= 0), 0, cache_locs)

        # Store to output
        output_ptrs = out_cache_loc_ptr + output_offset + token_offsets
        tl.store(output_ptrs, cache_locs, mask=mask)


def compute_out_cache_loc(
    out_cache_loc_ptr,
    input_lengths: torch.Tensor,  # [batch_size]
    cache_start: torch.Tensor,  # [batch_size]
    page_table: torch.Tensor,  # [batch_size, max_pages], batch-ordered
    page_size: int,
) -> None:
    batch_size = input_lengths.shape[0]
    max_pages = page_table.shape[1]

    cumsum_lengths = torch.cumsum(input_lengths, dim=0)

    BLOCK_SIZE = 128
    grid = (batch_size,)

    compute_out_cache_loc_kernel[grid](
        input_lengths,
        cache_start,
        page_table,
        cumsum_lengths,
        out_cache_loc_ptr,
        0,  # uniform_input_length unused when input_lengths_ptr is not None
        page_size=page_size,
        max_pages=max_pages,
        window_pages=0,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def fused_decode_input_prep_kernel(
    # Inputs
    req_pool_indices_ptr,  # [batch_size]
    valid_cache_lengths_ptr,  # [req_pool_size+1]
    page_table_ptr,  # [batch_size, max_pages], batch-ordered
    # Outputs
    out_cache_loc_ptr,  # [batch_size * uniform_input_length]
    positions_ptr,  # [batch_size * uniform_input_length]
    seq_lens_out_ptr,  # [batch_size]
    # Scalars
    uniform_input_length,
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One launch fuses the decode-uniform path's four small kernels.

    Replaces:
      valid_cache_lengths.index_select(0, req_pool_indices)
      compute_out_cache_loc_uniform
      compute_position_triton (decode branch)
      torch.add(input_lengths, valid_cache_lengths, out=seq_lens)

    Each program handles one request. We do one GMEM read of
    `valid_cache_lengths[pool_idx]` (runtime state is pool-indexed) and reuse
    it for the seq_lens write, the position writes, and the out_cache_loc
    page-table lookup; the page table itself is batch-ordered.
    """
    req_idx = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    cache_start = tl.load(valid_cache_lengths_ptr + pool_idx)

    # seq_lens[req_idx] = cache_start + uniform_input_length
    tl.store(seq_lens_out_ptr + req_idx, cache_start + uniform_input_length)

    output_offset = req_idx * uniform_input_length

    num_blocks = tl.cdiv(uniform_input_length, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        token_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = token_offsets < uniform_input_length

        positions_local = cache_start + token_offsets
        page_indices = positions_local // page_size
        overflow = page_indices >= max_pages
        # Clamp to last valid page to avoid OOB GMEM read.
        page_indices = tl.minimum(page_indices, max_pages - 1)
        offsets_in_page = positions_local % page_size

        page_ptrs = page_table_ptr + req_idx * max_pages + page_indices
        page_ids = tl.load(page_ptrs, mask=mask, other=0)
        cache_locs = page_ids * page_size + offsets_in_page
        # Null/hole pages share the same fixed dummy target as overflow.
        cache_locs = tl.where(overflow | (page_ids <= 0), 0, cache_locs)

        tl.store(
            out_cache_loc_ptr + output_offset + token_offsets,
            cache_locs,
            mask=mask,
        )
        tl.store(
            positions_ptr + output_offset + token_offsets,
            positions_local,
            mask=mask,
        )


def fused_decode_input_prep(
    out_cache_loc_ptr,
    positions_ptr,
    seq_lens_out_ptr,
    req_pool_indices: torch.Tensor,  # [batch_size]
    valid_cache_lengths: torch.Tensor,  # [req_pool_size+1]
    uniform_input_length: int,
    page_table: torch.Tensor,  # [batch_size, max_pages], batch-ordered
    page_size: int,
) -> None:
    """Decode-only fast path: one Triton launch writes out_cache_loc,
    positions, and seq_lens, reading `valid_cache_lengths[pool_idx]`
    directly so the per-iter indexSelect + add are gone too.
    """
    batch_size = req_pool_indices.shape[0]
    max_pages = page_table.shape[1]
    BLOCK_SIZE = 128
    grid = (batch_size,)
    fused_decode_input_prep_kernel[grid](
        req_pool_indices,
        valid_cache_lengths,
        page_table,
        out_cache_loc_ptr,
        positions_ptr,
        seq_lens_out_ptr,
        uniform_input_length,
        page_size=page_size,
        max_pages=max_pages,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def dflash_prepare_decode_kernel(
    output_tokens_ptr,
    accept_lengths_ptr,
    req_pool_indices_ptr,
    valid_cache_lengths_ptr,
    page_table_ptr,
    draft_seq_lens_ptr,
    block_ids_ptr,
    block_positions_ptr,
    out_cache_loc_ptr,
    verify_width: tl.constexpr,
    draft_query_width: tl.constexpr,
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    max_draft_prefix,
    block_ids_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    accept_len = tl.load(accept_lengths_ptr + req_idx)
    old_len = tl.load(valid_cache_lengths_ptr + pool_idx)
    prefix_len = old_len + accept_len
    prefix_len = tl.minimum(prefix_len, max_draft_prefix)
    tl.store(draft_seq_lens_ptr + req_idx, prefix_len)

    safe_accept = tl.minimum(tl.maximum(accept_len, 1), verify_width)
    current_token = tl.load(
        output_tokens_ptr + req_idx * verify_width + safe_accept - 1
    )
    tl.store(block_ids_ptr + req_idx * block_ids_stride, current_token)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < draft_query_width
    positions = prefix_len + offsets
    tl.store(
        block_positions_ptr + req_idx * draft_query_width + offsets,
        positions,
        mask=mask,
    )

    page_indices = positions // page_size
    overflow = page_indices >= max_pages
    page_indices = tl.minimum(page_indices, max_pages - 1)
    offsets_in_page = positions % page_size
    # The drafter's page table is batch-ordered (row i == batch position i).
    page_ptrs = page_table_ptr + req_idx * max_pages + page_indices
    page_ids = tl.load(page_ptrs, mask=mask, other=0)
    cache_locs = page_ids * page_size + offsets_in_page
    cache_locs = tl.where(overflow | (page_ids <= 0), 0, cache_locs)
    tl.store(
        out_cache_loc_ptr + req_idx * draft_query_width + offsets,
        cache_locs,
        mask=mask,
    )


def dflash_prepare_decode(
    output_tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    page_table: torch.Tensor,
    draft_seq_lens: torch.Tensor,
    block_ids: torch.Tensor,
    block_positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    verify_width: int,
    draft_query_width: int,
    page_size: int,
    max_draft_prefix: int,
) -> None:
    batch_size = req_pool_indices.shape[0]
    max_pages = page_table.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(draft_query_width)
    grid = (batch_size,)
    dflash_prepare_decode_kernel[grid](
        output_tokens,
        accept_lengths,
        req_pool_indices,
        valid_cache_lengths,
        page_table,
        draft_seq_lens,
        block_ids,
        block_positions,
        out_cache_loc,
        verify_width=verify_width,
        draft_query_width=draft_query_width,
        page_size=page_size,
        max_pages=max_pages,
        max_draft_prefix=max_draft_prefix,
        block_ids_stride=block_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )


def compute_out_cache_loc_uniform(
    out_cache_loc_ptr,
    uniform_input_length: int,
    cache_start: torch.Tensor,  # [batch_size]
    page_table: torch.Tensor,  # [batch_size, max_pages], batch-ordered
    page_size: int,
) -> None:
    """Specialized entry point when every request has the same ``input_length``.

    Skips the per-call ``torch.full`` + ``cumsum`` host-side work and the
    corresponding GMEM reads inside the kernel. Used by the multi-step drafter
    where each request decodes exactly ``spec_num_steps - 1`` tokens.
    """
    batch_size = cache_start.shape[0]
    max_pages = page_table.shape[1]

    BLOCK_SIZE = 128
    grid = (batch_size,)

    compute_out_cache_loc_kernel[grid](
        None,  # input_lengths_ptr is None → kernel uses uniform_input_length
        cache_start,
        page_table,
        None,  # cumsum_lengths_ptr is None → kernel computes offset analytically
        out_cache_loc_ptr,
        uniform_input_length,
        page_size=page_size,
        max_pages=max_pages,
        window_pages=0,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def compute_out_cache_loc_sliding(
    out_cache_loc_ptr,
    uniform_input_length: int,
    cache_start: torch.Tensor,  # [batch_size]
    page_table: torch.Tensor,  # [batch_size, window_pages], batch-ordered ring
    page_size: int,
) -> None:
    """Sliding-window write locations over a ring-mapped page table.

    Same unified kernel, ring column mapping: the table holds exactly the
    window's pages (``window_pages == table width``) and old columns are
    reused in place as the window slides, so the staged table stays
    capture-stable at a fixed width instead of growing with context.

    Args:
        out_cache_loc_ptr: [batch_size * uniform_input_length] output slots.
        uniform_input_length: Tokens per request (uniform across the batch).
        cache_start: [batch_size] absolute first write position per request.
        page_table: [batch_size, window_pages] ring table; entries are kernel
            page ids, 0 for null/reclaimed columns.
        page_size: Kernel page size in tokens.
    """
    batch_size = cache_start.shape[0]
    window_pages = page_table.shape[1]

    BLOCK_SIZE = 128
    grid = (batch_size,)

    compute_out_cache_loc_kernel[grid](
        None,
        cache_start,
        page_table,
        None,
        out_cache_loc_ptr,
        uniform_input_length,
        page_size=page_size,
        max_pages=window_pages,
        window_pages=window_pages,
        BLOCK_SIZE=BLOCK_SIZE,
    )
