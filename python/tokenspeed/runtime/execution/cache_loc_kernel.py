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

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@triton.jit
def update_req_to_page_kernel(
    # Input pointers
    req_pool_indices_ptr,  # [batch_size]
    new_occupied_pages_ptr,  # [total_entries] - flattened refresh page IDs
    new_occupied_pages_num_ptr,  # [batch_size] - refresh entries per request
    pages_copy_starts_ptr,  # [batch_size]
    cumsum_pages_ptr,  # [batch_size] - cumulative refresh-entry counts
    # Output pointer
    req_to_page_ptr,  # [req_pool_size+1, context_len]
    # Scalars
    context_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply emitted page-table refresh slices to req_to_page.
    Each program handles one request in the batch.
    """
    req_idx = tl.program_id(0)

    # Load request metadata
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    num_pages = tl.load(new_occupied_pages_num_ptr + req_idx)
    copy_start = tl.load(pages_copy_starts_ptr + req_idx)

    # Get the request's offset into the flattened refresh entries.
    offset_idx = tl.where(req_idx > 0, req_idx - 1, 0)
    pages_offset = tl.load(cumsum_pages_ptr + offset_idx)
    pages_offset = tl.where(req_idx > 0, pages_offset, 0)

    # Process pages in blocks
    num_blocks = tl.cdiv(num_pages, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE

        # Compute page indices within this block
        page_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = page_offsets < num_pages

        # Load the page IDs to refresh.
        page_ptrs = new_occupied_pages_ptr + pages_offset + page_offsets
        new_page_ids = tl.load(page_ptrs, mask=mask, other=0)

        # Compute target positions in req_to_page
        target_positions = copy_start + page_offsets

        # Store to req_to_page[req_pool_idx, target_positions]
        output_ptrs = req_to_page_ptr + req_pool_idx * context_len + target_positions
        tl.store(output_ptrs, new_page_ids, mask=mask)


def update_req_to_page(
    req_to_page: torch.Tensor,
    req_pool_indices: torch.Tensor,
    new_occupied_pages: torch.Tensor,
    new_occupied_pages_num: torch.Tensor,
    pages_copy_starts: torch.Tensor,
) -> None:
    """
    Apply emitted page-table refresh slices to req_to_page using Triton.

    Args:
        req_to_page: Request to page table [req_pool_size+1, context_len]
        req_pool_indices: Request pool indices [batch_size]
        new_occupied_pages: Refresh page IDs [total_entries] - flattened
        new_occupied_pages_num: Number of refresh entries per request [batch_size]
        pages_copy_starts: Start position in req_to_page for each request [batch_size]
    """
    batch_size = req_pool_indices.shape[0]
    context_len = req_to_page.shape[1]

    if new_occupied_pages.shape[0] == 0:
        return

    # Compute cumulative sum for offset calculation.
    cumsum_pages = torch.cumsum(new_occupied_pages_num, dim=0)

    # Launch kernel - one program per request
    BLOCK_SIZE = 128
    grid = (batch_size,)

    update_req_to_page_kernel[grid](
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
        cumsum_pages,
        req_to_page,
        context_len=context_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def compute_out_cache_loc_kernel(
    # Input pointers
    req_pool_indices_ptr,  # [batch_size]
    input_lengths_ptr,  # [batch_size] or None for uniform mode
    cache_start_ptr,  # [batch_size]
    req_to_pages_ptr,  # [req_pool_size+1, max_pages]
    cumsum_lengths_ptr,  # [batch_size] or None for uniform mode
    # Output pointer
    out_cache_loc_ptr,  # [total_tokens]
    # Scalars
    uniform_input_length,  # used when input_lengths_ptr is None
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unified kernel to compute out_cache_loc for both prefill and decode.

    For each token in each request, compute:
        position = cache_start[req_idx] + token_offset_in_seq
        page_idx = position // page_size
        offset_in_page = position % page_size
        page_id = req_to_pages[req_pool_idx, page_idx]
        out_cache_loc = page_id * page_size + offset_in_page

    For decode, input_lengths are all 1.
    For prefill, input_lengths vary.

    When all requests share the same input_length (the multi-step drafter
    case), callers pass ``input_lengths_ptr=None`` (and ``cumsum_lengths_ptr=None``)
    together with ``uniform_input_length`` set to the shared length. Triton
    specializes the kernel on the None-ness of the pointers at JIT time and
    dead-code-eliminates the corresponding GMEM reads.
    """
    # Program ID represents which request we're processing
    req_idx = tl.program_id(0)

    # Load request metadata.
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
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
        overflow = page_indices >= max_pages
        # Clamp to last valid page to avoid OOB GMEM read.
        page_indices = tl.minimum(page_indices, max_pages - 1)
        offsets_in_page = positions % page_size

        # Load page IDs from req_to_pages
        # req_to_pages is [req_pool_size+1, max_pages]
        page_ptrs = req_to_pages_ptr + req_pool_idx * max_pages + page_indices
        page_ids = tl.load(page_ptrs, mask=mask, other=0)

        # Compute physical cache locations
        cache_locs = page_ids * page_size + offsets_in_page
        # For overflow tokens, route to slot 0 (a fixed safe dummy target that
        # never aliases a real request's KV data). This avoids using a dynamic
        # req_to_pages[0][0] load whose value can change at runtime and corrupt
        # other requests' KV cache or trigger IndexKernel out-of-bounds.
        cache_locs = tl.where(overflow, 0, cache_locs)

        # Store to output
        output_ptrs = out_cache_loc_ptr + output_offset + token_offsets
        tl.store(output_ptrs, cache_locs, mask=mask)


def compute_out_cache_loc(
    out_cache_loc_ptr,
    req_pool_indices: torch.Tensor,  # [batch_size]
    input_lengths: torch.Tensor,  # [batch_size]
    cache_start: torch.Tensor,  # [batch_size]
    req_to_pages: torch.Tensor,  # [req_pool_size+1, max_pages]
    page_size: int,
) -> None:
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]

    cumsum_lengths = torch.cumsum(input_lengths, dim=0)

    BLOCK_SIZE = 128
    grid = (batch_size,)

    compute_out_cache_loc_kernel[grid](
        req_pool_indices,
        input_lengths,
        cache_start,
        req_to_pages,
        cumsum_lengths,
        out_cache_loc_ptr,
        0,  # uniform_input_length unused when input_lengths_ptr is not None
        page_size=page_size,
        max_pages=max_pages,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def fused_decode_input_prep_kernel(
    # Inputs
    req_pool_indices_ptr,  # [batch_size]
    valid_cache_lengths_ptr,  # [req_pool_size+1]
    req_to_pages_ptr,  # [req_pool_size+1, max_pages]
    # Outputs
    out_cache_loc_ptr,  # [batch_size * uniform_input_length]
    positions_ptr,  # [batch_size * uniform_input_length]
    seq_lens_out_ptr,  # [batch_size]
    # Scalars
    uniform_input_length,
    dummy_kv_slot,
    USE_DUMMY_CACHE_LOC: tl.constexpr,
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
    `valid_cache_lengths[pool_idx]` and reuse it for the seq_lens and position
    writes plus, in legacy scalar-location mode, the page-table lookup.
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
        if USE_DUMMY_CACHE_LOC:
            # Group-keyed flat KV writers resolve their real destinations from
            # per-group table/base metadata. The scalar location is a fixed
            # compatibility sentinel, so do not touch the legacy page table.
            cache_locs = tl.full((BLOCK_SIZE,), dummy_kv_slot, tl.int32)
        else:
            page_indices = positions_local // page_size
            overflow = page_indices >= max_pages
            # Clamp to last valid page to avoid OOB GMEM read.
            page_indices = tl.minimum(page_indices, max_pages - 1)
            offsets_in_page = positions_local % page_size

            page_ptrs = req_to_pages_ptr + pool_idx * max_pages + page_indices
            page_ids = tl.load(page_ptrs, mask=mask, other=0)
            cache_locs = page_ids * page_size + offsets_in_page
            # Route overflow tokens to slot 0 (fixed safe dummy target).
            cache_locs = tl.where(overflow, 0, cache_locs)

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
    req_to_pages: torch.Tensor,  # [req_pool_size+1, max_pages]
    page_size: int,
    *,
    use_dummy_cache_loc: bool = False,
    dummy_kv_slot: int = 0,
) -> None:
    """Decode-only fast path: one Triton launch writes out_cache_loc,
    positions, and seq_lens, reading `valid_cache_lengths[pool_idx]`
    directly so the per-iter indexSelect + add are gone too.
    """
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]
    BLOCK_SIZE = 128
    grid = (batch_size,)
    fused_decode_input_prep_kernel[grid](
        req_pool_indices,
        valid_cache_lengths,
        req_to_pages,
        out_cache_loc_ptr,
        positions_ptr,
        seq_lens_out_ptr,
        uniform_input_length,
        dummy_kv_slot,
        USE_DUMMY_CACHE_LOC=use_dummy_cache_loc,
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
    req_to_pages_ptr,
    draft_seq_lens_ptr,
    block_ids_ptr,
    block_positions_ptr,
    out_cache_loc_ptr,
    spec_num_tokens: tl.constexpr,
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

    safe_accept = tl.minimum(tl.maximum(accept_len, 1), spec_num_tokens)
    current_token = tl.load(
        output_tokens_ptr + req_idx * spec_num_tokens + safe_accept - 1
    )
    tl.store(block_ids_ptr + req_idx * block_ids_stride, current_token)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spec_num_tokens
    positions = prefix_len + offsets
    tl.store(
        block_positions_ptr + req_idx * spec_num_tokens + offsets, positions, mask=mask
    )

    page_indices = positions // page_size
    overflow = page_indices >= max_pages
    page_indices = tl.minimum(page_indices, max_pages - 1)
    offsets_in_page = positions % page_size
    page_ptrs = req_to_pages_ptr + pool_idx * max_pages + page_indices
    page_ids = tl.load(page_ptrs, mask=mask, other=0)
    cache_locs = page_ids * page_size + offsets_in_page
    cache_locs = tl.where(overflow, 0, cache_locs)
    tl.store(
        out_cache_loc_ptr + req_idx * spec_num_tokens + offsets, cache_locs, mask=mask
    )


def dflash_prepare_decode(
    output_tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    req_to_pages: torch.Tensor,
    draft_seq_lens: torch.Tensor,
    block_ids: torch.Tensor,
    block_positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    spec_num_tokens: int,
    page_size: int,
    max_draft_prefix: int,
) -> None:
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(spec_num_tokens)
    grid = (batch_size,)
    dflash_prepare_decode_kernel[grid](
        output_tokens,
        accept_lengths,
        req_pool_indices,
        valid_cache_lengths,
        req_to_pages,
        draft_seq_lens,
        block_ids,
        block_positions,
        out_cache_loc,
        spec_num_tokens=spec_num_tokens,
        page_size=page_size,
        max_pages=max_pages,
        max_draft_prefix=max_draft_prefix,
        block_ids_stride=block_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )


def compute_out_cache_loc_uniform(
    out_cache_loc_ptr,
    req_pool_indices: torch.Tensor,  # [batch_size]
    uniform_input_length: int,
    cache_start: torch.Tensor,  # [batch_size]
    req_to_pages: torch.Tensor,  # [req_pool_size+1, max_pages]
    page_size: int,
) -> None:
    """Specialized entry point when every request has the same ``input_length``.

    Skips the per-call ``torch.full`` + ``cumsum`` host-side work and the
    corresponding GMEM reads inside the kernel. Used by the multi-step drafter
    where each request decodes exactly ``spec_num_steps - 1`` tokens.
    """
    batch_size = req_pool_indices.shape[0]
    max_pages = req_to_pages.shape[1]

    BLOCK_SIZE = 128
    grid = (batch_size,)

    compute_out_cache_loc_kernel[grid](
        req_pool_indices,
        None,  # input_lengths_ptr is None → kernel uses uniform_input_length
        cache_start,
        req_to_pages,
        None,  # cumsum_lengths_ptr is None → kernel computes offset analytically
        out_cache_loc_ptr,
        uniform_input_length,
        page_size=page_size,
        max_pages=max_pages,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def update_block_table(forward_op, device, req_to_page):
    def flatten_and_to_device(data, dtype=torch.int32):
        if not data:
            return torch.tensor([], dtype=dtype, device=device)

        # Flatten one level if data is a list of lists
        if isinstance(data[0], (list, tuple)):
            flat = [x for inner in data for x in inner]
        else:
            flat = data

        if not flat:
            return torch.tensor([], dtype=dtype, device=device)

        tensor = torch.tensor(flat, dtype=dtype, device="cpu", pin_memory=True)
        return tensor.to(device, non_blocking=True)

    # sizes[i] is the number of page-table entries to refresh. Besides newly
    # allocated pages, this can include an existing radix prefix whose physical
    # IDs changed when duplicate request-local pages were canonicalized.
    if all(n == 0 for n in forward_op.sizes):
        return

    max_pages = req_to_page.shape[1]
    # Clamp a request that would overflow req_to_page instead of crashing the
    # engine. Happens when MTP accept-rate collapse keeps a request alive past
    # context_len; its KV drops but it will be finished shortly.
    sizes = list(forward_op.sizes)
    begins = list(forward_op.begins)
    # new_occupied_pages is a list-of-lists [batch, size_i] of page ids;
    # take a shallow copy so we can trim the offending request's row.
    new_occupied_pages = [list(row) for row in forward_op.new_occupied_pages]
    request_ids = list(forward_op.request_ids)
    for i, (begin, size) in enumerate(zip(begins, sizes)):
        if begin + size > max_pages:
            clamped = max(0, max_pages - begin)
            logger.warning(
                "page copy would exceed req_to_page capacity for req %s: "
                "begin=%s + size=%s = %s > req_to_page.shape[1]=%s; "
                "clamping size to %s to avoid engine crash. The request is past "
                "its context-length bound and will be finished by the length "
                "check; KV writes after this point are dropped.",
                request_ids[i] if i < len(request_ids) else "?",
                begin,
                size,
                begin + size,
                max_pages,
                clamped,
            )
            sizes[i] = clamped
            # Keep new_occupied_pages[i] consistent with the clamped size so
            # the kernel's cumsum-based offsets stay aligned across the batch.
            new_occupied_pages[i] = new_occupied_pages[i][:clamped]

    new_occupied_pages_num = flatten_and_to_device(sizes, dtype=torch.int32)
    pages_copy_starts = flatten_and_to_device(begins, dtype=torch.int32)
    new_occupied_pages_t = flatten_and_to_device(new_occupied_pages, dtype=torch.int32)
    request_pool_indices = flatten_and_to_device(
        forward_op.request_pool_indices, dtype=torch.int64
    )
    update_req_to_page(
        req_to_page=req_to_page,
        req_pool_indices=request_pool_indices,
        new_occupied_pages=new_occupied_pages_t,
        new_occupied_pages_num=new_occupied_pages_num,
        pages_copy_starts=pages_copy_starts,
    )
