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
    new_occupied_pages_ptr,  # [total_pages] - flattened
    new_occupied_pages_num_ptr,  # [batch_size]
    pages_copy_starts_ptr,  # [batch_size]
    cumsum_pages_ptr,  # [batch_size] - cumulative sum of new_occupied_pages_num
    # Output pointer
    req_to_page_ptr,  # [req_pool_size+1, context_len]
    # Scalars
    context_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Update req_to_page table with new occupied pages.
    Each program handles one request in the batch.
    """
    req_idx = tl.program_id(0)

    # Load request metadata
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    num_pages = tl.load(new_occupied_pages_num_ptr + req_idx)
    copy_start = tl.load(pages_copy_starts_ptr + req_idx)

    # Get offset into flattened new_occupied_pages
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

        # Load new page IDs
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
    Update req_to_page table with new occupied pages using Triton kernel.

    Args:
        req_to_page: Request to page table [req_pool_size+1, context_len]
        req_pool_indices: Request pool indices [batch_size]
        new_occupied_pages: New page IDs [total_pages] - flattened
        new_occupied_pages_num: Number of new pages per request [batch_size]
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
        offsets_in_page = positions % page_size

        # Load page IDs from req_to_pages
        # req_to_pages is [req_pool_size+1, max_pages]
        page_ptrs = req_to_pages_ptr + req_pool_idx * max_pages + page_indices
        page_ids = tl.load(page_ptrs, mask=mask, other=0)

        # Compute physical cache locations
        cache_locs = page_ids * page_size + offsets_in_page

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

    # sizes[i] is the number of newly allocated pages for request i.
    if all(n == 0 for n in forward_op.sizes):
        return

    max_pages = req_to_page.shape[1]
    for begin, size in zip(forward_op.begins, forward_op.sizes):
        if begin + size > max_pages:
            raise RuntimeError(
                f"page copy would exceed req_to_page capacity: "
                f"begin={begin} + size={size} = {begin + size} "
                f"> req_to_page.shape[1]={max_pages}"
            )

    new_occupied_pages_num = flatten_and_to_device(forward_op.sizes, dtype=torch.int32)
    pages_copy_starts = flatten_and_to_device(forward_op.begins, dtype=torch.int32)
    new_occupied_pages = flatten_and_to_device(
        forward_op.new_occupied_pages, dtype=torch.int32
    )
    request_pool_indices = flatten_and_to_device(
        forward_op.request_pool_indices, dtype=torch.int32
    )
    update_req_to_page(
        req_to_page=req_to_page,
        req_pool_indices=request_pool_indices,
        new_occupied_pages=new_occupied_pages,
        new_occupied_pages_num=new_occupied_pages_num,
        pages_copy_starts=pages_copy_starts,
    )
