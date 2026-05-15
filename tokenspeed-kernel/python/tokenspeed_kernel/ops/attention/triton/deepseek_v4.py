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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["deepseek_v4_indexer_decode_metadata_compute"]


@triton.jit
def _deepseek_v4_indexer_decode_metadata_kernel(
    out_block_tables_ptr,
    out_block_tables_stride,
    out_context_lens_ptr,
    positions_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    rows: tl.constexpr,
    cols: tl.constexpr,
    compress_ratio: tl.constexpr,
    cache_block_size: tl.constexpr,
    max_blocks: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    compressed_lens = tl.maximum((pos + 1) // compress_ratio, 0)
    req = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)
    req_valid = (req >= 0) & (req < rows)
    safe_req = tl.maximum(0, tl.minimum(req, rows - 1))
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


def deepseek_v4_indexer_decode_metadata_compute(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    max_blocks: int,
    out_context_lens: torch.Tensor,
    out_block_tables: torch.Tensor,
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
    _deepseek_v4_indexer_decode_metadata_kernel[(num_tokens,)](
        out_block_tables,
        out_block_tables.stride(0),
        out_context_lens,
        positions_i64,
        token_to_req_indices_i32,
        block_table_i32,
        block_table_i32.stride(0),
        rows=rows,
        cols=cols,
        compress_ratio=int(compress_ratio),
        cache_block_size=int(cache_block_size),
        max_blocks=int(max_blocks),
        candidate_block=candidate_block,
    )
