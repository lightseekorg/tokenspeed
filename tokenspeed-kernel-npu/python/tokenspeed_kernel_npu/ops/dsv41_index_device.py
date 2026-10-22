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


"""Device-bounded FP32 row selection and BF16-rounded candidate generation.

Native row selection uses FP32 scoring for every query count. Candidate-pool
scans preserve BF16 intermediate rounding and use ordered-key selection for
larger batches or native TopK for small batches. Both retain fixed output
storage for graph replay.
"""

import math

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_index_finish import finish
from tokenspeed_kernel_npu.ops.dsv41_index_scan import _index_score_scan_kernel


def topk_tile(
    q,
    weights,
    cache,
    table,
    visible,
    candidates,
    topk,
    candidate_topk,
    score_chunk_size,
    shards,
    out,
    *,
    prepared_reindex=None,
):
    if cache.dtype != torch.bfloat16:
        raise ValueError("Ascend V4.1 requires BF16 index storage")
    rows, lengths, blocks, block_lengths = out
    if prepared_reindex is not None and (candidates is None or candidate_topk):
        raise ValueError("Prepared reindex requires the native BF16 candidate path")
    n = q.shape[0]
    if not n or not table.shape[1] or not cache.shape[0]:
        rows.fill_(-1)
        lengths.zero_()
        blocks.fill_(-1)
        block_lengths.zero_()
        return out
    if candidates is not None and candidate_topk:
        raise ValueError("Candidate generation requires a full scan")
    width = table.shape[1] * 64 if candidates is None else candidates.shape[1] * 8
    if width and not candidate_topk:
        # Native finish overwrites every row/id and length, including invalid
        # queries. Only the block outputs (not produced on this path) need
        # their empty-result contract initialized here.
        blocks.fill_(-1)
        block_lengths.zero_()
        from tokenspeed_kernel_npu.ops.dsv41_index_native import native_index

        native_index(
            q,
            weights,
            cache,
            table,
            visible,
            candidates,
            topk,
            out=(rows, lengths),
            prepared_reindex=prepared_reindex,
        )
        return out
    rows.fill_(-1)
    lengths.zero_()
    blocks.fill_(-1)
    block_lengths.zero_()
    if not width:
        return out
    capacity = max(width, topk, candidate_topk * 8)
    capacity = (capacity + 7) // 8 * 8
    # Small batches favor native TopK; larger batches avoid scanning the
    # whole reserved context. Both branches have fixed graph buffer shapes.
    scores = (
        torch.full((n, capacity), -torch.inf, device=q.device, dtype=torch.float32)
        if n < 16
        else torch.empty((n, capacity), device=q.device, dtype=torch.float32)
    )
    from tokenspeed_kernel_npu.ops.dsv41_index_scan_native import (
        scan_scores,
    )

    use_native_scan = (
        q.shape[1:] == (32, 128)
        and q.dtype == weights.dtype == cache.dtype == torch.bfloat16
        and cache.is_contiguous()
        and table.dtype == visible.dtype == torch.int32
        and scores.shape[1] == width
    )
    if use_native_scan:
        # Balance complete query-pair/partition jobs across the 24 Cube cores.
        # Partitioning changes only ownership of columns, never their reduction.
        parts = 24 // math.gcd(24, (n + 1) // 2)
        scan_scores(
            q, weights, cache, table, visible, scores, shards=shards, parts=parts
        )
    else:
        parts = 16
        # BF16 full scans consume a complete 64-row page per matrix tile.
        tile = 64
        _index_score_scan_kernel[n, parts](
            q,
            weights,
            cache,
            table,
            visible,
            scores,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            weights.stride(0),
            weights.stride(1),
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            table.stride(0),
            table.stride(1),
            visible.stride(0),
            capacity,
            cache.shape[0],
            table.shape[1],
            q.shape[1],
            q.shape[1] // shards,
            shards,
            q.shape[1],
            parts,
            tile,
            tile,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    if n < 16:
        values, columns = scores.topk(topk, dim=-1, sorted=False)
        finish(values, columns, rows, lengths)
        from tokenspeed_kernel_npu.ops.dsv41_index_select import block_scores_for_topk

        block_scores = block_scores_for_topk(scores, visible)
        values, ids = block_scores.topk(candidate_topk, dim=-1, sorted=False)
        finish(values, ids, blocks, block_lengths)
        return out
    from tokenspeed_kernel_npu.ops.dsv41_index_select import select_full_scores

    selected = select_full_scores(
        scores, visible, topk, candidate_topk, logical_capacity=width
    )
    for (values, ids), dest, lens in zip(
        selected, (rows, blocks), (lengths, block_lengths)
    ):
        if values.shape[1]:
            finish(values, ids, dest, lens)
    return out


@triton.jit(do_not_specialize=["TS0", "TS1", "LS", "TC", "PAGES", "K", "C"])
def _select_all(
    Table,
    Visible,
    Rows,
    Lengths,
    Blocks,
    BlockLengths,
    TS0,
    TS1,
    LS,
    TC,
    PAGES,
    K,
    C,
    RB: tl.constexpr,
    CB: tl.constexpr,
):
    row = tl.program_id(0)
    visible = tl.load(Visible + row * LS)
    i = tl.arange(0, RB)
    page = tl.full((RB,), -1, Table.dtype.element_ty)
    if TC > 0:
        page = tl.load(
            Table + row * TS0 + (i // 64) * TS1, (i < K) & (i // 64 < TC), other=-1
        )
    valid = (i < K) & (i < visible) & (page > 0) & (page < PAGES)
    count = tl.sum(valid.to(tl.int32), 0)
    prefix = tl.cumsum(valid.to(tl.int32))
    tl.store(Rows + row * K + prefix - 1, i, valid)
    tl.store(Rows + row * K + i, -1, (i < K) & (i >= count))
    tl.store(Lengths + row, count)
    block_count = tl.full((), 0, tl.int32)
    if C:
        j = tl.arange(0, CB)
        offset = j * 8
        candidate_page = tl.full((CB,), -1, Table.dtype.element_ty)
        if TC > 0:
            candidate_page = tl.load(
                Table + row * TS0 + (offset // 64) * TS1,
                (j < C) & (offset // 64 < TC),
                other=-1,
            )
        valid_block = (
            (j < C)
            & (offset < visible)
            & (candidate_page > 0)
            & (candidate_page < PAGES)
        )
        block_count = tl.sum(valid_block.to(tl.int32), 0)
        prefix_block = tl.cumsum(valid_block.to(tl.int32))
        tl.store(Blocks + row * C + prefix_block - 1, j, valid_block)
        tl.store(Blocks + row * C + j, -1, (j < C) & (j >= block_count))
    tl.store(BlockLengths + row, block_count)


def select_all(page_table, visible_lens, pages, candidate_block_size, out):
    """Compact ascending visible IDs in one launch, without scores or a sort."""
    if candidate_block_size != 8:
        raise ValueError("V4.1 candidate blocks contain eight rows")
    if page_table.shape[0]:
        _select_all[(page_table.shape[0],)](
            page_table,
            visible_lens,
            *out,
            *page_table.stride(),
            visible_lens.stride(0),
            page_table.shape[1],
            pages,
            out[0].shape[1],
            out[2].shape[1],
            RB=triton.next_power_of_2(out[0].shape[1]),
            CB=triton.next_power_of_2(max(1, out[2].shape[1])),
            num_warps=4,
        )
    return out
