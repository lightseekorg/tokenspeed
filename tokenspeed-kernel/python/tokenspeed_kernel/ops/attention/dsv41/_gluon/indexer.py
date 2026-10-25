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

"""Shared validation, query preparation, and CSA2 selection for AMD scorers."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _index_gather_heads,
    _index_topk_outputs,
    _mxfp4_index_rows,
    _ue8m0_word,
)

_PAGE_SIZE = 64
# Match the native paged scorer's transient score budget. Heads are gathered
# once per API chunk; only score/selection work is split at wider histories.
_LOGITS_BUDGET_BYTES = 32 << 20
_FP32_BYTES = 4


def _score_query_tile(queries: int, width: int) -> int:
    """Fit as many queries as possible in one bounded FP32 logits tile."""
    return min(
        queries,
        max(1, _LOGITS_BUDGET_BYTES // (width * _FP32_BYTES)),
    )


@triton.jit(
    do_not_specialize=["Q0", "Q1", "Q2", "W0", "W1"],
)
def _pack_index_queries_padded(
    Q,
    W,
    V,
    S,
    F,
    Q0,
    Q1,
    Q2,
    W0,
    W1,
    HEADS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 128)
    live = head < HEADS
    # Inner-strided views can span more than int32 elements within one head.
    x = tl.load(Q + token * Q0 + head * Q1 + d.to(tl.int64) * Q2, live, 0.0).to(
        tl.float32
    )
    grouped = tl.reshape(x, (4, 32))
    # Match cache_pack(index): multiplication by the rounded reciprocal of six.
    raw = tl.maximum(tl.max(tl.abs(grouped), 1), 6 * 2.0**-126) * (1.0 / 6.0)
    packed, exponent = _mxfp4_index_rows(
        tl.reshape(grouped, (1, 4, 32)), tl.reshape(raw, (1, 4)), 1
    )
    packed = tl.reshape(packed, (64,))
    word = tl.sum(_ue8m0_word(exponent), 0)
    row = token * 32 + head
    tl.store(V + row * 64 + tl.arange(0, 64), tl.where(live, packed, 0))
    tl.store(S + row, tl.where(live, word.to(tl.int32), 0))
    weight = tl.load(W + token * W0 + head * W1, live, 0.0).to(tl.float32)
    tl.store(F + row, weight)


def _pack_index_q(q: torch.Tensor, weights: torch.Tensor):
    """Write AMD's dense 32-head FP4 planes and FP32 weights without relayouts."""
    tokens, heads, _ = q.shape
    if not 1 <= heads <= 32:
        raise ValueError("AMD CSA2 MXFP4 indexer supports 1..32 heads")
    values = torch.empty((tokens, 32, 64), dtype=torch.uint8, device=q.device)
    scales = torch.empty((tokens, 32), dtype=torch.int32, device=q.device)
    folded = torch.empty((tokens, 32), dtype=torch.float32, device=q.device)
    if tokens:
        _pack_index_queries_padded[(tokens, 32)](
            q,
            weights,
            values,
            scales,
            folded,
            *q.stride(),
            *weights.stride(),
            heads,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return values, scales, folded


def _logical_from_scan(columns: torch.Tensor, candidate_blocks: torch.Tensor | None):
    if candidate_blocks is None:
        return columns
    block = candidate_blocks.gather(
        1, (columns // 8).clamp(max=candidate_blocks.shape[1] - 1)
    )
    logical = block.to(torch.int64) * 8 + columns % 8
    return torch.where(block >= 0, logical, torch.iinfo(torch.int64).max)


def select_rows_torch(logits, candidates, live_widths, topk, row_out, row_lens):
    """Write each row's top-``topk`` finite logits as position-sorted ids.

    Column c maps to logical row c, or ``candidates[q, c // 8] * 8 + c % 8``.
    ``live_widths`` (None or [Q]) bounds the columns that may be finite; the
    rest are -inf, so this reference may scan whole rows and ignore it.
    Row q of ``row_out`` gets the ids ascending, then -1; ``row_lens`` the
    count. Equal-score boundary ties may pick any equivalent subset.
    """
    width = logits.shape[1]
    values_topk, columns = logits.topk(min(int(topk), width), dim=1, sorted=False)
    _write_selected_rows(values_topk, columns, candidates, row_out, row_lens)


def _write_selected_rows(values_topk, columns, candidates, row_out, row_lens):
    logical = _logical_from_scan(columns, candidates)
    packed = torch.where(
        values_topk > -torch.inf,
        logical,
        torch.iinfo(torch.int64).max,
    )
    ordered = packed.sort(dim=1).values
    ordered = ordered.masked_fill(ordered == torch.iinfo(torch.int64).max, -1)
    take = ordered.shape[1]
    row_out[:, :take].copy_(ordered.to(row_out.dtype))
    row_out[:, take:].fill_(-1)
    row_lens.copy_((values_topk > -torch.inf).sum(dim=1).to(row_lens.dtype))


def run_dsv41_csa2_index_topk(
    index_q,
    weights,
    index_cache,
    page_table,
    visible_lens,
    candidate_blocks,
    topk,
    candidate_topk,
    candidate_block_size,
    query_chunk_size,
    score_chunk_size,
    process_group,
    out,
    index_metadata,
    launch_logits,
    select_rows,
    select_blocks,
):
    """Gather, score, and select CSA2 rows with shape-bounded, graph-safe scratch.

    ``launch_logits`` scores a query tile into -inf-initialized FP32 logits;
    ``select_rows`` has the ``select_rows_torch`` contract and
    ``select_blocks`` the ``index_block_select`` contract.
    """
    del index_metadata
    out = _index_topk_outputs(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
    )
    if index_cache.shape[2] != 68:
        raise ValueError("AMD Gluon CSA2 requires 68-byte MXFP4 index rows")
    if process_group is not None:
        raise ValueError("AMD Gluon CSA2 requires local or replicated index heads")
    # The shared contract requires eight-row candidate blocks.
    need = (
        int(candidate_blocks.shape[1]) * 8
        if candidate_blocks is not None
        else int(page_table.shape[1]) * _PAGE_SIZE
    )

    tokens = index_q.shape[0]
    row_out, row_lens, block_out, block_lens = out
    if not tokens or not page_table.shape[1] or need < 1:
        row_out.fill_(-1)
        row_lens.zero_()
        block_out.fill_(-1)
        block_lens.zero_()
        return out

    # Arena pages have gaps between them but contiguous bytes within each page.
    # Flatten only the inner dimensions to preserve their storage and page stride.
    cache_2d = index_cache.flatten(1)
    if cache_2d.stride(1) != 1 or cache_2d.stride(0) % 16:
        # Flatten can retain a non-unit byte stride (for example [..., ::2]).
        # The scorers address adjacent page bytes, and the gfx1250 scorer
        # loads 16-byte row chunks, so normalize only those layouts. Runtime
        # arenas align index pages to 256 bytes and never take this copy.
        cache_2d = cache_2d.contiguous()
    query_chunk_size = min(int(query_chunk_size), 256)
    make_blocks = bool(candidate_topk)
    if not make_blocks:
        block_lens.zero_()

    for start in range(0, tokens, query_chunk_size):
        end = min(start + query_chunk_size, tokens)
        q, w, _shards = _index_gather_heads(
            index_q[start:end], weights[start:end], process_group
        )
        table = page_table[start:end].contiguous()
        # Consumers bound raw int64 lengths before narrowing. A dense input
        # needs no launch here; preserve the previous strided-input support.
        visible = visible_lens[start:end].contiguous()
        candidates = (
            None
            if candidate_blocks is None
            else candidate_blocks[start:end].contiguous()
        )
        queries = end - start
        width = need
        score_query_tile = _score_query_tile(queries, width)
        for query_begin in range(0, queries, score_query_tile):
            query_end = min(query_begin + score_query_tile, queries)
            output_begin = start + query_begin
            output_end = start + query_end
            tile_candidates = (
                None if candidates is None else candidates[query_begin:query_end]
            )
            tile_visible = visible[query_begin:query_end]
            tile_queries = query_end - query_begin
            logits = torch.full(
                (tile_queries, width),
                -float("inf"),
                dtype=torch.float32,
                device=q.device,
            )
            launch_logits(
                q[query_begin:query_end],
                w[query_begin:query_end],
                cache_2d,
                table[query_begin:query_end],
                tile_visible,
                tile_candidates,
                logits,
                score_chunk_size,
            )
            select_rows(
                logits,
                tile_candidates,
                # Full selection scores only each query's visible rows; the
                # rest of its page-table-capacity row stays -inf.
                tile_visible if tile_candidates is None else None,
                int(topk),
                row_out[output_begin:output_end],
                row_lens[output_begin:output_end],
            )
            if make_blocks:
                select_blocks(
                    logits,
                    tile_visible,
                    int(candidate_topk),
                    block_out[output_begin:output_end],
                    block_lens[output_begin:output_end],
                    visible_capacity=int(table.shape[1]) * _PAGE_SIZE,
                )
    return out


def launch_gfx950_logits(
    q, w, cache_2d, table, visible, candidates, logits, score_chunk_size
):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41 import (
        dsv41_index_logits_gfx950,
    )

    values, scales, w = _pack_index_q(q, w)
    dsv41_index_logits_gfx950(
        values,
        scales,
        w,
        cache_2d,
        table,
        visible,
        candidates,
        logits,
        score_chunk_size,
    )


def launch_gfx1250_logits(
    q, w, cache_2d, table, visible, candidates, logits, score_chunk_size
):
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41 import (
        dsv41_index_logits_gfx1250,
    )

    values, scales, w = _pack_index_q(q, w)
    dsv41_index_logits_gfx1250(
        values,
        scales,
        w,
        cache_2d,
        table,
        visible,
        candidates,
        logits,
        score_chunk_size,
    )


def select_rows_gfx950(logits, candidates, live_widths, topk, row_out, row_lens):
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41 import (
        launch_gluon_dsv41_index_topk_select_gfx950,
    )

    launch_gluon_dsv41_index_topk_select_gfx950(
        logits, candidates, live_widths, topk, row_out, row_lens
    )


def select_rows_gfx1250(logits, candidates, live_widths, topk, row_out, row_lens):
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41 import (
        launch_gluon_dsv41_index_topk_select_gfx1250,
    )

    launch_gluon_dsv41_index_topk_select_gfx1250(
        logits, candidates, live_widths, topk, row_out, row_lens
    )
