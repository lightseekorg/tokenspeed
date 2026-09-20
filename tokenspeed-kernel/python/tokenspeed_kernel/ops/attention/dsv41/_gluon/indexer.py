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
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _index_gather_heads,
    _index_topk_outputs,
    cache_pack,
    index_q_quantize,
)

_MFMA_HEADS = 32
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


def _pack_index_q(q: torch.Tensor, weights: torch.Tensor):
    tokens, heads, _ = q.shape
    packed = cache_pack(q.contiguous().reshape(tokens * heads, 128), "index", None)
    values = packed[:, :64].reshape(tokens, heads, 64)
    scales = packed[:, 64:68].contiguous().view(torch.int32).reshape(tokens, heads)
    pad = _MFMA_HEADS - heads
    if pad < 0:
        raise ValueError(
            f"GFX950 CSA2 MFMA indexer supports at most {_MFMA_HEADS} heads"
        )
    if pad:
        values = torch.nn.functional.pad(values, (0, 0, 0, pad))
        scales = torch.nn.functional.pad(scales, (0, pad))
        weights = torch.nn.functional.pad(weights.float(), (0, pad))
    else:
        weights = weights.float()
    return values.contiguous(), scales.contiguous(), weights.contiguous()


def _pad_query(q: torch.Tensor, weights: torch.Tensor):
    q = index_q_quantize(q.contiguous(), "index", None)
    tokens, heads, _ = q.shape
    pad = _MFMA_HEADS - heads
    if pad < 0:
        raise ValueError(f"GFX1250 CSA2 indexer supports at most {_MFMA_HEADS} heads")
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        weights = torch.nn.functional.pad(weights.float(), (0, pad))
    else:
        weights = weights.float()
    return q.contiguous(), weights.contiguous()


def _select_sorted(scores: torch.Tensor, k: int, out: torch.Tensor, lens: torch.Tensor):
    out.fill_(-1)
    lens.zero_()
    width = scores.shape[1]
    take = min(int(k), width)
    if not scores.shape[0] or take < 1:
        return
    values, indices = scores.topk(take, dim=1, sorted=False)
    valid = values > -torch.inf
    ordered = (
        indices.masked_fill(~valid, torch.iinfo(torch.int64).max).sort(dim=1).values
    )
    ordered = ordered.masked_fill(ordered == torch.iinfo(torch.int64).max, -1)
    out[:, :take].copy_(ordered.to(out.dtype))
    lens.copy_(valid.sum(dim=1).to(lens.dtype))


def _logical_from_scan(columns: torch.Tensor, candidate_blocks: torch.Tensor | None):
    if candidate_blocks is None:
        return columns
    block = candidate_blocks.gather(
        1, (columns // 8).clamp(max=candidate_blocks.shape[1] - 1)
    )
    logical = block.to(torch.int64) * 8 + columns % 8
    return torch.where(block >= 0, logical, torch.iinfo(torch.int64).max)


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
    launch_logits,
):
    """Gather, score, and select CSA2 rows with shape-bounded, graph-safe scratch."""
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
    row_out.fill_(-1)
    row_lens.zero_()
    block_out.fill_(-1)
    block_lens.zero_()
    if not tokens or not page_table.shape[1] or need < 1:
        return out

    # Arena pages have gaps between them but contiguous bytes within each page.
    # Flatten only the inner dimensions to preserve their storage and page stride.
    cache_2d = index_cache.flatten(1)
    if cache_2d.stride(1) != 1:
        # Flatten can retain a non-unit byte stride (for example [..., ::2]).
        # The scorers address adjacent page bytes, so normalize that layout only.
        cache_2d = cache_2d.contiguous()
    query_chunk_size = min(int(query_chunk_size), 256)
    make_blocks = bool(candidate_topk)

    for start in range(0, tokens, query_chunk_size):
        end = min(start + query_chunk_size, tokens)
        q, w, _shards = _index_gather_heads(
            index_q[start:end], weights[start:end], process_group
        )
        table = page_table[start:end].contiguous()
        visible = visible_lens[start:end].clamp(0, int(table.shape[1]) * _PAGE_SIZE)
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
            values_topk, columns = logits.topk(
                min(int(topk), width), dim=1, sorted=False
            )
            logical = _logical_from_scan(columns, tile_candidates)
            packed = torch.where(
                values_topk > -torch.inf,
                logical,
                torch.iinfo(torch.int64).max,
            )
            ordered = packed.sort(dim=1).values
            ordered = ordered.masked_fill(ordered == torch.iinfo(torch.int64).max, -1)
            take = ordered.shape[1]
            row_out[output_begin:output_end, :take].copy_(ordered.to(row_out.dtype))
            row_lens[output_begin:output_end].copy_(
                (values_topk > -torch.inf).sum(dim=1).to(row_lens.dtype)
            )
            if make_blocks:
                n_blocks = width // 8
                block_scores = (
                    logits[:, : n_blocks * 8]
                    .reshape(tile_queries, n_blocks, 8)
                    .amax(-1)
                )
                latest = (tile_visible.to(torch.int64) - 1) // 8
                live = (tile_visible > 0) & (latest < n_blocks)
                latest = latest.clamp(0, n_blocks - 1)
                rows = torch.arange(tile_queries, device=q.device)
                current = block_scores[rows, latest]
                block_scores[rows, latest] = torch.where(
                    live & (current > -torch.inf),
                    torch.full_like(current, float("inf")),
                    current,
                )
                _select_sorted(
                    block_scores,
                    int(candidate_topk),
                    block_out[output_begin:output_end],
                    block_lens[output_begin:output_end],
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

    q, w = _pad_query(q, w)
    dsv41_index_logits_gfx1250(
        q,
        w,
        cache_2d,
        table,
        visible,
        candidates,
        logits,
        score_chunk_size,
    )
