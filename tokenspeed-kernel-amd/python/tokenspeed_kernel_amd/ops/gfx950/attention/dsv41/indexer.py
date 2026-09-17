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

"""GFX950 DeepSeek V4.1 CSA2 indexer.

Scores page-planar MXFP4 index-K with the V4 ``mfma_scaled`` e2m1 tile, then
selects Top512 (and CSA2 blocks) on ATen. Histories wider than 32K fall back
to the portable Triton scan so serving at 1M does not materialize a dense
score matrix.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.indexer import (
    _BLOCK_N,
    _CHUNK_N,
    _HEADS_PER_MFMA,
    _PACKED_DIM,
    _PAGE_SIZE,
    _SCALE_DIM,
    _indexer_mfma_layouts,
    _load_query_group,
)

__all__ = [
    "gluon_dsv41_index_topk_gfx950",
    "run_dsv41_csa2_index_topk",
]

_MFMA_HEADS = 32
_MAX_LOGITS = 32768


@gluon.jit
def _csa2_page_rows(
    positions,
    valid,
    page_table,
    candidates,
    query,
    table_stride,
    cand_stride,
    num_pages,
    visible,
    PAGE_SIZE: gl.constexpr,
    TABLE_WIDTH: gl.constexpr,
    CANDIDATES: gl.constexpr,
):
    if CANDIDATES >= 0:
        block = gl.amd.cdna4.buffer_load(
            ptr=candidates,
            offsets=(query * cand_stride + positions // 8).to(gl.int32),
            mask=valid,
            other=-1,
        ).to(gl.int64)
        logical = gl.where(block >= 0, block * 8 + positions % 8, -1)
        valid = valid & (logical >= 0) & (logical < visible)
    else:
        logical = positions.to(gl.int64)
        valid = valid & (logical < visible)
    logical_page = logical // PAGE_SIZE
    safe_page = gl.minimum(gl.maximum(logical_page, 0), TABLE_WIDTH - 1)
    physical = gl.amd.cdna4.buffer_load(
        ptr=page_table,
        offsets=(query * table_stride + safe_page).to(gl.int32),
        mask=valid,
        other=-1,
    ).to(gl.int64)
    valid = valid & (physical >= 0) & (physical < num_pages)
    return gl.where(valid, physical, 0), logical % PAGE_SIZE, valid


@gluon.jit
def _score_csa2_group(
    query,
    query_scales,
    head_weights,
    index_k_cache,
    page_table,
    candidates,
    token,
    tile_start,
    candidate_end,
    table_stride,
    cand_stride,
    page_stride_bytes,
    num_pages,
    visible,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    TABLE_WIDTH: gl.constexpr,
    CANDIDATES: gl.constexpr,
):
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
        :, None
    ]
    columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    positions = tile_start + columns
    valid = positions < candidate_end
    pages, page_rows, valid = _csa2_page_rows(
        positions,
        valid,
        page_table,
        candidates,
        token,
        table_stride,
        cand_stride,
        num_pages,
        visible,
        PAGE_SIZE,
        TABLE_WIDTH,
        CANDIDATES,
    )
    key_offsets = (
        pages * page_stride_bytes + page_rows.to(gl.int64) * _PACKED_DIM + packed_dims
    )
    key = gl.load(index_k_cache + key_offsets, mask=valid, other=0)

    scale_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[
        :, None
    ]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout))[
        None, :
    ]
    scale_positions = tile_start + scale_columns
    scale_valid = scale_positions < candidate_end
    scale_pages, scale_page_rows, scale_valid = _csa2_page_rows(
        scale_positions,
        scale_valid,
        page_table,
        candidates,
        token,
        table_stride,
        cand_stride,
        num_pages,
        visible,
        PAGE_SIZE,
        TABLE_WIDTH,
        CANDIDATES,
    )
    key_scale_offsets = (
        scale_pages * page_stride_bytes
        + PAGE_SIZE * _PACKED_DIM
        + scale_page_rows.to(gl.int64) * _SCALE_DIM
        + scale_groups
    )
    key_scales = gl.load(
        index_k_cache + key_scale_offsets,
        mask=scale_valid,
        other=127,
    )
    accumulator = gl.zeros(
        [_HEADS_PER_MFMA, _BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    head_scores = gl.amd.cdna4.mfma_scaled(
        a=query,
        a_scale=query_scales,
        a_format="e2m1",
        b=key,
        b_scale=key_scales,
        b_format="e2m1",
        acc=accumulator,
    )
    head_scores = gl.maximum(
        head_scores,
        0.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return gl.sum(head_scores * head_weights[:, None], axis=0)


@gluon.jit(
    do_not_specialize=(
        "stride_q_token",
        "stride_q_head",
        "stride_q_scale_token",
        "stride_q_scale_head",
        "stride_w_token",
        "stride_w_head",
        "table_stride",
        "cand_stride",
        "logits_stride",
        "page_stride_bytes",
        "num_pages",
    )
)
def _dsv41_mxfp4_logits_kernel(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table,
    candidates,
    logits,
    stride_q_token,
    stride_q_head,
    stride_q_scale_token,
    stride_q_scale_head,
    stride_w_token,
    stride_w_head,
    table_stride,
    cand_stride,
    logits_stride,
    page_stride_bytes,
    num_pages,
    max_candidates,
    NUM_HEADS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    TABLE_WIDTH: gl.constexpr,
    CANDIDATES: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    vis = gl.minimum(
        gl.maximum(gl.load(visible + token).to(gl.int32), 0),
        TABLE_WIDTH * PAGE_SIZE,
    )
    if CANDIDATES >= 0:
        width = gl.where(vis > 0, CANDIDATES * 8, 0)
    else:
        width = vis
    candidate_start = split * CHUNK_N
    candidate_end = gl.minimum(width, candidate_start + CHUNK_N)
    candidate_end = gl.minimum(candidate_end, max_candidates)
    if candidate_start >= candidate_end:
        return

    layouts: gl.constexpr = _indexer_mfma_layouts(NUM_WARPS)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    query_0, query_scale_0, weight_0 = _load_query_group(
        q,
        q_scales,
        weights,
        token,
        0,
        stride_q_token,
        stride_q_head,
        stride_q_scale_token,
        stride_q_scale_head,
        stride_w_token,
        stride_w_head,
        mfma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    query_1, query_scale_1, weight_1 = _load_query_group(
        q,
        q_scales,
        weights,
        token,
        16,
        stride_q_token,
        stride_q_head,
        stride_q_scale_token,
        stride_q_scale_head,
        stride_w_token,
        stride_w_head,
        mfma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
        scores = _score_csa2_group(
            query_0,
            query_scale_0,
            weight_0,
            index_k_cache,
            page_table,
            candidates,
            token,
            tile_start,
            candidate_end,
            table_stride,
            cand_stride,
            page_stride_bytes,
            num_pages,
            vis,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            TABLE_WIDTH,
            CANDIDATES,
        )
        scores += _score_csa2_group(
            query_1,
            query_scale_1,
            weight_1,
            index_k_cache,
            page_table,
            candidates,
            token,
            tile_start,
            candidate_end,
            table_stride,
            cand_stride,
            page_stride_bytes,
            num_pages,
            vis,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            TABLE_WIDTH,
            CANDIDATES,
        )
        positions = tile_start + output_columns
        live = positions < candidate_end
        _, _, live = _csa2_page_rows(
            positions,
            live,
            page_table,
            candidates,
            token,
            table_stride,
            cand_stride,
            num_pages,
            vis,
            PAGE_SIZE,
            TABLE_WIDTH,
            CANDIDATES,
        )
        gl.store(
            logits + token * logits_stride + positions,
            gl.where(live, scores, -float("inf")),
            mask=positions < max_candidates,
        )


def _pack_index_q(q: torch.Tensor, weights: torch.Tensor):
    from tokenspeed_kernel.ops.attention.dsv41.triton import cache_pack

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
    """Shared CSA2 host: gather, score into logits, ATen TopK. Wide histories use Triton."""
    from tokenspeed_kernel.ops.attention.dsv41.triton import (
        _index_gather_heads,
        _index_topk_outputs,
        index_topk as portable_index_topk,
    )

    need = (
        int(candidate_blocks.shape[1]) * 8
        if candidate_blocks is not None
        else int(page_table.shape[1]) * _PAGE_SIZE
    )
    if need > _MAX_LOGITS:
        return portable_index_topk(
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
    tokens = index_q.shape[0]
    row_out, row_lens, block_out, block_lens = out
    row_out.fill_(-1)
    row_lens.zero_()
    block_out.fill_(-1)
    block_lens.zero_()
    if not tokens or not page_table.shape[1] or need < 1:
        return out

    cache = index_cache.contiguous()
    cache_2d = cache.view(cache.shape[0], cache.shape[1] * cache.shape[2])
    query_chunk_size = min(int(query_chunk_size), 256)
    make_blocks = bool(candidate_topk)

    for start in range(0, tokens, query_chunk_size):
        end = min(start + query_chunk_size, tokens)
        q, w, _shards = _index_gather_heads(
            index_q[start:end], weights[start:end], process_group
        )
        table = page_table[start:end].contiguous()
        visible = visible_lens[start:end].contiguous()
        candidates = (
            None
            if candidate_blocks is None
            else candidate_blocks[start:end].contiguous()
        )
        queries = end - start
        width = (
            int(candidates.shape[1]) * 8
            if candidates is not None
            else int(table.shape[1]) * _PAGE_SIZE
        )
        width = min(width, need, _MAX_LOGITS)
        if width < 1:
            continue
        logits = torch.full(
            (queries, width),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
        launch_logits(q, w, cache_2d, table, visible, candidates, logits)
        values_topk, columns = logits.topk(min(int(topk), width), dim=1, sorted=False)
        logical = _logical_from_scan(columns, candidates)
        packed = torch.where(
            values_topk > -torch.inf, logical, torch.iinfo(torch.int64).max
        )
        ordered = packed.sort(dim=1).values
        ordered = ordered.masked_fill(ordered == torch.iinfo(torch.int64).max, -1)
        take = ordered.shape[1]
        row_out[start:end, :take].copy_(ordered.to(row_out.dtype))
        row_lens[start:end].copy_(
            (values_topk > -torch.inf).sum(dim=1).to(row_lens.dtype)
        )
        if make_blocks:
            n_blocks = width // 8
            block_scores = (
                logits[:, : n_blocks * 8].reshape(queries, n_blocks, 8).amax(-1)
            )
            latest = ((visible.to(torch.int64) - 1) // 8).clamp(0, n_blocks - 1)
            live = (visible > 0) & (latest < n_blocks)
            rows = torch.arange(queries, device=q.device)
            current = block_scores[rows, latest]
            block_scores[rows, latest] = torch.where(
                live, torch.full_like(current, float("inf")), current
            )
            _select_sorted(
                block_scores,
                int(candidate_topk),
                block_out[start:end],
                block_lens[start:end],
            )
    return out


def _launch_gfx950_logits(q, w, cache_2d, table, visible, candidates, logits):
    values, scales, w = _pack_index_q(q, w)
    queries, width = logits.shape
    cand = table if candidates is None else candidates
    scale_dim = 4
    _dsv41_mxfp4_logits_kernel[(queries, triton.cdiv(width, _CHUNK_N))](
        values,
        scales.view(torch.uint8).reshape(queries, _MFMA_HEADS, scale_dim),
        w,
        cache_2d,
        visible,
        table,
        cand,
        logits,
        values.stride(0),
        values.stride(1),
        scale_dim * _MFMA_HEADS,
        scale_dim,
        w.stride(0),
        w.stride(1),
        table.stride(0),
        cand.stride(0),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        NUM_HEADS=_MFMA_HEADS,
        PAGE_SIZE=_PAGE_SIZE,
        TABLE_WIDTH=int(table.shape[1]),
        CANDIDATES=-1 if candidates is None else int(candidates.shape[1]),
        BLOCK_N=32,
        CHUNK_N=_CHUNK_N,
        NUM_WARPS=2,
        num_warps=2,
        waves_per_eu=2,
    )


def gluon_dsv41_index_topk_gfx950(
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
):
    """GFX950 CSA2 indexer: MXFP4 MFMA score plus ATen TopK."""
    return run_dsv41_csa2_index_topk(
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
        _launch_gfx950_logits,
    )
