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

"""QSA indexer block scoring and top-k selection for GFX950.

``gluon_qsa_block_scores_gfx950`` scores compressed KV blocks for a tile of
query rows with one MFMA: the rows' index heads form the M dimension, so a
gathered key tile is shared by every row of the tile that belongs to its
request. ``gluon_qsa_block_topk_select_gfx950`` radix-selects each row's top
blocks from the FP32 scores. See the "gfx950 QSA indexer" section of
``ops/README.md``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

__all__ = [
    "launch_gluon_qsa_block_scores_gfx950",
    "launch_gluon_qsa_block_topk_select_gfx950",
]

# MFMA M rows (query rows x padded index heads) per large-batch score tile.
_SCORE_BLOCK_M = 64
# Batches from this many rows (prefill) use the large score tile.
_SCORE_LARGE_ROWS = 512
# Score programs targeted per launch (about four per CU).
_SCORE_TARGET_PROGRAMS = 1024
# Compressed blocks per score tile.
_SCORE_BLOCK_N = 128
# Columns scanned per selection step.
_SELECT_CHUNK = 4096


def _score_metadata(grid, kernel, args):
    rows = args["query"].shape[0]
    heads = args["NUM_HEADS"]
    head_dim = args["HEAD_DIM"]
    num_blocks = args["num_blocks"]
    return {
        "name": kernel.name,
        "flops16": 2 * rows * heads * head_dim * num_blocks,
        "bytes": grid[0] * num_blocks * head_dim * 2
        + rows * (heads * head_dim * 2 + num_blocks * 4),
    }


def _select_metadata(grid, kernel, args):
    rows = grid[0]
    # Four radix passes plus the gather pass read the row's scores.
    return {
        "name": kernel.name,
        "bytes": rows * (5 * args["n_cols"] * 4 + args["TOPK"] * 4),
    }


@gluon.jit
def _load_key_tile(
    page_table,
    key_cache,
    request,
    tile,
    frontier,
    page_size,
    stride_k_n,
    stride_pt_b,
    k_layout: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HEAD_DIM_PAD: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    # Gather the compressed keys of blocks [tile * BLOCK_N, ...) below the
    # run's frontier straight into the MFMA B-operand layout.
    k_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_layout))
    k_d = gl.arange(0, HEAD_DIM_PAD, layout=gl.SliceLayout(1, k_layout))
    live = k_n < frontier
    # Int64 lengths can widen tile offsets; the host bounds the page table to
    # 2 GiB, so valid buffer offsets fit in int32. Key addresses stay int64.
    pages = cdna4.buffer_load(
        page_table,
        (request * stride_pt_b + k_n // page_size).to(gl.int32),
        mask=live,
        other=0,
    )
    slots = pages.to(gl.int64) * page_size + k_n % page_size
    # 64-bit addressing: the compressed cache may exceed the 2 GiB reach of
    # buffer loads.
    return gl.load(
        key_cache + slots[None, :] * stride_k_n + k_d[:, None],
        mask=live[None, :] & (k_d < HEAD_DIM)[:, None],
        other=0.0,
    )


@gluon.jit(
    launch_metadata=_score_metadata,
    do_not_specialize=["rows", "num_blocks", "col_splits"],
)
def gluon_qsa_block_scores_gfx950(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    logits,
    rows,
    num_blocks,
    page_size,
    col_splits,
    stride_q_n,
    stride_q_h,
    stride_k_n,
    stride_pt_b,
    stride_l_n,
    NUM_HEADS: gl.constexpr,
    HEADS_PAD: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HEAD_DIM_PAD: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_N: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = BLOCK_Q * HEADS_PAD
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        # Warps split the blocks: each gathered key is loaded by one warp,
        # while the small query tile is shared through L1.
        warps_per_cta=[1, num_warps],
    )
    q_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    k_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 32], [num_warps, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, out_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, out_layout)

    # A program owns BLOCK_Q query rows and every col_splits-th block tile.
    row0 = gl.program_id(0) * BLOCK_Q
    split = gl.program_id(1)
    q_idx = gl.arange(0, BLOCK_Q, layout=row_layout)
    q_rows = row0 + q_idx
    row_mask = q_rows < rows
    requests = cdna4.buffer_load(request_indices, q_rows, mask=row_mask, other=-1).to(
        gl.int32
    )
    complete = cdna4.buffer_load(complete_blocks, q_rows, mask=row_mask, other=0)
    out_rows = logits + q_rows[:, None].to(gl.int64) * stride_l_n

    # M index m = row * HEADS_PAD + head, so each row's heads are adjacent.
    m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_layout))
    d = gl.arange(0, HEAD_DIM_PAD, layout=gl.SliceLayout(0, q_layout))
    m_row = row0 + m // HEADS_PAD
    m_head = m % HEADS_PAD
    q_tile = cdna4.buffer_load(
        query,
        (m_row * stride_q_n + m_head * stride_q_h)[:, None] + d[None, :],
        mask=((m_row < rows) & (m_head < NUM_HEADS))[:, None] & (d < HEAD_DIM)[None, :],
        other=0.0,
    )

    # Rows of one request are contiguous: score one request run at a time,
    # sharing each gathered key tile across the run's rows.
    start = 0
    while start < BLOCK_Q:
        request = gl.sum(gl.where(q_idx == start, requests, 0), axis=0)
        run_end = gl.min(
            gl.where((q_idx > start) & (requests != request), q_idx, BLOCK_Q), axis=0
        )
        in_run = row_mask & (q_idx >= start) & (q_idx < run_end)
        frontier = gl.where(
            request >= 0,
            gl.minimum(gl.max(gl.where(in_run, complete, 0), axis=0), num_blocks),
            0,
        )
        num_tiles = gl.cdiv(frontier, BLOCK_N)
        # Key gathers run one tile ahead of the MFMA.
        keys = _load_key_tile(
            page_table,
            key_cache,
            request,
            split,
            frontier,
            page_size,
            stride_k_n,
            stride_pt_b,
            k_layout,
            HEAD_DIM,
            HEAD_DIM_PAD,
            BLOCK_N,
        )
        for tile in range(split, num_tiles, col_splits):
            next_keys = _load_key_tile(
                page_table,
                key_cache,
                request,
                tile + col_splits,
                frontier,
                page_size,
                stride_k_n,
                stride_pt_b,
                k_layout,
                HEAD_DIM,
                HEAD_DIM_PAD,
                BLOCK_N,
            )
            block_ids = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=col_layout)
            acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mma)
            acc = cdna4.mfma(q_tile, keys, acc)
            acc = gl.maximum(acc, 0.0)
            scores = gl.sum(gl.reshape(acc, [BLOCK_Q, HEADS_PAD, BLOCK_N]), axis=1)
            scores = gl.convert_layout(scores, out_layout)
            # Selection reads each row's complete prefix; leave its tail unwritten.
            gl.store(
                out_rows + block_ids[None, :],
                scores,
                mask=in_run[:, None]
                & (block_ids < num_blocks)[None, :]
                & (block_ids[None, :] < complete[:, None]),
            )
            keys = next_keys
        start = run_end


@gluon.jit
def _ordered_key(values):
    # Monotonic map from FP32 to uint32: larger floats give larger keys.
    bits = values.to(gl.uint32, bitcast=True)
    return bits ^ gl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(gl.uint32)


@gluon.jit
def _load_chunk(row_ptr, start, cols, n_cols, resident, IN_REGS: gl.constexpr):
    # A row that fits one chunk stays in registers across the passes.
    if IN_REGS:
        values = resident
    else:
        values = gl.load(
            row_ptr + start + cols, mask=start + cols < n_cols, other=float("-inf")
        )
    return values


@gluon.jit
def _radix_pass(
    row_ptr,
    n_cols,
    cols,
    bins,
    resident,
    hist_smem,
    prefix,
    remaining,
    SHIFT: gl.constexpr,
    CHUNK: gl.constexpr,
    IN_REGS: gl.constexpr,
):
    # Histogram the 8-bit digit at SHIFT of the finite keys that match the
    # digits chosen so far, then pick the digit holding the remaining-th key.
    # LDS atomics build the histogram: one ds_add per score.
    hist_smem.store(gl.zeros_like(bins))
    for start in range(0, n_cols, CHUNK):
        values = _load_chunk(row_ptr, start, cols, n_cols, resident, IN_REGS)
        keys = _ordered_key(values)
        match = values != float("-inf")
        if SHIFT < 24:
            match = match & ((keys >> (SHIFT + 8)) == (prefix >> (SHIFT + 8)))
        digits = ((keys >> SHIFT) & 0xFF).to(gl.int32)
        hist_smem.atomic_scatter_add(gl.full_like(digits, 1), digits, 0, mask=match)
    hist = hist_smem.load(bins.type.layout)
    # at_least[b]: matching keys whose digit is >= b.
    at_least = gl.sum(hist, axis=0) - gl.cumsum(hist, axis=0) + hist
    digit = gl.sum((at_least >= remaining).to(gl.int32), axis=0) - 1
    above = gl.sum(gl.where(bins > digit, hist, 0), axis=0)
    return prefix | (digit.to(gl.uint32) << SHIFT), remaining - above


@gluon.jit(
    launch_metadata=_select_metadata,
    do_not_specialize=["n_cols", "stride_l_n", "stride_o_n", "stride_len"],
)
def gluon_qsa_block_topk_select_gfx950(
    logits,
    lengths,
    out,
    n_cols,
    stride_l_n,
    stride_o_n,
    stride_len,
    HAS_LENGTHS: gl.constexpr,
    TOPK: gl.constexpr,
    CHUNK: gl.constexpr,
    IN_REGS: gl.constexpr,
    num_warps: gl.constexpr,
):
    # Radix select of the TOPK largest finite scores, 8 bits per pass. Ties
    # at the threshold keep the smallest block ids, matching the portable
    # bitonic selection; ids are written in increasing order.
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [num_warps], [0])
    bins_layout: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    out_layout: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    row = gl.program_id(0)
    row_ptr = logits + row.to(gl.int64) * stride_l_n
    if HAS_LENGTHS:
        # Score tails may be uninitialized. Bound every pass by the live prefix.
        n_cols = gl.minimum(gl.maximum(gl.load(lengths + row * stride_len), 0), n_cols)
    cols = gl.arange(0, CHUNK, layout=layout)
    bins = gl.arange(0, 256, layout=bins_layout)
    resident = gl.load(row_ptr + cols, mask=cols < n_cols, other=float("-inf"))
    hist_smem = gl.allocate_shared_memory(
        gl.int32, [256], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )

    # Count the finite scores; with at most TOPK of them, keep them all.
    num_finite = 0
    for start in range(0, n_cols, CHUNK):
        values = _load_chunk(row_ptr, start, cols, n_cols, resident, IN_REGS)
        num_finite += gl.sum((values != float("-inf")).to(gl.int32), axis=0)

    prefix = gl.to_tensor(0).to(gl.uint32)
    remaining = gl.minimum(num_finite, TOPK)
    if num_finite > TOPK:
        for p in gl.static_range(4):
            prefix, remaining = _radix_pass(
                row_ptr,
                n_cols,
                cols,
                bins,
                resident,
                hist_smem,
                prefix,
                remaining,
                24 - 8 * p,
                CHUNK,
                IN_REGS,
            )

    # Gather in column order: keys above the threshold, then the first
    # ``remaining`` keys equal to it. When every finite score is selected the
    # threshold is the smallest key and nothing is cut.
    written = 0
    equal_seen = 0
    for start in range(0, n_cols, CHUNK):
        values = _load_chunk(row_ptr, start, cols, n_cols, resident, IN_REGS)
        finite = values != float("-inf")
        keys = _ordered_key(values)
        equal = finite & (keys == prefix)
        equal_rank = equal_seen + gl.cumsum(equal.to(gl.int32), axis=0) - 1
        take = (finite & (keys > prefix)) | (equal & (equal_rank < remaining))
        equal_seen += gl.sum(equal.to(gl.int32), axis=0)
        position = written + gl.cumsum(take.to(gl.int32), axis=0) - 1
        gl.store(
            out + row.to(gl.int64) * stride_o_n + position,
            start + cols,
            mask=take & (position < TOPK),
        )
        written += gl.sum(take.to(gl.int32), axis=0)
    # Fewer than TOPK finite scores: pad with -1.
    pad = gl.arange(0, TOPK, layout=out_layout)
    gl.store(
        out + row.to(gl.int64) * stride_o_n + pad,
        gl.full([TOPK], -1, gl.int32, out_layout),
        mask=pad >= written,
    )


def launch_gluon_qsa_block_scores_gfx950(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Score every compressed block for each query row.

    Args:
        query: Index queries shaped ``[rows, heads, head_dim]`` with a
            contiguous last dimension; row and head strides are free.
        key_cache: Compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Compressed-cache page ids shaped ``[requests, pages]``.
        request_indices: Owning request per row; a request's rows are
            contiguous.
        complete_blocks: Fully compressed block count per row.
        page_size: Compressed rows per page.

    Returns:
        FP32 scores shaped ``[rows, pages * page_size]``: ``sum_h relu(q_h .
        k)`` below each row's ``complete_blocks``. Values outside that prefix
        are undefined.
    """

    rows, num_heads, head_dim = query.shape
    num_blocks = page_table.shape[1] * page_size
    logits = torch.empty((rows, num_blocks), dtype=torch.float32, device=query.device)
    if rows == 0 or num_blocks == 0:
        return logits
    if query.stride(-1) != 1 or key_cache.stride(-1) != 1:
        raise ValueError("QSA indexer query and key head dims must be contiguous")
    # Query and page-table buffer loads take 32-bit byte offsets; the key
    # cache is gathered with 64-bit pointers.
    for tensor in (query, page_table):
        span = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if (span + 1) * tensor.element_size() > 1 << 31:
            raise ValueError("Gluon QSA indexer query/page table exceed 2 GiB")
    heads_pad = triton.next_power_of_2(num_heads)
    # Large (prefill) batches share each key tile across 16 query rows; small
    # (decode) batches keep one MFMA M tile so a tile rarely spans requests.
    block_m = _SCORE_BLOCK_M if rows >= _SCORE_LARGE_ROWS else 16
    block_q = max(1, block_m // heads_pad)
    row_tiles = triton.cdiv(rows, block_q)
    # Spread the block tiles over programs until the GPU is full; each
    # program prefetches its next tile while scoring the current one.
    col_splits = min(
        triton.cdiv(num_blocks, _SCORE_BLOCK_N),
        triton.cdiv(_SCORE_TARGET_PROGRAMS, row_tiles),
    )
    gluon_qsa_block_scores_gfx950[(row_tiles, col_splits)](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        logits,
        rows,
        num_blocks,
        page_size,
        col_splits,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        page_table.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEADS_PAD=heads_pad,
        HEAD_DIM=head_dim,
        # The MFMA reduces 32 elements per step; padded dims load zeros.
        HEAD_DIM_PAD=max(32, triton.next_power_of_2(head_dim)),
        BLOCK_Q=block_q,
        BLOCK_N=_SCORE_BLOCK_N,
        num_warps=4,
    )
    return logits


def launch_gluon_qsa_block_topk_select_gfx950(
    logits: torch.Tensor, topk: int, *, lengths: torch.Tensor | None
) -> torch.Tensor:
    """Select each row's ``topk`` largest finite scores.

    Args:
        logits: FP32 scores shaped ``[rows, cols]`` with a contiguous last
            dimension; ``-inf`` entries are never selected.
        topk: Entries selected per row.
        lengths: Per-row valid column counts, clamped to the score width.
            Pass None to scan full rows. Columns past the counts are not read.

    Returns:
        Int32 column ids shaped ``[rows, topk]`` in increasing order, padded
        with ``-1`` when a row has fewer than ``topk`` finite scores. Ties at
        the threshold keep the smaller column id, like the portable selection.
    """

    rows, cols = logits.shape
    selected = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return selected
    if logits.stride(-1) != 1:
        raise ValueError("QSA top-k logits must be contiguous per row")
    # Rows up to _SELECT_CHUNK scores stay in registers; the chunk is a
    # power-of-two bucket of the score width.
    chunk = min(_SELECT_CHUNK, max(256, triton.next_power_of_2(cols)))
    gluon_qsa_block_topk_select_gfx950[(rows,)](
        logits,
        lengths,
        selected,
        cols,
        logits.stride(0),
        selected.stride(0),
        lengths.stride(0) if lengths is not None else 0,
        HAS_LENGTHS=lengths is not None,
        TOPK=topk,
        CHUNK=chunk,
        IN_REGS=cols <= chunk,
        num_warps=4,
    )
    return selected
