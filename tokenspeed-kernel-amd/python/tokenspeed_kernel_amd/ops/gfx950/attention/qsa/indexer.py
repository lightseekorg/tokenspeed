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
request. ``gluon_qsa_block_topk_select_gfx950`` selects each row's top
blocks from the FP32 scores. ``launch_gluon_qsa_block_topk_stream_gfx950``
runs both over row chunks whose scores fit a scratch budget. See the "gfx950
QSA indexer" section of ``ops/README.md``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options, wave_inclusive_sum_i32
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

__all__ = [
    "launch_gluon_qsa_block_scores_gfx950",
    "launch_gluon_qsa_block_topk_select_gfx950",
    "launch_gluon_qsa_block_topk_stream_gfx950",
]

# MFMA M rows (query rows x padded index heads) per large-batch score tile.
_SCORE_BLOCK_M = 64
# Batches from this many rows (prefill) use the large score tile.
_SCORE_LARGE_ROWS = 512
# Score programs targeted per launch (about four per CU).
_SCORE_TARGET_PROGRAMS = 1024
# GFX950 accelerator complex dies (each with its own L2); workgroups go to
# them round-robin by program id.
_NUM_XCDS = 8
# Compressed blocks per score tile.
_SCORE_BLOCK_N = 128
# Warps per selection program.
_SELECT_WARPS = 8
# Minimum LDS list entries for the selection candidates.
_SELECT_LIST = 4096
# The lowest finite FP32 value: ``x >= _FP32_LOWEST`` excludes -inf and NaN.
_FP32_LOWEST = gl.constexpr(-3.4028234663852886e38)


def _score_metadata(grid, kernel, args):
    rows = args["query"].shape[0]
    heads = args["NUM_HEADS"]
    head_dim = args["HEAD_DIM"]
    num_blocks = args["num_blocks"]
    return {
        "name": kernel.name,
        "flops16": 2 * rows * heads * head_dim * num_blocks,
        # Every row tile gathers the keys once.
        "bytes": triton.cdiv(rows, args["BLOCK_Q"]) * num_blocks * head_dim * 2
        + rows * (heads * head_dim * 2 + num_blocks * 4),
    }


def _select_metadata(grid, kernel, args):
    rows = grid[0]
    # The group and compaction passes each read the row's scores.
    return {
        "name": kernel.name,
        "bytes": rows * (2 * args["n_cols"] * 4 + args["TOPK"] * 4),
    }


@gluon.jit
def _key_tile_slots(
    page_table,
    request,
    tile,
    frontier,
    page_size,
    stride_pt_b,
    k_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    # Cache slots of blocks [tile * BLOCK_N, ...); -1 at or past the run's
    # frontier.
    k_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_layout))
    live = k_n < frontier
    # Int64 lengths can widen tile offsets; the host bounds the page table to
    # 2 GiB, so valid buffer offsets fit in int32. Key addresses stay int64.
    pages = cdna4.buffer_load(
        page_table,
        (request * stride_pt_b + k_n // page_size).to(gl.int32),
        mask=live,
        other=0,
    )
    return gl.where(live, pages.to(gl.int64) * page_size + k_n % page_size, -1)


@gluon.jit
def _load_key_tile(
    key_cache,
    slots,
    stride_k_n,
    k_layout: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HEAD_DIM_PAD: gl.constexpr,
):
    # Gather the keys of a tile's slots straight into the MFMA B-operand
    # layout, with 64-bit addressing: the compressed cache may exceed the
    # 2 GiB reach of buffer loads.
    k_d = gl.arange(0, HEAD_DIM_PAD, layout=gl.SliceLayout(1, k_layout))
    return gl.load(
        key_cache + slots[None, :] * stride_k_n + k_d[:, None],
        mask=(slots >= 0)[None, :] & (k_d < HEAD_DIM)[:, None],
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
    NUM_XCDS: gl.constexpr,
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
    # Programs go round-robin to the XCDs; the programs of an XCD take the
    # splits congruent to it modulo NUM_XCDS, for every row tile, so the key
    # tiles they share stay in that XCD's L2.
    pid = gl.program_id(0)
    xcd_index = pid // NUM_XCDS
    row_tiles = gl.cdiv(rows, BLOCK_Q)
    row0 = (xcd_index % row_tiles) * BLOCK_Q
    split = pid % NUM_XCDS + NUM_XCDS * (xcd_index // row_tiles)
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
        # Key gathers run one tile ahead of the MFMA and their page ids one
        # more, so neither dependent load waits in the loop.
        keys = _load_key_tile(
            key_cache,
            _key_tile_slots(
                page_table,
                request,
                split,
                frontier,
                page_size,
                stride_pt_b,
                k_layout,
                BLOCK_N,
            ),
            stride_k_n,
            k_layout,
            HEAD_DIM,
            HEAD_DIM_PAD,
        )
        next_slots = _key_tile_slots(
            page_table,
            request,
            split + col_splits,
            frontier,
            page_size,
            stride_pt_b,
            k_layout,
            BLOCK_N,
        )
        for tile in range(split, num_tiles, col_splits):
            next_keys = _load_key_tile(
                key_cache, next_slots, stride_k_n, k_layout, HEAD_DIM, HEAD_DIM_PAD
            )
            next_slots = _key_tile_slots(
                page_table,
                request,
                tile + 2 * col_splits,
                frontier,
                page_size,
                stride_pt_b,
                k_layout,
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
def _select_key(values):
    # Order-reversing map from FP32 to uint32: larger floats give smaller
    # keys, so ascending histogram prefix sums count the scores at least as
    # large as a bin.
    bits = values.to(gl.uint32, bitcast=True)
    return bits ^ gl.where((bits >> 31) != 0, 0, 0x7FFFFFFF).to(gl.uint32)


@gluon.jit
def _key_score(keys):
    # Inverse of _select_key.
    return (keys ^ gl.where(keys < 0x80000000, 0x7FFFFFFF, 0).to(gl.uint32)).to(
        gl.float32, bitcast=True
    )


@gluon.jit
def _chunk_scores(row_desc, start, cols, n_cols):
    # Scores [start, start + CHUNK) of a row; columns past the live prefix
    # read -inf. Unmasked loads vectorize, and the descriptor bounds them to
    # the live prefix.
    values = cdna4.buffer.load(row_desc, cols, soffset=start)
    return gl.where(start + cols < n_cols, values, float("-inf"))


@gluon.jit
def _pick_digit(hist, topk, BINS: gl.constexpr, num_warps: gl.constexpr):
    # Find the digit holding the k-th smallest counted key, with k the
    # smaller of topk and the count. Every wave reads the whole histogram,
    # lane i holding bins [i * V, (i + 1) * V), so the search stays within
    # the wave: DPP prefix sums and reductions, no LDS exchange. Returns the
    # digit, the keys before it, its own count, and the count.
    V: gl.constexpr = BINS // 64
    layout: gl.constexpr = gl.BlockedLayout([1, V], [64, 1], [num_warps, 1], [1, 0])
    counts = hist.slice(0, BINS).load(gl.BlockedLayout([V], [64], [num_warps], [0]))
    counts = gl.convert_layout(gl.reshape(counts, [64, V]), layout)
    lane_total = gl.sum(counts, axis=1)
    lane_inclusive = wave_inclusive_sum_i32(lane_total)
    total = gl.max(lane_inclusive, axis=0)
    k = gl.minimum(total, topk)
    inclusive = (lane_inclusive - lane_total)[:, None] + gl.cumsum(counts, 1)
    before = inclusive - counts
    chosen = (before < k) & (inclusive >= k)
    bins = (
        gl.arange(0, 64, layout=gl.SliceLayout(1, layout))[:, None] * V
        + gl.arange(0, V, layout=gl.SliceLayout(0, layout))[None, :]
    )
    # One bin is chosen (none without keys): a sum publishes each field.
    digit = gl.sum(gl.sum(gl.where(chosen, bins, 0), axis=1), axis=0)
    above = gl.sum(gl.sum(gl.where(chosen, before, 0), axis=1), axis=0)
    bucket = gl.sum(gl.sum(gl.where(chosen, counts, 0), axis=1), axis=0)
    return digit, above, bucket, total


@gluon.jit
def _exclusive_scan(counts, num_warps: gl.constexpr):
    # Exclusive prefix sums, in lane-major order, of [lanes, n] counts held
    # one row per lane, and their total: in-lane sums, a DPP scan of the
    # lane totals in each wave, then the wave totals across waves.
    lane_total = gl.sum(counts, axis=1)
    lane_inclusive = wave_inclusive_sum_i32(lane_total)
    waves_layout: gl.constexpr = gl.BlockedLayout(
        [1, num_warps], [64, 1], [num_warps, 1], [1, 0]
    )
    wave_total = gl.max(gl.reshape(lane_inclusive, [num_warps, 64]), axis=1)
    totals = gl.convert_layout(wave_total, gl.SliceLayout(0, waves_layout))
    waves = gl.arange(0, num_warps, layout=gl.SliceLayout(1, waves_layout))
    lower = gl.arange(0, num_warps, layout=gl.SliceLayout(0, waves_layout))
    wave_below = gl.sum(
        gl.where(lower[None, :] < waves[:, None], totals[None, :], 0), axis=1
    )
    wave_below = gl.convert_layout(wave_below, wave_total.type.layout)
    lane_below = gl.reshape(
        wave_below[:, None] + gl.reshape(lane_inclusive - lane_total, [num_warps, 64]),
        [num_warps * 64],
    )
    lane_below = gl.convert_layout(lane_below, lane_total.type.layout)
    return gl.cumsum(counts, 1) - counts + lane_below[:, None], gl.sum(totals, axis=0)


@gluon.jit
def _write_selected(
    out_row,
    ids,
    keys,
    live,
    threshold,
    remaining,
    written_above,
    equal_seen,
    num_warps: gl.constexpr,
):
    # Write the ids of the live keys below ``threshold`` (the larger scores)
    # and of the first ``remaining`` keys equal to it, in lane-major order.
    # One scan counts both: the keys below the threshold (fewer than TOPK <
    # 65536) in the low half and the ties (fewer than 65536 per call) in the
    # high half.
    above = live & (keys < threshold)
    equal = live & (keys == threshold)
    exclusive, total = _exclusive_scan(
        above.to(gl.int32) | (equal.to(gl.int32) << 16), num_warps
    )
    equal_rank = equal_seen + (exclusive >> 16)
    take = above | (equal & (equal_rank < remaining))
    position = written_above + (exclusive & 0xFFFF) + gl.minimum(equal_rank, remaining)
    gl.store(out_row + position, ids, mask=take)
    return written_above + (total & 0xFFFF), equal_seen + (total >> 16)


@gluon.jit
def _spare_slots(values, base):
    # Per-lane LDS slots base + lane % 64 for one [lanes, n] tensor row per
    # lane: masked-off lanes of an unmasked scatter write there, apart from
    # each other. Unmasked LDS ops avoid a branch per element.
    lane = gl.arange(0, values.shape[0], layout=gl.SliceLayout(1, values.type.layout))
    return base + lane[:, None] % 64


@gluon.jit
def _count_digits(hist, keys, mask, shift):
    # LDS-atomic histogram of the 8-bit digit at ``shift`` of the masked keys
    # into the first 256 entries of ``hist``.
    digits = gl.where(
        mask, ((keys >> shift) & 0xFF).to(gl.int32), _spare_slots(keys, 256)
    )
    digits = gl.reshape(digits, [keys.numel])
    hist.atomic_scatter_add(gl.full_like(digits, 1), digits, 0)


@gluon.jit
def _radix_digits(hist_smem, keys, live, topk, num_warps: gl.constexpr):
    # Resolve the threshold key of the topk smallest live keys, 8 bits at a
    # time, stopping once the bucket holding the topk-th key is taken whole.
    # Returns the threshold's resolved high bits, their shift, and how many
    # keys equal to them are taken.
    hist = hist_smem.slice(0, 512)
    _count_digits(hist, keys, live, 24)
    digit, before, bucket, total = _pick_digit(hist, topk, 256, num_warps)
    prefix = digit.to(gl.uint32)
    remaining = gl.minimum(total, topk) - before
    shift = gl.to_tensor(24).to(gl.uint32)
    for p in gl.static_range(1, 4):
        if bucket > remaining:
            _count_digits(
                hist_smem.slice(512 * p, 512),
                keys,
                live & ((keys >> shift) == prefix),
                24 - 8 * p,
            )
            digit, before, bucket, _ = _pick_digit(
                hist_smem.slice(512 * p, 512), remaining, 256, num_warps
            )
            prefix = (prefix << 8) | digit.to(gl.uint32)
            remaining -= before
            shift = gl.to_tensor(24 - 8 * p).to(gl.uint32)
    return prefix, shift, remaining


@gluon.jit
def _select_full_rows(
    out_row,
    row_desc,
    n_cols,
    hist_smem,
    TOPK: gl.constexpr,
    VEC: gl.constexpr,
    num_warps: gl.constexpr,
):
    # Recovery when the candidates overflow the LDS list, as with many equal
    # or near-equal scores: four 8-bit radix passes over the whole row, then
    # a column-order write. Returns the selected count.
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [64, 1], [num_warps, 1], [1, 0])
    CHUNK: gl.constexpr = 64 * num_warps * VEC
    lane = gl.arange(0, 64 * num_warps, layout=gl.SliceLayout(1, layout))
    cols = (
        lane[:, None] * VEC
        + gl.arange(0, VEC, layout=gl.SliceLayout(0, layout))[None, :]
    )
    prefix = gl.to_tensor(0).to(gl.uint32)
    remaining = gl.to_tensor(TOPK)
    selected = gl.to_tensor(0)
    for p in gl.static_range(4):
        hist = hist_smem.slice(512 * (4 + p), 512)
        for start in range(0, n_cols, CHUNK):
            values = _chunk_scores(row_desc, start, cols, n_cols)
            keys = _select_key(values)
            live = values >= _FP32_LOWEST
            if p > 0:
                live = live & ((keys >> (32 - 8 * p)) == prefix)
            _count_digits(hist, keys, live, 24 - 8 * p)
        digit, before, _, total = _pick_digit(hist, remaining, 256, num_warps)
        if p == 0:
            selected = gl.minimum(total, TOPK)
            remaining = selected
        prefix = (prefix << 8) | digit.to(gl.uint32)
        remaining -= before
    written_above = gl.to_tensor(0)
    equal_seen = gl.to_tensor(0)
    for start in range(0, n_cols, CHUNK):
        values = _chunk_scores(row_desc, start, cols, n_cols)
        written_above, equal_seen = _write_selected(
            out_row,
            start + cols,
            _select_key(values),
            values >= _FP32_LOWEST,
            prefix,
            remaining,
            written_above,
            equal_seen,
            num_warps,
        )
    return selected


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
    GROUP: gl.constexpr,
    LIST_VEC: gl.constexpr,
    num_warps: gl.constexpr,
):
    # Top-TOPK selection of a row's finite scores, one program per row:
    #  1. Each lane's GROUP consecutive scores form a group. The TOPK-th
    #     largest group maximum is a lower bound of the TOPK-th largest
    #     score: an 11-bit histogram of the group maxima bounds it by the
    #     low edge of its bucket.
    #  2. The scores at or above the bound are compacted, in column order,
    #     into an LDS list of at most CAP entries.
    #  3. 8-bit radix passes over the list resolve the threshold, stopping
    #     once the bucket holding the TOPK-th score is taken whole.
    #  4. The ids above the threshold and the first ones equal to it are
    #     written in increasing column order: ties keep the smallest block
    #     ids, like the portable bitonic selection.
    # Candidates overflowing the list take _select_full_rows instead.
    THREADS: gl.constexpr = 64 * num_warps
    CAP: gl.constexpr = THREADS * LIST_VEC
    BINS: gl.constexpr = 2048
    # Lanes own consecutive columns (16-byte vector loads), so lane order is
    # column order. A group is GROUP columns of a lane; compaction takes
    # COMPACT_VEC columns a lane.
    group_layout: gl.constexpr = gl.BlockedLayout(
        [1, GROUP], [64, 1], [num_warps, 1], [1, 0]
    )
    COMPACT_VEC: gl.constexpr = 32
    compact_layout: gl.constexpr = gl.BlockedLayout(
        [1, COMPACT_VEC], [64, 1], [num_warps, 1], [1, 0]
    )
    list_layout: gl.constexpr = gl.BlockedLayout(
        [1, LIST_VEC], [64, 1], [num_warps, 1], [1, 0]
    )
    flat_layout: gl.constexpr = gl.BlockedLayout([LIST_VEC], [64], [num_warps], [0])
    out_layout: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    row = gl.program_id(0)
    # The launcher checks 16-byte aligned rows.
    row_ptr = gl.multiple_of(logits + row.to(gl.int64) * stride_l_n, 16)
    out_row = out + row.to(gl.int64) * stride_o_n
    if HAS_LENGTHS:
        # Score tails may be uninitialized. Bound every pass by the live prefix.
        n_cols = gl.minimum(
            gl.maximum(gl.load(lengths + row * stride_len), 0), n_cols
        ).to(gl.int32)
    # Columns are 32-bit; the row fits the descriptor's 32-bit byte range.
    row_desc = cdna4.buffer.make_buffer_descriptor(row_ptr, n_cols * 4)

    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    # The group histogram's BINS bins, then 512-entry regions for eight 8-bit
    # histograms; the lists hold LIST_CAP entries. Each also has 64 spare
    # slots past its end, see _spare_slots.
    LIST_CAP: gl.constexpr = CAP - 64
    hist_smem = gl.allocate_shared_memory(gl.int32, [4 * BINS], smem_layout)
    list_scores = gl.allocate_shared_memory(gl.int32, [CAP], smem_layout)
    list_cols = gl.allocate_shared_memory(gl.int32, [CAP], smem_layout)
    hist_smem.store(
        gl.zeros(
            [4 * BINS],
            gl.int32,
            gl.BlockedLayout([4 * BINS // THREADS], [64], [num_warps], [0]),
        )
    )
    group_hist = hist_smem.slice(0, 2 * BINS)

    # 1. Histogram the live group maxima; the next chunk loads ahead.
    GROUP_CHUNK: gl.constexpr = THREADS * GROUP
    lane = gl.arange(0, THREADS, layout=gl.SliceLayout(1, group_layout))
    cols = (
        lane[:, None] * GROUP
        + gl.arange(0, GROUP, layout=gl.SliceLayout(0, group_layout))[None, :]
    )
    ahead = _chunk_scores(row_desc, 0, cols, n_cols)
    for start in range(0, n_cols, GROUP_CHUNK):
        values = ahead
        ahead = _chunk_scores(row_desc, start + GROUP_CHUNK, cols, n_cols)
        group_max = gl.max(values, axis=1)
        digits = gl.where(
            group_max > float("-inf"),
            (_select_key(group_max) >> 21).to(gl.int32),
            BINS + lane % 64,
        )
        group_hist.atomic_scatter_add(gl.full_like(digits, 1), digits, 0)
    digit, _, _, live_groups = _pick_digit(group_hist, TOPK, BINS, num_warps)
    # With more than TOPK live groups, the TOPK-th group maximum's bucket
    # bounds the selection; otherwise every finite score is a candidate.
    # Group maxima are never NaN, so the bucket's low edge is not either.
    bound = gl.where(
        live_groups > TOPK,
        _key_score(((digit.to(gl.uint32) + 1) << 21) - 1),
        _FP32_LOWEST,
    )

    # 2. Compact the candidates into the list.
    COMPACT_CHUNK: gl.constexpr = THREADS * COMPACT_VEC
    lane = gl.arange(0, THREADS, layout=gl.SliceLayout(1, compact_layout))
    cols = (
        lane[:, None] * COMPACT_VEC
        + gl.arange(0, COMPACT_VEC, layout=gl.SliceLayout(0, compact_layout))[None, :]
    )
    listed = gl.to_tensor(0)
    ahead = _chunk_scores(row_desc, 0, cols, n_cols)
    for start in range(0, n_cols, COMPACT_CHUNK):
        values = ahead
        ahead = _chunk_scores(row_desc, start + COMPACT_CHUNK, cols, n_cols)
        keep = values >= bound
        slot, kept = _exclusive_scan(keep.to(gl.int32), num_warps)
        # Overflowing candidates only need to stay in bounds: the list is
        # then unused.
        slot = gl.where(
            keep, gl.minimum(slot + listed, LIST_CAP), _spare_slots(values, LIST_CAP)
        )
        slot = gl.reshape(slot, [COMPACT_CHUNK])
        list_scores.scatter(
            gl.reshape(values.to(gl.int32, bitcast=True), [COMPACT_CHUNK]), slot, 0
        )
        list_cols.scatter(gl.reshape(start + cols, [COMPACT_CHUNK]), slot, 0)
        listed += kept

    if listed <= LIST_CAP:
        # 3. Resolve the threshold on the list.
        shape: gl.constexpr = [THREADS, LIST_VEC]
        keys = _select_key(
            list_scores.reshape(shape).load(list_layout).to(gl.float32, bitcast=True)
        )
        list_lane = gl.arange(0, THREADS, layout=gl.SliceLayout(1, list_layout))
        index = (
            list_lane[:, None] * LIST_VEC
            + gl.arange(0, LIST_VEC, layout=gl.SliceLayout(0, list_layout))[None, :]
        )
        live = index < listed
        prefix, shift, remaining = _radix_digits(
            hist_smem.slice(2 * BINS, 2 * BINS), keys, live, TOPK, num_warps
        )
        # 4. Write the selection in column order.
        _write_selected(
            out_row,
            list_cols.reshape(shape).load(list_layout),
            keys >> shift,
            live,
            prefix,
            remaining,
            0,
            0,
            num_warps,
        )
        selected = gl.minimum(listed, TOPK)
    else:
        selected = _select_full_rows(
            out_row,
            row_desc,
            n_cols,
            hist_smem.slice(2 * BINS, 2 * BINS),
            TOPK,
            COMPACT_VEC,
            num_warps,
        )
    # Fewer than TOPK finite scores: pad with -1.
    pad = gl.arange(0, TOPK, layout=out_layout)
    gl.store(
        out_row + pad,
        gl.full([TOPK], -1, gl.int32, out_layout),
        mask=pad >= selected,
    )


def _launch_block_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    out: torch.Tensor | None,
    block_m: int,
) -> torch.Tensor:
    # launch_gluon_qsa_block_scores_gfx950 with an explicit MFMA M tile:
    # block_m // heads query rows share each gathered key tile.

    rows, num_heads, head_dim = query.shape
    num_blocks = page_table.shape[1] * page_size
    if out is None:
        out = torch.empty((rows, num_blocks), dtype=torch.float32, device=query.device)
    elif out.shape != (rows, num_blocks) or out.stride(-1) != 1:
        raise ValueError("QSA score output must be [rows, blocks] with contiguous rows")
    logits = out
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
    block_q = max(1, block_m // heads_pad)
    row_tiles = triton.cdiv(rows, block_q)
    # Spread the block tiles over programs until the GPU is full; each
    # program prefetches its next tile while scoring the current one.
    # Splits come in multiples of the XCD count (see the kernel); the extra
    # splits past the last tile have no work.
    col_splits = min(
        triton.cdiv(num_blocks, _SCORE_BLOCK_N),
        triton.cdiv(_SCORE_TARGET_PROGRAMS, row_tiles),
    )
    col_splits = triton.cdiv(col_splits, _NUM_XCDS) * _NUM_XCDS
    gluon_qsa_block_scores_gfx950[(row_tiles * col_splits,)](
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
        NUM_XCDS=_NUM_XCDS,
        num_warps=4,
    )
    return logits


def launch_gluon_qsa_block_scores_gfx950(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    out: torch.Tensor | None = None,
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
        out: Optional FP32 destination shaped ``[rows, pages * page_size]``
            with a contiguous last dimension.

    Returns:
        FP32 scores shaped ``[rows, pages * page_size]``: ``sum_h relu(q_h .
        k)`` below each row's ``complete_blocks``. Values outside that prefix
        are undefined.
    """

    # Large (prefill) batches share each key tile across 16 query rows; small
    # (decode) batches keep one MFMA M tile so a tile rarely spans requests.
    return _launch_block_scores(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        out=out,
        block_m=_SCORE_BLOCK_M if query.shape[0] >= _SCORE_LARGE_ROWS else 16,
    )


def launch_gluon_qsa_block_topk_select_gfx950(
    logits: torch.Tensor,
    topk: int,
    *,
    lengths: torch.Tensor | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select each row's ``topk`` largest finite scores.

    Args:
        logits: FP32 scores shaped ``[rows, cols]`` with 16-byte aligned,
            contiguous rows; ``-inf`` and NaN entries are never selected.
        topk: Entries selected per row, a power of two below 65536.
        lengths: Per-row valid column counts, clamped to the score width.
            Pass None to scan full rows. Columns past the counts are not read.
        out: Optional int32 destination shaped ``[rows, topk]`` with a
            contiguous last dimension.

    Returns:
        Int32 column ids shaped ``[rows, topk]`` in increasing order, padded
        with ``-1`` when a row has fewer than ``topk`` finite scores. Ties at
        the threshold keep the smaller column id, like the portable selection.
    """

    rows, cols = logits.shape
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    elif out.shape != (rows, topk) or out.stride(-1) != 1:
        raise ValueError("QSA top-k output must be [rows, topk] with contiguous rows")
    selected = out
    if rows == 0:
        return selected
    if logits.stride(-1) != 1:
        raise ValueError("QSA top-k logits must be contiguous per row")
    # Rows load as 16-byte vectors.
    if logits.stride(0) % 4 or logits.data_ptr() % 16:
        raise ValueError("QSA top-k logits rows must be 16-byte aligned")
    if cols >= 1 << 29:
        raise ValueError("QSA top-k rows exceed the 2 GiB buffer-load reach")
    if topk >= 1 << 16 or topk & (topk - 1):
        raise ValueError("QSA top-k selection needs a power-of-two topk < 65536")
    # Rows of up to GROUP * topk scores list every finite score, so the
    # list must hold them: its 4096 entries keep 64 spare slots.
    cap = max(_SELECT_LIST, 2 * topk)
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
        GROUP=cap // (2 * topk),
        LIST_VEC=cap // (64 * _SELECT_WARPS),
        num_warps=_SELECT_WARPS,
        **extern_compile_options(),
    )
    return selected


def launch_gluon_qsa_block_topk_stream_gfx950(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    max_partial_bytes: int,
) -> torch.Tensor:
    """Select each query row's ``block_topk`` best blocks within a score budget.

    Scores rows in chunks whose FP32 scores fit ``max_partial_bytes`` and
    selects each chunk's blocks before scoring the next, so the selection
    equals that of :func:`launch_gluon_qsa_block_scores_gfx950` followed by
    :func:`launch_gluon_qsa_block_topk_select_gfx950` over all rows.

    Args:
        query, key_cache, page_table, request_indices, complete_blocks,
            page_size: As :func:`launch_gluon_qsa_block_scores_gfx950`.
        block_topk: Blocks selected per row, a power of two below 65536.
        max_partial_bytes: Scratch budget for one chunk's scores; it must
            hold at least one row of ``pages * page_size`` FP32 scores.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]`` in increasing order,
        padded with ``-1`` for rows with fewer complete blocks.
    """

    rows = query.shape[0]
    num_blocks = page_table.shape[1] * page_size
    selected = torch.empty((rows, block_topk), dtype=torch.int32, device=query.device)
    if rows == 0:
        return selected
    chunk_rows = min(rows, max_partial_bytes // (4 * max(num_blocks, 1)))
    if chunk_rows == 0:
        raise ValueError(
            f"QSA streaming budget of {max_partial_bytes} bytes cannot hold one "
            f"row of {num_blocks} FP32 block scores"
        )
    scores = torch.empty(
        (chunk_rows, num_blocks), dtype=torch.float32, device=query.device
    )
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        # Streamed chunks are prefill-sized: wide M tiles share each key tile.
        chunk = _launch_block_scores(
            query[start:end],
            key_cache,
            page_table,
            request_indices[start:end],
            complete_blocks[start:end],
            page_size=page_size,
            out=scores[: end - start],
            block_m=_SCORE_BLOCK_M,
        )
        launch_gluon_qsa_block_topk_select_gfx950(
            chunk,
            block_topk,
            lengths=complete_blocks[start:end],
            out=selected[start:end],
        )
    return selected
