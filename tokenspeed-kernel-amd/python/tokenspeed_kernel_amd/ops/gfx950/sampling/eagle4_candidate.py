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

"""Gluon EAGLE-4 candidate head selections for GFX950: the vocab shard top-k
before the candidate all-gather and the tail after it."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "launch_gluon_eagle4_candidate_tail_gfx950",
    "launch_gluon_eagle4_shard_topk_gfx950",
]

# CDNA4 waves are 64 lanes wide; torch's softmax warp is as wide.
_WAVE = 64


@gluon.constexpr_function
def _vector_layout(BLOCK, NUM_WARPS):
    # A [BLOCK] vector over the workgroup, contiguous elements per lane.
    per_lane = max(1, BLOCK // (_WAVE * NUM_WARPS))
    return gl.BlockedLayout([per_lane], [_WAVE], [NUM_WARPS], [0])


@gluon.constexpr_function
def _split_hist_layout(NUM_WARPS):
    # [splits, 256]: four bins per lane, one split per warp.
    return gl.BlockedLayout([1, 4], [1, _WAVE], [NUM_WARPS, 1], [1, 0])


@gluon.jit
def _bf16_key(x):
    # torch.topk's order-preserving key of a bf16 value; NaN maps to the maximum.
    bits = x.to(gl.uint16, bitcast=True).to(gl.int32)
    key = gl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return gl.where(x != x, 0xFFFF, key)


@gluon.jit
def _kth_bin(hist, target, NUM_WARPS: gl.constexpr):
    """The bin of the ``target``-th largest key of a 256-bin histogram, and the
    count of keys in higher bins."""
    layout: gl.constexpr = _vector_layout(256, NUM_WARPS)
    hist = gl.convert_layout(hist, layout)  # scans take blocked layouts only
    bins = gl.arange(0, 256, layout=layout)
    at_least = gl.sum(hist, 0) - gl.cumsum(hist, 0) + hist  # keys in bins >= b
    digit = gl.max(gl.where(at_least >= target, bins, 0), 0)
    above = gl.sum(gl.where(bins == digit, at_least - hist, 0), 0)
    return digit, above


@gluon.jit
def _shard_keys(scores, stride_s, row, cols, n_cols):
    valid = cols < n_cols
    x = gl.load(scores + row.to(gl.int64) * stride_s + cols, mask=valid, other=0.0)
    return valid, gl.where(valid, _bf16_key(x), -1)


@gluon.jit
def _split_hists(hist, SPLITS: gl.constexpr, SPLITS_PAD: gl.constexpr, layout):
    # [SPLITS_PAD, 256]: one 256-bin histogram per split of the row, zero padded.
    splits = gl.arange(0, SPLITS_PAD, layout=gl.SliceLayout(1, layout))
    bins = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    return gl.load(
        hist + splits[:, None] * 256 + bins[None, :],
        mask=(splits < SPLITS)[:, None],
        other=0,
    )


def _shard_topk_hist_metadata(grid, kernel, args):
    """Report one read of the scores plus the split histogram writes."""
    rows, splits = grid[0], grid[1]
    return {
        "name": kernel.name,
        "bytes": rows * args["n_cols"] * 2 + rows * splits * 256 * 4,
    }


@gluon.jit(
    launch_metadata=_shard_topk_hist_metadata,
    do_not_specialize=["stride_s", "n_cols"],
)
def gluon_eagle4_shard_topk_hist_gfx950(
    scores,  # [M, N] bf16
    hist,  # [M, 2, SPLITS, 256] int32: high byte, then low byte in its bin
    stride_s,
    n_cols,
    K: gl.constexpr,
    PASS: gl.constexpr,
    SPLITS: gl.constexpr,
    SPLITS_PAD: gl.constexpr,
    CHUNK: gl.constexpr,
    TILE: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    tile_layout: gl.constexpr = _vector_layout(TILE, NUM_WARPS)
    split_layout: gl.constexpr = _split_hist_layout(NUM_WARPS)
    bin_layout: gl.constexpr = _vector_layout(256, NUM_WARPS)
    row = gl.program_id(0)
    chunk = gl.program_id(1)
    row_hist = hist + row * (2 * SPLITS * 256)
    high = 0
    if PASS == 1:
        high, _ = _kth_bin(
            gl.sum(_split_hists(row_hist, SPLITS, SPLITS_PAD, split_layout), 0),
            K,
            NUM_WARPS,
        )
    counts = gl.zeros([256], gl.int32, layout=bin_layout)
    offs = gl.arange(0, TILE, layout=tile_layout)
    for start in range(0, CHUNK, TILE):
        cols = chunk * CHUNK + start + offs
        valid, key = _shard_keys(scores, stride_s, row, cols, n_cols)
        if PASS == 0:
            counts += gl.histogram(key >> 8, 256, mask=valid, layout=bin_layout)
        else:
            counts += gl.histogram(
                key & 255, 256, mask=valid & ((key >> 8) == high), layout=bin_layout
            )
    bins = gl.arange(0, 256, layout=bin_layout)
    gl.store(row_hist + (PASS * SPLITS + chunk) * 256 + bins, counts)


def _shard_topk_select_metadata(grid, kernel, args):
    """Report the split histogram reads plus the placement writes."""
    rows, splits = grid[0], args["SPLITS"]
    return {
        "name": kernel.name,
        "bytes": rows * (2 * splits * 256 * 4 + (2 * splits + 2) * 4),
    }


@gluon.jit(launch_metadata=_shard_topk_select_metadata)
def gluon_eagle4_shard_topk_select_gfx950(
    hist,
    sel,  # [M, 2 * SPLITS + 2] int32: per-split gt and tie bases, kth, n_gt
    K: gl.constexpr,
    SPLITS: gl.constexpr,
    SPLITS_PAD: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    layout: gl.constexpr = _split_hist_layout(NUM_WARPS)
    row = gl.program_id(0)
    row_hist = hist + row * (2 * SPLITS * 256)
    bins = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    hist_hi = _split_hists(row_hist, SPLITS, SPLITS_PAD, layout)
    hist_lo = _split_hists(row_hist + SPLITS * 256, SPLITS, SPLITS_PAD, layout)
    high, above_hi = _kth_bin(gl.sum(hist_hi, 0), K, NUM_WARPS)
    low, above_lo = _kth_bin(gl.sum(hist_lo, 0), K - above_hi, NUM_WARPS)
    # Keys above kth and kth ties of each split, to place each split after the
    # earlier ones.
    split_gt = gl.sum(gl.where(bins[None, :] > high, hist_hi, 0), 1) + gl.sum(
        gl.where(bins[None, :] > low, hist_lo, 0), 1
    )
    split_eq = gl.sum(gl.where(bins[None, :] == low, hist_lo, 0), 1)
    split_layout: gl.constexpr = _vector_layout(SPLITS_PAD, NUM_WARPS)
    split_gt = gl.convert_layout(split_gt, split_layout)
    split_eq = gl.convert_layout(split_eq, split_layout)
    splits = gl.arange(0, SPLITS_PAD, layout=split_layout)
    row_sel = sel + row * (2 * SPLITS + 2)
    gl.store(row_sel + splits, gl.cumsum(split_gt, 0) - split_gt, mask=splits < SPLITS)
    gl.store(
        row_sel + SPLITS + splits,
        gl.cumsum(split_eq, 0) - split_eq,
        mask=splits < SPLITS,
    )
    gl.store(row_sel + 2 * SPLITS, (high << 8) | low)
    gl.store(row_sel + 2 * SPLITS + 1, above_hi + above_lo)


def _shard_topk_metadata(grid, kernel, args):
    """Report one read of the scores plus the id writes."""
    rows = grid[0]
    return {
        "name": kernel.name,
        "bytes": rows * args["n_cols"] * 2 + rows * args["K"] * 8,
    }


@gluon.jit(
    launch_metadata=_shard_topk_metadata,
    do_not_specialize=["stride_s", "stride_g", "n_cols", "vocab_start"],
)
def gluon_eagle4_shard_topk_gfx950(
    scores,
    sel,
    global_ids,  # [M, K] int64
    stride_s,
    stride_g,
    n_cols,
    vocab_start,
    K: gl.constexpr,
    SPLITS: gl.constexpr,
    CHUNK: gl.constexpr,
    TILE: gl.constexpr,
):
    layout: gl.constexpr = _vector_layout(TILE, gl.num_warps())
    row = gl.program_id(0)
    chunk = gl.program_id(1)
    row_sel = sel + row * (2 * SPLITS + 2)
    base_gt = gl.load(row_sel + chunk)
    base_eq = gl.load(row_sel + SPLITS + chunk)
    kth = gl.load(row_sel + 2 * SPLITS)
    # Keys above kth; then K - n_gt of its ties.
    n_gt = gl.load(row_sel + 2 * SPLITS + 1)

    out_row = global_ids + row.to(gl.int64) * stride_g
    offs = gl.arange(0, TILE, layout=layout)
    for start in range(0, CHUNK, TILE):
        cols = chunk * CHUNK + start + offs
        _, key = _shard_keys(scores, stride_s, row, cols, n_cols)
        gt = (key > kth).to(gl.int32)
        eq = (key == kth).to(gl.int32)
        tile_gt = gl.sum(gt, 0)
        tile_eq = gl.sum(eq, 0)
        # Most tiles hold none of the K keys: skip their scans.
        if tile_gt + tile_eq > 0:
            rank_gt = base_gt + gl.cumsum(gt, 0) - gt
            rank_eq = base_eq + gl.cumsum(eq, 0) - eq
            take = (gt != 0) | ((eq != 0) & (rank_eq < K - n_gt))
            slot = gl.where(gt != 0, rank_gt, n_gt + rank_eq)
            gl.store(out_row + slot, cols.to(gl.int64) + vocab_start, mask=take)
        base_gt += tile_gt
        base_eq += tile_eq


# Splits per row: the CUDA kernel's 8-CTA cluster, doubled (no cluster sync here).
_SHARD_SPLITS = 16
# Columns a workgroup holds in registers at a time (four per lane, up to eight
# waves); it loops over its split.
_SHARD_MIN_TILE = 4 * _WAVE
_SHARD_MAX_TILE = 2048


def launch_gluon_eagle4_shard_topk_gfx950(
    scores: torch.Tensor, k: int, vocab_start: int, global_ids: torch.Tensor
) -> None:
    """``torch.topk(scores, k, dim=-1, sorted=False).indices + vocab_start`` in
    CUDA's order: the ids of keys above the k-th largest by position, then the
    first k-th largest ties by position.

    Args:
        scores: ``[M, N]`` bf16 scores of a vocab shard, unit column stride.
        k: Ids kept per row, at most ``N``.
        vocab_start: The vocab id of the shard's column 0.
        global_ids: ``[M, k]`` int64 destination, unit column stride.
    """
    rows, n_cols = scores.shape
    if rows == 0:
        return
    if scores.dtype != torch.bfloat16 or scores.stride(1) != 1:
        raise ValueError("shard top-k takes bf16 scores with unit column stride")
    if global_ids.dtype != torch.int64 or global_ids.stride(1) != 1:
        raise ValueError("shard top-k writes int64 ids with unit column stride")
    if not 1 <= k <= n_cols:
        raise ValueError(f"shard top-k keeps 1..{n_cols} ids, got {k}")
    # N is fixed per server, so the split layout compiles once.
    tile = min(
        _SHARD_MAX_TILE,
        max(
            _SHARD_MIN_TILE, triton.next_power_of_2(triton.cdiv(n_cols, _SHARD_SPLITS))
        ),
    )
    chunk = triton.cdiv(triton.cdiv(n_cols, _SHARD_SPLITS), tile) * tile
    splits = triton.cdiv(n_cols, chunk)
    hist = torch.empty((rows, 2, splits, 256), dtype=torch.int32, device=scores.device)
    sel = torch.empty((rows, 2 * splits + 2), dtype=torch.int32, device=scores.device)
    splits_pad = triton.next_power_of_2(splits)
    tiles = dict(CHUNK=chunk, TILE=tile, num_warps=tile // _SHARD_MIN_TILE)
    for p in range(2):
        gluon_eagle4_shard_topk_hist_gfx950[(rows, splits)](
            scores,
            hist,
            scores.stride(0),
            n_cols,
            K=k,
            PASS=p,
            SPLITS=splits,
            SPLITS_PAD=splits_pad,
            **tiles,
        )
    gluon_eagle4_shard_topk_select_gfx950[(rows,)](
        hist, sel, K=k, SPLITS=splits, SPLITS_PAD=splits_pad, num_warps=4
    )
    gluon_eagle4_shard_topk_gfx950[(rows, splits)](
        scores,
        sel,
        global_ids,
        scores.stride(0),
        global_ids.stride(0),
        n_cols,
        vocab_start,
        K=k,
        SPLITS=splits,
        **tiles,
    )


@gluon.jit
def _fold_halves(x, HALF: gl.constexpr, NUM_WARPS: gl.constexpr):
    # Element l of the result is x[l] + x[l + HALF].
    folded = gl.sum(gl.reshape(x, [2, HALF]), 0)
    return gl.convert_layout(folded, _vector_layout(HALF, NUM_WARPS))


@gluon.jit
def _butterfly_sum(x, WIDTH: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Sum of a ``[WIDTH]`` vector in a warp's xor-butterfly order: element l
    first adds element l + WIDTH / 2, and so on down to its neighbour."""
    if WIDTH >= 64:
        x = _fold_halves(x, 32, NUM_WARPS)
    if WIDTH >= 32:
        x = _fold_halves(x, 16, NUM_WARPS)
    if WIDTH >= 16:
        x = _fold_halves(x, 8, NUM_WARPS)
    if WIDTH >= 8:
        x = _fold_halves(x, 4, NUM_WARPS)
    if WIDTH >= 4:
        x = _fold_halves(x, 2, NUM_WARPS)
    return gl.sum(x, 0)


def _candidate_tail_metadata(grid, kernel, args):
    """Report the gathered candidate reads plus the per-row writes."""
    rows = grid[0]
    gathered = args["R"] * args["NC"]
    return {
        "name": kernel.name,
        "bytes": rows
        * (gathered * (2 + 8) + args["NC"] * (4 + 8) + args["TOP_K"] * (4 + 8)),
    }


@gluon.jit(
    launch_metadata=_candidate_tail_metadata,
    do_not_specialize=[
        "stride_cl_r",
        "stride_cl_m",
        "stride_ct_r",
        "stride_ct_m",
    ],
)
def gluon_eagle4_candidate_tail_gfx950(
    cand_logits,  # [R, M, NC] bf16
    cand_tokens,  # [R, M, NC] int64
    logits_out,  # [M, NC] fp32
    tokens_out,  # [M, NC] int64, written when R > 1
    topk_probs,  # [M, TOP_K] fp32
    topk_tokens,  # [M, TOP_K] int64
    stride_cl_r,
    stride_cl_m,
    stride_ct_r,
    stride_ct_m,
    R: gl.constexpr,
    NC: gl.constexpr,
    TOP_K: gl.constexpr,
    TOP_K_PAD: gl.constexpr,
    WARP: gl.constexpr,
    BLOCK: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    layout: gl.constexpr = _vector_layout(BLOCK, NUM_WARPS)
    # One workgroup per row, one element per gathered candidate (candidate
    # order r * NC + j).
    m = gl.program_id(0).to(gl.int64)
    i = gl.arange(0, BLOCK, layout=layout)
    valid = i < R * NC
    r = i // NC
    j = i % NC
    x = gl.load(
        cand_logits + r * stride_cl_r + m * stride_cl_m + j, mask=valid, other=0.0
    )
    tok = gl.load(
        cand_tokens + r * stride_ct_r + m * stride_ct_m + j, mask=valid, other=0
    )
    logit = x.to(gl.float32)
    if R > 1:
        # The NC-th largest key, a byte at a time, then torch.topk's placement:
        # keys above it in candidate order, then its first ties.
        key = gl.where(valid, _bf16_key(x), -1)
        high, above_hi = _kth_bin(
            gl.histogram(
                key >> 8, 256, mask=valid, layout=_vector_layout(256, NUM_WARPS)
            ),
            NC,
            NUM_WARPS,
        )
        hist_lo = gl.histogram(
            key & 255,
            256,
            mask=valid & ((key >> 8) == high),
            layout=_vector_layout(256, NUM_WARPS),
        )
        low, above_lo = _kth_bin(hist_lo, NC - above_hi, NUM_WARPS)
        kth = (high << 8) | low
        n_gt = above_hi + above_lo
        gt = (key > kth).to(gl.int32)
        eq = (key == kth).to(gl.int32)
        rank_eq = gl.cumsum(eq, 0) - eq
        selected = (gt != 0) | ((eq != 0) & (rank_eq < NC - n_gt))
        slot = gl.where(gt != 0, gl.cumsum(gt, 0) - gt, n_gt + rank_eq)
        gl.store(tokens_out + m * NC + slot, tok, mask=selected)
    else:
        selected = valid
        slot = i
    gl.store(logits_out + m * NC + slot, logit, mask=selected)

    if TOP_K > 0:
        # torch.softmax's warp kernel, bit for bit: lane l of a WARP-wide warp
        # sums exp(logit - max) of slots l, l + WARP, ..., then the lanes add up
        # by xor butterfly. Read the logits back in slot order to sum them so;
        # the barrier orders the scattered stores above before those reads.
        gl.barrier()
        slot_layout: gl.constexpr = _vector_layout(NC, NUM_WARPS)
        lane_layout: gl.constexpr = _vector_layout(WARP, NUM_WARPS)
        row_logits = logits_out + m * NC
        slots = gl.arange(0, NC, layout=slot_layout)
        logits = gl.load(row_logits + slots)
        top = gl.max(logits, 0)
        lanes = gl.arange(0, WARP, layout=lane_layout)
        lane_sum = gl.zeros([WARP], gl.float32, layout=lane_layout)
        for it in gl.static_range(NC // WARP):
            lane_sum += gl.extra.libdevice.exp(
                gl.load(row_logits + it * WARP + lanes) - top
            )
        total = _butterfly_sum(lane_sum, WARP, NUM_WARPS)
        score = gl.div_rn(gl.extra.libdevice.exp(logits - top), total)
        if R > 1:
            tokens = gl.load(tokens_out + m * NC + slots)
        else:
            tokens = gl.load(cand_tokens + m * stride_ct_m + slots)
        # The top TOP_K by repeated argmax: the largest probability, ties to
        # the lower slot.
        out_layout: gl.constexpr = _vector_layout(TOP_K_PAD, NUM_WARPS)
        out = gl.arange(0, TOP_K_PAD, layout=out_layout)
        top_probs = gl.zeros([TOP_K_PAD], gl.float32, layout=out_layout)
        top_tokens = gl.zeros([TOP_K_PAD], gl.int64, layout=out_layout)
        for t in gl.static_range(TOP_K):
            best = gl.max(score, 0)
            best_slot = gl.min(gl.where(score == best, slots, NC), 0)
            pick = slots == best_slot
            top_probs = gl.where(out == t, best, top_probs)
            top_tokens = gl.where(
                out == t, gl.sum(gl.where(pick, tokens, 0), 0), top_tokens
            )
            score = gl.where(pick, -1.0, score)
        gl.store(topk_probs + m * TOP_K + out, top_probs, mask=out < TOP_K)
        gl.store(topk_tokens + m * TOP_K + out, top_tokens, mask=out < TOP_K)


def launch_gluon_eagle4_candidate_tail_gfx950(
    cand_logits: torch.Tensor,
    cand_tokens: torch.Tensor,
    top_k: int,
    logits_out: torch.Tensor,
    tokens_out: torch.Tensor,
    topk_probs: torch.Tensor | None,
    topk_tokens: torch.Tensor | None,
) -> None:
    """Everything after the candidate all-gather, in one launch::

        logits, pos = torch.topk(gathered_logits, NC, sorted=False)  # R > 1 only
        tokens = torch.gather(gathered_tokens, -1, pos)
        logits = logits.float()
        top = torch.topk(torch.softmax(logits, -1), top_k, -1)  # top_k > 0 only
        top_tokens = torch.gather(tokens, -1, top.indices)

    The NC-way top-k keeps CUDA's order, as the shard top-k, and the softmax is
    torch's bit for bit; the draft top-k puts tied probabilities in slot order.

    Args:
        cand_logits: ``[R, M, NC]`` bf16 view of the gathered rank blocks, any
            rank and row strides, unit column stride.
        cand_tokens: ``[R, M, NC]`` int64 ids alike.
        top_k: Draft top-k, 0 to skip it.
        logits_out: Contiguous ``[M, NC]`` fp32 candidate logits.
        tokens_out: Contiguous ``[M, NC]`` int64 candidate ids; not written
            when R == 1.
        topk_probs: Contiguous ``[M, top_k]`` fp32, or None when top_k is 0.
        topk_tokens: Contiguous ``[M, top_k]`` int64, or None when top_k is 0.

    Covers NC in 32, 64, 128, 256, ``R * NC <= 1024`` and ``top_k <= 32``.
    """
    ranks, rows, nc = cand_logits.shape
    if rows == 0:
        return
    if cand_logits.stride(2) != 1 or cand_tokens.stride(2) != 1:
        raise ValueError("the candidate tail takes candidates with unit column stride")
    if nc not in (32, 64, 128, 256) or ranks * nc > 1024 or not 0 <= top_k <= 32:
        raise ValueError(
            f"the candidate tail covers NC in 32, 64, 128, 256, R * NC <= 1024 "
            f"and top_k <= 32; got NC {nc}, R {ranks}, top_k {top_k}"
        )
    if top_k == 0:  # the kernel does not touch them
        topk_probs, topk_tokens = logits_out, tokens_out
    block = triton.next_power_of_2(ranks * nc)
    gluon_eagle4_candidate_tail_gfx950[(rows,)](
        cand_logits,
        cand_tokens,
        logits_out,
        tokens_out,
        topk_probs,
        topk_tokens,
        cand_logits.stride(0),
        cand_logits.stride(1),
        cand_tokens.stride(0),
        cand_tokens.stride(1),
        R=ranks,
        NC=nc,
        TOP_K=top_k,
        TOP_K_PAD=triton.next_power_of_2(max(top_k, 1)),
        WARP=min(nc, _WAVE),
        BLOCK=block,
        # One element per lane until four waves are busy.
        num_warps=max(1, min(4, block // _WAVE)),
    )
