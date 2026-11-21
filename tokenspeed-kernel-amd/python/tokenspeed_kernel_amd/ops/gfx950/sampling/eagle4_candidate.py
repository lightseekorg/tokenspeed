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


@gluon.jit
def _bf16_key(x):
    # torch.topk's order-preserving key of a bf16 value; NaN maps to the maximum.
    bits = x.to(gl.uint16, bitcast=True).to(gl.int32)
    key = gl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return gl.where(x != x, 0xFFFF, key)


@gluon.jit
def _shard_keys(scores, stride_s, row, cols, n_cols):
    valid = cols < n_cols
    x = gl.load(scores + row.to(gl.int64) * stride_s + cols, mask=valid, other=0.0)
    return valid, gl.where(valid, _bf16_key(x), -1)


@gluon.jit
def _kth_key(key, target):
    """The ``target``-th largest of a register vector's 16-bit keys (-1 for
    none; at least ``target`` valid), and the count of keys above it.

    A binary search over the key range, one count reduction a step. It costs
    less than a byte-wise select's two gl.histogram passes and their scans:
    those cost several times a count per element."""
    lo = 0  # count(key >= lo) >= target
    n_above = 0  # count(key >= lo + 2 * q) < target
    for step in gl.static_range(16):
        q = 1 << (15 - step)
        c = gl.sum((key >= lo + q).to(gl.int32), 0)
        n_above = gl.where(c >= target, n_above, c)
        lo = gl.where(c >= target, lo + q, lo)
    return lo, n_above


@gluon.jit
def _select_top(key, target):
    """torch.topk's placement of a register vector's ``target`` largest keys:
    keys above the ``target``-th largest in vector order, then its first ties.
    Keys are 16-bit, -1 for none; at least ``target`` are valid. Returns which
    elements are kept and their output slots."""
    kth, n_gt = _kth_key(key, target)
    gt = (key > kth).to(gl.int32)
    eq = (key == kth).to(gl.int32)
    # Both ranks in one scan: gt in the low 16 bits, eq above.
    packed = gl.cumsum(gt + (eq << 16), 0)
    rank_gt = (packed & 0xFFFF) - gt
    rank_eq = (packed >> 16) - eq
    take = (gt != 0) | ((eq != 0) & (rank_eq < target - n_gt))
    slot = gl.where(gt != 0, rank_gt, n_gt + rank_eq)
    return take, slot


def _shard_topk_local_metadata(grid, kernel, args):
    """Report one read of the scores plus the pool writes."""
    rows, splits = grid[0], grid[1]
    return {
        "name": kernel.name,
        "bytes": rows * args["n_cols"] * 2 + rows * splits * args["K"] * 8,
    }


@gluon.jit(
    launch_metadata=_shard_topk_local_metadata,
    do_not_specialize=["stride_s", "n_cols"],
)
def gluon_eagle4_shard_topk_local_gfx950(
    scores,  # [M, N] bf16
    pool_key,  # [M, SPLITS, K] int32: a split's K largest keys, -1 padded
    pool_col,  # [M, SPLITS, K] int32: their columns
    stride_s,
    n_cols,
    K: gl.constexpr,
    K_PAD: gl.constexpr,
    CHUNK: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    layout: gl.constexpr = _vector_layout(CHUNK, NUM_WARPS)
    row = gl.program_id(0)
    split = gl.program_id(1)
    # One read of the split's columns into registers; the search and the
    # placement run on them.
    cols = split * CHUNK + gl.arange(0, CHUNK, layout=layout)
    valid, key = _shard_keys(scores, stride_s, row, cols, n_cols)
    # The last split may hold fewer than K columns: it keeps them all.
    target = gl.minimum(gl.sum(valid.to(gl.int32), 0), K)
    take, slot = _select_top(key, target)
    base = (row.to(gl.int64) * gl.num_programs(1) + split) * K
    gl.store(pool_key + base + slot, key, mask=take)
    gl.store(pool_col + base + slot, cols, mask=take)
    pad_layout: gl.constexpr = _vector_layout(K_PAD, NUM_WARPS)
    pad = gl.arange(0, K_PAD, layout=pad_layout)
    gl.store(
        pool_key + base + pad,
        gl.full([K_PAD], -1, gl.int32, layout=pad_layout),
        mask=(pad >= target) & (pad < K),
    )


def _shard_topk_merge_metadata(grid, kernel, args):
    """Report the pool reads plus the id writes."""
    rows = grid[0]
    return {
        "name": kernel.name,
        "bytes": rows * (args["POOL"] * 4 + args["K"] * (4 + 8)),
    }


@gluon.jit(
    launch_metadata=_shard_topk_merge_metadata,
    do_not_specialize=["stride_g", "vocab_start"],
)
def gluon_eagle4_shard_topk_merge_gfx950(
    pool_key,
    pool_col,
    global_ids,  # [M, K] int64
    stride_g,
    vocab_start,
    K: gl.constexpr,
    POOL: gl.constexpr,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = _vector_layout(BLOCK, gl.num_warps())
    row = gl.program_id(0).to(gl.int64)
    i = gl.arange(0, BLOCK, layout=layout)
    in_pool = i < POOL
    key = gl.load(pool_key + row * POOL + i, mask=in_pool, other=-1)
    # The row's K largest keys are among the splits' K largest. Each split's
    # pool lists its keys above its own k-th largest by column, then that key's
    # ties by column, and the row's k-th largest is at least the split's: so
    # the pool's keys above the row's k-th largest, and its ties, are each in
    # column order, and a split keeps every tie the row takes from it.
    take, slot = _select_top(key, K)
    col = gl.load(pool_col + row * POOL + i, mask=take, other=0)
    gl.store(
        global_ids + row * stride_g + slot,
        col.to(gl.int64) + vocab_start,
        mask=take,
    )


# Columns a split workgroup holds in registers: eight per lane over eight waves.
_SHARD_CHUNK = 4096
_SHARD_CHUNK_WARPS = 8
# The merge holds every split's K keys in registers, 16 a lane (fewer when
# under one wave).
_SHARD_MERGE_MAX_LANES = 1024
_SHARD_MAX_K = 256


def launch_gluon_eagle4_shard_topk_gfx950(
    scores: torch.Tensor, k: int, vocab_start: int, global_ids: torch.Tensor
) -> None:
    """``torch.topk(scores, k, dim=-1, sorted=False).indices + vocab_start`` in
    CUDA's order: the ids of keys above the k-th largest by position, then the
    first k-th largest ties by position.

    Args:
        scores: ``[M, N]`` bf16 scores of a vocab shard, unit column stride.
        k: Ids kept per row, 1 to ``min(N, 256)``.
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
    if not 1 <= k <= min(n_cols, _SHARD_MAX_K):
        raise ValueError(
            f"shard top-k keeps 1..{min(n_cols, _SHARD_MAX_K)} ids, got {k}"
        )
    # N is fixed per server, so the split layout compiles once.
    splits = triton.cdiv(n_cols, _SHARD_CHUNK)
    pool = splits * k
    block = triton.next_power_of_2(pool)
    merge_warps = max(1, min(_SHARD_MERGE_MAX_LANES, block // 16) // _WAVE)
    if block > 16 * _SHARD_MERGE_MAX_LANES:
        raise ValueError(
            f"shard top-k merges at most {16 * _SHARD_MERGE_MAX_LANES} keys a "
            f"row; {splits} splits of {k} need {pool}"
        )
    pool_key = torch.empty((rows, pool), dtype=torch.int32, device=scores.device)
    pool_col = torch.empty((rows, pool), dtype=torch.int32, device=scores.device)
    gluon_eagle4_shard_topk_local_gfx950[(rows, splits)](
        scores,
        pool_key,
        pool_col,
        scores.stride(0),
        n_cols,
        K=k,
        K_PAD=triton.next_power_of_2(k),
        CHUNK=_SHARD_CHUNK,
        num_warps=_SHARD_CHUNK_WARPS,
    )
    gluon_eagle4_shard_topk_merge_gfx950[(rows,)](
        pool_key,
        pool_col,
        global_ids,
        global_ids.stride(0),
        vocab_start,
        K=k,
        POOL=pool,
        BLOCK=block,
        num_warps=merge_warps,
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
        # The NC largest in torch.topk's placement over candidate order.
        selected, slot = _select_top(gl.where(valid, _bf16_key(x), -1), NC)
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
