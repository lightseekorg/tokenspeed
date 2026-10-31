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

"""EAGLE-4 low-rank candidate head: the exact logits of each row's candidates,
``out[b, k] = weight[indices[b, k] - id_offset] . hidden[b]`` in fp32, and the
selections around it (``triton_eagle4_shard_topk`` before,
``triton_eagle4_candidate_tail`` after the all-gather), Triton ports of
``thirdparty/cuda/csrc/eagle4_candidate.cu`` registered as the portable leaves
of ``tokenspeed_kernel.ops.sampling.eagle4_shard_topk`` /
``eagle4_candidate_tail``."""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["gather_dot", "triton_eagle4_candidate_tail", "triton_eagle4_shard_topk"]

_PORTABLE_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))


@triton.jit
def _gather_dot_kernel(
    weight_ptr,  # [V, H] LM head weight
    indices_ptr,  # [B, K] candidate vocab ids, id_offset at weight row 0
    hidden_ptr,  # [B, H]
    output_ptr,  # [B, K]
    stride_w_v,
    stride_w_h,
    stride_i_b,
    stride_i_k,
    stride_h_b,
    stride_h_h,
    stride_o_b,
    stride_o_k,
    K,
    id_offset,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program per BLOCK_K candidates of a row loads the row's hidden once per tile.
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    # int64: vocab_id * stride_w_v can exceed int32 for a large vocab x hidden.
    vocab_ids = tl.load(
        indices_ptr + pid_b * stride_i_b + k_offs * stride_i_k, mask=k_mask, other=0
    ).to(tl.int64)
    vocab_ids -= id_offset

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    EVEN_H: tl.constexpr = H % BLOCK_H == 0
    for off_h in range(0, H, BLOCK_H):
        cols = off_h + tl.arange(0, BLOCK_H)
        if EVEN_H:
            h_val = tl.load(hidden_ptr + pid_b * stride_h_b + cols * stride_h_h)
            w_val = tl.load(
                weight_ptr
                + vocab_ids[:, None] * stride_w_v
                + cols[None, :] * stride_w_h,
                mask=k_mask[:, None],
                other=0.0,
            )
        else:
            h_mask = cols < H
            h_val = tl.load(
                hidden_ptr + pid_b * stride_h_b + cols * stride_h_h,
                mask=h_mask,
                other=0.0,
            )
            w_val = tl.load(
                weight_ptr
                + vocab_ids[:, None] * stride_w_v
                + cols[None, :] * stride_w_h,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
        # Products and sums in fp32: the one rounding is the store's.
        acc += tl.sum(w_val.to(tl.float32) * h_val.to(tl.float32)[None, :], axis=1)

    tl.store(
        output_ptr + pid_b * stride_o_b + k_offs * stride_o_k,
        acc.to(output_ptr.dtype.element_ty),
        mask=k_mask,
    )


# (candidates, hidden) where BLOCK_K 1 was validated against BLOCK_K 4 (same H reduction layout).
_ONE_CANDIDATE_SHAPES = frozenset({(128, 5120)})


def _tile_config(num_rows: int, num_candidates: int, hidden: int, dtype: torch.dtype):
    """Static (BLOCK_K, BLOCK_H, num_warps, num_stages) per row bucket: few rows
    are latency bound and want deep H tiles, many rows bandwidth bound and
    want wide K tiles. Every bucket up to 32 rows reduces along H in the same
    layout, so their results agree bit for bit."""
    if num_rows <= 2:
        return 2, 2048, 8, 3
    if num_rows <= 32:
        shape = (num_candidates, hidden)
        one = shape in _ONE_CANDIDATE_SHAPES and dtype == torch.bfloat16
        return (1 if one else 4), 2048, 8, 3
    return 32, 256, 4, 3


def gather_dot(
    weight: torch.Tensor,
    indices: torch.Tensor,
    hidden_states: torch.Tensor,
    id_offset: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score gathered LM head rows:
    ``out[b, k] = weight[indices[b, k] - id_offset] . hidden[b]``.

    Args:
        weight: ``[V, H]`` LM head weight (this rank's vocab shard).
        indices: ``[B, K]`` vocab ids of each row's candidates, any strides.
        hidden_states: ``[B, H]`` in the weight's dtype.
        id_offset: The vocab id of ``weight`` row 0 (the shard's first id).
        out: ``[B, K]`` destination in the hidden dtype, any strides (e.g. a
            block of a packed all-gather buffer); a new tensor when None.

    Returns:
        The ``[B, K]`` scores, computed in fp32 and stored in the hidden dtype.
    """
    num_rows, num_candidates = indices.shape
    hidden = weight.shape[1]
    if out is None:
        out = torch.empty(
            (num_rows, num_candidates), device=indices.device, dtype=hidden_states.dtype
        )
    if num_rows == 0:
        return out
    block_k, block_h, num_warps, num_stages = _tile_config(
        num_rows, num_candidates, hidden, hidden_states.dtype
    )
    block_h = min(block_h, triton.next_power_of_2(hidden))
    _gather_dot_kernel[(num_rows, triton.cdiv(num_candidates, block_k))](
        weight,
        indices,
        hidden_states,
        out,
        weight.stride(0),
        weight.stride(1),
        indices.stride(0),
        indices.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        out.stride(0),
        out.stride(1),
        num_candidates,
        id_offset,
        H=hidden,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def _bf16_key(x):
    # torch.topk's order-preserving key of a bf16 value; NaN maps to the maximum.
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return tl.where(x != x, 0xFFFF, key)


@triton.jit
def _kth_bin(hist, target):
    """The bin of the ``target``-th largest key of a 256-bin histogram, and the
    count of keys in higher bins."""
    bins = tl.arange(0, 256)
    at_least = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist  # keys in bins >= b
    digit = tl.max(tl.where(at_least >= target, bins, 0), 0)
    above = tl.sum(tl.where(bins == digit, at_least - hist, 0), 0)
    return digit, above


@triton.jit
def _shard_keys(scores_ptr, stride_s, row, cols, n_cols):
    valid = cols < n_cols
    x = tl.load(scores_ptr + row.to(tl.int64) * stride_s + cols, mask=valid, other=0.0)
    return valid, tl.where(valid, _bf16_key(x), -1)


@triton.jit
def _split_hists(hist_ptr, SPLITS: tl.constexpr, SPLITS_PAD: tl.constexpr):
    # [SPLITS_PAD, 256]: one 256-bin histogram per split of the row, zero padded.
    splits = tl.arange(0, SPLITS_PAD)
    bins = tl.arange(0, 256)
    return tl.load(
        hist_ptr + splits[:, None] * 256 + bins[None, :],
        mask=(splits < SPLITS)[:, None],
        other=0,
    )


@triton.jit(do_not_specialize=["stride_s", "n_cols"])
def _shard_topk_hist_kernel(
    scores_ptr,  # [M, N] bf16
    hist_ptr,  # [M, 2, SPLITS, 256] int32: high byte, then low byte in its bin
    stride_s,
    n_cols,
    K: tl.constexpr,
    PASS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    CHUNK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    row_hist = hist_ptr + row * (2 * SPLITS * 256)
    if PASS == 1:
        high, _ = _kth_bin(tl.sum(_split_hists(row_hist, SPLITS, SPLITS_PAD), 0), K)
    counts = tl.zeros([256], dtype=tl.int32)
    for start in range(0, CHUNK, TILE):
        cols = chunk * CHUNK + start + tl.arange(0, TILE)
        valid, key = _shard_keys(scores_ptr, stride_s, row, cols, n_cols)
        if PASS == 0:
            counts += tl.histogram(key >> 8, 256, mask=valid)
        else:
            counts += tl.histogram(key & 255, 256, mask=valid & ((key >> 8) == high))
    tl.store(row_hist + (PASS * SPLITS + chunk) * 256 + tl.arange(0, 256), counts)


@triton.jit
def _shard_topk_select_kernel(
    hist_ptr,
    sel_ptr,  # [M, 2 * SPLITS + 2] int32: per-split gt and tie bases, kth, n_gt
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    row_hist = hist_ptr + row * (2 * SPLITS * 256)
    bins = tl.arange(0, 256)
    hist_hi = _split_hists(row_hist, SPLITS, SPLITS_PAD)
    hist_lo = _split_hists(row_hist + SPLITS * 256, SPLITS, SPLITS_PAD)
    high, above_hi = _kth_bin(tl.sum(hist_hi, 0), K)
    low, above_lo = _kth_bin(tl.sum(hist_lo, 0), K - above_hi)
    # Keys above kth and kth ties of each split, to place each split after the earlier ones.
    split_gt = tl.sum(tl.where(bins[None, :] > high, hist_hi, 0), 1) + tl.sum(
        tl.where(bins[None, :] > low, hist_lo, 0), 1
    )
    split_eq = tl.sum(tl.where(bins[None, :] == low, hist_lo, 0), 1)
    splits = tl.arange(0, SPLITS_PAD)
    row_sel = sel_ptr + row * (2 * SPLITS + 2)
    tl.store(row_sel + splits, tl.cumsum(split_gt, 0) - split_gt, mask=splits < SPLITS)
    tl.store(
        row_sel + SPLITS + splits,
        tl.cumsum(split_eq, 0) - split_eq,
        mask=splits < SPLITS,
    )
    tl.store(row_sel + 2 * SPLITS, (high << 8) | low)
    tl.store(row_sel + 2 * SPLITS + 1, above_hi + above_lo)


@triton.jit(do_not_specialize=["stride_s", "stride_g", "n_cols", "vocab_start"])
def _shard_topk_emit_kernel(
    scores_ptr,
    sel_ptr,
    global_ids_ptr,  # [M, K] int64
    stride_s,
    stride_g,
    n_cols,
    vocab_start,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    row_sel = sel_ptr + row * (2 * SPLITS + 2)
    base_gt = tl.load(row_sel + chunk)
    base_eq = tl.load(row_sel + SPLITS + chunk)
    kth = tl.load(row_sel + 2 * SPLITS)
    n_gt = tl.load(
        row_sel + 2 * SPLITS + 1
    )  # keys above kth; then K - n_gt of its ties

    out_row = global_ids_ptr + row.to(tl.int64) * stride_g
    for start in range(0, CHUNK, TILE):
        cols = chunk * CHUNK + start + tl.arange(0, TILE)
        _, key = _shard_keys(scores_ptr, stride_s, row, cols, n_cols)
        gt = (key > kth).to(tl.int32)
        eq = (key == kth).to(tl.int32)
        tile_gt = tl.sum(gt, 0)
        tile_eq = tl.sum(eq, 0)
        # Most tiles hold none of the K keys: skip their scans.
        if tile_gt + tile_eq > 0:
            rank_gt = base_gt + tl.cumsum(gt, 0) - gt
            rank_eq = base_eq + tl.cumsum(eq, 0) - eq
            take = (gt != 0) | ((eq != 0) & (rank_eq < K - n_gt))
            slot = tl.where(gt != 0, rank_gt, n_gt + rank_eq)
            tl.store(out_row + slot, cols.to(tl.int64) + vocab_start, mask=take)
        base_gt += tile_gt
        base_eq += tile_eq


# Splits per row: the CUDA kernel's 8-CTA cluster, doubled (no cluster sync here).
_SHARD_SPLITS = 16
# Columns a program holds in registers at a time; it loops over its split.
_SHARD_TILE = 2048


@register_kernel(
    "sampling",
    "eagle4_shard_topk",
    name="triton_eagle4_shard_topk",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures("scores", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def triton_eagle4_shard_topk(
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
        raise ValueError("shard_topk takes bf16 scores with unit column stride")
    if global_ids.dtype != torch.int64 or global_ids.stride(1) != 1:
        raise ValueError("shard_topk writes int64 ids with unit column stride")
    if not 1 <= k <= n_cols:
        raise ValueError(f"shard_topk keeps 1..{n_cols} ids, got {k}")
    # N is fixed per server, so the split layout compiles once.
    tile = min(_SHARD_TILE, triton.next_power_of_2(triton.cdiv(n_cols, _SHARD_SPLITS)))
    chunk = triton.cdiv(triton.cdiv(n_cols, _SHARD_SPLITS), tile) * tile
    splits = triton.cdiv(n_cols, chunk)
    hist = torch.empty((rows, 2, splits, 256), dtype=torch.int32, device=scores.device)
    sel = torch.empty((rows, 2 * splits + 2), dtype=torch.int32, device=scores.device)
    splits_pad = triton.next_power_of_2(splits)
    tiles = dict(CHUNK=chunk, TILE=tile, num_warps=max(1, min(8, tile // 256)))
    for p in range(2):
        _shard_topk_hist_kernel[(rows, splits)](
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
    _shard_topk_select_kernel[(rows,)](
        hist, sel, K=k, SPLITS=splits, SPLITS_PAD=splits_pad, num_warps=4
    )
    _shard_topk_emit_kernel[(rows, splits)](
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


@triton.jit
def _butterfly_sum(x, WIDTH: tl.constexpr):
    """Sum of a ``[WIDTH]`` vector in a warp's xor-butterfly order: element l
    first adds element l + WIDTH / 2, and so on down to its neighbour."""
    if WIDTH >= 64:
        x = tl.sum(tl.reshape(x, (2, 32)), 0)
    if WIDTH >= 32:
        x = tl.sum(tl.reshape(x, (2, 16)), 0)
    if WIDTH >= 16:
        x = tl.sum(tl.reshape(x, (2, 8)), 0)
    if WIDTH >= 8:
        x = tl.sum(tl.reshape(x, (2, 4)), 0)
    if WIDTH >= 4:
        x = tl.sum(tl.reshape(x, (2, 2)), 0)
    return tl.sum(x, 0)


@triton.jit(
    do_not_specialize=[
        "stride_cl_r",
        "stride_cl_m",
        "stride_ct_r",
        "stride_ct_m",
    ]
)
def _candidate_tail_kernel(
    cand_logits_ptr,  # [R, M, NC] bf16
    cand_tokens_ptr,  # [R, M, NC] int64
    logits_out_ptr,  # [M, NC] fp32
    tokens_out_ptr,  # [M, NC] int64, written when R > 1
    topk_probs_ptr,  # [M, TOP_K] fp32
    topk_tokens_ptr,  # [M, TOP_K] int64
    stride_cl_r,
    stride_cl_m,
    stride_ct_r,
    stride_ct_m,
    R: tl.constexpr,
    NC: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    WARP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per row, one lane per gathered candidate (candidate order r * NC + j).
    m = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BLOCK)
    valid = i < R * NC
    r = i // NC
    j = i % NC
    x = tl.load(
        cand_logits_ptr + r * stride_cl_r + m * stride_cl_m + j, mask=valid, other=0.0
    )
    tok = tl.load(
        cand_tokens_ptr + r * stride_ct_r + m * stride_ct_m + j, mask=valid, other=0
    )
    logit = x.to(tl.float32)
    if R > 1:
        # The NC-th largest key, a byte at a time, then torch.topk's placement:
        # keys above it in candidate order, then its first ties.
        key = tl.where(valid, _bf16_key(x), -1)
        high, above_hi = _kth_bin(tl.histogram(key >> 8, 256, mask=valid), NC)
        hist_lo = tl.histogram(key & 255, 256, mask=valid & ((key >> 8) == high))
        low, above_lo = _kth_bin(hist_lo, NC - above_hi)
        kth = (high << 8) | low
        n_gt = above_hi + above_lo
        gt = (key > kth).to(tl.int32)
        eq = (key == kth).to(tl.int32)
        rank_eq = tl.cumsum(eq, 0) - eq
        selected = (gt != 0) | ((eq != 0) & (rank_eq < NC - n_gt))
        slot = tl.where(gt != 0, tl.cumsum(gt, 0) - gt, n_gt + rank_eq)
        tl.store(tokens_out_ptr + m * NC + slot, tok, mask=selected)
    else:
        selected = valid
        slot = i
    tl.store(logits_out_ptr + m * NC + slot, logit, mask=selected)

    if TOP_K > 0:
        # torch.softmax's warp kernel, bit for bit: lane l of a WARP-wide warp
        # sums exp(logit - max) of slots l, l + WARP, ..., then the lanes add up
        # by xor butterfly. Read the logits back in slot order to sum them so.
        tl.debug_barrier()
        row_logits = logits_out_ptr + m * NC
        slots = tl.arange(0, NC)
        logits = tl.load(row_logits + slots)
        top = tl.max(logits, 0)
        lanes = tl.arange(0, WARP)
        lane_sum = tl.zeros([WARP], dtype=tl.float32)
        for it in tl.static_range(NC // WARP):
            lane_sum += libdevice.exp(tl.load(row_logits + it * WARP + lanes) - top)
        total = _butterfly_sum(lane_sum, WARP)
        score = tl.div_rn(libdevice.exp(logits - top), total)
        if R > 1:
            tokens = tl.load(tokens_out_ptr + m * NC + slots)
        else:
            tokens = tl.load(cand_tokens_ptr + m * stride_ct_m + slots)
        # The top TOP_K by repeated argmax: the largest probability, ties to
        # the lower slot.
        out = tl.arange(0, TOP_K_PAD)
        top_probs = tl.zeros(out.shape, dtype=tl.float32)
        top_tokens = tl.zeros(out.shape, dtype=tl.int64)
        for t in tl.static_range(TOP_K):
            best = tl.max(score, 0)
            best_slot = tl.min(tl.where(score == best, slots, NC), 0)
            pick = slots == best_slot
            top_probs = tl.where(out == t, best, top_probs)
            top_tokens = tl.where(
                out == t, tl.sum(tl.where(pick, tokens, 0), 0), top_tokens
            )
            score = tl.where(pick, -1.0, score)
        tl.store(topk_probs_ptr + m * TOP_K + out, top_probs, mask=out < TOP_K)
        tl.store(topk_tokens_ptr + m * TOP_K + out, top_tokens, mask=out < TOP_K)


@functools.cache
def _warp_size(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).warp_size


@register_kernel(
    "sampling",
    "eagle4_candidate_tail",
    name="triton_eagle4_candidate_tail",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures("cand_logits", "dense", {torch.bfloat16}),
    priority=Priority.PORTABLE,
)
def triton_eagle4_candidate_tail(
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

    The NC-way top-k keeps CUDA's order, as ``triton_eagle4_shard_topk``, and the softmax is
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
    """
    ranks, rows, nc = cand_logits.shape
    if rows == 0:
        return
    if cand_logits.stride(2) != 1 or cand_tokens.stride(2) != 1:
        raise ValueError("candidate_tail takes candidates with unit column stride")
    if top_k == 0:  # the kernel does not touch them
        topk_probs, topk_tokens = logits_out, tokens_out
    _candidate_tail_kernel[(rows,)](
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
        WARP=min(nc, _warp_size(cand_logits.device)),
        BLOCK=triton.next_power_of_2(ranks * nc),
        num_warps=4,
    )
