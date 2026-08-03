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

"""Triton kernels for short causal convolution (sconv).

Five kernels backing the public API in :mod:`tokenspeed_kernel.ops.conv`:

Dense conv-cache family:

- ``_sconv_prefill_kernel``: varlen-packed causal conv with per-request
  initial state loaded directly from the conv cache (read-only on the cache).
- ``_sconv_cache_update_kernel``: final-state writeback of the last ``W - 1``
  tokens per request into the conv cache, handling ``query_len < W - 1`` by
  shifting the old cache contents.
- ``_sconv_decode_kernel``: fused single-token decode — conv over
  ``[cache ++ x_t]`` plus in-place cache shift-update in one launch.

Paged (page-table) family, position-addressing input columns in the paged
pools (no shift-update; every processed token's column is persisted):

- ``_sconv_prefill_paged_kernel``: varlen-packed causal conv reading
  cached-prefix taps from the pools through the page table.
- ``_sconv_decode_paged_kernel``: single-token decode — the last ``W - 1``
  taps read through the page table plus the incoming token.

All kernels take explicit strides for the conv cache so channel-sliced views
(``cache[:, :, off:off + D]``) work without a copy. ``PAD_SLOT_ID`` (-1) rows
never read from or write to a real cache slot.

No autotuning is used: block configurations come from static heuristics so
the kernels stay CUDA-graph friendly.

Weight taps are loaded once as a 2D ``[BLOCK_D, W_POW2]`` tile and selected
per tap with an equality reduction instead of ``W`` separate 1D gathers at
constant offsets ``0..W-1``: the separate-gather pattern miscompiles on the
current tokenspeed_triton release (wrong values with ``num_warps > 1`` when
the compiler merges the strided gathers).
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "select_decode_config",
    "select_prefill_config",
]


@triton.jit
def _load_weight_taps(
    weight_ptr,
    d_off,
    d_mask,
    stride_w_d,
    stride_w_w,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """Load all ``W`` taps for a channel block as one ``[BLOCK_D, W_POW2]`` tile."""
    w_off = tl.arange(0, W_POW2)
    return tl.load(
        weight_ptr + d_off[:, None] * stride_w_d + w_off[None, :] * stride_w_w,
        mask=d_mask[:, None] & (w_off[None, :] < W),
        other=0,
    ).to(tl.float32)


@triton.jit
def _select_weight_tap(w_all, iw: tl.constexpr, W_POW2: tl.constexpr):
    """Extract tap ``iw`` (a ``[BLOCK_D]`` vector) from the 2D weight tile."""
    w_off = tl.arange(0, W_POW2)
    return tl.sum(tl.where(w_off[None, :] == iw, w_all, 0.0), axis=1)


# -----------------------------------------------------------------------------
# Prefill: causal conv with per-request initial state (cache is read-only)
# -----------------------------------------------------------------------------


@triton.jit
def _sconv_prefill_kernel(
    x_ptr,  # [T, D]
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, W-1, D]
    cu_seqlens_ptr,  # [B+1] int32
    seq_idx_ptr,  # [T] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_c_slot,
    stride_c_w,
    stride_c_d,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """Depthwise causal conv1d over ``[cached-prefix ++ chunk]`` per request.

    Grid: ``(cdiv(T, BLOCK_T), cdiv(D, BLOCK_D))``. Tiles may straddle
    sequence boundaries, so every conv tap is masked per element: positions
    ``shifted_t >= bos`` read from ``x``; positions before ``bos`` read from
    the request's cache slot (row ``shifted_t - bos + (W - 1)``), zeroed when
    the request has no initial state or its slot is ``PAD_SLOT_ID``.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    si = tl.load(seq_idx_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens_ptr + si, mask=t_mask, other=0).to(tl.int64)
    ci = tl.load(cache_indices_ptr + si, mask=t_mask, other=-1)
    has_state = tl.load(has_initial_state_ptr + si, mask=t_mask, other=0)
    # PAD_SLOT_ID = -1 (can't reference a Python global inside @jit)
    use_cache = (ci != -1) & (has_state != 0)
    safe_ci = tl.maximum(ci, 0).to(tl.int64)

    w_all = _load_weight_taps(
        weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
    )
    cache_base = (
        conv_cache_ptr + safe_ci[:, None] * stride_c_slot + d_off[None, :] * stride_c_d
    )

    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    t64 = t_off.to(tl.int64)

    # Keep tap W-1 in-loop: a split unconditional load miscompiles tokenspeed_triton (num_warps>1).
    for iw in tl.static_range(W):
        shifted_t = t64 - (W - 1) + iw
        in_x = (shifted_t >= bos) & t_mask
        x_val = tl.load(
            x_ptr + shifted_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=in_x[:, None] & d_mask[None, :],
            other=0,
        )
        prefix_pos = shifted_t - bos + (W - 1)
        in_prefix = (shifted_t < bos) & (prefix_pos >= 0) & use_cache & t_mask
        p_val = tl.load(
            cache_base + prefix_pos[:, None] * stride_c_w,
            mask=in_prefix[:, None] & d_mask[None, :],
            other=0,
        )
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        v = tl.where(in_x[:, None], x_val.to(tl.float32), p_val.to(tl.float32))
        acc += v * w_val[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        xv = tl.load(
            x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=td_mask,
            other=0,
        )
        acc += xv.to(tl.float32)

    tl.store(
        y_ptr + t64[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask,
    )


def select_prefill_config(T: int, D: int) -> tuple[int, int, int, int]:
    """Select ``(BLOCK_T, BLOCK_D, num_warps, num_stages)`` for prefill.

    Static heuristic (no autotune) so the kernel stays CUDA-graph friendly.
    Swept on B200 across (T, D) in {512..8192} x {512, 6144}: a 32x128 tile
    with 8 warps wins or ties everywhere (D is the contiguous axis, so the
    wider channel block doubles the burst size; 4096x6144 drops 66 -> 56 us,
    8192x6144 130 -> 110 us, small shapes improve slightly).

    Args:
        T: Total number of packed tokens.
        D: Number of channels.

    Returns:
        Tuple ``(BLOCK_T, BLOCK_D, num_warps, num_stages)``.
    """
    del T, D
    return 32, 128, 8, 3


# -----------------------------------------------------------------------------
# Cache update: write back the final W-1 states per request
# -----------------------------------------------------------------------------


@triton.jit
def _sconv_cache_update_kernel(
    x_ptr,  # [T, D]
    conv_cache_ptr,  # [num_slots, W-1, D]
    cu_seqlens_ptr,  # [B+1] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    stride_x_t,
    stride_x_d,
    stride_c_slot,
    stride_c_w,
    stride_c_d,
    D,
    BLOCK_D: tl.constexpr,
    W_MINUS_1: tl.constexpr,
):
    """Write the final ``W - 1`` conv states of each request into its slot.

    Grid: ``(B, cdiv(D, BLOCK_D))``; one program per (request, channel block).

    For each output cache position ``w`` the new value is either:

    - ``x[end - (W-1) + w]`` when ``query_len >= W-1 - w`` (token from x);
    - the shifted old cache ``cache[ci, w + query_len]`` when
      ``query_len < W-1 - w`` and the request has an initial state;
    - zero otherwise.

    Rows with ``cache_indices == PAD_SLOT_ID`` (or ``query_len <= 0``) exit
    before touching memory — unlike the TML reference, which clamped PAD rows
    to slot 0 and raced against the real occupant of that slot.

    The shift path reads ``cache[ci, w + query_len]`` with ``w + query_len >
    w`` while writes ascend from ``w = 0``, so every read observes the
    pre-update value.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    ci = tl.load(cache_indices_ptr + pid_b)
    if ci == -1:  # PAD_SLOT_ID: fully skip, never clamp to slot 0
        return
    start = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
    end = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
    query_len = end - start
    if query_len <= 0:
        return
    has_state = tl.load(has_initial_state_ptr + pid_b)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    cache_base = conv_cache_ptr + ci.to(tl.int64) * stride_c_slot + d_off * stride_c_d

    for w in tl.static_range(W_MINUS_1):
        if query_len >= W_MINUS_1 - w:
            x_val = tl.load(
                x_ptr + (end - W_MINUS_1 + w) * stride_x_t + d_off * stride_x_d,
                mask=d_mask,
                other=0,
            )
            val = x_val.to(conv_cache_ptr.dtype.element_ty)
        else:
            if has_state != 0:
                val = tl.load(
                    cache_base + (w + query_len) * stride_c_w, mask=d_mask, other=0
                )
            else:
                val = tl.zeros([BLOCK_D], dtype=conv_cache_ptr.dtype.element_ty)
        tl.store(cache_base + w * stride_c_w, val, mask=d_mask)


# -----------------------------------------------------------------------------
# Fused decode: conv + cache shift-update in one launch
# -----------------------------------------------------------------------------


@triton.jit
def _sconv_decode_kernel(
    x_ptr,  # [B, D]
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, W-1, D]
    cache_indices_ptr,  # [B] int32
    y_ptr,  # [B, D]
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_c_slot,
    stride_c_w,
    stride_c_d,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Fused depthwise causal conv1d + cache shift-update for decode.

    Decode invariant: token ``t`` belongs to request ``t``.

    - taps ``iw = 0..W-2`` read from ``conv_cache`` (the state history);
    - tap ``iw = W-1`` reads from ``x`` (the current token);
    - afterwards the cache row is shifted left by one and ``x_t`` is stored
      at position ``W-2``.

    Rows with ``cache_indices == PAD_SLOT_ID`` compute their output with a
    zeroed prefix and never write to the cache.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    ci = tl.load(cache_indices_ptr + t_off, mask=t_mask, other=-1)
    safe_ci = tl.maximum(ci, 0).to(tl.int64)
    valid = ci != -1  # PAD_SLOT_ID = -1 (can't reference Python global in @jit)
    cache_mask = td_mask & valid[:, None]

    cache_base = (
        conv_cache_ptr + safe_ci[:, None] * stride_c_slot + d_off[None, :] * stride_c_d
    )
    w_all = _load_weight_taps(
        weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
    )

    # ---- CONV ----
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    # Cache taps: iw = 0..W-2
    for iw in tl.static_range(W - 1):
        pv = tl.load(
            cache_base + iw * stride_c_w,
            mask=cache_mask,
            other=0,
            eviction_policy="evict_last",
        )
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        acc += pv.to(tl.float32) * w_val[None, :]

    # Current token: iw = W-1
    xv = tl.load(
        x_ptr + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    xv_f32 = xv.to(tl.float32)
    w_last = _select_weight_tap(w_all, W - 1, W_POW2)
    acc += xv_f32 * w_last[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if USE_RESIDUAL:
        acc += xv_f32

    tl.store(
        y_ptr + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask,
    )

    # Update cache: shift left one step, then write x_t at column W-2.
    for iw in tl.static_range(W - 2):
        shifted = tl.load(cache_base + (iw + 1) * stride_c_w, mask=cache_mask, other=0)
        tl.store(cache_base + iw * stride_c_w, shifted, mask=cache_mask)
    tl.store(
        cache_base + (W - 2) * stride_c_w,
        xv.to(conv_cache_ptr.dtype.element_ty),
        mask=cache_mask,
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def select_decode_config(T: int, D: int) -> tuple[int, int, int, int]:
    """Select ``(BLOCK_T, BLOCK_D, num_warps, num_stages)`` for fused decode.

    Static heuristic (no autotune, ported from the TML reference): keep
    ``BLOCK_T`` small (1-2) since ``T = B`` is moderate during decode, and
    scale ``BLOCK_D`` so that the grid has enough blocks to fill the GPU.

    Args:
        T: Number of decode tokens (equals the batch size).
        D: Number of channels.

    Returns:
        Tuple ``(BLOCK_T, BLOCK_D, num_warps, num_stages)``.
    """
    if current_platform().is_amd and T == 1 and D == 6144:
        return 1, 32, 1, 3

    if T <= 2048:
        block_t = 2
    else:
        # Round down to a power of 2: tl.arange sizes must be powers of 2.
        raw = min(T // 1024, 8)
        block_t = 1 << (raw.bit_length() - 1)

    target_blocks = 1024
    t_blocks = max(T // block_t, 1)
    needed_d_blocks = max(target_blocks // t_blocks, 1)
    block_d = max(D // needed_d_blocks, 64)
    block_d = 1 << max(min(block_d.bit_length() - 1, 9), 6)

    tile_elems = block_t * block_d
    if tile_elems <= 128:
        num_warps = 1
    elif tile_elems <= 512:
        num_warps = 2
    else:
        num_warps = 4

    return block_t, block_d, num_warps, 3


# -----------------------------------------------------------------------------
# Paged sconv: per-token input columns in paged storage (SWA-semantics group)
# -----------------------------------------------------------------------------


@triton.jit
def _sconv_decode_paged_kernel(
    x_ptr,  # [B, D]
    weight_ptr,  # [D, W]
    pool_ptr,  # [num_blocks, BT, C]; slot/row strides may carry slack
    pool2_ptr,  # second pool for the d >= HALF_D columns (fused K+V), or dummy
    page_table_ptr,  # [B, max_blocks] int32; -1 = hole/pad
    seq_lens_ptr,  # [B] int32, length INCLUDING the current token
    y_ptr,  # [B, D]
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_p_slot,
    stride_p_row,
    stride_p_c,
    stride_pt_b,
    col_offset,
    T,
    D,
    BT: tl.constexpr,  # tokens per block (block_size)
    HALF_D: tl.constexpr,  # 0 = single pool; else columns >= HALF_D use pool2
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    EARLY_RELEASE: tl.constexpr,
):
    """Decode conv over paged per-token input columns.

    Tap ``iw < W-1`` reads position ``pos - (W-1) + iw`` through the page
    table (zero when the position is negative or its block is a hole); tap
    ``W-1`` reads the current token from ``x``, which is also persisted to
    the column slice ``[col_offset, col_offset + D)`` of the token's pool
    row. No shift: columns are position-addressed, so MTP rollback and
    prefix restores are free.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    # Fused K+V: BLOCK_D divides HALF_D, so each program's pool base/column offset are scalars.
    if HALF_D > 0 and tl.program_id(1) * BLOCK_D >= HALF_D:
        pbase = pool2_ptr
        pcol = col_offset + d_off - HALF_D
    else:
        pbase = pool_ptr
        pcol = col_offset + d_off

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    seq = tl.load(seq_lens_ptr + t_off, mask=t_mask, other=1)
    pos = (seq - 1).to(tl.int64)

    w_all = _load_weight_taps(
        weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
    )
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for iw in tl.static_range(W - 1):
        p = pos - (W - 1) + iw
        in_range = p >= 0
        p_safe = tl.maximum(p, 0)
        blk = tl.load(
            page_table_ptr + t_off * stride_pt_b + p_safe // BT,
            mask=t_mask & in_range,
            other=-1,
        )
        live = in_range & (blk >= 0)
        off = (
            tl.maximum(blk, 0).to(tl.int64) * stride_p_slot
            + (p_safe % BT) * stride_p_row
        )
        pv = tl.load(
            pbase + off[:, None] + pcol[None, :] * stride_p_c,
            mask=td_mask & live[:, None],
            other=0.0,  # float literal: fp8 pools cannot cast an int fill
            eviction_policy="evict_last",
        )
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        acc += pv.to(tl.float32) * w_val[None, :]

    xv = tl.load(
        x_ptr + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    xv_f32 = xv.to(tl.float32)
    w_last = _select_weight_tap(w_all, W - 1, W_POW2)
    acc += xv_f32 * w_last[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if USE_RESIDUAL:
        acc += xv_f32

    tl.store(
        y_ptr + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask,
    )

    if ENABLE_PDL and EARLY_RELEASE:
        # Safe early release: the persist below touches only the conv pool, unread within this step.
        tl.extra.cuda.gdc_launch_dependents()

    # Persist the current token's input column slice.
    blk_cur = tl.load(
        page_table_ptr + t_off * stride_pt_b + pos // BT, mask=t_mask, other=-1
    )
    live_cur = blk_cur >= 0
    off_cur = (
        tl.maximum(blk_cur, 0).to(tl.int64) * stride_p_slot + (pos % BT) * stride_p_row
    )
    tl.store(
        pbase + off_cur[:, None] + pcol[None, :] * stride_p_c,
        xv.to(pool_ptr.dtype.element_ty),
        mask=td_mask & live_cur[:, None],
    )

    # Default: signal PDL dependents at kernel end (EARLY_RELEASE signals sooner).
    if ENABLE_PDL and not EARLY_RELEASE:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _sconv_prefill_paged_kernel(
    x_ptr,  # [T_total, D] varlen-packed chunk tokens
    weight_ptr,  # [D, W]
    pool_ptr,  # [num_blocks, BT, C]; slot/row strides may carry slack
    pool2_ptr,  # second pool for the d >= HALF_D columns (fused K+V), or dummy
    page_table_ptr,  # [num_reqs, max_blocks] int32; -1 = hole
    seq_idx_ptr,  # [T_total] int32 request index per token
    cu_seqlens_ptr,  # [num_reqs + 1] int32 chunk boundaries
    prefix_lens_ptr,  # [num_reqs] int32 cached-prefix length per request
    y_ptr,  # [T_total, D]
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_p_slot,
    stride_p_row,
    stride_p_c,
    stride_pt_b,
    col_offset,
    T,
    D,
    BT: tl.constexpr,
    HALF_D: tl.constexpr,  # 0 = single pool; else columns >= HALF_D use pool2
    LCM_ALIGN: tl.constexpr,  # restore-boundary alignment; 0 = persist all
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """Varlen prefill conv over paged input columns.

    Token at absolute position ``p = prefix_len + local_t`` reads taps from:
    the chunk itself (``pj >= prefix_len``: x rows, always resident), the
    paged prefix (``0 <= pj < prefix_len``: pool via page table; holes read
    zero — SWA-punched history contributes nothing), or zero (``pj < 0``).
    Every chunk token's input column is persisted to the pool.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    # Fused K+V: BLOCK_D divides HALF_D, so each program's pool base/column offset are scalars.
    if HALF_D > 0 and tl.program_id(1) * BLOCK_D >= HALF_D:
        pbase = pool2_ptr
        pcol = col_offset + d_off - HALF_D
    else:
        pbase = pool_ptr
        pcol = col_offset + d_off

    req = tl.load(seq_idx_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    chunk_start = tl.load(cu_seqlens_ptr + req, mask=t_mask, other=0)
    prefix = tl.load(prefix_lens_ptr + req, mask=t_mask, other=0).to(tl.int64)
    local_t = t_off - chunk_start
    pos = prefix + local_t

    w_all = _load_weight_taps(
        weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
    )
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for iw in tl.static_range(W - 1):
        pj = pos - (W - 1) + iw
        in_chunk = pj >= prefix
        in_prefix = (pj >= 0) & (pj < prefix)
        # chunk source: x row = chunk_start + (pj - prefix)
        xrow = tl.maximum(chunk_start + (pj - prefix), 0)
        cv = tl.load(
            x_ptr + xrow[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=td_mask & in_chunk[:, None],
            other=0,
        )
        # prefix source: pool row via table
        pj_safe = tl.maximum(pj, 0)
        blk = tl.load(
            page_table_ptr + req * stride_pt_b + pj_safe // BT,
            mask=t_mask & in_prefix,
            other=-1,
        )
        live = in_prefix & (blk >= 0)
        off = (
            tl.maximum(blk, 0).to(tl.int64) * stride_p_slot
            + (pj_safe % BT) * stride_p_row
        )
        pv = tl.load(
            pbase + off[:, None] + pcol[None, :] * stride_p_c,
            mask=td_mask & live[:, None],
            other=0.0,  # float literal: fp8 pools cannot cast an int fill
        )
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        acc += (cv.to(tl.float32) + pv.to(tl.float32)) * w_val[None, :]

    xv = tl.load(
        x_ptr + t_off[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    xv_f32 = xv.to(tl.float32)
    w_last = _select_weight_tap(w_all, W - 1, W_POW2)
    acc += xv_f32 * w_last[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if USE_RESIDUAL:
        acc += xv_f32

    tl.store(
        y_ptr + t_off[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask,
    )

    # Restores land on LCM_ALIGN boundaries: only the last W-1 columns per window/chunk are read.
    chunk_len = tl.load(cu_seqlens_ptr + req + 1, mask=t_mask, other=0) - chunk_start
    end_pos = prefix + chunk_len
    persist = t_mask
    if LCM_ALIGN > 0:
        tail_of_window = pos % LCM_ALIGN >= LCM_ALIGN - (W - 1)
        tail_of_chunk = pos >= end_pos - (W - 1)
        persist = persist & (tail_of_window | tail_of_chunk)
    blk_cur = tl.load(
        page_table_ptr + req * stride_pt_b + pos // BT, mask=persist, other=-1
    )
    live_cur = blk_cur >= 0
    off_cur = (
        tl.maximum(blk_cur, 0).to(tl.int64) * stride_p_slot + (pos % BT) * stride_p_row
    )
    tl.store(
        pbase + off_cur[:, None] + pcol[None, :] * stride_p_c,
        xv.to(pool_ptr.dtype.element_ty),
        mask=(persist[:, None] & d_mask[None, :]) & live_cur[:, None],
    )
