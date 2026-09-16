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

Two kernels backing the public API in :mod:`tokenspeed_kernel.ops.conv`,
split by forward family — each hard-codes its state source so neither
carries the other's paths:

- ``_inkling_ring_sconv_prefill_kernel``: varlen extend chunks. Pre-chunk
  conv taps read the PAGED checkpoint at the chunk-start boundary (extends
  always start on an aligned boundary; zeros when the request has no
  prefix); the ring is never read. Persists the last
  ``min(chunk_len + (W-1), R)`` positions to the ring — the chunk rows plus
  the checkpoint window itself, one writer per row via the
  ``pos >= through - R`` gate — and publishes the conv window at every
  covered boundary.
- ``_inkling_ring_sconv_decode_kernel``: fixed-shape decode/verify
  chunks (uniform tokens per request). Pre-chunk taps read the RING; the
  paged cache is never read. Persists every chunk row (the wrapper asserts
  the uniform chunk fits ``R - (W - 1)``, so writes never alias another
  request's taps) and publishes the at-most-one covered boundary.

Both publish unconditionally (there is no non-paged mode); the publish body
sits behind a tile-uniform early-exit so tiles covering no boundary skip it.
The ring row of absolute position ``p`` is ``p % R``; positions derive from
the through-chunk ``seq_lens``, there is no stored cursor, and rejected
speculative rows are overwritten when their positions recur.

Both kernels take explicit strides for the conv ring so channel-sliced views
(``ring[:, :, off:off + D]``) work without a copy. ``PAD_SLOT_ID`` (-1) rows
never read from or write to a real slot.

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

__all__ = [
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
# The sconv compute kernels: ring-addressed state, split by forward family
# -----------------------------------------------------------------------------


@triton.jit
def _inkling_ring_sconv_decode_kernel(
    x_ptr,  # [T, D]
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, R, D] ring
    cu_seqlens_ptr,  # [B+1] int32
    seq_idx_ptr,  # [T] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    seq_lens_ptr,  # [B] int32 lengths THROUGH the chunk
    block_table_ptr,  # [B, num_blocks] int32, the stream's conv group
    ckpt_a_ptr,  # [pages, W-1, a_width] checkpoint field
    ckpt_b_ptr,  # second field for d >= a_width (fused K+V), or ckpt_a
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_c_slot,
    stride_c_w,
    stride_c_d,
    stride_bt_b,
    stride_bt_c,
    stride_a_page,
    stride_a_w,
    stride_a_c,
    stride_b_page,
    stride_b_w,
    stride_b_c,
    a_width,
    num_blocks,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """Decode/verify sconv: ring taps, write-all, one boundary.

    Chunks are uniform per request (1 / K tokens) and fit
    ``R - (W - 1)``, so every chunk row persists at its position's ring row
    with no aliasing (the wrapper asserts the bound). Pre-chunk taps read
    the request's ring; the paged cache is never read. A token whose
    absolute length is page-aligned publishes the conv window at that
    boundary's page — accept-independent, healed by overwrite.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    si = tl.load(seq_idx_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens_ptr + si, mask=t_mask, other=0).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + si + 1, mask=t_mask, other=0).to(tl.int32)
    through = tl.load(seq_lens_ptr + si, mask=t_mask, other=0).to(tl.int32)
    ci = tl.load(cache_indices_ptr + si, mask=t_mask, other=-1)
    has_state = tl.load(has_initial_state_ptr + si, mask=t_mask, other=0)
    # PAD_SLOT_ID = -1 (can't reference a Python global inside @jit)
    use_cache = (ci != -1) & (has_state != 0)
    safe_ci = tl.maximum(ci, 0).to(tl.int64)

    bos32 = bos.to(tl.int32)
    chunk_len = eos - bos32
    pre_len = through - chunk_len  # absolute position of chunk row 0

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
        tap_pos = pre_len + (shifted_t - bos).to(tl.int32)
        in_ring = (shifted_t < bos) & (tap_pos >= 0) & use_cache & t_mask
        tap_row = (tl.maximum(tap_pos, 0) % R).to(tl.int64)
        p_val = tl.load(
            cache_base + tap_row[:, None] * stride_c_w,
            mask=in_ring[:, None] & d_mask[None, :],
            other=0,
        )
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        v = tl.where(in_x[:, None], x_val.to(tl.float32), p_val.to(tl.float32))
        acc += v * w_val[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    xv = tl.load(
        x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    if USE_RESIDUAL:
        acc += xv.to(tl.float32)

    tl.store(
        y_ptr + t64[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask,
    )

    # Every chunk row persists (the wrapper asserts the uniform chunk fits
    # R - (W - 1), so writes never alias another request's tap reads).
    pos = through - eos + t_off  # 0-based absolute position of token t
    do_write = t_mask & (ci != -1)
    ring_row = (tl.maximum(pos, 0) % R).to(tl.int64)
    tl.store(
        cache_base + ring_row[:, None] * stride_c_w,
        xv.to(conv_cache_ptr.dtype.element_ty),
        mask=do_write[:, None] & d_mask[None, :],
    )

    # Absolute length after this token (int32: lengths fit); pad rows
    # carry stale lengths, so gate on the slot BEFORE the alignment test.
    abs_len = pos + 1
    valid_tok = t_mask & (ci != -1)
    is_boundary = valid_tok & (abs_len > 0) & (abs_len % PAGE_SIZE == 0)
    # Boundaries are one token in PAGE_SIZE: skip the whole publish body
    # unless this tile actually holds one (tile-uniform branch).
    if tl.max(is_boundary.to(tl.int32), axis=0) > 0:
        blk = (abs_len // PAGE_SIZE - 1).to(tl.int64)
        blk_ok = is_boundary & (blk >= 0) & (blk < num_blocks)
        page = tl.load(
            block_table_ptr + si * stride_bt_b + blk * stride_bt_c,
            mask=blk_ok,
            other=0,
        ).to(tl.int64)
        pub = blk_ok & (page > 0)
        use_a = d_off < a_width
        col_b = tl.maximum(d_off - a_width, 0)
        for w in tl.static_range(W - 1):
            src_t = t64 - (W - 2) + w
            in_x2 = src_t >= bos
            x_row = tl.load(
                x_ptr + src_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                mask=(pub & in_x2)[:, None] & d_mask[None, :],
                other=0,
            )
            src_pos = pre_len + (src_t - bos).to(tl.int32)
            in_ring2 = (src_t < bos) & (src_pos >= 0) & use_cache
            src_row = (tl.maximum(src_pos, 0) % R).to(tl.int64)
            p_row = tl.load(
                cache_base + src_row[:, None] * stride_c_w,
                mask=(pub & in_ring2)[:, None] & d_mask[None, :],
                other=0,
            )
            val = tl.where(in_x2[:, None], x_row.to(tl.float32), p_row.to(tl.float32))
            tl.store(
                ckpt_a_ptr
                + page[:, None] * stride_a_page
                + w * stride_a_w
                + d_off[None, :] * stride_a_c,
                val.to(ckpt_a_ptr.dtype.element_ty),
                mask=pub[:, None] & d_mask[None, :] & use_a[None, :],
            )
            tl.store(
                ckpt_b_ptr
                + page[:, None] * stride_b_page
                + w * stride_b_w
                + col_b[None, :] * stride_b_c,
                val.to(ckpt_b_ptr.dtype.element_ty),
                mask=pub[:, None] & d_mask[None, :] & (d_off >= a_width)[None, :],
            )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _inkling_ring_sconv_prefill_kernel(
    x_ptr,  # [T, D]
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, R, D] ring
    cu_seqlens_ptr,  # [B+1] int32
    seq_idx_ptr,  # [T] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    seq_lens_ptr,  # [B] int32 lengths THROUGH the chunk
    block_table_ptr,  # [B, num_blocks] int32, the stream's conv group
    ckpt_a_ptr,  # [pages, W-1, a_width] checkpoint field
    ckpt_b_ptr,  # second field for d >= a_width (fused K+V), or ckpt_a
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_c_slot,
    stride_c_w,
    stride_c_d,
    stride_bt_b,
    stride_bt_c,
    stride_a_page,
    stride_a_w,
    stride_a_c,
    stride_b_page,
    stride_b_w,
    stride_b_c,
    a_width,
    num_blocks,
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """Prefill/extend sconv: checkpoint taps, gated write, all boundaries.

    Extend chunks always start on an aligned boundary, so pre-chunk taps
    read the PAGED checkpoint window at ``pre_len`` (zeros when the request
    has no prefix or the entry is a hole); the ring is never read. The last
    ``min(chunk_len + (W - 1), R)`` positions persist to the ring — chunk
    rows from ``x`` plus the checkpoint window itself (materialized by the
    request's first-token lane), one writer per row via the
    ``pos >= through - R`` gate. Every covered boundary publishes its conv
    window; boundaries are >= PAGE_SIZE past the aligned chunk start, so
    publish sources are always in-chunk ``x`` rows.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    si = tl.load(seq_idx_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens_ptr + si, mask=t_mask, other=0).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + si + 1, mask=t_mask, other=0).to(tl.int32)
    through = tl.load(seq_lens_ptr + si, mask=t_mask, other=0).to(tl.int32)
    ci = tl.load(cache_indices_ptr + si, mask=t_mask, other=-1)
    has_state = tl.load(has_initial_state_ptr + si, mask=t_mask, other=0)
    # PAD_SLOT_ID = -1 (can't reference a Python global inside @jit)
    use_cache = (ci != -1) & (has_state != 0)
    safe_ci = tl.maximum(ci, 0).to(tl.int64)

    bos32 = bos.to(tl.int32)
    chunk_len = eos - bos32
    pre_len = through - chunk_len  # aligned chunk-start boundary (0 = cold)

    # The chunk-start boundary's page: the tap (and restore) source.
    blk0 = (pre_len // PAGE_SIZE - 1).to(tl.int64)
    blk0_ok = use_cache & t_mask & (pre_len > 0) & (blk0 >= 0) & (blk0 < num_blocks)
    page0 = tl.load(
        block_table_ptr + si * stride_bt_b + tl.maximum(blk0, 0) * stride_bt_c,
        mask=blk0_ok,
        other=0,
    ).to(tl.int64)
    ckpt_ok = blk0_ok & (page0 > 0)
    use_a = d_off < a_width
    col_b = tl.maximum(d_off - a_width, 0)

    w_all = _load_weight_taps(
        weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
    )
    cache_base = (
        conv_cache_ptr + safe_ci[:, None] * stride_c_slot + d_off[None, :] * stride_c_d
    )

    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    t64 = t_off.to(tl.int64)

    # Only the tile holding a chunk's first W-1 tokens has pre-chunk taps;
    # every other tile takes the pure-x fast path (tile-uniform branch, so
    # the checkpoint machinery costs interior tiles nothing).
    has_prechunk = tl.max((t_mask & ((t64 - bos) < (W - 1))).to(tl.int32), axis=0) > 0
    # Keep tap W-1 in-loop: a split unconditional load miscompiles tokenspeed_triton (num_warps>1).
    if has_prechunk:
        for iw in tl.static_range(W):
            shifted_t = t64 - (W - 1) + iw
            in_x = (shifted_t >= bos) & t_mask
            x_val = tl.load(
                x_ptr + shifted_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                mask=in_x[:, None] & d_mask[None, :],
                other=0,
            )
            # Pre-chunk taps: checkpoint window row j at the chunk-start
            # page. No `other` filler (an int cannot cast to fp8 sources);
            # masked-off lanes are discarded by the selects below.
            j = (shifted_t - bos + (W - 1)).to(tl.int64)
            in_ck = (shifted_t < bos) & ckpt_ok
            a_val = tl.load(
                ckpt_a_ptr
                + page0[:, None] * stride_a_page
                + j[:, None] * stride_a_w
                + d_off[None, :] * stride_a_c,
                mask=in_ck[:, None] & d_mask[None, :] & use_a[None, :],
            )
            b_val = tl.load(
                ckpt_b_ptr
                + page0[:, None] * stride_b_page
                + j[:, None] * stride_b_w
                + col_b[None, :] * stride_b_c,
                mask=in_ck[:, None] & d_mask[None, :] & (d_off >= a_width)[None, :],
            )
            ck_val = tl.where(
                use_a[None, :], a_val.to(tl.float32), b_val.to(tl.float32)
            )
            ck_val = tl.where(in_ck[:, None] & d_mask[None, :], ck_val, 0.0)
            w_val = _select_weight_tap(w_all, iw, W_POW2)
            v = tl.where(in_x[:, None], x_val.to(tl.float32), ck_val)
            acc += v * w_val[None, :]
            if iw < W - 1:
                # Fused restore: the request's first-token lane materializes
                # the checkpoint window into the ring so later decode rounds
                # can rewind past the boundary. Gated to the last R
                # positions — one writer per row beside the chunk stores.
                tap_pos = pre_len - (W - 1) + iw
                do_restore = (
                    (t64 == bos) & in_ck & (tap_pos >= through - R) & (tap_pos >= 0)
                )
                restore_row = (tl.maximum(tap_pos, 0) % R).to(tl.int64)
                tl.store(
                    cache_base + restore_row[:, None] * stride_c_w,
                    ck_val.to(conv_cache_ptr.dtype.element_ty),
                    mask=do_restore[:, None] & d_mask[None, :],
                )
    else:
        # Interior tiles: every tap of every lane is an in-chunk x row.
        for iw in tl.static_range(W):
            shifted_t = t64 - (W - 1) + iw
            x_val = tl.load(
                x_ptr + shifted_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                mask=td_mask,
                other=0,
            )
            w_val = _select_weight_tap(w_all, iw, W_POW2)
            acc += x_val.to(tl.float32) * w_val[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    xv = tl.load(
        x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        mask=td_mask,
        other=0,
    )
    if USE_RESIDUAL:
        acc += xv.to(tl.float32)

    tl.store(
        y_ptr + t64[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask,
    )

    # Ring persistence, gated to the last R positions: one writer per row
    # regardless of chunk length (no epilogue, no aliasing — this kernel
    # never reads the ring).
    pos = through - eos + t_off  # 0-based absolute position of token t
    do_write = t_mask & (ci != -1) & (pos >= through - R)
    ring_row = (tl.maximum(pos, 0) % R).to(tl.int64)
    tl.store(
        cache_base + ring_row[:, None] * stride_c_w,
        xv.to(conv_cache_ptr.dtype.element_ty),
        mask=do_write[:, None] & d_mask[None, :],
    )

    # Publish every covered boundary. Boundaries sit >= PAGE_SIZE past the
    # aligned chunk start, so the W-1 window rows are always in-chunk.
    abs_len = pos + 1
    valid_tok = t_mask & (ci != -1)
    is_boundary = valid_tok & (abs_len > 0) & (abs_len % PAGE_SIZE == 0)
    if tl.max(is_boundary.to(tl.int32), axis=0) > 0:
        blk = (abs_len // PAGE_SIZE - 1).to(tl.int64)
        blk_ok = is_boundary & (blk >= 0) & (blk < num_blocks)
        page = tl.load(
            block_table_ptr + si * stride_bt_b + blk * stride_bt_c,
            mask=blk_ok,
            other=0,
        ).to(tl.int64)
        pub = blk_ok & (page > 0)
        for w in tl.static_range(W - 1):
            src_t = t64 - (W - 2) + w
            x_row = tl.load(
                x_ptr + src_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                mask=pub[:, None] & d_mask[None, :],
                other=0,
            )
            val = x_row.to(tl.float32)
            tl.store(
                ckpt_a_ptr
                + page[:, None] * stride_a_page
                + w * stride_a_w
                + d_off[None, :] * stride_a_c,
                val.to(ckpt_a_ptr.dtype.element_ty),
                mask=pub[:, None] & d_mask[None, :] & use_a[None, :],
            )
            tl.store(
                ckpt_b_ptr
                + page[:, None] * stride_b_page
                + w * stride_b_w
                + col_b[None, :] * stride_b_c,
                val.to(ckpt_b_ptr.dtype.element_ty),
                mask=pub[:, None] & d_mask[None, :] & (d_off >= a_width)[None, :],
            )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


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
