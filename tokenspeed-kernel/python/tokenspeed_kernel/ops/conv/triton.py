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

Four kernels backing the public API in :mod:`tokenspeed_kernel.ops.conv`:

Dense conv-cache family:

- ``_inkling_sconv_kernel``: THE sconv compute kernel — varlen-packed causal
  conv with per-request initial state loaded directly from the conv cache
  (read-only on the cache). Decode is its T=1-per-request case. Optional
  per-token side writes: verify-scratch copy and speculative
  boundary-checkpoint publication (see the kernel docstring).
- ``_sconv_cache_update_kernel``: final-state writeback of the last ``W - 1``
  tokens per request into the conv cache, handling ``query_len < W - 1`` by
  shifting the old cache contents (also serves single-token decode, whose
  fused-shift kernel was merged into the compute kernel).

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
# Prefill: causal conv with per-request initial state (cache is read-only)
# -----------------------------------------------------------------------------


@triton.jit
def _inkling_sconv_kernel(
    x_ptr,  # [T, D]
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, W-1, D]
    cu_seqlens_ptr,  # [B+1] int32
    seq_idx_ptr,  # [T] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    scratch_ptr,  # [>= T, D] verify-scratch view; dummy when !WRITE_SCRATCH
    col_seq_lens_ptr,  # [B] int32 lengths THROUGH the chunk; dummy when !PUBLISH
    block_table_ptr,  # [B, num_blocks] int32; dummy when !PUBLISH
    ckpt_a_ptr,  # [pages, W-1, a_width] checkpoint field; dummy when !PUBLISH
    ckpt_b_ptr,  # second field for d >= a_width (fused K+V), or dummy
    stride_x_t,
    stride_x_d,
    stride_y_t,
    stride_y_d,
    stride_w_d,
    stride_w_w,
    stride_c_slot,
    stride_c_w,
    stride_c_d,
    stride_s_t,
    stride_s_d,
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
    WRITE_SCRATCH: tl.constexpr,
    PUBLISH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
):
    """Depthwise causal conv1d over ``[cached-prefix ++ chunk]`` per request.

    The single sconv compute kernel: decode is the T=1-per-request case
    (uniform ``cu_seqlens``), verify the T=K case, prefill the varlen case.
    It never writes the working state (persistence stays with
    ``_sconv_cache_update_kernel`` or the post-verify select).

    Grid: ``(cdiv(T, BLOCK_T), cdiv(D, BLOCK_D))``. Tiles may straddle
    sequence boundaries, so every conv tap is masked per element: positions
    ``shifted_t >= bos`` read from ``x``; positions before ``bos`` read from
    the request's cache slot (row ``shifted_t - bos + (W - 1)``), zeroed when
    the request has no initial state or its slot is ``PAD_SLOT_ID``.

    Optional per-token side writes:

    - ``WRITE_SCRATCH``: copy every chunk row into the verify scratch
      (padded rows included — replaces the per-stream eager copy).
    - ``PUBLISH``: speculative boundary-checkpoint publication. A token
      whose absolute length ``col_seq_lens[si] - (eos - t) `` is
      page-aligned writes the window rows ``L-W+2..L`` (x rows, or cache
      rows for positions before the chunk) to the stream's checkpoint
      field(s) at that boundary's page — accept-independent; a later round
      covering the same boundary overwrites rejected content. PAD rows,
      out-of-table blocks and hole/pad pages skip the write.
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

    if WRITE_SCRATCH:
        sv = tl.load(
            x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            mask=td_mask,
            other=0,
        )
        tl.store(
            scratch_ptr + t64[:, None] * stride_s_t + d_off[None, :] * stride_s_d,
            sv.to(scratch_ptr.dtype.element_ty),
            mask=td_mask,
        )

    if PUBLISH:
        eos = tl.load(cu_seqlens_ptr + si + 1, mask=t_mask, other=0).to(tl.int32)
        through = tl.load(col_seq_lens_ptr + si, mask=t_mask, other=0).to(tl.int32)
        # Absolute length after this token (int32: lengths fit); pad rows
        # carry stale lengths, so gate on the slot BEFORE the alignment test.
        abs_len = through - eos + t_off + 1
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
                prefix_pos2 = src_t - bos + (W - 1)
                in_prefix2 = (src_t < bos) & (prefix_pos2 >= 0) & use_cache
                p_row = tl.load(
                    cache_base + prefix_pos2[:, None] * stride_c_w,
                    mask=(pub & in_prefix2)[:, None] & d_mask[None, :],
                    other=0,
                )
                val = tl.where(
                    in_x2[:, None], x_row.to(tl.float32), p_row.to(tl.float32)
                )
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
