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

Kernels backing the public API in :mod:`tokenspeed_kernel.ops.conv`. The two
conv kernels are split by forward family -- each hard-codes its state source
so neither carries the other's paths. All address the conv checkpoint cache through a
two-slot block table per request: slot 0 is the input checkpoint (the conv
window at the chunk start: a prefix hit, or the previous chunk's window the
scheduler rotated there; a hole for the zero state), slot 1 the live block
that receives the window at the chunk end.

- ``_inkling_ring_sconv_prefill_kernel``: varlen extend chunks. Pre-chunk
  conv taps read the checkpoint in slot 0 (zeros when the request has no
  prefix); the ring is never read. Persists the last
  ``min(chunk_len + (W-1), R)`` positions to the ring -- the chunk rows plus
  the checkpoint window itself, one writer per row via the
  ``pos >= through - R`` gate -- and publishes the chunk-end window to slot 1.
- ``_inkling_ring_sconv_decode_kernel``: fixed-shape decode/verify
  chunks (uniform tokens per request). Pre-chunk taps read the RING; the
  checkpoint cache is never read. Persists every chunk row (the wrapper
  asserts the uniform chunk fits ``R - (W - 1)``, so writes never alias
  another request's taps) and, when ``PUBLISH`` is set, publishes the
  chunk-end window to slot 1.
- ``_inkling_ring_commit_kernel``: after a target verify, publishes the
  window at each request's accepted frontier from the ring to slot 1, for
  every layer's streams in one launch (a verify's decode kernel publishes
  nothing: the frontier is known only after acceptance).

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


@triton.jit
def _load_input(ptrs, mask, stride_x_s, SLOTS: tl.constexpr):
    """Load input rows in their dtype. With ``SLOTS > 1`` each row is that
    many slot rows ``stride_x_s`` apart (MoE route rows), added in FP32 slot
    order from +0.0 and rounded once."""
    if SLOTS == 1:
        return tl.load(ptrs, mask=mask, other=0)
    acc = tl.zeros(ptrs.shape, dtype=tl.float32)
    for slot in tl.static_range(SLOTS):
        acc += tl.load(ptrs + slot * stride_x_s, mask=mask, other=0).to(tl.float32)
    return acc.to(ptrs.dtype.element_ty)


@triton.jit
def _await_mailbox_sum(ptrs, mask, stride_x_s, SLOTS: tl.constexpr):
    """``_load_input`` of BF16 Lamport mailbox slots (``ops.communication.mailbox``):
    the slots re-load (all at once) until no masked value is the armed
    ``-0.0``, which producers never store; the last pass's rows add in FP32
    slot order and round once. The loads bypass L1."""
    pending = 1
    acc = tl.zeros(ptrs.shape, dtype=tl.float32)
    while pending != 0:
        armed = tl.zeros(ptrs.shape, dtype=tl.int1)
        acc = tl.zeros(ptrs.shape, dtype=tl.float32)
        for slot in tl.static_range(SLOTS):
            bits_ptrs = (ptrs + slot * stride_x_s).to(tl.pointer_type(tl.int16))
            bits = tl.load(bits_ptrs, mask=mask, other=0, volatile=True)
            armed = armed | (bits == -32768)
            acc += bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        # One block-wide reduction a pass.
        pending = tl.max(armed.to(tl.int32))
    return acc.to(ptrs.dtype.element_ty)


@triton.jit
def _load_rows(ptrs, mask, stride_x_s, SLOTS: tl.constexpr, MAILBOX: tl.constexpr):
    if MAILBOX:
        # Slot 0 holds the program's summed rows (written by the program);
        # L1 may still hold the slot's previous use, so the load skips it.
        rows = tl.load(ptrs, mask=mask, other=0, cache_modifier=".cg")
    else:
        rows = _load_input(ptrs, mask, stride_x_s, SLOTS)
    return rows


# -----------------------------------------------------------------------------
# The sconv compute kernels: ring-addressed state, split by forward family
# -----------------------------------------------------------------------------


@triton.jit
def _publish_window_row(
    ckpt_a_ptr,
    ckpt_b_ptr,
    page,
    row: tl.constexpr,
    d_off,
    value,
    mask,
    a_width,
    stride_a_page,
    stride_a_w,
    stride_a_c,
    stride_b_page,
    stride_b_w,
    stride_b_c,
):
    """Store ``value`` as row ``row`` of the published conv window: columns
    below ``a_width`` go to the first checkpoint field, the rest to the
    second (fused K+V)."""
    use_a = d_off < a_width
    col_b = tl.maximum(d_off - a_width, 0)
    tl.store(
        ckpt_a_ptr
        + page[:, None] * stride_a_page
        + row * stride_a_w
        + d_off[None, :] * stride_a_c,
        value.to(ckpt_a_ptr.dtype.element_ty),
        mask=mask & use_a[None, :],
    )
    tl.store(
        ckpt_b_ptr
        + page[:, None] * stride_b_page
        + row * stride_b_w
        + col_b[None, :] * stride_b_c,
        value.to(ckpt_b_ptr.dtype.element_ty),
        mask=mask & (d_off >= a_width)[None, :],
    )


@triton.jit
def _inkling_ring_sconv_decode_kernel(
    x_ptr,  # [T, D], or [T, SLOTS, D] rows folded on load
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, R, D] ring
    cu_seqlens_ptr,  # [B+1] int32
    seq_idx_ptr,  # [T] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    seq_lens_ptr,  # [B] int32 lengths THROUGH the chunk
    block_table_ptr,  # [B, 2] int32: (input checkpoint, live block)
    ckpt_a_ptr,  # [pages, W-1, a_width] checkpoint field
    ckpt_b_ptr,  # second field for d >= a_width (fused K+V), or ckpt_a
    dynamic_ptr,
    stride_dyn_t,
    stride_dyn_h,
    stride_dyn_w,
    output_scale,
    stride_x_t,
    stride_x_s,
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
    T,
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    PUBLISH: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    HAS_DYNAMIC: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOTS: tl.constexpr,
    MAILBOX: tl.constexpr,
):
    """Decode/verify sconv: ring taps, write-all, chunk-end publish.

    Chunks are uniform per request (1 / K tokens) and fit
    ``R - (W - 1)``, so every chunk row persists at its position's ring row
    with no aliasing (the wrapper asserts the bound). Pre-chunk taps read
    the request's ring; the checkpoint cache is never read. With ``PUBLISH``
    the request's last token writes the conv window ending at ``through`` to
    the live block -- the committed window of a one-token decode, or of a
    draft chunk re-anchored at the committed frontier.
    """
    t_off = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    # Predecessors wait before they trigger, so these older inputs load pre-wait.
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

    # The last token publishes [through - (W-1), through); pad rows have stale lengths.
    pub = t_mask & (ci != -1) & (t_off == eos - 1)
    page = 0
    if PUBLISH:
        page = tl.load(
            block_table_ptr + si * stride_bt_b + stride_bt_c, mask=pub, other=0
        ).to(tl.int64)
        pub = pub & (page > 0)

    if not HAS_DYNAMIC:
        # Ring taps precede x taps in tap order, so their FMA chain part runs pre-wait.
        for iw in tl.static_range(W):
            lag = W - 1 - iw
            shifted_t = t64 - lag
            in_x = (shifted_t >= bos) & t_mask
            tap_pos = pre_len + (shifted_t - bos).to(tl.int32)
            in_ring = (shifted_t < bos) & (tap_pos >= 0) & use_cache & t_mask
            tap_row = (tl.maximum(tap_pos, 0) % R).to(tl.int64)
            p_val = tl.load(
                cache_base + tap_row[:, None] * stride_c_w,
                mask=in_ring[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            w_val = _select_weight_tap(w_all, iw, W_POW2)
            coefficient = w_val.to(x_ptr.dtype.element_ty).to(tl.float32)[None, :]
            acc = tl.where(in_x[:, None], acc, tl.fma(p_val, coefficient, acc))
            if PUBLISH and lag <= W - 2:
                _publish_window_row(
                    ckpt_a_ptr,
                    ckpt_b_ptr,
                    page,
                    W - 2 - lag,
                    d_off,
                    p_val,
                    (pub & ~in_x)[:, None] & d_mask[None, :],
                    a_width,
                    stride_a_page,
                    stride_a_w,
                    stride_a_c,
                    stride_b_page,
                    stride_b_w,
                    stride_b_c,
                )

    if ENABLE_PDL:
        # Dependents read only constants before their own wait, so release them first.
        tl.extra.cuda.gdc_launch_dependents()
        if not MAILBOX:
            tl.extra.cuda.gdc_wait()

    x_own = tl.zeros([BLOCK_T, BLOCK_D], dtype=x_ptr.dtype.element_ty)
    if MAILBOX:
        # The program's rows, summed once; the taps below read them back from
        # slot 0, which only this program reads (it holds every token).
        own_ptrs = x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d
        x_own = _await_mailbox_sum(own_ptrs, td_mask, stride_x_s, SLOTS)
        tl.store(own_ptrs, x_own, mask=td_mask)
        tl.debug_barrier()
        # Only slot 0 is read from here on: re-arm the others now.
        for slot in tl.static_range(1, SLOTS):
            tl.store(
                (own_ptrs + slot * stride_x_s).to(tl.pointer_type(tl.int16)),
                tl.full([BLOCK_T, BLOCK_D], -32768, tl.int16),
                mask=td_mask,
            )

    # Keep tap W-1 in-loop: a split unconditional load miscompiles tokenspeed_triton (num_warps>1).
    for iw in tl.static_range(W):
        lag = iw if HAS_DYNAMIC else W - 1 - iw
        shifted_t = t64 - lag
        in_x = (shifted_t >= bos) & t_mask
        x_val = _load_rows(
            x_ptr + shifted_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            in_x[:, None] & d_mask[None, :],
            stride_x_s,
            SLOTS,
            MAILBOX,
        ).to(tl.float32)
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        if HAS_DYNAMIC:
            # Dynamic taps run lag 0 first, so the ring is read here as well.
            tap_pos = pre_len + (shifted_t - bos).to(tl.int32)
            in_ring = (shifted_t < bos) & (tap_pos >= 0) & use_cache & t_mask
            tap_row = (tl.maximum(tap_pos, 0) % R).to(tl.int64)
            p_val = tl.load(
                cache_base + tap_row[:, None] * stride_c_w,
                mask=in_ring[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            v = tl.where(in_x[:, None], x_val, p_val)
            dynamic = (
                tl.load(
                    dynamic_ptr
                    + t64[:, None] * stride_dyn_t
                    + (d_off[None, :] // HEAD_DIM) * stride_dyn_h
                    + iw * stride_dyn_w,
                    mask=td_mask,
                    other=0,
                )
                .to(x_ptr.dtype.element_ty)
                .to(tl.float32)
            )
            coefficient = w_val[None, :] + dynamic
            if iw == 0:
                acc = v * coefficient
            else:
                acc = tl.fma(v, coefficient, acc)
            if PUBLISH and lag <= W - 2:
                _publish_window_row(
                    ckpt_a_ptr,
                    ckpt_b_ptr,
                    page,
                    W - 2 - lag,
                    d_off,
                    v,
                    pub[:, None] & d_mask[None, :],
                    a_width,
                    stride_a_page,
                    stride_a_w,
                    stride_a_c,
                    stride_b_page,
                    stride_b_w,
                    stride_b_c,
                )
        else:
            coefficient = w_val.to(x_ptr.dtype.element_ty).to(tl.float32)[None, :]
            acc = tl.where(in_x[:, None], tl.fma(x_val, coefficient, acc), acc)
            if PUBLISH and lag <= W - 2:
                _publish_window_row(
                    ckpt_a_ptr,
                    ckpt_b_ptr,
                    page,
                    W - 2 - lag,
                    d_off,
                    x_val,
                    (pub & in_x)[:, None] & d_mask[None, :],
                    a_width,
                    stride_a_page,
                    stride_a_w,
                    stride_a_c,
                    stride_b_page,
                    stride_b_w,
                    stride_b_c,
                )

    if MAILBOX:
        # The program holds every token of its channels, so it alone read
        # slot 0's rows: re-arm them once every thread has.
        tl.debug_barrier()
        tl.store(
            (x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d).to(
                tl.pointer_type(tl.int16)
            ),
            tl.full([BLOCK_T, BLOCK_D], -32768, tl.int16),
            mask=td_mask,
        )

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    if MAILBOX:
        xv = x_own
    else:
        xv = _load_input(
            x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            td_mask,
            stride_x_s,
            SLOTS,
        )
    if USE_RESIDUAL:
        acc += xv.to(tl.float32)
    acc = acc * output_scale
    if MAILBOX and ENABLE_PDL:
        # The mailbox signals its own rows; only the stores below wait.
        tl.extra.cuda.gdc_wait()

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


@triton.jit
def _prefill_edges(
    req,
    d_off,
    x_ptr,  # [T, D], or [T, SLOTS, D] rows folded on load
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, R, D] ring
    cu_seqlens_ptr,  # [B+1] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    seq_lens_ptr,  # [B] int32 lengths THROUGH the chunk
    block_table_ptr,  # [B, 2] int32: (input checkpoint, live block)
    ckpt_a_ptr,  # [pages, W-1, a_width] checkpoint field
    ckpt_b_ptr,  # second field for d >= a_width (fused K+V), or ckpt_a
    dynamic_ptr,
    stride_dyn_t,
    stride_dyn_h,
    stride_dyn_w,
    output_scale,
    stride_x_t,
    stride_x_s,
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
    D,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    PUBLISH: tl.constexpr,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    HAS_DYNAMIC: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """One request's chunk-start rows, checkpoint restore and publish.

    For a request with an input checkpoint (block-table slot 0), the first
    ``W - 1`` chunk rows take their pre-chunk taps from the checkpoint
    window, with the token tiles' per-element arithmetic, and that window is
    restored into the ring so later decode rounds can rewind past the chunk
    start (gated to the last R positions, so each ring row has one writer).
    With ``PUBLISH``, the window ending at ``through`` goes to the live block
    in slot 1; its rows come from ``x`` or, for a chunk shorter than
    ``W - 1``, from the checkpoint.
    """
    d_mask = d_off < D
    bos = tl.load(cu_seqlens_ptr + req).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + req + 1).to(tl.int64)
    through = tl.load(seq_lens_ptr + req).to(tl.int64)
    ci = tl.load(cache_indices_ptr + req)
    has_state = tl.load(has_initial_state_ptr + req)
    # PAD_SLOT_ID = -1 (can't reference a Python global inside @jit)
    use_cache = (ci != -1) & (has_state != 0)
    safe_ci = tl.maximum(ci, 0).to(tl.int64)
    pre_len = through - (eos - bos)  # chunk-start position (0 = cold)
    page0 = tl.load(block_table_ptr + req * stride_bt_b, mask=use_cache, other=0).to(
        tl.int64
    )
    ckpt_ok = use_cache & (pre_len > 0) & (page0 > 0)
    use_a = d_off < a_width
    col_b = tl.maximum(d_off - a_width, 0)
    cache_base = conv_cache_ptr + safe_ci * stride_c_slot + d_off[None, :] * stride_c_d

    if ckpt_ok:
        # Rows bos .. bos + W - 2 of the chunk, as a [W_POW2, BLOCK_D] tile.
        r_off = tl.arange(0, W_POW2)
        t64 = bos + r_off
        t_mask = (r_off < W - 1) & (t64 < eos)
        td_mask = t_mask[:, None] & d_mask[None, :]
        w_all = _load_weight_taps(
            weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
        )
        acc = tl.zeros([W_POW2, BLOCK_D], dtype=tl.float32)
        for iw in tl.static_range(W):
            lag = iw if HAS_DYNAMIC else W - 1 - iw
            shifted_t = t64 - lag
            in_x = (shifted_t >= bos) & t_mask
            x_val = _load_input(
                x_ptr + shifted_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
                in_x[:, None] & d_mask[None, :],
                stride_x_s,
                SLOTS,
            )
            # Pre-chunk taps: checkpoint window row j. No `other` filler (an
            # int cannot cast to fp8 sources); masked-off lanes are discarded
            # by the selects below.
            j = (shifted_t - bos + (W - 1)).to(tl.int64)
            in_ck = (shifted_t < bos) & t_mask
            a_val = tl.load(
                ckpt_a_ptr
                + page0 * stride_a_page
                + j[:, None] * stride_a_w
                + d_off[None, :] * stride_a_c,
                mask=in_ck[:, None] & d_mask[None, :] & use_a[None, :],
            )
            b_val = tl.load(
                ckpt_b_ptr
                + page0 * stride_b_page
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
            if HAS_DYNAMIC:
                dynamic = (
                    tl.load(
                        dynamic_ptr
                        + t64[:, None] * stride_dyn_t
                        + (d_off[None, :] // HEAD_DIM) * stride_dyn_h
                        + iw * stride_dyn_w,
                        mask=td_mask,
                        other=0,
                    )
                    .to(x_ptr.dtype.element_ty)
                    .to(tl.float32)
                )
                coefficient = w_val[None, :] + dynamic
            else:
                coefficient = w_val.to(x_ptr.dtype.element_ty).to(tl.float32)[None, :]
            if HAS_DYNAMIC and iw == 0:
                acc = v * coefficient
            else:
                acc = tl.fma(v, coefficient, acc)
            if lag > 0:
                # Fused restore: the request's first-token lane materializes
                # the checkpoint window into the ring so later decode rounds
                # can rewind past the chunk start. Gated to the last R
                # positions -- one writer per row beside the chunk stores.
                tap_pos = pre_len - lag
                do_restore = (
                    (r_off == 0) & in_ck & (tap_pos >= through - R) & (tap_pos >= 0)
                )
                restore_row = tl.maximum(tap_pos, 0) % R
                tl.store(
                    cache_base + restore_row * stride_c_w + r_off[:, None] * 0,
                    ck_val.to(conv_cache_ptr.dtype.element_ty),
                    mask=do_restore[:, None] & d_mask[None, :],
                )

        if USE_SILU:
            acc = acc * tl.sigmoid(acc)
        xv = _load_input(
            x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            td_mask,
            stride_x_s,
            SLOTS,
        )
        if USE_RESIDUAL:
            acc += xv.to(tl.float32)
        acc = acc * output_scale
        tl.store(
            y_ptr + t64[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
            acc.to(y_ptr.dtype.element_ty),
            mask=td_mask,
        )

    if PUBLISH:
        # The request's last token publishes the window [through - (W-1),
        # through) to the live block; rows before the chunk start come from
        # the input checkpoint (a chunk shorter than W-1 tokens).
        page = tl.load(
            block_table_ptr + req * stride_bt_b + stride_bt_c,
            mask=ci != -1,
            other=0,
        ).to(tl.int64)
        if (ci != -1) & (page > 0) & (eos > bos):
            last = eos - 1
            for w in tl.static_range(W - 1):
                src_t = last - (W - 2) + w
                in_x2 = src_t >= bos
                x_row = _load_input(
                    x_ptr + src_t * stride_x_t + d_off * stride_x_d,
                    in_x2 & d_mask,
                    stride_x_s,
                    SLOTS,
                )
                j2 = (src_t - bos + (W - 1)).to(tl.int64)
                in_ck2 = (src_t < bos) & ckpt_ok
                a_row = tl.load(
                    ckpt_a_ptr
                    + page0 * stride_a_page
                    + j2 * stride_a_w
                    + d_off * stride_a_c,
                    mask=in_ck2 & d_mask & use_a,
                )
                b_row = tl.load(
                    ckpt_b_ptr
                    + page0 * stride_b_page
                    + j2 * stride_b_w
                    + col_b * stride_b_c,
                    mask=in_ck2 & d_mask & (d_off >= a_width),
                )
                ck_row = tl.where(use_a, a_row.to(tl.float32), b_row.to(tl.float32))
                ck_row = tl.where(in_ck2 & d_mask, ck_row, 0.0)
                val = tl.where(in_x2, x_row.to(tl.float32), ck_row)
                tl.store(
                    ckpt_a_ptr
                    + page * stride_a_page
                    + w * stride_a_w
                    + d_off * stride_a_c,
                    val.to(ckpt_a_ptr.dtype.element_ty),
                    mask=d_mask & use_a,
                )
                tl.store(
                    ckpt_b_ptr
                    + page * stride_b_page
                    + w * stride_b_w
                    + col_b * stride_b_c,
                    val.to(ckpt_b_ptr.dtype.element_ty),
                    mask=d_mask & (d_off >= a_width),
                )


@triton.jit(do_not_specialize=["tile_lo", "n_tiles", "n_tail_programs"])
def _inkling_ring_sconv_prefill_kernel(
    x_ptr,  # [T, D], or [T, SLOTS, D] rows folded on load
    weight_ptr,  # [D, W]
    conv_cache_ptr,  # [num_slots, R, D] ring
    cu_seqlens_ptr,  # [B+1] int32
    seq_idx_ptr,  # [T] int32
    cache_indices_ptr,  # [B] int32
    has_initial_state_ptr,  # [B] bool
    y_ptr,  # [T, D]
    seq_lens_ptr,  # [B] int32 lengths THROUGH the chunk
    block_table_ptr,  # [B, 2] int32: (input checkpoint, live block)
    ckpt_a_ptr,  # [pages, W-1, a_width] checkpoint field
    ckpt_b_ptr,  # second field for d >= a_width (fused K+V), or ckpt_a
    dynamic_ptr,
    stride_dyn_t,
    stride_dyn_h,
    stride_dyn_w,
    output_scale,
    stride_x_t,
    stride_x_s,
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
    T,
    D,
    tile_lo,
    n_tiles,
    n_tail_programs,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    PUBLISH: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    HAS_DYNAMIC: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """Prefill/extend sconv: chunk rows, ring write, checkpoint taps, publish.

    Program rows past the token tiles are edge programs, one per request
    (:func:`_prefill_edges`): they compute the chunk-start rows of a request
    with an input checkpoint, restore its window into the ring and publish.
    Token tiles compute all other rows, with pre-chunk taps reading zero (a
    cold start's value), and persist the last ``min(chunk_len, R)`` chunk
    rows to the ring. Keeping the checkpoint work in small edge tiles keeps
    the token tiles lean; the two never write the same row.

    Token tiles are ``n_tiles`` tiles from ``tile_lo``, then
    ``n_tail_programs`` tail programs: each request's ``TAIL_TILES`` tiles
    around its last ``R`` rows that the first range does not hold, so a
    sequence-parallel rank that computes only its share still writes every
    request's ring.
    """
    d_off = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    if pid >= n_tiles + n_tail_programs:
        _prefill_edges(
            pid - n_tiles - n_tail_programs,
            d_off,
            x_ptr,
            weight_ptr,
            conv_cache_ptr,
            cu_seqlens_ptr,
            cache_indices_ptr,
            has_initial_state_ptr,
            y_ptr,
            seq_lens_ptr,
            block_table_ptr,
            ckpt_a_ptr,
            ckpt_b_ptr,
            dynamic_ptr,
            stride_dyn_t,
            stride_dyn_h,
            stride_dyn_w,
            output_scale,
            stride_x_t,
            stride_x_s,
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
            D,
            USE_SILU,
            USE_RESIDUAL,
            PUBLISH,
            R,
            BLOCK_D,
            W,
            W_POW2,
            HAS_DYNAMIC,
            HEAD_DIM,
            SLOTS,
        )
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    if pid < n_tiles:
        tile = tile_lo + pid
    else:
        TAIL_TILES: tl.constexpr = (R + BLOCK_T - 1) // BLOCK_T + 1
        req = (pid - n_tiles) // TAIL_TILES
        req_bos = tl.load(cu_seqlens_ptr + req)
        req_eos = tl.load(cu_seqlens_ptr + req + 1)
        tile = (
            tl.maximum(req_eos - R, req_bos) // BLOCK_T + (pid - n_tiles) % TAIL_TILES
        )
        if (tile * BLOCK_T >= req_eos) | (
            (tile >= tile_lo) & (tile < tile_lo + n_tiles)
        ):
            if ENABLE_PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    t_off = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < T
    d_mask = d_off < D
    td_mask = t_mask[:, None] & d_mask[None, :]

    si = tl.load(seq_idx_ptr + t_off, mask=t_mask, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens_ptr + si, mask=t_mask, other=0).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + si + 1, mask=t_mask, other=0).to(tl.int32)
    through = tl.load(seq_lens_ptr + si, mask=t_mask, other=0).to(tl.int32)
    ci = tl.load(cache_indices_ptr + si, mask=t_mask, other=-1)
    has_state = tl.load(has_initial_state_ptr + si, mask=t_mask, other=0)
    # PAD_SLOT_ID = -1 (can't reference a Python global inside @jit)
    use_cache = (ci != -1) & (has_state != 0)
    # The input checkpoint in slot 0: the tap (and restore) source.
    page0 = tl.load(
        block_table_ptr + si * stride_bt_b, mask=use_cache & t_mask, other=0
    )
    pre_len = through - (eos - bos.to(tl.int32))

    w_all = _load_weight_taps(
        weight_ptr, d_off, d_mask, stride_w_d, stride_w_w, W, W_POW2
    )
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    t64 = t_off.to(tl.int64)
    HEADS: tl.constexpr = max(BLOCK_D // HEAD_DIM, 1)
    # Keep tap W-1 in-loop: a split unconditional load miscompiles tokenspeed_triton (num_warps>1).
    for iw in tl.static_range(W):
        lag = iw if HAS_DYNAMIC else W - 1 - iw
        shifted_t = t64 - lag
        tap_mask = ((shifted_t >= bos) & t_mask)[:, None] & d_mask[None, :]
        x_val = _load_input(
            x_ptr + shifted_t[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
            tap_mask,
            stride_x_s,
            SLOTS,
        )
        w_val = _select_weight_tap(w_all, iw, W_POW2)
        v = x_val.to(tl.float32)
        if HAS_DYNAMIC and BLOCK_D % HEAD_DIM == 0:
            # One coefficient per (token, head), broadcast over its channels.
            h_off = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
            per_head = (
                tl.load(
                    dynamic_ptr
                    + t64[:, None] * stride_dyn_t
                    + h_off[None, :] * stride_dyn_h
                    + iw * stride_dyn_w,
                    mask=t_mask[:, None] & (h_off * HEAD_DIM < D)[None, :],
                    other=0,
                )
                .to(x_ptr.dtype.element_ty)
                .to(tl.float32)
            )
            dynamic = tl.reshape(
                tl.broadcast_to(per_head[:, :, None], (BLOCK_T, HEADS, HEAD_DIM)),
                (BLOCK_T, BLOCK_D),
            )
            coefficient = w_val[None, :] + dynamic
        elif HAS_DYNAMIC:
            dynamic = (
                tl.load(
                    dynamic_ptr
                    + t64[:, None] * stride_dyn_t
                    + (d_off[None, :] // HEAD_DIM) * stride_dyn_h
                    + iw * stride_dyn_w,
                    mask=td_mask,
                    other=0,
                )
                .to(x_ptr.dtype.element_ty)
                .to(tl.float32)
            )
            coefficient = w_val[None, :] + dynamic
        else:
            coefficient = w_val.to(x_ptr.dtype.element_ty).to(tl.float32)[None, :]
        if HAS_DYNAMIC and iw == 0:
            acc = v * coefficient
        else:
            acc = tl.fma(v, coefficient, acc)

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)

    xv = _load_input(
        x_ptr + t64[:, None] * stride_x_t + d_off[None, :] * stride_x_d,
        td_mask,
        stride_x_s,
        SLOTS,
    )
    if USE_RESIDUAL:
        acc += xv.to(tl.float32)
    acc = acc * output_scale

    # Rows an edge program computes from the checkpoint are left to it.
    edge_row = use_cache & (pre_len > 0) & (page0 > 0) & (t64 - bos < W - 1)
    tl.store(
        y_ptr + t64[:, None] * stride_y_t + d_off[None, :] * stride_y_d,
        acc.to(y_ptr.dtype.element_ty),
        mask=td_mask & ~edge_row[:, None],
    )

    # Ring persistence, gated to the last R positions: one writer per row
    # regardless of chunk length (no epilogue, no aliasing -- this kernel
    # never reads the ring).
    pos = through - eos + t_off  # 0-based absolute position of token t
    do_write = t_mask & (ci != -1) & (pos >= through - R)
    # Tile-uniform: only tiles holding such a row store.
    if tl.max(do_write.to(tl.int32), axis=0) > 0:
        safe_ci = tl.maximum(ci, 0).to(tl.int64)
        ring_row = (tl.maximum(pos, 0) % R).to(tl.int64)
        tl.store(
            conv_cache_ptr
            + safe_ci[:, None] * stride_c_slot
            + ring_row[:, None] * stride_c_w
            + d_off[None, :] * stride_c_d,
            xv.to(conv_cache_ptr.dtype.element_ty),
            mask=do_write[:, None] & d_mask[None, :],
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["stride_live_g", "stride_live_r"])
def _inkling_ring_commit_kernel(
    ring_ptr,  # [L, num_slots, R, C] conv rings of every layer
    src_offsets_ptr,  # int64 [E]: entry's first channel, layer * stride_l + channel
    widths_ptr,  # int32 [E]: channels the entry's field holds
    ckpt_addresses_ptr,  # uint64 [E]: the field's page-0 address
    ckpt_page_strides_ptr,  # int64 [E], elements
    ckpt_row_strides_ptr,  # int64 [E], elements
    groups_ptr,  # int32 [E]: the entry's row in live_pages
    live_pages_ptr,  # int32 [G, >= B]: live block per group and request (strided)
    cache_indices_ptr,  # int32 [B]: ring slot, PAD_SLOT_ID for padded rows
    seq_lens_ptr,  # [B]: lengths through the verify chunk
    accepted_ptr,  # [B]: accepted tokens of the verify chunk
    stride_ring_slot,
    stride_ring_row,
    stride_live_g,
    stride_live_r,
    CKPT_DTYPE: tl.constexpr,
    STATE_ROWS: tl.constexpr,  # W - 1
    R: tl.constexpr,
    VERIFY_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Publish one entry's accepted window of one request to its live block.

    The window is the ``STATE_ROWS`` ring rows ending at the committed
    frontier ``seq_lens - VERIFY_WIDTH + clamp(accepted, 1, VERIFY_WIDTH)``;
    positions before 0 are zeros. Padded rows and hole live blocks store
    nothing.
    """
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    entry = tl.program_id(0)
    req = tl.program_id(1)
    cols = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < tl.load(widths_ptr + entry)
    slot = tl.load(cache_indices_ptr + req).to(tl.int64)
    group = tl.load(groups_ptr + entry).to(tl.int64)
    page = tl.load(live_pages_ptr + group * stride_live_g + req * stride_live_r).to(
        tl.int64
    )
    if (slot >= 0) & (page > 0):
        steps = tl.load(accepted_ptr + req).to(tl.int64)
        steps = tl.minimum(tl.maximum(steps, 1), VERIFY_WIDTH)
        frontier = tl.load(seq_lens_ptr + req).to(tl.int64) - VERIFY_WIDTH + steps
        src = (
            ring_ptr
            + slot * stride_ring_slot
            + tl.load(src_offsets_ptr + entry)
            + cols.to(tl.int64)
        )
        dst = (
            tl.cast(tl.load(ckpt_addresses_ptr + entry), tl.pointer_type(CKPT_DTYPE))
            + page * tl.load(ckpt_page_strides_ptr + entry)
            + cols.to(tl.int64)
        )
        row_stride = tl.load(ckpt_row_strides_ptr + entry)
        for w in tl.static_range(STATE_ROWS):
            pos = frontier - STATE_ROWS + w
            row = (tl.maximum(pos, 0) % R).to(tl.int64)
            val = tl.load(
                src + row * stride_ring_row, mask=col_mask & (pos >= 0), other=0
            )
            tl.store(dst + w * row_stride, val.to(CKPT_DTYPE), mask=col_mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def select_prefill_config(T: int, D: int) -> tuple[int, int, int, int]:
    """Select ``(BLOCK_T, BLOCK_D, num_warps, num_stages)`` for prefill.

    Static heuristic (no autotune) so the kernel stays CUDA-graph friendly;
    every tile sums an element's taps alike. A token-count tier would
    compile a second variant at the first long prefill. Swept on B300 (CUDA
    graphs) over T in {64..8192} and D in {256..4608}: 16x128 tiles with 4
    warps are within 3% of the best tile everywhere (D 4608 dynamic: T 2048
    26 -> 18 us, T 8192 128 -> 70 us against 8x512).

    Args:
        T: Total number of packed tokens.
        D: Number of channels.

    Returns:
        Tuple ``(BLOCK_T, BLOCK_D, num_warps, num_stages)``.
    """
    del T, D
    return 16, 128, 4, 2


def select_decode_config(T: int, D: int, dynamic: bool) -> tuple[int, int, int, int]:
    """Select ``(BLOCK_T, BLOCK_D, num_warps, num_stages)`` for decode and
    verify chunks.

    A verify round holds a few rows per request, so narrow channel tiles
    spread it over more programs. On B300 under CUDA graphs (8 rows per
    request), a PDL chain of a producer that triggers at its end, the
    convolution and a consumer that waits first takes 3.2 / 3.7 us per step
    with 8x64 tiles for static taps at D 4608 and 1 or 4 requests, against
    4.1 / 4.6 us with 8x128 (D 1152: 3.1 / 3.2 against 3.3 / 4.1 us); up to
    32 rows, 4x64 tiles take 0.3 / 0.2 us less than 8x64 at D 4608, and 0.5
    us more at 80 rows. Dynamic taps, which also load a coefficient per
    head, keep 8x128 (D 1024: 2.9 / 3.3 against 3.3 / 3.4 us). Every tile
    sums an element's taps alike; the two row tiles are the only variants
    the batch size selects.

    Args:
        T: Rows in the batch.
        D: Number of channels.
        dynamic: Whether the convolution has dynamic coefficients.

    Returns:
        Tuple ``(BLOCK_T, BLOCK_D, num_warps, num_stages)``.
    """
    del D
    if dynamic:
        return 8, 128, 4, 2
    return (4, 64, 4, 2) if T <= 32 else (8, 64, 4, 2)
