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

"""The relative-attention head clipping and value norm in one kernel."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops._triton_l2 import prefetch_l2
from tokenspeed_kernel.ops.attention.prologue.types import (
    GQAPrologueOutput,
    HeadKVCache,
    KVCacheFormat,
    KVShortConv,
    RelativeHeadNorm,
)
from tokenspeed_kernel.ops.gemm.triton_skinny import _prefetch_weight_rows
from tokenspeed_kernel.ops.kvcache.triton import (
    _mxfp8_quantize_row,
    _sf_interleaved_offset,
)
from tokenspeed_kernel.ops.layernorm._triton.precision import (
    _l2_clipped,
    _rms_normalized,
)
from tokenspeed_kernel.ops.layernorm.precision import l2_clip, rmsnorm_precision
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit
def _clipped_row(
    X,
    Scale,
    OutputScale,
    token,
    head,
    x_stride,
    threshold,
    eps,
    WIDTH: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    cols = tl.arange(0, WIDTH)
    return _clipped_value(
        tl.load(X + token * x_stride + head * WIDTH + cols).to(tl.float32),
        Scale,
        OutputScale,
        token,
        head,
        threshold,
        eps,
        WIDTH,
        HAS_SCALE,
        HAS_OUTPUT_SCALE,
        OUT_DTYPE,
    )


@triton.jit
def _clipped_value(
    value,
    Scale,
    OutputScale,
    token,
    head,
    threshold,
    eps,
    WIDTH: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # The l2_clip equation for one (token, head) row, before its store.
    cols = tl.arange(0, WIDTH)
    if HAS_SCALE:
        value = value * tl.load(Scale + head * WIDTH + cols).to(tl.float32)
    output = _l2_clipped(value, threshold, eps).to(OUT_DTYPE).to(tl.float32)
    if HAS_OUTPUT_SCALE:
        output = output * tl.load(OutputScale + token).to(tl.float32)
    return output


@triton.jit
def _clip_head(
    X,
    Scale,
    OutputScale,
    Out,
    token,
    head,
    x_stride,
    threshold,
    eps,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
):
    output = _clipped_row(
        X,
        Scale,
        OutputScale,
        token,
        head,
        x_stride,
        threshold,
        eps,
        WIDTH,
        HAS_SCALE,
        HAS_OUTPUT_SCALE,
        Out.dtype.element_ty,
    )
    cols = tl.arange(0, WIDTH)
    tl.store(Out + (token * HEADS + head) * WIDTH + cols, output)


@triton.jit
def _store_kv_row(
    Out,
    Cache,
    ScaleCache,
    Slots,
    slot_count,
    cache_stride,
    sf_page_stride,
    sf_page_tokens,
    value,
    token,
    head,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    STORE: tl.constexpr,
    RETURN_KV: tl.constexpr,
    MXFP8: tl.constexpr,
):
    # The returned row and, under STORE, the cache slot of a written row get
    # the same rounding; rows past the slots (graph padding) are not written.
    # An MXFP8 cache stores that rounded row as quantize_store_kv_mxfp8 does:
    # fp8 data at the slot, its packed UE8M0 word at the interleaved position.
    cols = tl.arange(0, WIDTH)
    if RETURN_KV:
        tl.store(Out + (token * HEADS + head) * WIDTH + cols, value)
    if STORE:
        if token < slot_count:
            slot = tl.load(Slots + token).to(tl.int64)
            if MXFP8:
                q8, packed = _mxfp8_quantize_row(value.to(Out.dtype.element_ty), WIDTH)
                tl.store(Cache + slot * cache_stride + head * WIDTH + cols, q8)
                word = _sf_interleaved_offset(slot, sf_page_tokens, sf_page_stride)
                tl.store(
                    ScaleCache + word + head * (sf_page_tokens // 128) * 128, packed
                )
            else:
                tl.store(Cache + slot * cache_stride + head * WIDTH + cols, value)


@triton.jit
def _hint_conv_row(
    Weight,
    Ring,
    Scale,
    Slots,
    SeqIdx,
    CuSeqlens,
    SeqLens,
    CacheIndices,
    HasInitialState,
    token,
    head,
    w_stride_c,
    w_stride_w,
    ring_stride_slot,
    ring_stride_row,
    WIDTH: tl.constexpr,
    TAPS: tl.constexpr,
    RING_ROWS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    STORE: tl.constexpr,
):
    # Before the wait: the reads of _conv_row, _clipped_value and
    # _store_kv_row after it, all older than the chain, start for one
    # (token, head). The routing loads into registers (reread from L1 after
    # the wait); the taps, scale and pre-chunk ring rows are hinted into L2.
    cols = head * WIDTH + tl.arange(0, WIDTH)
    lines = (cols % 64) == 0
    seq = tl.load(SeqIdx + token).to(tl.int64)
    bos = tl.load(CuSeqlens + seq).to(tl.int64)
    eos = tl.load(CuSeqlens + seq + 1).to(tl.int32)
    through = tl.load(SeqLens + seq).to(tl.int32)
    slot = tl.load(CacheIndices + seq)
    use_ring = (slot != -1) & (tl.load(HasInitialState + seq) != 0)
    pre_len = through - (eos - bos.to(tl.int32))
    ring = Ring + tl.maximum(slot, 0).to(tl.int64) * ring_stride_slot
    for iw in tl.static_range(TAPS):
        lag = TAPS - 1 - iw
        shifted = token - lag
        position = pre_len + (shifted - bos).to(tl.int32)
        in_ring = (shifted < bos) & (position >= 0) & use_ring
        row = (tl.maximum(position, 0) % RING_ROWS).to(tl.int64)
        prefetch_l2(ring + row * ring_stride_row + cols, lines & in_ring)
        prefetch_l2(Weight + cols * w_stride_c + iw * w_stride_w, lines)
    if HAS_SCALE:
        prefetch_l2(Scale + cols, lines)
    if STORE:
        prefetch_l2(Slots + token + tl.arange(0, 1), tl.arange(0, 1) == 0)


@triton.jit
def _conv_row(
    X,
    Weight,
    Dynamic,
    Ring,
    Checkpoint,
    SeqIdx,
    CuSeqlens,
    SeqLens,
    CacheIndices,
    HasInitialState,
    BlockTable,
    token,
    head,
    x_stride,
    w_stride_c,
    w_stride_w,
    dyn_stride_t,
    dyn_stride_h,
    dyn_stride_w,
    ring_stride_slot,
    ring_stride_row,
    ckpt_stride_page,
    ckpt_stride_row,
    bt_stride_row,
    bt_stride_col,
    conv_scale,
    WIDTH: tl.constexpr,
    TAPS: tl.constexpr,
    RING_ROWS: tl.constexpr,
    DYNAMIC: tl.constexpr,
    CONV_HEAD_DIM: tl.constexpr,
    PUBLISH: tl.constexpr,
):
    # One (token, head) row of inkling_ring_sconv's decode convolution, in
    # its tap order and rounding: pre-chunk taps read the request's ring rows,
    # the raw row persists at its position's ring row, and with PUBLISH the
    # request's last row writes its window to the live checkpoint block.
    cols = head * WIDTH + tl.arange(0, WIDTH)
    seq = tl.load(SeqIdx + token).to(tl.int64)
    bos = tl.load(CuSeqlens + seq).to(tl.int64)
    eos = tl.load(CuSeqlens + seq + 1).to(tl.int32)
    through = tl.load(SeqLens + seq).to(tl.int32)
    slot = tl.load(CacheIndices + seq)
    # PAD_SLOT_ID (-1) marks padded requests, which never touch the ring.
    use_ring = (slot != -1) & (tl.load(HasInitialState + seq) != 0)
    pre_len = through - (eos - bos.to(tl.int32))
    ring = Ring + tl.maximum(slot, 0).to(tl.int64) * ring_stride_slot
    publish = (slot != -1) & (token == eos - 1)
    page = tl.zeros([], dtype=tl.int64)
    if PUBLISH:
        # Column 1 holds the live block.
        page = tl.load(
            BlockTable + seq * bt_stride_row + bt_stride_col, mask=publish, other=0
        ).to(tl.int64)
        publish = publish & (page > 0)
    acc = tl.zeros([WIDTH], dtype=tl.float32)
    for iw in tl.static_range(TAPS):
        # Dynamic taps run lag 0 first; static ones oldest first.
        lag = iw if DYNAMIC else TAPS - 1 - iw
        shifted = token - lag
        in_x = shifted >= bos
        position = pre_len + (shifted - bos).to(tl.int32)
        in_ring = (shifted < bos) & (position >= 0) & use_ring
        row = (tl.maximum(position, 0) % RING_ROWS).to(tl.int64)
        x_tap = tl.load(X + shifted * x_stride + cols, mask=in_x & (cols >= 0), other=0)
        ring_tap = tl.load(
            ring + row * ring_stride_row + cols, mask=in_ring & (cols >= 0), other=0
        )
        tap = tl.where(in_x, x_tap.to(tl.float32), ring_tap.to(tl.float32))
        # +0.0 as the conv kernel's tap select sums the tap with zeros.
        weight = tl.load(Weight + cols * w_stride_c + iw * w_stride_w).to(tl.float32)
        weight = weight + 0.0
        if DYNAMIC:
            coefficient = weight + tl.load(
                Dynamic
                + token * dyn_stride_t
                + (cols // CONV_HEAD_DIM) * dyn_stride_h
                + iw * dyn_stride_w
            ).to(X.dtype.element_ty).to(tl.float32)
            if iw == 0:
                acc = tap * coefficient
            else:
                acc = tl.fma(tap, coefficient, acc)
        else:
            acc = tl.fma(tap, weight.to(X.dtype.element_ty).to(tl.float32), acc)
        if PUBLISH and lag <= TAPS - 2:
            tl.store(
                Checkpoint
                + page * ckpt_stride_page
                + (TAPS - 2 - lag) * ckpt_stride_row
                + cols,
                tap.to(Checkpoint.dtype.element_ty),
                mask=publish & (cols >= 0),
            )
    current = tl.load(X + token * x_stride + cols)
    acc = (acc + current.to(tl.float32)) * conv_scale
    ring_row = (tl.maximum(through - eos + token.to(tl.int32), 0) % RING_ROWS).to(
        tl.int64
    )
    tl.store(
        ring + ring_row * ring_stride_row + cols,
        current.to(Ring.dtype.element_ty),
        mask=(slot != -1) & (cols >= 0),
    )
    return acc.to(X.dtype.element_ty).to(tl.float32)


@triton.jit
def _project_rel_head(
    R,
    RScale,
    OutputScale,
    Projection,
    Logits,
    token,
    head,
    chunk,
    r_stride,
    threshold,
    eps,
    HQ: tl.constexpr,
    DR: tl.constexpr,
    EXTENT: tl.constexpr,
    REL_BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
):
    # Row 0 of a 16-row tensor-core product sums each logit as cuBLAS's matmul does.
    output = _clipped_row(
        R,
        RScale,
        OutputScale,
        token,
        head,
        r_stride,
        threshold,
        eps,
        DR,
        HAS_SCALE,
        HAS_OUTPUT_SCALE,
        Logits.dtype.element_ty,
    ).to(Logits.dtype.element_ty)
    rows = tl.arange(0, 16)
    stored = tl.where(rows[:, None] == 0, output.to(tl.float32)[None, :], 0.0)
    a = stored.to(Logits.dtype.element_ty)
    k = tl.arange(0, DR)
    e = chunk * REL_BLOCK + tl.arange(0, REL_BLOCK)
    b = tl.load(Projection + k[:, None] * EXTENT + e[None, :])
    logits = tl.dot(a, b)
    tl.store(
        Logits + (token * HQ + head) * EXTENT + rows[:, None] * 0 + e[None, :],
        logits.to(Logits.dtype.element_ty),
        mask=(rows == 0)[:, None],
    )


@triton.jit
def _norm_token_head(
    Q,
    K,
    V,
    R,
    QScale,
    KScale,
    RScale,
    OutputScale,
    QOut,
    KOut,
    VOut,
    ROut,
    Projection,
    KCache,
    VCache,
    KScaleCache,
    VScaleCache,
    Slots,
    slot_count,
    k_cache_stride,
    v_cache_stride,
    sf_page_stride,
    sf_page_tokens,
    token,
    head,
    q_stride,
    k_stride,
    v_stride,
    r_stride,
    q_threshold,
    k_threshold,
    r_threshold,
    clip_eps,
    norm_eps,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    EXTENT: tl.constexpr,
    REL_BLOCK: tl.constexpr,
    HAS_Q_SCALE: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_R_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
    HAS_PROJECTION: tl.constexpr,
    STORE: tl.constexpr,
    RETURN_KV: tl.constexpr,
    NORM_V: tl.constexpr,
    MXFP8: tl.constexpr,
    KWeight,
    VWeight,
    KDynamic,
    VDynamic,
    KRing,
    VRing,
    KCheckpoint,
    VCheckpoint,
    SeqIdx,
    CuSeqlens,
    SeqLens,
    CacheIndices,
    HasInitialState,
    BlockTable,
    w_stride_c,
    w_stride_w,
    dyn_stride_t,
    dyn_stride_h,
    dyn_stride_w,
    ring_stride_slot,
    ring_stride_row,
    ckpt_stride_page,
    ckpt_stride_row,
    bt_stride_row,
    bt_stride_col,
    conv_scale,
    CONV: tl.constexpr,
    TAPS: tl.constexpr,
    RING_ROWS: tl.constexpr,
    DYNAMIC: tl.constexpr,
    CONV_HEAD_DIM: tl.constexpr,
    PUBLISH: tl.constexpr,
):
    # Head axis order: Q, K, relative (RP programs each when projecting), then V.
    RP: tl.constexpr = EXTENT // REL_BLOCK if HAS_PROJECTION else 1
    if head < HQ:
        _clip_head(
            Q,
            QScale,
            OutputScale,
            QOut,
            token,
            head,
            q_stride,
            q_threshold,
            clip_eps,
            HQ,
            D,
            HAS_Q_SCALE,
            HAS_OUTPUT_SCALE,
        )
    elif head < HQ + HKV:
        if CONV:
            value = _conv_row(
                K,
                KWeight,
                KDynamic,
                KRing,
                KCheckpoint,
                SeqIdx,
                CuSeqlens,
                SeqLens,
                CacheIndices,
                HasInitialState,
                BlockTable,
                token,
                head - HQ,
                k_stride,
                w_stride_c,
                w_stride_w,
                dyn_stride_t,
                dyn_stride_h,
                dyn_stride_w,
                ring_stride_slot,
                ring_stride_row,
                ckpt_stride_page,
                ckpt_stride_row,
                bt_stride_row,
                bt_stride_col,
                conv_scale,
                D,
                TAPS,
                RING_ROWS,
                DYNAMIC,
                CONV_HEAD_DIM,
                PUBLISH,
            )
        else:
            value = tl.load(
                K + token * k_stride + (head - HQ) * D + tl.arange(0, D)
            ).to(tl.float32)
        output = _clipped_value(
            value,
            KScale,
            OutputScale,
            token,
            head - HQ,
            k_threshold,
            clip_eps,
            D,
            HAS_K_SCALE,
            False,
            KOut.dtype.element_ty,
        )
        _store_kv_row(
            KOut,
            KCache,
            KScaleCache,
            Slots,
            slot_count,
            k_cache_stride,
            sf_page_stride,
            sf_page_tokens,
            output,
            token,
            head - HQ,
            HKV,
            D,
            STORE,
            RETURN_KV,
            MXFP8,
        )
    elif head < HQ + HKV + HQ * RP:
        if HAS_PROJECTION:
            _project_rel_head(
                R,
                RScale,
                OutputScale,
                Projection,
                ROut,
                token,
                (head - HQ - HKV) // RP,
                (head - HQ - HKV) % RP,
                r_stride,
                r_threshold,
                clip_eps,
                HQ,
                DR,
                EXTENT,
                REL_BLOCK,
                HAS_R_SCALE,
                HAS_OUTPUT_SCALE,
            )
        else:
            _clip_head(
                R,
                RScale,
                OutputScale,
                ROut,
                token,
                head - HQ - HKV,
                r_stride,
                r_threshold,
                clip_eps,
                HQ,
                DR,
                HAS_R_SCALE,
                HAS_OUTPUT_SCALE,
            )
    else:
        v_head = head - HQ - HKV - HQ * RP
        if CONV:
            value = _conv_row(
                V,
                VWeight,
                VDynamic,
                VRing,
                VCheckpoint,
                SeqIdx,
                CuSeqlens,
                SeqLens,
                CacheIndices,
                HasInitialState,
                BlockTable,
                token,
                v_head,
                v_stride,
                w_stride_c,
                w_stride_w,
                dyn_stride_t,
                dyn_stride_h,
                dyn_stride_w,
                ring_stride_slot,
                ring_stride_row,
                ckpt_stride_page,
                ckpt_stride_row,
                bt_stride_row,
                bt_stride_col,
                conv_scale,
                D,
                TAPS,
                RING_ROWS,
                DYNAMIC,
                CONV_HEAD_DIM,
                PUBLISH,
            )
        else:
            value = tl.load(V + token * v_stride + v_head * D + tl.arange(0, D)).to(
                tl.float32
            )
        if NORM_V:
            value = _rms_normalized(value, D, norm_eps, False)
        _store_kv_row(
            VOut,
            VCache,
            VScaleCache,
            Slots,
            slot_count,
            v_cache_stride,
            sf_page_stride,
            sf_page_tokens,
            value,
            token,
            v_head,
            HKV,
            D,
            STORE,
            RETURN_KV,
            MXFP8,
        )


@triton.jit(
    do_not_specialize=["tokens", "q_stride", "k_stride", "v_stride", "r_stride"]
)
def _rel_mha_head_norms_kernel(
    Q,
    K,
    V,
    R,
    QScale,
    KScale,
    RScale,
    OutputScale,
    QOut,
    KOut,
    VOut,
    ROut,
    Projection,
    KCache,
    VCache,
    KScaleCache,
    VScaleCache,
    Slots,
    slot_count,
    k_cache_stride,
    v_cache_stride,
    sf_page_stride,
    sf_page_tokens,
    tokens,
    q_stride,
    k_stride,
    v_stride,
    r_stride,
    q_threshold,
    k_threshold,
    r_threshold,
    clip_eps,
    norm_eps,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    EXTENT: tl.constexpr,
    REL_BLOCK: tl.constexpr,
    HAS_Q_SCALE: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_R_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
    HAS_PROJECTION: tl.constexpr,
    STORE: tl.constexpr,
    RETURN_KV: tl.constexpr,
    NORM_V: tl.constexpr,
    MXFP8: tl.constexpr,
    KWeight,
    VWeight,
    KDynamic,
    VDynamic,
    KRing,
    VRing,
    KCheckpoint,
    VCheckpoint,
    SeqIdx,
    CuSeqlens,
    SeqLens,
    CacheIndices,
    HasInitialState,
    BlockTable,
    w_stride_c,
    w_stride_w,
    dyn_stride_t,
    dyn_stride_h,
    dyn_stride_w,
    ring_stride_slot,
    ring_stride_row,
    ckpt_stride_page,
    ckpt_stride_row,
    bt_stride_row,
    bt_stride_col,
    conv_scale,
    CONV: tl.constexpr,
    TAPS: tl.constexpr,
    RING_ROWS: tl.constexpr,
    DYNAMIC: tl.constexpr,
    CONV_HEAD_DIM: tl.constexpr,
    PUBLISH: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # Program axis 1 selects the head (see _norm_token_head).
    if ENABLE_PDL:
        if HAS_PROJECTION:
            rel = tl.program_id(1) - HQ - HKV
            if rel >= 0 and rel < HQ * (EXTENT // REL_BLOCK):
                # Constant projection: prefetch this program's columns before waiting.
                chunk = rel % (EXTENT // REL_BLOCK)
                _prefetch_weight_rows(
                    Projection, 0, chunk * REL_BLOCK, DR, EXTENT, DR, REL_BLOCK * 2
                )
        if CONV:
            # A K or V head's convolution reads a chain of older rows.
            first = tl.program_id(0).to(tl.int64)
            head = tl.program_id(1)
            if head >= HQ and head < HQ + HKV:
                _hint_conv_row(
                    KWeight,
                    KRing,
                    KScale,
                    Slots,
                    SeqIdx,
                    CuSeqlens,
                    SeqLens,
                    CacheIndices,
                    HasInitialState,
                    first,
                    head - HQ,
                    w_stride_c,
                    w_stride_w,
                    ring_stride_slot,
                    ring_stride_row,
                    D,
                    TAPS,
                    RING_ROWS,
                    HAS_K_SCALE,
                    STORE,
                )
            if head >= HQ + HKV + HQ * (EXTENT // REL_BLOCK if HAS_PROJECTION else 1):
                _hint_conv_row(
                    VWeight,
                    VRing,
                    KScale,
                    Slots,
                    SeqIdx,
                    CuSeqlens,
                    SeqLens,
                    CacheIndices,
                    HasInitialState,
                    first,
                    head
                    - HQ
                    - HKV
                    - HQ * (EXTENT // REL_BLOCK if HAS_PROJECTION else 1),
                    w_stride_c,
                    w_stride_w,
                    ring_stride_slot,
                    ring_stride_row,
                    D,
                    TAPS,
                    RING_ROWS,
                    False,
                    STORE,
                )
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    # A program walks tokens of one head, so a prefill chunk needs only a few waves.
    head = tl.program_id(1)
    _norm_token_head(
        Q,
        K,
        V,
        R,
        QScale,
        KScale,
        RScale,
        OutputScale,
        QOut,
        KOut,
        VOut,
        ROut,
        Projection,
        KCache,
        VCache,
        KScaleCache,
        VScaleCache,
        Slots,
        slot_count,
        k_cache_stride,
        v_cache_stride,
        sf_page_stride,
        sf_page_tokens,
        tl.program_id(0).to(tl.int64),
        head,
        q_stride,
        k_stride,
        v_stride,
        r_stride,
        q_threshold,
        k_threshold,
        r_threshold,
        clip_eps,
        norm_eps,
        HQ,
        HKV,
        D,
        DR,
        EXTENT,
        REL_BLOCK,
        HAS_Q_SCALE,
        HAS_K_SCALE,
        HAS_R_SCALE,
        HAS_OUTPUT_SCALE,
        HAS_PROJECTION,
        STORE,
        RETURN_KV,
        NORM_V,
        MXFP8,
        KWeight,
        VWeight,
        KDynamic,
        VDynamic,
        KRing,
        VRing,
        KCheckpoint,
        VCheckpoint,
        SeqIdx,
        CuSeqlens,
        SeqLens,
        CacheIndices,
        HasInitialState,
        BlockTable,
        w_stride_c,
        w_stride_w,
        dyn_stride_t,
        dyn_stride_h,
        dyn_stride_w,
        ring_stride_slot,
        ring_stride_row,
        ckpt_stride_page,
        ckpt_stride_row,
        bt_stride_row,
        bt_stride_col,
        conv_scale,
        CONV,
        TAPS,
        RING_ROWS,
        DYNAMIC,
        CONV_HEAD_DIM,
        PUBLISH,
    )
    for token in range(
        tl.program_id(0) + tl.num_programs(0), tokens, tl.num_programs(0)
    ):
        _norm_token_head(
            Q,
            K,
            V,
            R,
            QScale,
            KScale,
            RScale,
            OutputScale,
            QOut,
            KOut,
            VOut,
            ROut,
            Projection,
            KCache,
            VCache,
            KScaleCache,
            VScaleCache,
            Slots,
            slot_count,
            k_cache_stride,
            v_cache_stride,
            sf_page_stride,
            sf_page_tokens,
            token.to(tl.int64),
            head,
            q_stride,
            k_stride,
            v_stride,
            r_stride,
            q_threshold,
            k_threshold,
            r_threshold,
            clip_eps,
            norm_eps,
            HQ,
            HKV,
            D,
            DR,
            EXTENT,
            REL_BLOCK,
            HAS_Q_SCALE,
            HAS_K_SCALE,
            HAS_R_SCALE,
            HAS_OUTPUT_SCALE,
            HAS_PROJECTION,
            STORE,
            RETURN_KV,
            NORM_V,
            MXFP8,
            KWeight,
            VWeight,
            KDynamic,
            VDynamic,
            KRing,
            VRing,
            KCheckpoint,
            VCheckpoint,
            SeqIdx,
            CuSeqlens,
            SeqLens,
            CacheIndices,
            HasInitialState,
            BlockTable,
            w_stride_c,
            w_stride_w,
            dyn_stride_t,
            dyn_stride_h,
            dyn_stride_w,
            ring_stride_slot,
            ring_stride_row,
            ckpt_stride_page,
            ckpt_stride_row,
            bt_stride_row,
            bt_stride_col,
            conv_scale,
            CONV,
            TAPS,
            RING_ROWS,
            DYNAMIC,
            CONV_HEAD_DIM,
            PUBLISH,
        )


# Token programs per head: decode batches keep one program per token.
_TOKEN_PROGRAMS = 256
# Fused projection spends a tensor-core tile per token; longer batches use a GEMM.
_FUSED_PROJECTION_MAX_TOKENS = 128


def _check_heads(name: str, x: torch.Tensor) -> None:
    if x.ndim != 3 or x.stride(2) != 1 or x.stride(1) != x.shape[2]:
        raise ValueError(f"{name} must be [tokens, heads, width] with packed heads")


def rel_mha_head_norms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    *,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    rel_scale: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    q_threshold: float,
    k_threshold: float,
    rel_threshold: float,
    clip_eps: float,
    value_norm_eps: float | None,
    rel_projection: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``l2_clip`` the Q, K and relative heads (FP32 norms summed in kernel
    order) and RMS-normalize the V heads exactly as ``rmsnorm_precision``
    would, in one launch; optionally project the clipped relative heads to
    relative logits in the same launch.

    Under PDL the next kernel may start before this one's predecessor
    finishes; it must read only constant operands before its own wait.

    Args:
        q: ``[tokens, q_heads, head_dim]`` rows; the token stride may be wide.
        k: ``[tokens, kv_heads, head_dim]`` rows.
        v: ``[tokens, kv_heads, head_dim]`` rows.
        rel: ``[tokens, q_heads, rel_dim]`` rows.
        q_scale: Optional ``[q_heads, head_dim]`` learned clip scale.
        k_scale: Optional ``[kv_heads, head_dim]`` learned clip scale.
        rel_scale: Optional ``[q_heads, rel_dim]`` learned clip scale.
        output_scale: Optional FP32 ``[tokens]`` scale of the clipped Q and
            relative heads, applied after their rounding (not to K).
        q_threshold: L2 limit of the Q heads.
        k_threshold: L2 limit of the K heads.
        rel_threshold: L2 limit of the relative heads.
        clip_eps: Epsilon added to every clipped head's norm.
        value_norm_eps: RMS epsilon of the V heads; None returns ``v`` as is.
        rel_projection: Optional ``[rel_dim, extent]`` weight in ``rel``'s
            dtype, ``rel_dim >= 16``. Given, the last result is
            ``clipped_rel @ rel_projection``, bit for bit the matmul of the
            stored clipped heads, instead of the clipped heads.

    Returns:
        Contiguous clipped Q, K, normalized V, and the clipped relative heads
        or their ``[tokens, q_heads, extent]`` relative logits.
    """
    for name, x in (("q", q), ("k", k), ("v", v), ("rel", rel)):
        _check_heads(name, x)
    tokens, q_heads, head_dim = q.shape
    kv_heads, rel_dim = k.shape[1], rel.shape[2]
    if (
        k.shape != (tokens, kv_heads, head_dim)
        or v.shape != k.shape
        or rel.shape[:2] != (tokens, q_heads)
        or head_dim & (head_dim - 1)
        or rel_dim & (rel_dim - 1)
    ):
        raise ValueError(
            "head norms need matching token counts and power-of-two widths"
        )
    if rel_projection is not None and (
        rel_projection.shape[0] != rel_dim
        or rel_dim < 16
        or rel_projection.dtype != rel.dtype
        or not rel_projection.is_contiguous()
        or rel_projection.shape[1] & (rel_projection.shape[1] - 1)
    ):
        raise ValueError(
            "the relative projection must be a contiguous [rel_dim >= 16, "
            "power-of-two extent] weight in the relative heads' dtype"
        )
    if rel_projection is not None and (
        not q.is_cuda or tokens > _FUSED_PROJECTION_MAX_TOKENS
    ):
        # Store the clipped heads, then project them with one GEMM of the same products.
        *heads, clipped_rel = rel_mha_head_norms(
            q,
            k,
            v,
            rel,
            q_scale=q_scale,
            k_scale=k_scale,
            rel_scale=rel_scale,
            output_scale=output_scale,
            q_threshold=q_threshold,
            k_threshold=k_threshold,
            rel_threshold=rel_threshold,
            clip_eps=clip_eps,
            value_norm_eps=value_norm_eps,
            rel_projection=None,
        )
        return (*heads, torch.matmul(clipped_rel, rel_projection))
    if not q.is_cuda:
        return (
            l2_clip(q, q_threshold, clip_eps, q_scale, output_scale),
            l2_clip(k, k_threshold, clip_eps, k_scale, None),
            (
                v
                if value_norm_eps is None
                else rmsnorm_precision(
                    v, None, value_norm_eps, v.dtype, mean_by_reciprocal=False
                )
            ),
            l2_clip(rel, rel_threshold, clip_eps, rel_scale, output_scale),
        )
    q_out, k_out = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (q, k)
    )
    v_out = (
        v
        if value_norm_eps is None
        else torch.empty(v.shape, dtype=v.dtype, device=v.device)
    )
    rel_out = _launch_head_norms(
        q,
        k,
        v,
        rel,
        q_out,
        k_out,
        v_out,
        q_scale=q_scale,
        k_scale=k_scale,
        rel_scale=rel_scale,
        output_scale=output_scale,
        q_threshold=q_threshold,
        k_threshold=k_threshold,
        rel_threshold=rel_threshold,
        clip_eps=clip_eps,
        value_norm_eps=value_norm_eps,
        rel_projection=rel_projection,
        cache=None,
        return_kv=True,
        conv=None,
    )
    return q_out, k_out, v_out, rel_out


def _launch_head_norms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    *,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    rel_scale: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    q_threshold: float,
    k_threshold: float,
    rel_threshold: float,
    clip_eps: float,
    value_norm_eps: float | None,
    rel_projection: torch.Tensor | None,
    cache: HeadKVCache | None,
    return_kv: bool,
    conv: KVShortConv | None,
) -> torch.Tensor:
    """One head-norms launch into the given outputs; with ``cache`` the K/V
    rows also go to their cache slots, and with ``conv`` (a decode batch,
    stored rows) the K/V programs convolve their input rows first. Returns
    the relative output."""
    tokens, q_heads, head_dim = q.shape
    kv_heads, rel_dim = k.shape[1], rel.shape[2]
    extent = rel_dim if rel_projection is None else rel_projection.shape[1]
    rel_out = torch.empty((tokens, q_heads, extent), dtype=rel.dtype, device=rel.device)
    store = cache is not None
    # A projected relative head takes one program per 256 logits.
    rel_block = min(extent, 256)
    rel_programs = extent // rel_block if rel_projection is not None else 1
    # Stored rows need their V program even when the values pass through.
    v_programs = kv_heads if store or value_norm_eps is not None else 0
    if conv is None:
        conv_args = (k_out,) * 14 + (0,) * 11 + (1.0, False, 1, 1, False, 1, False)
    else:
        k_stream, v_stream, batch = conv.k, conv.v, conv.batch
        dynamic = k_stream.dynamic_weights is not None
        conv_args = (
            k_stream.weight,
            v_stream.weight,
            k_stream.dynamic_weights if dynamic else k_stream.weight,
            v_stream.dynamic_weights if dynamic else v_stream.weight,
            k_stream.ring,
            v_stream.ring,
            k_stream.checkpoint,
            v_stream.checkpoint,
            batch.seq_idx,
            batch.cu_seqlens,
            batch.seq_lens,
            batch.cache_indices,
            batch.has_initial_state,
            batch.block_table,
            *k_stream.weight.stride(),
            *(k_stream.dynamic_weights.stride() if dynamic else (0, 0, 0)),
            *k_stream.ring.stride()[:2],
            *k_stream.checkpoint.stride()[:2],
            *batch.block_table.stride(),
            conv.output_scale,
            True,
            k_stream.weight.shape[1],
            k_stream.ring.shape[1],
            dynamic,
            # Static taps never read the head width: one value, one compilation.
            (
                kv_heads * head_dim // k_stream.dynamic_weights.shape[1]
                if dynamic
                else 1
            ),
            batch.publish_live,
        )
    programs = q_heads + kv_heads + q_heads * rel_programs + v_programs
    mxfp8 = store and cache.format is KVCacheFormat.MXFP8
    if mxfp8:
        # fp8 rows as bytes; the scale slabs as packed u32 words.
        k_cache = cache.k_cache.view(torch.uint8)
        v_cache = cache.v_cache.view(torch.uint8)
        k_sf, v_sf = (
            sf.view(torch.uint8).reshape(-1, 4).view(torch.int32).reshape(-1)
            for sf in (cache.scales.k, cache.scales.v)
        )
        sf_tokens = cache.scales.page_tokens
        sf_page_stride = kv_heads * (sf_tokens // 128) * 128
    elif store:
        k_cache, v_cache, k_sf, v_sf = cache.k_cache, cache.v_cache, k_out, v_out
        sf_tokens, sf_page_stride = 128, 0
    if tokens:
        _rel_mha_head_norms_kernel[(min(tokens, _TOKEN_PROGRAMS), programs)](
            q,
            k,
            v,
            rel,
            q_scale,
            k_scale,
            rel_scale,
            output_scale,
            q_out,
            k_out,
            v_out,
            rel_out,
            rel_projection,
            k_cache if store else k_out,
            v_cache if store else v_out,
            k_sf if store else k_out,
            v_sf if store else v_out,
            cache.slots if store else k_out,
            cache.slots.numel() if store else 0,
            cache.k_cache.stride(0) if store else 0,
            cache.v_cache.stride(0) if store else 0,
            sf_page_stride if store else 0,
            sf_tokens if store else 128,
            tokens,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            rel.stride(0),
            q_threshold,
            k_threshold,
            rel_threshold,
            clip_eps,
            1.0 if value_norm_eps is None else value_norm_eps,
            q_heads,
            kv_heads,
            head_dim,
            rel_dim,
            extent,
            rel_block,
            q_scale is not None,
            k_scale is not None,
            rel_scale is not None,
            output_scale is not None,
            rel_projection is not None,
            store,
            return_kv,
            value_norm_eps is not None,
            mxfp8,
            *conv_args,
            pdl_enabled(),
            # A program holds one head row: on B300 one warp beats four from
            # 8 tokens (2.4 against 2.5 us) to 8192 (81 against 134 us).
            num_warps=1,
            enable_fp_fusion=False,
            **({"launch_pdl": True} if pdl_enabled() else {}),
        )
    return rel_out


def relative_head_prologue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm: RelativeHeadNorm,
    cache: HeadKVCache,
    return_kv: bool,
) -> GQAPrologueOutput:
    """The GQA prologue under relative attention's head steps in one launch:
    ``rel_mha_head_norms``'s outputs, the K and V rows stored once more into
    their native cache slots, as its outputs written by the cache store.
    With a K/V convolution over a decode batch, the K and V programs first
    convolve their rows as ``inkling_ring_sconv`` does.

    Args:
        q: Query ``[tokens, q_heads * head_dim]`` or ``[tokens, q_heads,
            head_dim]``; rows may be strided.
        k, v: ``[tokens, kv_heads * head_dim]`` rows, the convolution's
            inputs when it has one.
        norm: The head steps; ``norm.relative`` is ``[tokens, q_heads, rel_dim]``.
        cache: KV cache destination: native (unconverted) rows, or an MXFP8
            cache the rows are quantized into as ``quantize_store_kv_mxfp8``
            does.
        return_kv: Also return the normalized K/V rows.

    Returns:
        The clipped query, the K/V rows when requested and the relative logits.
    """
    kv_heads, head_dim = cache.k_cache.shape[1:]
    tokens = q.shape[0]
    q_heads = q.shape[1:].numel() // head_dim
    q3 = q.view(tokens, q_heads, head_dim)
    k3 = k.view(tokens, kv_heads, head_dim)
    v3 = v.view(tokens, kv_heads, head_dim)
    if norm.kv_conv is not None and norm.kv_conv.batch.num_extends:
        raise ValueError("the fused K/V convolution takes decode batches only")
    projection = norm.projection
    # Longer batches clip the relative heads here and project them with one GEMM.
    fused_projection = tokens <= _FUSED_PROJECTION_MAX_TOKENS
    q_out = torch.empty(q3.shape, dtype=q.dtype, device=q.device)
    k_out, v_out = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (k3, v3)
    )
    rel = _launch_head_norms(
        q3,
        k3,
        v3,
        norm.relative,
        q_out,
        k_out,
        v_out,
        q_scale=norm.q_scale,
        k_scale=norm.k_scale,
        rel_scale=norm.rel_scale,
        output_scale=norm.q_output_scale,
        q_threshold=norm.q_threshold,
        k_threshold=norm.k_threshold,
        rel_threshold=norm.rel_threshold,
        clip_eps=norm.clip_eps,
        value_norm_eps=norm.value_norm_eps,
        rel_projection=projection if fused_projection else None,
        cache=cache,
        return_kv=return_kv,
        conv=norm.kv_conv,
    )
    if not fused_projection:
        rel = torch.matmul(rel, projection)
    return GQAPrologueOutput(
        q=q_out.view(tokens, -1),
        k=k_out.view(tokens, -1) if return_kv else None,
        v=v_out.view(tokens, -1) if return_kv else None,
        rel_logits=rel,
    )
