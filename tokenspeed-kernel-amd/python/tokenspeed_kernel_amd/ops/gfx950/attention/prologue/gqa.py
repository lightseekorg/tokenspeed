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

"""GQA attention prologue of one head, as a gfx950 device function.

For kernels that run the prologue as one of their stages (a fused decoder
layer). Each head repeats the Triton GQA prologue's operations in its
layouts, so the same BF16 q/k/v rows give the same bytes: per-head RMSNorm
summed by one 64-lane reduction, the rotation ``fma(x1, cos, -x2 * sin)`` /
``fma(x2, cos, x1 * sin)``, and one rounding to FP8 (RTNE).
"""

from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = ["gqa_prologue_head_gfx950"]


@gluon.jit
def _head_tile(
    src,
    weight_ptr,
    cos,
    sin,
    row_mask,
    rows_l: gl.constexpr,
    pairs_l: gl.constexpr,
    BLOCK_T: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HALF_ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    SRC_CACHE: gl.constexpr,
):
    """One head's normalized fp32 rows and rotated pairs (rows, o1, o2, i1, i2).

    ``src`` is a [BLOCK_T] tensor of row base pointers in ``rows_l``'s row
    slice; the pair pointers are rebuilt in ``pairs_l``.
    """
    offs = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, rows_l))
    mask = row_mask[:, None] & (offs < HEAD_DIM)[None, :]
    rows = gl.load(
        src[:, None] + offs[None, :], mask=mask, other=0.0, cache_modifier=SRC_CACHE
    ).to(gl.float32)
    inv_rms = gl.rsqrt(gl.sum(rows * rows, axis=1) / HEAD_DIM + EPS)
    w = gl.load(weight_ptr + offs).to(gl.float32)
    if WEIGHT_OFFSET != 0.0:
        w = w + WEIGHT_OFFSET
    rows = rows * inv_rms[:, None] * w[None, :]
    if HALF_ROTARY > 0:
        half = gl.arange(0, HALF_ROTARY, gl.SliceLayout(0, pairs_l))
        # Interleaved (GPT-J) pairs: channels 2i and 2i + 1.
        i1 = 2 * half
        i2 = 2 * half + 1
        psrc = gl.convert_layout(src, gl.SliceLayout(1, pairs_l))
        pmask = (
            gl.convert_layout(row_mask, gl.SliceLayout(1, pairs_l))[:, None]
            & (half >= 0)[None, :]
        )
        x1 = gl.load(
            psrc[:, None] + i1[None, :], mask=pmask, other=0.0, cache_modifier=SRC_CACHE
        ).to(gl.float32)
        x2 = gl.load(
            psrc[:, None] + i2[None, :], mask=pmask, other=0.0, cache_modifier=SRC_CACHE
        ).to(gl.float32)
        w1 = gl.load(weight_ptr + i1).to(gl.float32)
        w2 = gl.load(weight_ptr + i2).to(gl.float32)
        if WEIGHT_OFFSET != 0.0:
            w1 = w1 + WEIGHT_OFFSET
            w2 = w2 + WEIGHT_OFFSET
        inv = gl.convert_layout(inv_rms, gl.SliceLayout(1, pairs_l))
        x1 = x1 * inv[:, None] * w1[None, :]
        x2 = x2 * inv[:, None] * w2[None, :]
        o1 = gl.fma(x1, cos, -x2 * sin)
        o2 = gl.fma(x2, cos, x1 * sin)
        return rows, o1, o2
    else:
        return rows, rows, rows


@gluon.jit
def _store_head(
    dst,
    rows,
    o1,
    o2,
    row_mask,
    inv_scale,
    rows_l: gl.constexpr,
    pairs_l: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HALF_ROTARY: gl.constexpr,
    SCALED: gl.constexpr,
):
    """Store one head's rows at ``dst`` (a [BLOCK_T] pointer tensor) in its dtype."""
    dtype: gl.constexpr = dst.dtype.element_ty
    offs = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, rows_l))
    if SCALED:
        # A per-tensor scaled FP8 cache holds x / scale, saturated to e4m3's range.
        rows = gl.clamp(rows * inv_scale, -448.0, 448.0)
    if HALF_ROTARY == 0:
        gl.store(
            dst[:, None] + offs[None, :],
            rows.to(dtype, fp_downcast_rounding="rtne"),
            mask=row_mask[:, None] & (offs < HEAD_DIM)[None, :],
        )
    else:
        if 2 * HALF_ROTARY < HEAD_DIM:
            gl.store(
                dst[:, None] + offs[None, :],
                rows.to(dtype, fp_downcast_rounding="rtne"),
                mask=row_mask[:, None] & (offs >= 2 * HALF_ROTARY)[None, :],
            )
        if SCALED:
            o1 = gl.clamp(o1 * inv_scale, -448.0, 448.0)
            o2 = gl.clamp(o2 * inv_scale, -448.0, 448.0)
        half = gl.arange(0, HALF_ROTARY, gl.SliceLayout(0, pairs_l))
        pdst = gl.convert_layout(dst, gl.SliceLayout(1, pairs_l))
        pmask = (
            gl.convert_layout(row_mask, gl.SliceLayout(1, pairs_l))[:, None]
            & (half >= 0)[None, :]
        )
        gl.store(
            pdst[:, None] + (2 * half)[None, :],
            o1.to(dtype, fp_downcast_rounding="rtne"),
            mask=pmask,
        )
        gl.store(
            pdst[:, None] + (2 * half + 1)[None, :],
            o2.to(dtype, fp_downcast_rounding="rtne"),
            mask=pmask,
        )


@gluon.jit
def gqa_prologue_head_gfx950(
    src_ptr,
    src_stride,
    num_rows,
    head,
    q_out_ptr,
    q_out_stride,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    positions_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_inv_scale,
    v_inv_scale,
    k_cache_stride_s,
    k_cache_stride_h,
    v_cache_stride_s,
    v_cache_stride_h,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HALF_ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    KV_SCALED: gl.constexpr,
    BLOCK_T: gl.constexpr,
    SRC_CACHE: gl.constexpr,
):
    """Run the prologue of one head over ``num_rows`` <= BLOCK_T rows.

    ``src_ptr`` holds BF16 rows ``[num_rows, src_stride]`` of
    ``[q heads | k heads | v heads | ...]``. Head ``head < NUM_Q_HEADS``
    writes its normalized, rotated query to ``q_out_ptr`` (row stride
    ``q_out_stride``); KV head ``head - NUM_Q_HEADS`` writes its key and value
    rows to the caches at ``slots_ptr``, scaled by the inverse scales when
    ``KV_SCALED``. ``HALF_ROTARY == 0`` skips the rotation (NoPE).
    ``SRC_CACHE`` is the cache modifier of the q/k/v loads: ``".cv"`` when
    another workgroup of the same launch wrote them. The query and the cache
    rows are stored plainly: their readers run in a later launch.
    """
    NUM_WARPS: gl.constexpr = gl.num_warps()
    # The Triton GQA prologue's layouts (num_warps = 4): rows [T, 64] one
    # token per subgroup row with one channel per lane, pairs [T, 32] two
    # tokens per subgroup.
    rows_l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])
    pairs_l: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [NUM_WARPS, 1], [1, 0])
    tokens = gl.arange(0, BLOCK_T, gl.SliceLayout(1, rows_l)).to(gl.int64)
    row_mask = tokens < num_rows
    if HALF_ROTARY > 0:
        ptok = gl.arange(0, BLOCK_T, gl.SliceLayout(1, pairs_l)).to(gl.int64)
        pmask = ptok < num_rows
        half = gl.arange(0, HALF_ROTARY, gl.SliceLayout(0, pairs_l))
        position = gl.load(positions_ptr + ptok, mask=pmask, other=0)
        base = (
            cos_sin_ptr
            + position.to(gl.int64)[:, None] * cos_sin_stride
            + half[None, :]
        )
        cmask = pmask[:, None] & (half >= 0)[None, :]
        cos = gl.load(base, mask=cmask, other=0.0)
        sin = gl.load(base + HALF_ROTARY, mask=cmask, other=0.0)
    else:
        cos = 0.0
        sin = 0.0
    if head < NUM_Q_HEADS:
        rows, o1, o2 = _head_tile(
            src_ptr + tokens * src_stride + head * HEAD_DIM,
            q_weight_ptr,
            cos,
            sin,
            row_mask,
            rows_l,
            pairs_l,
            BLOCK_T,
            HEAD_DIM,
            HALF_ROTARY,
            EPS,
            WEIGHT_OFFSET,
            SRC_CACHE,
        )
        _store_head(
            q_out_ptr + tokens * q_out_stride + head * HEAD_DIM,
            rows,
            o1,
            o2,
            row_mask,
            1.0,
            rows_l,
            pairs_l,
            HEAD_DIM,
            HALF_ROTARY,
            False,
        )
    else:
        kv_head = head - NUM_Q_HEADS
        rows, o1, o2 = _head_tile(
            src_ptr + tokens * src_stride + (NUM_Q_HEADS + kv_head) * HEAD_DIM,
            k_weight_ptr,
            cos,
            sin,
            row_mask,
            rows_l,
            pairs_l,
            BLOCK_T,
            HEAD_DIM,
            HALF_ROTARY,
            EPS,
            WEIGHT_OFFSET,
            SRC_CACHE,
        )
        slot = gl.load(slots_ptr + tokens, mask=row_mask, other=0).to(gl.int64)
        _store_head(
            k_cache_ptr + slot * k_cache_stride_s + kv_head * k_cache_stride_h,
            rows,
            o1,
            o2,
            row_mask,
            k_inv_scale,
            rows_l,
            pairs_l,
            HEAD_DIM,
            HALF_ROTARY,
            KV_SCALED,
        )
        offs = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, rows_l))
        mask = row_mask[:, None] & (offs < HEAD_DIM)[None, :]
        value = gl.load(
            (
                src_ptr
                + tokens * src_stride
                + (NUM_Q_HEADS + NUM_KV_HEADS + kv_head) * HEAD_DIM
            )[:, None]
            + offs[None, :],
            mask=mask,
            cache_modifier=SRC_CACHE,
        )
        dst = (v_cache_ptr + slot * v_cache_stride_s + kv_head * v_cache_stride_h)[
            :, None
        ] + offs[None, :]
        if KV_SCALED:
            value = gl.clamp(value.to(gl.float32) * v_inv_scale, -448.0, 448.0)
        gl.store(
            dst,
            value.to(dst.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=mask,
        )
