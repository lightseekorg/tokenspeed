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

"""GQA attention prologue on gfx950: device functions per head, and a kernel.

The functions repeat the Triton GQA prologue's operations in its layouts, so
the same BF16 q/k/v rows give the same bytes: a per-head RMSNorm summed by one
64-lane reduction, the interleaved (GPT-J) rotation ``fma(x1, cos, -x2 *
sin)`` / ``fma(x2, cos, x1 * sin)`` over the whole head, and one rounding
(RTNE) to the destination dtype.

``gqa_prologue_query_gfx950`` / ``_key_gfx950`` / ``_value_gfx950`` take one
head's rows as values, so a kernel that computes them (for example by
summing split-K partials) needs no round trip through memory;
``gqa_prologue_head_gfx950`` loads them first. ``gluon_gqa_prologue_gfx950``
runs every head of a token block in its own program.
"""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "gluon_gqa_prologue_gfx950",
    "gqa_prologue_head_gfx950",
    "gqa_prologue_key_gfx950",
    "gqa_prologue_query_gfx950",
    "gqa_prologue_rope_gfx950",
    "gqa_prologue_value_gfx950",
    "launch_gluon_gqa_prologue_gfx950",
    "supports_gluon_gqa_prologue_gfx950",
]


@gluon.constexpr_function
def gqa_prologue_rows_layout(num_warps: int):
    """One token per subgroup row with one channel per lane, as the Triton
    prologue's ``[T, 64]`` rows at four warps."""
    return gl.BlockedLayout([1, 1], [1, 64], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _pairs3_layout(num_warps: int):
    """Interleaved channel pairs ``[T, pairs, 2]``, two tokens per subgroup
    (the Triton prologue's ``[T, 32]`` pairs), the pair in one thread."""
    return gl.BlockedLayout([1, 1, 2], [2, 32, 1], [num_warps, 1, 1], [2, 1, 0])


@gluon.jit
def gqa_prologue_rope_gfx950(
    cos_sin_ptr,
    cos_sin_stride,
    positions_ptr,
    tokens,
    row_mask,
    HALF_ROTARY: gl.constexpr,
):
    """The ``[T, HALF_ROTARY]`` cos and sin of ``tokens``' positions, in the
    rotation's pair layout (``tokens``/``row_mask`` in the rows layout)."""
    pairs_l: gl.constexpr = gl.SliceLayout(2, _pairs3_layout(gl.num_warps()))
    ptok = gl.convert_layout(tokens, gl.SliceLayout(1, pairs_l))
    pmask = gl.convert_layout(row_mask, gl.SliceLayout(1, pairs_l))
    half = gl.arange(0, HALF_ROTARY, gl.SliceLayout(0, pairs_l))
    position = gl.load(positions_ptr + ptok, mask=pmask, other=0)
    base = cos_sin_ptr + position.to(gl.int64)[:, None] * cos_sin_stride + half[None, :]
    cmask = pmask[:, None] & (half >= 0)[None, :]
    cos = gl.load(base, mask=cmask, other=0.0)
    sin = gl.load(base + HALF_ROTARY, mask=cmask, other=0.0)
    return cos, sin


@gluon.jit
def _inv_rms(rows, EPS: gl.constexpr):
    return gl.rsqrt(gl.sum(rows * rows, axis=1) / rows.shape[1] + EPS)


@gluon.jit
def _normed_rows(rows, weight_ptr, EPS: gl.constexpr, WEIGHT_OFFSET: gl.constexpr):
    """One head's FP32 rows times the RMS scale and the weight."""
    offs = gl.arange(0, rows.shape[1], gl.SliceLayout(0, rows.type.layout))
    w = gl.load(weight_ptr + offs).to(gl.float32)
    if WEIGHT_OFFSET != 0.0:
        w = w + WEIGHT_OFFSET
    return rows * _inv_rms(rows, EPS)[:, None] * w[None, :]


@gluon.jit
def _rotated_pairs(
    rows, weight_ptr, cos, sin, EPS: gl.constexpr, WEIGHT_OFFSET: gl.constexpr
):
    """One head's interleaved pairs, normalized and rotated: ``(o1, o2)``
    ``[T, D / 2]`` in the pair layout."""
    HEAD_DIM: gl.constexpr = rows.shape[1]
    pairs3_l: gl.constexpr = _pairs3_layout(gl.num_warps())
    pairs_l: gl.constexpr = gl.SliceLayout(2, pairs3_l)
    x1, x2 = gl.split(
        gl.convert_layout(gl.reshape(rows, [rows.shape[0], HEAD_DIM // 2, 2]), pairs3_l)
    )
    x1 = gl.convert_layout(x1, pairs_l)
    x2 = gl.convert_layout(x2, pairs_l)
    half = gl.arange(0, HEAD_DIM // 2, gl.SliceLayout(0, pairs_l))
    w1 = gl.load(weight_ptr + 2 * half).to(gl.float32)
    w2 = gl.load(weight_ptr + 2 * half + 1).to(gl.float32)
    if WEIGHT_OFFSET != 0.0:
        w1 = w1 + WEIGHT_OFFSET
        w2 = w2 + WEIGHT_OFFSET
    inv = gl.convert_layout(_inv_rms(rows, EPS), gl.SliceLayout(1, pairs_l))
    x1 = x1 * inv[:, None] * w1[None, :]
    x2 = x2 * inv[:, None] * w2[None, :]
    return gl.fma(x1, cos, -x2 * sin), gl.fma(x2, cos, x1 * sin)


@gluon.jit
def _store_rows(dst, rows, row_mask, inv_scale, SCALED: gl.constexpr):
    """Store ``[T, D]`` rows at ``dst`` (``[T]`` row pointers) in ``dst``'s
    dtype (``SCALED``: times ``inv_scale``, saturated to e4m3's range)."""
    offs = gl.arange(0, rows.shape[1], gl.SliceLayout(0, rows.type.layout))
    if SCALED:
        rows = gl.clamp(rows * inv_scale, -448.0, 448.0)
    gl.store(
        dst[:, None] + offs[None, :],
        rows.to(dst.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=row_mask[:, None] & (offs >= 0)[None, :],
    )


@gluon.jit
def _store_pairs(dst, o1, o2, row_mask, inv_scale, SCALED: gl.constexpr):
    """Store rotated pairs at channels ``2i`` / ``2i + 1`` of ``dst``'s rows."""
    dtype: gl.constexpr = dst.dtype.element_ty
    if SCALED:
        o1 = gl.clamp(o1 * inv_scale, -448.0, 448.0)
        o2 = gl.clamp(o2 * inv_scale, -448.0, 448.0)
    pairs_l: gl.constexpr = o1.type.layout
    half = gl.arange(0, o1.shape[1], gl.SliceLayout(0, pairs_l))
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
def _normed_head(
    rows,
    dst,
    weight_ptr,
    cos,
    sin,
    row_mask,
    inv_scale,
    ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    SCALED: gl.constexpr,
):
    if ROTARY:
        o1, o2 = _rotated_pairs(rows, weight_ptr, cos, sin, EPS, WEIGHT_OFFSET)
        _store_pairs(dst, o1, o2, row_mask, inv_scale, SCALED)
    else:
        _store_rows(
            dst,
            _normed_rows(rows, weight_ptr, EPS, WEIGHT_OFFSET),
            row_mask,
            inv_scale,
            SCALED,
        )


@gluon.jit
def gqa_prologue_query_gfx950(
    rows,
    dst,
    weight_ptr,
    cos,
    sin,
    row_mask,
    ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
):
    """Normalize, rotate (``ROTARY``) and store one query head.

    Args:
        rows: FP32 ``[T, HEAD_DIM]`` holding BF16 values, in
            ``gqa_prologue_rows_layout``.
        dst: ``[T]`` pointers to the head's output rows (the query dtype).
        cos, sin: ``gqa_prologue_rope_gfx950``'s tables (unused without
            ``ROTARY``).
        row_mask: ``[T]`` rows to store.
    """
    _normed_head(
        rows,
        dst,
        weight_ptr,
        cos,
        sin,
        row_mask,
        1.0,
        ROTARY,
        EPS,
        WEIGHT_OFFSET,
        False,
    )


@gluon.jit
def gqa_prologue_key_gfx950(
    rows,
    dst,
    weight_ptr,
    cos,
    sin,
    row_mask,
    inv_scale,
    ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    SCALED: gl.constexpr,
):
    """As :func:`gqa_prologue_query_gfx950` for a key head written to its
    cache rows ``dst`` (``SCALED``: an FP8 cache with per-tensor scale)."""
    _normed_head(
        rows, dst, weight_ptr, cos, sin, row_mask, inv_scale, ROTARY, EPS,
        WEIGHT_OFFSET, SCALED,
    )  # fmt: skip


@gluon.jit
def gqa_prologue_value_gfx950(rows, dst, row_mask, inv_scale, SCALED: gl.constexpr):
    """Write a value head's rows (FP32 holding BF16 values, rows layout) to
    its cache rows ``dst``."""
    _store_rows(dst, rows, row_mask, inv_scale, SCALED)


@gluon.jit
def _load_rows(src, row_mask, HEAD_DIM: gl.constexpr, SRC_CACHE: gl.constexpr):
    rows_l: gl.constexpr = gqa_prologue_rows_layout(gl.num_warps())
    offs = gl.arange(0, HEAD_DIM, gl.SliceLayout(0, rows_l))
    return gl.load(
        src[:, None] + offs[None, :],
        mask=row_mask[:, None] & (offs >= 0)[None, :],
        other=0.0,
        cache_modifier=SRC_CACHE,
    ).to(gl.float32)


@gluon.jit
def gqa_prologue_head_gfx950(
    head,
    first_token,
    num_tokens,
    num_rows,
    q_ptr,
    q_stride,
    q_head_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
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
    HEAD_DIM: gl.constexpr,
    ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    KV_SCALED: gl.constexpr,
    BLOCK_T: gl.constexpr,
    SRC_CACHE: gl.constexpr,
):
    """Run head ``head``'s prologue for tokens ``first_token ..`` (``BLOCK_T``
    of them, those below ``num_tokens``).

    A query head (``head < NUM_Q_HEADS``) stores its rows at ``q_out_ptr``; a
    KV head writes its key and value rows to the cache slots of tokens below
    ``num_rows``. q/k/v rows are strided, each head contiguous. ``SRC_CACHE``
    is the q/k/v loads' cache modifier (``".cv"`` when another workgroup of
    the same launch wrote them).
    """
    rows_l: gl.constexpr = gqa_prologue_rows_layout(gl.num_warps())
    tokens = first_token + gl.arange(0, BLOCK_T, gl.SliceLayout(1, rows_l)).to(gl.int64)
    row_mask = tokens < num_tokens
    if ROTARY:
        cos, sin = gqa_prologue_rope_gfx950(
            cos_sin_ptr, cos_sin_stride, positions_ptr, tokens, row_mask, HEAD_DIM // 2
        )
    else:
        cos = 0.0
        sin = 0.0
    if head < NUM_Q_HEADS:
        rows = _load_rows(
            q_ptr + tokens * q_stride + head * q_head_stride,
            row_mask,
            HEAD_DIM,
            SRC_CACHE,
        )
        gqa_prologue_query_gfx950(
            rows,
            q_out_ptr + tokens * q_out_stride + head * HEAD_DIM,
            q_weight_ptr,
            cos,
            sin,
            row_mask,
            ROTARY,
            EPS,
            WEIGHT_OFFSET,
        )
    else:
        kv_head = head - NUM_Q_HEADS
        write_mask = tokens < num_rows
        slot = gl.load(slots_ptr + tokens, mask=write_mask, other=0).to(gl.int64)
        rows = _load_rows(
            k_ptr + tokens * k_stride + kv_head * HEAD_DIM,
            row_mask,
            HEAD_DIM,
            SRC_CACHE,
        )
        gqa_prologue_key_gfx950(
            rows,
            k_cache_ptr + slot * k_cache_stride_s + kv_head * k_cache_stride_h,
            k_weight_ptr,
            cos,
            sin,
            write_mask,
            k_inv_scale,
            ROTARY,
            EPS,
            WEIGHT_OFFSET,
            KV_SCALED,
        )
        values = _load_rows(
            v_ptr + tokens * v_stride + kv_head * HEAD_DIM,
            write_mask,
            HEAD_DIM,
            SRC_CACHE,
        )
        gqa_prologue_value_gfx950(
            values,
            v_cache_ptr + slot * v_cache_stride_s + kv_head * v_cache_stride_h,
            write_mask,
            v_inv_scale,
            KV_SCALED,
        )


@gluon.jit(do_not_specialize=("num_tokens", "num_rows"))
def gluon_gqa_prologue_gfx950(
    q_ptr,
    q_stride,
    q_head_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
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
    num_tokens,
    num_rows,
    NUM_Q_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROTARY: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    KV_SCALED: gl.constexpr,
    BLOCK_T: gl.constexpr,
):
    """Program ``(block, head)`` runs head ``head`` over token block ``block``."""
    gqa_prologue_head_gfx950(
        gl.program_id(1), gl.program_id(0) * BLOCK_T, num_tokens, num_rows,
        q_ptr, q_stride, q_head_stride, k_ptr, k_stride, v_ptr, v_stride, q_out_ptr,
        q_out_stride, q_weight_ptr, k_weight_ptr, cos_sin_ptr, cos_sin_stride, positions_ptr,
        k_cache_ptr, v_cache_ptr, slots_ptr, k_inv_scale, v_inv_scale,
        k_cache_stride_s, k_cache_stride_h, v_cache_stride_s, v_cache_stride_h,
        NUM_Q_HEADS, HEAD_DIM, ROTARY, EPS, WEIGHT_OFFSET, KV_SCALED, BLOCK_T, ""
    )  # fmt: skip


def supports_gluon_gqa_prologue_gfx950(
    q: torch.Tensor, head_dim: int, rotary_dim: int | None
) -> bool:
    """BF16 or FP16 q with contiguous heads, a head dim of 64 or 128 (the
    rows layout's lanes), and a whole-head rotation or none (``rotary_dim``
    None)."""
    return (
        head_dim in (64, 128)
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.stride(-1) == 1
        and rotary_dim in (None, head_dim)
    )


def launch_gluon_gqa_prologue_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    cos_sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    k_inv_scale: float,
    v_inv_scale: float,
    kv_scaled: bool,
    query_dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize and rotate the query, and write normalized, rotated keys and
    the values to the KV cache.

    Args:
        q: ``[T, Hq * D]`` or ``[T, Hq, D]``; rows and heads strided, each
            head contiguous.
        k, v: ``[T, Hkv * D]``, rows strided.
        q_weight, k_weight: per-head RMSNorm weights ``[D]``; ``eps`` and
            ``weight_offset`` as in ``HeadNorm``.
        cos_sin: ``[positions, D]`` cos then sin halves for a whole-head
            GPT-J rotation, with ``positions`` ``[T]``; both None for none.
        k_cache, v_cache: ``[slots, Hkv, D]`` caches; rows for tokens below
            ``slots.numel()`` go to ``slots``.
        k_inv_scale, v_inv_scale: inverse per-tensor FP8 scales when
            ``kv_scaled`` (values are saturated to e4m3's range).
        query_dtype: dtype of the returned query.

    Returns:
        The query ``[T, Hq * D]`` in ``query_dtype``.
    """
    num_tokens = q.shape[0]
    num_kv_heads, head_dim = k_cache.shape[1:]
    num_q_heads = q.shape[1:].numel() // head_dim
    rotary = cos_sin is not None
    if not supports_gluon_gqa_prologue_gfx950(
        q, head_dim, head_dim if rotary else None
    ):
        raise ValueError(
            "gluon_gqa_prologue_gfx950 needs BF16/FP16 q with contiguous heads "
            f"and head dim 64 or 128; got {q.dtype} {tuple(q.shape)} strides "
            f"{q.stride()}, head dim {head_dim}"
        )
    q_out = torch.empty(
        (num_tokens, num_q_heads * head_dim), dtype=query_dtype, device=q.device
    )
    if num_tokens == 0:
        return q_out
    block_t = 16
    grid = (triton.cdiv(num_tokens, block_t), num_q_heads + num_kv_heads)
    gluon_gqa_prologue_gfx950[grid](
        q, q.stride(0), q.stride(1) if q.dim() == 3 else head_dim, k, k.stride(0),
        v, v.stride(0), q_out, q_out.stride(0),
        q_weight, k_weight,
        cos_sin if rotary else q, cos_sin.stride(0) if rotary else 0,
        positions if rotary else slots,
        k_cache, v_cache, slots, k_inv_scale, v_inv_scale,
        k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1),
        num_tokens, slots.numel(),
        NUM_Q_HEADS=num_q_heads, HEAD_DIM=head_dim, ROTARY=rotary, EPS=eps,
        WEIGHT_OFFSET=weight_offset, KV_SCALED=kv_scaled, BLOCK_T=block_t,
        num_warps=4,
    )  # fmt: skip
    return q_out
