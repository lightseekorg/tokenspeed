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

"""Triton fused rotary embedding kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.embedding import (
        FusedMLASetKVBufferArg,
        FusedSetKVBufferArg,
    )


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _rope_apply_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    offsets_ptr,
    value_ptr,
    k_buffer_ptr,
    v_buffer_ptr,
    cache_loc_ptr,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    q_out_stride_t,
    q_out_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    value_stride_t,
    value_stride_h,
    k_buffer_stride_t,
    k_buffer_stride_h,
    v_buffer_stride_t,
    v_buffer_stride_h,
    cache_stride_p,
    num_q_heads,
    num_k_heads,
    head_size,
    rotary_dim,
    HALF_DIM_PADDED: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    HAS_Q_OUT: tl.constexpr,
    HAS_K_OUT: tl.constexpr,
    HAS_FUSED_KV: tl.constexpr,
    IS_NEOX: tl.constexpr,
    POSITION_INT64: tl.constexpr,
):
    """Apply rotary embedding to one (token, head) pair in-place.

    Grid: (num_tokens, num_q_heads + num_k_heads).
    Heads in [0, num_q_heads) belong to Q; heads in
    [num_q_heads, num_q_heads + num_k_heads) belong to K.

    Each program loads cos/sin for `rotary_dim // 2` channels, applies the
    NEOX or GPT-J style rotation to the first `rotary_dim` lanes of the
    head, and leaves the trailing `head_size - rotary_dim` lanes untouched.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    is_query = head_idx < num_q_heads
    kv_head_idx = head_idx - num_q_heads
    if is_query:
        base_ptr = q_ptr + token_idx * q_stride_t + head_idx * q_stride_h
        out_ptr = (
            q_out_ptr + token_idx * q_out_stride_t + head_idx * q_out_stride_h
            if HAS_Q_OUT
            else base_ptr
        )
    else:
        base_ptr = k_ptr + token_idx * k_stride_t + kv_head_idx * k_stride_h
        out_ptr = (
            k_out_ptr + token_idx * k_out_stride_t + kv_head_idx * k_out_stride_h
            if HAS_K_OUT
            else base_ptr
        )

    if POSITION_INT64:
        pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    else:
        pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    if HAS_OFFSETS:
        if POSITION_INT64:
            pos = pos + tl.load(offsets_ptr + token_idx).to(tl.int64)
        else:
            pos = pos + tl.load(offsets_ptr + token_idx).to(tl.int32)

    half = rotary_dim // 2
    half_offs = tl.arange(0, HALF_DIM_PADDED)
    half_mask = half_offs < half

    cos = tl.load(
        cos_sin_cache_ptr + pos * cache_stride_p + half_offs,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * cache_stride_p + half + half_offs,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)

    if IS_NEOX:
        # NEOX layout: x is split into [first_half | second_half].
        # Output: [x1 * cos - x2 * sin, x2 * cos + x1 * sin].
        x1 = tl.load(base_ptr + half_offs, mask=half_mask, other=0.0)
        x2 = tl.load(base_ptr + half + half_offs, mask=half_mask, other=0.0)
        x1_f = x1.to(tl.float32)
        x2_f = x2.to(tl.float32)
        o1 = x1_f * cos - x2_f * sin
        o2 = x2_f * cos + x1_f * sin
        tl.store(out_ptr + half_offs, o1.to(x1.dtype), mask=half_mask)
        tl.store(out_ptr + half + half_offs, o2.to(x2.dtype), mask=half_mask)
    else:
        # GPT-J layout: x is interleaved [x0, x1, x0, x1, ...].
        # Pairs are (x[2i], x[2i+1]); output:
        #   y[2i]   = x[2i] * cos - x[2i+1] * sin
        #   y[2i+1] = x[2i+1] * cos + x[2i] * sin
        x1 = tl.load(base_ptr + 2 * half_offs, mask=half_mask, other=0.0)
        x2 = tl.load(base_ptr + 2 * half_offs + 1, mask=half_mask, other=0.0)
        x1_f = x1.to(tl.float32)
        x2_f = x2.to(tl.float32)
        o1 = x1_f * cos - x2_f * sin
        o2 = x2_f * cos + x1_f * sin
        tl.store(out_ptr + 2 * half_offs, o1.to(x1.dtype), mask=half_mask)
        tl.store(out_ptr + 2 * half_offs + 1, o2.to(x2.dtype), mask=half_mask)

    head_offs = tl.arange(0, HEAD_DIM_PADDED)
    tail_mask = (head_offs >= rotary_dim) & (head_offs < head_size)
    if HAS_Q_OUT or HAS_K_OUT:
        tail = tl.load(base_ptr + head_offs, mask=tail_mask, other=0.0)
        tl.store(out_ptr + head_offs, tail, mask=tail_mask)

    if HAS_FUSED_KV and not is_query:
        # Slot IDs fit in int32, but slot * row stride may not.
        cache_loc = tl.load(cache_loc_ptr + token_idx).to(tl.int64)
        head_mask = head_offs < head_size
        k_value = tl.load(out_ptr + head_offs, mask=head_mask, other=0.0)
        v_value = tl.load(
            value_ptr
            + token_idx * value_stride_t
            + kv_head_idx * value_stride_h
            + head_offs,
            mask=head_mask,
            other=0.0,
        )
        tl.store(
            k_buffer_ptr
            + cache_loc * k_buffer_stride_t
            + kv_head_idx * k_buffer_stride_h
            + head_offs,
            k_value,
            mask=head_mask,
        )
        tl.store(
            v_buffer_ptr
            + cache_loc * v_buffer_stride_t
            + kv_head_idx * v_buffer_stride_h
            + head_offs,
            v_value,
            mask=head_mask,
        )


@triton.jit
def _mla_rope_set_kv_buffer_kernel(
    q_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    q_out_rope_ptr,
    kv_buffer_ptr,
    loc_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_nope_ptr,
    num_tokens,
    q_rope_stride_t: tl.constexpr,
    q_rope_stride_h: tl.constexpr,
    k_nope_stride_t: tl.constexpr,
    k_rope_stride_t: tl.constexpr,
    q_out_rope_stride_t: tl.constexpr,
    q_out_rope_stride_h: tl.constexpr,
    kv_buffer_stride_t: tl.constexpr,
    cos_sin_stride_p: tl.constexpr,
    q_nope_stride_t: tl.constexpr,
    q_nope_stride_h: tl.constexpr,
    num_q_heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    IS_NEOX: tl.constexpr,
    POSITION_INT64: tl.constexpr,
    ASSEMBLE_FULL_QUERY: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEX_INT64: tl.constexpr,
):
    """One launch for the MLA query assembly and the latent KV write.

    Programs with ``head_idx < num_q_heads`` build one query head; the extra
    program per token writes that token's latent row into the cache. Stores
    convert, so an FP8 destination needs no pre-cast. ``APPLY_ROPE=False``
    serves NoPE models, where the same halves are copied without rotation.
    """
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    half = rope_dim // 2
    half_offsets = tl.arange(0, HALF_BLOCK)
    half_mask = half_offsets < half
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < nope_dim
    if IS_NEOX:
        pair_lo = half_offsets
        pair_hi = half + half_offsets
    else:
        pair_lo = half_offsets * 2
        pair_hi = half_offsets * 2 + 1

    # Wait only after the address-independent index math, so the producer's
    # tail overlaps work that touches no global memory.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    # One program covers BLOCK_N tokens. The token grid dimension is what sets
    # the CTA count, and PDL costs a fixed amount per CTA, so widening this is
    # what stops the wait from outgrowing the overlap it buys.
    for tok in tl.static_range(BLOCK_N):
        # The query destination spans nope+rope per head, so token_idx times
        # its row stride passes 2**31 at token counts a public caller can
        # reach, and the wrap is an out-of-bounds write. Widen only when the
        # shape can actually reach it -- the wider arithmetic costs up to 9%
        # at large token counts.
        if INDEX_INT64:
            token_idx = (block_idx * BLOCK_N + tok).to(tl.int64)
        else:
            token_idx = block_idx * BLOCK_N + tok
        if token_idx < num_tokens:
            if APPLY_ROPE:
                if POSITION_INT64:
                    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
                else:
                    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
                cos = tl.load(
                    cos_sin_cache_ptr + pos * cos_sin_stride_p + half_offsets,
                    mask=half_mask,
                    other=0.0,
                ).to(tl.float32)
                sin = tl.load(
                    cos_sin_cache_ptr + pos * cos_sin_stride_p + half + half_offsets,
                    mask=half_mask,
                    other=0.0,
                ).to(tl.float32)

            if head_idx < num_q_heads:
                q_base = (
                    q_rope_ptr
                    + token_idx * q_rope_stride_t
                    + head_idx * q_rope_stride_h
                )
                q_out_base = (
                    q_out_rope_ptr
                    + token_idx * q_out_rope_stride_t
                    + head_idx * q_out_rope_stride_h
                )
                if ASSEMBLE_FULL_QUERY:
                    q_nope = tl.load(
                        q_nope_ptr
                        + token_idx * q_nope_stride_t
                        + head_idx * q_nope_stride_h
                        + nope_offsets,
                        mask=nope_mask,
                        other=0.0,
                    )
                    tl.store(q_out_base + nope_offsets, q_nope, mask=nope_mask)
                    q_out_base = q_out_base + nope_dim

                q1 = tl.load(q_base + pair_lo, mask=half_mask, other=0.0)
                q2 = tl.load(q_base + pair_hi, mask=half_mask, other=0.0)
                if APPLY_ROPE:
                    q1_f = q1.to(tl.float32)
                    q2_f = q2.to(tl.float32)
                    tl.store(
                        q_out_base + pair_lo, q1_f * cos - q2_f * sin, mask=half_mask
                    )
                    tl.store(
                        q_out_base + pair_hi, q2_f * cos + q1_f * sin, mask=half_mask
                    )
                else:
                    tl.store(q_out_base + pair_lo, q1, mask=half_mask)
                    tl.store(q_out_base + pair_hi, q2, mask=half_mask)
            else:
                loc = tl.load(loc_ptr + token_idx).to(tl.int64)
                kv_base = kv_buffer_ptr + loc * kv_buffer_stride_t
                k_nope = tl.load(
                    k_nope_ptr + token_idx * k_nope_stride_t + nope_offsets,
                    mask=nope_mask,
                    other=0.0,
                )
                tl.store(kv_base + nope_offsets, k_nope, mask=nope_mask)

                k_base = k_rope_ptr + token_idx * k_rope_stride_t
                k1 = tl.load(k_base + pair_lo, mask=half_mask, other=0.0)
                k2 = tl.load(k_base + pair_hi, mask=half_mask, other=0.0)
                if APPLY_ROPE:
                    k1_f = k1.to(tl.float32)
                    k2_f = k2.to(tl.float32)
                    tl.store(
                        kv_base + nope_dim + pair_lo,
                        k1_f * cos - k2_f * sin,
                        mask=half_mask,
                    )
                    tl.store(
                        kv_base + nope_dim + pair_hi,
                        k2_f * cos + k1_f * sin,
                        mask=half_mask,
                    )
                else:
                    tl.store(kv_base + nope_dim + pair_lo, k1, mask=half_mask)
                    tl.store(kv_base + nope_dim + pair_hi, k2, mask=half_mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def select_mla_kv_block_n(num_tokens: int, num_q_heads: int) -> int:
    """Tokens per program for the fused MLA write.

    PDL's cost is a fixed amount per CTA (measured ~0.57us per 1000 CTAs on
    B200) while the overlap it buys is roughly constant, so the grid has to
    stay near the point where the two meet. Capped at 6: past that the wider
    per-program footprint costs more than the CTA reduction returns.
    """
    ctas_per_token = num_q_heads + 1
    return max(1, min(6, -(-num_tokens * ctas_per_token // 1100)))


def apply_rope_mla_set_kv_buffer_triton(
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    is_neox: bool,
    fused_mla_set_kv_buffer_arg: FusedMLASetKVBufferArg,
    q_rope_out: torch.Tensor | None,
    enable_pdl: bool = False,
) -> None:
    """Apply MLA RoPE and write the query and KV cache with store conversion.

    A ``cos_sin_cache`` of ``None`` selects the NoPE form: the same halves are
    assembled without rotation, which is what a model with no rotary embedding
    needs.

    Contract the caller owns, unchecked here because the decode scheduler
    already guarantees it: ``fused_mla_set_kv_buffer_arg.cache_loc`` holds one
    write destination per token and every entry is in range and unique for
    the duration of this call. A duplicate tears the row between whichever
    programs race for it; an out-of-range entry writes outside the pool. The
    split path this replaces relies on the same guarantee. Every tensor's
    last dimension must be contiguous (stride 1) -- true for a fresh
    allocation and for the ``Q[..., a:b]`` / ``K[..., a:b]`` slices every
    caller in this tree passes.
    """
    q_rope_out = q_rope if q_rope_out is None else q_rope_out
    k_nope = fused_mla_set_kv_buffer_arg.k_nope
    kv_buffer = fused_mla_set_kv_buffer_arg.kv_buffer
    loc = fused_mla_set_kv_buffer_arg.cache_loc
    q_nope = fused_mla_set_kv_buffer_arg.q_nope

    num_tokens = q_rope.shape[0]
    if num_tokens == 0:
        return

    nope_dim = k_nope.shape[2]
    assert q_rope.ndim == 3
    assert k_nope.ndim == 3 and k_nope.shape[1] == 1
    assert k_rope.ndim == 3 and k_rope.shape[1] == 1
    assert kv_buffer.ndim == 2
    assert loc.numel() == num_tokens
    assert positions.numel() == num_tokens
    assert q_rope.dtype == k_nope.dtype == k_rope.dtype
    if q_nope is None:
        assert q_rope_out.shape == q_rope.shape
    else:
        assert q_nope.shape[:2] == q_rope.shape[:2]
        assert q_nope.shape[2] == nope_dim
        assert q_nope.dtype == q_rope.dtype
        assert q_rope_out.shape[:2] == q_rope.shape[:2]
        assert q_rope_out.shape[2] == nope_dim + q_rope.shape[2]

    num_q_heads = q_rope.shape[1]
    rope_dim = q_rope.shape[2]
    assert k_rope.shape == (num_tokens, 1, rope_dim)
    assert kv_buffer.shape[1] == nope_dim + rope_dim
    assert rope_dim % 2 == 0
    assert loc.dtype in (torch.int32, torch.int64)
    # Indexed as ``ptr + token_idx``, so a strided view would silently read
    # or write the wrong rows -- flatten() would not catch it, it returns a
    # 1-D input unchanged however it is strided.
    assert loc.stride(-1) == 1
    apply_rope = cos_sin_cache is not None
    if apply_rope:
        assert cos_sin_cache.shape[-1] == rope_dim
        assert positions.dtype in (torch.int32, torch.int64)
        assert positions.ndim == 1 and positions.stride(-1) == 1

    # Every token-indexed term the kernel forms, so the widening turns on
    # before any of them can wrap rather than only for the largest.
    max_token_offset = num_tokens * max(
        q_rope_out.stride(0),
        q_rope.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        0 if q_nope is None else q_nope.stride(0),
    )
    index_int64 = max_token_offset >= 2**31

    half_block = max(_next_power_of_2(rope_dim // 2), 16)
    nope_block = max(_next_power_of_2(nope_dim), 16)
    block_n = select_mla_kv_block_n(num_tokens, num_q_heads)
    grid = (triton.cdiv(num_tokens, block_n), num_q_heads + 1)
    _mla_rope_set_kv_buffer_kernel[grid](
        q_rope,
        k_nope,
        k_rope,
        q_rope_out,
        kv_buffer,
        loc,
        cos_sin_cache,
        positions,
        q_nope,
        num_tokens,
        q_rope.stride(0),
        q_rope.stride(1),
        k_nope.stride(0),
        k_rope.stride(0),
        q_rope_out.stride(0),
        q_rope_out.stride(1),
        kv_buffer.stride(0),
        0 if cos_sin_cache is None else cos_sin_cache.stride(0),
        0 if q_nope is None else q_nope.stride(0),
        0 if q_nope is None else q_nope.stride(1),
        num_q_heads,
        nope_dim,
        rope_dim,
        NOPE_BLOCK=nope_block,
        HALF_BLOCK=half_block,
        IS_NEOX=bool(is_neox),
        POSITION_INT64=positions.dtype == torch.int64,
        ASSEMBLE_FULL_QUERY=q_nope is not None,
        ENABLE_PDL=enable_pdl,
        APPLY_ROPE=apply_rope,
        BLOCK_N=block_n,
        INDEX_INT64=index_int64,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


def apply_rope_triton(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    offsets: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    fused_set_kv_buffer_arg: FusedSetKVBufferArg | None = None,
    fused_mla_set_kv_buffer_arg: FusedMLASetKVBufferArg | None = None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Apply rotary positional embedding to query and key in-place.

    Args:
        positions: Token positions, 1D [num_tokens]. int32 or int64.
        query: [num_tokens, num_q_heads * head_size] (will be viewed
            as [num_tokens, num_q_heads, head_size]).
        key: [num_tokens, num_k_heads * head_size] (will be viewed as
            [num_tokens, num_k_heads, head_size]).
        head_size: Per-head dimension.
        cos_sin_cache: [max_position, rotary_dim] packed as
            concat(cos, sin) along the last dimension. Float32 is strongly
            recommended for numerical stability; other dtypes are accepted.
        is_neox: If True, use NEOX-style rotation (x split in halves). If
            False, use GPT-J-style rotation (interleaved pairs).
        offsets: Optional [num_tokens] int tensor added to positions.
        rotary_dim: Rotary dimension. Defaults to
            cos_sin_cache.shape[-1]. Must be even and <= head_size.
    """
    assert (
        positions.dim() == 1
    ), f"triton rope expects 1D positions, got shape {tuple(positions.shape)}"
    assert positions.dtype in (
        torch.int32,
        torch.int64,
    ), f"positions dtype must be int32 or int64, got {positions.dtype}"
    assert (
        query.dtype == key.dtype
    ), f"query/key dtype mismatch: {query.dtype} vs {key.dtype}"

    if rotary_dim is None:
        rotary_dim = cos_sin_cache.shape[-1]
    assert rotary_dim % 2 == 0, f"rotary_dim must be even, got {rotary_dim}"
    assert (
        rotary_dim <= head_size
    ), f"rotary_dim ({rotary_dim}) must be <= head_size ({head_size})"
    assert cos_sin_cache.shape[-1] == rotary_dim, (
        f"cos_sin_cache last dim ({cos_sin_cache.shape[-1]}) must equal "
        f"rotary_dim ({rotary_dim})"
    )

    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return

    if fused_mla_set_kv_buffer_arg is not None:
        if offsets is not None:
            raise ValueError("MLA fused KV write does not support offsets")
        if output_k_rope is not None:
            raise ValueError("MLA fused KV write stores rotated K directly in cache")
        apply_rope_mla_set_kv_buffer_triton(
            positions=positions,
            q_rope=query,
            k_rope=key,
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
            fused_mla_set_kv_buffer_arg=fused_mla_set_kv_buffer_arg,
            q_rope_out=output_q_rope,
            enable_pdl=enable_pdl,
        )
        return

    q_view = query.view(num_tokens, -1, head_size)
    k_view = key.view(num_tokens, -1, head_size)
    num_q_heads = q_view.shape[1]
    num_k_heads = k_view.shape[1]

    if offsets is not None:
        assert (
            offsets.dim() == 1 and offsets.shape[0] == num_tokens
        ), f"offsets must have shape [{num_tokens}], got {tuple(offsets.shape)}"
    if fused_set_kv_buffer_arg is not None:
        if (
            fused_set_kv_buffer_arg.k_scale is not None
            or fused_set_kv_buffer_arg.v_scale is not None
        ):
            raise ValueError("k_scale/v_scale are not supported yet")
        if fused_set_kv_buffer_arg.cache_loc is None:
            raise ValueError("fused_set_kv_buffer_arg.cache_loc is required")
        if fused_set_kv_buffer_arg.cache_loc.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"cache_loc must be int32 or int64, got {fused_set_kv_buffer_arg.cache_loc.dtype}"
            )

    half = rotary_dim // 2
    half_padded = max(_next_power_of_2(half), 16)
    head_padded = max(_next_power_of_2(head_size), 16)

    q_out_view = (
        output_q_rope.view(num_tokens, num_q_heads, head_size)
        if output_q_rope is not None
        else q_view
    )
    k_out_view = (
        output_k_rope.view(num_tokens, num_k_heads, head_size)
        if output_k_rope is not None
        else k_view
    )

    if fused_set_kv_buffer_arg is not None:
        value = fused_set_kv_buffer_arg.value
        value_view = value.view(num_tokens, num_k_heads, -1)
        assert (
            value_view.shape[-1] == head_size
        ), f"fused value head size {value_view.shape[-1]} must match head_size {head_size}"
        k_buffer_view = fused_set_kv_buffer_arg.k_buffer.view(
            fused_set_kv_buffer_arg.k_buffer.shape[0], num_k_heads, head_size
        )
        v_buffer_view = fused_set_kv_buffer_arg.v_buffer.view(
            fused_set_kv_buffer_arg.v_buffer.shape[0], num_k_heads, head_size
        )
        cache_loc = fused_set_kv_buffer_arg.cache_loc
    else:
        value_view = k_view
        k_buffer_view = k_view
        v_buffer_view = k_view
        cache_loc = positions

    grid = (num_tokens, num_q_heads + num_k_heads)
    _rope_apply_kernel[grid](
        q_view,
        k_view,
        q_out_view,
        k_out_view,
        cos_sin_cache,
        positions,
        offsets if offsets is not None else positions,
        value_view,
        k_buffer_view,
        v_buffer_view,
        cache_loc,
        q_view.stride(0),
        q_view.stride(1),
        k_view.stride(0),
        k_view.stride(1),
        q_out_view.stride(0),
        q_out_view.stride(1),
        k_out_view.stride(0),
        k_out_view.stride(1),
        value_view.stride(0),
        value_view.stride(1),
        k_buffer_view.stride(0),
        k_buffer_view.stride(1),
        v_buffer_view.stride(0),
        v_buffer_view.stride(1),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_k_heads,
        head_size,
        rotary_dim,
        HALF_DIM_PADDED=half_padded,
        HEAD_DIM_PADDED=head_padded,
        HAS_OFFSETS=offsets is not None,
        HAS_Q_OUT=output_q_rope is not None,
        HAS_K_OUT=output_k_rope is not None,
        HAS_FUSED_KV=fused_set_kv_buffer_arg is not None,
        IS_NEOX=bool(is_neox),
        POSITION_INT64=positions.dtype == torch.int64,
    )


@triton.jit
def _fp8_quantize_kernel(
    x,
    out,
    scale,
    x_stride_t: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    num_heads: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols
    # PDL: this kernel is launched with launch_pdl and may start while the
    # producer (the projection GEMM) is still writing x; its stores are only
    # guaranteed visible after griddepcontrol.wait.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    values = tl.load(
        x + token * x_stride_t + head * x_stride_h + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_SCALE_TENSOR:
        scale = tl.load(scale)
    values = values * scale
    values_fp8 = values.to(tl.float8e4nv)
    tl.store(
        out + token * out_stride_t + head * out_stride_h + offsets,
        values_fp8,
        mask=(head < num_heads) & mask,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _fp8_quantize(
    x: torch.Tensor,
    out: torch.Tensor,
    scale: float | torch.Tensor,
    *,
    enable_pdl: bool,
) -> None:
    if x.dim() != 3 or out.dim() != 3:
        raise ValueError(
            f"MLA FP8 quantize expects rank-3 tensors, got {x.shape} and {out.shape}"
        )
    if x.shape != out.shape:
        raise ValueError(f"MLA FP8 quantize shape mismatch: {x.shape} vs {out.shape}")
    if out.dtype != torch.float8_e4m3fn:
        raise TypeError(f"MLA FP8 quantize output must be e4m3fn, got {out.dtype}")
    if isinstance(scale, torch.Tensor):
        scale = scale.contiguous()
    block_n = max(16, _next_power_of_2(x.shape[-1]))
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _fp8_quantize_kernel[(x.shape[0], x.shape[1])](
        x,
        out,
        scale,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        num_heads=x.shape[1],
        n_cols=x.shape[2],
        BLOCK_N=block_n,
        HAS_SCALE_TENSOR=isinstance(scale, torch.Tensor),
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=1,
        **extra_kwargs,
    )


@triton.jit
def _mla_nope_quantize_fp8_kernel(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    q_nope_out,
    q_rope_out,
    k_nope_out,
    k_rope_out,
    scale_q,
    scale_kv,
    qn_stride_t,
    qn_stride_h,
    qr_stride_t,
    qr_stride_h,
    kn_stride_t,
    kn_stride_h,
    kr_stride_t,
    kr_stride_h,
    qno_stride_t,
    qno_stride_h,
    qro_stride_t,
    qro_stride_h,
    kno_stride_t,
    kno_stride_h,
    kro_stride_t,
    kro_stride_h,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE_Q_TENSOR: tl.constexpr,
    HAS_SCALE_KV_TENSOR: tl.constexpr,
    BROADCAST_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    # PDL: launched with launch_pdl; the q/k inputs may still be in flight from
    # the producer until griddepcontrol.wait orders them.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SCALE_Q_TENSOR:
        scale_q = tl.load(scale_q)
    if HAS_SCALE_KV_TENSOR:
        scale_kv = tl.load(scale_kv)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < nope_dim
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < rope_dim

    qn = tl.load(
        q_nope + token * qn_stride_t + head * qn_stride_h + offs_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        q_nope_out + token * qno_stride_t + head * qno_stride_h + offs_n,
        (qn * scale_q).to(tl.float8e4nv),
        mask=mask_n,
    )
    qr = tl.load(
        q_rope + token * qr_stride_t + head * qr_stride_h + offs_r,
        mask=mask_r,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        q_rope_out + token * qro_stride_t + head * qro_stride_h + offs_r,
        (qr * scale_q).to(tl.float8e4nv),
        mask=mask_r,
    )

    if BROADCAST_K:
        store_k = head == 0
    else:
        store_k = True

    kn = tl.load(
        k_nope + token * kn_stride_t + head * kn_stride_h + offs_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    if store_k:
        tl.store(
            k_nope_out + token * kno_stride_t + head * kno_stride_h + offs_n,
            (kn * scale_kv).to(tl.float8e4nv),
            mask=mask_n,
        )
    kr = tl.load(
        k_rope + token * kr_stride_t + head * kr_stride_h + offs_r,
        mask=mask_r,
        other=0.0,
    ).to(tl.float32)
    if store_k:
        tl.store(
            k_rope_out + token * kro_stride_t + head * kro_stride_h + offs_r,
            (kr * scale_kv).to(tl.float8e4nv),
            mask=mask_r,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def mla_nope_quantize_fp8_triton(
    *,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> None:
    """The no-RoPE tail of apply_rope_mla: quantize the four query/key parts
    straight into their FP8 output slices in one launch. ``k_rope`` may carry
    a single head; it broadcasts across the output heads via a zero stride."""
    num_tokens, num_heads, nope_dim = q_nope.shape
    rope_dim = q_rope.shape[-1]
    kv_heads = k_nope.shape[1]
    if (k_nope.shape[0], k_nope.shape[2]) != (num_tokens, nope_dim) or kv_heads not in (
        1,
        num_heads,
    ):
        raise ValueError(
            f"k_nope {tuple(k_nope.shape)} must be [{num_tokens}, 1 or {num_heads}, "
            f"{nope_dim}] to match q_nope {tuple(q_nope.shape)}"
        )
    if k_rope.shape[1] not in (1, num_heads):
        raise ValueError(
            f"k_rope heads must be 1 or {num_heads}, got {k_rope.shape[1]}"
        )
    broadcast_k = kv_heads == 1 and num_heads > 1
    if broadcast_k and k_rope.shape[1] != 1:
        raise ValueError(
            f"k_nope is single-head so k_rope must be too, got {k_rope.shape[1]}"
        )
    expected_outputs = (
        ("q_nope_out", q_nope_out, q_nope.shape, q_nope.device),
        ("q_rope_out", q_rope_out, q_rope.shape, q_rope.device),
        ("k_nope_out", k_nope_out, k_nope.shape, k_nope.device),
        (
            "k_rope_out",
            k_rope_out,
            (num_tokens, kv_heads, rope_dim),
            k_rope.device,
        ),
    )
    for name, output, expected_shape, expected_device in expected_outputs:
        if output.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, got {tuple(output.shape)}"
            )
        if output.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"{name} must have dtype {torch.float8_e4m3fn}, got {output.dtype}"
            )
        if output.device != expected_device:
            raise ValueError(
                f"{name} must be on device {expected_device}, got {output.device}"
            )
    if isinstance(quant_scale_q, torch.Tensor):
        quant_scale_q = quant_scale_q.contiguous()
    if isinstance(quant_scale_kv, torch.Tensor):
        quant_scale_kv = quant_scale_kv.contiguous()

    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _mla_nope_quantize_fp8_kernel[(num_tokens, num_heads)](
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_nope_out,
        q_rope_out,
        k_nope_out,
        k_rope_out,
        quant_scale_q,
        quant_scale_kv,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        k_nope.stride(0),
        0 if broadcast_k else k_nope.stride(1),
        k_rope.stride(0),
        0 if k_rope.shape[1] == 1 else k_rope.stride(1),
        q_nope_out.stride(0),
        q_nope_out.stride(1),
        q_rope_out.stride(0),
        q_rope_out.stride(1),
        k_nope_out.stride(0),
        k_nope_out.stride(1),
        k_rope_out.stride(0),
        k_rope_out.stride(1),
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        BLOCK_N=max(16, _next_power_of_2(nope_dim)),
        BLOCK_R=max(16, _next_power_of_2(rope_dim)),
        HAS_SCALE_Q_TENSOR=isinstance(quant_scale_q, torch.Tensor),
        HAS_SCALE_KV_TENSOR=isinstance(quant_scale_kv, torch.Tensor),
        BROADCAST_K=broadcast_k,
        ENABLE_PDL=enable_pdl,
        **extra_kwargs,
    )


def mla_rope_quantize_fp8_triton(
    *,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> None:
    if q_rope.shape[-1] != k_rope.shape[-1]:
        raise ValueError(
            "q_rope and k_rope must have the same rope dim, got "
            f"{q_rope.shape[-1]} and {k_rope.shape[-1]}"
        )
    if q_rope.shape[0] != k_rope.shape[0] or q_rope.shape[0] != positions.numel():
        raise ValueError(
            "MLA RoPE token count mismatch: "
            f"q={q_rope.shape[0]}, k={k_rope.shape[0]}, pos={positions.numel()}"
        )

    q_rope_tmp = torch.empty(q_rope.shape, dtype=q_rope.dtype, device=q_rope.device)
    k_rope_tmp = torch.empty(k_rope.shape, dtype=k_rope.dtype, device=k_rope.device)
    apply_rope_triton(
        positions=positions,
        query=q_rope,
        key=k_rope,
        head_size=q_rope.shape[-1],
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        rotary_dim=q_rope.shape[-1],
        output_q_rope=q_rope_tmp,
        output_k_rope=k_rope_tmp,
    )
    _fp8_quantize(q_rope_tmp, q_rope_out, quant_scale_q, enable_pdl=enable_pdl)
    _fp8_quantize(k_rope_tmp, k_rope_out, quant_scale_kv, enable_pdl=enable_pdl)
    _fp8_quantize(q_nope, q_nope_out, quant_scale_q, enable_pdl=enable_pdl)
    _fp8_quantize(k_nope, k_nope_out, quant_scale_kv, enable_pdl=enable_pdl)


@register_kernel(
    "embedding",
    "rope",
    name="triton_embedding_rope",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(("q", "k"), "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PORTABLE,
    traits={
        "partial_rotary": frozenset({True, False}),
        "is_neox": frozenset({True, False}),
        "has_fused_kv": frozenset({True, False}),
        "has_fused_mla_kv": frozenset({True, False}),
        "fused_mla_full_query": frozenset({True, False}),
        "has_q_out": frozenset({True, False}),
        "has_k_out": frozenset({True, False}),
    },
    tags={"portability"},
)
def triton_embedding_rope(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg: Any = None,
    fused_mla_set_kv_buffer_arg: Any = None,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    apply_rope_triton(
        positions=positions,
        query=q,
        key=k,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
        fused_mla_set_kv_buffer_arg=fused_mla_set_kv_buffer_arg,
        output_q_rope=q_rope_out,
        output_k_rope=k_rope_out,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "embedding",
    "rope_mla_set_kv",
    name="triton_embedding_rope_mla_set_kv",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(
        ("q_rope", "k_rope"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    priority=Priority.PORTABLE,
    traits={
        "has_rope": frozenset({True, False}),
        "is_neox": frozenset({True, False}),
        "fused_mla_full_query": frozenset({True, False}),
    },
    tags={"portability"},
)
def triton_embedding_rope_mla_set_kv(
    *,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    is_neox: bool,
    fused_mla_set_kv_buffer_arg: FusedMLASetKVBufferArg,
    q_rope_out: torch.Tensor,
    enable_pdl: bool = False,
) -> None:
    apply_rope_mla_set_kv_buffer_triton(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        fused_mla_set_kv_buffer_arg=fused_mla_set_kv_buffer_arg,
        q_rope_out=q_rope_out,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "embedding",
    "rope_mla",
    name="triton_embedding_nope_mla",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(
        ("q_rope", "k_rope", "q_nope", "k_nope"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    priority=Priority.PORTABLE,
    traits={
        "has_rope": frozenset({False}),
        "is_neox": frozenset({True, False}),
        "quantize_dtype": frozenset({torch.float8_e4m3fn}),
        "has_scale_q_tensor": frozenset({True, False}),
        "has_scale_kv_tensor": frozenset({True, False}),
    },
    tags={"portability"},
)
def triton_embedding_nope_mla(
    *,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> None:
    mla_nope_quantize_fp8_triton(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        q_rope_out=q_rope_out,
        k_rope_out=k_rope_out,
        q_nope_out=q_nope_out,
        k_nope_out=k_nope_out,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "embedding",
    "rope_mla",
    name="triton_embedding_rope_mla",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(
        ("q_rope", "k_rope", "q_nope", "k_nope"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    priority=Priority.PORTABLE,
    traits={
        "has_rope": frozenset({True}),
        "is_neox": frozenset({True, False}),
        "quantize_dtype": frozenset({torch.float8_e4m3fn}),
        "has_scale_q_tensor": frozenset({True, False}),
        "has_scale_kv_tensor": frozenset({True, False}),
    },
    tags={"portability"},
)
def triton_embedding_rope_mla(
    *,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> None:
    mla_rope_quantize_fp8_triton(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        q_rope_out=q_rope_out,
        k_rope_out=k_rope_out,
        q_nope_out=q_nope_out,
        k_nope_out=k_nope_out,
        is_neox=is_neox,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        enable_pdl=enable_pdl,
    )
