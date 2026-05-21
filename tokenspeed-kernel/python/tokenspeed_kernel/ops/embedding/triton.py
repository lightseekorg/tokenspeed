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

from typing import Any, Optional

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel


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
    CACHE_LOC_INT64: tl.constexpr,
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
        if CACHE_LOC_INT64:
            cache_loc = tl.load(cache_loc_ptr + token_idx).to(tl.int64)
        else:
            cache_loc = tl.load(cache_loc_ptr + token_idx).to(tl.int32)
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


def apply_rope_triton(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    offsets: Optional[torch.Tensor] = None,
    rotary_dim: Optional[int] = None,
    fused_set_kv_buffer_arg=None,
    output_q_rope: Optional[torch.Tensor] = None,
    output_k_rope: Optional[torch.Tensor] = None,
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
        CACHE_LOC_INT64=cache_loc.dtype == torch.int64,
    )


@register_kernel(
    "embedding",
    "rope",
    name="triton_embedding_rope",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    dtypes={torch.float16, torch.bfloat16},
    priority=Priority.PORTABLE,
    traits={
        "partial_rotary": frozenset({True, False}),
        "is_neox": frozenset({True, False}),
        "has_fused_kv": frozenset({True, False}),
        "has_q_out": frozenset({True, False}),
        "has_k_out": frozenset({True, False}),
    },
    tags={"portability"},
)
def triton_embedding_rope(
    *,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    rotary_dim: int | None = None,
    fused_set_kv_buffer_arg: Any = None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        rotary_dim=rotary_dim,
        fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
        output_q_rope=output_q_rope,
        output_k_rope=output_k_rope,
    )
