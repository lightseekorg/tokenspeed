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

"""One-launch attention prologues."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.prologue.types import (
    BOOLS,
    ROPE_STYLES,
    GQAPrologueOutput,
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    LatentKVCache,
    MLAPrologueOutput,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel.ops.embedding import FusedMLASetKVBufferArg
from tokenspeed_kernel.ops.embedding.triton import apply_rope_mla_set_kv_buffer_triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Above this the one-launch write loses to the composite (measured at 16-64 heads).
_MLA_MAX_TOKEN_HEADS = 2048 * 16


@triton.jit
def _rope_tables(
    cos_sin_ptr,
    positions_ptr,
    tokens,
    row_mask,
    cos_sin_stride,
    positions_stride,
    half_rotary: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    MROPE_ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    """cos/sin per (token, rotary pair); multimodal RoPE takes each pair's
    position from the T, H or W row its section names."""
    half = tl.arange(0, HALF_BLOCK)
    mask = row_mask[:, None] & (half < half_rotary)[None, :]
    if MROPE_ROWS:
        if INTERLEAVED:
            row = tl.where((half % 3 == 1) & (half < 3 * S1), 1, 0)
            row = tl.where((half % 3 == 2) & (half < 3 * S2), 2, row)
        else:
            row = tl.where(half < S0, 0, tl.where(half < S0 + S1, 1, 2))
        position = tl.load(
            positions_ptr
            + row[None, :].to(tl.int64) * positions_stride
            + tokens[:, None],
            mask=mask,
            other=0,
        )
    else:
        position = tl.load(positions_ptr + tokens, mask=row_mask, other=0)[:, None]
    base = cos_sin_ptr + position.to(tl.int64) * cos_sin_stride + half[None, :]
    cos = tl.load(base, mask=mask, other=0.0)
    sin = tl.load(base + half_rotary, mask=mask, other=0.0)
    return cos, sin


@triton.jit
def _gqa_prologue_tile(
    src,
    weight_ptr,
    cos,
    sin,
    inv_rms,
    row_mask,
    head_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    weight_offset: tl.constexpr,
    HAS_NORM: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    """A block of one head's fp32 rows and their rotated pairs: (rows, first
    half, second half)."""
    offs = tl.arange(0, BLOCK)
    mask = row_mask[:, None] & (offs < head_dim)[None, :]
    rows = tl.load(src[:, None] + offs[None, :], mask=mask, other=0.0).to(tl.float32)
    half = tl.arange(0, HALF_BLOCK)
    half_mask = half < half_rotary
    pair_mask = row_mask[:, None] & half_mask[None, :]
    if IS_NEOX:
        i1 = half
        i2 = half + half_rotary
    else:
        i1 = 2 * half
        i2 = 2 * half + 1
    x1 = tl.load(src[:, None] + i1[None, :], mask=pair_mask, other=0.0).to(tl.float32)
    x2 = tl.load(src[:, None] + i2[None, :], mask=pair_mask, other=0.0).to(tl.float32)
    if HAS_NORM:
        w = tl.load(weight_ptr + offs, mask=offs < head_dim, other=0.0).to(tl.float32)
        w1 = tl.load(weight_ptr + i1, mask=half_mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_ptr + i2, mask=half_mask, other=0.0).to(tl.float32)
        # Adding +0.0 would turn a -0.0 weight into +0.0.
        if weight_offset != 0.0:
            w, w1, w2 = w + weight_offset, w1 + weight_offset, w2 + weight_offset
        rows = rows * inv_rms[:, None] * w[None, :]
        x1 = x1 * inv_rms[:, None] * w1[None, :]
        x2 = x2 * inv_rms[:, None] * w2[None, :]
    # The association the CUDA embedding.rope kernel compiles to.
    o1 = tl.fma(x1, cos, -x2 * sin)
    o2 = tl.fma(x2, cos, x1 * sin)
    return rows, o1, o2, i1, i2, half_mask


@triton.jit
def _gqa_store_tile(
    dst,
    rows,
    o1,
    o2,
    i1,
    i2,
    row_mask,
    half_mask,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    dtype = dst.dtype.element_ty
    tl.store(
        dst[:, None] + offs[None, :],
        rows.to(dtype),
        mask=row_mask[:, None] & ((offs >= rotary_dim) & (offs < head_dim))[None, :],
    )
    pair_mask = row_mask[:, None] & half_mask[None, :]
    tl.store(dst[:, None] + i1[None, :], o1.to(dtype), mask=pair_mask)
    tl.store(dst[:, None] + i2[None, :], o2.to(dtype), mask=pair_mask)


@triton.jit
def _gqa_prologue_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    num_tokens,
    num_rows,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    v_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    cos_sin_stride,
    positions_stride,
    k_cache_stride_s,
    k_cache_stride_h,
    v_cache_stride_s,
    v_cache_stride_h,
    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    weight_offset: tl.constexpr,
    eps: tl.constexpr,
    HAS_NORM: tl.constexpr,
    IS_NEOX: tl.constexpr,
    MROPE_ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    RETURN_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    tokens = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T).to(
        tl.int64
    )
    row_mask = tokens < num_tokens
    head = tl.program_id(1).to(tl.int64)
    is_k = head >= num_q_heads
    kv_head = head - num_q_heads
    if is_k:
        src = k_ptr + tokens * k_stride_t + kv_head * head_dim
        weight_ptr = k_weight_ptr
    else:
        src = q_ptr + tokens * q_stride_t + head * q_stride_h
        weight_ptr = q_weight_ptr

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    inv_rms = 0.0
    if HAS_NORM:
        offs = tl.arange(0, BLOCK)
        mask = row_mask[:, None] & (offs < head_dim)[None, :]
        x = tl.load(src[:, None] + offs[None, :], mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
    cos = 0.0
    sin = 0.0
    if half_rotary > 0:
        cos, sin = _rope_tables(
            cos_sin_ptr,
            positions_ptr,
            tokens,
            row_mask,
            cos_sin_stride,
            positions_stride,
            half_rotary,
            HALF_BLOCK,
            MROPE_ROWS,
            INTERLEAVED,
            S0,
            S1,
            S2,
        )
    rows, o1, o2, i1, i2, half_mask = _gqa_prologue_tile(
        src,
        weight_ptr,
        cos,
        sin,
        inv_rms,
        row_mask,
        head_dim,
        half_rotary,
        weight_offset,
        HAS_NORM,
        IS_NEOX,
        BLOCK,
        HALF_BLOCK,
    )

    if not is_k:
        _gqa_store_tile(
            q_out_ptr + tokens * q_out_stride_t + head * head_dim,
            rows,
            o1,
            o2,
            i1,
            i2,
            row_mask,
            half_mask,
            head_dim,
            2 * half_rotary,
            BLOCK,
        )
    else:
        if RETURN_K:
            _gqa_store_tile(
                k_out_ptr + tokens * k_out_stride_t + kv_head * head_dim,
                rows,
                o1,
                o2,
                i1,
                i2,
                row_mask,
                half_mask,
                head_dim,
                2 * half_rotary,
                BLOCK,
            )
        write_mask = tokens < num_rows
        slot = tl.load(slots_ptr + tokens, mask=write_mask, other=0).to(tl.int64)
        _gqa_store_tile(
            k_cache_ptr + slot * k_cache_stride_s + kv_head * k_cache_stride_h,
            rows,
            o1,
            o2,
            i1,
            i2,
            write_mask,
            half_mask,
            head_dim,
            2 * half_rotary,
            BLOCK,
        )
        offs = tl.arange(0, BLOCK)
        mask = write_mask[:, None] & (offs < head_dim)[None, :]
        value = tl.load(
            v_ptr[:, None]
            + (tokens * v_stride_t + kv_head * head_dim)[:, None]
            + offs[None, :],
            mask=mask,
        )
        dst = (
            v_cache_ptr
            + (slot * v_cache_stride_s + kv_head * v_cache_stride_h)[:, None]
            + offs[None, :]
        )
        tl.store(dst, value.to(dst.dtype.element_ty), mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _block_tokens(num_tokens: int, block: int) -> int:
    """Tokens per program: at least four; 512 elements at decode sizes, 2048 past 256 tokens."""
    elements = 512 if num_tokens <= 64 else 1024 if num_tokens <= 256 else 2048
    return max(4, elements // block)


@register_kernel(
    "attention",
    "gqa_prologue",
    name="triton_gqa_prologue",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(("q",), "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PERFORMANT,
    traits={
        "has_norm": BOOLS,
        "kv_format": frozenset({"native", "fp8"}),
        "kv_convert": BOOLS,
        "mrope": BOOLS,
        "partial_rotary": BOOLS,
        "return_kv": BOOLS,
        "rope_style": ROPE_STYLES,
    },
)
def triton_gqa_prologue(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm: HeadNorm | None,
    rotary: Rotary | None,
    cache: HeadKVCache,
    return_kv: bool,
    enable_pdl: bool,
) -> GQAPrologueOutput:
    num_tokens = q.shape[0]
    num_kv_heads, head_dim = cache.k_cache.shape[1:]
    num_q_heads = q.shape[1:].numel() // head_dim
    q_out = torch.empty(
        (num_tokens, num_q_heads * head_dim), dtype=q.dtype, device=q.device
    )
    k_out = (
        torch.empty(
            (num_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
        )
        if return_kv
        else q_out
    )
    if num_tokens > 0:
        half_rotary = 0 if rotary is None else rotary.rotary_dim // 2
        positions = cache.slots if rotary is None else rotary.positions
        mrope = None if rotary is None else rotary.mrope
        mrope_rows = positions.ndim == 2
        s0, s1, s2 = mrope.section if mrope_rows else (0, 0, 0)
        block_t = _block_tokens(num_tokens, triton.next_power_of_2(head_dim))
        _gqa_prologue_kernel[
            (triton.cdiv(num_tokens, block_t), num_q_heads + num_kv_heads)
        ](
            q,
            k,
            v,
            q_out,
            k_out,
            q if norm is None else norm.q_weight,
            k if norm is None else norm.k_weight,
            q if rotary is None else rotary.cos_sin_cache,
            positions,
            cache.k_cache,
            cache.v_cache,
            cache.slots,
            num_tokens,
            cache.slots.numel(),
            q.stride(0),
            q.stride(1) if q.dim() == 3 else head_dim,
            k.stride(0),
            v.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            0 if rotary is None else rotary.cos_sin_cache.stride(0),
            positions.stride(0) if mrope_rows else 0,
            cache.k_cache.stride(0),
            cache.k_cache.stride(1),
            cache.v_cache.stride(0),
            cache.v_cache.stride(1),
            num_q_heads,
            head_dim,
            half_rotary,
            0.0 if norm is None else norm.weight_offset,
            0.0 if norm is None else norm.eps,
            HAS_NORM=norm is not None,
            IS_NEOX=rotary is not None and rotary.style is RopeStyle.NEOX,
            MROPE_ROWS=mrope_rows,
            INTERLEAVED=mrope_rows and mrope.interleaved,
            S0=s0,
            S1=s1,
            S2=s2,
            RETURN_K=return_kv,
            BLOCK_T=block_t,
            BLOCK=triton.next_power_of_2(head_dim),
            HALF_BLOCK=max(triton.next_power_of_2(half_rotary), 1),
            ENABLE_PDL=enable_pdl,
            num_warps=4,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return GQAPrologueOutput(
        q=q_out, k=k_out if return_kv else None, v=v if return_kv else None
    )


@register_kernel(
    "attention",
    "mla_prologue",
    name="triton_mla_prologue",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures(("query",), "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PERFORMANT,
    traits={
        "token_heads_max": frozenset({_MLA_MAX_TOKEN_HEADS}),
        "expanded": frozenset({False}),
        "full_write": frozenset({True}),
        "kv_format": frozenset({"native", "fp8"}),
        "kv_convert": BOOLS,
        "rope_style": ROPE_STYLES,
        "sanitize": BOOLS,
    },
)
def triton_mla_prologue(
    *,
    query: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    expanded: None,
    rotary: Rotary | None,
    cache: LatentKVCache,
    enable_pdl: bool,
) -> MLAPrologueOutput:
    del expanded
    rank = latent_cache.shape[-1] - q_pe.shape[-1]
    latent = latent_cache.unsqueeze(1)
    cos_sin_cache = None if rotary is None else rotary.cos_sin_cache
    is_neox = rotary is not None and rotary.style is RopeStyle.NEOX
    fp8 = cache.format is KVCacheFormat.FP8
    out = (
        torch.empty(query.shape, dtype=torch.float8_e4m3fn, device=query.device)
        if fp8
        else query
    )
    apply_rope_mla_set_kv_buffer_triton(
        # NoPE never reads positions; the launch only takes their length and dtype.
        positions=cache.slots if rotary is None else rotary.positions,
        q_rope=q_pe,
        k_rope=latent[..., rank:],
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
            k_nope=latent[..., :rank],
            kv_buffer=cache.kv_cache.view(cache.kv_cache.shape[0], -1),
            cache_loc=cache.slots,
            q_nope=query[..., :rank] if fp8 else None,
            sanitize=cache.sanitize,
        ),
        q_rope_out=out if fp8 else query[..., rank:],
        enable_pdl=enable_pdl,
    )
    return MLAPrologueOutput(query=out, key=None, value=None)
