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

"""Common helper utilities for mem-cache operations."""

import torch
import triton
import triton.language as tl


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def set_mla_kv_buffer_per_loc_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    n_loc,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_LOC: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Each CTA writes BLOCK_LOC locs (the full nope+rope span for each).
    Grid is ceil(n_loc / BLOCK_LOC). With BLOCK_LOC > 1 each CTA processes
    a [BLOCK_LOC, nope_dim] tile, exposing more parallelism / vectorization
    width and better amortizing launch overhead at large n_loc.
    Pairs with the block-split set_mla_kv_buffer_kernel above:
    set_mla_kv_buffer_triton dispatches between them.
    """
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    loc_indices = pid * BLOCK_LOC + tl.arange(0, BLOCK_LOC)
    loc_mask = loc_indices < n_loc
    locs = tl.load(loc_ptr + loc_indices, mask=loc_mask, other=0)

    # Nope tile: [BLOCK_LOC, nope_dim]
    nope_offs = tl.arange(0, nope_dim)
    src_nope = tl.load(
        cache_k_nope_ptr + loc_indices[:, None] * nope_stride + nope_offs[None, :],
        mask=loc_mask[:, None],
    )
    tl.store(
        kv_buffer_ptr + locs[:, None] * buffer_stride + nope_offs[None, :],
        src_nope,
        mask=loc_mask[:, None],
    )

    # Rope tile: [BLOCK_LOC, rope_dim]
    rope_offs = tl.arange(0, rope_dim)
    src_rope = tl.load(
        cache_k_rope_ptr + loc_indices[:, None] * rope_stride + rope_offs[None, :],
        mask=loc_mask[:, None],
    )
    tl.store(
        kv_buffer_ptr + locs[:, None] * buffer_stride + nope_dim + rope_offs[None, :],
        src_rope,
        mask=loc_mask[:, None],
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    enable_pdl: bool = False,
):
    # Dispatch buckets from experiments on B200 GPUs.
    #   n_loc <  512  : block-split kernel — more CTAs/loc fills SMs at decode
    #                   batch sizes.
    #   n_loc >= 512  : per-loc kernel — fat tiles saturate bandwidth at
    #                   prefill chunk sizes; (BLOCK_LOC, num_warps, num_stages)
    #                   widens with n_loc. Above 16K each loc has enough
    #                   elements to vectorize at 32 threads (16-byte loads).
    n_loc = loc.numel()
    nope_dim = cache_k_nope.size(-1)
    rope_dim = cache_k_rope.size(-1)

    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    if n_loc >= 512:
        if n_loc >= 16384:
            block_loc, num_warps, num_stages = 4, 1, 2
        elif n_loc >= 2048:
            block_loc, num_warps, num_stages = 4, 4, 2
        else:
            block_loc, num_warps, num_stages = 2, 4, 2
        grid = (triton.cdiv(n_loc, block_loc),)
        set_mla_kv_buffer_per_loc_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            n_loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK_LOC=block_loc,
            ENABLE_PDL=enable_pdl,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra_kwargs,
        )
    else:
        BLOCK = 256
        assert (
            nope_dim % BLOCK == 0
        ), f"nope_dim ({nope_dim}) must be a multiple of BLOCK ({BLOCK})"
        grid = (n_loc, triton.cdiv(nope_dim + rope_dim, BLOCK))
        set_mla_kv_buffer_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK=BLOCK,
            ENABLE_PDL=enable_pdl,
            **extra_kwargs,
        )


@triton.jit
def get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Block-split variant: grid (n_loc, ceil(total_dim/BLOCK)), each CTA reads
    BLOCK elements of one source (nope OR rope, never straddling). More CTAs/loc
    fills SMs better at small n_loc — mirrors the block-split
    set_mla_kv_buffer_kernel. Pairs with get_mla_kv_buffer_per_loc_kernel below:
    get_mla_kv_buffer_triton dispatches between them.

    Requires BLOCK to divide nope_dim so each block is purely nope or purely
    rope (with masking on the trailing rope block). Wrapper picks BLOCK=128.
    """
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    src = tl.load(kv_buffer_ptr + loc * buffer_stride + offs, mask=mask)

    if base + BLOCK <= nope_dim:
        tl.store(cache_k_nope_ptr + pid_loc * nope_stride + offs, src, mask=mask)
    else:
        offs_rope = offs - nope_dim
        tl.store(cache_k_rope_ptr + pid_loc * rope_stride + offs_rope, src, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def get_mla_kv_buffer_per_loc_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    n_loc,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_LOC: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Each CTA reads BLOCK_LOC locs from kv_buffer (gather) and writes them
    contiguously to cache_k_nope / cache_k_rope. Grid is ceil(n_loc / BLOCK_LOC).
    Mirror of set_mla_kv_buffer_per_loc_kernel with read/write directions
    flipped. get_mla_kv_buffer_triton dispatches between this kernel and the
    block-split get_mla_kv_buffer_kernel above based on n_loc.
    """
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    loc_indices = pid * BLOCK_LOC + tl.arange(0, BLOCK_LOC)
    loc_mask = loc_indices < n_loc
    locs = tl.load(loc_ptr + loc_indices, mask=loc_mask, other=0)

    # Nope tile: [BLOCK_LOC, nope_dim] — gather from kv_buffer at locs.
    nope_offs = tl.arange(0, nope_dim)
    src_nope = tl.load(
        kv_buffer_ptr + locs[:, None] * buffer_stride + nope_offs[None, :],
        mask=loc_mask[:, None],
    )
    tl.store(
        cache_k_nope_ptr + loc_indices[:, None] * nope_stride + nope_offs[None, :],
        src_nope,
        mask=loc_mask[:, None],
    )

    # Rope tile: [BLOCK_LOC, rope_dim]
    rope_offs = tl.arange(0, rope_dim)
    src_rope = tl.load(
        kv_buffer_ptr + locs[:, None] * buffer_stride + nope_dim + rope_offs[None, :],
        mask=loc_mask[:, None],
    )
    tl.store(
        cache_k_rope_ptr + loc_indices[:, None] * rope_stride + rope_offs[None, :],
        src_rope,
        mask=loc_mask[:, None],
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def get_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    enable_pdl: bool = False,
):
    # Dispatch buckets from experiments on B200 GPUs.
    #   n_loc <  512  : block-split kernel — more CTAs/loc fills SMs at decode
    #                   batch sizes.
    #   n_loc >= 512  : per-loc kernel — fat tiles saturate bandwidth.
    #                   The W=4→W=1 transition lands earlier than for set
    #                   (gather reads benefit from fewer threads / wider
    #                   per-thread elements / extra pipeline stages).
    n_loc = loc.numel()
    nope_dim = cache_k_nope.size(-1)
    rope_dim = cache_k_rope.size(-1)

    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    if n_loc >= 512:
        if n_loc >= 16384:
            block_loc, num_warps, num_stages = 8, 1, 2
        elif n_loc >= 2048:
            block_loc, num_warps, num_stages = 8, 1, 3
        else:
            block_loc, num_warps, num_stages = 2, 4, 2
        grid = (triton.cdiv(n_loc, block_loc),)
        get_mla_kv_buffer_per_loc_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            n_loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK_LOC=block_loc,
            ENABLE_PDL=enable_pdl,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra_kwargs,
        )
    else:
        BLOCK = 256
        assert (
            nope_dim % BLOCK == 0
        ), f"nope_dim ({nope_dim}) must be a multiple of BLOCK ({BLOCK})"
        grid = (n_loc, triton.cdiv(nope_dim + rope_dim, BLOCK))
        get_mla_kv_buffer_kernel[grid](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK=BLOCK,
            ENABLE_PDL=enable_pdl,
            **extra_kwargs,
        )
