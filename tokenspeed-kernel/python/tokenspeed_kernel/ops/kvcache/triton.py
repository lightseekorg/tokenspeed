# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""Triton implementation of KVStore transfer kernels."""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform

_PER_LAYER_GRID_CAP = int(os.environ.get("TOKENSPEED_KV_GRID_CAP", "64"))
_ALL_LAYER_GRID_CAP = int(os.environ.get("TOKENSPEED_KV_ALL_LAYER_GRID_CAP", "32"))

_is_nvidia = current_platform().is_nvidia

__all__ = [
    "fused_fp8_set_kv_buffer",
    "flat_decode_locs",
    "flat_tables_unpack",
    "gather_page_table_with_padding",
    "index_k_block_split_scatter",
    "quantize_mxfp8_rows",
    "quantize_store_kv_mxfp8",
    "store_kv_cache",
    "store_sf_interleaved",
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_mla",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_mla",
]


# -----------------------------------------------------------------------------
# MXFP8 Scale-Factor Scatter (interleaved FA4 atom layout)
# -----------------------------------------------------------------------------


@triton.jit
def _sf_interleaved_offset(slot, page_tokens, sf_page_stride):
    """Head-0 offset (u32 words) of ``slot``'s packed-SF word.

    Page-major: page ``slot // page_tokens`` at ``sf_page_stride`` words
    apart; within the page, the slot's 128-row chunk, then the
    BlockScaledBasicChunk position ``(row % 32) * 4 + row // 32``.
    Callers add ``h * chunks_per_page * 128`` for head ``h``.
    """
    page_idx = slot // page_tokens
    page_off = slot % page_tokens
    chunk_idx = page_off // 128
    row = page_off % 128
    interleaved = chunk_idx * 128 + (row % 32) * 4 + (row // 32)
    return page_idx * sf_page_stride + interleaved


@triton.jit
def _mxfp8_quantize_row(x, HEAD_DIM: tl.constexpr):
    """Quantize one [HEAD_DIM] row to MXFP8 (flashinfer bit-parity).

    Per 32-element group: amax -> ``e8m0 = clamp(ceil(log2(amax / 448)),
    -127, 127) + 127`` and ``fp8 = rn(x * 2^-exp)`` (zero groups quantize
    to exponent -127, data 0). Returns ``(q8, packed_sf)``: the fp8-e4m3
    bits as u8 and the HEAD_DIM // 32 e8m0 bytes packed little-endian in
    one u32.
    """
    xf = x.to(tl.float32)
    # Per-32 groups: amax -> e8m0 exponent (flashinfer rounding).
    g = tl.reshape(tl.abs(xf), (HEAD_DIM // 32, 32))
    amax = tl.max(g, axis=1)
    exp = tl.ceil(tl.log2(amax / 448.0))
    exp = tl.clamp(exp, -127.0, 127.0)
    exp = tl.where(amax > 0, exp, -127.0)
    sf_bytes = (exp + 127.0).to(tl.uint32)  # [HEAD_DIM // 32]
    # Quantize: x * 2^-exp, RN to e4m3.
    scale = tl.exp2(-exp)  # [HEAD_DIM // 32]
    q = tl.reshape(tl.reshape(xf, (HEAD_DIM // 32, 32)) * scale[:, None], (HEAD_DIM,))
    q8 = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # Pack the e8m0 bytes little-endian into one u32.
    idx = tl.arange(0, HEAD_DIM // 32)
    packed = tl.sum(sf_bytes << (8 * idx))
    return q8, packed


@triton.jit
def _store_sf_interleaved_kernel(
    sf_in_ptr,
    sf_out_ptr,
    loc_ptr,
    num_tokens,
    nheads: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Scatter per-token MXFP8 scale rows into the FA4 interleaved layout.

    Input is viewed as [num_tokens, nheads] of u32 (4 packed e8m0 scales,
    i.e. head_dim 128 at one scale per 32 elements). Output is
    [num_pages, nheads, page_size // 128, 128] of u32: pages hold
    ``page_size // 128`` consecutive 128-row chunks per head (the
    tile_to_shape order the blockscaled kernel derives for k*128-token
    paged TMA), and within a chunk row ``r`` lands at
    ``(r % 32) * 4 + (r // 32)`` — the BlockScaledBasicChunk (32, 4, 4)
    atom loaded directly under ``kv_sf_interleaved``.
    """
    pid = tl.program_id(0)
    tok_offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = tok_offsets < num_tokens

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    slots = tl.load(loc_ptr + tok_offsets, mask=mask, other=0).to(tl.int64)
    chunks_per_page: tl.constexpr = page_size // 128
    page_stride: tl.constexpr = nheads * chunks_per_page * 128
    sf_base = _sf_interleaved_offset(slots, page_size, page_stride)

    for h in tl.static_range(nheads):
        vals = tl.load(sf_in_ptr + tok_offsets * nheads + h, mask=mask, other=0)
        tl.store(sf_out_ptr + sf_base + h * chunks_per_page * 128, vals, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def store_sf_interleaved(
    sf_in: torch.Tensor,
    sf_out: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = 128,
    enable_pdl: bool = False,
) -> None:
    """Scatter per-token MXFP8 scale factors into the interleaved page layout.

    Args:
        sf_in: Per-token scales with shape [num_tokens, num_kv_heads, 4]
            in float8_e8m0fnu (head_dim 128, one scale per 32 elements).
        sf_out: Paged scale buffer in float8_e8m0fnu with shape
            [num_pages, num_kv_heads, 32, 4, 4] (page_size 128) or
            [num_pages, num_kv_heads, page_size // 128, 32, 4, 4]
            (page_size = k*128), laid out as consecutive
            BlockScaledBasicChunk atoms per head — what the blockscaled
            kernel's paged TMA consumes under ``kv_sf_interleaved``.
        loc: Destination slot index per token, shape [num_tokens], integer.
        page_size: Tokens per page; must be a multiple of 128.
        enable_pdl: Launch with Programmatic Dependent Launch (Hopper+).
    """
    assert (
        page_size % 128 == 0
    ), f"interleaved SF layout requires page_size % 128 == 0, got {page_size}"
    num_tokens, nheads, sf_dim = sf_in.shape
    assert sf_dim == 4, f"expected sf_dim=4 (head_dim 128 / 32), got {sf_dim}"
    if num_tokens == 0:
        return

    sf_in_u32 = (
        sf_in.view(torch.uint8)
        .reshape(num_tokens, nheads, 4)
        .contiguous()
        .view(torch.int32)
        .reshape(num_tokens, nheads)
    )
    sf_out_u32 = sf_out.view(torch.uint8).reshape(-1, 4).view(torch.int32).reshape(-1)

    BLOCK_T = 128
    grid = ((num_tokens + BLOCK_T - 1) // BLOCK_T,)
    use_pdl = bool(enable_pdl and _is_nvidia)
    kwargs = {}
    if use_pdl:
        kwargs["launch_pdl"] = True
    _store_sf_interleaved_kernel[grid](
        sf_in_u32,
        sf_out_u32,
        loc,
        num_tokens,
        nheads=nheads,
        page_size=page_size,
        BLOCK_T=BLOCK_T,
        ENABLE_PDL=use_pdl,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Per-Layer KV Cache Scatter
# -----------------------------------------------------------------------------


@triton.jit
def _store_kv_cache_kernel(
    k_src_ptr,
    v_src_ptr,
    k_dst_ptr,
    v_dst_ptr,
    loc_ptr,
    k_src_token_stride,
    v_src_token_stride,
    k_dst_row_stride,
    v_dst_row_stride,
    n_kv_per_token: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Scatter rows of k_src/v_src into k_dst/v_dst at indices loc_ptr.

    Stride-aware: leading axis of src/dst can have any stride; the only
    requirement is ``stride(-1) == 1`` so we can use linear addressing on
    the flattened head_dim×num_kv_heads axis.
    """
    is_v = tl.program_id(0)
    row = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_kv_per_token

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    dst_row = tl.load(loc_ptr + row).to(tl.int64)

    if is_v == 1:
        src = tl.load(
            v_src_ptr + row * v_src_token_stride + offsets, mask=mask, other=0
        )
        tl.store(v_dst_ptr + dst_row * v_dst_row_stride + offsets, src, mask=mask)
    else:
        src = tl.load(
            k_src_ptr + row * k_src_token_stride + offsets, mask=mask, other=0
        )
        tl.store(k_dst_ptr + dst_row * k_dst_row_stride + offsets, src, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def store_kv_cache(
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    loc: torch.Tensor,
    enable_pdl: bool = False,
) -> None:
    """Fused per-token KV cache scatter for one layer.

    Replaces ``k_dst[loc] = k_src; v_dst[loc] = v_src`` with a single triton
    launch handling both k and v rows. The last dim of all four tensors must
    be contiguous (stride == 1); the leading axis may have any stride — this
    lets src tensors come from a qkv-split view directly (no contiguous copy
    required).

    ``enable_pdl`` launches with Programmatic Dependent Launch (Hopper+):
    the kernel waits for its producer before the first load and signals
    dependents after its last store.
    """
    n_tokens = k_src.shape[0]
    if n_tokens == 0:
        return
    n_kv_k = k_src.numel() // n_tokens
    n_kv_v = v_src.numel() // n_tokens
    assert (
        n_kv_k == n_kv_v
    ), f"k/v must share per-token element count, got {n_kv_k} vs {n_kv_v}"
    assert k_src.stride(-1) == 1 and v_src.stride(-1) == 1
    assert k_dst.stride(-1) == 1 and v_dst.stride(-1) == 1

    k_src_stride = k_src.stride(0) if k_src.dim() > 1 else k_src.shape[-1]
    v_src_stride = v_src.stride(0) if v_src.dim() > 1 else v_src.shape[-1]
    k_dst_stride = k_dst.stride(0) if k_dst.dim() > 1 else k_dst.shape[-1]
    v_dst_stride = v_dst.stride(0) if v_dst.dim() > 1 else v_dst.shape[-1]

    block = triton.next_power_of_2(n_kv_k)
    use_pdl = bool(enable_pdl and _is_nvidia)
    kwargs = {}
    if use_pdl:
        kwargs["launch_pdl"] = True
    _store_kv_cache_kernel[(2, n_tokens)](
        k_src,
        v_src,
        k_dst,
        v_dst,
        loc,
        k_src_stride,
        v_src_stride,
        k_dst_stride,
        v_dst_stride,
        n_kv_k,
        BLOCK=block,
        ENABLE_PDL=use_pdl,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# FP8 KV Cache Write
# -----------------------------------------------------------------------------


@triton.jit
def _process_fp8_kv_tensor(
    token_id,
    head_block_id,
    page_id,
    page_offset,
    input_ptr,
    cache_ptr,
    inv_scale,
    use_provided_scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    input_stride_token: tl.constexpr,
    input_stride_head: tl.constexpr,
    input_stride_dim: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    head_idx = head_block_id * BLOCK_HEAD
    num_heads_in_block = min(BLOCK_HEAD, num_kv_heads - head_idx)

    for dim_idx in range(0, head_dim, BLOCK_DIM):
        num_dims_in_block = min(BLOCK_DIM, head_dim - dim_idx)

        head_offsets = head_idx + tl.arange(0, BLOCK_HEAD)
        dim_offsets = dim_idx + tl.arange(0, BLOCK_DIM)

        head_mask = head_offsets < (head_idx + num_heads_in_block)
        dim_mask = dim_offsets < (dim_idx + num_dims_in_block)
        mask = head_mask[:, None] & dim_mask[None, :]

        input_offsets = (
            token_id * input_stride_token
            + head_offsets[:, None] * input_stride_head
            + dim_offsets[None, :] * input_stride_dim
        )
        block = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

        if use_provided_scale:
            block_fp8 = (block * inv_scale).to(tl.float8e4nv)
        else:
            block_fp8 = block.to(tl.float8e4nv)

        cache_offsets = (
            page_id * cache_stride_page
            + page_offset * cache_stride_offset
            + head_offsets[:, None] * cache_stride_head
            + dim_offsets[None, :] * cache_stride_dim
        )
        tl.store(cache_ptr + cache_offsets, block_fp8, mask=mask)


@triton.jit
def _fused_fp8_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_loc_ptr,
    inv_k_scale_ptr,
    inv_v_scale_ptr,
    use_provided_scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    k_cache_stride_page: tl.constexpr,
    k_cache_stride_offset: tl.constexpr,
    k_cache_stride_head: tl.constexpr,
    k_cache_stride_dim: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    v_cache_stride_page: tl.constexpr,
    v_cache_stride_offset: tl.constexpr,
    v_cache_stride_head: tl.constexpr,
    v_cache_stride_dim: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    kv_idx = tl.program_id(2)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    cache_loc = tl.load(cache_loc_ptr + token_id).to(tl.int64)
    page_id = cache_loc // page_size
    page_offset = cache_loc % page_size

    if kv_idx == 0:
        if use_provided_scale:
            inv_scale = tl.load(inv_k_scale_ptr)
        else:
            inv_scale = 1.0
        _process_fp8_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            k_ptr,
            k_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )
    else:
        if use_provided_scale:
            inv_scale = tl.load(inv_v_scale_ptr)
        else:
            inv_scale = 1.0
        _process_fp8_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            v_ptr,
            v_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def fused_fp8_set_kv_buffer(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
    page_size: int = 16,
    enable_pdl: bool = False,
) -> None:
    """Quantize K/V tensors to FP8 and scatter them into a paged KV cache.

    Args:
        k: Key tensor with shape ``[num_tokens, num_kv_heads, head_dim]`` or
            ``[num_tokens, num_kv_heads * head_dim]``.
        v: Value tensor with the same shape convention as ``k``.
        k_cache: Destination K cache, either flattened slots
            ``[total_slots, num_kv_heads, head_dim]`` or paged layout
            ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Destination V cache with the same shape convention as
            ``k_cache``.
        cache_loc: Cache slot index for each input token.
        k_scale: Optional scalar K scale. When provided with ``v_scale``, K is
            divided by this scale before FP8 conversion.
        v_scale: Optional scalar V scale. When provided with ``k_scale``, V is
            divided by this scale before FP8 conversion.
        page_size: Number of tokens per cache page.
        enable_pdl: Request Programmatic Dependent Launch (SM90+); no-op on
            non-NVIDIA platforms.
    """
    num_tokens = k.shape[0]
    if num_tokens == 0:
        return

    if k_cache.ndim == 3:
        total_slots, num_kv_heads, head_dim = k_cache.shape
        assert (
            total_slots % page_size == 0
        ), f"total_slots ({total_slots}) must be divisible by page_size ({page_size})"
    elif k_cache.ndim == 4:
        _, ps, num_kv_heads, head_dim = k_cache.shape
        assert (
            ps == page_size
        ), f"page_size mismatch: cache has {ps}, expected {page_size}"
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}, expected 3 or 4")

    if k.ndim == 3:
        assert (
            k.shape[1] == num_kv_heads
        ), f"num_kv_heads mismatch: k.shape[1]={k.shape[1]} vs cache={num_kv_heads}"
        assert (
            k.shape[2] == head_dim
        ), f"head_dim mismatch: k.shape[2]={k.shape[2]} vs cache={head_dim}"
        assert v.shape[1] == num_kv_heads and v.shape[2] == head_dim, "v shape mismatch"
        k_3d = k
        v_3d = v
    elif k.ndim == 2:
        assert (
            k.shape[1] == num_kv_heads * head_dim
        ), f"k.shape[1]={k.shape[1]} != {num_kv_heads * head_dim}"
        assert (
            v.shape[1] == num_kv_heads * head_dim
        ), f"v.shape[1]={v.shape[1]} != {num_kv_heads * head_dim}"
        k_3d = k.view(num_tokens, num_kv_heads, head_dim)
        v_3d = v.view(num_tokens, num_kv_heads, head_dim)
    else:
        raise ValueError(f"Unsupported k.ndim={k.ndim}, expected 2 or 3")

    if k_cache.ndim == 3:
        k_cache_stride_page = k_cache.stride(0) * page_size
        k_cache_stride_offset = k_cache.stride(0)
        k_cache_stride_head = k_cache.stride(1)
        k_cache_stride_dim = k_cache.stride(2)

        v_cache_stride_page = v_cache.stride(0) * page_size
        v_cache_stride_offset = v_cache.stride(0)
        v_cache_stride_head = v_cache.stride(1)
        v_cache_stride_dim = v_cache.stride(2)
    else:
        k_cache_stride_page = k_cache.stride(0)
        k_cache_stride_offset = k_cache.stride(1)
        k_cache_stride_head = k_cache.stride(2)
        k_cache_stride_dim = k_cache.stride(3)

        v_cache_stride_page = v_cache.stride(0)
        v_cache_stride_offset = v_cache.stride(1)
        v_cache_stride_head = v_cache.stride(2)
        v_cache_stride_dim = v_cache.stride(3)

    use_provided_scale = k_scale is not None and v_scale is not None

    block_head = min(num_kv_heads, 8)
    block_dim = min(head_dim, 128)
    num_head_blocks = (num_kv_heads + block_head - 1) // block_head
    grid = (num_tokens, num_head_blocks, 2)
    device = k_3d.device

    def _to_tensor_scale(scale):
        if isinstance(scale, torch.Tensor):
            return scale.to(device=device, dtype=torch.float32)
        return torch.tensor(float(scale), device=device, dtype=torch.float32)

    if use_provided_scale:
        k_scale_tensor = _to_tensor_scale(k_scale)
        v_scale_tensor = _to_tensor_scale(v_scale)
        inv_k_scale_ptr = (1.0 / k_scale_tensor).to(device=device, dtype=torch.float32)
        inv_v_scale_ptr = (1.0 / v_scale_tensor).to(device=device, dtype=torch.float32)
    else:
        inv_k_scale_ptr = k_3d
        inv_v_scale_ptr = k_3d

    use_pdl = bool(enable_pdl and _is_nvidia)
    kwargs = {}
    if use_pdl:
        kwargs["launch_pdl"] = True

    _fused_fp8_set_kv_buffer_kernel[grid](
        k_3d,
        v_3d,
        k_cache,
        v_cache,
        cache_loc,
        inv_k_scale_ptr,
        inv_v_scale_ptr,
        use_provided_scale,
        num_kv_heads,
        head_dim,
        page_size,
        k_3d.stride(0),
        k_3d.stride(1),
        k_3d.stride(2),
        k_cache_stride_page,
        k_cache_stride_offset,
        k_cache_stride_head,
        k_cache_stride_dim,
        v_3d.stride(0),
        v_3d.stride(1),
        v_3d.stride(2),
        v_cache_stride_page,
        v_cache_stride_offset,
        v_cache_stride_head,
        v_cache_stride_dim,
        BLOCK_HEAD=block_head,
        BLOCK_DIM=block_dim,
        ENABLE_PDL=use_pdl,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Page Table Gather
# -----------------------------------------------------------------------------


@triton.jit
def _gather_page_table_with_padding_kernel(
    req_to_page_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    out_ptr,
    src_stride0,
    out_stride0,
    max_num_pages: tl.constexpr,
    page_size: tl.constexpr,
    dummy_slot: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    sl = tl.load(seq_lens_ptr + pid_row).to(tl.int32)
    n_pages = (sl + page_size - 1) // page_size

    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    in_bounds = col_offsets < max_num_pages
    valid = col_offsets < n_pages

    req_idx = tl.load(req_pool_indices_ptr + pid_row).to(tl.int64)
    src_addr = req_to_page_ptr + req_idx * src_stride0 + col_offsets
    gathered = tl.load(src_addr, mask=valid & in_bounds, other=dummy_slot)

    out_addr = out_ptr + pid_row * out_stride0 + col_offsets
    tl.store(out_addr, gathered, mask=in_bounds)


def gather_page_table_with_padding(
    req_to_page: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    bs: int,
    max_num_pages: int,
    page_size: int,
    dummy_slot: int = 0,
) -> None:
    """Gather active request page tables and clear padding columns.

    Args:
        req_to_page: Source page table with request rows.
        req_pool_indices: Request row indices to gather, shape ``[bs]``.
        seq_lens: Per-request KV lengths, shape ``[bs]``.
        out: Destination page table, shape ``[max_bs, max_num_pages]``.
        bs: Number of active rows to gather.
        max_num_pages: Number of destination page-table columns.
        page_size: Number of tokens per page.
        dummy_slot: Value written into padding columns.
    """
    block_cols = 128
    grid = (bs, triton.cdiv(max_num_pages, block_cols))
    _gather_page_table_with_padding_kernel[grid](
        req_to_page,
        req_pool_indices,
        seq_lens,
        out,
        req_to_page.stride(0),
        out.stride(0),
        max_num_pages,
        page_size,
        dummy_slot,
        BLOCK_COLS=block_cols,
        num_warps=4,
    )


# -----------------------------------------------------------------------------
# KV Cache Transfer
# -----------------------------------------------------------------------------


@triton.jit
def _kv_transfer_per_layer_capped_kernel(
    k_cache_dst_ptr,
    v_cache_dst_ptr,
    indices_dst_ptr,
    k_cache_src_ptr,
    v_cache_src_ptr,
    indices_src_ptr,
    kv_cache_src_stride,
    kv_cache_dst_stride,
    length,
    BLOCK_SIZE: tl.constexpr,
):
    """Grid-capped variant: each program strides over multiple indices."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_SIZE)
    for i in range(pid, length, nprog):
        pos_src = tl.load(indices_src_ptr + i).to(tl.int64)
        pos_dst = tl.load(indices_dst_ptr + i).to(tl.int64)
        src_offset = pos_src * kv_cache_src_stride
        dst_offset = pos_dst * kv_cache_dst_stride
        k_src = tl.load(k_cache_src_ptr + src_offset + offs)
        tl.store(k_cache_dst_ptr + dst_offset + offs, k_src)
        v_src = tl.load(v_cache_src_ptr + src_offset + offs)
        tl.store(v_cache_dst_ptr + dst_offset + offs, v_src)


@triton.jit
def _kv_transfer_per_layer_kernel(
    k_cache_dst_ptr,
    v_cache_dst_ptr,
    indices_dst_ptr,
    k_cache_src_ptr,
    v_cache_src_ptr,
    indices_src_ptr,
    kv_cache_src_stride,
    kv_cache_dst_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transfer KV cache entries for one layer based on src/dst indices.

    Each program handles one index pair (src_idx -> dst_idx) and copies
    BLOCK_SIZE elements at a time.
    """
    pid = tl.program_id(0)

    # Load src and dst positions
    pos_src = tl.load(indices_src_ptr + pid).to(tl.int64)
    pos_dst = tl.load(indices_dst_ptr + pid).to(tl.int64)

    # Calculate base offsets in elements (not bytes, since we use element-based pointers)
    src_offset = pos_src * kv_cache_src_stride
    dst_offset = pos_dst * kv_cache_dst_stride

    # Copy K cache
    offs = tl.arange(0, BLOCK_SIZE)
    k_src = tl.load(k_cache_src_ptr + src_offset + offs)
    tl.store(k_cache_dst_ptr + dst_offset + offs, k_src)

    # Copy V cache
    v_src = tl.load(v_cache_src_ptr + src_offset + offs)
    tl.store(v_cache_dst_ptr + dst_offset + offs, v_src)


@triton.jit
def _kv_transfer_all_layer_kernel(
    k_ptr_dst_ptr,
    v_ptr_dst_ptr,
    indices_dst_ptr,
    k_ptr_src_ptr,
    v_ptr_src_ptr,
    indices_src_ptr,
    length,
    num_layers: tl.constexpr,
    kv_cache_src_stride_words,
    kv_cache_dst_stride_words,
    total_words,
    WORDS_PER_CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """
    Transfer KV cache entries for all layers based on src/dst indices.

    Mirror the JIT kernel's execution model: each program iterates over index
    pairs and copies all layers for that pair in 128-byte chunks.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    word_offsets = tl.arange(0, WORDS_PER_CHUNK)

    for idx in range(pid, length, num_programs):
        pos_src = tl.load(indices_src_ptr + idx).to(tl.int64)
        pos_dst = tl.load(indices_dst_ptr + idx).to(tl.int64)
        src_slot_offset = pos_src * kv_cache_src_stride_words
        dst_slot_offset = pos_dst * kv_cache_dst_stride_words

        for layer in range(num_layers):
            k_cache_src_ptr = tl.load(k_ptr_src_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )
            v_cache_src_ptr = tl.load(v_ptr_src_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )
            k_cache_dst_ptr = tl.load(k_ptr_dst_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )
            v_cache_dst_ptr = tl.load(v_ptr_dst_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )

            for chunk in range(NUM_CHUNKS):
                chunk_offsets = chunk * WORDS_PER_CHUNK + word_offsets
                mask = chunk_offsets < total_words
                src_offsets = src_slot_offset + chunk_offsets
                dst_offsets = dst_slot_offset + chunk_offsets
                src_offsets = tl.max_contiguous(
                    tl.multiple_of(src_offsets, 4), WORDS_PER_CHUNK
                )
                dst_offsets = tl.max_contiguous(
                    tl.multiple_of(dst_offsets, 4), WORDS_PER_CHUNK
                )

                k_src = tl.load(
                    k_cache_src_ptr + src_offsets,
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                )
                v_src = tl.load(
                    v_cache_src_ptr + src_offsets,
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                )
                tl.store(
                    k_cache_dst_ptr + dst_offsets,
                    k_src,
                    mask=mask,
                    cache_modifier=".cs",
                )
                tl.store(
                    v_cache_dst_ptr + dst_offsets,
                    v_src,
                    mask=mask,
                    cache_modifier=".cs",
                )


@triton.jit
def _load_cs_u32(ptrs):
    return tl.inline_asm_elementwise(
        "ld.global.cs.b32 $0, [$1];",
        "=r,l",
        [ptrs],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_cs_u32(values, ptrs):
    return tl.inline_asm_elementwise(
        "st.global.cs.b32 [$2], $1; mov.b32 $0, $1;",
        "=r,r,l",
        [values, ptrs],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _kv_transfer_all_layer_cs32_kernel(
    k_ptr_dst_ptr,
    v_ptr_dst_ptr,
    indices_dst_ptr,
    k_ptr_src_ptr,
    v_ptr_src_ptr,
    indices_src_ptr,
    length,
    num_layers: tl.constexpr,
    kv_cache_src_stride_words,
    kv_cache_dst_stride_words,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane_offsets = tl.arange(0, 32)

    for idx in range(pid, length, num_programs):
        pos_src = tl.load(indices_src_ptr + idx).to(tl.int64)
        pos_dst = tl.load(indices_dst_ptr + idx).to(tl.int64)
        src_slot_offset = pos_src * kv_cache_src_stride_words
        dst_slot_offset = pos_dst * kv_cache_dst_stride_words

        for layer in range(num_layers):
            k_cache_src_ptr = tl.load(k_ptr_src_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )
            v_cache_src_ptr = tl.load(v_ptr_src_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )
            k_cache_dst_ptr = tl.load(k_ptr_dst_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )
            v_cache_dst_ptr = tl.load(v_ptr_dst_ptr + layer).to(
                tl.pointer_type(tl.uint32)
            )

            for chunk in range(NUM_CHUNKS):
                chunk_offsets = chunk * 32 + lane_offsets
                src_offsets = src_slot_offset + chunk_offsets
                dst_offsets = dst_slot_offset + chunk_offsets
                k_src = _load_cs_u32(k_cache_src_ptr + src_offsets)
                v_src = _load_cs_u32(v_cache_src_ptr + src_offsets)
                _store_cs_u32(k_src, k_cache_dst_ptr + dst_offsets)
                _store_cs_u32(v_src, v_cache_dst_ptr + dst_offsets)


def _next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


def _recommended_program_count(
    *,
    length: int,
    element_size: int,
    num_layers: int,
    device: torch.device,
) -> int:
    # Each program copies one indexed token across all layers, so the amount of
    # work scales with both slot size and layer count.
    bytes_per_index = element_size * num_layers * 2
    if bytes_per_index <= 16 * 1024:
        programs_per_sm = 8
    elif bytes_per_index <= 64 * 1024:
        programs_per_sm = 4
    else:
        programs_per_sm = 2

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, min(length, sm_count * programs_per_sm))


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
) -> None:
    """
    Transfer KV cache entries for one layer based on src/dst indices.

    Args:
        src_k: Source K cache tensor [num_slots, num_heads, head_dim]
        dst_k: Destination K cache tensor [num_slots, num_heads, head_dim]
        src_v: Source V cache tensor [num_slots, num_heads, head_dim]
        dst_v: Destination V cache tensor [num_slots, num_heads, head_dim]
        src_indices: Source indices tensor [length]
        dst_indices: Destination indices tensor [length]
        item_size: Number of bytes per cache slot
    """
    if item_size % src_k.element_size() != 0:
        raise ValueError("item_size must be divisible by the KV cache element size.")
    element_dim = item_size // src_k.element_size()

    length = src_indices.numel()
    if length == 0:
        return

    # Flatten to 2D view: [num_slots, element_dim]
    k_cache_src_flat = src_k.view(-1, element_dim)
    v_cache_src_flat = src_v.view(-1, element_dim)
    k_cache_dst_flat = dst_k.view(-1, element_dim)
    v_cache_dst_flat = dst_v.view(-1, element_dim)

    # Strides in elements
    kv_cache_src_stride = k_cache_src_flat.stride(0)
    kv_cache_dst_stride = k_cache_dst_flat.stride(0)

    # BLOCK_SIZE is in elements, must be power of two and cover element_dim
    block_size = _next_power_of_two(element_dim)

    cap = _PER_LAYER_GRID_CAP
    if cap > 0 and length > cap:
        _kv_transfer_per_layer_capped_kernel[(cap,)](
            k_cache_dst_flat,
            v_cache_dst_flat,
            dst_indices,
            k_cache_src_flat,
            v_cache_src_flat,
            src_indices,
            kv_cache_src_stride,
            kv_cache_dst_stride,
            length,
            BLOCK_SIZE=block_size,
        )
        return

    grid = (length,)
    _kv_transfer_per_layer_kernel[grid](
        k_cache_dst_flat,
        v_cache_dst_flat,
        dst_indices,
        k_cache_src_flat,
        v_cache_src_flat,
        src_indices,
        kv_cache_src_stride,
        kv_cache_dst_stride,
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _kv_transfer_per_layer_mla_kernel(
    cache_dst_ptr,
    indices_dst_ptr,
    cache_src_ptr,
    indices_src_ptr,
    cache_src_stride,
    cache_dst_stride,
    ELEMENT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    pos_src = tl.load(indices_src_ptr + pid).to(tl.int64)
    pos_dst = tl.load(indices_dst_ptr + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < ELEMENT_DIM

    src = tl.load(cache_src_ptr + pos_src * cache_src_stride + offs, mask=mask)
    tl.store(cache_dst_ptr + pos_dst * cache_dst_stride + offs, src, mask=mask)


@triton.jit
def _kv_transfer_all_layer_mla_kernel(
    ptr_dst_ptr,
    indices_dst_ptr,
    ptr_src_ptr,
    indices_src_ptr,
    length,
    num_layers: tl.constexpr,
    cache_src_stride_words,
    cache_dst_stride_words,
    total_words,
    WORDS_PER_CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    word_offsets = tl.arange(0, WORDS_PER_CHUNK)

    for idx in range(pid, length, num_programs):
        pos_src = tl.load(indices_src_ptr + idx).to(tl.int64)
        pos_dst = tl.load(indices_dst_ptr + idx).to(tl.int64)
        src_slot_offset = pos_src * cache_src_stride_words
        dst_slot_offset = pos_dst * cache_dst_stride_words

        for layer in range(num_layers):
            cache_src_ptr = tl.load(ptr_src_ptr + layer).to(tl.pointer_type(tl.uint32))
            cache_dst_ptr = tl.load(ptr_dst_ptr + layer).to(tl.pointer_type(tl.uint32))

            for chunk in range(NUM_CHUNKS):
                chunk_offsets = chunk * WORDS_PER_CHUNK + word_offsets
                mask = chunk_offsets < total_words
                src_offsets = src_slot_offset + chunk_offsets
                dst_offsets = dst_slot_offset + chunk_offsets
                src_offsets = tl.max_contiguous(
                    tl.multiple_of(src_offsets, 4), WORDS_PER_CHUNK
                )
                dst_offsets = tl.max_contiguous(
                    tl.multiple_of(dst_offsets, 4), WORDS_PER_CHUNK
                )

                src = tl.load(
                    cache_src_ptr + src_offsets,
                    mask=mask,
                    other=0,
                    cache_modifier=".cg",
                )
                tl.store(
                    cache_dst_ptr + dst_offsets,
                    src,
                    mask=mask,
                    cache_modifier=".cs",
                )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int | None = None,
) -> None:
    del block_quota

    if item_size % src.element_size() != 0:
        raise ValueError("item_size must be divisible by the MLA cache element size.")
    element_dim = item_size // src.element_size()

    length = src_indices.numel()
    if length == 0:
        return

    cache_src_flat = src.view(-1, element_dim)
    cache_dst_flat = dst.view(-1, element_dim)
    block_size = _next_power_of_two(element_dim)

    _kv_transfer_per_layer_mla_kernel[(length,)](
        cache_dst_flat,
        dst_indices,
        cache_src_flat,
        src_indices,
        cache_src_flat.stride(0),
        cache_dst_flat.stride(0),
        ELEMENT_DIM=element_dim,
        BLOCK_SIZE=block_size,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int | None = None,
) -> None:
    del block_quota

    length = src_indices.numel()
    if length == 0:
        return

    if item_size % 4 != 0:
        raise ValueError(
            "Triton MLA all-layer kernel requires item_size to be a multiple of "
            "4 bytes."
        )

    words_per_chunk = 32
    total_words = item_size // 4
    num_chunks = triton.cdiv(total_words, words_per_chunk)
    grid = (
        _recommended_program_count(
            length=length,
            element_size=item_size,
            num_layers=num_layers,
            device=src_indices.device,
        ),
    )
    _kv_transfer_all_layer_mla_kernel[grid](
        dst_layers,
        dst_indices,
        src_layers,
        src_indices,
        length,
        num_layers=num_layers,
        cache_src_stride_words=item_size // 4,
        cache_dst_stride_words=item_size // 4,
        total_words=total_words,
        WORDS_PER_CHUNK=words_per_chunk,
        NUM_CHUNKS=num_chunks,
        num_warps=1,
        num_stages=1,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
) -> None:
    """
    Transfer KV cache entries for all layers based on src/dst indices.

    Args:
        src_k_layers: Tensor of source K cache pointers per layer [num_layers]
        dst_k_layers: Tensor of destination K cache pointers per layer [num_layers]
        src_v_layers: Tensor of source V cache pointers per layer [num_layers]
        dst_v_layers: Tensor of destination V cache pointers per layer [num_layers]
        src_indices: Source indices tensor [length]
        dst_indices: Destination indices tensor [length]
        item_size: Number of bytes per cache slot
        num_layers: Number of layers to copy
    """
    length = src_indices.numel()

    if length == 0:
        return

    if item_size % 4 != 0:
        raise ValueError(
            "Triton KV cache all-layer kernel requires item_size to be a multiple of 4 bytes."
        )

    words_per_chunk = 32
    total_words = item_size // 4
    num_chunks = triton.cdiv(total_words, words_per_chunk)
    num_programs = _recommended_program_count(
        length=length,
        element_size=item_size,
        num_layers=num_layers,
        device=src_indices.device,
    )
    if _ALL_LAYER_GRID_CAP > 0:
        num_programs = min(num_programs, _ALL_LAYER_GRID_CAP)
    grid = (num_programs,)
    if _is_nvidia and total_words % words_per_chunk == 0:
        _kv_transfer_all_layer_cs32_kernel[grid](
            dst_k_layers,
            dst_v_layers,
            dst_indices,
            src_k_layers,
            src_v_layers,
            src_indices,
            length,
            num_layers=num_layers,
            kv_cache_src_stride_words=item_size // 4,
            kv_cache_dst_stride_words=item_size // 4,
            NUM_CHUNKS=num_chunks,
            num_warps=1,
            num_stages=1,
        )
        return

    _kv_transfer_all_layer_kernel[grid](
        dst_k_layers,
        dst_v_layers,
        dst_indices,
        src_k_layers,
        src_v_layers,
        src_indices,
        length,
        num_layers=num_layers,
        kv_cache_src_stride_words=item_size // 4,
        kv_cache_dst_stride_words=item_size // 4,
        total_words=total_words,
        WORDS_PER_CHUNK=words_per_chunk,
        NUM_CHUNKS=num_chunks,
        num_warps=1,
        num_stages=1,
    )


# -----------------------------------------------------------------------------
# Fused MXFP8 quantize + KV store + SF scatter (one launch per decode store)
# -----------------------------------------------------------------------------


@triton.jit
def _quantize_store_kv_mxfp8_kernel(
    k_src_ptr,  # [T, H*D] bf16
    v_src_ptr,
    k_dst_ptr,  # fp8 slab rows as u8, row = loc
    v_dst_ptr,
    k_sf_ptr,  # SF slabs as u32 (4 packed e8m0), interleaved atom layout
    v_sf_ptr,
    loc_ptr,
    k_src_token_stride,
    v_src_token_stride,
    k_dst_row_stride,
    v_dst_row_stride,
    sf_page_stride,  # nheads * chunks_per_page * 128 (u32 units)
    page_tokens,  # tokens per page of this layer's group
    nheads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  # 128
    ENABLE_PDL: tl.constexpr,
):
    """Quantize one token's K or V row to MXFP8 and store data + scales.

    Replaces the five-launch sequence (k/v quantize_mxfp8, store_kv_cache,
    2x store_sf_interleaved) with one launch. Bit-parity contract with
    flashinfer's mxfp8_quantize: per 32-element group,
    ``e8m0 = clamp(ceil(log2(amax / 448)), -127, 127) + 127`` and
    ``fp8 = rn(x * 2^-exp)`` (zero rows quantize to exponent -127, data 0).
    SF layout matches _store_sf_interleaved_kernel: page-major, per-head
    chunks_per_page consecutive 128-row BlockScaledBasicChunk atoms,
    row -> (row % 32) * 4 + row // 32, 4 head_dim-group bytes packed
    little-endian in one u32.
    """
    is_v = tl.program_id(0)
    tok = tl.program_id(1)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    slot = tl.load(loc_ptr + tok).to(tl.int64)
    d_off = tl.arange(0, HEAD_DIM)  # one head row at a time

    chunks_per_page = page_tokens // 128
    sf_base = _sf_interleaved_offset(slot, page_tokens, sf_page_stride)

    for h in tl.static_range(nheads):
        if is_v == 1:
            x = tl.load(v_src_ptr + tok * v_src_token_stride + h * HEAD_DIM + d_off)
        else:
            x = tl.load(k_src_ptr + tok * k_src_token_stride + h * HEAD_DIM + d_off)
        q8, packed = _mxfp8_quantize_row(x, HEAD_DIM)
        if is_v == 1:
            tl.store(v_dst_ptr + slot * v_dst_row_stride + h * HEAD_DIM + d_off, q8)
        else:
            tl.store(k_dst_ptr + slot * k_dst_row_stride + h * HEAD_DIM + d_off, q8)
        # Scatter the packed-SF u32 into the interleaved slab.
        sf_out_off = sf_base + h * chunks_per_page * 128
        if is_v == 1:
            tl.store(v_sf_ptr + sf_out_off, packed)
        else:
            tl.store(k_sf_ptr + sf_out_off, packed)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def quantize_store_kv_mxfp8(
    k: torch.Tensor,
    v: torch.Tensor,
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    loc: torch.Tensor,
    page_tokens: int = 128,
    enable_pdl: bool = False,
) -> None:
    """Fused per-token MXFP8 quantize + KV data store + interleaved SF store.

    Args:
        k, v: Per-token bf16 rows, [T, H, 128] or [T, H * 128]; the leading
            axis may be strided, the trailing element stride must be 1.
        k_dst, v_dst: fp8-e4m3 slab row views (the same per-layer views
            ``set_kv_buffer`` targets); rows are addressed by ``loc``.
        k_sf, v_sf: e8m0 scale slabs in the interleaved atom layout of
            ``store_sf_interleaved`` for this layer's ``page_tokens``.
        loc: [T] destination row per token (layer-view row units).
        page_tokens: Tokens per page of the layer's group (multiple of 128).
        enable_pdl: Launch with Programmatic Dependent Launch (Hopper+).
    """
    assert page_tokens % 128 == 0
    t = k.shape[0]
    if t == 0:
        return
    head_dim = 128
    nheads = k.numel() // (t * head_dim)
    assert k.stride(-1) == 1 and v.stride(-1) == 1
    k2 = k.reshape(t, nheads * head_dim)
    v2 = v.reshape(t, nheads * head_dim)
    k_dst_u8 = k_dst.reshape(k_dst.shape[0], -1).view(torch.uint8)
    v_dst_u8 = v_dst.reshape(v_dst.shape[0], -1).view(torch.uint8)
    k_sf_u32 = k_sf.view(torch.uint8).reshape(-1, 4).view(torch.int32).reshape(-1)
    v_sf_u32 = v_sf.view(torch.uint8).reshape(-1, 4).view(torch.int32).reshape(-1)
    chunks_per_page = page_tokens // 128
    sf_page_stride = nheads * chunks_per_page * 128

    grid = (2, t)
    use_pdl = bool(enable_pdl and _is_nvidia)
    kwargs = {}
    if use_pdl:
        kwargs["launch_pdl"] = True
    _quantize_store_kv_mxfp8_kernel[grid](
        k2,
        v2,
        k_dst_u8,
        v_dst_u8,
        k_sf_u32,
        v_sf_u32,
        loc,
        k2.stride(0),
        v2.stride(0),
        k_dst_u8.stride(0),
        v_dst_u8.stride(0),
        sf_page_stride,
        page_tokens,
        nheads=nheads,
        HEAD_DIM=head_dim,
        ENABLE_PDL=use_pdl,
        **kwargs,
    )


@triton.jit
def _quantize_mxfp8_rows_kernel(
    x_ptr,  # [R, 128] bf16 rows (R = tokens * heads)
    data_ptr,  # [R, 128] u8 (fp8-e4m3 storage)
    sf_ptr,  # [R] u32 (4 packed e8m0 bytes)
    x_row_stride,
    HEAD_DIM: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    R,
    ENABLE_PDL: tl.constexpr,
):
    """Per-row MXFP8 quantize (bit-parity with flashinfer mxfp8_quantize),
    PDL-capable so it keeps the qk_rmsnorm -> shear -> fwd chain intact."""
    pid = tl.program_id(0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    d_off = tl.arange(0, HEAD_DIM)
    for i in tl.static_range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + i
        if row < R:
            x = tl.load(x_ptr + row * x_row_stride + d_off)
            q8, packed = _mxfp8_quantize_row(x, HEAD_DIM)
            tl.store(data_ptr + row * HEAD_DIM + d_off, q8)
            tl.store(sf_ptr + row, packed)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def quantize_mxfp8_rows(
    x: torch.Tensor,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantize [R, 128] rows: (fp8-e4m3 [R, 128], e8m0 [R, 4]).

    PDL-capable row quantizer intended for the decode-Q fusion follow-up
    (inkling_mxfp8_attn.md); parity-tested against flashinfer's
    mxfp8_quantize, no runtime caller yet.
    """
    r, d = x.shape
    assert d == 128 and x.stride(-1) == 1
    data = torch.empty(r, d, dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty(r, 4, dtype=torch.uint8, device=x.device)
    if r == 0:
        return data, sf
    rows_per_prog = 4
    grid = ((r + rows_per_prog - 1) // rows_per_prog,)
    use_pdl = bool(enable_pdl and _is_nvidia)
    kwargs = {}
    if use_pdl:
        kwargs["launch_pdl"] = True
    _quantize_mxfp8_rows_kernel[grid](
        x,
        data.view(torch.uint8),
        sf.view(torch.int32).reshape(-1),
        x.stride(0),
        HEAD_DIM=d,
        ROWS_PER_PROG=rows_per_prog,
        R=r,
        ENABLE_PDL=use_pdl,
        **kwargs,
    )
    return data, sf


@triton.jit
def _flat_decode_locs_kernel(
    tab_ptr,  # [G, max_bs, Wmax] int32 stacked group tables
    ps_ptr,  # [G] int32 page size per group
    seq_ptr,  # [bs] int32 current lengths (incl. the newest token)
    out_ptr,  # [G, >= bs*N] int32 write locs (token-major per request)
    stride_g,
    stride_b,
    out_stride_g,
    num_rows,  # bs * N live output rows per group
    N,  # tokens per request
    BLOCK_B: tl.constexpr,
):
    """All groups' decode write locs in one launch. Row i = b*N + t maps to
    position seq[b] - N + t (clamped at 0 for graph-padded rows, which
    dereference the dummy page harmlessly): loc = table[g, b, pos//ps] * ps
    + pos % ps. N = 1 is plain decode (pos = seq-1); N > 1 is the spec
    verify layout. Replaces the per-group python gather/mul/mod chains
    between graph replays."""
    g = tl.program_id(0)
    i = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = i < num_rows
    b = i // N
    t = i - b * N
    ps = tl.load(ps_ptr + g).to(tl.int64)
    seq = tl.load(seq_ptr + b, mask=mask, other=1).to(tl.int64)
    pos = tl.maximum(seq - N + t, 0)
    col = pos // ps
    page = tl.load(tab_ptr + g * stride_g + b * stride_b + col, mask=mask, other=0).to(
        tl.int64
    )
    # Negative pages (-1 column-tail pad / holes) route to dummy page 0, never a negative slot.
    loc = tl.maximum(page, 0) * ps + pos % ps
    tl.store(out_ptr + g * out_stride_g + i, loc.to(tl.int32), mask=mask)


def flat_decode_locs(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    bs: int,
    tokens_per_req: int = 1,
) -> None:
    """Fused per-group decode write-loc computation over stacked tables.

    Args:
        tables: [G, max_bs, Wmax] int32 stacked per-group page tables.
        page_sizes: [G] int32 tokens-per-page per group.
        seq_lens: [bs] int32 lengths including the newest token.
        out: [G, cap] int32 destination, cap >= bs * tokens_per_req; rows
            [:, : bs * tokens_per_req] written token-major per request
            (request b's tokens at b*N .. b*N + N - 1, the spec verify
            layout).
        bs: live batch size.
        tokens_per_req: write locs per request; token t of request b lands
            at position seq_lens[b] - tokens_per_req + t (clamped at 0).
    """
    g = tables.shape[0]
    num_rows = bs * tokens_per_req
    assert out.shape[0] >= g and out.shape[1] >= num_rows
    BLOCK_B = 128
    grid = (g, (num_rows + BLOCK_B - 1) // BLOCK_B)
    _flat_decode_locs_kernel[grid](
        tables,
        page_sizes,
        seq_lens,
        out,
        tables.stride(0),
        tables.stride(1),
        out.stride(0),
        num_rows,
        tokens_per_req,
        BLOCK_B=BLOCK_B,
    )


@triton.jit
def _flat_tables_unpack_kernel(
    src_ptr,  # packed int32 device buffer (bridge upload)
    meta_ptr,  # [G, 2] int32: (src element offset, cols) per group
    dst_ptr,  # [G, max_bs, Wmax] int32 stacked graph tables
    stride_g,
    stride_b,
    wmax,
    actual_bs,
    TAIL_PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Fill every group's graph-table rows from the packed upload in one
    launch: row (g, b < actual_bs) gets src[off_g + b*cols_g : +cols_g]
    followed by a TAIL_PAD tail up to Wmax; padded rows b >= actual_bs
    (grid axis 1 covers the padded batch size) are written all-0 (the
    flat dummy-page row contract). Replaces the per-group D2D copy +
    tail fill (+ F.pad row padding) per decode step."""
    g = tl.program_id(0)
    b = tl.program_id(1)  # grid axis 1 is exactly bs, no bounds check needed
    off = tl.load(meta_ptr + g * 2).to(tl.int64)
    cols = tl.load(meta_ptr + g * 2 + 1).to(tl.int64)
    w_off = tl.arange(0, BLOCK_W)
    real = b < actual_bs
    for w0 in range(0, wmax, BLOCK_W):
        w = w0 + w_off
        in_row = (w < cols) & real
        vals = tl.load(src_ptr + off + b * cols + w, mask=in_row & (w < wmax), other=0)
        vals = tl.where(in_row, vals, tl.where(real, TAIL_PAD, 0))
        tl.store(
            dst_ptr + g * stride_g + b * stride_b + w,
            vals,
            mask=w < wmax,
        )


@triton.jit
def _flat_tables_unpack_ragged_kernel(
    src_ptr,  # packed int32 device buffer (header + active table/base payload)
    meta_ptr,  # [G, 6]: src table off/cols, dst table off/cols, src/dst base off
    dst_ptr,  # exact-width packed graph table/base destination
    actual_bs,
    TAIL_PAD: tl.constexpr,
    MAX_COLS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Unpack one group's table and base vector per program row."""
    g = tl.program_id(0)
    b = tl.program_id(1)
    row = meta_ptr + g * 6
    src_table_off = tl.load(row).to(tl.int64)
    src_cols = tl.load(row + 1).to(tl.int64)
    dst_table_off = tl.load(row + 2).to(tl.int64)
    dst_cols = tl.load(row + 3).to(tl.int64)
    src_base_off = tl.load(row + 4).to(tl.int64)
    dst_base_off = tl.load(row + 5).to(tl.int64)
    real = b < actual_bs
    w_off = tl.arange(0, BLOCK_W)
    for w0 in range(0, MAX_COLS, BLOCK_W):
        w = w0 + w_off
        in_row = real & (w < src_cols)
        vals = tl.load(
            src_ptr + src_table_off + b * src_cols + w,
            mask=in_row & (w < dst_cols),
            other=0,
        )
        vals = tl.where(in_row, vals, tl.where(real, TAIL_PAD, 0))
        tl.store(
            dst_ptr + dst_table_off + b * dst_cols + w,
            vals,
            mask=w < dst_cols,
        )
    base = tl.load(src_ptr + src_base_off + b, mask=real, other=0)
    tl.store(dst_ptr + dst_base_off + b, base)


def flat_tables_unpack(
    src: torch.Tensor,
    meta: torch.Tensor,
    dst: torch.Tensor,
    bs: int,
    actual_bs: int | None = None,
    tail_pad: int = -1,
    *,
    max_cols: int | None = None,
) -> None:
    """Unpack one packed table upload into graph buffers in one launch.

    Args:
        src: 1-D int32 device buffer holding every group's rows
            back-to-back (rows x cols per group).
        meta: ``[G, 2]`` source offset/width rows for a stacked destination,
            or ``[G, 6]`` source/destination table offsets/widths plus
            source/destination base offsets for an exact-width packed
            destination.
        dst: ``[G, max_bs, Wmax]`` stacked destination, or one exact-width
            packed 1-D table/base destination.
        bs: rows to fill per group (padded batch size).
        actual_bs: live batch size; rows [actual_bs, bs) are written all-0
            (the flat dummy-page row contract). Defaults to bs (no padded
            rows).
        tail_pad: value for columns past the group's width.
        max_cols: Maximum destination table width. Required for a packed
            1-D destination and ignored for a stacked destination.
    """
    if isinstance(bs, bool) or not isinstance(bs, int) or bs < 0:
        raise ValueError("flat unpack bs must be an integer >= 0")
    if actual_bs is None:
        actual_bs = bs
    if (
        isinstance(actual_bs, bool)
        or not isinstance(actual_bs, int)
        or actual_bs < 0
        or actual_bs > bs
    ):
        raise ValueError("flat unpack actual_bs must be an integer in [0, bs]")
    if dst.ndim == 1:
        if meta.ndim != 2 or meta.shape[1] != 6:
            raise ValueError("ragged flat unpack meta must have shape [G, 6]")
        if isinstance(max_cols, bool) or not isinstance(max_cols, int) or max_cols <= 0:
            raise ValueError("ragged flat unpack requires max_cols > 0")
        g = int(meta.shape[0])
        if bs == 0 or g == 0:
            return
        BLOCK_W = 128 if max_cols >= 128 else 64
        _flat_tables_unpack_ragged_kernel[(g, bs)](
            src,
            meta,
            dst,
            actual_bs,
            TAIL_PAD=tail_pad,
            MAX_COLS=max_cols,
            BLOCK_W=BLOCK_W,
        )
        return
    if dst.ndim != 3:
        raise ValueError("flat unpack destination must be rank 1 or rank 3")
    if meta.ndim != 2 or meta.shape[1] != 2:
        raise ValueError("stacked flat unpack meta must have shape [G, 2]")
    g, _, wmax = dst.shape
    if int(meta.shape[0]) != g:
        raise ValueError(
            "stacked flat unpack group count mismatch: "
            f"meta={int(meta.shape[0])}, destination={g}"
        )
    if bs > int(dst.shape[1]):
        raise ValueError(
            "stacked flat unpack row count exceeds destination: "
            f"bs={bs}, destination={int(dst.shape[1])}"
        )
    if bs == 0 or g == 0:
        return
    BLOCK_W = 128 if wmax >= 128 else 64
    grid = (g, bs)
    _flat_tables_unpack_kernel[grid](
        src,
        meta,
        dst,
        dst.stride(0),
        dst.stride(1),
        wmax,
        actual_bs,
        TAIL_PAD=tail_pad,
        BLOCK_W=BLOCK_W,
    )


# GLM-5 DSA block-split index-K scatter


@triton.jit
def _index_k_scatter_kernel(
    fp8_buf_ptr,  # uint8 flat view of buf
    scale_buf_ptr,  # float32 flat view of buf (aliases fp8_buf_ptr)
    k_fp8_ptr,  # uint8 [tokens, HD]
    k_scale_ptr,  # float32 [tokens, NG]
    loc_ptr,  # int [tokens] global slot index (non-negative)
    page_bytes,  # fp8 elements per page
    scale_page_off,  # float32 elements per page (page_bytes // 4)
    scale_base_off,  # float32 offset of the scale region ((ps*hd)//4)
    PAGE_SIZE: tl.constexpr,
    HD: tl.constexpr,
    NG: tl.constexpr,
    BLOCK_HD: tl.constexpr,  # next_pow2(HD); masked so HD need not be pow2
    BLOCK_NG: tl.constexpr,  # next_pow2(NG)
):
    t = tl.program_id(0)
    # loc >= 0 makes // and % exact.
    loc = tl.load(loc_ptr + t).to(tl.int64)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    d = tl.arange(0, BLOCK_HD)
    hd_mask = d < HD
    fp8_dst = page * page_bytes + slot * HD + d
    tl.store(
        fp8_buf_ptr + fp8_dst,
        tl.load(k_fp8_ptr + t * HD + d, mask=hd_mask),
        mask=hd_mask,
    )

    g = tl.arange(0, BLOCK_NG)
    ng_mask = g < NG
    sc_dst = scale_base_off + page * scale_page_off + slot * NG + g
    tl.store(
        scale_buf_ptr + sc_dst,
        tl.load(k_scale_ptr + t * NG + g, mask=ng_mask),
        mask=ng_mask,
    )


def index_k_block_split_scatter(
    buf: torch.Tensor,
    index_k_fp8: torch.Tensor,
    index_k_scale: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    head_dim: int,
    group_size: int,
) -> None:
    """Scatter FP8 index-K rows + scales into the block-split paged buffer.

    Byte-exact single-launch equivalent of two ``index_put`` writes through
    the block-split ``as_strided`` views (see
    ``DSATokenToKVPool._index_k_block_views``). The ``(page, slot_in_page) =
    (loc // page_size, loc % page_size)`` mapping is derived per token inside
    the kernel, so callers pass raw cache locations.

    Args:
        buf: uint8 ``[num_slots, head_dim + num_groups*4]`` packed buffer.
        index_k_fp8: ``[tokens, head_dim]`` FP8 values.
        index_k_scale: ``[tokens, num_groups]`` float32 scales.
        loc: ``[tokens]`` non-negative int global slot indices (any integer
            dtype).
        page_size, head_dim, group_size: layout; ``num_groups = head_dim //
            group_size``.

    Returns:
        None; ``buf`` is written in place.
    """
    tokens = index_k_fp8.shape[0]
    if tokens == 0:
        return
    ng = head_dim // group_size
    row_bytes = head_dim + ng * 4
    page_bytes = page_size * row_bytes

    fp8_buf = buf.reshape(-1)  # uint8
    scale_buf = fp8_buf.view(torch.float32)  # aliases the same storage
    k_fp8 = index_k_fp8.reshape(-1, head_dim).contiguous().view(torch.uint8)
    k_scale = index_k_scale.reshape(-1, ng).contiguous()

    _index_k_scatter_kernel[(tokens,)](
        fp8_buf,
        scale_buf,
        k_fp8,
        k_scale,
        loc.reshape(-1),
        page_bytes,
        page_bytes // 4,
        (page_size * head_dim) // 4,
        PAGE_SIZE=page_size,
        HD=head_dim,
        NG=ng,
        BLOCK_HD=_next_power_of_two(head_dim),
        BLOCK_NG=_next_power_of_two(ng),
    )
