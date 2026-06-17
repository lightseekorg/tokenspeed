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
    "gather_page_table_with_padding",
    "store_kv_cache",
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_mla",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_mla",
]


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
):
    """Scatter rows of k_src/v_src into k_dst/v_dst at indices loc_ptr.

    Stride-aware: leading axis of src/dst can have any stride; the only
    requirement is ``stride(-1) == 1`` so we can use linear addressing on
    the flattened head_dim×num_kv_heads axis.
    """
    is_v = tl.program_id(0)
    row = tl.program_id(1)

    dst_row = tl.load(loc_ptr + row).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_kv_per_token

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


def store_kv_cache(
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    """Fused per-token KV cache scatter for one layer.

    Replaces ``k_dst[loc] = k_src; v_dst[loc] = v_src`` with a single triton
    launch handling both k and v rows. The last dim of all four tensors must
    be contiguous (stride == 1); the leading axis may have any stride — this
    lets src tensors come from a qkv-split view directly (no contiguous copy
    required).
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
    )


# -----------------------------------------------------------------------------
# FP8 KV Cache Write
# -----------------------------------------------------------------------------


# Adapted from meituan-longcat/SGLang-FluentLLM. This code may incorporate
# material from ModelTC/lightllm, vllm-project/vllm, and sgl-project/sglang.
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
):
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    kv_idx = tl.program_id(2)

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


def fused_fp8_set_kv_buffer(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
    page_size: int = 16,
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
