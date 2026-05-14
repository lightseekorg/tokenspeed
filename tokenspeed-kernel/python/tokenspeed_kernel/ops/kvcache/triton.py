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

_is_nvidia = current_platform().is_nvidia

__all__ = [
    "transfer_kv_all_layer",
    "transfer_kv_per_layer",
]


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
    grid = (
        _recommended_program_count(
            length=length,
            element_size=item_size,
            num_layers=num_layers,
            device=src_indices.device,
        ),
    )
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
