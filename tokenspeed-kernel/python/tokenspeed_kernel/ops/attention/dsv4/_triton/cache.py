# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""DSV4 SWA cache insertion and FP8/MXFP4 cache reads and writes."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv4._triton.common import (
    DEEPSEEK_V4_FP8_MAX,
    DEEPSEEK_V4_FP8_QUANT_BLOCK,
    DEEPSEEK_V4_HEAD_DIM,
    DEEPSEEK_V4_INDEXER_DIM,
    DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
    DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    DEEPSEEK_V4_NOPE_DIM,
    DEEPSEEK_V4_ROPE_DIM,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    _as_int32_block_table,
    _dsv4_mxfp4_e2m1_nibble,
)
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _dsv4_qnorm_rope_kv_insert_kernel(
    q_ptr,
    q_out_ptr,
    kv_ptr,
    cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_token,
    q_stride_head,
    q_out_stride_token,
    q_out_stride_head,
    kv_stride_token,
    cache_block_stride,
    cos_sin_stride,
    num_q_tokens,
    num_insert,
    rms_norm_eps,
    block_size,
    max_cache_slots,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    role = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM

    if role < NUM_HEADS:
        if token_idx < num_q_tokens:
            q_base = q_ptr + token_idx * q_stride_token + role * q_stride_head
            q_out_base = (
                q_out_ptr + token_idx * q_out_stride_token + role * q_out_stride_head
            )
            q = tl.load(q_base + offsets, mask=mask, other=0.0).to(tl.float32)
            q *= tl.rsqrt(tl.sum(q * q, axis=0) / HEAD_DIM + rms_norm_eps)

            NUM_PAIRS: tl.constexpr = BLOCK_DIM // 2
            NOPE_PAIRS: tl.constexpr = NOPE_DIM // 2
            pair_2d = tl.reshape(q, (NUM_PAIRS, 2))
            even, odd = tl.split(pair_2d)
            pair_idx = tl.arange(0, NUM_PAIRS)
            rope_pair = pair_idx - NOPE_PAIRS
            is_rope = (rope_pair >= 0) & (rope_pair < ROPE_DIM // 2)
            cs_idx = tl.maximum(rope_pair, 0)
            position = tl.load(positions_ptr + token_idx)
            cs_base = cos_sin_cache_ptr + position * cos_sin_stride
            cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(tl.float32)
            sin_v = tl.load(
                cs_base + ROPE_DIM // 2 + cs_idx,
                mask=is_rope,
                other=0.0,
            ).to(tl.float32)
            rotated = tl.interleave(
                even * cos_v - odd * sin_v,
                even * sin_v + odd * cos_v,
            )
            tl.store(q_out_base + offsets, rotated, mask=mask)
    else:
        if token_idx < num_insert:
            slot = tl.load(slot_mapping_ptr + token_idx)
            if slot >= 0 and slot < max_cache_slots:
                kv = tl.load(
                    kv_ptr + token_idx * kv_stride_token + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

                NUM_PAIRS: tl.constexpr = BLOCK_DIM // 2
                NOPE_PAIRS: tl.constexpr = NOPE_DIM // 2
                pair_2d = tl.reshape(kv, (NUM_PAIRS, 2))
                even, odd = tl.split(pair_2d)
                pair_idx = tl.arange(0, NUM_PAIRS)
                rope_pair = pair_idx - NOPE_PAIRS
                is_rope = (rope_pair >= 0) & (rope_pair < ROPE_DIM // 2)
                cs_idx = tl.maximum(rope_pair, 0)
                position = tl.load(positions_ptr + token_idx)
                cs_base = cos_sin_cache_ptr + position * cos_sin_stride
                cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(
                    tl.float32
                )
                sin_v = tl.load(
                    cs_base + ROPE_DIM // 2 + cs_idx,
                    mask=is_rope,
                    other=0.0,
                ).to(tl.float32)
                rotated = tl.interleave(
                    even * cos_v - odd * sin_v,
                    even * sin_v + odd * cos_v,
                )

                cache_block = slot // block_size
                cache_position = slot % block_size
                block_base = cache_ptr + cache_block.to(tl.int64) * cache_block_stride
                token_base = block_base + cache_position * TOKEN_STRIDE
                scale_base = (
                    block_base + block_size * TOKEN_STRIDE + cache_position * SCALE_DIM
                )

                N_QUANT_BLOCKS: tl.constexpr = BLOCK_DIM // QUANT_BLOCK
                N_NOPE_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK
                values_2d = tl.reshape(
                    rotated.to(tl.bfloat16).to(tl.float32),
                    (N_QUANT_BLOCKS, QUANT_BLOCK),
                )
                block_absmax = tl.maximum(tl.max(tl.abs(values_2d), axis=1), 1.0e-4)
                exponents = tl.ceil(tl.log2(block_absmax / FP8_MAX))
                inv_scales = tl.exp2(-exponents)
                quantized = tl.clamp(
                    values_2d * tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1)),
                    -FP8_MAX,
                    FP8_MAX,
                ).to(tl.float8e4nv)
                quantized_u8 = tl.reshape(
                    quantized.to(tl.uint8, bitcast=True), (BLOCK_DIM,)
                )
                tl.store(
                    token_base + offsets,
                    quantized_u8,
                    mask=offsets < NOPE_DIM,
                )

                scale_offsets = tl.arange(0, N_QUANT_BLOCKS)
                encoded_scales = tl.maximum(tl.minimum(exponents + 127.0, 255.0), 0.0)
                tl.store(
                    scale_base + scale_offsets,
                    encoded_scales.to(tl.uint8),
                    mask=scale_offsets < N_NOPE_BLOCKS,
                )
                tl.store(
                    scale_base + N_NOPE_BLOCKS,
                    tl.zeros((), dtype=tl.uint8),
                )

                rope_offsets = tl.arange(0, ROPE_DIM)
                rope_values = tl.load(
                    kv_ptr + token_idx * kv_stride_token + NOPE_DIM + rope_offsets
                ).to(tl.float32)
                rope_pairs = tl.reshape(rope_values, (ROPE_DIM // 2, 2))
                rope_even, rope_odd = tl.split(rope_pairs)
                rope_idx = tl.arange(0, ROPE_DIM // 2)
                rope_cos = tl.load(cs_base + rope_idx).to(tl.float32)
                rope_sin = tl.load(cs_base + ROPE_DIM // 2 + rope_idx).to(tl.float32)
                rope_rotated = tl.interleave(
                    rope_even * rope_cos - rope_odd * rope_sin,
                    rope_even * rope_sin + rope_odd * rope_cos,
                )
                rope_ptr = (token_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
                tl.store(
                    rope_ptr + rope_offsets,
                    rope_rotated.to(tl.bfloat16),
                )


@register_kernel(
    "attention",
    "dsv4_swa_cache_insert",
    name="triton_dsv4_swa_cache_insert",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        format_signature(
            q=dense_tensor_format(dtype),
            kv=dense_tensor_format(dtype),
            swa_kv_cache=dense_tensor_format(torch.uint8),
        )
        for dtype in (torch.float16, torch.bfloat16)
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "rope_dim": frozenset({DEEPSEEK_V4_ROPE_DIM}),
        "quant_block_size": frozenset({DEEPSEEK_V4_FP8_QUANT_BLOCK}),
        "cache_layout": frozenset({"fp8_swa_page_planar"}),
        "has_q_out": frozenset({True, False}),
    },
    priority=Priority.PORTABLE,
    tags={"portability", "cache_insert"},
)
def triton_dsv4_swa_cache_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    page_size: int,
    q_out: torch.Tensor | None = None,
) -> None:
    """Normalize/rotate Q and insert rotated K into the V4 SWA cache."""

    q_destination = q if q_out is None else q_out
    if q_destination.shape != q.shape or q_destination.dtype != q.dtype:
        raise ValueError("DeepSeek V4 q_out must match q shape and dtype")

    num_q_tokens, num_heads, head_dim = q.shape
    if head_dim != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"DeepSeek V4 Q head dimension must be 512, got {head_dim}")
    num_insert = min(kv.shape[0], slot_mapping.numel(), positions.numel())
    grid_tokens = max(num_q_tokens, num_insert)
    if grid_tokens == 0:
        return
    _dsv4_qnorm_rope_kv_insert_kernel[(grid_tokens, num_heads + 1)](
        q,
        q_destination,
        kv,
        swa_kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        q_destination.stride(0),
        q_destination.stride(1),
        kv.stride(0),
        swa_kv_cache.stride(0),
        cos_sin_cache.stride(0),
        num_q_tokens,
        num_insert,
        rms_norm_eps,
        page_size,
        swa_kv_cache.shape[0] * page_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        NOPE_DIM=DEEPSEEK_V4_NOPE_DIM,
        ROPE_DIM=DEEPSEEK_V4_ROPE_DIM,
        QUANT_BLOCK=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        TOKEN_STRIDE=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        SCALE_DIM=DEEPSEEK_V4_SWA_SCALE_DIM,
        FP8_MAX=DEEPSEEK_V4_FP8_MAX,
        BLOCK_DIM=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        num_warps=4,
    )


@triton.jit
def _dsv4_indexer_mxfp4_cache_write_kernel(
    rows_ptr,
    row_stride,
    cache_ptr,
    cache_stride0,
    slot_mapping_ptr,
    valid_ptr,
    cache_block_size,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    valid = tl.load(valid_ptr + row_idx)
    if valid == 0:
        return
    slot = tl.load(slot_mapping_ptr + row_idx)
    if slot < 0:
        return

    offsets = tl.arange(0, HALF_BLOCK)
    block_base = block_idx * QUANT_BLOCK
    row_base = rows_ptr + row_idx * row_stride + block_base
    x_lo = tl.load(row_base + offsets * 2).to(tl.float32)
    x_hi = tl.load(row_base + offsets * 2 + 1).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 1.0e-4)
    exponent = tl.ceil(tl.log2(amax / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    inv_scale = tl.exp2(-exponent)
    lo = _dsv4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _dsv4_mxfp4_e2m1_nibble(x_hi * inv_scale)
    packed = lo | (hi << 4)
    scale = (exponent + 127.0).to(tl.uint8)

    page = slot // cache_block_size
    pos = slot % cache_block_size
    page_base = cache_ptr + page.to(tl.int64) * cache_stride0
    value_base = page_base + pos * TOKEN_STRIDE + block_base // 2
    scale_base = page_base + cache_block_size * TOKEN_STRIDE + pos * SCALE_DIM
    tl.store(value_base + offsets, packed)
    tl.store(scale_base + block_idx, scale)


def write_dsv4_indexer_mxfp4_cache_cuda(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    block_size: int,
) -> None:
    num_rows = min(index_k.shape[0], slot_mapping.numel(), valid.numel())
    if num_rows == 0:
        return
    index_k = index_k[:num_rows]
    if index_k.stride(-1) != 1:
        index_k = index_k.contiguous()
    _dsv4_indexer_mxfp4_cache_write_kernel[
        (num_rows, DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM)
    ](
        index_k,
        index_k.stride(0),
        cache_2d,
        cache_2d.stride(0),
        slot_mapping[:num_rows],
        valid[:num_rows],
        block_size,
        HEAD_DIM=DEEPSEEK_V4_INDEXER_DIM,
        QUANT_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        HALF_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2,
        TOKEN_STRIDE=DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
        SCALE_DIM=DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
        num_warps=1,
    )


@triton.jit
def _dsv4_gather_indexer_mxfp4_cache_kernel(
    cache_ptr,
    slot_mapping_ptr,
    values_out_ptr,
    scales_out_ptr,
    rows: tl.constexpr,
    slot_stride: tl.constexpr,
    value_stride: tl.constexpr,
    scale_stride: tl.constexpr,
    cache_block_stride: tl.constexpr,
    block_size: tl.constexpr,
    value_bytes: tl.constexpr,
    scale_bytes: tl.constexpr,
    block_rows: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    row_mask = row_offsets < rows
    slots = tl.load(
        slot_mapping_ptr + row_offsets * slot_stride,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    valid_slots = row_mask & (slots >= 0)
    pages = slots // block_size
    pos = slots - pages * block_size
    page_base = pages * cache_block_stride

    value_cols = tl.arange(0, value_bytes)
    value_base = page_base + pos * value_bytes
    values = tl.load(
        cache_ptr + value_base[:, None] + value_cols[None, :],
        mask=valid_slots[:, None],
        other=0,
    )
    tl.store(
        values_out_ptr + row_offsets[:, None] * value_stride + value_cols[None, :],
        values,
        mask=row_mask[:, None],
    )

    scale_cols = tl.arange(0, scale_bytes)
    scale_base = page_base + block_size * value_bytes + pos * scale_bytes
    scales = tl.load(
        cache_ptr + scale_base[:, None] + scale_cols[None, :],
        mask=valid_slots[:, None],
        other=0,
    )
    tl.store(
        scales_out_ptr + row_offsets[:, None] * scale_stride + scale_cols[None, :],
        scales,
        mask=row_mask[:, None],
    )


def dsv4_gather_indexer_mxfp4_cache(
    *,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    values_out: torch.Tensor,
    scales_out: torch.Tensor,
    block_size: int,
) -> None:
    """Gather MXFP4 indexer cache bytes into DeepGEMM-ready workspaces."""

    rows = int(slot_mapping.numel())
    if rows == 0:
        return
    if not cache_2d.is_cuda:
        raise ValueError("dsv4_gather_indexer_mxfp4_cache requires CUDA cache")
    if not slot_mapping.is_cuda:
        raise ValueError("dsv4_gather_indexer_mxfp4_cache requires CUDA slots")
    if values_out.dtype != torch.uint8 or scales_out.dtype != torch.uint8:
        raise TypeError("MXFP4 gather workspaces must be uint8 tensors")
    if values_out.stride(1) != 1 or scales_out.stride(1) != 1:
        raise ValueError("MXFP4 gather workspaces must be contiguous in the last dim")
    if values_out.shape[0] < rows or scales_out.shape[0] < rows:
        raise ValueError("MXFP4 gather workspaces are smaller than slot_mapping")
    if values_out.shape[1] < DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES:
        raise ValueError("values_out has insufficient value bytes")
    if scales_out.shape[1] < DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM:
        raise ValueError("scales_out has insufficient scale bytes")

    block_rows = 16
    _dsv4_gather_indexer_mxfp4_cache_kernel[(triton.cdiv(rows, block_rows),)](
        cache_2d,
        slot_mapping,
        values_out,
        scales_out,
        rows=rows,
        slot_stride=slot_mapping.stride(0),
        value_stride=values_out.stride(0),
        scale_stride=scales_out.stride(0),
        cache_block_stride=cache_2d.stride(0),
        block_size=block_size,
        value_bytes=DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
        scale_bytes=DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
        block_rows=block_rows,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["block_table_stride", "max_blocks_per_seq"])
def _dsv4_dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    offset,
    gather_lens_ptr,
    block_table_stride,
    max_blocks_per_seq,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        if block_table_base_offsets_ptr is not None:
            block_in_seq -= tl.load(block_table_base_offsets_ptr + batch_idx)
        pos_in_block = pos % cache_block_size

        block_table_row = block_table_ptr + batch_idx * block_table_stride
        valid_block = (block_in_seq >= 0) & (block_in_seq < max_blocks_per_seq)
        physical_block_idx = tl.load(
            block_table_row + block_in_seq,
            mask=valid_block,
            other=-1,
        )
        valid_block = valid_block & (physical_block_idx >= 0)
        cache_block = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        token_data = cache_block + pos_in_block * token_data_size
        token_scales = (
            cache_block + cache_block_size * token_data_size + pos_in_block * scale_dim
        )
        out_row = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim
            x_uint8 = tl.load(token_data + offsets, mask=mask & valid_block, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            exponent = (
                tl.load(token_scales + qblock_idx, mask=valid_block, other=127).to(
                    tl.float32
                )
                - 127.0
            )
            scale = tl.exp2(exponent)
            tl.store(
                out_row + offsets,
                (x_fp8.to(tl.float32) * scale).to(tl.bfloat16),
                mask=mask,
            )

        bf16_out_offset = fp8_dim
        bf16_cache = (token_data + fp8_dim).to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            values = tl.load(bf16_cache + chunk_offsets, mask=valid_block, other=0.0)
            tl.store(out_row + bf16_out_offset + chunk_offsets, values)


def _dsv4_gather_launch_config(
    num_reqs: int,
    max_rows: int,
) -> tuple[int, int]:
    """Choose the Blackwell per-request grid width and warp count."""

    max_rows = max(1, max_rows)
    if max_rows <= 512:
        return 128, 4
    if max_rows <= 3072:
        return 512, 1
    if max_rows <= 6144:
        return 1024, 1
    if 3 <= num_reqs <= 4 and max_rows <= 12288:
        return 1024, 1
    return 2048, 1


def dsv4_dequantize_and_gather_k_cache(
    *,
    out: torch.Tensor,
    cache_2d: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    block_table_base_offsets: torch.Tensor | None = None,
    max_gather_len: int | None = None,
) -> None:
    """Gather/dequantize fp8_ds_mla cache rows for sparse prefill."""

    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be bfloat16, got {out.dtype}")
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if seq_lens.numel() == 0:
        return

    num_reqs = int(seq_lens.numel())
    max_rows = (
        int(out.shape[1]) - int(offset)
        if max_gather_len is None
        else int(max_gather_len)
    )
    if current_platform().is_blackwell:
        num_workers, num_warps = _dsv4_gather_launch_config(num_reqs, max_rows)
    else:
        num_workers, num_warps = 128, 4
    block_table_i32 = _as_int32_block_table(block_table)
    _dsv4_dequantize_and_gather_k_kernel[(num_reqs, num_workers)](
        out,
        out.stride(0),
        out.stride(1),
        cache_2d,
        seq_lens.to(torch.int32),
        block_table_i32,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        offset,
        gather_lens.to(torch.int32) if gather_lens is not None else None,
        block_table_stride=block_table_i32.stride(0),
        max_blocks_per_seq=block_table_i32.shape[-1],
        fp8_dim=DEEPSEEK_V4_NOPE_DIM,
        bf16_dim=DEEPSEEK_V4_ROPE_DIM,
        scale_dim=DEEPSEEK_V4_SWA_SCALE_DIM,
        quant_block=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        cache_block_size=block_size,
        token_data_size=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        block_stride=cache_2d.stride(0),
        fp8_max=DEEPSEEK_V4_FP8_MAX,
        n_quant_blocks=DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK,
        num_warps=num_warps,
    )
