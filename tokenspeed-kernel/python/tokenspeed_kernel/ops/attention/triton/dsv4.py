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

from __future__ import annotations

import functools
import logging

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

logger = logging.getLogger(__name__)

DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_ROPE_DIM = 64
DEEPSEEK_V4_NOPE_DIM = DEEPSEEK_V4_HEAD_DIM - DEEPSEEK_V4_ROPE_DIM
DEEPSEEK_V4_FP8_MAX = 448.0
DEEPSEEK_V4_FP8_QUANT_BLOCK = 64
DEEPSEEK_V4_MXFP4_BLOCK_SIZE = 32
DEEPSEEK_V4_INDEXER_DIM = 128
DEEPSEEK_V4_SWA_TOKEN_STRIDE = DEEPSEEK_V4_NOPE_DIM + DEEPSEEK_V4_ROPE_DIM * 2
DEEPSEEK_V4_SWA_SCALE_DIM = DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK + 1
DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES = DEEPSEEK_V4_INDEXER_DIM // 2
DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM = (
    DEEPSEEK_V4_INDEXER_DIM // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
)
DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT = 128

__all__ = [
    "dsv4_build_dense_prefill_local_compressed_indices",
    "dsv4_combine_dense_swa_indices",
    "dsv4_combine_topk_swa_indices",
    "dsv4_compressed_slot_mapping",
    "dsv4_compute_global_topk_indices_and_lens",
    "dsv4_decode_swa_indices_and_lens",
    "dsv4_dequantize_and_gather_k_cache",
    "dsv4_fused_csa_indexer_fp8_cache_insert",
    "dsv4_fused_csa_indexer_mxfp4_cache_insert",
    "dsv4_fused_indexer_q_rope_hadamard_mxfp4",
    "dsv4_fused_qnorm_rope_kv_insert",
    "dsv4_fused_sparse_compress_cache_insert",
    "dsv4_gather_indexer_mxfp4_cache",
    "dsv4_indexer_decode_metadata_compute",
    "dsv4_save_compressor_state",
    "dsv4_sparse_attention",
    "write_dsv4_indexer_mxfp4_cache_cuda",
]


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
def dsv4_fused_qnorm_rope_kv_insert(
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
def _dsv4_sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    q_stride_token,
    q_stride_head,
    kv_stride_row,
    indices_stride_token,
    out_stride_token,
    out_stride_head,
    softmax_scale,
    num_kv_rows,
    TOPK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim = tl.arange(0, BLOCK_DIM)
    dim_mask = dim < HEAD_DIM
    q = tl.load(
        q_ptr + token_idx * q_stride_token + head_idx * q_stride_head + dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    max_logit = tl.load(sink_ptr + head_idx).to(tl.float32)
    denominator = tl.full((), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_DIM,), tl.float32)
    valid_len = tl.minimum(tl.maximum(tl.load(lens_ptr + token_idx), 0), TOPK)
    topk_offsets = tl.arange(0, BLOCK_TOPK)

    for start in range(0, TOPK, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        rows = tl.load(
            indices_ptr + token_idx * indices_stride_token + cols,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (rows >= 0) & (rows < num_kv_rows)
        rows = tl.where(valid, rows, 0)
        kv = tl.load(
            kv_ptr + rows[:, None] * kv_stride_row + dim[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(kv * q[None, :], axis=1) * softmax_scale
        logits = tl.where(valid, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(max_logit, block_max)
        previous_scale = tl.exp(max_logit - next_max)
        probabilities = tl.exp(logits - next_max)
        probabilities = tl.where(valid, probabilities, 0.0)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * kv,
            axis=0,
        )
        denominator = denominator * previous_scale + tl.sum(probabilities, axis=0)
        max_logit = next_max

    output = tl.where(denominator > 0.0, accumulator / denominator, 0.0)
    tl.store(
        out_ptr + token_idx * out_stride_token + head_idx * out_stride_head + dim,
        output,
        mask=dim_mask,
    )


@register_kernel(
    "attention",
    "dsv4_prefill",
    name="triton_dsv4_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                kv=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "cache_layout": frozenset({"dense_workspace"}),
        "support_sink": frozenset({True}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def dsv4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run selected shared-KV attention for DeepSeek V4 geometry."""

    if q.dim() != 3 or q.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected q [tokens, heads, 512], got {tuple(q.shape)}")
    kv_2d = kv.reshape(-1, kv.shape[-1])
    if kv_2d.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected kv rows of width 512, got {tuple(kv.shape)}")
    indices_2d = indices.reshape(indices.shape[0], -1).contiguous()
    lens = lens.reshape(-1).contiguous()
    if indices_2d.shape[0] != q.shape[0] or lens.shape[0] != q.shape[0]:
        raise ValueError("selected-attention metadata must have one row per query")
    if attn_sink.numel() < q.shape[1]:
        raise ValueError("attention sink must provide one value per query head")

    output = out if out is not None else torch.empty_like(q)
    _dsv4_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        kv_2d,
        indices_2d,
        lens,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        kv_2d.stride(0),
        indices_2d.stride(0),
        output.stride(0),
        output.stride(1),
        softmax_scale,
        kv_2d.shape[0],
        TOPK=indices_2d.shape[1],
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        BLOCK_TOPK=16,
        BLOCK_DIM=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _dsv4_dequantize_selected_cache_rows_kernel(
    cache_ptr,
    slots_ptr,
    lens_ptr,
    out_ptr,
    indices_ptr,
    selected_lens_ptr,
    cache_block_stride,
    slots_stride_token,
    indices_stride_token,
    out_stride_token,
    out_stride_row,
    block_size,
    cache_capacity,
    OUTPUT_ROW_OFFSET: tl.constexpr,
    INDEX_OFFSET: tl.constexpr,
    WORKSPACE_WIDTH: tl.constexpr,
    HAS_METADATA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    slot = tl.load(slots_ptr + token_idx * slots_stride_token + row_idx).to(tl.int64)
    valid = (slot >= 0) & (slot < cache_capacity)
    if HAS_METADATA:
        valid &= row_idx < tl.load(lens_ptr + token_idx)
    safe_slot = tl.where(valid, slot, 0)
    cache_block = safe_slot // block_size
    cache_position = safe_slot % block_size
    block_base = cache_ptr + cache_block * cache_block_stride
    token_base = block_base + cache_position * TOKEN_STRIDE
    scale_base = block_base + block_size * TOKEN_STRIDE + cache_position * SCALE_DIM
    out_base = (
        out_ptr
        + token_idx * out_stride_token
        + (OUTPUT_ROW_OFFSET + row_idx) * out_stride_row
    )

    dim = tl.arange(0, BLOCK_DIM)
    nope_mask = dim < NOPE_DIM
    values_u8 = tl.load(token_base + dim, mask=valid & nope_mask, other=0)
    values_fp8 = values_u8.to(tl.float8e4nv, bitcast=True)
    scale_idx = dim // QUANT_BLOCK
    exponent = (
        tl.load(
            scale_base + scale_idx,
            mask=valid & nope_mask,
            other=127,
        ).to(tl.float32)
        - 127.0
    )
    nope = values_fp8.to(tl.float32) * tl.exp2(exponent)
    tl.store(out_base + dim, nope, mask=nope_mask)

    rope_offsets = tl.arange(0, ROPE_DIM)
    rope_ptr = (token_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rope_ptr + rope_offsets, mask=valid, other=0.0)
    tl.store(out_base + NOPE_DIM + rope_offsets, rope)
    if HAS_METADATA:
        flat_index = token_idx * WORKSPACE_WIDTH + INDEX_OFFSET + row_idx
        tl.store(
            indices_ptr + token_idx * indices_stride_token + INDEX_OFFSET + row_idx,
            tl.where(valid, flat_index, -1),
        )
        tl.store(
            selected_lens_ptr + token_idx,
            WORKSPACE_WIDTH,
            mask=row_idx == 0,
        )


def _dsv4_dequantize_selected_cache_segment(
    cache_2d: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    output: torch.Tensor,
    indices: torch.Tensor,
    selected_lens: torch.Tensor,
    output_row_offset: int,
    workspace_width: int,
) -> None:
    """Dequantize one cache segment and produce flattened attention metadata."""
    slots_2d = slots.reshape(slots.shape[0], -1).contiguous()
    lens_1d = lens.reshape(-1).contiguous()
    if slots_2d.shape[0] != output.shape[0] or lens_1d.shape[0] != output.shape[0]:
        raise ValueError("selected cache segment must have one row per query")
    if output_row_offset + slots_2d.shape[1] > output.shape[1]:
        raise ValueError("selected cache segment exceeds the output workspace")
    if slots_2d.numel() == 0:
        return
    _dsv4_dequantize_selected_cache_rows_kernel[(slots_2d.shape[0], slots_2d.shape[1])](
        cache_2d,
        slots_2d,
        lens_1d,
        output,
        indices,
        selected_lens,
        cache_2d.stride(0),
        slots_2d.stride(0),
        indices.stride(0),
        output.stride(0),
        output.stride(1),
        block_size,
        cache_2d.shape[0] * block_size,
        OUTPUT_ROW_OFFSET=output_row_offset,
        INDEX_OFFSET=output_row_offset,
        WORKSPACE_WIDTH=workspace_width,
        HAS_METADATA=True,
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        NOPE_DIM=DEEPSEEK_V4_NOPE_DIM,
        ROPE_DIM=DEEPSEEK_V4_ROPE_DIM,
        QUANT_BLOCK=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        TOKEN_STRIDE=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        SCALE_DIM=DEEPSEEK_V4_SWA_SCALE_DIM,
        BLOCK_DIM=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        num_warps=4,
    )


@register_kernel(
    "attention",
    "dsv4_decode",
    name="triton_dsv4_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                swa_kv_cache=dense_tensor_format(torch.uint8),
            )
        }
    ),
    traits={
        "head_dim": frozenset({DEEPSEEK_V4_HEAD_DIM}),
        "cache_layout": frozenset({"fp8_swa_page_planar"}),
        "topk_layout": frozenset({"global_slots"}),
        "support_sink": frozenset({True}),
        "has_extra_segment": frozenset({False, True}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
    },
    priority=Priority.PORTABLE,
    tags={"portability", "paged_cache", "selected_attention"},
)
def triton_dsv4_decode(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extra_kv_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose page-planar dequantization with registered dense attention."""
    from tokenspeed_kernel.ops.attention import dsv4_prefill

    tokens = q.shape[0]
    swa_width = swa_slots.numel() // tokens
    extra_width = 0 if extra_slots is None else extra_slots.numel() // tokens
    workspace_width = swa_width + extra_width
    kv_workspace = torch.empty(
        (tokens, workspace_width, DEEPSEEK_V4_HEAD_DIM),
        dtype=torch.bfloat16,
        device=q.device,
    )
    selected_indices = torch.empty(
        (tokens, workspace_width), dtype=torch.int32, device=q.device
    )
    selected_lens = torch.empty((tokens,), dtype=torch.int32, device=q.device)
    _dsv4_dequantize_selected_cache_segment(
        swa_kv_cache,
        swa_slots,
        swa_lens,
        swa_page_size,
        kv_workspace,
        selected_indices,
        selected_lens,
        0,
        workspace_width,
    )
    if extra_kv_cache is not None:
        assert extra_slots is not None
        assert extra_lens is not None
        assert extra_page_size is not None
        _dsv4_dequantize_selected_cache_segment(
            extra_kv_cache,
            extra_slots,
            extra_lens,
            extra_page_size,
            kv_workspace,
            selected_indices,
            selected_lens,
            swa_width,
            workspace_width,
        )
    return dsv4_prefill(
        q=q,
        kv=kv_workspace,
        indices=selected_indices,
        lens=selected_lens,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        out=out,
    )


def _as_int32_block_table(block_table: torch.Tensor) -> torch.Tensor:
    """Return an int32 table with unit column stride for Triton row indexing."""

    block_table_i32 = block_table.to(torch.int32)
    if block_table_i32.stride(-1) != 1:
        block_table_i32 = block_table_i32.contiguous()
    return block_table_i32


@triton.jit
def _dsv4_mxfp4_e2m1_nibble(x):
    abs_x = tl.minimum(tl.abs(x), 6.0)
    code = tl.where(
        abs_x <= 0.25,
        0.0,
        tl.where(
            abs_x <= 0.75,
            1.0,
            tl.where(
                abs_x <= 1.25,
                2.0,
                tl.where(
                    abs_x <= 1.75,
                    3.0,
                    tl.where(
                        abs_x <= 2.5,
                        4.0,
                        tl.where(abs_x <= 3.5, 5.0, tl.where(abs_x <= 5.0, 6.0, 7.0)),
                    ),
                ),
            ),
        ),
    )
    code_u8 = code.to(tl.uint8)
    sign = ((x < 0) & (code_u8 != 0)).to(tl.uint8)
    return code_u8 | (sign << 3)


@triton.jit
def _dsv4_fused_indexer_q_rope_hadamard_mxfp4_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    q_packed_ptr,
    q_packed_stride0,
    q_packed_stride1,
    q_scale_ptr,
    q_scale_stride0,
    q_scale_stride1,
    weights_ptr,
    weights_stride,
    weights_softmax_scale,
    weights_head_scale,
    weights_out_ptr,
    weights_out_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    HADAMARD_SCALE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    quant_block_idx = tl.program_id(2)

    pos = tl.load(positions_ptr + token_idx)
    dim = tl.arange(0, TRITON_BLOCK_SIZE)
    q_base = index_q_ptr + token_idx * index_q_stride0 + head_idx * index_q_stride1
    q = tl.load(q_base + dim, mask=dim < HEAD_DIM, other=0.0).to(tl.float32)

    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_DIM // 2

    pair_2d = tl.reshape(q, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair = pair_idx - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cs_base = cos_sin_cache_ptr + pos * cos_sin_cache_stride
    cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(tl.float32)
    sin_v = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0).to(
        tl.float32
    )
    rotated_even = even * cos_v - odd * sin_v
    rotated_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(rotated_even, rotated_odd)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    in_idx = tl.arange(0, TRITON_BLOCK_SIZE)
    out_idx = quant_block_idx * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)
    bits = (in_idx[:, None] & out_idx[None, :]).to(tl.int32)
    parity = bits ^ (bits >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    hadamard = tl.sum(rotated[:, None] * signs, axis=0) * HADAMARD_SCALE
    hadamard = hadamard.to(tl.bfloat16).to(tl.float32)

    hadamard_2d = tl.reshape(hadamard, (HALF_BLOCK, 2))
    x_lo, x_hi = tl.split(hadamard_2d)
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 1.0e-4)
    exponent = tl.ceil(tl.log2(amax / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    inv_scale = tl.exp2(-exponent)
    lo = _dsv4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _dsv4_mxfp4_e2m1_nibble(x_hi * inv_scale)
    packed = lo | (hi << 4)
    scale = (exponent + 127.0).to(tl.uint8)

    packed_base = (
        q_packed_ptr
        + token_idx * q_packed_stride0
        + head_idx * q_packed_stride1
        + quant_block_idx * HALF_BLOCK
    )
    scale_base = (
        q_scale_ptr
        + token_idx * q_scale_stride0
        + head_idx * q_scale_stride1
        + quant_block_idx
    )
    tl.store(packed_base + tl.arange(0, HALF_BLOCK), packed)
    tl.store(scale_base, scale)

    weights = tl.load(weights_ptr + token_idx * weights_stride + head_idx).to(
        tl.float32
    )
    weights = weights * weights_softmax_scale * weights_head_scale
    tl.store(
        weights_out_ptr + token_idx * weights_out_stride + head_idx,
        weights,
        mask=quant_block_idx == 0,
    )


def dsv4_fused_indexer_q_rope_hadamard_mxfp4(
    *,
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    num_tokens, num_heads, head_dim = index_q.shape
    q_packed = torch.empty(
        (num_tokens, num_heads, head_dim // 2),
        dtype=torch.uint8,
        device=index_q.device,
    )
    q_scale_bytes = torch.empty(
        (num_tokens, num_heads, head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE),
        dtype=torch.uint8,
        device=index_q.device,
    )
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    if num_tokens == 0:
        return (q_packed, q_scale_bytes.view(torch.int32).squeeze(-1)), weights_out

    _dsv4_fused_indexer_q_rope_hadamard_mxfp4_kernel[
        (num_tokens, num_heads, head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE)
    ](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_packed,
        q_packed.stride(0),
        q_packed.stride(1),
        q_scale_bytes,
        q_scale_bytes.stride(0),
        q_scale_bytes.stride(1),
        weights,
        weights.stride(0),
        softmax_scale,
        head_scale,
        weights_out,
        weights_out.stride(0),
        HEAD_DIM=head_dim,
        ROPE_DIM=DEEPSEEK_V4_ROPE_DIM,
        QUANT_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        HALF_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2,
        HADAMARD_SCALE=head_dim**-0.5,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return (
        q_packed,
        q_scale_bytes.view(torch.int32).squeeze(-1).contiguous(),
    ), weights_out


@triton.jit(do_not_specialize=["block_table_stride", "block_table_width"])
def _dsv4_fused_sparse_compress_cache_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    block_table_width,
    state_block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    state_slot = tl.load(slot_mapping_ptr + token_idx)
    if state_slot < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_slot = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot < 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    if block_table_base_offsets_ptr is not None:
        base_logical_page = tl.load(block_table_base_offsets_ptr + req_idx)
    else:
        base_logical_page = tl.full((), 0, tl.int32)
    window: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    start = position - window + 1
    tokens = tl.arange(0, window)
    pos = start + tokens
    valid_pos = pos >= 0

    table_idx = pos // state_block_size - base_logical_page
    valid_pos = valid_pos & (table_idx >= 0) & (table_idx < block_table_width)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + table_idx,
        mask=valid_pos,
        other=-1,
    ).to(tl.int64)
    pos_in_block = pos % state_block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers[:, None] * state_cache_stride0
        + pos_in_block[:, None] * state_cache_stride1
        + head_offset[:, None]
    )
    combined_mask = valid_pos[:, None] & (block_numbers[:, None] >= 0) & mask[None, :]

    score = tl.load(
        row_base + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base + block[None, :], mask=combined_mask, other=0.0)
    compressed = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_SIZE
    normed = compressed * tl.rsqrt(variance + rms_norm_eps) * rms_w

    kv_block = kv_slot // kv_cache_block_size
    kv_pos = kv_slot % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr + kv_cache_block_size * TOKEN_STRIDE + kv_pos * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    block_absmax = tl.max(tl.abs(quant_2d), axis=1)
    block_absmax = tl.maximum(block_absmax, 1.0e-4)
    exponents = tl.ceil(tl.log2(block_absmax * INV_FP8_MAX))
    inv_scales = tl.exp2(-exponents)
    x_scaled = quant_2d * tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_fp8 = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    x_uint8 = tl.reshape(x_fp8.to(tl.uint8, bitcast=True), (TRITON_BLOCK_SIZE,))

    tl.store(fp8_ptr + block, x_uint8, mask=block < NOPE_HEAD_DIM)
    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = tl.maximum(tl.minimum(exponents + 127.0, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx, encoded.to(tl.uint8), mask=scale_idx < N_NOPE_BLOCKS
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair = pair_idx - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cs_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(new_even, new_odd)

    rope_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    tl.store(
        rope_ptr + rope_local,
        rotated.to(tl.bfloat16),
        mask=(block >= NOPE_HEAD_DIM) & mask,
    )


@functools.cache
def _wide_compress_launch_supported(device: torch.device | int | None) -> bool:
    """Return whether the wide sparse-compress launch is supported."""

    try:
        if not torch.cuda.is_available() or torch.version.hip is not None:
            return False
        capability = torch.cuda.get_device_capability(device)
        supported = capability == (10, 0)
        logger.info(
            "DeepSeek V4 sparse-compress launch selection: capability=%s num_warps=%d",
            capability,
            16 if supported else 4,
        )
        return supported
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


def dsv4_fused_sparse_compress_cache_insert(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    overlap: bool,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    num_actual = min(
        compressor_slot_mapping.numel(),
        positions.numel(),
        kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return
    block_table_i32 = _as_int32_block_table(block_table)
    _dsv4_fused_sparse_compress_cache_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices[:num_actual],
        positions[:num_actual],
        compressor_slot_mapping[:num_actual],
        block_table_i32,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table_i32.stride(0),
        block_table_i32.shape[-1],
        compressor_block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache_2d,
        kv_slot_mapping[:num_actual],
        kv_cache_block_size,
        HEAD_SIZE=DEEPSEEK_V4_HEAD_DIM,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(DEEPSEEK_V4_HEAD_DIM),
        STATE_WIDTH=state_cache.shape[-1] // 2,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=DEEPSEEK_V4_ROPE_DIM,
        FP8_MAX=DEEPSEEK_V4_FP8_MAX,
        QUANT_BLOCK=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        TOKEN_STRIDE=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        SCALE_DIM=DEEPSEEK_V4_SWA_SCALE_DIM,
        KV_BLOCK_STRIDE=kv_cache_2d.stride(0),
        num_warps=(
            16
            if compress_ratio >= 128
            and _wide_compress_launch_supported(state_cache.device)
            else 4
        ),
    )


@triton.jit(do_not_specialize=["block_table_stride", "block_table_width"])
def _dsv4_fused_csa_indexer_fp8_cache_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    block_table_width,
    state_block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    state_cache_blocks,
    block_table_rows,
    cos_sin_rows,
    kv_cache_blocks,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    HADAMARD_SCALE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    state_slot = tl.load(slot_mapping_ptr + token_idx)
    if state_slot < 0 or state_slot >= state_cache_blocks * state_block_size:
        return

    position = tl.load(positions_ptr + token_idx)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    if (
        position < 0
        or (position + 1) % COMPRESS_RATIO != 0
        or compressed_pos >= cos_sin_rows
    ):
        return

    kv_slot = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot < 0 or kv_slot >= kv_cache_blocks * kv_cache_block_size:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    if req_idx < 0 or req_idx >= block_table_rows:
        return
    if block_table_base_offsets_ptr is not None:
        base_logical_page = tl.load(block_table_base_offsets_ptr + req_idx)
    else:
        base_logical_page = tl.full((), 0, tl.int32)
    window: tl.constexpr = 2 * COMPRESS_RATIO
    window_offsets = tl.arange(0, window)
    pos = position - window + 1 + window_offsets
    valid_pos = pos >= 0

    table_idx = pos // state_block_size - base_logical_page
    valid_pos = valid_pos & (table_idx >= 0) & (table_idx < block_table_width)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + table_idx,
        mask=valid_pos,
        other=-1,
    ).to(tl.int64)
    pos_in_block = pos % state_block_size
    head_offset = (window_offsets >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    dim = tl.arange(0, TRITON_BLOCK_SIZE)
    row_base = (
        state_cache_ptr
        + block_numbers[:, None] * state_cache_stride0
        + pos_in_block[:, None] * state_cache_stride1
        + head_offset[:, None]
    )
    valid_rows = (
        valid_pos[:, None]
        & (block_numbers[:, None] >= 0)
        & (block_numbers[:, None] < state_cache_blocks)
    )
    score = tl.load(
        row_base + STATE_WIDTH + dim[None, :],
        mask=valid_rows,
        other=-1.0e30,
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base + dim[None, :], mask=valid_rows, other=0.0)
    compressed = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + dim)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_SIZE
    normed = compressed * tl.rsqrt(variance + rms_norm_eps) * rms_w

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair = pair_idx - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)

    cs_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(new_even, new_odd)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    in_idx = tl.arange(0, TRITON_BLOCK_SIZE)
    out_idx = tl.arange(0, TRITON_BLOCK_SIZE)
    bits = (in_idx[:, None] & out_idx[None, :]).to(tl.int32)
    parity = bits ^ (bits >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    hadamard = tl.sum(rotated[:, None] * signs, axis=0) * HADAMARD_SCALE
    hadamard = hadamard.to(tl.bfloat16).to(tl.float32)

    scale_input = tl.maximum(
        tl.max(tl.abs(hadamard), axis=0) / FP8_MAX,
        1.0e-10,
    )
    scale = tl.exp2(tl.ceil(tl.log2(scale_input)))
    quantized = tl.clamp(hadamard / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    value_bytes = quantized.to(tl.uint8, bitcast=True)

    kv_block = kv_slot // kv_cache_block_size
    kv_pos = kv_slot % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block.to(tl.int64) * KV_BLOCK_STRIDE
    value_ptr = cache_block_ptr + kv_pos * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr + kv_cache_block_size * TOKEN_STRIDE + kv_pos * SCALE_DIM
    ).to(tl.pointer_type(tl.float32))
    tl.store(value_ptr + out_idx, value_bytes)
    tl.store(scale_ptr, scale)


@register_kernel(
    "attention",
    "dsv4_csa_indexer_fp8_cache_insert",
    name="triton_dsv4_csa_indexer_fp8_cache_insert",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                state_cache=dense_tensor_format(torch.float32),
                kv_cache=dense_tensor_format(torch.uint8),
            )
        }
    ),
    traits={
        "index_head_dim": frozenset({DEEPSEEK_V4_INDEXER_DIM}),
        "compress_ratio": frozenset({4}),
        "page_size": frozenset({64}),
        "cache_format": frozenset({"fp8_scaled_page_planar"}),
    },
    priority=Priority.PORTABLE,
    tags={"portability", "cache_insert"},
)
def dsv4_fused_csa_indexer_fp8_cache_insert(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress CSA indexer state and insert page-planar FP8 cache rows.

    The input state is FP32 `[pages, state_block, 512]` and the output pages
    contain `[64, 128]` E4M3 value bytes followed by `[64, 4]` FP32 scale
    bytes. Rows are written only at the end of each four-token CSA group.

    Args:
        state_cache: Paged compressor values and scores.
        token_to_req_indices: Request index for each input token.
        positions: Absolute input token positions.
        compressor_slot_mapping: State slots; negative slots suppress writes.
        block_table: Logical-to-physical state page table.
        compressor_block_size: Number of state rows per page.
        rms_norm_weight: Width-128 RMSNorm weight.
        rms_norm_eps: RMSNorm epsilon.
        cos_sin_cache: Width-64 fused cosine and sine cache.
        kv_cache_2d: Uint8 page-planar FP8 indexer cache.
        kv_slot_mapping: Output cache slots; negative slots suppress writes.
        kv_cache_block_size: Output page size, which must be 64.
        compress_ratio: CSA compression ratio, which must be 4.
        block_table_base_offsets: Optional logical page base per request.

    Returns:
        None.
    """

    if kv_cache_block_size != 64:
        raise ValueError(
            "DeepSeek V4 FP8 indexer insertion requires "
            f"kv_cache_block_size=64, got {kv_cache_block_size}"
        )
    if compress_ratio != 4:
        raise ValueError(
            "DeepSeek V4 CSA indexer insertion requires "
            f"compress_ratio=4, got {compress_ratio}"
        )
    num_actual = min(
        compressor_slot_mapping.numel(),
        positions.numel(),
        kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return
    block_table_i32 = _as_int32_block_table(block_table)
    _dsv4_fused_csa_indexer_fp8_cache_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices[:num_actual],
        positions[:num_actual],
        compressor_slot_mapping[:num_actual],
        block_table_i32,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table_i32.stride(0),
        block_table_i32.shape[-1],
        compressor_block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache_2d,
        kv_slot_mapping[:num_actual],
        kv_cache_block_size,
        state_cache.shape[0],
        block_table_i32.shape[0],
        cos_sin_cache.shape[0],
        kv_cache_2d.shape[0],
        HEAD_SIZE=DEEPSEEK_V4_INDEXER_DIM,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(DEEPSEEK_V4_INDEXER_DIM),
        STATE_WIDTH=state_cache.shape[-1] // 2,
        COMPRESS_RATIO=compress_ratio,
        ROPE_HEAD_DIM=DEEPSEEK_V4_ROPE_DIM,
        FP8_MAX=DEEPSEEK_V4_FP8_MAX,
        TOKEN_STRIDE=DEEPSEEK_V4_INDEXER_DIM,
        SCALE_DIM=4,
        KV_BLOCK_STRIDE=kv_cache_2d.stride(0),
        HADAMARD_SCALE=DEEPSEEK_V4_INDEXER_DIM**-0.5,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["block_table_stride", "block_table_width"])
def _dsv4_fused_csa_indexer_mxfp4_cache_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    block_table_width,
    state_block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    HADAMARD_SCALE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    quant_block_idx = tl.program_id(1)

    state_slot = tl.load(slot_mapping_ptr + token_idx)
    if state_slot < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_slot = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot < 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    if block_table_base_offsets_ptr is not None:
        base_logical_page = tl.load(block_table_base_offsets_ptr + req_idx)
    else:
        base_logical_page = tl.full((), 0, tl.int32)
    window: tl.constexpr = 2 * COMPRESS_RATIO
    window_offsets = tl.arange(0, window)
    pos = position - window + 1 + window_offsets
    valid_pos = pos >= 0

    table_idx = pos // state_block_size - base_logical_page
    valid_pos = valid_pos & (table_idx >= 0) & (table_idx < block_table_width)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + table_idx,
        mask=valid_pos,
        other=-1,
    ).to(tl.int64)
    pos_in_block = pos % state_block_size
    head_offset = (window_offsets >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    dim = tl.arange(0, TRITON_BLOCK_SIZE)
    row_base = (
        state_cache_ptr
        + block_numbers[:, None] * state_cache_stride0
        + pos_in_block[:, None] * state_cache_stride1
        + head_offset[:, None]
    )
    score = tl.load(
        row_base + STATE_WIDTH + dim[None, :],
        mask=valid_pos[:, None] & (block_numbers[:, None] >= 0),
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(
        row_base + dim[None, :],
        mask=valid_pos[:, None] & (block_numbers[:, None] >= 0),
        other=0.0,
    )
    compressed = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + dim)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_SIZE
    normed = compressed * tl.rsqrt(variance + rms_norm_eps) * rms_w

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair = pair_idx - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cs_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    rotated = tl.interleave(new_even, new_odd)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    in_idx = tl.arange(0, TRITON_BLOCK_SIZE)
    out_idx = quant_block_idx * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)
    bits = (in_idx[:, None] & out_idx[None, :]).to(tl.int32)
    parity = bits ^ (bits >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    hadamard = tl.sum(rotated[:, None] * signs, axis=0) * HADAMARD_SCALE
    hadamard = hadamard.to(tl.bfloat16).to(tl.float32)

    hadamard_2d = tl.reshape(hadamard, (HALF_BLOCK, 2))
    x_lo, x_hi = tl.split(hadamard_2d)
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 1.0e-4)
    exponent = tl.ceil(tl.log2(amax / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    inv_scale = tl.exp2(-exponent)
    lo = _dsv4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _dsv4_mxfp4_e2m1_nibble(x_hi * inv_scale)
    packed = lo | (hi << 4)
    scale = (exponent + 127.0).to(tl.uint8)

    kv_block = kv_slot // kv_cache_block_size
    kv_pos = kv_slot % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr + kv_cache_block_size * TOKEN_STRIDE + kv_pos * SCALE_DIM
    )
    tl.store(val_ptr + quant_block_idx * HALF_BLOCK + tl.arange(0, HALF_BLOCK), packed)
    tl.store(scale_ptr + quant_block_idx, scale)


def dsv4_fused_csa_indexer_mxfp4_cache_insert(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    num_actual = min(
        compressor_slot_mapping.numel(),
        positions.numel(),
        kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return
    block_table_i32 = _as_int32_block_table(block_table)
    _dsv4_fused_csa_indexer_mxfp4_cache_kernel[
        (num_actual, DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM)
    ](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices[:num_actual],
        positions[:num_actual],
        compressor_slot_mapping[:num_actual],
        block_table_i32,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table_i32.stride(0),
        block_table_i32.shape[-1],
        compressor_block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache_2d,
        kv_slot_mapping[:num_actual],
        kv_cache_block_size,
        HEAD_SIZE=DEEPSEEK_V4_INDEXER_DIM,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(DEEPSEEK_V4_INDEXER_DIM),
        STATE_WIDTH=state_cache.shape[-1] // 2,
        COMPRESS_RATIO=compress_ratio,
        ROPE_HEAD_DIM=DEEPSEEK_V4_ROPE_DIM,
        QUANT_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        HALF_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2,
        TOKEN_STRIDE=DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
        SCALE_DIM=DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
        KV_BLOCK_STRIDE=kv_cache_2d.stride(0),
        HADAMARD_SCALE=DEEPSEEK_V4_INDEXER_DIM**-0.5,
        num_warps=4,
    )


@triton.jit
def _dsv4_save_compressor_state_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    state_block_size,
    STATE_WIDTH: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    C4_OVERLAP: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    block_idx = slot_id // state_block_size
    pos_in_block = slot_id % state_block_size
    base_ptr = (
        state_cache_ptr
        + block_idx.to(tl.int64) * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    offsets = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = offsets < STATE_WIDTH
    kv = tl.load(kv_ptr + token_idx * kv_stride + offsets, mask=mask, other=0.0)
    score = tl.load(
        score_ptr + token_idx * score_stride + offsets,
        mask=mask,
        other=0.0,
    )

    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    if C4_OVERLAP:
        HEAD_DIM: tl.constexpr = STATE_WIDTH // 2
        ape_offsets = tl.where(
            offsets < HEAD_DIM,
            ape_row * HEAD_DIM + offsets,
            (ape_row + COMPRESS_RATIO) * HEAD_DIM + offsets - HEAD_DIM,
        )
    else:
        ape_offsets = ape_row * STATE_WIDTH + offsets
    ape = tl.load(ape_ptr + ape_offsets, mask=mask, other=0.0)

    tl.store(base_ptr + offsets, kv, mask=mask)
    tl.store(base_ptr + STATE_WIDTH + offsets, score + ape, mask=mask)


def dsv4_save_compressor_state(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> None:
    num_actual = min(slot_mapping.numel(), kv.shape[0])
    if num_actual == 0:
        return
    state_width = kv.shape[-1]
    _dsv4_save_compressor_state_kernel[(num_actual,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        positions[:num_actual],
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping[:num_actual],
        block_size,
        STATE_WIDTH=state_width,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(state_width),
        COMPRESS_RATIO=compress_ratio,
        C4_OVERLAP=compress_ratio == 4
        and state_width == ape.shape[1]
        and state_width % 2 == 0,
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


@triton.jit
def _dsv4_compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    is_valid_token_ptr,
    has_valid_token: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if has_valid_token:
        is_valid_token = tl.load(is_valid_token_ptr + token_idx)
        if not is_valid_token:
            tl.store(topk_lens_ptr + token_idx, 0)
            return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    count = tl.zeros((), dtype=tl.int32)

    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        valid = local_idx >= 0
        block_indices = local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & valid,
            other=0,
        )
        block_offsets = local_idx % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, count)


def dsv4_compute_global_topk_indices_and_lens(
    *,
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local CSA top-k indices to global KV slots in one Triton kernel."""

    if topk_indices.dtype != torch.int32:
        raise TypeError(f"topk_indices must be int32, got {topk_indices.dtype}")
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be 2-D, got {tuple(topk_indices.shape)}")
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    if num_tokens == 0:
        return global_topk_indices, topk_lens
    if is_valid_token is not None:
        is_valid_token = is_valid_token[:num_tokens].to(
            device=topk_indices.device,
            dtype=torch.bool,
        )
    if not topk_indices.is_cuda:
        valid = topk_indices >= 0
        if is_valid_token is not None:
            valid = valid & is_valid_token[:, None]
        req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
        rows = int(block_table.shape[0]) if block_table.dim() >= 1 else 0
        cols = int(block_table.shape[1]) if block_table.dim() >= 2 else 0
        if rows <= 0 or cols <= 0:
            global_topk_indices.fill_(-1)
            topk_lens.zero_()
            return global_topk_indices, topk_lens
        safe_local = torch.where(valid, topk_indices, torch.zeros_like(topk_indices))
        block_indices = torch.div(safe_local, block_size, rounding_mode="floor")
        block_offsets = safe_local % block_size
        req_valid = (req_idx >= 0) & (req_idx < rows)
        block_valid = (block_indices >= 0) & (block_indices < cols)
        valid = valid & req_valid[:, None] & block_valid
        safe_req = req_idx.clamp(0, rows - 1)
        safe_block = block_indices.long().clamp(0, cols - 1)
        block_numbers = block_table[safe_req[:, None], safe_block]
        global_topk_indices.copy_(
            torch.where(
                valid,
                block_numbers.to(torch.int32) * block_size + block_offsets,
                torch.full_like(topk_indices, -1),
            )
        )
        topk_lens.copy_(valid.sum(dim=1, dtype=torch.int32))
        return global_topk_indices, topk_lens
    if is_valid_token is None:
        is_valid_token = torch.empty(0, dtype=torch.bool, device=topk_indices.device)

    _dsv4_compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        token_to_req_indices.to(torch.int32),
        block_table.to(torch.int32),
        block_table.stride(0),
        is_valid_token,
        is_valid_token.numel() != 0,
        block_size=block_size,
        topk=topk_indices.shape[-1],
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit
def _dsv4_combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_base_offsets_ptr,
    workspace_width,
    compressed_base,
    compressed_block_size,
    compressed_table_capacity,
    has_block_table_base_offsets: tl.constexpr,
    topk: tl.constexpr,
    compress_ratio: tl.constexpr,
    window_size: tl.constexpr,
    padded_topk: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        base_row = tl.zeros((), dtype=tl.int32)
        if has_block_table_base_offsets:
            base_row = (
                tl.load(block_table_base_offsets_ptr + batch_idx).to(tl.int32)
                * compressed_block_size
            )
        live_compressed_len = tl.maximum(
            tl.minimum(
                (pos + 1) // compress_ratio - base_row, compressed_table_capacity
            ),
            0,
        )
        topk_len = tl.minimum(live_compressed_len, topk)
        swa_len = tl.minimum(pos + 1, window_size)

        topk_offsets = tl.arange(0, padded_topk)
        topk_mask = topk_offsets < topk_len
        topk_values = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + topk_offsets,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = topk_mask & (topk_values >= 0)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offsets,
            topk_values + workspace_width * batch_idx,
            mask=valid_topk,
        )

        swa_offsets = tl.arange(0, window_size)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offsets,
            workspace_width * batch_idx
            + compressed_base
            + swa_offsets
            + pos
            - swa_len
            + 1
            - gather_start,
            mask=swa_offsets < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def dsv4_combine_topk_swa_indices(
    *,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    workspace_width: int,
    compressed_base: int,
    block_table_base_offsets: torch.Tensor | None = None,
    compressed_block_size: int = 1,
    compressed_table_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FlashMLA sparse prefill indices from compressed prefix and SWA."""

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
        * DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        -1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0 or num_reqs == 0:
        return combined_indices, combined_lens
    if compressed_block_size <= 0:
        raise ValueError("compressed_block_size must be positive")
    if compressed_table_capacity is None:
        compressed_table_capacity = compressed_base

    _dsv4_combine_topk_swa_indices_kernel[(num_reqs, 128)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        gather_lens.to(torch.int32),
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else seq_lens
        ),
        workspace_width,
        compressed_base,
        compressed_block_size,
        compressed_table_capacity,
        has_block_table_base_offsets=block_table_base_offsets is not None,
        topk=topk,
        compress_ratio=compress_ratio,
        window_size=window_size,
        padded_topk=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _dsv4_build_dense_prefill_local_compressed_indices_kernel(
    out_ptr,
    out_stride,
    positions_ptr,
    token_to_req_indices_ptr,
    block_table_base_offsets_ptr,
    compressed_block_size,
    compressed_table_capacity,
    has_block_table_base_offsets: tl.constexpr,
    width: tl.constexpr,
    compress_ratio: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    base_row = tl.zeros((), dtype=tl.int64)
    if has_block_table_base_offsets:
        req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)
        base_row = (
            tl.load(block_table_base_offsets_ptr + req_idx).to(tl.int64)
            * compressed_block_size
        )
    compressed_len = tl.minimum(
        tl.maximum((position + 1) // compress_ratio - base_row, 0),
        tl.minimum(width, compressed_table_capacity),
    )
    for start in range(0, width, block):
        offsets = start + tl.arange(0, block)
        mask = offsets < width
        values = tl.where(offsets < compressed_len, base_row + offsets, -1)
        tl.store(out_ptr + token_idx * out_stride + offsets, values, mask=mask)


def dsv4_build_dense_prefill_local_compressed_indices(
    *,
    positions: torch.Tensor,
    compress_ratio: int,
    width: int,
    out: torch.Tensor,
    token_to_req_indices: torch.Tensor | None = None,
    block_table_base_offsets: torch.Tensor | None = None,
    compressed_block_size: int = 1,
    compressed_table_capacity: int | None = None,
) -> torch.Tensor:
    """Build C128A/HCA prefill-local compressed prefix indices into `out`."""

    result = out[: positions.numel(), :width]
    if positions.numel() == 0 or width <= 0:
        return result
    if result.stride(1) != 1:
        raise ValueError(
            "dense prefill compressed indices output must be contiguous in the last dim"
        )
    if block_table_base_offsets is not None and token_to_req_indices is None:
        raise ValueError(
            "token_to_req_indices is required with block_table_base_offsets"
        )
    if compressed_table_capacity is None:
        compressed_table_capacity = width
    metadata_arg = positions if token_to_req_indices is None else token_to_req_indices
    base_offsets_arg = (
        positions if block_table_base_offsets is None else block_table_base_offsets
    )
    if positions.is_cuda:
        _dsv4_build_dense_prefill_local_compressed_indices_kernel[(positions.numel(),)](
            result,
            result.stride(0),
            positions,
            metadata_arg,
            base_offsets_arg,
            compressed_block_size,
            compressed_table_capacity,
            has_block_table_base_offsets=block_table_base_offsets is not None,
            width=width,
            compress_ratio=compress_ratio,
            block=1024,
        )
        return result

    compressed_ends = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    )
    if block_table_base_offsets is None:
        base_rows = torch.zeros_like(compressed_ends)
    else:
        base_rows = block_table_base_offsets.to(torch.int64)[
            token_to_req_indices.to(torch.int64)
        ] * int(compressed_block_size)
    compressed_lens = (compressed_ends - base_rows).clamp(
        0, min(width, int(compressed_table_capacity))
    )
    offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
    local = base_rows[:, None] + offsets[None, :]
    valid = offsets[None, :] < compressed_lens[:, None]
    result.copy_(torch.where(valid, local, torch.full_like(local, -1)).to(torch.int32))
    return result


@triton.jit
def _dsv4_combine_dense_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    positions_ptr,
    token_to_req_indices_ptr,
    seq_lens_ptr,
    compressed_lens_ptr,
    gather_lens_ptr,
    workspace_width,
    compressed_base,
    combined_topk: tl.constexpr,
    compress_ratio: tl.constexpr,
    window_size: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * candidate_block + tl.arange(0, candidate_block)
    mask = offsets < combined_topk

    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)
    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    gather_len = tl.load(gather_lens_ptr + req_idx).to(tl.int32)
    gather_start = seq_len - gather_len
    if compress_ratio > 1:
        compressed_len = tl.minimum(
            (pos + 1) // compress_ratio,
            tl.load(compressed_lens_ptr + req_idx).to(tl.int32),
        )
    else:
        compressed_len = tl.full((), 0, tl.int32)
    swa_len = tl.minimum(pos + 1, window_size)
    total_len = compressed_len + swa_len

    request_base = workspace_width * req_idx
    values = tl.full((candidate_block,), -1, tl.int32)
    is_compressed = offsets < compressed_len
    values = tl.where(is_compressed, request_base + offsets, values)

    swa_offsets = offsets - compressed_len
    is_swa = (offsets >= compressed_len) & (offsets < total_len)
    swa_values = (
        request_base + compressed_base + swa_offsets + pos - swa_len + 1 - gather_start
    )
    values = tl.where(is_swa, swa_values, values)

    tl.store(
        combined_indices_ptr + token_idx * combined_indices_stride + offsets,
        values,
        mask=mask,
    )
    tl.store(combined_lens_ptr + token_idx, total_len, mask=block_idx == 0)


def dsv4_combine_dense_swa_indices(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    compressed_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    workspace_width: int,
    compressed_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense-compressed plus SWA sparse prefill indices."""

    num_tokens = positions.numel()
    combined_topk = (
        (
            max(compressed_base + window_size, 1)
            + DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
            - 1
        )
        // DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
        * DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=positions.device)
    if num_tokens == 0:
        return combined_indices, combined_lens

    candidate_block = 128
    _dsv4_combine_dense_swa_indices_kernel[
        (num_tokens, triton.cdiv(combined_topk, candidate_block))
    ](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        positions,
        token_to_req_indices.to(torch.int32),
        seq_lens.to(torch.int32),
        compressed_lens.to(torch.int32),
        gather_lens.to(torch.int32),
        workspace_width,
        compressed_base,
        combined_topk=combined_topk,
        compress_ratio=compress_ratio,
        window_size=window_size,
        candidate_block=candidate_block,
    )
    return combined_indices, combined_lens


@triton.jit(do_not_specialize=["block_table_stride", "max_blocks_per_seq"])
def _dsv4_decode_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    max_blocks_per_seq,
    has_valid_token: tl.constexpr,
    window_size: tl.constexpr,
    block_size: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if has_valid_token:
        is_valid = tl.load(is_valid_token_ptr + token_idx)
        if not is_valid:
            tl.store(swa_lens_ptr + token_idx, 0)
            return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)

    query_start = tl.load(query_start_loc_ptr + req_idx).to(tl.int32)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int32)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    prefix_len = seq_len - query_len
    pos = prefix_len + token_idx - query_start

    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1
    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + token_idx, swa_len)

    for i in range(0, window_size, candidate_block):
        offsets = i + tl.arange(0, candidate_block)
        mask = offsets < window_size
        pos_offsets = start_pos + offsets
        valid = offsets < swa_len
        block_indices = pos_offsets // block_size
        if block_table_base_offsets_ptr is not None:
            block_indices -= tl.load(block_table_base_offsets_ptr + req_idx)
        valid = valid & (block_indices >= 0) & (block_indices < max_blocks_per_seq)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=valid,
            other=-1,
        )
        block_offsets = pos_offsets % block_size
        slot_ids = block_numbers * block_size + block_offsets
        values = tl.where(valid & (block_numbers >= 0), slot_ids, -1)
        tl.store(
            swa_indices_ptr + token_idx * swa_indices_stride + offsets,
            values,
            mask=mask,
        )


def dsv4_decode_swa_indices_and_lens(
    *,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    window_size: int,
    block_size: int,
    block_table_base_offsets: torch.Tensor | None = None,
    is_valid_token: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    out_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DeepSeek V4 decode SWA KV slot indices once per metadata step."""

    num_tokens = token_to_req_indices.shape[0]
    if out_indices is None:
        out_indices = torch.empty(
            (num_tokens, window_size),
            dtype=torch.int32,
            device=seq_lens.device,
        )
    if out_lens is None:
        out_lens = torch.empty(num_tokens, dtype=torch.int32, device=seq_lens.device)
    if num_tokens == 0:
        return out_indices, out_lens
    if is_valid_token is None:
        is_valid_token = torch.empty(0, dtype=torch.bool, device=seq_lens.device)
    else:
        is_valid_token = is_valid_token[:num_tokens].to(
            device=seq_lens.device,
            dtype=torch.bool,
        )

    candidate_block = min(1024, triton.next_power_of_2(window_size))
    block_table_i32 = _as_int32_block_table(block_table)
    _dsv4_decode_swa_indices_and_lens_kernel[(num_tokens,)](
        out_indices,
        out_indices.stride(0),
        out_lens,
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        token_to_req_indices.to(torch.int32),
        is_valid_token,
        block_table_i32,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table_i32.stride(0),
        block_table_i32.shape[-1],
        is_valid_token.numel() != 0,
        window_size=window_size,
        block_size=block_size,
        candidate_block=candidate_block,
    )
    return out_indices, out_lens


@triton.jit
def _dsv4_compressed_slot_mapping_kernel(
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    pad_id: tl.constexpr,
    candidate_block: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + req_idx).to(tl.int32)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int32)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
    start_pos = seq_len - query_len

    for i in range(0, query_len, candidate_block):
        offsets = i + tl.arange(0, candidate_block)
        mask = offsets < query_len
        pos = start_pos + offsets
        valid = (pos + 1) % compress_ratio == 0
        compressed_pos = pos // compress_ratio
        block_ids = compressed_pos // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_ids,
            mask=mask & valid,
            other=0,
        ).to(tl.int64)
        slot_ids = block_numbers * block_size + compressed_pos % block_size
        values = tl.where(valid, slot_ids, pad_id)
        tl.store(slot_mapping_ptr + query_start + offsets, values, mask=mask)


def dsv4_compressed_slot_mapping(
    *,
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build compressed KV slot mapping for DeepSeek V4."""

    if out is None:
        out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
    out.fill_(-1)
    slot_mapping = out[:num_tokens]
    if num_tokens == 0:
        return slot_mapping

    _dsv4_compressed_slot_mapping_kernel[(block_table.shape[0],)](
        slot_mapping,
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        block_table.stride(0),
        block_size=block_size,
        compress_ratio=compress_ratio,
        pad_id=-1,
        candidate_block=1024,
    )
    return slot_mapping


@triton.jit
def _dsv4_indexer_decode_metadata_kernel(
    out_block_tables_ptr,
    out_block_tables_stride,
    out_context_lens_ptr,
    positions_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_table_base_offsets_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    compress_ratio: tl.constexpr,
    cache_block_size: tl.constexpr,
    max_blocks: tl.constexpr,
    candidate_block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    req = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int32)
    req_valid = (req >= 0) & (req < rows)
    safe_req = tl.maximum(0, tl.minimum(req, rows - 1))
    base_logical_page = tl.zeros((), dtype=tl.int64)
    if block_table_base_offsets_ptr is not None:
        base_logical_page = tl.load(block_table_base_offsets_ptr + safe_req).to(
            tl.int64
        )
    compressed_lens = tl.maximum(
        ((pos + 1) // compress_ratio) - base_logical_page * cache_block_size,
        0,
    )
    num_valid_pages = tl.zeros((), dtype=tl.int64)
    for col_start in range(0, max_blocks, candidate_block):
        col_offsets = col_start + tl.arange(0, candidate_block)
        col_mask = col_offsets < max_blocks
        in_cols = col_offsets < cols
        safe_col = tl.where(in_cols, col_offsets, 0)
        bt_load_mask = col_mask & in_cols & req_valid
        bt_vals = tl.load(
            block_table_ptr + safe_req * block_table_stride + safe_col,
            mask=bt_load_mask,
            other=0,
        )
        page_valid = (bt_vals >= 0) & in_cols
        final_mask = page_valid & req_valid & col_mask
        masked_bt = tl.where(final_mask, bt_vals, 0)
        tl.store(
            out_block_tables_ptr + token_idx * out_block_tables_stride + col_offsets,
            masked_bt,
            mask=col_mask,
        )
        num_valid_pages += tl.sum(final_mask.to(tl.int64), axis=0)
    available_lens = num_valid_pages * cache_block_size
    context_len_val = tl.minimum(compressed_lens, available_lens)
    context_len_val = tl.where(req_valid, context_len_val, 0)
    tl.store(out_context_lens_ptr + token_idx, context_len_val.to(tl.int32))


def dsv4_indexer_decode_metadata_compute(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    max_blocks: int,
    out_context_lens: torch.Tensor,
    out_block_tables: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Build decode-indexer context lengths and block tables in one Triton pass."""
    num_tokens = int(positions.shape[0]) if positions.ndim >= 1 else 0
    if num_tokens == 0:
        return
    if out_context_lens.dtype != torch.int32 or out_block_tables.dtype != torch.int32:
        raise TypeError("output buffers must be int32")
    positions_i64 = positions.to(torch.int64)
    token_to_req_indices_i32 = token_to_req_indices.to(torch.int32)
    block_table_i32 = block_table.to(torch.int32)
    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    candidate_block = min(1024, max(16, triton.next_power_of_2(max_blocks)))
    _dsv4_indexer_decode_metadata_kernel[(num_tokens,)](
        out_block_tables,
        out_block_tables.stride(0),
        out_context_lens,
        positions_i64,
        token_to_req_indices_i32,
        block_table_i32,
        block_table_i32.stride(0),
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        rows=rows,
        cols=cols,
        compress_ratio=int(compress_ratio),
        cache_block_size=int(cache_block_size),
        max_blocks=int(max_blocks),
        candidate_block=candidate_block,
    )


# Fused inverse-RoPE + block-scaled FP8 quant for the V4 attention output
# projection. The caller selects either canonical FP32 scales for portable BMM
# and Hopper DeepGEMM or packed, TMA-aligned UE8M0 scales for Blackwell.
@triton.jit(do_not_specialize=["num_tokens"])
def _dsv4_fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)
    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD
    if pid_token >= num_tokens:
        # Zero-fill the TMA-aligned padding rows of the scale buffer.
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return
    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head
    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start
    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)
    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))
    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)
    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)
    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def dsv4_fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse RoPE + grouped block-scaled FP8 quant of the attention output.

    Returns ``(o_fp8, o_scale)`` in the scale layout requested by the selected
    grouped output projection implementation.
    """
    num_tokens, num_heads, head_dim = o.shape
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    tma_aligned_t = ((num_tokens + 3) // 4) * 4  # get_tma_aligned_size(T, int32)
    scale_inner = (
        (num_scale_blocks + 3) // 4 if tma_aligned_scales else num_scale_blocks
    )
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d), dtype=torch.float8_e4m3fn, device=o.device
    )
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_t, dtype=scale_dtype, device=o.device
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )
    grid = (tma_aligned_t, n_groups * heads_per_group)
    _dsv4_fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_stages=1,
        num_warps=1,
    )
    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)
