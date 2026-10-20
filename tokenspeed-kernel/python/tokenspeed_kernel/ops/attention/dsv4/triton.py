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

"""Triton DSV4 kernel exports.

TODO: Moe kernel implementations into their own dedicated files.
"""

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
    "dsv4_compact_compressed_slot_mapping",
    "dsv4_compressed_slot_mapping",
    "dsv4_fused_csa_indexer_mxfp4_cache_insert",
    "dsv4_fused_indexer_q_rope_hadamard_mxfp4",
    "dsv4_fused_sparse_compress_cache_insert",
    "dsv4_save_compressor_state",
    "triton_dsv4_csa_indexer_fp8_cache_insert",
    "triton_dsv4_prefill",
    "triton_dsv4_swa_cache_insert",
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
        "quant_block_size": frozenset({DEEPSEEK_V4_FP8_QUANT_BLOCK}),
        "rope_dim": frozenset({DEEPSEEK_V4_ROPE_DIM}),
        "cache_layout": frozenset({"fp8_swa_page_planar"}),
        "has_q_out": frozenset({True, False}),
    },
    priority=Priority.PORTABLE,
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
def _dsv4_sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    num_heads,
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

    if sink_ptr is not None:
        max_logit = tl.load(sink_ptr + head_idx).to(tl.float32)
        denominator = tl.full((), 1.0, tl.float32)
    else:
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
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
        # Empty shards/tiles must not introduce exp(-inf - -inf).
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_scale = tl.exp(max_logit - safe_max)
        probabilities = tl.exp(logits - safe_max)
        probabilities = tl.where(valid, probabilities, 0.0)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * kv,
            axis=0,
        )
        denominator = denominator * previous_scale + tl.sum(probabilities, axis=0)
        max_logit = next_max

    if lse_ptr is not None:
        logsumexp = tl.where(
            denominator > 0.0, max_logit + tl.log(denominator), -float("inf")
        )
        tl.store(lse_ptr + token_idx * num_heads + head_idx, logsumexp)
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
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
        "sinks": frozenset({True, False}),
        "return_lse": frozenset({True, False}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
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
    if attn_sink is not None and attn_sink.numel() < q.shape[1]:
        raise ValueError("attention sink must provide one value per query head")

    output = out if out is not None else torch.empty_like(q)
    _dsv4_sparse_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        kv_2d,
        indices_2d,
        lens,
        attn_sink,
        output,
        lse,
        q.shape[1],
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
        "has_extra_segment": frozenset({False, True}),
        "metadata_dtypes": frozenset({torch.int32, torch.int64}),
        "sinks": frozenset({True}),
        "topk_layout": frozenset({"global_slots"}),
    },
    priority=Priority.PORTABLE,
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
    from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

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


@triton.jit
def _dsv4_fused_indexer_q_rope_hadamard_mxfp4_serial_four_block_kernel(
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
    NUM_QUANT_BLOCKS: tl.constexpr,
    HADAMARD_SCALE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Reuse Q/RoPE setup while producing the four fixed MXFP4 blocks."""

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

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
    for quant_block_idx in tl.static_range(0, NUM_QUANT_BLOCKS):
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
    tl.store(weights_out_ptr + token_idx * weights_out_stride + head_idx, weights)


def _dsv4_use_serial_four_block_indexer_q(
    *,
    num_tokens: int,
    head_dim: int,
    rope_dim: int,
    quant_block: int,
    is_cuda: bool,
    is_hip: bool,
    capability: tuple[int, int] | None,
    shapes_valid: bool,
    dtypes_valid: bool,
    devices_valid: bool,
    inner_strides: tuple[int, int, int, int],
) -> bool:
    """Return whether the exact 8192-token SM100 specialization is eligible."""

    return (
        num_tokens == 8192
        and head_dim == DEEPSEEK_V4_INDEXER_DIM
        and rope_dim == DEEPSEEK_V4_ROPE_DIM
        and quant_block == DEEPSEEK_V4_MXFP4_BLOCK_SIZE
        and is_cuda
        and not is_hip
        and capability == (10, 0)
        and shapes_valid
        and dtypes_valid
        and devices_valid
        and inner_strides == (1, 1, 1, 1)
    )


@functools.cache
def _dsv4_indexer_q_cuda_capability(
    device: torch.device | int | None,
) -> tuple[int, int] | None:
    try:
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return None
        return torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None


@functools.cache
def _log_serial_four_block_indexer_q_selection(capability: tuple[int, int]) -> None:
    logger.info(
        "DeepSeek V4 Indexer-Q launch selection: serial_four_block=True "
        f"tokens=8192 capability={capability!s}",
    )


def _dsv4_serial_four_block_indexer_q_supported(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    if index_q.dim() != 3:
        return False
    num_tokens, num_heads, head_dim = index_q.shape
    capability = (
        _dsv4_indexer_q_cuda_capability(index_q.device) if num_tokens == 8192 else None
    )
    supported = _dsv4_use_serial_four_block_indexer_q(
        num_tokens=num_tokens,
        head_dim=head_dim,
        rope_dim=DEEPSEEK_V4_ROPE_DIM,
        quant_block=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        is_cuda=index_q.is_cuda,
        is_hip=getattr(torch.version, "hip", None) is not None,
        capability=capability,
        shapes_valid=(
            positions.shape == (num_tokens,)
            and cos_sin_cache.dim() == 2
            and cos_sin_cache.shape[1] == DEEPSEEK_V4_ROPE_DIM
            and weights.shape == (num_tokens, num_heads)
        ),
        dtypes_valid=(
            index_q.dtype == torch.bfloat16
            and positions.dtype == torch.int64
            and cos_sin_cache.dtype == torch.float32
            and weights.dtype in (torch.bfloat16, torch.float32)
        ),
        devices_valid=(
            positions.device == index_q.device
            and cos_sin_cache.device == index_q.device
            and weights.device == index_q.device
        ),
        inner_strides=(
            index_q.stride(-1),
            positions.stride(-1) if positions.dim() == 1 else 0,
            cos_sin_cache.stride(-1) if cos_sin_cache.dim() == 2 else 0,
            weights.stride(-1) if weights.dim() == 2 else 0,
        ),
    )
    if supported:
        assert capability is not None
        _log_serial_four_block_indexer_q_selection(capability)
    return supported


def dsv4_fused_indexer_q_rope_hadamard_mxfp4(
    *,
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    prefer_serial_four_block: bool,
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

    use_serial_four_block = prefer_serial_four_block and (
        _dsv4_serial_four_block_indexer_q_supported(
            index_q,
            positions,
            cos_sin_cache,
            weights,
        )
    )
    kernel = (
        _dsv4_fused_indexer_q_rope_hadamard_mxfp4_serial_four_block_kernel
        if use_serial_four_block
        else _dsv4_fused_indexer_q_rope_hadamard_mxfp4_kernel
    )
    grid = (
        (num_tokens, num_heads)
        if use_serial_four_block
        else (num_tokens, num_heads, head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE)
    )
    launch_kwargs = {
        "HEAD_DIM": head_dim,
        "ROPE_DIM": DEEPSEEK_V4_ROPE_DIM,
        "QUANT_BLOCK": DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        "HALF_BLOCK": DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2,
        "HADAMARD_SCALE": head_dim**-0.5,
        "TRITON_BLOCK_SIZE": triton.next_power_of_2(head_dim),
        "num_warps": 4,
    }
    if use_serial_four_block:
        launch_kwargs["NUM_QUANT_BLOCKS"] = head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    kernel[grid](
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
        **launch_kwargs,
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
    kv_write_mask_ptr=None,
):
    token_idx = tl.program_id(0)

    state_slot = tl.load(slot_mapping_ptr + token_idx)
    if state_slot < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_slot = tl.load(kv_slot_mapping_ptr + token_idx)
    write_valid = kv_slot >= 0
    if kv_write_mask_ptr is not None:
        write_valid = write_valid & tl.load(kv_write_mask_ptr + token_idx)
    if not write_valid:
        return
    kv_slot = tl.maximum(kv_slot, 0)

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
        mask=(block >= NOPE_HEAD_DIM) & mask & write_valid,
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
            f"DeepSeek V4 sparse-compress launch selection: capability={capability!s} "
            f"num_warps={(16 if supported else 4):d}",
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
    block_table_base_offsets: torch.Tensor | None,
    kv_write_mask: torch.Tensor | None,
) -> None:
    """Compress replicated state and store owned FP8 payload, scale and RoPE.

    Args:
        block_table_base_offsets: Optional per-request base offsets added to
            the compressor block-table row before it is indexed.
        kv_write_mask: Optional contiguous boolean vector covering the input
            slots on the KV slot device. False entries suppress every store,
            including scale padding, and may safely address reserved page 0.
            None preserves the unmasked compression path.

    Returns:
        None. Selected cache rows are updated in place.
    """
    num_actual = min(
        compressor_slot_mapping.numel(),
        positions.numel(),
        kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return
    if kv_write_mask is not None and (
        kv_write_mask.ndim != 1
        or kv_write_mask.dtype != torch.bool
        or kv_write_mask.device != kv_slot_mapping.device
        or kv_write_mask.numel() < num_actual
        or not kv_write_mask.is_contiguous()
    ):
        raise ValueError(
            "compressed cache write mask must be a contiguous bool vector "
            "on the KV slot device and cover all slots"
        )
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
        kv_write_mask_ptr=(
            kv_write_mask[:num_actual] if kv_write_mask is not None else None
        ),
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
        "page_size": frozenset({64}),
        "compress_ratio": frozenset({4}),
        "cache_format": frozenset({"fp8_scaled_page_planar"}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsv4_csa_indexer_fp8_cache_insert(
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
    base_offsets_ptr,
    valid_lens_ptr,
    num_requests,
    table_width,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx).to(tl.int64)
    query_valid = (req_idx >= 0) & (req_idx < num_requests)
    if is_valid_token_ptr is not None:
        query_valid &= tl.load(is_valid_token_ptr + token_idx)
    base = tl.full((), 0, tl.int64)
    if base_offsets_ptr is not None:
        base = tl.load(base_offsets_ptr + req_idx, mask=query_valid, other=0).to(
            tl.int64
        )
    local_count = tl.zeros((), dtype=tl.int32)
    scan_end = tl.zeros((), dtype=tl.int32)

    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE).to(tl.int64)
        mask = offset < topk
        selected = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        candidates_valid = mask & (selected >= 0) & query_valid
        # Trim only upstream padding. Base/owner filtering may leave holes,
        # and its per-layer readable count must not shorten the scan prefix.
        scan_end = tl.maximum(
            scan_end,
            tl.max(tl.where(candidates_valid, offset + 1, 0).to(tl.int32), axis=0),
        )
        local_idx = selected.to(tl.int64) - base * block_size
        block_indices = local_idx // block_size
        valid = candidates_valid & (local_idx >= 0) & (block_indices < table_width)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid &= block_numbers >= 0
        block_offsets = local_idx % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        if valid_lens_ptr is not None:
            local_count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, scan_end)
    if valid_lens_ptr is not None:
        tl.store(valid_lens_ptr + token_idx, local_count)


def dsv4_compute_global_topk_indices_and_lens(
    *,
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor | None = None,
    block_table_base_offsets: torch.Tensor | None = None,
    out_valid_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CSA top-k entries through a physical page table in one kernel.

    Args:
        topk_indices: Int32 compressed entry IDs [queries, width], padded by -1.
            The upstream indexer must enforce each query's causal bound and
            keep the candidate prefix length unchanged across layers within a
            forward when the attention backend reuses its schedule.
        token_to_req_indices: Int32/int64 vector covering every query's request
            row. Noncontiguous metadata vectors are copied to contiguous storage.
        block_table: Int32/int64 physical page IDs [requests, pages]; negative
            pages are unreadable. DCP callers prepare local pages before this op.
            Physical slots must fit int32. Both tables allow strided rows, with
            unit column stride. All input tensors must share a device.
        block_size: Positive number of compressed entries per physical page.
        is_valid_token: Optional boolean vector covering every query; false
            entries identify padded queries.
        block_table_base_offsets: Optional int32/int64 vector of absolute first
            logical pages [requests]. Top-k IDs stay absolute when supplied.
        out_valid_lens: Optional contiguous int32 [queries] output receiving
            the number of readable slots after all filtering, on the same device.

    Returns:
        Int32 physical slots [queries, width], with invalid entries set to -1,
        and int32 scan lengths [queries]: one past the last nonnegative input
        candidate, zero for empty, masked or out-of-range queries. For upstream
        candidates with trailing padding this equals their valid count. Base
        and page filtering preserve this prefix even when every output is -1.
        Every slot is written, preserving candidate order, holes and duplicates.

    Callers provide metadata with the documented shapes, dtypes and device.
    Validation covers the top-k tensor, column strides and output count buffer.
    """

    if topk_indices.dtype != torch.int32:
        raise TypeError(f"topk_indices must be int32, got {topk_indices.dtype}")
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be 2-D, got {tuple(topk_indices.shape)}")
    num_tokens = topk_indices.shape[0]
    if topk_indices.stride(1) != 1 or block_table.stride(1) != 1:
        raise ValueError("top-k mapping requires contiguous rows")
    if out_valid_lens is not None and (
        out_valid_lens.shape != (num_tokens,)
        or out_valid_lens.dtype != torch.int32
        or out_valid_lens.device != topk_indices.device
        or not out_valid_lens.is_contiguous()
    ):
        raise ValueError("local counts must be contiguous int32 on the query device")
    global_topk_indices = torch.empty_like(
        topk_indices, memory_format=torch.contiguous_format
    )
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    rows, cols = block_table.shape
    if num_tokens == 0 or topk_indices.shape[1] == 0 or rows == 0:
        global_topk_indices.fill_(-1)
        topk_lens.zero_()
        if out_valid_lens is not None:
            out_valid_lens.zero_()
        return global_topk_indices, topk_lens
    if is_valid_token is not None:
        is_valid_token = is_valid_token[:num_tokens].contiguous()
    token_to_req_indices = token_to_req_indices[:num_tokens].contiguous()
    if block_table_base_offsets is not None:
        block_table_base_offsets = block_table_base_offsets[:rows].contiguous()
    if not topk_indices.is_cuda:
        req_idx = token_to_req_indices.to(torch.int64)
        req_valid = (req_idx >= 0) & (req_idx < rows)
        if is_valid_token is not None:
            req_valid &= is_valid_token
        candidates_valid = (topk_indices >= 0) & req_valid[:, None]
        candidate_ends = torch.arange(
            1, topk_indices.shape[1] + 1, dtype=torch.int32, device=topk_indices.device
        )
        topk_lens.copy_(torch.where(candidates_valid, candidate_ends, 0).amax(dim=1))
        if cols == 0:
            global_topk_indices.fill_(-1)
            if out_valid_lens is not None:
                out_valid_lens.zero_()
            return global_topk_indices, topk_lens
        safe_req = req_idx.clamp(0, rows - 1)
        local = topk_indices.to(torch.int64)
        if block_table_base_offsets is not None:
            local = (
                local
                - block_table_base_offsets.to(torch.int64)[safe_req, None] * block_size
            )
        block_indices = torch.div(local, block_size, rounding_mode="floor")
        valid = candidates_valid & (block_indices >= 0) & (block_indices < cols)
        safe_block = block_indices.long().clamp(0, cols - 1)
        block_numbers = block_table[safe_req[:, None], safe_block]
        valid &= block_numbers >= 0
        global_topk_indices.copy_(
            torch.where(
                valid,
                block_numbers.to(torch.int64) * block_size + local % block_size,
                -1,
            )
        )
        if out_valid_lens is not None:
            out_valid_lens.copy_(valid.sum(dim=1, dtype=torch.int32))
        return global_topk_indices, topk_lens

    _dsv4_compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        is_valid_token,
        block_table_base_offsets,
        out_valid_lens,
        rows,
        cols,
        block_size=block_size,
        topk=topk_indices.shape[-1],
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


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


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_reqs",
        "max_blocks_per_seq",
    ]
)
def _dsv4_compact_compressed_slot_mapping_kernel(
    slot_mapping_ptr,
    token_to_req_indices_ptr,
    token_to_req_indices_stride,
    query_start_loc_ptr,
    query_start_loc_stride,
    seq_lens_ptr,
    seq_lens_stride,
    is_valid_token_ptr,
    is_valid_token_stride,
    block_table_ptr,
    block_table_stride,
    block_table_base_offsets_ptr,
    block_table_base_offsets_stride,
    num_tokens,
    num_reqs,
    max_blocks_per_seq,
    has_valid_token: tl.constexpr,
    has_block_table_base_offsets: tl.constexpr,
    block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_is_valid = token_idx < num_tokens
    if has_valid_token:
        token_is_valid &= tl.load(
            is_valid_token_ptr + token_idx * is_valid_token_stride,
            mask=token_is_valid,
            other=False,
        )

    req_idx = tl.load(
        token_to_req_indices_ptr + token_idx * token_to_req_indices_stride,
        mask=token_is_valid,
        other=-1,
    ).to(tl.int32)
    req_is_valid = token_is_valid & (req_idx >= 0) & (req_idx < num_reqs)
    query_start = tl.load(
        query_start_loc_ptr + req_idx * query_start_loc_stride,
        mask=req_is_valid,
        other=0,
    ).to(tl.int64)
    query_end = tl.load(
        query_start_loc_ptr + (req_idx + 1) * query_start_loc_stride,
        mask=req_is_valid,
        other=0,
    ).to(tl.int64)
    seq_len = tl.load(
        seq_lens_ptr + req_idx * seq_lens_stride,
        mask=req_is_valid,
        other=0,
    ).to(tl.int64)
    position = seq_len - (query_end - query_start) + token_idx - query_start
    compressed_position = position // compress_ratio
    logical_page = compressed_position // block_size
    offset = compressed_position % block_size

    base_page = tl.zeros((), dtype=tl.int64)
    if has_block_table_base_offsets:
        base_page = tl.load(
            block_table_base_offsets_ptr + req_idx * block_table_base_offsets_stride,
            mask=req_is_valid,
            other=0,
        ).to(tl.int64)
    table_page = logical_page - base_page
    page_is_valid = (
        req_is_valid
        & (position >= 0)
        & ((position + 1) % compress_ratio == 0)
        & (table_page >= 0)
        & (table_page < max_blocks_per_seq)
    )
    page_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + table_page,
        mask=page_is_valid,
        other=-1,
    ).to(tl.int64)
    slot = page_id * block_size + offset
    tl.store(
        slot_mapping_ptr + token_idx,
        tl.where(page_is_valid & (page_id >= 0), slot, -1),
    )


def dsv4_compact_compressed_slot_mapping(
    *,
    num_tokens: int,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    block_table_base_offsets: torch.Tensor | None,
    is_valid_token: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Build slots for a grouped DeepSeek V4 compressed-cache page table.

    The table may contain full logical rows or compact rows accompanied by a
    per-request base logical-page offset. Invalid and non-boundary tokens map
    to ``-1``.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if token_to_req_indices.numel() < num_tokens:
        raise ValueError(
            "token_to_req_indices must cover every token: "
            f"tokens={num_tokens}, request_indices={token_to_req_indices.numel()}"
        )
    if query_start_loc.dim() != 1 or seq_lens.dim() != 1:
        raise ValueError("query_start_loc and seq_lens must be 1-D")
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2-D, got {tuple(block_table.shape)}")
    if block_table_base_offsets is not None and block_table_base_offsets.dim() != 1:
        raise ValueError("block_table_base_offsets must be 1-D")
    if is_valid_token is not None:
        if is_valid_token.numel() != num_tokens:
            if is_valid_token.numel() <= 0 or num_tokens % is_valid_token.numel() != 0:
                raise ValueError(
                    "is_valid_token must cover or evenly expand across every token: "
                    f"tokens={num_tokens}, validity={is_valid_token.numel()}"
                )
            is_valid_token = is_valid_token.repeat_interleave(
                num_tokens // is_valid_token.numel()
            )
        is_valid_token = is_valid_token.to(device=seq_lens.device, dtype=torch.bool)
    if out is None:
        out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
    if out.dim() != 1 or out.dtype != torch.int64 or out.numel() < num_tokens:
        raise ValueError(
            "out must be a 1-D int64 tensor with at least num_tokens entries, got "
            f"shape={tuple(out.shape)} dtype={out.dtype} tokens={num_tokens}"
        )
    if out.stride(0) != 1:
        raise ValueError("out must be contiguous")

    slot_mapping = out[:num_tokens]
    if out.numel() == 0:
        return slot_mapping

    num_reqs = min(
        seq_lens.numel(),
        max(0, query_start_loc.numel() - 1),
        block_table.shape[0],
        (
            block_table_base_offsets.numel()
            if block_table_base_offsets is not None
            else seq_lens.numel()
        ),
    )
    if seq_lens.is_cuda:
        req_indices_i32 = token_to_req_indices.to(torch.int32)
        query_start_i32 = query_start_loc.to(torch.int32)
        seq_lens_i32 = seq_lens.to(torch.int32)
        block_table_i32 = _as_int32_block_table(block_table)
        validity_arg = seq_lens_i32 if is_valid_token is None else is_valid_token
        base_offsets_arg = (
            seq_lens_i32
            if block_table_base_offsets is None
            else block_table_base_offsets.to(torch.int32)
        )
        _dsv4_compact_compressed_slot_mapping_kernel[(out.numel(),)](
            out,
            req_indices_i32,
            req_indices_i32.stride(0),
            query_start_i32,
            query_start_i32.stride(0),
            seq_lens_i32,
            seq_lens_i32.stride(0),
            validity_arg,
            validity_arg.stride(0),
            block_table_i32,
            block_table_i32.stride(0),
            base_offsets_arg,
            base_offsets_arg.stride(0),
            num_tokens,
            num_reqs,
            block_table_i32.shape[1],
            has_valid_token=is_valid_token is not None,
            has_block_table_base_offsets=block_table_base_offsets is not None,
            block_size=block_size,
            compress_ratio=compress_ratio,
        )
        return slot_mapping

    out.fill_(-1)
    if num_tokens == 0 or num_reqs == 0:
        return slot_mapping
    req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
    valid_req = (req_idx >= 0) & (req_idx < num_reqs)
    safe_req = req_idx.clamp(0, num_reqs - 1)
    query_starts = query_start_loc[safe_req].to(torch.int64)
    query_lens = query_start_loc[safe_req + 1].to(torch.int64) - query_starts
    positions = (
        seq_lens[safe_req].to(torch.int64)
        - query_lens
        + torch.arange(num_tokens, dtype=torch.int64, device=seq_lens.device)
        - query_starts
    )
    compressed_positions = torch.div(positions, compress_ratio, rounding_mode="floor")
    table_pages = torch.div(compressed_positions, block_size, rounding_mode="floor")
    if block_table_base_offsets is not None:
        table_pages -= block_table_base_offsets[safe_req].to(torch.int64)
    valid_page = (
        valid_req
        & (positions >= 0)
        & (((positions + 1) % compress_ratio) == 0)
        & (table_pages >= 0)
        & (table_pages < block_table.shape[1])
    )
    safe_page = table_pages.clamp(0, max(0, block_table.shape[1] - 1))
    if block_table.shape[1] == 0:
        page_ids = torch.full_like(table_pages, -1)
    else:
        page_ids = block_table.to(torch.int64)[safe_req, safe_page]
    valid_page &= page_ids >= 0
    if is_valid_token is not None:
        valid_page &= is_valid_token[:num_tokens]
    slots = page_ids * block_size + compressed_positions % block_size
    slot_mapping.copy_(torch.where(valid_page, slots, torch.full_like(slots, -1)))
    return slot_mapping


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
