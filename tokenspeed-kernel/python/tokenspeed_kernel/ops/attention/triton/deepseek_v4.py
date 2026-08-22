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
from tokenspeed_kernel.platform import current_platform

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
    "deepseek_v4_build_dense_prefill_local_compressed_indices",
    "deepseek_v4_combine_dense_swa_indices",
    "deepseek_v4_combine_topk_swa_indices",
    "deepseek_v4_compressed_slot_mapping",
    "deepseek_v4_compute_global_topk_indices_and_lens",
    "deepseek_v4_decode_swa_indices_and_lens",
    "deepseek_v4_dequantize_and_gather_k_cache",
    "deepseek_v4_fused_csa_indexer_mxfp4_cache_insert",
    "deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4",
    "deepseek_v4_fused_paged_sparse_attn",
    "deepseek_v4_fused_sparse_compress_cache_insert",
    "deepseek_v4_gather_indexer_mxfp4_cache",
    "deepseek_v4_gather_paged_indexer_fp8_padded",
    "deepseek_v4_gather_slots_bf16",
    "deepseek_v4_indexer_decode_metadata_compute",
    "deepseek_v4_indexer_select_all",
    "deepseek_v4_indexer_mqa_logits",
    "deepseek_v4_kv_rope_quant_insert",
    "deepseek_v4_q_norm_rope",
    "deepseek_v4_sparse_attn",
    "deepseek_v4_save_compressor_state",
    "write_deepseek_v4_indexer_mxfp4_cache_cuda",
]


def _as_int32_block_table(block_table: torch.Tensor) -> torch.Tensor:
    """Return an int32 table with unit column stride for Triton row indexing."""

    block_table_i32 = block_table.to(torch.int32)
    if block_table_i32.stride(-1) != 1:
        block_table_i32 = block_table_i32.contiguous()
    return block_table_i32


@triton.jit
def _deepseek_v4_mxfp4_e2m1_nibble(x):
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
def _deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4_kernel(
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
    lo = _deepseek_v4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _deepseek_v4_mxfp4_e2m1_nibble(x_hi * inv_scale)
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


def deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4(
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

    _deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4_kernel[
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
def _deepseek_v4_fused_sparse_compress_cache_kernel(
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


def deepseek_v4_fused_sparse_compress_cache_insert(
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
    _deepseek_v4_fused_sparse_compress_cache_kernel[(num_actual,)](
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
def _deepseek_v4_fused_csa_indexer_mxfp4_cache_kernel(
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
    lo = _deepseek_v4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _deepseek_v4_mxfp4_e2m1_nibble(x_hi * inv_scale)
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


def deepseek_v4_fused_csa_indexer_mxfp4_cache_insert(
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
    _deepseek_v4_fused_csa_indexer_mxfp4_cache_kernel[
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
def _deepseek_v4_save_compressor_state_kernel(
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


def deepseek_v4_save_compressor_state(
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
    _deepseek_v4_save_compressor_state_kernel[(num_actual,)](
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
def _deepseek_v4_indexer_mxfp4_cache_write_kernel(
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
    lo = _deepseek_v4_mxfp4_e2m1_nibble(x_lo * inv_scale)
    hi = _deepseek_v4_mxfp4_e2m1_nibble(x_hi * inv_scale)
    packed = lo | (hi << 4)
    scale = (exponent + 127.0).to(tl.uint8)

    page = slot // cache_block_size
    pos = slot % cache_block_size
    page_base = cache_ptr + page.to(tl.int64) * cache_stride0
    value_base = page_base + pos * TOKEN_STRIDE + block_base // 2
    scale_base = page_base + cache_block_size * TOKEN_STRIDE + pos * SCALE_DIM
    tl.store(value_base + offsets, packed)
    tl.store(scale_base + block_idx, scale)


def write_deepseek_v4_indexer_mxfp4_cache_cuda(
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
    _deepseek_v4_indexer_mxfp4_cache_write_kernel[
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
def _deepseek_v4_gather_indexer_mxfp4_cache_kernel(
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


def deepseek_v4_gather_indexer_mxfp4_cache(
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
        raise ValueError("deepseek_v4_gather_indexer_mxfp4_cache requires CUDA cache")
    if not slot_mapping.is_cuda:
        raise ValueError("deepseek_v4_gather_indexer_mxfp4_cache requires CUDA slots")
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
    _deepseek_v4_gather_indexer_mxfp4_cache_kernel[(triton.cdiv(rows, block_rows),)](
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
def _deepseek_v4_dequantize_and_gather_k_kernel(
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


def _deepseek_v4_gather_launch_config(
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


def deepseek_v4_dequantize_and_gather_k_cache(
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
        num_workers, num_warps = _deepseek_v4_gather_launch_config(num_reqs, max_rows)
    else:
        num_workers, num_warps = 128, 4
    block_table_i32 = _as_int32_block_table(block_table)
    _deepseek_v4_dequantize_and_gather_k_kernel[(num_reqs, num_workers)](
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


# Indexer selection when every candidate survives. ``topk_tokens`` keys are
# taken per row, so a row with no more than that many candidates selects all of
# them and the scores decide nothing. The dense path still pays for a
# ``[tokens, max_len]`` masked copy, a sort and the -1 rewrite to reach that
# answer; this writes it directly.
#
# Order within a row is not part of the contract: the selected indices address
# rows of a cache that attention reduces under a softmax, so any permutation of
# the same set gives the same output. What does matter is that the valid
# entries stay packed at the front, because the consumer reports a length
# rather than a mask.
@triton.jit(do_not_specialize=["num_tokens", "topk"])
def _deepseek_v4_indexer_select_all_kernel(
    out_ptr,
    lens_ptr,
    row_starts_ptr,
    out_stride_token,
    num_tokens,
    topk,
    HAS_ROW_STARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_t >= num_tokens:
        return

    length = tl.load(lens_ptr + pid_t)
    start = tl.load(row_starts_ptr + pid_t) if HAS_ROW_STARTS else 0

    offsets = pid_k * BLOCK + tl.arange(0, BLOCK)
    in_row = offsets < topk
    values = tl.where(offsets < length, start + offsets, -1)
    tl.store(out_ptr + pid_t * out_stride_token + offsets, values, mask=in_row)


def deepseek_v4_indexer_select_all(
    out: torch.Tensor,
    lens: torch.Tensor,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill indexer top-k rows with every candidate, in order.

    Valid for rows whose candidate count does not exceed the top-k width, where
    selection cannot discard anything.

    Args:
        out: ``[num_tokens, topk_tokens]`` int32 destination.
        lens: ``[num_tokens]`` candidate count per row.
        row_starts: optional ``[num_tokens]`` first key index per row; absent
            means every row starts at zero.

    Returns:
        ``out``, for chaining.
    """
    num_tokens, topk = out.shape
    if num_tokens == 0 or topk == 0:
        return out
    block = min(1024, triton.next_power_of_2(topk))
    _deepseek_v4_indexer_select_all_kernel[(num_tokens, triton.cdiv(topk, block))](
        out,
        lens,
        row_starts,
        out.stride(0),
        num_tokens,
        topk,
        HAS_ROW_STARTS=row_starts is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _deepseek_v4_compute_global_topk_indices_and_lens_kernel(
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


def deepseek_v4_compute_global_topk_indices_and_lens(
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

    _deepseek_v4_compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
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
def _deepseek_v4_combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    workspace_width,
    compressed_base,
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
        topk_len = tl.minimum((pos + 1) // compress_ratio, topk)
        swa_len = tl.minimum(pos + 1, window_size)

        topk_offsets = tl.arange(0, padded_topk)
        topk_mask = topk_offsets < topk_len
        topk_values = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + topk_offsets,
            mask=topk_mask,
            other=-1,
        )
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offsets,
            topk_values + workspace_width * batch_idx,
            mask=topk_mask,
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


def deepseek_v4_combine_topk_swa_indices(
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

    _deepseek_v4_combine_topk_swa_indices_kernel[(num_reqs, 128)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        gather_lens.to(torch.int32),
        workspace_width,
        compressed_base,
        topk=topk,
        compress_ratio=compress_ratio,
        window_size=window_size,
        padded_topk=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _deepseek_v4_build_dense_prefill_local_compressed_indices_kernel(
    out_ptr,
    out_stride,
    positions_ptr,
    width: tl.constexpr,
    compress_ratio: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    compressed_len = tl.minimum((position + 1) // compress_ratio, width)
    for start in range(0, width, block):
        offsets = start + tl.arange(0, block)
        mask = offsets < width
        values = tl.where(offsets < compressed_len, offsets, -1)
        tl.store(out_ptr + token_idx * out_stride + offsets, values, mask=mask)


def deepseek_v4_build_dense_prefill_local_compressed_indices(
    *,
    positions: torch.Tensor,
    compress_ratio: int,
    width: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Build C128A/HCA prefill-local compressed prefix indices into `out`."""

    result = out[: positions.numel(), :width]
    if positions.numel() == 0 or width <= 0:
        return result
    if result.stride(1) != 1:
        raise ValueError(
            "dense prefill compressed indices output must be contiguous in the last dim"
        )
    if positions.is_cuda:
        _deepseek_v4_build_dense_prefill_local_compressed_indices_kernel[
            (positions.numel(),)
        ](
            result,
            result.stride(0),
            positions,
            width=width,
            compress_ratio=compress_ratio,
            block=1024,
        )
        return result

    compressed_lens = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp(0, width)
    offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
    local = offsets[None, :].expand(positions.numel(), -1)
    valid = offsets[None, :] < compressed_lens[:, None]
    result.copy_(torch.where(valid, local, torch.full_like(local, -1)).to(torch.int32))
    return result


@triton.jit
def _deepseek_v4_combine_dense_swa_indices_kernel(
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


def deepseek_v4_combine_dense_swa_indices(
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
    _deepseek_v4_combine_dense_swa_indices_kernel[
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
def _deepseek_v4_decode_swa_indices_and_lens_kernel(
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


def deepseek_v4_decode_swa_indices_and_lens(
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
    _deepseek_v4_decode_swa_indices_and_lens_kernel[(num_tokens,)](
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
def _deepseek_v4_compressed_slot_mapping_kernel(
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


def deepseek_v4_compressed_slot_mapping(
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

    _deepseek_v4_compressed_slot_mapping_kernel[(block_table.shape[0],)](
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
def _deepseek_v4_indexer_decode_metadata_kernel(
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


def deepseek_v4_indexer_decode_metadata_compute(
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
    _deepseek_v4_indexer_decode_metadata_kernel[(num_tokens,)](
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


# Weightless per-head RMS norm + forward RoPE over the query's trailing rope
# dims, matching the reference:
#     q *= rsqrt(q.square().mean(-1) + eps)
#     apply_rotary_emb(q[..., -rope_dim:], freqs_cis)
# RoPE pairs adjacent lanes (``offs ^ 1``), so cos/sin are indexed by the pair
# index and the cache holds cos in ``[0, half_rope)`` and sin in
# ``[half_rope, rope_dim)`` -- the same layout the inverse-RoPE kernel below
# consumes, with the sin term's sign flipped.
@triton.jit(do_not_specialize=["num_tokens"])
def _deepseek_v4_q_norm_rope_per_head(
    q_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    num_tokens,
    q_stride_token,
    q_stride_head,
    cache_stride_pos,
    rms_eps,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)
    if pid_token >= num_tokens:
        return

    q_base = q_ptr + pid_token * q_stride_token + pid_head * q_stride_head
    offsets = tl.arange(0, HEAD_DIM)
    x_raw = tl.load(q_base + offsets).to(tl.float32)

    # Weightless RMS norm across the full head, before RoPE.
    var = tl.sum(x_raw * x_raw, axis=0) / HEAD_DIM
    scale = tl.rsqrt(var + rms_eps)
    x = x_raw * scale

    rope_start: tl.constexpr = HEAD_DIM - ROPE_DIM
    is_rope = offsets >= rope_start
    rope_local = offsets - rope_start

    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)

    # The norm is a single per-head scale, so the paired lane can be reloaded
    # raw and scaled rather than shuffled out of ``x``.
    partner = (
        tl.load(q_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32) * scale
    )

    is_even = (rope_local & 1) == 0
    rotated = tl.where(
        is_even, x * cos_v - partner * sin_v, x * cos_v + partner * sin_v
    )
    x = tl.where(is_rope, rotated, x)

    tl.store(q_base + offsets, x.to(q_ptr.dtype.element_ty))


def deepseek_v4_q_norm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int = 64,
) -> None:
    """Weightless per-head RMS norm + forward RoPE on the query, in place.

    Args:
        q: ``[num_tokens, num_heads, head_dim]``, mutated in place. The norm
            spans ``head_dim``; RoPE applies only to the trailing ``rope_dim``.
        positions: ``[num_tokens]`` absolute token positions.
        cos_sin_cache: ``[max_position, rope_dim]`` float32, cos in the first
            half and sin in the second.
        rms_norm_eps: epsilon added to the mean square before the reciprocal
            square root.
        rope_dim: width of the rotary section at the end of each head.
    """
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if rope_dim > head_dim:
        raise ValueError(f"rope_dim={rope_dim} exceeds head_dim={head_dim}")
    _deepseek_v4_q_norm_rope_per_head[(num_tokens, num_heads)](
        q,
        positions,
        cos_sin_cache,
        num_tokens,
        q.stride(0),
        q.stride(1),
        cos_sin_cache.stride(0),
        rms_norm_eps,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HALF_ROPE=rope_dim // 2,
        num_warps=4,
    )


# Capture-safe paged gather for the indexer's FP8 keys. The torch gather reads
# the cumulative length back to the host to size its output, which a CUDA graph
# capture forbids; this writes into a fixed [tokens, max_len] window instead, so
# every shape is known before capture begins. Rows past a token's context are
# left as written by the caller.
@triton.jit(do_not_specialize=["num_tokens", "max_len"])
def _deepseek_v4_gather_paged_indexer_fp8_padded_kernel(
    cache_ptr,
    context_lens_ptr,
    block_table_ptr,
    values_ptr,
    scales_ptr,
    cache_stride_page,
    block_table_stride,
    max_blocks,
    num_tokens,
    max_len,
    cache_block_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale_bytes: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_j = tl.program_id(1)
    if pid_t >= num_tokens:
        return

    context_len = tl.load(context_lens_ptr + pid_t)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    active = (offs_j < max_len) & (offs_j < context_len)

    logical_block = offs_j // cache_block_size
    in_block = offs_j % cache_block_size
    block_ok = active & (logical_block < max_blocks)
    phys = tl.load(
        block_table_ptr + pid_t * block_table_stride + logical_block,
        mask=block_ok,
        other=-1,
    )
    ok = block_ok & (phys >= 0)

    page_base = cache_ptr + phys.to(tl.int64) * cache_stride_page

    offs_d = tl.arange(0, head_dim)
    values = tl.load(
        page_base[:, None] + in_block[:, None] * head_dim + offs_d[None, :],
        mask=ok[:, None],
        other=0,
    )
    tl.store(
        values_ptr + (pid_t * max_len + offs_j)[:, None] * head_dim + offs_d[None, :],
        values,
        mask=ok[:, None],
    )

    # Per-token fp32 scale stored as ``scale_bytes`` raw bytes after the values.
    offs_b = tl.arange(0, scale_bytes)
    scale_raw = tl.load(
        page_base[:, None]
        + cache_block_size * head_dim
        + in_block[:, None] * scale_bytes
        + offs_b[None, :],
        mask=ok[:, None],
        other=0,
    )
    tl.store(
        scales_ptr
        + (pid_t * max_len + offs_j)[:, None] * scale_bytes
        + offs_b[None, :],
        scale_raw,
        mask=ok[:, None],
    )


def deepseek_v4_gather_paged_indexer_fp8_padded(
    cache_2d: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    head_dim: int,
    max_len: int,
    scale_bytes: int = 4,
    values_out: torch.Tensor | None = None,
    scales_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged FP8 indexer keys into a padded ``[tokens * max_len, dim]`` buffer.

    Unlike the torch gather this performs no device-to-host reads, so it is safe
    to call inside a CUDA graph capture. Token ``t`` occupies rows
    ``[t * max_len, t * max_len + context_lens[t])``.

    Args:
        cache_2d: ``[pages, block_size * (head_dim + scale_bytes)]`` uint8 cache.
        context_lens: ``[tokens]`` valid key count per token.
        block_table: ``[tokens, max_blocks]`` logical-to-physical page map.
        block_size: tokens per page.
        head_dim: indexer head dimension.
        max_len: padded per-token width.
        scale_bytes: bytes per stored scale (4 for fp32).
        values_out: optional preallocated uint8 value buffer.
        scales_out: optional preallocated uint8 scale buffer.

    Returns:
        ``(values, scales)`` as ``float8_e4m3fn`` ``[tokens * max_len, head_dim]``
        and ``float32`` ``[tokens * max_len]``.
    """
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    num_tokens = int(context_lens.reshape(-1).shape[0])
    rows = max(1, num_tokens * max_len)
    device = cache_2d.device
    if values_out is None:
        values_out = torch.zeros(rows, head_dim, dtype=torch.uint8, device=device)
    if scales_out is None:
        scales_out = torch.zeros(rows, scale_bytes, dtype=torch.uint8, device=device)
    if num_tokens == 0 or max_len == 0:
        return (
            values_out.view(torch.float8_e4m3fn),
            scales_out.view(torch.float32).reshape(-1),
        )
    block_j = 32
    block_table_i32 = _as_int32_block_table(block_table)
    _deepseek_v4_gather_paged_indexer_fp8_padded_kernel[
        (num_tokens, triton.cdiv(max_len, block_j))
    ](
        cache_2d,
        context_lens.reshape(-1).to(torch.int32),
        block_table_i32,
        values_out,
        scales_out,
        cache_2d.stride(0),
        block_table_i32.stride(0),
        block_table_i32.shape[-1],
        num_tokens,
        max_len,
        cache_block_size=block_size,
        head_dim=head_dim,
        scale_bytes=scale_bytes,
        BLOCK_J=block_j,
        num_warps=4,
    )
    return (
        values_out.view(torch.float8_e4m3fn),
        scales_out.view(torch.float32).reshape(-1),
    )


# Dequantize slot-addressed rows of a paged fp8_ds_mla cache into a contiguous
# bf16 workspace. Decode selects arbitrary slots per token, so it cannot use the
# sequential gather; dequantizing first lets the sparse-attention kernel run
# unchanged over both the sliding-window and compressed caches, exactly as the
# prefill path already does with its own workspace.
@triton.jit(do_not_specialize=["num_tokens", "topk"])
def _deepseek_v4_gather_slots_bf16_kernel(
    cache_ptr,
    slots_ptr,
    lens_ptr,
    out_ptr,
    block_stride,
    slots_stride_token,
    out_stride_token,
    out_stride_row,
    dst_offset,
    num_tokens,
    topk,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    scale_dim: tl.constexpr,
    fp8_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    quant_block: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_t >= num_tokens or pid_k >= topk:
        return

    length = tl.load(lens_ptr + pid_t)
    out_row = out_ptr + pid_t * out_stride_token + (dst_offset + pid_k) * out_stride_row
    if pid_k >= length:
        return

    slot = tl.load(slots_ptr + pid_t * slots_stride_token + pid_k)
    if slot < 0:
        return

    block_idx = slot // cache_block_size
    pos_in_block = slot % cache_block_size
    cache_block = cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data = cache_block + pos_in_block * token_data_size
    token_scales = (
        cache_block + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    for qblock_idx in tl.static_range(n_quant_blocks):
        offsets = qblock_idx * quant_block + tl.arange(0, quant_block)
        mask = offsets < fp8_dim
        raw = tl.load(token_data + offsets, mask=mask, other=0)
        values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        exponent = tl.load(token_scales + qblock_idx).to(tl.float32) - 127.0
        tl.store(
            out_row + offsets,
            (values * tl.exp2(exponent)).to(tl.bfloat16),
            mask=mask,
        )

    rope_offsets = tl.arange(0, rope_dim)
    rope_src = (token_data + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        out_row + fp8_dim + rope_offsets,
        tl.load(rope_src + rope_offsets),
    )


def deepseek_v4_gather_slots_bf16(
    cache: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
    dst_offset: int = 0,
) -> torch.Tensor:
    """Dequantize slot-addressed fp8_ds_mla rows into a bf16 workspace.

    Args:
        cache: uint8 paged cache viewed as ``[num_blocks, block_bytes]``.
        slots: ``[num_tokens, topk]`` slot indices; negative entries are skipped.
        lens: ``[num_tokens]`` valid entry count per row of ``slots``.
        block_size: tokens per cache block.
        out: ``[num_tokens, workspace_width, head_dim]`` bf16 destination.
        dst_offset: column offset into ``out`` for this cache's rows, so that
            several caches can share one workspace.

    Returns:
        ``out``, for chaining.
    """
    if cache.dtype != torch.uint8:
        raise TypeError(f"cache must be uint8, got {cache.dtype}")
    # Callers pass [tokens, topk] or a [tokens, 1, topk]-style view.
    slots = slots.reshape(slots.shape[0], -1)
    num_tokens, topk = slots.shape
    if num_tokens == 0 or topk == 0:
        return out
    _deepseek_v4_gather_slots_bf16_kernel[(num_tokens, topk)](
        cache,
        slots,
        lens,
        out,
        cache.stride(0),
        slots.stride(0),
        out.stride(0),
        out.stride(1),
        dst_offset,
        num_tokens,
        topk,
        cache_block_size=block_size,
        token_data_size=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        scale_dim=DEEPSEEK_V4_SWA_SCALE_DIM,
        fp8_dim=DEEPSEEK_V4_NOPE_DIM,
        rope_dim=DEEPSEEK_V4_ROPE_DIM,
        quant_block=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        n_quant_blocks=DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK,
        num_warps=4,
    )
    return out


# Weighted MQA logits for the V4 sparse indexer. The reference scores each head
# independently, rectifies, then takes the weighted sum over heads::
#
#     score = einsum("bshd,btd->bsht", q, kv).relu_()
#     logits = (score * weights.unsqueeze(-1)).sum(dim=2)
#
# The ReLU is per-head and precedes the reduction, so it cannot be folded into
# the weights. Query scales are already folded into ``weights`` by the FP8
# preparation step; the per-key scale is positive and independent of the head,
# so it factors out of the rectified sum.
@triton.jit(do_not_specialize=["max_len"])
def _deepseek_v4_indexer_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    q_stride_token,
    q_stride_head,
    k_stride_row,
    w_stride_token,
    logits_stride_token,
    max_len,
    num_heads,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_j = tl.program_id(1)

    start = tl.load(cu_start_ptr + pid_t)
    end = tl.load(cu_end_ptr + pid_t)
    num_keys = end - start

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    valid = (offs_j < num_keys) & (offs_j < max_len)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    head_ok = offs_h < num_heads

    q = tl.load(
        q_ptr
        + pid_t * q_stride_token
        + offs_h[:, None] * q_stride_head
        + offs_d[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    k = tl.load(
        k_ptr + (start + offs_j)[:, None].to(tl.int64) * k_stride_row + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )

    scores = tl.dot(q, tl.trans(k)).to(tl.float32)
    scores = tl.maximum(scores, 0.0)

    w = tl.load(
        weights_ptr + pid_t * w_stride_token + offs_h,
        mask=head_ok,
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(scores * w[:, None], axis=0)

    k_scale = tl.load(k_scale_ptr + start + offs_j, mask=valid, other=0.0)
    logits = logits * k_scale

    tl.store(
        logits_ptr + pid_t * logits_stride_token + offs_j,
        logits,
        mask=valid,
    )


def deepseek_v4_indexer_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_start: torch.Tensor,
    cu_end: torch.Tensor,
    max_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rectified, weight-summed MQA logits for the V4 sparse indexer.

    Args:
        q: ``[tokens, heads, index_head_dim]`` FP8 (or bf16) indexer queries,
            with the softmax and head scales already folded into ``weights``.
        k: ``[total_keys, index_head_dim]`` FP8 (or bf16) indexer keys, laid out
            contiguously in request order.
        k_scale: ``[total_keys]`` float32 per-key dequant scales.
        weights: ``[tokens, heads]`` float32 per-head weights.
        cu_start: ``[tokens]`` int32 first key index for each token.
        cu_end: ``[tokens]`` int32 one-past-last key index for each token.
        max_len: width of the logits buffer.
        out: optional ``[tokens, max_len]`` float32 destination.

    Returns:
        ``[tokens, max_len]`` float32 logits; entries beyond a token's key range
        are left untouched.
    """
    num_tokens, num_heads, head_dim = q.shape
    if out is None:
        out = torch.zeros(num_tokens, max_len, dtype=torch.float32, device=q.device)
    if num_tokens == 0 or max_len == 0:
        return out
    if k.shape[-1] != head_dim:
        raise ValueError(f"k head_dim {k.shape[-1]} must match q head_dim {head_dim}")
    if weights.dtype != torch.float32:
        weights = weights.float()
    weights = weights.reshape(num_tokens, -1).contiguous()
    # tl.dot needs at least 16 rows; pad the head axis and mask the extras.
    h_block = max(16, triton.next_power_of_2(num_heads))
    block_j = 64
    grid = (num_tokens, triton.cdiv(max_len, block_j))
    _deepseek_v4_indexer_mqa_logits_kernel[grid](
        q,
        k,
        k_scale,
        weights,
        cu_start,
        cu_end,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        out.stride(0),
        max_len,
        num_heads,
        H=h_block,
        D=head_dim,
        BLOCK_J=block_j,
        num_warps=4,
    )
    return out


# Sparse MQA attention over gathered top-k rows, with DeepSeek V4's learned
# per-head attention sink. Mirrors the reference kernel: q and the latent KV
# share one 512-wide space (KV is both key and value), scores are taken over
# valid gathered rows under an online softmax, and the sink enters the
# denominator once at the end as ``exp(sink - running_max)`` -- it has no
# matching value row, so it never contributes to the numerator.
_DEEPSEEK_V4_ATTN_NEG = -1.0e30


@triton.jit(do_not_specialize=["num_tokens", "topk"])
def _deepseek_v4_sparse_attn_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    q_stride_token,
    q_stride_head,
    kv_stride_row,
    idx_stride_token,
    out_stride_token,
    out_stride_head,
    scale,
    num_tokens,
    topk,
    neg: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + pid * q_stride_token + offs_h[:, None] * q_stride_head + offs_d[None, :]
    )

    length = tl.load(lens_ptr + pid)

    acc_o = tl.zeros((H, D), dtype=tl.float32)
    m_i = tl.full((H,), neg, tl.float32)
    l_i = tl.zeros((H,), dtype=tl.float32)

    for k0 in range(0, topk, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        in_range = (offs_k < topk) & (offs_k < length)
        idx = tl.load(
            indices_ptr + pid * idx_stride_token + offs_k,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (idx >= 0)
        kv = tl.load(
            kv_ptr + idx[:, None].to(tl.int64) * kv_stride_row + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
        scores = tl.where(valid[None, :], scores, neg)

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(kv.dtype), kv).to(tl.float32)
        m_i = m_new

    # The sink is a denominator-only term, applied against the final running max.
    sink = tl.load(sink_ptr + offs_h).to(tl.float32)
    l_i = l_i + tl.exp(sink - m_i)

    out = acc_o / l_i[:, None]
    tl.store(
        out_ptr
        + pid * out_stride_token
        + offs_h[:, None] * out_stride_head
        + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
    )


def deepseek_v4_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse latent attention with a per-head sink.

    Args:
        q: ``[num_tokens, num_heads, head_dim]`` queries; ``num_heads`` must be
            a power of two (pad before calling).
        kv: ``[num_rows, head_dim]`` gathered latent KV, serving as both key and
            value.
        indices: ``[num_tokens, topk]`` int32 row indices into ``kv``; negative
            entries are masked out.
        lens: ``[num_tokens]`` count of valid entries per row of ``indices``.
        attn_sink: ``[num_heads]`` float32 learned sink logits.
        softmax_scale: multiplier applied to the raw scores.
        out: optional ``[num_tokens, num_heads, head_dim]`` destination.

    Returns:
        The attention output, ``out`` when provided.
    """
    num_tokens, num_heads, head_dim = q.shape
    topk = indices.shape[-1]
    if out is None:
        out = torch.empty_like(q)
    if num_tokens == 0:
        return out
    if kv.shape[-1] != head_dim:
        raise ValueError(f"kv head_dim {kv.shape[-1]} must match q head_dim {head_dim}")
    if attn_sink.numel() < num_heads:
        raise ValueError(f"attn_sink has {attn_sink.numel()} entries, need {num_heads}")
    indices_2d = indices.reshape(num_tokens, topk)
    _deepseek_v4_sparse_attn_kernel[(num_tokens,)](
        q,
        kv,
        indices_2d,
        lens,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices_2d.stride(0),
        out.stride(0),
        out.stride(1),
        softmax_scale,
        num_tokens,
        topk,
        neg=_DEEPSEEK_V4_ATTN_NEG,
        H=num_heads,
        D=head_dim,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


# Decode attention that dequantizes the paged cache inside the softmax loop.
#
# ``deepseek_v4_gather_slots_bf16`` followed by ``deepseek_v4_sparse_attn`` is
# correct but pays for a full bf16 workspace in HBM that attention reads back
# immediately. This kernel removes that round trip: it walks the same slot lists
# and dequantizes each row into registers as the online softmax consumes it.
#
# The two caches keep their own slot list, length vector and block size, and are
# walked as two phases sharing one set of accumulators -- which is what makes a
# single softmax span both, without the host-side index arithmetic that packing
# them into a shared workspace required.
@triton.jit
def _deepseek_v4_paged_kv_tile(
    cache_ptr,
    slots_ptr,
    lens_ptr,
    pid_t,
    slots_stride_token,
    block_stride,
    k0,
    topk,
    offs_d,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    scale_dim: tl.constexpr,
    fp8_dim: tl.constexpr,
    quant_block: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_QB: tl.constexpr,
):
    """Load and dequantize ``BLOCK_K`` slot-addressed cache rows.

    Returns ``(kv, valid)`` where ``kv`` is a ``[BLOCK_K, D]`` bfloat16 tile and
    ``valid`` marks the rows backed by a real slot. Rounding matches the bf16
    workspace the gather kernel writes, so results are bit-identical.
    """
    length = tl.load(lens_ptr + pid_t)
    offs_k = k0 + tl.arange(0, BLOCK_K)
    in_range = (offs_k < topk) & (offs_k < length)
    slot = tl.load(
        slots_ptr + pid_t * slots_stride_token + offs_k, mask=in_range, other=-1
    )
    valid = in_range & (slot >= 0)
    slot = tl.where(valid, slot, 0)

    block_idx = slot // cache_block_size
    pos_in_block = slot % cache_block_size
    block_base = block_idx.to(tl.int64) * block_stride
    data_off = block_base + pos_in_block.to(tl.int64) * token_data_size
    scale_off = (
        block_base
        + cache_block_size * token_data_size
        + pos_in_block.to(tl.int64) * scale_dim
    )

    is_nope = offs_d < fp8_dim
    raw = tl.load(
        cache_ptr + data_off[:, None] + offs_d[None, :],
        mask=valid[:, None] & is_nope[None, :],
        other=0,
    )

    # One exponent per quantization block, broadcast across the block's columns.
    # Addressing it per output column instead issues D byte-gathers per row to
    # fetch the handful of exponents the row actually has.
    offs_qb = tl.arange(0, N_QB)
    exponent = tl.load(
        cache_ptr + scale_off[:, None] + offs_qb[None, :],
        mask=valid[:, None] & (offs_qb < fp8_dim // quant_block)[None, :],
        other=127,
    )
    exponent = tl.reshape(
        tl.broadcast_to(exponent[:, :, None], (BLOCK_K, N_QB, quant_block)),
        (BLOCK_K, N_QB * quant_block),
    )
    nope = (
        raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        * tl.exp2(exponent.to(tl.float32) - 127.0)
    ).to(tl.bfloat16)

    # Payload byte offsets and ``fp8_dim`` are both even, so the rope tail is
    # addressable through a bfloat16 view of the cache. It is stored bfloat16,
    # so it is loaded that way -- keeping two [BLOCK_K, D] float32 tiles from
    # being live at once is what lets the kernel hold more than one wave.
    cache_bf16 = cache_ptr.to(tl.pointer_type(tl.bfloat16))
    rope_off = (data_off + fp8_dim) // 2
    rope = tl.load(
        cache_bf16 + rope_off[:, None] + (offs_d - fp8_dim)[None, :],
        mask=valid[:, None] & (offs_d >= fp8_dim)[None, :],
        other=0.0,
    )

    return tl.where(is_nope[None, :], nope, rope), valid


@triton.jit
def _deepseek_v4_online_softmax_step(
    q, kv, valid, scale, m_i, l_i, acc_o, neg: tl.constexpr
):
    """Fold one ``[BLOCK_K, D]`` key/value tile into the running softmax."""
    scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * scale
    scores = tl.where(valid[None, :], scores, neg)

    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)

    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc_o = acc_o * alpha[:, None] + tl.dot(p.to(kv.dtype), kv).to(tl.float32)
    return m_new, l_i, acc_o


@triton.jit(do_not_specialize=["num_tokens", "swa_topk", "extra_topk"])
def _deepseek_v4_fused_paged_sparse_attn_kernel(
    q_ptr,
    acc_ptr,
    m_ptr,
    l_ptr,
    swa_cache_ptr,
    swa_slots_ptr,
    swa_lens_ptr,
    swa_block_stride,
    swa_slots_stride_token,
    extra_cache_ptr,
    extra_slots_ptr,
    extra_lens_ptr,
    extra_block_stride,
    extra_slots_stride_token,
    q_stride_token,
    q_stride_head,
    scale,
    num_tokens,
    swa_topk,
    extra_topk,
    split_width,
    neg: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    scale_dim: tl.constexpr,
    fp8_dim: tl.constexpr,
    quant_block: tl.constexpr,
    N_QB: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    if pid >= num_tokens:
        return

    # Splits carve the concatenated slot range so each covers one contiguous
    # run; a split that straddles the cache boundary walks the tail of one and
    # the head of the other.
    lo = split_id * split_width
    hi = lo + split_width
    # The last split's range runs past the end of the concatenated slot list,
    # so both ends are clamped to their own cache rather than leaving the
    # overhang to be masked off by the lengths.
    swa_lo = tl.minimum(lo, swa_topk)
    swa_hi = tl.minimum(hi, swa_topk)
    extra_lo = tl.minimum(tl.maximum(lo - swa_topk, 0), extra_topk)
    extra_hi = tl.minimum(tl.maximum(hi - swa_topk, 0), extra_topk)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + pid * q_stride_token + offs_h[:, None] * q_stride_head + offs_d[None, :]
    )

    acc_o = tl.zeros((H, D), dtype=tl.float32)
    m_i = tl.full((H,), neg, tl.float32)
    l_i = tl.zeros((H,), dtype=tl.float32)

    for k0 in range(swa_lo, swa_hi, BLOCK_K):
        kv, valid = _deepseek_v4_paged_kv_tile(
            swa_cache_ptr,
            swa_slots_ptr,
            swa_lens_ptr,
            pid,
            swa_slots_stride_token,
            swa_block_stride,
            k0,
            swa_hi,
            offs_d,
            cache_block_size=swa_block_size,
            token_data_size=token_data_size,
            scale_dim=scale_dim,
            fp8_dim=fp8_dim,
            quant_block=quant_block,
            BLOCK_K=BLOCK_K,
            N_QB=N_QB,
        )
        m_i, l_i, acc_o = _deepseek_v4_online_softmax_step(
            q, kv, valid, scale, m_i, l_i, acc_o, neg=neg
        )

    if HAS_EXTRA:
        for k0 in range(extra_lo, extra_hi, BLOCK_K):
            kv, valid = _deepseek_v4_paged_kv_tile(
                extra_cache_ptr,
                extra_slots_ptr,
                extra_lens_ptr,
                pid,
                extra_slots_stride_token,
                extra_block_stride,
                k0,
                extra_hi,
                offs_d,
                cache_block_size=extra_block_size,
                token_data_size=token_data_size,
                scale_dim=scale_dim,
                fp8_dim=fp8_dim,
                quant_block=quant_block,
                BLOCK_K=BLOCK_K,
                N_QB=N_QB,
            )
            m_i, l_i, acc_o = _deepseek_v4_online_softmax_step(
                q, kv, valid, scale, m_i, l_i, acc_o, neg=neg
            )

    base = (pid * NUM_SPLITS + split_id) * H
    tl.store(m_ptr + base + offs_h, m_i)
    tl.store(l_ptr + base + offs_h, l_i)
    tl.store(acc_ptr + base * D + offs_h[:, None] * D + offs_d[None, :], acc_o)


# Merge the per-split partials. Each split ran its own online softmax over a
# disjoint slot range, so rescaling every partial against the global running
# max and summing gives the same result as one sequential pass. The sink joins
# the denominator once, against that global max.
@triton.jit(do_not_specialize=["num_tokens"])
def _deepseek_v4_fused_paged_sparse_attn_combine_kernel(
    acc_ptr,
    m_ptr,
    l_ptr,
    sink_ptr,
    out_ptr,
    out_stride_token,
    out_stride_head,
    num_tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, NUM_SPLITS)

    sm = (pid * NUM_SPLITS + offs_s)[:, None] * H + offs_h[None, :]
    m_s = tl.load(m_ptr + sm)
    l_s = tl.load(l_ptr + sm)

    m_g = tl.max(m_s, axis=0)
    alpha = tl.exp(m_s - m_g[None, :])
    l_g = tl.sum(l_s * alpha, axis=0)

    acc = tl.zeros((H, D), dtype=tl.float32)
    for split_id in tl.static_range(0, NUM_SPLITS):
        part = tl.load(
            acc_ptr
            + ((pid * NUM_SPLITS + split_id) * H + offs_h)[:, None] * D
            + offs_d[None, :]
        )
        scale_s = tl.exp(
            tl.load(m_ptr + (pid * NUM_SPLITS + split_id) * H + offs_h) - m_g
        )
        acc += part * scale_s[:, None]

    sink = tl.load(sink_ptr + offs_h).to(tl.float32)
    l_g = l_g + tl.exp(sink - m_g)

    out = acc / l_g[:, None]
    tl.store(
        out_ptr
        + pid * out_stride_token
        + offs_h[:, None] * out_stride_head
        + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
    )


def deepseek_v4_fused_paged_sparse_attn(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_block_size: int,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extra_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_block_size: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse latent attention read straight from the paged fp8_ds_mla caches.

    Equivalent to dequantizing both caches into one bf16 workspace with
    ``deepseek_v4_gather_slots_bf16`` and calling ``deepseek_v4_sparse_attn``
    over it, but without materializing the workspace.

    Args:
        q: ``[num_tokens, num_heads, head_dim]`` queries; ``num_heads`` must be
            a power of two (pad before calling).
        swa_cache: uint8 sliding-window cache viewed as
            ``[num_blocks, block_bytes]``.
        swa_slots: ``[num_tokens, swa_topk]`` slot indices; negatives are
            skipped.
        swa_lens: ``[num_tokens]`` valid entry count per row of ``swa_slots``.
        swa_block_size: tokens per sliding-window cache block.
        attn_sink: ``[num_heads]`` float32 learned sink logits, entering the
            denominator once against the final running max.
        softmax_scale: multiplier applied to the raw scores.
        extra_cache: optional uint8 compressed cache in the same layout.
        extra_slots: ``[num_tokens, extra_topk]`` slots into ``extra_cache``.
        extra_lens: ``[num_tokens]`` valid entry count for ``extra_slots``.
        extra_block_size: tokens per compressed cache block.
        out: optional ``[num_tokens, num_heads, head_dim]`` destination.

    Returns:
        The attention output, ``out`` when provided.
    """
    num_tokens, num_heads, head_dim = q.shape
    if out is None:
        out = torch.empty_like(q)
    if num_tokens == 0:
        return out
    if swa_cache.dtype != torch.uint8:
        raise TypeError(f"swa_cache must be uint8, got {swa_cache.dtype}")
    if head_dim != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} must be {DEEPSEEK_V4_HEAD_DIM} for the "
            "fp8_ds_mla cache layout"
        )
    if attn_sink.numel() < num_heads:
        raise ValueError(f"attn_sink has {attn_sink.numel()} entries, need {num_heads}")
    if head_dim % DEEPSEEK_V4_FP8_QUANT_BLOCK != 0:
        raise ValueError(
            f"head_dim {head_dim} must be a multiple of the quantization block "
            f"{DEEPSEEK_V4_FP8_QUANT_BLOCK}: the dequantized tile broadcasts one "
            "exponent per block across the block's columns"
        )
    if head_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK > DEEPSEEK_V4_SWA_SCALE_DIM:
        raise ValueError(
            "scale vector is too short to cover the head dimension: "
            f"{DEEPSEEK_V4_SWA_SCALE_DIM} entries for "
            f"{head_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK} blocks"
        )

    swa_slots = swa_slots.reshape(num_tokens, -1)
    swa_topk = swa_slots.shape[1]

    has_extra = (
        extra_cache is not None and extra_slots is not None and extra_lens is not None
    )
    if has_extra:
        if extra_cache.dtype != torch.uint8:
            raise TypeError(f"extra_cache must be uint8, got {extra_cache.dtype}")
        extra_slots = extra_slots.reshape(num_tokens, -1)
        extra_topk = extra_slots.shape[1]
        has_extra = extra_topk > 0
    if not has_extra:
        # Triton still needs well-formed arguments for the disabled branch.
        extra_cache = swa_cache
        extra_slots = swa_slots
        extra_lens = swa_lens
        extra_topk = 0
        extra_block_size = swa_block_size

    block_k = _deepseek_v4_fused_attn_block_k(swa_topk, extra_topk)
    num_splits, split_width = _deepseek_v4_fused_attn_split(
        num_tokens, swa_topk + extra_topk, block_k, q.device
    )

    partial_acc = torch.empty(
        (num_tokens, num_splits, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    partial_m = torch.empty(
        (num_tokens, num_splits, num_heads), dtype=torch.float32, device=q.device
    )
    partial_l = torch.empty_like(partial_m)

    _deepseek_v4_fused_paged_sparse_attn_kernel[(num_tokens, num_splits)](
        q,
        partial_acc,
        partial_m,
        partial_l,
        swa_cache,
        swa_slots,
        swa_lens,
        swa_cache.stride(0),
        swa_slots.stride(0),
        extra_cache,
        extra_slots,
        extra_lens,
        extra_cache.stride(0),
        extra_slots.stride(0),
        q.stride(0),
        q.stride(1),
        softmax_scale,
        num_tokens,
        swa_topk,
        extra_topk,
        split_width,
        neg=_DEEPSEEK_V4_ATTN_NEG,
        H=num_heads,
        D=head_dim,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        swa_block_size=swa_block_size,
        extra_block_size=extra_block_size,
        token_data_size=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        scale_dim=DEEPSEEK_V4_SWA_SCALE_DIM,
        fp8_dim=DEEPSEEK_V4_NOPE_DIM,
        quant_block=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        N_QB=head_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK,
        HAS_EXTRA=has_extra,
        num_warps=4,
        # Software pipelining double-buffers the [BLOCK_K, head_dim] tiles, and
        # at these widths the second buffer costs more in occupancy than the
        # overlap returns.
        num_stages=1,
    )
    _deepseek_v4_fused_paged_sparse_attn_combine_kernel[(num_tokens,)](
        partial_acc,
        partial_m,
        partial_l,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        num_tokens,
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=4,
    )
    return out


@functools.lru_cache(maxsize=None)
def _deepseek_v4_attn_core_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _deepseek_v4_fused_attn_split(
    num_tokens: int, total_topk: int, block_k: int, device: torch.device
) -> tuple[int, int]:
    """Choose a K-split count that keeps the machine busy at decode widths.

    One program per token leaves all but ``num_tokens`` compute units idle,
    which is the common case at low concurrency. Splitting the slot range
    trades a small combine pass for parallelism, so the split count only needs
    to grow until the grid covers the device.

    Returns ``(num_splits, split_width)`` with ``split_width`` a multiple of
    ``block_k`` so no split starts mid-tile.
    """
    tiles = max(1, triton.cdiv(total_topk, block_k))
    cores = _deepseek_v4_attn_core_count(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    wanted = min(tiles, max(1, triton.cdiv(cores, max(1, num_tokens))))
    # The combine kernel reduces the splits as one tile, so keep the count a
    # power of two rather than rounding up past the tiles that exist.
    num_splits = 1 << (wanted.bit_length() - 1)
    return num_splits, triton.cdiv(tiles, num_splits) * block_k


def _deepseek_v4_fused_attn_block_k(swa_topk: int, extra_topk: int) -> int:
    """Pick a K tile that does not overshoot short slot lists.

    Both phases share the tile width, so it is capped by the shorter list to
    keep masked-off lanes from dominating the sliding-window phase, whose length
    is the model's sliding window rather than the indexer top-k.
    """
    widths = [w for w in (swa_topk, extra_topk) if w > 0]
    smallest = min(widths) if widths else 1
    block_k = 16
    while block_k < 64 and block_k * 2 <= smallest:
        block_k *= 2
    return block_k


# Forward RoPE + block-scaled FP8 quant + paged scatter of the KV latent into
# the sliding-window cache, matching the reference::
#
#     apply_rotary_emb(kv[..., -rope_dim:], freqs_cis)
#     act_quant(kv[..., :-rope_dim], 64, ...)   # rope dims stay bf16
#
# The written layout is the one _deepseek_v4_dequantize_and_gather_k_kernel
# reads back: within a block the token payloads come first as
# ``cache_block_size * token_data_size`` bytes, then the per-token scale
# vectors. Each payload is ``fp8_dim`` bytes of e4m3 followed by ``rope_dim``
# bf16 values; each scale is a ``uint8`` e8m0 exponent biased by 127.
@triton.jit(do_not_specialize=["num_tokens"])
def _deepseek_v4_kv_rope_quant_insert_kernel(
    kv_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    k_cache_ptr,
    kv_stride_token,
    cache_stride_pos,
    block_stride,
    num_tokens,
    eps,
    fp8_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    slot = tl.load(slot_mapping_ptr + pid)
    # Masked or out-of-capacity tokens arrive as -1 and must not be written.
    if slot < 0:
        return

    block_idx = slot // cache_block_size
    pos_in_block = slot % cache_block_size
    cache_block = k_cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data = cache_block + pos_in_block * token_data_size
    token_scales = (
        cache_block + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    kv_base = kv_ptr + pid * kv_stride_token

    # --- rope section: rotate, keep bf16 ---
    rope_offsets = tl.arange(0, rope_dim)
    r = tl.load(kv_base + fp8_dim + rope_offsets).to(tl.float32)
    partner = tl.load(kv_base + fp8_dim + (rope_offsets ^ 1)).to(tl.float32)

    pos = tl.load(positions_ptr + pid)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    cs_idx = rope_offsets >> 1
    cos_v = tl.load(cache_base + cs_idx)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx)
    is_even = (rope_offsets & 1) == 0
    rotated = tl.where(
        is_even, r * cos_v - partner * sin_v, r * cos_v + partner * sin_v
    )

    bf16_cache = (token_data + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    tl.store(bf16_cache + rope_offsets, rotated.to(tl.bfloat16))

    # --- nope section: per-block e8m0 scale + e4m3 payload ---
    for qblock_idx in tl.static_range(n_quant_blocks):
        offsets = qblock_idx * quant_block + tl.arange(0, quant_block)
        mask = offsets < fp8_dim
        x = tl.load(kv_base + offsets, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x), axis=0), eps)
        # exp2/log2 round-trip keeps the scale a power of two, which is what an
        # e8m0 exponent can represent exactly.
        exponent = tl.ceil(tl.log2(absmax / fp8_max))
        exponent = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
        scale = tl.exp2(exponent - 127.0)
        q = tl.clamp(x / scale, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(token_data + offsets, q.to(tl.uint8, bitcast=True), mask=mask)
        tl.store(token_scales + qblock_idx, exponent.to(tl.uint8))


def deepseek_v4_kv_rope_quant_insert(
    kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    block_size: int,
) -> None:
    """RoPE, quantize, and scatter the KV latent into the paged SWA cache.

    Args:
        kv: ``[num_tokens, head_dim]`` KV latent, already RMS-normalized. The
            leading ``head_dim - rope_dim`` values are quantized to e4m3; the
            trailing ``rope_dim`` are rotated and stored bf16.
        slot_mapping: ``[num_tokens]`` destination slots; negative entries are
            skipped.
        positions: ``[num_tokens]`` absolute token positions for RoPE.
        cos_sin_cache: ``[max_position, rope_dim]`` float32, cos then sin.
        k_cache: uint8 cache viewed as ``[num_blocks, block_bytes]``.
        block_size: tokens per cache block.
    """
    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be uint8, got {k_cache.dtype}")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError(f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}")
    num_tokens = kv.shape[0]
    if num_tokens == 0:
        return
    if kv.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(
            f"kv last dim must be {DEEPSEEK_V4_HEAD_DIM}, got {kv.shape[-1]}"
        )
    _deepseek_v4_kv_rope_quant_insert_kernel[(num_tokens,)](
        kv,
        slot_mapping.to(torch.int64),
        positions.to(torch.int64),
        cos_sin_cache,
        k_cache,
        kv.stride(0),
        cos_sin_cache.stride(0),
        k_cache.stride(0),
        num_tokens,
        1e-10,
        fp8_dim=DEEPSEEK_V4_NOPE_DIM,
        rope_dim=DEEPSEEK_V4_ROPE_DIM,
        scale_dim=DEEPSEEK_V4_SWA_SCALE_DIM,
        quant_block=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        cache_block_size=block_size,
        token_data_size=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        fp8_max=DEEPSEEK_V4_FP8_MAX,
        n_quant_blocks=DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK,
        HALF_ROPE=DEEPSEEK_V4_ROPE_DIM // 2,
        num_warps=4,
    )


# Fused inverse-RoPE + block-scaled FP8 quant for the V4 attention output
# projection. Output scale is pre-transformed (MN-major TMA-aligned;
# INT32-packed UE8M0 on SM100, FP32 on SM90) so deep_gemm.fp8_einsum can
# consume it without re-transforming.
@triton.jit(do_not_specialize=["num_tokens"])
def _deepseek_v4_fused_inv_rope_fp8_quant_per_head(
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


def deepseek_v4_fused_inv_rope_fp8_quant(
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

    Returns (o_fp8 [T, G, D] float8_e4m3fn, o_scale [T, G, scale_inner])
    pre-laid-out for ``deep_gemm.fp8_einsum("bhr,hdr->bhd")``.
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
    _deepseek_v4_fused_inv_rope_fp8_quant_per_head[grid](
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
