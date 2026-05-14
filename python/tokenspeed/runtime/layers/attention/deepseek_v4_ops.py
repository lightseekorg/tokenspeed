# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#
# DeepSeek V4 compressor/indexer/attention helpers keep the boundary unfused
# and framework-local so the correctness contract can be tested before the
# optimized fused kernel lands.

"""DeepSeek V4 attention kernel boundaries.

Keep the model layer independent from the CUDA extension import details. The
runtime requires TokenSpeed's own built DeepSeek V4 attention op.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

QNORM_ROPE_KV_INSERT_OP = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
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


class DeepseekV4AttentionOpUnavailable(RuntimeError):
    pass


def deepseek_v4_profile_scope(name: str):
    """V4 NVTX scope — wraps main's nvtx_range so callsites stay V4-specific.

    Off by default; enabled via `--enable-nvtx` or `TOKENSPEED_NVTX=1`.
    """
    from tokenspeed.runtime.utils.nvtx import nvtx_range

    return nvtx_range(f"v4:{name}")


def _deepseek_v4_fused_compressor_cache_enabled(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _deepseek_v4_fused_indexer_mxfp4_enabled(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _get_tokenspeed_op() -> object | None:
    try:
        from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
            fused_qnorm_rope_kv_insert as op,
        )
    except Exception:
        return None
    return op


def _has_tokenspeed_op() -> bool:
    try:
        from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
            has_fused_qnorm_rope_kv_insert as has_op,
        )
    except Exception:
        return False
    return has_op()


def has_fused_qnorm_rope_kv_insert() -> bool:
    return _has_tokenspeed_op()


def _require_op():
    tokenspeed_op = _get_tokenspeed_op()
    if tokenspeed_op is not None and _has_tokenspeed_op():
        return tokenspeed_op
    raise DeepseekV4AttentionOpUnavailable(
        f"DeepSeek V4 fused SWA cache insert op {QNORM_ROPE_KV_INSERT_OP} "
        "is unavailable. Build `tokenspeed-kernel/python` so the "
        "deepseek_v4_attention CUDA library is present before running this path."
    )


def fused_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    block_size: int,
) -> None:
    """Run the DeepSeek V4 fused SWA cache insert op.

    Expected contract:
    - q: [tokens, local_heads, 512], mutated in place by RMSNorm/RoPE
    - kv: [tokens, 512], source KV latent before RoPE/quant insert
    - swa_kv_cache_2d: uint8 cache blocks flattened as [num_blocks, block_bytes]
    - slot_mapping: output token slots in the paged SWA cache
    - positions: absolute token positions
    """

    op = _require_op()
    op(
        q,
        kv,
        swa_kv_cache_2d,
        slot_mapping,
        positions.to(torch.int64),
        cos_sin_cache,
        rms_norm_eps,
        block_size,
    )


def _apply_gptj_rope_tail(
    x: torch.Tensor,
    position: int,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    half_rope = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cos_sin = cos_sin_cache[position]
    cos = cos_sin[:half_rope].float()
    sin = cos_sin[half_rope:rope_dim].float()
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    out[..., nope_dim::2] = even * cos - odd * sin
    out[..., nope_dim + 1 :: 2] = even * sin + odd * cos
    return out


def _apply_gptj_rope_tail_rows(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    half_rope = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cos = cos_sin_cache[positions.long(), :half_rope].float()
    sin = cos_sin_cache[positions.long(), half_rope:rope_dim].float()
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos - odd * sin
    out[..., nope_dim + 1 :: 2] = even * sin + odd * cos
    return out


def _apply_inverse_gptj_rope_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    half_rope = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cos = cos_sin_cache[positions.long(), :half_rope].float()
    sin = cos_sin_cache[positions.long(), half_rope:rope_dim].float()
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos + odd * sin
    out[..., nope_dim + 1 :: 2] = odd * cos - even * sin
    return out


def _encode_ue8m0_exponent(exponent: float) -> int:
    return int(max(min(exponent + 127, 255), 0))


def _fp8_e4m3_ue8m0_bytes(block: torch.Tensor) -> tuple[torch.Tensor, int]:
    absmax = max(float(block.detach().abs().max()), 1.0e-4)
    exponent = math.ceil(math.log2(absmax / DEEPSEEK_V4_FP8_MAX))
    scaled = torch.clamp(
        block * (2.0**-exponent),
        -DEEPSEEK_V4_FP8_MAX,
        DEEPSEEK_V4_FP8_MAX,
    )
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), _encode_ue8m0_exponent(
        exponent
    )


def _fp8_e4m3_pow2_bytes(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(float(block.detach().abs().max()) / DEEPSEEK_V4_FP8_MAX, 1.0e-10)
    scale = 2.0 ** math.ceil(math.log2(scale))
    scaled = torch.clamp(block / scale, -DEEPSEEK_V4_FP8_MAX, DEEPSEEK_V4_FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), block.new_tensor(scale)


def _fp8_e4m3_pow2_dequant_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_f = rows.float()
    scale = (rows_f.detach().abs().amax(dim=-1) / DEEPSEEK_V4_FP8_MAX).clamp_min(
        1.0e-10
    )
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    scaled = torch.clamp(
        rows_f / scale.unsqueeze(-1),
        -DEEPSEEK_V4_FP8_MAX,
        DEEPSEEK_V4_FP8_MAX,
    )
    dequant = scaled.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return dequant, scale


def _e2m1_nibbles(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.clamp(x.abs(), max=6.0)
    code = torch.zeros_like(abs_x, dtype=torch.uint8)
    for idx, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        code = torch.where(abs_x > boundary, idx + 1, code)
    sign = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (sign << 3)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


def _mxfp4_e2m1_ue8m0_bytes(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if block.numel() != DEEPSEEK_V4_MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 block must have {DEEPSEEK_V4_MXFP4_BLOCK_SIZE} values, "
            f"got {block.numel()}"
        )
    even = block.float()[0::2]
    odd = block.float()[1::2]
    absmax = max(
        float(torch.maximum(even.detach().abs().max(), odd.detach().abs().max())),
        1.0e-4,
    )
    exponent = min(max(math.ceil(math.log2(absmax / 6.0)), -127), 127)
    inv_scale = 2.0**-exponent
    lo = _e2m1_nibbles(even * inv_scale)
    hi = _e2m1_nibbles(odd * inv_scale)
    return lo | (hi << 4), block.new_tensor(exponent + 127, dtype=torch.uint8)


def _mxfp4_e2m1_ue8m0_dequant_rows(rows: torch.Tensor) -> torch.Tensor:
    orig_shape = rows.shape
    if orig_shape[-1] % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 rows require last dim divisible by {DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, "
            f"got {orig_shape[-1]}"
        )
    blocks = rows.float().reshape(
        -1, orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE, DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    )
    absmax = blocks.detach().abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / 6.0)).clamp(-127, 127)
    scale = torch.pow(2.0, exponent)
    nibbles = _e2m1_nibbles(blocks / scale.unsqueeze(-1))
    dequant = _e2m1_values(nibbles) * scale.unsqueeze(-1)
    return dequant.reshape(orig_shape)


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
def _deepseek_v4_mxfp4_quantize_rows_kernel(
    rows_ptr,
    row_stride,
    packed_ptr,
    packed_stride,
    scale_ptr,
    scale_stride,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
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

    packed_base = packed_ptr + row_idx * packed_stride + block_base // 2
    tl.store(packed_base + offsets, packed)
    tl.store(scale_ptr + row_idx * scale_stride + block_idx, scale)


def _deepseek_v4_mxfp4_quantize_rows_cuda(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = rows.shape
    if orig_shape[-1] % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 rows require last dim divisible by {DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, "
            f"got {orig_shape[-1]}"
        )
    rows_2d = rows.reshape(-1, orig_shape[-1]).contiguous()
    packed = torch.empty(
        *orig_shape[:-1],
        orig_shape[-1] // 2,
        device=rows.device,
        dtype=torch.uint8,
    )
    scale_bytes = torch.empty(
        *orig_shape[:-1],
        orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        device=rows.device,
        dtype=torch.uint8,
    )
    if rows_2d.shape[0] == 0:
        return packed, scale_bytes
    packed_2d = packed.reshape(-1, orig_shape[-1] // 2)
    scale_2d = scale_bytes.reshape(-1, orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE)
    _deepseek_v4_mxfp4_quantize_rows_kernel[
        (rows_2d.shape[0], orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE)
    ](
        rows_2d,
        rows_2d.stride(0),
        packed_2d,
        packed_2d.stride(0),
        scale_2d,
        scale_2d.stride(0),
        HEAD_DIM=orig_shape[-1],
        QUANT_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        HALF_BLOCK=DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2,
        num_warps=1,
    )
    return packed.contiguous(), scale_bytes.contiguous()


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


def _deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4(
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


def _mxfp4_e2m1_ue8m0_quantize_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = rows.shape
    if orig_shape[-1] % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 rows require last dim divisible by {DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, "
            f"got {orig_shape[-1]}"
        )
    if _deepseek_v4_fused_indexer_mxfp4_enabled(rows):
        return _deepseek_v4_mxfp4_quantize_rows_cuda(rows)
    blocks = rows.float().reshape(
        -1,
        orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    )
    absmax = blocks.detach().abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / 6.0)).clamp(-127, 127)
    inv_scale = torch.pow(2.0, -exponent)
    nibbles = _e2m1_nibbles(blocks * inv_scale.unsqueeze(-1))
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).reshape(
        *orig_shape[:-1],
        orig_shape[-1] // 2,
    )
    scale_bytes = (
        (exponent.to(torch.int32) + 127)
        .to(torch.uint8)
        .reshape(
            *orig_shape[:-1],
            orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
        )
    )
    return packed.contiguous(), scale_bytes.contiguous()


def _deepseek_v4_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    try:
        from tokenspeed_kernel.thirdparty.fast_hadamard_transform import (
            hadamard_transform,
        )
    except Exception as exc:
        raise DeepseekV4AttentionOpUnavailable(
            "DeepSeek V4 CSA indexer requires fast_hadamard_transform. "
            "Build/install `tokenspeed-kernel/python` before serving V4."
        ) from exc

    shape = x.shape
    rotated = hadamard_transform(
        x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    )
    return rotated.reshape(shape)


def deepseek_v4_inv_rope_reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = DEEPSEEK_V4_NOPE_DIM,
    rope_dim: int = DEEPSEEK_V4_ROPE_DIM,
) -> torch.Tensor:
    """Inverse-RoPE and group V4 attention output without FP8 activation rounding."""

    if o.dim() != 3:
        raise ValueError(f"o must be [tokens, heads, dim], got {tuple(o.shape)}")
    if o.shape[1] != n_groups * heads_per_group:
        raise ValueError(
            f"heads={o.shape[1]} does not match n_groups={n_groups} "
            f"* heads_per_group={heads_per_group}"
        )
    if o.shape[2] != nope_dim + rope_dim:
        raise ValueError(f"head dim must be {nope_dim + rope_dim}, got {o.shape[2]}")

    inv = _apply_inverse_gptj_rope_tail(o, positions, cos_sin_cache, rope_dim)
    grouped = inv.reshape(o.shape[0], n_groups, heads_per_group * o.shape[2])
    return grouped.to(o.dtype)


def dequantize_deepseek_v4_fp8_ds_mla_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize DeepSeek V4 `fp8_ds_mla` rows selected by global slots."""

    min_stride = block_size * (DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), DEEPSEEK_V4_HEAD_DIM)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.bfloat16)

    flat_cache = cache_2d.reshape(-1)
    num_nope_blocks = DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK

    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_SWA_TOKEN_STRIDE
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        + pos * DEEPSEEK_V4_SWA_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_SWA_TOKEN_STRIDE, device=cache_2d.device, dtype=torch.int64
        )[None, :]
    )
    row_bytes = flat_cache[value_offsets]
    nope = row_bytes[:, :DEEPSEEK_V4_NOPE_DIM].contiguous().view(torch.float8_e4m3fn)

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(num_nope_blocks, device=cache_2d.device, dtype=torch.int64)[
            None, :
        ]
    )
    scales = torch.pow(2.0, flat_cache[scale_offsets].to(torch.int32) - 127)
    scales = scales.float().repeat_interleave(DEEPSEEK_V4_FP8_QUANT_BLOCK, dim=1)

    rope = row_bytes[:, DEEPSEEK_V4_NOPE_DIM:DEEPSEEK_V4_SWA_TOKEN_STRIDE].contiguous()
    out = torch.cat([nope.float() * scales, rope.view(torch.bfloat16).float()], dim=1)
    out = out.to(torch.bfloat16)
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def deepseek_v4_prepare_indexer_q_reference(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    use_fp4: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply indexer Q RoPE and quant/dequant with folded weight rules."""

    if index_q.dim() != 3 or index_q.shape[-1] != DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_q must be [tokens, heads, {DEEPSEEK_V4_INDEXER_DIM}], "
            f"got {tuple(index_q.shape)}"
        )
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")

    rotated = index_q.float().clone()
    half_rope = DEEPSEEK_V4_ROPE_DIM // 2
    nope_dim = index_q.shape[-1] - DEEPSEEK_V4_ROPE_DIM
    cos = cos_sin_cache[positions.long(), :half_rope].float().unsqueeze(1)
    sin = (
        cos_sin_cache[positions.long(), half_rope:DEEPSEEK_V4_ROPE_DIM]
        .float()
        .unsqueeze(1)
    )
    even = rotated[..., nope_dim::2].clone()
    odd = rotated[..., nope_dim + 1 :: 2].clone()
    rotated[..., nope_dim::2] = even * cos - odd * sin
    rotated[..., nope_dim + 1 :: 2] = even * sin + odd * cos

    rotated = _deepseek_v4_hadamard_rotate(rotated).float()
    weights_out = weights.float().clone()
    if use_fp4:
        q_out = _mxfp4_e2m1_ue8m0_dequant_rows(rotated)
        weights_out *= softmax_scale * head_scale
    else:
        q_out, q_scale = _fp8_e4m3_pow2_dequant_rows(rotated)
        weights_out *= q_scale * softmax_scale * head_scale
    return q_out.to(index_q.dtype), weights_out


def deepseek_v4_prepare_indexer_q_mxfp4(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Apply indexer Q RoPE and return DeepGEMM-ready MXFP4 values/scales."""

    if index_q.dim() != 3 or index_q.shape[-1] != DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_q must be [tokens, heads, {DEEPSEEK_V4_INDEXER_DIM}], "
            f"got {tuple(index_q.shape)}"
        )
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")
    if _deepseek_v4_fused_indexer_mxfp4_enabled(index_q):
        return _deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4(
            index_q=index_q,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=weights,
            softmax_scale=softmax_scale,
            head_scale=head_scale,
        )

    rotated = index_q.float().clone()
    half_rope = DEEPSEEK_V4_ROPE_DIM // 2
    nope_dim = index_q.shape[-1] - DEEPSEEK_V4_ROPE_DIM
    cos = cos_sin_cache[positions.long(), :half_rope].float().unsqueeze(1)
    sin = (
        cos_sin_cache[positions.long(), half_rope:DEEPSEEK_V4_ROPE_DIM]
        .float()
        .unsqueeze(1)
    )
    even = rotated[..., nope_dim::2].clone()
    odd = rotated[..., nope_dim + 1 :: 2].clone()
    rotated[..., nope_dim::2] = even * cos - odd * sin
    rotated[..., nope_dim + 1 :: 2] = even * sin + odd * cos
    rotated = _deepseek_v4_hadamard_rotate(rotated).float()

    q_values, q_scale_bytes = _mxfp4_e2m1_ue8m0_quantize_rows(rotated)
    q_scales = q_scale_bytes.view(torch.int32).squeeze(-1).contiguous()
    weights_out = weights.float().clone()
    weights_out *= softmax_scale * head_scale
    return (q_values, q_scales), weights_out


def _write_fp8_ds_mla_cache_row(
    normed: torch.Tensor,
    position: int,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot: int,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    quant_input = normed.to(torch.bfloat16).float()
    kv_block_idx = kv_slot // kv_cache_block_size
    kv_pos_in_block = kv_slot % kv_cache_block_size
    token_base = (
        kv_block_idx * kv_cache_2d.stride(0)
        + kv_pos_in_block * DEEPSEEK_V4_SWA_TOKEN_STRIDE
    )
    scale_base = (
        kv_block_idx * kv_cache_2d.stride(0)
        + kv_cache_block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        + kv_pos_in_block * DEEPSEEK_V4_SWA_SCALE_DIM
    )
    flat_cache = kv_cache_2d.reshape(-1)
    num_nope_blocks = DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK
    for block_id in range(num_nope_blocks):
        lo = block_id * DEEPSEEK_V4_FP8_QUANT_BLOCK
        hi = lo + DEEPSEEK_V4_FP8_QUANT_BLOCK
        fp8_bytes, encoded_scale = _fp8_e4m3_ue8m0_bytes(quant_input[lo:hi])
        flat_cache[token_base + lo : token_base + hi].copy_(fp8_bytes)
        flat_cache[scale_base + block_id] = encoded_scale
    flat_cache[scale_base + num_nope_blocks] = 0

    compressed_position = (position // compress_ratio) * compress_ratio
    rotated = _apply_gptj_rope_tail(
        normed, compressed_position, cos_sin_cache, DEEPSEEK_V4_ROPE_DIM
    ).to(torch.bfloat16)
    flat_cache[
        token_base + DEEPSEEK_V4_NOPE_DIM : token_base + DEEPSEEK_V4_SWA_TOKEN_STRIDE
    ].copy_(rotated[DEEPSEEK_V4_NOPE_DIM:].view(torch.uint8))


@triton.jit
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
    block_table_width: tl.constexpr,
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


def _deepseek_v4_fused_sparse_compress_cache_insert(
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
    _deepseek_v4_fused_sparse_compress_cache_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices[:num_actual],
        positions[:num_actual],
        compressor_slot_mapping[:num_actual],
        block_table,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table.stride(0),
        block_table.shape[-1],
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
        num_warps=4,
    )


@triton.jit
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
    block_table_width: tl.constexpr,
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


def _deepseek_v4_fused_csa_indexer_mxfp4_cache_insert(
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
    _deepseek_v4_fused_csa_indexer_mxfp4_cache_kernel[
        (num_actual, DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM)
    ](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices[:num_actual],
        positions[:num_actual],
        compressor_slot_mapping[:num_actual],
        block_table,
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table.stride(0),
        block_table.shape[-1],
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


def _fp8_ds_mla_cache_rows(
    normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows = normed.shape[0]
    quant_input = normed.to(torch.bfloat16).float()
    nope_blocks = quant_input[:, :DEEPSEEK_V4_NOPE_DIM].reshape(
        num_rows,
        DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK,
        DEEPSEEK_V4_FP8_QUANT_BLOCK,
    )
    absmax = nope_blocks.detach().abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / DEEPSEEK_V4_FP8_MAX))
    scaled = torch.clamp(
        nope_blocks * torch.pow(2.0, -exponent).unsqueeze(-1),
        -DEEPSEEK_V4_FP8_MAX,
        DEEPSEEK_V4_FP8_MAX,
    )
    value_bytes = (
        scaled.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(
            num_rows,
            DEEPSEEK_V4_NOPE_DIM,
        )
    )
    scale_bytes = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.uint8)
    scale_bytes = torch.cat([scale_bytes, torch.zeros_like(scale_bytes[:, :1])], dim=-1)

    compressed_positions = (
        torch.div(positions.to(torch.int64), compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    rotated = _apply_gptj_rope_tail_rows(
        normed,
        compressed_positions,
        cos_sin_cache,
        DEEPSEEK_V4_ROPE_DIM,
    ).to(torch.bfloat16)
    rope_bytes = rotated[:, DEEPSEEK_V4_NOPE_DIM:].contiguous().view(torch.uint8)
    rope_bytes = rope_bytes.reshape(num_rows, DEEPSEEK_V4_ROPE_DIM * 2)
    return value_bytes, scale_bytes, rope_bytes


def _write_fp8_ds_mla_cache_rows_capturable(
    normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    num_rows = normed.shape[0]
    if num_rows == 0:
        return

    slots = kv_slot_mapping[:num_rows].to(torch.int64)
    valid = valid[:num_rows] & (slots >= 0)
    if not (slots.is_cuda and torch.cuda.is_current_stream_capturing()):
        if not bool(valid.any()):
            return
        normed = normed[:num_rows][valid]
        positions = positions[:num_rows][valid]
        slots = slots[valid]
        valid = torch.ones_like(slots, dtype=torch.bool)
        num_rows = slots.numel()
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_idx = torch.div(safe_slots, kv_cache_block_size, rounding_mode="floor")
    pos_in_block = safe_slots % kv_cache_block_size
    block_base = block_idx * kv_cache_2d.stride(0)
    token_base = block_base + pos_in_block * DEEPSEEK_V4_SWA_TOKEN_STRIDE
    scale_base = (
        block_base
        + kv_cache_block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        + pos_in_block * DEEPSEEK_V4_SWA_SCALE_DIM
    )

    value_bytes, scale_bytes, rope_bytes = _fp8_ds_mla_cache_rows(
        normed[:num_rows], positions[:num_rows], cos_sin_cache, compress_ratio
    )
    flat_cache = kv_cache_2d.reshape(-1)
    value_offsets = (
        token_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_NOPE_DIM,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_SWA_SCALE_DIM,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    rope_offsets = (
        token_base[:, None]
        + DEEPSEEK_V4_NOPE_DIM
        + torch.arange(
            DEEPSEEK_V4_ROPE_DIM * 2,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    flat_cache[value_offsets] = value_bytes
    flat_cache[scale_offsets] = scale_bytes
    flat_cache[rope_offsets] = rope_bytes


def save_deepseek_v4_compressor_state(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> None:
    """Save DeepSeek V4 compressor residual state into paged SWA-style cache.

    This correctness-first state write packs `[kv_state, score_state]`, each
    with width `coff * head_dim`; score state includes the APE row selected by
    `position % compress_ratio`.
    """

    if kv.shape != score.shape:
        raise ValueError(
            f"kv and score shapes must match, got {kv.shape} vs {score.shape}"
        )
    if kv.dim() != 2:
        raise ValueError(f"kv/score must be [tokens, state_width], got {kv.shape}")
    if state_cache.dim() != 3:
        raise ValueError(
            "state_cache must be [blocks, block_size, 2 * state_width], "
            f"got {state_cache.shape}"
        )
    if block_size != state_cache.shape[1]:
        raise ValueError(
            f"block_size={block_size} does not match "
            f"state_cache.shape[1]={state_cache.shape[1]}"
        )
    state_width = kv.shape[-1]
    if state_cache.shape[-1] != state_width * 2:
        raise ValueError(
            f"state_cache last dim must be {state_width * 2}, "
            f"got {state_cache.shape[-1]}"
        )
    if ape.shape != (compress_ratio, state_width):
        raise ValueError(
            f"ape must be [{compress_ratio}, {state_width}], got {tuple(ape.shape)}"
        )

    num_actual = min(slot_mapping.numel(), kv.shape[0])
    if num_actual == 0:
        return

    if _deepseek_v4_fused_compressor_cache_enabled(state_cache):
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
        return

    slots = slot_mapping[:num_actual].to(torch.int64)
    capturing = slots.is_cuda and torch.cuda.is_current_stream_capturing()
    if capturing:
        # CUDA graph replay pads inactive decode slots to the pool's reserved
        # dummy slot. Avoid boolean indexing here because it is not capturable.
        valid_slots = slots.clamp_min(0)
        kv_rows = kv[:num_actual].float()
        score_rows = score[:num_actual].float()
        valid_positions = positions[:num_actual].to(torch.int64)
    else:
        valid = slots >= 0
        valid_slots = slots[valid]
        kv_rows = kv[:num_actual][valid].float()
        score_rows = score[:num_actual][valid].float()
        valid_positions = positions[:num_actual][valid].to(torch.int64)

    block_idx = torch.div(valid_slots, block_size, rounding_mode="floor")
    pos_in_block = valid_slots % block_size
    state_cache[block_idx, pos_in_block, :state_width] = kv_rows

    if compress_ratio == 4 and state_width == ape.shape[1] and state_width % 2 == 0:
        head_dim = state_width // 2
        ape_slots = ape.view(-1, head_dim).float()
        slot_idx = valid_positions % compress_ratio
        scored = score_rows.clone()
        scored[:, :head_dim] += ape_slots[slot_idx]
        scored[:, head_dim:] += ape_slots[slot_idx + compress_ratio]
    else:
        scored = score_rows + ape[valid_positions % compress_ratio].float()
    state_cache[block_idx, pos_in_block, state_width:] = scored


def write_deepseek_v4_indexer_fp8_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write FP8 indexer keys using `[values | fp32 scale]` page layout."""

    if index_k.dim() != 2 or index_k.shape[-1] != 128:
        raise ValueError(f"index_k must be [tokens, 128], got {tuple(index_k.shape)}")
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < block_size * 132:
        raise ValueError(
            f"cache_2d must be [pages, >= {block_size * 132}], "
            f"got {tuple(cache_2d.shape)}"
        )

    flat_cache = cache_2d.reshape(-1)
    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    for token_idx in range(num_actual):
        slot = int(slot_mapping[token_idx].item())
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        page_base = page * cache_2d.stride(0)
        value_base = page_base + pos * 128
        scale_base = page_base + block_size * 128 + pos * 4
        q_bytes, scale = _fp8_e4m3_pow2_bytes(index_k[token_idx].float())
        flat_cache[value_base : value_base + 128].copy_(q_bytes)
        flat_cache[scale_base : scale_base + 4].copy_(
            scale.reshape(1).view(torch.uint8)
        )


def write_deepseek_v4_indexer_mxfp4_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write MXFP4 indexer keys using the `[values | ue8m0 scales]` layout."""

    if index_k.dim() != 2 or index_k.shape[-1] != DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_k must be [tokens, {DEEPSEEK_V4_INDEXER_DIM}], "
            f"got {tuple(index_k.shape)}"
        )
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    min_stride = block_size * (
        DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES + DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    if num_actual == 0:
        return

    if _deepseek_v4_fused_indexer_mxfp4_enabled(index_k):
        valid = torch.ones(num_actual, device=index_k.device, dtype=torch.bool)
        _write_deepseek_v4_indexer_mxfp4_cache_cuda(
            index_k[:num_actual],
            cache_2d,
            slot_mapping[:num_actual],
            valid,
            block_size,
        )
        return

    flat_cache = cache_2d.reshape(-1)
    slots = slot_mapping[:num_actual].to(torch.int64)
    valid = slots >= 0
    if not bool(valid.any()):
        return

    valid_slots = slots[valid]
    valid_rows = index_k[:num_actual][valid].float()
    packed, scale_bytes = _mxfp4_e2m1_ue8m0_quantize_rows(valid_rows)

    pages = torch.div(valid_slots, block_size, rounding_mode="floor")
    pos = valid_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        + pos * DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    flat_cache[value_offsets] = packed
    flat_cache[scale_offsets] = scale_bytes


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


def _write_deepseek_v4_indexer_mxfp4_cache_cuda(
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


def _write_deepseek_v4_indexer_mxfp4_cache_capturable(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    block_size: int = 64,
) -> None:
    num_rows = min(slot_mapping.numel(), index_k.shape[0])
    if num_rows == 0:
        return

    slots = slot_mapping[:num_rows].to(torch.int64)
    valid = valid[:num_rows] & (slots >= 0)
    if _deepseek_v4_fused_indexer_mxfp4_enabled(index_k):
        _write_deepseek_v4_indexer_mxfp4_cache_cuda(
            index_k[:num_rows],
            cache_2d,
            slots,
            valid,
            block_size,
        )
        return
    if not (slots.is_cuda and torch.cuda.is_current_stream_capturing()):
        if not bool(valid.any()):
            return
        index_k = index_k[:num_rows][valid]
        slots = slots[valid]
        valid = torch.ones_like(slots, dtype=torch.bool)
        num_rows = slots.numel()
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    packed, scale_bytes = _mxfp4_e2m1_ue8m0_quantize_rows(index_k[:num_rows].float())

    flat_cache = cache_2d.reshape(-1)
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        + pos * DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    flat_cache[value_offsets] = packed
    flat_cache[scale_offsets] = scale_bytes


def _write_deepseek_v4_indexer_fp8_cache_capturable(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    block_size: int = 64,
) -> None:
    num_rows = min(slot_mapping.numel(), index_k.shape[0])
    if num_rows == 0:
        return

    rows = index_k[:num_rows].float()
    scale = (rows.detach().abs().amax(dim=-1) / DEEPSEEK_V4_FP8_MAX).clamp_min(1.0e-10)
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    value_bytes = (
        torch.clamp(
            rows / scale.unsqueeze(-1),
            -DEEPSEEK_V4_FP8_MAX,
            DEEPSEEK_V4_FP8_MAX,
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )

    slots = slot_mapping[:num_rows].to(torch.int64)
    valid = valid[:num_rows] & (slots >= 0)
    if not (slots.is_cuda and torch.cuda.is_current_stream_capturing()):
        if not bool(valid.any()):
            return
        rows = rows[valid]
        slots = slots[valid]
        scale = scale[valid]
        value_bytes = value_bytes[valid]
        valid = torch.ones_like(slots, dtype=torch.bool)
        num_rows = slots.numel()
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_INDEXER_DIM
    scale_base = page_base + block_size * DEEPSEEK_V4_INDEXER_DIM + pos * 4

    flat_cache = cache_2d.reshape(-1)
    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_DIM,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(4, device=cache_2d.device, dtype=torch.int64)[None, :]
    )
    flat_cache[value_offsets] = value_bytes
    flat_cache[scale_offsets] = scale.view(torch.uint8).reshape(num_rows, 4)


def read_deepseek_v4_indexer_mxfp4_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize MXFP4 indexer cache rows selected by `slot_mapping`."""

    min_stride = block_size * (
        DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES + DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), DEEPSEEK_V4_INDEXER_DIM)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)

    flat_cache = cache_2d.reshape(-1)
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        + pos * DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    packed = flat_cache[value_offsets]

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scales = torch.pow(2.0, flat_cache[scale_offsets].to(torch.int32) - 127)
    byte_scales = scales.float().repeat_interleave(
        DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2, dim=1
    )

    even = _e2m1_values(packed & 0xF) * byte_scales
    odd = _e2m1_values(packed >> 4) * byte_scales
    out = torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)
    out[:, 0::2] = even
    out[:, 1::2] = odd
    return torch.where(valid[:, None], out, torch.zeros_like(out))


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


def read_deepseek_v4_indexer_fp8_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize FP8 indexer cache rows selected by `slot_mapping`."""

    min_stride = block_size * (DEEPSEEK_V4_INDEXER_DIM + 4)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out = torch.zeros(
        slot_mapping.numel(),
        DEEPSEEK_V4_INDEXER_DIM,
        device=cache_2d.device,
        dtype=torch.float32,
    )
    flat_cache = cache_2d.reshape(-1)
    for token_idx, raw_slot in enumerate(slot_mapping.tolist()):
        slot = int(raw_slot)
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        page_base = page * cache_2d.stride(0)
        value_base = page_base + pos * DEEPSEEK_V4_INDEXER_DIM
        scale_base = page_base + block_size * DEEPSEEK_V4_INDEXER_DIM + pos * 4
        scale = flat_cache[scale_base : scale_base + 4].view(torch.float32)[0]
        values = flat_cache[value_base : value_base + DEEPSEEK_V4_INDEXER_DIM].view(
            torch.float8_e4m3fn
        )
        out[token_idx].copy_(values.float() * scale)
    return out


@triton.jit
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
    max_blocks_per_seq: tl.constexpr,
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

        block_table_row = block_table_ptr + batch_idx * max_blocks_per_seq
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
) -> None:
    """Gather/dequantize fp8_ds_mla cache rows for sparse prefill."""

    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be bfloat16, got {out.dtype}")
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if seq_lens.numel() == 0:
        return

    _deepseek_v4_dequantize_and_gather_k_kernel[(seq_lens.numel(), 128)](
        out,
        out.stride(0),
        out.stride(1),
        cache_2d,
        seq_lens.to(torch.int32),
        block_table.to(torch.int32),
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        offset,
        gather_lens.to(torch.int32) if gather_lens is not None else None,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=DEEPSEEK_V4_NOPE_DIM,
        bf16_dim=DEEPSEEK_V4_ROPE_DIM,
        scale_dim=DEEPSEEK_V4_SWA_SCALE_DIM,
        quant_block=DEEPSEEK_V4_FP8_QUANT_BLOCK,
        cache_block_size=block_size,
        token_data_size=DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        block_stride=cache_2d.stride(0),
        fp8_max=DEEPSEEK_V4_FP8_MAX,
        n_quant_blocks=DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK,
    )


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


@triton.jit
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
    max_blocks_per_seq: tl.constexpr,
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
    _deepseek_v4_decode_swa_indices_and_lens_kernel[(num_tokens,)](
        out_indices,
        out_indices.stride(0),
        out_lens,
        query_start_loc.to(torch.int32),
        seq_lens.to(torch.int32),
        token_to_req_indices.to(torch.int32),
        is_valid_token,
        block_table.to(torch.int32),
        (
            block_table_base_offsets.to(torch.int32)
            if block_table_base_offsets is not None
            else None
        ),
        block_table.stride(0),
        block_table.shape[-1],
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


def deepseek_v4_indexer_topk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference weighted ReLU MQA top-k for the CSA sparse indexer."""

    if q.dim() != 3 or k.dim() != 2:
        raise ValueError(
            f"expected q [tokens, heads, dim], k [kv, dim], got {q.shape}, {k.shape}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k dims must match, got {q.shape[-1]} and {k.shape[-1]}")
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {weights.shape}")

    logits = torch.einsum("thd,kd->thk", q.float(), k.float()).relu()
    logits = (logits * weights.float().unsqueeze(-1)).sum(dim=1)
    if lengths is not None:
        if row_starts is None:
            row_starts = torch.zeros_like(lengths)
        cols = torch.arange(k.shape[0], device=k.device)
        valid = (cols.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
            cols.unsqueeze(0) < (row_starts + lengths).unsqueeze(1)
        )
        logits = logits.masked_fill(~valid, -float("inf"))
    return torch.topk(logits, k=top_k, dim=-1, sorted=False).indices.to(torch.int32)


def _compress_v4_state_window(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    token_idx: int,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
) -> torch.Tensor | None:
    state_slot = int(compressor_slot_mapping[token_idx].item())
    if state_slot < 0:
        return None
    position = int(positions[token_idx].item())
    if (position + 1) % compress_ratio != 0:
        return None

    state_width = state_cache.shape[-1] // 2
    window = (2 if overlap else 1) * compress_ratio
    req_idx = int(token_to_req_indices[token_idx].item())
    base_logical_page = (
        int(block_table_base_offsets[req_idx].item())
        if block_table_base_offsets is not None
        else 0
    )
    start = position - window + 1
    kv_rows = []
    score_rows = []
    for offset in range(window):
        pos = start + offset
        if pos < 0:
            continue
        table_idx = pos // compressor_block_size - base_logical_page
        if table_idx < 0:
            continue
        if table_idx >= block_table.shape[1]:
            continue
        block_number = int(block_table[req_idx, table_idx].item())
        if block_number < 0:
            continue
        head_offset = head_dim if overlap and offset >= compress_ratio else 0
        row = state_cache[block_number, pos % compressor_block_size]
        kv_rows.append(row[head_offset : head_offset + head_dim].float())
        score_rows.append(
            row[
                state_width + head_offset : state_width + head_offset + head_dim
            ].float()
        )
    if not kv_rows:
        return None

    kv_stack = torch.stack(kv_rows, dim=0)
    score_stack = torch.stack(score_rows, dim=0)
    weights = torch.softmax(score_stack, dim=0)
    compressed = torch.sum(kv_stack * weights, dim=0)
    variance = compressed.square().sum() / float(head_dim)
    normed = compressed * torch.rsqrt(variance + rms_norm_eps)
    return normed * rms_norm_weight.float()


def _compress_v4_state_windows_capturable(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual == 0:
        return (
            torch.empty((0, head_dim), device=state_cache.device, dtype=torch.float32),
            torch.empty((0,), device=state_cache.device, dtype=torch.bool),
        )

    token_positions = positions[:num_actual].to(torch.int64)
    state_slots = compressor_slot_mapping[:num_actual].to(torch.int64)
    valid_token = (state_slots >= 0) & (
        torch.remainder(token_positions + 1, compress_ratio) == 0
    )

    window = (2 if overlap else 1) * compress_ratio
    offsets = torch.arange(window, device=state_cache.device, dtype=torch.int64)
    window_positions = token_positions[:, None] - window + 1 + offsets[None, :]
    table_idx_raw = torch.div(
        window_positions, compressor_block_size, rounding_mode="floor"
    )
    req_idx = token_to_req_indices[:num_actual].to(torch.int64).clamp_min(0)
    if block_table_base_offsets is not None:
        safe_req_for_base = req_idx.clamp(
            0, max(int(block_table_base_offsets.shape[0]) - 1, 0)
        )
        base_logical_page = block_table_base_offsets.to(
            device=state_cache.device,
            dtype=torch.int64,
        )[safe_req_for_base]
        table_idx_raw = table_idx_raw - base_logical_page[:, None]
    valid_window = (
        (window_positions >= 0)
        & (table_idx_raw >= 0)
        & (table_idx_raw < block_table.shape[1])
    )
    table_idx = table_idx_raw.clamp(0, max(block_table.shape[1] - 1, 0))
    block_number = block_table[req_idx[:, None], table_idx]
    valid_window = valid_window & (block_number >= 0)

    safe_block = block_number.to(torch.int64).clamp_min(0)
    pos_in_block = torch.remainder(window_positions.clamp_min(0), compressor_block_size)
    rows = state_cache[safe_block, pos_in_block]
    state_width = state_cache.shape[-1] // 2

    if overlap:
        head_offsets = torch.where(
            offsets >= compress_ratio,
            torch.full_like(offsets, head_dim),
            torch.zeros_like(offsets),
        )
    else:
        head_offsets = torch.zeros_like(offsets)
    dim_indices = (
        head_offsets[:, None]
        + torch.arange(head_dim, device=state_cache.device, dtype=torch.int64)[None, :]
    )
    dim_indices = dim_indices[None, :, :].expand(num_actual, -1, -1)

    kv_rows = torch.gather(rows[..., :state_width], -1, dim_indices).float()
    score_rows = torch.gather(rows[..., state_width:], -1, dim_indices).float()
    valid_window_f = valid_window.unsqueeze(-1)
    score_rows = torch.where(
        valid_window_f, score_rows, score_rows.new_full((), -1.0e30)
    )
    weights = torch.softmax(score_rows, dim=1)
    kv_rows = torch.where(valid_window_f, kv_rows, torch.zeros_like(kv_rows))
    compressed = torch.sum(kv_rows * weights, dim=1)
    variance = compressed.square().sum(dim=-1, keepdim=True) / float(head_dim)
    normed = compressed * torch.rsqrt(variance + rms_norm_eps)
    return normed * rms_norm_weight.float(), valid_token


def deepseek_v4_hca_compress_kv_cache_insert(
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
    compress_ratio: int = 128,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress HCA state, normalize/RoPE/FP8-quantize, and insert KV cache.

    The HCA path writes one compressed cache entry only at positions where
    `(position + 1) % 128 == 0`. This path is deliberately unfused so
    TokenSpeed has an auditable correctness boundary before the optimized
    kernel lands.
    """

    if compress_ratio != 128:
        raise ValueError(
            f"HCA cache insert requires compress_ratio=128, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    if state_width != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(
            f"HCA state width must be {DEEPSEEK_V4_HEAD_DIM}, got {state_width}"
        )
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    min_block_stride = kv_cache_block_size * (
        DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(
        compressor_slot_mapping.numel(),
        positions.numel(),
        kv_slot_mapping.numel(),
    )
    if num_actual > 0 and _deepseek_v4_fused_compressor_cache_enabled(state_cache):
        _deepseek_v4_fused_sparse_compress_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            overlap=False,
            block_table_base_offsets=block_table_base_offsets,
        )
        return
    if (
        num_actual > 0
        and compressor_slot_mapping.is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        normed, valid = _compress_v4_state_windows_capturable(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=False,
        )
        _write_fp8_ds_mla_cache_rows_capturable(
            normed=normed,
            positions=positions[:num_actual],
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping[:num_actual],
            valid=valid,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
        )
        return
    if num_actual > 0 and compressor_slot_mapping.is_cuda:
        token_positions = positions[:num_actual].to(torch.int64)
        state_slots = compressor_slot_mapping[:num_actual].to(torch.int64)
        kv_slots = kv_slot_mapping[:num_actual].to(torch.int64)
        boundary = (
            (state_slots >= 0)
            & (kv_slots >= 0)
            & (torch.remainder(token_positions + 1, compress_ratio) == 0)
        )
        normed, valid = _compress_v4_state_windows_capturable(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices[:num_actual][boundary],
            positions=token_positions[boundary],
            compressor_slot_mapping=state_slots[boundary],
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=False,
        )
        _write_fp8_ds_mla_cache_rows_capturable(
            normed=normed,
            positions=token_positions[boundary],
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slots[boundary],
            valid=valid,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
        )
        return

    for token_idx in range(num_actual):
        state_slot = int(compressor_slot_mapping[token_idx].item())
        if state_slot < 0:
            continue
        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue

        normed = _compress_v4_state_window(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_idx=token_idx,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=False,
        )
        if normed is None:
            continue

        _write_fp8_ds_mla_cache_row(
            normed,
            position,
            cos_sin_cache,
            kv_cache_2d,
            kv_slot,
            kv_cache_block_size,
            compress_ratio,
        )


def deepseek_v4_csa_compress_kv_cache_insert(
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
    compress_ratio: int = 4,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress CSA state and insert one `fp8_ds_mla` row per 4 tokens.

    CSA uses overlap: the compression window spans eight token positions and
    selects the first 512-wide slice from the older four positions and the
    second slice from the newer four positions before the softmax-weighted sum.
    """

    if compress_ratio != 4:
        raise ValueError(
            f"CSA cache insert requires compress_ratio=4, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    expected_width = DEEPSEEK_V4_HEAD_DIM * 2
    if state_width != expected_width:
        raise ValueError(f"CSA state width must be {expected_width}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    min_block_stride = kv_cache_block_size * (
        DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual > 0 and _deepseek_v4_fused_compressor_cache_enabled(state_cache):
        _deepseek_v4_fused_sparse_compress_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            overlap=True,
            block_table_base_offsets=block_table_base_offsets,
        )
        return
    if num_actual > 0 and compressor_slot_mapping.is_cuda:
        normed, valid = _compress_v4_state_windows_capturable(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=True,
        )
        _write_fp8_ds_mla_cache_rows_capturable(
            normed=normed,
            positions=positions[:num_actual],
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping[:num_actual],
            valid=valid,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
        )
        return

    for token_idx in range(num_actual):
        state_slot = int(compressor_slot_mapping[token_idx].item())
        if state_slot < 0:
            continue
        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue

        normed = _compress_v4_state_window(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_idx=token_idx,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=True,
        )
        if normed is None:
            continue
        _write_fp8_ds_mla_cache_row(
            normed,
            position,
            cos_sin_cache,
            kv_cache_2d,
            kv_slot,
            kv_cache_block_size,
            compress_ratio,
        )


def deepseek_v4_csa_indexer_cache_insert(
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
    use_fp4_cache: bool,
    compress_ratio: int = 4,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress CSA indexer state and insert FP8/MXFP4 indexer cache rows."""

    if compress_ratio != 4:
        raise ValueError(
            f"CSA indexer cache insert requires compress_ratio=4, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    expected_width = DEEPSEEK_V4_INDEXER_DIM * 2
    if state_width != expected_width:
        raise ValueError(
            f"CSA indexer state width must be {expected_width}, got {state_width}"
        )

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if (
        num_actual > 0
        and use_fp4_cache
        and _deepseek_v4_fused_indexer_mxfp4_enabled(state_cache)
    ):
        _deepseek_v4_fused_csa_indexer_mxfp4_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            block_table_base_offsets=block_table_base_offsets,
        )
        return
    if num_actual > 0 and compressor_slot_mapping.is_cuda:
        normed, valid = _compress_v4_state_windows_capturable(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_INDEXER_DIM,
            overlap=True,
        )
        compressed_positions = (
            torch.div(
                positions[:num_actual].to(torch.int64),
                compress_ratio,
                rounding_mode="floor",
            )
            * compress_ratio
        )
        rotated = _apply_gptj_rope_tail_rows(
            normed,
            compressed_positions,
            cos_sin_cache,
            DEEPSEEK_V4_ROPE_DIM,
        )
        rotated = _deepseek_v4_hadamard_rotate(rotated).float()
        if use_fp4_cache:
            _write_deepseek_v4_indexer_mxfp4_cache_capturable(
                rotated,
                kv_cache_2d,
                kv_slot_mapping[:num_actual],
                valid,
                block_size=kv_cache_block_size,
            )
        else:
            _write_deepseek_v4_indexer_fp8_cache_capturable(
                rotated,
                kv_cache_2d,
                kv_slot_mapping[:num_actual],
                valid,
                block_size=kv_cache_block_size,
            )
        return

    for token_idx in range(num_actual):
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue
        position = int(positions[token_idx].item())
        normed = _compress_v4_state_window(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_idx=token_idx,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_INDEXER_DIM,
            overlap=True,
        )
        if normed is None:
            continue
        compressed_position = (position // compress_ratio) * compress_ratio
        rotated = _apply_gptj_rope_tail(
            normed,
            compressed_position,
            cos_sin_cache,
            DEEPSEEK_V4_ROPE_DIM,
        )
        rotated = _deepseek_v4_hadamard_rotate(rotated).float()
        writer = (
            write_deepseek_v4_indexer_mxfp4_cache
            if use_fp4_cache
            else write_deepseek_v4_indexer_fp8_cache
        )
        writer(
            rotated.unsqueeze(0),
            kv_cache_2d,
            kv_slot_mapping[token_idx : token_idx + 1],
            block_size=kv_cache_block_size,
        )
