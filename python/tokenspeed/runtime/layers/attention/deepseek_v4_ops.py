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
# DeepSeek V4 attention helpers keep runtime validation here; production Triton
# kernels live under tokenspeed-kernel ops.

"""DeepSeek V4 attention kernel boundaries.

Keep the model layer independent from the CUDA extension import details. The
runtime requires TokenSpeed's own built DeepSeek V4 attention op.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    fused_qnorm_rope_kv_insert as _cuda_fused_qnorm_rope_kv_insert,
)
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    has_fused_qnorm_rope_kv_insert as _cuda_has_fused_qnorm_rope_kv_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_build_dense_prefill_local_compressed_indices,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compressed_slot_mapping,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_csa_indexer_mxfp4_cache_insert as _triton_fused_csa_indexer_mxfp4_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_hca_direct_compress_cache_insert as _triton_fused_hca_direct_compress_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4 as _triton_fused_indexer_q_rope_hadamard_mxfp4,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_inv_rope_fp8_quant,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_sparse_compress_cache_insert as _triton_fused_sparse_compress_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_save_compressor_state as _triton_save_compressor_state,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    write_deepseek_v4_indexer_mxfp4_cache_cuda as _triton_write_indexer_mxfp4_cache_cuda,
)
from tokenspeed_kernel.ops.transform import hadamard_transform

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    DEEPSEEK_V4_FP8_MAX,
    DEEPSEEK_V4_FP8_QUANT_BLOCK,
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    deepseek_v4_indexer_fp8_layout_from_row_bytes,
    deepseek_v4_indexer_fp8_row_bytes,
    deepseek_v4_indexer_fp8_scale_bytes,
    deepseek_v4_indexer_mxfp4_layout_from_row_bytes,
    deepseek_v4_indexer_mxfp4_row_bytes,
    deepseek_v4_nope_dim,
    deepseek_v4_swa_row_bytes,
    deepseek_v4_swa_scale_dim,
    deepseek_v4_swa_token_stride,
)

__all__ = (
    "deepseek_v4_build_dense_prefill_local_compressed_indices",
    "deepseek_v4_combine_dense_swa_indices",
    "deepseek_v4_combine_topk_swa_indices",
    "deepseek_v4_compressed_slot_mapping",
    "deepseek_v4_compute_global_topk_indices_and_lens",
    "deepseek_v4_decode_swa_indices_and_lens",
    "deepseek_v4_dequantize_and_gather_k_cache",
    "deepseek_v4_fused_inv_rope_fp8_quant",
    "deepseek_v4_csa_compress_kv_cache_insert",
    "deepseek_v4_csa_indexer_cache_insert",
    "deepseek_v4_hca_compress_kv_cache_insert",
    "deepseek_v4_hca_direct_compress_kv_cache_insert",
    "deepseek_v4_prepare_indexer_q_mxfp4",
    "dequantize_deepseek_v4_fp8_ds_mla_cache",
    "fused_qnorm_rope_kv_insert",
    "read_deepseek_v4_indexer_fp8_cache",
    "read_deepseek_v4_indexer_mxfp4_cache",
    "save_deepseek_v4_compressor_state",
    "write_deepseek_v4_indexer_fp8_cache",
    "write_deepseek_v4_indexer_mxfp4_cache",
)


def _indexer_mxfp4_layout_from_cache(
    cache_2d: torch.Tensor,
    block_size: int,
) -> tuple[int, int, int]:
    if cache_2d.dim() != 2:
        raise ValueError(f"cache_2d must be 2-D, got {tuple(cache_2d.shape)}")
    row_bytes = cache_2d.shape[1] // block_size
    if cache_2d.shape[1] % block_size != 0:
        raise ValueError(
            "MXFP4 indexer cache row size must match value+scale layout, "
            f"got cache shape {tuple(cache_2d.shape)} and block_size={block_size}"
        )
    return deepseek_v4_indexer_mxfp4_layout_from_row_bytes(row_bytes)


def _indexer_fp8_layout_from_cache(
    cache_2d: torch.Tensor,
    block_size: int,
) -> tuple[int, int]:
    if cache_2d.dim() != 2:
        raise ValueError(f"cache_2d must be 2-D, got {tuple(cache_2d.shape)}")
    row_bytes = cache_2d.shape[1] // block_size
    if cache_2d.shape[1] % block_size != 0:
        raise ValueError(
            "FP8 indexer cache row size must match value+scale layout, "
            f"got cache shape {tuple(cache_2d.shape)} and block_size={block_size}"
        )
    return deepseek_v4_indexer_fp8_layout_from_row_bytes(row_bytes)


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

    if not _cuda_has_fused_qnorm_rope_kv_insert():
        raise RuntimeError(
            "DeepSeek V4 fused SWA cache insert op "
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert is unavailable. "
            "Build `tokenspeed-kernel/python` so the deepseek_v4_attention CUDA "
            "library is present before running this path."
        )

    _cuda_fused_qnorm_rope_kv_insert(
        q,
        kv,
        swa_kv_cache_2d,
        slot_mapping,
        positions.to(torch.int64),
        cos_sin_cache,
        rms_norm_eps,
        block_size,
    )


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


def _fp8_e4m3_pow2_bytes(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(float(block.detach().abs().max()) / DEEPSEEK_V4_FP8_MAX, 1.0e-10)
    scale = 2.0 ** math.ceil(math.log2(scale))
    scaled = torch.clamp(block / scale, -DEEPSEEK_V4_FP8_MAX, DEEPSEEK_V4_FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), block.new_tensor(scale)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


def _deepseek_v4_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    rotated = hadamard_transform(
        x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    )
    return rotated.reshape(shape)


def dequantize_deepseek_v4_fp8_ds_mla_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
    *,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Dequantize DeepSeek V4 `fp8_ds_mla` rows selected by global slots."""

    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    token_stride = deepseek_v4_swa_token_stride(head_dim, rope_dim)
    scale_dim = deepseek_v4_swa_scale_dim(head_dim, rope_dim)
    min_stride = block_size * (token_stride + scale_dim)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), head_dim)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.bfloat16)

    flat_cache = cache_2d.reshape(-1)
    num_nope_blocks = nope_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK

    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * token_stride
    scale_base = page_base + block_size * token_stride + pos * scale_dim

    value_offsets = (
        value_base[:, None]
        + torch.arange(token_stride, device=cache_2d.device, dtype=torch.int64)[None, :]
    )
    row_bytes = flat_cache[value_offsets]
    nope = row_bytes[:, :nope_dim].contiguous().view(torch.float8_e4m3fn)

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(num_nope_blocks, device=cache_2d.device, dtype=torch.int64)[
            None, :
        ]
    )
    scales = torch.pow(2.0, flat_cache[scale_offsets].to(torch.int32) - 127)
    scales = scales.float().repeat_interleave(DEEPSEEK_V4_FP8_QUANT_BLOCK, dim=1)

    rope = row_bytes[:, nope_dim:token_stride].contiguous()
    out = torch.cat([nope.float() * scales, rope.view(torch.bfloat16).float()], dim=1)
    out = out.to(torch.bfloat16)
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def deepseek_v4_prepare_indexer_q_mxfp4(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Apply indexer Q RoPE and return DeepGEMM-ready MXFP4 values/scales."""

    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [tokens, heads, dim], got {index_q.shape}")
    if index_q.shape[-1] % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "MXFP4 index_q dim must be divisible by "
            f"{DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, got {index_q.shape[-1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    if index_q.shape[-1] <= rope_dim:
        raise ValueError(
            f"index_q dim must be larger than rope_dim={rope_dim}, got {index_q.shape}"
        )
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")
    if not index_q.is_cuda:
        raise ValueError(
            "deepseek_v4_prepare_indexer_q_mxfp4 only supports CUDA tensors."
        )
    return _triton_fused_indexer_q_rope_hadamard_mxfp4(
        index_q=index_q,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
    )


def _fp8_ds_mla_cache_rows(
    normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows = normed.shape[0]
    head_dim = int(normed.shape[-1])
    rope_dim = int(cos_sin_cache.shape[-1])
    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    scale_dim = deepseek_v4_swa_scale_dim(head_dim, rope_dim)
    quant_input = normed.to(torch.bfloat16).float()
    nope_blocks = quant_input[:, :nope_dim].reshape(
        num_rows,
        nope_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK,
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
            nope_dim,
        )
    )
    scale_bytes = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.uint8)
    scale_pad = scale_dim - scale_bytes.shape[1]
    if scale_pad > 0:
        scale_bytes = torch.cat(
            [scale_bytes, torch.zeros_like(scale_bytes[:, :scale_pad])],
            dim=-1,
        )

    compressed_positions = (
        torch.div(positions.to(torch.int64), compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    rotated = _apply_gptj_rope_tail_rows(
        normed,
        compressed_positions,
        cos_sin_cache,
        rope_dim,
    ).to(torch.bfloat16)
    rope_bytes = rotated[:, nope_dim:].contiguous().view(torch.uint8)
    rope_bytes = rope_bytes.reshape(num_rows, rope_dim * 2)
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

    head_dim = int(normed.shape[-1])
    rope_dim = int(cos_sin_cache.shape[-1])
    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    token_stride = deepseek_v4_swa_token_stride(head_dim, rope_dim)
    scale_dim = deepseek_v4_swa_scale_dim(head_dim, rope_dim)
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
    token_base = block_base + pos_in_block * token_stride
    scale_base = (
        block_base + kv_cache_block_size * token_stride + pos_in_block * scale_dim
    )

    value_bytes, scale_bytes, rope_bytes = _fp8_ds_mla_cache_rows(
        normed[:num_rows], positions[:num_rows], cos_sin_cache, compress_ratio
    )
    flat_cache = kv_cache_2d.reshape(-1)
    value_offsets = (
        token_base[:, None]
        + torch.arange(
            nope_dim,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            scale_dim,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    rope_offsets = (
        token_base[:, None]
        + nope_dim
        + torch.arange(
            rope_dim * 2,
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
    if not state_cache.is_cuda:
        raise ValueError(
            "save_deepseek_v4_compressor_state only supports CUDA tensors."
        )

    _triton_save_compressor_state(
        kv=kv,
        score=score,
        ape=ape,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        block_size=block_size,
        compress_ratio=compress_ratio,
    )


def write_deepseek_v4_indexer_fp8_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write FP8 indexer keys using `[values | fp32 scale]` page layout."""

    if index_k.dim() != 2:
        raise ValueError(f"index_k must be [tokens, dim], got {tuple(index_k.shape)}")
    index_head_dim = int(index_k.shape[-1])
    scale_bytes = deepseek_v4_indexer_fp8_scale_bytes(index_head_dim)
    row_bytes = deepseek_v4_indexer_fp8_row_bytes(index_head_dim)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    min_stride = block_size * row_bytes
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], "
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
        value_base = page_base + pos * index_head_dim
        scale_base = page_base + block_size * index_head_dim + pos * scale_bytes
        q_bytes, scale = _fp8_e4m3_pow2_bytes(index_k[token_idx].float())
        flat_cache[value_base : value_base + index_head_dim].copy_(q_bytes)
        flat_cache[scale_base : scale_base + scale_bytes].copy_(
            scale.reshape(1).view(torch.uint8)
        )


def write_deepseek_v4_indexer_mxfp4_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write MXFP4 indexer keys using the `[values | ue8m0 scales]` layout."""

    if index_k.dim() != 2:
        raise ValueError(f"index_k must be [tokens, dim], got {tuple(index_k.shape)}")
    index_head_dim = int(index_k.shape[-1])
    row_bytes = deepseek_v4_indexer_mxfp4_row_bytes(index_head_dim)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    min_stride = block_size * row_bytes
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    if num_actual == 0:
        return
    if not index_k.is_cuda:
        raise ValueError(
            "write_deepseek_v4_indexer_mxfp4_cache only supports CUDA tensors."
        )
    valid = torch.ones(num_actual, device=index_k.device, dtype=torch.bool)
    _triton_write_indexer_mxfp4_cache_cuda(
        index_k[:num_actual],
        cache_2d,
        slot_mapping[:num_actual],
        valid,
        block_size,
    )


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

    index_head_dim = int(index_k.shape[-1])
    scale_bytes = deepseek_v4_indexer_fp8_scale_bytes(index_head_dim)
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
    value_base = page_base + pos * index_head_dim
    scale_base = page_base + block_size * index_head_dim + pos * scale_bytes

    flat_cache = cache_2d.reshape(-1)
    value_offsets = (
        value_base[:, None]
        + torch.arange(
            index_head_dim,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(scale_bytes, device=cache_2d.device, dtype=torch.int64)[None, :]
    )
    flat_cache[value_offsets] = value_bytes
    flat_cache[scale_offsets] = scale.view(torch.uint8).reshape(num_rows, scale_bytes)


def read_deepseek_v4_indexer_mxfp4_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize MXFP4 indexer cache rows selected by `slot_mapping`."""

    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    index_head_dim, value_bytes, scale_bytes = _indexer_mxfp4_layout_from_cache(
        cache_2d, block_size
    )
    min_stride = block_size * (value_bytes + scale_bytes)
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), index_head_dim)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)

    flat_cache = cache_2d.reshape(-1)
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * value_bytes
    scale_base = page_base + block_size * value_bytes + pos * scale_bytes

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            value_bytes,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    packed = flat_cache[value_offsets]

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            scale_bytes,
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


def read_deepseek_v4_indexer_fp8_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize FP8 indexer cache rows selected by `slot_mapping`."""

    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    index_head_dim, scale_bytes = _indexer_fp8_layout_from_cache(cache_2d, block_size)
    min_stride = block_size * (index_head_dim + scale_bytes)
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out = torch.zeros(
        slot_mapping.numel(),
        index_head_dim,
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
        value_base = page_base + pos * index_head_dim
        scale_base = page_base + block_size * index_head_dim + pos * scale_bytes
        scale = flat_cache[scale_base : scale_base + scale_bytes].view(torch.float32)[0]
        values = flat_cache[value_base : value_base + index_head_dim].view(
            torch.float8_e4m3fn
        )
        out[token_idx].copy_(values.float() * scale)
    return out


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
    `(position + 1) % 128 == 0`.
    """

    if compress_ratio != 128:
        raise ValueError(
            f"HCA cache insert requires compress_ratio=128, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    head_dim = int(rms_norm_weight.numel())
    if state_width != head_dim:
        raise ValueError(f"HCA state width must be {head_dim}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    min_block_stride = kv_cache_block_size * deepseek_v4_swa_row_bytes(
        state_width, rope_dim
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
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "deepseek_v4_hca_compress_kv_cache_insert only supports CUDA tensors."
        )

    _triton_fused_sparse_compress_cache_insert(
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


def deepseek_v4_hca_direct_compress_kv_cache_insert(
    state_cache: torch.Tensor,
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    active_token_indices: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress HCA active rows without falling back to paged row gathering.

    The kernel reads current-step tokens directly from ``kv``/``score`` and
    prefix-boundary tokens from the loaded paged ``state_cache``. This keeps HCA
    cached-prefill rows on the direct path even when their 128-token window
    crosses the prefix/tail boundary.

    Args:
        state_cache: Paged HCA compressor state with shape
            ``[blocks, block_size, 2 * head_dim]``.
        kv: Current forward HCA value states with shape ``[tokens, head_dim]``.
        score: Current forward HCA score states with shape ``[tokens, head_dim]``.
        ape: HCA APE table with shape ``[128, head_dim]``.
        token_to_req_indices: Request index for each current-step token.
        query_start_loc: Prefix-sum token offsets for current-step requests.
        positions: Absolute token positions for the current forward step.
        block_table: Paged compressor-state block table.
        compressor_block_size: Compressor-state page size.
        rms_norm_weight: RMSNorm weight applied after weighted reduction.
        rms_norm_eps: RMSNorm epsilon.
        cos_sin_cache: RoPE cache indexed by compressed positions.
        kv_cache_2d: Destination compressed KV cache byte storage.
        kv_slot_mapping: Destination compressed KV slot per source token.
        kv_cache_block_size: Destination compressed KV cache page size.
        active_token_indices: Source token offsets for HCA compressed rows.
        block_table_base_offsets: Optional logical-page offset per request.
    """

    if kv.shape != score.shape:
        raise ValueError(
            f"kv and score shapes must match, got {kv.shape} vs {score.shape}"
        )
    if kv.dim() != 2:
        raise ValueError(f"kv/score must be [tokens, head_dim], got {kv.shape}")
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = kv.shape[-1]
    head_dim = int(rms_norm_weight.numel())
    if state_width != head_dim:
        raise ValueError(f"HCA state width must be {head_dim}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    if state_cache.shape[-1] != state_width * 2:
        raise ValueError(
            f"state_cache last dim must be {state_width * 2}, "
            f"got {state_cache.shape[-1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    min_block_stride = kv_cache_block_size * deepseek_v4_swa_row_bytes(
        state_width, rope_dim
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")
    if not kv.is_cuda:
        raise ValueError(
            "deepseek_v4_hca_direct_compress_kv_cache_insert only supports CUDA tensors."
        )
    if token_to_req_indices.device != positions.device:
        token_to_req_indices = token_to_req_indices.to(
            positions.device, non_blocking=True
        )
    if query_start_loc.device != positions.device:
        query_start_loc = query_start_loc.to(positions.device, non_blocking=True)

    _triton_fused_hca_direct_compress_cache_insert(
        state_cache=state_cache,
        kv=kv,
        score=score,
        ape=ape,
        token_to_req_indices=token_to_req_indices,
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=block_table,
        compressor_block_size=compressor_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache_2d=kv_cache_2d,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        active_token_indices=active_token_indices,
        block_table_base_offsets=block_table_base_offsets,
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
    head_dim = int(rms_norm_weight.numel())
    expected_width = head_dim * 2
    if state_width != expected_width:
        raise ValueError(f"CSA state width must be {expected_width}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    min_block_stride = kv_cache_block_size * deepseek_v4_swa_row_bytes(
        head_dim, rope_dim
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "deepseek_v4_csa_compress_kv_cache_insert only supports CUDA tensors."
        )

    _triton_fused_sparse_compress_cache_insert(
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
    index_head_dim = int(rms_norm_weight.numel())
    expected_width = index_head_dim * 2
    if state_width != expected_width:
        raise ValueError(
            f"CSA indexer state width must be {expected_width}, got {state_width}"
        )

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "deepseek_v4_csa_indexer_cache_insert only supports CUDA tensors."
        )
    if use_fp4_cache:
        _triton_fused_csa_indexer_mxfp4_cache_insert(
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
        head_dim=index_head_dim,
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
        int(cos_sin_cache.shape[-1]),
    )
    rotated = _deepseek_v4_hadamard_rotate(rotated).float()
    _write_deepseek_v4_indexer_fp8_cache_capturable(
        rotated,
        kv_cache_2d,
        kv_slot_mapping[:num_actual],
        valid,
        block_size=kv_cache_block_size,
    )
