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

import torch
from tokenspeed_kernel import (
    dsv4_csa_indexer_fp8_cache_insert,
    dsv4_swa_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.dsv4 import (
    dsv4_fused_csa_indexer_mxfp4_cache_insert,
    dsv4_fused_indexer_q_rope_hadamard_mxfp4,
    dsv4_fused_sparse_compress_cache_insert,
    dsv4_save_compressor_state,
)
from tokenspeed_kernel.ops.transform import hadamard_transform

from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    DEEPSEEK_V4_FP8_MAX,
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    deepseek_v4_indexer_fp8_layout_from_row_bytes,
    deepseek_v4_indexer_fp8_scale_bytes,
    deepseek_v4_swa_row_bytes,
)

__all__ = (
    "deepseek_v4_csa_compress_kv_cache_insert",
    "deepseek_v4_csa_indexer_cache_insert",
    "deepseek_v4_hca_compress_kv_cache_insert",
    "deepseek_v4_prepare_indexer_q",
    "deepseek_v4_prepare_indexer_q_fp8",
    "deepseek_v4_prepare_indexer_q_mxfp4",
    "gather_paged_indexer_fp8_cache",
    "fused_qnorm_rope_kv_insert",
    "read_deepseek_v4_indexer_fp8_cache",
    "save_deepseek_v4_compressor_state",
)


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
    q_out: torch.Tensor | None = None,
) -> None:
    """Run the DeepSeek V4 fused SWA cache insert op.

    Expected contract:
    - q: [tokens, local_heads, 512], mutated in place by RMSNorm/RoPE unless
      q_out is provided
    - kv: [tokens, 512], source KV latent before RoPE/quant insert
    - swa_kv_cache_2d: uint8 cache blocks flattened as [num_blocks, block_bytes]
    - slot_mapping: output token slots in the paged SWA cache
    - positions: absolute token positions
    """

    dsv4_swa_cache_insert(
        q=q,
        kv=kv,
        swa_kv_cache=swa_kv_cache_2d,
        slot_mapping=slot_mapping,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        rms_norm_eps=rms_norm_eps,
        page_size=block_size,
        q_out=q_out,
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


def _deepseek_v4_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    rotated = hadamard_transform(
        x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    )
    return rotated.reshape(shape)


def deepseek_v4_prepare_indexer_q_mxfp4(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Apply indexer Q RoPE and return packed MXFP4 values and scales."""

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
    return dsv4_fused_indexer_q_rope_hadamard_mxfp4(
        index_q=index_q,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
    )


def deepseek_v4_prepare_indexer_q(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare BF16 index queries and FP32 weights for public DSA top-k."""

    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [tokens, heads, dim], got {index_q.shape}")
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")
    rope_dim = int(cos_sin_cache.shape[-1])
    rotated = _apply_gptj_rope_tail_rows(
        index_q,
        positions,
        cos_sin_cache,
        rope_dim,
    )
    query = _deepseek_v4_hadamard_rotate(rotated).to(torch.bfloat16).contiguous()
    return query, (weights.float() * float(head_scale)).contiguous()


def deepseek_v4_prepare_indexer_q_fp8(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply indexer Q RoPE + Hadamard and return DeepGEMM-ready FP8 values.

    Non-Blackwell (SM90) analogue of ``deepseek_v4_prepare_indexer_q_mxfp4``.
    ``deep_gemm.fp8_paged_mqa_logits`` / ``fp8_mqa_logits`` consume the query as
    a plain ``float8_e4m3fn`` tensor (no per-token query scale) with the softmax
    and head scales folded into ``weights`` -- mirroring SGLang's
    ``fused_q_indexer_rope_hadamard_quant``.

    Args:
        index_q: ``[tokens, heads, index_head_dim]`` bf16/fp32 indexer queries.
        positions: ``[tokens]`` absolute token positions for RoPE.
        cos_sin_cache: ``[max_pos, rope_dim]`` fused cos/sin cache.
        weights: ``[tokens, heads]`` (or trailing singleton) indexer weights.
        softmax_scale: ``index_head_dim**-0.5`` attention softmax scale.
        head_scale: ``n_head**-0.5`` head normalization scale.

    Returns:
        Tuple of ``(q_fp8, weights_out)`` where ``q_fp8`` is
        ``[tokens, heads, index_head_dim]`` ``float8_e4m3fn`` and
        ``weights_out`` is ``[tokens, heads]`` float32 with both scales folded
        in.
    """

    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [tokens, heads, dim], got {index_q.shape}")
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
            "deepseek_v4_prepare_indexer_q_fp8 only supports CUDA tensors."
        )

    weights_out = (weights.float() * float(softmax_scale) * float(head_scale)).float()
    if index_q.shape[0] == 0:
        q_fp8 = index_q.new_empty(index_q.shape, dtype=torch.float8_e4m3fn)
        return q_fp8, weights_out

    rotated = _apply_gptj_rope_tail_rows(
        index_q,
        positions,
        cos_sin_cache,
        rope_dim,
    )
    rotated = _deepseek_v4_hadamard_rotate(rotated)
    q_fp8 = rotated.to(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    return q_fp8, weights_out


def gather_paged_indexer_fp8_cache(
    cache_2d: torch.Tensor,
    page_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ragged FP8 indexer keys for ``deep_gemm.fp8_mqa_logits``.

    Torch analogue of the CUDA MXFP4 paged gather. Produces the fp8 keys and
    their fp32 per-token scales laid out contiguously in ``cu_seq_lens`` order.

    Args:
        cache_2d: ``[pages, block_size * (index_head_dim + 4)]`` uint8 fp8 cache.
        page_table: ``[num_reqs, max_blocks]`` int32 logical->physical pages.
        cu_seq_lens: ``[num_reqs + 1]`` int32 cumulative compressed key lengths.
        block_size: cache block size (tokens per page), e.g. 64.

    Returns:
        Tuple ``(k_fp8, k_scale)`` with ``k_fp8`` of shape ``[total_k, dim]``
        ``float8_e4m3fn`` and ``k_scale`` of shape ``[total_k]`` float32.
    """

    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    index_head_dim, scale_bytes = _indexer_fp8_layout_from_cache(cache_2d, block_size)
    device = cache_2d.device
    cu_seq_lens = cu_seq_lens.to(device=device, dtype=torch.int64)
    total_rows = int(cu_seq_lens[-1].item()) if cu_seq_lens.numel() else 0
    if total_rows == 0:
        return (
            torch.empty((0, index_head_dim), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    page_table = page_table.to(device=device, dtype=torch.int64)
    row_ids = torch.arange(total_rows, device=device, dtype=torch.int64)
    # searchsorted over the per-req end offsets maps each output row to its req.
    req = torch.searchsorted(cu_seq_lens[1:].contiguous(), row_ids, right=True)
    req = req.clamp_max(page_table.shape[0] - 1)
    local = row_ids - cu_seq_lens[req]
    logical_block = torch.div(local, block_size, rounding_mode="floor")
    logical_block = logical_block.clamp_max(page_table.shape[1] - 1)
    in_block = local % block_size
    phys = page_table[req, logical_block]

    # Index pages via advanced indexing on the 2-D cache, NOT via
    # page * stride(0) into reshape(-1): the indexer cache can be a strided
    # field view of a larger LCM arena (stride(0) > shape[1]), where a
    # flattened view only covers the logical elements and physical-stride
    # offsets read past its end.
    value_offsets = (
        in_block[:, None] * index_head_dim
        + torch.arange(index_head_dim, device=device, dtype=torch.int64)[None, :]
    )
    scale_offsets = (
        block_size * index_head_dim
        + in_block[:, None] * scale_bytes
        + torch.arange(scale_bytes, device=device, dtype=torch.int64)[None, :]
    )
    k_fp8 = (
        cache_2d[phys[:, None], value_offsets].contiguous().view(torch.float8_e4m3fn)
    )
    k_scale = (
        cache_2d[phys[:, None], scale_offsets]
        .contiguous()
        .view(torch.float32)
        .reshape(total_rows)
    )
    return k_fp8, k_scale


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

    dsv4_save_compressor_state(
        kv=kv,
        score=score,
        ape=ape,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        block_size=block_size,
        compress_ratio=compress_ratio,
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

    # Index pages via advanced indexing on the 2-D cache, NOT via
    # page * stride(0) into reshape(-1): the indexer cache can be a strided
    # field view of a larger LCM arena (stride(0) > shape[1]), where a
    # flattened view only covers the logical elements and physical-stride
    # offsets run past its end (or silently corrupt neighboring fields).
    value_offsets = (
        pos[:, None] * index_head_dim
        + torch.arange(
            index_head_dim,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        block_size * index_head_dim
        + pos[:, None] * scale_bytes
        + torch.arange(scale_bytes, device=cache_2d.device, dtype=torch.int64)[None, :]
    )
    cache_2d[pages[:, None], value_offsets] = value_bytes
    cache_2d[pages[:, None], scale_offsets] = scale.view(torch.uint8).reshape(
        num_rows, scale_bytes
    )


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
    for token_idx, raw_slot in enumerate(slot_mapping.tolist()):
        slot = int(raw_slot)
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        # Row-index the 2-D cache (strided-view safe) instead of computing
        # page * stride(0) offsets into reshape(-1).
        page_row = cache_2d[page]
        value_base = pos * index_head_dim
        scale_base = block_size * index_head_dim + pos * scale_bytes
        scale = page_row[scale_base : scale_base + scale_bytes].view(torch.float32)[0]
        values = page_row[value_base : value_base + index_head_dim].view(
            torch.float8_e4m3fn
        )
        out[token_idx].copy_(values.float() * scale)
    return out


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

    dsv4_fused_sparse_compress_cache_insert(
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

    dsv4_fused_sparse_compress_cache_insert(
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
        dsv4_fused_csa_indexer_mxfp4_cache_insert(
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
        )
        return

    dsv4_csa_indexer_fp8_cache_insert(
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
    )
