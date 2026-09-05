# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Pure-PyTorch DSpark captured-context attention.

The math follows the public DeepSeek DSpark reference. The production entry
point is fixed-shape and tensorized so it can execute inside TokenSpeed's target
CUDA Graph. These local primitives intentionally preserve reference-level
position, FP8, and normalization semantics during correctness bring-up instead
of mixing in runtime kernels with different contracts. Follow-up performance
work may replace them with library or fused kernels only after parity and
endpoint gains are demonstrated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel import fp8_quantize_dequantize
from tokenspeed_kernel.ops.layernorm import grouped_rmsnorm as kernel_grouped_rmsnorm
from tokenspeed_kernel.ops.layernorm import rmsnorm as kernel_rmsnorm


def dspark_fp8_quant_dequant(
    x: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply the DSpark UE8M0-scaled FP8 quantize/dequantize contract."""

    if x.shape[-1] % block_size != 0:
        raise ValueError(
            "DSpark FP8 activation width must be divisible by block_size; "
            f"got width={x.shape[-1]}, block_size={block_size}."
        )
    if x.is_cuda:
        return fp8_quantize_dequantize(x, group_size=block_size)
    original_dtype = x.dtype
    blocks = x.float().unflatten(-1, (-1, block_size))
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    quantized = torch.clamp(blocks / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).flatten(-2).to(original_dtype)


def _dspark_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Emulate the public DSpark FP8 linear activation contract."""

    return F.linear(dspark_fp8_quant_dequant(x, 128), weight)


def _quantize_dspark_non_rope(
    tensor: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """Quantize/dequantize the non-RoPE KV channels in 64-value groups."""

    non_rope = dspark_fp8_quant_dequant(tensor[..., :-rope_head_dim], 64)
    return torch.cat([non_rope, tensor[..., -rope_head_dim:]], dim=-1)


def _dspark_output_projection(
    output: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Apply the BF16 grouped projection followed by the FP8 linear."""

    batch, sequence = output.shape[:2]
    output = output.reshape(batch, sequence, n_groups, -1)
    grouped_wo_a = wo_a.view(n_groups, o_lora_rank, -1)
    output = torch.einsum("bsgd,grd->bsgr", output, grouped_wo_a)
    return _dspark_fp8_linear(output.flatten(2), wo_b)


def precompute_dspark_freqs_cis(
    rope_head_dim: int,
    seqlen: int,
    rope_theta: float = 10000.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return plain-RoPE complex phases for the DSpark draft."""

    freqs = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                rope_head_dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / rope_head_dim
        )
    )
    positions = torch.arange(seqlen, dtype=torch.float32, device=device)
    phases = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(phases), phases)


def apply_dspark_rotary_batched(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply per-request adjacent-pair RoPE phases."""

    original_dtype = x.dtype
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    groups, sequence, half = freqs_cis.shape
    phases = (
        freqs_cis.view(groups, sequence, half)
        if complex_x.ndim == 3
        else freqs_cis.view(groups, sequence, 1, half)
    )
    return torch.view_as_real(complex_x * phases).flatten(-2).to(original_dtype)


def get_dspark_topk_idxs_batched(
    window_size: int,
    block_size: int,
    start_pos: torch.Tensor,
) -> torch.Tensor:
    """Build fixed-width, masked DSpark context indices."""

    device = start_pos.device
    groups = start_pos.shape[0]
    context_columns = torch.arange(window_size, device=device)
    valid = context_columns.unsqueeze(0) <= start_pos.unsqueeze(1)
    context_indices = torch.where(
        valid,
        context_columns.unsqueeze(0).expand(groups, -1),
        torch.full_like(valid, -1, dtype=torch.long),
    )
    block_indices = window_size + torch.arange(block_size, device=device)
    block_indices = block_indices.unsqueeze(0).expand(groups, -1)
    row = torch.cat([context_indices, block_indices], dim=1).to(torch.int32)
    return row.unsqueeze(1).expand(groups, block_size, -1).contiguous()


def dspark_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run index-gathered MQA with an attention-sink denominator term."""

    batch, queries, heads, dim = q.shape
    indices = topk_indices.long()
    valid = indices >= 0
    safe_indices = indices.clamp_min(0)
    expanded_kv = kv.unsqueeze(1).expand(batch, queries, kv.shape[1], dim)
    gathered = torch.gather(
        expanded_kv,
        2,
        safe_indices.unsqueeze(-1).expand(
            batch,
            queries,
            safe_indices.shape[-1],
            dim,
        ),
    ).float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered)
    scores.mul_(softmax_scale)
    scores.masked_fill_(~valid.unsqueeze(2), float("-inf"))
    maximum = scores.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isinf(maximum), torch.zeros_like(maximum), maximum)
    probabilities = torch.exp(scores - maximum)
    sink = torch.exp(attn_sink.float().view(1, 1, heads) - maximum.squeeze(-1))
    denominator = probabilities.sum(dim=-1) + sink
    output = torch.einsum("bmhk,bmkd->bmhd", probabilities, gathered)
    output.div_(denominator.unsqueeze(-1))
    return output.to(q.dtype)


def _rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply DSpark RMSNorm through the platform kernel on accelerators."""

    if x.is_cuda:
        return kernel_rmsnorm(x, weight, eps)
    original_dtype = x.dtype
    normalized = x.float()
    normalized.mul_(torch.rsqrt(normalized.square().mean(-1, keepdim=True) + eps))
    return (weight.float() * normalized).to(original_dtype)


# Public name for direct contract tests while preserving the historical private
# helper imported by the DSpark model implementation.
dspark_rmsnorm = _rmsnorm


def _normalize_query_per_head(
    query: torch.Tensor,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Normalize every query head without materializing intermediate tensors."""

    if query.is_cuda:
        return kernel_grouped_rmsnorm(query, head_dim, eps, out=query)
    query.mul_(torch.rsqrt(query.square().mean(-1, keepdim=True) + eps))
    return query


def _rope_last_dims_batched(
    tensor: torch.Tensor,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    non_rope = tensor[..., :-rope_head_dim]
    rope = apply_dspark_rotary_batched(
        tensor[..., -rope_head_dim:],
        freqs_cis,
        inverse=inverse,
    )
    return torch.cat([non_rope, rope], dim=-1)


def dspark_attention_forward_batched(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm_w: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    attn_sink: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Run graph-safe DSpark attention for a fixed request batch."""

    groups, block_size, _ = x.shape
    main_freqs = freqs_cis[start_pos].unsqueeze(1)
    block_positions = (
        start_pos.unsqueeze(1)
        + 1
        + torch.arange(
            block_size,
            device=x.device,
        )
    )
    block_freqs = freqs_cis[block_positions]

    main_kv = dspark_rmsnorm(_dspark_fp8_linear(main_x, wkv), kv_norm_w, eps)
    main_kv = _rope_last_dims_batched(
        main_kv,
        rope_head_dim,
        main_freqs,
    )
    main_kv = _quantize_dspark_non_rope(main_kv, rope_head_dim)

    query = dspark_rmsnorm(_dspark_fp8_linear(x, wq_a), q_norm_w, eps)
    query = _dspark_fp8_linear(query, wq_b).unflatten(-1, (n_heads, head_dim))
    query = _normalize_query_per_head(query, head_dim, eps)
    query = _rope_last_dims_batched(
        query,
        rope_head_dim,
        block_freqs,
    )

    block_kv = dspark_rmsnorm(_dspark_fp8_linear(x, wkv), kv_norm_w, eps)
    block_kv = _rope_last_dims_batched(
        block_kv,
        rope_head_dim,
        block_freqs,
    )
    block_kv = _quantize_dspark_non_rope(block_kv, rope_head_dim)

    slot_positions = start_pos % window_size
    kv_cache[slots, slot_positions] = main_kv.squeeze(1).to(kv_cache.dtype)
    cache_rows = kv_cache[slots]
    all_kv = torch.cat([cache_rows, block_kv], dim=1)
    topk_indices = get_dspark_topk_idxs_batched(
        window_size,
        block_size,
        start_pos,
    )
    output = dspark_sparse_attn(
        query,
        all_kv,
        attn_sink,
        topk_indices,
        softmax_scale,
    )
    output = _rope_last_dims_batched(
        output,
        rope_head_dim,
        block_freqs,
        inverse=True,
    )

    return _dspark_output_projection(
        output,
        wo_a,
        wo_b,
        n_groups,
        o_lora_rank,
    )
