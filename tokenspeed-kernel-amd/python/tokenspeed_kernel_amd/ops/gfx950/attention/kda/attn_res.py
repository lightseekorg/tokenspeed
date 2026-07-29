# Copyright (c) 2026 LightSeek Foundation

"""Fused Kimi K3 AttnRes mixing and output RMSNorm for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna4 = gl.amd.cdna4
_BLOCK_H = gl.constexpr(8192)
_BLOCK_N = gl.constexpr(16)
_LOAD_ELEMS = gl.constexpr(2)


@gluon.jit
def _load_candidate(
    layer_residual,
    block_residual,
    token,
    hidden,
    hidden_mask,
    stride_layer_t: gl.constexpr,
    stride_block_t: gl.constexpr,
    stride_block_n: gl.constexpr,
    candidate: gl.constexpr,
    N: gl.constexpr,
):
    if candidate == N - 1:
        ptr = layer_residual
        offset = token * stride_layer_t + hidden
    else:
        ptr = block_residual
        offset = token * stride_block_t + candidate * stride_block_n + hidden
    return cdna4.buffer_load(
        ptr,
        offset.to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)


@gluon.jit
def _attn_res_rmsnorm_kernel(
    layer_residual,
    block_residual,
    res_weight,
    score_rms_weight,
    output_rms_weight,
    output,
    stride_layer_t: gl.constexpr,
    stride_block_t: gl.constexpr,
    stride_block_n: gl.constexpr,
    stride_output_t: gl.constexpr,
    H: gl.constexpr,
    N: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    hidden_layout: gl.constexpr = gl.BlockedLayout(
        [_LOAD_ELEMS], [64], [NUM_WARPS], [0]
    )
    candidate_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, _BLOCK_H, layout=hidden_layout)
    hidden_mask = hidden < H
    candidates = gl.arange(0, _BLOCK_N, layout=candidate_layout)

    scorer = cdna4.buffer_load(
        res_weight,
        hidden.to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)
    scorer *= cdna4.buffer_load(
        score_rms_weight,
        hidden.to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)

    logits = gl.full([_BLOCK_N], -float("inf"), gl.float32, candidate_layout)
    for candidate in gl.static_range(N):
        value = _load_candidate(
            layer_residual,
            block_residual,
            token,
            hidden,
            hidden_mask,
            stride_layer_t,
            stride_block_t,
            stride_block_n,
            candidate,
            N,
        )
        square_sum = gl.sum(value * value, axis=0)
        # FP32 products preserve the model's arithmetic; FP64 accumulation
        # reduces sensitivity to the reduction tree before the FP32 softmax.
        dot = gl.sum((value * scorer).to(gl.float64), axis=0).to(gl.float32)
        score = dot * gl.rsqrt(square_sum / H + SCORE_EPS)
        logits = gl.where(candidates == candidate, score, logits)

    logits -= gl.max(logits, axis=0)
    probabilities = gl.exp(logits)
    probabilities /= gl.sum(probabilities, axis=0)

    mixed = gl.zeros([_BLOCK_H], gl.float32, hidden_layout)
    for candidate in gl.static_range(N):
        probability = gl.sum(
            gl.where(candidates == candidate, probabilities, 0.0), axis=0
        )
        mixed += probability * _load_candidate(
            layer_residual,
            block_residual,
            token,
            hidden,
            hidden_mask,
            stride_layer_t,
            stride_block_t,
            stride_block_n,
            candidate,
            N,
        )

    # Preserve the existing AttnRes BF16 boundary before output RMSNorm.
    mixed = mixed.to(gl.bfloat16).to(gl.float32)
    inverse_rms = gl.rsqrt(gl.sum(mixed * mixed, axis=0) / H + OUTPUT_EPS)
    output_weight = cdna4.buffer_load(
        output_rms_weight,
        hidden.to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)
    cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(output.dtype.element_ty),
        output,
        (token * stride_output_t + hidden).to(gl.int32),
        mask=hidden_mask,
    )


def attn_res_rmsnorm_gfx950(
    *,
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    score_eps: float,
    output_rms_weight: torch.Tensor,
    output_eps: float,
    num_valid_blocks: int,
) -> torch.Tensor:
    """Mix AttnRes candidates and apply the following RMSNorm in one launch."""
    tokens, hidden = layer_residual.shape
    if layer_residual.dtype != torch.bfloat16 or hidden not in {
        4096,
        5120,
        6144,
        7168,
        8192,
    }:
        raise ValueError(
            "gfx950 AttnRes requires BF16 input with "
            "H in {4096, 5120, 6144, 7168, 8192}"
        )
    if not 0 <= num_valid_blocks <= 11:
        raise ValueError("gfx950 AttnRes supports at most 11 block snapshots")
    if layer_residual.stride(1) != 1 or block_residual.stride(2) != 1:
        raise ValueError("gfx950 AttnRes requires a contiguous hidden dimension")

    output = torch.empty_like(layer_residual)
    num_warps = 4 if num_valid_blocks <= 4 else 8
    _attn_res_rmsnorm_kernel[(tokens,)](
        layer_residual,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        output,
        stride_layer_t=layer_residual.stride(0),
        stride_block_t=block_residual.stride(0),
        stride_block_n=block_residual.stride(1),
        stride_output_t=output.stride(0),
        H=hidden,
        N=num_valid_blocks + 1,
        SCORE_EPS=score_eps,
        OUTPUT_EPS=output_eps,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return output


__all__ = ["attn_res_rmsnorm_gfx950"]
