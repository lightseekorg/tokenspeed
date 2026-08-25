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

"""Fused Kimi K3 AttnRes mixing and output RMSNorm for gfx1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

gfx1250 = gl.amd.gfx1250
_BLOCK_H = gl.constexpr(8192)
_LOAD_ELEMS = gl.constexpr(2)


@gluon.jit
def _load_candidate(
    prefix,
    block_residual,
    token,
    hidden,
    hidden_mask,
    stride_block_t: gl.constexpr,
    stride_block_n: gl.constexpr,
    candidate: gl.constexpr,
    N: gl.constexpr,
):
    if candidate == N - 1:
        return prefix
    # Candidate offsets can exceed int32; fold the block stride into the pointer.
    ptr = block_residual + candidate * stride_block_n
    return gfx1250.buffer_load(
        ptr,
        (token * stride_block_t + hidden).to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)


@gluon.jit
def _attn_res_rmsnorm_kernel(
    layer_residual,
    delta,
    block_residual,
    res_weight,
    score_rms_weight,
    output_rms_weight,
    output,
    stride_layer_t: gl.constexpr,
    stride_delta_t: gl.constexpr,
    stride_block_t: gl.constexpr,
    stride_block_n: gl.constexpr,
    stride_output_t: gl.constexpr,
    H: gl.constexpr,
    N: gl.constexpr,
    BLOCK_WRITE_IDX: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    hidden_layout: gl.constexpr = gl.BlockedLayout(
        [_LOAD_ELEMS], [32], [NUM_WARPS], [0]
    )
    hidden = gl.arange(0, _BLOCK_H, layout=hidden_layout)
    hidden_mask = hidden < H

    prefix = gfx1250.buffer_load(
        layer_residual,
        (token * stride_layer_t + hidden).to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)
    if HAS_DELTA:
        prefix += gfx1250.buffer_load(
            delta,
            (token * stride_delta_t + hidden).to(gl.int32),
            mask=hidden_mask,
            other=0.0,
        ).to(gl.float32)
        prefix = prefix.to(layer_residual.dtype.element_ty).to(gl.float32)
        gfx1250.buffer_store(
            prefix.to(layer_residual.dtype.element_ty),
            layer_residual,
            (token * stride_layer_t + hidden).to(gl.int32),
            mask=hidden_mask,
        )
    if WRITE_BLOCK:
        block_write_ptr = block_residual + BLOCK_WRITE_IDX * stride_block_n
        gfx1250.buffer_store(
            prefix.to(block_residual.dtype.element_ty),
            block_write_ptr,
            (token * stride_block_t + hidden).to(gl.int32),
            mask=hidden_mask,
        )

    if N == 1:
        mixed = prefix
    else:
        scorer = gfx1250.buffer_load(
            res_weight,
            hidden.to(gl.int32),
            mask=hidden_mask,
            other=0.0,
        ).to(gl.float32)
        scorer *= gfx1250.buffer_load(
            score_rms_weight,
            hidden.to(gl.int32),
            mask=hidden_mask,
            other=0.0,
        ).to(gl.float32)
        max_logit = -float("inf")
        denominator = 0.0
        mixed = gl.zeros([_BLOCK_H], gl.float32, hidden_layout)
        for candidate in gl.static_range(N):
            value = _load_candidate(
                prefix,
                block_residual,
                token,
                hidden,
                hidden_mask,
                stride_block_t,
                stride_block_n,
                candidate,
                N,
            )
            square_sum = gl.sum(value * value, axis=0)
            # Match the established high-precision score reduction.
            dot = gl.sum((value * scorer).to(gl.float64), axis=0).to(gl.float32)
            score = dot * gl.rsqrt(square_sum / H + SCORE_EPS)
            next_max = gl.maximum(max_logit, score)
            old_scale = gl.exp(max_logit - next_max)
            candidate_scale = gl.exp(score - next_max)
            denominator = denominator * old_scale + candidate_scale
            mixed = mixed * old_scale + candidate_scale * value
            max_logit = next_max
        mixed /= denominator

    # Preserve the AttnRes BF16 boundary before output RMSNorm.
    mixed = mixed.to(gl.bfloat16).to(gl.float32)
    inverse_rms = gl.rsqrt(gl.sum(mixed * mixed, axis=0) / H + OUTPUT_EPS)
    output_weight = gfx1250.buffer_load(
        output_rms_weight,
        hidden.to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)
    gfx1250.buffer_store(
        (mixed * inverse_rms * output_weight).to(output.dtype.element_ty),
        output,
        (token * stride_output_t + hidden).to(gl.int32),
        mask=hidden_mask,
    )


def attn_res_rmsnorm_gfx1250(
    *,
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    score_eps: float,
    output_rms_weight: torch.Tensor,
    output_eps: float,
    num_valid_blocks: int,
    delta: torch.Tensor | None = None,
    block_write_idx: int = -1,
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
            "gfx1250 AttnRes requires BF16 input with "
            "H in {4096, 5120, 6144, 7168, 8192}"
        )
    if not 0 <= num_valid_blocks <= 11:
        raise ValueError("gfx1250 AttnRes supports at most 11 block snapshots")
    if (
        block_residual.ndim != 3
        or block_residual.shape[0] != tokens
        or block_residual.shape[2] != hidden
    ):
        raise ValueError(
            "gfx1250 AttnRes requires token-major block storage [T, blocks, H]"
        )
    if block_residual.shape[1] < num_valid_blocks:
        raise ValueError("num_valid_blocks is outside block_residual")
    if block_write_idx >= 0 and (
        block_write_idx != num_valid_blocks
        or block_write_idx >= block_residual.shape[1]
    ):
        raise ValueError("block_write_idx must append within block_residual")
    if delta is not None and (
        delta.shape != layer_residual.shape
        or delta.dtype != layer_residual.dtype
        or delta.device != layer_residual.device
        or delta.stride(1) != 1
    ):
        raise ValueError("gfx1250 AttnRes delta must match layer_residual")
    if layer_residual.stride(1) != 1 or block_residual.stride(2) != 1:
        raise ValueError("gfx1250 AttnRes requires a contiguous hidden dimension")

    output = torch.empty_like(layer_residual)
    num_warps = 4
    delta_tensor = layer_residual if delta is None else delta
    _attn_res_rmsnorm_kernel[(tokens,)](
        layer_residual,
        delta_tensor,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        output,
        stride_layer_t=layer_residual.stride(0),
        stride_delta_t=delta_tensor.stride(0),
        stride_block_t=block_residual.stride(0),
        stride_block_n=block_residual.stride(1),
        stride_output_t=output.stride(0),
        H=hidden,
        N=num_valid_blocks + 1,
        BLOCK_WRITE_IDX=0 if block_write_idx < 0 else block_write_idx,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        SCORE_EPS=score_eps,
        OUTPUT_EPS=output_eps,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return output


__all__ = ["attn_res_rmsnorm_gfx1250"]
