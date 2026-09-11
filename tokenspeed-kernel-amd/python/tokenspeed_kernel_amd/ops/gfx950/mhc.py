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

"""GFX950 mHC reduction and Sinkhorn kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = [
    "gluon_mhc_pre_reduce_apply_gfx950",
]


@gluon.jit
def _mhc_pre_reduce_apply_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    layer_input,
    post_mix,
    comb_mix,
    num_tokens,
    HIDDEN_SIZE: gl.constexpr,
    RMS_EPS: gl.constexpr,
    HC_EPS: gl.constexpr,
    SINKHORN_ITERS: gl.constexpr,
    N_SPLITS: gl.constexpr,
    BLOCK_H: gl.constexpr,
):
    token = gl.program_id(0)
    hidden_block = gl.program_id(1)
    split_layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    splits = gl.arange(0, 128, layout=split_layout)
    split_mask = splits < N_SPLITS

    square_parts = gl.load(
        gemm_out_sqrsum + splits * num_tokens + token,
        mask=split_mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(square_parts, axis=0)
    inverse_rms = gl.rsqrt(square_sum / (4 * HIDDEN_SIZE) + RMS_EPS)
    pre_scale = gl.load(hc_scale).to(gl.float32)

    hidden_layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    dims = hidden_block * BLOCK_H + gl.arange(0, BLOCK_H, layout=hidden_layout)
    hidden_acc = gl.zeros([BLOCK_H], gl.float32, layout=hidden_layout)
    for head in gl.static_range(4):
        projection_parts = gl.load(
            gemm_out_mul + splits * num_tokens * 24 + token * 24 + head,
            mask=split_mask,
            other=0.0,
        ).to(gl.float32)
        projection = gl.sum(projection_parts, axis=0)
        pre = (
            1.0
            / (
                1.0
                + gl.exp(
                    -(
                        projection * inverse_rms * pre_scale
                        + gl.load(hc_base + head).to(gl.float32)
                    )
                )
            )
            + HC_EPS
        )
        values = gl.load(
            residual + token * 4 * HIDDEN_SIZE + head * HIDDEN_SIZE + dims
        ).to(gl.float32)
        hidden_acc += pre * values
    gl.store(layer_input + token * HIDDEN_SIZE + dims, hidden_acc.to(gl.bfloat16))

    if hidden_block == 0:
        vector_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        lane = gl.arange(0, 64, layout=vector_layout)
        post_projection = gl.zeros([64], gl.float32, layout=vector_layout)
        for split in gl.static_range(N_SPLITS):
            post_projection += gl.load(
                gemm_out_mul + split * num_tokens * 24 + token * 24 + lane,
                mask=(lane >= 4) & (lane < 8),
                other=0.0,
            ).to(gl.float32)
        post_scale = gl.load(hc_scale + 1).to(gl.float32)
        post_values = 2.0 / (
            1.0
            + gl.exp(
                -(
                    post_projection * inverse_rms * post_scale
                    + gl.load(
                        hc_base + lane, mask=(lane >= 4) & (lane < 8), other=0.0
                    ).to(gl.float32)
                )
            )
        )
        gl.store(
            post_mix + token * 4 + lane - 4,
            post_values,
            mask=(lane >= 4) & (lane < 8),
        )

        matrix_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [1, 1], [1, 0])
        rows = gl.arange(0, 4, layout=gl.SliceLayout(1, matrix_layout))
        cols = gl.arange(0, 16, layout=gl.SliceLayout(0, matrix_layout))
        active = cols[None, :] < 4
        comb_offsets = rows[:, None] * 4 + cols[None, :]
        comb_projection = gl.zeros([4, 16], gl.float32, layout=matrix_layout)
        for split in gl.static_range(N_SPLITS):
            comb_projection += gl.load(
                gemm_out_mul + split * num_tokens * 24 + token * 24 + 8 + comb_offsets,
                mask=active,
                other=0.0,
            ).to(gl.float32)
        comb_scale = gl.load(hc_scale + 2).to(gl.float32)
        comb = gl.where(
            active,
            comb_projection * inverse_rms * comb_scale
            + gl.load(hc_base + 8 + comb_offsets, mask=active, other=0.0).to(
                gl.float32
            ),
            0.0,
        )
        row_max = gl.max(gl.where(active, comb, -float("inf")), axis=1)
        comb = gl.where(active, gl.exp(comb - row_max[:, None]), comb)
        row_sum = gl.sum(gl.where(active, comb, 0.0), axis=1)
        comb = gl.where(active, comb / row_sum[:, None] + HC_EPS, comb)
        col_sum = gl.sum(gl.where(active, comb, 0.0), axis=0)
        comb = gl.where(active, comb / (col_sum[None, :] + HC_EPS), comb)
        for _ in gl.static_range(1, SINKHORN_ITERS):
            row_sum = gl.sum(gl.where(active, comb, 0.0), axis=1)
            comb = gl.where(active, comb / (row_sum[:, None] + HC_EPS), comb)
            col_sum = gl.sum(gl.where(active, comb, 0.0), axis=0)
            comb = gl.where(active, comb / (col_sum[None, :] + HC_EPS), comb)
        gl.store(comb_mix + token * 16 + comb_offsets, comb, mask=active)


def _mhc_pre_reduce_apply_block_h(
    num_tokens: int, hidden_size: int, n_splits: int
) -> int:
    if (num_tokens, hidden_size, n_splits) == (64, 4096, 64):
        return 1024
    return 512


def gluon_mhc_pre_reduce_apply_gfx950(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    layer_input: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Reduce mHC partials, apply the pre mix, and emit post coefficients."""
    if gemm_out_mul.shape != (n_splits, num_tokens, 24):
        raise ValueError("gemm_out_mul must have shape [n_splits, tokens, 24]")
    if gemm_out_sqrsum.shape != (n_splits, num_tokens):
        raise ValueError("gemm_out_sqrsum must have shape [n_splits, tokens]")
    if residual.shape != (num_tokens, 4, hidden_size):
        raise ValueError("residual must have shape [tokens, 4, hidden_size]")
    if layer_input.shape != (num_tokens, hidden_size):
        raise ValueError("layer_input must have shape [tokens, hidden_size]")
    expected_splits = hidden_size // 64
    if hidden_size not in (4096, 7168) or n_splits != expected_splits:
        raise ValueError(
            "GFX950 mHC specialization requires hidden_size=4096 or 7168 "
            "with the matching split-K decomposition"
        )
    if sinkhorn_iters != 20:
        raise ValueError("GFX950 mHC specialization requires 20 Sinkhorn iterations")
    if not 1 <= num_tokens <= 64:
        raise ValueError("GFX950 mHC specialization requires 1-64 tokens")

    block_h = _mhc_pre_reduce_apply_block_h(num_tokens, hidden_size, n_splits)
    _mhc_pre_reduce_apply_kernel[(num_tokens, hidden_size // block_h)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        layer_input,
        post_mix,
        comb_mix,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        SINKHORN_ITERS=sinkhorn_iters,
        N_SPLITS=n_splits,
        BLOCK_H=block_h,
        num_warps=1,
    )
