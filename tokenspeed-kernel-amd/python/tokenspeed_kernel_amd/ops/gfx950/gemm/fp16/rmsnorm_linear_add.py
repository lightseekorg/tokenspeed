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

"""Fused K3 RMSNorm, latent projection, and residual additions for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LATENT = gl.constexpr(3584)
_BLOCK_N = gl.constexpr(32)
_BLOCK_K = gl.constexpr(512)
_NUM_WARPS = gl.constexpr(8)
_LANES = gl.constexpr(64)


@gluon.jit
def _rmsnorm_linear_add_kernel(
    latent_ptr,
    norm_weight_ptr,
    projection_weight_ptr,
    residual_ptr,
    shared_ptr,
    output_ptr,
    latent_stride_m,
    residual_stride_m,
    shared_stride_m,
    output_stride_m,
    eps,
    HAS_RESIDUAL: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, _BLOCK_K // _LANES],
        [1, _LANES],
        [_NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_n = pid_n * _BLOCK_N + gl.arange(0, _BLOCK_N, layout=n_layout)

    square_sum = gl.full((), 0.0, gl.float32)
    for k0 in range(0, _LATENT, _BLOCK_K):
        offs_k = k0 + gl.arange(0, _BLOCK_K, layout=k_layout)
        k_mask = offs_k < _LATENT
        latent = gl.amd.cdna4.buffer_load(
            latent_ptr,
            (pid_m * latent_stride_m + offs_k).to(gl.int32),
            mask=k_mask,
            other=0.0,
        ).to(gl.float32)
        square_sum += gl.sum(latent * latent, axis=0)
    inverse_rms = gl.rsqrt(square_sum / _LATENT + eps)

    acc = gl.zeros([_BLOCK_N], gl.float32, n_layout)
    for k0 in range(0, _LATENT, _BLOCK_K):
        offs_k = k0 + gl.arange(0, _BLOCK_K, layout=k_layout)
        k_mask = offs_k < _LATENT
        latent = gl.amd.cdna4.buffer_load(
            latent_ptr,
            (pid_m * latent_stride_m + offs_k).to(gl.int32),
            mask=k_mask,
            other=0.0,
        ).to(gl.float32)
        norm_weight = gl.amd.cdna4.buffer_load(
            norm_weight_ptr,
            offs_k.to(gl.int32),
            mask=k_mask,
            other=0.0,
        ).to(gl.float32)
        normalized = (latent * inverse_rms * norm_weight).to(gl.bfloat16)
        projection_weight = gl.amd.cdna4.buffer_load(
            projection_weight_ptr,
            (offs_n[:, None].to(gl.int64) * _LATENT + offs_k[None, :].to(gl.int64)).to(
                gl.int32
            ),
            mask=k_mask[None, :],
            other=0.0,
            cache=".cg",
        )
        normalized = gl.convert_layout(normalized[None, :], layout)
        acc += gl.sum(
            projection_weight.to(gl.float32) * normalized.to(gl.float32),
            axis=1,
        )

    # Match the materialized BF16 projection before the residual additions.
    acc = acc.to(gl.bfloat16).to(gl.float32)
    residual_offset = pid_m * residual_stride_m + offs_n
    shared_offset = pid_m * shared_stride_m + offs_n
    output_offset = pid_m * output_stride_m + offs_n
    if HAS_RESIDUAL:
        acc += gl.amd.cdna4.buffer_load(residual_ptr, residual_offset.to(gl.int32)).to(
            gl.float32
        )
    acc += gl.amd.cdna4.buffer_load(shared_ptr, shared_offset.to(gl.int32)).to(
        gl.float32
    )
    gl.store(output_ptr + output_offset, acc)


def gluon_rmsnorm_linear_add_gfx950(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    residual: torch.Tensor | None,
    shared: torch.Tensor,
    *,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(shared)
    m = latent.shape[0]
    residual_ptr = shared if residual is None else residual
    _rmsnorm_linear_add_kernel[(m, projection_weight.shape[0] // _BLOCK_N)](
        latent,
        norm_weight,
        projection_weight,
        residual_ptr,
        shared,
        out,
        latent.stride(0),
        0 if residual is None else residual.stride(0),
        shared.stride(0),
        out.stride(0),
        float(eps),
        HAS_RESIDUAL=residual is not None,
        num_warps=8,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_rmsnorm_linear_add_gfx950"]
