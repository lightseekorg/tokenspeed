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

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton


@triton.jit
def _kimi3_projection_gemv_kernel(
    a,
    weight,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Bandwidth-oriented small-M kernel.

    Each program owns one input row and a small group of output rows.  The
    activation tile is reused across that group while K is reduced in FP32.
    Unlike an MFMA GEMM tile this does not execute fifteen padded rows for a
    single-token decode.
    """

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    weight_row = weight + offs_n[:, None] * K

    if EVEN:
        for k_start in range(0, K, BLOCK_K):
            k_offsets = tl.max_contiguous(
                tl.multiple_of(k_start + offs_k, BLOCK_K), BLOCK_K
            )
            activation = tl.load(a + pid_m * K + k_offsets)
            weight_tile = tl.load(weight_row + k_offsets[None, :], cache_modifier=".cg")
            accumulator += tl.sum(
                weight_tile.to(tl.float32) * activation[None, :], axis=1
            )
    else:
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + offs_k
            k_mask = k_offsets < K
            activation = tl.load(a + pid_m * K + k_offsets, mask=k_mask, other=0.0)
            weight_tile = tl.load(
                weight_row + k_offsets[None, :],
                mask=(offs_n[:, None] < N) & k_mask[None, :],
                other=0.0,
            )
            accumulator += tl.sum(
                weight_tile.to(tl.float32) * activation[None, :], axis=1
            )

    output_mask = None if EVEN else ((pid_m < M) & (offs_n < N))
    tl.store(output + pid_m * N + offs_n, accumulator, mask=output_mask)


@triton.jit
def _kimi3_shared_situ_projection_gemv_kernel(
    hidden_states,
    gate_up_weight,
    output,
    beta,
    linear_beta,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """Fuse K3's local shared-expert gate/up GEMV and SiTU activation."""

    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    gate_row = gate_up_weight + offs_n[:, None] * K
    up_row = gate_up_weight + (N + offs_n[:, None]) * K

    for k_start in range(0, K, BLOCK_K):
        k_offsets = tl.max_contiguous(
            tl.multiple_of(k_start + offs_k, BLOCK_K), BLOCK_K
        )
        activation = tl.load(hidden_states + k_offsets)
        gate_weight = tl.load(gate_row + k_offsets[None, :], cache_modifier=".cg")
        up_weight = tl.load(up_row + k_offsets[None, :], cache_modifier=".cg")
        gate_acc += tl.sum(gate_weight.to(tl.float32) * activation[None, :], axis=1)
        up_acc += tl.sum(up_weight.to(tl.float32) * activation[None, :], axis=1)

    # Preserve the materialized BF16 projection boundary used by the unfused
    # MergedColumnParallelLinear -> SituAndMul path.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    gate = beta * libdevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * libdevice.tanh(up / linear_beta)
    tl.store(output + offs_n, gate * up)


def kimi3_projection_gemv(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
    config: tuple[int, int, int, int],
) -> torch.Tensor:
    m, k = hidden_states.shape
    n = weight.shape[0]
    if out is None:
        out = hidden_states.new_empty((m, n))
    block_n, block_k, num_warps, waves_per_eu = config
    _kimi3_projection_gemv_kernel[(triton.cdiv(n, block_n), m)](
        hidden_states,
        weight,
        out,
        m,
        N=n,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN=(n % block_n == 0) and (k % block_k == 0),
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=waves_per_eu,
    )
    return out


def kimi3_shared_situ_projection_gemv(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    out: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    k = hidden_states.shape[1]
    n = gate_up_weight.shape[0] // 2
    block_n, block_k, num_warps = 4, 1024, 4
    _kimi3_shared_situ_projection_gemv_kernel[(triton.cdiv(n, block_n),)](
        hidden_states,
        gate_up_weight,
        out,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        K=k,
        N=n,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_LINEAR_BETA=linear_beta is not None,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=1,
    )
    return out
