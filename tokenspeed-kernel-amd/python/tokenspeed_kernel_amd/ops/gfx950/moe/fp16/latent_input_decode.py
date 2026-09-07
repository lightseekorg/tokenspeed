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

"""Decode input projections for latent MoE models on gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_HIDDEN = gl.constexpr(7168)
_SHARED = gl.constexpr(768)
_ROUTER_BLOCK_N = gl.constexpr(4)
_LATENT_BLOCK_N = gl.constexpr(16)
_SHARED_BLOCK_N = gl.constexpr(4)
_BLOCK_K = gl.constexpr(1024)
_NUM_WARPS = gl.constexpr(4)
_LANES = gl.constexpr(64)
_ROUTER_GRID = gl.constexpr(896 // 4)
_LATENT_GRID = gl.constexpr(3584 // 16)
_TOTAL_GRID = 896 // 4 + 3584 // 16 + 768 // 4


@gluon.jit
def _latent_input_decode_kernel(
    hidden_ptr,
    router_weight_ptr,
    routed_weight_ptr,
    shared_weight_ptr,
    router_out_ptr,
    routed_out_ptr,
    shared_out_ptr,
    beta,
    linear_beta,
    HAS_LINEAR_BETA: gl.constexpr,
):
    """Compute independent router, routed-down, and shared-up regions."""

    pid = gl.program_id(0)
    if pid < _ROUTER_GRID:
        layout: gl.constexpr = gl.BlockedLayout(
            [1, _BLOCK_K // _LANES],
            [1, _LANES],
            [_NUM_WARPS, 1],
            [1, 0],
        )
        n_layout: gl.constexpr = gl.SliceLayout(1, layout)
        k_layout: gl.constexpr = gl.SliceLayout(0, layout)
        offs_n = pid * _ROUTER_BLOCK_N + gl.arange(0, _ROUTER_BLOCK_N, layout=n_layout)
        acc = gl.zeros([_ROUTER_BLOCK_N], gl.float32, n_layout)
        for k0 in range(0, _HIDDEN, _BLOCK_K):
            offs_k = k0 + gl.arange(0, _BLOCK_K, layout=k_layout)
            activation = gl.amd.cdna4.buffer_load(
                ptr=hidden_ptr,
                offsets=offs_k.to(gl.int32),
            ).to(gl.float32)
            weight = gl.amd.cdna4.buffer_load(
                ptr=router_weight_ptr,
                offsets=(
                    offs_n[:, None].to(gl.int64) * _HIDDEN
                    + offs_k[None, :].to(gl.int64)
                ).to(gl.int32),
            )
            activation = gl.convert_layout(activation[None, :], layout)
            acc += gl.sum(weight.to(gl.float32) * activation, axis=1)
        gl.store(router_out_ptr + offs_n, acc)
        return

    if pid < _ROUTER_GRID + _LATENT_GRID:
        pid_n = pid - _ROUTER_GRID
        layout: gl.constexpr = gl.BlockedLayout(
            [1, _BLOCK_K // _LANES],
            [1, _LANES],
            [_NUM_WARPS, 1],
            [1, 0],
        )
        n_layout: gl.constexpr = gl.SliceLayout(1, layout)
        k_layout: gl.constexpr = gl.SliceLayout(0, layout)
        offs_n = pid_n * _LATENT_BLOCK_N + gl.arange(
            0, _LATENT_BLOCK_N, layout=n_layout
        )
        acc = gl.zeros([_LATENT_BLOCK_N], gl.float32, n_layout)
        for k0 in range(0, _HIDDEN, _BLOCK_K):
            offs_k = k0 + gl.arange(0, _BLOCK_K, layout=k_layout)
            activation = gl.amd.cdna4.buffer_load(
                ptr=hidden_ptr,
                offsets=offs_k.to(gl.int32),
            ).to(gl.float32)
            weight = gl.amd.cdna4.buffer_load(
                ptr=routed_weight_ptr,
                offsets=(
                    offs_n[:, None].to(gl.int64) * _HIDDEN
                    + offs_k[None, :].to(gl.int64)
                ).to(gl.int32),
            )
            activation = gl.convert_layout(activation[None, :], layout)
            acc += gl.sum(weight.to(gl.float32) * activation, axis=1)
        gl.store(routed_out_ptr + offs_n, acc)
        return

    pid_n = pid - _ROUTER_GRID - _LATENT_GRID
    layout: gl.constexpr = gl.BlockedLayout(
        [1, _BLOCK_K // _LANES],
        [1, _LANES],
        [_NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_n = pid_n * _SHARED_BLOCK_N + gl.arange(0, _SHARED_BLOCK_N, layout=n_layout)
    gate_acc = gl.zeros([_SHARED_BLOCK_N], gl.float32, n_layout)
    up_acc = gl.zeros([_SHARED_BLOCK_N], gl.float32, n_layout)
    for k0 in range(0, _HIDDEN, _BLOCK_K):
        offs_k = k0 + gl.arange(0, _BLOCK_K, layout=k_layout)
        activation = gl.amd.cdna4.buffer_load(
            ptr=hidden_ptr,
            offsets=offs_k.to(gl.int32),
        ).to(gl.float32)
        gate_weight = gl.amd.cdna4.buffer_load(
            ptr=shared_weight_ptr,
            offsets=(
                offs_n[:, None].to(gl.int64) * _HIDDEN + offs_k[None, :].to(gl.int64)
            ).to(gl.int32),
        )
        up_weight = gl.amd.cdna4.buffer_load(
            ptr=shared_weight_ptr,
            offsets=(
                (_SHARED + offs_n[:, None]).to(gl.int64) * _HIDDEN
                + offs_k[None, :].to(gl.int64)
            ).to(gl.int32),
        )
        activation = gl.convert_layout(activation[None, :], layout)
        gate_acc += gl.sum(gate_weight.to(gl.float32) * activation, axis=1)
        up_acc += gl.sum(up_weight.to(gl.float32) * activation, axis=1)
    # Preserve both BF16 projection outputs before the FP32 SiTU epilogue.
    gate_raw = gate_acc.to(gl.bfloat16).to(gl.float32)
    up = up_acc.to(gl.bfloat16).to(gl.float32)
    gate = beta * gl.extra.libdevice.tanh(gate_raw / beta)
    gate *= 1.0 / (1.0 + gl.exp(-gate_raw))
    if HAS_LINEAR_BETA:
        up = linear_beta * gl.extra.libdevice.tanh(up / linear_beta)
    gl.store(shared_out_ptr + offs_n, gate * up)


def gluon_latent_input_decode_gfx950(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project router, routed-latent, and shared-expert inputs in one launch.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[1, 7168]``.
        router_weight: Contiguous BF16 router weight shaped ``[896, 7168]``.
        routed_down_weight: Contiguous BF16 latent weight shaped ``[3584, 7168]``.
        shared_gate_up_weight: Contiguous BF16 TP8 shared-expert weight shaped
            ``[1536, 7168]``.
        beta: Positive SiTU gate clipping scale.
        linear_beta: Optional positive SiTU linear-branch clipping scale.

    Returns:
        FP32 router logits ``[1, 896]``, BF16 routed latent ``[1, 3584]``,
        and BF16 activated shared-expert row ``[1, 768]``.
    """

    expected = (
        (hidden_states, (1, 7168), "hidden states"),
        (router_weight, (896, 7168), "router weight"),
        (routed_down_weight, (3584, 7168), "routed-down weight"),
        (shared_gate_up_weight, (1536, 7168), "shared gate/up weight"),
    )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Kimi K3 {name} must have shape {shape}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"Kimi K3 {name} must be BF16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"Kimi K3 {name} must be contiguous on GPU")
        if tensor.device != hidden_states.device:
            raise ValueError("Kimi K3 MoE input tensors must be colocated")
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("Kimi K3 SiTU beta values must be positive")
    router_out = torch.empty((1, 896), dtype=torch.float32, device=hidden_states.device)
    routed_out = torch.empty(
        (1, 3584), dtype=torch.bfloat16, device=hidden_states.device
    )
    shared_out = torch.empty(
        (1, 768), dtype=torch.bfloat16, device=hidden_states.device
    )
    _latent_input_decode_kernel[(_TOTAL_GRID,)](
        hidden_states,
        router_weight,
        routed_down_weight,
        shared_gate_up_weight,
        router_out,
        routed_out,
        shared_out,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        HAS_LINEAR_BETA=linear_beta is not None,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
    )
    return router_out, routed_out, shared_out


__all__ = ["gluon_latent_input_decode_gfx950"]
