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

"""Route-direct gfx950 A16W4 SiTU MoE kernels for small-batch decode.

The Triton MXFP4 weight preprocessor stores values as `(E, K / 2, N)` and
CDNA4-swizzles their `(E, K / 32, N)` scales.  Decode can consume those
existing buffers directly: one wave reduces K for a small output-column block,
without sorting routes or padding each one to a 64-row grouped GEMM. gfx950's
native scaled upcast expands each E2M1/UE8M0 tile directly to exact BF16 weight
values, avoiding scalar nibble and exponent decoding in both GEMVs.

Stage 1 writes one BF16 SiTU row per local route.  Stage 2 visits the original
top-k slots, skips remote EP ids (`-1`), preserves the per-route BF16 W2
boundary, and combines all local routes directly into the rank's output.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.a4w4.decode_kernels import (
    _cdna4_swizzled_mxfp4_scale_offset,
)

_LANES = gl.constexpr(64)
# Kimi K3's packed K dimensions exactly fit these maximum tiles. Smaller
# supported shapes step down to the largest exact 128-byte multiple below.
WARP_DECODE_STAGE1_BLOCK_N = 4
WARP_DECODE_STAGE1_BLOCK_KB = 256
WARP_DECODE_STAGE1_NUM_WARPS = 4
WARP_DECODE_STAGE2_BLOCK_N = 8
WARP_DECODE_STAGE2_BLOCK_KB = 512
WARP_DECODE_STAGE2_NUM_WARPS = 8
_MIN_WARP_DECODE_BLOCK_KB = 128


def _largest_exact_block_kb(packed_k: int, max_block_kb: int) -> int:
    """Select the widest power-of-two K tile that does not need masking."""

    block_kb = max_block_kb
    while block_kb > _MIN_WARP_DECODE_BLOCK_KB and packed_k % block_kb:
        block_kb //= 2
    return block_kb


@gluon.jit
def _stage1_a16w4_situ_warp_gemv(
    x_ptr,
    w13_ptr,
    w13_scale_ptr,
    inter_ptr,
    local_ids_ptr,
    hidden_dim,
    intermediate_dim,
    top_k: gl.constexpr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_se,
    stride_slin,
    stride_snb,
    stride_im,
    stride_in,
    stride_idm,
    stride_ids,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    LINEAR_WEIGHTS: gl.constexpr,
    W13_INTERLEAVED: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    route = pid // NUM_PID_N
    pid_n = pid % NUM_PID_N
    token = route // top_k
    slot = route % top_k
    expert = (
        gl.load(local_ids_ptr + token * stride_idm + slot * stride_ids) - EXPERT_START
    )
    if expert < 0:
        return
    if expert >= NUM_LOCAL_EXPERTS:
        return

    # Warps span output neurons while each wave's lanes reduce packed K bytes.
    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_KB // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, (2 * BLOCK_KB) // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    expanded_offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=expanded_n_layout)

    packed_k = hidden_dim // 2
    x_row = token.to(gl.int64) * stride_xm
    w_expert = expert.to(gl.int64) * stride_we
    scale_expert = expert.to(gl.int64) * stride_se
    if LINEAR_WEIGHTS and not W13_INTERLEAVED:
        gate_col = offs_n
        up_col = intermediate_dim + offs_n
        expanded_gate_col = expanded_offs_n
        expanded_up_col = intermediate_dim + expanded_offs_n
    else:
        # Triton's SiTU preprocessor and the optional linear interleaved layout
        # store adjacent gate/up output columns.
        gate_col = 2 * offs_n
        up_col = gate_col + 1
        expanded_gate_col = 2 * expanded_offs_n
        expanded_up_col = expanded_gate_col + 1
    gate_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)
    up_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)

    for kb0 in range(0, packed_k, BLOCK_KB):
        offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=k_layout)
        expanded_k = 2 * kb0 + gl.arange(0, 2 * BLOCK_KB, layout=expanded_k_layout)
        x = gl.amd.cdna4.buffer_load(
            ptr=x_ptr,
            offsets=(x_row + expanded_k * stride_xk).to(gl.int32),
        ).to(gl.float32)
        gate_w_offsets = (
            w_expert
            + offs_kb[None, :].to(gl.int64) * stride_wk
            + gate_col[:, None].to(gl.int64) * stride_wn
        )
        up_w_offsets = (
            w_expert
            + offs_kb[None, :].to(gl.int64) * stride_wk
            + up_col[:, None].to(gl.int64) * stride_wn
        )
        if LINEAR_WEIGHTS:
            gate_scale_offsets = (
                scale_expert
                + expanded_gate_col[:, None].to(gl.int64) * stride_slin
                + (expanded_k[None, :] // 32).to(gl.int64) * stride_snb
            )
            up_scale_offsets = (
                scale_expert
                + expanded_up_col[:, None].to(gl.int64) * stride_slin
                + (expanded_k[None, :] // 32).to(gl.int64) * stride_snb
            )
        else:
            gate_scale_offsets = _cdna4_swizzled_mxfp4_scale_offset(
                scale_expert,
                expanded_gate_col[:, None],
                expanded_k[None, :] // 32,
                stride_slin,
                stride_snb,
            )
            up_scale_offsets = _cdna4_swizzled_mxfp4_scale_offset(
                scale_expert,
                expanded_up_col[:, None],
                expanded_k[None, :] // 32,
                stride_slin,
                stride_snb,
            )
        gate_packed = gl.amd.cdna4.buffer_load(
            ptr=w13_ptr,
            offsets=gate_w_offsets.to(gl.int32),
        )
        up_packed = gl.amd.cdna4.buffer_load(
            ptr=w13_ptr,
            offsets=up_w_offsets.to(gl.int32),
        )
        gate_w = gl.amd.cdna4.scaled_upcast(
            gate_packed,
            gl.amd.cdna4.buffer_load(
                ptr=w13_scale_ptr,
                offsets=gate_scale_offsets.to(gl.int32),
            ),
            gl.bfloat16,
            axis=1,
        )
        up_w = gl.amd.cdna4.scaled_upcast(
            up_packed,
            gl.amd.cdna4.buffer_load(
                ptr=w13_scale_ptr,
                offsets=up_scale_offsets.to(gl.int32),
            ),
            gl.bfloat16,
            axis=1,
        )
        x_tile = gl.convert_layout(x[None, :], expanded_layout)
        gate_acc += gl.sum(gate_w.to(gl.float32) * x_tile, axis=1)
        up_acc += gl.sum(up_w.to(gl.float32) * x_tile, axis=1)

    # Match Kimi's BF16 W13 output before evaluating SiTU in FP32.
    gate = gate_acc.to(gl.bfloat16).to(gl.float32)
    up = up_acc.to(gl.bfloat16).to(gl.float32)
    gate = (
        SITU_BETA
        * gl.extra.libdevice.tanh(gate / SITU_BETA)
        * (1.0 / (1.0 + gl.exp(-gate)))
    )
    if HAS_LINEAR_BETA:
        up = SITU_LINEAR_BETA * gl.extra.libdevice.tanh(up / SITU_LINEAR_BETA)
    activated = (gate * up).to(inter_ptr.dtype.element_ty)
    gl.store(
        inter_ptr + route.to(gl.int64) * stride_im + expanded_offs_n * stride_in,
        activated,
    )


@gluon.jit
def _stage2_a16w4_warp_gemv_combine(
    inter_ptr,
    w2_ptr,
    w2_scale_ptr,
    out_ptr,
    local_ids_ptr,
    topk_weights_ptr,
    hidden_dim,
    intermediate_dim,
    stride_ipm,
    stride_ipk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_se,
    stride_slin,
    stride_snb,
    stride_om,
    stride_on,
    stride_idm,
    stride_ids,
    stride_twm,
    stride_tws,
    TOP_K: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    LINEAR_WEIGHTS: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    token = pid // NUM_PID_N
    pid_n = pid % NUM_PID_N
    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_KB // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, (2 * BLOCK_KB) // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    expanded_offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=expanded_n_layout)
    packed_k = intermediate_dim // 2
    acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)

    for slot in gl.static_range(0, TOP_K):
        expert = (
            gl.load(local_ids_ptr + token * stride_idm + slot * stride_ids)
            - EXPERT_START
        )
        if (expert >= 0) & (expert < NUM_LOCAL_EXPERTS):
            route_weight = gl.load(
                topk_weights_ptr + token * stride_twm + slot * stride_tws
            ).to(gl.float32)
            inter_row = (token * TOP_K + slot).to(gl.int64) * stride_ipm
            w_expert = expert.to(gl.int64) * stride_we
            scale_expert = expert.to(gl.int64) * stride_se
            route_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)
            for kb0 in range(0, packed_k, BLOCK_KB):
                offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=k_layout)
                expanded_k = 2 * kb0 + gl.arange(
                    0, 2 * BLOCK_KB, layout=expanded_k_layout
                )
                inter = gl.amd.cdna4.buffer_load(
                    ptr=inter_ptr,
                    offsets=(inter_row + expanded_k * stride_ipk).to(gl.int32),
                ).to(gl.float32)
                w_offsets = (
                    w_expert
                    + offs_kb[None, :].to(gl.int64) * stride_wk
                    + offs_n[:, None].to(gl.int64) * stride_wn
                )
                if LINEAR_WEIGHTS:
                    scale_offsets = (
                        scale_expert
                        + expanded_offs_n[:, None].to(gl.int64) * stride_slin
                        + (expanded_k[None, :] // 32).to(gl.int64) * stride_snb
                    )
                else:
                    scale_offsets = _cdna4_swizzled_mxfp4_scale_offset(
                        scale_expert,
                        expanded_offs_n[:, None],
                        expanded_k[None, :] // 32,
                        stride_slin,
                        stride_snb,
                    )
                packed = gl.amd.cdna4.buffer_load(
                    ptr=w2_ptr,
                    offsets=w_offsets.to(gl.int32),
                )
                weight = gl.amd.cdna4.scaled_upcast(
                    packed,
                    gl.amd.cdna4.buffer_load(
                        ptr=w2_scale_ptr,
                        offsets=scale_offsets.to(gl.int32),
                    ),
                    gl.bfloat16,
                    axis=1,
                )
                inter_tile = gl.convert_layout(inter[None, :], expanded_layout)
                route_acc += gl.sum(weight.to(gl.float32) * inter_tile, axis=1)
            # Match the reference's BF16 W2 result before route weighting.
            acc += route_weight * route_acc.to(gl.bfloat16).to(gl.float32)

    gl.store(
        out_ptr + token * stride_om + expanded_offs_n * stride_on,
        acc.to(out_ptr.dtype.element_ty),
    )


def gluon_a16w4_situ_warp_decode_ep_gfx950(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    local_topk_ids: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    expert_start: int = 0,
    linear_weights: bool = False,
    w13_interleaved: bool = False,
) -> torch.Tensor:
    """Compute a tiny-M EP contribution from packed MXFP4 weights.

    `linear_weights=False` consumes Triton's K-packed values and
    CDNA4-swizzled scales. `linear_weights=True` consumes the Gluon EP8
    plan's original contiguous `[E, N, K / 2]` values and linear scales.
    Kimi K3's exact packed-K sizes use the widest tuned tiles; other supported
    widths retain unmasked execution by stepping down to an exact tile.
    """
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("gfx950 warp decode requires rank-2 BF16 activations")
    if not hidden_states.is_contiguous():
        raise ValueError("gfx950 warp decode requires contiguous hidden states")
    if topk_weights.shape != local_topk_ids.shape or local_topk_ids.ndim != 2:
        raise ValueError("top-k weights and local ids must have the same rank-2 shape")
    if local_topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("top-k token count must match hidden states")
    if situ_beta <= 0.0:
        raise ValueError("SiTU beta must be positive")
    if situ_linear_beta is not None and situ_linear_beta <= 0.0:
        raise ValueError("SiTU linear beta must be positive")
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")

    num_tokens, hidden_dim = hidden_states.shape
    top_k = int(local_topk_ids.shape[1])
    if linear_weights:
        num_experts, two_intermediate, packed_hidden = w13_weight.shape
        intermediate_dim = two_intermediate // 2
        if two_intermediate % 2 or packed_hidden * 2 != hidden_dim:
            raise ValueError("linear W13 shape is inconsistent with activations")
        if tuple(w13_scale.shape) != (
            num_experts,
            two_intermediate,
            hidden_dim // 32,
        ):
            raise ValueError("linear W13 scale shape mismatch")
        if tuple(w2_weight.shape) != (
            num_experts,
            hidden_dim,
            intermediate_dim // 2,
        ):
            raise ValueError("linear W2 shape mismatch")
        if tuple(w2_scale.shape) != (
            num_experts,
            hidden_dim,
            intermediate_dim // 32,
        ):
            raise ValueError("linear W2 scale shape mismatch")
        if any(
            not tensor.is_contiguous()
            for tensor in (w13_weight, w13_scale, w2_weight, w2_scale)
        ):
            raise ValueError("linear MXFP4 weights and scales must be contiguous")
        w13_stride_k = w13_weight.stride(2)
        w13_stride_n = w13_weight.stride(1)
        w2_stride_k = w2_weight.stride(2)
        w2_stride_n = w2_weight.stride(1)
    else:
        num_experts, packed_hidden, two_intermediate = w13_weight.shape
        intermediate_dim = two_intermediate // 2
        if two_intermediate % 2 or packed_hidden * 2 != hidden_dim:
            raise ValueError("preprocessed W13 shape is inconsistent with activations")
        if tuple(w13_scale.shape) != (
            num_experts,
            hidden_dim,
            two_intermediate // 32,
        ):
            raise ValueError("preprocessed W13 scale shape mismatch")
        if tuple(w2_weight.shape) != (
            num_experts,
            intermediate_dim // 2,
            hidden_dim,
        ):
            raise ValueError("preprocessed W2 shape mismatch")
        if tuple(w2_scale.shape) != (
            num_experts,
            intermediate_dim,
            hidden_dim // 32,
        ):
            raise ValueError("preprocessed W2 scale shape mismatch")
        w13_stride_k = w13_weight.stride(1)
        w13_stride_n = w13_weight.stride(2)
        w2_stride_k = w2_weight.stride(1)
        w2_stride_n = w2_weight.stride(2)
    if hidden_dim % 256 or intermediate_dim % 256:
        raise ValueError(
            "gfx950 warp decode requires hidden and intermediate dimensions "
            "divisible by 256"
        )

    local_topk_ids = local_topk_ids.to(torch.int32)
    inter = torch.empty(
        (num_tokens * top_k, intermediate_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    stage1_block_n = WARP_DECODE_STAGE1_BLOCK_N
    stage1_block_kb = _largest_exact_block_kb(
        packed_hidden,
        WARP_DECODE_STAGE1_BLOCK_KB,
    )
    stage1_warps = WARP_DECODE_STAGE1_NUM_WARPS
    if intermediate_dim % stage1_block_n or packed_hidden % stage1_block_kb:
        raise ValueError("unmasked stage1 requires exact N and packed-K tiles")
    stage1_grid = num_tokens * top_k * triton.cdiv(intermediate_dim, stage1_block_n)
    _stage1_a16w4_situ_warp_gemv[(stage1_grid,)](
        hidden_states,
        w13_weight,
        w13_scale,
        inter,
        local_topk_ids,
        hidden_dim,
        intermediate_dim,
        top_k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w13_weight.stride(0),
        w13_stride_k,
        w13_stride_n,
        w13_scale.stride(0),
        w13_scale.stride(1),
        w13_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        local_topk_ids.stride(0),
        local_topk_ids.stride(1),
        SITU_BETA=float(situ_beta),
        SITU_LINEAR_BETA=(1.0 if situ_linear_beta is None else float(situ_linear_beta)),
        HAS_LINEAR_BETA=situ_linear_beta is not None,
        EXPERT_START=int(expert_start),
        NUM_LOCAL_EXPERTS=num_experts,
        LINEAR_WEIGHTS=linear_weights,
        W13_INTERLEAVED=(w13_interleaved if linear_weights else True),
        NUM_PID_N=intermediate_dim // stage1_block_n,
        BLOCK_N=stage1_block_n,
        BLOCK_KB=stage1_block_kb,
        NUM_WARPS=stage1_warps,
        num_warps=stage1_warps,
    )

    out = torch.empty_like(hidden_states)
    stage2_block_n = WARP_DECODE_STAGE2_BLOCK_N
    packed_intermediate = intermediate_dim // 2
    stage2_block_kb = _largest_exact_block_kb(
        packed_intermediate,
        WARP_DECODE_STAGE2_BLOCK_KB,
    )
    stage2_warps = WARP_DECODE_STAGE2_NUM_WARPS
    if hidden_dim % stage2_block_n or packed_intermediate % stage2_block_kb:
        raise ValueError("unmasked stage2 requires exact N and packed-K tiles")
    stage2_grid = num_tokens * triton.cdiv(hidden_dim, stage2_block_n)
    _stage2_a16w4_warp_gemv_combine[(stage2_grid,)](
        inter,
        w2_weight,
        w2_scale,
        out,
        local_topk_ids,
        topk_weights,
        hidden_dim,
        intermediate_dim,
        inter.stride(0),
        inter.stride(1),
        w2_weight.stride(0),
        w2_stride_k,
        w2_stride_n,
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        out.stride(0),
        out.stride(1),
        local_topk_ids.stride(0),
        local_topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        TOP_K=top_k,
        EXPERT_START=int(expert_start),
        NUM_LOCAL_EXPERTS=num_experts,
        LINEAR_WEIGHTS=linear_weights,
        NUM_PID_N=hidden_dim // stage2_block_n,
        BLOCK_N=stage2_block_n,
        BLOCK_KB=stage2_block_kb,
        NUM_WARPS=stage2_warps,
        num_warps=stage2_warps,
    )
    return out


__all__ = ["gluon_a16w4_situ_warp_decode_ep_gfx950"]
