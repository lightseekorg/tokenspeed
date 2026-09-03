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

"""Route-direct gfx950 A16W4 gated MoE kernels for small-batch decode.

The Triton MXFP4 weight preprocessor stores values as ``(E, K / 2, N)`` and
CDNA4-swizzles their ``(E, K / 32, N)`` scales.  Decode can consume those
existing buffers directly: one wave reduces K for a small output-column block,
without sorting routes or padding each one to a 64-row grouped GEMM. gfx950's
native scaled upcast expands each E2M1/UE8M0 tile directly to exact BF16 weight
values, avoiding scalar nibble and exponent decoding in both GEMVs.

Kimi K3's 1792-byte packed hidden dimension uses a masked 1024-byte W13 tile.
The 14.3% padded tail is cheaper than five additional loop iterations on
gfx950; masked values and scales are zero-filled before the reduction.

Stage 1 writes one BF16 activated row per local route. Stage 2 visits the original
top-k slots, skips remote EP ids (``-1``), preserves the per-route BF16 W2
boundary, and combines all local routes directly into the rank's output.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _cdna4_swizzled_mxfp4_scale_offset,
    _gluon_dot_preshuffled_w_offset,
)

_LANES = gl.constexpr(64)
# Kimi K3 W13 uses a deliberately masked wide-K tile. W2 and other supported
# shapes retain exact tiles selected by ``_largest_exact_block_kb`` below.
WARP_DECODE_STAGE1_BLOCK_N = 8
WARP_DECODE_STAGE1_BLOCK_KB = 1024
WARP_DECODE_STAGE1_NUM_WARPS = 4
WARP_DECODE_STAGE2_BLOCK_N = 8
WARP_DECODE_STAGE2_BATCHED_BLOCK_N = 4
WARP_DECODE_STAGE2_BLOCK_KB = 512
WARP_DECODE_STAGE2_NUM_WARPS = 8
WARP_DECODE_STAGE2_BATCHED_NUM_WARPS = 4
WARP_DECODE_TP_STAGE1_BLOCK_N = 2
WARP_DECODE_TP_STAGE1_NUM_WARPS = 1
WARP_DECODE_TP_STAGE2_BLOCK_N = 8
WARP_DECODE_TP_STAGE2_BLOCK_KB = 256
WARP_DECODE_TP_STAGE2_NUM_WARPS = 1
WARP_DECODE_TP_TOPK_PER_CTA = 1
_MIN_WARP_DECODE_BLOCK_KB = 128
_ROUTE_DIRECT_DECODE_MAX_TOKENS = 16
_BUFFER_LOAD_OFFSET_LIMIT = 1 << 31
_KIMI3_SHARED_K = gl.constexpr(768)
_KIMI3_SHARED_BLOCK_K = gl.constexpr(512)
_GDOT_DECODE_BLOCK_KB = gl.constexpr(128)
_GDOT_STAGE1_BLOCK_N = gl.constexpr(16)
_GDOT_STAGE1_NUM_WARPS = gl.constexpr(4)
_GDOT_STAGE2_BLOCK_N = gl.constexpr(16)
_GDOT_STAGE2_NUM_WARPS = gl.constexpr(4)


def _largest_exact_block_kb(packed_k: int, max_block_kb: int) -> int:
    """Select the widest power-of-two K tile that does not need masking."""

    block_kb = max_block_kb
    while block_kb > _MIN_WARP_DECODE_BLOCK_KB and packed_k % block_kb:
        block_kb //= 2
    return block_kb


def _supports_a16w4_warp_decode_ep_gfx950(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    linear_weights: bool = True,
) -> bool:
    """Return whether route-direct decode supports this weight representation."""
    if hidden_states.ndim != 2 or any(
        tensor.ndim != 3 for tensor in (w13_weight, w13_scale, w2_weight, w2_scale)
    ):
        return False

    hidden = int(hidden_states.shape[1])
    if linear_weights:
        num_experts, two_intermediate, packed_hidden = w13_weight.shape
        intermediate = int(two_intermediate) // 2
        shapes_match = (
            int(two_intermediate) % 2 == 0
            and int(packed_hidden) * 2 == hidden
            and tuple(w13_scale.shape)
            == (int(num_experts), int(two_intermediate), hidden // 32)
            and tuple(w2_weight.shape) == (int(num_experts), hidden, intermediate // 2)
            and tuple(w2_scale.shape) == (int(num_experts), hidden, intermediate // 32)
        )
    else:
        gdot_weights = all(
            bool(getattr(weight, "is_shuffled_for_gluon_dot", False))
            and int(getattr(weight, "gluon_dot_block_k_pk", 0)) == 128
            and int(getattr(weight, "gluon_dot_block_n", 0)) == 128
            for weight in (w13_weight, w2_weight)
        )
        num_experts, packed_hidden_phys, two_intermediate = w13_weight.shape
        packed_hidden = int(getattr(w13_weight, "original_k_pk", packed_hidden_phys))
        intermediate = int(two_intermediate) // 2
        w2_packed_phys = int(w2_weight.shape[1])
        w2_packed = int(getattr(w2_weight, "original_k_pk", w2_packed_phys))
        w2_output = int(getattr(w2_weight, "original_n", w2_weight.shape[2]))
        shapes_match = (
            gdot_weights
            and int(two_intermediate) % 2 == 0
            and packed_hidden * 2 == hidden
            and tuple(w13_scale.shape)
            == (int(num_experts), hidden, int(two_intermediate) // 32)
            and int(w2_weight.shape[0]) == int(num_experts)
            and w2_packed * 2 == intermediate
            and w2_output == hidden
            and tuple(w2_scale.shape)
            == (int(num_experts), w2_packed_phys * 2, hidden // 32)
        )

    offsets_fit = all(
        tensor.numel() * tensor.element_size() < _BUFFER_LOAD_OFFSET_LIMIT
        for tensor in (w13_weight, w13_scale, w2_weight, w2_scale)
    )
    return (
        0 < hidden_states.shape[0] <= _ROUTE_DIRECT_DECODE_MAX_TOKENS
        and hidden_states.is_contiguous()
        and offsets_fit
        and hidden % 256 == 0
        and intermediate % 256 == 0
        and shapes_match
    )


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
    SWIGLU_LIMIT: gl.constexpr,
    HAS_SWIGLU_LIMIT: gl.constexpr,
    USE_SWIGLU: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    LINEAR_WEIGHTS: gl.constexpr,
    W13_INTERLEAVED: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    MASK_K_TAIL: gl.constexpr,
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
        if MASK_K_TAIL:
            packed_k_valid = offs_kb < packed_k
            expanded_k_valid = expanded_k < hidden_dim
        else:
            packed_k_valid = gl.full([BLOCK_KB], True, gl.int1, layout=k_layout)
            expanded_k_valid = gl.full(
                [2 * BLOCK_KB], True, gl.int1, layout=expanded_k_layout
            )
        x = gl.amd.cdna4.buffer_load(
            ptr=x_ptr,
            offsets=(x_row + expanded_k * stride_xk).to(gl.int32),
            mask=expanded_k_valid,
            other=0.0,
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
            mask=packed_k_valid[None, :],
            other=0,
        )
        up_packed = gl.amd.cdna4.buffer_load(
            ptr=w13_ptr,
            offsets=up_w_offsets.to(gl.int32),
            mask=packed_k_valid[None, :],
            other=0,
        )
        gate_w = gl.amd.cdna4.scaled_upcast(
            gate_packed,
            gl.amd.cdna4.buffer_load(
                ptr=w13_scale_ptr,
                offsets=gate_scale_offsets.to(gl.int32),
                mask=expanded_k_valid[None, :],
                other=0,
            ),
            gl.bfloat16,
            axis=1,
        )
        up_w = gl.amd.cdna4.scaled_upcast(
            up_packed,
            gl.amd.cdna4.buffer_load(
                ptr=w13_scale_ptr,
                offsets=up_scale_offsets.to(gl.int32),
                mask=expanded_k_valid[None, :],
                other=0,
            ),
            gl.bfloat16,
            axis=1,
        )
        x_tile = gl.convert_layout(x[None, :], expanded_layout)
        gate_acc += gl.sum(gate_w.to(gl.float32) * x_tile, axis=1)
        up_acc += gl.sum(up_w.to(gl.float32) * x_tile, axis=1)

    # Preserve the BF16 W13 boundary before evaluating the activation in FP32.
    gate = gate_acc.to(gl.bfloat16).to(gl.float32)
    up = up_acc.to(gl.bfloat16).to(gl.float32)
    if USE_SWIGLU:
        if HAS_SWIGLU_LIMIT:
            gate = gl.minimum(gate, SWIGLU_LIMIT)
            up = gl.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = gate * (1.0 / (1.0 + gl.exp(-gate)))
    else:
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
def _stage1_a16w4_situ_gdot_gemv(
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
    SWIGLU_LIMIT: gl.constexpr,
    HAS_SWIGLU_LIMIT: gl.constexpr,
    USE_SWIGLU: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    N_PHYS: gl.constexpr,
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

    # Keep each lane's K values within the gdot128 layout's 16-byte subtiles.
    packed_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N // (4 * NUM_WARPS), BLOCK_KB // 16],
        [4, 16],
        [NUM_WARPS, 1],
        [1, 0],
    )
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N // (4 * NUM_WARPS), (2 * BLOCK_KB) // 16],
        [4, 16],
        [NUM_WARPS, 1],
        [1, 0],
    )
    packed_n_layout: gl.constexpr = gl.SliceLayout(1, packed_layout)
    packed_k_layout: gl.constexpr = gl.SliceLayout(0, packed_layout)
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=packed_n_layout)
    expanded_offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=expanded_n_layout)

    packed_k = hidden_dim // 2
    x_row = token.to(gl.int64) * stride_xm
    w_expert = expert.to(gl.int64) * stride_we
    scale_expert = expert.to(gl.int64) * stride_se
    acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)

    for kb0 in range(0, packed_k, BLOCK_KB):
        offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=packed_k_layout)
        expanded_k = 2 * kb0 + gl.arange(0, 2 * BLOCK_KB, layout=expanded_k_layout)
        packed_k_valid = offs_kb < packed_k
        expanded_k_valid = expanded_k < hidden_dim
        x = gl.amd.cdna4.buffer_load(
            ptr=x_ptr,
            offsets=(x_row + expanded_k * stride_xk).to(gl.int32),
            mask=expanded_k_valid,
            other=0.0,
        ).to(gl.float32)
        packed = gl.amd.cdna4.buffer_load(
            ptr=w13_ptr,
            offsets=_gluon_dot_preshuffled_w_offset(
                w_expert,
                offs_kb[None, :],
                offs_n[:, None],
                N_PHYS,
            ).to(gl.int32),
            mask=packed_k_valid[None, :],
            other=0,
        )
        scale = gl.amd.cdna4.buffer_load(
            ptr=w13_scale_ptr,
            offsets=_cdna4_swizzled_mxfp4_scale_offset(
                scale_expert,
                expanded_offs_n[:, None],
                expanded_k[None, :] // 32,
                stride_slin,
                stride_snb,
            ).to(gl.int32),
            mask=expanded_k_valid[None, :],
            other=0,
        )
        weight = gl.amd.cdna4.scaled_upcast(
            packed,
            scale,
            gl.bfloat16,
            axis=1,
        )
        x_tile = gl.convert_layout(x[None, :], expanded_layout)
        acc += gl.sum(weight.to(gl.float32) * x_tile, axis=1)

    # W13 is preshuffled with adjacent gate/up columns. Preserve the BF16 GEMM
    # boundary before applying K3's SiTU activation in FP32.
    split_layout: gl.constexpr = gl.BlockedLayout(
        [2],
        [64],
        [NUM_WARPS],
        [0],
    )
    gate_up = gl.convert_layout(
        acc.to(gl.bfloat16).to(gl.float32), split_layout
    ).reshape((BLOCK_N // 2, 2))
    gate, up = gl.split(gate_up)
    if USE_SWIGLU:
        if HAS_SWIGLU_LIMIT:
            gate = gl.minimum(gate, SWIGLU_LIMIT)
            up = gl.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = gate * (1.0 / (1.0 + gl.exp(-gate)))
    else:
        gate = (
            SITU_BETA
            * gl.extra.libdevice.tanh(gate / SITU_BETA)
            * (1.0 / (1.0 + gl.exp(-gate)))
        )
        if HAS_LINEAR_BETA:
            up = SITU_LINEAR_BETA * gl.extra.libdevice.tanh(up / SITU_LINEAR_BETA)
    activated = (gate * up).reshape((BLOCK_N // 2,))
    out_n = pid_n * (BLOCK_N // 2) + gl.arange(
        0, BLOCK_N // 2, layout=activated.type.layout
    )
    gl.store(
        inter_ptr + route.to(gl.int64) * stride_im + out_n * stride_in,
        activated.to(inter_ptr.dtype.element_ty),
        mask=out_n < intermediate_dim,
    )


@gluon.jit
def _stage2_a16w4_warp_gemv_combine(
    inter_ptr,
    w2_ptr,
    w2_scale_ptr,
    out_ptr,
    local_ids_ptr,
    topk_weights_ptr,
    shared_input_ptr,
    shared_weight_ptr,
    shared_out_ptr,
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
    stride_sim,
    stride_sik,
    stride_som,
    stride_son,
    FUSE_SHARED_DOWN: gl.constexpr,
    NUM_ROUTED_PROGRAMS: gl.constexpr,
    TOP_K: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    LINEAR_WEIGHTS: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    NUM_TOPK_GROUPS: gl.constexpr,
    NUM_SHARED_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    MASK_K_TAIL: gl.constexpr,
):
    pid = gl.program_id(0)
    if FUSE_SHARED_DOWN and pid >= NUM_ROUTED_PROGRAMS:
        shared_pid = pid - NUM_ROUTED_PROGRAMS
        shared_token = shared_pid // NUM_SHARED_PID_N
        shared_pid_n = shared_pid % NUM_SHARED_PID_N
        shared_layout: gl.constexpr = gl.BlockedLayout(
            [1, _KIMI3_SHARED_BLOCK_K // _LANES],
            [1, _LANES],
            [NUM_WARPS, 1],
            [1, 0],
        )
        shared_n_layout: gl.constexpr = gl.SliceLayout(1, shared_layout)
        shared_k_layout: gl.constexpr = gl.SliceLayout(0, shared_layout)
        shared_offs_n = shared_pid_n * BLOCK_N + gl.arange(
            0, BLOCK_N, layout=shared_n_layout
        )
        shared_acc = gl.zeros([BLOCK_N], gl.float32, shared_n_layout)
        for k0 in range(0, _KIMI3_SHARED_K, _KIMI3_SHARED_BLOCK_K):
            shared_offs_k = k0 + gl.arange(
                0, _KIMI3_SHARED_BLOCK_K, layout=shared_k_layout
            )
            shared_k_valid = shared_offs_k < _KIMI3_SHARED_K
            shared_input = gl.amd.cdna4.buffer_load(
                ptr=shared_input_ptr,
                offsets=(shared_token * stride_sim + shared_offs_k * stride_sik).to(
                    gl.int32
                ),
                mask=shared_k_valid,
                other=0.0,
            ).to(gl.float32)
            shared_weight = gl.amd.cdna4.buffer_load(
                ptr=shared_weight_ptr,
                offsets=(
                    shared_offs_n[:, None].to(gl.int64) * _KIMI3_SHARED_K
                    + shared_offs_k[None, :].to(gl.int64)
                ).to(gl.int32),
                mask=shared_k_valid[None, :],
                other=0.0,
            )
            shared_input = gl.convert_layout(
                shared_input[None, :],
                shared_layout,
            )
            shared_acc += gl.sum(
                shared_weight.to(gl.float32) * shared_input,
                axis=1,
            )
        gl.store(
            shared_out_ptr + shared_token * stride_som + shared_offs_n * stride_son,
            shared_acc.to(shared_out_ptr.dtype.element_ty),
        )
        return

    token = pid // (NUM_TOPK_GROUPS * NUM_PID_N)
    token_pid = pid % (NUM_TOPK_GROUPS * NUM_PID_N)
    topk_group = token_pid // NUM_PID_N
    pid_n = token_pid % NUM_PID_N
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

    for group_slot in gl.static_range(
        0, (TOP_K + NUM_TOPK_GROUPS - 1) // NUM_TOPK_GROUPS
    ):
        slot = (
            topk_group * ((TOP_K + NUM_TOPK_GROUPS - 1) // NUM_TOPK_GROUPS) + group_slot
        )
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
                if MASK_K_TAIL:
                    packed_k_valid = offs_kb < packed_k
                    expanded_k_valid = expanded_k < intermediate_dim
                else:
                    packed_k_valid = gl.full([BLOCK_KB], True, gl.int1, layout=k_layout)
                    expanded_k_valid = gl.full(
                        [2 * BLOCK_KB], True, gl.int1, layout=expanded_k_layout
                    )
                inter = gl.amd.cdna4.buffer_load(
                    ptr=inter_ptr,
                    offsets=(inter_row + expanded_k * stride_ipk).to(gl.int32),
                    mask=expanded_k_valid,
                    other=0.0,
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
                    mask=packed_k_valid[None, :],
                    other=0,
                )
                weight = gl.amd.cdna4.scaled_upcast(
                    packed,
                    gl.amd.cdna4.buffer_load(
                        ptr=w2_scale_ptr,
                        offsets=scale_offsets.to(gl.int32),
                        mask=expanded_k_valid[None, :],
                        other=0,
                    ),
                    gl.bfloat16,
                    axis=1,
                )
                inter_tile = gl.convert_layout(inter[None, :], expanded_layout)
                route_acc += gl.sum(weight.to(gl.float32) * inter_tile, axis=1)
            # Match the reference's BF16 W2 result before route weighting.
            acc += route_weight * route_acc.to(gl.bfloat16).to(gl.float32)

    out_row = token * NUM_TOPK_GROUPS + topk_group
    gl.store(
        out_ptr + out_row * stride_om + expanded_offs_n * stride_on,
        acc.to(out_ptr.dtype.element_ty),
    )


@gluon.jit
def _stage2_a16w4_gdot_gemv_combine(
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
    NUM_TOPK_GROUPS: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    N_PHYS: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    token = pid // (NUM_TOPK_GROUPS * NUM_PID_N)
    token_pid = pid % (NUM_TOPK_GROUPS * NUM_PID_N)
    topk_group = token_pid // NUM_PID_N
    pid_n = token_pid % NUM_PID_N
    packed_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N // (4 * NUM_WARPS), BLOCK_KB // 16],
        [4, 16],
        [NUM_WARPS, 1],
        [1, 0],
    )
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_N // (4 * NUM_WARPS), (2 * BLOCK_KB) // 16],
        [4, 16],
        [NUM_WARPS, 1],
        [1, 0],
    )
    packed_n_layout: gl.constexpr = gl.SliceLayout(1, packed_layout)
    packed_k_layout: gl.constexpr = gl.SliceLayout(0, packed_layout)
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=packed_n_layout)
    expanded_offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=expanded_n_layout)
    packed_k = intermediate_dim // 2
    acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)

    for group_slot in gl.static_range(
        0, (TOP_K + NUM_TOPK_GROUPS - 1) // NUM_TOPK_GROUPS
    ):
        slot = (
            topk_group * ((TOP_K + NUM_TOPK_GROUPS - 1) // NUM_TOPK_GROUPS) + group_slot
        )
        expert = (
            gl.load(local_ids_ptr + token * stride_idm + slot * stride_ids)
            - EXPERT_START
        )
        if (expert >= 0) & (expert < NUM_LOCAL_EXPERTS):
            inter_row = (token * TOP_K + slot).to(gl.int64) * stride_ipm
            w_expert = expert.to(gl.int64) * stride_we
            scale_expert = expert.to(gl.int64) * stride_se
            route_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)
            for kb0 in range(0, packed_k, BLOCK_KB):
                offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=packed_k_layout)
                expanded_k = 2 * kb0 + gl.arange(
                    0, 2 * BLOCK_KB, layout=expanded_k_layout
                )
                packed_k_valid = offs_kb < packed_k
                expanded_k_valid = expanded_k < intermediate_dim
                inter = gl.amd.cdna4.buffer_load(
                    ptr=inter_ptr,
                    offsets=(inter_row + expanded_k * stride_ipk).to(gl.int32),
                    mask=expanded_k_valid,
                    other=0.0,
                ).to(gl.float32)
                packed = gl.amd.cdna4.buffer_load(
                    ptr=w2_ptr,
                    offsets=_gluon_dot_preshuffled_w_offset(
                        w_expert,
                        offs_kb[None, :],
                        offs_n[:, None],
                        N_PHYS,
                    ).to(gl.int32),
                    mask=packed_k_valid[None, :],
                    other=0,
                )
                scale = gl.amd.cdna4.buffer_load(
                    ptr=w2_scale_ptr,
                    offsets=_cdna4_swizzled_mxfp4_scale_offset(
                        scale_expert,
                        expanded_offs_n[:, None],
                        expanded_k[None, :] // 32,
                        stride_slin,
                        stride_snb,
                    ).to(gl.int32),
                    mask=expanded_k_valid[None, :],
                    other=0,
                )
                weight = gl.amd.cdna4.scaled_upcast(
                    packed,
                    scale,
                    gl.bfloat16,
                    axis=1,
                )
                inter_tile = gl.convert_layout(inter[None, :], expanded_layout)
                route_acc += gl.sum(weight.to(gl.float32) * inter_tile, axis=1)

            # Match the A16 reference boundary: round the W2 result to BF16,
            # then apply the FP32 routing weight before combining routes.
            partial = route_acc.to(gl.bfloat16).to(gl.float32)
            route_weight = gl.load(
                topk_weights_ptr + token * stride_twm + slot * stride_tws
            ).to(gl.float32)
            acc += partial * route_weight

    out_row = token * NUM_TOPK_GROUPS + topk_group
    gl.store(
        out_ptr + out_row * stride_om + expanded_offs_n * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=expanded_offs_n < hidden_dim,
    )


@gluon.jit
def _reduce_topk_groups(
    partial_ptr,
    out_ptr,
    hidden_dim,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    NUM_GROUPS: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(hidden_dim, BLOCK_N)
    token = pid // num_pid_n
    pid_n = pid % num_pid_n
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    acc = gl.zeros([BLOCK_N], gl.float32, layout=layout)
    for group in gl.static_range(0, NUM_GROUPS):
        acc += gl.load(
            partial_ptr + (token * NUM_GROUPS + group) * stride_pm + offs_n * stride_pn,
            mask=offs_n < hidden_dim,
            other=0.0,
        ).to(gl.float32)
    gl.store(
        out_ptr + token * stride_om + offs_n * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=offs_n < hidden_dim,
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
    activation: str = "situ",
    swiglu_limit: float | None = None,
    expert_start: int = 0,
    linear_weights: bool = False,
    w13_interleaved: bool = False,
    shared_input: torch.Tensor | None = None,
    shared_weight: torch.Tensor | None = None,
    routed_out: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute a tiny-M EP contribution from packed MXFP4 weights.

    ``linear_weights=False`` consumes K-packed values and CDNA4-swizzled
    scales, including the gdot128 layout used by the prefill kernels.
    ``linear_weights=True`` consumes the Gluon EP8 plan's original contiguous
    ``[E, N, K / 2]`` values and linear scales.
    Kimi K3's W13 packed-K size uses a tuned masked tail; W2 and other
    supported widths retain unmasked execution by stepping down to an exact
    tile. ``activation="swiglu"`` selects standard alpha=1, beta=0 SwiGLU
    with an optional symmetric clamp on the gate/up inputs.
    """
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("gfx950 warp decode requires rank-2 BF16 activations")
    if not hidden_states.is_contiguous():
        raise ValueError("gfx950 warp decode requires contiguous hidden states")
    if topk_weights.shape != local_topk_ids.shape or local_topk_ids.ndim != 2:
        raise ValueError("top-k weights and local ids must have the same rank-2 shape")
    if local_topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("top-k token count must match hidden states")
    if activation not in {"situ", "swiglu"}:
        raise ValueError(f"unsupported A16W4 activation: {activation}")
    if activation == "situ":
        if situ_beta <= 0.0:
            raise ValueError("SiTU beta must be positive")
        if situ_linear_beta is not None and situ_linear_beta <= 0.0:
            raise ValueError("SiTU linear beta must be positive")
    elif swiglu_limit is not None and swiglu_limit <= 0.0:
        raise ValueError("SwiGLU limit must be positive when set")
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    fuse_shared_down = shared_input is not None or shared_weight is not None
    if fuse_shared_down:
        if shared_input is None or shared_weight is None:
            raise ValueError(
                "Kimi K3 shared input and weight must be provided together"
            )
        if tuple(shared_input.shape) != (hidden_states.shape[0], 768) or tuple(
            shared_weight.shape
        ) != (7168, 768):
            raise ValueError(
                "Kimi K3 shared down fusion requires [M, 768] input and "
                "[7168, 768] weight"
            )
        if (
            shared_input.dtype != torch.bfloat16
            or shared_weight.dtype != torch.bfloat16
        ):
            raise TypeError("Kimi K3 shared down fusion requires BF16 tensors")
        if (
            not shared_input.is_cuda
            or not shared_weight.is_cuda
            or not shared_input.is_contiguous()
            or not shared_weight.is_contiguous()
            or shared_input.device != hidden_states.device
            or shared_weight.device != hidden_states.device
        ):
            raise ValueError(
                "Kimi K3 shared down tensors must be contiguous and colocated on GPU"
            )
    num_tokens, hidden_dim = hidden_states.shape
    top_k = int(local_topk_ids.shape[1])
    gdot_weights = not linear_weights and all(
        bool(getattr(weight, "is_shuffled_for_gluon_dot", False))
        and int(getattr(weight, "gluon_dot_block_k_pk", 0)) == 128
        and int(getattr(weight, "gluon_dot_block_n", 0)) == 128
        for weight in (w13_weight, w2_weight)
    )
    if gdot_weights and (fuse_shared_down or shared_out is not None):
        raise ValueError("gdot A16W4 decode does not support shared down fusion")
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
        num_experts, packed_hidden_phys, two_intermediate = w13_weight.shape
        packed_hidden = int(getattr(w13_weight, "original_k_pk", packed_hidden_phys))
        intermediate_dim = two_intermediate // 2
        if two_intermediate % 2 or packed_hidden * 2 != hidden_dim:
            raise ValueError("preprocessed W13 shape is inconsistent with activations")
        if tuple(w13_scale.shape) != (
            num_experts,
            hidden_dim,
            two_intermediate // 32,
        ):
            raise ValueError("preprocessed W13 scale shape mismatch")
        w2_packed_phys = int(w2_weight.shape[1])
        w2_packed = int(getattr(w2_weight, "original_k_pk", w2_packed_phys))
        w2_output_phys = int(w2_weight.shape[2])
        w2_output = int(getattr(w2_weight, "original_n", w2_output_phys))
        if (
            int(w2_weight.shape[0]) != num_experts
            or w2_packed * 2 != intermediate_dim
            or w2_output != hidden_dim
        ):
            raise ValueError("preprocessed W2 shape mismatch")
        # The scale swizzle pads logical K to the gdot weight's physical K tile.
        if tuple(w2_scale.shape) != (
            num_experts,
            w2_packed_phys * 2,
            hidden_dim // 32,
        ):
            raise ValueError("preprocessed W2 scale shape mismatch")
        w13_stride_k = w13_weight.stride(1)
        w13_stride_n = w13_weight.stride(2)
        w2_stride_k = w2_weight.stride(1)
        w2_stride_n = w2_weight.stride(2)
    if hidden_dim % 256 or intermediate_dim % 128:
        raise ValueError(
            "gfx950 warp decode requires hidden dimensions divisible by 256 "
            "and intermediate dimensions divisible by 128"
        )

    local_topk_ids = local_topk_ids.to(torch.int32)
    inter = torch.empty(
        (num_tokens * top_k, intermediate_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if gdot_weights:
        stage1_grid = (
            num_tokens * top_k * triton.cdiv(2 * intermediate_dim, _GDOT_STAGE1_BLOCK_N)
        )
        _stage1_a16w4_situ_gdot_gemv[(stage1_grid,)](
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
            w13_scale.stride(0),
            w13_scale.stride(1),
            w13_scale.stride(2),
            inter.stride(0),
            inter.stride(1),
            local_topk_ids.stride(0),
            local_topk_ids.stride(1),
            SITU_BETA=float(situ_beta),
            SITU_LINEAR_BETA=(
                1.0 if situ_linear_beta is None else float(situ_linear_beta)
            ),
            HAS_LINEAR_BETA=situ_linear_beta is not None,
            SWIGLU_LIMIT=(1.0 if swiglu_limit is None else float(swiglu_limit)),
            HAS_SWIGLU_LIMIT=swiglu_limit is not None,
            USE_SWIGLU=activation == "swiglu",
            EXPERT_START=int(expert_start),
            NUM_LOCAL_EXPERTS=num_experts,
            N_PHYS=two_intermediate,
            NUM_PID_N=triton.cdiv(2 * intermediate_dim, _GDOT_STAGE1_BLOCK_N),
            BLOCK_N=_GDOT_STAGE1_BLOCK_N,
            BLOCK_KB=_GDOT_DECODE_BLOCK_KB,
            NUM_WARPS=_GDOT_STAGE1_NUM_WARPS,
            num_warps=_GDOT_STAGE1_NUM_WARPS,
        )
    else:
        tp_local = num_experts == 896 and intermediate_dim == 384 and top_k == 16
        stage1_block_n = (
            WARP_DECODE_TP_STAGE1_BLOCK_N if tp_local else WARP_DECODE_STAGE1_BLOCK_N
        )
        stage1_block_kb = WARP_DECODE_STAGE1_BLOCK_KB
        stage1_warps = (
            WARP_DECODE_TP_STAGE1_NUM_WARPS
            if tp_local
            else WARP_DECODE_STAGE1_NUM_WARPS
        )
        if (
            intermediate_dim % stage1_block_n
            or stage1_block_kb < _MIN_WARP_DECODE_BLOCK_KB
            or stage1_block_kb % 64
        ):
            raise ValueError(
                "stage1 requires exact N tiles and a lane-aligned packed-K tile"
            )
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
            SITU_LINEAR_BETA=(
                1.0 if situ_linear_beta is None else float(situ_linear_beta)
            ),
            HAS_LINEAR_BETA=situ_linear_beta is not None,
            SWIGLU_LIMIT=(1.0 if swiglu_limit is None else float(swiglu_limit)),
            HAS_SWIGLU_LIMIT=swiglu_limit is not None,
            USE_SWIGLU=activation == "swiglu",
            EXPERT_START=int(expert_start),
            NUM_LOCAL_EXPERTS=num_experts,
            LINEAR_WEIGHTS=linear_weights,
            W13_INTERLEAVED=(w13_interleaved if linear_weights else True),
            NUM_PID_N=intermediate_dim // stage1_block_n,
            BLOCK_N=stage1_block_n,
            BLOCK_KB=stage1_block_kb,
            NUM_WARPS=stage1_warps,
            MASK_K_TAIL=packed_hidden % stage1_block_kb != 0,
            num_warps=stage1_warps,
        )

    out = torch.empty_like(hidden_states) if routed_out is None else routed_out
    if out.shape != hidden_states.shape or out.dtype != hidden_states.dtype:
        raise ValueError("routed output must match the hidden-state shape and dtype")
    if (
        out.stride(1) != 1
        or out.stride(0) < hidden_dim
        or out.device != hidden_states.device
    ):
        raise ValueError("routed output must be row-major and colocated")
    if gdot_weights:
        gdot_topk_groups = top_k
        stage2_out = torch.empty(
            (num_tokens * gdot_topk_groups, hidden_dim),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        stage2_grid = (
            num_tokens
            * gdot_topk_groups
            * triton.cdiv(hidden_dim, _GDOT_STAGE2_BLOCK_N)
        )
        _stage2_a16w4_gdot_gemv_combine[(stage2_grid,)](
            inter,
            w2_weight,
            w2_scale,
            stage2_out,
            local_topk_ids,
            topk_weights,
            hidden_dim,
            intermediate_dim,
            inter.stride(0),
            inter.stride(1),
            w2_weight.stride(0),
            w2_scale.stride(0),
            w2_scale.stride(1),
            w2_scale.stride(2),
            stage2_out.stride(0),
            stage2_out.stride(1),
            local_topk_ids.stride(0),
            local_topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            TOP_K=top_k,
            NUM_TOPK_GROUPS=gdot_topk_groups,
            EXPERT_START=int(expert_start),
            NUM_LOCAL_EXPERTS=num_experts,
            N_PHYS=w2_output_phys,
            NUM_PID_N=triton.cdiv(hidden_dim, _GDOT_STAGE2_BLOCK_N),
            BLOCK_N=_GDOT_STAGE2_BLOCK_N,
            BLOCK_KB=_GDOT_DECODE_BLOCK_KB,
            NUM_WARPS=_GDOT_STAGE2_NUM_WARPS,
            num_warps=_GDOT_STAGE2_NUM_WARPS,
        )
        _reduce_topk_groups[(num_tokens * triton.cdiv(hidden_dim, 256),)](
            stage2_out,
            out,
            hidden_dim,
            stage2_out.stride(0),
            stage2_out.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_GROUPS=gdot_topk_groups,
            BLOCK_N=256,
            num_warps=1,
        )
        return out

    tp_local = num_experts == 896 and intermediate_dim == 384 and top_k == 16
    # Finer output tiles keep work balanced when local route counts differ.
    stage2_block_n = (
        WARP_DECODE_TP_STAGE2_BLOCK_N
        if tp_local
        else (
            WARP_DECODE_STAGE2_BATCHED_BLOCK_N
            if num_tokens in (2, 4)
            else WARP_DECODE_STAGE2_BLOCK_N
        )
    )
    packed_intermediate = intermediate_dim // 2
    stage2_block_kb = (
        WARP_DECODE_TP_STAGE2_BLOCK_KB
        if tp_local
        else _largest_exact_block_kb(
            packed_intermediate,
            WARP_DECODE_STAGE2_BLOCK_KB,
        )
    )
    # The M=1 joint path remains faster with eight waves.
    stage2_warps = (
        WARP_DECODE_TP_STAGE2_NUM_WARPS
        if tp_local
        else (
            WARP_DECODE_STAGE2_BATCHED_NUM_WARPS
            if num_tokens in (2, 4)
            else WARP_DECODE_STAGE2_NUM_WARPS
        )
    )
    if hidden_dim % stage2_block_n:
        raise ValueError("stage2 requires exact output-column tiles")
    num_topk_groups = triton.cdiv(top_k, WARP_DECODE_TP_TOPK_PER_CTA) if tp_local else 1
    stage2_grid = num_tokens * num_topk_groups * triton.cdiv(hidden_dim, stage2_block_n)
    stage2_out = out
    if num_topk_groups > 1:
        stage2_out = torch.empty(
            (num_tokens * num_topk_groups, hidden_dim),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    if fuse_shared_down:
        if shared_out is None:
            shared_out = torch.empty(
                (num_tokens, 7168),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        if (
            shared_out.shape != (num_tokens, 7168)
            or shared_out.dtype != torch.bfloat16
            or not shared_out.is_contiguous()
            or shared_out.device != hidden_states.device
        ):
            raise ValueError("shared output must be contiguous BF16 [M, 7168]")
    else:
        if shared_out is not None:
            raise ValueError("shared output requires fused shared down")
        shared_out = out
    total_stage2_grid = stage2_grid + (
        num_tokens * triton.cdiv(7168, stage2_block_n) if fuse_shared_down else 0
    )
    _stage2_a16w4_warp_gemv_combine[(total_stage2_grid,)](
        inter,
        w2_weight,
        w2_scale,
        stage2_out,
        local_topk_ids,
        topk_weights,
        out if shared_input is None else shared_input,
        out if shared_weight is None else shared_weight,
        shared_out,
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
        stage2_out.stride(0),
        stage2_out.stride(1),
        local_topk_ids.stride(0),
        local_topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        shared_input.stride(0) if shared_input is not None else out.stride(0),
        shared_input.stride(1) if shared_input is not None else out.stride(1),
        shared_out.stride(0),
        shared_out.stride(1),
        FUSE_SHARED_DOWN=fuse_shared_down,
        NUM_ROUTED_PROGRAMS=stage2_grid,
        TOP_K=top_k,
        EXPERT_START=int(expert_start),
        NUM_LOCAL_EXPERTS=num_experts,
        LINEAR_WEIGHTS=linear_weights,
        NUM_PID_N=hidden_dim // stage2_block_n,
        NUM_TOPK_GROUPS=num_topk_groups,
        NUM_SHARED_PID_N=7168 // stage2_block_n,
        BLOCK_N=stage2_block_n,
        BLOCK_KB=stage2_block_kb,
        NUM_WARPS=stage2_warps,
        MASK_K_TAIL=packed_intermediate % stage2_block_kb != 0,
        num_warps=stage2_warps,
    )
    if num_topk_groups > 1:
        _reduce_topk_groups[(num_tokens * triton.cdiv(hidden_dim, 256),)](
            stage2_out,
            out,
            hidden_dim,
            stage2_out.stride(0),
            stage2_out.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_GROUPS=num_topk_groups,
            BLOCK_N=256,
            num_warps=1,
        )
    if fuse_shared_down:
        return out, shared_out
    return out


gluon_a16w4_warp_decode_ep_gfx950 = gluon_a16w4_situ_warp_decode_ep_gfx950


__all__ = [
    "gluon_a16w4_situ_warp_decode_ep_gfx950",
    "gluon_a16w4_warp_decode_ep_gfx950",
]
