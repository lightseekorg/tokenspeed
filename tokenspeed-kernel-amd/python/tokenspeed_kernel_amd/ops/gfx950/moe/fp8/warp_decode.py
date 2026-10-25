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

"""Direct block-FP8 warp-reduce GEMV MoE for gfx950 decode.

The compact checkpoint stores each expert as E4M3 values with one floating
point inverse scale per 128x128 weight block. Decode is route-sparse and each
expert normally receives very few rows, so an MFMA grouped GEMM wastes most of
its M tile. These kernels instead assign reduction lanes to K and dequantize
the FP8 values after their global load. This preserves the compact HBM traffic,
avoids routing/sort scratch, and fuses the routed-weight combine into stage 2.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

_LANES = gl.constexpr(64)
_SCALE_BLOCK = gl.constexpr(128)


@gluon.jit
def _packed_weight_offsets(rows, columns, input_size):
    offsets = (
        (((rows // 16) * (input_size // 32) + columns // 32) * 2 + (columns % 32) // 16)
        * 16
        + rows % 16
    ) * 16 + columns % 16
    return gl.multiple_of(gl.max_contiguous(offsets, (1, 1, 16)), (1, 1, 16))


@gluon.jit
def _packed_stage1(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    ids_ptr,
    expert_start,
    num_experts,
    D,
    I,
    top_k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_se,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    SWIGLU_LIMIT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    blocks_n = gl.cdiv(I, BLOCK_N)
    slot, block_n = pid // blocks_n, pid % blocks_n
    token, route = slot // top_k, slot % top_k
    expert = gl.load(ids_ptr + token * stride_tit + route * stride_tis) - expert_start
    if (expert < 0) | (expert >= num_experts):
        return

    # Axes are wave, output row, and the wave's natural-order K segment.
    # Keeping the wave axis in partial sums prevents cross-wave communication
    # inside the K loop without retaining a full tile of FP32 accumulators.
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [1, 16, 4], [NUM_WARPS, 1, 1], [1, 2, 0]
    )
    partial_layout: gl.constexpr = gl.SliceLayout(2, layout)
    n_layout: gl.constexpr = gl.SliceLayout(0, partial_layout)
    wk_layout: gl.constexpr = gl.SliceLayout(1, layout)
    wave_k_layout: gl.constexpr = gl.SliceLayout(1, wk_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, wk_layout)
    wave_scale_layout: gl.constexpr = gl.SliceLayout(1, partial_layout)
    wave_k = gl.arange(0, NUM_WARPS, layout=wave_k_layout)
    wave_scale = gl.arange(0, NUM_WARPS, layout=wave_scale_layout)
    local_k: gl.constexpr = BLOCK_K // NUM_WARPS
    within_k = gl.arange(0, local_k, layout=k_layout)
    rows = block_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    row_mask = rows < I
    x_row = x_ptr + token.to(gl.int64) * stride_xm
    expert_weight = weight_ptr + expert.to(gl.int64) * stride_we
    expert_scale = scale_ptr + expert.to(gl.int64) * stride_se
    gate_scale_row = (block_n * BLOCK_N) // 128
    up_scale_row = (I + block_n * BLOCK_N) // 128
    gate_partial = gl.zeros([NUM_WARPS, BLOCK_N], gl.float32, partial_layout)
    up_partial = gl.zeros([NUM_WARPS, BLOCK_N], gl.float32, partial_layout)
    for k0 in range(0, D, BLOCK_K):
        columns = k0 + wave_k[:, None] * local_k + within_k[None, :]
        k_mask = columns < D
        activation = gl.load(x_row + columns * stride_xk, mask=k_mask, other=0.0).to(
            gl.float32
        )
        # K128 per wave: one quantization block, independently of the CTA K.
        scale_start = k0 + wave_scale * local_k
        scale_k = (scale_start // 128).to(gl.int64) * stride_sk
        gate_scale = gl.load(
            expert_scale + gate_scale_row * stride_sn + scale_k,
            mask=scale_start < D,
            other=0.0,
        )
        up_scale = gl.load(
            expert_scale + up_scale_row * stride_sn + scale_k,
            mask=scale_start < D,
            other=0.0,
        )
        mask = row_mask[None, :, None] & k_mask[:, None, :]
        gate = gl.load(
            expert_weight
            + _packed_weight_offsets(rows[None, :, None], columns[:, None, :], D),
            mask=mask,
            other=0.0,
        ).to(gl.float32)
        up = gl.load(
            expert_weight
            + _packed_weight_offsets((I + rows)[None, :, None], columns[:, None, :], D),
            mask=mask,
            other=0.0,
        ).to(gl.float32)
        gate_partial += gl.sum(
            gate * gate_scale[:, None, None] * activation[:, None, :], axis=2
        )
        up_partial += gl.sum(
            up * up_scale[:, None, None] * activation[:, None, :], axis=2
        )
    gate = gl.sum(gate_partial, axis=0)
    up = gl.sum(up_partial, axis=0)
    if SWIGLU_LIMIT > 0.0:
        gate = gl.minimum(gate, SWIGLU_LIMIT)
        up = gl.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    value = gate * (1.0 / (1.0 + gl.exp(-gate))) * up
    gl.store(
        out_ptr + slot.to(gl.int64) * stride_om + rows * stride_on,
        value.to(out_ptr.dtype.element_ty),
        mask=row_mask,
    )


@gluon.jit
def _packed_stage2(
    inter_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    ids_ptr,
    probabilities_ptr,
    expert_start,
    num_experts,
    D,
    I,
    top_k,
    stride_im,
    stride_ik,
    stride_we,
    stride_se,
    stride_sd,
    stride_sk,
    stride_om,
    stride_od,
    stride_tit,
    stride_tis,
    stride_twt,
    stride_tws,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    blocks_d = gl.cdiv(D, BLOCK_D)
    token, block_d = pid // blocks_d, pid % blocks_d
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [1, 16, 4], [NUM_WARPS, 1, 1], [1, 2, 0]
    )
    partial_layout: gl.constexpr = gl.SliceLayout(2, layout)
    d_layout: gl.constexpr = gl.SliceLayout(0, partial_layout)
    wk_layout: gl.constexpr = gl.SliceLayout(1, layout)
    wave_k_layout: gl.constexpr = gl.SliceLayout(1, wk_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, wk_layout)
    wave_scale_layout: gl.constexpr = gl.SliceLayout(1, partial_layout)
    wave_k = gl.arange(0, NUM_WARPS, layout=wave_k_layout)
    wave_scale = gl.arange(0, NUM_WARPS, layout=wave_scale_layout)
    local_k: gl.constexpr = BLOCK_K // NUM_WARPS
    within_k = gl.arange(0, local_k, layout=k_layout)
    rows = block_d * BLOCK_D + gl.arange(0, BLOCK_D, layout=d_layout)
    row_mask = rows < D
    scale_d = (block_d * BLOCK_D) // 128
    acc = gl.zeros([BLOCK_D], gl.float32, d_layout)
    for slot in range(0, top_k):
        expert = (
            gl.load(ids_ptr + token * stride_tit + slot * stride_tis) - expert_start
        )
        valid = (expert >= 0) & (expert < num_experts)
        probability = gl.load(
            probabilities_ptr + token * stride_twt + slot * stride_tws
        ).to(gl.float32)
        probability = gl.where(valid, probability, 0.0)
        inter_row = inter_ptr + (token * top_k + slot).to(gl.int64) * stride_im
        expert_weight = weight_ptr + expert.to(gl.int64) * stride_we
        expert_scale = scale_ptr + expert.to(gl.int64) * stride_se + scale_d * stride_sd
        partial = gl.zeros([NUM_WARPS, BLOCK_D], gl.float32, partial_layout)
        for k0 in range(0, I, BLOCK_K):
            columns = k0 + wave_k[:, None] * local_k + within_k[None, :]
            k_mask = columns < I
            activation = gl.load(
                inter_row + columns * stride_ik, mask=k_mask & valid, other=0.0
            ).to(gl.float32)
            # Each wave owns K64, so adjacent waves share a K128 scale.
            scale_start = k0 + wave_scale * local_k
            scale = gl.load(
                expert_scale + (scale_start // 128).to(gl.int64) * stride_sk,
                mask=(scale_start < I) & valid,
                other=0.0,
            )
            weight = gl.load(
                expert_weight
                + _packed_weight_offsets(rows[None, :, None], columns[:, None, :], I),
                mask=row_mask[None, :, None] & k_mask[:, None, :] & valid,
                other=0.0,
            ).to(gl.float32)
            partial += gl.sum(
                weight * scale[:, None, None] * activation[:, None, :], axis=2
            )
        # Reduce before weighting each route, retaining the original route order.
        dot = gl.sum(partial, axis=0)
        acc += probability * dot
    gl.store(
        out_ptr + token * stride_om + rows * stride_od,
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask,
    )


@gluon.jit
def _block_fp8_dequantize_kernel(
    weight_ptr,
    scale_ptr,
    output_ptr,
    numel,
    n_size,
    k_size,
    scale_n,
    scale_k,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK // (_LANES * NUM_WARPS)],
        [_LANES],
        [NUM_WARPS],
        [0],
    )
    offsets = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offsets < numel
    k_offsets = offsets % k_size
    n_offsets = (offsets // k_size) % n_size
    expert_offsets = offsets // (n_size * k_size)
    scale_offsets = (
        expert_offsets * scale_n * scale_k
        + (n_offsets // _SCALE_BLOCK) * scale_k
        + k_offsets // _SCALE_BLOCK
    )
    values = gl.load(weight_ptr + offsets, mask=mask, other=0.0).to(gl.float32)
    scales = gl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(gl.float32)
    gl.store(output_ptr + offsets, values * scales, mask=mask)


@gluon.jit
def _stage1_fp8_warp_gemv(
    x_ptr,
    w1_ptr,
    w1_scale_ptr,
    out_ptr,
    topk_ids_ptr,
    expert_start,
    num_experts,
    D,
    I,
    top_k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    PACKED_WEIGHTS: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    if PACKED_WEIGHTS:
        _packed_stage1(
            x_ptr,
            w1_ptr,
            w1_scale_ptr,
            out_ptr,
            topk_ids_ptr,
            expert_start,
            num_experts,
            D,
            I,
            top_k,
            stride_xm,
            stride_xk,
            stride_we,
            stride_se,
            stride_sn,
            stride_sk,
            stride_om,
            stride_on,
            stride_tit,
            stride_tis,
            SWIGLU_LIMIT,
            BLOCK_N,
            BLOCK_K,
            NUM_WARPS,
        )
        return
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(I, BLOCK_N)
    slot = pid // num_pid_n
    pid_n = pid % num_pid_n
    token = slot // top_k
    route = slot % top_k
    expert = (
        gl.load(topk_ids_ptr + token * stride_tit + route * stride_tis) - expert_start
    )
    if expert < 0:
        return
    if expert >= num_experts:
        return

    block_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_K // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, block_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, block_layout)

    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    n_mask = offs_n < I
    x_row = x_ptr + token.to(gl.int64) * stride_xm
    expert_weight = w1_ptr + expert.to(gl.int64) * stride_we
    expert_scale = w1_scale_ptr + expert.to(gl.int64) * stride_se
    gate_row = expert_weight + offs_n[:, None].to(gl.int64) * stride_wn
    up_row = expert_weight + (I + offs_n)[:, None].to(gl.int64) * stride_wn

    # BLOCK_N divides the 128-row scale block, so every output in one program
    # shares a scale row. Only the K-block scale vector needs to be loaded.
    gate_scale_row = (pid_n * BLOCK_N) // _SCALE_BLOCK
    up_scale_row = (I + pid_n * BLOCK_N) // _SCALE_BLOCK
    acc_gate = gl.zeros([BLOCK_N], gl.float32, n_layout)
    acc_up = gl.zeros([BLOCK_N], gl.float32, n_layout)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + gl.arange(0, BLOCK_K, layout=k_layout)
        k_mask = offs_k < D
        tile_mask = n_mask[:, None] & k_mask[None, :]
        weight_k = offs_k[None, :].to(gl.int64) * stride_wk
        scale_k = (offs_k // _SCALE_BLOCK).to(gl.int64) * stride_sk
        x = gl.load(x_row + offs_k * stride_xk, mask=k_mask, other=0.0).to(gl.float32)
        gate_scale = gl.load(
            expert_scale + gate_scale_row * stride_sn + scale_k,
            mask=k_mask,
            other=0.0,
        ).to(gl.float32)
        up_scale = gl.load(
            expert_scale + up_scale_row * stride_sn + scale_k,
            mask=k_mask,
            other=0.0,
        ).to(gl.float32)
        gate_ptr, up_ptr = gate_row + weight_k, up_row + weight_k
        gate = gl.load(gate_ptr, mask=tile_mask, other=0.0).to(gl.float32)
        up = gl.load(up_ptr, mask=tile_mask, other=0.0).to(gl.float32)
        acc_gate += gl.sum(gate * gate_scale[None, :] * x[None, :], axis=1)
        acc_up += gl.sum(up * up_scale[None, :] * x[None, :], axis=1)

    if SWIGLU_LIMIT > 0.0:
        acc_gate = gl.minimum(acc_gate, SWIGLU_LIMIT)
        acc_up = gl.clamp(acc_up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    inter = acc_gate * (1.0 / (1.0 + gl.exp(-acc_gate))) * acc_up
    gl.store(
        out_ptr + slot.to(gl.int64) * stride_om + offs_n * stride_on,
        inter.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


@gluon.jit
def _stage2_fp8_warp_gemv(
    inter_ptr,
    w2_ptr,
    w2_scale_ptr,
    out_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_start,
    num_experts,
    D,
    I,
    top_k,
    stride_im,
    stride_ik,
    stride_we,
    stride_wd,
    stride_wk,
    stride_se,
    stride_sd,
    stride_sk,
    stride_om,
    stride_od,
    stride_tit,
    stride_tis,
    stride_twt,
    stride_tws,
    PACKED_WEIGHTS: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    if PACKED_WEIGHTS:
        _packed_stage2(
            inter_ptr,
            w2_ptr,
            w2_scale_ptr,
            out_ptr,
            topk_ids_ptr,
            topk_weights_ptr,
            expert_start,
            num_experts,
            D,
            I,
            top_k,
            stride_im,
            stride_ik,
            stride_we,
            stride_se,
            stride_sd,
            stride_sk,
            stride_om,
            stride_od,
            stride_tit,
            stride_tis,
            stride_twt,
            stride_tws,
            BLOCK_D,
            BLOCK_K,
            NUM_WARPS,
        )
        return
    pid = gl.program_id(0)
    num_pid_d = gl.cdiv(D, BLOCK_D)
    token = pid // num_pid_d
    pid_d = pid % num_pid_d

    block_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_D + NUM_WARPS - 1) // NUM_WARPS, BLOCK_K // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    d_layout: gl.constexpr = gl.SliceLayout(1, block_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, block_layout)
    offs_d = pid_d * BLOCK_D + gl.arange(0, BLOCK_D, layout=d_layout)
    d_mask = offs_d < D
    scale_d = (pid_d * BLOCK_D) // _SCALE_BLOCK

    acc = gl.zeros([BLOCK_D], gl.float32, d_layout)
    for slot in range(0, top_k):
        global_expert = gl.load(topk_ids_ptr + token * stride_tit + slot * stride_tis)
        expert = global_expert - expert_start
        expert_valid = (expert >= 0) & (expert < num_experts)
        probability = gl.load(
            topk_weights_ptr + token * stride_twt + slot * stride_tws
        ).to(gl.float32)
        probability = gl.where(expert_valid, probability, 0.0)
        inter_row = inter_ptr + (token * top_k + slot).to(gl.int64) * stride_im
        weight_row = (
            w2_ptr
            + expert.to(gl.int64) * stride_we
            + offs_d[:, None].to(gl.int64) * stride_wd
        )
        scale_row = w2_scale_ptr + expert.to(gl.int64) * stride_se + scale_d * stride_sd
        dot = gl.zeros([BLOCK_D], gl.float32, d_layout)
        for k0 in range(0, I, BLOCK_K):
            offs_k = k0 + gl.arange(0, BLOCK_K, layout=k_layout)
            k_mask = offs_k < I
            scale_k = (offs_k // _SCALE_BLOCK).to(gl.int64) * stride_sk
            tile_mask = d_mask[:, None] & k_mask[None, :] & expert_valid
            activation = gl.load(
                inter_row + offs_k * stride_ik,
                mask=k_mask & expert_valid,
                other=0.0,
            ).to(gl.float32)
            scale = gl.load(
                scale_row + scale_k,
                mask=k_mask & expert_valid,
                other=0.0,
            ).to(gl.float32)
            weight_ptr = weight_row + offs_k[None, :].to(gl.int64) * stride_wk
            weight = gl.load(
                weight_ptr,
                mask=tile_mask,
                other=0.0,
            ).to(gl.float32)
            dot += gl.sum(weight * scale[None, :] * activation[None, :], axis=1)
        acc += probability * dot

    gl.store(
        out_ptr + token * stride_om + offs_d * stride_od,
        acc.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


def gluon_fp8_block_dequantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Expand a contiguous 128x128 block-scaled E4M3 expert tensor to BF16.

    Args:
        weight: E4M3 experts ``[experts, output, input]``.
        scale: FP32 inverse scales ``[experts, ceil(output/128), ceil(input/128)]``.

    Returns:
        A contiguous BF16 tensor with the same shape as ``weight``.
    """
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert weight.ndim == 3 and scale.ndim == 3
    assert weight.dtype in fp8_dtypes and scale.dtype == torch.float32
    assert weight.is_contiguous() and scale.is_contiguous()
    expected_scale_shape = (
        weight.shape[0],
        triton.cdiv(weight.shape[1], 128),
        triton.cdiv(weight.shape[2], 128),
    )
    assert scale.shape == expected_scale_shape
    output = torch.empty_like(weight, dtype=torch.bfloat16)
    block = 256
    num_warps = 4
    _block_fp8_dequantize_kernel[(triton.cdiv(weight.numel(), block),)](
        weight,
        scale,
        output,
        weight.numel(),
        weight.shape[1],
        weight.shape[2],
        scale.shape[1],
        scale.shape[2],
        BLOCK=block,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return output


def gluon_fp8_block_warp_decode_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float | None = None,
    expert_start: int = 0,
    expert_parallel: bool = False,
    *,
    packed_weights: bool,
) -> torch.Tensor:
    """Apply compact 128x128 block-E4M3 experts to a decode-shaped batch.

    Args:
        hidden_states: BF16 token states ``[tokens, hidden]``.
        w13: E4M3 gate/up experts ``[experts, 2 * intermediate, hidden]``.
        w2: E4M3 down experts ``[experts, hidden, intermediate]``.
        w13_scale: FP32 inverse scales for 128x128 blocks of ``w13``.
        w2_scale: FP32 inverse scales for 128x128 blocks of ``w2``.
        topk_ids: Routed expert ids ``[tokens, top_k]``.
        topk_weights: Routed expert weights ``[tokens, top_k]``.
        swiglu_limit: Optional GLM SwiGLU clamp limit.
        expert_start: First global expert ID owned by this rank. Routes outside
            the local expert range contribute zero for expert parallelism.
        expert_parallel: Whether top-k IDs span experts owned by multiple ranks.
        packed_weights: Whether expert banks use the canonical N16/K32 layout.

    Returns:
        BF16 finalized states ``[tokens, hidden]``.
    """
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert hidden_states.dtype == torch.bfloat16
    assert w13.dtype in fp8_dtypes and w2.dtype in fp8_dtypes
    assert w13_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32
    num_tokens, hidden_size = hidden_states.shape
    num_experts, twice_intermediate, weight_hidden = w13.shape
    intermediate_size = twice_intermediate // 2
    assert weight_hidden == hidden_size
    assert w2.shape == (num_experts, hidden_size, intermediate_size)
    assert expert_start >= 0
    assert hidden_size % 512 == 0 and intermediate_size % 512 == 0
    top_k = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    intermediate = torch.empty(
        (num_tokens * top_k, intermediate_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if intermediate_size == 512 and num_tokens == 1:
        block_n, stage1_block_k, stage1_warps = 32, 1024, 8
    elif intermediate_size == 512 and num_tokens <= 2:
        block_n, stage1_block_k, stage1_warps = 8, 1024, 2
    elif intermediate_size == 512 and num_tokens <= 4:
        block_n, stage1_block_k, stage1_warps = 8, 1024, 4
    elif intermediate_size == 512 and num_tokens <= 8:
        block_n, stage1_block_k, stage1_warps = 32, 512, 4
    elif intermediate_size == 512 and num_tokens <= 16:
        block_n, stage1_block_k, stage1_warps = 4, 1024, 2
    elif intermediate_size == 512:
        block_n, stage1_block_k, stage1_warps = 32, 512, 4
    elif num_tokens <= 4:
        block_n, stage1_block_k, stage1_warps = 4, 1024, 2
    elif num_tokens <= 8:
        block_n, stage1_block_k, stage1_warps = 8, 1024, 2
    else:
        block_n, stage1_block_k, stage1_warps = 16, 512, 2
    if expert_parallel:
        block_n, stage1_block_k, stage1_warps = 32, 512, 4
    if packed_weights:
        block_n, stage1_block_k, stage1_warps = 16, 1024, 8
    stage1_grid = (num_tokens * top_k * triton.cdiv(intermediate_size, block_n),)
    _stage1_fp8_warp_gemv[stage1_grid](
        hidden_states,
        w13,
        w13_scale,
        intermediate,
        topk_ids,
        expert_start,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w13.stride(0),
        w13.stride(1),
        w13.stride(2),
        w13_scale.stride(0),
        w13_scale.stride(1),
        w13_scale.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        PACKED_WEIGHTS=packed_weights,
        SWIGLU_LIMIT=-1.0 if swiglu_limit is None else swiglu_limit,
        BLOCK_N=block_n,
        BLOCK_K=stage1_block_k,
        NUM_WARPS=stage1_warps,
        num_warps=stage1_warps,
    )

    output = torch.empty_like(hidden_states)
    if expert_parallel:
        block_d, stage2_warps = 16, 4
        stage2_block_k = 1024
    elif intermediate_size == 512 and num_tokens <= 4:
        block_d, stage2_warps = 8, 4
        stage2_block_k = 512
    elif intermediate_size == 512 and num_tokens <= 8:
        block_d, stage2_warps = 8, 2
        stage2_block_k = 512
    elif intermediate_size == 512:
        block_d, stage2_warps = 64, 8
        stage2_block_k = 512
    elif num_tokens <= 4:
        block_d, stage2_warps = 8, 4
        stage2_block_k = 1024
    elif num_tokens <= 8:
        block_d, stage2_warps = 32, 4
        stage2_block_k = 1024
    elif num_tokens <= 16:
        block_d, stage2_warps = 16, 4
        stage2_block_k = 1024
    else:
        block_d, stage2_warps = 8, 2
        stage2_block_k = 1024
    if packed_weights:
        block_d, stage2_block_k, stage2_warps = 16, 512, 8
    stage2_grid = (num_tokens * triton.cdiv(hidden_size, block_d),)
    _stage2_fp8_warp_gemv[stage2_grid](
        intermediate,
        w2,
        w2_scale,
        output,
        topk_ids,
        topk_weights,
        expert_start,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        intermediate.stride(0),
        intermediate.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        output.stride(0),
        output.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        PACKED_WEIGHTS=packed_weights,
        BLOCK_D=block_d,
        BLOCK_K=stage2_block_k,
        NUM_WARPS=stage2_warps,
        num_warps=stage2_warps,
    )
    return output
