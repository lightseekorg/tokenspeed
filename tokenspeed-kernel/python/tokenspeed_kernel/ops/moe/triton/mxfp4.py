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
from tokenspeed_kernel._triton import TensorDescriptor, libdevice, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _routing_kernel(
    topk_ids_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_routes,
    BLOCK_ROUTES: tl.constexpr,
):
    expert_id = tl.program_id(0)
    count = 0
    num_blocks = tl.cdiv(num_routes, BLOCK_ROUTES)
    for block_id in range(num_blocks):
        route_ids = block_id * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
        route_mask = route_ids < num_routes
        selected_experts = tl.load(topk_ids_ptr + route_ids, mask=route_mask, other=-1)
        matches = route_mask & (selected_experts == expert_id)
        local_rank = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        tl.store(
            expert_route_ids_ptr + expert_id * num_routes + count + local_rank,
            route_ids,
            mask=matches,
        )
        count += tl.sum(matches.to(tl.int32), axis=0)
    tl.store(expert_counts_ptr + expert_id, count)


def _routing(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    topk_ids = topk_ids.to(torch.int32).contiguous()
    num_routes = topk_ids.numel()
    expert_route_ids = torch.empty(
        (num_experts, num_routes), device=topk_ids.device, dtype=torch.int32
    )
    expert_counts = torch.empty(num_experts, device=topk_ids.device, dtype=torch.int32)
    block_routes = 128 if num_routes <= 128 else 1024
    _routing_kernel[(num_experts,)](
        topk_ids,
        expert_route_ids,
        expert_counts,
        num_routes,
        BLOCK_ROUTES=block_routes,
        num_warps=4,
    )
    return expert_route_ids, expert_counts


@triton.jit
def _combine_kernel(
    route_output_ptr,
    topk_weights_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for slot in range(top_k):
        route_offsets = token_offsets * top_k + slot
        values = tl.load(
            route_output_ptr
            + route_offsets[:, None] * hidden_size
            + hidden_offsets[None, :],
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weights = tl.load(topk_weights_ptr + route_offsets, mask=token_mask, other=0.0)
        acc += values.to(tl.float32) * weights[:, None]
    tl.store(
        output_ptr + token_offsets[:, None] * hidden_size + hidden_offsets[None, :],
        acc,
        mask=token_mask[:, None] & hidden_mask[None, :],
    )


def _combine(
    route_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    num_tokens, top_k = topk_weights.shape
    hidden_size = output.shape[1]
    _combine_kernel[(triton.cdiv(num_tokens, 4), triton.cdiv(hidden_size, 256))](
        route_output,
        topk_weights.contiguous(),
        output,
        num_tokens,
        hidden_size,
        top_k=top_k,
        BLOCK_M=4,
        BLOCK_N=256,
        num_warps=4,
    )


def _validate(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
    upcast_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not do_finalize:
        raise ValueError("Triton MoE does not support deferred finalization")
    if int(getattr(w, "ep_size", 1)) != 1:
        raise ValueError("Triton MoE does not support expert parallelism")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Triton MoE requires precomputed topk weights and ids")

    w13 = w.w13_weight
    w13_scale = w.w13_weight_scale
    w2 = w.w2_weight
    w2_scale = w.w2_weight_scale
    w13_bias = getattr(w, "w13_weight_bias", None)
    w2_bias = getattr(w, "w2_weight_bias", None)
    if x.ndim != 2:
        raise ValueError("x must have shape [num_tokens, hidden_size]")
    if any(t.dtype != torch.uint8 for t in (w13, w13_scale, w2, w2_scale)):
        raise TypeError("MXFP4 packed values and scales must use torch.uint8")
    if not all(
        t.is_cuda and t.is_contiguous() for t in (x, w13, w13_scale, w2, w2_scale)
    ):
        raise ValueError("x and weights must be contiguous GPU tensors")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use torch.int32 or torch.int64")
    if not topk_weights.is_floating_point():
        raise TypeError("topk_weights must use a floating-point dtype")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and x must be on the same device")
    if any(t.device != x.device for t in (w13, w13_scale, w2, w2_scale)):
        raise ValueError("x and weights must be on the same device")

    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, packed_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_experts == 0 or twice_intermediate_size % 2:
        raise ValueError("MXFP4 MoE requires at least one expert and paired w13 rows")
    if topk_ids.shape[0] != num_tokens or top_k == 0:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k > 0]")
    if packed_hidden_size * 2 != hidden_size:
        raise ValueError("w13_weight has an incompatible packed shape")
    if w13_scale.shape != (num_experts, 2 * intermediate_size, hidden_size // 32):
        raise ValueError("w13_weight_scale must have shape [E, 2I, H/32]")
    if w2.shape != (num_experts, hidden_size, intermediate_size // 2):
        raise ValueError("w2_weight must have shape [E, H, I/2]")
    if w2_scale.shape != (num_experts, hidden_size, intermediate_size // 32):
        raise ValueError("w2_weight_scale must have shape [E, H, I/32]")

    layout = getattr(w, "w13_input_layout", "concatenated")
    if upcast_weights:
        activation = getattr(w, "activation", "swiglu")
        if activation != "swiglu":
            raise ValueError("Weight-only MXFP4 MoE supports only SwiGLU")
        if x.dtype != torch.bfloat16:
            raise TypeError("Weight-only MXFP4 MoE requires BF16 input")
        if layout not in {"concatenated", "interleaved"}:
            raise ValueError("MXFP4 MoE received an invalid gate/up weight layout")
        if (w13_bias is None) != (w2_bias is None):
            raise ValueError("w13 and w2 expert biases must both be present or absent")
        if w13_bias is not None:
            if w13_bias.dtype != x.dtype or w2_bias.dtype != x.dtype:
                raise TypeError("expert biases must match the input dtype")
            if not all(t.is_cuda and t.is_contiguous() for t in (w13_bias, w2_bias)):
                raise ValueError("expert biases must be contiguous GPU tensors")
            if w13_bias.device != x.device or w2_bias.device != x.device:
                raise ValueError("expert biases must be on the input device")
            if w13_bias.shape != (
                num_experts,
                2 * intermediate_size,
            ) or w2_bias.shape != (num_experts, hidden_size):
                raise ValueError("expert biases have incompatible shapes")
        swiglu_arg = getattr(w, "swiglu_arg", None)
        limit = getattr(swiglu_arg, "limit", None)
        if limit is not None and limit <= 0:
            raise ValueError("SwiGLU limit must be positive")
        if hidden_size % 64 or intermediate_size % 32:
            raise ValueError(
                "Weight-only MXFP4 requires hidden size aligned to 64 and "
                "intermediate size aligned to 32"
            )
    else:
        activation = plan.get("activation") or getattr(w, "activation", "silu")
        if activation not in {"silu", "situ", "swiglu"}:
            raise ValueError(f"Triton MoE does not support activation {activation!r}")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("x must use torch.float16 or torch.bfloat16")
        if w13_bias is not None or w2_bias is not None:
            raise ValueError("Triton MoE does not support expert bias")
        if layout != "concatenated":
            raise ValueError("Triton MoE requires concatenated gate/up weights")
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None and (
            getattr(swiglu_arg, "alpha", None) not in {None, 1.0}
            or getattr(swiglu_arg, "limit", None) is not None
        ):
            raise ValueError("Triton MoE supports only standard SwiGLU")
        if getattr(w, "swiglu_beta", None) not in {None, 0.0}:
            raise ValueError("Triton MoE supports only standard SwiGLU")
        if activation == "situ":
            situ_beta = getattr(w, "activation_situ_beta", None)
            situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)
            if situ_beta is None or situ_beta <= 0:
                raise ValueError("SiTU beta must be positive")
            if situ_linear_beta is not None and situ_linear_beta <= 0:
                raise ValueError("SiTU linear beta must be positive")
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError("hidden and intermediate sizes must be multiples of 128")
    return topk_weights, topk_ids, activation


@triton.jit
def _quantize_mxfp4_routine(values, valid_mask):
    block_m: tl.constexpr = values.shape[0]
    block_k: tl.constexpr = values.shape[1]
    num_scale_blocks: tl.constexpr = block_k // 32
    values_f32 = values.to(tl.float32)
    abs_values = tl.where(valid_mask, tl.abs(values_f32), 0.0)
    abs_values = abs_values.reshape((block_m, num_scale_blocks, 32))
    max_values = tl.max(abs_values, axis=2, keep_dims=True)

    dequant_scale = max_values / 6.0
    scale_bits = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    dequant_scale = scale_bits.to(tl.float32, bitcast=True)
    scales = (scale_bits.reshape((block_m, num_scale_blocks)) >> 23).to(tl.uint8)
    quant_scale = tl.where(dequant_scale == 0, 0.0, 1.0 / dequant_scale)
    normalized = values_f32.reshape((block_m, num_scale_blocks, 32)) * quant_scale
    normalized = normalized.reshape((block_m, block_k))
    normalized = tl.where(valid_mask, normalized, 0.0)

    bits = normalized.to(tl.uint32, bitcast=True)
    signs = bits & 0x80000000
    exponents = (bits >> 23) & 0xFF
    original_mantissas = bits & 0x7FFFFF
    is_subnormal = exponents < 127
    exponent_shift = 127 - (exponents + 1)
    subnormal_mantissas = 0x400000 | (original_mantissas >> 1)
    mantissas = tl.where(
        is_subnormal,
        subnormal_mantissas >> exponent_shift,
        original_mantissas,
    )
    exponents = tl.maximum(exponents, 126) - 126
    mantissa_bits = mantissas >> 21
    kept_lsb = (mantissa_bits >> 1) & 1
    guard = mantissa_bits & 1
    sticky = (mantissas & 0x1FFFFF) != 0
    round_up = guard & (sticky | kept_lsb)
    codes = tl.minimum((((exponents << 2) | mantissa_bits) + round_up) >> 1, 7)
    codes = ((signs >> 28) | codes).to(tl.uint8)
    code_pairs = codes.reshape((block_m, block_k // 2, 2))
    low, high = tl.split(code_pairs)
    return low | (high << 4), scales


@triton.jit
def _quantize_mxfp4_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    num_rows,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k_start = tl.program_id(1) * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    valid_mask = (row_offsets[:, None] < num_rows) & (k_offsets[None, :] < K)
    values = tl.load(
        x_ptr + row_offsets[:, None] * K + k_offsets[None, :],
        mask=valid_mask,
        other=0.0,
    )
    packed, scales = _quantize_mxfp4_routine(values, valid_mask)
    packed_offsets = (
        row_offsets[:, None] * (K // 2)
        + k_start // 2
        + tl.arange(0, BLOCK_K // 2)[None, :]
    )
    scale_offsets = (
        row_offsets[:, None] * (K // 32)
        + k_start // 32
        + tl.arange(0, BLOCK_K // 32)[None, :]
    )
    row_mask = row_offsets < num_rows
    tl.store(packed_ptr + packed_offsets, packed, mask=row_mask[:, None])
    tl.store(scale_ptr + scale_offsets, scales, mask=row_mask[:, None])


@triton.jit
def _upcast_mxfp4_tile(packed, scales):
    """Decode a linear E2M1/E8M0 weight tile to BF16 for a standard dot."""
    em0 = packed & 0x07
    em1 = packed & 0x70
    x0 = (em0.to(tl.uint16) << 6) | ((packed & 0x08).to(tl.uint16) << 12)
    x1 = (em1.to(tl.uint16) << 2) | ((packed & 0x80).to(tl.uint16) << 8)
    x0 = tl.where((em0 & 0x06) != 0, x0 + (126 << 7), x0)
    x1 = tl.where((em1 & 0x60) != 0, x1 + (126 << 7), x1)
    x0 = tl.where(em0 == 0x01, 16128 | (x0 & 0x8000), x0)
    x1 = tl.where(em1 == 0x10, 16128 | (x1 & 0x8000), x1)
    values = tl.interleave(x0, x1).to(tl.bfloat16, bitcast=True)
    values = values.reshape((packed.shape[0], scales.shape[1], 32))
    scale_values = (scales.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
    return (values * scale_values[:, :, None]).reshape(
        (packed.shape[0], packed.shape[1] * 2)
    )


@triton.jit
def _stage1_kernel(
    x_data,
    x_scale_ptr,
    w13_data,
    w13_scale_ptr,
    w13_bias_ptr,
    inter_ptr,
    inter_scale_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    situ_beta,
    situ_linear_beta,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    swiglu_alpha: tl.constexpr,
    swiglu_limit: tl.constexpr,
    swiglu_beta: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    UPCAST_WEIGHTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_count = num_tokens * top_k
    packed_k = hidden_size // 2
    scale_k = hidden_size // 32
    tile_idx = tl.program_id(0)
    problem_start = 0

    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(intermediate_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles
            local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = local_rows < group_m
            route_ids = tl.load(
                expert_route_ids_ptr + expert_id * route_count + local_rows,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            token_ids = tl.where(row_mask, route_ids // top_k, 0).to(tl.int32)
            n_offset = tile_n * BLOCK_N
            logical_rows = n_offset + tl.arange(0, BLOCK_N)
            if INTERLEAVED:
                gate_rows = logical_rows * 2
                up_rows = gate_rows + 1
            else:
                gate_rows = logical_rows
                up_rows = intermediate_size + logical_rows
            gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, hidden_size, BLOCK_K):
                scale_cols = tl.arange(0, BLOCK_K // 32)
                expert_scale = (
                    w13_scale_ptr + expert_id * 2 * intermediate_size * scale_k
                )
                if UPCAST_WEIGHTS:
                    x = tl.load(
                        x_data
                        + token_ids[:, None] * hidden_size
                        + k_offset
                        + tl.arange(0, BLOCK_K)[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    packed_offsets = k_offset // 2 + tl.arange(0, BLOCK_K // 2)
                    expert_weight = (
                        w13_data + expert_id * 2 * intermediate_size * packed_k
                    )
                    gate_packed = tl.load(
                        expert_weight
                        + gate_rows[:, None] * packed_k
                        + packed_offsets[None, :]
                    )
                    up_packed = tl.load(
                        expert_weight
                        + up_rows[:, None] * packed_k
                        + packed_offsets[None, :]
                    )
                    gate_scale = tl.load(
                        expert_scale
                        + gate_rows[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :]
                    )
                    up_scale = tl.load(
                        expert_scale
                        + up_rows[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :]
                    )
                    gate = _upcast_mxfp4_tile(gate_packed, gate_scale)
                    up = _upcast_mxfp4_tile(up_packed, up_scale)
                    gate_acc += tl.dot(x, gate.T)
                    up_acc += tl.dot(x, up.T)
                else:
                    x = x_data.gather(token_ids, k_offset // 2)
                    x_scale = tl.load(
                        x_scale_ptr
                        + token_ids[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :],
                        mask=row_mask[:, None],
                        other=0,
                    )
                    gate = w13_data.load([expert_id, n_offset, k_offset // 2]).reshape(
                        (BLOCK_N, BLOCK_K // 2)
                    )
                    up = w13_data.load(
                        [expert_id, intermediate_size + n_offset, k_offset // 2]
                    ).reshape((BLOCK_N, BLOCK_K // 2))
                    gate_scale = tl.load(
                        expert_scale
                        + logical_rows[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :]
                    )
                    up_scale = tl.load(
                        expert_scale
                        + up_rows[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :]
                    )
                    gate_acc = tl.dot_scaled(
                        x,
                        x_scale,
                        "e2m1",
                        gate.T,
                        gate_scale,
                        "e2m1",
                        gate_acc,
                        fast_math=False,
                        lhs_k_pack=True,
                        rhs_k_pack=True,
                    )
                    up_acc = tl.dot_scaled(
                        x,
                        x_scale,
                        "e2m1",
                        up.T,
                        up_scale,
                        "e2m1",
                        up_acc,
                        fast_math=False,
                        lhs_k_pack=True,
                        rhs_k_pack=True,
                    )

            if UPCAST_WEIGHTS:
                if HAS_BIAS:
                    expert_bias = w13_bias_ptr + expert_id * 2 * intermediate_size
                    gate_acc += tl.load(expert_bias + gate_rows)[None, :]
                    up_acc += tl.load(expert_bias + up_rows)[None, :]
                if HAS_LIMIT:
                    gate_acc = tl.minimum(gate_acc, swiglu_limit)
                    up_acc = tl.clamp(up_acc, -swiglu_limit, swiglu_limit)
                activated = (
                    gate_acc
                    * tl.sigmoid(swiglu_alpha * gate_acc)
                    * (up_acc + swiglu_beta)
                ).to(tl.bfloat16)
                inter_offsets = (
                    route_ids[:, None] * intermediate_size
                    + n_offset
                    + tl.arange(0, BLOCK_N)[None, :]
                )
                tl.store(inter_ptr + inter_offsets, activated, mask=row_mask[:, None])
            else:
                output_dtype: tl.constexpr = (
                    tl.bfloat16 if OUTPUT_DTYPE == "bf16" else tl.float16
                )
                gate = gate_acc.to(output_dtype)
                up = up_acc.to(output_dtype)
                if ACTIVATION == "situ":
                    gate = gate.to(tl.float32)
                    up = up.to(tl.float32)
                    gate = (
                        situ_beta * libdevice.tanh(gate / situ_beta) * tl.sigmoid(gate)
                    )
                    if HAS_LINEAR_BETA:
                        up = situ_linear_beta * libdevice.tanh(up / situ_linear_beta)
                    activated = (gate * up).to(output_dtype)
                else:
                    silu = (gate.to(tl.float32) * tl.sigmoid(gate.to(tl.float32))).to(
                        output_dtype
                    )
                    activated = (silu * up).to(output_dtype)
                valid_mask = row_mask[:, None] & (
                    tl.arange(0, BLOCK_N)[None, :] < BLOCK_N
                )
                activated_packed, activated_scales = _quantize_mxfp4_routine(
                    activated, valid_mask
                )
                inter_packed_offsets = (
                    route_ids[:, None] * (intermediate_size // 2)
                    + n_offset // 2
                    + tl.arange(0, BLOCK_N // 2)[None, :]
                )
                inter_scale_offsets = (
                    route_ids[:, None] * (intermediate_size // 32)
                    + n_offset // 32
                    + tl.arange(0, BLOCK_N // 32)[None, :]
                )
                tl.store(
                    inter_ptr + inter_packed_offsets,
                    activated_packed,
                    mask=row_mask[:, None],
                )
                tl.store(
                    inter_scale_ptr + inter_scale_offsets,
                    activated_scales,
                    mask=row_mask[:, None],
                )
            tile_idx += NUM_PROGRAMS

        problem_start += problem_tiles


@triton.jit
def _stage2_kernel(
    inter_ptr,
    inter_scale_ptr,
    w2_data,
    w2_scale_ptr,
    w2_bias_ptr,
    route_output_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    UPCAST_WEIGHTS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_count = num_tokens * top_k
    packed_k = intermediate_size // 2
    scale_k = intermediate_size // 32
    tile_idx = tl.program_id(0)
    problem_start = 0

    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(hidden_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles
            local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = local_rows < group_m
            route_ids = tl.load(
                expert_route_ids_ptr + expert_id * route_count + local_rows,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            route_ids = tl.where(row_mask, route_ids, -1).to(tl.int32)
            n_offset = tile_n * BLOCK_N
            weight_rows = n_offset + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, intermediate_size, BLOCK_K):
                scale_cols = tl.arange(0, BLOCK_K // 32)
                expert_scale = w2_scale_ptr + expert_id * hidden_size * scale_k
                if UPCAST_WEIGHTS:
                    intermediate = tl.load(
                        inter_ptr
                        + route_ids[:, None] * intermediate_size
                        + k_offset
                        + tl.arange(0, BLOCK_K)[None, :],
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    packed_offsets = k_offset // 2 + tl.arange(0, BLOCK_K // 2)
                    expert_weight = w2_data + expert_id * hidden_size * packed_k
                    weight_packed = tl.load(
                        expert_weight
                        + weight_rows[:, None] * packed_k
                        + packed_offsets[None, :]
                    )
                    weight_scale = tl.load(
                        expert_scale
                        + weight_rows[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :]
                    )
                    weight = _upcast_mxfp4_tile(weight_packed, weight_scale)
                    acc += tl.dot(intermediate, weight.T)
                else:
                    intermediate = tl.load(
                        inter_ptr
                        + route_ids[:, None] * (intermediate_size // 2)
                        + k_offset // 2
                        + tl.arange(0, BLOCK_K // 2)[None, :],
                        mask=row_mask[:, None],
                        other=0,
                    )
                    intermediate_scale = tl.load(
                        inter_scale_ptr
                        + route_ids[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :],
                        mask=row_mask[:, None],
                        other=0,
                    )
                    weight = w2_data.load([expert_id, n_offset, k_offset // 2]).reshape(
                        (BLOCK_N, BLOCK_K // 2)
                    )
                    weight_scale = tl.load(
                        expert_scale
                        + weight_rows[:, None] * scale_k
                        + k_offset // 32
                        + scale_cols[None, :]
                    )
                    acc = tl.dot_scaled(
                        intermediate,
                        intermediate_scale,
                        "e2m1",
                        weight.T,
                        weight_scale,
                        "e2m1",
                        acc,
                        fast_math=False,
                        lhs_k_pack=True,
                        rhs_k_pack=True,
                    )

            if UPCAST_WEIGHTS and HAS_BIAS:
                acc += tl.load(w2_bias_ptr + expert_id * hidden_size + weight_rows)[
                    None, :
                ]
            output_offsets = (
                route_ids[:, None] * hidden_size
                + n_offset
                + tl.arange(0, BLOCK_N)[None, :]
            )
            tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])
            tile_idx += NUM_PROGRAMS

        problem_start += problem_tiles


def _prepare_routed_output(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_route_ids, expert_counts = _routing(topk_ids, num_experts)
    # Invalid expert ids are absent from routing; zero their canonical rows.
    route_output = torch.zeros(
        (topk_ids.numel(), x.shape[1]), device=x.device, dtype=x.dtype
    )
    return expert_route_ids, expert_counts, route_output, torch.empty_like(x)


def _moe(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> torch.Tensor:
    internal_activation_dtype = plan.get("internal_activation_dtype")
    if internal_activation_dtype == "mxfp4":
        upcast_weights = False
    elif internal_activation_dtype == "input":
        upcast_weights = True
    else:
        raise ValueError(
            "Triton MXFP4 MoE requires internal_activation_dtype "
            "to be 'input' or 'mxfp4'"
        )

    topk_weights, topk_ids, activation = _validate(
        plan, x, w, topk_weights, topk_ids, do_finalize, upcast_weights
    )
    if upcast_weights:
        swiglu_arg = getattr(w, "swiglu_arg", None)
        swiglu_alpha = float(getattr(swiglu_arg, "alpha", 1.0) or 1.0)
        swiglu_limit = getattr(swiglu_arg, "limit", None)
        swiglu_beta = float(getattr(w, "swiglu_beta", 0.0) or 0.0)
        interleaved = getattr(w, "w13_input_layout", "concatenated") == "interleaved"
        situ_beta = 1.0
        situ_linear_beta = None
    else:
        situ_beta = float(getattr(w, "activation_situ_beta", 1.0) or 1.0)
        situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)
        swiglu_alpha = 1.0
        swiglu_limit = None
        swiglu_beta = 0.0
        interleaved = False

    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w.w13_weight.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_tokens == 0:
        return torch.empty_like(x)

    route_count = num_tokens * top_k
    expert_route_ids, expert_counts, route_output, output = _prepare_routed_output(
        x, topk_ids, num_experts
    )
    block_m = 16 if num_tokens <= 16 else 64
    w13_bias = getattr(w, "w13_weight_bias", None)
    w2_bias = getattr(w, "w2_weight_bias", None)
    has_bias = w13_bias is not None

    if upcast_weights:
        x_data = x
        x_scale = x
        w13_data = w.w13_weight
        intermediate = torch.empty(
            (route_count, intermediate_size), device=x.device, dtype=x.dtype
        )
        intermediate_scale = w.w13_weight_scale
        w2_data = w.w2_weight
        stage1_block_n = 32
        stage2_block_n = 32
        stage1_block_k = 64
        stage2_block_k = 64 if intermediate_size % 64 == 0 else 32
        backend_options = {}
    else:
        x_packed = torch.empty(
            (num_tokens, hidden_size // 2), device=x.device, dtype=torch.uint8
        )
        x_scale = torch.empty(
            (num_tokens, hidden_size // 32), device=x.device, dtype=torch.uint8
        )
        intermediate = torch.empty(
            (route_count, intermediate_size // 2),
            device=x.device,
            dtype=torch.uint8,
        )
        intermediate_scale = torch.empty(
            (route_count, intermediate_size // 32),
            device=x.device,
            dtype=torch.uint8,
        )
        stage1_block_n = 64
        stage2_block_n = 128
        stage1_block_k = 128
        stage2_block_k = 128
        quant_block_k = 256 if hidden_size % 256 == 0 else 128
        _quantize_mxfp4_kernel[
            (
                triton.cdiv(num_tokens, block_m),
                triton.cdiv(hidden_size, quant_block_k),
            )
        ](
            x,
            x_packed,
            x_scale,
            num_tokens,
            K=hidden_size,
            BLOCK_M=block_m,
            BLOCK_K=quant_block_k,
            num_warps=4,
        )
        x_data = TensorDescriptor.from_tensor(x_packed, [1, stage1_block_k // 2])
        w13_data = TensorDescriptor.from_tensor(
            w.w13_weight, [1, stage1_block_n, stage1_block_k // 2]
        )
        w2_data = TensorDescriptor.from_tensor(
            w.w2_weight, [1, stage2_block_n, stage2_block_k // 2]
        )
        backend = triton.runtime.driver.active.get_current_target().backend
        backend_options = (
            {"matrix_instr_nonkdim": 32, "kpack": 1} if backend == "hip" else {}
        )

    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    stage1_programs = min(
        num_sms, route_count * triton.cdiv(intermediate_size, stage1_block_n)
    )
    stage2_programs = min(
        num_sms, route_count * triton.cdiv(hidden_size, stage2_block_n)
    )
    _stage1_kernel[(stage1_programs,)](
        x_data,
        x_scale,
        w13_data,
        w.w13_weight_scale,
        w.w13_weight if w13_bias is None else w13_bias,
        intermediate,
        intermediate_scale,
        expert_route_ids,
        expert_counts,
        num_tokens,
        situ_beta=situ_beta,
        situ_linear_beta=(1.0 if situ_linear_beta is None else situ_linear_beta),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=1.0 if swiglu_limit is None else swiglu_limit,
        swiglu_beta=swiglu_beta,
        OUTPUT_DTYPE="bf16" if x.dtype == torch.bfloat16 else "fp16",
        ACTIVATION=activation,
        UPCAST_WEIGHTS=upcast_weights,
        INTERLEAVED=interleaved,
        HAS_BIAS=has_bias,
        HAS_LIMIT=swiglu_limit is not None,
        HAS_LINEAR_BETA=situ_linear_beta is not None,
        NUM_PROGRAMS=stage1_programs,
        BLOCK_M=block_m,
        BLOCK_N=stage1_block_n,
        BLOCK_K=stage1_block_k,
        num_warps=4 if block_m == 16 else 8,
        num_stages=3,
        **backend_options,
    )
    _stage2_kernel[(stage2_programs,)](
        intermediate,
        intermediate_scale,
        w2_data,
        w.w2_weight_scale,
        w.w2_weight if w2_bias is None else w2_bias,
        route_output,
        expert_route_ids,
        expert_counts,
        num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        UPCAST_WEIGHTS=upcast_weights,
        HAS_BIAS=has_bias,
        NUM_PROGRAMS=stage2_programs,
        BLOCK_M=block_m,
        BLOCK_N=stage2_block_n,
        BLOCK_K=stage2_block_k,
        num_warps=4 if block_m == 16 else 8,
        num_stages=3,
        **backend_options,
    )
    _combine(route_output, topk_weights, output)
    return output


# ===-----------------------------------------------------------------------===#
# Kernel Registry
# ===-----------------------------------------------------------------------===#


def triton_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Validate that the module retains the linear OCP MXFP4 weight layout."""
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    if any(not hasattr(w, name) for name in names):
        raise ValueError("linear MXFP4 MoE weights are incomplete")
    tensors = tuple(getattr(w, name) for name in names)
    if any(t.dtype != torch.uint8 for t in tensors):
        raise TypeError("MXFP4 packed values and scales must use torch.uint8")
    if any(not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise ValueError("linear MXFP4 weights must be contiguous GPU tensors")
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("linear MXFP4 weights must be rank-3")
    if len({t.shape[0] for t in tensors}) != 1:
        raise ValueError("linear MXFP4 weights must share an expert axis")
    layout = getattr(w, "w13_input_layout", "concatenated")
    if layout not in {
        "concatenated",
        "interleaved",
    }:
        raise ValueError("Triton MXFP4 MoE received an invalid gate/up weight layout")
    if layout == "interleaved" and plan.get("internal_activation_dtype") != "input":
        raise ValueError("dynamic MXFP4 MoE requires concatenated gate/up weights")
    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "situ", "swiglu"}:
        raise ValueError(f"Triton MXFP4 MoE does not support {activation!r}")


@register_kernel(
    "moe",
    "apply",
    name="triton_mxfp4_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_mxfp4_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"amd", "nvidia"})),
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32, 128}),
        "internal_activation_dtype": frozenset({"input", "mxfp4"}),
        "supports_bias": frozenset({False, True}),
    },
    priority=Priority.PORTABLE,
)
def triton_mxfp4_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply experts with dynamic MXFP4 activations or weight-only MXFP4.

    Args:
        plan: MoE plan selecting standard SiLU/SwiGLU or SiTU activation.
        x: Contiguous FP16/BF16 hidden states `[tokens, hidden]`.
        w: Module holding linear packed E2M1 `w13_weight`/`w2_weight` and
            E8M0 `w13_weight_scale`/`w2_weight_scale` tensors.
        router_logits: Unused because routing must be precomputed.
        topk_weights: Route weights `[tokens, top_k]`.
        topk_ids: Expert ids `[tokens, top_k]`. Out-of-range ids contribute zero.
        num_tokens_global: Unused; distributed expert parallelism is unsupported.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized hidden states `[tokens, hidden]` with dtype matching `x`.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    return _moe(
        plan,
        x,
        w,
        topk_weights,
        topk_ids,
        do_finalize,
    )
