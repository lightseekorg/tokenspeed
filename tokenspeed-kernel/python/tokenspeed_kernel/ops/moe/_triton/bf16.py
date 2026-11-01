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
from tokenspeed_kernel.ops.moe._triton.common import (
    _combine,
    _routing,
    _validate_launch,
    _validate_topk,
)


def _check_route_tensors(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, topk_ids: torch.Tensor
) -> None:
    """Reject activations, weights and expert ids the route stages cannot launch on."""
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("x and unquantized MoE weights must be rank-2/rank-3")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("x must use torch.float16 or torch.bfloat16")
    if w13.dtype != x.dtype or w2.dtype != x.dtype:
        raise TypeError("x, w13_weight, and w2_weight must have the same dtype")
    if not all(t.is_cuda and t.is_contiguous() for t in (x, w13, w2)):
        raise ValueError("x and weights must be contiguous GPU tensors")
    if topk_ids.ndim != 2:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use torch.int32 or torch.int64")
    if topk_ids.device != x.device:
        raise ValueError("top-k tensors and x must be on the same device")
    if w13.device != x.device or w2.device != x.device:
        raise ValueError("x and weights must be on the same device")

    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, weight_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    if num_experts == 0:
        raise ValueError("unquantized MoE requires at least one expert")
    if topk_ids.shape[0] != num_tokens or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k > 0]")
    if twice_intermediate_size % 2 or weight_hidden_size != hidden_size:
        raise ValueError("w13_weight has an incompatible shape")
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("w2_weight has an incompatible shape")
    if hidden_size % 128 or intermediate_size % 32:
        raise ValueError(
            "hidden size must be a multiple of 128 and intermediate size of 32"
        )


def _validate(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_launch(w, topk_weights, topk_ids, do_finalize)
    if any(
        getattr(w, name, None) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("Triton MoE does not support expert bias")

    activation = plan.get("activation") or getattr(w, "activation", "silu")
    precision_policy = plan.get("precision_policy", "default")
    if precision_policy not in {"default", "bf16_route_accum_fp32"}:
        raise ValueError(f"Unsupported MoE precision policy: {precision_policy!r}")
    if precision_policy == "bf16_route_accum_fp32":
        if (
            x.dtype != torch.bfloat16
            or topk_weights.dtype != torch.float32
            or activation != "silu"
        ):
            raise ValueError(
                "bf16_route_accum_fp32 requires BF16 SiLU inputs and FP32 route weights"
            )
        if not 0.0 < float(plan["output_scale"]) < float("inf"):
            raise ValueError("MoE output_scale must be finite and positive")
    elif plan.get("output_scale", 1.0) != 1.0:
        raise ValueError("MoE output_scale requires an explicit precision policy")
    if activation not in {"silu", "situ", "swiglu"}:
        raise ValueError(f"Triton MoE does not support activation {activation!r}")
    swiglu_arg = getattr(w, "swiglu_arg", None)
    if swiglu_arg is not None and (
        getattr(swiglu_arg, "alpha", None) not in {None, 1.0}
        or getattr(swiglu_arg, "limit", None) is not None
    ):
        raise ValueError("Triton MoE supports only standard SwiGLU")
    if getattr(w, "swiglu_beta", None) not in {None, 0.0}:
        raise ValueError("Triton MoE supports only standard SwiGLU")
    if getattr(w, "w13_input_layout", "concatenated") != "concatenated":
        raise ValueError("Triton MoE requires concatenated gate/up weights")
    if activation == "situ":
        situ_beta = getattr(w, "activation_situ_beta", None)
        situ_linear_beta = getattr(w, "activation_situ_linear_beta", None)
        if situ_beta is None or situ_beta <= 0:
            raise ValueError("SiTU beta must be positive")
        if situ_linear_beta is not None and situ_linear_beta <= 0:
            raise ValueError("SiTU linear beta must be positive")

    _check_route_tensors(x, w.w13_weight, w.w2_weight, topk_ids)
    _validate_topk(x, topk_weights, topk_ids)
    return topk_weights, topk_ids, activation


@triton.jit
def _stage1_kernel(
    x_ptr,
    w13_ptr,
    inter_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    situ_beta,
    situ_linear_beta,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    ROUND_GEMM_OUTPUTS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_count = num_tokens * top_k
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
            gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, hidden_size, BLOCK_K):
                x = tl.load(
                    x_ptr
                    + token_ids[:, None] * hidden_size
                    + k_offset
                    + tl.arange(0, BLOCK_K)[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                # Advance the base in int64: one expert's w13 stride passes 2^31.
                w13_expert = w13_ptr + expert_id.to(tl.int64) * (
                    2 * intermediate_size * hidden_size
                )
                weight_offsets = (
                    (n_offset + tl.arange(0, BLOCK_N))[:, None] * hidden_size
                    + k_offset
                    + tl.arange(0, BLOCK_K)[None, :]
                )
                gate = tl.load(w13_expert + weight_offsets)
                up = tl.load(
                    w13_expert + weight_offsets + intermediate_size * hidden_size
                )
                gate_acc += tl.dot(x, gate.T)
                up_acc += tl.dot(x, up.T)

            if ACTIVATION == "situ":
                gate = gate_acc.to(inter_ptr.dtype.element_ty).to(tl.float32)
                up = up_acc.to(inter_ptr.dtype.element_ty).to(tl.float32)
                gate = situ_beta * libdevice.tanh(gate / situ_beta) * tl.sigmoid(gate)
                if HAS_LINEAR_BETA:
                    up = situ_linear_beta * libdevice.tanh(up / situ_linear_beta)
                activated = (gate * up).to(inter_ptr.dtype.element_ty)
            else:
                if ROUND_GEMM_OUTPUTS:
                    gate_acc = gate_acc.to(inter_ptr.dtype.element_ty).to(tl.float32)
                    up_acc = up_acc.to(inter_ptr.dtype.element_ty).to(tl.float32)
                activated = (gate_acc * tl.sigmoid(gate_acc) * up_acc).to(
                    inter_ptr.dtype.element_ty
                )
            inter_offsets = (
                route_ids[:, None] * intermediate_size
                + n_offset
                + tl.arange(0, BLOCK_N)[None, :]
            )
            tl.store(inter_ptr + inter_offsets, activated, mask=row_mask[:, None])
            tile_idx += NUM_PROGRAMS

        problem_start += problem_tiles


@triton.jit
def _stage2_kernel(
    inter_ptr,
    w2_ptr,
    route_output_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_count = num_tokens * top_k
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
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, intermediate_size, BLOCK_K):
                intermediate_offsets = (
                    route_ids[:, None] * intermediate_size
                    + k_offset
                    + tl.arange(0, BLOCK_K)[None, :]
                )
                intermediate = tl.load(
                    inter_ptr + intermediate_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                # Advance the base in int64: one expert's w2 stride passes 2^31.
                w2_expert = w2_ptr + expert_id.to(tl.int64) * (
                    hidden_size * intermediate_size
                )
                weight_offsets = (
                    (n_offset + tl.arange(0, BLOCK_N))[:, None] * intermediate_size
                    + k_offset
                    + tl.arange(0, BLOCK_K)[None, :]
                )
                weight = tl.load(w2_expert + weight_offsets)
                acc += tl.dot(intermediate, weight.T)

            output_offsets = (
                route_ids[:, None] * hidden_size
                + n_offset
                + tl.arange(0, BLOCK_N)[None, :]
            )
            tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])
            tile_idx += NUM_PROGRAMS

        problem_start += problem_tiles


def _deep_pipeline(device: torch.device, block_m: int) -> bool:
    """Whether the stages may pipeline four slabs per CTA at this tile height."""
    from tokenspeed_kernel.platform import current_platform

    if device.type != "cuda" or not current_platform().is_nvidia:
        return False
    # Four stages of block_m x 128 activations plus 2x64x128 gate/up (or 128x128 w2) BF16 tiles, and barriers.
    smem_bytes = 4 * (block_m * 128 * 2 + 2 * 64 * 128 * 2) + 1024
    properties = torch.cuda.get_device_properties(device)
    return properties.shared_memory_per_block_optin >= smem_bytes


def _route_outputs(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    situ_beta: float,
    situ_linear_beta: float | None,
    expert_start: int,
    round_intermediates: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w13.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_tokens == 0:
        if output is not None:
            return output.view(0, top_k, hidden_size)
        return torch.empty((0, top_k, hidden_size), dtype=x.dtype, device=x.device)

    expert_route_ids, expert_counts = _routing(topk_ids, num_experts, expert_start)
    route_count = num_tokens * top_k
    intermediate = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=x.dtype
    )
    # Invalid expert ids are absent from routing; zero their canonical rows.
    if output is None:
        route_output = torch.zeros(
            (route_count, hidden_size), device=x.device, dtype=x.dtype
        )
    else:
        route_output = output.view(route_count, hidden_size)
        route_output.zero_()
    if intermediate_size % 128 == 0 and _deep_pipeline(x.device, 16):
        # Cold B300: deep 16-row tiles win up to 32 routes (counted over every rank) per local expert.
        block_m = 16 if route_count <= 32 * num_experts else 64
    else:
        block_m = 16 if num_tokens <= 16 else 64
    stage1_block_n = 64 if intermediate_size % 64 == 0 else 32
    stage2_block_n = 128
    block_k = 64 if intermediate_size % 64 == 0 else 32
    num_stages, num_warps, programs_per_sm = 3, 4 if block_m == 16 else 8, 1
    if intermediate_size % 128 == 0 and _deep_pipeline(x.device, block_m):
        # Wider K slabs keep cold decode weights in flight; K accumulation order is unchanged.
        block_k, num_stages, num_warps, programs_per_sm = 128, 4, 8, 4
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    stage1_programs = min(
        num_sms * programs_per_sm,
        route_count * triton.cdiv(intermediate_size, stage1_block_n),
    )
    stage2_programs = min(
        num_sms * programs_per_sm,
        route_count * triton.cdiv(hidden_size, stage2_block_n),
    )
    _stage1_kernel[(stage1_programs,)](
        x,
        w13,
        intermediate,
        expert_route_ids,
        expert_counts,
        num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        situ_beta=situ_beta,
        situ_linear_beta=(1.0 if situ_linear_beta is None else situ_linear_beta),
        ACTIVATION=activation,
        HAS_LINEAR_BETA=situ_linear_beta is not None,
        ROUND_GEMM_OUTPUTS=round_intermediates,
        NUM_PROGRAMS=stage1_programs,
        BLOCK_M=block_m,
        BLOCK_N=stage1_block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=not round_intermediates,
    )
    _stage2_kernel[(stage2_programs,)](
        intermediate,
        w2,
        route_output,
        expert_route_ids,
        expert_counts,
        num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        NUM_PROGRAMS=stage2_programs,
        BLOCK_M=block_m,
        BLOCK_N=stage2_block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return route_output.view(num_tokens, top_k, hidden_size)


def _moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    situ_beta: float,
    situ_linear_beta: float | None,
    round_intermediates: bool,
    output_scale: float,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if round_intermediates:
        # Match ascending-expert accumulation, independently of routing's
        # score-sorted output order. Keep the original top-k tensors unchanged.
        topk_ids, order = topk_ids.sort(dim=-1)
        topk_weights = topk_weights.gather(1, order)
    route_output = _route_outputs(
        x,
        w13,
        w2,
        topk_ids,
        activation,
        situ_beta,
        situ_linear_beta,
        0,
        round_intermediates,
    )
    output = torch.empty_like(x)
    _combine(
        route_output,
        topk_weights,
        output,
        None,
        0,
        round_weighted_routes=round_intermediates,
        output_scale=output_scale,
    )
    return output
