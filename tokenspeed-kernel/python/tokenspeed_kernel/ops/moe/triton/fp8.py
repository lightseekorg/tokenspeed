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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe.triton._common import (
    _combine,
    _num_programs,
    _prepare_routed_output,
    _swiglu,
    _swiglu_params,
    _validate_launch,
    _validate_topk,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_BLOCK = 128


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
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Triton FP8 MoE does not support activation {activation!r}")
    limit = getattr(getattr(w, "swiglu_arg", None), "limit", None)
    if limit is not None and limit <= 0:
        raise ValueError("SwiGLU limit must be positive")
    if getattr(w, "w13_input_layout", "concatenated") != "concatenated":
        raise ValueError("Triton MoE requires concatenated gate/up weights")

    w13 = w.w13_weight
    w2 = w.w2_weight
    w13_scale = w.w13_weight_scale_inv
    w2_scale = w.w2_weight_scale_inv
    weights = (w13, w2, w13_scale, w2_scale)
    if x.ndim != 2 or any(t.ndim != 3 for t in weights):
        raise ValueError("x and block-FP8 MoE weights must be rank-2/rank-3")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must use torch.bfloat16")
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError("w13_weight and w2_weight must use torch.float8_e4m3fn")
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise TypeError("block-FP8 inverse scales must use torch.float32")
    if not all(t.is_cuda and t.is_contiguous() for t in (x, *weights)):
        raise ValueError("x, weights, and scales must be contiguous GPU tensors")
    _validate_topk(x, topk_weights, topk_ids)
    if any(t.device != x.device for t in weights):
        raise ValueError("x and weights must be on the same device")

    hidden_size = x.shape[1]
    num_experts, twice_intermediate_size, weight_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    if num_experts == 0:
        raise ValueError("block-FP8 MoE requires at least one expert")
    if twice_intermediate_size % 2 or weight_hidden_size != hidden_size:
        raise ValueError("w13_weight has an incompatible shape")
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("w2_weight has an incompatible shape")
    if hidden_size % _FP8_BLOCK or intermediate_size % _FP8_BLOCK:
        raise ValueError(
            f"hidden and intermediate sizes must be multiples of {_FP8_BLOCK}"
        )
    hidden_blocks = hidden_size // _FP8_BLOCK
    intermediate_blocks = intermediate_size // _FP8_BLOCK
    if w13_scale.shape != (num_experts, 2 * intermediate_blocks, hidden_blocks):
        raise ValueError("w13_weight_scale_inv has an incompatible shape")
    if w2_scale.shape != (num_experts, hidden_blocks, intermediate_blocks):
        raise ValueError("w2_weight_scale_inv has an incompatible shape")
    return topk_weights, topk_ids


@triton.jit
def _stage1_kernel(
    x_ptr,
    w13_ptr,
    w13_scale_ptr,
    inter_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    swiglu_alpha: tl.constexpr,
    swiglu_limit: tl.constexpr,
    swiglu_beta: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(SCALE_BLOCK % BLOCK_K == 0)
    route_count = num_tokens * top_k
    scale_k = hidden_size // SCALE_BLOCK
    up_scale_offset = intermediate_size // SCALE_BLOCK * scale_k
    tile_idx = tl.program_id(0)
    problem_start = 0

    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(intermediate_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles
        expert_weight = w13_ptr + expert_id.to(tl.int64) * (
            2 * intermediate_size * hidden_size
        )
        expert_scale = w13_scale_ptr + expert_id * 2 * up_scale_offset

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
            gate_rows = n_offset + tl.arange(0, BLOCK_N)
            up_rows = intermediate_size + gate_rows
            gate_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
            up_scale = gate_scale + up_scale_offset
            gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, hidden_size, BLOCK_K):
                k_cols = k_offset + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr + token_ids[:, None] * hidden_size + k_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                gate = tl.load(
                    expert_weight + gate_rows[:, None] * hidden_size + k_cols[None, :]
                ).to(tl.bfloat16)
                up = tl.load(
                    expert_weight + up_rows[:, None] * hidden_size + k_cols[None, :]
                ).to(tl.bfloat16)
                scale_col = k_offset // SCALE_BLOCK
                gate_acc += tl.dot(x, gate.T) * tl.load(gate_scale + scale_col)
                up_acc += tl.dot(x, up.T) * tl.load(up_scale + scale_col)

            activated = _swiglu(
                gate_acc,
                up_acc,
                swiglu_alpha,
                swiglu_limit,
                swiglu_beta,
                HAS_LIMIT,
            ).to(tl.bfloat16)
            inter_offsets = route_ids[:, None] * intermediate_size + gate_rows[None, :]
            tl.store(inter_ptr + inter_offsets, activated, mask=row_mask[:, None])
            tile_idx += num_programs

        problem_start += problem_tiles


@triton.jit
def _stage2_kernel(
    inter_ptr,
    w2_ptr,
    w2_scale_ptr,
    route_output_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(SCALE_BLOCK % BLOCK_N == 0)
    tl.static_assert(SCALE_BLOCK % BLOCK_K == 0)
    route_count = num_tokens * top_k
    scale_k = intermediate_size // SCALE_BLOCK
    tile_idx = tl.program_id(0)
    problem_start = 0

    for expert_id in range(num_experts):
        group_m = tl.load(expert_counts_ptr + expert_id)
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)
        num_n_tiles = tl.cdiv(hidden_size, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles
        expert_weight = w2_ptr + expert_id.to(tl.int64) * (
            hidden_size * intermediate_size
        )
        expert_scale = w2_scale_ptr + expert_id * (hidden_size // SCALE_BLOCK * scale_k)

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
            weight_scale = expert_scale + n_offset // SCALE_BLOCK * scale_k
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_offset in range(0, intermediate_size, BLOCK_K):
                k_cols = k_offset + tl.arange(0, BLOCK_K)
                intermediate = tl.load(
                    inter_ptr
                    + route_ids[:, None] * intermediate_size
                    + k_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                weight = tl.load(
                    expert_weight
                    + weight_rows[:, None] * intermediate_size
                    + k_cols[None, :]
                ).to(tl.bfloat16)
                acc += tl.dot(intermediate, weight.T) * tl.load(
                    weight_scale + k_offset // SCALE_BLOCK
                )

            output_offsets = route_ids[:, None] * hidden_size + weight_rows[None, :]
            tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])
            tile_idx += num_programs

        problem_start += problem_tiles


def _moe(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w.w13_weight.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_tokens == 0:
        return torch.empty_like(x)

    swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_params(w)
    expert_route_ids, expert_counts, route_output, output = _prepare_routed_output(
        x, topk_ids, num_experts
    )
    route_count = num_tokens * top_k
    intermediate = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=x.dtype
    )
    block_m = 16 if num_tokens <= 16 else 64
    block_n = 32
    num_warps = 4 if block_m == 16 else 8
    stage1_programs = _num_programs(x.device, route_count, intermediate_size, block_n)
    stage2_programs = _num_programs(x.device, route_count, hidden_size, block_n)

    _stage1_kernel[(stage1_programs,)](
        x,
        w.w13_weight,
        w.w13_weight_scale_inv,
        intermediate,
        expert_route_ids,
        expert_counts,
        num_tokens,
        stage1_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=1.0 if swiglu_limit is None else swiglu_limit,
        swiglu_beta=swiglu_beta,
        HAS_LIMIT=swiglu_limit is not None,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_FP8_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _stage2_kernel[(stage2_programs,)](
        intermediate,
        w.w2_weight,
        w.w2_weight_scale_inv,
        route_output,
        expert_route_ids,
        expert_counts,
        num_tokens,
        stage2_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        SCALE_BLOCK=_FP8_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_FP8_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _combine(route_output, topk_weights, output)
    return output


# ===-----------------------------------------------------------------------===#
# Kernel Registry
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_block_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd", "nvidia"}),
        vendor_min_arch_versions={
            "amd": ArchVersion(9, 5),
            "nvidia": ArchVersion(8, 9),
        },
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({_FP8_BLOCK}),
        "hidden_alignment": frozenset({_FP8_BLOCK}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(_FP8_BLOCK, _FP8_BLOCK)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_block_precomputed_moe_apply(
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
    """Apply 128x128 block-scaled E4M3 experts to BF16 activations.

    Weight tiles are upcast to BF16 exactly and each block's FP32 inverse scale
    multiplies its FP32 partial product, so neither the weights nor the
    activations are requantized.

    Args:
        plan: MoE plan selecting SiLU/SwiGLU activation. A `swiglu_arg` limit,
            alpha, or `swiglu_beta` on `w` applies to either name.
        x: Contiguous BF16 hidden states `[tokens, hidden]`.
        w: Module with contiguous E4M3 `w13_weight` `[E, 2I, H]` and
            `w2_weight` `[E, H, I]`, plus FP32 `w13_weight_scale_inv`
            `[E, 2I/128, H/128]` and `w2_weight_scale_inv` `[E, H/128, I/128]`.
        router_logits: Unused because routing must be precomputed.
        topk_weights: Route weights `[tokens, top_k]`.
        topk_ids: Expert ids `[tokens, top_k]`. Out-of-range ids contribute zero.
        num_tokens_global: Unused; distributed expert parallelism is unsupported.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized BF16 hidden states `[tokens, hidden]`.
    """
    topk_weights, topk_ids = _validate(plan, x, w, topk_weights, topk_ids, do_finalize)
    return _moe(x, w, topk_weights, topk_ids)
