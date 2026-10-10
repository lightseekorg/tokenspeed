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
    _routing,
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


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    route_ids_ptr,
    counts_ptr,
    num_routes,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    INPUT_TOP_K: tl.constexpr,
    GATED: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    NUM_PROGRAMS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Match the checkpoint's K scale block exactly; scale each partial dot
    # before accumulating, as in the portable block-FP8 dense GEMM.
    BLOCK_K: tl.constexpr = 128
    WEIGHT_N: tl.constexpr = 2 * N if GATED else N
    tile_idx = tl.program_id(0)
    problem_start = 0
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    for expert in range(NUM_EXPERTS):
        count = tl.load(counts_ptr + expert)
        n_tiles = tl.cdiv(N, BLOCK_N)
        problem_tiles = tl.cdiv(count, BLOCK_M) * n_tiles
        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            local_tile = tile_idx - problem_start
            rows = local_tile // n_tiles * BLOCK_M + tl.arange(0, BLOCK_M)
            cols = local_tile % n_tiles * BLOCK_N + tl.arange(0, BLOCK_N)
            routes = tl.load(
                route_ids_ptr + expert * num_routes + rows,
                mask=rows < count,
                other=0,
            )
            input_rows = routes // INPUT_TOP_K
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            if GATED:
                up_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for block in range(K // BLOCK_K):
                ks = block * BLOCK_K + tl.arange(0, BLOCK_K)
                a = tl.load(
                    a_ptr + input_rows[:, None] * K + ks[None, :],
                    mask=rows[:, None] < count,
                    other=0.0,
                )
                weight_offsets = (
                    expert.to(tl.int64) * WEIGHT_N * K + cols[:, None] * K + ks[None, :]
                )
                scale_offsets = (
                    expert * (WEIGHT_N // 128) * (K // 128)
                    + cols // 128 * (K // 128)
                    + block
                )
                # Every finite E4M3 value is exact in BF16. Only this loaded
                # tile is converted; no dequantized expert copy is allocated.
                weight = tl.load(weight_ptr + weight_offsets).to(tl.bfloat16)
                scale = tl.load(scale_ptr + scale_offsets)
                acc += tl.dot(a, weight.T) * scale[None, :]
                if GATED:
                    up = tl.load(weight_ptr + weight_offsets + N * K).to(tl.bfloat16)
                    up_scale = tl.load(
                        scale_ptr + scale_offsets + N // 128 * (K // 128)
                    )
                    up_acc += tl.dot(a, up.T) * up_scale[None, :]
            if GATED:
                acc = acc * tl.sigmoid(acc) * up_acc
            tl.store(
                output_ptr + routes[:, None] * N + cols[None, :],
                acc,
                mask=rows[:, None] < count,
            )
            tile_idx += NUM_PROGRAMS
        problem_start += problem_tiles
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def triton_fp8_moe_process_weights(plan: dict, w: torch.nn.Module) -> None:
    """Validate checkpoint block-128 FP8 experts without changing their storage.

    Args:
        plan: Plan returned by ``moe_plan`` for this implementation.
        w: MoE module with concatenated gate/up E4M3 weights, FP32 inverse
            scales, and explicit activation and contiguous EP ownership fields.
    """
    if plan["a2a_backend"] not in (None, "none"):
        raise ValueError("Triton block-FP8 MoE does not own all-to-all communication")
    if plan["internal_activation_dtype"] not in (None, "input"):
        raise ValueError("Triton block-FP8 MoE requires BF16 internal activations")
    if w.activation not in {"silu", "swiglu"} or plan["activation"] not in {
        None,
        w.activation,
    }:
        raise ValueError("Triton block-FP8 MoE requires standard SiLU/SwiGLU")
    if w.swiglu_arg is not None and (
        w.swiglu_arg.alpha not in (None, 1.0) or w.swiglu_arg.limit is not None
    ):
        raise ValueError(
            "Triton block-FP8 MoE does not support generalized/clamped SwiGLU"
        )
    if w.swiglu_beta not in (None, 0.0) or w.w13_input_layout != "concatenated":
        raise ValueError("Triton block-FP8 MoE requires standard concatenated gate/up")
    if any(
        w._parameters.get(name) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("Triton block-FP8 MoE does not support expert bias")
    w13, w2 = w.w13_weight, w.w2_weight
    s13, s2 = w.w13_weight_scale_inv, w.w2_weight_scale_inv
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("block-FP8 expert weights must be rank-3")
    experts, twice_intermediate, hidden = w13.shape
    intermediate = twice_intermediate // 2
    if (
        experts <= 0
        or hidden <= 0
        or intermediate <= 0
        or twice_intermediate % 2
        or hidden % 128
        or intermediate % 128
        or w2.shape != (experts, hidden, intermediate)
    ):
        raise ValueError(
            "invalid expert shapes; hidden and intermediate must be multiples of 128"
        )
    if (
        w.ep_size < 1
        or not 0 <= w.ep_rank < w.ep_size
        or w.num_local_experts != experts
        or w.num_experts != experts * w.ep_size
    ):
        raise ValueError("invalid contiguous expert-parallel ownership")
    if s13.shape != (experts, twice_intermediate // 128, hidden // 128) or s2.shape != (
        experts,
        hidden // 128,
        intermediate // 128,
    ):
        raise ValueError("FP8 inverse scales must have block shape (128, 128)")
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError("expert weights must use torch.float8_e4m3fn")
    if s13.dtype != torch.float32 or s2.dtype != torch.float32:
        raise TypeError("FP8 inverse scales must use torch.float32")
    if not all(
        t.is_cuda and t.is_contiguous() and t.device == w13.device
        for t in (w13, w2, s13, s2)
    ):
        raise ValueError("expert weights and scales must be contiguous on the same GPU")


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_fp8_moe_process_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "expert_id_repeats": frozenset({False, True}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "hidden_alignment": frozenset({128}),
        "ispp_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
        "persistent_workspace": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Apply SM90 block-FP8 experts with BF16 activations and FP32 accumulation.

    Args:
        plan: Execution plan returned by ``moe_plan``.
        x: Contiguous BF16 hidden states ``[tokens, hidden]``.
        w: Module validated by ``triton_fp8_moe_process_weights``.
        router_logits: Unused; precomputed routing is authoritative.
        topk_weights: Floating-point route weights ``[tokens, top_k]``, including
            any normalization and routed scaling factor.
        topk_ids: Int32/int64 global expert IDs ``[tokens, top_k]``. IDs outside
            this rank's contiguous expert range contribute zero, including -1.
        num_tokens_global: Communication hint; the caller owns collectives.
        max_num_tokens_per_gpu: Communication hint; all input rows are processed.
        do_finalize: Must be true; reduce the weighted routes to token order.
        enable_pdl: Whether the grouped GEMM launches may use PDL.

    Returns:
        BF16 states ``[tokens, hidden]``, containing this EP/TP rank's partial
        result. The caller owns any cross-rank reduction.
    """
    if not do_finalize:
        raise ValueError("Triton block-FP8 MoE requires finalization")
    triton_fp8_moe_process_weights(plan, w)
    if topk_ids is None or topk_weights is None:
        raise ValueError(
            "Triton block-FP8 MoE requires precomputed top-k weights and IDs"
        )
    if x.ndim != 2 or x.shape[1] != w.w13_weight.shape[2]:
        raise ValueError("x must have shape [tokens, hidden]")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must use torch.bfloat16")
    if not x.is_contiguous() or x.device != w.w13_weight.device:
        raise ValueError("x must be contiguous on the expert weights' GPU")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != x.shape[0]
        or topk_ids.shape[1] <= 0
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if topk_ids.dtype not in (torch.int32, torch.int64) or topk_weights.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ):
        raise TypeError("top-k IDs must be int32/int64 and weights FP32/BF16/FP16")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("routing tensors must be on x's GPU")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    # The none/allgather path supplies global IDs, not dispatch-local IDs.
    local_ids = topk_ids - w.ep_rank * w.num_local_experts
    routes, counts = _routing(local_ids, w.num_local_experts)
    num_routes = topk_ids.numel()
    intermediate_size = w.w2_weight.shape[2]
    intermediate = torch.empty(
        (num_routes, intermediate_size), device=x.device, dtype=x.dtype
    )
    # Non-local/padding routes are absent from the GEMMs. Keep down-projection
    # partials in FP32 until weighted reduction to avoid an extra BF16 rounding.
    route_output = torch.zeros(
        (num_routes, x.shape[1]), device=x.device, dtype=torch.float32
    )
    output = torch.empty_like(x)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    block_m = 16 if x.shape[0] <= 16 else 32
    for a, weight, scale, out, gated, input_top_k in (
        (
            x,
            w.w13_weight,
            w.w13_weight_scale_inv,
            intermediate,
            True,
            topk_ids.shape[1],
        ),
        (intermediate, w.w2_weight, w.w2_weight_scale_inv, route_output, False, 1),
    ):
        programs = min(num_sms, num_routes * triton.cdiv(out.shape[1], 64))
        _grouped_gemm_kernel[(programs,)](
            a,
            weight,
            scale,
            out,
            routes,
            counts,
            num_routes,
            N=out.shape[1],
            K=a.shape[1],
            NUM_EXPERTS=w.num_local_experts,
            INPUT_TOP_K=input_top_k,
            GATED=gated,
            ENABLE_PDL=enable_pdl,
            NUM_PROGRAMS=programs,
            BLOCK_M=block_m,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    _combine(route_output, topk_weights, output)
    return output
