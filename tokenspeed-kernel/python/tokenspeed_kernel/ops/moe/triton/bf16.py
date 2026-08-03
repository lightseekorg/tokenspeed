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
from tokenspeed_kernel._triton import TensorDescriptor, tl, triton
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _validate_apply_contract(
    plan: dict,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not do_finalize:
        raise ValueError("Triton MoE does not support deferred finalization")
    if int(getattr(w, "ep_size", 1)) != 1:
        raise ValueError("Triton MoE does not support expert parallelism")
    if any(
        getattr(w, name, None) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("Triton MoE does not support expert bias")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Triton MoE requires precomputed topk weights and ids")
    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "swiglu"}:
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
    return topk_weights, topk_ids


@triton.jit
def _stage1_kernel(
    x_desc,
    w13_desc,
    inter_ptr,
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
                x = x_desc.gather(token_ids, k_offset)
                gate = w13_desc.load([expert_id, n_offset, k_offset]).reshape(
                    (BLOCK_N, BLOCK_K)
                )
                up = w13_desc.load(
                    [expert_id, intermediate_size + n_offset, k_offset]
                ).reshape((BLOCK_N, BLOCK_K))
                gate_acc += tl.dot(x, gate.T)
                up_acc += tl.dot(x, up.T)

            activated = (gate_acc * tl.sigmoid(gate_acc) * up_acc).to(x_desc.dtype)
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
    w2_desc,
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
                weight = w2_desc.load([expert_id, n_offset, k_offset]).reshape(
                    (BLOCK_N, BLOCK_K)
                )
                acc += tl.dot(intermediate, weight.T)

            output_offsets = (
                route_ids[:, None] * hidden_size
                + n_offset
                + tl.arange(0, BLOCK_N)[None, :]
            )
            tl.store(route_output_ptr + output_offsets, acc, mask=row_mask[:, None])
            tile_idx += NUM_PROGRAMS

        problem_start += problem_tiles


def _moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("x and unquantized MoE weights must be rank-2/rank-3")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("x must use torch.float16 or torch.bfloat16")
    if w13.dtype != x.dtype or w2.dtype != x.dtype:
        raise TypeError("x, w13_weight, and w2_weight must have the same dtype")
    if not all(t.is_cuda and t.is_contiguous() for t in (x, w13, w2)):
        raise ValueError("x and weights must be contiguous GPU tensors")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use torch.int32 or torch.int64")
    if not topk_weights.is_floating_point():
        raise TypeError("topk_weights must use a floating-point dtype")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and x must be on the same device")
    if w13.device != x.device or w2.device != x.device:
        raise ValueError("x and weights must be on the same device")

    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, weight_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_experts == 0:
        raise ValueError("unquantized MoE requires at least one expert")
    if topk_ids.shape[0] != num_tokens or top_k == 0:
        raise ValueError("top-k tensors must have shape [num_tokens, top_k > 0]")
    if twice_intermediate_size % 2 or weight_hidden_size != hidden_size:
        raise ValueError("w13_weight has an incompatible shape")
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("w2_weight has an incompatible shape")
    if hidden_size % 128 or intermediate_size % 32:
        raise ValueError(
            "hidden size must be a multiple of 128 and intermediate size of 32"
        )
    if num_tokens == 0:
        return torch.empty_like(x)

    expert_route_ids, expert_counts = _routing(topk_ids, num_experts)
    route_count = num_tokens * top_k
    intermediate = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=x.dtype
    )
    # Invalid expert ids are absent from routing; zero their canonical rows.
    route_output = torch.zeros(
        (route_count, hidden_size), device=x.device, dtype=x.dtype
    )
    output = torch.empty_like(x)
    block_m = 16 if num_tokens <= 16 else 64
    stage1_block_n = 64 if intermediate_size % 64 == 0 else 32
    stage2_block_n = 128
    block_k = 64 if intermediate_size % 64 == 0 else 32
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    stage1_programs = min(
        num_sms, route_count * triton.cdiv(intermediate_size, stage1_block_n)
    )
    stage2_programs = min(
        num_sms, route_count * triton.cdiv(hidden_size, stage2_block_n)
    )
    x_desc = TensorDescriptor.from_tensor(x, [1, block_k])
    w13_desc = TensorDescriptor.from_tensor(w13, [1, stage1_block_n, block_k])
    w2_desc = TensorDescriptor.from_tensor(w2, [1, stage2_block_n, block_k])

    _stage1_kernel[(stage1_programs,)](
        x_desc,
        w13_desc,
        intermediate,
        expert_route_ids,
        expert_counts,
        num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        NUM_PROGRAMS=stage1_programs,
        BLOCK_M=block_m,
        BLOCK_N=stage1_block_n,
        BLOCK_K=block_k,
        num_warps=4 if block_m == 16 else 8,
        num_stages=3,
    )
    _stage2_kernel[(stage2_programs,)](
        intermediate,
        w2_desc,
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
        num_warps=4 if block_m == 16 else 8,
        num_stages=3,
    )
    _combine(route_output, topk_weights, output)
    return output


@register_kernel(
    "moe",
    "apply",
    name="triton_bf16_precomputed_moe_apply",
    solution="triton",
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_bf16_precomputed_moe_apply(
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
    """Apply an FP16/BF16 Triton MoE using precomputed top-k routing.

    Args:
        plan: MoE plan selecting standard SiLU/SwiGLU activation.
        x: Contiguous hidden states `[tokens, hidden]` in FP16 or BF16.
        w: Module with contiguous `w13_weight` `[E, 2I, H]` and
            `w2_weight` `[E, H, I]` tensors matching `x.dtype`.
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
    topk_weights, topk_ids = _validate_apply_contract(
        plan, w, topk_weights, topk_ids, do_finalize
    )
    return _moe(x, w.w13_weight, w.w2_weight, topk_weights, topk_ids)
