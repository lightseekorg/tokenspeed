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

"""Triton MoE for FP8 experts with one FP32 scale per output channel.

The ``fp8_channel`` weights (compressed-tensors FP8 dynamic W8A8) run on the
BF16 Triton MoE's pipeline (``bf16.py``): routing, an FC1 stage with the gated
SiLU in its epilogue, an FC2 stage and the FP32 route combine. Activations
are quantized dynamically to FP8 E4M3 with one FP32 scale per row: the FC1
input per token and the FC2 input per route (token and expert) over this
rank's intermediate partition. The rounding points follow the scheme:

- quantize: ``s = max|row| / 448`` and ``q = RNE_sat(row / s)``, both FP32
  divisions correctly rounded; an all-zero row gets the smallest normal scale.
- FC1: ``s_x * s_w13 * sum(q_x * q_w13)`` with FP32 accumulation; gate and up
  are rounded to BF16, ``silu(gate) * up`` is computed in FP32 and rounded to
  BF16.
- FC2: ``s_act * s_w2 * sum(q_act * q_w2)`` with FP32 accumulation, rounded to
  BF16 per route.
- combine: FP32 route weights times the BF16 route outputs, accumulated in
  FP32 and rounded to BF16 once.

The FC2-input quantization is a separate row kernel: a route's full
intermediate width per rank need not fit one power-of-two FC1 tile, and one
tile per route would cut the FC1 parallelism of small batches.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import TensorDescriptor, tl, triton
from tokenspeed_kernel.ops.moe.triton._common import (
    _combine,
    _prepare_routed_output,
    _validate_topk,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_E4M3_MAX = tl.constexpr(448.0)
# Smallest normal FP32: the scale of an all-zero row, whose codes are all zero.
_MIN_SCALE = tl.constexpr(2.0**-126)


@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    expert_ids_ptr,
    num_experts,
    num_cols: tl.constexpr,
    CHECK_EXPERT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if CHECK_EXPERT:
        # Routes of other ranks' experts or invalid ids have no FC1 output.
        expert_id = tl.load(expert_ids_ptr + row)
        if (expert_id < 0) | (expert_id >= num_experts):
            return
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_cols
    values = tl.load(x_ptr + row * num_cols + offsets, mask=mask, other=0.0)
    values = values.to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(tl.math.div_rn(amax, _E4M3_MAX), _MIN_SCALE)
    scales = tl.zeros((BLOCK,), dtype=tl.float32) + scale
    quantized = tl.math.div_rn(values, scales).to(tl.float8e4nv)
    tl.store(q_ptr + row * num_cols + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row, scale)


def _quantize_rows(
    x: torch.Tensor, expert_ids: torch.Tensor | None = None, num_experts: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row of ``x`` to FP8 E4M3 with one FP32 scale.

    With ``expert_ids`` (one int32 per row), rows whose id lies outside
    ``[0, num_experts)`` are skipped: no codes and no scale are written.
    """
    rows, cols = x.shape
    quantized = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(cols)
    _quantize_rows_kernel[(rows,)](
        x,
        quantized,
        scales,
        x if expert_ids is None else expert_ids,
        num_experts,
        num_cols=cols,
        CHECK_EXPERT=expert_ids is not None,
        BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return quantized, scales


@triton.jit
def _expert_tiles(
    expert_counts_ptr,
    num_experts: tl.constexpr,
    EXPERTS: tl.constexpr,
    num_n_tiles: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each expert's route count, tile count and running tile end, in registers.

    Locating a tile's expert from these vectors costs a few reductions instead
    of a dependent load per expert in every program.
    """
    experts = tl.arange(0, EXPERTS)
    counts = tl.load(expert_counts_ptr + experts, mask=experts < num_experts, other=0)
    tiles = tl.cdiv(counts, BLOCK_M) * num_n_tiles
    return experts, counts, tiles, tl.cumsum(tiles, axis=0)


@triton.jit
def _locate_tile(tile_idx, experts, counts, tiles, tile_ends):
    expert_id = tl.sum((tile_ends <= tile_idx).to(tl.int32), axis=0)
    this_expert = experts == expert_id
    first_tile = tl.sum(tl.where(this_expert, tile_ends - tiles, 0), axis=0)
    group_m = tl.sum(tl.where(this_expert, counts, 0), axis=0)
    return expert_id, tile_idx - first_tile, group_m


@triton.jit
def _fc1_kernel(
    xq_ptr,
    x_desc,
    x_scale_ptr,
    w13_desc,
    w13_scale_ptr,
    act_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    EXPERTS: tl.constexpr,
    top_k: tl.constexpr,
    USE_TMA_GATHER: tl.constexpr,
    MAX_IMPRECISE_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_count = num_tokens * top_k
    num_n_tiles: tl.constexpr = intermediate_size // BLOCK_N
    experts, counts, tiles, tile_ends = _expert_tiles(
        expert_counts_ptr, num_experts, EXPERTS, num_n_tiles, BLOCK_M
    )
    for tile_idx in range(tl.program_id(0), tl.sum(tiles, axis=0), num_programs):
        expert_id, tile_in_problem, group_m = _locate_tile(
            tile_idx, experts, counts, tiles, tile_ends
        )
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
        offs_k = tl.arange(0, BLOCK_K)
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_offset in range(0, hidden_size, BLOCK_K):
            if USE_TMA_GATHER:
                x = x_desc.gather(token_ids, k_offset)
            else:
                # sm_90 has no TMA gather (tile::gather4 needs sm_100).
                x = tl.load(
                    xq_ptr
                    + token_ids.to(tl.int64)[:, None] * hidden_size
                    + k_offset
                    + offs_k[None, :]
                )
            gate_w = w13_desc.load([expert_id, n_offset, k_offset]).reshape(
                (BLOCK_N, BLOCK_K)
            )
            up_w = w13_desc.load(
                [expert_id, intermediate_size + n_offset, k_offset]
            ).reshape((BLOCK_N, BLOCK_K))
            gate_acc = tl.dot(
                x, gate_w.T, gate_acc, max_num_imprecise_acc=MAX_IMPRECISE_ACC
            )
            up_acc = tl.dot(x, up_w.T, up_acc, max_num_imprecise_acc=MAX_IMPRECISE_ACC)

        offs_n = n_offset + tl.arange(0, BLOCK_N)
        x_scale = tl.load(x_scale_ptr + token_ids)
        scale_row = w13_scale_ptr + expert_id * (2 * intermediate_size)
        gate_scale = tl.load(scale_row + offs_n)
        up_scale = tl.load(scale_row + intermediate_size + offs_n)
        gate = gate_acc * x_scale[:, None] * gate_scale[None, :]
        up = up_acc * x_scale[:, None] * up_scale[None, :]
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        activated = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
        act_offsets = (
            route_ids.to(tl.int64)[:, None] * intermediate_size + offs_n[None, :]
        )
        tl.store(act_ptr + act_offsets, activated, mask=row_mask[:, None])


@triton.jit
def _fc2_kernel(
    act_q_ptr,
    act_scale_ptr,
    w2_desc,
    w2_scale_ptr,
    route_output_ptr,
    expert_route_ids_ptr,
    expert_counts_ptr,
    num_tokens,
    num_programs,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    EXPERTS: tl.constexpr,
    top_k: tl.constexpr,
    MAX_IMPRECISE_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_count = num_tokens * top_k
    num_n_tiles: tl.constexpr = hidden_size // BLOCK_N
    experts, counts, tiles, tile_ends = _expert_tiles(
        expert_counts_ptr, num_experts, EXPERTS, num_n_tiles, BLOCK_M
    )
    for tile_idx in range(tl.program_id(0), tl.sum(tiles, axis=0), num_programs):
        expert_id, tile_in_problem, group_m = _locate_tile(
            tile_idx, experts, counts, tiles, tile_ends
        )
        tile_m = tile_in_problem // num_n_tiles
        tile_n = tile_in_problem % num_n_tiles
        local_rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = local_rows < group_m
        route_ids = tl.load(
            expert_route_ids_ptr + expert_id * route_count + local_rows,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        n_offset = tile_n * BLOCK_N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_offset in range(0, intermediate_size, BLOCK_K):
            act = tl.load(
                act_q_ptr
                + route_ids[:, None] * intermediate_size
                + k_offset
                + offs_k[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            weight = w2_desc.load([expert_id, n_offset, k_offset]).reshape(
                (BLOCK_N, BLOCK_K)
            )
            acc = tl.dot(act, weight.T, acc, max_num_imprecise_acc=MAX_IMPRECISE_ACC)

        offs_n = n_offset + tl.arange(0, BLOCK_N)
        act_scale = tl.load(act_scale_ptr + route_ids, mask=row_mask, other=0.0)
        w_scale = tl.load(w2_scale_ptr + expert_id * hidden_size + offs_n)
        output = acc * act_scale[:, None] * w_scale[None, :]
        output_offsets = route_ids[:, None] * hidden_size + offs_n[None, :]
        tl.store(
            route_output_ptr + output_offsets,
            output.to(tl.bfloat16),
            mask=row_mask[:, None],
        )


def _largest_block(size: int, candidates: tuple[int, ...]) -> int:
    return next(block for block in candidates if size % block == 0)


def _launch_config(
    num_tokens: int, hidden_size: int, intermediate_size: int
) -> dict[str, int]:
    """Tile sizes and launch options of the FC1 and FC2 kernels.

    Tuned on a GB200 for 256 experts with top-8 routing, where
    ``num_tokens / 32`` is an expert's mean route count: 16-row tiles up to
    16 routes per expert, 64-row tiles above. Other shapes run correctly with
    these tiles but were not tuned.
    ``*_ctas_per_sm`` sizes each persistent grid in programs per SM (capped by
    the tile count), so co-resident programs hide each other's load latency.
    """
    # A whole expert per rank (EP) has a long FC2 reduction: wider FC2 K tiles.
    wide = intermediate_size >= 2048
    k1 = _largest_block(hidden_size, (256, 128))
    n2 = _largest_block(hidden_size, (256, 128))
    n1 = _largest_block(intermediate_size, (64, 32))
    n1_wide = _largest_block(intermediate_size, (128, 64, 32))
    k2 = _largest_block(intermediate_size, (64, 32))
    k2_mid = _largest_block(intermediate_size, (128, 64, 32))
    k2_wide = _largest_block(intermediate_size, (256, 128, 64, 32))
    # fc1 / fc2: block_m, block_n, block_k, warps, stages, ctas_per_sm.
    if num_tokens <= 4:
        fc1 = (16, 32, k1, 4, 5 if wide else 4, 2)
        fc2 = (16, 64, k2_wide, 4, 4, 3) if wide else (16, 128, k2_mid, 4, 3, 3)
    elif num_tokens <= 16:
        fc1 = (16, n1_wide, k1, 4, 4, 3) if wide else (16, n1, k1, 4, 3, 3)
        fc2 = (16, 128, k2_wide, 4, 3, 4) if wide else (16, n2, k2, 4, 3, 4)
    elif num_tokens <= 128:
        fc1 = (16, n1, k1, 4, 4, 4)
        fc2 = (16, n2, k2, 4, 3, 4)
    elif num_tokens <= 512:
        fc1 = (16, n1, 128, 4, 4, 3)
        fc2 = (16, n2, k2, 4, 5, 3)
    elif num_tokens <= 1024:
        fc1 = (64, n1, k1, 8, 4, 2)
        fc2 = (64, n2, k2_mid, 8, 5, 2) if wide else (64, n2, k2, 4, 5, 2)
    else:
        fc1 = (64, n1_wide, 128, 8, 5 if n1_wide == 128 else 4, 2)
        fc2 = (64, n2, k2, 4, 5, 2)
    keys = ("block_m", "block_n", "block_k", "warps", "stages", "ctas_per_sm")
    return {
        **{f"fc1_{key}": value for key, value in zip(keys, fc1)},
        **{f"fc2_{key}": value for key, value in zip(keys, fc2)},
    }


def _moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_offset: int,
    use_tma_gather: bool | None = None,
) -> torch.Tensor:
    num_tokens, hidden_size = x.shape
    num_experts, twice_intermediate_size, _ = w13.shape
    intermediate_size = twice_intermediate_size // 2
    top_k = topk_ids.shape[1]
    if num_tokens == 0:
        return torch.empty_like(x)

    # Global ids to this rank's experts; ids outside [0, E) contribute zero.
    local_ids = topk_ids.to(torch.int32)
    if expert_offset:
        local_ids = local_ids - expert_offset
    local_ids = local_ids.contiguous()
    expert_route_ids, expert_counts, route_output, output = _prepare_routed_output(
        x, local_ids, num_experts
    )
    route_count = num_tokens * top_k

    device = torch.cuda.get_device_properties(x.device)
    if use_tma_gather is None:
        use_tma_gather = device.major == 10
    # sm_90 FP8 wgmma accumulates imprecisely by default; promote every block.
    promote_acc = device.major == 9
    config = _launch_config(num_tokens, hidden_size, intermediate_size)
    fc1_block_n, fc1_block_k = config["fc1_block_n"], config["fc1_block_k"]
    fc2_block_n, fc2_block_k = config["fc2_block_n"], config["fc2_block_k"]
    fc1_programs = min(
        config["fc1_ctas_per_sm"] * device.multi_processor_count,
        route_count * triton.cdiv(intermediate_size, fc1_block_n),
    )
    fc2_programs = min(
        config["fc2_ctas_per_sm"] * device.multi_processor_count,
        route_count * triton.cdiv(hidden_size, fc2_block_n),
    )

    x_q, x_scale = _quantize_rows(x)
    activated = torch.empty(
        (route_count, intermediate_size), device=x.device, dtype=torch.bfloat16
    )
    x_desc = TensorDescriptor.from_tensor(x_q, [1, fc1_block_k])
    w13_desc = TensorDescriptor.from_tensor(w13, [1, fc1_block_n, fc1_block_k])
    w2_desc = TensorDescriptor.from_tensor(w2, [1, fc2_block_n, fc2_block_k])

    _fc1_kernel[(fc1_programs,)](
        x_q,
        x_desc,
        x_scale,
        w13_desc,
        w13_scale,
        activated,
        expert_route_ids,
        expert_counts,
        num_tokens,
        fc1_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        EXPERTS=triton.next_power_of_2(num_experts),
        top_k=top_k,
        USE_TMA_GATHER=use_tma_gather,
        MAX_IMPRECISE_ACC=fc1_block_k if promote_acc else None,
        BLOCK_M=config["fc1_block_m"],
        BLOCK_N=fc1_block_n,
        BLOCK_K=fc1_block_k,
        num_warps=config["fc1_warps"],
        num_stages=config["fc1_stages"],
    )
    act_q, act_scale = _quantize_rows(activated, local_ids.view(-1), num_experts)
    _fc2_kernel[(fc2_programs,)](
        act_q,
        act_scale,
        w2_desc,
        w2_scale,
        route_output,
        expert_route_ids,
        expert_counts,
        num_tokens,
        fc2_programs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        EXPERTS=triton.next_power_of_2(num_experts),
        top_k=top_k,
        MAX_IMPRECISE_ACC=fc2_block_k if promote_acc else None,
        BLOCK_M=config["fc2_block_m"],
        BLOCK_N=fc2_block_n,
        BLOCK_K=fc2_block_k,
        num_warps=config["fc2_warps"],
        num_stages=config["fc2_stages"],
    )
    _combine(route_output, topk_weights, output)
    return output


def _validate(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    do_finalize: bool,
) -> int:
    if not do_finalize:
        raise ValueError("Triton FP8 MoE does not support deferred finalization")
    if any(
        getattr(w, name, None) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("Triton FP8 MoE does not support expert bias")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Triton FP8 MoE requires precomputed topk weights and ids")
    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Triton FP8 MoE does not support activation {activation!r}")
    swiglu_arg = getattr(w, "swiglu_arg", None)
    if swiglu_arg is not None and (
        getattr(swiglu_arg, "alpha", None) not in {None, 1.0}
        or getattr(swiglu_arg, "limit", None) is not None
    ):
        raise ValueError("Triton FP8 MoE supports only standard SwiGLU")
    if getattr(w, "swiglu_beta", None) not in {None, 0.0}:
        raise ValueError("Triton FP8 MoE supports only standard SwiGLU")
    if getattr(w, "w13_input_layout", "concatenated") != "concatenated":
        raise ValueError("Triton FP8 MoE requires concatenated gate/up weights")

    w13, w13_scale = w.w13_weight, w.w13_weight_scale
    w2, w2_scale = w.w2_weight, w.w2_weight_scale
    tensors = (x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError(
            "x, weights, scales and top-k tensors must be contiguous GPU tensors"
        )
    if any(t.device != x.device for t in tensors):
        raise ValueError("x, weights, scales and top-k tensors must share a device")
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise TypeError("x must be a rank-2 torch.bfloat16 tensor")
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError("w13_weight and w2_weight must use torch.float8_e4m3fn")
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise TypeError("FP8 channel weight scales must use torch.float32")
    _validate_topk(x, topk_weights, topk_ids)
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must use torch.float32")

    hidden_size = x.shape[1]
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("w13_weight and w2_weight must be rank-3")
    num_experts, twice_intermediate_size, weight_hidden_size = w13.shape
    intermediate_size = twice_intermediate_size // 2
    if num_experts == 0:
        raise ValueError("FP8 channel MoE requires at least one expert")
    if twice_intermediate_size % 2 or weight_hidden_size != hidden_size:
        raise ValueError("w13_weight has an incompatible shape")
    if w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("w2_weight has an incompatible shape")
    if w13_scale.shape != (num_experts, twice_intermediate_size):
        raise ValueError("w13_weight_scale must have shape [experts, 2 * ispp]")
    if w2_scale.shape != (num_experts, hidden_size):
        raise ValueError("w2_weight_scale must have shape [experts, hidden]")
    if hidden_size % 128 or intermediate_size % 32:
        raise ValueError(
            "hidden size must be a multiple of 128 and intermediate size of 32"
        )
    ep_size = int(getattr(w, "ep_size", 1))
    ep_rank = int(getattr(w, "ep_rank", 0))
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank {ep_rank} is outside ep_size {ep_size}")
    return ep_rank * num_experts


# ===-----------------------------------------------------------------------===#
# Kernel Registry
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "apply",
    name="triton_fp8_channel_precomputed_moe_apply",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8_channel"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({32}),
        "hidden_alignment": frozenset({128}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_fp8_channel_precomputed_moe_apply(
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
    """Apply FP8 per-output-channel experts with dynamic per-token activations.

    Args:
        plan: MoE plan for standard SiLU/SwiGLU. The rounding points are
            those of the module docstring; fold any routed scale into
            ``topk_weights``.
        x: Contiguous BF16 hidden states ``[tokens, hidden]``.
        w: Module with contiguous FP8 E4M3 ``w13_weight`` ``[E, 2I, H]`` (gate
            rows, then up rows), FP32 ``w13_weight_scale`` ``[E, 2I]``, FP8
            E4M3 ``w2_weight`` ``[E, H, I]`` and FP32 ``w2_weight_scale``
            ``[E, H]``; ``E`` is the local expert count. Under expert
            parallelism ``ep_rank`` places the local experts at global ids
            ``[ep_rank * E, (ep_rank + 1) * E)``.
        router_logits: Unused because routing must be precomputed.
        topk_weights: FP32 route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``. Ids outside this
            rank's experts contribute zero, so each rank's output sums its
            local routes only.
        num_tokens_global: Unused; there is no all-to-all.
        max_num_tokens_per_gpu: Unused token-capacity hint.
        do_finalize: Must be true.
        enable_pdl: Unused launch hint.

    Returns:
        Finalized BF16 hidden states ``[tokens, hidden]``.
    """
    expert_offset = _validate(plan, x, w, topk_weights, topk_ids, do_finalize)
    return _moe(
        x,
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        topk_weights,
        topk_ids,
        expert_offset,
    )
