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

"""Marlin W4A16 MXFP4 MoE over DeepEP all-to-all (SM90).

The replicated-token Marlin path runs every rank over the full token batch and
all-reduces; with attention DP that additionally needs a cross-DP all-gather.
This kernel replaces both collectives with DeepEP's token-level legs: dispatch
routes each token only to the ranks owning its top-k experts, the local Marlin
grouped GEMM computes those experts' contributions, and combine reduces the
per-expert partials back to the token's home rank.

Activations stay bf16 on the wire (Marlin is weight-only quant; there is no
FP8 activation form to exploit), so dispatch carries a single bf16 tensor.

Both DeepEP modes are supported, and their receive layouts differ:

- Normal (extend-shaped): flat rows with local ``topk_ids`` whose non-local
  slots arrive as -1. Marlin consumes these local IDs directly; route weights
  fold into GEMM2 and combine reduces unweighted.
- Low-latency (decode-shaped): a padded per-expert buffer with device counts.
  The adapter compacts valid rows, directly builds Marlin's aligned block
  schedule, and reuses the receive buffer for the result. Only combine
  applies the real route weights.

Intranode traffic runs the NVLink P2P fast path in both modes; the IBGDA
requirement only bites for internode low-latency traffic.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.communication.deep_ep import DeepEPDispatcher, DeepEPMode
from tokenspeed_kernel.ops.moe.marlin.mxfp4 import (
    MXFP4_BLOCK,
    marlin_mxfp4_local_moe_apply,
    marlin_mxfp4_masked_moe_apply,
    marlin_mxfp4_moe_weights,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def _get_dispatcher(
    plan: dict,
    w: torch.nn.Module,
    x: torch.Tensor,
) -> DeepEPDispatcher:
    """Build (once per plan) the dispatcher owning this layer's DeepEP legs.

    Sizing comes from the plan rather than the live batch: the DeepEP buffer is
    allocated on first use and reused for every later forward.
    """
    dispatcher = plan.get("_deepep_dispatcher")
    if dispatcher is not None:
        return dispatcher

    group = plan.get("deepep_group")
    if group is None:
        raise ValueError("DeepEP MoE plan is missing deepep_group")
    deepep_mode = DeepEPMode(plan.get("deepep_mode") or DeepEPMode.auto.value)
    capacity = plan.get("deepep_low_latency_max_num_tokens_per_gpu")
    if deepep_mode.enable_low_latency() and not capacity:
        raise ValueError(
            f"DeepEP plan with mode {deepep_mode.value} is missing "
            "deepep_low_latency_max_num_tokens_per_gpu"
        )
    config = SimpleNamespace(
        top_k=getattr(w, "top_k"),
        num_experts=getattr(w, "num_experts"),
        low_latency_max_num_tokens_per_gpu=capacity,
        hidden_size=x.shape[1],
        world_size=getattr(w, "ep_size", group.size()),
        group=group,
        params_dtype=torch.bfloat16,
    )
    dispatcher = DeepEPDispatcher(
        config,
        deepep_mode=deepep_mode,
        async_finish=False,
        return_recv_hook=True,
        # bf16 on the wire: Marlin consumes bf16 activations directly, so the
        # FP8 wire cast of the DeepGEMM path would only add quantization error.
        use_fp8=False,
    )
    plan["_deepep_dispatcher"] = dispatcher
    return dispatcher


def marlin_mxfp4_deepep_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack local weights and reserve DeepEP memory before cache sizing.

    Args:
        plan: DeepEP MoE plan including process group, mode, and capacity.
        w: Local expert weights with num_experts/top_k and EP ownership.

    Returns:
        None; weights are repacked in place and the plan owns its dispatcher.
    """
    marlin_mxfp4_moe_weights(plan, w)
    like = torch.empty(
        (0, w._marlin_hidden_size), dtype=torch.bfloat16, device=w.w13_weight.device
    )
    _get_dispatcher(plan, w, like).prepare()


def _apply_normal(
    dispatcher: DeepEPDispatcher,
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Extend-shaped path: flat recv rows, local ids with -1 masking."""
    dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=False)
    # Keep shared work in the same callback contract as low-latency mode.
    # The legacy normal implementation starts its transfer in dispatch_b.
    if overlap_fn is not None:
        overlap_fn()
    (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        _,
        _,
        _,
        _,
    ) = dispatcher.dispatch_b()

    if recv_x.shape[0]:
        expert_out = marlin_mxfp4_local_moe_apply(
            plan, recv_x, w, recv_topk_weights, recv_topk_ids
        )
    else:
        expert_out = torch.zeros_like(recv_x)

    dispatcher.combine_a(
        expert_out, recv_topk_ids, recv_topk_weights, low_latency=False
    )
    return dispatcher.combine_b()


def _apply_low_latency(
    dispatcher: DeepEPDispatcher,
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_global_tokens: int,
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Pack valid device-counted routes; combine applies routing weights."""
    dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=True)
    if overlap_fn is not None:
        overlap_fn()
    recv_x, _, _, _, _, _, masked_m = dispatcher.dispatch_b()
    expert_out = marlin_mxfp4_masked_moe_apply(
        plan, recv_x, w, masked_m, num_global_tokens, topk_ids.shape[1]
    )
    # K3 has no identity expert: its shared experts are computed outside EP.
    # The supported DeepEP 2.x Legacy API has no x_ori combine parameter.
    dispatcher.combine_a(expert_out, topk_ids, topk_weights, low_latency=True)
    return dispatcher.combine_b()


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp4_deepep_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp4_deepep_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({True}),
        "deepep_modes": frozenset({"normal", "low_latency"}),
        "ispp_alignment": frozenset({MXFP4_BLOCK}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def marlin_mxfp4_deepep_moe_apply(
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
    low_latency: bool | None = None,
    overlap_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Run one Marlin MXFP4 MoE layer over DeepEP all-to-all.

    Args:
        plan: Execution plan from ``moe_plan``; owns the DeepEP group, mode,
            low-latency capacity, and the lazily built dispatcher.
        x: ``[local_tokens, hidden]`` bf16 hidden states of this rank's tokens.
        w: Module holding Marlin-repacked expert weights plus ``top_k``,
            ``num_experts``, ``num_local_experts``, ``ep_rank``, ``ep_size``.
        router_logits: Unused; routing is precomputed.
        topk_weights: ``[local_tokens, top_k]`` route weights.
        topk_ids: ``[local_tokens, top_k]`` global expert ids.
        num_tokens_global: Upper bound on unique source token rows across EP;
            sizes compact LL intermediates. Graph callers pass bucket rows.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (combine is the finalize).
        enable_pdl: Unused launch hint.
        low_latency: Which DeepEP legs to run when the plan mode is "auto".
            Every rank of the EP group must pass the same value.
        overlap_fn: Optional work queued inside the dispatch window (e.g. the
            shared experts); it must not read the dispatch result or write
            ``x``.

    Returns:
        ``[local_tokens, hidden]`` bf16 combined MoE output, already reduced
        across the EP group (no outer all-reduce needed).
    """
    del router_logits, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Marlin MXFP4 DeepEP MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP4 DeepEP MoE requires precomputed top-k")

    dispatcher = _get_dispatcher(plan, w, x)
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(torch.float32)

    if dispatcher.deepep_mode.resolve(low_latency) == DeepEPMode.normal:
        return _apply_normal(dispatcher, plan, x, w, topk_weights, topk_ids, overlap_fn)
    # Callers with a model-level token partition pass its unique global row
    # bound. Generic kernel callers may omit it and use the configured bound.
    global_bound = (
        num_tokens_global
        if num_tokens_global is not None
        else plan["deepep_low_latency_max_num_tokens_per_gpu"] * w.ep_size
    )
    return _apply_low_latency(
        dispatcher, plan, x, w, topk_weights, topk_ids, global_bound, overlap_fn
    )
