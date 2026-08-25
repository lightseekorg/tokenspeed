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

- Normal (extend-shaped): flat rows with global ``topk_ids`` whose non-local
  slots arrive as -1; the Marlin apply's global->local remap handles that
  layout as-is, and the route weights fold into GEMM2 so combine reduces
  unweighted.
- Low-latency (decode-shaped): a padded per-expert buffer
  ``[num_local_experts, recv_m, hidden]`` with ``masked_m`` valid rows per
  expert and no ids. Each row belongs to exactly one local expert, so the
  apply flattens the buffer and synthesizes a top-1 routing (weight 1.0);
  the real route weights are applied by the low-latency combine leg.

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
    marlin_mxfp4_moe_weights,
    marlin_mxfp4_precomputed_moe_apply,
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


def _apply_normal(
    dispatcher: DeepEPDispatcher,
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    enable_pdl: bool,
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Extend-shaped path: flat recv rows, global ids with -1 masking."""
    dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=False)
    # Normal-mode dispatch finishes asynchronously; work queued here overlaps
    # the transfer instead of waiting behind it.
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
        # Local experts' contribution per received token. Non-local ids arrive
        # as -1 and fall into the Marlin EP mask (zero contribution); the
        # route weights are folded in by GEMM2, so combine reduces unweighted.
        expert_out = marlin_mxfp4_precomputed_moe_apply(
            plan,
            recv_x,
            w,
            None,
            topk_weights=recv_topk_weights,
            topk_ids=recv_topk_ids,
            enable_pdl=enable_pdl,
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
    enable_pdl: bool,
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Decode-shaped path: padded per-expert recv buffer, masked rows.

    The buffer arrives as ``[num_local_experts, recv_m, hidden]`` with
    ``masked_m`` valid rows per expert and no routing metadata. Every valid row
    belongs to exactly one known local expert, so the Marlin grouped GEMM runs
    on the flattened buffer with a synthesized top-1 routing (weight 1.0);
    padded rows get id -1 and fall into the EP mask instead of computing
    garbage. The real route weights are applied by the low-latency combine.
    """
    dispatcher.dispatch_a(x, topk_ids, topk_weights, low_latency=True)
    # dispatch_a only launched the send phase; work queued here runs while the
    # transfer lands instead of leaving the GPU spinning in the recv phase.
    if overlap_fn is not None:
        overlap_fn()
    recv_x, _, _, _, _, _, masked_m = dispatcher.dispatch_b()

    num_local_experts, recv_m, hidden = recv_x.shape
    ep_rank = int(getattr(w, "ep_rank", 0))
    expert_start = ep_rank * num_local_experts
    device = recv_x.device

    # Row r of expert e is valid iff r < masked_m[e]; valid rows route to the
    # (global) id of their owning expert, padded rows to -1. All device-side,
    # no sync on masked_m.
    global_ids = torch.arange(
        expert_start,
        expert_start + num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    valid = torch.arange(recv_m, dtype=torch.int32, device=device).unsqueeze(
        0
    ) < masked_m.to(torch.int32).unsqueeze(1)
    flat_ids = torch.where(valid, global_ids.unsqueeze(1).expand(-1, recv_m), -1)
    flat_ids = flat_ids.reshape(-1, 1)
    flat_weights = torch.ones(
        (num_local_experts * recv_m, 1), dtype=torch.float32, device=device
    )

    expert_out = marlin_mxfp4_precomputed_moe_apply(
        plan,
        recv_x.reshape(num_local_experts * recv_m, hidden),
        w,
        None,
        topk_weights=flat_weights,
        topk_ids=flat_ids,
        enable_pdl=enable_pdl,
    ).view(num_local_experts, recv_m, hidden)

    # The low-latency combine leg applies the routing weights itself. This
    # deep_ep tree's LL combine carries an identity-expert extension (-1 ids
    # add x_ori * weight) and device-asserts x_ori whenever any weight is
    # nonzero; all our ids are valid so x is never actually read, but the
    # pointer must be supplied.
    dispatcher.combine_a(
        expert_out, topk_ids, topk_weights, low_latency=True, moe_origin_input=x
    )
    return dispatcher.combine_b()


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp4_deepep_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp4_moe_weights,
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
        num_tokens_global: Unused; DeepEP owns the token exchange.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (combine is the finalize).
        enable_pdl: Forwarded to the Marlin apply's activation epilogue.
        low_latency: Which DeepEP legs to run when the plan mode is "auto".
            Every rank of the EP group must pass the same value.
        overlap_fn: Optional work queued inside the dispatch window (e.g. the
            shared experts); it must not read the dispatch result or write
            ``x``.

    Returns:
        ``[local_tokens, hidden]`` bf16 combined MoE output, already reduced
        across the EP group (no outer all-reduce needed).
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu
    if not do_finalize:
        raise ValueError("Marlin MXFP4 DeepEP MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP4 DeepEP MoE requires precomputed top-k")

    dispatcher = _get_dispatcher(plan, w, x)
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(torch.float32)

    if dispatcher.deepep_mode.resolve(low_latency) == DeepEPMode.normal:
        return _apply_normal(
            dispatcher, plan, x, w, topk_weights, topk_ids, enable_pdl, overlap_fn
        )
    return _apply_low_latency(
        dispatcher, plan, x, w, topk_weights, topk_ids, enable_pdl, overlap_fn
    )
