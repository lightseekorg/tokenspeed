# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered bf16 MoE apply using MORI all-to-all EP (real dispatch/combine).

Dispatch -> per-expert grouped-GEMM -> combine, backed by MORI's v2 op on AMD.
Selected for vendor=amd, gfx950, bf16/unquant, expert-parallel, all-to-all EP.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


def masked_grouped_gemm(
    packed_x: torch.Tensor,  # [E_local, cap, hidden]  (dispatched tokens grouped by local expert)
    counts: torch.Tensor,  # [E_local] int32         (valid rows per expert)
    w13: torch.Tensor,  # [E_local, 2*I, hidden]  gate rows [0:I], up [I:2I]
    w2: torch.Tensor,  # [E_local, hidden, I]
) -> torch.Tensor:
    """Per-expert SwiGLU FFN over the first counts[e] rows of each expert slot; padding -> 0.
    Pure FFN (combine applies the top-k weighting). Returns packed_out [E_local, cap, hidden].

    Two batched GEMMs over all experts at once, with padding rows (>= counts[e]) masked to
    0 so MORI combine -- which reads only valid rows via src_info -- is unaffected. This is the
    bf16 analog of the mxfp4 ``_grouped_mxfp4_gemm_3d`` (ops/moe/mori/mxfp4.py).
    """
    E, cap, H = packed_x.shape
    I = w13.shape[1] // 2
    # zero padding rows (>= counts[e]) so padding never produces NaN/garbage that could leak
    valid = torch.arange(cap, device=packed_x.device)[None, :] < counts[:, None].to(
        torch.long
    )
    xin = (packed_x * valid[..., None]).to(torch.bfloat16)  # [E, cap, H]
    gate_up = torch.bmm(xin, w13.transpose(1, 2))  # [E, cap, 2I]
    inter = torch.nn.functional.silu(gate_up[..., :I]) * gate_up[..., I:]  # [E, cap, I]
    out = torch.bmm(inter.to(torch.bfloat16), w2.transpose(1, 2))  # [E, cap, H]
    return (out * valid[..., None]).to(packed_x.dtype)


if platform.is_amd:
    from tokenspeed_kernel.ops.communication.mori_ep import get_dispatcher

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_bf16_moe_apply",
        solution="mori",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            # silu only: masked_grouped_gemm computes plain silu(gate)*up and does not apply
            # swiglu_arg/alpha/limit, so do not advertise "swiglu" (would select this kernel
            # for clamped-swiglu configs and silently compute the wrong activation).
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def mori_ep_bf16_moe_apply(
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
    ):
        """bf16 MoE FFN with real EP dispatch/combine via MORI.

        ``w`` exposes ``w13_weight`` ``[E_local, 2*I, H]`` (gate [0:I], up [I:2I]),
        ``w2_weight`` ``[E_local, H, I]`` (bf16), ``top_k``, and EP mapping
        ``ep_rank`` / ``ep_size`` / ``num_local_experts``. ``topk_ids`` are GLOBAL
        expert ids (MORI routes by global id); do NOT pre-mask to local space.

        ``low_latency`` / ``overlap_fn`` are the all-to-all apply-kernel contract: MORI
        has one dispatch/combine path, so ``low_latency`` is accepted and ignored and
        ``overlap_fn`` (if any) runs after the dispatch send.
        """
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        ep_size = int(getattr(w, "ep_size", dist.get_world_size()))
        ep_rank = int(getattr(w, "ep_rank", dist.get_rank()))
        num_local = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
        # Fixed dispatch capacity (per rank). Must upper-bound tokens/rank across all
        # calls; MORI allocates symmetric buffers for this at op creation. One
        # dispatcher is shared across all same-shape MoE layers (see get_dispatcher).
        cap = max(int(max_num_tokens_per_gpu or 0), int(x.shape[0]))
        dispatcher = get_dispatcher(
            rank=ep_rank,
            world_size=ep_size,
            hidden_dim=x.shape[1],
            num_local_experts=num_local,
            num_experts_per_token=int(getattr(w, "top_k")),
            max_num_inp_token_per_rank=cap,
            data_type=torch.bfloat16,
        )

        handle = dispatcher.dispatch(x, topk_weights.float(), topk_ids)
        if overlap_fn is not None:
            overlap_fn()
        packed = handle["packed_x"]  # [E_local, cap, H]
        packed.copy_(
            masked_grouped_gemm(packed, handle["counts"], w.w13_weight, w.w2_weight)
        )
        # Return the COMPLETE per-token routed result on every rank; do NOT reduce/scale here.
        return dispatcher.combine(handle)
