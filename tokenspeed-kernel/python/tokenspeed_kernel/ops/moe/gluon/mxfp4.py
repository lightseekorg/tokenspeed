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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

# Captured TP8/EP8 Kimi K3 execution crosses from route-direct warp GEMV to
# grouped MFMA between M=4 and M=8. A standalone single-rank kernel sweep puts
# the crossover much later, but extending that result regresses model TPOT at
# both M=8 and M=16. Keep production selection tied to the model measurement.
_ROUTE_DIRECT_DECODE_MAX_TOKENS = 4


def _use_route_direct_decode(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
) -> bool:
    """Return whether a K3-shaped activation should use routed warp GEMV."""
    two_intermediate = int(w13_weight.shape[1])
    intermediate = two_intermediate // 2
    return (
        0 < x.shape[0] <= _ROUTE_DIRECT_DECODE_MAX_TOKENS
        and x.is_contiguous()
        and x.shape[1] % 256 == 0
        and two_intermediate % 2 == 0
        and intermediate % 256 == 0
        and int(w13_weight.shape[2]) * 2 == x.shape[1]
    )


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
        gluon_mxfp_dynamic_mxfp4_fused_moe,
        gluon_mxfp_fused_moe,
        gluon_mxfp_precomputed_mxfp4_fused_moe,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_grouped import (
        gluon_a16w4_situ_grouped_ep_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import fused as fused_mxfp_gfx1250
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx1250_moe_weights,
    )

    def gluon_mxfp4_gfx950_moe_weights(plan: dict, w: torch.nn.Module):
        return preprocess_gluon_mxfp4_gfx950_moe_weights(plan, w, preshuffle=True)

    def gluon_mxfp4_gfx1250_moe_weights(plan: dict, w: torch.nn.Module):
        return preprocess_gluon_mxfp4_gfx1250_moe_weights(plan, w)

    def validate_linear_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
        del plan
        names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
        if any(not hasattr(w, name) for name in names):
            raise ValueError("linear MXFP4 MoE weights are incomplete")
        tensors = [getattr(w, name) for name in names]
        if any(t.dtype != torch.uint8 for t in tensors):
            raise TypeError("linear MXFP4 weights must use uint8 storage")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("linear MXFP4 weights must be contiguous")
        if any(t.ndim != 3 for t in tensors) or len({t.shape[0] for t in tensors}) != 1:
            raise ValueError("linear MXFP4 weights must share a rank-3 expert axis")
        expected = int(getattr(w, "num_local_experts", tensors[0].shape[0]))
        if tensors[0].shape[0] != expected:
            raise ValueError("linear MXFP4 weights have the wrong local expert count")

    def _swiglu_args(w: torch.nn.Module) -> tuple[float, float, float]:
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is None:
            # alpha=1, limit=0, beta=0 makes the reducer use an unclamped
            # SiLU gate multiplied by the linear branch.
            return 1.0, 0.0, 0.0
        swiglu_beta = getattr(w, "swiglu_beta", None)
        return (
            1.0 if swiglu_arg.alpha is None else swiglu_arg.alpha,
            0.0 if swiglu_arg.limit is None else swiglu_arg.limit,
            0.0 if swiglu_beta is None else swiglu_beta,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=validate_linear_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            # This grouped kernel was validated for K3's TP1/EP8 placement.
            # Keep auto-selection scoped to that exact EP degree; Triton
            # remains the fallback for other AMD EP layouts.
            "ep_size": frozenset({8}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
        },
        # Gfx950 A16W4 SiTU EP8 is the measured K3 fast path. Capability and
        # ep_size traits keep this specialized priority from affecting other
        # platforms or EP degrees.
        priority=Priority.SPECIALIZED + 1,
    )
    def gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply(
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
    ):
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del enable_pdl
        if not do_finalize:
            raise ValueError("gfx950 A16W4 Gluon MoE cannot defer finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError("gfx950 A16W4 Gluon MoE requires precomputed top-k")
        num_local_experts = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
        expert_start = int(getattr(w, "ep_rank", 0)) * num_local_experts
        use_route_direct_decode = _use_route_direct_decode(x, w.w13_weight)
        if use_route_direct_decode:
            from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (
                gluon_a16w4_situ_warp_decode_ep_gfx950,
            )

            # Consume the Gluon plan's linear checkpoint layout directly. EP
            # localization stays inside both route-direct stages, avoiding the
            # four pointwise kernels in ``_local_topk_for_ep``. Batches above
            # the measured M=4 model crossover retain grouped MFMA below.
            return gluon_a16w4_situ_warp_decode_ep_gfx950(
                x,
                w.w13_weight,
                w.w13_weight_scale,
                w.w2_weight,
                w.w2_weight_scale,
                topk_weights,
                topk_ids,
                situ_beta=float(getattr(w, "activation_situ_beta", 1.0)),
                situ_linear_beta=getattr(w, "activation_situ_linear_beta", None),
                expert_start=expert_start,
                linear_weights=True,
                w13_interleaved=(
                    getattr(w, "w13_input_layout", "concatenated") == "interleaved"
                ),
            )
        return gluon_a16w4_situ_grouped_ep_gfx950(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_weights,
            topk_ids,
            situ_beta=float(getattr(w, "activation_situ_beta", 1.0)),
            situ_linear_beta=getattr(w, "activation_situ_linear_beta", None),
            expert_start=expert_start,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_gfx1250_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx1250_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"fp8"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 4,
    )
    def gluon_mxfp4_gfx1250_precomputed_moe_apply(
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
    ):
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        if topk_weights is None or topk_ids is None:
            raise ValueError(
                "gluon_mxfp4_gfx1250_precomputed_moe_apply requires "
                "topk_weights and topk_ids"
            )

        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_pc = w.w13_precision_config
        w2_pc = w.w2_precision_config

        return fused_mxfp_gfx1250.gluon_mxfp_precomputed_mxfp4_fused_moe(
            x,
            topk_weights,
            topk_ids,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_pc.b_mx_scale,
            w2_mx_scale=w2_pc.b_mx_scale,
            out_dtype=w2_pc.out_dtype or torch.bfloat16,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_dynamic_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 3,
    )
    def gluon_mxfp4_dynamic_moe_apply(
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
    ):
        del num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        top_k = getattr(w, "top_k")
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config
        # Forward the caller's precomputed top-k whenever it is supplied. The
        # downstream dispatch picks the tuned kernel by batch size (direct
        # decode owns M <= _DIRECT_DECODE_MAX_M, precomputed-MFMA decode owns
        # M >= _PRECOMPUTED_MFMA_MIN_M) and otherwise builds ragged metadata
        # directly from the forwarded top-k. Dropping it for any M in between
        # (e.g. M == 3) would silently recompute routing from router_logits,
        # so we always forward and let the dispatch choose.
        forward_precomputed_topk = topk_weights is not None and topk_ids is not None

        return gluon_mxfp_dynamic_mxfp4_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            top_k=top_k,
            correction_bias=getattr(w, "_correction_bias", None),
            n_group=int(getattr(w, "_n_group", 0) or 0),
            topk_group=int(getattr(w, "_topk_group", 0) or 0),
            routed_scaling_factor=float(
                getattr(w, "_routed_scaling_factor", 1.0) or 1.0
            ),
            normalize_topk_weights=bool(getattr(w, "_normalize_topk_weights", True)),
            routing_method_type=int(getattr(w, "_routing_method_type", 0)),
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            precomputed_topk_weights=topk_weights if forward_precomputed_topk else None,
            precomputed_topk_ids=topk_ids if forward_precomputed_topk else None,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_precomputed_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def gluon_mxfp4_precomputed_moe_apply(
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
    ):
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        if topk_weights is None or topk_ids is None:
            raise ValueError(
                "gluon_mxfp4_precomputed_moe_apply requires topk_weights and topk_ids"
            )
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config

        return gluon_mxfp_precomputed_mxfp4_fused_moe(
            x,
            topk_weights,
            topk_ids,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=getattr(w, "w13_weight_bias", None),
            w2_bias=getattr(w, "w2_weight_bias", None),
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
        )

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_moe_apply",
        solution="gluon",
        weight_preprocessor=gluon_mxfp4_gfx950_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({False}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"fp8"}),
            "supports_bias": frozenset({True}),
        },
        # gluon is narrowly gated to gfx950
        priority=Priority.SPECIALIZED,
    )
    def gluon_mxfp4_moe_apply(
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
    ):
        del topk_weights, topk_ids, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        top_k = getattr(w, "top_k")
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config

        return gluon_mxfp_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=(
                None
                if getattr(w, "_gluon_w13_bias_is_zero", False)
                else getattr(w, "w13_weight_bias", None)
            ),
            w2_bias=(
                None
                if getattr(w, "_gluon_w2_bias_is_zero", False)
                else getattr(w, "w2_weight_bias", None)
            ),
            w13_mx_scale=w13_precision_config.b_mx_scale,
            w2_mx_scale=w2_precision_config.b_mx_scale,
            w13_act_scale=w.w13_act_scale,
            w2_act_scale=w.w2_act_scale,
            out_dtype=w2_precision_config.out_dtype or torch.bfloat16,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
        )
