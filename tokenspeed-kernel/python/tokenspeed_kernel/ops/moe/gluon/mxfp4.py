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


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
        gluon_mxfp_dynamic_mxfp4_fused_moe,
        gluon_mxfp_fused_moe,
    )
    from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )

    def gluon_mxfp4_gfx950_moe_weights(plan: dict, w: torch.nn.Module):
        return preprocess_gluon_mxfp4_gfx950_moe_weights(plan, w, preshuffle=True)

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
        del topk_weights, topk_ids, num_tokens_global, max_num_tokens_per_gpu
        del do_finalize, enable_pdl
        top_k = getattr(w, "top_k")
        swiglu_alpha, swiglu_limit, swiglu_beta = _swiglu_args(w)
        w13_precision_config = w.w13_precision_config
        w2_precision_config = w.w2_precision_config

        return gluon_mxfp_dynamic_mxfp4_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=getattr(w, "w13_weight_bias", None),
            w2_bias=getattr(w, "w2_weight_bias", None),
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
            w13_bias=getattr(w, "w13_weight_bias", None),
            w2_bias=getattr(w, "w2_weight_bias", None),
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
