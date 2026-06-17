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
from tokenspeed_kernel.preprocessing import register_weight_preprocessor
from tokenspeed_kernel.registry import Priority, WeightPreprocessorRef, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import gluon_mxfp_fused_moe
    from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights as _preprocess_impl,
    )

    @register_weight_preprocessor(
        "moe",
        name="gluon_mxfp4_gfx950_moe_weights",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
    )
    def gluon_mxfp4_gfx950_moe_weights(plan: dict, w: torch.nn.Module):
        return _preprocess_impl(plan, w)

    @register_kernel(
        "moe",
        "apply",
        name="gluon_mxfp4_moe_apply",
        solution="gluon",
        weight_preprocessor=WeightPreprocessorRef(
            "gluon_mxfp4_gfx950_moe_weights", required=True
        ),
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
        swiglu_arg = getattr(w, "swiglu_arg", None)

        router_logits = router_logits
        top_k = getattr(w, "top_k")

        swiglu_alpha = swiglu_arg.alpha if swiglu_arg else 1.702
        swiglu_limit = swiglu_arg.limit if swiglu_arg else 7.0

        return gluon_mxfp_fused_moe(
            x,
            router_logits,
            w.w13_weight_triton_tensor,
            w.w2_weight_triton_tensor,
            w13_bias=getattr(w, "w13_weight_bias", None),
            w2_bias=getattr(w, "w2_weight_bias", None),
            w13_precision_config=getattr(w, "w13_precision_config", None),
            w2_precision_config=getattr(w, "w2_precision_config", None),
            w13_act_scale=w.w13_act_scale,
            w2_act_scale=w.w2_act_scale,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )
