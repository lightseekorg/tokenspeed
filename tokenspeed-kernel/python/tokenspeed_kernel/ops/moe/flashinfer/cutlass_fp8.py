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
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe

    def flashinfer_cutlass_fp8_moe_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half

        half_s = w.w13_weight_scale_inv.shape[1] // 2
        first_scale = w.w13_weight_scale_inv.data[:, :half_s, :].clone()
        w.w13_weight_scale_inv.data[:, :half_s, :] = w.w13_weight_scale_inv.data[
            :, half_s:, :
        ]
        w.w13_weight_scale_inv.data[:, half_s:, :] = first_scale
        w.w13_weight_scale_inv.data.clamp_(min=1e-10)
        w.w2_weight_scale_inv.data.clamp_(min=1e-10)
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None:
            # The cutlass gated activation runs post-dequant, so alpha/beta/
            # limit are passed in the model's actual-value domain; default
            # parameters stay None to keep the plain Swiglu kernel path.
            num_experts = w.w13_weight.shape[0]
            device = w.w13_weight.device

            def _per_expert(value: float) -> torch.nn.Parameter:
                return torch.nn.Parameter(
                    torch.full(
                        (num_experts,), float(value), dtype=torch.float32, device=device
                    ),
                    requires_grad=False,
                )

            alpha = swiglu_arg.alpha
            w.swiglu_alpha_t = (
                _per_expert(alpha)
                if alpha is not None and float(alpha) != 1.0
                else None
            )
            beta = getattr(w, "swiglu_beta", None)
            w.swiglu_beta_t = (
                _per_expert(beta) if beta is not None and float(beta) != 0.0 else None
            )
            w.swiglu_limit_t = (
                _per_expert(swiglu_arg.limit) if swiglu_arg.limit is not None else None
            )
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_fp8_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_fp8_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"fp8"}),
            # The kernel invokes ActivationType.Swiglu (gated SiLU); accept both
            # names so DeepSeek-V4 (activation="swiglu") selects this path.
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(128, 128)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_fp8_moe_apply(
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
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)
        output = torch.empty(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        swiglu_alpha = getattr(w, "swiglu_alpha_t", None)
        swiglu_beta = getattr(w, "swiglu_beta_t", None)
        swiglu_limit = getattr(w, "swiglu_limit_t", None)
        if swiglu_alpha is None and swiglu_beta is None and swiglu_limit is None:
            activation_type = ActivationType.Swiglu
        else:
            activation_type = ActivationType.SwigluBias
        return cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=x.dtype,
            input_sf=None,
            quant_scales=[w.w13_weight_scale_inv, w.w2_weight_scale_inv],
            ep_size=getattr(w, "ep_size", 1),
            ep_rank=getattr(w, "ep_rank", 0),
            tp_size=getattr(w, "tp_size", 1),
            tp_rank=getattr(w, "tp_rank", 0),
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            activation_type=activation_type,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            use_deepseek_fp8_block_scale=True,
            enable_pdl=enable_pdl,
        )[0]
