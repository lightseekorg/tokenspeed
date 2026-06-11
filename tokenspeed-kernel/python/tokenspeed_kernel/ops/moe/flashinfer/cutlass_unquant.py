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
from tokenspeed_kernel.signature import format_signature, format_signatures

platform = current_platform()
next_power_of_2 = lambda value: 1 if value <= 1 else 1 << (value - 1).bit_length()


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe
    from flashinfer.autotuner import autotune as flashinfer_autotune

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_cutlass_unquant_moe_process_weights",
        solution="flashinfer_cutlass",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(8, 9),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"unquant"})},
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_unquant_moe_process_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_unquant_moe_apply",
        solution="flashinfer_cutlass",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(8, 9),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_unquant_moe_apply(
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
            topk_weights = topk_weights.to(x.dtype)
        with flashinfer_autotune():
            return cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=w.w13_weight,
                fc2_expert_weights=w.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=getattr(w, "ep_size", 1),
                ep_rank=getattr(w, "ep_rank", 0),
                tp_size=getattr(w, "tp_size", 1),
                tp_rank=getattr(w, "tp_rank", 0),
                tune_max_num_tokens=max(8192, next_power_of_2(x.shape[0])),
                activation_type=ActivationType.Swiglu,
            )[0]
