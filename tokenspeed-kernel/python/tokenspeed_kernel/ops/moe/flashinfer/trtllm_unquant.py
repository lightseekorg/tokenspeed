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
    from flashinfer import trtllm_bf16_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_trtllm_unquant_moe_process_weights",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"unquant"})},
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_unquant_moe_process_weights(plan: dict, w: torch.nn.Module):
        cache_permute_indices = {}
        num_experts = w.w13_weight.shape[0]
        epilogue_tile_m = 128
        block_k = 128

        half_w = w.w13_weight.shape[1] // 2
        w1_weight = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = w1_weight

        old_shape_w13 = w.w13_weight.data[0].shape
        old_shape_w2 = w.w2_weight.data[0].shape
        new_shape_w13 = old_shape_w13
        new_shape_w2 = old_shape_w2

        for idx in range(num_experts):
            permute_indices = maybe_get_cached_w3_w1_permute_indices(
                cache_permute_indices,
                w.w13_weight.data[idx].view(torch.uint8),
                epilogue_tile_m,
            )
            tmp_weights1 = (
                w.w13_weight.data[idx]
                .clone()
                .view(torch.uint8)[permute_indices.to(w.w13_weight.data.device)]
                .contiguous()
            )
            permute_indices = get_w2_permute_indices_with_cache(
                cache_permute_indices,
                w.w2_weight.data[idx].view(torch.uint8),
                epilogue_tile_m,
            )
            tmp_weights2 = (
                w.w2_weight.data[idx]
                .clone()
                .view(torch.uint8)[permute_indices.to(w.w2_weight.data.device)]
                .contiguous()
            )
            tmp_weights1 = convert_to_block_layout(
                tmp_weights1.view(torch.uint8), block_k
            )
            tmp_weights2 = convert_to_block_layout(
                tmp_weights2.view(torch.uint8), block_k
            )
            new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
            new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
            w.w13_weight.data[idx] = (
                tmp_weights1.view(torch.bfloat16).contiguous().reshape(old_shape_w13)
            )
            w.w2_weight.data[idx] = (
                tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
            )

        w.w13_weight.data = w.w13_weight.data.reshape(num_experts, *new_shape_w13)
        w.w2_weight.data = w.w2_weight.data.reshape(num_experts, *new_shape_w2)
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_unquant_moe_apply",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({128}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_unquant_moe_apply(
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
        routing_config = getattr(w, "routing_config", {})
        if not isinstance(routing_config, dict):
            routing_config = {}
        routing_value = lambda name, default: (
            routing_config[name]
            if name in routing_config
            else getattr(w, name, default)
        )
        routing_method_type = routing_value("routing_method_type", 1)
        routing_logits_dtype = (
            torch.float32 if int(routing_method_type) in {2, 7} else torch.bfloat16
        )
        routing_bias = routing_value("correction_bias", None)
        if routing_bias is not None:
            routing_bias = routing_bias.to(routing_logits_dtype)
        local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        result = trtllm_bf16_moe(
            routing_logits=router_logits.to(routing_logits_dtype),
            routing_bias=routing_bias,
            hidden_states=x,
            gemm1_weights=w.w13_weight,
            gemm2_weights=w.w2_weight,
            num_experts=getattr(w, "num_experts"),
            top_k=getattr(w, "top_k"),
            n_group=routing_value("n_group", None),
            topk_group=routing_value("topk_group", None),
            intermediate_size=getattr(w, "intermediate_size")
            // getattr(w, "tp_size", 1),
            local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
            local_num_experts=local_experts,
            routed_scaling_factor=routing_value("routed_scaling_factor", None),
            routing_method_type=routing_method_type,
            do_finalize=do_finalize,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
        )
        if do_finalize and isinstance(result, (list, tuple)):
            return result[0]
        if not do_finalize:
            gemm2_out, expert_weights, expanded_idx = result
            if expert_weights.dtype == torch.float32:
                n, k = expert_weights.size()
                expert_weights = expert_weights.view(torch.bfloat16).view(-1, k)[:n]
            return (gemm2_out, expert_weights, expanded_idx)
        return result
