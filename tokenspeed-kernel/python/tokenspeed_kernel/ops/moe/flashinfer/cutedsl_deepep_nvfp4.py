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

from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.communication.deep_ep import DeepEPDispatcher, DeepEPMode
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

platform = current_platform()


if platform.is_nvidia:
    from flashinfer import (
        scaled_fp4_grouped_quantize,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )
    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_cutedsl_deepep_nvfp4_moe_process_weights",
        solution="flashinfer_cutedsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"nvfp4"})},
        # DeepEP needs an EP process group in the plan, so it
        # must not become the generic NVFP4 auto-selected solution.
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutedsl_deepep_nvfp4_moe_process_weights(
        plan: dict, w: torch.nn.Module
    ):
        w13_ws2 = w.w13_weight_scale_2[:, 0]
        w13_input_scale = w.w13_input_scale.max().to(torch.float32)
        w2_input_scale = w.w2_input_scale.max().to(torch.float32)
        w.w13_weight_scale_2 = torch.nn.Parameter(w13_ws2, requires_grad=False)
        w.w13_input_scale_quant = torch.nn.Parameter(
            (1.0 / w13_input_scale).to(torch.float32), requires_grad=False
        )
        w.w2_input_scale_quant = torch.nn.Parameter(
            (1.0 / w2_input_scale).to(torch.float32), requires_grad=False
        )
        w.g1_alphas = torch.nn.Parameter(
            (w13_input_scale * w13_ws2).to(torch.float32), requires_grad=False
        )
        w.g2_alphas = torch.nn.Parameter(
            (w2_input_scale * w.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        scales = w.w13_weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        batches, rows, cols = scales.shape
        rows_padded = (rows + 127) // 128 * 128
        cols_padded = (cols + 3) // 4 * 4
        padded = torch.zeros(
            (batches, rows_padded, cols_padded),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:batches, :rows, :cols] = scales
        padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
        if scale_ndim == 2:
            swizzled = padded.reshape(rows_padded, cols_padded)
        else:
            swizzled = padded.reshape(batches, rows_padded, cols_padded)
        w.w13_blockscale_swizzled = torch.nn.Parameter(swizzled, requires_grad=False)

        scales = w.w2_weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        batches, rows, cols = scales.shape
        rows_padded = (rows + 127) // 128 * 128
        cols_padded = (cols + 3) // 4 * 4
        padded = torch.zeros(
            (batches, rows_padded, cols_padded),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:batches, :rows, :cols] = scales
        padded = padded.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
        padded = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
        if scale_ndim == 2:
            swizzled = padded.reshape(rows_padded, cols_padded)
        else:
            swizzled = padded.reshape(batches, rows_padded, cols_padded)
        w.w2_blockscale_swizzled = torch.nn.Parameter(swizzled, requires_grad=False)
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        solution="flashinfer_cutedsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({64}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutedsl_deepep_nvfp4_moe_apply(
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

        dispatcher = plan.get("_deepep_dispatcher")
        if dispatcher is None:
            group = plan.get("deepep_group")
            if group is None:
                raise ValueError("DeepEP MoE plan is missing deepep_group")
            config = SimpleNamespace(
                top_k=getattr(w, "top_k"),
                num_experts=getattr(w, "num_experts"),
                low_latency_max_num_tokens_per_gpu=max_num_tokens_per_gpu or x.shape[0],
                hidden_size=x.shape[1],
                world_size=getattr(w, "ep_size", group.size()),
                group=group,
                params_dtype=torch.bfloat16,
            )
            dispatcher = DeepEPDispatcher(
                config,
                deepep_mode=DeepEPMode.low_latency,
                async_finish=False,
                return_recv_hook=True,
                use_fp8=False,
            )
            plan["_deepep_dispatcher"] = dispatcher

        dispatcher.dispatch_a(x, topk_ids, topk_weights, None)
        recv_hidden, _, _, _, _, _, masked_m = dispatcher.dispatch_b()

        num_local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        a_q, a_q_sf = scaled_fp4_grouped_quantize(
            recv_hidden,
            masked_m,
            w.w13_input_scale_quant.expand(num_local_experts),
        )
        sf_vec_size = 16
        gateup_output = torch.empty(
            (num_local_experts, recv_hidden.shape[1], w.w2_weight.shape[-1] * 4),
            dtype=torch.bfloat16,
            device=x.device,
        ).permute(1, 2, 0)
        grouped_gemm_nt_masked(
            (a_q, a_q_sf),
            (w.w13_weight.permute(1, 2, 0), w.w13_blockscale_swizzled),
            gateup_output,
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=sf_vec_size,
            alpha=w.g1_alphas.view(1, 1, num_local_experts),
            alpha_dtype="float32",
        )

        diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
            gateup_output.permute(2, 0, 1),
            masked_m,
            w.w2_input_scale_quant.expand(num_local_experts),
        )
        output = torch.empty(
            (num_local_experts, recv_hidden.shape[1], x.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        ).permute(1, 2, 0)
        grouped_gemm_nt_masked(
            (diq, diq_sf),
            (w.w2_weight.permute(1, 2, 0), w.w2_blockscale_swizzled),
            output,
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=sf_vec_size,
            alpha=w.g2_alphas.view(1, 1, num_local_experts),
            alpha_dtype="float32",
        )

        dispatcher.combine_a(output.permute(2, 0, 1), topk_ids, topk_weights, None)
        return dispatcher.combine_b()
