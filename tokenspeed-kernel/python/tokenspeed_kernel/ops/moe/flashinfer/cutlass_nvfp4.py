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

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_cutlass_nvfp4_moe_process_weights",
        solution="flashinfer_cutlass",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"nvfp4"})},
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_nvfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
        half_w = w.w13_weight.shape[1] // 2
        first_half = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_half

        half_s = w.w13_weight_scale.shape[1] // 2
        first_scale = w.w13_weight_scale.data[:, :half_s, :].clone()
        w.w13_weight_scale.data[:, :half_s, :] = w.w13_weight_scale.data[:, half_s:, :]
        w.w13_weight_scale.data[:, half_s:, :] = first_scale

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
        name="flashinfer_cutlass_nvfp4_moe_apply",
        solution="flashinfer_cutlass",
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
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_nvfp4_moe_apply(
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
        output = torch.empty(
            x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device
        )
        return cutlass_fused_moe(
            output=output,
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w.w13_weight.view(torch.long),
            fc2_expert_weights=w.w2_weight.view(torch.long),
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=[
                w.w13_input_scale_quant,
                w.w13_blockscale_swizzled.view(torch.int32),
                w.g1_alphas,
                w.w2_input_scale_quant,
                w.w2_blockscale_swizzled.view(torch.int32),
                w.g2_alphas,
            ],
            ep_size=getattr(w, "ep_size", 1),
            ep_rank=getattr(w, "ep_rank", 0),
            tp_size=getattr(w, "tp_size", 1),
            tp_rank=getattr(w, "tp_rank", 0),
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            activation_type=ActivationType.Swiglu,
        )[0]
