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
next_power_of_2 = lambda value: 1 if value <= 1 else 1 << (value - 1).bit_length()


def _swap_halves(x: torch.Tensor, dim: int) -> torch.Tensor:
    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {size}")
    first, second = x.split(size // 2, dim=dim)
    return torch.cat((second, first), dim=dim).contiguous()


def _swizzle_mxfp4_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert linear E8M0 scales to CUTLASS' padded 128x4 SF layout."""
    if scales.ndim != 3 or scales.dtype != torch.uint8:
        raise ValueError(
            "MXFP4 scales must be a 3D uint8 tensor with shape "
            f"[experts, rows, cols], got {tuple(scales.shape)} {scales.dtype}"
        )

    experts, rows, cols = scales.shape
    rows_padded = (rows + 127) // 128 * 128
    cols_padded = (cols + 3) // 4 * 4
    padded = torch.zeros(
        (experts, rows_padded, cols_padded),
        dtype=scales.dtype,
        device=scales.device,
    )
    padded[:, :rows, :cols] = scales
    return (
        padded.reshape(
            experts,
            rows_padded // 128,
            4,
            32,
            cols_padded // 4,
            4,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows_padded, cols_padded)
    )


def _expert_float_parameter(
    w: torch.nn.Module,
    value: float | None,
) -> torch.nn.Parameter | None:
    if value is None:
        return None
    return torch.nn.Parameter(
        torch.full(
            (w.w13_weight.shape[0],),
            float(value),
            dtype=torch.float32,
            device=w.w13_weight.device,
        ),
        requires_grad=False,
    )


if platform.is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe, mxfp8_quantize

    def flashinfer_cutlass_mxfp4_moe_weights(
        plan: dict,
        w: torch.nn.Module,
    ) -> None:
        """Prepare checkpoint MXFP4 weights for FlashInfer CUTLASS on SM12x."""
        del plan
        intermediate_size = w.w13_weight.shape[1] // 2
        hidden_size = w.w2_weight.shape[1]
        if intermediate_size % 128 != 0 or hidden_size % 128 != 0:
            raise ValueError(
                "FlashInfer SM12x MXFP4 MoE requires hidden and intermediate "
                f"sizes divisible by 128, got {hidden_size} and {intermediate_size}"
            )

        # TokenSpeed checkpoints store [w1 (gate) | w3 (up)]. FlashInfer's
        # CUTLASS SwiGLU path consumes [w3 (up) | w1 (gate)].
        w13_weight = _swap_halves(w.w13_weight.data, 1)
        w13_scale = _swap_halves(w.w13_weight_scale.data, 1)
        w2_weight = w.w2_weight.data.contiguous()
        w2_scale = w.w2_weight_scale.data.contiguous()

        w.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        w.w13_weight_scale = torch.nn.Parameter(
            _swizzle_mxfp4_block_scales(w13_scale),
            requires_grad=False,
        )
        w.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        w.w2_weight_scale = torch.nn.Parameter(
            _swizzle_mxfp4_block_scales(w2_scale),
            requires_grad=False,
        )

        if hasattr(w, "w13_weight_bias"):
            w.w13_weight_bias = torch.nn.Parameter(
                _swap_halves(w.w13_weight_bias.data, 1),
                requires_grad=False,
            )
        if hasattr(w, "w2_weight_bias"):
            w.w2_weight_bias = torch.nn.Parameter(
                w.w2_weight_bias.data.contiguous(),
                requires_grad=False,
            )

        num_local_experts = w.w13_weight.shape[0]
        ones = torch.ones(
            num_local_experts,
            dtype=torch.float32,
            device=w.w13_weight.device,
        )
        w.w13_weight_global_scale = torch.nn.Parameter(ones, requires_grad=False)
        w.w2_weight_global_scale = torch.nn.Parameter(ones.clone(), requires_grad=False)

        swiglu_arg = getattr(w, "swiglu_arg", None)
        w.gemm1_alpha = _expert_float_parameter(
            w, None if swiglu_arg is None else swiglu_arg.alpha
        )
        w.gemm1_beta = _expert_float_parameter(w, getattr(w, "swiglu_beta", None))
        w.gemm1_clamp_limit = _expert_float_parameter(
            w, None if swiglu_arg is None else swiglu_arg.limit
        )
        w.intermediate_size_per_partition = intermediate_size
        w.hidden_size_padded = hidden_size
        w.hidden_size_original = getattr(w, "hidden_size", hidden_size)

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_cutlass_mxfp4_moe_apply",
        solution="flashinfer_cutlass",
        weight_preprocessor=flashinfer_cutlass_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(12, 0),
            max_arch_version=ArchVersion(12, 1),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_cutlass_mxfp4_moe_apply(
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
    ) -> torch.Tensor:
        del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu
        if not do_finalize:
            raise ValueError("FlashInfer CUTLASS MXFP4 MoE requires finalization")
        if topk_weights is None or topk_ids is None:
            raise ValueError(
                "FlashInfer CUTLASS MXFP4 MoE requires precomputed top-k routing"
            )

        hidden_padded = getattr(w, "hidden_size_padded", w.w2_weight.shape[1])
        hidden_original = getattr(w, "hidden_size_original", hidden_padded)
        if x.shape[0] == 0:
            return x.new_empty((0, hidden_original), dtype=torch.bfloat16)
        if x.shape[-1] > hidden_padded:
            raise ValueError(
                f"input hidden size {x.shape[-1]} exceeds prepared size {hidden_padded}"
            )
        if x.shape[-1] < hidden_padded:
            x = torch.nn.functional.pad(x, (0, hidden_padded - x.shape[-1]))

        x_quant, x_scale = mxfp8_quantize(
            x,
            False,
            alignment=hidden_padded,
            enable_pdl=enable_pdl,
        )
        output = torch.empty(
            (x.shape[0], hidden_padded),
            dtype=torch.bfloat16,
            device=x.device,
        )
        result = cutlass_fused_moe(
            input=x_quant,
            input_sf=x_scale,
            swizzled_input_sf=False,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.to(torch.float32),
            fc1_expert_weights=w.w13_weight.view(torch.long),
            fc2_expert_weights=w.w2_weight.view(torch.long),
            fc1_expert_biases=getattr(w, "w13_weight_bias", None),
            fc2_expert_biases=getattr(w, "w2_weight_bias", None),
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=[
                w.w13_weight_scale.contiguous().view(torch.int32),
                w.w13_weight_global_scale,
                w.w2_weight_scale.contiguous().view(torch.int32),
                w.w2_weight_global_scale,
            ],
            swiglu_alpha=getattr(w, "gemm1_alpha", None),
            swiglu_beta=getattr(w, "gemm1_beta", None),
            swiglu_limit=getattr(w, "gemm1_clamp_limit", None),
            ep_size=getattr(w, "ep_size", 1),
            ep_rank=getattr(w, "ep_rank", 0),
            tp_size=getattr(w, "tp_size", 1),
            tp_rank=getattr(w, "tp_rank", 0),
            use_mxfp8_act_scaling=True,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu,
        )[0]
        if hidden_original != hidden_padded:
            result = result[:, :hidden_original].contiguous()
        return result
