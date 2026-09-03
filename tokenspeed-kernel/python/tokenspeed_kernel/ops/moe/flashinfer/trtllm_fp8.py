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

"""FP8 block-scale MoE via the TensorRT-LLM-Gen fused kernel.

Checkpoint weights are converted once at load time to TRT-LLM's shuffled
``BlockMajorK`` layout.  Keeping that layout resident avoids the slower
unshuffled MajorK load path on Blackwell without changing the block scales or
the logical expert weights.
"""

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
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import (
        RoutingMethodType,
        convert_to_block_layout,
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_block_scale_routed_moe,
    )
    from flashinfer.tllm_enums import WeightLayout
    from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

    _FP8_BLOCK = 128
    # DeepSeek block-FP8 kernels use a 64-row epilogue tile when weights are
    # shuffled.  This differs from the 128-row MXFP8/BF16 preparation path.
    _FP8_EPILOGUE_TILE_M = 64

    def _shuffle_block_major_(weight: torch.nn.Parameter) -> None:
        """Shuffle an expert weight in place, then expose BlockMajorK shape."""
        num_experts = weight.shape[0]
        old_expert_shape = weight.shape[1:]
        new_expert_shape = None
        for expert_idx in range(num_experts):
            shuffled = shuffle_matrix_a(
                weight.data[expert_idx].view(torch.uint8),
                _FP8_EPILOGUE_TILE_M,
            )
            blocked = convert_to_block_layout(shuffled, _FP8_BLOCK)
            new_expert_shape = blocked.shape
            # Reuse the checkpoint allocation while processing experts.  The
            # physical element count is unchanged; only the final view differs.
            weight.data[expert_idx] = (
                blocked.view(weight.dtype).contiguous().reshape(old_expert_shape)
            )
        if new_expert_shape is not None:
            weight.data = weight.data.reshape(num_experts, *new_expert_shape)

    def _routing_value(w: torch.nn.Module, name: str, default):
        routing_config = getattr(w, "routing_config", {})
        if not isinstance(routing_config, dict):
            routing_config = {}
        if name in routing_config:
            return routing_config[name]
        return getattr(w, name, default)

    def flashinfer_trtllm_fp8_moe_process_weights(plan: dict, w: torch.nn.Module):
        # The shared MoE checkpoint loader stores w13 as a concatenated
        # ``[w1(gate) | w3(up)]`` block; the TRT-LLM-Gen gated kernel consumes
        # ``[w3 | w1]`` ordering (same swap flashinfer_cutlass applies). Swap the
        # gate/up halves of both the weight and its block-scale in place.
        half_w = w.w13_weight.shape[1] // 2
        first_w = w.w13_weight.data[:, :half_w, :].clone()
        w.w13_weight.data[:, :half_w, :] = w.w13_weight.data[:, half_w:, :]
        w.w13_weight.data[:, half_w:, :] = first_w

        half_s = w.w13_weight_scale_inv.shape[1] // 2
        first_s = w.w13_weight_scale_inv.data[:, :half_s, :].clone()
        w.w13_weight_scale_inv.data[:, :half_s, :] = w.w13_weight_scale_inv.data[
            :, half_s:, :
        ]
        w.w13_weight_scale_inv.data[:, half_s:, :] = first_s

        w.w13_weight_scale_inv.data.clamp_(min=1e-10)
        w.w2_weight_scale_inv.data.clamp_(min=1e-10)

        # Preserve the logical size before BlockMajorK changes the final tensor
        # dimension from intermediate_size to block_k.
        w.intermediate_size_per_partition = w.w2_weight.shape[-1]
        _shuffle_block_major_(w.w13_weight)
        _shuffle_block_major_(w.w2_weight)

        # DeepSeek block-FP8 applies SwiGLU after GEMM1 dequantization, so
        # activation parameters stay in the model's actual-value domain.
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is not None:
            num_experts = w.w13_weight.shape[0]
            device = w.w13_weight.device

            def _per_expert(value: float) -> torch.nn.Parameter:
                return torch.nn.Parameter(
                    torch.full(
                        (num_experts,), float(value), dtype=torch.float32, device=device
                    ),
                    requires_grad=False,
                )

            if swiglu_arg.alpha is not None and float(swiglu_arg.alpha) != 1.0:
                w.gemm1_alpha = _per_expert(swiglu_arg.alpha)
            beta = getattr(w, "swiglu_beta", None)
            if beta is not None and float(beta) != 0.0:
                w.gemm1_beta = _per_expert(beta)
            if swiglu_arg.limit is not None:
                w.gemm1_clamp_limit = _per_expert(swiglu_arg.limit)
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_fp8_moe_process_weights,
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
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(128, 128)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_fp8_moe_apply(
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
        hidden_size = x.shape[1]
        if x.shape[0] == 0:
            empty_output = x.new_empty(0, hidden_size, dtype=torch.bfloat16)
            if do_finalize:
                return empty_output
            expert_weights = (
                topk_weights
                if topk_weights is not None
                else x.new_empty((0, getattr(w, "top_k")), dtype=torch.bfloat16)
            )
            return (
                empty_output,
                expert_weights,
                x.new_empty((0,), dtype=torch.int32),
            )

        # Per-token group (block=128) FP8 quantization of activations. The
        # TRT-LLM helper emits the group-major scale layout consumed by the
        # fused MoE kernel.
        x_fp8, x_scale = per_token_group_quant_fp8(
            x,
            _FP8_BLOCK,
            column_major_scales=False,
            scale_tma_aligned=False,
        )
        x_scale = x_scale.to(torch.float32).contiguous()
        hidden_blocks = hidden_size // _FP8_BLOCK
        if x_scale.shape != (hidden_blocks, x_fp8.shape[0]):
            raise RuntimeError(
                "unexpected hidden_states_scale shape "
                f"{tuple(x_scale.shape)}; expected "
                f"{(hidden_blocks, x_fp8.shape[0])}"
            )

        local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        num_experts = getattr(w, "num_experts", local_experts)
        correction_bias = _routing_value(w, "correction_bias", None)
        routing_bias = (
            correction_bias.to(x.dtype)
            if isinstance(correction_bias, torch.Tensor)
            else None
        )
        n_group = _routing_value(w, "n_group", 0) or 1
        topk_group = _routing_value(w, "topk_group", 0) or 1
        routed_scaling_factor = _routing_value(w, "routed_scaling_factor", None)
        routing_method_type = _routing_value(w, "routing_method_type", 1)

        intermediate_size = w.intermediate_size_per_partition

        common_kwargs = dict(
            hidden_states=x_fp8,
            hidden_states_scale=x_scale,
            gemm1_weights=w.w13_weight,
            gemm1_weights_scale=w.w13_weight_scale_inv,
            gemm2_weights=w.w2_weight,
            gemm2_weights_scale=w.w2_weight_scale_inv,
            num_experts=num_experts,
            top_k=getattr(w, "top_k"),
            intermediate_size=intermediate_size,
            local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
            local_num_experts=local_experts,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
        )
        if topk_weights is not None and topk_ids is not None:
            result = trtllm_fp8_block_scale_routed_moe(
                topk_ids=(
                    topk_ids.to(torch.int32).contiguous(),
                    topk_weights.contiguous(),
                ),
                routing_bias=None,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
                # Routing is already complete. Use the generic permutation
                # pipeline instead of DeepSeekV3's <=8 groups / <=32 experts
                # per group logits router, which cannot represent GLM's E=288,
                # n_group=1 topology.
                routing_method_type=int(RoutingMethodType.Renormalize),
                **common_kwargs,
            )
        else:
            result = trtllm_fp8_block_scale_moe(
                routing_logits=router_logits.to(torch.float32),
                routing_bias=routing_bias,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=int(routing_method_type),
                **common_kwargs,
            )
        if do_finalize:
            if isinstance(result, (list, tuple)):
                result = result[0]
            return result

        if not isinstance(result, (list, tuple)) or len(result) < 3:
            raise RuntimeError(
                "FlashInfer deferred FP8 MoE must return gemm2 output, "
                "expert weights, and the expanded-to-permuted index"
            )
        gemm2_out, expert_weights, expanded_idx = result[:3]
        if topk_weights is None and expert_weights.dtype == torch.float32:
            # In-kernel DeepSeek routing allocates this carrier with the fp32
            # logits dtype, while the TRT-LLM launcher writes bf16 route
            # weights into it. Recover the live bf16 prefix before handing it
            # to the external fused finalize kernel. The pre-routed path
            # borrows the caller's real tensor and does not need this fixup.
            num_tokens, top_k = expert_weights.shape
            expert_weights = expert_weights.view(torch.bfloat16).view(-1, top_k)[
                :num_tokens
            ]
        return gemm2_out, expert_weights, expanded_idx

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_routed_moe_apply",
        solution="flashinfer_trtllm",
        weight_preprocessor=flashinfer_trtllm_fp8_moe_process_weights,
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
            "weight_dtype": frozenset({"fp8"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({True}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "fp8_scale_block_shape": frozenset({(128, 128)}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_trtllm_fp8_routed_moe_apply(
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
            raise RuntimeError(
                "precomputed_topk plan requires topk_weights and topk_ids"
            )
        return flashinfer_trtllm_fp8_moe_apply(
            plan,
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            num_tokens_global,
            max_num_tokens_per_gpu,
            do_finalize,
            enable_pdl,
        )
