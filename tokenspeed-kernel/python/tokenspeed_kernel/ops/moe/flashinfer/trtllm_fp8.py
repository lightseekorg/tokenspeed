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

"""DeepSeek-style FP8 block-scale MoE via the TensorRT-LLM-Gen fused kernel.

FlashInfer's ``cutlass_fused_moe`` gates ``use_deepseek_fp8_block_scale`` to
SM90, so it cannot serve fp8 block-scale MoE (GLM-5.1 / DeepSeek recipe) on
Blackwell. The TensorRT-LLM-Gen ``trtllm_fp8_block_scale_moe`` kernel *is* the
Blackwell (SM100+) path for the same recipe -- it fuses routing + dispatch +
grouped GEMM + finalize in a single CUDA-graph-safe call. Upstream wired the
TRT-LLM-Gen solutions for mxfp4 / nvfp4 / unquant but not for fp8; this module
fills that gap so the generic ``MoELayer`` auto-selects a working fp8 MoE on
Blackwell.

Registered at the ``SPECIALIZED`` priority band and gated to ``min_arch 10.0``
so it wins over the ``flashinfer_cutlass`` fp8 apply (``PERFORMANT`` band,
``min_arch 9.0``) on SM100 while leaving SM90 to cutlass.
"""

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
    from flashinfer.fused_moe import RoutingMethodType, trtllm_fp8_block_scale_moe
    from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

    try:
        from flashinfer import autotune
    except ImportError:  # pragma: no cover - older flashinfer
        autotune = None

    _FP8_BLOCK = 128
    _DEEPSEEK_V3_ROUTING = int(RoutingMethodType.DeepSeekV3)
    _MIN_TUNE_MAX_NUM_TOKENS = 8192

    def _tune_max_num_tokens(num_tokens: int) -> int:
        return max(_MIN_TUNE_MAX_NUM_TOKENS, next_power_of_2(int(num_tokens)))

    def _routing_value(w: torch.nn.Module, name: str, default):
        routing_config = getattr(w, "routing_config", {})
        if not isinstance(routing_config, dict):
            routing_config = {}
        if name in routing_config:
            return routing_config[name]
        return getattr(w, name, default)

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_trtllm_fp8_moe_process_weights",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"fp8"})},
        priority=Priority.SPECIALIZED,
    )
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
        w._flashinfer_trtllm_fp8_autotuned = False
        w._flashinfer_trtllm_fp8_autotuned_buckets = set()
        return None

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_fp8_moe_apply",
        solution="flashinfer_trtllm",
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
            "supports_deferred_finalize": frozenset({False}),
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
            return x.new_empty(0, hidden_size, dtype=torch.bfloat16)

        # Per-token group (block=128) FP8 quantization of activations. The
        # TRT-LLM-Gen kernel expects ``hidden_states_scale`` as a 2D
        # ``[hidden_size // 128, num_tokens]`` float32 tensor for the DeepSeekFp8
        # recipe. ``per_token_group_quant_fp8`` already emits the scale in that
        # ``[K, M]`` (group-major) orientation, so no transpose is needed.
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
        n_group = _routing_value(w, "n_group", 0) or None
        topk_group = _routing_value(w, "topk_group", 0) or None
        routed_scaling_factor = _routing_value(w, "routed_scaling_factor", None)
        tune_max_num_tokens = _tune_max_num_tokens(x_fp8.shape[0])

        def _call():
            return trtllm_fp8_block_scale_moe(
                routing_logits=router_logits.to(torch.float32),
                routing_bias=routing_bias,
                hidden_states=x_fp8,
                hidden_states_scale=x_scale,
                gemm1_weights=w.w13_weight,
                gemm1_weights_scale=w.w13_weight_scale_inv,
                gemm2_weights=w.w2_weight,
                gemm2_weights_scale=w.w2_weight_scale_inv,
                num_experts=num_experts,
                top_k=getattr(w, "top_k"),
                n_group=n_group,
                topk_group=topk_group,
                intermediate_size=getattr(w, "intermediate_size"),
                local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
                local_num_experts=local_experts,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=_DEEPSEEK_V3_ROUTING,
                do_finalize=True,
                tune_max_num_tokens=tune_max_num_tokens,
            )

        if autotune is not None:
            tuned_buckets = getattr(w, "_flashinfer_trtllm_fp8_autotuned_buckets", set())
            if tune_max_num_tokens not in tuned_buckets:
                with autotune():
                    _call()
                tuned_buckets = set(tuned_buckets)
                tuned_buckets.add(tune_max_num_tokens)
                w._flashinfer_trtllm_fp8_autotuned_buckets = tuned_buckets
                w._flashinfer_trtllm_fp8_autotuned = True

        result = _call()
        if isinstance(result, (list, tuple)):
            result = result[0]
        return result
