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

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "moe",
    "combine_quant_fp8",
    name="cute_dsl_combine_quant_fp8",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("partial", "dense", {torch.bfloat16}),
    priority=Priority.SPECIALIZED,
)
def cute_dsl_combine_quant_fp8(partial, inverse, weights, valid_rows, out):
    """Run the ordered route reduction and projection quantizer on Hopper."""
    from tokenspeed_kernel.ops.moe._cute_dsl.combine_quant import run_combine_quant

    return run_combine_quant(partial, inverse, weights, valid_rows, out)


@dataclass(frozen=True)
class _DeferredMoEState:
    small_batch_native: Callable
    original_native: Callable

    def native_for_tokens(self, tokens: int) -> Callable:
        # Keep the original schedule beyond the measured small-batch benefit.
        return self.small_batch_native if tokens <= 16 else self.original_native


def cute_dsl_fp8_moe_process_weights(plan: dict, w: torch.nn.Module) -> None:
    if w._moe_backend_state is not None:
        raise ValueError("FP8 MoE weights have already been prepared")
    if w.hidden_size <= 0 or w.hidden_size % 128 or not 1 <= w.top_k <= 32:
        raise ValueError(
            "CuTe DSL FP8 MoE requires a positive 128-aligned hidden size and top-k in 1..32"
        )
    if w.swiglu_arg is not None and (
        w.swiglu_arg.alpha not in (None, 1.0) or w.swiglu_arg.limit is not None
    ):
        raise ValueError("CuTe DSL FP8 MoE requires standard SwiGLU")
    if w.swiglu_beta not in (None, 0.0):
        raise ValueError("CuTe DSL FP8 MoE requires standard SwiGLU")
    from tokenspeed_kernel.ops.moe.flashinfer import flashinfer_cutlass_fp8_moe_weights
    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import make_cutlass_fused_moe

    # Build before mutating checkpoint storage; a failed build can be retried.
    small_batch_native = make_cutlass_fused_moe(
        mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=-1
    )
    original_native = make_cutlass_fused_moe(
        mode="expected_m", max_num_stages_fc1=-1, max_num_stages_fc2=-1
    )
    flashinfer_cutlass_fp8_moe_weights(plan, w)
    w._moe_backend_state = _DeferredMoEState(small_batch_native, original_native)


@register_kernel(
    "moe",
    "apply",
    name="cute_dsl_fp8_moe_apply",
    solution="cute_dsl",
    weight_preprocessor=cute_dsl_fp8_moe_process_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_deferred_routes": frozenset({True}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({128}),
        "hidden_alignment": frozenset({128}),
        "top_k": frozenset(range(1, 33)),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False}),
        "expert_id_repeats": frozenset({False}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def cute_dsl_fp8_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
):
    """Execute the planned FP8 experts with precomputed route IDs and weights.

    ``x`` is BF16 [tokens, hidden]; ``w`` owns processed checkpoint weights and
    the native callables initialized by ``moe_process_weights``. The remaining
    arguments follow ``moe_apply``. Normal calls return finalized BF16
    [tokens, hidden]; ``moe_apply_routes`` requests owned unweighted route rows.
    """
    if not do_finalize and plan["cutlass_fused_finalize"]:
        raise ValueError("Deferred routes require cutlass_fused_finalize=False")
    if topk_ids is None or topk_weights is None:
        raise ValueError("CuTe DSL FP8 MoE requires precomputed top-k")
    if (
        x.ndim != 2
        or x.shape[1] != w.hidden_size
        or x.dtype != torch.bfloat16
        or topk_ids.shape != (x.shape[0], w.top_k)
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("FP8 MoE inputs differ from the planned model geometry")
    state = w._moe_backend_state
    if not isinstance(state, _DeferredMoEState):
        raise RuntimeError("Prepare FP8 MoE weights before execution")
    from flashinfer import ActivationType
    from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens

    if not do_finalize and not 1 <= x.shape[0] <= 4096:
        raise ValueError("Deferred block-FP8 MoE supports 1..4096 tokens")
    output = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device) if do_finalize else None
    )
    if x.shape[0] == 0:
        return output
    ids = topk_ids.to(torch.int32).contiguous()
    weights = topk_weights.to(torch.float32).contiguous()
    native = state.native_for_tokens(x.shape[0])
    route_outputs = {}
    if not do_finalize:
        from tokenspeed_kernel.ops.moe.combine_quant import allocate_deferred_routes

        routes = allocate_deferred_routes(x, weights, w.w13_weight.shape[0], None)
        output = routes.partial
        route_outputs = dict(
            route_map=routes.inverse.T.reshape(-1), expert_offsets=routes.expert_offsets
        )
    result = native(
        output=output,
        input=x,
        token_selected_experts=ids,
        token_final_scales=weights,
        fc1_expert_weights=w.w13_weight,
        fc2_expert_weights=w.w2_weight,
        output_dtype=x.dtype,
        input_sf=None,
        quant_scales=[w.w13_weight_scale_inv, w.w2_weight_scale_inv],
        ep_size=w.ep_size,
        ep_rank=w.ep_rank,
        tp_size=w.tp_size,
        tp_rank=w.tp_rank,
        tune_max_num_tokens=get_autotune_max_num_tokens(),
        activation_type=ActivationType.Swiglu,
        use_deepseek_fp8_block_scale=True,
        use_fused_finalize=plan["cutlass_fused_finalize"],
        enable_pdl=enable_pdl,
        expected_m=None,
        **route_outputs,
    )
    return result[0] if do_finalize else routes


# ===-----------------------------------------------------------------------===#
# NVFP4 MegaMoE: TRT-LLM (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


def trtllm_mega_moe_process_weights(plan: dict, w: torch.nn.Module) -> None:
    from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.blocked_scale import to_blocked

    del plan
    if w.tp_size != 1:
        raise ValueError("NVFP4 MegaMoE requires MoE TP=1")
    gate_scale, up_scale = w.w13_weight_scale_2.unbind(dim=1)
    if not torch.equal(gate_scale, up_scale):
        raise ValueError("NVFP4 MegaMoE requires equal gate/up weight_scale_2")
    experts = w.num_local_experts
    intermediate = w.intermediate_size
    for name in ("w13_weight", "w13_weight_scale"):
        tensor = getattr(w, name).data.view(torch.uint8)
        gate, up = tensor.split(intermediate, dim=1)
        packed = torch.stack(
            (
                gate.reshape(experts, intermediate // 16, 16, -1),
                up.reshape(experts, intermediate // 16, 16, -1),
            ),
            dim=2,
        )
        getattr(w, name).data = packed.reshape_as(tensor).contiguous()
    for name in ("w13_weight_scale", "w2_weight_scale"):
        tensor = getattr(w, name).data.view(torch.float8_e4m3fn)
        getattr(w, name).data = torch.stack(
            [to_blocked(sf).view(torch.uint8) for sf in tensor]
        )
    w.w13_input_scale_quant = torch.nn.Parameter(
        w.w13_input_scale.reciprocal(), requires_grad=False
    )
    w.mega_fc1_alpha = torch.nn.Parameter(
        (w.w13_input_scale * gate_scale).float(), requires_grad=False
    )
    w.mega_fc2_alpha = torch.nn.Parameter(
        (w.w2_input_scale * w.w2_weight_scale_2).float(), requires_grad=False
    )
    w.mega_fc1_norm = torch.nn.Parameter(
        w.w2_input_scale.reciprocal().expand(experts).contiguous(), requires_grad=False
    )
    for name in (
        "w13_input_scale",
        "w2_input_scale",
        "w13_weight_scale_2",
        "w2_weight_scale_2",
    ):
        delattr(w, name)


@register_kernel(
    "moe",
    "apply",
    name="trtllm_nvfp4_mega_moe_apply",
    solution="mega_moe",
    weight_preprocessor=trtllm_mega_moe_process_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.uint8}),
    traits={
        "weight_dtype": frozenset({"nvfp4"}),
        "activation": frozenset({"silu", "situ"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({16}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE - 1,
)
def trtllm_nvfp4_mega_moe_apply(
    plan: dict,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_tokens_global: int,
    max_num_tokens_per_gpu: int,
    do_finalize: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Dispatch NVFP4 tokens, run gated SiLU or SiTU experts, and combine on the source ranks.

    Uses the upstream heuristic by default. Set MEGAMOE_TACTIC_AUTOTUNE=1
    identically on every EP rank before startup to participate in FlashInfer's
    autotuning context and reuse tuned tactics across compatible layers.
    Autotuning includes input staging in each measured call. One buffer set
    sized by the configured autotuning maximum backs all token views and tactics.
    Fixed scratch offsets let captured calls alternate sizes without host resets.
    Outside tuning, cache misses use the heuristic tactic. Tune and warm up
    before CUDA graph capture; all EP ranks must enter with identical caches.

    Args:
        plan: Selected NVFP4 MoE plan containing the EP process group.
        x: BF16 activations, or packed E2M1 activations and linear E4M3 scales.
        w: Local expert module with MegaMoE-prepared weights.
        router_logits: Unused; routes are supplied explicitly.
        topk_weights: Final routing weights, including the model routing scale.
        topk_ids: Global expert indices for local tokens.
        num_tokens_global: Global token capacity for this collective step.
        max_num_tokens_per_gpu: Rank-identical local token bound for this call.
        do_finalize: Must be true; this operation owns the complete routed result.
        enable_pdl: Reserved; the vendored kernel controls its own launch ordering.

    Returns:
        Local BF16 routed output, valid until the next use of the shared workspace.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.runner import get_runner

    del router_logits, num_tokens_global, enable_pdl
    if not do_finalize:
        raise ValueError("MegaMoE requires complete finalization")
    prequantized = isinstance(x, tuple)
    if not prequantized and x.dtype != torch.bfloat16:
        raise ValueError("MegaMoE requires BF16 or prequantized NVFP4 inputs")
    runner = get_runner(plan, w, prequantized=prequantized)
    weights = (
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        w.mega_fc1_alpha,
        w.mega_fc2_alpha,
        w.mega_fc1_norm,
    )
    routed_input = x if prequantized else (x, w.w13_input_scale_quant)
    return runner.run(
        routed_input, topk_ids, topk_weights, weights, max_num_tokens_per_gpu
    )
