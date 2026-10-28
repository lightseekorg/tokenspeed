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

import functools
import logging
import warnings

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.megamoe import (
    MegaMoeState,
    megamoe_unavailable_reason,
    prepare_megamoe,
)


def flashinfer_megamoe_nvfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Replace checkpoint weights with the single upstream MegaMoE NVFP4 layout.

    Args:
        plan: Selected MoE plan, including the explicit fast-math policy.
        w: Expert module whose registered weights and backend state share the
            prepared MegaMoE storage.

    Returns:
        None; preparation updates the expert module in place.
    """
    if w.w13_weight.shape != (512, 1280, 1280) or w.top_k != 10:
        raise ValueError("MegaMoE requires E512/H2560/I640/K10")
    w._moe_backend_state = prepare_megamoe(w, fast_math=plan["fast_math"])


@functools.cache
def _warn_pdl_disabled() -> None:
    warnings.warn(
        "FlashInfer MegaMoE uses ordinary stream ordering; enable_pdl=True "
        "does not enable PDL for this implementation.",
        stacklevel=3,
    )


def flashinfer_megamoe_nvfp4_moe_apply(
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
) -> torch.Tensor:
    """Run Generic Rubin MegaMoE for every nonempty input, including decode.

    Args:
        plan: Selected MoE plan.
        x: Contiguous BF16 [tokens, hidden] activations.
        w: Prepared expert module.
        router_logits: BF16 [tokens, experts] logits for Qwen softmax top-k routing.
            Optional when top-k IDs and weights are supplied; explicit routes
            always take precedence and are not recomputed.
        topk_weights: Optional contiguous FP32 [tokens, top_k] late route weights.
        topk_ids: Optional contiguous int32 [tokens, top_k] IDs; -1 pads routes.
            IDs and weights must be supplied together or both omitted.
        num_tokens_global: Optional single-rank token count, equal to input rows.
        max_num_tokens_per_gpu: Optional upper bound on input rows used to reserve
            workspace. Without a hint, reserve for the current input. Capacity
            grows in power-of-two buckets during eager execution, never capture.
        do_finalize: Must be True; the kernel combines the expert outputs.
        enable_pdl: MegaMoE warns and uses ordinary stream ordering when True.

    Returns:
        BF16 [tokens, hidden] finalized output with independent storage.
    """
    if not do_finalize:
        raise ValueError("MegaMoE requires finalized output")
    if (topk_weights is None) != (topk_ids is None):
        raise ValueError("MegaMoE requires top-k IDs and FP32 weights together")
    from_logits = topk_ids is None
    if router_logits is not None and (
        router_logits.ndim != 2 or router_logits.shape[0] != x.shape[0]
    ):
        raise ValueError("MegaMoE source logits must have one row per input token")
    if num_tokens_global is not None and num_tokens_global != x.shape[0]:
        raise ValueError("Single-rank MegaMoE requires num_tokens_global == input rows")
    if max_num_tokens_per_gpu is not None and max_num_tokens_per_gpu < x.shape[0]:
        raise ValueError("MegaMoE capacity hint is smaller than the input")
    state = w._moe_backend_state
    if not isinstance(state, MegaMoeState):
        raise RuntimeError("FlashInfer MegaMoE weights have not been prepared")
    if from_logits:
        state.validate_hidden(x)
        if (
            router_logits is None
            or router_logits.shape != (x.shape[0], 512)
            or router_logits.dtype != torch.bfloat16
            or router_logits.device != x.device
            or not router_logits.is_contiguous()
        ):
            raise ValueError("MegaMoE requires contiguous BF16 [tokens, 512] logits")
        # FlashInfer method 4 is Qwen's softmax -> top-k -> renormalize.
        if (
            w._routing_method_type != 4
            or not w._normalize_topk_weights
            or w._correction_bias is not None
            or w._n_group != 0
            or w._topk_group != 0
            or w._routed_scaling_factor != 1.0
            or w._routing_logits_dtype != torch.bfloat16
        ):
            raise ValueError("MegaMoE logits routing requires Qwen RenormalizeNaive")
    else:
        state.validate_inputs(x, topk_ids, topk_weights)
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if from_logits:
        from tokenspeed_kernel.ops.moe import moe_topk

        topk_weights, topk_ids = moe_topk(
            router_logits,
            state.top_k,
            score_function="softmax",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=1.0,
            topk_indices_dtype=torch.int32,
            topk_weights_dtype=torch.float32,
        )
    if enable_pdl:
        _warn_pdl_disabled()
    capacity = x.shape[0] if max_num_tokens_per_gpu is None else max_num_tokens_per_gpu
    return state.apply(x, topk_ids, topk_weights, capacity=capacity)


platform = current_platform()
if platform.is_nvidia and platform.arch_version == ArchVersion(10, 7):
    reason = megamoe_unavailable_reason()
    if reason is not None:
        logging.getLogger(__name__).info(f"Skipping flashinfer_megamoe: {reason}")
    else:
        register_kernel(
            "moe",
            "apply",
            name="flashinfer_megamoe_nvfp4_moe_apply",
            solution="flashinfer_megamoe",
            weight_preprocessor=flashinfer_megamoe_nvfp4_moe_weights,
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(10, 7),
                max_arch_version=ArchVersion(10, 7),
            ),
            signatures=format_signatures("x", "dense", {torch.bfloat16}),
            traits={
                "weight_dtype": frozenset({"nvfp4"}),
                "activation": frozenset({"silu", "swiglu"}),
                "swiglu_form": frozenset({"standard"}),
                "activation_clamped": frozenset({False}),
                "routing_mode": frozenset({"kernel_routing", "precomputed_topk"}),
                "supports_deferred_finalize": frozenset({False}),
                "supports_ep": frozenset({False}),
                "ep_size": frozenset({1}),
                "num_experts": frozenset({512}),
                "top_k": frozenset({10}),
                "hidden": frozenset({2560}),
                "ispp": frozenset({640}),
                "supports_all_to_all_ep": frozenset({False}),
                "ispp_alignment": frozenset({128}),
                "hidden_alignment": frozenset({128}),
                "internal_activation_dtype": frozenset({"input"}),
                "topk_weights_dtype": frozenset({torch.float32}),
                "supports_bias": frozenset({False}),
            },
            # Experimental opt-in: model-level performance gains are unproven.
            priority=Priority.PORTABLE,
        )(flashinfer_megamoe_nvfp4_moe_apply)
