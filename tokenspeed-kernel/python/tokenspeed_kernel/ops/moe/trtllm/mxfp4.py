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
from tokenspeed_kernel.thirdparty.trtllm_native_moe import (
    has_native_mxfp4_moe,
    run_native_mxfp4_moe,
    select_native_mxfp4_moe_tactic,
)

platform = current_platform()

if platform.is_nvidia:
    # Reuse the established FlashInfer weight transformation and activation
    # quantization so the first native-runner A/B changes as little as possible.
    # Only the source runner's required physical padding differs. These helpers
    # can move behind the native TRT-LLM boundary after parity is established.
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        flashinfer_trtllm_mxfp4_moe_weights,
        mxfp8_quantize,
    )


def _validate_topk(
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk_weights is None or topk_ids is None:
        raise ValueError(
            "native TRT-LLM MXFP4 MoE requires precomputed topk_weights and topk_ids"
        )

    expected_shape = (x.shape[0], w.top_k)
    if topk_weights.shape != expected_shape or topk_ids.shape != expected_shape:
        raise ValueError(
            f"expected topk_weights and topk_ids shape {expected_shape}, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if topk_weights.device != x.device or topk_ids.device != x.device:
        raise ValueError(
            "topk_weights, topk_ids, and hidden states must share a device"
        )
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("topk_weights and topk_ids must be contiguous")
    return topk_weights.to(torch.bfloat16), topk_ids


def trtllm_mxfp4_moe_apply(
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
    """Run the native TRT-LLM MXFP8-by-MXFP4 routed MoE kernel.

    The selected expert ids are global int32 ids. Routing weights are converted
    to BF16, matching the native runner's precomputed-routing ABI.
    """
    del plan, router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl

    if not do_finalize:
        raise ValueError("native TRT-LLM MXFP4 MoE requires do_finalize=True")
    if getattr(w, "ep_num_redundant_experts", 0) not in (None, 0):
        raise ValueError(
            "native TRT-LLM MXFP4 MoE does not support redundant experts or EPLB"
        )

    hidden_padded = getattr(w, "hidden_size_padded", w.w13_weight.shape[-1] * 2)
    hidden_valid = getattr(
        w,
        "hidden_size_original",
        getattr(w, "hidden_size", hidden_padded),
    )
    if hidden_valid != hidden_padded:
        raise ValueError(
            "native TRT-LLM MXFP4 MoE currently requires hidden_size divisible "
            "by 512 because TokenSpeed stores one padded hidden dimension for "
            "both expert GEMMs"
        )
    if x.shape[0] == 0:
        return x.new_empty((0, hidden_valid), dtype=torch.bfloat16)

    topk_weights_bf16, topk_ids = _validate_topk(x, w, topk_weights, topk_ids)
    x_quant, x_scale = mxfp8_quantize(x, False, alignment=hidden_padded)
    if x_quant.shape[-1] != hidden_padded:
        raise RuntimeError(
            f"expected quantized hidden size {hidden_padded}, got {x_quant.shape[-1]}"
        )

    local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
    valid_intermediate_size = w.intermediate_size // getattr(w, "tp_size", 1)
    output = torch.empty(
        x_quant.shape[0], hidden_valid, dtype=torch.bfloat16, device=x_quant.device
    )
    tactic = select_native_mxfp4_moe_tactic(
        num_tokens=x_quant.shape[0],
        hidden_size=hidden_padded,
        intermediate_size=w.intermediate_size_per_partition,
        valid_hidden_size=hidden_valid,
        valid_intermediate_size=valid_intermediate_size,
        local_num_experts=local_experts,
        top_k=w.top_k,
        device=x_quant.device,
    )
    return run_native_mxfp4_moe(
        hidden_states=x_quant,
        hidden_states_scale=x_scale.view(torch.uint8).flatten(),
        gemm1_weights=w.w13_weight.view(torch.uint8),
        gemm1_weights_scale=w.w13_weight_scale.view(torch.uint8),
        gemm1_bias=getattr(w, "w13_weight_bias", None),
        gemm1_alpha=getattr(w, "gemm1_alpha", None),
        gemm1_beta=getattr(w, "gemm1_beta", None),
        gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
        gemm2_weights=w.w2_weight.view(torch.uint8),
        gemm2_weights_scale=w.w2_weight_scale.view(torch.uint8),
        gemm2_bias=getattr(w, "w2_weight_bias", None),
        num_experts=w.num_experts,
        top_k=w.top_k,
        intermediate_size=w.intermediate_size_per_partition,
        valid_hidden_size=hidden_valid,
        valid_intermediate_size=valid_intermediate_size,
        local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
        local_num_experts=local_experts,
        topk_weights=topk_weights_bf16,
        topk_ids=topk_ids,
        output=output,
        tactic=tactic,
    )


if platform.is_nvidia and has_native_mxfp4_moe():
    register_kernel(
        "moe",
        "apply",
        name="trtllm_mxfp4_moe_apply",
        solution="trtllm",
        weight_preprocessor=flashinfer_trtllm_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({True}),
        },
        # Keep auto selection on the established FlashInfer backend until the
        # native runner has completed correctness and full-sweep validation.
        priority=Priority.PERFORMANT + 3,
    )(trtllm_mxfp4_moe_apply)


__all__ = ["trtllm_mxfp4_moe_apply"]
