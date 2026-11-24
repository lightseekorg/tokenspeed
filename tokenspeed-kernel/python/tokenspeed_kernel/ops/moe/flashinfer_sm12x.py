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
from tokenspeed_kernel.ops.moe._triton.mxfp4 import _validate
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.sm12x_fused_moe import (
    Sm12xFusedMoEWeights,
    apply_sm12x_fused_moe,
    prepare_sm12x_fused_moe,
)


def flashinfer_sm12x_mxfp4_weights(plan: dict, w: torch.nn.Module) -> None:
    if isinstance(w._moe_backend_state, Sm12xFusedMoEWeights):
        return
    if w._moe_backend_state is not None:
        raise ValueError("Cannot reuse another backend's prepared MoE weights")
    w._moe_backend_state = prepare_sm12x_fused_moe(w)


@register_kernel(
    "moe",
    "apply",
    name="flashinfer_sm12x_mxfp4_moe_apply",
    solution="flashinfer",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False, True}),
        "expert_id_repeats": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({128}),
        "hidden_alignment": frozenset({512}),
        "internal_activation_dtype": frozenset({"fp8"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    weight_preprocessor=flashinfer_sm12x_mxfp4_weights,
)
def flashinfer_sm12x_mxfp4_moe_apply(
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
    """Compute local contributions while the caller retains EP communication."""
    if not do_finalize:
        raise ValueError("SM12x fused MoE requires finalization")
    _, _, _, expert_start = _validate(
        plan, x, w, topk_weights, topk_ids, do_finalize, upcast_weights=True
    )
    state = w._moe_backend_state
    if not isinstance(state, Sm12xFusedMoEWeights):
        raise RuntimeError("Weights must be prepared before graph capture")
    out = torch.empty_like(x)
    for start in range(0, x.shape[0], 4096):
        stop = min(start + 4096, x.shape[0])
        ids = topk_ids[start:stop].to(torch.int32) - expert_start
        ids = torch.where((ids >= 0) & (ids < w.num_local_experts), ids, -1)
        out[start:stop] = apply_sm12x_fused_moe(
            x[start:stop],
            state,
            ids,
            topk_weights[start:stop].float().contiguous(),
            enable_pdl,
        )
    return out


from tokenspeed_kernel.thirdparty.flashinfer.sm12x_nvfp4_moe import (
    Sm12xNvfp4MoEWeights,
    apply_sm12x_nvfp4,
    prepare_sm12x_nvfp4,
)


def flashinfer_sm12x_nvfp4_weights(plan: dict, w: torch.nn.Module) -> None:
    if isinstance(w._moe_backend_state, Sm12xNvfp4MoEWeights):
        return
    if w._moe_backend_state is not None:
        raise ValueError("Cannot reuse another backend's prepared MoE weights")
    w._moe_backend_state = prepare_sm12x_nvfp4(w)


@register_kernel(
    "moe",
    "apply",
    name="flashinfer_sm12x_nvfp4_moe_apply",
    solution="flashinfer",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"nvfp4"}),
        "activation": frozenset({"swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({True}),
        "expert_id_repeats": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({128}),
        "hidden_alignment": frozenset({512}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    weight_preprocessor=flashinfer_sm12x_nvfp4_weights,
)
def flashinfer_sm12x_nvfp4_moe_apply(
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
):
    """Compute local contributions while the caller owns EP communication."""
    if not do_finalize or topk_ids is None or topk_weights is None:
        raise ValueError("NVFP4 requires finalized precomputed routes")
    state = w._moe_backend_state
    if not isinstance(state, Sm12xNvfp4MoEWeights):
        raise RuntimeError("NVFP4 weights must be prepared before graph capture")
    ids = topk_ids.to(torch.int32) - w.ep_rank * w.num_local_experts
    ids = torch.where((ids >= 0) & (ids < w.num_local_experts), ids, -1)
    return apply_sm12x_nvfp4(x, state, ids, topk_weights, enable_pdl)
