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
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401
    import triton_kernels.matmul  # noqa: F401
    import triton_kernels.tensor  # noqa: F401
    import triton_kernels.tensor_details  # noqa: F401
    import triton_kernels.tensor_details.layout  # noqa: F401
    import triton_kernels.topk  # noqa: F401

from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
    _local_topk_for_ep,
    _release_parameter,
    _routing_from_topk,
    _silu_gate_up,
)
from triton_kernels.matmul import FnSpecs, FusedActivation, matmul
from triton_kernels.swiglu import swiglu_fn


def triton_unquant_moe_weights(plan: dict, w: torch.nn.Module):
    w.w13_weight_triton_tensor = w.w13_weight.transpose(-2, -1).contiguous()
    w.w2_weight_triton_tensor = w.w2_weight.transpose(-2, -1).contiguous()

    _release_parameter(w, "w13_weight")
    _release_parameter(w, "w2_weight")
    torch.cuda.empty_cache()


@register_kernel(
    "moe",
    "apply",
    name="triton_unquant_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_unquant_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE + 1,
)
@register_kernel(
    "moe",
    "apply",
    name="triton_unquant_ep_precomputed_moe_apply",
    solution="triton",
    weight_preprocessor=triton_unquant_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def triton_unquant_moe_apply(
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
    del plan, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise RuntimeError("triton unquant MoE does not support deferred finalize")
    if topk_weights is None or topk_ids is None:
        raise RuntimeError(
            "triton unquant MoE requires precomputed topk_weights/topk_ids"
        )
    if topk_weights.shape != topk_ids.shape:
        raise RuntimeError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )

    top_k = getattr(w, "top_k", topk_ids.shape[1])
    n_tokens = x.shape[0]
    topk_weights, topk_ids, num_experts = _local_topk_for_ep(
        topk_weights,
        topk_ids,
        w,
    )
    ragged_metadata, gather_indx, scatter_indx, gate_scal = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=router_logits.dtype,
    )

    swiglu_arg = getattr(w, "swiglu_arg", None)
    act = None
    if swiglu_arg is not None:
        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (swiglu_arg.alpha, swiglu_arg.limit),
        )

    intermediate_cache = matmul(
        x,
        w.w13_weight_triton_tensor,
        None,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=act,
    )
    if act is None:
        intermediate_cache = _silu_gate_up(
            intermediate_cache,
            output_dtype=x.dtype,
        )

    output = matmul(
        intermediate_cache,
        w.w2_weight_triton_tensor,
        None,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        gammas=gate_scal,
    )
    if top_k > 1:
        return output.view(n_tokens, top_k, output.shape[-1]).sum(dim=1)
    return output
