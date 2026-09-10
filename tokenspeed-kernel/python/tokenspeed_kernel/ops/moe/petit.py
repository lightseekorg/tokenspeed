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

"""Petit fused communication-and-compute MegaMoE integration for GPT-OSS and DeepSeek."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.petit import (
    import_petit_kernel as _import_petit_kernel,
)

_WORLD_SIZE = 8
_MAX_TOKENS_PER_RANK = 1024


@dataclass(frozen=True)
class _Profile:
    num_experts: int
    top_k: int
    model_dim: int
    logical_intermediate: int
    inter_dim: int
    has_bias: bool


_PROFILES = (
    _Profile(128, 4, 2880, 2880, 3072, True),
    _Profile(256, 8, 7168, 2048, 2048, False),
)


def _release_parameter(module: torch.nn.Module, name: str) -> None:
    if name in module._parameters:
        module.register_parameter(name, None)
    elif hasattr(module, name):
        delattr(module, name)


def _deinterleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    return (
        tensor.reshape(shape[0], shape[1] // 2, 2, *shape[2:])
        .permute(0, 2, 1, *range(3, tensor.dim() + 1))
        .contiguous()
        .reshape(shape)
    )


def _pad_mxfp4_pair(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_weight = weight.new_zeros((weight.shape[0], rows, columns // 2))
    padded_scale = scale.new_zeros((scale.shape[0], rows, columns // 32))
    padded_weight[:, : weight.shape[1], : weight.shape[2]].copy_(weight)
    padded_scale[:, : scale.shape[1], : scale.shape[2]].copy_(scale)
    return padded_weight, padded_scale


def _pad_gate_up_pair(
    weight: torch.Tensor,
    scale: torch.Tensor,
    profile: _Profile,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts = weight.shape[0]
    logical_intermediate = weight.shape[1] // 2
    padded_weight = weight.new_zeros(
        (experts, 2, profile.inter_dim, ((profile.model_dim + 511) // 512 * 512) // 2)
    )
    padded_scale = scale.new_zeros(
        (experts, 2, profile.inter_dim, ((profile.model_dim + 511) // 512 * 512) // 32)
    )
    weight = weight.reshape(experts, 2, logical_intermediate, weight.shape[2])
    scale = scale.reshape(experts, 2, logical_intermediate, scale.shape[2])
    padded_weight[:, :, :logical_intermediate, : weight.shape[3]].copy_(weight)
    padded_scale[:, :, :logical_intermediate, : scale.shape[3]].copy_(scale)
    return padded_weight.reshape(
        experts, 2 * profile.inter_dim, -1
    ), padded_scale.reshape(experts, 2 * profile.inter_dim, -1)


def _validate_layer(w: torch.nn.Module) -> _Profile:
    geometry = (w.num_experts, w.top_k, w.hidden_size, w.intermediate_size)
    profile = next(
        (
            p
            for p in _PROFILES
            if geometry == (p.num_experts, p.top_k, p.model_dim, p.logical_intermediate)
        ),
        None,
    )
    if profile is None:
        raise ValueError(
            f"Unsupported Petit MegaMoE geometry: num_experts={w.num_experts}, "
            f"top_k={w.top_k}, hidden_size={w.hidden_size}, "
            f"intermediate_size={w.intermediate_size}"
        )
    if (
        w.ep_size != 8
        or w.tp_size != 1
        or w.num_local_experts != profile.num_experts // 8
    ):
        raise ValueError("Petit MegaMoE requires EP8, TP1 and contiguous expert shards")
    if profile.has_bias:
        if w.activation != "swiglu" or w.swiglu_beta != 1.0:
            raise ValueError("Petit GPT-OSS MegaMoE requires OpenAI SWIGLU beta=1.0")
        if w.w13_input_layout != "interleaved":
            raise ValueError("Petit GPT-OSS MegaMoE requires interleaved W13 input")
    else:
        if w.activation not in {"silu", "swiglu"} or w.swiglu_beta is not None:
            raise ValueError("Petit DeepSeek MegaMoE requires standard SiLU activation")
        if w.w13_input_layout != "concatenated":
            raise ValueError("Petit DeepSeek MegaMoE requires concatenated W13 input")
        arg = getattr(w, "swiglu_arg", None)
        if arg is not None and arg.limit is not None:
            raise ValueError(
                "Petit DeepSeek MegaMoE does not support activation clamps"
            )
        if arg is not None and arg.alpha is not None:
            raise ValueError(
                "Petit DeepSeek MegaMoE does not support nonstandard SiLU alpha"
            )
        if (
            getattr(w, "w13_weight_bias", None) is not None
            or getattr(w, "w2_weight_bias", None) is not None
        ):
            raise ValueError("Petit DeepSeek MegaMoE requires bias-free experts")
    return profile


@dataclass
class _Workspace:
    config: Any
    heap: Any
    inputs: Any


_workspace_cache: dict[tuple[int, _Profile], _Workspace] = {}


def _workspace_key(device: torch.device, profile: _Profile) -> tuple[int, _Profile]:
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    return index, profile


def _get_workspace(device: torch.device, profile: _Profile) -> _Workspace:
    key = _workspace_key(device, profile)
    workspace = _workspace_cache.get(key)
    if workspace is not None:
        return workspace
    if not dist.is_initialized():
        raise RuntimeError("Petit MegaMoE requires torch.distributed initialization")
    if dist.get_world_size() != _WORLD_SIZE:
        raise RuntimeError(
            "Petit MegaMoE requires the default distributed world size 8"
        )
    petit_kernel = _import_petit_kernel()
    config = petit_kernel.MegaMoeConfig(
        world_size=_WORLD_SIZE,
        num_experts=profile.num_experts,
        topk=profile.top_k,
        model_dim=profile.model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=(
            petit_kernel.MegaMoeActivationFunction.swiglu
            if profile.has_bias
            else petit_kernel.MegaMoeActivationFunction.silu
        ),
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=profile.inter_dim,
        has_bias=profile.has_bias,
    )
    heap = petit_kernel.create_vmm_symmetric_heap(_WORLD_SIZE)
    inputs = config.input_views(heap, _MAX_TOKENS_PER_RANK)
    workspace = _Workspace(config=config, heap=heap, inputs=inputs)
    _workspace_cache[key] = workspace
    return workspace


def _slice_inputs(inputs: Any, num_tokens: int) -> Any:
    """Return exact-row views required by Petit's quantize and run APIs."""
    return type(inputs)(
        tokens=inputs.tokens[:num_tokens],
        scales=None if inputs.scales is None else inputs.scales[:num_tokens],
        expert_ids=inputs.expert_ids[:num_tokens],
        expert_weights=inputs.expert_weights[:num_tokens],
    )


def petit_mxfp4_megamoe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack a loaded MXFP4 expert shard for Petit and initialize its heap."""
    del plan
    profile = _validate_layer(w)
    w.petit_profile = profile
    _get_workspace(w.w13_weight.device, profile)
    petit_kernel = _import_petit_kernel()

    w13 = w.w13_weight.detach().contiguous()
    s13 = w.w13_weight_scale.detach().contiguous()
    if profile.has_bias:
        w13 = _deinterleave_gate_up(w13)
        s13 = _deinterleave_gate_up(s13)
    w13, s13 = _pad_gate_up_pair(w13, s13, profile)
    w2, s2 = _pad_mxfp4_pair(
        w.w2_weight.detach().contiguous(),
        w.w2_weight_scale.detach().contiguous(),
        rows=(profile.model_dim + 511) // 512 * 512,
        columns=profile.inter_dim,
    )
    layout = petit_kernel.MoeKernelLayout.native_mxfp4
    w13, s13 = petit_kernel.repack_moe_kernel_layout(w13, s13, layout=layout)
    w2, s2 = petit_kernel.repack_moe_kernel_layout(w2, s2, layout=layout)

    b13 = b2 = None
    if profile.has_bias:
        b13 = _deinterleave_gate_up(w.w13_weight_bias.detach().contiguous())
        logical_intermediate = b13.shape[1] // 2
        padded_b13 = b13.new_zeros((b13.shape[0], 2, profile.inter_dim))
        padded_b13[:, :, :logical_intermediate].copy_(
            b13.reshape(b13.shape[0], 2, logical_intermediate)
        )
        b13 = petit_kernel.repack_moe_kernel_layout(
            padded_b13.reshape(-1, profile.inter_dim), layout=layout
        ).reshape(b13.shape[0], -1)
        padded_b2 = w.w2_weight_bias.new_zeros(
            (w.w2_weight_bias.shape[0], profile.inter_dim)
        )
        padded_b2[:, : profile.model_dim].copy_(w.w2_weight_bias.detach())
        b2 = petit_kernel.repack_moe_kernel_layout(
            padded_b2.contiguous(), layout=layout
        )

    w.petit_w13_weight = w13.contiguous()
    w.petit_w2_weight = w2.contiguous()
    w.petit_w13_scale = s13.contiguous()
    w.petit_w2_scale = s2.contiguous()
    w.petit_w13_bias = None if b13 is None else b13.contiguous()
    w.petit_w2_bias = None if b2 is None else b2.contiguous()
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_bias",
        "w2_weight_bias",
    ):
        _release_parameter(w, name)
    torch.cuda.empty_cache()


@register_kernel(
    "moe",
    "apply",
    name="petit_mxfp4_megamoe_apply",
    solution="petit",
    weight_preprocessor=petit_mxfp4_megamoe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"swiglu", "silu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({True}),
        "ep_size": frozenset({_WORLD_SIZE}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input", "mxfp4"}),
        "supports_bias": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 3,
)
def petit_mxfp4_megamoe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
    low_latency: bool | None,
    overlap_fn: Callable[[], None] | None,
) -> torch.Tensor:
    """Run Petit MegaMoE for one rank's local tokens."""
    del plan, router_logits, num_tokens_global, enable_pdl, low_latency
    if not do_finalize:
        raise ValueError("Petit MegaMoE does not support deferred finalize")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Petit MegaMoE requires precomputed top-k ids and weights")
    if (
        max_num_tokens_per_gpu is not None
        and max_num_tokens_per_gpu > _MAX_TOKENS_PER_RANK
    ):
        raise ValueError(
            "Petit MegaMoE per-rank token count exceeds its 1024-token capacity"
        )
    num_tokens = int(x.shape[0])
    if num_tokens > _MAX_TOKENS_PER_RANK:
        raise ValueError("Petit MegaMoE received more than 1024 tokens on one rank")

    profile = w.petit_profile
    workspace = _get_workspace(x.device, profile)
    inputs = _slice_inputs(workspace.inputs, num_tokens)
    if num_tokens:
        workspace.config.quantize(x.contiguous(), out=inputs)
        inputs.expert_ids.copy_(topk_ids.to(torch.int32))
        inputs.expert_weights.copy_(topk_weights.to(torch.float32))
    if overlap_fn is not None:
        overlap_fn()
    output = x.new_empty((num_tokens, profile.model_dim))
    return workspace.config.run(
        workspace.heap,
        w.petit_w13_weight,
        w.petit_w2_weight,
        w.petit_w13_scale,
        w.petit_w2_scale,
        num_tokens,
        w13_bias=w.petit_w13_bias,
        w2_bias=w.petit_w2_bias,
        out=output,
        inputs=inputs,
    )
