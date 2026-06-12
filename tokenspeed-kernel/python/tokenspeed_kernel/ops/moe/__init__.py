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
from typing import Any

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.moe.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.moe.gluon  # noqa: F401
import tokenspeed_kernel.ops.moe.triton  # noqa: F401
import torch
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
]


def _normalize_weight_dtype(weight_dtype: str) -> str:
    if weight_dtype in {"bf16", "fp16", "float16", "bfloat16", "unquantized"}:
        return "unquant"
    return weight_dtype


def _uses_all_to_all_ep(a2a_backend: str | None) -> bool:
    return a2a_backend not in {None, "none"}


def _validate_a2a_backend(a2a_backend: str | None) -> None:
    if a2a_backend in {None, "none", "deepep"}:
        return
    raise NotImplementedError(f"MoE all-to-all backend is unsupported: {a2a_backend}")


def _build_traits(
    *,
    weight_dtype: str,
    activation: str | None,
    requires_deferred_finalize: bool,
    a2a_backend: str | None,
    ep_size: int | None,
    ispp: int | None,
    fp8_scale_block_shape: tuple[int, int] | None,
    internal_activation_dtype: str | None,
    with_bias: bool,
) -> dict[str, Any]:
    traits: dict[str, Any] = {"weight_dtype": weight_dtype}
    if activation is not None:
        traits["activation"] = activation
    if requires_deferred_finalize:
        traits["supports_deferred_finalize"] = True

    all_to_all_ep = _uses_all_to_all_ep(a2a_backend)
    traits["supports_all_to_all_ep"] = all_to_all_ep
    if all_to_all_ep or (ep_size is not None and ep_size > 1):
        traits["supports_ep"] = True

    if ispp is not None:
        traits["ispp"] = int(ispp)
    if fp8_scale_block_shape is not None:
        traits["fp8_scale_block_shape"] = tuple(fp8_scale_block_shape)
    if internal_activation_dtype is not None:
        traits["internal_activation_dtype"] = internal_activation_dtype
    if with_bias:
        traits["supports_bias"] = True
    return traits


def moe_plan(
    weight_dtype: str,
    input_dtype: torch.dtype = torch.bfloat16,
    activation: str | None = None,
    requires_deferred_finalize: bool = False,
    a2a_backend: str | None = None,
    ep_size: int | None = None,
    ispp: int | None = None,
    fp8_scale_block_shape: tuple[int, int] | None = None,
    internal_activation_dtype: str | None = None,
    with_bias: bool = False,
    deepep_group: object | None = None,
    solution: str | None = None,
) -> dict:
    """Create a MoE execution plan.

    Args:
        weight_dtype: Logical MoE weight dtype. fp16, bf16, float16,
            bfloat16, and unquantized aliases map to unquant.
        input_dtype: Hidden-state dtype used for the apply-kernel signature.
        activation: Optional activation name required by the layer.
        requires_deferred_finalize: Require a kernel that can defer finalize.
        a2a_backend: Optional all-to-all backend. deepep selects the DeepEP
            solution when solution is not set.
        ep_size: Optional expert-parallel size. Values > 1 require EP support.
        ispp: Optional intermediate size per partition for alignment checks.
        fp8_scale_block_shape: Optional FP8 block-scale shape requirement.
        internal_activation_dtype: Optional internal activation dtype requirement.
        with_bias: Whether the selected kernel must support expert bias tensors.
        deepep_group: Runtime-created process group used by DeepEP plans.
        solution: Optional kernel solution to force through normal selection.
            None leaves the concrete kernel choice to the registry.

    The selected apply kernel owns plan metadata. A plan with support_routing
    false requires precomputed top-k ids and weights when calling moe_apply.
    Process-weights is pinned to the processor selected for the chosen apply
    solution when weights are loaded.
    """
    weight_dtype = _normalize_weight_dtype(weight_dtype)
    _validate_a2a_backend(a2a_backend)
    if solution is None and a2a_backend == "deepep":
        solution = "flashinfer_cutedsl_deepep"

    traits = _build_traits(
        weight_dtype=weight_dtype,
        activation=activation,
        requires_deferred_finalize=requires_deferred_finalize,
        a2a_backend=a2a_backend,
        ep_size=ep_size,
        ispp=ispp,
        fp8_scale_block_shape=fp8_scale_block_shape,
        internal_activation_dtype=internal_activation_dtype,
        with_bias=with_bias,
    )

    kernel = select_kernel(
        "moe",
        "apply",
        format_signature(x=dense_tensor_format(input_dtype)),
        traits=traits,
        solution=solution,
    )
    registry = KernelRegistry.get()
    apply_spec = registry.get_by_name(kernel.name)
    if apply_spec is None:
        raise RuntimeError(f"Kernel spec not found for selected kernel {kernel.name}")

    process_weights_kernel = select_kernel(
        "moe",
        "process_weights",
        format_signature(),
        traits={"weight_dtype": weight_dtype},
        solution=apply_spec.solution,
    )
    process_weights_spec = registry.get_by_name(process_weights_kernel.name)
    if process_weights_spec is None:
        raise RuntimeError(
            f"Kernel spec not found for selected kernel {process_weights_kernel.name}"
        )

    routing_modes = apply_spec.traits.get("routing_mode", frozenset())
    support_routing = "kernel_routing" in routing_modes
    supports_deferred_finalize = True in apply_spec.traits.get(
        "supports_deferred_finalize", frozenset({False})
    )
    return {
        "weight_dtype": weight_dtype,
        "apply_kernel_name": apply_spec.name,
        "process_weights_kernel_name": process_weights_spec.name,
        "a2a_backend": a2a_backend,
        "deepep_group": deepep_group,
        "support_routing": support_routing,
        "supports_deferred_finalize": supports_deferred_finalize,
        "solution": apply_spec.solution,
        "internal_activation_dtype": internal_activation_dtype,
    }


def moe_process_weights(plan: dict, w: torch.nn.Module):
    """Process loaded MoE weights according to a plan.

    Args:
        plan: Execution plan returned by moe_plan.
        w: Module containing loaded MoE weights. This module is mutated in
            place to prepare solution-specific layouts and scales.
    """
    kernel = select_kernel(
        "moe",
        "process_weights",
        format_signature(),
        override=plan["process_weights_kernel_name"],
    )
    return kernel(plan=plan, w=w)


def moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    # top-k routing inputs
    router_logits: torch.Tensor,
    # top-k routing results
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    # token length
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    # launch config
    enable_pdl: bool = False,
):
    """Apply a planned MoE kernel.

    Args:
        plan: Execution plan returned by moe_plan.
        x: Hidden states with shape [tokens, hidden_size].
        w: Module containing processed MoE weights.
        router_logits: Router logits with shape [tokens, num_experts].
        topk_weights: Optional precomputed expert weights with shape
            [tokens, top_k]. Required when plan support_routing is false.
        topk_ids: Optional precomputed expert ids with shape [tokens, top_k].
            Required when plan support_routing is false.
        num_tokens_global: Optional global token count for distributed MoE.
        max_num_tokens_per_gpu: Optional per-GPU token capacity hint.

    Solutions may use precomputed top-k tensors or route from logits directly.
    """
    kernel = select_kernel(
        "moe",
        "apply",
        format_signature(x=dense_tensor_format(x.dtype)),
        override=plan["apply_kernel_name"],
    )
    return kernel(
        plan=plan,
        x=x,
        w=w,
        router_logits=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=num_tokens_global,
        max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
    )
