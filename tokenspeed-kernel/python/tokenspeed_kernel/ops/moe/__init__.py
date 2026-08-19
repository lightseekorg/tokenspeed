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
from collections.abc import Callable
from typing import Any

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.moe.deep_gemm  # noqa: F401
import tokenspeed_kernel.ops.moe.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.moe.gluon  # noqa: F401
import tokenspeed_kernel.ops.moe.marlin  # noqa: F401
import tokenspeed_kernel.ops.moe.triton  # noqa: F401
import torch
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "native_latent_moe_available",
    "latent_moe_decode_pipeline_available",
    "latent_moe_expert_shared",
    "latent_moe_input_projections",
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
    "moe_sigmoid_bias_topk",
    "moe_softmax_topk",
]

from tokenspeed_kernel.ops.moe.latent_decode import (  # noqa: E402
    latent_moe_decode_pipeline_available,
    latent_moe_expert_shared,
)
from tokenspeed_kernel.ops.moe.latent_input import (  # noqa: E402
    latent_moe_input_projections,
)
from tokenspeed_kernel.ops.moe.native import native_latent_moe_available  # noqa: E402
from tokenspeed_kernel.ops.moe.sigmoid_topk import moe_sigmoid_bias_topk  # noqa: E402
from tokenspeed_kernel.ops.moe.softmax_topk import moe_softmax_topk  # noqa: E402


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


def _validate_routing_mode(routing_mode: str | None) -> None:
    if routing_mode in {None, "kernel_routing", "precomputed_topk"}:
        return
    raise ValueError(
        f"routing_mode must be 'kernel_routing' or 'precomputed_topk', "
        f"got {routing_mode!r}"
    )


def _validate_deepep_mode(a2a_backend: str | None, deepep_mode: str | None) -> None:
    if deepep_mode is None:
        return
    if deepep_mode not in {"auto", "normal", "low_latency"}:
        raise ValueError(
            "deepep_mode must be 'auto', 'normal' or 'low_latency', got "
            f"{deepep_mode!r}"
        )
    if not _uses_all_to_all_ep(a2a_backend):
        raise ValueError(
            f"deepep_mode={deepep_mode!r} requires an all-to-all backend, got "
            f"a2a_backend={a2a_backend!r}"
        )


def _validate_selected_deepep_mode(
    a2a_backend: str | None,
    deepep_mode: str | None,
    kernel_name: str,
    kernel_traits: dict[str, frozenset[Any]],
) -> None:
    """Reject a selected DeepEP kernel that lacks a requested collective leg."""
    if a2a_backend != "deepep":
        return
    supported_modes = kernel_traits.get("deepep_modes")
    if supported_modes is None:
        return

    requested_mode = deepep_mode or "auto"
    required_modes = (
        frozenset({"normal", "low_latency"})
        if requested_mode == "auto"
        else frozenset({requested_mode})
    )
    if required_modes.issubset(supported_modes):
        return

    supported = ", ".join(sorted(supported_modes))
    auto_note = (
        " 'auto' requires both normal and low_latency legs."
        if requested_mode == "auto"
        else ""
    )
    raise ValueError(
        f"MoE kernel {kernel_name!r} does not support "
        f"deepep_mode={requested_mode!r}; supported modes: {supported}."
        f"{auto_note}"
    )


def _build_traits(
    *,
    weight_dtype: str,
    activation: str | None,
    requires_deferred_finalize: bool,
    routing_mode: str | None,
    a2a_backend: str | None,
    ep_size: int | None,
    ispp: int | None,
    fp8_scale_block_shape: tuple[int, int] | None,
    internal_activation_dtype: str | None,
    with_bias: bool,
) -> dict[str, Any]:
    if internal_activation_dtype is None:
        internal_activation_dtype = "input"

    traits: dict[str, Any] = {"weight_dtype": weight_dtype}
    if activation is not None:
        traits["activation"] = activation
    if requires_deferred_finalize:
        traits["supports_deferred_finalize"] = True
    if routing_mode is not None:
        traits["routing_mode"] = routing_mode

    all_to_all_ep = _uses_all_to_all_ep(a2a_backend)
    traits["supports_all_to_all_ep"] = all_to_all_ep
    if all_to_all_ep or (ep_size is not None and ep_size > 1):
        traits["supports_ep"] = True
    if ep_size is not None:
        # ``supports_ep`` distinguishes EP from non-EP plans. Keep the exact
        # degree as a separate selection trait so narrowly tuned EP kernels
        # (for example the gfx950 K3 EP8 Gluon path) do not become automatic
        # winners for unvalidated EP degrees.
        traits["ep_size"] = int(ep_size)

    if ispp is not None:
        traits["ispp"] = int(ispp)
    if fp8_scale_block_shape is not None:
        traits["fp8_scale_block_shape"] = tuple(fp8_scale_block_shape)
    traits["internal_activation_dtype"] = internal_activation_dtype
    if with_bias:
        traits["supports_bias"] = True
    return traits


def moe_plan(
    weight_dtype: str,
    input_dtype: torch.dtype = torch.bfloat16,
    activation: str | None = None,
    requires_deferred_finalize: bool = False,
    routing_mode: str | None = None,
    a2a_backend: str | None = None,
    ep_size: int | None = None,
    ispp: int | None = None,
    fp8_scale_block_shape: tuple[int, int] | None = None,
    internal_activation_dtype: str | None = None,
    with_bias: bool = False,
    deepep_group: object | None = None,
    deepep_mode: str | None = None,
    deepep_low_latency_max_num_tokens_per_gpu: int | None = None,
    solution: str | None = None,
) -> dict:
    """Create a MoE execution plan.

    Args:
        weight_dtype: Logical MoE weight dtype. fp16, bf16, float16,
            bfloat16, and unquantized aliases map to unquant.
        input_dtype: Hidden-state dtype used for the apply-kernel signature.
        activation: Optional activation name required by the layer.
        requires_deferred_finalize: Require a kernel that can defer finalize.
        routing_mode: Optional routing-mode requirement. "precomputed_topk"
            requires a kernel that consumes externally computed top-k ids and
            weights (for models whose routing function the fused kernels
            cannot reproduce); "kernel_routing" requires in-kernel routing
            from logits. None (default) leaves routing mode unconstrained.
        a2a_backend: Optional all-to-all backend. deepep selects the DeepEP
            solution when solution is not set.
        ep_size: Optional expert-parallel size. Values > 1 require EP support.
            The exact value is also passed as a selection trait when a kernel
            declares an ``ep_size`` constraint.
        ispp: Optional intermediate size per partition for alignment checks.
        fp8_scale_block_shape: Optional FP8 block-scale shape requirement.
        internal_activation_dtype: Optional internal activation dtype requirement.
            "input" is a special value that uses the whatever dtype the input
            activations have. "mxfp4" requests dynamic MXFP4 activation
            quantization. Defaults to "input" if not set.
        with_bias: Whether the selected kernel must support expert bias tensors.
        deepep_group: Runtime-created process group used by DeepEP plans.
        deepep_mode: Optional DeepEP mode for all-to-all plans: "low_latency"
            (decode-shaped batches only), "normal" (extend-shaped batches only),
            or "auto" to let each ``moe_apply`` pick through its ``low_latency``
            argument. Defaults to "auto".
        deepep_low_latency_max_num_tokens_per_gpu: Per-GPU token capacity the
            DeepEP low-latency buffer is sized for. Required whenever the mode
            can run the low-latency legs; batches above it must use normal mode.
        solution: Optional kernel solution to force through normal selection.
            None leaves the concrete kernel choice to the registry.

    The selected apply kernel owns plan metadata. A plan with support_routing
    false requires precomputed top-k ids and weights when calling moe_apply.
    Weight preprocessing is selected from the ordered candidates advertised by
    the selected apply kernel, then pinned by callable in the returned plan so load
    time does not rerun selection or conflict resolution.
    """
    weight_dtype = _normalize_weight_dtype(weight_dtype)
    _validate_a2a_backend(a2a_backend)
    _validate_routing_mode(routing_mode)
    _validate_deepep_mode(a2a_backend, deepep_mode)
    # DeepEP does not pin a solution: the ``supports_all_to_all_ep`` trait plus
    # ``weight_dtype`` already narrow the candidates to the apply kernels that
    # own the dispatch/combine legs (nvfp4 cutedsl, block-scale fp8 DeepGEMM).
    # Callers may still force one explicitly through ``solution``.

    traits = _build_traits(
        weight_dtype=weight_dtype,
        activation=activation,
        requires_deferred_finalize=requires_deferred_finalize,
        routing_mode=routing_mode,
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
    _validate_selected_deepep_mode(
        a2a_backend,
        deepep_mode,
        apply_spec.name,
        apply_spec.traits,
    )

    routing_modes = apply_spec.traits.get("routing_mode", frozenset())
    support_routing = "kernel_routing" in routing_modes
    supports_deferred_finalize = True in apply_spec.traits.get(
        "supports_deferred_finalize", frozenset({False})
    )
    return {
        "weight_dtype": weight_dtype,
        "activation": activation,
        "apply_kernel_name": apply_spec.name,
        "weight_preprocessor": apply_spec.weight_preprocessor,
        "a2a_backend": a2a_backend,
        "deepep_group": deepep_group,
        "deepep_mode": deepep_mode or "auto",
        "deepep_low_latency_max_num_tokens_per_gpu": (
            deepep_low_latency_max_num_tokens_per_gpu
        ),
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
    preprocessor = plan.get("weight_preprocessor")
    if preprocessor is None:
        return None
    if not callable(preprocessor):
        raise RuntimeError(f"Weight preprocessor is not callable: {preprocessor!r}")
    return preprocessor(plan=plan, w=w)


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
    # all-to-all EP
    low_latency: bool | None = None,
    overlap_fn: Callable[[], None] | None = None,
    shared_input: torch.Tensor | None = None,
    shared_weight: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
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
        do_finalize: Whether the kernel must produce the finalized output.
        enable_pdl: Whether kernels may honor programmatic dependent launch.
        low_latency: Only forwarded to all-to-all EP plans, and only meaningful
            when the plan mode is "auto": True selects the latency-optimized
            dispatch/combine legs (decode-shaped batches), False the
            throughput-optimized ones (extend-shaped batches). Every rank of the
            EP group must pass the same value, since the two legs are different
            collectives.
        overlap_fn: Only forwarded to all-to-all EP plans. Work queued here runs
            inside the dispatch window (tokens sent, not yet awaited), so it
            overlaps the transfer. It must not read the dispatch result or write
            ``x``.

    Solutions may use precomputed top-k tensors or route from logits directly.
    """
    kernel = select_kernel(
        "moe",
        "apply",
        format_signature(x=dense_tensor_format(x.dtype)),
        override=plan["apply_kernel_name"],
    )
    # Only the all-to-all EP kernels own dispatch/combine legs, so the mode
    # decision stays off the signature every other apply kernel implements.
    a2a_kwargs = (
        {"low_latency": low_latency, "overlap_fn": overlap_fn}
        if _uses_all_to_all_ep(plan.get("a2a_backend"))
        else {}
    )
    shared_tensors = (shared_input, shared_weight, shared_out)
    if any(value is not None for value in shared_tensors) and not all(
        value is not None for value in shared_tensors
    ):
        raise ValueError("joint shared projection requires input, weight, and output")
    shared_kwargs = (
        {
            "shared_input": shared_input,
            "shared_weight": shared_weight,
            "shared_out": shared_out,
        }
        if all(value is not None for value in shared_tensors)
        else {}
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
        **a2a_kwargs,
        **shared_kwargs,
    )
