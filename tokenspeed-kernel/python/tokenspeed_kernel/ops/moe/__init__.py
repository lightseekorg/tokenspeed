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
from typing import Any, Literal

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.moe.cuda  # noqa: F401
import tokenspeed_kernel.ops.moe.deep_gemm  # noqa: F401
import tokenspeed_kernel.ops.moe.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.moe.gluon  # noqa: F401
import tokenspeed_kernel.ops.moe.marlin  # noqa: F401
import tokenspeed_kernel.ops.moe.mega_moe  # noqa: F401
import tokenspeed_kernel.ops.moe.triton  # noqa: F401
import torch
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
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
    "moe_topk",
]

from tokenspeed_kernel.ops.moe.latent_decode import (  # noqa: E402
    latent_moe_decode_pipeline_available,
    latent_moe_expert_shared,
)
from tokenspeed_kernel.ops.moe.latent_input import (  # noqa: E402
    latent_moe_input_projections,
)
from tokenspeed_kernel.ops.moe.native import native_latent_moe_available  # noqa: E402
from tokenspeed_kernel.ops.moe.sigmoid_topk import (  # noqa: E402
    _moe_sigmoid_bias_topk,
)
from tokenspeed_kernel.ops.moe.softmax_topk import _moe_softmax_topk  # noqa: E402


def _assert_indices_in_range(
    indices: torch.Tensor,
    upper_bound: int,
    name: str,
) -> None:
    valid = ((indices >= 0) & (indices < upper_bound)).all()
    message = f"{name} entries must be in [0, {upper_bound})"
    if indices.device.type == "cpu":
        if not bool(valid.item()):
            raise ValueError(message)
    else:
        torch._assert_async(valid, message)


def _routing_kind(
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> str:
    if hash_indices_table is not None:
        return "hash"
    if correction_bias is not None:
        return "bias"
    return "plain"


def moe_topk(
    router_logits: torch.Tensor,
    top_k: int,
    score_function: Literal["softmax", "sigmoid", "sqrt_softplus"],
    selection_method: Literal["topk", "hash"],
    renormalize: bool,
    routed_scaling_factor: float | None,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    topk_indices_dtype: torch.dtype = torch.int32,
    topk_weights_dtype: torch.dtype = torch.float32,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce expert weights and ids using a registered MoE TopK kernel.

    Correction bias affects selection only; returned weights are gathered from
    the unbiased scores. Hash routing uses the checkpoint table for expert ids.

    Args:
        router_logits: Router logits shaped [tokens, experts].
        top_k: Number of experts selected for each token.
        score_function: Transformation from router logits to routing scores.
        selection_method: Select experts by score or token hash.
        renormalize: Whether selected routing weights sum to one.
        routed_scaling_factor: Optional scale applied to selected weights.
        correction_bias: Optional selection-only bias shaped [experts] or
            [tokens, experts].
        hash_indices_table: Optional token-id to expert-id table.
        input_ids: Token ids used with hash_indices_table.
        logical_to_physical_map: Optional expert-id map for sigmoid routing.
        topk_indices_dtype: Integer dtype for returned expert ids.
        topk_weights_dtype: Floating-point dtype for returned routing weights.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    Returns:
        Weights and expert ids shaped [tokens, top_k], using the requested
        output dtypes (FP32 weights and INT32 ids by default).
    """
    if selection_method not in {"topk", "hash"}:
        raise ValueError(f"unsupported MoE selection method: {selection_method!r}")
    if selection_method == "hash":
        if correction_bias is not None:
            raise ValueError("hash selection does not accept correction_bias")
        if hash_indices_table is None or input_ids is None:
            raise ValueError("hash selection requires hash_indices_table and input_ids")
    elif hash_indices_table is not None or input_ids is not None:
        raise ValueError("hash routing inputs require hash selection")
    if topk_indices_dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_indices_dtype must be torch.int32 or torch.int64")
    if not topk_weights_dtype.is_floating_point:
        raise ValueError("topk_weights_dtype must be a floating-point dtype")

    scaling_factor = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    if score_function == "softmax":
        if selection_method != "topk":
            raise ValueError("softmax routing only supports topk selection")
        if correction_bias is not None:
            raise ValueError("softmax routing does not accept correction_bias")
        if logical_to_physical_map is not None:
            raise ValueError("softmax routing does not accept an expert-id map")
        topk_weights, topk_ids = _moe_softmax_topk(
            router_logits,
            top_k,
            topk_indices_dtype=topk_indices_dtype,
            renormalize=renormalize,
            routed_scaling_factor=scaling_factor,
            override=override,
            solution=solution,
        )
        return topk_weights.to(topk_weights_dtype), topk_ids

    if score_function == "sigmoid":
        if selection_method != "topk":
            raise ValueError("sigmoid routing only supports topk selection")
        if correction_bias is None:
            raise ValueError("sigmoid routing requires correction_bias")
        topk_weights, topk_ids = _moe_sigmoid_bias_topk(
            router_logits,
            correction_bias,
            top_k,
            routed_scaling_factor=scaling_factor,
            normalize_topk_weights=renormalize,
            logical_to_physical_map=logical_to_physical_map,
            weights_dtype=topk_weights_dtype,
            override=override,
            solution=solution,
        )
        return topk_weights, topk_ids.to(topk_indices_dtype)

    if score_function != "sqrt_softplus":
        raise ValueError(f"unsupported MoE score function: {score_function!r}")
    if logical_to_physical_map is not None:
        raise ValueError("sqrt_softplus routing does not accept an expert-id map")
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must be a floating-point tensor")
    tokens, experts = router_logits.shape
    if not 0 < top_k <= experts:
        raise ValueError(f"top_k must be in [1, {experts}], got {top_k}")
    valid_bias_shapes = {(experts,), (tokens, experts)}
    if (
        correction_bias is not None
        and tuple(correction_bias.shape) not in valid_bias_shapes
    ):
        raise ValueError(
            f"correction_bias must have shape [{experts}] or [{tokens}, {experts}]"
        )
    if hash_indices_table is not None:
        if (
            hash_indices_table.ndim != 2
            or hash_indices_table.shape[0] == 0
            or hash_indices_table.shape[1] != top_k
        ):
            raise ValueError("hash_indices_table must have shape [vocabulary, top_k]")
        if hash_indices_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("hash_indices_table must have dtype int32 or int64")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must have dtype int32 or int64")
        if input_ids.numel() != tokens:
            raise ValueError(f"input_ids must contain {tokens} token ids")
        if hash_indices_table.device != router_logits.device:
            raise ValueError(
                "hash_indices_table must be on the same device as router_logits"
            )
        if input_ids.device != router_logits.device:
            raise ValueError("input_ids must be on the same device as router_logits")
        _assert_indices_in_range(input_ids, hash_indices_table.shape[0], "input_ids")
        safe_input_ids = input_ids.clamp(0, hash_indices_table.shape[0] - 1)
        selected_experts = hash_indices_table[safe_input_ids.reshape(-1).long()]
        _assert_indices_in_range(selected_experts, experts, "hash_indices_table")
        input_ids = safe_input_ids

    routing_kind = _routing_kind(correction_bias, hash_indices_table)
    traits = {
        "tokens": int(tokens),
        "experts": experts,
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
        "score_function": score_function,
    }
    signature = format_signature(router_logits=dense_tensor_format(router_logits.dtype))
    per_token_bias = correction_bias is not None and correction_bias.ndim == 2
    if per_token_bias and solution not in {None, "torch"}:
        raise ValueError(
            f"per-token correction bias does not support solution {solution!r}"
        )
    if per_token_bias and override not in {
        None,
        "torch",
        "torch_sqrt_softplus_topk",
    }:
        raise ValueError(
            f"per-token correction bias does not support override {override!r}"
        )
    routing_solution = "torch" if per_token_bias else solution
    kernel = select_kernel(
        "moe",
        "topk",
        signature,
        traits=traits,
        override=override,
        solution=routing_solution,
    )
    shape_params = {
        "tokens": int(tokens),
        "experts": int(experts),
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
        "score_function": score_function,
    }
    ShapeCapture.get().record(
        "moe",
        "topk",
        kernel.name,
        router_logits.dtype,
        shape_params,
    )
    with kernel_scope(
        "moe",
        "topk",
        router_logits.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        if tokens == 0:
            return (
                torch.empty(
                    (0, top_k), dtype=topk_weights_dtype, device=router_logits.device
                ),
                torch.empty(
                    (0, top_k), dtype=topk_indices_dtype, device=router_logits.device
                ),
            )
        topk_weights, topk_ids, _ = kernel(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            False,
        )
        if scaling_factor != 1.0:
            topk_weights = topk_weights * scaling_factor
        return topk_weights.to(topk_weights_dtype), topk_ids.to(topk_indices_dtype)


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
    hidden: int | None,
    swiglu_form: str | None,
    activation_clamped: bool,
    expert_id_repeats: bool,
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
    if hidden is not None:
        traits["hidden"] = int(hidden)
    if swiglu_form is not None:
        traits["swiglu_form"] = swiglu_form
    traits["activation_clamped"] = activation_clamped
    if expert_id_repeats:
        traits["expert_id_repeats"] = True
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
    *,
    hidden: int | None,
    swiglu_form: str | None,
    activation_clamped: bool,
    expert_id_repeats: bool,
    fp8_scale_block_shape: tuple[int, int] | None = None,
    internal_activation_dtype: str | None = None,
    with_bias: bool = False,
    process_group: object | None = None,
    deepep_mode: str | None = None,
    deepep_low_latency_max_num_tokens_per_gpu: int | None = None,
    persistent_max_num_tokens_per_gpu: int | None = None,
    fast_math: bool,
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
        hidden: MoE input width (hidden size) for alignment checks; kernels
            declare ``hidden`` / ``hidden_alignment`` traits the same way as
            ``ispp`` / ``ispp_alignment``. Required keyword: pass None only
            to leave the width unconstrained on purpose.
        swiglu_form: For SwiGLU layers, ``"standard"`` (silu(gate) * up with an
            optional clamp) or ``"generalized"`` (a sigmoid multiplier alpha or
            an up-branch offset beta). Kernels whose epilogue implements only
            the standard form declare ``swiglu_form={"standard"}``. Required
            keyword: None for activations other than SwiGLU.
        activation_clamped: True when the activation's output is bounded by
            the checkpoint (a SwiGLU clamp limit). Kernels whose FP8
            activation path relies on a fixed scale declare
            ``activation_clamped={True}`` so unbounded layers never select
            them. Required keyword.
        expert_id_repeats: True when the routing may hand a kernel the same
            expert id more than once for one token (zero-expert placeholders).
            Kernels whose permutation needs distinct ids per token declare
            ``expert_id_repeats={False}``. Required keyword.
        fp8_scale_block_shape: Optional FP8 block-scale shape requirement.
        internal_activation_dtype: Optional internal activation dtype requirement.
            "input" is a special value that uses the whatever dtype the input
            activations have. "mxfp4" requests dynamic MXFP4 activation
            quantization. Defaults to "input" if not set.
        with_bias: Whether the selected kernel must support expert bias tensors.
        process_group: Runtime-created process group for DeepEP or MegaMoE
            communication. Defaults to None for backends that do not use it.
        deepep_mode: Optional DeepEP mode for all-to-all plans: "low_latency"
            (decode-shaped batches only), "normal" (extend-shaped batches only),
            or "auto" to let each ``moe_apply`` pick through its ``low_latency``
            argument. Defaults to "auto".
        deepep_low_latency_max_num_tokens_per_gpu: Per-GPU token capacity the
            DeepEP low-latency buffer is sized for. Required whenever the mode
            can run the low-latency legs; batches above it must use normal mode.
        persistent_max_num_tokens_per_gpu: Optional fixed per-GPU capacity for
            implementations that own persistent communication buffers.
        fast_math: Whether the selected implementation may use fast math.
            Implementations without a fast-math path always compute precisely.
            Required keyword.
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
        hidden=hidden,
        swiglu_form=swiglu_form,
        activation_clamped=activation_clamped,
        expert_id_repeats=expert_id_repeats,
        fp8_scale_block_shape=fp8_scale_block_shape,
        internal_activation_dtype=internal_activation_dtype,
        with_bias=with_bias,
    )
    traits["persistent_workspace"] = persistent_max_num_tokens_per_gpu is not None

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
    if persistent_max_num_tokens_per_gpu is not None and True not in (
        apply_spec.traits.get("persistent_workspace", frozenset())
    ):
        raise ValueError(
            f"MoE kernel {apply_spec.name!r} does not support persistent workspace"
        )

    routing_modes = apply_spec.traits.get("routing_mode", frozenset())
    support_routing = "kernel_routing" in routing_modes
    supports_precomputed_topk = "precomputed_topk" in routing_modes
    supports_deferred_finalize = True in apply_spec.traits.get(
        "supports_deferred_finalize", frozenset({False})
    )
    return {
        "weight_dtype": weight_dtype,
        "activation": activation,
        "apply_kernel_name": apply_spec.name,
        "weight_preprocessor": apply_spec.weight_preprocessor,
        "warmup": getattr(kernel.impl, "_tokenspeed_warmup", None),
        "a2a_backend": a2a_backend,
        "process_group": process_group,
        "deepep_mode": deepep_mode or "auto",
        "deepep_low_latency_max_num_tokens_per_gpu": (
            deepep_low_latency_max_num_tokens_per_gpu
        ),
        "persistent_max_num_tokens_per_gpu": persistent_max_num_tokens_per_gpu,
        "fast_math": fast_math,
        "support_routing": support_routing,
        "supports_precomputed_topk": supports_precomputed_topk,
        "supports_deferred_finalize": supports_deferred_finalize,
        "solution": apply_spec.solution,
        "internal_activation_dtype": internal_activation_dtype,
    }


def moe_process_weights(plan: dict, w: torch.nn.Module):
    """Process loaded MoE weights and prepare persistent communication storage.

    Args:
        plan: Execution plan returned by moe_plan.
        w: Module containing loaded MoE weights. This module is mutated in
            place to prepare solution-specific layouts and scales. DeepEP
            modules must declare hidden_size (the unquantized input width)
            and num_experts (the global expert count).

    Returns:
        The selected weight preprocessor's result, or None without one.
    """
    # Preserve input geometry before a kernel transforms its weight storage.
    # Every DeepEP backend reserves through this one lifecycle entry point.
    deepep_geometry = (
        (w.hidden_size, w.num_experts) if plan.get("a2a_backend") == "deepep" else None
    )
    preprocessor = plan.get("weight_preprocessor")
    result = None
    if preprocessor is not None:
        if not callable(preprocessor):
            raise RuntimeError(f"Weight preprocessor is not callable: {preprocessor!r}")
        result = preprocessor(plan=plan, w=w)
    if deepep_geometry is not None:
        # Keep the optional communication dependency out of non-DeepEP plans.
        from tokenspeed_kernel.ops.communication.deep_ep import prepare_deepep_buffer

        hidden_size, num_experts = deepep_geometry
        prepare_deepep_buffer(
            group=plan["process_group"],
            hidden_size=hidden_size,
            num_experts=num_experts,
            deepep_mode=plan["deepep_mode"],
            max_dispatch_tokens_per_rank=plan[
                "deepep_low_latency_max_num_tokens_per_gpu"
            ],
        )
    return result


def moe_apply(
    plan: dict,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    w: torch.nn.Module,
    # top-k routing inputs
    router_logits: torch.Tensor | None,
    # top-k routing results
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    # token length
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
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
        x: Hidden states with shape [tokens, hidden_size], or a
            (packed_nvfp4, block_scales) pair for a kernel supporting prequantized
            input. Packed data is uint8 [tokens, hidden_size // 2]; scales are
            linear uint8/float8 [tokens, hidden_size // 16].
        w: Module containing processed MoE weights.
        router_logits: Router logits with shape [tokens, num_experts], or None
            for a precomputed-TopK kernel that consumes only IDs and weights.
        topk_weights: Optional precomputed expert weights with shape
            [tokens, top_k]. Required when plan support_routing is false.
        topk_ids: Optional precomputed expert ids with shape [tokens, top_k].
            Required when plan support_routing is false.
        num_tokens_global: Optional global token count for distributed MoE.
        max_num_tokens_per_gpu: Optional per-GPU token capacity hint.
        do_finalize: Whether the kernel must produce the finalized output.
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
        shared_input: Optional activated shared-expert input with shape
            [tokens, shared_size]. Must be provided together with
            ``shared_weight`` and ``shared_out`` to request a joint routed/shared
            projection from kernels that support it.
        shared_weight: Optional shared-expert down-projection weight with shape
            [output_size, shared_size]. Must be provided together with
            ``shared_input`` and ``shared_out``.
        shared_out: Optional destination for the shared-expert down projection
            with shape [tokens, output_size]. Must be provided together with
            ``shared_input`` and ``shared_weight``.

    Solutions may use precomputed top-k tensors or route from logits directly.
    """
    data = x[0] if isinstance(x, tuple) else x
    kernel = select_kernel(
        "moe",
        "apply",
        format_signature(x=dense_tensor_format(data.dtype)),
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
        enable_pdl=pdl_enabled(),
        **a2a_kwargs,
        **shared_kwargs,
    )
