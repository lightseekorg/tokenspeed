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
from dataclasses import dataclass
from typing import Any

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.moe.cuda  # noqa: F401
import tokenspeed_kernel.ops.moe.deep_gemm  # noqa: F401
import tokenspeed_kernel.ops.moe.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.moe.gluon  # noqa: F401
import tokenspeed_kernel.ops.moe.marlin  # noqa: F401
import tokenspeed_kernel.ops.moe.triton  # noqa: F401
import torch
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "dsv4_mega_moe_apply",
    "dsv4_mega_moe_plan",
    "dsv4_mega_moe_process_weights",
    "dsv4_mega_moe_warmup",
    "dsv4_select_experts",
    "native_latent_moe_available",
    "latent_moe_decode_pipeline_available",
    "latent_moe_expert_shared",
    "latent_moe_input_projections",
    "moe_apply",
    "moe_plan",
    "pack_topk_router_logits",
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
from tokenspeed_kernel.ops.moe.pack_topk import pack_topk_router_logits  # noqa: E402
from tokenspeed_kernel.ops.moe.sigmoid_topk import moe_sigmoid_bias_topk  # noqa: E402
from tokenspeed_kernel.ops.moe.softmax_topk import moe_softmax_topk  # noqa: E402


@dataclass(frozen=True)
class _MegaMoEPlan:
    kernel: SelectedKernel
    weight_preprocessor: Callable
    warmup: Callable | None
    input_dtype: torch.dtype
    num_experts: int
    num_local_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    max_num_tokens: int
    process_group: object | None
    activation_clamp: float | None


@dataclass(frozen=True)
class _MegaMoEState:
    plan: _MegaMoEPlan
    backend_state: object


def dsv4_mega_moe_plan(
    *,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    max_num_tokens: int,
    process_group: object | None = None,
    activation_clamp: float | None = None,
    input_dtype: torch.dtype = torch.bfloat16,
    solution: str | None = None,
) -> object:
    """Create an opaque DeepSeek V4 MegaMoE execution plan.

    Kernel selection applies registry capability requirements. The currently
    registered implementation requires NVIDIA SM100 and optional DeepGEMM
    MegaMoE symbols, so planning fails cleanly when either is unavailable.

    Args:
        num_experts: Total number of routed experts across the EP group.
        num_local_experts: Number of checkpoint experts held by this rank.
        top_k: Number of experts selected per token.
        hidden_size: Model hidden dimension.
        intermediate_size: Per-expert intermediate dimension.
        max_num_tokens: Per-rank token capacity of the symmetric input buffer.
        process_group: Optional expert-parallel process group. When omitted, the
            backend uses the default initialized distributed process group.
        activation_clamp: Optional SwiGLU activation clamp compiled into the
            MegaMoE kernel. The same value is used for execution and warmup.
        input_dtype: Hidden-state dtype consumed by the implementation.
        solution: Optional implementation family selected through the registry.

    Returns:
        An opaque plan accepted by the related process, apply, and warmup APIs.
    """
    dimensions = {
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "max_num_tokens": max_num_tokens,
    }
    invalid = [name for name, value in dimensions.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"MegaMoE dimensions must be positive: {', '.join(invalid)}")
    if num_local_experts > num_experts:
        raise ValueError("num_local_experts cannot exceed num_experts")
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "DeepSeek V4 MegaMoE hidden and intermediate sizes must be "
            "multiples of 128"
        )

    kernel = select_kernel(
        "moe",
        "dsv4_mega_moe",
        format_signature(hidden_states=dense_tensor_format(input_dtype)),
        traits={
            "weight_dtype": "mxfp4",
            "scale_format": "ue8m0",
            "scale_block_size": 32,
            "supports_ep": True,
        },
        solution=solution,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None or spec.weight_preprocessor is None:
        raise RuntimeError(f"MegaMoE kernel {kernel.name!r} has no weight preprocessor")
    return _MegaMoEPlan(
        kernel=kernel,
        weight_preprocessor=spec.weight_preprocessor,
        warmup=getattr(kernel.impl, "_tokenspeed_warmup", None),
        input_dtype=input_dtype,
        num_experts=int(num_experts),
        num_local_experts=int(num_local_experts),
        top_k=int(top_k),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        max_num_tokens=int(max_num_tokens),
        process_group=process_group,
        activation_clamp=activation_clamp,
    )


def _require_mega_moe_plan(plan: object) -> _MegaMoEPlan:
    if not isinstance(plan, _MegaMoEPlan):
        raise TypeError("plan must be returned by dsv4_mega_moe_plan")
    return plan


def _require_mega_moe_state(plan: _MegaMoEPlan, state: object) -> _MegaMoEState:
    if not isinstance(state, _MegaMoEState):
        raise TypeError("state must be returned by dsv4_mega_moe_process_weights")
    if state.plan is not plan:
        raise ValueError("MegaMoE state was created by a different plan")
    return state


def dsv4_mega_moe_process_weights(
    plan: object,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> object:
    """Process canonical checkpoint tensors into opaque MegaMoE state.

    Callers may retain ordinary checkpoint tensors while loading and replace
    them with the returned state only after all shards are populated. Scale
    conversion and implementation-specific weight layouts are backend-owned.

    Args:
        plan: Opaque plan returned by :func:`dsv4_mega_moe_plan`.
        w13_weight: Packed canonical gate/up weight tensor.
        w13_weight_scale: Canonical gate/up UE8M0 scale tensor.
        w2_weight: Packed canonical down-projection weight tensor.
        w2_weight_scale: Canonical down-projection UE8M0 scale tensor.

    Returns:
        Opaque processed state accepted by apply and warmup.
    """
    typed_plan = _require_mega_moe_plan(plan)
    backend_state = typed_plan.weight_preprocessor(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        num_local_experts=typed_plan.num_local_experts,
        hidden_size=typed_plan.hidden_size,
        intermediate_size=typed_plan.intermediate_size,
    )
    return _MegaMoEState(plan=typed_plan, backend_state=backend_state)


def dsv4_mega_moe_apply(
    plan: object,
    state: object,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    fast_math: bool = True,
) -> torch.Tensor:
    """Execute DeepSeek V4 MegaMoE using opaque processed state.

    Args:
        plan: Opaque plan returned by :func:`dsv4_mega_moe_plan`.
        state: Opaque state returned by
            :func:`dsv4_mega_moe_process_weights` for the same plan.
        hidden_states: Input activations shaped ``[tokens, hidden_size]``.
        topk_weights: Routing weights shaped ``[tokens, top_k]``.
        topk_ids: Global expert ids shaped ``[tokens, top_k]``.
        fast_math: Whether the backend may use its fast-math execution path.

    Returns:
        BF16 routed-expert output shaped like ``hidden_states``.
    """
    typed_plan = _require_mega_moe_plan(plan)
    typed_state = _require_mega_moe_state(typed_plan, state)
    if hidden_states.ndim != 2 or hidden_states.shape[1] != typed_plan.hidden_size:
        raise ValueError(
            "MegaMoE hidden_states must have shape "
            f"[tokens, {typed_plan.hidden_size}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != typed_plan.input_dtype:
        raise ValueError(
            f"MegaMoE expected hidden dtype {typed_plan.input_dtype}, "
            f"got {hidden_states.dtype}"
        )
    expected_routing_shape = (hidden_states.shape[0], typed_plan.top_k)
    if tuple(topk_weights.shape) != expected_routing_shape:
        raise ValueError(
            f"topk_weights must have shape {expected_routing_shape}, "
            f"got {tuple(topk_weights.shape)}"
        )
    if tuple(topk_ids.shape) != expected_routing_shape:
        raise ValueError(
            f"topk_ids must have shape {expected_routing_shape}, "
            f"got {tuple(topk_ids.shape)}"
        )
    if hidden_states.shape[0] > typed_plan.max_num_tokens:
        raise ValueError(
            f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, but the "
            f"symmetric buffer was sized for {typed_plan.max_num_tokens}"
        )

    return typed_plan.kernel(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        state=typed_state.backend_state,
        process_group=typed_plan.process_group,
        num_experts=typed_plan.num_experts,
        top_k=typed_plan.top_k,
        hidden_size=typed_plan.hidden_size,
        intermediate_size=typed_plan.intermediate_size,
        max_num_tokens=typed_plan.max_num_tokens,
        activation_clamp=typed_plan.activation_clamp,
        fast_math=fast_math,
    )


def dsv4_mega_moe_warmup(plan: object, state: object) -> None:
    """Warm all serving token shapes for an opaque MegaMoE plan and state.

    The backend performs the EP barrier, reuses its symmetric buffer, and uses
    the plan's token capacity and activation clamp so compiled variants match
    serving.

    Args:
        plan: Opaque plan returned by :func:`dsv4_mega_moe_plan`.
        state: Opaque state returned by
            :func:`dsv4_mega_moe_process_weights` for the same plan.

    Returns:
        None.
    """
    typed_plan = _require_mega_moe_plan(plan)
    typed_state = _require_mega_moe_state(typed_plan, state)
    if typed_plan.warmup is None:
        return
    typed_plan.warmup(
        state=typed_state.backend_state,
        process_group=typed_plan.process_group,
        num_experts=typed_plan.num_experts,
        top_k=typed_plan.top_k,
        hidden_size=typed_plan.hidden_size,
        intermediate_size=typed_plan.intermediate_size,
        max_num_tokens=typed_plan.max_num_tokens,
        activation_clamp=typed_plan.activation_clamp,
    )


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


def dsv4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    need_scores: bool = True,
    override: str | None = None,
    solution: str | None = None,
    *,
    hash_table_values_validated: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select DeepSeek V4 experts from sqrt-softplus router scores.

    Correction bias affects selection only; returned weights are gathered from
    the unbiased scores. Hash routing uses the checkpoint table for expert ids.

    Args:
        router_logits: Router logits shaped [tokens, experts].
        top_k: Number of experts selected for each token.
        renormalize: Normalize selected weights to sum to one when true.
        correction_bias: Optional selection-only bias shaped [experts].
        hash_indices_table: Optional token-id to expert-id table.
        input_ids: Token ids used with hash_indices_table.
        need_scores: Whether callers consume the full score tensor. Specialized
            kernels avoid materializing it when false.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
        hash_table_values_validated: Whether the caller has already validated
            every expert id in the immutable hash table. This only skips the
            repeated table-value check; runtime token ids are always checked.

    Returns:
        FP32 weights, INT32 expert ids, and a tensor shaped [tokens, experts].
        The first two tensors have shape [tokens, top_k]. When need_scores is
        false, a specialized kernel may return router_logits as the ignored
        third value instead of materializing scores.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must be a floating-point tensor")
    tokens, experts = router_logits.shape
    if not 0 < top_k <= experts:
        raise ValueError(f"top_k must be in [1, {experts}], got {top_k}")
    if not isinstance(hash_table_values_validated, bool):
        raise TypeError("hash_table_values_validated must be a bool")
    if hash_table_values_validated and hash_indices_table is None:
        raise ValueError("hash_table_values_validated requires hash_indices_table")
    if correction_bias is not None and correction_bias.shape != (experts,):
        raise ValueError(f"correction_bias must have shape [{experts}]")
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
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
        if not hash_table_values_validated:
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
    }
    signature = format_signature(router_logits=dense_tensor_format(router_logits.dtype))
    kernel = select_kernel(
        "moe",
        "dsv4_select_experts",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    shape_params = {
        "tokens": int(tokens),
        "experts": int(experts),
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
        "need_scores": bool(need_scores),
    }
    ShapeCapture.get().record(
        "moe",
        "dsv4_select_experts",
        kernel.name,
        router_logits.dtype,
        shape_params,
    )
    with kernel_scope(
        "moe",
        "dsv4_select_experts",
        router_logits.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            need_scores,
        )


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
    supports_precomputed_topk = "precomputed_topk" in routing_modes
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
        "supports_precomputed_topk": supports_precomputed_topk,
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
        enable_pdl=pdl_enabled(),
        **a2a_kwargs,
        **shared_kwargs,
    )
