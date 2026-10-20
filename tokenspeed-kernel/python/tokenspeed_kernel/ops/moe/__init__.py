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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import (
    dense_format_signature,
    dense_tensor_format,
    format_signature,
)

__all__ = [
    "bf16_moe_routes",
    "COMBINE_ORDERS",
    "ExpertDispatch",
    "dispatch_topk_ids",
    "native_latent_moe_available",
    "latent_moe_decode_pipeline_available",
    "latent_moe_expert_shared",
    "latent_moe_input_projections",
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
    "moe_warmup",
    "moe_topk",
    "packed_projection_weight_view",
    "moe_sigmoid_bias_topk_routing",
    "FUSED_ROUTING_MAX_TOKENS",
    "PreparedNvfp4Routing",
    "prepare_nvfp4_routing",
    "routed_input_scale_rmsnorm_sum",
    "routed_input_scale_rmsnorm_sum_gathered",
    "prepare_sm90_fp8_moe",
]


def prepare_sm90_fp8_moe(
    *,
    mode: Literal["default", "expected_m", "optimized"],
    max_num_stages_fc1: int = -1,
    max_num_stages_fc2: int = -1,
) -> Callable:
    """Prepare an explicit FlashInfer block-FP8 MoE implementation at init.

    Args:
        mode: ``default`` binds the installed FlashInfer implementation.
            ``expected_m`` binds the SM90 per-call load-hint backport.
            ``optimized`` adds tile-prefix and direct tile-to-expert lookup.
        max_num_stages_fc1: Optional FC1 grouped GEMM pipeline stage limit. ``-1`` keeps
            upstream selection; a positive integer of at least four caps the
            supported FC1 candidates. Requires a private mode.
        max_num_stages_fc2: Independent FC2 pipeline stage limit, with the
            same semantics and default as ``max_num_stages_fc1``.

    Returns:
        A callable accepting FlashInfer's ``cutlass_fused_moe`` arguments and
        returning its output list. Private modes additionally accept a positive
        uint32 ``expected_m`` or None. Callers compute that hint from their own
        routing layout. Initialize before warmup and CUDA graph capture;
        private modes require FlashInfer 0.7.2, Hopper, and a working JIT compiler.
    """
    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import make_cutlass_fused_moe

    return make_cutlass_fused_moe(
        mode=mode,
        max_num_stages_fc1=max_num_stages_fc1,
        max_num_stages_fc2=max_num_stages_fc2,
    )


# ===-----------------------------------------------------------------------===#
# Expert placement dispatch
# ===-----------------------------------------------------------------------===#


# Expert placement dispatch: logical top-k ids onto physical replicas.
#
# With redundant experts a logical expert has several physical replicas, each
# on some rank. Routing then emits physical ids: every rank runs the same
# routing over the same tokens, so the replica must be a pure function of the
# token row and route rank -- ``replicas[logical, (row + rank) % count]`` --
# for exactly one rank to own each (token, expert) pair. ``dispatch_topk_ids``
# is the production implementation of that choice, in tensor ops (two gathers
# and a modulo over ``[tokens, top_k]``); a fused Triton kernel is a
# follow-up, see ``README.md``. Load counting is the runtime's
# (``record_expert_load``), applied to the mapped ids after routing.


@dataclass(frozen=True)
class ExpertDispatch:
    """One MoE layer's logical-to-physical routing tables.

    Attributes:
        replicas: ``[num_logical, X]`` int32 physical ids of every logical
            expert's replicas, ``-1`` padded; contiguous.
        num_replicas: ``[num_logical]`` int32 valid entries per row, all >= 1.
    """

    replicas: torch.Tensor
    num_replicas: torch.Tensor

    def __post_init__(self) -> None:
        if self.replicas.ndim != 2 or self.replicas.dtype != torch.int32:
            raise ValueError("replicas must be an int32 [num_logical, X] table")
        if (
            self.num_replicas.shape != (self.replicas.shape[0],)
            or self.num_replicas.dtype != torch.int32
        ):
            raise ValueError("num_replicas must be int32 [num_logical]")
        if not (self.replicas.is_contiguous() and self.num_replicas.is_contiguous()):
            raise ValueError("dispatch tables must be contiguous")

    @property
    def max_replicas(self) -> int:
        return self.replicas.shape[1]


def dispatch_topk_ids(topk_ids: torch.Tensor, dispatch: ExpertDispatch) -> torch.Tensor:
    """Map logical top-k ids onto physical replicas (the serving path).

    Replica ``(row + route rank) % count`` of each logical expert, so every
    rank running the same routing over the same rows picks the same replica
    and the replicas of a hot expert share its routes evenly. Tensor ops,
    graph-capturable; the ids must all be real experts in ``[0, num_logical)``
    (the runtime masks zero experts and padding around this call).

    Args:
        topk_ids: ``[tokens, top_k]`` logical ids, every entry in
            ``[0, num_logical)``.
        dispatch: The layer's replica tables.

    Returns:
        ``[tokens, top_k]`` physical ids in ``topk_ids``' dtype.
    """
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)[:, None]
    ranks = torch.arange(topk_ids.shape[1], device=topk_ids.device)[None, :]
    logical = topk_ids.long()
    count = dispatch.num_replicas.long()[logical]
    physical = dispatch.replicas.long()[logical, (rows + ranks) % count]
    return physical.to(topk_ids.dtype)


# How a token's routed-expert contributions are combined across the MoE
# TP-EP group (``moe_plan(combine_order=...)``):
#
# ``"rank"``
#     The apply kernel returns this rank's partial (its experts, or its K-split
#     of the down projection); the caller sums the partials rank by rank with
#     its all-reduce / reduce-scatter and adds anything else (an identity
#     zero-expert residual) around that reduction. Every registered kernel
#     serves this form; declaring the trait is optional.
# ``"slot"``
#     The apply kernel exchanges the per-route outputs across the EP group
#     itself (``plan["process_group"]``) and folds each token's top-k slots in
#     fp32 in slot order, zero-expert residual included, returning the
#     complete routed output: the caller must not reduce it again and must
#     hand the kernel the raw top-k (zero-expert ids intact, weights kept).
#     Only kernels declaring ``combine_order={"slot", ...}`` are eligible.
COMBINE_ORDERS = ("rank", "slot")

# ===-----------------------------------------------------------------------===#
# Latent-MoE input projections
# ===-----------------------------------------------------------------------===#


# Column tiles of a packed projection GEMM must not straddle a projection
# boundary, so every projection width has to be a multiple of the tile width.
REGION_ALIGNMENT = 128


def packed_projection_weight_view(
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
) -> torch.Tensor | None:
    """Return the single tensor backing three consecutive projection weights.

    Returns ``None`` unless the three weights are contiguous rows of one
    allocation in router/routed/shared order, which is what lets one GEMM read
    them as a single operand.
    """
    parts = (router_weight, routed_weight, shared_gate_up_weight)
    if not all(part.ndim == 2 and part.is_contiguous() for part in parts):
        return None
    hidden_size = router_weight.shape[1]
    if any(part.shape[1] != hidden_size for part in parts):
        return None
    # Adjacent addresses alone are not enough: separate allocations can land
    # back to back, and reading across them would run off the end of the first.
    storage = router_weight.untyped_storage()
    if any(part.untyped_storage().data_ptr() != storage.data_ptr() for part in parts):
        return None
    row_bytes = hidden_size * router_weight.element_size()
    address = router_weight.data_ptr()
    for part in parts:
        if part.data_ptr() != address:
            return None
        address += part.shape[0] * row_bytes
    if address > storage.data_ptr() + storage.nbytes():
        return None
    total_rows = sum(part.shape[0] for part in parts)
    return router_weight.as_strided((total_rows, hidden_size), (hidden_size, 1))


def _weights_packed(weights: tuple[torch.Tensor, ...]) -> bool:
    """Return whether one GEMM can read the three weights as a single operand.

    Beyond sharing an allocation, every projection width must be a multiple of
    the column-tile width so that no tile straddles two projections and the
    epilogue stays uniform across a program.
    """
    if packed_projection_weight_view(*weights) is None:
        return False
    return all(weight.shape[0] % REGION_ALIGNMENT == 0 for weight in weights)


def latent_moe_input_projections(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    *,
    gate_clamp: float,
    up_clamp: float | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the three independent consumers of a latent-MoE input.

    The operation computes FP32 router logits, a materialized routed latent,
    and a materialized gated shared-expert input. A backend may fuse the three
    independent projections; unsupported shapes execute the same composition
    with PyTorch.

    Args:
        hidden_states: Input shaped ``[tokens, hidden_size]``.
        router_weight: Router weight shaped ``[experts, hidden_size]``.
        routed_weight: Latent projection shaped ``[latent_size, hidden_size]``.
        shared_gate_up_weight: Stacked shared gate/up projection shaped
            ``[2 * shared_size, hidden_size]``.
        gate_clamp: Positive tanh clamp for the shared gate branch.
        up_clamp: Optional positive tanh clamp for the shared up branch.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        ``(router_logits, routed_input, shared_input)``.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    tokens, hidden_size = hidden_states.shape
    weights = (router_weight, routed_weight, shared_gate_up_weight)
    if any(
        weight.ndim != 2
        or weight.shape[1] != hidden_size
        or weight.dtype != hidden_states.dtype
        or weight.device != hidden_states.device
        for weight in weights
    ):
        raise ValueError(
            "projection weights must match the input width, dtype, and device"
        )
    if shared_gate_up_weight.shape[0] % 2:
        raise ValueError("shared gate/up output width must be even")
    if gate_clamp <= 0.0 or (up_clamp is not None and up_clamp <= 0.0):
        raise ValueError("activation clamp values must be positive")

    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        router_weight=dense_tensor_format(router_weight.dtype),
        routed_weight=dense_tensor_format(routed_weight.dtype),
        shared_gate_up_weight=dense_tensor_format(shared_gate_up_weight.dtype),
    )
    traits = {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "num_experts": router_weight.shape[0],
        "latent_size": routed_weight.shape[0],
        "shared_size": shared_gate_up_weight.shape[0] // 2,
        "inputs_contiguous": all(
            tensor.is_contiguous() for tensor in (hidden_states, *weights)
        ),
        "weights_packed": _weights_packed(weights),
        "hidden_size_multiple_64": hidden_size % 64 == 0,
    }
    try:
        kernel = select_kernel(
            "moe",
            "latent_input",
            signature,
            traits=traits,
            override=override,
            solution=solution,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None

    if kernel is not None:
        ShapeCapture.get().record(
            "moe",
            "latent_input",
            kernel.name,
            hidden_states.dtype,
            traits,
        )
        with kernel_scope(
            "moe",
            "latent_input",
            hidden_states.dtype,
            kernel_name=kernel.name,
            **traits,
        ):
            return kernel(
                hidden_states=hidden_states,
                router_weight=router_weight,
                routed_weight=routed_weight,
                shared_gate_up_weight=shared_gate_up_weight,
                gate_clamp=gate_clamp,
                up_clamp=up_clamp,
            )

    router_logits = torch.nn.functional.linear(
        hidden_states.float(), router_weight.float()
    )
    routed_input = torch.nn.functional.linear(hidden_states, routed_weight)
    shared_raw = torch.nn.functional.linear(hidden_states, shared_gate_up_weight)
    gate, up = shared_raw.chunk(2, dim=-1)
    gate_fp32 = gate.float()
    up_fp32 = up.float()
    gate_fp32 = gate_clamp * torch.tanh(gate_fp32 / gate_clamp)
    gate_fp32 *= torch.sigmoid(gate.float())
    if up_clamp is not None:
        up_fp32 = up_clamp * torch.tanh(up_fp32 / up_clamp)
    shared_input = (gate_fp32 * up_fp32).to(hidden_states.dtype)
    return router_logits, routed_input, shared_input


# ===-----------------------------------------------------------------------===#
# Latent-MoE joint routed/shared decode
# ===-----------------------------------------------------------------------===#


class _UnsetLinearClamp:
    pass


_UNSET_LINEAR_CLAMP = _UnsetLinearClamp()


def _selection(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_weight: torch.Tensor,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
):
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        w13_weight=dense_tensor_format(w13_weight.dtype),
        w13_scale=dense_tensor_format(w13_scale.dtype),
        w2_weight=dense_tensor_format(w2_weight.dtype),
        w2_scale=dense_tensor_format(w2_scale.dtype),
        topk_weights=dense_tensor_format(topk_weights.dtype),
        topk_ids=dense_tensor_format(topk_ids.dtype),
        shared_input=dense_tensor_format(shared_input.dtype),
        shared_weight=dense_tensor_format(shared_weight.dtype),
        routed_out=dense_tensor_format(routed_out.dtype),
        shared_out=dense_tensor_format(shared_out.dtype),
    )
    traits = {
        "tokens": hidden_states.shape[0],
        "latent_size": hidden_states.shape[1],
        "topk": topk_ids.shape[1],
        "num_local_experts": w13_weight.shape[0],
        "intermediate_size": (
            w13_weight.shape[1] * 8
            if len(w13_weight.shape) == 6
            else w2_weight.shape[-1] * 2
        ),
        "shared_size": shared_input.shape[1],
        "output_size": shared_weight.shape[0],
        "linear_weights": len(w13_weight.shape) != 6,
        "inputs_contiguous": all(
            tensor.is_contiguous()
            for tensor in (
                hidden_states,
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                topk_weights,
                topk_ids,
                shared_input,
                shared_weight,
                routed_out,
                shared_out,
            )
        ),
    }
    return signature, traits


def latent_moe_decode_pipeline_available(
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_plan: Mapping[str, Any],
    *,
    topk: int,
    linear_clamp: float | None | _UnsetLinearClamp = _UNSET_LINEAR_CLAMP,
) -> bool:
    """Return whether joint routed/shared small-batch decode is available.

    This construction-time probe owns backend, tensor-format, expert-plan, and
    exact-shape constraints needed by the orchestration-level optimization. An
    A8 input-policy plan can share its N16 bank with this A16 decoder. Explicit
    FP8 plans cannot use the decoder. An A8 plan must provide ``linear_clamp``
    explicitly; callers of the pre-existing A16 probe may omit it.
    """
    projection_weights = (
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )
    expert_tensors = (w13_weight, w13_scale, w2_weight, w2_scale)
    apply_kernel_name = expert_plan.get("apply_kernel_name")
    is_a8 = apply_kernel_name == "gluon_mxfp4_a8w4_situ_ep_precomputed_moe_apply"
    # Construction precedes preprocessing. The selected plan promises N16
    # cells; execution validates their physical axes, not transient attributes.
    uses_n16_weights = (
        is_a8
        and linear_clamp is not _UNSET_LINEAR_CLAMP
        and linear_clamp is not None
        and expert_plan.get("internal_activation_dtype") != "fp8"
    )
    uses_linear_weights = (
        apply_kernel_name == "gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply"
        or (
            is_a8
            and linear_clamp is None
            and expert_plan.get("internal_activation_dtype") != "fp8"
        )
    ) and all(len(t.shape) == 3 for t in expert_tensors)
    if not (
        (uses_linear_weights or uses_n16_weights)
        and router_weight.dtype
        == routed_weight.dtype
        == shared_gate_up_weight.dtype
        == shared_down_weight.dtype
        == torch.bfloat16
        and shared_gate_up_weight.shape[0] % 2 == 0
        and all(
            tensor.is_cuda and tensor.is_contiguous()
            for tensor in (*projection_weights, *expert_tensors)
        )
    ):
        return False

    traits = {
        "tokens": 1,
        "latent_size": routed_weight.shape[0],
        "topk": topk,
        "num_local_experts": w13_weight.shape[0],
        "intermediate_size": (
            w13_weight.shape[1] * 8
            if len(w13_weight.shape) == 6
            else w2_weight.shape[-1] * 2
        ),
        "shared_size": shared_gate_up_weight.shape[0] // 2,
        "output_size": shared_down_weight.shape[0],
        "linear_weights": uses_linear_weights,
        "inputs_contiguous": True,
    }
    for tokens in range(1, 5):
        traits["tokens"] = tokens
        for topk_weight_dtype in (torch.bfloat16, torch.float32):
            signature = format_signature(
                hidden_states=dense_tensor_format(routed_weight.dtype),
                w13_weight=dense_tensor_format(w13_weight.dtype),
                w13_scale=dense_tensor_format(w13_scale.dtype),
                w2_weight=dense_tensor_format(w2_weight.dtype),
                w2_scale=dense_tensor_format(w2_scale.dtype),
                topk_weights=dense_tensor_format(topk_weight_dtype),
                topk_ids=dense_tensor_format(torch.int32),
                shared_input=dense_tensor_format(shared_gate_up_weight.dtype),
                shared_weight=dense_tensor_format(shared_down_weight.dtype),
                routed_out=dense_tensor_format(routed_weight.dtype),
                shared_out=dense_tensor_format(shared_down_weight.dtype),
            )
            try:
                select_kernel("moe", "latent_expert_shared", signature, traits=traits)
            except NoKernelFoundError:
                continue
            break
        else:
            return False
    return True


def latent_moe_expert_shared(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_weight: torch.Tensor,
    *,
    activation_clamp: float,
    linear_clamp: float | None,
    expert_start: int,
    w13_interleaved: bool,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run routed experts and an independent shared down-projection jointly.

    This operation is optional and has no opaque fallback: callers use
    :func:`latent_moe_decode_pipeline_available` before choosing the joint
    orchestration. Ordinary model execution remains the fallback.
    """
    signature, traits = _selection(
        hidden_states,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        topk_weights,
        topk_ids,
        shared_input,
        shared_weight,
        routed_out,
        shared_out,
    )
    kernel = select_kernel(
        "moe",
        "latent_expert_shared",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    result = kernel(
        hidden_states=hidden_states,
        w13_weight=w13_weight,
        w13_scale=w13_scale,
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_input=shared_input,
        shared_weight=shared_weight,
        activation_clamp=activation_clamp,
        linear_clamp=linear_clamp,
        expert_start=expert_start,
        w13_interleaved=w13_interleaved,
        routed_out=routed_out,
        shared_out=shared_out,
    )
    if not isinstance(result, tuple):
        raise RuntimeError("joint latent MoE kernel did not return both outputs")
    return result


def native_latent_moe_available() -> bool:
    """Return whether the active backend provides native latent-space MoE."""

    return current_platform().is_amd


# ===-----------------------------------------------------------------------===#
# Routing: softmax top-k
# ===-----------------------------------------------------------------------===#


_TRITON_MAX_EXPERTS = 1024
_TRITON_MAX_TOPK = 32
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _triton_eligible(router_logits: torch.Tensor, topk: int) -> bool:
    return (
        router_logits.is_cuda
        and router_logits.dtype in _SUPPORTED_DTYPES
        and router_logits.shape[0] > 0
        and router_logits.stride(1) == 1
        and router_logits.shape[1] <= _TRITON_MAX_EXPERTS
        and topk <= _TRITON_MAX_TOPK
    )


def _moe_softmax_topk(
    router_logits: torch.Tensor,
    topk: int,
    *,
    topk_indices_dtype: torch.dtype,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using softmax routing in one fused launch when supported.

    Args:
        router_logits: Router logits shaped ``[tokens, experts]``. Inputs
            matching a registered Triton implementation use it; others use
            the PyTorch fallback.
        topk: Number of experts selected per token.
        topk_indices_dtype: Integer dtype for returned expert ids.
        renormalize: Normalize the selected weights to sum to one. When false,
            return probabilities from the softmax over all experts.
        routed_scaling_factor: Scale applied to every selected route weight.
        override: Optional exact registered kernel name.
        solution: Optional implementation solution, such as ``"triton"`` or
            ``"torch"``.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, topk]``. Weights are
        FP32 and ids use ``topk_indices_dtype``.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    tokens, experts = router_logits.shape
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must have a floating-point dtype")
    if not 0 < topk <= experts:
        raise ValueError(f"topk must be in [1, {experts}], got {topk}")
    if topk_indices_dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_indices_dtype must be torch.int32 or torch.int64")
    if tokens == 0:
        shape = (0, topk)
        return (
            torch.empty(shape, device=router_logits.device, dtype=torch.float32),
            torch.empty(
                shape,
                device=router_logits.device,
                dtype=topk_indices_dtype,
            ),
        )

    if (
        override is None
        and solution is None
        and not _triton_eligible(router_logits, topk)
    ):
        solution = "torch"
    enable_pdl = pdl_enabled()
    if override is None and solution == "torch":
        return torch_softmax_topk(
            router_logits=router_logits,
            topk=topk,
            topk_indices_dtype=topk_indices_dtype,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            enable_pdl=enable_pdl,
        )

    kernel = select_kernel(
        "moe",
        "softmax_topk",
        format_signature(
            router_logits=dense_tensor_format(router_logits.dtype),
        ),
        traits={"tokens": int(tokens), "experts": int(experts), "topk": int(topk)},
        override=override,
        solution=solution,
    )
    return kernel(
        router_logits=router_logits,
        topk=topk,
        topk_indices_dtype=topk_indices_dtype,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        enable_pdl=enable_pdl,
    )


# ===-----------------------------------------------------------------------===#
# Routing: sigmoid-bias top-k
# ===-----------------------------------------------------------------------===#


def _gluon_eligible(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
) -> bool:
    tokens, experts = router_logits.shape
    return (
        router_logits.is_cuda
        and router_logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and tokens > 0
        and 0 < topk <= 16
        and topk <= experts <= 1024
        and router_logits.stride(1) == 1
        and correction_bias.is_contiguous()
    )


def _moe_sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
    logical_to_physical_map: torch.Tensor | None = None,
    weights_dtype: torch.dtype = torch.float32,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using ``sigmoid(logits) + correction_bias``.

    Args:
        router_logits: Router logits with shape ``[tokens, experts]``.
        correction_bias: Expert-selection bias with shape ``[experts]``.
        topk: Number of experts selected per token.
        routed_scaling_factor: Scale applied after optional normalization.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        logical_to_physical_map: Optional ``[experts]`` integer map applying
            static expert-location dispatch to the selected ids (fused into
            the specialized decode kernel; a gather elsewhere).
        weights_dtype: Output dtype for the route weights (selection always
            runs in FP32).
        override: Optional exact registered kernel name.
        solution: Optional implementation solution.

    Returns:
        ``(topk_weights, topk_ids)`` with shapes ``[tokens, topk]``. Weights
        are ``weights_dtype`` and ids are INT32 (physical ids when a dispatch
        map is given).
    """
    if router_logits.dim() != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    tokens, experts = router_logits.shape
    if correction_bias.shape != (experts,):
        raise ValueError("correction_bias must have shape [experts]")
    if correction_bias.device != router_logits.device:
        raise ValueError("correction_bias and router_logits must share a device")
    if logical_to_physical_map is not None and (
        logical_to_physical_map.shape != (experts,)
        or logical_to_physical_map.dtype not in (torch.int32, torch.int64)
        or logical_to_physical_map.device != router_logits.device
        or not logical_to_physical_map.is_contiguous()
    ):
        raise ValueError(
            "logical_to_physical_map must be contiguous colocated INT32/INT64 "
            "[experts]"
        )
    if not 0 < topk <= experts:
        raise ValueError(f"topk must be in [1, {experts}], got {topk}")
    if tokens == 0:
        shape = (0, topk)
        return (
            torch.empty(shape, device=router_logits.device, dtype=weights_dtype),
            torch.empty(shape, device=router_logits.device, dtype=torch.int32),
        )
    if (
        override is None
        and solution is None
        and not _gluon_eligible(router_logits, correction_bias, topk)
    ):
        solution = "torch"

    signature = format_signature(router_logits=dense_tensor_format(router_logits.dtype))
    traits = {"tokens": tokens, "experts": experts, "topk": topk}
    if logical_to_physical_map is not None and (
        (override is not None and override.endswith("_mapped"))
        or (
            logical_to_physical_map.dtype == torch.int32
            and solution is None
            and override is None
        )
    ):
        try:
            mapped_kernel = select_kernel(
                "moe",
                "sigmoid_bias_topk_mapped",
                signature,
                traits=traits,
                override=override,
            )
        except NoKernelFoundError:
            mapped_kernel = None
        if mapped_kernel is not None:
            topk_weights, topk_ids = mapped_kernel(
                router_logits=router_logits,
                correction_bias=correction_bias,
                topk=topk,
                routed_scaling_factor=routed_scaling_factor,
                normalize_topk_weights=normalize_topk_weights,
                logical_to_physical_map=logical_to_physical_map,
                weights_dtype=weights_dtype,
            )
            return topk_weights, topk_ids

    kernel = select_kernel(
        "moe",
        "sigmoid_bias_topk",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    topk_weights, topk_ids = kernel(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        weights_dtype=weights_dtype,
    )
    if logical_to_physical_map is not None:
        topk_ids = logical_to_physical_map[topk_ids.long()].to(torch.int32)
    return topk_weights, topk_ids


# ===-----------------------------------------------------------------------===#
# Routing and expert APIs
# ===-----------------------------------------------------------------------===#

from tokenspeed_kernel.ops.moe._triton.sigmoid_topk_routing import (  # noqa: E402
    MAX_TOKENS as FUSED_ROUTING_MAX_TOKENS,
)
from tokenspeed_kernel.ops.moe.flashinfer import (  # noqa: E402
    PreparedNvfp4Routing,
    prepare_nvfp4_routing,
)
from tokenspeed_kernel.ops.moe.routed_norm import (  # noqa: E402
    routed_input_scale_rmsnorm_sum,
    routed_input_scale_rmsnorm_sum_gathered,
)


def moe_sigmoid_bias_topk_routing(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    local_expert_offset: int,
    num_local_experts: int,
    routing: object,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts as ``moe_sigmoid_bias_topk`` does and fill ``routing`` in the same launch.

    Serves at most ``FUSED_ROUTING_MAX_TOKENS`` tokens, the batches where one
    program beats the separate top-k and routing kernels. ``routing`` is a
    flashinfer trtllm-gen routing metadata slot the expert GEMMs will read.
    """
    kernel = select_kernel(
        "moe",
        "sigmoid_bias_topk_routing",
        format_signature(router_logits=dense_tensor_format(router_logits.dtype)),
    )
    return kernel(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        routing=routing,
        enable_pdl=enable_pdl,
    )


logger = logging.getLogger(__name__)

if current_platform().is_npu:
    import tokenspeed_kernel.ops.moe.ascend  # noqa: F401


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
            Sqrt-softplus scoring, normalization and scaling use FP32; CUDA
            and Triton can write BF16 directly after scaling.
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
            scaling_factor,
            (
                topk_weights_dtype
                if topk_weights_dtype in (torch.float32, torch.bfloat16)
                else torch.float32
            ),
        )
        return topk_weights.to(topk_weights_dtype), topk_ids.to(topk_indices_dtype)


def _normalize_weight_dtype(weight_dtype: str) -> str:
    if weight_dtype in {"bf16", "fp16", "float16", "bfloat16", "unquantized"}:
        return "unquant"
    return weight_dtype


def _uses_all_to_all_ep(a2a_backend: str | None) -> bool:
    return a2a_backend not in {None, "none"}


def _validate_a2a_backend(a2a_backend: str | None) -> None:
    if a2a_backend in {None, "none", "deepep", "gluon_petit"}:
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


def _validate_combine_order(
    combine_order: str, kernel_name: str, kernel_traits: dict[str, frozenset[Any]]
) -> None:
    """Reject a selected apply kernel that cannot combine in the planned order.

    A kernel that declares ``combine_order`` must list the planned value. A
    kernel that declares nothing serves ``"rank"`` (the caller reduces) but
    cannot serve ``"slot"``, which only an in-kernel EP exchange provides.
    """
    declared = kernel_traits.get("combine_order")
    if declared is None:
        if combine_order == "slot":
            raise ValueError(
                f"MoE kernel {kernel_name!r} does not declare the combine_order "
                "trait: it returns a per-rank partial for the caller's "
                "rank-ordered reduction and cannot fold the top-k slots across "
                "the EP group itself (combine_order='slot'); select a kernel "
                "declaring the trait through solution="
            )
        return
    if combine_order not in declared:
        supported = ", ".join(sorted(declared))
        raise ValueError(
            f"MoE kernel {kernel_name!r} does not support "
            f"combine_order={combine_order!r}; supported orders: {supported}"
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
    num_experts: int | None,
    top_k: int | None,
    swiglu_form: str | None,
    activation_clamped: bool,
    expert_id_repeats: bool,
    fp8_scale_block_shape: tuple[int, int] | None,
    internal_activation_dtype: str | None,
    with_bias: bool,
    combine_order: str,
) -> dict[str, Any]:
    if internal_activation_dtype is None:
        internal_activation_dtype = "input"

    traits: dict[str, Any] = {"weight_dtype": weight_dtype}
    # Unknown counts must not admit a kernel restricted to particular counts.
    traits["num_experts"] = num_experts
    traits["top_k"] = top_k
    # Seller-declared: excludes only kernels declaring another order; the
    # post-selection check turns a silent kernel down for "slot".
    traits["combine_order"] = combine_order
    if activation is not None:
        traits["activation"] = activation
    if requires_deferred_finalize:
        traits["supports_deferred_finalize"] = True
    if routing_mode is not None:
        traits["routing_mode"] = routing_mode

    all_to_all_ep = _uses_all_to_all_ep(a2a_backend)
    traits["supports_all_to_all_ep"] = all_to_all_ep
    traits["a2a_backend"] = a2a_backend
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


def _select_moe_apply(
    input_dtype: torch.dtype, traits: dict[str, Any], solution: str | None
):
    kernel = select_kernel(
        "moe",
        "apply",
        format_signature(x=dense_tensor_format(input_dtype)),
        traits=traits,
        solution=solution,
        ignore_layout=True,
    )
    apply_spec = KernelRegistry.get().get_by_name(kernel.name)
    if apply_spec is None:
        raise RuntimeError(f"Kernel spec not found for selected kernel {kernel.name}")
    return kernel, apply_spec


@functools.cache
def _log_fp32_correction_bias_fallback(kernel_name: str) -> None:
    logger.info(
        f"MoE kernel {kernel_name} cannot add the correction bias in FP32 while "
        "routing; layers that need it plan precomputed top-k"
    )


def bf16_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
) -> torch.Tensor:
    """Return unweighted BF16 SwiGLU outputs for every selected local expert.

    Args:
        x: Contiguous BF16 hidden states ``[tokens, hidden]``.
        w13: Local concatenated gate/up weights ``[experts, 2I, hidden]``.
        w2: Local down weights ``[experts, hidden, I]``.
        topk_ids: Global selected expert ids ``[tokens, topk]``.
        expert_start: First global expert represented by ``w13`` and ``w2``.

    Returns:
        Route outputs ``[tokens, topk, hidden]``; routes to other ranks' experts are zero.
    """
    kernel = select_kernel("moe", "routes", dense_format_signature(("x",), (x.dtype,)))
    return kernel(x, w13, w2, topk_ids, expert_start)


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
    num_experts: int | None = None,
    top_k: int | None = None,
    fp8_scale_block_shape: tuple[int, int] | None = None,
    internal_activation_dtype: str | None = None,
    with_bias: bool = False,
    process_group: object | None = None,
    deepep_mode: str | None = None,
    deepep_low_latency_max_num_tokens_per_gpu: int | None = None,
    persistent_max_num_tokens_per_gpu: int | None = None,
    fast_math: bool,
    combine_order: str,
    solution: str | None = None,
    fp32_correction_bias: bool = False,
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
        num_experts: Global expert count. None means unknown and excludes
            kernels restricted to a particular expert count.
        top_k: Number of selected experts per token. None means unknown and
            excludes kernels restricted to a particular route count.
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
        combine_order: How a token's routed contributions meet across the MoE
            TP-EP group, one of ``COMBINE_ORDERS``. ``"rank"``: the kernel
            returns this rank's partial and the caller reduces. ``"slot"``: the
            kernel folds the top-k slots in fp32 slot order across the EP group
            through ``process_group`` and returns the complete output, which
            the caller must not reduce; it needs a kernel declaring the
            ``combine_order`` trait with ``"slot"`` and, for ``ep_size > 1``,
            the EP process group. Required keyword.
        solution: Optional kernel solution to force through normal selection.
            None leaves the concrete kernel choice to the registry.
        fp32_correction_bias: Whether routing must add the DeepSeekV3
            correction bias in FP32. A kernel that routes from logits is kept
            only if its ``_tokenspeed_fp32_correction_bias`` hook prepares
            such a router and returns True; otherwise the plan uses
            precomputed top-k, whose routers add the bias in FP32, and a
            "kernel_routing" requirement raises ValueError.

    The selected apply kernel owns plan metadata. A plan with support_routing
    false requires precomputed top-k ids and weights when calling moe_apply.
    Weight preprocessing is selected from the ordered candidates advertised by
    the selected apply kernel, then pinned by callable in the returned plan so load
    time does not rerun selection or conflict resolution. The plan carries
    ``combine_order`` and ``process_group`` for the apply kernel to read.
    """
    weight_dtype = _normalize_weight_dtype(weight_dtype)
    _validate_a2a_backend(a2a_backend)
    _validate_routing_mode(routing_mode)
    _validate_deepep_mode(a2a_backend, deepep_mode)
    if combine_order not in COMBINE_ORDERS:
        raise ValueError(
            f"combine_order must be one of {list(COMBINE_ORDERS)}, got "
            f"{combine_order!r}"
        )
    if combine_order == "slot" and ep_size is not None and ep_size > 1:
        if process_group is None:
            raise ValueError(
                "combine_order='slot' with ep_size > 1 exchanges per-route "
                "outputs inside the apply kernel and needs the EP process group"
            )
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
        num_experts=num_experts,
        top_k=top_k,
        swiglu_form=swiglu_form,
        activation_clamped=activation_clamped,
        expert_id_repeats=expert_id_repeats,
        fp8_scale_block_shape=fp8_scale_block_shape,
        internal_activation_dtype=internal_activation_dtype,
        with_bias=with_bias,
        combine_order=combine_order,
    )
    traits["persistent_workspace"] = persistent_max_num_tokens_per_gpu is not None

    kernel, apply_spec = _select_moe_apply(input_dtype, traits, solution)
    # A kernel that routes from logits keeps an FP32 correction bias only if
    # its hook says so; TokenSpeed's top-k routers always add it in FP32.
    fp32_bias_hook = getattr(kernel.impl, "_tokenspeed_fp32_correction_bias", None)
    if (
        fp32_correction_bias
        and routing_mode != "precomputed_topk"
        and "kernel_routing" in apply_spec.traits.get("routing_mode", frozenset())
        and not (fp32_bias_hook is not None and fp32_bias_hook())
    ):
        if routing_mode == "kernel_routing":
            raise ValueError(
                f"MoE kernel {apply_spec.name!r} cannot add the correction bias "
                "in FP32 while routing"
            )
        _log_fp32_correction_bias_fallback(apply_spec.name)
        routing_mode = traits["routing_mode"] = "precomputed_topk"
        kernel, apply_spec = _select_moe_apply(input_dtype, traits, solution)
    _validate_selected_deepep_mode(
        a2a_backend,
        deepep_mode,
        apply_spec.name,
        apply_spec.traits,
    )
    _validate_combine_order(combine_order, apply_spec.name, apply_spec.traits)
    if persistent_max_num_tokens_per_gpu is not None and True not in (
        apply_spec.traits.get("persistent_workspace", frozenset())
    ):
        raise ValueError(
            f"MoE kernel {apply_spec.name!r} does not support persistent workspace"
        )

    routing_modes = apply_spec.traits.get("routing_mode", frozenset())
    # With an FP32 correction bias, precomputed top-k also means routing
    # outside a kernel that could route itself.
    support_routing = "kernel_routing" in routing_modes and not (
        fp32_correction_bias and routing_mode == "precomputed_topk"
    )
    supports_precomputed_topk = "precomputed_topk" in routing_modes
    supports_deferred_finalize = True in apply_spec.traits.get(
        "supports_deferred_finalize", frozenset({False})
    )
    topk_weights_dtypes = apply_spec.traits.get(
        "topk_weights_dtype", frozenset({torch.float32})
    )
    if len(topk_weights_dtypes) != 1:
        raise ValueError(
            f"MoE kernel {apply_spec.name!r} must declare exactly one "
            "topk_weights_dtype"
        )
    topk_weights_dtype = next(iter(topk_weights_dtypes))
    if supports_deferred_finalize and topk_weights_dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError(
            f"MoE kernel {apply_spec.name!r} uses unsupported deferred-finalize "
            f"weight dtype {topk_weights_dtype}"
        )
    supports_prequantized_input = True in apply_spec.traits.get(
        "supports_prequantized_input", frozenset({False})
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
        "combine_order": combine_order,
        "support_routing": support_routing,
        "supports_precomputed_topk": supports_precomputed_topk,
        "supports_deferred_finalize": supports_deferred_finalize,
        "topk_weights_dtype": topk_weights_dtype,
        "supports_prequantized_input": supports_prequantized_input,
        # All-to-all EP (DeepEP, Petit): each rank runs its own tokens'
        # routes, so routing may pick per-rank replicas. Otherwise every
        # rank routes every token and must agree on one replica per route.
        "supports_all_to_all_ep": traits["supports_all_to_all_ep"],
        "solution": apply_spec.solution,
        "internal_activation_dtype": internal_activation_dtype,
        "fp32_correction_bias": fp32_correction_bias,
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


def moe_warmup(plan: dict, w: torch.nn.Module, *, max_num_tokens: int) -> None:
    """Warm ``plan``'s selected backend using ``w``'s processed expert weights.

    ``max_num_tokens`` bounds global routing rows. Plans without a warmup
    callback need no work; this function returns nothing.
    """
    warmup = plan.get("warmup")
    if warmup is not None:
        warmup({**plan, "warmup_max_num_tokens": max_num_tokens}, w)


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
    x_quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
    routing_prepared: object | None = None,
):
    """Apply a planned MoE kernel.

    Args:
        plan: Execution plan returned by moe_plan.
        x: Hidden states with shape [tokens, hidden_size], or a
            (packed_fp4, block_scales) pair for a kernel supporting prequantized
            input. Packed data is uint8 [tokens, hidden_size // 2]; scales are
            linear uint8/float8 [tokens, hidden_size // 16] for NVFP4 or
            CDNA4-swizzled E8M0 bytes for the gfx950 MXFP4 kernel.
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
        x_quantized: Optional ``x`` already quantized in the kernel's
            activation format, so the kernel skips its own quantize. Only
            forwarded when given; kernels that cannot consume it reject it.
        routing_prepared: Routing metadata the top-k kernel filled ahead of
            this call, so the kernel skips its routing launch. Only forwarded
            when given; kernels that cannot consume it reject it.

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
    # Kernels that cannot consume a pre-quantized activation never see the argument.
    quantized_kwargs = {"x_quantized": x_quantized} if x_quantized is not None else {}
    if routing_prepared is not None:
        quantized_kwargs["routing_prepared"] = routing_prepared
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
        **quantized_kwargs,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.moe.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.deep_ep  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.marlin  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.native  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.triton  # noqa: E402,F401
from tokenspeed_kernel.ops.moe.native import torch_softmax_topk  # noqa: E402

# isort: on
