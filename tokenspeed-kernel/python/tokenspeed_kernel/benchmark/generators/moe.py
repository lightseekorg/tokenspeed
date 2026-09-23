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

import math
from collections.abc import Collection
from types import SimpleNamespace

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.ops import moe as moe_ops
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["prepare_moe_apply", "prepare_sigmoid_bias_topk"]


_IMPLEMENTED_MODEL_PROFILES = frozenset({"glm53_flash_tp4"})
_IMPLEMENTED_INPUT_DTYPES = {
    "bfloat16": torch.bfloat16,
}
_IMPLEMENTED_ROUTER_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_IMPLEMENTED_ROUTING_WEIGHT_DTYPES = {
    "float32": torch.float32,
}
_IMPLEMENTED_WEIGHT_FORMATS = {
    "fp8": torch.float8_e4m3fn,
}
_IMPLEMENTED_ACTIVATIONS = frozenset({"silu", "swiglu"})
_IMPLEMENTED_ROUTING_MODES = frozenset({"precomputed_topk"})
_IMPLEMENTED_ROUTE_SCOPES = frozenset({"global", "local"})
_IMPLEMENTED_ROUTE_DISTRIBUTIONS = frozenset({"router"})
_IMPLEMENTED_TOKEN_COUNT_SCOPES = frozenset({"local"})
_IMPLEMENTED_INTERNAL_ACTIVATION_DTYPES = frozenset({"input"})
_IMPLEMENTED_FP8_BLOCK_SHAPES = frozenset({(128, 128)})


def _implemented_value(
    name: str,
    value: object,
    implemented: Collection[object],
):
    if value not in implemented:
        accepted = ", ".join(str(item) for item in sorted(implemented))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Implemented MoE {name} values: {accepted}",
        )
    return value


def _parse_dtype(
    name: str,
    value: object,
    implemented: dict[str, torch.dtype],
) -> torch.dtype:
    dtype_name = _implemented_value(name, value, implemented)
    return implemented[dtype_name]


def _parse_weight_dtype(value: object) -> tuple[str, torch.dtype]:
    name = _implemented_value("weight_dtype", value, _IMPLEMENTED_WEIGHT_FORMATS)
    return name, _IMPLEMENTED_WEIGHT_FORMATS[name]


def _parse_block_shape(value: object) -> tuple[int, int]:
    block_shape = tuple(value)
    return _implemented_value(
        "fp8_scale_block_shape",
        block_shape,
        _IMPLEMENTED_FP8_BLOCK_SHAPES,
    )


def _validate_request_options(request: BenchmarkRequest) -> None:
    if request.parameters.get("validation") is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE benchmark correctness validation is not implemented yet",
        )
    if request.solution is not None or request.registration is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE benchmarks exercise normal kernel selection",
        )


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)


def _select_sigmoid_bias_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    *,
    router_logits_dtype: torch.dtype,
    tokens: int,
    experts: int,
    topk: int,
) -> KernelSpec:
    signature = format_signature(router_logits=dense_tensor_format(router_logits_dtype))
    try:
        selected = select_kernel(
            request.family,
            request.mode,
            signature,
            platform=platform,
            traits={"tokens": tokens, "experts": experts, "topk": topk},
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(
            BenchmarkStatus.NOT_APPLICABLE,
            str(error),
        ) from error

    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec


def _correction_bias(
    experts: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.linspace(
        -0.05,
        0.05,
        experts,
        dtype=torch.float32,
        device=device,
    )


def _routing_tensors(
    *,
    tokens: int,
    experts: int,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    router_logits_dtype: torch.dtype,
    weights_dtype: torch.dtype,
    generator: torch.Generator,
    expert_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = _randn(
        (tokens, experts),
        generator=generator,
        dtype=router_logits_dtype,
    )
    topk_weights, topk_ids = moe_ops.moe_topk(
        router_logits,
        topk,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        correction_bias=_correction_bias(experts, device=router_logits.device),
        topk_weights_dtype=weights_dtype,
    )
    if expert_start:
        topk_ids = topk_ids + expert_start
    return router_logits, topk_weights.contiguous(), topk_ids.contiguous()


def _intermediate_per_partition(intermediate_size: int, tp_size: int) -> int:
    if intermediate_size % tp_size:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE intermediate_size must be divisible by tp_size",
        )
    return intermediate_size // tp_size


def _num_local_experts(num_experts: int, ep_size: int) -> int:
    if num_experts % ep_size:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE num_experts must be divisible by ep_size",
        )
    return num_experts // ep_size


def _fp8_weight_shapes(
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    block_shape: tuple[int, int],
) -> dict[str, tuple[int, ...]]:
    block_n, block_k = block_shape
    return {
        "w13": (
            num_local_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        ),
        "w13_scale": (
            num_local_experts,
            math.ceil((2 * intermediate_size_per_partition) / block_n),
            math.ceil(hidden_size / block_k),
        ),
        "w2": (
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
        ),
        "w2_scale": (
            num_local_experts,
            math.ceil(hidden_size / block_n),
            math.ceil(intermediate_size_per_partition / block_k),
        ),
    }


def _zero_fp8(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cuda").zero_()


def _ones(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.float32, device="cuda")


def _make_fp8_weights(
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    block_shape: tuple[int, int],
    weight_dtype: torch.dtype,
    activation: str,
    swiglu_limit: float | None,
    ep_rank: int,
    ep_size: int,
) -> SimpleNamespace:
    shapes = _fp8_weight_shapes(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        block_shape=block_shape,
    )
    swiglu_arg = (
        SimpleNamespace(alpha=1.0, limit=swiglu_limit)
        if activation == "swiglu"
        else None
    )
    return SimpleNamespace(
        w13_weight=_zero_fp8(shapes["w13"], weight_dtype),
        w13_weight_scale_inv=_ones(shapes["w13_scale"]),
        w2_weight=_zero_fp8(shapes["w2"], weight_dtype),
        w2_weight_scale_inv=_ones(shapes["w2_scale"]),
        w13_weight_bias=None,
        w2_weight_bias=None,
        activation=activation,
        swiglu_arg=swiglu_arg,
        swiglu_beta=None,
        w13_input_layout="concatenated",
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_local_experts=num_local_experts,
    )


def prepare_sigmoid_bias_topk(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one biased sigmoid top-k router benchmark."""

    _validate_request_options(request)
    parameters = request.parameters
    model_profile = _implemented_value(
        "model_profile",
        parameters["model_profile"],
        _IMPLEMENTED_MODEL_PROFILES,
    )
    tokens = parameters["tokens"]
    experts = parameters["num_experts"]
    topk = parameters["topk"]
    router_logits_dtype = _parse_dtype(
        "router_logits_dtype",
        parameters["router_logits_dtype"],
        _IMPLEMENTED_ROUTER_DTYPES,
    )
    weights_dtype = _parse_dtype(
        "weights_dtype",
        parameters["weights_dtype"],
        _IMPLEMENTED_ROUTING_WEIGHT_DTYPES,
    )
    routed_scaling_factor = float(parameters["routed_scaling_factor"])
    normalize_topk_weights = parameters["normalize_topk_weights"]

    load_builtin_kernels()
    spec = _select_sigmoid_bias_topk(
        request,
        platform,
        router_logits_dtype=router_logits_dtype,
        tokens=tokens,
        experts=experts,
        topk=topk,
    )
    generator = _generator(request.seed)
    router_logits = _randn(
        (tokens, experts),
        generator=generator,
        dtype=router_logits_dtype,
    )
    correction_bias = _correction_bias(experts, device=router_logits.device)

    def invoke() -> object:
        return moe_ops.moe_topk(
            router_logits,
            topk,
            score_function="sigmoid",
            selection_method="topk",
            renormalize=normalize_topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias,
            topk_weights_dtype=weights_dtype,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "num_experts": experts,
            "topk": topk,
            "router_logits_dtype": str(router_logits_dtype).removeprefix("torch."),
            "weights_dtype": str(weights_dtype).removeprefix("torch."),
            "routed_scaling_factor": routed_scaling_factor,
            "normalize_topk_weights": normalize_topk_weights,
        },
        validation=None,
    )


def prepare_moe_apply(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one precomputed-routing MoE apply benchmark."""

    _validate_request_options(request)
    parameters = request.parameters
    model_profile = _implemented_value(
        "model_profile",
        parameters["model_profile"],
        _IMPLEMENTED_MODEL_PROFILES,
    )
    tokens = parameters["tokens"]
    hidden_size = parameters["hidden_size"]
    intermediate_size = parameters["intermediate_size"]
    num_experts = parameters["num_experts"]
    topk = parameters["topk"]
    tp_size = parameters["tp_size"]
    ep_size = parameters["ep_size"]
    ep_rank = parameters["ep_rank"]
    input_dtype = _parse_dtype(
        "input_dtype",
        parameters["input_dtype"],
        _IMPLEMENTED_INPUT_DTYPES,
    )
    router_logits_dtype = _parse_dtype(
        "router_logits_dtype",
        parameters["router_logits_dtype"],
        _IMPLEMENTED_ROUTER_DTYPES,
    )
    weight_dtype_name, weight_dtype = _parse_weight_dtype(parameters["weight_dtype"])
    block_shape = _parse_block_shape(parameters["fp8_scale_block_shape"])
    activation = _implemented_value(
        "activation",
        parameters["activation"],
        _IMPLEMENTED_ACTIVATIONS,
    )
    swiglu_limit = None
    if activation == "swiglu":
        swiglu_limit = float(parameters["swiglu_limit"])
    routing_mode = _implemented_value(
        "routing_mode",
        parameters["routing_mode"],
        _IMPLEMENTED_ROUTING_MODES,
    )
    route_scope = _implemented_value(
        "route_scope",
        parameters["route_scope"],
        _IMPLEMENTED_ROUTE_SCOPES,
    )
    route_distribution = _implemented_value(
        "route_distribution",
        parameters["route_distribution"],
        _IMPLEMENTED_ROUTE_DISTRIBUTIONS,
    )
    token_count_scope = _implemented_value(
        "token_count_scope",
        parameters["token_count_scope"],
        _IMPLEMENTED_TOKEN_COUNT_SCOPES,
    )
    routed_scaling_factor = float(parameters["routed_scaling_factor"])
    normalize_topk_weights = parameters["normalize_topk_weights"]
    internal_activation_dtype = _implemented_value(
        "internal_activation_dtype",
        parameters["internal_activation_dtype"],
        _IMPLEMENTED_INTERNAL_ACTIVATION_DTYPES,
    )
    intermediate_size_per_partition = _intermediate_per_partition(
        intermediate_size,
        tp_size,
    )
    expected_local_experts = _num_local_experts(num_experts, ep_size)
    num_local_experts = parameters["num_local_experts"]
    if num_local_experts != expected_local_experts:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE num_local_experts must match num_experts / ep_size",
        )
    if route_scope == "local" and topk > num_local_experts:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE topk exceeds the selected expert pool",
        )

    load_builtin_kernels()
    plan = moe_ops.moe_plan(
        weight_dtype=weight_dtype_name,
        input_dtype=input_dtype,
        activation=activation,
        requires_deferred_finalize=False,
        routing_mode=routing_mode,
        a2a_backend="none",
        ep_size=ep_size,
        ispp=intermediate_size_per_partition,
        hidden=hidden_size,
        swiglu_form="standard" if activation == "swiglu" else None,
        activation_clamped=swiglu_limit is not None,
        expert_id_repeats=False,
        fp8_scale_block_shape=block_shape,
        internal_activation_dtype=internal_activation_dtype,
        with_bias=False,
        fast_math=False,
        solution=None,
    )
    spec = KernelRegistry.get().get_by_name(plan["apply_kernel_name"])
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {plan['apply_kernel_name']!r} is not available",
        )

    generator = _generator(request.seed)
    hidden_states = _randn(
        (tokens, hidden_size),
        generator=generator,
        dtype=input_dtype,
    )
    route_experts = num_local_experts if route_scope == "local" else num_experts
    expert_start = ep_rank * num_local_experts if route_scope == "local" else 0
    router_logits, topk_weights, topk_ids = _routing_tensors(
        tokens=tokens,
        experts=route_experts,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        router_logits_dtype=router_logits_dtype,
        weights_dtype=torch.float32,
        generator=generator,
        expert_start=expert_start,
    )
    weights = _make_fp8_weights(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        block_shape=block_shape,
        weight_dtype=weight_dtype,
        activation=activation,
        swiglu_limit=swiglu_limit,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    moe_ops.moe_process_weights(plan, weights)

    def invoke() -> object:
        return moe_ops.moe_apply(
            plan,
            hidden_states,
            weights,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_tokens_global=tokens * ep_size,
            max_num_tokens_per_gpu=tokens,
            do_finalize=True,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "intermediate_size_per_partition": intermediate_size_per_partition,
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "topk": topk,
            "tp_size": tp_size,
            "ep_size": ep_size,
            "ep_rank": ep_rank,
            "input_dtype": str(input_dtype).removeprefix("torch."),
            "router_logits_dtype": str(router_logits_dtype).removeprefix("torch."),
            "weight_dtype": weight_dtype_name,
            "activation": activation,
            "swiglu_limit": swiglu_limit,
            "routing_mode": routing_mode,
            "route_scope": route_scope,
            "route_distribution": route_distribution,
            "token_count_scope": token_count_scope,
            "routed_scaling_factor": routed_scaling_factor,
            "normalize_topk_weights": normalize_topk_weights,
            "fp8_scale_block_shape": block_shape,
            "topk_generation": "sigmoid_bias_topk",
        },
        validation=None,
    )
