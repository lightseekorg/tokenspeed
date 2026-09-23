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
from tokenspeed_kernel.signature import (
    FormatSignature,
    dense_tensor_format,
    format_signature,
)

__all__ = [
    "prepare_latent_expert_shared",
    "prepare_latent_input",
    "prepare_moe_apply",
    "prepare_sigmoid_bias_topk",
]


_IMPLEMENTED_MODEL_PROFILES = frozenset({"glm53_flash_tp4", "kimi_k3_tp8"})
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
    "mxfp4": torch.uint8,
}
_IMPLEMENTED_ACTIVATIONS = frozenset({"silu", "situ", "swiglu"})
_IMPLEMENTED_ROUTING_MODES = frozenset({"precomputed_topk"})
_IMPLEMENTED_ROUTE_SCOPES = frozenset({"global", "local"})
_IMPLEMENTED_ROUTE_DISTRIBUTIONS = frozenset({"router"})
_IMPLEMENTED_TOKEN_COUNT_SCOPES = frozenset({"global", "local"})
_IMPLEMENTED_INTERNAL_ACTIVATION_DTYPES = frozenset({"input"})
_IMPLEMENTED_FP8_BLOCK_SHAPES = frozenset({(128, 128)})
_MXFP4_GROUP_SIZE = 32


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


def _selected_spec(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    signature: FormatSignature,
    traits: dict[str, object],
) -> KernelSpec:
    try:
        selected = select_kernel(
            request.family,
            request.mode,
            signature,
            platform=platform,
            traits=traits,
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


def _mxfp4_weight_shapes(
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
) -> dict[str, tuple[int, ...]]:
    if hidden_size % 2 or intermediate_size_per_partition % 2:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE MXFP4 packed dimensions must be even",
        )
    if (
        hidden_size % _MXFP4_GROUP_SIZE
        or intermediate_size_per_partition % _MXFP4_GROUP_SIZE
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE MXFP4 dimensions must be divisible by group size {_MXFP4_GROUP_SIZE}",
        )
    return {
        "w13": (
            num_local_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
        ),
        "w13_scale": (
            num_local_experts,
            2 * intermediate_size_per_partition,
            hidden_size // _MXFP4_GROUP_SIZE,
        ),
        "w2": (
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
        ),
        "w2_scale": (
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition // _MXFP4_GROUP_SIZE,
        ),
    }


def _make_mxfp4_weights(
    *,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    activation: str,
    swiglu_limit: float | None,
    situ_beta: float | None,
    situ_linear_beta: float | None,
    ep_rank: int,
    ep_size: int,
) -> SimpleNamespace:
    shapes = _mxfp4_weight_shapes(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
    )
    swiglu_arg = (
        SimpleNamespace(alpha=1.0, limit=swiglu_limit)
        if activation == "swiglu"
        else None
    )
    return SimpleNamespace(
        w13_weight=torch.zeros(shapes["w13"], dtype=torch.uint8, device="cuda"),
        w13_weight_scale=torch.full(
            shapes["w13_scale"], 127, dtype=torch.uint8, device="cuda"
        ),
        w2_weight=torch.zeros(shapes["w2"], dtype=torch.uint8, device="cuda"),
        w2_weight_scale=torch.full(
            shapes["w2_scale"], 127, dtype=torch.uint8, device="cuda"
        ),
        activation=activation,
        activation_situ_beta=situ_beta,
        activation_situ_linear_beta=situ_linear_beta,
        swiglu_arg=swiglu_arg,
        swiglu_beta=None,
        w13_input_layout="concatenated",
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        quant_config=None,
    )


def _prepare_routed_weights(
    *,
    weight_dtype_name: str,
    weight_dtype: torch.dtype,
    input_dtype: torch.dtype,
    activation: str,
    routing_mode: str,
    ep_size: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    swiglu_limit: float | None,
    situ_beta: float | None,
    situ_linear_beta: float | None,
    block_shape: tuple[int, int] | None,
    internal_activation_dtype: str,
) -> tuple[dict, SimpleNamespace]:
    from tokenspeed_kernel.ops import moe as moe_ops

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
    if weight_dtype_name == "fp8":
        assert block_shape is not None
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
    else:
        weights = _make_mxfp4_weights(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            activation=activation,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
    moe_ops.moe_process_weights(plan, weights)
    return plan, weights


def _latent_expert_shared_weights(
    weights: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return (
            weights.w13_weight,
            weights.w13_weight_scale,
            weights.w2_weight,
            weights.w2_weight_scale,
        )
    except AttributeError as error:
        raise BenchmarkCaseError(
            BenchmarkStatus.NOT_APPLICABLE,
            "Selected MoE weight preprocessing is not compatible with the joint "
            "latent-expert/shared operation",
        ) from error


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
    signature = format_signature(router_logits=dense_tensor_format(router_logits_dtype))
    spec = _selected_spec(
        request,
        platform,
        signature,
        {"tokens": tokens, "experts": experts, "topk": topk},
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
    block_shape = (
        _parse_block_shape(parameters["fp8_scale_block_shape"])
        if weight_dtype_name == "fp8"
        else None
    )
    activation = _implemented_value(
        "activation",
        parameters["activation"],
        _IMPLEMENTED_ACTIVATIONS,
    )
    swiglu_limit = None
    if activation == "swiglu":
        swiglu_limit = float(parameters["swiglu_limit"])
    situ_beta = None
    situ_linear_beta = None
    if activation == "situ":
        if weight_dtype_name != "mxfp4":
            raise BenchmarkCaseError(
                BenchmarkStatus.INVALID_CASE,
                "MoE SiTU apply benchmarks require MXFP4 weights",
            )
        situ_beta = float(parameters["activation_situ_beta"])
        linear_beta = parameters["activation_situ_linear_beta"]
        situ_linear_beta = None if linear_beta is None else float(linear_beta)
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
    token_partition_count = tp_size * ep_size
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
    plan, weights = _prepare_routed_weights(
        weight_dtype_name=weight_dtype_name,
        weight_dtype=weight_dtype,
        input_dtype=input_dtype,
        activation=activation,
        routing_mode=routing_mode,
        ep_size=ep_size,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        ep_rank=ep_rank,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        block_shape=block_shape,
        internal_activation_dtype=internal_activation_dtype,
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

    def invoke() -> object:
        return moe_ops.moe_apply(
            plan,
            hidden_states,
            weights,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_tokens_global=(
                tokens * ep_size if token_count_scope == "local" else tokens
            ),
            max_num_tokens_per_gpu=(
                tokens
                if token_count_scope == "local"
                else math.ceil(tokens / token_partition_count)
            ),
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
            "activation_situ_beta": situ_beta,
            "activation_situ_linear_beta": situ_linear_beta,
            "routing_mode": routing_mode,
            "route_scope": route_scope,
            "route_distribution": route_distribution,
            "token_count_scope": token_count_scope,
            "routed_scaling_factor": routed_scaling_factor,
            "normalize_topk_weights": normalize_topk_weights,
            "fp8_scale_block_shape": block_shape,
            "mxfp4_group_size": (
                _MXFP4_GROUP_SIZE if weight_dtype_name == "mxfp4" else None
            ),
            "topk_generation": "sigmoid_bias_topk",
        },
        validation=None,
    )


def _packed_projection_weights(
    *,
    hidden_size: int,
    num_experts: int,
    latent_size: int,
    shared_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = num_experts + latent_size + 2 * shared_size
    packed = torch.zeros((rows, hidden_size), dtype=dtype, device="cuda")
    router_weight = packed.narrow(0, 0, num_experts)
    routed_weight = packed.narrow(0, num_experts, latent_size)
    shared_gate_up_weight = packed.narrow(
        0,
        num_experts + latent_size,
        2 * shared_size,
    )
    return router_weight, routed_weight, shared_gate_up_weight


def prepare_latent_input(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one packed latent-MoE input-projection benchmark."""

    _validate_request_options(request)
    parameters = request.parameters
    model_profile = _implemented_value(
        "model_profile",
        parameters["model_profile"],
        _IMPLEMENTED_MODEL_PROFILES,
    )
    tokens = parameters["tokens"]
    hidden_size = parameters["hidden_size"]
    num_experts = parameters["num_experts"]
    latent_size = parameters["latent_size"]
    shared_size = parameters["shared_size"]
    input_dtype = _parse_dtype(
        "input_dtype",
        parameters["input_dtype"],
        _IMPLEMENTED_INPUT_DTYPES,
    )
    situ_beta = float(parameters["activation_situ_beta"])
    linear_beta = parameters["activation_situ_linear_beta"]
    situ_linear_beta = None if linear_beta is None else float(linear_beta)

    generator = _generator(request.seed)
    hidden_states = _randn(
        (tokens, hidden_size),
        generator=generator,
        dtype=input_dtype,
    )
    router_weight, routed_weight, shared_gate_up_weight = _packed_projection_weights(
        hidden_size=hidden_size,
        num_experts=num_experts,
        latent_size=latent_size,
        shared_size=shared_size,
        dtype=input_dtype,
    )

    load_builtin_kernels()
    from tokenspeed_kernel.ops.moe.latent_input import (
        REGION_ALIGNMENT,
        latent_moe_input_projections,
        packed_projection_weight_view,
    )

    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        router_weight=dense_tensor_format(router_weight.dtype),
        routed_weight=dense_tensor_format(routed_weight.dtype),
        shared_gate_up_weight=dense_tensor_format(shared_gate_up_weight.dtype),
    )
    weights = (router_weight, routed_weight, shared_gate_up_weight)
    weights_packed = packed_projection_weight_view(*weights) is not None and all(
        weight.shape[0] % REGION_ALIGNMENT == 0 for weight in weights
    )
    traits = {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "latent_size": latent_size,
        "shared_size": shared_size,
        "inputs_contiguous": all(tensor.is_contiguous() for tensor in weights),
        "weights_packed": weights_packed,
        "hidden_size_multiple_64": hidden_size % 64 == 0,
    }
    spec = _selected_spec(request, platform, signature, traits)

    def invoke() -> object:
        return latent_moe_input_projections(
            hidden_states,
            router_weight,
            routed_weight,
            shared_gate_up_weight,
            gate_clamp=situ_beta,
            up_clamp=situ_linear_beta,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "latent_size": latent_size,
            "shared_size": shared_size,
            "input_dtype": str(input_dtype).removeprefix("torch."),
            "activation_situ_beta": situ_beta,
            "activation_situ_linear_beta": situ_linear_beta,
            "weights_packed": weights_packed,
        },
        validation=None,
    )


def prepare_latent_expert_shared(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one joint latent routed-expert and shared-projection benchmark."""

    _validate_request_options(request)
    parameters = request.parameters
    model_profile = _implemented_value(
        "model_profile",
        parameters["model_profile"],
        _IMPLEMENTED_MODEL_PROFILES,
    )
    tokens = parameters["tokens"]
    latent_size = parameters["latent_size"]
    intermediate_size = parameters["intermediate_size"]
    num_experts = parameters["num_experts"]
    num_local_experts = parameters["num_local_experts"]
    topk = parameters["topk"]
    ep_size = parameters["ep_size"]
    ep_rank = parameters["ep_rank"]
    shared_size = parameters["shared_size"]
    output_size = parameters["output_size"]
    if ep_rank >= ep_size or num_local_experts * ep_size != num_experts:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Latent MoE EP placement must evenly cover the global experts",
        )
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
    situ_beta = float(parameters["activation_situ_beta"])
    linear_beta = parameters["activation_situ_linear_beta"]
    situ_linear_beta = None if linear_beta is None else float(linear_beta)
    routed_scaling_factor = float(parameters["routed_scaling_factor"])
    normalize_topk_weights = parameters["normalize_topk_weights"]

    generator = _generator(request.seed)
    hidden_states = _randn(
        (tokens, latent_size),
        generator=generator,
        dtype=input_dtype,
    )
    _router_logits, topk_weights, topk_ids = _routing_tensors(
        tokens=tokens,
        experts=num_experts,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        router_logits_dtype=router_logits_dtype,
        weights_dtype=torch.float32,
        generator=generator,
        expert_start=0,
    )
    load_builtin_kernels()
    _plan, weights = _prepare_routed_weights(
        weight_dtype_name="mxfp4",
        weight_dtype=torch.uint8,
        input_dtype=input_dtype,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        hidden_size=latent_size,
        intermediate_size_per_partition=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        ep_rank=ep_rank,
        swiglu_limit=None,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        block_shape=None,
        internal_activation_dtype="input",
    )
    w13_weight, w13_scale, w2_weight, w2_scale = _latent_expert_shared_weights(weights)
    shared_input = _randn(
        (tokens, shared_size),
        generator=generator,
        dtype=input_dtype,
    )
    shared_weight = torch.zeros(
        (output_size, shared_size), dtype=input_dtype, device="cuda"
    )
    routed_out = torch.empty_like(hidden_states)
    shared_out = torch.empty((tokens, output_size), dtype=input_dtype, device="cuda")
    expert_start = ep_rank * num_local_experts

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
    weights_are_linear = w2_weight.ndim != 6
    processed_intermediate_size = (
        w2_weight.shape[-1] * 2 if weights_are_linear else w2_weight.shape[2] * 128
    )
    traits = {
        "tokens": tokens,
        "latent_size": latent_size,
        "topk": topk,
        "num_local_experts": w13_weight.shape[0],
        "intermediate_size": processed_intermediate_size,
        "shared_size": shared_size,
        "output_size": output_size,
        "linear_weights": weights_are_linear,
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
    spec = _selected_spec(request, platform, signature, traits)

    from tokenspeed_kernel.ops.moe.latent_decode import latent_moe_expert_shared

    def reset() -> None:
        routed_out.zero_()
        shared_out.zero_()

    def invoke() -> object:
        return latent_moe_expert_shared(
            hidden_states,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            topk_weights,
            topk_ids,
            shared_input,
            shared_weight,
            activation_clamp=situ_beta,
            linear_clamp=situ_linear_beta,
            expert_start=expert_start,
            w13_interleaved=weights.w13_input_layout == "interleaved",
            routed_out=routed_out,
            shared_out=shared_out,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke, reset=reset),
        parameters={
            "model_profile": model_profile,
            "tokens": tokens,
            "latent_size": latent_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "topk": topk,
            "ep_size": ep_size,
            "ep_rank": ep_rank,
            "expert_start": expert_start,
            "shared_size": shared_size,
            "output_size": output_size,
            "input_dtype": str(input_dtype).removeprefix("torch."),
            "router_logits_dtype": str(router_logits_dtype).removeprefix("torch."),
            "weight_dtype": "mxfp4",
            "mxfp4_group_size": _MXFP4_GROUP_SIZE,
            "activation": "situ",
            "activation_situ_beta": situ_beta,
            "activation_situ_linear_beta": situ_linear_beta,
            "route_scope": "global",
            "route_distribution": "router",
            "routed_scaling_factor": routed_scaling_factor,
            "normalize_topk_weights": normalize_topk_weights,
            "topk_generation": "sigmoid_bias_topk",
        },
        validation=None,
    )
