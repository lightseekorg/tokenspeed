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
from types import SimpleNamespace
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["prepare_moe_apply", "prepare_sigmoid_bias_topk"]


_DTYPE_NAMES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
_FP8_DTYPE_NAMES = {
    "fp8": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
_SUPPORTED_FP8_BLOCK = 128


def _positive_int(parameters: dict[str, Any], name: str) -> int:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must be a positive integer",
        )
    return value


def _optional_positive_int(
    parameters: dict[str, Any],
    name: str,
    fallback: int,
) -> int:
    value = parameters.get(name, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must be a positive integer",
        )
    return value


def _nonnegative_int(
    parameters: dict[str, Any],
    name: str,
    fallback: int,
) -> int:
    value = parameters.get(name, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must be a nonnegative integer",
        )
    return value


def _finite_float(parameters: dict[str, Any], name: str, fallback: float) -> float:
    value = parameters.get(name, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must be a finite number",
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must be finite",
        )
    return converted


def _bool_parameter(parameters: dict[str, Any], name: str, fallback: bool) -> bool:
    value = parameters.get(name, fallback)
    if not isinstance(value, bool):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must be a boolean",
        )
    return value


def _parse_dtype(parameters: dict[str, Any], name: str, fallback: str) -> torch.dtype:
    value = parameters.get(name, fallback)
    if isinstance(value, str):
        dtype = _DTYPE_NAMES.get(value.lower())
    else:
        dtype = value if isinstance(value, torch.dtype) else None
    if dtype not in {torch.bfloat16, torch.float32}:
        supported = ", ".join(sorted(_DTYPE_NAMES))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE parameter {name!r} must use one of: {supported}",
        )
    return dtype


def _parse_weight_dtype(parameters: dict[str, Any]) -> tuple[str, torch.dtype]:
    value = parameters.get("weight_dtype", "fp8")
    if not isinstance(value, str):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE weight_dtype must be a string",
        )
    normalized = value.lower()
    dtype = _FP8_DTYPE_NAMES.get(normalized)
    if dtype is None:
        supported = ", ".join(sorted(_FP8_DTYPE_NAMES))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"MoE apply benchmarks currently support weight_dtype names: {supported}",
        )
    return "fp8", dtype


def _parse_block_shape(parameters: dict[str, Any]) -> tuple[int, int]:
    value = parameters.get("fp8_scale_block_shape", [_SUPPORTED_FP8_BLOCK] * 2)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE fp8_scale_block_shape must contain two integer dimensions",
        )
    block_shape = (int(value[0]), int(value[1]))
    if block_shape != (_SUPPORTED_FP8_BLOCK, _SUPPORTED_FP8_BLOCK):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE FP8 apply benchmarks currently support 128x128 scale blocks",
        )
    return block_shape


def _normalize_parameters(
    request: BenchmarkRequest,
    *,
    allowed: set[str],
) -> None:
    unknown = sorted(set(request.parameters) - allowed)
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown MoE parameters: {', '.join(unknown)}",
        )
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
    from tokenspeed_kernel.ops import moe as moe_ops

    router_logits = _randn(
        (tokens, experts),
        generator=generator,
        dtype=router_logits_dtype,
    )
    topk_weights, topk_ids = moe_ops.moe_sigmoid_bias_topk(
        router_logits,
        _correction_bias(experts, device=router_logits.device),
        topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        weights_dtype=weights_dtype,
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

    _normalize_parameters(
        request,
        allowed={
            "tokens",
            "num_experts",
            "topk",
            "router_logits_dtype",
            "weights_dtype",
            "routed_scaling_factor",
            "normalize_topk_weights",
            "validation",
        },
    )
    tokens = _positive_int(request.parameters, "tokens")
    experts = _positive_int(request.parameters, "num_experts")
    topk = _positive_int(request.parameters, "topk")
    if topk > experts:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE topk cannot exceed num_experts",
        )
    router_logits_dtype = _parse_dtype(
        request.parameters,
        "router_logits_dtype",
        "float32",
    )
    weights_dtype = _parse_dtype(request.parameters, "weights_dtype", "float32")
    routed_scaling_factor = _finite_float(
        request.parameters,
        "routed_scaling_factor",
        1.0,
    )
    normalize_topk_weights = _bool_parameter(
        request.parameters,
        "normalize_topk_weights",
        True,
    )

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

    from tokenspeed_kernel.ops import moe as moe_ops

    def invoke() -> object:
        return moe_ops.moe_sigmoid_bias_topk(
            router_logits,
            correction_bias,
            topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
            weights_dtype=weights_dtype,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
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

    _normalize_parameters(
        request,
        allowed={
            "tokens",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "num_local_experts",
            "topk",
            "tp_size",
            "ep_size",
            "ep_rank",
            "input_dtype",
            "router_logits_dtype",
            "weight_dtype",
            "activation",
            "swiglu_limit",
            "routing_mode",
            "routed_scaling_factor",
            "normalize_topk_weights",
            "fp8_scale_block_shape",
            "internal_activation_dtype",
            "route_scope",
            "validation",
        },
    )
    tokens = _positive_int(request.parameters, "tokens")
    hidden_size = _positive_int(request.parameters, "hidden_size")
    intermediate_size = _positive_int(request.parameters, "intermediate_size")
    num_experts = _positive_int(request.parameters, "num_experts")
    topk = _positive_int(request.parameters, "topk")
    tp_size = _optional_positive_int(request.parameters, "tp_size", 1)
    ep_size = _optional_positive_int(request.parameters, "ep_size", 1)
    ep_rank = _nonnegative_int(request.parameters, "ep_rank", 0)
    if ep_rank >= ep_size:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE ep_rank must be smaller than ep_size",
        )
    input_dtype = _parse_dtype(request.parameters, "input_dtype", "bfloat16")
    if input_dtype is not torch.bfloat16:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE apply benchmarks currently support bfloat16 inputs",
        )
    router_logits_dtype = _parse_dtype(
        request.parameters,
        "router_logits_dtype",
        "bfloat16",
    )
    weight_dtype_name, weight_dtype = _parse_weight_dtype(request.parameters)
    block_shape = _parse_block_shape(request.parameters)
    activation = request.parameters.get("activation", "silu")
    if activation not in {"silu", "swiglu"}:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE apply benchmarks currently support silu and swiglu activations",
        )
    swiglu_limit = None
    if activation == "swiglu":
        swiglu_limit = _finite_float(request.parameters, "swiglu_limit", 0.0)
        if swiglu_limit <= 0.0:
            raise BenchmarkCaseError(
                BenchmarkStatus.INVALID_CASE,
                "MoE swiglu_limit must be positive for swiglu activation",
            )
    routing_mode = request.parameters.get("routing_mode", "precomputed_topk")
    if routing_mode != "precomputed_topk":
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE apply benchmarks currently use precomputed_topk routing",
        )
    route_scope = request.parameters.get("route_scope", "local")
    if route_scope not in {"local", "global"}:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE route_scope must be 'local' or 'global'",
        )
    routed_scaling_factor = _finite_float(
        request.parameters,
        "routed_scaling_factor",
        1.0,
    )
    normalize_topk_weights = _bool_parameter(
        request.parameters,
        "normalize_topk_weights",
        True,
    )
    intermediate_size_per_partition = _intermediate_per_partition(
        intermediate_size,
        tp_size,
    )
    expected_local_experts = _num_local_experts(num_experts, ep_size)
    num_local_experts = _optional_positive_int(
        request.parameters,
        "num_local_experts",
        expected_local_experts,
    )
    if num_local_experts != expected_local_experts:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE num_local_experts must match num_experts / ep_size",
        )
    if topk > num_experts or (route_scope == "local" and topk > num_local_experts):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MoE topk exceeds the selected expert pool",
        )

    load_builtin_kernels()
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
        internal_activation_dtype=request.parameters.get(
            "internal_activation_dtype",
            "input",
        ),
        with_bias=False,
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
            "routed_scaling_factor": routed_scaling_factor,
            "normalize_topk_weights": normalize_topk_weights,
            "fp8_scale_block_shape": block_shape,
            "topk_generation": "sigmoid_bias_topk",
        },
        validation=None,
    )
