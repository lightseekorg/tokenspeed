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
from collections.abc import Callable
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
    PreparedValidation,
    ValidationInvocation,
)
from tokenspeed_kernel.benchmark.validation import OutputValidationSpec
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.tolerance import get_family_tolerance
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    ref_compatible_with_spec,
    select_kernel,
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    FormatSignature,
    ScaleFormat,
    dense_format_signature,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

# isort: split
import tokenspeed_kernel.numerics.gemm  # noqa: F401

__all__ = [
    "prepare_mv",
    "prepare_dense_bmm",
    "prepare_dense_mm",
    "prepare_grouped_bf16_projection",
    "prepare_linear_attnres_partials",
    "prepare_mm",
    "prepare_mxfp8_mm",
    "prepare_prepared_fp32_router_gemv",
]


_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
}

_DEFAULT_VALIDATION_RUNS = 5
_MXFP8_DTYPE = torch.float8_e4m3fn
_MXFP8_BLOCK_SIZE = (1, 32)
_MXFP8_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=_MXFP8_BLOCK_SIZE,
)
_MXFP8_SIGNATURE = format_signature(
    a=tensor_format("mxfp8", _MXFP8_DTYPE, scale=_MXFP8_SCALE),
    b=tensor_format("mxfp8", _MXFP8_DTYPE, scale=_MXFP8_SCALE),
)
_FP32_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_FP32_BLOCK_SIGNATURE = format_signature(
    a=tensor_format("mxfp8", _MXFP8_DTYPE, scale=_FP32_BLOCK_SCALE),
    b=tensor_format("mxfp8", _MXFP8_DTYPE, scale=_FP32_BLOCK_SCALE),
)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"GEMM parameter {name!r} must be a positive integer",
        )
    return value


def _parse_dtype(value: object) -> torch.dtype:
    dtype = _DTYPE_NAMES.get(value) if isinstance(value, str) else None
    if dtype is None:
        supported = ", ".join(sorted(_DTYPE_NAMES))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"GEMM currently supports dtype names: {supported}",
        )
    return dtype


def _tolerance(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"GEMM validation parameter {name!r} must be finite and nonnegative",
        )
    return float(value)


def _parse_validation(
    value: object,
    *,
    dtype: torch.dtype,
    K: int,
) -> dict[str, float | int] | None:
    if value is None:
        return None
    unknown = sorted(set(value) - {"runs", "atol", "rtol"})
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown GEMM validation parameters: {', '.join(unknown)}",
        )

    tolerance = get_family_tolerance("gemm")(dtype, K=K)
    return {
        "runs": _positive_int(value.get("runs", _DEFAULT_VALIDATION_RUNS), "runs"),
        "atol": _tolerance(value.get("atol", tolerance.atol), "atol"),
        "rtol": _tolerance(value.get("rtol", tolerance.rtol), "rtol"),
    }


def _bmm_traits(
    batch: int,
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> dict[str, object]:
    return {
        "batch": batch,
        "m": M,
        "n": N,
        "k": K,
        "a_inner_stride_one": True,
        "b_n_stride_one": True,
        "out_dtype": out_dtype,
        "out_inner_stride_one": True,
    }


def _validate_exact_registration(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    signature: FormatSignature,
    traits: dict[str, object],
    shape: dict[str, int],
) -> tuple[KernelSpec, SelectedKernel]:
    assert request.registration is not None
    registry = KernelRegistry.get()
    spec = registry.get_by_name(request.registration)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Required registration {request.registration!r} is not available",
        )
    if (spec.family, spec.mode) != (request.family, request.mode):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Registration {spec.name!r} belongs to {spec.family}.{spec.mode}, "
            f"not {request.family}.{request.mode}",
        )
    if not spec.capability.satisfied_by(platform):
        raise BenchmarkCaseError(
            BenchmarkStatus.NOT_APPLICABLE,
            f"Registration {spec.name!r} does not support {platform.device_name}",
        )
    if not spec.supports_format_signature(signature):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Registration {spec.name!r} does not support dense BF16 inputs",
        )
    if not spec_matches_traits(spec, traits) or not spec_matches_shape_traits(
        spec, traits
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Registration {spec.name!r} does not support parameters {shape}",
        )
    impl = registry.get_impl(spec.name)
    if impl is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Registration {spec.name!r} has no callable implementation",
        )
    return spec, SelectedKernel(spec.name, impl)


def _select_registration(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    signature: FormatSignature,
    traits: dict[str, object],
    shape: dict[str, int],
) -> tuple[KernelSpec, SelectedKernel]:
    if request.registration is not None:
        return _validate_exact_registration(request, platform, signature, traits, shape)

    try:
        selected = select_kernel(
            request.family,
            request.mode,
            signature,
            platform=platform,
            traits=traits,
            solution=request.solution,
        )
    except NoKernelFoundError as exc:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(exc)) from exc

    registry = KernelRegistry.get()
    spec = registry.get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )
    return spec, selected


def _select_reference_registration(
    spec: KernelSpec,
    signature: FormatSignature,
    traits: dict[str, object],
    platform: PlatformInfo,
) -> tuple[KernelSpec, SelectedKernel]:
    registry = KernelRegistry.get()
    references = registry.get_for_operator(
        spec.family,
        spec.mode,
        platform=platform,
        format_signature=signature,
        solution="torch",
    )
    for reference in references:
        if (
            reference.name == spec.name
            or not ref_compatible_with_spec(reference, spec)
            or not spec_matches_shape_traits(reference, traits)
        ):
            continue
        implementation = registry.get_impl(reference.name)
        if implementation is not None:
            return reference, SelectedKernel(reference.name, implementation)

    raise BenchmarkCaseError(
        BenchmarkStatus.REGISTRATION_MISSING,
        f"No compatible registered reference exists for {spec.name!r}",
    )


def _dense_mm_traits(
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> dict[str, object]:
    return {
        "m": M,
        "n": N,
        "k": K,
        "a_inner_stride_one": True,
        "b_inner_stride_one": True,
        "out_dtype": out_dtype,
    }


def _prepare_dense_gemm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
    *,
    dtype: torch.dtype,
    shape: dict[str, int],
    traits: dict[str, object],
    input_traits: dict[str, object],
    out_shape: tuple[int, ...],
    normalized_parameters: dict[str, object],
    validation_config: dict[str, float | int] | None,
) -> PreparedBenchmark:
    """Prepare a dense GEMM benchmark whose registrations write into ``out``."""

    signature = format_signature(
        a=dense_tensor_format(dtype),
        b=dense_tensor_format(dtype),
    )

    load_builtin_kernels()
    spec, selected = _select_registration(request, platform, signature, traits, shape)

    def generate_inputs(seed: int) -> dict[str, Any]:
        generator = get_input_generator(
            request.family,
            request.mode,
            dtype=dtype,
            traits=input_traits,
            format_signature=signature,
            device="cuda",
            seed=seed,
        )
        return generator.generate(**shape)

    def prepare_invocation(
        inputs: dict[str, Any],
        kernel: Callable[..., object],
        kernel_spec: KernelSpec,
    ) -> Callable[[], torch.Tensor]:
        out = torch.empty(out_shape, dtype=dtype, device=inputs["A"].device)
        call_kwargs = dict(inputs)
        call_kwargs["out"] = out

        def invoke() -> torch.Tensor:
            result = kernel(**call_kwargs)
            if result is not out:
                raise RuntimeError(
                    f"Registration {kernel_spec.name!r} did not return the "
                    "prepared output buffer"
                )
            return result

        return invoke

    performance_inputs = generate_inputs(request.seed)
    performance_invoke = prepare_invocation(performance_inputs, selected, spec)

    validation: PreparedValidation | None = None
    if validation_config is not None:
        reference_spec, reference = _select_reference_registration(
            spec,
            signature,
            traits,
            platform,
        )

        def prepare_validation_run(run_index: int) -> ValidationInvocation:
            inputs = generate_inputs(request.seed + run_index + 1)
            candidate = prepare_invocation(inputs, selected, spec)
            expected = prepare_invocation(inputs, reference, reference_spec)
            return ValidationInvocation(
                candidate=lambda: (candidate(),),
                reference=lambda: (expected(),),
            )

        validation = PreparedValidation(
            output_specs=(
                OutputValidationSpec(
                    "close",
                    {
                        "atol": validation_config["atol"],
                        "rtol": validation_config["rtol"],
                    },
                ),
            ),
            runs=validation_config["runs"],
            prepare_run=prepare_validation_run,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(
            invoke=performance_invoke,
        ),
        parameters=normalized_parameters,
        validation=validation,
    )


def prepare_dense_bmm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a dense BF16 batched GEMM registration benchmark."""

    allowed = {"batch", "M", "N", "K", "dtype", "validation"}
    unknown = sorted(set(request.parameters) - allowed)
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown dense gemm.bmm parameters: {', '.join(unknown)}",
        )

    batch = _positive_int(request.parameters.get("batch"), "batch")
    M = _positive_int(request.parameters.get("M"), "M")
    N = _positive_int(request.parameters.get("N"), "N")
    K = _positive_int(request.parameters.get("K"), "K")
    dtype = _parse_dtype(request.parameters.get("dtype", "bfloat16"))
    validation_config = _parse_validation(
        request.parameters.get("validation"),
        dtype=dtype,
        K=K,
    )
    normalized_parameters = {
        "batch": batch,
        "M": M,
        "N": N,
        "K": K,
        "dtype": "bfloat16",
        "a_layout": "BMK",
        "b_layout": "BNK",
        "b_n_stride_one": True,
        "out_layout": "BMN",
    }
    if validation_config is not None:
        normalized_parameters["validation"] = validation_config

    return _prepare_dense_gemm(
        request,
        platform,
        dtype=dtype,
        shape={"batch": batch, "M": M, "N": N, "K": K},
        traits=_bmm_traits(batch, M, N, K, dtype),
        input_traits={"b_n_stride_one": frozenset({True})},
        out_shape=(batch, M, N),
        normalized_parameters=normalized_parameters,
        validation_config=validation_config,
    )


def prepare_dense_mm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a dense BF16 GEMM registration benchmark.

    The case computes ``A @ B.T`` from K-contiguous ``A[M, K]`` and
    ``B[N, K]`` into a caller-owned ``out[M, N]``.
    """

    allowed = {"M", "N", "K", "quant", "dtype", "validation"}
    unknown = sorted(set(request.parameters) - allowed)
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown dense gemm.mm parameters: {', '.join(unknown)}",
        )

    M = _positive_int(request.parameters.get("M"), "M")
    N = _positive_int(request.parameters.get("N"), "N")
    K = _positive_int(request.parameters.get("K"), "K")
    if request.parameters.get("quant") != "none":
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "Dense gemm.mm requires quant='none'",
        )
    dtype = _parse_dtype(request.parameters.get("dtype"))
    validation_config = _parse_validation(
        request.parameters.get("validation"),
        dtype=dtype,
        K=K,
    )
    normalized_parameters: dict[str, object] = {
        "M": M,
        "N": N,
        "K": K,
        "quant": "none",
        "dtype": "bfloat16",
        "a_layout": "MK",
        "b_layout": "NK",
        "out_layout": "MN",
    }
    if validation_config is not None:
        normalized_parameters["validation"] = validation_config

    return _prepare_dense_gemm(
        request,
        platform,
        dtype=dtype,
        shape={"M": M, "N": N, "K": K},
        traits=_dense_mm_traits(M, N, K, dtype),
        input_traits={"a_layout": "MK", "b_layout": "NK"},
        out_shape=(M, N),
        normalized_parameters=normalized_parameters,
        validation_config=validation_config,
    )


def _mxfp8_mm_traits(
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> dict[str, object]:
    return {
        "m": M,
        "n": N,
        "k": K,
        "a_inner_stride_one": True,
        "b_inner_stride_one": True,
        "block_scale_layout": "canonical",
        "out_dtype": out_dtype,
        "out_inner_stride_one": True,
    }


def prepare_mxfp8_mm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare canonical E4M3 GEMM with E8M0 K32 or FP32 block128 scales."""

    allowed = {
        "M",
        "N",
        "K",
        "quant",
        "block_size",
        "out_dtype",
        "validation",
    }
    unknown = sorted(set(request.parameters) - allowed)
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown MXFP8 gemm.mm parameters: {', '.join(unknown)}",
        )

    M = _positive_int(request.parameters.get("M"), "M")
    N = _positive_int(request.parameters.get("N"), "N")
    K = _positive_int(request.parameters.get("K"), "K")
    if request.parameters.get("quant") != "mxfp8":
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MXFP8 gemm.mm requires quant='mxfp8'",
        )
    block_size = request.parameters.get("block_size")
    if block_size == list(_MXFP8_BLOCK_SIZE):
        signature = _MXFP8_SIGNATURE
        scale_dtype = "uint8"
    elif block_size == [128, 128]:
        signature = _FP32_BLOCK_SIGNATURE
        scale_dtype = "float32"
    else:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MXFP8 gemm.mm requires block_size=[1, 32] or [128, 128]",
        )
    out_dtype = _parse_dtype(request.parameters.get("out_dtype"))
    validation_config = _parse_validation(
        request.parameters.get("validation"),
        dtype=out_dtype,
        K=K,
    )

    normalized_parameters: dict[str, object] = {
        "M": M,
        "N": N,
        "K": K,
        "quant": "mxfp8",
        "value_dtype": "float8_e4m3fn",
        "scale_dtype": scale_dtype,
        "block_size": block_size,
        "out_dtype": "bfloat16",
        "a_layout": "MK",
        "b_layout": "NK",
        "scale_layout": "canonical",
        "out_layout": "MN",
    }
    if validation_config is not None:
        normalized_parameters["validation"] = validation_config

    shape = {"M": M, "N": N, "K": K}
    traits = _mxfp8_mm_traits(M, N, K, out_dtype)
    load_builtin_kernels()
    spec, selected = _select_registration(
        request,
        platform,
        signature,
        traits,
        shape,
    )

    def generate_inputs(seed: int) -> dict[str, Any]:
        generator = get_input_generator(
            request.family,
            request.mode,
            dtype=_MXFP8_DTYPE,
            traits={"a_layout": "MK", "b_layout": "NK"},
            format_signature=signature,
            device="cuda",
            seed=seed,
        )
        inputs = generator.generate(**shape)
        if scale_dtype == "uint8":
            # Match the quantized projection range used by the kernel's exact
            # correctness tests. Full-scale random FP8 values combined with large
            # UE8M0 exponents create unrealistic outputs near BF16 rounding ties.
            inputs["A"] = (inputs["A"].float() * 0.05).to(_MXFP8_DTYPE)
            inputs["B"] = (inputs["B"].float() * 0.05).to(_MXFP8_DTYPE)
            inputs["A_scales"].clamp_max_(128)
            inputs["B_scales"].clamp_max_(128)
        return inputs

    def prepare_invocation(
        inputs: dict[str, Any],
        kernel: Callable[..., object],
        kernel_spec: KernelSpec,
        *,
        validate_layout: bool,
    ) -> Callable[[], torch.Tensor]:
        A = inputs["A"]
        B = inputs["B"]
        out = torch.empty((M, N), dtype=out_dtype, device=A.device)
        if validate_layout:
            actual_traits = _mxfp8_mm_traits(M, N, K, out_dtype)
            actual_traits["a_inner_stride_one"] = A.stride(-1) == 1
            actual_traits["b_inner_stride_one"] = B.stride(-1) == 1
            if not spec_matches_shape_traits(
                kernel_spec, actual_traits
            ) or not spec_matches_traits(kernel_spec, actual_traits):
                raise BenchmarkCaseError(
                    BenchmarkStatus.INVALID_CASE,
                    "Generated tensor layouts do not satisfy registration "
                    f"{kernel_spec.name!r}",
                )

        call_kwargs = dict(inputs)
        call_kwargs["out"] = out

        def invoke() -> torch.Tensor:
            result = kernel(**call_kwargs)
            if result is not out:
                raise RuntimeError(
                    f"Registration {kernel_spec.name!r} did not return the "
                    "prepared output buffer"
                )
            return result

        return invoke

    performance_inputs = generate_inputs(request.seed)
    performance_invoke = prepare_invocation(
        performance_inputs,
        selected,
        spec,
        validate_layout=True,
    )

    validation: PreparedValidation | None = None
    if validation_config is not None:
        reference_spec, reference = _select_reference_registration(
            spec,
            signature,
            traits,
            platform,
        )

        def prepare_validation_run(run_index: int) -> ValidationInvocation:
            inputs = generate_inputs(request.seed + run_index + 1)
            candidate = prepare_invocation(
                inputs,
                selected,
                spec,
                validate_layout=True,
            )
            expected = prepare_invocation(
                inputs,
                reference,
                reference_spec,
                validate_layout=False,
            )
            return ValidationInvocation(
                candidate=lambda: (candidate(),),
                reference=lambda: (expected(),),
            )

        validation = PreparedValidation(
            output_specs=(
                OutputValidationSpec(
                    "close",
                    {
                        "atol": validation_config["atol"],
                        "rtol": validation_config["rtol"],
                    },
                ),
            ),
            runs=validation_config["runs"],
            prepare_run=prepare_validation_run,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(
            invoke=performance_invoke,
        ),
        parameters=normalized_parameters,
        validation=validation,
    )


_MM_GENERATORS: dict[str, Callable[..., PreparedBenchmark]] = {
    "mxfp8": prepare_mxfp8_mm,
    "none": prepare_dense_mm,
}


def prepare_mm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a ``gemm.mm`` benchmark for the requested quantization.

    ``quant`` is required: ``"none"`` selects dense GEMM and ``"mxfp8"``
    selects canonical E4M3 GEMM, with the explicit ``block_size`` choosing
    E8M0 K32 or FP32 block128 scales.
    """

    quant = request.parameters.get("quant")
    generator = _MM_GENERATORS.get(quant) if isinstance(quant, str) else None
    if generator is None:
        accepted = ", ".join(sorted(_MM_GENERATORS))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"gemm.mm requires quant to be one of: {accepted}",
        )
    return generator(request, platform)


def _reject_unknown_parameters(request: BenchmarkRequest, allowed: set[str]) -> None:
    unknown = sorted(set(request.parameters) - allowed)
    if unknown:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"Unknown gemm.{request.mode} parameters: {', '.join(unknown)}",
        )


def _random_tensors(
    seed: int,
    dtype: torch.dtype,
    *shapes: tuple[int, ...],
) -> list[torch.Tensor]:
    """Return seeded standard-normal device tensors, one per shape."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        for shape in shapes
    ]


def _prepared_timing(
    spec: KernelSpec,
    invoke: Callable[[], torch.Tensor],
    parameters: dict[str, object],
) -> PreparedBenchmark:
    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters=parameters,
        validation=None,
    )


def prepare_mv(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a small-M dense GEMV benchmark computing ``x @ weight.T``."""

    _reject_unknown_parameters(request, {"M", "N", "K", "dtype"})
    M = _positive_int(request.parameters.get("M"), "M")
    N = _positive_int(request.parameters.get("N"), "N")
    K = _positive_int(request.parameters.get("K"), "K")
    dtype = _parse_dtype(request.parameters.get("dtype"))
    shape = {"M": M, "N": N, "K": K}
    signature = format_signature(
        x=dense_tensor_format(dtype),
        weight=dense_tensor_format(dtype),
    )

    load_builtin_kernels()
    spec, selected = _select_registration(
        request,
        platform,
        signature,
        {"m": M, "n": N, "k": K},
        shape,
    )

    x, weight = _random_tensors(request.seed, dtype, (M, K), (N, K))
    out = torch.empty((M, N), dtype=dtype, device=x.device)

    def invoke() -> torch.Tensor:
        return selected(x, weight, out)

    return _prepared_timing(
        spec,
        invoke,
        {
            "M": M,
            "N": N,
            "K": K,
            "dtype": "bfloat16",
            "x_layout": "MK",
            "weight_layout": "NK",
            "out_layout": "MN",
        },
    )


def prepare_prepared_fp32_router_gemv(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare an FP32 router GEMV computing ``(x @ weight) * multiplier``.

    The router weight is stored ``[K, N]`` and converted once to the selected
    kernel's persistent layout, as model loading does; each invocation runs
    the planned kernel over ``x[M, K]``.
    """

    if request.registration is not None:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "gemm.prepared_fp32_router_gemv selects through its plan; "
            "use solution instead of registration",
        )
    _reject_unknown_parameters(request, {"M", "N", "K", "multiplier", "dtype"})
    M = _positive_int(request.parameters.get("M"), "M")
    N = _positive_int(request.parameters.get("N"), "N")
    K = _positive_int(request.parameters.get("K"), "K")
    multiplier = request.parameters.get("multiplier")
    if not isinstance(multiplier, float) or not math.isfinite(multiplier):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "gemm.prepared_fp32_router_gemv multiplier must be a finite float",
        )
    if request.parameters.get("dtype") != "float32":
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "gemm.prepared_fp32_router_gemv dtype must be float32",
        )

    from tokenspeed_kernel.ops import gemm as gemm_ops

    load_builtin_kernels()
    try:
        # The plan's own selection, repeated to name the timed registration.
        selected = select_kernel(
            request.family,
            request.mode,
            dense_format_signature(("x", "weight"), (torch.float32, torch.float32)),
            features=frozenset(),
            platform=platform,
            solution=request.solution,
        )
        plan = gemm_ops.fp32_router_gemv_plan(
            hidden_size=K,
            output_size=N,
            dtype=torch.float32,
            solution=request.solution,
            batch_invariant=False,
        )
    except NoKernelFoundError as exc:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(exc)) from exc
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )

    x, weight = _random_tensors(request.seed, torch.float32, (M, K), (K, N))
    prepared_weight = gemm_ops.fp32_router_gemv_process_weight(plan, weight)
    counters = gemm_ops.fp32_router_gemv_counters(N, x.device)

    def invoke() -> torch.Tensor:
        return gemm_ops.fp32_router_gemv_execute(
            plan, x, prepared_weight, multiplier, counters, False
        )

    return _prepared_timing(
        spec,
        invoke,
        {
            "M": M,
            "N": N,
            "K": K,
            "multiplier": multiplier,
            "dtype": "float32",
            "x_layout": "MK",
            "weight_layout": "KN",
            "out_layout": "MN",
        },
    )


def prepare_grouped_bf16_projection(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a grouped projection computing ``einsum('tgd,grd->tgr')``.

    ``x[M, groups, K]`` is a view whose rows are ``x_token_stride`` elements
    apart: ``groups * K`` for a contiguous activation, or more for a slice of
    padded attention heads. ``weight[groups, N, K]`` is contiguous. As in the
    models, no ``out`` buffer is passed, so the registration allocates its
    ``[M, groups, N]`` result.
    """

    _reject_unknown_parameters(
        request, {"groups", "M", "N", "K", "x_token_stride", "dtype"}
    )
    groups = _positive_int(request.parameters.get("groups"), "groups")
    M = _positive_int(request.parameters.get("M"), "M")
    N = _positive_int(request.parameters.get("N"), "N")
    K = _positive_int(request.parameters.get("K"), "K")
    x_token_stride = _positive_int(
        request.parameters.get("x_token_stride"), "x_token_stride"
    )
    if x_token_stride < groups * K:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "gemm.grouped_bf16_projection x_token_stride must be at least "
            "groups * K",
        )
    dtype = _parse_dtype(request.parameters.get("dtype"))
    shape = {"groups": groups, "M": M, "N": N, "K": K}

    rows, weight = _random_tensors(
        request.seed, dtype, (M, x_token_stride), (groups, N, K)
    )
    x = rows[:, : groups * K].unflatten(1, (groups, K))
    signature = format_signature(
        x=dense_tensor_format(dtype),
        weight=dense_tensor_format(dtype),
    )
    # Mirror the operation's selection traits for these inputs.
    traits = {
        "batch": groups,
        "m": M,
        "n": N,
        "k": K,
        "a_inner_stride_one": x.stride(-1) == 1,
        "b_contiguous": weight.is_contiguous(),
        "b_inner_stride_one": weight.stride(-1) == 1,
        "is_cuda": x.is_cuda,
    }

    load_builtin_kernels()
    spec, selected = _select_registration(request, platform, signature, traits, shape)

    def invoke() -> torch.Tensor:
        return selected(x, weight, None)

    return _prepared_timing(
        spec,
        invoke,
        {
            **shape,
            "x_token_stride": x_token_stride,
            "dtype": "bfloat16",
            "x_layout": "MGK",
            "weight_layout": "GNK",
            "out_layout": "MGN",
        },
    )


def prepare_linear_attnres_partials(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a projection fused with two AttnRes block partials.

    The case projects ``hidden_states[tokens, input_size]`` through
    ``weight[output_size, input_size]`` and reduces ``num_blocks`` residual
    blocks into two FP32 ``(max, sum, accumulator)`` scratch tuples.
    """

    _reject_unknown_parameters(
        request,
        {"tokens", "input_size", "output_size", "num_blocks", "dtype", "eps"},
    )
    tokens = _positive_int(request.parameters.get("tokens"), "tokens")
    input_size = _positive_int(request.parameters.get("input_size"), "input_size")
    output_size = _positive_int(request.parameters.get("output_size"), "output_size")
    num_blocks = _positive_int(request.parameters.get("num_blocks"), "num_blocks")
    dtype = _parse_dtype(request.parameters.get("dtype"))
    eps = request.parameters.get("eps")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "gemm.linear_attnres_partials parameter 'eps' must be a positive float",
        )

    shape = {
        "tokens": tokens,
        "input_size": input_size,
        "output_size": output_size,
        "num_blocks": num_blocks,
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(dtype),
        weight=dense_tensor_format(dtype),
        blocks=dense_tensor_format(dtype),
        score_weight_a=dense_tensor_format(dtype),
        score_weight_b=dense_tensor_format(dtype),
        out=dense_tensor_format(dtype),
    )
    traits: dict[str, object] = {**shape, "inputs_contiguous": True}
    if platform.is_cdna5:
        # Mirror the operation's dispatch, which opts CDNA5 into its kernel.
        traits["gfx1250_linear_attnres_enabled"] = True

    load_builtin_kernels()
    spec, selected = _select_registration(request, platform, signature, traits, shape)

    hidden_states, weight, blocks, score_weight_a, score_weight_b = _random_tensors(
        request.seed,
        dtype,
        (tokens, input_size),
        (output_size, input_size),
        (num_blocks, tokens, input_size),
        (input_size,),
        (input_size,),
    )

    def scratch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(tokens, dtype=torch.float32, device=blocks.device),
            torch.empty(tokens, dtype=torch.float32, device=blocks.device),
            torch.empty(
                (tokens, input_size), dtype=torch.float32, device=blocks.device
            ),
        )

    scratch_a = scratch()
    scratch_b = scratch()
    out = torch.empty((tokens, output_size), dtype=dtype, device=blocks.device)

    def invoke() -> torch.Tensor:
        return selected(
            hidden_states=hidden_states,
            weight=weight,
            blocks=blocks,
            score_weight_a=score_weight_a,
            score_weight_b=score_weight_b,
            scratch_a=scratch_a,
            scratch_b=scratch_b,
            eps=eps,
            out=out,
        )

    return _prepared_timing(
        spec,
        invoke,
        {
            **shape,
            "dtype": "bfloat16",
            "eps": eps,
            "scratch_dtype": "float32",
        },
    )
