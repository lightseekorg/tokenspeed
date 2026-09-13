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
from tokenspeed_kernel.benchmark.validation import (
    MAX_VALIDATION_RUNS,
    OutputValidationSpec,
)
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
    dense_tensor_format,
    format_signature,
)

# isort: split
import tokenspeed_kernel.numerics.gemm  # noqa: F401

__all__ = ["prepare_dense_bmm"]


_DTYPE_NAMES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

_DEFAULT_VALIDATION_RUNS = 5


def _positive_dimension(parameters: dict[str, Any], name: str) -> int:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"gemm.bmm parameter {name!r} must be a positive integer",
        )
    return value


def _parse_dtype(value: object) -> torch.dtype:
    if isinstance(value, str):
        dtype = _DTYPE_NAMES.get(value.lower())
    else:
        dtype = value if isinstance(value, torch.dtype) else None
    if dtype is not torch.bfloat16:
        supported = ", ".join(sorted(_DTYPE_NAMES))
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"dense gemm.bmm currently supports dtype names: {supported}",
        )
    return dtype


def _nonnegative_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"gemm.bmm validation parameter {name!r} must be a number",
        )
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            f"gemm.bmm validation parameter {name!r} must be finite and nonnegative",
        )
    return converted


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
            f"Unknown gemm.bmm validation parameters: {', '.join(unknown)}",
        )

    runs = value.get("runs", _DEFAULT_VALIDATION_RUNS)
    if (
        isinstance(runs, bool)
        or not isinstance(runs, int)
        or not 0 < runs <= MAX_VALIDATION_RUNS
    ):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "gemm.bmm validation parameter 'runs' must be an integer between "
            f"1 and {MAX_VALIDATION_RUNS}",
        )
    tolerance = get_family_tolerance("gemm")(dtype, K=K)
    atol = _nonnegative_finite_number(value.get("atol", tolerance.atol), "atol")
    rtol = _nonnegative_finite_number(value.get("rtol", tolerance.rtol), "rtol")
    return {"runs": runs, "atol": atol, "rtol": rtol}


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
        "out_inner_stride_one": True,
        "out_dtype": out_dtype,
        "n_align_16": N % 16 == 0,
        "k_align_16": K % 16 == 0,
        "k_align_32": K % 32 == 0,
        "n_align_64": N % 64 == 0,
        "n_align_128": N % 128 == 0,
        "k_align_64": K % 64 == 0,
        "k_align_128": K % 128 == 0,
        "n_min_128": N >= 128,
        "k_min_128": K >= 128,
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
    if not spec_matches_shape_traits(spec, shape) or not spec_matches_traits(
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
        status = _selection_miss_status(request, platform)
        raise BenchmarkCaseError(status, str(exc)) from exc

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
    shape: dict[str, int],
    platform: PlatformInfo,
) -> tuple[KernelSpec, SelectedKernel]:
    registry = KernelRegistry.get()
    references = registry.get_for_operator(
        spec.family,
        spec.mode,
        platform=platform,
        format_signature=signature,
        solution="reference",
    )
    for reference in references:
        if (
            reference.name == spec.name
            or not ref_compatible_with_spec(reference, spec)
            or not spec_matches_shape_traits(reference, shape)
        ):
            continue
        implementation = registry.get_impl(reference.name)
        if implementation is not None:
            return reference, SelectedKernel(reference.name, implementation)

    raise BenchmarkCaseError(
        BenchmarkStatus.REGISTRATION_MISSING,
        f"No compatible registered reference exists for {spec.name!r}",
    )


def _selection_miss_status(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> BenchmarkStatus:
    if request.solution is None:
        return BenchmarkStatus.NOT_APPLICABLE

    registry = KernelRegistry.get()
    solution_specs = registry.get_for_operator(
        request.family,
        request.mode,
        solution=request.solution,
    )
    if not solution_specs:
        return BenchmarkStatus.BACKEND_UNAVAILABLE
    if not any(spec.capability.satisfied_by(platform) for spec in solution_specs):
        return BenchmarkStatus.NOT_APPLICABLE
    return BenchmarkStatus.INVALID_CASE


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

    batch = _positive_dimension(request.parameters, "batch")
    M = _positive_dimension(request.parameters, "M")
    N = _positive_dimension(request.parameters, "N")
    K = _positive_dimension(request.parameters, "K")
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
    shape = {"batch": batch, "M": M, "N": N, "K": K}
    signature = format_signature(
        a=dense_tensor_format(dtype),
        b=dense_tensor_format(dtype),
    )
    traits = _bmm_traits(batch, M, N, K, dtype)

    load_builtin_kernels()
    spec, selected = _select_registration(request, platform, signature, traits, shape)

    def generate_inputs(seed: int) -> dict[str, Any]:
        generator = get_input_generator(
            request.family,
            request.mode,
            dtype=dtype,
            traits={"b_n_stride_one": frozenset({True})},
            format_signature=signature,
            device="cuda",
            seed=seed,
        )
        return generator.generate(**shape)

    def prepare_invocation(
        inputs: dict[str, Any],
        kernel: Callable[..., object],
        kernel_spec: KernelSpec,
        *,
        validate_layout: bool,
    ) -> Callable[[], torch.Tensor]:
        A = inputs["A"]
        B = inputs["B"]
        out = torch.empty((batch, M, N), dtype=dtype, device=A.device)
        if validate_layout:
            actual_traits = _bmm_traits(batch, M, N, K, dtype)
            actual_traits["a_inner_stride_one"] = A.stride(-1) == 1
            actual_traits["b_n_stride_one"] = B.stride(1) == 1
            actual_traits["out_inner_stride_one"] = out.stride(-1) == 1
            if not spec_matches_traits(kernel_spec, actual_traits):
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
            shape,
            platform,
        )
        validation_runs = int(validation_config["runs"])
        validation_kwargs = {
            "atol": float(validation_config["atol"]),
            "rtol": float(validation_config["rtol"]),
        }

        def output_specs() -> tuple[OutputValidationSpec]:
            return (
                OutputValidationSpec(
                    "close",
                    validation_runs,
                    validation_kwargs,
                ),
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
            output_specs=output_specs,
            prepare_run=prepare_validation_run,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(
            invoke=performance_invoke,
            repeat_safe=True,
        ),
        parameters=normalized_parameters,
        validation=validation,
    )
