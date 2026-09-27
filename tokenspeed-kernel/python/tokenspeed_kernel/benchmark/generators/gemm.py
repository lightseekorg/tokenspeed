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
    dense_tensor_format,
    format_signature,
    tensor_format,
)

# isort: split
import tokenspeed_kernel.numerics.gemm  # noqa: F401

__all__ = ["prepare_dense_bmm", "prepare_mxfp8_mm"]


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
        solution="reference",
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
    ) -> Callable[[], torch.Tensor]:
        out = torch.empty((batch, M, N), dtype=dtype, device=inputs["A"].device)
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
    }


def prepare_mxfp8_mm(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare a canonical E4M3/UE8M0 MXFP8 GEMM benchmark."""

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
    if request.parameters.get("block_size") != list(_MXFP8_BLOCK_SIZE):
        raise BenchmarkCaseError(
            BenchmarkStatus.INVALID_CASE,
            "MXFP8 gemm.mm requires block_size=[1, 32]",
        )
    out_dtype = _parse_dtype(request.parameters.get("out_dtype"))
    validation_config = _parse_validation(
        request.parameters.get("validation"),
        dtype=out_dtype,
        K=K,
    )

    block_size = list(_MXFP8_BLOCK_SIZE)
    normalized_parameters: dict[str, object] = {
        "M": M,
        "N": N,
        "K": K,
        "quant": "mxfp8",
        "value_dtype": "float8_e4m3fn",
        "scale_dtype": "uint8",
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
        _MXFP8_SIGNATURE,
        traits,
        shape,
    )

    def generate_inputs(seed: int) -> dict[str, Any]:
        generator = get_input_generator(
            request.family,
            request.mode,
            dtype=_MXFP8_DTYPE,
            traits={"a_layout": "MK", "b_layout": "NK"},
            format_signature=_MXFP8_SIGNATURE,
            device="cuda",
            seed=seed,
        )
        inputs = generator.generate(**shape)
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
            _MXFP8_SIGNATURE,
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
