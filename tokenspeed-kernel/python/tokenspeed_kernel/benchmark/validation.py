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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.numerics.comparison import compare_outputs
from tokenspeed_kernel.numerics.tolerance import Tolerance

__all__ = [
    "MAX_VALIDATION_RUNS",
    "OutputValidationSpec",
    "ValidationDatum",
    "ValidationOutcome",
    "get_output_validator",
    "set_output_validator",
    "validate_output",
]

MAX_VALIDATION_RUNS = 100
_CLOSE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@dataclass(frozen=True)
class OutputValidationSpec:
    """Select a validator and the data it needs for one logical output."""

    validator: str
    runs: int
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.runs, bool)
            or not isinstance(self.runs, int)
            or not 0 < self.runs <= MAX_VALIDATION_RUNS
        ):
            raise ValueError(f"runs must be between 1 and {MAX_VALIDATION_RUNS}")
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class ValidationDatum:
    """One candidate output and its reference output."""

    actual: object
    expected: object


@dataclass(frozen=True)
class ValidationOutcome:
    """A validator verdict and an optional diagnostic message."""

    passed: bool
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")


OutputValidator = Callable[
    [OutputValidationSpec, tuple[ValidationDatum, ...]], ValidationOutcome
]
_OUTPUT_VALIDATORS: dict[str, OutputValidator] = {}


def set_output_validator(name: str, validator: OutputValidator) -> None:
    """Associate a name with an output-validation function.

    Args:
        name: Name used by output validation specifications.
        validator: Callable that receives an output specification and the
            requested sequence of candidate/reference pairs, then returns a
            validation outcome.

    Returns:
        None.
    """

    _OUTPUT_VALIDATORS[name] = validator


def get_output_validator(name: str) -> OutputValidator:
    """Return the output validator registered under ``name``.

    Args:
        name: Validator name from an output validation specification.

    Returns:
        Callable that receives an output specification and candidate/reference
        data, then returns a validation outcome.

    Raises:
        KeyError: If no validator is registered under ``name``.
    """

    validator = _OUTPUT_VALIDATORS.get(name)
    if validator is None:
        known = ", ".join(sorted(_OUTPUT_VALIDATORS)) or "none"
        raise KeyError(f"Unknown output validator {name!r}. Known: {known}")
    return validator


def validate_output(
    spec: OutputValidationSpec,
    data: Sequence[ValidationDatum],
) -> ValidationOutcome:
    """Validate exactly the number of candidate/reference pairs in ``spec``."""

    pairs = tuple(data)
    if len(pairs) != spec.runs:
        raise ValueError(
            f"validation data has {len(pairs)} run(s), but spec requires {spec.runs}"
        )
    return get_output_validator(spec.validator)(spec, pairs)


def _finite_nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


def _max_metric(current: float, value: float) -> float:
    if not math.isfinite(value):
        return value
    return max(current, value)


def _validate_close(
    spec: OutputValidationSpec,
    data: tuple[ValidationDatum, ...],
) -> ValidationOutcome:
    expected_kwargs = {"atol", "rtol"}
    if set(spec.kwargs) != expected_kwargs:
        missing = sorted(expected_kwargs - spec.kwargs.keys())
        unknown = sorted(spec.kwargs.keys() - expected_kwargs)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError("close validator kwargs: " + "; ".join(details))

    tolerance = Tolerance(
        atol=_finite_nonnegative_number(spec.kwargs["atol"], "atol"),
        rtol=_finite_nonnegative_number(spec.kwargs["rtol"], "rtol"),
    )
    total_elements = 0
    total_mismatches = 0
    weighted_abs_diff = 0.0
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for run_index, datum in enumerate(data):
        if not isinstance(datum.actual, torch.Tensor) or not isinstance(
            datum.expected, torch.Tensor
        ):
            return ValidationOutcome(
                False,
                f"run={run_index} requires tensor outputs, got "
                f"actual={type(datum.actual).__name__} "
                f"expected={type(datum.expected).__name__}",
            )
        if datum.actual.dtype != datum.expected.dtype:
            return ValidationOutcome(
                False,
                f"run={run_index} requires matching dtypes, got "
                f"actual={datum.actual.dtype} expected={datum.expected.dtype}",
            )
        if datum.actual.dtype not in _CLOSE_DTYPES:
            return ValidationOutcome(
                False,
                f"run={run_index} close validation does not support "
                f"dtype={datum.actual.dtype}",
            )

        try:
            comparison = compare_outputs(
                datum.actual,
                datum.expected,
                tolerance=tolerance,
            )
        except (TypeError, ValueError) as error:
            return ValidationOutcome(
                False,
                f"run={run_index} comparison failed: {type(error).__name__}: {error}",
            )

        total_elements += comparison.total_elements
        total_mismatches += comparison.num_mismatches
        weighted_abs_diff += comparison.mean_abs_diff * comparison.total_elements
        max_abs_diff = _max_metric(max_abs_diff, comparison.max_abs_diff)
        max_rel_diff = _max_metric(max_rel_diff, comparison.max_rel_diff)

    mean_abs_diff = weighted_abs_diff / total_elements if total_elements else 0.0
    diagnostic = (
        f"runs={len(data)} elements={total_elements} mismatches={total_mismatches} "
        f"max_abs_diff={max_abs_diff:.6e} mean_abs_diff={mean_abs_diff:.6e} "
        f"max_rel_diff={max_rel_diff:.6e}"
    )
    return ValidationOutcome(total_mismatches == 0, diagnostic)


set_output_validator("close", _validate_close)
