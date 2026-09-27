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
    "OutputValidationSpec",
    "ValidationDatum",
    "ValidationOutcome",
    "get_output_validator",
    "set_output_validator",
    "validate_output",
]


@dataclass(frozen=True)
class OutputValidationSpec:
    """Select a validator and its options for one logical output."""

    validator: str
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
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
    """Validate every candidate/reference pair of one output with ``spec``."""

    return get_output_validator(spec.validator)(spec, tuple(data))


def _max_metric(current: float, value: float) -> float:
    if not math.isfinite(value):
        return value
    return max(current, value)


def _validate_close(
    spec: OutputValidationSpec,
    data: tuple[ValidationDatum, ...],
) -> ValidationOutcome:
    tolerance = Tolerance(
        atol=float(spec.kwargs["atol"]),
        rtol=float(spec.kwargs["rtol"]),
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
