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

import pytest
import torch
from tokenspeed_kernel.benchmark.validation import (
    OutputValidationSpec,
    ValidationDatum,
    ValidationOutcome,
    get_output_validator,
    set_output_validator,
    validate_output,
)


def _close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> ValidationOutcome:
    return validate_output(
        OutputValidationSpec("close", 1, {"atol": atol, "rtol": rtol}),
        [ValidationDatum(actual, expected)],
    )


def test_output_validator_registry_dispatches() -> None:
    calls: list[tuple[OutputValidationSpec, tuple[ValidationDatum, ...]]] = []

    def custom(spec, data):
        calls.append((spec, data))
        return ValidationOutcome(True, "checked")

    set_output_validator("unit_custom", custom)
    spec = OutputValidationSpec("unit_custom", 1, {"setting": 1})
    datum = ValidationDatum("candidate", "reference")

    assert get_output_validator("unit_custom") is custom
    assert validate_output(spec, [datum]) == ValidationOutcome(True, "checked")
    assert calls == [(spec, (datum,))]

    with pytest.raises(TypeError, match="passed"):
        ValidationOutcome(1)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"validator": "close", "runs": 0, "kwargs": {}}, ValueError, "between"),
        (
            {"validator": "close", "runs": 101, "kwargs": {}},
            ValueError,
            "between",
        ),
        (
            {"validator": "close", "runs": True, "kwargs": {}},
            ValueError,
            "between",
        ),
    ],
)
def test_output_validation_spec_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        OutputValidationSpec(**kwargs)


def test_close_aggregates_all_runs() -> None:
    outcome = validate_output(
        OutputValidationSpec("close", 2, {"atol": 0.0, "rtol": 0.0}),
        [
            ValidationDatum(
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0]),
            ),
            ValidationDatum(
                torch.tensor([1.0, 4.0]),
                torch.tensor([1.0, 2.0]),
            ),
        ],
    )

    assert outcome.passed is False
    assert outcome.diagnostic is not None
    assert "runs=2" in outcome.diagnostic
    assert "elements=4" in outcome.diagnostic
    assert "mismatches=1" in outcome.diagnostic
    assert "max_abs_diff=2.000000e+00" in outcome.diagnostic
    assert "mean_abs_diff=5.000000e-01" in outcome.diagnostic


def test_close_uses_independent_absolute_and_relative_tolerances() -> None:
    absolute = _close(
        torch.tensor([0.05]),
        torch.tensor([0.0]),
        atol=0.1,
        rtol=0.0,
    )
    relative = _close(
        torch.tensor([105.0]),
        torch.tensor([100.0]),
        atol=1.0,
        rtol=0.1,
    )

    assert absolute.passed is True
    assert relative.passed is True


def test_close_treats_nonfinite_values_as_mismatches() -> None:
    outcome = _close(
        torch.tensor([float("nan")]),
        torch.tensor([float("nan")]),
        atol=1e6,
        rtol=1e6,
    )

    assert outcome.passed is False
    assert "mismatches=1" in (outcome.diagnostic or "")
    assert "max_abs_diff=nan" in (outcome.diagnostic or "")


@pytest.mark.parametrize(
    ("actual", "expected", "detail"),
    [
        ("not a tensor", torch.tensor([1.0]), "requires tensor outputs"),
        (torch.tensor([1.0]), torch.tensor([1.0, 2.0]), "Shape mismatch"),
        (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float64),
            "matching dtypes",
        ),
        (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            "does not support dtype",
        ),
    ],
)
def test_close_reports_unsupported_output_pairs(actual, expected, detail) -> None:
    outcome = validate_output(
        OutputValidationSpec("close", 1, {"atol": 0.0, "rtol": 0.0}),
        [ValidationDatum(actual, expected)],
    )

    assert outcome.passed is False
    assert detail in (outcome.diagnostic or "")


def test_validate_output_requires_known_validator_and_expected_run_count() -> None:
    with pytest.raises(KeyError, match="Unknown output validator"):
        validate_output(
            OutputValidationSpec("missing", 1, {}),
            [ValidationDatum("candidate", "reference")],
        )

    with pytest.raises(ValueError, match="has 1 run.*requires 2"):
        validate_output(
            OutputValidationSpec("close", 2, {"atol": 0.0, "rtol": 0.0}),
            [ValidationDatum(torch.tensor([1.0]), torch.tensor([1.0]))],
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"atol": 0.0}, ValueError, "missing rtol"),
        ({"atol": 0.0, "rtol": 0.0, "extra": 1}, ValueError, "unknown extra"),
        ({"atol": -1.0, "rtol": 0.0}, ValueError, "nonnegative"),
        ({"atol": True, "rtol": 0.0}, TypeError, "number"),
    ],
)
def test_close_rejects_invalid_kwargs(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        validate_output(
            OutputValidationSpec("close", 1, kwargs),
            [ValidationDatum(torch.tensor([1.0]), torch.tensor([1.0]))],
        )
