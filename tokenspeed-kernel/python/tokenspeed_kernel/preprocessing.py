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

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass

__all__ = [
    "WeightPreprocessorConflictError",
    "WeightPreprocessorRequest",
    "resolve_weight_preprocessor_conflict",
]


class WeightPreprocessorConflictError(RuntimeError):
    """Raised when selected kernels require incompatible module weight layouts."""


def _normalize_weight_preprocessors(
    preprocessors: Callable | Iterable[Callable],
) -> tuple[Callable, ...]:
    if callable(preprocessors):
        normalized = (preprocessors,)
    else:
        normalized = tuple(preprocessors)
    if not normalized:
        raise ValueError("weight preprocessor request must contain at least one item")
    for preprocessor in normalized:
        if not callable(preprocessor):
            raise TypeError("weight preprocessors must be callable")
    if len(set(normalized)) != len(normalized):
        raise ValueError("weight preprocessors must be unique")
    return normalized


def _preprocessor_name(preprocessor: Callable) -> str:
    return getattr(preprocessor, "__name__", repr(preprocessor))


@dataclass(frozen=True, init=False)
class WeightPreprocessorRequest:
    """Ordered weight-preprocessor candidates requested by one kernel path."""

    preprocessors: tuple[Callable, ...]
    required: bool

    def __init__(
        self,
        preprocessors: Callable | Iterable[Callable],
        *,
        required: bool = True,
    ) -> None:
        object.__setattr__(
            self,
            "preprocessors",
            _normalize_weight_preprocessors(preprocessors),
        )
        object.__setattr__(self, "required", required)
        if not isinstance(self.required, bool):
            raise TypeError("weight preprocessor request required must be a bool")

    @property
    def preprocessor(self) -> Callable:
        """Primary candidate, for single-preprocessor callers and display."""
        return self.preprocessors[0]


def _ordered_common_preprocessors(
    requests: list[WeightPreprocessorRequest],
) -> tuple[Callable, ...]:
    common = set(requests[0].preprocessors)
    for request in requests[1:]:
        common.intersection_update(request.preprocessors)
    if not common:
        return ()

    def rank(preprocessor: Callable) -> tuple[int, int, tuple[int, ...], str]:
        ranks = tuple(request.preprocessors.index(preprocessor) for request in requests)
        return max(ranks), sum(ranks), ranks, _preprocessor_name(preprocessor)

    return tuple(sorted(common, key=rank))


def resolve_weight_preprocessor_conflict(
    requests: Iterable[WeightPreprocessorRequest | None],
) -> WeightPreprocessorRequest | None:
    """Resolve module-scoped preprocessing requests.

    ``None`` is a required no-preprocessing constraint. Callable requests are
    compatible when they share at least one candidate preprocessor. The resolved
    request names the highest-priority shared callable across the inputs.
    """
    required_no_preprocessing = False
    preprocessor_requests: list[WeightPreprocessorRequest] = []

    for request in requests:
        if request is None:
            required_no_preprocessing = True
            continue
        preprocessor_requests.append(request)

    if not preprocessor_requests:
        return None

    if required_no_preprocessing:
        names = sorted(
            {
                _preprocessor_name(preprocessor)
                for request in preprocessor_requests
                for preprocessor in request.preprocessors
            }
        )
        if any(request.required for request in preprocessor_requests):
            raise WeightPreprocessorConflictError(
                "Conflicting module preprocessing requirements: required "
                f"no-preprocessing and required preprocessors {names!r}"
            )
        warnings.warn(
            "skipping conflicting preprocessors: "
            f"{names!r} conflict with required no preprocessing; "
            "this may hurt performance",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    common_preprocessors = _ordered_common_preprocessors(preprocessor_requests)
    if not common_preprocessors:
        names = sorted(
            {
                _preprocessor_name(preprocessor)
                for request in preprocessor_requests
                for preprocessor in request.preprocessors
            }
        )
        if any(request.required for request in preprocessor_requests):
            required_names = sorted(
                {
                    _preprocessor_name(preprocessor)
                    for request in preprocessor_requests
                    if request.required
                    for preprocessor in request.preprocessors
                }
            )
            raise WeightPreprocessorConflictError(
                f"Conflicting module preprocessors {names!r}; "
                f"required preprocessors: {required_names!r}"
            )
        warnings.warn(
            f"skipping conflicting preprocessors: {names!r}; "
            "this may hurt performance",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return WeightPreprocessorRequest(
        common_preprocessors[0],
        required=any(request.required for request in preprocessor_requests),
    )
