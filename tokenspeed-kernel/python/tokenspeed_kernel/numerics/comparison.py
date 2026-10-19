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

from dataclasses import dataclass

import torch
from tokenspeed_kernel.numerics.tolerance import Tolerance

__all__ = [
    "ComparisonResult",
    "compare_outputs",
    "format_comparison",
]


@dataclass
class ComparisonResult:
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    num_mismatches: int
    total_elements: int
    mismatch_fraction: float
    worst_indices: list[tuple[int, ...]]
    worst_values: list[tuple[float, float]]


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords: list[int] = []
    remaining = index
    for dim in reversed(shape):
        coords.append(remaining % dim)
        remaining //= dim
    return tuple(reversed(coords))


def _get_worst_flat_indices(diff: torch.Tensor, top_k: int) -> list[int]:
    flat = diff.reshape(-1)
    k = min(top_k, flat.numel())
    if k == 0:
        return []
    _, idx = torch.topk(flat, k=k)
    return [int(i.item()) for i in idx]


def _get_worst_indices(
    shape: tuple[int, ...],
    flat_indices: list[int],
) -> list[tuple[int, ...]]:
    return [_unravel_index(index, shape) for index in flat_indices]


def _get_worst_values(
    actual: torch.Tensor,
    expected: torch.Tensor,
    flat_indices: list[int],
) -> list[tuple[float, float]]:
    if not flat_indices:
        return []

    flat_actual = actual.reshape(-1)
    flat_expected = expected.reshape(-1)

    values: list[tuple[float, float]] = []
    for j in flat_indices:
        values.append((float(flat_actual[j].item()), float(flat_expected[j].item())))
    return values


def compare_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: Tolerance,
) -> ComparisonResult:
    """An element mismatches if it exceeds both atol and rtol, or is non-finite."""
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )

    if actual.numel() == 0:
        return ComparisonResult(
            passed=True,
            max_abs_diff=0.0,
            max_rel_diff=0.0,
            mean_abs_diff=0.0,
            num_mismatches=0,
            total_elements=0,
            mismatch_fraction=0.0,
            worst_indices=[],
            worst_values=[],
        )

    diff = (actual.float() - expected.float()).abs()
    abs_expected = expected.float().abs()
    rel_diff = diff / (abs_expected + 1e-12)

    exceeds = (diff > tolerance.atol) & (rel_diff > tolerance.rtol)
    non_finite = (
        ~torch.isfinite(actual)
        | ~torch.isfinite(expected)
        | ~torch.isfinite(diff)
        | ~torch.isfinite(rel_diff)
    )
    mismatches = exceeds | non_finite
    num_mismatches = int(mismatches.sum().item())
    total_elements = int(actual.numel())
    worst_flat_indices = _get_worst_flat_indices(diff, 10)

    return ComparisonResult(
        passed=num_mismatches == 0,
        max_abs_diff=float(diff.max().item()),
        max_rel_diff=float(rel_diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        num_mismatches=num_mismatches,
        total_elements=total_elements,
        mismatch_fraction=num_mismatches / total_elements,
        worst_indices=_get_worst_indices(tuple(diff.shape), worst_flat_indices),
        worst_values=_get_worst_values(actual, expected, worst_flat_indices),
    )


def format_comparison(result: ComparisonResult, kernel_name: str = "") -> str:
    status = "PASS" if result.passed else "FAIL"
    header = f"[{status}]"
    if kernel_name:
        header = f"{header} {kernel_name}"

    lines = [
        header,
        (
            f"  max_abs_diff={result.max_abs_diff:.6e}  "
            f"max_rel_diff={result.max_rel_diff:.6e}"
        ),
        f"  mean_abs_diff={result.mean_abs_diff:.6e}",
        (
            f"  mismatches={result.num_mismatches}/{result.total_elements} "
            f"({result.mismatch_fraction:.4%})"
        ),
    ]

    if not result.passed:
        lines.append("  Worst mismatches:")
        for idx, (actual, expected) in zip(
            result.worst_indices[:5], result.worst_values[:5], strict=False
        ):
            lines.append(f"    [{idx}] actual={actual:.6e} expected={expected:.6e}")

    return "\n".join(lines)
