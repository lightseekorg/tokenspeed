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

from collections.abc import Collection

from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import KernelSpec


def _capabilities_overlap(a: CapabilityRequirement, b: CapabilityRequirement) -> bool:
    if a.vendors and b.vendors:
        vendors = a.vendors & b.vendors
        if not vendors:
            return False
    else:
        vendors = a.vendors or b.vendors or {""}
    for vendor in vendors:
        minima = [
            v
            for v in (a.min_arch_version_for(vendor), b.min_arch_version_for(vendor))
            if v is not None
        ]
        maxima = [v for v in (a.max_arch_version, b.max_arch_version) if v is not None]
        if not minima or not maxima or max(minima) <= min(maxima):
            return True
    return False


def _extent(values: Collection[int]) -> tuple[int, int]:
    # Ranges in registrations can cover large domains; never materialize them.
    if isinstance(values, range):
        return min(values[0], values[-1]), max(values[0], values[-1])
    return min(values), max(values)


def _dimension_may_overlap(a: KernelSpec, b: KernelSpec, dim: str) -> bool:
    exact = []
    minima = []
    maxima = []
    alignments = []
    for spec in (a, b):
        values = spec.traits.get(dim)
        if values is not None:
            exact.append(values)
        for suffix, target in (
            ("_min", minima),
            ("_max", maxima),
            ("_align", alignments),
        ):
            values = spec.traits.get(f"{dim}{suffix}")
            if values is not None:
                target.append(values)
        if dim in {"ispp", "hidden"}:
            values = spec.traits.get(f"{dim}_alignment")
            if values is not None:
                alignments.append(values)
    groups = exact + minima + maxima + alignments
    if any(not values for values in groups):
        return False
    if any(
        not isinstance(values, range) and not all(isinstance(v, int) for v in values)
        for values in groups
    ):
        return True
    if any(_extent(values)[0] <= 0 for values in alignments):
        return True
    lower = [_extent(values)[0] for values in exact + minima]
    upper = [_extent(values)[1] for values in exact + maxima]
    low = max(lower) if lower else None
    high = min(upper) if upper else None
    if low is not None and high is not None and low > high:
        return False
    finite = [values for values in exact if not isinstance(values, range)]
    if finite:
        return any(
            (low is None or value >= low)
            and (high is None or value <= high)
            and all(value in values for values in exact)
            and all(
                any(value % alignment == 0 for alignment in values)
                for values in alignments
            )
            for value in min(finite, key=len)
        )
    # Unresolved range steps and alignment intersections remain possible overlaps.
    return True


def domains_may_overlap(a: KernelSpec, b: KernelSpec) -> bool:
    """Conservatively diagnose overlap without executing kernels or predicates.

    Only a proven empty intersection suppresses a warning. Layouts can be
    ignored by selection, positive feature requirements can coexist, and
    arbitrary predicates cannot be evaluated safely during discovery.
    """
    if not _capabilities_overlap(a.capability, b.capability):
        return False
    if not any(
        left.roles == right.roles
        for left in a.format_signatures
        for right in b.format_signatures
    ):
        return False
    # These exact dimensions and all bound dimensions require a request value.
    # Other traits can be omitted by callers, even if their value sets differ.
    dimensions = {"batch", "m", "n", "k"}
    for spec in (a, b):
        dimensions.update({"ispp", "hidden"} & spec.traits.keys())
        for name in spec.traits:
            for suffix in ("_min", "_max", "_align"):
                if name.endswith(suffix):
                    dimensions.add(name[: -len(suffix)])
    return all(_dimension_may_overlap(a, b, dim) for dim in dimensions)
