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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable

from tokenspeed_kernel.platform import CapabilityRequirement, PlatformInfo
from tokenspeed_kernel.registry import (
    KernelSpec,
    WeightPreprocessorRef,
)

__all__ = [
    "WeightPreprocessorConflictError",
    "WeightPreprocessorRegistry",
    "WeightPreprocessorResolutionError",
    "WeightPreprocessorSpec",
    "WeightPreprocessorRef",
    "register_weight_preprocessor",
    "resolve_weight_preprocessor_conflict",
    "resolve_weight_preprocessor_ref",
]


@dataclass(frozen=True)
class WeightPreprocessorSpec:
    """Complete specification of a registered weight preprocessor."""

    name: str
    family: str
    capability: CapabilityRequirement = field(default_factory=CapabilityRequirement)
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _validate_name("weight preprocessor", self.name)
        _validate_name("weight preprocessor family", self.family)
        if not isinstance(self.tags, frozenset):
            raise TypeError("weight preprocessor tags must be a frozenset")


class WeightPreprocessorResolutionError(RuntimeError):
    """Raised when a selected preprocessor cannot be resolved or applied."""


class WeightPreprocessorConflictError(RuntimeError):
    """Raised when selected kernels require incompatible module weight layouts."""


class WeightPreprocessorRegistry:
    """Central registry for weight preprocessing implementations."""

    _instance: WeightPreprocessorRegistry | None = None

    def __init__(self) -> None:
        self._by_name: dict[str, WeightPreprocessorSpec] = {}
        self._impls: dict[str, Callable] = {}

    @classmethod
    def get(cls) -> WeightPreprocessorRegistry:
        """Get singleton registry instance."""
        if cls._instance is None:
            cls._instance = WeightPreprocessorRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton registry. Intended for tests."""
        cls._instance = None

    def register(self, spec: WeightPreprocessorSpec, impl: Callable) -> None:
        """Register a weight preprocessor specification and implementation."""
        self._by_name[spec.name] = spec
        self._impls[spec.name] = impl

    def get_by_name(self, name: str) -> WeightPreprocessorSpec | None:
        """Get a specific preprocessor spec by name."""
        return self._by_name.get(name)

    def get_impl(self, name: str) -> Callable | None:
        """Get a preprocessor callable by name."""
        return self._impls.get(name)

    def list_preprocessors(
        self,
        family: str | None = None,
    ) -> list[WeightPreprocessorSpec]:
        """List registered preprocessor specs, optionally filtered by family."""
        specs = list(self._by_name.values())
        if family is not None:
            specs = [spec for spec in specs if spec.family == family]
        return specs


def register_weight_preprocessor(
    family: str,
    *,
    name: str,
    capability: CapabilityRequirement | None = None,
    tags: set[str] | None = None,
) -> Callable:
    """Decorator to register a weight preprocessor function."""
    _validate_name("weight preprocessor", name)
    _validate_name("weight preprocessor family", family)

    def decorator(fn: Callable) -> Callable:
        spec = WeightPreprocessorSpec(
            name=name,
            family=family,
            capability=capability or CapabilityRequirement(),
            tags=frozenset(tags or set()),
        )
        WeightPreprocessorRegistry.get().register(spec, fn)
        return fn

    return decorator


def _validate_name(kind: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} name must be a non-empty string")


def resolve_weight_preprocessor_ref(
    ref: WeightPreprocessorRef | None,
    *,
    kernel_spec: KernelSpec,
    platform: PlatformInfo,
    registry: WeightPreprocessorRegistry | None = None,
) -> WeightPreprocessorSpec | None:
    """Resolve an ordered preprocessor reference for a selected kernel path."""
    if ref is None:
        return None

    registry = registry or WeightPreprocessorRegistry.get()
    errors: list[str] = []
    candidate_specs: list[WeightPreprocessorSpec] = []
    for name in ref.names:
        spec = registry.get_by_name(name)
        if spec is None:
            errors.append(
                f"Selected kernel references missing weight preprocessor {name!r}"
            )
            continue
        if spec.family != kernel_spec.family:
            errors.append(
                f"{kernel_spec.name}: preprocessor {spec.name!r} belongs to family "
                f"{spec.family!r}, expected {kernel_spec.family!r}"
            )
            continue
        candidate_specs.append(spec)
    if errors:
        raise WeightPreprocessorResolutionError("\n".join(errors))

    capability_skips = [
        spec.name
        for spec in candidate_specs
        if not spec.capability.satisfied_by(platform)
    ]
    for spec in candidate_specs:
        if spec.capability.satisfied_by(platform):
            return spec

    skipped = ", ".join(repr(name) for name in capability_skips)
    message = f"Weight preprocessors [{skipped}] cannot run on {platform.device_name}"
    if ref.required:
        raise WeightPreprocessorResolutionError(message)
    warnings.warn(
        "skipping preprocessors due to hardware capabilities: "
        f"[{skipped}]; this may hurt performance",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _ordered_common_preprocessors(
    refs: list[WeightPreprocessorRef],
) -> tuple[str, ...]:
    common = set(refs[0].names)
    for ref in refs[1:]:
        common.intersection_update(ref.names)
    if not common:
        return ()

    def rank(name: str) -> tuple[int, int, tuple[int, ...], str]:
        ranks = tuple(ref.names.index(name) for ref in refs)
        return max(ranks), sum(ranks), ranks, name

    return tuple(sorted(common, key=rank))


def resolve_weight_preprocessor_conflict(
    refs: Iterable[WeightPreprocessorRef | None],
) -> WeightPreprocessorRef | None:
    """Resolve module-scoped preprocessing requests.

    ``None`` is a required no-preprocessing constraint. Named refs are compatible
    when they share at least one candidate preprocessor. The resolved ref names
    the highest-priority shared candidate across the inputs.
    """
    required_no_preprocessing = False
    named_refs: list[WeightPreprocessorRef] = []

    for ref in refs:
        if ref is None:
            required_no_preprocessing = True
            continue
        named_refs.append(ref)

    if not named_refs:
        return None

    if required_no_preprocessing:
        if any(ref.required for ref in named_refs):
            required_names = sorted({name for ref in named_refs for name in ref.names})
            raise WeightPreprocessorConflictError(
                "Conflicting module preprocessing requirements: required "
                f"no-preprocessing and required preprocessors {required_names!r}"
            )
        optional_names = sorted({name for ref in named_refs for name in ref.names})
        warnings.warn(
            "skipping conflicting preprocessors: "
            f"{optional_names!r} conflict with required no preprocessing; "
            "this may hurt performance",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    common_names = _ordered_common_preprocessors(named_refs)
    if not common_names:
        all_names = sorted({name for ref in named_refs for name in ref.names})
        if any(ref.required for ref in named_refs):
            required_names = sorted(
                {name for ref in named_refs if ref.required for name in ref.names}
            )
            raise WeightPreprocessorConflictError(
                f"Conflicting module preprocessors {all_names!r}; "
                f"required preprocessors: {required_names!r}"
            )
        warnings.warn(
            "skipping conflicting preprocessors: "
            f"{all_names!r}; this may hurt performance",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    required = any(ref.required for ref in named_refs)
    return WeightPreprocessorRef(common_names[0], required=required)
