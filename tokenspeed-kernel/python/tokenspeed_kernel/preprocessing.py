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
from typing import Any, Callable

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
    traits: dict[str, frozenset[Any]] = field(default_factory=dict)
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _validate_name("weight preprocessor", self.name)
        _validate_name("weight preprocessor family", self.family)
        _validate_traits(self.traits)
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
    traits: dict[str, frozenset[Any]] | None = None,
    tags: set[str] | None = None,
) -> Callable:
    """Decorator to register a weight preprocessor function."""
    _validate_name("weight preprocessor", name)
    _validate_name("weight preprocessor family", family)
    if traits is not None:
        _validate_traits(traits)

    def decorator(fn: Callable) -> Callable:
        spec = WeightPreprocessorSpec(
            name=name,
            family=family,
            capability=capability or CapabilityRequirement(),
            traits=traits or {},
            tags=frozenset(tags or set()),
        )
        WeightPreprocessorRegistry.get().register(spec, fn)
        return fn

    return decorator


def _validate_name(kind: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} name must be a non-empty string")


def _validate_traits(traits: dict[str, frozenset[Any]]) -> None:
    if not isinstance(traits, dict):
        raise TypeError("weight preprocessor traits must be a dict")
    for trait_name, values in traits.items():
        if not isinstance(trait_name, str) or not trait_name:
            raise ValueError("weight preprocessor trait names must be non-empty strings")
        if not isinstance(values, frozenset):
            raise TypeError(
                f"weight preprocessor trait {trait_name!r} values must be a frozenset"
            )


def _values_overlap(lhs: frozenset[Any], rhs: frozenset[Any]) -> bool:
    return bool(lhs & rhs)


def _validate_trait_compatibility(
    *,
    kernel_spec: KernelSpec,
    preprocessor_spec: WeightPreprocessorSpec,
    errors: list[str],
) -> None:
    shared_traits = set(kernel_spec.traits) & set(preprocessor_spec.traits)
    for trait_name in sorted(shared_traits):
        kernel_values = kernel_spec.traits[trait_name]
        preprocessor_values = preprocessor_spec.traits[trait_name]
        if not _values_overlap(kernel_values, preprocessor_values):
            errors.append(
                f"{kernel_spec.name}: preprocessor {preprocessor_spec.name!r} "
                f"trait {trait_name!r} has values {sorted(preprocessor_values)!r} "
                f"that do not overlap kernel values {sorted(kernel_values)!r}"
            )

    if (
        kernel_spec.family == "moe"
        and kernel_spec.mode == "apply"
        and "weight_dtype" in kernel_spec.traits
        and "weight_dtype" not in preprocessor_spec.traits
    ):
        errors.append(
            f"{kernel_spec.name}: preprocessor {preprocessor_spec.name!r} must "
            "declare weight_dtype for MoE apply validation"
        )


def resolve_weight_preprocessor_ref(
    ref: WeightPreprocessorRef | None,
    *,
    kernel_spec: KernelSpec,
    platform: PlatformInfo,
    registry: WeightPreprocessorRegistry | None = None,
) -> WeightPreprocessorSpec | None:
    """Resolve an exact preprocessor reference for a selected kernel path."""
    if ref is None:
        return None

    registry = registry or WeightPreprocessorRegistry.get()
    spec = registry.get_by_name(ref.name)
    if spec is None:
        raise WeightPreprocessorResolutionError(
            f"Selected kernel references missing weight preprocessor {ref.name!r}"
        )
    errors: list[str] = []
    if spec.family != kernel_spec.family:
        errors.append(
            f"{kernel_spec.name}: preprocessor {spec.name!r} belongs to family "
            f"{spec.family!r}, expected {kernel_spec.family!r}"
        )
    _validate_trait_compatibility(
        kernel_spec=kernel_spec,
        preprocessor_spec=spec,
        errors=errors,
    )
    if errors:
        raise WeightPreprocessorResolutionError("\n".join(errors))
    if not spec.capability.satisfied_by(platform):
        message = (
            f"Weight preprocessor {ref.name!r} cannot run on "
            f"{platform.device_name}"
        )
        if ref.required:
            raise WeightPreprocessorResolutionError(message)
        warnings.warn(
            f"{message}; skipping optional preprocessing",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return spec


def resolve_weight_preprocessor_conflict(
    refs: Iterable[WeightPreprocessorRef | None],
    *,
    scope: str = "module",
) -> WeightPreprocessorRef | None:
    """Resolve module-scoped preprocessing requests.

    ``None`` is a required no-preprocessing constraint. Same-name preprocessors
    are compatible and deduped. Distinct named preprocessors conflict because
    they own the module weight layout as a whole.
    """
    required_no_preprocessing = False
    named: dict[str, bool] = {}

    for ref in refs:
        if ref is None:
            required_no_preprocessing = True
            continue
        named[ref.name] = named.get(ref.name, False) or ref.required

    if not named:
        return None

    if required_no_preprocessing:
        required_named = [name for name, required in named.items() if required]
        if required_named:
            raise WeightPreprocessorConflictError(
                f"Conflicting {scope} preprocessing requirements: required "
                f"no-preprocessing and required preprocessors {sorted(required_named)!r}"
            )
        warnings.warn(
            f"Skipping optional {scope} preprocessors {sorted(named)!r} because "
            "the module also has a no-preprocessing constraint",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if len(named) > 1:
        required_named = [name for name, required in named.items() if required]
        if required_named:
            raise WeightPreprocessorConflictError(
                f"Conflicting {scope} preprocessors {sorted(named)!r}; "
                f"required preprocessors: {sorted(required_named)!r}"
            )
        warnings.warn(
            f"Skipping optional conflicting {scope} preprocessors {sorted(named)!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    name, required = next(iter(named.items()))
    return WeightPreprocessorRef(name=name, required=required)
