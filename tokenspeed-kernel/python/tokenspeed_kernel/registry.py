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

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    import torch

    from tokenspeed_kernel.selection import SelectedKernel

from tokenspeed_kernel.platform import CapabilityRequirement, PlatformInfo
from tokenspeed_kernel.signature import FormatSignature

logger = logging.getLogger(__name__)

__all__ = [
    "KernelSpec",
    "KernelRegistry",
    "Priority",
    "load_builtin_kernels",
    "register_kernel",
    "describe_kernel",
]


def _normalize_roles(roles: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(roles, str):
        role_names = (roles,)
    else:
        role_names = tuple(roles)
    if not role_names:
        raise ValueError("at least one dtype filter role is required")
    return role_names


# Hard upper bound on priority values; selection scoring clamps to this range.
_PRIORITY_MAX = 20


class Priority(IntEnum):
    """Selection-priority bands for registered kernels.

    Priority is the tiebreaker among kernels that already match the request's
    capability and trait gates. A kernel that does not satisfy the platform's
    capability requirement is filtered out before priority is consulted.

    Bands group kernels by their portability/performance contract so that an
    out-of-tree plugin author can predict what offset they need to win without
    auditing every in-tree registration. Use band members directly, or add a
    small intra-band offset for relative preference within the band:

        priority=Priority.PERFORMANT       # band start (8)
        priority=Priority.PERFORMANT + 2   # +2 within the band (10)

    Band layout (each occupies a contiguous range of ints in [0, 20).
    ints in 1..3 are unused — that range previously held a separate FALLBACK
    band that was folded into PORTABLE):

    +--------------+--------+----------------------------------------------------+
    | Band         | Range  | When to use                                        |
    +==============+========+====================================================+
    | REFERENCE    |    0   | Correctness reference. Never auto-selected when a  |
    |              |        | real implementation is available; useful as a      |
    |              |        | numeric ground truth in tests.                     |
    +--------------+--------+----------------------------------------------------+
    | PORTABLE     |  4..7  | In-tree generic implementation with no arch or     |
    |              |        | shape gating beyond the family contract — e.g.     |
    |              |        | default Triton, or PyTorch reference patsh used as |
    |              |        | last-resort coverage.                              |
    +--------------+--------+----------------------------------------------------+
    | PERFORMANT   | 8..11  | In-tree generally optimized kernel, covering a     |
    |              |        | broad arch range — e.g. optimizied Triton for      |
    |              |        | Hopper+. The default winner on supported vendor.   |
    +--------------+--------+----------------------------------------------------+
    | SPECIALIZED  | 12..15 | In-tree highly optimized kernel, narrowly gated on |
    |              |        | arch + shape, e.g., Gluon/CuTe DSL fp8 attention   |
    |              |        | for Blackwell with specific head_dim.              |
    +--------------+--------+----------------------------------------------------+
    | PLUGIN       | 16..19 | Reserved for out-of-tree plugins to override the   |
    |              |        | in-tree default. In-tree kernels should not use    |
    |              |        | this band so plugins always have headroom.         |
    +--------------+--------+----------------------------------------------------+

    Notes:

    * Keep offsets within the band's width — band+offset returns a plain int,
      so crossing into the next band cannot be flagged.
    * Pick the lowest band that fits. Inflating a kernel's band to win on one
      platform makes it win everywhere it isn't actually specialized.
    """

    REFERENCE = 0
    PORTABLE = 4
    PERFORMANT = 8
    SPECIALIZED = 12
    PLUGIN = 16


def _band_for(value: int) -> Priority:
    """Return the band that contains ``value`` (the largest band start ≤ value)."""
    return max((b for b in Priority if int(b) <= value), key=int)


def _validate_priority(value: int | Priority) -> int:
    """Validate a priority value and return it as a plain ``int``.

    Accepts a :class:`Priority` band, a band plus offset (e.g.
    ``Priority.PERFORMANT + 2``), or a raw int. Raises ``ValueError`` if the
    final value falls outside ``[0, 20)``.
    """
    ivalue = int(value)
    if not 0 <= ivalue < _PRIORITY_MAX:
        bands = ", ".join(f"{b.name}={int(b)}" for b in Priority)
        raise ValueError(
            f"priority must be in [0, {_PRIORITY_MAX}), got {ivalue}. "
            f"Use a Priority band ({bands}) optionally with a small +offset."
        )
    return ivalue


@dataclass(frozen=True)
class KernelSpec:
    """Complete specification of a registered kernel."""

    # Identity
    name: str  # Unique name, e.g., "flashinfer_decode_sm90"
    family: str  # "attention", "gemm", "moe", etc.
    mode: str  # "decode", "mm", "experts", etc.
    features: frozenset[str] = (
        frozenset()
    )  # Orthogonal features, e.g., {"paged", "mla"}
    solution: str = ""  # "triton", "flashinfer", "cutlass", "reference", etc.

    # Capabilities
    capability: CapabilityRequirement = field(default_factory=CapabilityRequirement)
    format_signatures: frozenset[FormatSignature] = frozenset()
    # Op-specific traits, e.g. {"head_dim": frozenset({64, 128, 256}), "persistent": frozenset({True})}
    traits: dict[str, frozenset[Any]] = field(default_factory=dict)

    # Selection metadata
    # Higher = preferred. See :class:`Priority` for the band layout. The default
    # places unannotated kernels in PERFORMANT so they win against PORTABLE but
    # lose to SPECIALIZED. Selection scoring clamps out-of-range values.
    priority: int = int(Priority.PERFORMANT) + 2
    tags: frozenset[str] = (
        frozenset()
    )  # Standard tags: "throughput", "latency", "determinism", "portability"

    def supports_format_signature(self, format_signature: FormatSignature) -> bool:
        return format_signature in self.format_signatures

    def format_signatures_for_storage_dtype(
        self,
        storage_dtype: torch.dtype,
        roles: str | Iterable[str],
    ) -> tuple[FormatSignature, ...]:
        """Return signatures whose selected role has storage_dtype.

        ``roles`` is explicit because the meaningful dtype role is an operator
        property, not a property of ``FormatSignature`` itself. Multiple roles
        are treated as alternatives, which is useful for operators whose dtype
        filter role depends on the concrete signature.
        """
        role_names = _normalize_roles(roles)
        return tuple(
            signature
            for signature in sorted(self.format_signatures, key=str)
            if any(
                signature.storage_dtype_for(role) == storage_dtype
                for role in role_names
            )
        )

    def format_signature_for_storage_dtype(
        self,
        storage_dtype: torch.dtype,
        roles: str | Iterable[str],
    ) -> FormatSignature | None:
        """Return the single matching signature, or raise if ambiguous."""
        matches = self.format_signatures_for_storage_dtype(storage_dtype, roles)
        if len(matches) > 1:
            role_list = ", ".join(_normalize_roles(roles)) or "none"
            raise ValueError(
                f"Kernel {self.name!r} has multiple format signatures for "
                f"storage dtype={storage_dtype} on role(s) {role_list}; "
                "use a full format signature"
            )
        return matches[0] if matches else None

    def storage_dtypes_for_role(
        self,
        roles: str | Iterable[str],
    ) -> frozenset[torch.dtype]:
        role_names = _normalize_roles(roles)
        return frozenset(
            dtype
            for dtype in (
                signature.storage_dtype_for(role)
                for signature in self.format_signatures
                for role in role_names
            )
            if dtype is not None
        )


class KernelRegistry:
    """Central registry for all kernel implementations."""

    _instance: KernelRegistry | None = None

    def __init__(self) -> None:
        self._by_operator: dict[tuple[str, str], list[KernelSpec]] = defaultdict(list)
        self._by_name: dict[str, KernelSpec] = {}
        self._impls: dict[str, Callable] = {}  # name -> callable
        self._selection_cache: dict[tuple, SelectedKernel] = {}

    @classmethod
    def get(cls) -> KernelRegistry:
        """Get singleton registry instance."""
        if cls._instance is None:
            cls._instance = KernelRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ---- Registration ----

    def register(self, spec: KernelSpec, impl: Callable) -> None:
        """Register a kernel specification and its implementation."""
        if spec.name in self._by_name:
            # Allow re-registration (plugin override)
            self._unregister(spec.name)

        self._by_name[spec.name] = spec
        self._impls[spec.name] = impl
        key = (spec.family, spec.mode)
        self._by_operator[key].append(spec)
        self._by_operator[key].sort(key=lambda s: s.priority, reverse=True)
        self._invalidate_cache(key)

    def _unregister(self, name: str) -> None:
        if name not in self._by_name:
            return
        spec = self._by_name.pop(name)
        self._impls.pop(name, None)
        key = (spec.family, spec.mode)
        self._by_operator[key] = [s for s in self._by_operator[key] if s.name != name]
        self._invalidate_cache(key)

    # ---- Queries ----

    def get_by_name(self, name: str) -> KernelSpec | None:
        """Get a specific kernel spec by name."""
        return self._by_name.get(name)

    def get_impl(self, name: str) -> Callable | None:
        """Get a kernel's callable implementation by name."""
        return self._impls.get(name)

    def get_for_operator(
        self,
        family: str,
        mode: str,
        *,
        features: frozenset[str] | None = None,
        platform: PlatformInfo | None = None,
        format_signature: FormatSignature | None = None,
        tags: set[str] | None = None,
        solution: str | None = None,
    ) -> list[KernelSpec]:
        """Get all kernels for an operator, optionally filtered."""
        specs = list(self._by_operator.get((family, mode), []))

        if features is not None:
            specs = [s for s in specs if features.issubset(s.features)]
        if platform:
            specs = [s for s in specs if s.capability.satisfied_by(platform)]
        if format_signature:
            specs = [s for s in specs if s.supports_format_signature(format_signature)]
        if tags:
            specs = [s for s in specs if tags.issubset(s.tags)]
        if solution:
            specs = [s for s in specs if s.solution == solution]

        return specs

    def list_operators(self) -> list[tuple[str, str]]:
        """List all registered (family, mode) pairs."""
        return list(self._by_operator.keys())

    def list_kernels(
        self,
        family: str | None = None,
        mode: str | None = None,
    ) -> list[KernelSpec]:
        """List registered kernel specs, optionally filtered."""
        if family and mode:
            return list(self._by_operator.get((family, mode), []))
        specs = list(self._by_name.values())
        if family:
            specs = [s for s in specs if s.family == family]
        if mode:
            specs = [s for s in specs if s.mode == mode]
        return specs

    def list_solutions(self, family: str, mode: str) -> list[str]:
        """List available solutions for an operator."""
        return list({s.solution for s in self._by_operator.get((family, mode), [])})

    # ---- Cache management ----

    def cache_get(self, key: tuple) -> SelectedKernel | None:
        """Look up a cached selection result."""
        return self._selection_cache.get(key)

    def cache_put(self, key: tuple, selected_kernel: SelectedKernel) -> None:
        """Store a selection result in the cache."""
        self._selection_cache[key] = selected_kernel

    def _invalidate_cache(self, key: tuple[str, str]) -> None:
        self._selection_cache = {
            k: v for k, v in self._selection_cache.items() if k[:2] != key
        }

    def clear_cache(self) -> None:
        self._selection_cache.clear()


def register_kernel(
    family: str,
    mode: str,
    *,
    name: str | None = None,
    features: set[str] | None = None,
    solution: str,
    capability: CapabilityRequirement | None = None,
    signatures: set[FormatSignature] | frozenset[FormatSignature],
    traits: dict[str, frozenset[Any]] | None = None,
    priority: Priority | int = Priority.PERFORMANT + 2,
    tags: set[str] | None = None,
) -> Callable:
    """Decorator to register a kernel function.

    ``priority`` accepts a :class:`Priority` band (recommended) or a raw ``int``
    in ``[0, 20)``. Within a band, add a small offset for relative preference,
    e.g. ``Priority.SPECIALIZED + 2``. See :class:`Priority` for the meaning of
    each band and how to choose between them.

    Example::

        from tokenspeed_kernel.signature import format_signatures

        @register_kernel(
            "attention", "decode",
            features={"paged"},
            solution="triton",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(10, 0),
                required_features=frozenset({"tensor_core:f8"}),
            ),
            signatures=format_signatures(
                ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
            ),
            # Narrowly gated on SM100 + tcgen05 → SPECIALIZED band.
            priority=Priority.SPECIALIZED + 1,
            tags={"latency", "determinism"},
        )
        def triton_decode_attention(query, key_cache, value_cache, ...):
            ...
    """
    priority_int = _validate_priority(priority)

    def decorator(fn: Callable) -> Callable:
        kernel_name = name or f"{solution}_{family}_{mode}"

        spec = KernelSpec(
            name=kernel_name,
            family=family,
            mode=mode,
            solution=solution,
            features=frozenset(features or set()),
            format_signatures=frozenset(signatures),
            capability=capability or CapabilityRequirement(),
            traits=traits or {},
            priority=priority_int,
            tags=frozenset(tags or set()),
        )

        KernelRegistry.get().register(spec, fn)
        return fn

    return decorator


def describe_kernel(name: str) -> str:
    """Generate human-readable description of a kernel."""
    registry = KernelRegistry.get()
    spec = registry.get_by_name(name)
    if not spec:
        return f"Kernel '{name}' not found"

    band = _band_for(spec.priority)
    offset = spec.priority - int(band)
    band_str = band.name if offset == 0 else f"{band.name}+{offset}"
    lines = [
        f"Kernel: {spec.name}",
        f"  Operator: {spec.family}.{spec.mode}",
        f"  Solution: {spec.solution}",
        f"  Priority: {spec.priority} ({band_str})",
        "  Format signatures: "
        + ("; ".join(str(p) for p in spec.format_signatures) or "none"),
        f"  Platform: {spec.capability}",
        f"  Tags: {', '.join(spec.tags) or 'none'}",
    ]
    return "\n".join(lines)


def load_builtin_kernels() -> None:
    from tokenspeed_kernel.registrations import amd, nvidia, portable

    portable.load()
    nvidia.load()
    amd.load()


def error_fn(*args, **kwargs):
    """A placeholder function when kernel is not properly imported or registered."""
    raise RuntimeError("Kernel implementation not found")


class ErrorClass:
    """A placeholder class when kernel implementation is not properly imported or registered."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Kernel implementation not found")
