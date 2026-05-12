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
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

import torch
from tokenspeed_kernel.platform import PlatformInfo, current_platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec

logger = logging.getLogger(__name__)

__all__ = [
    "NoKernelFoundError",
    "SelectedKernel",
    "SelectionObjective",
    "SelectionStrategy",
    "ScoreBreakdown",
    "SelectionOracle",
    "AutotuneParams",
    "SelectionPolicy",
    "select_kernel",
    "set_selection_policy",
    "register_oracle",
    "kernel_override",
    "load_config_overrides",
    "clear_config_overrides",
    "explain_selection",
    "spec_matches_traits",
    "ref_compatible_with_spec",
    "spec_matches_shape_traits",
    "warmup_selection",
]


class NoKernelFoundError(RuntimeError):
    """Raised when no kernel matches the requested operation."""

    pass


class SelectedKernel:
    """Result of kernel selection — a callable that also carries the kernel name."""

    __slots__ = ("name", "impl")

    def __init__(self, name: str, impl: Callable) -> None:
        self.name = name
        self.impl = impl

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.impl(*args, **kwargs)

    def __repr__(self) -> str:
        return f"SelectedKernel(name={self.name!r})"


class SelectionObjective(Enum):
    """Objectives are a closed enum — they directly control scoring logic,
    so an unknown value would silently fall through with no effect.
    """

    DEFAULT = "default"  # Balanced heuristic (priority-weighted)
    LATENCY = "latency"  # Minimize per-call latency
    THROUGHPUT = "throughput"  # Maximize tokens/second for large batches
    PORTABILITY = "portability"  # Prefer solutions that work across vendors (Triton)
    DETERMINISM = "determinism"  # Prefer bit-reproducible implementations
    DEBUG = "debug"  # Prefer readable implementations (reference, Triton)


class SelectionStrategy(Enum):
    HEURISTIC = "heuristic"  # Score-based ranking (default, instant)
    AUTOTUNE = "autotune"  # Benchmark candidates, pick fastest


@dataclass
class ScoreBreakdown:
    """Per-kernel scoring breakdown across all dimensions.

    Ranking is lexicographic on ``(oracle, objective, priority)`` — the oracle's
    per-family knowledge wins first, then objective alignment, with the kernel's
    declared priority band as the final tiebreaker.
    """

    priority: int  # [0, 20) — from KernelSpec.priority
    objective: int  # 0 or 1 — 1 if the kernel matches the requested objective
    oracle: int  # [0, 20) — per-family oracle adjustment

    def sort_key(self) -> tuple[int, int, int]:
        """Lex sort key (descending — higher is better)."""
        return (self.oracle, self.objective, self.priority)

    def __str__(self) -> str:
        return f"ora={self.oracle} obj={self.objective} pri={self.priority}"


class SelectionOracle:
    """Base class for per-family selection adjustments.

    Return a score in [0, 20). 10 = neutral. Higher = better fit.
    """

    def adjust(
        self,
        spec: KernelSpec,
        platform: PlatformInfo,
        traits: dict[str, Any] | None,
    ) -> int:
        return 10


@dataclass
class AutotuneParams:
    """Tuning knobs for autotune strategy."""

    warmup_iters: int = 3
    bench_iters: int = 10
    use_cuda_events: bool = True


@dataclass
class SelectionPolicy:
    """Per-op selection strategy configuration."""

    # Default strategy for all ops
    default_strategy: SelectionStrategy = SelectionStrategy.HEURISTIC

    # Per-op overrides: (family, mode) -> strategy
    op_strategies: dict[tuple[str, str], SelectionStrategy] = field(
        default_factory=dict
    )

    # Autotune parameters (used when strategy is AUTOTUNE)
    autotune_params: AutotuneParams = field(default_factory=AutotuneParams)

    def get_strategy(self, family: str, mode: str) -> SelectionStrategy:
        return self.op_strategies.get((family, mode), self.default_strategy)


@dataclass
class _ConfigOverrideEntry:
    """A single override entry parsed from overrides.yaml."""

    name: str | None = None  # Exact kernel name
    solution: str | None = None  # Solution backend to match
    objective: str | None = None  # SelectionObjective value string


_policy = SelectionPolicy()
_oracles: dict[str, SelectionOracle] = {}
_global_overrides: dict[tuple[str, str], str] = {}
_config_overrides: dict[tuple[str, str], _ConfigOverrideEntry] | None = None


def set_selection_policy(policy: SelectionPolicy) -> None:
    """Set per-op selection strategy. Clears all cached selections."""
    global _policy
    _policy = policy
    KernelRegistry.get().clear_cache()


def register_oracle(family: str, oracle: SelectionOracle) -> None:
    """Register a per-family selection oracle."""
    _oracles[family] = oracle


def _get_oracle(family: str) -> SelectionOracle | None:
    return _oracles.get(family)


def _parse_overrides(
    raw: dict,
) -> dict[tuple[str, str], _ConfigOverrideEntry]:
    """Parse the ``overrides`` section of the YAML config.

    Accepted formats::

        overrides:
          attention.decode:
            solution: flashinfer
          gemm.mm:
            name: gluon_gemm_mm_fp8
          moe.experts:
            objective: determinism
          norm.rmsnorm: triton_rmsnorm   # shorthand: treated as kernel name
    """
    result: dict[tuple[str, str], _ConfigOverrideEntry] = {}
    if not isinstance(raw, dict):
        return result

    for op_key, entry in raw.items():
        parts = str(op_key).split(".", 1)
        if len(parts) != 2:
            logger.warning("Invalid override key '%s' (expected 'family.mode')", op_key)
            continue
        family, mode = parts

        if isinstance(entry, str):
            result[(family, mode)] = _ConfigOverrideEntry(name=entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            solution = entry.get("solution")
            objective = entry.get("objective")
            result[(family, mode)] = _ConfigOverrideEntry(
                name=str(name) if name else None,
                solution=str(solution) if solution else None,
                objective=str(objective) if objective else None,
            )
        else:
            logger.warning("Invalid override value for '%s': %r", op_key, entry)

    return result


def load_config_overrides(path: str | os.PathLike[str] | None = None) -> None:
    """Load kernel overrides from a YAML config file.

    Args:
        path: Path to the YAML file.  If *None*, uses the
            ``TOKENSPEED_KERNEL_OVERRIDES_FILE`` env var or falls back to
            ``~/.config/tokenspeed-kernel/overrides.yaml``.
    """
    global _config_overrides

    if path is None:
        env_path = os.environ.get("TOKENSPEED_KERNEL_OVERRIDES_FILE")
        if env_path:
            path = Path(env_path)
        else:
            path = Path("~/.config/tokenspeed-kernel/overrides.yaml").expanduser()
    else:
        path = Path(path)

    _config_overrides = {}

    if not path.exists():
        return

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "PyYAML not installed; cannot load overrides from %s. "
            "Install with: pip install pyyaml",
            path,
        )
        return

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.warning("Failed to load overrides from %s", path, exc_info=True)
        return

    if not isinstance(data, dict):
        return

    _config_overrides = _parse_overrides(data.get("overrides", {}))

    if _config_overrides:
        logger.debug(
            "Loaded %d config override(s) from %s",
            len(_config_overrides),
            path,
        )
        KernelRegistry.get().clear_cache()


def clear_config_overrides() -> None:
    """Clear loaded config overrides.

    After this call no config-file overrides are active.  Call
    :func:`load_config_overrides` again to reload from a file.
    """
    global _config_overrides
    _config_overrides = {}


def _get_config_override(family: str, mode: str) -> _ConfigOverrideEntry | None:
    """Return the config-file override for *(family, mode)*, lazily loading
    from the default path on first access."""
    global _config_overrides
    if _config_overrides is None:
        load_config_overrides()
    return _config_overrides.get((family, mode))  # type: ignore[union-attr]


def _make_cache_key(
    family: str,
    mode: str,
    dtype: object,
    arch: str,
    objective: SelectionObjective,
    features: frozenset[str] | None,
    traits: dict[str, Any] | None,
    solution: str | None = None,
) -> tuple:
    """Build a hashable cache key including selection-relevant traits."""
    traits_key = tuple(sorted(traits.items())) if traits else ()
    mods_key = frozenset(features) if features else frozenset()
    return (family, mode, dtype, arch, objective, mods_key, traits_key, solution)


def _score_priority(spec: KernelSpec) -> int:
    """Priority dimension: kernel's inherent quality/maturity."""
    return max(0, min(19, spec.priority))


_OBJECTIVE_TAG: dict[SelectionObjective, str] = {
    SelectionObjective.LATENCY: "latency",
    SelectionObjective.THROUGHPUT: "throughput",
    SelectionObjective.PORTABILITY: "portability",
    SelectionObjective.DETERMINISM: "determinism",
    SelectionObjective.DEBUG: "determinism",
}


def _score_objective(spec: KernelSpec, objective: SelectionObjective) -> int:
    """Objective dimension: 1 if the kernel declares the matching tag, else 0.

    DEFAULT returns 0 so every kernel ties on this dimension and ranking
    falls through to oracle/priority.
    """
    tag = _OBJECTIVE_TAG.get(objective)
    return 1 if tag is not None and tag in spec.tags else 0


def _score_oracle(
    spec: KernelSpec,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
) -> int:
    """Oracle dimension: per-family domain-specific scoring."""
    oracle = _get_oracle(spec.family)
    if oracle is None:
        return 10  # Neutral when no oracle is registered
    score = oracle.adjust(spec, platform, traits)
    return max(0, min(19, score))


def _score(
    spec: KernelSpec,
    objective: SelectionObjective,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
) -> ScoreBreakdown:
    """Score a kernel across all ranking dimensions."""
    return ScoreBreakdown(
        priority=_score_priority(spec),
        objective=_score_objective(spec, objective),
        oracle=_score_oracle(spec, platform, traits),
    )


def _rank_by_objective(
    specs: list[KernelSpec],
    objective: SelectionObjective,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
) -> list[tuple[KernelSpec, ScoreBreakdown]]:
    """Rank kernels lexicographically by (oracle, objective, priority).

    Higher is better. Oracle wins first because per-family oracles encode the
    most domain knowledge; objective alignment breaks ties next; the kernel's
    declared priority band is the final tiebreaker.
    """
    scored = [(spec, _score(spec, objective, platform, traits)) for spec in specs]
    scored.sort(key=lambda x: x[1].sort_key(), reverse=True)
    return scored


def _trait_value_matches(spec_values: frozenset[Any], trait_value: Any) -> bool:
    if not isinstance(trait_value, (set, frozenset)):
        trait_value = frozenset({trait_value})
    return trait_value.issubset(spec_values)


def spec_matches_traits(
    spec: KernelSpec,
    traits: dict[str, Any],
    *,
    require_all_traits: bool = False,
) -> bool:
    """Return whether a spec's declared traits match the requested traits.

    Args:
        spec: Registered kernel specification to test.
        traits: Trait requirements. Values may be concrete scalars (for example,
            ``{"head_dim": 128}``) or sets/frozensets of allowed values.
        require_all_traits: When ``False`` (selection behavior), unknown traits on
            the spec are ignored. When ``True`` (reference compatibility checks),
            every requested trait must be explicitly present on the spec.
    """
    for trait_name, trait_value in traits.items():
        spec_values = spec.traits.get(trait_name)
        if spec_values is None:
            if require_all_traits:
                return False
            continue
        if not _trait_value_matches(spec_values, trait_value):
            return False
    return True


def ref_compatible_with_spec(ref: KernelSpec, spec: KernelSpec) -> bool:
    """Return whether a reference kernel can handle the same inputs as a test kernel.

    For each trait the reference declares, the spec must declare that same trait
    with values that fully cover the reference's required values.  Traits the
    reference does not declare are unconstrained (the reference is general with
    respect to those traits).
    """
    for trait_name, ref_values in ref.traits.items():
        spec_values = spec.traits.get(trait_name)
        if spec_values is None:
            return False
        if not ref_values.issubset(spec_values):
            return False
    return True


def spec_matches_shape_traits(spec: KernelSpec, shape: dict[str, Any]) -> bool:
    """Return whether a spec's alignment traits match a concrete shape."""
    alignment_traits: dict[str, tuple[str, int]] = {
        "n_align_16": ("N", 16),
        "n_align_64": ("N", 64),
        "n_align_128": ("N", 128),
        "k_align_16": ("K", 16),
        "k_align_128": ("K", 128),
    }

    for trait_name, (dim_name, alignment) in alignment_traits.items():
        values = spec.traits.get(trait_name)
        if values is None or True not in values:
            continue

        dim = shape.get(dim_name)
        if isinstance(dim, int) and dim % alignment != 0:
            return False

    return True


def _filter_by_traits(
    specs: list[KernelSpec],
    traits: dict[str, Any],
) -> list[KernelSpec]:
    """Filter kernels by op-specific trait compatibility."""
    return [spec for spec in specs if spec_matches_traits(spec, traits)]


def _resolve_override(
    registry: KernelRegistry,
    family: str,
    mode: str,
    dtype: object,
    override: str,
    platform: PlatformInfo,
) -> SelectedKernel:
    impl = registry.get_impl(override)
    if impl is not None:
        return SelectedKernel(name=override, impl=impl)

    specs = registry.get_for_operator(family, mode, solution=override)
    if specs:
        kernel_name = specs[0].name
        impl = registry.get_impl(kernel_name)
        if impl is not None:
            return SelectedKernel(name=kernel_name, impl=impl)

    raise NoKernelFoundError(
        f"Override '{override}' not found for {family}.{mode} ({dtype})"
    )


def _log_selection(
    family: str,
    mode: str,
    dtype: object,
    winner: KernelSpec,
    scored: list[tuple[KernelSpec, ScoreBreakdown]],
    platform: PlatformInfo,
    objective: SelectionObjective,
) -> None:
    """Log selection result if verbose mode is enabled."""
    if not os.environ.get("TOKENSPEED_KERNEL_VERBOSE"):
        return

    breakdown = next((s for spec, s in scored if spec.name == winner.name), None)
    if breakdown:
        logger.info(
            "[tokenspeed_kernel] %s.%s(%s) -> %s (%s, %s)",
            family,
            mode,
            dtype,
            winner.name,
            breakdown,
            platform.arch,
        )
    else:
        logger.info(
            "[tokenspeed_kernel] %s.%s(%s) -> %s (%s)",
            family,
            mode,
            dtype,
            winner.name,
            platform.arch,
        )


def select_kernel(
    family: str,
    mode: str,
    dtype: torch.dtype,
    *,
    features: frozenset[str] | None = None,
    platform: PlatformInfo | None = None,
    objective: SelectionObjective = SelectionObjective.DEFAULT,
    traits: dict[str, Any] | None = None,
    solution: str | None = None,
    override: str | None = None,
    expected_kernel_name: str | None = None,
) -> SelectedKernel:
    """Select the best kernel for an operation.

    On first call for a given (family, mode, dtype, platform, objective, traits,
    solution) combination, runs the full selection pipeline. Subsequent calls
    with the same arguments return the cached result — a single dict lookup.

    Args:
        family: Operator family (e.g., "attention")
        mode: Operator mode (e.g., "decode")
        dtype: Primary data type
        features: Required operator features (e.g., {"paged"})
        platform: Hardware to match (auto-detected if None)
        objective: Selection objective (see SelectionObjective enum)
        traits: Op-specific trait values that affect kernel applicability
               (e.g., {"head_dim": 128, "num_kv_heads": 8})
        solution: Restrict selection to a registered solution while preserving
            normal platform, dtype, and trait filtering.
        override: Force a specific kernel name or solution string
        expected_kernel_name: Debug-only hint. When set, a warning is
            logged if the selected kernel differs from this name. The
            selected kernel is still used regardless of the mismatch.

    Returns:
        A :class:`SelectedKernel` that is directly callable and also
        exposes the winning kernel's ``name``.
    """
    platform = platform or current_platform()

    # --- Override resolution (lowest → highest priority) ---

    # 1. Config file (lowest priority)
    config_entry = _get_config_override(family, mode)
    if config_entry is not None:
        if override is None and solution is None:
            if config_entry.name:
                override = config_entry.name
            elif config_entry.solution:
                solution = config_entry.solution
        if config_entry.objective and objective == SelectionObjective.DEFAULT:
            try:
                objective = SelectionObjective(config_entry.objective)
            except ValueError:
                logger.warning(
                    "Invalid objective '%s' in overrides config for %s.%s",
                    config_entry.objective,
                    family,
                    mode,
                )

    # 2. Context-manager global overrides
    global_override = _global_overrides.get((family, mode))
    if global_override:
        override = global_override

    # 3. Environment variable
    env_key = f"TOKENSPEED_KERNEL_OVERRIDE_{family.upper()}_{mode.upper()}"
    env_override = os.environ.get(env_key)
    if env_override:
        override = env_override
    registry = KernelRegistry.get()

    # Fast path: check cache (skipped when override is active)
    cache_key = _make_cache_key(
        family, mode, dtype, platform.arch, objective, features, traits, solution
    )
    if override is None:
        cached = registry.cache_get(cache_key)
        if cached is not None:
            return cached

    if override:
        return _resolve_override(registry, family, mode, dtype, override, platform)

    # Get candidates (same filtering for both strategies)
    candidates = registry.get_for_operator(
        family,
        mode,
        features=features,
        platform=platform,
        dtype=dtype,
        solution=solution,
    )

    solution_clause = f" with solution {solution!r}" if solution else ""
    if not candidates:
        raise NoKernelFoundError(
            f"No kernel found for {family}.{mode} ({dtype})"
            f"{solution_clause} on {platform.device_name}"
        )

    if traits:
        candidates = _filter_by_traits(candidates, traits)

    if not candidates:
        raise NoKernelFoundError(
            f"No kernel found for {family}.{mode} ({dtype})"
            f"{solution_clause} with traits {traits} on {platform.device_name}"
        )

    # Strategy dispatch
    strategy = _policy.get_strategy(family, mode)

    if strategy == SelectionStrategy.AUTOTUNE:
        winner, scored = _autotune_select(
            candidates,
            family,
            mode,
            dtype,
            platform,
            traits,
            _policy.autotune_params,
        )
    else:
        scored = _rank_by_objective(candidates, objective, platform, traits)
        winner = scored[0][0]

    _log_selection(family, mode, dtype, winner, scored, platform, objective)

    if expected_kernel_name and winner.name != expected_kernel_name:
        logger.warning(
            "[tokenspeed_kernel] select_kernel(%s.%s, %s) chose '%s' but "
            "expected '%s'. Score breakdown — selected: %s",
            family,
            mode,
            dtype,
            winner.name,
            expected_kernel_name,
            next((s for sp, s in scored if sp.name == winner.name), "N/A"),
        )

    impl = registry.get_impl(winner.name)
    result = SelectedKernel(name=winner.name, impl=impl)
    registry.cache_put(cache_key, result)
    return result


def _autotune_select(
    candidates: list[KernelSpec],
    family: str,
    mode: str,
    dtype: object,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
    params: AutotuneParams,
) -> tuple[KernelSpec, list[tuple[KernelSpec, ScoreBreakdown]]]:
    """Benchmark candidates and return the fastest.

    Falls back to heuristic ranking when the autotuning infrastructure
    (input generators, benchmark runner) is not yet available.
    """
    scored = _rank_by_objective(
        candidates, SelectionObjective.DEFAULT, platform, traits
    )
    winner = scored[0][0]
    logger.debug(
        "[tokenspeed_kernel:autotune] falling back to heuristic for %s.%s(%s)",
        family,
        mode,
        dtype,
    )
    return winner, scored


@contextmanager
def kernel_override(
    family: str, mode: str, kernel_name: str
) -> Generator[None, None, None]:
    """Context manager for scoped kernel override."""
    key = (family, mode)
    old = _global_overrides.get(key)
    _global_overrides[key] = kernel_name
    try:
        yield
    finally:
        if old is None:
            _global_overrides.pop(key, None)
        else:
            _global_overrides[key] = old


def explain_selection(
    family: str,
    mode: str,
    dtype: torch.dtype,
    *,
    features: frozenset[str] | None = None,
    platform: PlatformInfo | None = None,
    objective: SelectionObjective = SelectionObjective.DEFAULT,
    traits: dict[str, Any] | None = None,
    solution: str | None = None,
) -> str:
    """Return a human-readable explanation of kernel selection.

    Example output::

        Op: attention.decode (bfloat16)
        Platform: NVIDIA H100 (sm_90)
        Objective: default
        Ranking: lex (oracle, objective, priority); higher wins

        Candidates (3 matched, 5 registered):
          1. flashinfer_decode  [SELECTED]
             ora=16 obj=1 pri=14
          2. triton_decode
             ora=10 obj=0 pri=10

        Filtered out:
          - aiter_decode: vendor mismatch (requires amd)
    """
    platform = platform or current_platform()
    registry = KernelRegistry.get()

    all_specs = registry.list_kernels(family=family, mode=mode)
    candidates = registry.get_for_operator(
        family,
        mode,
        features=features,
        platform=platform,
        dtype=dtype,
        solution=solution,
    )

    if traits:
        candidates = _filter_by_traits(candidates, traits)

    scored = _rank_by_objective(candidates, objective, platform, traits)

    filtered_names = {s.name for s in candidates}
    filtered_out = [s for s in all_specs if s.name not in filtered_names]

    lines = [
        f"Op: {family}.{mode} ({dtype})",
        f"Platform: {platform.device_name} ({platform.arch})",
        f"Solution: {solution or 'any'}",
        f"Objective: {objective.value}",
        "Ranking: lex (oracle, objective, priority); higher wins",
        "",
        f"Candidates ({len(scored)} matched, {len(all_specs)} registered):",
    ]

    for i, (spec, breakdown) in enumerate(scored):
        marker = "  [SELECTED]" if i == 0 else ""
        lines.append(f"  {i + 1}. {spec.name}{marker}")
        lines.append(f"     {breakdown}")

    if filtered_out:
        lines.append("")
        lines.append("Filtered out:")
        for spec in filtered_out:
            reasons: list[str] = []
            if (
                spec.capability.vendors
                and platform.vendor not in spec.capability.vendors
            ):
                reasons.append(
                    f"vendor mismatch (requires "
                    f"{', '.join(spec.capability.vendors)})"
                )
            missing = spec.capability.missing_features(platform)
            if missing:
                reasons.append(f"missing features: {', '.join(missing)}")
            if spec.capability.min_arch_version:
                if not (platform.arch_version >= spec.capability.min_arch_version):
                    reasons.append(
                        f"arch mismatch (requires "
                        f"{spec.capability.min_arch_version})"
                    )
            if dtype and dtype not in spec.dtypes:
                reasons.append(
                    f"dtype mismatch (supports "
                    f"{', '.join(str(d) for d in spec.dtypes)})"
                )
            if solution and spec.solution != solution:
                reasons.append(f"solution mismatch (is {spec.solution!r})")
            reason_str = "; ".join(reasons) if reasons else "unknown"
            lines.append(f"  - {spec.name}: {reason_str}")

    return "\n".join(lines)


def warmup_selection(
    ops: list[tuple[str, str, torch.dtype, dict | None]] | None = None,
) -> None:
    """Pre-resolve kernel selection for the given ops (or all registered ops).

    Call this at model init to front-load both heuristic and autotune costs.
    After warmup, every select_kernel() call on the hot path is a cache hit.
    """

    if ops is None:
        ops = [
            (f, m, torch.bfloat16, None)
            for f, m in KernelRegistry.get().list_operators()
        ]

    for family, mode, dtype, traits in ops:
        try:
            select_kernel(family, mode, dtype, traits=traits)
        except NoKernelFoundError:
            logger.debug(
                "[tokenspeed_kernel] warmup: no kernel for %s.%s(%s)",
                family,
                mode,
                dtype,
            )
