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

"""Report Triton and CuTe DSL JIT compilations that happen while serving.

Every compile-time kernel parameter (``tl.constexpr``, ``gl.constexpr``,
``cutlass.Constexpr``, a static tensor dimension) is part of the kernel's
compile-cache key. A per-batch value passed as one -- a token or row count, a
table width, a sequence length -- compiles a new binary for every new batch
shape, on the forward thread, stalling serving for 100+ ms each time.

The monitor hooks Triton's JIT, which Gluon kernels share, and CuTe DSL's
(:mod:`tokenspeed_kernel.compile_monitor.cute_dsl`), and records every
compilation. Compilations during startup are expected. After
:func:`mark_serving`, each one is logged with its duration and with what
changed against the kernel's earlier specializations. A compile-time parameter
that keeps taking new values from one call site is reported by name once more
than :data:`UNBOUNDED_VALUE_LIMIT` distinct values were first compiled there
while serving, and so is a call site that keeps compiling specializations the
kernel already has -- a compile cache that is missing, or keyed on a value the
kernel takes at runtime. The call site is the first Python frame outside the
JITs, torch and the kernel packages, so the layers of a model that
legitimately launch a kernel with different dimensions count separately.
Powers of two are not counted: rounding up to one is how a kernel buckets a
compile-time bound, and it is log-bounded.

The serving mark is also the package's compile switch, kept whether or not the
monitor is installed: :func:`is_serving` turns true, and a kernel whose library
compiles once per batch shape -- with no compile key this package can bucket --
runs only before then. Afterwards its callers take another
implementation; CUDA graphs captured during startup keep what they recorded.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from tokenspeed_kernel._triton import _IS_NPU, triton

if TYPE_CHECKING:
    from tokenspeed_kernel.compile_monitor.cute_dsl import CuteDslHooks

__all__ = [
    "UNBOUNDED_VALUE_LIMIT",
    "CompileMonitor",
    "CompileStats",
    "ServingJitError",
    "UnboundedSpecializationError",
    "compile_stats",
    "install_compile_monitor",
    "is_serving",
    "mark_serving",
    "uninstall_compile_monitor",
]

logger = logging.getLogger(__name__)

# A compile-time parameter may take this many new non-power-of-two values
# from one call site while serving before it is reported, and a call site may
# recompile this many known specializations. A call site passes one model
# dimension; a per-batch value crosses the limit within the first few distinct
# batch shapes.
UNBOUNDED_VALUE_LIMIT = 4

OnViolation = Literal["warn", "error"]


class ServingJitError(RuntimeError):
    """JIT work on the serving path that recurs, raised under ``"error"``."""


class UnboundedSpecializationError(ServingJitError):
    """A compile-time kernel parameter keeps taking new values while serving."""


@dataclass(frozen=True)
class CompileStats:
    """Cumulative Triton and CuTe DSL compilations, split at :func:`mark_serving`."""

    startup_compiles: int
    startup_seconds: float
    serving_compiles: int
    serving_seconds: float


@dataclass
class _KernelHistory:
    # Rendered specialization of every distinct compiled variant:
    # parameter -> text.
    specializations: list[dict[str, str]] = field(default_factory=list)
    # Every value each numeric compile-time parameter was compiled with.
    values: dict[str, set[int | float]] = field(default_factory=dict)
    # Per (parameter, call site): values first compiled while serving, powers
    # of two excluded.
    serving_values: dict[tuple[str, str], list[int | float]] = field(
        default_factory=dict
    )
    # serving_values count at the last report, so reports repeat on doubling.
    reported: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per call site: compilations while serving of a specialization compiled
    # before, and that count at the last report.
    serving_repeats: dict[str, int] = field(default_factory=dict)
    reported_repeats: dict[str, int] = field(default_factory=dict)


def _is_power_of_two(value: int | float) -> bool:
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _param_name(names: list[str], path: tuple[int, ...]) -> str:
    return names[path[0]] + "".join(f"[{index}]" for index in path[1:])


class CompileMonitor:
    """Bookkeeping behind the JIT hooks; usable without a JIT for tests.

    Args:
        on_violation: ``"warn"`` logs JIT work that recurs while serving -- a
            parameter that keeps taking new values, a call site that keeps
            recompiling known specializations, a direct ``@cute.jit`` call;
            ``"error"`` raises :class:`ServingJitError` (for a parameter,
            :class:`UnboundedSpecializationError`) from the compiling launch.
        value_limit: How many new non-power-of-two values one compile-time
            parameter may take, and how many known specializations a kernel
            may recompile, from one call site while serving before it is
            reported.
    """

    def __init__(self, on_violation: OnViolation, value_limit: int) -> None:
        if on_violation not in ("warn", "error"):
            raise ValueError(
                f"on_violation must be 'warn' or 'error', got {on_violation!r}"
            )
        self.on_violation: OnViolation = on_violation
        self.value_limit = value_limit
        self.serving = False
        self._lock = threading.Lock()
        self._kernels: dict[str, _KernelHistory] = {}
        # (kernel, call site) of every direct @cute.jit call reported.
        self._direct_calls: set[tuple[str, str]] = set()
        self._startup_compiles = 0
        self._startup_seconds = 0.0
        self._serving_compiles = 0
        self._serving_seconds = 0.0

    def stats(self) -> CompileStats:
        with self._lock:
            return CompileStats(
                startup_compiles=self._startup_compiles,
                startup_seconds=self._startup_seconds,
                serving_compiles=self._serving_compiles,
                serving_seconds=self._serving_seconds,
            )

    def record(
        self,
        jit: str,
        kernel: str,
        specialization: dict[str, str],
        constexprs: dict[str, Any],
        seconds: float,
        site: str,
    ) -> None:
        """Record one compilation and report it if it happened while serving.

        Args:
            jit: The compiling JIT, as logged (``"Triton"``, ``"CuTe DSL"``).
            kernel: Qualified kernel name.
            specialization: Rendered compile-cache key, parameter -> text, for
                describing what a new compilation changed.
            constexprs: Values of the parameters the kernel declares
                compile-time, flattened for tuple parameters.
            seconds: Wall time of the compilation.
            site: The launching call site, ``file:line (function)``.
        """
        unbounded: list[tuple[str, list[int | float]]] = []
        repeats = 0
        with self._lock:
            history = self._kernels.setdefault(kernel, _KernelHistory())
            change = _describe_change(history.specializations, specialization)
            repeated = specialization in history.specializations
            if not repeated:
                history.specializations.append(specialization)
            serving = self.serving
            if serving:
                self._serving_compiles += 1
                self._serving_seconds += seconds
            else:
                self._startup_compiles += 1
                self._startup_seconds += seconds
            if serving and repeated:
                count = history.serving_repeats.get(site, 0) + 1
                history.serving_repeats[site] = count
                reported = history.reported_repeats.get(site, 0)
                if count > max(self.value_limit, 2 * reported):
                    history.reported_repeats[site] = count
                    repeats = count
            for name, value in constexprs.items():
                if not _numeric(value):
                    continue
                seen = history.values.setdefault(name, set())
                if value in seen:
                    continue
                seen.add(value)
                if not serving or _is_power_of_two(value):
                    continue
                new_values = history.serving_values.setdefault((name, site), [])
                new_values.append(value)
                reported = history.reported.get((name, site), 0)
                if len(new_values) > max(self.value_limit, 2 * reported):
                    history.reported[(name, site)] = len(new_values)
                    unbounded.append((name, list(new_values)))
        if not serving:
            return
        logger.info(
            f"{jit} JIT compiled {kernel} while serving in "
            f"{seconds * 1e3:.0f} ms ({change}) from {site}"
        )
        for name, values in unbounded:
            shown = ", ".join(str(value) for value in values[:12])
            more = ", ..." if len(values) > 12 else ""
            message = (
                f"{kernel}: compile-time parameter {name} has compiled "
                f"{len(values)} new values while serving from {site} "
                f"({shown}{more}). Each "
                "new value is a JIT compilation on the forward thread; a value "
                "that varies per batch must be a runtime argument, or be "
                "bucketed to a power of two where the kernel needs a "
                "compile-time bound."
            )
            if self.on_violation == "error":
                raise UnboundedSpecializationError(message)
            logger.warning(message)
        if repeats:
            self._report(
                f"{kernel} has compiled specializations it already had "
                f"{repeats} times while serving from {site}. Each is a JIT "
                "compilation on the forward thread; the caller's compile cache "
                "is missing or keyed on a value the kernel takes at runtime."
            )

    def record_direct_call(self, kernel: str, seconds: float, site: str) -> None:
        """Report a ``@cute.jit`` function called directly while serving.

        A direct call traces the function in Python on every call, even when
        CuTe DSL has its binary cached. Each call site is reported once.

        Args:
            kernel: Qualified name of the ``@cute.jit`` function.
            seconds: Wall time of the call's trace, and compilation if any.
            site: The calling site, ``file:line (function)``.
        """
        with self._lock:
            if not self.serving or (kernel, site) in self._direct_calls:
                return
            self._direct_calls.add((kernel, site))
        self._report(
            f"{kernel} was called directly as a @cute.jit function while "
            f"serving from {site}, taking {seconds * 1e3:.0f} ms. A direct call "
            "traces the kernel in Python on every call; compile it with "
            "cute.compile during startup and call the compiled function."
        )

    def _report(self, message: str) -> None:
        if self.on_violation == "error":
            raise ServingJitError(message)
        logger.warning(message)


def _describe_change(
    previous: list[dict[str, str]], specialization: dict[str, str]
) -> str:
    """Name what distinguishes a compilation from its nearest earlier one."""
    if not previous:
        return "first specialization of this kernel"

    def differing(other: dict[str, str]) -> list[str]:
        return [
            name
            for name in specialization.keys() | other.keys()
            if specialization.get(name) != other.get(name)
        ]

    nearest = min(previous, key=lambda other: len(differing(other)))
    changes = [
        f"{name}: {nearest.get(name, '-')} -> {specialization.get(name, '-')}"
        for name in sorted(differing(nearest))
    ]
    return "; ".join(changes) if changes else "same specialization recompiled"


# Triton hook payload entries that key the compile cache besides the signature:
# the launch options and the device.
_KEYED_OPTIONS = (
    "num_warps",
    "num_ctas",
    "num_stages",
    "enable_fp_fusion",
    "launch_cooperative_grid",
    "device",
)


def _render(kind: str, value: Any, attrs: list[list[Any]]) -> str:
    """Render one parameter's part of the compile-cache key."""
    if kind == "constexpr":
        return repr(value)
    flags = "".join(f" {name.removeprefix('tt.')}={setting}" for name, setting in attrs)
    return f"{kind}{flags}"


def _specialization(fn: Any, compile_info: dict[str, Any]) -> tuple[
    dict[str, str],
    dict[str, Any],
]:
    """Split a hook payload into its rendered key and declared constexprs."""
    params = fn.jit_function.params
    names = [param.name for param in params]
    constants = compile_info["constants"] or {}
    attrs = compile_info["configs"][0] if compile_info["configs"] else {}
    rendered: dict[str, str] = {}
    for name, kind in compile_info["signature"].items():
        index = names.index(name)
        rendered[name] = _render(kind, constants.get((index,)), attrs.get((index,), []))
    for option in _KEYED_OPTIONS:
        rendered[option] = repr(compile_info[option])
    constexprs = {
        _param_name(names, path): value
        for path, value in constants.items()
        if params[path[0]].is_constexpr
    }
    # A tuple-valued compile-time parameter arrives as one value; count each
    # element as its own parameter so a varying element is named.
    for name, value in list(constexprs.items()):
        if isinstance(value, tuple):
            del constexprs[name]
            for index, element in enumerate(value):
                constexprs[f"{name}[{index}]"] = element
    return rendered, constexprs


def _internal_dirs() -> tuple[str, ...]:
    dirs = [
        os.path.dirname(module.__file__) + os.sep
        for module in (triton, torch, sys.modules["tokenspeed_kernel"])
    ]
    # AMD Gluon kernels launch from tokenspeed_kernel_amd, MLA CuTe DSL kernels
    # from tokenspeed_mla, and CuTe DSL compiles in cutlass; locate them
    # without importing them where they are absent or unused.
    for package in ("tokenspeed_kernel_amd", "tokenspeed_mla", "cutlass"):
        spec = importlib.util.find_spec(package)
        if spec is not None and spec.submodule_search_locations:
            dirs.extend(path + os.sep for path in spec.submodule_search_locations)
    return tuple(dirs)


# Frames under these directories are the JIT and the kernel wrappers; the
# call site worth naming is the first frame outside them.
_INTERNAL_DIRS = _internal_dirs()


def _call_site() -> str:
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_INTERNAL_DIRS):
        frame = frame.f_back
    if frame is None:
        return "<unknown>"
    path = frame.f_code.co_filename.split(os.sep)
    return f"{os.sep.join(path[-3:])}:{frame.f_lineno} ({frame.f_code.co_name})"


class _Hooks:
    """Triton's pre- and post-compile hooks, chained onto earlier ones."""

    def __init__(self, monitor: CompileMonitor) -> None:
        self.monitor = monitor
        self.previous_cache_hook = triton.knobs.runtime.jit_cache_hook
        self.previous_post_compile_hook = triton.knobs.runtime.jit_post_compile_hook
        self._started: dict[tuple[int, int, str], float] = {}

    def _timer_key(self, fn: Any, key: str) -> tuple[int, int, str]:
        return threading.get_ident(), id(fn.jit_function), key

    def cache_hook(self, **kwargs: Any) -> bool | None:
        if self.previous_cache_hook is not None:
            skip = self.previous_cache_hook(**kwargs)
            if skip:
                return skip
        self._started[self._timer_key(kwargs["fn"], kwargs["key"])] = (
            time.perf_counter()
        )
        return None

    def post_compile_hook(self, **kwargs: Any) -> bool | None:
        result = None
        if self.previous_post_compile_hook is not None:
            result = self.previous_post_compile_hook(**kwargs)
        fn = kwargs["fn"]
        started = self._started.pop(self._timer_key(fn, kwargs["key"]), None)
        seconds = 0.0 if started is None else time.perf_counter() - started
        specialization, constexprs = _specialization(fn, kwargs["compile"])
        self.monitor.record(
            "Triton",
            f"{fn.module}.{fn.name}",
            specialization,
            constexprs,
            seconds,
            _call_site(),
        )
        return result


_hooks: _Hooks | None = None
# Set while monitoring where CuTe DSL is installed and its JIT entry is hookable.
_cute_dsl_hooks: CuteDslHooks | None = None
_serving = False


def install_compile_monitor(on_violation: OnViolation) -> None:
    """Start recording every Triton and CuTe DSL compilation in this process.

    Args:
        on_violation: ``"warn"`` or ``"error"``; see :class:`CompileMonitor`.

    Chains onto hooks installed earlier. Installing again replaces the
    monitor, which resets its history. Where CuTe DSL is not installed only
    Triton is monitored. A no-op on NPU, whose Triton has no JIT hooks.
    """
    global _hooks, _cute_dsl_hooks
    if _IS_NPU:
        logger.info("Triton compile monitor is not available on NPU")
        return
    uninstall_compile_monitor()
    _hooks = _Hooks(CompileMonitor(on_violation, UNBOUNDED_VALUE_LIMIT))
    triton.knobs.runtime.jit_cache_hook = _hooks.cache_hook
    triton.knobs.runtime.jit_post_compile_hook = _hooks.post_compile_hook
    if importlib.util.find_spec("cutlass") is None:
        return
    try:
        # Imported here: it imports this package, and cutlass, which only
        # NVIDIA platforms install.
        from tokenspeed_kernel.compile_monitor.cute_dsl import install_cute_dsl_hooks
    except ImportError as error:
        logger.warning(f"CuTe DSL compilations are not monitored: {error}")
        return
    _cute_dsl_hooks = install_cute_dsl_hooks(_hooks.monitor)


def uninstall_compile_monitor() -> None:
    """Remove the monitor and restore the hooks it chained onto."""
    global _hooks, _cute_dsl_hooks
    if _cute_dsl_hooks is not None:
        _cute_dsl_hooks.uninstall()
        _cute_dsl_hooks = None
    if _hooks is None:
        return
    triton.knobs.runtime.jit_cache_hook = _hooks.previous_cache_hook
    triton.knobs.runtime.jit_post_compile_hook = _hooks.previous_post_compile_hook
    _hooks = None


def mark_serving() -> None:
    """Mark the end of startup: close the compile switch and report later compilations."""
    global _serving
    _serving = True
    if _hooks is not None:
        _hooks.monitor.serving = True


def is_serving() -> bool:
    """Whether startup has ended, after which no kernel may compile per batch shape."""
    return _serving


def compile_stats() -> CompileStats | None:
    """Cumulative compilations, or ``None`` when the monitor is not installed."""
    return None if _hooks is None else _hooks.monitor.stats()
