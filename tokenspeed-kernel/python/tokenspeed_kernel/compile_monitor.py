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

"""Report Triton JIT compilations that happen while serving.

Every compile-time kernel parameter (``tl.constexpr``, ``gl.constexpr``) is
part of the kernel's compile-cache key. A per-batch value passed as one -- a
token or row count, a table width, a sequence length -- compiles a new binary
for every new batch shape, on the forward thread, stalling serving for 100+ ms
each time.

The monitor hooks Triton's JIT, which Gluon kernels share, and records every
compilation. Compilations during startup are expected. After
:func:`mark_serving`, each one is logged with its duration and with what
changed against the kernel's earlier specializations, and a compile-time
parameter that keeps taking new values from one call site is reported by name
once more than :data:`UNBOUNDED_VALUE_LIMIT` distinct values were first
compiled there while serving. The call site is the first Python frame outside
Triton, torch and this package, so the layers of a model that legitimately
launch a kernel with different dimensions count separately. Powers of two are
not counted: rounding up to one is how a kernel buckets a compile-time bound,
and it is log-bounded.

The serving mark is also the package's compile switch, kept whether or not the
monitor is installed: :func:`is_serving` turns true, and a kernel whose library
compiles once per batch shape -- outside Triton, so no key can be bucketed for
it -- runs only before then. Afterwards its callers take another
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
from typing import Any, Literal

import torch
from tokenspeed_kernel._triton import _IS_NPU, triton

__all__ = [
    "UNBOUNDED_VALUE_LIMIT",
    "CompileMonitor",
    "CompileStats",
    "UnboundedSpecializationError",
    "compile_stats",
    "install_compile_monitor",
    "is_serving",
    "mark_serving",
    "uninstall_compile_monitor",
]

logger = logging.getLogger(__name__)

# A compile-time parameter may take this many new non-power-of-two values
# from one call site while serving before it is reported. A call site passes
# one model dimension; a per-batch value crosses the limit within the first
# few distinct batch shapes.
UNBOUNDED_VALUE_LIMIT = 4

OnUnbounded = Literal["warn", "error"]


class UnboundedSpecializationError(RuntimeError):
    """A compile-time kernel parameter keeps taking new values while serving."""


@dataclass(frozen=True)
class CompileStats:
    """Cumulative Triton compilations, split at :func:`mark_serving`."""

    startup_compiles: int
    startup_seconds: float
    serving_compiles: int
    serving_seconds: float


@dataclass
class _KernelHistory:
    # Rendered specialization of every compiled variant: parameter -> text.
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


def _is_power_of_two(value: int | float) -> bool:
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _param_name(names: list[str], path: tuple[int, ...]) -> str:
    return names[path[0]] + "".join(f"[{index}]" for index in path[1:])


class CompileMonitor:
    """Bookkeeping behind the JIT hooks; usable without Triton for tests.

    Args:
        on_unbounded: ``"warn"`` logs a parameter that keeps taking new values
            while serving; ``"error"`` raises
            :class:`UnboundedSpecializationError` from the compiling launch.
        value_limit: How many new non-power-of-two values one compile-time
            parameter may take from one call site while serving before it is
            reported.
    """

    def __init__(self, on_unbounded: OnUnbounded, value_limit: int) -> None:
        if on_unbounded not in ("warn", "error"):
            raise ValueError(
                f"on_unbounded must be 'warn' or 'error', got {on_unbounded!r}"
            )
        self.on_unbounded: OnUnbounded = on_unbounded
        self.value_limit = value_limit
        self.serving = False
        self._lock = threading.Lock()
        self._kernels: dict[str, _KernelHistory] = {}
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
        kernel: str,
        specialization: dict[str, str],
        constexprs: dict[str, Any],
        seconds: float,
        site: str,
    ) -> None:
        """Record one compilation and report it if it happened while serving.

        Args:
            kernel: Qualified kernel name.
            specialization: Rendered compile-cache key, parameter -> text, for
                describing what a new compilation changed.
            constexprs: Values of the parameters the kernel declares
                compile-time, flattened for tuple parameters.
            seconds: Wall time of the compilation.
            site: The launching call site, ``file:line (function)``.
        """
        unbounded: list[tuple[str, list[int | float]]] = []
        with self._lock:
            history = self._kernels.setdefault(kernel, _KernelHistory())
            change = _describe_change(history.specializations, specialization)
            history.specializations.append(specialization)
            serving = self.serving
            if serving:
                self._serving_compiles += 1
                self._serving_seconds += seconds
            else:
                self._startup_compiles += 1
                self._startup_seconds += seconds
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
            f"Triton JIT compiled {kernel} while serving in "
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
            if self.on_unbounded == "error":
                raise UnboundedSpecializationError(message)
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
    return "; ".join(changes) if changes else "same key recompiled"


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
        for module in (triton, torch, sys.modules[__package__])
    ]
    # AMD Gluon kernels launch from tokenspeed_kernel_amd; locate it without
    # importing it on other platforms.
    amd = importlib.util.find_spec("tokenspeed_kernel_amd")
    if amd is not None and amd.submodule_search_locations:
        dirs.extend(path + os.sep for path in amd.submodule_search_locations)
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
            f"{fn.module}.{fn.name}", specialization, constexprs, seconds, _call_site()
        )
        return result


_hooks: _Hooks | None = None
_serving = False


def install_compile_monitor(on_unbounded: OnUnbounded) -> None:
    """Start recording every Triton compilation in this process.

    Args:
        on_unbounded: ``"warn"`` or ``"error"``; see :class:`CompileMonitor`.

    Chains onto hooks installed earlier. Installing again replaces the
    monitor, which resets its history. A no-op on NPU, whose Triton has no
    JIT hooks.
    """
    global _hooks
    if _IS_NPU:
        logger.info("Triton compile monitor is not available on NPU")
        return
    uninstall_compile_monitor()
    _hooks = _Hooks(CompileMonitor(on_unbounded, UNBOUNDED_VALUE_LIMIT))
    triton.knobs.runtime.jit_cache_hook = _hooks.cache_hook
    triton.knobs.runtime.jit_post_compile_hook = _hooks.post_compile_hook


def uninstall_compile_monitor() -> None:
    """Remove the monitor and restore the hooks it chained onto."""
    global _hooks
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
