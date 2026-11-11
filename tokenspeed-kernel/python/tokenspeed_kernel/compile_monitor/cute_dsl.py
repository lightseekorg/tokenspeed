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

"""Feed CuTe DSL compilations and direct ``@cute.jit`` calls to the monitor.

``cute.compile`` traces and compiles a ``@cute.jit`` function every time it is
called; callers keep the compiled function in their own cache, so a
compilation while serving is a miss of that cache. Calling a ``@cute.jit``
function directly traces it in Python on every call, even when CuTe DSL's
in-memory cache holds its binary, and that cache keys on every Python scalar
argument, which the generated function's name spells out.

Both enter cutlass-dsl's ``BaseDSL.generate_mlir``: ``cute.compile`` with
``compile_only`` set, a direct call without it. A ``@cute.jit`` function called
inside a trace is inlined and does not enter it again. The hooks wrap it and
``BaseDSL.compile_and_cache``, which runs only when a binary is compiled (the
DSL subclasses that override it call the base). Both are private to
cutlass-dsl, so installing checks what the hooks read and leaves CuTe DSL
unmonitored, with a warning, if a cutlass-dsl upgrade changed it.
"""

from __future__ import annotations

import enum
import functools
import inspect
import logging
import threading
import time
from typing import Any, get_origin

import torch
from cutlass import Constexpr, Numeric
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute import Pointer, Tensor
from tokenspeed_kernel.compile_monitor import CompileMonitor, _call_site, _numeric

__all__ = ["CuteDslHooks", "install_cute_dsl_hooks"]

logger = logging.getLogger(__name__)

# The ``BaseDSL.generate_mlir`` parameters the hooks read.
_GENERATE_MLIR_PARAMETERS = frozenset(
    {"funcBody", "args", "kwonlyargs", "sig", "compile_only"}
)


class _ThreadState(threading.local):
    def __init__(self) -> None:
        # Whether compile_and_cache ran within this thread's current
        # generate_mlir call.
        self.compiled = False


class CuteDslHooks:
    """``BaseDSL.generate_mlir`` and ``compile_and_cache``, wrapped to feed a monitor.

    Args:
        monitor: Receives every compilation and every direct ``@cute.jit``
            call.
    """

    def __init__(self, monitor: CompileMonitor) -> None:
        self.monitor = monitor
        self.previous_generate_mlir = BaseDSL.generate_mlir
        self.previous_compile_and_cache = BaseDSL.compile_and_cache
        self._signature = inspect.signature(self.previous_generate_mlir)
        self._state = _ThreadState()

    def install(self) -> None:
        hooks = self

        @functools.wraps(self.previous_generate_mlir)
        def generate_mlir(dsl: BaseDSL, *args: Any, **kwargs: Any) -> Any:
            return hooks._generate_mlir(dsl, args, kwargs)

        @functools.wraps(self.previous_compile_and_cache)
        def compile_and_cache(dsl: BaseDSL, *args: Any, **kwargs: Any) -> Any:
            hooks._state.compiled = True
            return hooks.previous_compile_and_cache(dsl, *args, **kwargs)

        BaseDSL.generate_mlir = generate_mlir
        BaseDSL.compile_and_cache = compile_and_cache

    def uninstall(self) -> None:
        BaseDSL.generate_mlir = self.previous_generate_mlir
        BaseDSL.compile_and_cache = self.previous_compile_and_cache

    def _generate_mlir(
        self, dsl: BaseDSL, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        state = self._state
        outer = state.compiled
        state.compiled = False
        started = time.perf_counter()
        try:
            result = self.previous_generate_mlir(dsl, *args, **kwargs)
            compiled = state.compiled
        finally:
            state.compiled = outer
        seconds = time.perf_counter() - started
        bound = self._signature.bind(dsl, *args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        direct = not call["compile_only"]
        if not (compiled or direct):
            return result
        func = call["funcBody"]
        kernel = f"{func.__module__}.{func.__qualname__}"
        site = _call_site()
        if compiled:
            specialization, constexprs = _specialization(
                func, call["sig"], call["args"], call["kwonlyargs"], direct
            )
            self.monitor.record(
                "CuTe DSL", kernel, specialization, constexprs, seconds, site
            )
        if direct:
            self.monitor.record_direct_call(kernel, seconds, site)
        return result


def install_cute_dsl_hooks(monitor: CompileMonitor) -> CuteDslHooks | None:
    """Feed ``monitor`` every CuTe DSL compilation and direct ``@cute.jit`` call.

    Args:
        monitor: The process's compile monitor.

    Returns:
        The installed hooks, or ``None`` when this cutlass-dsl changed the JIT
        entry the hooks read.
    """
    unsupported = _unsupported_change()
    if unsupported is not None:
        logger.warning(
            f"CuTe DSL compilations are not monitored: {unsupported} in this "
            "cutlass-dsl; update tokenspeed_kernel.compile_monitor.cute_dsl"
        )
        return None
    hooks = CuteDslHooks(monitor)
    hooks.install()
    return hooks


def _unsupported_change() -> str | None:
    """What of the JIT entry the hooks read this cutlass-dsl changed, if anything."""
    methods = vars(BaseDSL)
    for name in ("generate_mlir", "compile_and_cache"):
        if name not in methods:
            return f"BaseDSL.{name} is gone"
    missing = (
        _GENERATE_MLIR_PARAMETERS
        - inspect.signature(methods["generate_mlir"]).parameters.keys()
    )
    if missing:
        return f"BaseDSL.generate_mlir lost {', '.join(sorted(missing))}"
    subclasses = list(BaseDSL.__subclasses__())
    while subclasses:
        subclass = subclasses.pop()
        if "generate_mlir" in vars(subclass):
            return f"{subclass.__name__} overrides BaseDSL.generate_mlir"
        subclasses.extend(subclass.__subclasses__())
    return None


def _specialization(
    func: Any,
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwonlyargs: dict[str, Any],
    direct: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Split a traced call into its rendered key and compile-time numbers.

    A tensor's dtype and static layout, a ``Constexpr`` argument, the kernel
    object's (``self``) attributes and closure variables are compile-time for
    both entries. ``cute.compile`` takes any other scalar as a runtime
    argument; a direct call keys on it too.
    """
    params = sig.parameters
    positional = [
        name
        for name, param in params.items()
        if param.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    rendered: dict[str, str] = {}
    constexprs: dict[str, Any] = {}
    for index, (name, value) in enumerate(
        (*zip(positional, args), *kwonlyargs.items())
    ):
        if index == 0 and name == "self":
            rendered[name] = type(value).__qualname__
            _add_constants("self.", _attributes(value), rendered, constexprs)
            continue
        compile_time = direct or _is_constexpr(params[name].annotation)
        rendered[name] = _render(value, compile_time)
        _collect(name, value, compile_time, constexprs)
    cells = {}
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            cells[name] = cell.cell_contents
        except ValueError:  # A closure variable not assigned yet.
            continue
    _add_constants("", cells, rendered, constexprs)
    return rendered, constexprs


def _is_constexpr(annotation: Any) -> bool:
    return (
        isinstance(annotation, type) and issubclass(annotation, Constexpr)
    ) or get_origin(annotation) is Constexpr


def _attributes(value: Any) -> dict[str, Any]:
    try:
        return vars(value)
    except TypeError:  # No instance dictionary.
        return {}


def _add_constants(
    prefix: str,
    values: dict[str, Any],
    rendered: dict[str, str],
    constexprs: dict[str, Any],
) -> None:
    """Add the plain constants among ``values``; a trace bakes them in."""
    for name, value in values.items():
        text = _constant(value)
        if text is not None:
            rendered[prefix + name] = text
            _collect(prefix + name, value, True, constexprs)


def _constant(value: Any) -> str | None:
    """Render a plain Python constant, or ``None`` for any other object."""
    if value is None or isinstance(value, (bool, int, float, str, enum.Enum)):
        return repr(value)
    if isinstance(value, type):
        return value.__qualname__
    if isinstance(value, (tuple, list)):
        texts = [_constant(element) for element in value]
        if None in texts:
            return None
        return f"({', '.join(texts)})"
    return None


def _render(value: Any, compile_time: bool) -> str:
    """Render one argument's part of the compile-cache key."""
    if isinstance(value, Tensor):
        shape = ",".join(
            "?" if dynamic else str(dim)
            for dim, dynamic in zip(value.shape, value.dynamic_shapes_mask)
        )
        stride = ",".join(
            "?" if dynamic else str(step)
            for step, dynamic in zip(value.stride, value.dynamic_strides_mask)
        )
        return (
            f"{value.element_type.__name__}({shape}):({stride})@{value.memspace.name}"
        )
    if isinstance(value, Pointer):
        return f"Ptr<{value.dtype.__name__}@{value.memspace.name}>"
    if isinstance(value, torch.Tensor):
        # CuTe DSL marks a torch tensor's layout dynamic but for unit strides.
        shape = ",".join("?" for _ in value.shape)
        stride = ",".join("1" if step == 1 else "?" for step in value.stride())
        return f"{value.dtype}({shape}):({stride})"
    if isinstance(value, Numeric):
        return type(value).__name__
    if isinstance(value, (tuple, list)):
        return f"({', '.join(_render(element, compile_time) for element in value)})"
    text = _constant(value)
    if text is None:
        return type(value).__qualname__
    if not compile_time and isinstance(value, (bool, int, float)):
        return type(value).__name__
    return text


def _collect(
    name: str, value: Any, compile_time: bool, constexprs: dict[str, Any]
) -> None:
    """Add the compile-time numbers in ``value`` to ``constexprs``."""
    if isinstance(value, Tensor):
        for axis, (dim, dynamic) in enumerate(
            zip(value.shape, value.dynamic_shapes_mask)
        ):
            if not dynamic:
                constexprs[f"{name}.shape[{axis}]"] = dim
        for axis, (step, dynamic) in enumerate(
            zip(value.stride, value.dynamic_strides_mask)
        ):
            if not dynamic:
                constexprs[f"{name}.stride[{axis}]"] = step
    elif isinstance(value, (tuple, list)):
        for index, element in enumerate(value):
            _collect(f"{name}[{index}]", element, compile_time, constexprs)
    elif compile_time and _numeric(value):
        constexprs[name] = value
