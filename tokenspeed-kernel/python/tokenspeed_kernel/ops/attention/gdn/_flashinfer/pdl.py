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

"""PDL launch adapters for FlashInfer CuTe kernels without upstream PDL APIs.

The original device body is inlined into a kernel that waits for predecessors
before any global-memory access and signals launch readiness after computation.
Host launch geometry and runtime ABI stay with FlashInfer. Private namespaces and
fresh caches keep PDL executors separate from the upstream non-PDL executors;
no installed module or process-wide CuTe launch API is modified.
"""

from __future__ import annotations

import functools
import inspect
import types
import typing

import cutlass
import cutlass.cute as cute


def _merge_arguments(dynamic_args, constant_args, positions):
    args = list(dynamic_args)
    for position, value in zip(positions, constant_args):
        args.insert(position, value)
    return tuple(args)


@cute.kernel
def _gdn_pdl_kernel(
    body: cutlass.Constexpr,
    dynamic_args,
    constant_args: cutlass.Constexpr,
    constant_positions: cutlass.Constexpr,
):
    # Wait covers in-place conv/state updates and side inputs. Signal late to
    # avoid scheduling waiting consumers throughout the recurrent computation.
    # All CTAs signal, including those whose body skips a padding row.
    cute.arch.griddepcontrol_wait()
    body(*_merge_arguments(dynamic_args, constant_args, constant_positions))
    cute.arch.griddepcontrol_launch_dependents()


class _PdlLaunch:
    def __init__(self, launcher):
        self.launcher = launcher

    def launch(self, **kwargs):
        kwargs["use_pdl"] = True
        return self.launcher.launch(**kwargs)


class _PdlKernel:
    def __init__(self, kernel):
        original = inspect.unwrap(kernel)
        raw = _clone_function(original, dict(original.__globals__))
        self.body = cute.jit(raw)
        self.signature = inspect.signature(raw)
        annotations = inspect.get_annotations(raw, eval_str=True)
        # A generic tuple would turn Python integers into runtime arguments.
        # Preserve the original constexpr contract (notably shared-memory
        # layouts); an instance method's self is also a compile-time object.
        self.constant_positions = tuple(
            i
            for i, name in enumerate(self.signature.parameters)
            if name == "self"
            or annotations.get(name) is cutlass.Constexpr
            or typing.get_origin(annotations.get(name)) is cutlass.Constexpr
        )

    def __get__(self, instance, owner):
        return self if instance is None else functools.partial(self, instance)

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.kwargs:
            raise TypeError("FlashInfer PDL requires a positional device-kernel ABI")
        args = bound.args
        dynamic_args = tuple(
            value for i, value in enumerate(args) if i not in self.constant_positions
        )
        constant_args = tuple(args[i] for i in self.constant_positions)
        return _PdlLaunch(
            _gdn_pdl_kernel(
                self.body, dynamic_args, constant_args, self.constant_positions
            )
        )


def _clone_function(function, namespace):
    raw = inspect.unwrap(function)
    # CuTe replaces a decorated function's code during its first compilation.
    # Rebind its saved Python code, never preprocess the transformed code twice.
    clone = types.FunctionType(
        getattr(raw, "_original_code", raw.__code__),
        namespace,
        raw.__name__,
        raw.__defaults__,
        raw.__closure__,
    )
    clone.__annotations__ = raw.__annotations__
    clone.__kwdefaults__ = raw.__kwdefaults__
    clone.__qualname__ = raw.__qualname__
    return clone


def _adapt_module(module, *, kernels, launchers, entrypoints, caches, overrides):
    """Bind explicit upstream symbols into a private PDL compilation namespace.

    Missing symbols intentionally fail rather than silently launching a kernel
    without dependency synchronization after an incompatible FlashInfer update.
    """
    namespace = dict(vars(module))
    namespace.update(overrides)
    for name in kernels:
        namespace[name] = _PdlKernel(getattr(module, name))
    for name in caches:
        original = getattr(module, name)
        namespace[name] = (
            {}
            if isinstance(original, dict)
            else functools.cache(_clone_function(original, namespace))
        )
    for name in launchers:
        namespace[name] = cute.jit(_clone_function(getattr(module, name), namespace))
    for name in entrypoints:
        namespace[name] = _clone_function(getattr(module, name), namespace)
    return namespace
