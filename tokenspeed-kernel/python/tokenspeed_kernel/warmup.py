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

"""Operator-owned finite Triton warmups, using the ordinary launch arguments."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import Any

import torch
from tokenspeed_kernel._triton import _IS_NPU
from tokenspeed_kernel.ops.tuning import is_warming_up, kernel_warmup


@dataclass(frozen=True)
class TensorVariant:
    """A pointer dtype and alignment class to compile without reading data."""

    dtype: torch.dtype
    aligned: bool


_compile_only: ContextVar[bool] = ContextVar("triton_compile_only", default=False)


@cache
def _pointer_probe(
    dtype: torch.dtype, aligned: bool, device: torch.device
) -> torch.Tensor:
    storage = torch.empty(8, dtype=dtype, device=device)
    offset = 0 if aligned else 1
    return storage[offset : offset + 1]


@dataclass(frozen=True)
class _WarmupHook:
    kernel: Any
    variants: (
        tuple[Mapping[str, tuple[Any, ...]], ...]
        | Callable[[Mapping[str, Any]], tuple[Mapping[str, tuple[Any, ...]], ...]]
    )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        names = self.kernel.arg_names
        if len(args) > len(names):
            raise TypeError(f"Too many arguments to {self.kernel.fn.__name__}")
        bound = dict(zip(names, args))
        duplicate = bound.keys() & kwargs.keys()
        if duplicate:
            raise TypeError(f"Arguments supplied twice: {sorted(duplicate)}")
        bound.update(kwargs)
        tensor = next(
            (value for value in bound.values() if isinstance(value, torch.Tensor)), None
        )
        variants = self.variants(bound) if callable(self.variants) else self.variants
        for choices in variants:
            if any(not values for values in choices.values()):
                raise ValueError("Warmup argument choices must not be empty")
            for values in product(*choices.values()):
                call = dict(bound)
                for name, value in zip(choices, values, strict=True):
                    if isinstance(value, TensorVariant):
                        if tensor is None:
                            raise ValueError("Pointer warmup needs a tensor device")
                        value = _pointer_probe(
                            value.dtype, value.aligned, tensor.device
                        )
                    call[name] = value
                compiled = self.kernel.run(**call, grid=(1,), warmup=True)
                if compiled is not None:
                    compiled._init_handles()


_warmup_hooks: dict[Any, _WarmupHook] = {}


def register_triton_warmup(
    kernel: Any,
    *,
    variants: (
        tuple[Mapping[str, tuple[Any, ...]], ...]
        | Callable[[Mapping[str, Any]], tuple[Mapping[str, tuple[Any, ...]], ...]]
    ),
) -> None:
    """Register finite variants owned by one kernel, without changing its API.

    Args:
        kernel: Triton or Gluon JIT function.
        variants: Alternative sets of named argument choices. Choices within
            a set form a Cartesian product. A callback may derive those sets
            from the ordinary launch arguments. Values may be scalars, None or
            TensorVariant pointer descriptions, and may override compiler
            options. All unspecified arguments and options are preserved.

    Returns:
        None. Variants compile only inside kernel_warmup; no variant launches.
    """
    if _IS_NPU:
        return
    if kernel in _warmup_hooks:
        raise ValueError(f"Warmup already registered for {kernel}")
    _warmup_hooks[kernel] = _WarmupHook(kernel, variants)


@contextmanager
def compile_triton_kernels() -> Generator[None]:
    """Prepare normal in-place operators at startup without launching their kernels.

    Callers must discard their outputs: tensor allocation and argument
    preparation still run, but a kernel_launcher call only compiles and loads.
    This scope is for component initialization before any requests are admitted.
    """
    token = _compile_only.set(True)
    try:
        with kernel_warmup(enabled=True):
            yield
    finally:
        _compile_only.reset(token)


def kernel_launcher(kernel: Any, grid: Any) -> Callable[..., Any]:
    """Return the ordinary launcher, or a compiler in a compile-only startup scope.

    Args:
        kernel: Triton or Gluon JIT function.
        grid: The operator's unchanged launch grid.

    Returns:
        A callable preserving every positional argument and keyword option.
    """
    warming_up = is_warming_up()
    compile_only = _compile_only.get()
    if not warming_up and not compile_only:
        return kernel[grid]

    hook = _warmup_hooks.get(kernel) if warming_up else None
    launch = kernel[grid]

    def prepare(*args: Any, **kwargs: Any) -> Any:
        if hook is not None:
            hook(*args, **kwargs)
        if not compile_only:
            return launch(*args, **kwargs)
        compiled = kernel.run(*args, grid=grid, warmup=True, **kwargs)
        if compiled is not None:
            compiled._init_handles()
        return compiled

    return prepare
