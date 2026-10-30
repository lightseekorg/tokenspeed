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

from collections.abc import Callable, Mapping
from itertools import product
from typing import Any, TypeAlias

from tokenspeed_kernel._triton import _IS_NPU
from tokenspeed_kernel.ops.tuning import is_warming_up

_VariantSets: TypeAlias = tuple[Mapping[str, tuple[Any, ...]], ...]
_Variants: TypeAlias = _VariantSets | Callable[[Mapping[str, Any]], _VariantSets]
# Operators register their finite warmup variants in this process-wide
# registry at import time, so kernel_launcher can prepare them during
# startup warmup without changing any operator's launch signature.
_warmup_variants: dict[Any, _Variants] = {}


def register_triton_warmup(
    kernel: Any,
    *,
    variants: _Variants,
) -> None:
    """Register finite variants owned by one kernel, without changing its API.

    Args:
        kernel: Triton or Gluon JIT function.
        variants: Alternative sets of named argument choices. Choices within
            a set form a Cartesian product. A callback may derive those sets
            from the ordinary launch arguments. Choices cover scalar alignment
            classes and tile/compiler options; tensor arguments are preserved.

    Returns:
        None. Variants compile only inside kernel_warmup; no variant launches.
    """
    if _IS_NPU:
        return
    if kernel in _warmup_variants:
        raise ValueError(f"Warmup already registered for {kernel}")
    _warmup_variants[kernel] = variants


def kernel_launcher(kernel: Any, grid: Any) -> Callable[..., Any]:
    """Prepare registered variants during warmup, then run the ordinary launch.

    Args:
        kernel: Triton or Gluon JIT function.
        grid: The operator's unchanged launch grid.

    Returns:
        A callable preserving every positional argument and keyword option.
    """
    variants = _warmup_variants.get(kernel) if is_warming_up() else None
    if variants is None:
        return kernel[grid]

    launch = kernel[grid]

    def prepare(*args: Any, **kwargs: Any) -> Any:
        names = kernel.arg_names
        if len(args) > len(names):
            raise TypeError(f"Too many arguments to {kernel.fn.__name__}")
        bound = dict(zip(names, args))
        duplicate = bound.keys() & kwargs.keys()
        if duplicate:
            raise TypeError(f"Arguments supplied twice: {sorted(duplicate)}")
        bound.update(kwargs)
        sets = variants(bound) if callable(variants) else variants
        for choices in sets:
            if any(not values for values in choices.values()):
                raise ValueError("Warmup argument choices must not be empty")
            for values in product(*choices.values()):
                call = bound | dict(zip(choices, values, strict=True))
                compiled = kernel.warmup(**call, grid=(1,))
                if compiled is not None:
                    compiled._init_handles()
        return launch(*args, **kwargs)

    return prepare
