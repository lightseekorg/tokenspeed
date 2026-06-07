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

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable


# TODO: Move vendor kernel registration metadata into tokenspeed_kernel.registrations
# and delete this vendor-local bootstrap once vendor modules are implementation-only.
class Priority(IntEnum):
    REFERENCE = 0
    PORTABLE = 4
    PERFORMANT = 8
    SPECIALIZED = 12
    PLUGIN = 16


@dataclass(frozen=True)
class KernelRegistration:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    impl: Callable


REGISTRATIONS: list[KernelRegistration] = []


def register_kernel(*args, **kwargs) -> Callable:
    def decorator(fn: Callable) -> Callable:
        REGISTRATIONS.append(KernelRegistration(args=args, kwargs=kwargs, impl=fn))
        return fn

    return decorator


def error_fn(*args, **kwargs):
    raise RuntimeError("Kernel implementation not found")


class ErrorClass:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Kernel implementation not found")
