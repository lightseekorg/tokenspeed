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

"""Return the current CUDA context's thread-stack reservation while its GPU rank sleeps."""

import ctypes
from functools import cache

_STACK_SIZE = 0  # CU_LIMIT_STACK_SIZE


@cache
def _driver() -> ctypes.CDLL:
    """Bind the CUDA driver only when a CUDA worker first hands off its device."""
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    lib.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    return lib


def _set_stack(size: int) -> None:
    """Resize the current context's stack or propagate the native CUDA error."""
    result = _driver().cuCtxSetLimit(_STACK_SIZE, size)
    if result:
        raise RuntimeError(
            f"cuCtxSetLimit(STACK_SIZE, {size}) failed with CUresult {result}"
        )


def trim_cuda_stack() -> int:
    """Shrink the stack after GPU work drains and return the size required on wake."""
    previous = ctypes.c_size_t()
    result = _driver().cuCtxGetLimit(ctypes.byref(previous), _STACK_SIZE)
    if result:
        raise RuntimeError(f"cuCtxGetLimit(STACK_SIZE) failed with CUresult {result}")
    if previous.value:
        _set_stack(0)
    return previous.value


def restore_cuda_stack(previous: int) -> None:
    """Restore the same context's pre-sleep limit before relaunching any kernels."""
    if previous:
        _set_stack(previous)
