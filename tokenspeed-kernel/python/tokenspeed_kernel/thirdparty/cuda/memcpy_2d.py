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

"""libcudart ``cudaMemcpy2DAsync`` wrapper."""

from __future__ import annotations

import ctypes

_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
_cudart = None


def _load_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    lib = ctypes.CDLL("libcudart.so")
    lib.cudaMemcpy2DAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cudaMemcpy2DAsync.restype = ctypes.c_int
    _cudart = lib
    return _cudart


def cuda_memcpy_2d_async(
    *,
    dst: int,
    dst_pitch: int,
    src: int,
    src_pitch: int,
    width: int,
    height: int,
    stream_ptr: int,
) -> None:
    runtime = _load_cudart()
    err = runtime.cudaMemcpy2DAsync(
        dst,
        dst_pitch,
        src,
        src_pitch,
        width,
        height,
        _CUDA_MEMCPY_DEVICE_TO_DEVICE,
        stream_ptr,
    )
    code = getattr(err, "value", err)
    if int(code) != 0:
        raise RuntimeError(f"cudaMemcpy2DAsync failed: {err}")
