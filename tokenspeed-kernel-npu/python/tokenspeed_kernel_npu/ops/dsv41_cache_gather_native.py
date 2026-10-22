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


"""Batched, byte-exact BF16 cache gather with 64-bit page addresses."""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library


@lru_cache(maxsize=1)
def _library():
    library = load_library("dsv41_cache_gather")
    library.launch_gather.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int64] * 4
        + [ctypes.c_int] * 3
        + [ctypes.c_void_p]
    )
    library.launch_gather.restype = None
    return library


def gather_rows(cache, indices, out):
    """Gather live INT32/INT64 indices; preserve BF16 bits or widen to FP32.

    Page/row padding is supported, and invalid slots produce positive zero.
    Output and metadata are contiguous. The page address is computed with
    scalar INT64 arithmetic, including cache storage beyond 2 GiB.
    """
    if (
        cache.dtype != torch.bfloat16
        or cache.ndim != 3
        or cache.shape[1:] != (64, 512)
        or cache.stride(2) != 1
        or cache.data_ptr() % 32
        or cache.stride(0) % 16
        or cache.stride(1) % 16
    ):
        raise ValueError("Native gather requires aligned BF16 [pages,64,512] rows")
    if (
        indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_contiguous()
        or not out.is_contiguous()
        or out.shape != (*indices.shape, 512)
        or out.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Unsupported native cache gather metadata/output")
    if (
        cache.device.type != "npu"
        or indices.device != cache.device
        or out.device != cache.device
    ):
        raise ValueError("Native cache gather tensors must share an NPU device")
    if indices.numel() == 0:
        return out
    library = _library()
    library.launch_gather(
        cache.data_ptr(),
        indices.data_ptr(),
        out.data_ptr(),
        indices.numel(),
        cache.shape[0] * 64,
        cache.stride(0),
        cache.stride(1),
        32 if out.dtype == torch.float32 else 64,
        int(indices.dtype == torch.int64),
        int(out.dtype == torch.float32),
        torch.npu.current_stream(cache.device).npu_stream,
    )
    return out
