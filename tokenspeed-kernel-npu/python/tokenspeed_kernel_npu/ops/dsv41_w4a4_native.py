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

"""Grouped INT4 Cube tiles with a fused per-token scaling epilogue."""

import ctypes
from functools import cache, lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library

SUPPORTED_PROJECTIONS = frozenset({(5120, 4608), (2304, 5120)})


@lru_cache(maxsize=1)
def _library():
    library = load_library("dsv41_w4a4")
    library.launch_w4a4.argtypes = (
        [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    library.launch_w4a4.restype = None
    library.make_tiling.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.make_tiling.restype = ctypes.c_int
    return library


def available():
    return _library() is not None


@cache
def _tiling(device, k, n):
    # Static model geometry only: no token counts or request metadata on host.
    raw = ctypes.create_string_buffer(2048)
    size = _library().make_tiling(raw, k, n)
    if size <= 0 or size > len(raw):
        raise RuntimeError("Failed to prepare Ascend INT4 Cube tiling")
    return torch.tensor(list(raw.raw[:size]), dtype=torch.uint8, device=device)


def grouped_w4a4(quantized, weight, weight_scale, token_scale, counts, out):
    """Write grouped BF16 output for packed INT4 inputs and live expert counts.

    Quantized is INT32 [capacity,K/8]; weight is NZ INT32 [E,N,K/8].
    Weight scales are UINT64 [E,1,N] carrying the existing Fixpipe scale bits;
    token scales are FP32 [capacity], counts INT64 [E], output BF16 [capacity,N].
    Nonnegative counts must sum to at most capacity. Only that active prefix
    is written. Empty groups and replay-time count changes require no host read.
    """
    import torch_npu

    if quantized.ndim != 2 or weight.ndim != 3:
        raise ValueError("Expected packed grouped INT4 matrices")
    capacity, packed_k = quantized.shape
    experts, n, weight_k = weight.shape
    k = packed_k * 8
    if (k, n) not in SUPPORTED_PROJECTIONS or weight_k != packed_k:
        raise ValueError("Unsupported V4.1 grouped INT4 geometry")
    if (
        weight_scale.shape != (experts, 1, n)
        or token_scale.shape != (capacity,)
        or counts.shape != (experts,)
        or out.shape != (capacity, n)
    ):
        raise ValueError("Incorrect grouped INT4 scale/count/output shape")
    tensors = (quantized, weight, weight_scale, token_scale, counts, out)
    dtypes = (
        torch.int32,
        torch.int32,
        torch.uint64,
        torch.float32,
        torch.int64,
        torch.bfloat16,
    )
    if any(t.dtype != dtype for t, dtype in zip(tensors, dtypes, strict=True)):
        raise ValueError("Incorrect grouped INT4 dtype")
    if quantized.device.type != "npu" or any(
        t.device != quantized.device for t in tensors
    ):
        raise ValueError("Grouped INT4 tensors must share an NPU device")
    if (
        any(not t.is_contiguous() for t in tensors)
        or torch_npu.get_npu_format(weight) != 29
    ):
        raise ValueError("Grouped INT4 requires contiguous tensors and NZ weights")
    if max(capacity, experts) >= 2**31:
        raise ValueError("Grouped INT4 dimensions exceed INT32")
    if capacity == 0:
        return out
    library = _library()
    tiling = _tiling(quantized.device, k, n)
    scratch = torch.empty(
        (24, 2, 64, 512), dtype=torch.float16, device=quantized.device
    )
    library.launch_w4a4(
        *[
            ctypes.c_void_p(t.data_ptr())
            for t in (
                quantized,
                weight,
                weight_scale,
                counts,
                token_scale,
                out,
                scratch,
                tiling,
            )
        ],
        k,
        n,
        experts,
        ctypes.c_void_p(torch.npu.current_stream(quantized.device).npu_stream),
    )
    return out
