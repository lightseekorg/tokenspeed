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


"""Lossless route-weight mapping for bounded decode capacities."""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library


@lru_cache(maxsize=1)
def _library():
    lib = load_library("dsv41_route_weights")
    lib.launch_route_weights.argtypes = [ctypes.c_void_p] * 3 + [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_route_weights.restype = None
    return lib


def route_weights_native(source_to_dest, weights):
    """Map unique destinations without accumulating remote slots at a sink.

    Return None outside the native kernel's geometry. The routing operator
    guarantees unique valid destinations; negative entries denote remote
    experts. FP32 values retain their bits, with scatter-add's +0/NaN handling.
    """
    n = source_to_dest.numel()
    if (
        not 512 < n <= 4096
        or source_to_dest.device.type != "npu"
        or source_to_dest.ndim != 1
        or source_to_dest.dtype != torch.int32
        or not source_to_dest.is_contiguous()
        or weights.device != source_to_dest.device
        or weights.dtype != torch.float32
        or weights.shape != (n,)
        or not weights.is_contiguous()
    ):
        return None
    lib = _library()
    output = torch.empty(n, device=weights.device, dtype=torch.float32)
    lib.launch_route_weights(
        source_to_dest.data_ptr(),
        weights.data_ptr(),
        output.data_ptr(),
        n,
        16,
        torch.npu.current_stream(weights.device).npu_stream,
    )
    return output
