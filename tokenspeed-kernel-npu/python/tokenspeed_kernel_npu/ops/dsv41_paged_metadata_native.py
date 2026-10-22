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


"""Byte-exact physical-address conversion and stable compaction."""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library


class _Shape(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_int) for name in ("rows", "w0", "w1")]
        + [(name, ctypes.c_int64) for name in ("s0", "s1", "l0", "l1")]
        + [
            (name, ctypes.c_int)
            for name in (
                "base0",
                "stride0",
                "capacity0",
                "base1",
                "stride1",
                "capacity1",
            )
        ]
    )


@lru_cache(maxsize=1)
def _library():
    library = load_library("dsv41_paged_metadata")
    library.launch_paged_rows.argtypes = [ctypes.c_void_p] * 6 + [
        _Shape,
        ctypes.c_void_p,
    ]
    library.launch_paged_rows.restype = None
    return library


def prepare_indices(
    slots0,
    lengths0,
    slots1,
    lengths1,
    geometry0,
    geometry1,
    second_width,
    physical_rows,
):
    """Return compact physical IDs/counts, or None for the portable fallback.

    GatherMask preserves input order and duplicates. Only aligned INT32 rows
    with widths divisible by 64 use this kernel. Float comparisons are exact
    for valid logical IDs because each bank's capacity is at most 2**24;
    physical addresses themselves always use INT32 arithmetic, including
    addresses above 2**24. No KV data is read or converted.
    """
    if slots0.ndim != 2 or slots1.ndim != 2:
        return None
    rows, first_width = slots0.shape
    width = first_width + second_width
    if (
        not rows
        or width <= 0
        or width > 4096
        or first_width % 64
        or second_width < 0
        or second_width % 64
        or slots1.shape[0] != rows
        or second_width > slots1.shape[1]
        or not 0 < physical_rows < 2**31
    ):
        return None
    tensors = (slots0, slots1, lengths0, lengths1)
    if (
        slots0.device.type != "npu"
        or any(t.dtype != torch.int32 or t.device != slots0.device for t in tensors)
        or lengths0.shape != (rows,)
        or lengths1.shape != (rows,)
    ):
        return None
    if any(
        t.stride(1) != 1 or t.stride(0) < 0 or t.stride(0) % 8 or t.data_ptr() % 32
        for t in (slots0, slots1)
    ):
        return None
    for base, stride, capacity in (geometry0, geometry1):
        if (
            base < 0
            or stride < 64
            or capacity <= 0
            or capacity > 2**24
            or capacity % 64
            or base + (capacity // 64 - 1) * stride + 64 > physical_rows
        ):
            return None
    library = _library()
    out = torch.empty((rows, width), device=slots0.device, dtype=torch.int32)
    counts = torch.empty(rows, device=slots0.device, dtype=torch.int32)
    shape = _Shape(
        rows,
        first_width,
        second_width,
        slots0.stride(0),
        slots1.stride(0),
        lengths0.stride(0),
        lengths1.stride(0),
        *geometry0,
        *geometry1,
    )
    library.launch_paged_rows(
        slots0.data_ptr(),
        lengths0.data_ptr(),
        slots1.data_ptr(),
        lengths1.data_ptr(),
        out.data_ptr(),
        counts.data_ptr(),
        shape,
        torch.npu.current_stream(slots0.device).npu_stream,
    )
    return out, counts
