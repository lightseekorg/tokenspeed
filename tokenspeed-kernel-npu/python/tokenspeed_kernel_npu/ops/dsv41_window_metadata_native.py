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


"""Native integer LCM metadata; no query or cache values are changed."""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library


class _Shape(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_int)
        for name in (
            "n",
            "tr",
            "tc",
            "pages",
            "pbytes",
            "rbytes",
            "tbytes",
            "pair_r_bytes",
            "read_bytes",
        )
    ] + [(name, ctypes.c_int64) for name in ("ps", "rs", "ts0", "ts1")]


@lru_cache(maxsize=1)
def _library():
    lib = load_library("dsv41_window_metadata")
    lib.launch_window.argtypes = [ctypes.c_void_p] * 6 + [_Shape, ctypes.c_void_p]
    lib.launch_window.restype = None
    lib.launch_compressor.argtypes = [ctypes.c_void_p] * 9 + [_Shape, ctypes.c_void_p]
    lib.launch_compressor.restype = None
    return lib


def _shape(p, r, table, pages, pair_bytes, read_bytes):
    return _Shape(
        p.numel(),
        *table.shape,
        pages,
        p.element_size(),
        r.element_size(),
        table.element_size(),
        pair_bytes,
        read_bytes,
        p.stride(0),
        r.stride(0),
        *table.stride(),
    )


def _supported(p, r, table, outputs):
    return (
        p.ndim == r.ndim == 1
        and p.shape == r.shape
        and table.ndim == 2
        and p.device.type == "npu"
        and all(
            t.dtype in (torch.int32, torch.int64) and t.device == p.device
            for t in (p, r, table)
        )
        and all(
            t.is_contiguous() and t.device == p.device and t.data_ptr() % 32 == 0
            for t in outputs
        )
    )


def decode_window(p, r, write, read, lens, table, pages):
    """Fill aligned caller-owned window buffers; return False if unsupported."""
    if not _supported(p, r, table, (write, read, lens)):
        return False
    n = p.numel()
    if (
        write.shape != (n,)
        or read.shape != (n, 128)
        or lens.shape != (n,)
        or write.dtype != torch.int64
        or lens.dtype != torch.int32
        or read.dtype not in (torch.int32, torch.int64)
    ):
        return False
    lib = _library()
    if n:
        lib.launch_window(
            p.data_ptr(),
            r.data_ptr(),
            table.data_ptr(),
            write.data_ptr(),
            read.data_ptr(),
            lens.data_ptr(),
            _shape(p, r, table, pages, 8, read.element_size()),
            torch.npu.current_stream(p.device).npu_stream,
        )
    return True


def compressor_metadata(
    p, r, table, pages, active, pair_p, pair_r, previous, read, write
):
    """Fill pair metadata with scalar INT64 arithmetic; False keeps CANN fallback."""
    outputs = (active, pair_p, pair_r, previous, read, write)
    if not _supported(p, r, table, outputs):
        return False
    if (
        any(t.shape != p.shape for t in outputs)
        or active.dtype != torch.bool
        or any(t.dtype != torch.int64 for t in (pair_p, previous, read, write))
        or pair_r.dtype not in (torch.int32, torch.int64)
    ):
        return False
    lib = _library()
    if p.numel():
        lib.launch_compressor(
            p.data_ptr(),
            r.data_ptr(),
            table.data_ptr(),
            *[t.data_ptr() for t in outputs],
            _shape(p, r, table, pages, pair_r.element_size(), 8),
            torch.npu.current_stream(p.device).npu_stream,
        )
    return True
