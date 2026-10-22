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


"""Stable INT32 page/candidate compaction for the native BF16 indexer."""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library


class _Shape(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_int)
        for name in ("rows", "width", "pages", "candidates", "stride")
    ] + [(name, ctypes.c_int64) for name in ("ts", "vs", "cs")]


@lru_cache(maxsize=1)
def _library():
    lib = load_library("dsv41_index_metadata")
    lib.launch_index_meta.argtypes = [ctypes.c_void_p] * 6 + [_Shape, ctypes.c_void_p]
    lib.launch_index_meta.restype = None
    return lib


def compact_index_metadata(
    table, visible, candidates, pages, mapping, native_table, counts
):
    """Fill preinitialized buffers; return False for unsupported layouts.

    Unaligned widths and GM row addresses use exact-length padded transfers.
    Mapping starts at -1 and native_table at zero, including unused tails.
    The result keeps valid logical IDs in input order, including duplicates.
    Counts are visible token counts (a final partial block contributes its
    actual length), repeated over each aligned 16-element row.
    """
    if table.ndim != 2 or table.device.type != "npu":
        return False
    if candidates is not None and candidates.ndim != 2:
        return False
    rows, width = table.shape
    count = width if candidates is None else candidates.shape[-1]
    tensors = [table, visible, mapping, native_table, counts]
    if candidates is not None:
        tensors.append(candidates)
    if (
        not 0 < rows < 2**31
        or not 0 < width <= 16384
        or not 0 < pages <= 2**24
        or not 0 < count < 2**31
        or visible.shape != (rows,)
        or mapping.ndim != 2
        or mapping.shape[0] != rows
        or mapping.shape[1] < count
        or mapping.shape[1] >= 2**31
        or native_table.shape != mapping.shape
        or counts.shape != (rows, 16)
        or any(t.dtype != torch.int32 or t.device != table.device for t in tensors)
        or not all(t.is_contiguous() for t in (mapping, native_table, counts))
    ):
        return False
    if candidates is not None and (candidates.ndim != 2 or candidates.shape[0] != rows):
        return False
    inputs = [table] if candidates is None else [table, candidates]
    if any(t.stride(1) != 1 or t.stride(0) < 0 for t in inputs):
        return False
    lib = _library()
    lib.launch_index_meta(
        table.data_ptr(),
        visible.data_ptr(),
        0 if candidates is None else candidates.data_ptr(),
        mapping.data_ptr(),
        native_table.data_ptr(),
        counts.data_ptr(),
        _Shape(
            rows,
            width,
            pages,
            -1 if candidates is None else count,
            mapping.shape[1],
            table.stride(0),
            visible.stride(0),
            0 if candidates is None else candidates.stride(0),
        ),
        torch.npu.current_stream(table.device).npu_stream,
    )
    return True
