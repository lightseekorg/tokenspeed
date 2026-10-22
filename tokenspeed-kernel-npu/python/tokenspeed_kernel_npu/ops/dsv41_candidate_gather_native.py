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


"""Byte-exact candidate KV block gathering; no intermediate per-token slots."""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops._native import load_library


@lru_cache(maxsize=1)
def _library():
    lib = load_library("dsv41_candidate_gather")
    lib.launch_blocks.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
    )
    lib.launch_blocks.restype = None
    return lib


@triton.jit
def _physical_blocks(
    C, T, V, O, CS0, CS1, TS0, TS1, VS, COUNT, WIDTH, PAGES, B: tl.constexpr
):
    q = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    c = tl.load(C + q.to(tl.int64) * CS0 + col * CS1, col < COUNT, other=-1)
    page = tl.load(
        T + q.to(tl.int64) * TS0 + tl.minimum(tl.maximum(c // 8, 0), WIDTH - 1) * TS1
    )
    valid = (c >= 0) & (c * 8 < tl.load(V + q * VS)) & (page > 0) & (page < PAGES)
    physical = tl.where(valid, page, 0) * 8 + tl.where(valid, c, col) % 8
    tl.store(O + q * COUNT + col, physical, col < COUNT)


def gather_candidates(c, t, v, cache):
    lib = _library()
    if c.device.type != "npu" or any(x.device != c.device for x in (t, v, cache)):
        return None
    if (
        c.ndim != 2
        or t.ndim != 2
        or v.ndim != 1
        or c.shape[0] != t.shape[0]
        or v.shape[0] != c.shape[0]
    ):
        return None
    if any(x.dtype not in (torch.int32, torch.int64) for x in (c, t, v)):
        return None
    if (
        not cache.is_contiguous()
        or cache.dtype != torch.bfloat16
        or cache.shape[1:] != (64, 128)
        or not 0 < cache.shape[0] * 8 < 2**31
        or not t.shape[1]
    ):
        return None
    n, count = c.shape
    ids = torch.empty((n, count), device=c.device, dtype=torch.int32)
    out = torch.empty((n, count * 8, 128), device=cache.device, dtype=cache.dtype)
    # Both launches must use the input device's context as well as its stream.
    with torch.npu.device(cache.device):
        _physical_blocks[(n, triton.cdiv(count, 256))](
            c,
            t,
            v,
            ids,
            *c.stride(),
            *t.stride(),
            v.stride(0),
            count,
            t.shape[1],
            cache.shape[0],
            256,
            multibuffer=False,
        )
        lib.launch_blocks(
            cache.data_ptr(),
            ids.data_ptr(),
            out.data_ptr(),
            ids.numel(),
            cache.shape[0] * 8,
            torch.npu.current_stream(cache.device).npu_stream,
        )
    return out
