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


"""Mixed Cube/Vector BF16 index scan for Ascend 910B.

Physical page equality is checked on device before sharing keys between two
query rows. Different requests, partial pages and replay-time page changes
retain independent scores. The native kernel preserves the existing BF16
rounding boundaries and rank-ordered shard reduction.
"""

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel_npu.ops._native import load_library


@lru_cache(maxsize=1)
def _library():
    library = load_library("dsv41_index_scan")
    library.launch_scan.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    )
    library.launch_scan.restype = None
    return library


def scan_scores(q, weights, cache, table, visible, out, *, shards, parts):
    """Write live full-history scores into caller-owned FP32 output.

    Inputs are BF16 Q [T,32,128], BF16 weights [T,32], BF16 cache
    [pages,64,128], INT32 page table [T,P] and visible lengths [T].
    Parts is the number of disjoint historical partitions per query pair.
    Unsupported metadata is rejected before launch. Graph replay reads fresh
    page IDs and lengths; no host tensor reads or persistent request state.
    """
    n = q.shape[0]
    width = table.shape[1] * 64
    if (
        q.shape != (n, 32, 128)
        or weights.shape != (n, 32)
        or cache.ndim != 3
        or cache.shape[1:] != (64, 128)
        or table.shape[0] != n
        or visible.shape != (n,)
        or out.shape != (n, width)
    ):
        raise ValueError("Unsupported native index scan shape")
    if (
        q.dtype != torch.bfloat16
        or weights.dtype != torch.bfloat16
        or cache.dtype != torch.bfloat16
        or table.dtype != torch.int32
        or visible.dtype != torch.int32
        or out.dtype != torch.float32
    ):
        raise ValueError("Native index scan requires BF16 data and INT32 metadata")
    tensors = (q, weights, cache, table, visible, out)
    if q.device.type != "npu" or any(t.device != q.device for t in tensors):
        raise ValueError("Native index scan tensors must share an NPU device")
    if not cache.is_contiguous() or not out.is_contiguous():
        raise ValueError("Native index cache and output must be contiguous")
    if shards not in (1, 2, 4, 8, 16, 32) or not 1 <= parts <= 24:
        raise ValueError("Unsupported index shard or partition count")
    if max(n, width, cache.shape[0]) >= 2**31:
        raise ValueError("Native index scan dimensions exceed INT32")
    if n == 0 or width == 0:
        return out
    library = _library()
    inputs = [
        q.contiguous(),
        weights.contiguous(),
        cache,
        table.contiguous(),
        visible.contiguous(),
        out,
    ]
    workspace = torch.empty(24 * 2 * 64 * 128, device=q.device, dtype=torch.bfloat16)
    inputs.append(workspace)
    library.launch_scan(
        *[ctypes.c_void_p(t.data_ptr()) for t in inputs],
        n,
        width,
        cache.shape[0],
        parts,
        shards,
        ctypes.c_void_p(torch.npu.current_stream(q.device).npu_stream),
    )
    return out
