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

"""Pack unequal token shards and clear padding without intermediate copies."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch
from tokenspeed_kernel_npu._triton import tl, triton


class _Layout(ctypes.Structure):
    _fields_ = [
        ("ranks", ctypes.c_int),
        ("maximum", ctypes.c_int),
        ("words", ctypes.c_int),
        ("offsets", ctypes.c_int * 16),
        ("counts", ctypes.c_int * 16),
    ]


@lru_cache(maxsize=1)
def _library():
    path = Path(__file__).resolve().parents[1] / "lib/libtoken_shards.so"
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    lib.launch_pad_token_shards.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        _Layout,
        ctypes.c_void_p,
    ]
    lib.launch_pad_token_shards.restype = None
    return lib


def pad_token_shards(tensor, counts):
    """Return rank-major rows with zero padding up to max(counts).

    The sixteen-shard native path copies aligned contiguous rows bit for bit.
    Other layouts retain the ordinary copy path. Counts travel by value in
    the kernel arguments, avoiding host copies or graph-owned metadata caches.
    """
    if tensor.ndim != 2 or tensor.device.type != "npu":
        raise ValueError("Expected a two-dimensional NPU tensor")
    if not counts or any(n < 0 for n in counts) or sum(counts) != tensor.shape[0]:
        raise ValueError("Nonnegative shard counts must cover all input rows")
    maximum = max(counts)
    row_bytes = tensor.shape[1] * tensor.element_size()
    native = (
        _library()
        if (
            len(counts) <= 16
            and tensor.is_contiguous()
            and row_bytes % 32 == 0
            and tensor.data_ptr() % 32 == 0
            and maximum * (row_bytes // 4) < 2**31
            and tensor.shape[0] < 2**31
        )
        else None
    )
    out = tensor.new_empty((len(counts) * maximum, tensor.shape[1]))
    if not out.numel():
        return out
    if native is None:
        out.zero_()
        start = 0
        for rank, count in enumerate(counts):
            out[rank * maximum : rank * maximum + count].copy_(
                tensor[start : start + count]
            )
            start += count
        return out
    layout = _Layout(len(counts), maximum, row_bytes // 4)
    start = 0
    for rank, count in enumerate(counts):
        layout.offsets[rank] = start
        layout.counts[rank] = count
        start += count
    native.launch_pad_token_shards(
        tensor.data_ptr(),
        out.data_ptr(),
        layout,
        torch.npu.current_stream(tensor.device).npu_stream,
    )
    return out


@triton.jit(do_not_specialize=["E"])
def _sum_token_shards(X, Y, E, SHARDS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < E, 0)
    for rank in tl.static_range(1, SHARDS):
        value = value + tl.load(X + rank * E + i, i < E, 0)
    tl.store(Y + i, value, i < E)


def sum_token_shards(tensor, shards):
    """Sum contiguous rank-major FP32 [shards*rows,hidden] to BF16 [rows,hidden].

    Addition follows group rank order in FP32; BF16 rounding occurs only once,
    after all partial projections have been accumulated.
    """
    if (
        tensor.ndim != 2
        or tensor.device.type != "npu"
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or shards < 1
        or tensor.shape[0] % shards
    ):
        raise ValueError("Expected contiguous FP32 NPU rows split into equal shards")
    out = torch.empty(
        (tensor.shape[0] // shards, tensor.shape[1]),
        device=tensor.device,
        dtype=torch.bfloat16,
    )
    if out.numel():
        block = 256 if out.numel() < 32768 else 1024
        _sum_token_shards[(triton.cdiv(out.numel(), block),)](
            tensor,
            out,
            out.numel(),
            shards,
            block,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
