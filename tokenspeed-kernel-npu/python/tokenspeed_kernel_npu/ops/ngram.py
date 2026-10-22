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


"""Ascend C preparation of exact Engram histories and hashes."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


class _Shape(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_int)
        for name in (
            "batch",
            "total",
            "capacity",
            "vocab",
            "uniform",
            "layers",
            "heads",
            "pad",
            "ids64",
            "ends64",
            "reset_bytes",
        )
    ]


@lru_cache(maxsize=1)
def _library():
    path = Path(__file__).resolve().parents[1] / "lib" / "libdsv41_ngram.so"
    if not path.is_file():
        raise RuntimeError("Build tokenspeed-kernel-npu/csrc/dsv41_ngram/build.sh")
    library = ctypes.CDLL(str(path))
    library.launch_ngram.argtypes = [ctypes.c_void_p] * 17 + [_Shape, ctypes.c_void_p]
    library.launch_ngram.restype = None
    return library


def prepare_ngram(
    snapshots,
    positions,
    reset,
    slots,
    cache_lengths,
    request_ends,
    tail,
    needs_seed,
    ids,
    previous,
    token_mask,
    prefix,
    hashes,
    total,
    vocab_size,
    uniform_length,
    params,
):
    """Fill caller-owned histories and hashes without mutating accepted state."""
    layers, _, heads = params.primes.shape
    tensors = [
        snapshots,
        positions,
        reset,
        slots,
        cache_lengths,
        request_ends,
        tail,
        needs_seed,
        ids,
        previous,
        token_mask,
        prefix,
        hashes,
        params.token_map,
        params.primes,
        params.offsets,
        params.multipliers,
    ]
    if any(t.device != ids.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Engram inputs must be contiguous on the same NPU")
    expected = [
        torch.int64,
        torch.int64,
        reset.dtype,
        torch.int64,
        torch.int32,
        request_ends.dtype,
        torch.int64,
        torch.bool,
        ids.dtype,
        torch.int64,
        torch.bool,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.int64,
    ]
    if (
        any(t.dtype != dtype for t, dtype in zip(tensors, expected))
        or reset.dtype not in (torch.bool, torch.int32, torch.int64)
        or ids.dtype not in (torch.int32, torch.int64)
        or request_ends.dtype not in (torch.int32, torch.int64)
        or not 0 < heads <= 32
    ):
        raise ValueError("Unsupported Engram metadata dtype or head count")
    if max(previous.shape[0], slots.numel()) == 0:
        return
    _library().launch_ngram(
        *(t.data_ptr() for t in tensors),
        _Shape(
            slots.numel(),
            total,
            previous.shape[0],
            vocab_size,
            uniform_length,
            layers,
            heads,
            params.pad_id,
            ids.dtype == torch.int64,
            request_ends.dtype == torch.int64,
            reset.element_size(),
        ),
        torch.npu.current_stream(ids.device).npu_stream,
    )
