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

"""DMA batched memcpy for KV cache transfers via ``cuMemcpyBatchAsync``.

One driver call dispatches all (K/V × layer × page) copies onto the
GPU's copy engines, so the transfer does not consume SMs or L2 and
cannot slow concurrent compute (allreduce, MoE bmm). Requires CUDA
12.8+; ``cuda-python`` is used to look the symbol up at module load,
then the hot-path call is a raw ``ctypes`` thunk that takes numpy
data pointers directly — avoiding ~10µs/descriptor of Python wrapper
cost we'd otherwise pay constructing ``CUdeviceptr`` objects.
``cuMemcpyBatchAsync`` rejects the legacy NULL stream, so callers
must issue on an explicit stream.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch
from tokenspeed_kernel.platform import current_platform


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


# CUresult cuMemcpyBatchAsync(
#     CUdeviceptr *dsts, CUdeviceptr *srcs, size_t *sizes, size_t count,
#     CUmemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs,
#     CUstream hStream
# );  -- v13 ABI; v12 had an extra failIdx out-param before hStream.
_BATCH_FN_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
)

_batch_fn = None
_attrs_singleton = None
_attrs_idxs_np = np.zeros(1, dtype=np.uint64)
_resolve_attempted = False


def _try_resolve():
    global _batch_fn, _attrs_singleton, _resolve_attempted
    if _resolve_attempted:
        return _batch_fn
    _resolve_attempted = True
    if not current_platform().is_nvidia:
        return None
    try:
        from cuda.bindings import driver as drv

        err, ver = drv.cuDriverGetVersion()
        if err != drv.CUresult.CUDA_SUCCESS:
            return None
        err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", ver, 0)
        if err != drv.CUresult.CUDA_SUCCESS or not ptr:
            return None
        _batch_fn = _BATCH_FN_TYPE(ptr)
        _attrs_singleton = _CUmemcpyAttributes(srcAccessOrder=1)  # STREAM
    except Exception:
        _batch_fn = None
    return _batch_fn


def is_available() -> bool:
    return _try_resolve() is not None


def batch_async_load_kv_pages(
    host_k_layers: list,
    host_v_layers: list,
    device_k_layers: list,
    device_v_layers: list,
    src_page_token_offsets: torch.Tensor,
    dst_page_token_offsets: torch.Tensor,
    page_bytes: int,
    token_stride_bytes: int,
    stream: torch.cuda.Stream,
) -> None:
    """Single ``cuMemcpyBatchAsync`` call for all (K/V × layer × page) blocks.

    Descriptors live entirely in three numpy uint64 arrays passed by their
    raw memory pointer — no per-descriptor Python work. Hot-path cost is
    dominated by the driver dispatch.
    """
    fn = _try_resolve()
    if fn is None:
        raise RuntimeError("cuMemcpyBatchAsync not available; install cuda-python")

    n_pages = int(src_page_token_offsets.numel())
    if n_pages == 0:
        return
    assert int(dst_page_token_offsets.numel()) == n_pages
    n_layers = len(host_k_layers)
    assert (
        n_layers == len(host_v_layers) == len(device_k_layers) == len(device_v_layers)
    )
    assert src_page_token_offsets.is_cpu and dst_page_token_offsets.is_cpu

    src_off = src_page_token_offsets.numpy().astype(np.uint64, copy=False)
    dst_off = dst_page_token_offsets.numpy().astype(np.uint64, copy=False)
    stride = np.uint64(token_stride_bytes)
    src_byte_off = src_off * stride
    dst_byte_off = dst_off * stride

    src_k_bases = np.fromiter(
        (t.data_ptr() for t in host_k_layers), dtype=np.uint64, count=n_layers
    )
    src_v_bases = np.fromiter(
        (t.data_ptr() for t in host_v_layers), dtype=np.uint64, count=n_layers
    )
    dst_k_bases = np.fromiter(
        (t.data_ptr() for t in device_k_layers), dtype=np.uint64, count=n_layers
    )
    dst_v_bases = np.fromiter(
        (t.data_ptr() for t in device_v_layers), dtype=np.uint64, count=n_layers
    )

    src_k = (src_k_bases[:, None] + src_byte_off[None, :]).ravel()
    src_v = (src_v_bases[:, None] + src_byte_off[None, :]).ravel()
    dst_k = (dst_k_bases[:, None] + dst_byte_off[None, :]).ravel()
    dst_v = (dst_v_bases[:, None] + dst_byte_off[None, :]).ravel()

    # Single uint64 arrays for dst/src/size; ascontiguousarray so .ctypes.data
    # is a stable pointer the driver can read directly.
    dst_all = np.ascontiguousarray(np.concatenate([dst_k, dst_v]), dtype=np.uint64)
    src_all = np.ascontiguousarray(np.concatenate([src_k, src_v]), dtype=np.uint64)
    sz_all = np.full(dst_all.size, page_bytes, dtype=np.uint64)

    err = fn(
        dst_all.ctypes.data,
        src_all.ctypes.data,
        sz_all.ctypes.data,
        ctypes.c_size_t(dst_all.size),
        ctypes.addressof(_attrs_singleton),
        _attrs_idxs_np.ctypes.data,
        ctypes.c_size_t(1),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if err != 0:
        raise RuntimeError(
            f"cuMemcpyBatchAsync failed: err={err} "
            f"total={dst_all.size} n_layers={n_layers} n_pages={n_pages}"
        )
