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

"""libascendcl asynchronous SDMA memcpy wrappers."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Literal

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

ACL_MEM_LOCATION_TYPE_HOST = 0
ACL_MEM_LOCATION_TYPE_DEVICE = 1

# CANN batch memcpy caps per-call descriptor count around 4096 entries.
ACL_MEMCPY_BATCH_MAX_COUNT = 4096

_SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


class _AclrtMemLocation(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_int32),
    ]


class _AclrtMemcpyBatchAttr(ctypes.Structure):
    _fields_ = [
        ("dstLoc", _AclrtMemLocation),
        ("srcLoc", _AclrtMemLocation),
        ("rsv", ctypes.c_uint8 * 16),
    ]


_ascendcl = None
_batch_async_fn = None
_memcpy_async_fn = None


def _load_ascendcl():
    global _ascendcl, _batch_async_fn, _memcpy_async_fn
    if _ascendcl is not None:
        return _ascendcl, _batch_async_fn, _memcpy_async_fn
    lib = ctypes.CDLL("libascendcl.so")
    memcpy_async = lib.aclrtMemcpyAsync
    memcpy_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    memcpy_async.restype = ctypes.c_int32

    batch_async = getattr(lib, "aclrtMemcpyBatchAsync", None)
    if batch_async is not None:
        batch_async.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(_AclrtMemcpyBatchAttr),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        batch_async.restype = ctypes.c_int32

    _ascendcl = lib
    _batch_async_fn = batch_async
    _memcpy_async_fn = memcpy_async
    return _ascendcl, _batch_async_fn, _memcpy_async_fn


def _check_acl_error(err, *, op_name: str, detail: str = "") -> None:
    code = int(getattr(err, "value", err))
    if code != 0:
        suffix = f" ({detail})" if detail else ""
        raise RuntimeError(f"{op_name} failed with error code {code}{suffix}")


def ascend_memcpy_batch_async(
    *,
    dst_addrs: Sequence[int],
    src_addrs: Sequence[int],
    sizes: Sequence[int],
    direction: Literal["d2h", "h2d"],
    device_id: int,
    stream_ptr: int,
) -> None:
    """Enqueue asynchronous H2D or D2H SDMA copies on an Ascend stream.

    Uses ``aclrtMemcpyBatchAsync`` (chunked to ``ACL_MEMCPY_BATCH_MAX_COUNT``)
    when exported by ``libascendcl.so``, and falls back to ``aclrtMemcpyAsync``
    on CANN builds without the batch symbol.

    Args:
        dst_addrs: Destination byte addresses.
        src_addrs: Source byte addresses.
        sizes: Byte count of each copy pair.
        direction: ``"d2h"`` (Device to Host) or ``"h2d"`` (Host to Device).
        device_id: NPU device index owning the Device addresses.
        stream_ptr: Raw ``aclrtStream`` pointer (already flushed against
            ``torch_npu``'s host task queue via ``stream.npu_stream``).
    """
    count = len(dst_addrs)
    if len(src_addrs) != count or len(sizes) != count:
        raise ValueError("dst_addrs, src_addrs, and sizes must have equal length")
    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown Ascend memcpy direction {direction!r}")
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
    if count == 0:
        return

    _, batch_async, memcpy_async = _load_ascendcl()
    stream_arg = ctypes.c_void_p(int(stream_ptr))

    if batch_async is not None:
        if direction == "d2h":
            src_loc = _AclrtMemLocation(
                id=int(device_id), type=ACL_MEM_LOCATION_TYPE_DEVICE
            )
            dst_loc = _AclrtMemLocation(id=0, type=ACL_MEM_LOCATION_TYPE_HOST)
        else:
            src_loc = _AclrtMemLocation(id=0, type=ACL_MEM_LOCATION_TYPE_HOST)
            dst_loc = _AclrtMemLocation(
                id=int(device_id), type=ACL_MEM_LOCATION_TYPE_DEVICE
            )
        attrs = (_AclrtMemcpyBatchAttr * 1)(
            _AclrtMemcpyBatchAttr(dstLoc=dst_loc, srcLoc=src_loc)
        )
        attrs_indexes = (ctypes.c_size_t * 1)(0)
        for start in range(0, count, ACL_MEMCPY_BATCH_MAX_COUNT):
            end = min(start + ACL_MEMCPY_BATCH_MAX_COUNT, count)
            chunk = end - start
            dst_arr = (ctypes.c_void_p * chunk)(*dst_addrs[start:end])
            src_arr = (ctypes.c_void_p * chunk)(*src_addrs[start:end])
            size_arr = (ctypes.c_size_t * chunk)(*sizes[start:end])
            fail_index = ctypes.c_size_t(_SIZE_MAX)
            err = batch_async(
                dst_arr,
                size_arr,
                src_arr,
                size_arr,
                chunk,
                attrs,
                attrs_indexes,
                1,
                ctypes.byref(fail_index),
                stream_arg,
            )
            _check_acl_error(
                err,
                op_name="aclrtMemcpyBatchAsync",
                detail=(
                    f"chunk_start={start}, chunk_size={chunk}, "
                    f"fail_index={fail_index.value}"
                ),
            )
        return

    kind = (
        ACL_MEMCPY_DEVICE_TO_HOST if direction == "d2h" else ACL_MEMCPY_HOST_TO_DEVICE
    )
    for index, (dst, src, size) in enumerate(zip(dst_addrs, src_addrs, sizes)):
        err = memcpy_async(
            ctypes.c_void_p(int(dst)),
            int(size),
            ctypes.c_void_p(int(src)),
            int(size),
            kind,
            stream_arg,
        )
        _check_acl_error(
            err,
            op_name="aclrtMemcpyAsync",
            detail=f"index={index}, size={size}",
        )


__all__ = [
    "ACL_MEMCPY_BATCH_MAX_COUNT",
    "ACL_MEMCPY_DEVICE_TO_HOST",
    "ACL_MEMCPY_HOST_TO_DEVICE",
    "ACL_MEM_LOCATION_TYPE_DEVICE",
    "ACL_MEM_LOCATION_TYPE_HOST",
    "ascend_memcpy_batch_async",
]
