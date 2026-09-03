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

"""A torch view over memory torch did not allocate.

The symmetric-memory rendezvous hands back the NVLS multicast address as a
plain integer, and a store to it lands in every peer's mailbox at once. Giving
that address to a vendor GEMM's ``out=`` needs a tensor over it, which torch
exposes no public way to build, so this wraps the pointer in a DLPack capsule
whose deleter does nothing: the mailbox owns the memory and outlives the view.
"""

from __future__ import annotations

import ctypes

import torch

# DLPack device type kDLCUDA, and the (type code, bits, lanes) triple of bf16.
_DL_CUDA = 2
_DL_BFLOAT = 4
_DL_BF16_BITS = 16
# Views are per mailbox slot and live as long as the process, so this cannot grow.
_KEEPALIVE: list = []


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManaged(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManaged))
_DLManaged._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER),
]


_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.restype = ctypes.py_object
_new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def bf16_tensor_on_pointer(
    pointer: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    device_index: int,
) -> torch.Tensor:
    """Alias a raw device address as a bfloat16 tensor that owns nothing.

    Args:
        pointer: Device address of element zero.
        shape: Shape of the view.
        strides: Strides in elements, one per dimension; the last must be 1 for
            a cuBLAS ``out=`` to accept the view without a staging copy.
        device_index: CUDA device the address belongs to.

    Returns:
        A bfloat16 CUDA tensor aliasing ``pointer``. It allocates nothing and
        frees nothing, so the caller must keep the underlying buffer alive.
    """
    from torch.utils.dlpack import from_dlpack

    if len(shape) != len(strides):
        raise ValueError("shape and strides must have the same rank")
    if not pointer:
        raise ValueError("a null address has no tensor over it")
    managed = _DLManaged()
    sizes = (ctypes.c_int64 * len(shape))(*shape)
    steps = (ctypes.c_int64 * len(strides))(*strides)
    managed.dl_tensor.data = ctypes.c_void_p(pointer)
    managed.dl_tensor.device = _DLDevice(_DL_CUDA, device_index)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _DLDataType(_DL_BFLOAT, _DL_BF16_BITS, 1)
    managed.dl_tensor.shape = sizes
    managed.dl_tensor.strides = steps
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    # NULL, not a no-op callback: freeing is the mailbox owner's job either way,
    # but a ctypes callback is no longer callable once Py_FinalizeEx has begun
    # clearing module dicts, and a view held to exit is deallocated exactly
    # there. DLPack permits a null deleter; a no-op one segfaults on shutdown.
    managed.deleter = ctypes.cast(None, _DELETER)
    # The consumer keeps the capsule, not the structures it points into.
    _KEEPALIVE.extend((managed, sizes, steps))
    return from_dlpack(_new_capsule(ctypes.byref(managed), b"dltensor", None))


__all__ = ["bf16_tensor_on_pointer"]
