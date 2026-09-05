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

"""Shared host-side floor for the fused MXFP4 MoE family: constants,
ragged-metadata helpers, and wrapped-tensor extraction."""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
    RaggedTensorMetadata,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    CDNA4_SCALE_K_BLOCK,
)

# The gfx950 MXFP4 kernels are dominated by uint8 activation/weight/scale buffer
# loads in the M=4/8 decode regime. Keep the backend's i8 buffer-load coalescer
# enabled unless the caller explicitly overrides it before import/compilation.
os.environ.setdefault("AMDGCN_COALESCE_BUFFER_LOAD_I8", "1")


def _as_int32(t):
    if t is None or t.dtype == torch.int32:
        return t
    return t.to(torch.int32)


def _wrapped_tensor_data(obj):
    storage = getattr(obj, "storage", None)
    data = getattr(storage, "data", None)
    return data if isinstance(data, torch.Tensor) else None


_BLOCK_SIZES_TUPLE = tuple(RaggedTensorMetadata.block_sizes())


_BLOCK_SIZES_FROZEN = frozenset(_BLOCK_SIZES_TUPLE)


_BLOCK_SIZE_TO_IDX = {bs: i for i, bs in enumerate(_BLOCK_SIZES_TUPLE)}


def _ragged_block_offs(metadata, block_size: int):
    return metadata.block_offs_data[_BLOCK_SIZE_TO_IDX[block_size]]


def _ragged_scale_block_offs(metadata):
    return _ragged_block_offs(metadata, _NON_K_PRESHUFFLE_BLOCK_SIZE)


def _ragged_block_schedule(metadata, block_size: int):
    return metadata.block_schedule_data[_BLOCK_SIZE_TO_IDX[block_size]]


def composition(cls):
    """A decorator lets aggregate type to directly access attributes from its aggregate member."""

    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        for member in self.__dict__.values():
            if getattr(member, "__triton_aggregate__", False) and hasattr(member, name):
                return getattr(member, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    cls.__getattr__ = __getattr__
    return cls


_CDNA4_NUM_CUS = 256


_PERSISTENT_OVERSUBSCRIBE = 2


_PERSISTENT_TILES_THRESHOLD = _CDNA4_NUM_CUS * 3


_GLUON_DOT_K_WIDTH = 16


_GLUON_DOT_N_LANE = 16


_GLUON_DOT_K_QUAD = 4


_GLUON_DOT_SUB_TILE_K = _GLUON_DOT_K_QUAD * _GLUON_DOT_K_WIDTH  # = 64


_TCP_INFLIGHT_CAP_BYTES = 32 * 1024  # gfx9 L1/TCP per-CU in-flight cap


_CDNA4_NUM_XCDS = 8  # MI355X has 8 XCDs (chiplets) per device.


_SCALE_LOAD_MODES = ("bypass", "transpose", "swizzle")


_SCALE_PRESHUFFLE_FACTOR = 32


# Constants matching triton_kernels' CDNA4MXScaleLayout.
_NON_K_PRESHUFFLE_BLOCK_SIZE = 32


_ALIGN_K_SCALE_SWIZZLE = CDNA4_SCALE_K_BLOCK


# Inner reshape factor for the 7-D unswizzle: K_SCALE_pad must be a
# multiple of this for `unswizzle_mx_scale_cdna4` to be well-defined.
_SWIZZLE_K_S_INNER = 8


def _make_dummy(device, dtype=torch.int32, n: int = 0) -> torch.Tensor:
    return torch.empty(max(n, 0), device=device, dtype=dtype)


_SCALED_FORMATS = {"e2m1", "e4m3", "e5m2"}


def _extract_gluon_raw_w(w):
    """Return the raw ``(E, K_packed, N) uint8`` W tensor.

    The upstream wrapper's ``storage.data`` is already K-contiguous
    so we pass it through. If a ``_gluon_shuffled`` attribute is
    attached (set by the backend's preshuffle hook) we return the
    shuffled view instead -- ``is_shuffled_for_gluon_dot=True`` then
    triggers the kernel's preshuffled W path.
    """
    if isinstance(w, torch.Tensor):
        shuffled = getattr(w, "_gluon_shuffled", None)
        if shuffled is not None:
            return shuffled
        return w
    raw = _wrapped_tensor_data(w)
    if raw is None:
        return w
    shuffled = getattr(raw, "_gluon_shuffled", None)
    if shuffled is not None:
        return shuffled
    return raw


def _extract_gluon_raw_w_unshuffled(w):
    """Return the canonical K-contiguous W storage, ignoring preshuffle attrs.

    M=8/16 medium-decode uses the direct-load body and must not be routed to the
    default preshuffled-W path. This helper preserves the main path's
    ``_extract_gluon_raw_w`` behavior by being opt-in at the call site.
    """
    if isinstance(w, torch.Tensor):
        return w
    raw = _wrapped_tensor_data(w)
    return w if raw is None else raw


def _extract_gluon_raw_s(s):
    """Return the raw uint8 scale tensor for Gluon's ``swizzle`` mode
    (bit-equivalent to upstream CDNA4MXScaleLayout.swizzle_data)."""
    if isinstance(s, torch.Tensor):
        return s
    raw = _wrapped_tensor_data(s)
    return s if raw is None else raw


def _maybe_extract_swiglu_args(fused_activation):
    """Pull ``(alpha, limit, beta)`` from an upstream ``FusedActivation`` object
    representing SwiGLU. Returns ``None`` for any other activation."""
    if fused_activation is None:
        return None
    specs = getattr(fused_activation, "specs", None)
    fn_name = getattr(specs, "name", None) if specs is not None else None
    if fn_name != "swiglu":
        return None
    args = getattr(fused_activation, "fn_args", None)
    if args is None:
        args = getattr(fused_activation, "args", None)
    if args is None or len(args) < 2:
        return None
    beta = args[2] if len(args) >= 3 else 1.0
    return float(args[0]), float(args[1]), float(beta)


def _global_scale_passthrough(scale):
    """Return the flex scale in a form the launcher can take without
    a host ``.item()`` (keeps HIP-graph capture working)."""
    if scale is None:
        return 1.0
    if isinstance(scale, torch.Tensor):
        return scale
    return float(scale)
