"""Petit Layout/Shape/Stride coordinate mapping for static logical layouts.

Shape and stride trees are compile-time integers. Coordinates may contain
runtime Gluon tensors. This is address arithmetic, not a thread distribution.
"""

from dataclasses import dataclass

import triton.experimental.gluon as g
from lib.tal.tensor.stride import crd2idx
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin


@g.constexpr_function
def make_shape(*values):
    return tuple(values)


@g.constexpr_function
def make_stride(*values):
    return tuple(values)


# Native Shape/Stride are tuple aliases; spell their construction explicitly.
Shape = make_shape
Stride = make_stride


@builtin
def make_coord(*values, _semantic=None):
    return l.tuple(values)


def _validate(shape, stride):
    if isinstance(shape, tuple):
        if not isinstance(stride, tuple) or len(shape) != len(stride):
            raise ValueError("Mismatched Ranks")
        if not shape:
            raise ValueError("Layout modes must be nonempty")
        for s, d in zip(shape, stride, strict=True):
            _validate(s, d)
    elif not isinstance(shape, int) or not isinstance(stride, int):
        raise TypeError("Layout requires static integer shapes and strides")
    elif shape <= 0:
        raise ValueError("Layout extents must be positive")


@dataclass(frozen=True)
class Layout:
    __triton_builtin__ = True
    shape_: tuple | int
    stride_: tuple | int

    def __post_init__(self):
        object.__setattr__(self, "shape_", _unwrap_if_constexpr(self.shape_))
        object.__setattr__(self, "stride_", _unwrap_if_constexpr(self.stride_))
        _validate(self.shape_, self.stride_)

    @property
    def cache_key(self):
        return f"Layout({self.shape_!r}, {self.stride_!r}, {crd2idx.cache_key})"

    def shape(self):
        return self.shape_

    def stride(self):
        return self.stride_

    def __call__(self, coord, _semantic=None, _generator=None):
        # Bridge Python's callable layout object to its ordinary Gluon JIT body.
        # No IR is emitted here; crd2idx performs the native recursive mapping.
        if _generator is None:
            raise TypeError("Layout coordinates must be evaluated inside @gluon.jit")
        return _generator.call_JitFunction(
            crd2idx, [coord, l.constexpr(self.shape_), l.constexpr(self.stride_)], {}
        )
