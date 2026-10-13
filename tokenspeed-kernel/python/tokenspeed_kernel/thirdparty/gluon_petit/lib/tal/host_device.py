"""One arithmetic body for unsigned native TAL_HOST_DEVICE helpers.

Host calls evaluate Python integers. Device calls inline the same function's
Gluon JIT body, with the layout/configuration object as a constexpr argument.
"""

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import _unwrap_if_constexpr


class host_device:
    def __init__(self, fn):
        fn.__annotations__["self"] = l.constexpr
        self.fn = fn
        self.jit = g.jit(fn)

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return _BoundHostDevice(self, obj)


class _BoundHostDevice:
    __triton_builtin__ = True

    def __init__(self, method, obj):
        self.method = method
        self.obj = obj

    @property
    def cache_key(self):
        return self.method.jit.cache_key + self.obj.cache_key

    def __call__(self, *args, _semantic=None, _generator=None, **kwargs):
        if _generator is None:
            return self.method.fn(self.obj, *args, **kwargs) & 0xFFFFFFFF
        from lib.tal.device import _CompileTimeView

        result = _generator.call_JitFunction(
            self.method.jit, [l.constexpr(_CompileTimeView(self.obj)), *args], kwargs
        )
        if isinstance(result, l.tensor):
            return _semantic.cast(result, l.uint32)
        return l.constexpr(_unwrap_if_constexpr(result) & 0xFFFFFFFF)
