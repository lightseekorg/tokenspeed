"""Compile-time template objects and native-style device method calls.

Objects contain only specialization constants. Runtime C++ fields are passed
explicitly as immutable state and returned by methods that update them.
"""

from functools import cache
from hashlib import sha256
from pathlib import Path

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l


@cache
def source_key():
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DeviceTemplate:
    __triton_builtin__ = True

    @property
    def cache_key(self):
        return f"{type(self).__module__}.{type(self).__name__}:{self._key!r}:{source_key()}"


class device_method:
    def __init__(self, fn):
        fn.__annotations__["self"] = l.constexpr
        self.jit = g.jit(fn)

    def __get__(self, obj, owner=None):
        return self if obj is None else _BoundDeviceMethod(self, obj)


class _BoundDeviceMethod:
    __triton_builtin__ = True

    def __init__(self, method, obj):
        self.method, self.obj = method, obj

    @property
    def cache_key(self):
        return self.obj.cache_key + self.method.jit.cache_key

    def __call__(self, *args, _semantic=None, _generator=None, **kwargs):
        if _generator is None:
            raise TypeError("Device methods must be called inside @gluon.jit")
        return _generator.call_JitFunction(
            self.method.jit, [l.constexpr(_CompileTimeView(self.obj)), *args], kwargs
        )


class _CompileTimeView:
    """Expose native static constexpr fields as Gluon constexpr values."""

    __triton_builtin__ = True

    def __init__(self, obj):
        self._object = obj

    @property
    def cache_key(self):
        return self._object.cache_key

    def __getattr__(self, name):
        value = getattr(self._object, name)
        if isinstance(value, (int, float, bool)):
            return l.constexpr(value)
        if isinstance(value, DeviceTemplate):
            value = _CompileTimeView(value)
            # LLVM 22's frontend does not normalize nested compile-time views.
            return l.constexpr(value) if hasattr(l, "thread_barrier") else value
        return value
