"""Compile-time method binding for Petit's class-based Gluon helpers.

This metaprogramming adapter preserves the port's C++-style organization of
configuration and device methods. Python objects hold specialization constants
and nested helpers; they are not mutable objects allocated on the GPU. Runtime
tensors and changing state are explicit method arguments and return values.

For example, a configured NativeMxFp4Matmul object's Matmul(acc, ...) call inside
a Gluon kernel binds the object to the decorated method. The compiler invokes
_BoundDeviceMethod, which passes a constexpr view of that object as self to the
Gluon JIT function. Reads such as self.kMRepeats then resolve at compile time,
while operations on acc become device code. There is no Python method dispatch
during GPU execution.
"""

from functools import cache
from hashlib import sha256
from pathlib import Path

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l


@cache
def source_key():
    """Return a digest of relative paths and contents of vendored lib/*.py files.

    The recursive digest conservatively invalidates specializations when helper
    sources change. It is computed once per process, not refreshed on each call;
    source edits require a new process to obtain a new key.
    """
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DeviceTemplate:
    """Base for configuration objects recognized by the Gluon compiler.

    Subclasses using this cache key initialize _key from their specialization
    settings. The class identity, settings and source digest identify the
    specialization; instance fields are treated as compile-time configuration.
    """

    __triton_builtin__ = True

    @property
    def cache_key(self):
        return f"{type(self).__module__}.{type(self).__name__}:{self._key!r}:{source_key()}"


class device_method:
    """JIT-compile a method with constexpr self and bind it on instance access.

    This descriptor keeps the usual obj.Method(...) syntax while handing calls
    from Gluon code to _BoundDeviceMethod instead of executing the Python body.
    """

    def __init__(self, fn):
        fn.__annotations__["self"] = l.constexpr
        self.jit = g.jit(fn)

    def __get__(self, obj, owner=None):
        return self if obj is None else _BoundDeviceMethod(self, obj)


class _BoundDeviceMethod:
    """Compiler-callable pairing of a JIT method and its configuration object.

    The builtin marker lets the compiler invoke this adapter with its code
    generator. Calls outside Gluon compilation are rejected.
    """

    __triton_builtin__ = True

    def __init__(self, method, obj):
        self.method, self.obj = method, obj

    @property
    def cache_key(self):
        """Identify both the instance specialization and the method's JIT code."""
        return self.obj.cache_key + self.method.jit.cache_key

    def __call__(self, *args, _semantic=None, _generator=None, **kwargs):
        """Compile the method call with the wrapped instance prepended as self."""
        if _generator is None:
            raise TypeError("Device methods must be called inside @gluon.jit")
        return _generator.call_JitFunction(
            self.method.jit, [l.constexpr(_CompileTimeView(self.obj)), *args], kwargs
        )


class _CompileTimeView:
    """Expose configuration fields through the compiler's constexpr interface.

    Numeric fields become Gluon constexpr values. Nested DeviceTemplate objects
    receive the same view, using the wrapping required by the compiler. Other
    attributes, including bound device methods, pass through unchanged.
    """

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
