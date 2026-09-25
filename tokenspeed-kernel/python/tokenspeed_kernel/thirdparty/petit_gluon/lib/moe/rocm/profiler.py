"""Native cycle-counter profiling helpers."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import _native_call
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class ClockProfiler(DeviceTemplate):
    _key = "ClockProfiler"

    @device_method
    def Start(self):
        return _native_call("llvm.readcyclecounter", "i64", (), (), False)

    @device_method
    def End(self, start):
        return _native_call("llvm.readcyclecounter", "i64", (), (), False) - start


class NoopProfiler(DeviceTemplate):
    _key = "NoopProfiler"

    @device_method
    def Start(self):
        return l.full((), 0, l.uint64)

    @device_method
    def End(self, start):
        return l.full((), 0, l.uint64)


def Profiler(kProfile):
    return ClockProfiler() if kProfile else NoopProfiler()


@g.jit
def RecordProfile(
    profile, counter, value, tid, block, kProfile: l.constexpr, kStride: l.constexpr
):
    if kProfile:
        if tid == 0 and profile.to(l.uint64) != 0:
            l.store(profile + counter * kStride + block, value)
