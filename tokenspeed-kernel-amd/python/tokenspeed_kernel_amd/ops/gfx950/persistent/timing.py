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

"""Clock and stage-stamp helpers for persistent gfx950 kernels."""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.persistent._intrinsics import memrealtime

# s_memrealtime counts a constant 100 MHz clock on gfx950: 10 ns per tick.
REALTIME_TICK_NS = 10


@gluon.jit
def realtime_ticks():
    """Return the 64-bit constant-rate clock (10 ns ticks); uniform per wave
    (``llvm.amdgcn.s.memrealtime``; launch with ``persistent_compile_options``).

    The clock is per GPU: stamps from different GPUs are not comparable.
    """
    return memrealtime()


@gluon.jit
def trace_stamp(trace_ptr, index):
    """Store the clock at ``trace_ptr[index]``.

    A plain store: it adds no wait to the stage it stamps.
    """
    gl.store(trace_ptr + index, realtime_ticks())
