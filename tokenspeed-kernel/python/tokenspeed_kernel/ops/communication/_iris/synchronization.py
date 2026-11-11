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

"""Vector-memory waits that Iris selects per architecture through its policy.

Callers handle workgroup synchronization, cache writeback/invalidation, and
system-scope ordering.
"""

from tokenspeed_kernel._triton import gl, gluon


# asm: the LLVM intrinsic form of these waits (tokenspeed_kernel_amd._extern)
# makes the backend's wait insertion add a vector-memory wait after the peer
# flag store or atomic that precedes a rendezvous poll or acquire, delaying the
# first poll until that store completes. An inline asm wait is invisible to
# that pass, so the publish and the first poll overlap as before. The memory
# clobber keeps loads and stores on their side of the wait. Retire these once
# the backend no longer adds that wait for a visible s_waitcnt.
@gluon.jit
def _wait(asm: gl.constexpr):
    gl.inline_asm_elementwise(
        asm=asm,
        constraints="=r,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def wait_vmem():
    """Wait for this subgroup's vector memory operations on one-counter targets.

    Targets such as gfx950 track vector loads and stores with one counter
    (VMcnt).
    """
    _wait("s_waitcnt vmcnt(0)")


# Targets such as gfx1250 track vector-memory loads and stores with separate
# counters (LOADcnt and STOREcnt) and have no VMcnt.
@gluon.jit
def wait_stores():
    """Wait for this subgroup's vector stores and non-returning atomics."""
    _wait("s_wait_storecnt 0x0")


@gluon.jit
def wait_loads_stores():
    """Wait for this subgroup's vector loads and stores.

    Async and tensor copies to LDS (ASYNCcnt and TENSORcnt) are not covered.
    """
    _wait("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0")
