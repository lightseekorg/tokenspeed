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

"""Per-wave event timestamps inside a megakernel.

A kernel profiler sees one launch. This module shows where the time goes
inside the launch: when each wave of each CTA starts a stage, finishes
polling, and finishes computing. Stamps are ``s_memrealtime`` (a constant
100 MHz clock shared by all CUs), so CTAs on different XCDs line up on one
timeline.

A stamp must not perturb what it measures. On gfx9 ``vmcnt`` counts stores
as well as loads, so a global store per stamp would make later load waits
also wait for the stamp. Stamps therefore go to an LDS buffer
(``ds_write_b64``, which ``lgkmcnt`` counts). At the end of the kernel,
:func:`flush` copies the LDS buffer to the trace buffer once.

The trace buffer is ``int64 [grid, EVENTS, num_warps]``. 0 means "not
reached". Instrumentation compiles away when the kernel's trace buffer is the
constexpr ``None``.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon

EVENTS = gl.constexpr(128)  # 4 points of up to 32 stages
# The points of a stage that each of its tasks stamps (Mailbox.stamp).
START = gl.constexpr(0)  # the task starts
READY = gl.constexpr(1)  # the task polled its inputs
COMPUTED = gl.constexpr(2)  # the task computed its result
END = gl.constexpr(3)  # the task published its outputs
TICK_NS = 10  # s_memrealtime period


@gluon.constexpr_function
def disabled(buf):
    """Whether ``buf`` is the "no trace" ``None`` (plain or constexpr)."""
    return buf is None or (isinstance(buf, gl.constexpr) and buf.value is None)


@gluon.jit
def now():
    """Wave-uniform ``s_memrealtime`` value."""
    return gl.inline_asm(
        "s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s", [], gl.int64, is_pure=False
    )


@gluon.constexpr_function
def _table_layout(num_warps):
    return gl.BlockedLayout([1, 1], [64, 1], [1, num_warps], [0, 1])


@gluon.jit
def allocate(buf):
    """The CTA's zeroed LDS stamp table, or None when ``buf`` is None."""
    if disabled(buf):
        table = gl.constexpr(None)
    else:
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        table = gl.allocate_shared_memory(gl.int64, [EVENTS, gl.num_warps()], flat)
        layout: gl.constexpr = _table_layout(gl.num_warps())
        table.store(gl.zeros([EVENTS, gl.num_warps()], gl.int64, layout))
        gl.barrier()
    return table


@gluon.constexpr_function
def _stamp_asm(outputs, inputs):
    (stamp_v,), (address,), (stamp_s,) = outputs
    (base,), (offset,) = inputs
    # Every lane of the wave writes the wave's stamp to the wave's slot.
    return "\n".join(
        [
            f"s_memrealtime {stamp_s}",
            f"v_add_u32 {address}, {base}, {offset}",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b64 {stamp_v}, {stamp_s}",
            f"ds_write_b64 {address}, {stamp_v}",
        ]
    )


@gluon.jit
def stamp(table, event: gl.constexpr):
    """Record this wave's time of ``event`` in the CTA's LDS stamp table."""
    if not disabled(table):
        lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        i = gl.arange(0, 64 * gl.num_warps(), lanes)
        offset = (event * gl.num_warps() + i // 64) * 8
        gl.inline_asm(
            _stamp_asm,
            "=&v,=&v,=&s,s,v",
            [table, offset],
            [offset.type.with_element_ty(gl.int64), offset.type, gl.int64],
            is_pure=False,
        )


@gluon.jit
def flush(table, buf):
    """Copy the CTA's stamps to ``buf``. Call it once, at the end of the kernel."""
    if not disabled(table):
        gl.barrier()
        layout: gl.constexpr = _table_layout(gl.num_warps())
        e = gl.arange(0, EVENTS, gl.SliceLayout(1, layout))
        w = gl.arange(0, gl.num_warps(), gl.SliceLayout(0, layout))
        slot = (gl.program_id(0) * EVENTS + e[:, None]) * gl.num_warps() + w[None, :]
        gl.store(buf + slot, table.load(layout))


__all__ = ["EVENTS", "TICK_NS", "allocate", "disabled", "flush", "now", "stamp"]
