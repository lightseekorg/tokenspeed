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


"""Per-wave escape hatches for persistent gfx950 kernels.

A persistent stream runs independent work per wave inside one workgroup,
while Gluon models a workgroup as one program. The helpers here state the
per-wave facts Gluon cannot express, until Gluon can state them itself.

- :func:`wave_scalar`: a value uniform within a wave but different across
  waves, in an SGPR (``llvm.amdgcn.readfirstlane``, visible to the
  compiler's hazard checks). Use it for addresses, trip counts and branches whose
  bodies hold no workgroup-wide operation (reduction across waves, layout
  conversion through LDS, barrier).
- :func:`opaque_zero`: an SGPR zero the compiler cannot fold. Indexing a
  ``[1, ...]`` shared view with it keeps loop-invariant LDS reads inside a
  loop instead of hoisting every iteration's reads into registers.
- :func:`after`: a value made data-dependent on another, so a store of it is
  issued only after the wave waited for the other's load.
- :func:`issue_order`: a point vector memory operations do not cross, so a
  wave issues the ones before it first (the queue then completes them first).
- :func:`lds_store_count`, :func:`lds_spin_until` and
  :func:`lds_spin_bounded`: an LDS exchange among
  the waves of a workgroup that the compiler does not model, so it adds no
  barrier. The waves share LDS and one wave's LDS operations complete in
  issue order: a wave that sees a counter value also sees every store its
  writers made before their increments.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.persistent._intrinsics import first_lane_scalar


@gluon.jit
def wave_scalar(x):
    """Return this wave's first lane of the int32 tensor ``x`` as a scalar.

    ``x`` must be uniform within each wave. Every wave gets its own value,
    while Gluon treats the result as one value for the workgroup. The read
    is ``llvm.amdgcn.readfirstlane`` (:func:`first_lane_scalar`), so the
    compiler places the wait states between it and memory instructions
    that use the result.
    """
    return first_lane_scalar(x)


@gluon.jit
def opaque_zero():
    """Return 0 in an SGPR the compiler cannot constant-fold."""
    # asm: a value the compiler can neither fold nor hoist; no intrinsic
    # gives one.
    return gl.inline_asm("s_mov_b32 $0, 0", "=s", [], gl.int32, is_pure=False)


@gluon.jit
def after(x, dep):
    """Return the int32 tensor ``x``, available only once ``dep`` (same
    shape and layout, int32) is.

    The asm reads ``dep``, so the backend waits for ``dep``'s load before it,
    and everything computed from the result follows that wait. A wave
    publishes a tag with it: when ``dep`` was loaded after the wave's payload
    stores, the wait for it covers those stores, since one wave's vector
    memory operations complete in issue order. That covers this wave's
    stores only; a tag for other waves' stores needs a barrier.
    """
    # asm: a dependence on ``dep``'s load that uses no arithmetic; no
    # intrinsic expresses "after this value" without consuming it.
    return gl.inline_asm_elementwise(
        "v_mov_b32 $0, $1 ; after $2",
        "=v,v,v",
        [x, dep],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def issue_order():
    """Keep this wave's memory operations on their side of this point (an
    asm memory clobber; one ``s_mov``): the compiler may otherwise reorder
    loads and stores to different buffers. One wave's vector memory
    operations complete in issue order, so with :func:`after` a wave can wait
    for the operations before the point while later loads stay in flight.

    Returns an int32 0 the compiler cannot fold: arithmetic that adds it
    stays after the point too (the point orders memory operations only).
    """
    # asm: a compiler memory barrier that also yields an unfoldable 0 (a
    # singlethread fence orders memory but returns nothing to chain on).
    return gl.inline_asm("s_mov_b32 $0, 0", "=s,~{memory}", [], gl.int32, is_pure=False)


@gluon.constexpr_function
def _store_count_asm(outputs, inputs):
    (old,), ((data,), offsets, values, (count,), counters) = outputs, inputs
    lines = []
    for offset, value in zip(offsets, values):
        lines += [f"v_add_u32 {old[0]}, {data}, {offset}"]
        lines += [f"ds_write_b32 {old[0]}, {value}"]
    lines += [
        f"v_add_u32 {old[0]}, {count}, {counters[0]}",
        f"v_mov_b32 {old[-1]}, 1",
        f"ds_add_rtn_u32 {old[0]}, {old[0]}, {old[-1]}",
        "s_waitcnt lgkmcnt(0)",
    ]
    lines += [f"v_mov_b32 {o}, {old[0]}" for o in old[1:]]
    return "\n".join(lines)


@gluon.jit
def lds_store_count(data, offsets, values, count, counters):
    """Store ``values`` (32-bit) at the byte ``offsets`` into the shared
    buffer ``data``, then add 1 per lane to the word at the lane's first
    ``counters`` byte offset into the shared buffer ``count``, so a wave adds
    64. ``offsets``, ``values`` and ``counters`` share shape and layout, with
    at least two elements per lane.

    Returns the counter before this lane's add, shaped like ``offsets``.

    Wave independence: no barrier joins the writer and its readers. A
    reader that sees the count also sees the stores, because one wave's LDS
    operations complete in issue order (stores, then the add); the readers
    are other waves of this workgroup in :func:`lds_spin_until` /
    :func:`lds_spin_bounded`.
    """
    # asm: Gluon has no LDS atomic on a shared-memory descriptor (the
    # ds_add_rtn), and the stores must precede it in this wave's LDS order.
    return gl.inline_asm(
        _store_count_asm,
        ("=&v", "s", "v", "v", "s", "v"),
        [data, offsets, values.to(gl.int32, bitcast=True), count, counters],
        offsets.type,
        is_pure=False,
    )


@gluon.jit
def lds_spin_until(count, counter, target):
    """Wait until the word at byte ``counter`` into the shared buffer ``count`` reaches
    ``target`` (unsigned). ``counter`` and ``target`` are int32 tensors with
    one element per lane, as in ``BlockedLayout([1], [64], [num_warps], [0])``.

    Reads LDS only (``lgkmcnt``), so the wave's vector loads stay in flight.
    Later shared-memory reads stay after it (memory clobber).

    There is no deadline or abort check: the spin is bounded only because the
    waited-for writers are waves of this same workgroup (resident with it)
    that never wait on this one before their increment. Do not use it for a
    wait on another workgroup or another GPU.
    """
    # asm: a shared-memory load in a loop is hoisted (the compiler assumes
    # no other writer); the spin rereads LDS each round.
    gl.inline_asm(
        "v_add_u32 $0, $2, $3\n1:\n\tds_read_b32 $1, $0\n\ts_waitcnt lgkmcnt(0)\n\t"
        "v_cmp_lt_u32 vcc, $1, $4\n\ts_cbranch_vccnz 1b",
        "=&v,=&v,s,v,v,~{vcc},~{memory}",
        [count, counter, target],
        [counter.type, counter.type],
        is_pure=False,
    )


# LDS reads per call of lds_spin_bounded: the caller checks its deadline
# between calls, as often as poll_check expects (every 64 retries).
LDS_SPIN_TRIES = 64
_SPIN_BOUNDED_ASM = gl.constexpr(
    f"v_add_u32 $1, $3, $4\n\ts_mov_b32 $2, {LDS_SPIN_TRIES}\n1:\n\t"
    "ds_read_b32 $0, $1\n\ts_waitcnt lgkmcnt(0)\n\t"
    "v_cmp_lt_u32 vcc, $0, $5\n\ts_cbranch_vccz 2f\n\t"
    "s_add_i32 $2, $2, -1\n\ts_cmp_lg_u32 $2, 0\n\ts_cbranch_scc1 1b\n2:"
)


@gluon.jit
def lds_spin_bounded(count, counter, target):
    """:func:`lds_spin_until` for at most ``LDS_SPIN_TRIES`` reads; returns
    the last value read (one per lane, the same in every lane). The caller
    loops until it reaches ``target``, checking its deadline between calls
    (``poll_check``), so a writer that never arrives fails the launch instead
    of hanging it.
    """
    # asm: as lds_spin_until.
    old, _addr, _left = gl.inline_asm(
        _SPIN_BOUNDED_ASM,
        "=&v,=&v,=&s,s,v,v,~{vcc},~{scc},~{memory}",
        [count, counter, target],
        [counter.type, counter.type, counter.type],
        is_pure=False,
    )
    return old
