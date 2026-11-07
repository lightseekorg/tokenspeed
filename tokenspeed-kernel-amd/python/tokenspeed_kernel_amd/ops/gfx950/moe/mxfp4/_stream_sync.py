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


"""Per-wave synchronization helpers of the gfx950 MXFP4 MoE stream kernel.

The stream (``moe_stream.py``) runs independent work per wave inside each
workgroup and hands results between workgroups through memory, while Gluon
models a workgroup as one program. The helpers here state the per-wave facts
Gluon cannot express:

- through ``_stream_sync.ll`` (the backend sees the instruction): the XCD id,
  the trap and a per-wave acquire (the store drain is the scheduling
  library's ``release_drain``);
- as inline asm, where no intrinsic exists: moving a wave-uniform value into
  Gluon's scalar type, an unfoldable zero, a data dependence on a load, the
  LDS exchange counter and its spin, and scalar loads that bypass the scalar
  data cache (``glc``). Each asm use says why.

Kernels using them launch with :func:`stream_sync_compile_options`.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
from tokenspeed_kernel_amd._scheduling import sched_compile_options, wave_uniform_i32
from tokenspeed_kernel_amd._triton import gl, gluon, tl

_LIBRARY_NAME = "tokenspeed_stream_sync"
_LIBRARY_PATH = str(Path(__file__).with_name("_stream_sync.ll"))
_SYMBOL = "__tokenspeed_stream_sync_"


def stream_sync_compile_options() -> dict:
    """Launch options for kernels using these helpers (with the scheduling
    library, whose ``readfirstlane`` and ``release_drain`` they also use)."""
    libs = dict(sched_compile_options()["extern_libs"])
    libs[_LIBRARY_NAME] = _LIBRARY_PATH
    return {"extern_libs": libs}


def _call(symbol, ret, args, types, semantic):
    return tl.core.extern_elementwise(
        _LIBRARY_NAME,
        _LIBRARY_PATH,
        args,
        {types: (_SYMBOL + symbol, ret)},
        is_pure=False,
        _semantic=semantic,
    )


@tl.core.extern
def xcc_id(_semantic=None):
    """The XCD running this wave (``HW_REG_XCC_ID``, 4 bits), int32."""
    return _call("xcc_id", tl.int32, [], (), _semantic)


@tl.core.extern
def trap(_semantic=None):
    """Raise a queue error (``s_trap 2``); never returns. The host sees the
    error at its next synchronization with the device."""
    return _call("trap", tl.int32, [], (), _semantic)


@tl.core.extern
def wave_acquire(_semantic=None):
    """An acquire ordering point for this wave alone (no instruction): loads
    after it stay after it in the IR."""
    return _call("wave_acquire", tl.int32, [], (), _semantic)


@gluon.constexpr_function
def _scalar_move_asm(outputs, inputs):
    # A thread may hold several (equal) elements; the first one is moved.
    ((out,),), (values,) = outputs, inputs
    return f"s_mov_b32 {out}, {values[0]}"


@gluon.jit
def wave_scalar(x):
    """This wave's first lane of the int32 tensor ``x`` (uniform within each
    wave) as a per-wave scalar in an SGPR.

    Every wave gets its own value while Gluon treats the result as one value
    for the workgroup, so use it only for addresses, trip counts and branches
    whose bodies hold no workgroup-wide operation (reduction across waves,
    layout conversion through LDS, barrier). The read is
    ``llvm.amdgcn.readfirstlane``, so the compiler places the wait states its
    memory consumers need.
    """
    # asm: Gluon has no op that turns a wave-uniform tensor into a scalar;
    # the asm only moves the (already scalar) value.
    return gl.inline_asm(
        _scalar_move_asm, ("=s", "s"), [wave_uniform_i32(x)], gl.int32, is_pure=True
    )


@gluon.jit
def opaque_zero():
    """0 in an SGPR the compiler cannot fold. Indexing a ``[1, ...]`` shared
    view with it keeps loop-invariant LDS reads inside a loop instead of
    hoisting every iteration's reads into registers."""
    # asm: a value the compiler can neither fold nor hoist; no intrinsic
    # gives one.
    return gl.inline_asm("s_mov_b32 $0, 0", "=s", [], gl.int32, is_pure=False)


@gluon.jit
def after(x, dep):
    """The int32 tensor ``x``, available only once ``dep`` (same shape and
    layout, int32) is.

    The backend waits for ``dep``'s load before the asm, and everything
    computed from the result follows that wait. A wave's vector memory
    operations complete in issue order, so when ``dep`` was loaded after the
    wave's payload stores, a store of the result follows them. That covers
    this wave's stores only.
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
    """Store 32-bit ``values`` at the byte ``offsets`` into the shared buffer
    ``data``, then add 1 per lane to the word at the lane's first
    ``counters`` byte offset into the shared buffer ``count`` (a wave adds
    64). ``offsets``, ``values`` and ``counters`` share shape and layout,
    with at least two elements per lane. Returns the counter before this
    lane's add, shaped like ``offsets``.

    No barrier joins the writer and its readers (:func:`lds_spin_until` in
    other waves of the workgroup): one wave's LDS operations complete in
    issue order, so a reader that sees the count also sees the stores.
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
    """Wait until the word at byte ``counter`` into the shared buffer
    ``count`` reaches ``target`` (unsigned). ``counter`` and ``target`` are
    int32 tensors with one element per lane.

    Reads LDS only (``lgkmcnt``), so the wave's vector loads stay in flight;
    later shared-memory reads stay after it (memory clobber). The writers are
    waves of this workgroup, resident with it, that never wait on this wave
    before their increment, so the spin ends.
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


@gluon.jit
def scalar_load_glc_i64(word_ptr):
    """Load an int64 with a scalar load that bypasses the scalar data cache
    (``glc``). It counts in ``lgkmcnt``, so it never waits behind the wave's
    vector loads, and its result never shares registers with its address
    (early clobber), so the load cannot overwrite its own base in flight."""
    # asm: no scalar load with glc and memory semantics exists: a volatile
    # constant-space load is an s_load without glc (it reads the scalar
    # cache), and llvm.amdgcn.s.buffer.load is readnone (hoisted out of a
    # loop). The wait is the load's own result.
    return gl.inline_asm(
        "s_load_dwordx2 $0, $1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)", "=&s,s,~{memory}",
        [word_ptr], gl.int64, is_pure=False,
    )  # fmt: skip


# OR of (tag_i ^ tag) over s64..s79 into $0, after the 16 tags were loaded.
_TAGS16_ASM = gl.constexpr(
    "s_load_dwordx16 s[64:79], $1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)\n\t"
    + "s_xor_b32 $0, s64, $2"
    + "".join(
        f"\n\ts_xor_b32 s80, s{64 + i}, $2\n\ts_or_b32 $0, $0, s80"
        for i in range(1, 16)
    )
)


@gluon.jit
def _tags16_differ(word_ptr, tag):
    # One s_load_dwordx16 of the 16 int32 tags at ``word_ptr`` (64 B
    # aligned, so it never straddles a line) past the scalar data cache;
    # returns the OR of tag_i ^ tag (0: all 16 match).
    # asm: as scalar_load_glc_i64, for 16 tags compared in scalar registers.
    return gl.inline_asm(
        _TAGS16_ASM, "=&s,s,s,~{memory},~{scc},~{s[64:79]},~{s80}",
        [word_ptr, tag], gl.int32, is_pure=False,
    )  # fmt: skip


@gluon.jit
def wait_tags16(word_ptr, tag):
    """Wait until all 16 int32 tags at ``word_ptr`` (64 B aligned) equal
    ``tag``, with scalar loads only, so the wave's vector loads stay in
    flight. Per wave, ending with :func:`wave_acquire`: the waiting wave
    itself must be the one that reads what the tags guard.

    The words must be in uncached memory (:class:`UncachedBuffer`) when other
    XCDs write them: a scalar ``glc`` load still reads this XCD's L2, which
    does not see another XCD's stores to cached memory.
    """
    differ = _tags16_differ(word_ptr, tag)
    while differ != 0:
        differ = _tags16_differ(word_ptr, tag)
    wave_acquire()


# hipExtMallocWithFlags flag: device memory no cache keeps lines of.
_HIP_DEVICE_MALLOC_UNCACHED = 0x3


def _hip() -> ctypes.CDLL:
    bundled = os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
    return ctypes.CDLL(bundled if os.path.exists(bundled) else "libamdhip64.so")


class UncachedBuffer:
    """A zeroed uncached device allocation, passed to kernels like a tensor
    (``data_ptr()`` and ``dtype``).

    Each XCD has its own L2, which does not snoop the others: a line one XCD
    cached can still be read there after another XCD wrote the word. Uncached
    memory (``hipDeviceMallocUncached``) is never kept in any cache, so a
    scalar ``glc`` load sees a word another XCD stored. Every access goes to
    memory, so keep only small flags here. Free it with :meth:`free` (torch
    does not count references to it).

    Args:
        numel: Elements of ``dtype``.
        dtype: The element type kernels see.
        device: The GPU.
    """

    def __init__(self, numel: int, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype = dtype
        self.device = torch.device(device)
        self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
        self._hip = _hip()
        pointer = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            status = self._hip.hipExtMallocWithFlags(
                ctypes.byref(pointer),
                ctypes.c_size_t(self.nbytes),
                ctypes.c_uint(_HIP_DEVICE_MALLOC_UNCACHED),
            )
            if status != 0:
                raise RuntimeError(
                    f"hipExtMallocWithFlags(uncached) failed ({status}) for "
                    f"{self.nbytes} bytes"
                )
            self._pointer: int = pointer.value
            status = self._hip.hipMemset(
                ctypes.c_void_p(self._pointer), 0, ctypes.c_size_t(self.nbytes)
            )
            if status != 0:
                raise RuntimeError(f"hipMemset failed ({status})")
            # hipMemset may return before the zeros land; finish it so the
            # first launch, on whatever stream, reads zeroed words.
            status = self._hip.hipDeviceSynchronize()
            if status != 0:
                raise RuntimeError(f"hipDeviceSynchronize failed ({status})")

    def data_ptr(self) -> int:
        return self._pointer

    def ptr_range(self) -> int:
        """Bytes a kernel may address from :meth:`data_ptr` (a launch
        specializes buffers within 2 GB for 32-bit buffer offsets)."""
        return self.nbytes

    def free(self) -> None:
        if self._pointer:
            self._hip.hipFree(ctypes.c_void_p(self._pointer))
            self._pointer = 0
