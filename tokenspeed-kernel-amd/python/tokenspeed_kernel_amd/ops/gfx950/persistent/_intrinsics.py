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


"""Escape hatches of persistent gfx950 kernels: LLVM intrinsics and inline asm.

Everything Gluon cannot express for persistent synchronization lives here,
one named helper per instruction pattern, each documented with the compiler
change that would remove it:

- through ``intrinsics.ll`` (the backend sees the instruction): the clock,
  ``HW_REG_XCC_ID``, the trap, the release drain, paced ``vmcnt(N)`` waits and
  the lane-0 returning atomic of the launch entry, and a per-wave acquire;
- as inline asm, where no intrinsic exists or the intrinsic is unusable: the
  per-wave vote, scalar loads past the scalar cache (``s.buffer.load`` is
  declared not to read memory, so LLVM would hoist it out of a poll loop),
  ``s_dcache_inv`` and the 64-bit claim compare pinned below the role's loads.

Kernels calling the intrinsics launch with :func:`persistent_compile_options`.
"""

from pathlib import Path

from tokenspeed_kernel_amd._scheduling import sched_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon, tl

_LIBRARY_NAME = "tokenspeed_persistent"
_LIBRARY_PATH = str(Path(__file__).with_name("intrinsics.ll"))
_SYMBOL = "__tokenspeed_persistent_"
# vmcnt(N) helpers exist for 1 <= N <= MAX_VMCNT_WAIT (release_drain is N = 0).
MAX_VMCNT_WAIT = 15


def persistent_compile_options() -> dict:
    """Launch options for kernels using these helpers (with the scheduling
    library, which persistent kernels also use). Merge ``extern_libs`` with
    any other device libraries the caller needs."""
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
def memrealtime(_semantic=None):
    """The constant 100 MHz clock (``s_memrealtime``), int64, uniform per wave."""
    return _call("memrealtime", tl.int64, [], (), _semantic)


@tl.core.extern
def xcc_id(_semantic=None):
    """The XCD running this wave (``HW_REG_XCC_ID``, 4 bits), int32."""
    return _call("xcc_id", tl.int32, [], (), _semantic)


@tl.core.extern
def trap(_semantic=None):
    """Raise a queue error (``s_trap 2``); never returns."""
    return _call("trap", tl.int32, [], (), _semantic)


@tl.core.extern
def release_drain(_semantic=None):
    """Wait for this wave's vector memory operations (``vmcnt(0)``), keeping
    earlier stores above the wait in the IR; no L2 write-back.

    Follow it with ``gl.barrier()`` when several waves stored, then publish.
    It also waits for loads the wave issued earlier (one in-order counter).
    """
    return _call("release_drain", tl.int32, [], (), _semantic)


@tl.core.extern
def wave_acquire(_semantic=None):
    """An acquire ordering point for this wave alone (no instruction): loads
    after it stay after it in the IR. Ends a per-wave poll whose result only
    this wave uses, in place of a workgroup barrier."""
    return _call("wave_acquire", tl.int32, [], (), _semantic)


@tl.core.extern
def wait_vmcnt(n, _semantic=None):
    """Wait until at most ``n`` (constexpr, 1-15) of this wave's vector memory
    operations are outstanding: the paced-issue wait."""
    n = tl.core._unwrap_if_constexpr(n)
    if not 1 <= n <= MAX_VMCNT_WAIT:
        raise ValueError(f"wait_vmcnt({n}): 1 <= n <= {MAX_VMCNT_WAIT}")
    return _call(f"vmcnt_{n}", tl.int32, [], (), _semantic)


@tl.core.extern
def lane0_add_i64(address, value, lane, _semantic=None):
    """Lane 0 adds ``value`` to the int64 at ``address`` (an int64 holding a
    global address), device scope, relaxed, returning the old value; other
    lanes get an undefined value. The wait for the result is placed at its
    first use. Pass a lane-indexed address (see ``intrinsics.ll``)."""
    return _call(
        "lane0_add_i64", tl.int64, [address, value, lane],
        (tl.int64, tl.int64, tl.int32), _semantic,
    )  # fmt: skip


@gluon.jit
def vote(flags):
    """Whether any lane of this wave holds a nonzero int32 flag, as a per-wave
    scalar. One ``v_cmp`` writes the lane mask to an SGPR pair; Gluon has no
    wave-level vote with a scalar result (an elementwise ballot stays a tensor,
    and reducing it crosses waves). ``flags``: exactly one int32 per lane."""
    mask = gl.inline_asm(
        "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v,~{memory}", [flags], gl.int64,
        is_pure=False,
    )  # fmt: skip
    return mask != 0


@gluon.jit
def lane_mask_differs_i64(a, b):
    """The lane mask of ``a != b`` (int64 per lane) as a per-wave scalar.

    A side-effecting asm: the compiler cannot move it above vector loads
    issued before it, so its wait for ``a`` and ``b`` counts only the
    operations issued after them (a plain compare was hoisted above the
    loads and forced ``vmcnt(0)``)."""
    return gl.inline_asm(
        "v_cmp_ne_u64_e64 $0, $1, $2", "=s,v,v,~{memory}", [a, b], gl.int64,
        is_pure=False,
    )  # fmt: skip


@gluon.jit
def lane_mask(flags):
    """The lane mask of ``flags != 0`` (int32 per lane) as a per-wave scalar."""
    return gl.inline_asm(
        "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v,~{memory}", [flags], gl.int64,
        is_pure=False,
    )  # fmt: skip


@gluon.jit
def scalar_load_glc(word_ptr):
    """Load an int32 with a scalar load past the scalar cache (``glc``).

    Counts in ``lgkmcnt``: it never waits behind the wave's vector loads.
    ``glc`` reads see same-XCD stores and uncached memory [measured, MB-S];
    words written on another XCD to coarse memory are unmeasured."""
    return gl.inline_asm(
        "s_load_dword $0, $1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,s,~{memory}",
        [word_ptr], gl.int32, is_pure=False,
    )  # fmt: skip


# OR of (tag_i ^ tag) over s64..s79 into $0, after the 16 tags were loaded.
_XOR_TAGS16 = "s_xor_b32 $0, s64, $2" + "".join(
    f"\n\ts_xor_b32 s80, s{64 + i}, $2\n\ts_or_b32 $0, $0, s80" for i in range(1, 16)
)
_LOAD_TAGS16 = "s_load_dwordx16 s[64:79], $1, 0x0{}\n\ts_waitcnt lgkmcnt(0)\n\t"
_TAGS16_GLC = gl.constexpr(_LOAD_TAGS16.format(" glc") + _XOR_TAGS16)
_TAGS16_CACHED = gl.constexpr(
    "s_dcache_inv\n\t" + _LOAD_TAGS16.format("") + _XOR_TAGS16
)
_TAGS16_CONSTRAINTS = gl.constexpr("=&s,s,s,~{memory},~{scc},~{s[64:79]},~{s80}")


@gluon.jit
def scalar_tags16_differ(word_ptr, tag, GLC: gl.constexpr):
    """One ``s_load_dwordx16`` of 16 int32 tags at ``word_ptr`` (64 B aligned,
    so it never straddles a line); returns the OR of ``tag_i ^ tag`` (0: all
    16 match). ``GLC``: past the scalar cache (``glc``), or through it after
    ``s_dcache_inv``. Clobbers s[64:80]."""
    if GLC:
        return gl.inline_asm(
            _TAGS16_GLC, _TAGS16_CONSTRAINTS, [word_ptr, tag], gl.int32, is_pure=False
        )
    else:
        return gl.inline_asm(
            _TAGS16_CACHED, _TAGS16_CONSTRAINTS, [word_ptr, tag], gl.int32,
            is_pure=False,
        )  # fmt: skip
