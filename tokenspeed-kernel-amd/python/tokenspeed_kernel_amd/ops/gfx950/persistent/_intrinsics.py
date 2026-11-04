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
  ``HW_REG_XCC_ID``, the trap, the release drain, paced ``vmcnt(N)`` waits
  and a per-wave acquire;
- as inline asm, where no intrinsic exists or the intrinsic is unusable: the
  per-wave vote, the wave index, scalar returning atomics (the launch
  entry's counters), scalar loads past the scalar cache (``s.buffer.load`` is
  declared not to read memory, so LLVM would hoist it out of a poll loop)
  and ``s_dcache_inv``.

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


@tl.core.extern
def _first_lane(x, _semantic=None):
    """``llvm.amdgcn.readfirstlane`` of every element of the int32 tensor
    ``x`` (uniform per wave, still a tensor)."""
    return _call("readfirstlane", tl.int32, [x], (tl.int32,), _semantic)


@gluon.jit
def first_lane_scalar(x):
    """This wave's first lane of the int32 tensor ``x`` (one element per
    lane, uniform within each wave) as a per-wave scalar in an SGPR.

    The read is the LLVM intrinsic, so the compiler sees a VALU write of the
    SGPR and inserts the wait states its memory consumers need; the asm
    only moves the (already scalar) value into Gluon's scalar type."""
    return gl.inline_asm(
        _scalar_move_asm, ("=s", "s"), [_first_lane(x)], gl.int32, is_pure=True
    )


@gluon.constexpr_function
def _scalar_move_asm(outputs, inputs):
    # A thread may hold several (equal) elements; the first one is moved.
    ((out,),), (values,) = outputs, inputs
    return f"s_mov_b32 {out}, {values[0]}"


@gluon.jit
def wave_index():
    """This wave's index in the workgroup, as a per-wave scalar (SGPR)."""
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), layout)
    return first_lane_scalar(lane // 64)


@gluon.jit
def scalar_load_glc(word_ptr):
    """Load an int32 with a scalar load past the scalar cache (``glc``).

    Counts in ``lgkmcnt``: it never waits behind the wave's vector loads.
    ``glc`` reads see same-XCD stores and uncached memory (measured); words
    written on another XCD to coarse memory are unmeasured. The result never
    shares registers with the address (early clobber), so the load cannot
    overwrite its own base while it is in flight."""
    return gl.inline_asm(
        "s_load_dword $0, $1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)", "=&s,s,~{memory}",
        [word_ptr], gl.int32, is_pure=False,
    )  # fmt: skip


@gluon.jit
def scalar_load_glc_i64(word_ptr):
    """Load an int64 with a scalar load past the scalar cache (``glc``).

    Counts in ``lgkmcnt``, like :func:`scalar_load_glc` (and likewise never
    shares registers with its address)."""
    return gl.inline_asm(
        "s_load_dwordx2 $0, $1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)", "=&s,s,~{memory}",
        [word_ptr], gl.int64, is_pure=False,
    )  # fmt: skip


# Eight lines of four tagged pairs (value low, tag high): loads into s16..s79,
# the OR of (tag ^ $5) into $4 (s80 scratch), and the four sums in line order
# into $0..$3. Line k holds pair i's value in s(16 + 8k + 2i), its tag in the
# next register.
@gluon.constexpr_function
def _pair_sums8_text(stride):
    loads = "\n\t".join(
        f"s_load_dwordx8 s[{16 + 8 * k}:{23 + 8 * k}], $6, {hex(k * stride)} glc"
        for k in range(8)
    )
    tags = ["s_xor_b32 $4, s17, $5"]
    for k in range(8):
        for i in range(4):
            if k or i:
                reg = 16 + 8 * k + 2 * i + 1
                tags += [f"s_xor_b32 s80, s{reg}, $5", "s_or_b32 $4, $4, s80"]
    sums = []
    for i in range(4):
        sums.append(f"v_mov_b32 ${i}, s{16 + 2 * i}")
        sums += [f"v_add_f32 ${i}, s{16 + 8 * k + 2 * i}, ${i}" for k in range(1, 8)]
    return "\n\t".join([loads, "s_waitcnt lgkmcnt(0)"] + tags + sums)


_PAIR_SUMS8_CONSTRAINTS = gl.constexpr(
    "=&v,=&v,=&v,=&v,=&s,s,s,~{memory},~{scc},~{s[16:79]},~{s80}"
)


@gluon.jit
def scalar_pair_sums8(first_ptr, tag, STRIDE: gl.constexpr):
    """Read eight lines of four tagged int64 pairs (``tag << 32 | value``)
    at ``first_ptr + k * STRIDE`` bytes (k = 0..7, each line 32 B aligned)
    with scalar loads past the scalar cache (``glc``: same-XCD words), so the
    read never waits behind the wave's vector loads.

    Returns ``(sum_0, sum_1, sum_2, sum_3, differ)``, per wave: ``sum_i`` the
    FP32 sum of the lines' pair-i values added in line order 0..7
    (``((v_0 + v_1) + v_2) + ...``), ``differ`` the OR of ``tag_ik ^ tag``
    (0: all 32 tags match, and the sums are of current values).
    Clobbers s[16:80]."""
    TEXT: gl.constexpr = _pair_sums8_text(STRIDE)
    return gl.inline_asm(
        TEXT, _PAIR_SUMS8_CONSTRAINTS, [tag, first_ptr],
        [gl.float32, gl.float32, gl.float32, gl.float32, gl.int32], is_pure=False,
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
