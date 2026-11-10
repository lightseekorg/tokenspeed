# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AMD intrinsics Gluon does not expose, from external LLVM IR libraries.

Each category is one library, ``_intrinsics/_<category>.ll``: ``arith``
holds floating-point arithmetic controls, ``isched`` instruction scheduling
hints, ``sync`` memory synchronization and ``wave`` wave-level operations.
Kernels calling any function here launch
with :func:`extern_compile_options`. For a plain scheduling barrier use
``gl.amd.hint.sched_barrier()``.
"""

from pathlib import Path

from tokenspeed_kernel_amd._triton import tl

_INTRINSICS_DIRECTORY = Path(__file__).with_name("_intrinsics")

# Library name -> LLVM IR file of one category.
_LIBRARIES = {
    f"tokenspeed_{category}": str(_INTRINSICS_DIRECTORY / f"_{category}.ll")
    for category in ("arith", "isched", "sync", "wave")
}


def extern_compile_options() -> dict:
    """Return launch options for kernels using this module's intrinsics.

    Every library is listed: Triton links only those a kernel calls into and
    keys compiled kernels on the contents of all of them. Merge
    ``extern_libs`` with any other device libraries required by the caller.
    """
    return {"extern_libs": dict(_LIBRARIES)}


def _call(category, function, args, arg_types, ret_type, is_pure, _semantic):
    # Triton links a library into a kernel only when the kernel calls a
    # symbol containing the library name, hence the symbol prefix.
    library = f"tokenspeed_{category}"
    return tl.core.extern_elementwise(
        library,
        _LIBRARIES[library],
        args,
        {arg_types: (f"__{library}_{function}", ret_type)},
        is_pure=is_pure,
        _semantic=_semantic,
    )


# Floating-point arithmetic (_arith.ll).


@tl.core.extern
def arithmetic_fence(value, _semantic):
    """Return the FP32 ``value`` as one LLVM must treat as new
    (``llvm.arithmetic.fence``; no instruction): arithmetic on the result is
    neither merged with the same arithmetic on ``value`` nor reassociated
    across this point. Launch with :func:`extern_compile_options`.
    """
    return _call(
        "arith",
        "fence_f32",
        [value],
        (tl.float32,),
        tl.float32,
        is_pure=True,
        _semantic=_semantic,
    )


# Instruction scheduling (_isched.ll).

# Instruction classes of ``llvm.amdgcn.sched.group.barrier`` masks.
_SCHED_GROUP_CLASSES = {
    "alu": 0x1,
    "valu": 0x2,
    "salu": 0x4,
    "mfma": 0x8,
    "vmem": 0x10,
    "vmem_read": 0x20,
    "vmem_write": 0x40,
    "ds": 0x80,
    "ds_read": 0x100,
    "ds_write": 0x200,
    "trans": 0x400,
}


def _sched_group_mask(mask) -> int:
    # A class name or a tuple of them; the classes are OR-ed together.
    mask = tl.core._unwrap_if_constexpr(mask)
    if isinstance(mask, tl.core.tuple):
        mask = mask.values
    if isinstance(mask, str):
        mask = (mask,)
    bits = 0
    for name in mask:
        name = tl.core._unwrap_if_constexpr(name)
        if name not in _SCHED_GROUP_CLASSES:
            raise ValueError(
                f"unknown sched_group class {name!r}; "
                f"expected one of {sorted(_SCHED_GROUP_CLASSES)}"
            )
        bits |= _SCHED_GROUP_CLASSES[name]
    return bits


@tl.core.extern
def sched_group(mask, size, _semantic):
    """Emit ``llvm.amdgcn.sched.group.barrier(mask, size, 0)``.

    ``mask`` names the instruction class, or a tuple of classes the group
    accepts: ``"alu"``, ``"valu"``, ``"salu"``, ``"mfma"``, ``"vmem"``,
    ``"vmem_read"``, ``"vmem_write"``, ``"ds"``, ``"ds_read"``, ``"ds_write"``
    and ``"trans"``. ``size`` is how many instructions of it the group takes.
    A sequence of these after a region's instructions pins their interleave.
    Direct-to-LDS ``buffer_load ... lds`` matches ``"vmem"`` but not
    ``"vmem_read"``. Only the (mask, size) pairs defined in
    ``_intrinsics/_isched.ll`` exist. Launch with
    :func:`extern_compile_options`.
    """
    mask = _sched_group_mask(mask)
    size = tl.core._unwrap_if_constexpr(size)
    return _call(
        "isched",
        f"group_barrier_{mask}_{size}",
        [],
        (),
        tl.int32,
        is_pure=False,
        _semantic=_semantic,
    )


# Memory synchronization (_sync.ll).


@tl.core.extern
def release_drain(_semantic):
    """Wait for this wave's vector memory operations (``s_waitcnt vmcnt(0)``),
    keeping its earlier stores above the wait; no L2 write-back.

    Only this wave's operations are waited for: follow it with
    ``gl.barrier()`` when every wave's stores must land before one thread
    publishes. Loads the wave issued earlier are waited for too (one
    in-order counter). Launch with :func:`extern_compile_options`.
    """
    return _call(
        "sync",
        "release_drain",
        [],
        (),
        tl.int32,
        is_pure=False,
        _semantic=_semantic,
    )


# Wave-level operations (_wave.ll).


@tl.core.extern
def wave_uniform_i32(value, _semantic):
    """Return lane 0's int32 ``value``, marking it wave-uniform (in an SGPR).

    Emits ``llvm.amdgcn.readfirstlane``. Unlike an inline-asm
    ``v_readfirstlane_b32``, the backend sees the instruction and inserts the
    wait states needed after a VALU write of its source; the inline-asm form
    can read a stale VGPR on gfx950. Launch with
    :func:`extern_compile_options`.
    """
    return _call(
        "wave",
        "readfirstlane_i32",
        [value],
        (tl.int32,),
        tl.int32,
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def wave_ballot_i64(predicate, _semantic):
    """Return an int64 mask of active lanes with nonzero int32 ``predicate``.

    Bit l represents lane l. Launch with :func:`extern_compile_options`.
    """
    return _call(
        "wave",
        "ballot_i64",
        [predicate],
        (tl.int32,),
        tl.int64,
        is_pure=False,
        _semantic=_semantic,
    )


@tl.core.extern
def wave_popcount_i64(mask, _semantic):
    """Return the int32 number of set bits in int64 ``mask``.

    Uses ``llvm.ctpop``. Launch with :func:`extern_compile_options`.
    """
    return _call(
        "wave",
        "popcount_i64",
        [mask],
        (tl.int64,),
        tl.int32,
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def wave_mbcnt(mask, _semantic):
    """Return the int32 count of set bits in int64 ``mask`` below this lane.

    Counts across a wave of 64 lanes. Launch with :func:`extern_compile_options`.
    """
    return _call(
        "wave",
        "mbcnt",
        [mask],
        (tl.int64,),
        tl.int32,
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def wave_all_equal(a, b, _semantic):
    """1 where ``a == b`` (FP32, ordered) holds in every active lane of the
    wave, else 0, as an int32 tensor whose value is the same across the
    wave (in an SGPR, so a branch on it is a scalar branch). One compare
    into a lane mask (``llvm.amdgcn.ballot``) tested against the active
    lanes. Launch with :func:`extern_compile_options`.
    """
    return _call(
        "wave",
        "all_equal_f32",
        [a, b],
        (tl.float32, tl.float32),
        tl.int32,
        is_pure=False,
        _semantic=_semantic,
    )
