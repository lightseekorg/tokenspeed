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

"""AMD scheduling and wave-level intrinsics from an external LLVM library."""

import hashlib
from functools import lru_cache
from pathlib import Path

from tokenspeed_kernel_amd._triton import tl

_SCHED_LIBRARY_NAME = "tokenspeed_sched"
_SCHED_SYMBOL = "__tokenspeed_sched_barrier0"
_READFIRSTLANE_SYMBOL = "__tokenspeed_readfirstlane_i32"
_SCHED_LIBRARY_PATH = str(Path(__file__).with_name("sched_barrier.ll"))


@lru_cache(maxsize=1)
def _scheduler_library_hash() -> str:
    # File contents are immutable within a running process, like JIT source.
    # A constexpr carries their digest into Triton's compiled-kernel cache key.
    return hashlib.sha256(Path(_SCHED_LIBRARY_PATH).read_bytes()).hexdigest()


def sched_barrier_compile_options() -> dict:
    """Return launch options for kernels using :func:`sched_barrier`.

    The kernel must accept an otherwise unused ``SCHED_LIBRARY_HASH`` constexpr
    so edits to the library invalidate its compiled binary. Merge ``extern_libs``
    with any other device libraries required by the caller.
    """
    return {
        "SCHED_LIBRARY_HASH": _scheduler_library_hash(),
        "extern_libs": {_SCHED_LIBRARY_NAME: _SCHED_LIBRARY_PATH},
    }


@tl.core.extern
def sched_barrier(_semantic):
    """Prevent instruction scheduling across this point; no workgroup sync.

    Emits ``llvm.amdgcn.sched.barrier(0)``. Returns an unused int32 value required
    by the elementwise extern interface. Launch with
    :func:`sched_barrier_compile_options` to link the library and key its content.
    """
    return tl.core.extern_elementwise(
        _SCHED_LIBRARY_NAME,
        _SCHED_LIBRARY_PATH,
        [],
        {(): (_SCHED_SYMBOL, tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@tl.core.extern
def wave_uniform_i32(value, _semantic):
    """Return lane 0's int32 ``value``, marking it wave-uniform (in an SGPR).

    Emits ``llvm.amdgcn.readfirstlane``. Unlike an inline-asm
    ``v_readfirstlane_b32``, the backend sees the instruction and inserts the
    wait states needed after a VALU write of its source; the inline-asm form
    can read a stale VGPR on gfx950. Launch with
    :func:`sched_barrier_compile_options`.
    """
    return tl.core.extern_elementwise(
        _SCHED_LIBRARY_NAME,
        _SCHED_LIBRARY_PATH,
        [value],
        {(tl.int32,): (_READFIRSTLANE_SYMBOL, tl.int32)},
        is_pure=True,
        _semantic=_semantic,
    )


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
    ``sched_barrier.ll`` exist.
    """
    mask = _sched_group_mask(mask)
    size = tl.core._unwrap_if_constexpr(size)
    return tl.core.extern_elementwise(
        _SCHED_LIBRARY_NAME,
        _SCHED_LIBRARY_PATH,
        [],
        {(): (f"__tokenspeed_sched_group_barrier_{mask}_{size}", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )
