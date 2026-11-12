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

"""Warm a launch's own code into every XCD's L2 at its start.

A monokernel's code (~100 KB) is larger than the instruction cache, and
the weights that stream through the L2 during a launch evict it. So each
stage of the next launch would first wait on its instructions from the
memory-side cache or HBM. Each wait is short, but the stages run one after
another on the critical path, and they add up, the more so when the
binary differs from the previous launch's (a full layer after a sliding
one).

A code task reads its share of the launch's ``.text`` once, at the start,
as data: the loads fill the L2 that instruction fetches then hit. Each XCD
has its own L2, and a launch's CTAs go to the XCDs in turn (CTA ``c`` to
XCD ``c % 8``), so the tasks of each XCD read the whole ``.text``.

The kernel finds its code from its program counter: ``s_getpc_b64`` next to
a marker literal. The host finds the marker in the compiled ``.text``
(:func:`text_range`) and passes its offset and the ``.text`` size as runtime
arguments, so the binary does not depend on them.
"""

from __future__ import annotations

import struct

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import trace

XCDS = gl.constexpr(8)
# The literal of the s_mov_b32 after the kernel's s_getpc_b64. It must occur
# once in a kernel's .text: call code_pointer from one place.
MARKER = 0x5EC0DE01
_GETPC = gl.constexpr("s_getpc_b64 $0\ns_mov_b32 $1, 0x5ec0de01")


@gluon.jit
def code_pointer(offset):
    """The start of this kernel's ``.text`` as an ``int32`` pointer, from
    ``offset`` (:func:`text_range`). Call it from one place in a kernel."""
    # asm: no Gluon op or intrinsic reads the program counter.
    pc, _ = gl.inline_asm(
        _GETPC,
        "=s,=s",
        [],
        [gl.int64, gl.int32],
        is_pure=False,
    )
    return (pc - offset).to(gl.pointer_type(gl.int32), bitcast=True)


@gluon.jit
def code_warm_stage(
    mb, task, TASKS: gl.constexpr, sink, offset, quads, STAGE: gl.constexpr
):
    """Code task ``task`` of ``TASKS`` (a multiple of 8): read its XCD's
    share of the ``4 * quads`` int32 words of this kernel's ``.text``.

    Args:
        sink: An ``int64`` buffer that the loads' values would go to; nothing
            is stored (``quads`` is never negative), but the store keeps the
            loads.
        offset, quads: :func:`text_range` of this kernel.
    """
    gl.static_assert(TASKS % XCDS == 0)
    mb.stamp(STAGE, trace.START)
    code = code_pointer(offset)
    NW: gl.constexpr = gl.num_warps()
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [NW], [0])
    BLOCK: gl.constexpr = NW * 64 * 4
    i = gl.arange(0, BLOCK, layout)
    PARTS: gl.constexpr = TASKS // XCDS
    words = quads * 4
    share = (words + PARTS * BLOCK - 1) // (PARTS * BLOCK) * BLOCK
    first = (task // XCDS) * share
    acc = gl.zeros([BLOCK], gl.int32, layout)
    for k in range(0, share, BLOCK):
        o = first + k + i
        acc = acc ^ gl.load(code + o, mask=o < words, other=0)
    gl.store(sink + i, acc.to(gl.int64), mask=(i < 16) & (quads < 0))
    mb.stamp(STAGE, trace.END)


_TEXT_RANGES: dict[int, tuple[tuple[int, int], object]] = {}


def text_range(kernel) -> tuple[int, int]:
    """``(offset, quads)`` of compiled kernel ``kernel`` (a Triton
    ``CompiledKernel``): the offset in its ``.text`` of the program counter
    that :func:`code_warm_stage` reads, and the ``.text`` size in 16-byte
    quads. Cached per kernel."""
    got = _TEXT_RANGES.get(id(kernel))
    if got is not None:
        return got[0]
    elf = kernel.asm["hsaco"]
    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    entsize, count, names_index = struct.unpack_from("<HHH", elf, 0x3A)
    # Section headers: (name, type, flags, addr, offset, size, ...).
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", elf, shoff + k * entsize)
        for k in range(count)
    ]
    names = sections[names_index][4]

    def name(at: int) -> str:
        return elf[names + at : elf.index(b"\0", names + at)].decode()

    text = next(s for s in sections if name(s[0]) == ".text")
    body = elf[text[4] : text[4] + text[5]]
    marker = struct.pack("<I", MARKER)
    if body.count(marker) != 1:
        raise RuntimeError(f"code warm: the marker occurs {body.count(marker)} times")
    # s_getpc_b64 returns the address of the next instruction: the s_mov_b32
    # whose 4-byte opcode the literal follows.
    got = (body.index(marker) - 4, text[5] // 16)
    _TEXT_RANGES[id(kernel)] = (got, kernel)  # holds the kernel: ids stay unique
    return got


__all__ = ["MARKER", "code_pointer", "code_warm_stage", "text_range"]
