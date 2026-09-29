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

"""Tensor register views and explicitly masked native memory instructions."""

import triton.experimental.gluon as g
from lib.gemm.rocm.intrinsics import _native_call, _resource_content
from triton.experimental.gluon import language as l


@g.jit
def prevent_loop_unroll():
    # Gluon's loop_unroll_factor controls its own unroller. Keep LLVM from
    # duplicating the publication loop as well, without emitting an instruction.
    _native_call("compiler.loop.barrier", "void", (), (), False)


@g.jit
def is_first_thread(tid):
    # Recompute this predicate at its use instead of keeping two SGPRs live
    # across the matrix loop. The tied register operand emits no instruction.
    tid = l.inline_asm_elementwise(
        "", constraints="=v,0", args=[tid], dtype=l.uint32, is_pure=False, pack=1
    )
    return tid == 0


@g.jit
def first(value):
    if value.type.is_block():
        index = l.full([1], 0, l.int32, value.type.layout)
        return l.gather(value, index, 0).reshape([])
    else:
        return value


@g.jit
def load_words(resource, voffset, soffset, width: l.constexpr, aux: l.constexpr, mask):
    offset = l.where(mask, (l.full((), 0, l.uint32) + voffset).to(l.uint32), 0xFFFFFFFF)
    if width == 4:
        return _native_call(
            "llvm.amdgcn.raw.buffer.load.v4i32",
            "v4i32",
            ("v4i32", "i32", "i32", "#i32"),
            (_resource_content(resource), offset, soffset, aux),
            False,
        )
    elif width == 2:
        return _native_call(
            "llvm.amdgcn.raw.buffer.load.v2i32",
            "v2i32",
            ("v4i32", "i32", "i32", "#i32"),
            (_resource_content(resource), offset, soffset, aux),
            False,
        )
    else:
        return _native_call(
            "llvm.amdgcn.raw.buffer.load.i32",
            "i32",
            ("v4i32", "i32", "i32", "#i32"),
            (_resource_content(resource), offset, soffset, aux),
            False,
        )


@g.jit
def store_words(
    resource, voffset, soffset, values, width: l.constexpr, aux: l.constexpr, mask
):
    if width == 4:
        _native_call(
            "when:llvm.amdgcn.raw.buffer.store.v4i32",
            "void",
            ("v4i32", "v4i32", "i32", "i32", "#i32", "i1"),
            (values, _resource_content(resource), voffset, soffset, aux, mask),
            False,
        )
    elif width == 2:
        _native_call(
            "when:llvm.amdgcn.raw.buffer.store.v2i32",
            "void",
            ("v2i32", "v4i32", "i32", "i32", "#i32", "i1"),
            (values, _resource_content(resource), voffset, soffset, aux, mask),
            False,
        )
    else:
        _native_call(
            "when:llvm.amdgcn.raw.buffer.store.i32",
            "void",
            ("i32", "v4i32", "i32", "i32", "#i32", "i1"),
            (values, _resource_content(resource), voffset, soffset, aux, mask),
            False,
        )


@g.jit
def atomic_add(resource, voffset, soffset, value, aux: l.constexpr, mask):
    return _native_call(
        "when:llvm.amdgcn.raw.buffer.atomic.add.i32",
        "i32",
        ("i32", "v4i32", "i32", "i32", "#i32", "i1"),
        (value, _resource_content(resource), voffset, soffset, aux, mask),
        False,
    )


@g.jit
def atomic_or(resource, voffset, soffset, value, aux: l.constexpr, mask):
    return _native_call(
        "when:llvm.amdgcn.raw.buffer.atomic.or.i32",
        "i32",
        ("i32", "v4i32", "i32", "i32", "#i32", "i1"),
        (value, _resource_content(resource), voffset, soffset, aux, mask),
        False,
    )


@g.jit
def store_vector4(ptr, words, mask):
    layout: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [l.num_warps(), 1], [1, 0])
    ptr = l.convert_layout(ptr, l.SliceLayout(1, layout))
    mask = l.convert_layout(mask, l.SliceLayout(1, layout))
    value = l.join(l.join(words[0], words[2]), l.join(words[1], words[3]))
    value = l.convert_layout(value.reshape([ptr.numel, 4]), layout)
    word = l.arange(0, 4, layout=l.SliceLayout(0, layout))
    l.store(ptr[:, None] + word[None, :], value, mask=mask[:, None])


@g.jit
def wave_id():
    # Wave-uniform control does not require a CTA-wide tensor reduction.
    thread = _native_call("llvm.amdgcn.workitem.id.x", "i32", (), (), True)
    return (
        _native_call("llvm.amdgcn.readfirstlane", "i32", ("i32",), (thread,), True)
        // 64
    )
