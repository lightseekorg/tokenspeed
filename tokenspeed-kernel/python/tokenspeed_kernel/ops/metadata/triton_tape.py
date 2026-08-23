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

"""Portable Triton interpreter for the prep tape (one program per op).

Descriptor fields and register semantics match ``prep_tape.py``; pointers
travel as int64 immediates or register references and are dereferenced via
``tl.cast(addr, tl.pointer_type(...))``, so the same kernel serves every
vendor tokenspeed-triton targets.
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton

_REG_FLAG = 1 << 62


@triton.jit
def _resolve(v, regs_ptr):
    is_ref = (v >= 0) & ((v & (1 << 62)) != 0)
    slot = v & 0xFFFF
    reg = tl.load(regs_ptr + slot, mask=is_ref, other=0)
    return tl.where(is_ref, reg, v)


@triton.jit
def _prep_tape_kernel(descs_ptr, regs_ptr, BLOCK: tl.constexpr):
    d = descs_ptr + tl.program_id(0) * 8
    opcode = tl.load(d)
    dst = tl.cast(_resolve(tl.load(d + 1), regs_ptr), tl.pointer_type(tl.int32))
    src = tl.cast(_resolve(tl.load(d + 2), regs_ptr), tl.pointer_type(tl.int32))
    idx = tl.cast(_resolve(tl.load(d + 3), regs_ptr), tl.pointer_type(tl.int32))
    n = _resolve(tl.load(d + 4), regs_ptr)
    m = _resolve(tl.load(d + 5), regs_ptr)
    stride = tl.load(d + 6)
    scalar = _resolve(tl.load(d + 7), regs_ptr)
    offs = tl.arange(0, BLOCK)

    if opcode == 0:  # FILL: dst[0:n] = scalar
        for base in range(0, n, BLOCK):
            o = base + offs
            tl.store(dst + o, tl.full([BLOCK], 0, tl.int32) + scalar, mask=o < n)
    elif opcode == 1:  # COPY: dst[0:n] = src[soff : soff + n]
        soff = _resolve(stride, regs_ptr)
        for base in range(0, n, BLOCK):
            o = base + offs
            v = tl.load(src + soff + o, mask=o < n, other=0)
            tl.store(dst + o, v, mask=o < n)
    elif opcode == 2:  # GATHER: dst[i] = src[idx[i]], idx<0 -> scalar
        for base in range(0, n, BLOCK):
            o = base + offs
            mask = o < n
            j = tl.load(idx + o, mask=mask, other=0)
            v = tl.load(src + j, mask=mask & (j >= 0), other=0)
            tl.store(dst + o, tl.where(j >= 0, v, scalar.to(tl.int32)), mask=mask)
    elif opcode == 3:  # COPY2D with column pad; stride = dst row stride
        total = n * stride
        for base in range(0, total, BLOCK):
            o = base + offs
            mask = o < total
            r = o // stride
            c = o - r * stride
            v = tl.load(src + r * m + c, mask=mask & (c < m), other=0)
            tl.store(dst + o, tl.where(c < m, v, scalar.to(tl.int32)), mask=mask)
    elif opcode == 4:  # FILLTAIL: dst[n:m] = scalar
        for base in range(0, m, BLOCK):
            o = base + offs
            mask = (o >= n) & (o < m)
            if stride == 0:
                tl.store(dst + o, tl.full([BLOCK], 0, tl.int32) + scalar, mask=mask)
            else:
                dst64 = tl.cast(dst, tl.pointer_type(tl.int64))
                tl.store(dst64 + o, tl.full([BLOCK], 0, tl.int64) + scalar, mask=mask)
    elif opcode == 5:  # STATE_PAGES (decode: before = after - 1)
        rows = src  # [n, m] page table
        state_out = idx  # second output rides the idx slot
        seq_ptr = tl.cast(scalar, tl.pointer_type(tl.int32))
        page_size = stride
        o = offs
        mask = o < n
        after = tl.load(seq_ptr + o, mask=mask, other=1).to(tl.int64)
        before = after - 1
        in_slot = tl.maximum((before - 1) // page_size, 0)
        out_slot = tl.minimum(tl.maximum((after - 1) // page_size, 0), m - 1)
        s_in = tl.load(rows + o * m + in_slot, mask=mask, other=0)
        s_in = tl.where(before > 0, s_in, 0)
        s_out = tl.load(rows + o * m + out_slot, mask=mask, other=0)
        tl.store(dst + o, s_in, mask=mask)
        tl.store(state_out + o, s_out, mask=mask)


def run_tape(descs, regs) -> None:
    """Execute ordered stages, with one parallel program per descriptor."""
    if isinstance(descs, tuple):
        for stage in descs:
            run_tape(stage, regs)
        return
    n_ops = descs.shape[0]
    if n_ops == 0:
        return
    _prep_tape_kernel[(n_ops,)](descs, regs, BLOCK=128, num_warps=1)
