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

"""Prepare active sampling slots in one launch, without a pool-wide scan."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["rows", "extends", "slot_stride", "length_stride"])
def _prepare_sampling_offsets(
    frontier,
    slots,
    lengths,
    out,
    rows,
    extends,
    slot_stride,
    length_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row * slot_stride, row < rows, 0).to(tl.int64)
    live = (row < rows) & (slot != 0)
    prefix = tl.load(frontier + slot, live, 0).to(tl.int64)
    increment = tl.load(lengths + row * length_stride, live & (row < extends), 1)
    tl.store(out + slot, prefix + increment, live)


def prepare_offsets(frontier, slots, lengths, out):
    extends = 0 if lengths is None else lengths.numel()
    _prepare_sampling_offsets[(triton.cdiv(slots.numel(), 128),)](
        frontier,
        slots,
        frontier if lengths is None else lengths,
        out,
        slots.numel(),
        extends,
        slots.stride(0),
        1 if lengths is None else lengths.stride(0),
        BLOCK=128,
    )
