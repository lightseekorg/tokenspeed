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

"""Synthetic-acceptance verify outputs in one launch."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _write_synthetic_outputs(
    candidates,
    targets,
    lengths,
    predict,
    accept_index,
    accept_length,
    candidates_stride,
    targets_stride,
    WIDTH: tl.constexpr,
    TARGET_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.minimum(tl.load(lengths + row).to(tl.int32), WIDTH)
    if TARGET_ROWS:
        target = tl.load(targets + row * targets_stride + length - 1)
    else:
        target = tl.load(targets + row * targets_stride)
    prefix = col < length - 1
    draft = tl.load(candidates + row * candidates_stride + col + 1, prefix, 0)
    token = tl.where(prefix, draft, tl.where(col == length - 1, target, 0))
    live = col < WIDTH
    tl.store(predict + row * WIDTH + col, token.to(predict.dtype.element_ty), live)
    index = tl.where(col < length, row * WIDTH + col, -1)
    tl.store(
        accept_index + row * WIDTH + col, index.to(accept_index.dtype.element_ty), live
    )
    tl.store(accept_length + row, (length - 1).to(accept_length.dtype.element_ty))


def write_synthetic_outputs(
    candidates, targets, lengths, predict, accept_index, accept_length
):
    rows, width = candidates.shape
    _write_synthetic_outputs[(rows,)](
        candidates,
        targets,
        lengths,
        predict,
        accept_index,
        accept_length,
        candidates.stride(0),
        targets.stride(0),
        WIDTH=width,
        TARGET_ROWS=targets.ndim == 2,
        BLOCK=triton.next_power_of_2(width),
    )
