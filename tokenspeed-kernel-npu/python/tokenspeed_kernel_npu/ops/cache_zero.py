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


"""Bounded NPU launch for scheduler-owned cache page initialization."""

from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["num_fields", "num_ranges"])
def _zero_page_fields(
    backing, pages, fields, num_fields, num_ranges, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    for entry in range(tl.program_id(0), num_ranges, tl.num_programs(0)):
        page = tl.load(pages + entry // num_fields).to(tl.int64)
        field = fields + (entry % num_fields) * 3
        base = tl.load(field) + page * tl.load(field + 1)
        size = tl.load(field + 2)
        for start in range(tl.program_id(1) * BLOCK, size, tl.num_programs(1) * BLOCK):
            byte = start + offsets
            tl.store(backing + base + byte, 0, byte < size)


def zero_page_fields(backing, pages, fields, *, max_field_bytes):
    """Zero validated page/field ranges while keeping launch size at most 2048.

    The public kernel wrapper validates tensors. Fields retain their runtime
    byte offsets, strides and sizes; only the launch geometry is bounded.
    Every physical address is computed in 64 bits from the field table.
    """
    num_ranges = pages.numel() * fields.shape[0]
    if not num_ranges:
        return
    tile_count = min(32, triton.cdiv(max_field_bytes, 1024))
    grid = (min(num_ranges, max(1, 2048 // tile_count)), tile_count)
    _zero_page_fields[grid](
        backing,
        pages,
        fields,
        fields.shape[0],
        num_ranges,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
        multibuffer=False,
    )
