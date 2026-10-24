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


"""Store BF16 index planes with optional original FP4 operand rounding."""

import torch
from tokenspeed_kernel_npu.ops.dsv41_index_qdq import index_qdq


def store_index(rows, cache, slots, *, quantize):
    """Write physical LCM rows, optionally preserving FP4 QDQ operand values."""
    if not cache.is_contiguous() or cache.dtype != torch.bfloat16:
        raise ValueError("BF16 index storage must be a contiguous LCM plane")
    values = index_qdq(rows, None) if quantize else rows.to(torch.bfloat16)
    valid = (slots >= 0) & (slots < cache.shape[0] * 64)
    safe = slots.long().masked_fill(~valid, 0)
    values = values.masked_fill(~valid[:, None], 0)
    cache.view(-1, 128).index_copy_(0, safe, values)
