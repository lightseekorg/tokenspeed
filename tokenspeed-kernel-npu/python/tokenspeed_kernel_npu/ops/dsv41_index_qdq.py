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

"""Original index FP4 quantize/dequantize without a byte-buffer round trip."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_cache import _e2m1_decode, _e2m1_encode


@triton.jit
def _index_qdq_kernel(X, O, XS0, OS0, ROWS):
    d = tl.arange(0, 128)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        x = tl.load(X + row.to(tl.int64) * XS0 + d).to(tl.float32).reshape(4, 32)
        amax = tl.max(tl.abs(x), axis=1)
        scaled_max = tl.maximum(amax, 6 * 2.0**-126) * (1.0 / 6.0)
        bits = scaled_max.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        normalized = tl.div_rn(x, scale[:, None])
        code = _e2m1_encode(normalized)
        decoded = (_e2m1_decode(code) * scale[:, None]).to(tl.bfloat16)
        tl.store(O + row.to(tl.int64) * OS0 + d, decoded.reshape(128))


def index_qdq(rows, out):
    """Apply group32 E8M0/FP4 RNE and return the original BF16 QDQ values."""
    if rows.ndim != 2 or rows.shape[1] != 128:
        raise ValueError("Index QDQ requires [rows,128]")
    rows = rows.contiguous()
    if out is None:
        out = torch.empty(rows.shape, device=rows.device, dtype=torch.bfloat16)
    if out.shape != rows.shape or not out.is_contiguous():
        raise ValueError("Index QDQ output must have the input shape and be contiguous")
    if rows.shape[0]:
        _index_qdq_kernel[(min(rows.shape[0], 32768),)](
            rows,
            out,
            rows.stride(0),
            out.stride(0),
            rows.shape[0],
            num_warps=4,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
