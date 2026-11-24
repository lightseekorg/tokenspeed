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

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, Q, S, H: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    tile = tl.arange(0, 8)
    lane = tl.arange(0, 128)
    col = block * 1024 + tile[:, None] * 128 + lane[None, :]
    v = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(v), 1), 1.0e-4)
    sf = amax / 448.0
    bits = sf.to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 255) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    value = (v * (1.0 / scale[:, None])).to(Q.dtype.element_ty)
    tl.store(Q + row * H + col, value, col < H)
    shift = (tile & 3) * 8
    s0 = tl.sum(tl.where(tile < 4, exp << shift, 0), 0)
    s1 = tl.sum(tl.where(tile >= 4, exp << shift, 0), 0)
    tl.store(S + row * (H // 512) + block * 2, s0, block * 2 < H // 512)
    tl.store(S + row * (H // 512) + block * 2 + 1, s1, block * 2 + 1 < H // 512)


def pack_q0(x):
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] % 512 == 0
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((x.shape[0], x.shape[1] // 512), device=x.device, dtype=torch.int32)
    if x.shape[0]:
        _pack[(x.shape[0], triton.cdiv(x.shape[1], 1024))](
            x, q, s, x.shape[1], num_warps=4
        )
    return q, s
