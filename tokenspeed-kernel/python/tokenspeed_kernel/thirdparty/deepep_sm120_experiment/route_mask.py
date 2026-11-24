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
def _mask(
    IDS,
    W,
    COUNT,
    OUT_IDS,
    OUT_W,
    pairs,
    topk: tl.constexpr,
    experts: tl.constexpr,
    ids_stride,
    weights_stride,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // topk
    slot = i % topk
    valid = (i < pairs) & (row < tl.load(COUNT))
    expert = tl.load(IDS + row * ids_stride + slot, valid, other=-1)
    valid = valid & (expert >= 0) & (expert < experts)
    weight = tl.load(W + row * weights_stride + slot, valid, other=0)
    tl.store(OUT_IDS + i, tl.where(valid, expert, -1), i < pairs)
    tl.store(OUT_W + i, weight, i < pairs)


def mask_routes(ids, weights, counts, experts):
    assert ids.stride(1) == weights.stride(1) == 1
    local = torch.empty(ids.shape, device=ids.device, dtype=torch.int32)
    safe_weights = torch.empty(
        weights.shape, device=weights.device, dtype=torch.float32
    )
    _mask[(triton.cdiv(ids.numel(), 256),)](
        ids,
        weights,
        counts[-1:],
        local,
        safe_weights,
        ids.numel(),
        ids.shape[1],
        experts,
        ids.stride(0),
        weights.stride(0),
        256,
    )
    return local, safe_weights
