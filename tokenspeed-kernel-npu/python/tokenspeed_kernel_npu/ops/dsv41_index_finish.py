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

"""Index selection finalization retaining the original FP32 sort semantics."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["K", "S0", "I0", "L0", "C0"])
def _prepare_finish(
    Scores, Ids, Labels, Counts, K, S0, I0, L0, C0, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    score = tl.load(Scores + row * S0 + col, col < K, other=-float("inf")).to(
        tl.float32
    )
    ids = tl.load(Ids + row * I0 + col, col < K, other=-1).to(tl.float32)
    valid = (col < K) & (score > -float("inf"))
    tl.store(Labels + row * L0 + col, tl.where(valid, ids, float("inf")), col < K)
    tl.store(Counts + row * C0, tl.sum(valid.to(tl.int32), 0))


@triton.jit(do_not_specialize=["K", "S0", "O0"])
def _finish_labels(Sorted, Out, K, S0, O0, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    label = tl.load(Sorted + row * S0 + col, col < K, other=float("inf"))
    finite = (label > -float("inf")) & (label < float("inf"))
    value = tl.where(finite, label, -1).to(tl.int32)
    tl.store(Out + row * O0 + col, value, col < K)


def finish(scores, ids, out, lengths):
    """Sort selected logical IDs and write count, preserving NaN/inf rejection."""
    n, k = scores.shape
    if not n:
        return
    if not k:
        lengths.zero_()
        return
    scores = scores.contiguous()
    ids = ids.contiguous()
    target = out if out.stride(1) == 1 else torch.empty_like(scores, dtype=out.dtype)
    stride = triton.cdiv(k, 8) * 8
    labels = torch.empty_strided(
        (n, k), (stride, 1), device=scores.device, dtype=torch.float32
    )
    block = max(8, triton.next_power_of_2(k))
    _prepare_finish[(n,)](
        scores,
        ids,
        labels,
        lengths,
        k,
        scores.stride(0),
        ids.stride(0),
        stride,
        lengths.stride(0),
        BLOCK=block,
        num_warps=4,
    )
    sorted_labels = labels.sort(-1).values
    _finish_labels[(n,)](
        sorted_labels,
        target,
        k,
        sorted_labels.stride(0),
        target.stride(0),
        BLOCK=block,
        num_warps=4,
    )
    if target is not out:
        out.copy_(target)
