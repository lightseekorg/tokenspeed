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

"""Fuse valid-row gather, FP32 expert summation and the final BF16 store."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _combine(
    Expanded,
    Map,
    Out,
    H: tl.constexpr,
    K: tl.constexpr,
    CAPACITY,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for k in tl.static_range(K):
        dst = tl.load(Map + row * K + k)
        x = tl.load(
            Expanded + dst.to(tl.int64) * H + col,
            (dst >= 0) & (dst < CAPACITY) & (col < H),
            other=0,
        ).to(tl.float32)
        value = value + x
    tl.store(Out + row * H + col, value, col < H)


def combine_routed_rows(expanded, source_to_dest, *, rows, topk):
    """Consume init_routing_v2 row_idx_type=0 maps; route weights are applied.

    Negative destinations denote remote experts. No padded expanded rows are
    read or cast, and FP32 summation follows the original source route order.
    """
    if (
        expanded.device.type != "npu"
        or expanded.dtype != torch.bfloat16
        or expanded.ndim != 2
        or not expanded.is_contiguous()
        or source_to_dest.device != expanded.device
        or source_to_dest.dtype not in (torch.int32, torch.int64)
        or source_to_dest.ndim != 1
        or not source_to_dest.is_contiguous()
        or source_to_dest.numel() != rows * topk
        or rows < 0
        or topk < 1
        or expanded.shape[1] % 32
    ):
        raise ValueError("Expected contiguous BF16 rows and a source-route map")
    out = torch.empty(
        (rows, expanded.shape[1]), device=expanded.device, dtype=expanded.dtype
    )
    if out.numel():
        _combine[(rows, triton.cdiv(expanded.shape[1], 4096))](
            expanded,
            source_to_dest,
            out,
            expanded.shape[1],
            topk,
            expanded.shape[0],
            4096,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
