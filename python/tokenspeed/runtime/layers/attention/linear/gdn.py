# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
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
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.utils.triton import tl, triton


@triton.jit(
    do_not_specialize=["stride_a", "num_rows"],
    do_not_specialize_on_alignment=["a"],
)
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    stride_a,
    dt_bias,
    num_rows,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_ROWS: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()
    # One program gates BLK_ROWS tokens across every head.
    rows = tl.program_id(0) * BLK_ROWS + tl.arange(0, BLK_ROWS)
    heads = tl.arange(0, BLK_HEADS)
    head_mask = heads < NUM_HEADS
    mask = (rows < num_rows)[:, None] & head_mask[None, :]
    rows = rows.to(tl.int64)
    blk_A_log = tl.load(A_log + heads, mask=head_mask)
    blk_bias = tl.load(dt_bias + heads, mask=head_mask)
    blk_a = tl.load(a + rows[:, None] * stride_a + heads[None, :], mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32))[None, :] * softplus_x
    tl.store(
        g + rows[:, None] * NUM_HEADS + heads[None, :],
        blk_g.to(g.dtype.element_ty),
        mask=mask,
    )


# Tokens per program. One program per token cost ~100 us per layer on a
# 16k-token prefill chunk.
_GATING_BLOCK_ROWS = 16


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    enable_pdl = pdl_enabled()
    num_rows, num_heads = a.shape
    grid = (triton.cdiv(num_rows, _GATING_BLOCK_ROWS),)
    # ``a`` may be a row-strided view (unit inner stride); ``g`` is contiguous.
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    fused_gdn_gating_kernel[grid](
        g,
        A_log,
        a,
        a.stride(0),
        dt_bias,
        num_rows,
        num_heads,
        beta,
        threshold,
        _GATING_BLOCK_ROWS,
        triton.next_power_of_2(num_heads),
        num_warps=4,
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return g
