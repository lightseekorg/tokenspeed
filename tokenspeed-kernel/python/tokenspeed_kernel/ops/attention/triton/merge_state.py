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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def attn_merge_state_kernel(
    OutA,
    LseA,
    OutB,
    LseB,
    Out,
    Lse,
    head_dim: tl.constexpr,
    lse_scale_log2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    value_offsets = row * head_dim + offs_d

    lse_a = tl.load(LseA + row).to(tl.float32)
    lse_b = tl.load(LseB + row).to(tl.float32)
    lse_a_log2 = lse_a * lse_scale_log2
    lse_b_log2 = lse_b * lse_scale_log2
    lse_max_log2 = tl.maximum(lse_a_log2, lse_b_log2)

    weight_a = tl.exp2(lse_a_log2 - lse_max_log2)
    weight_b = tl.exp2(lse_b_log2 - lse_max_log2)
    denom = weight_a + weight_b

    out_a = tl.load(OutA + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out_b = tl.load(OutB + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out = (out_a * weight_a + out_b * weight_b) / denom
    merged_lse = (lse_max_log2 + tl.log2(denom)) / lse_scale_log2

    tl.store(Out + value_offsets, out, mask=mask_d)
    tl.store(Lse + row, merged_lse)


@register_kernel(
    "attention",
    "attn_merge_state",
    name="triton_attn_merge_state",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("out_a", "out_b"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={},
    tags={"portability"},
)
def triton_attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(out_a)
    lse = torch.empty_like(lse_a)
    total_rows = out_a.shape[0] * out_a.shape[1]
    head_dim = out_a.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    attn_merge_state_kernel[(total_rows,)](
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        lse,
        head_dim,
        float(lse_scale_log2),
        BLOCK_D=block_d,
    )
    return out, lse
