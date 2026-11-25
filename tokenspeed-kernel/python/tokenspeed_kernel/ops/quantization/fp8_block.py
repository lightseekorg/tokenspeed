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

"""FP32 block dequantization with one final BF16 rounding."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _dequantize_fp8_block(
    q,
    scales,
    out,
    size,
    N: tl.constexpr,
    K: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    col = index % K
    row = index // K % N
    expert = index // (N * K)
    code = tl.load(q + index, index < size, other=0.0).to(tl.float32)
    scale = tl.load(
        scales + (expert * SN + row // 128) * SK + col // 128, index < size, other=0.0
    )
    tl.store(out + index, (code * scale).to(tl.bfloat16), index < size)


@register_kernel(
    "quantization",
    "dequantize_fp8_block",
    name="triton_dequantize_fp8_block",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("x", "dense", (torch.float8_e4m3fn,)),
    features={"batch_invariant", "cuda_graph"},
    priority=Priority.REFERENCE,
)
def triton_dequantize_fp8_block(codes, scales):
    out = torch.empty(codes.shape, dtype=torch.bfloat16, device=codes.device)
    n, k = codes.shape[-2:]
    if out.numel():
        _dequantize_fp8_block[(triton.cdiv(out.numel(), 1024),)](
            codes,
            scales,
            out,
            out.numel(),
            n,
            k,
            (n + 127) // 128,
            (k + 127) // 128,
            1024,
        )
    return out
