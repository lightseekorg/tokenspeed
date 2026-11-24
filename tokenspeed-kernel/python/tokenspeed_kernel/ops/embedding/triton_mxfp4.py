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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit(do_not_specialize=["rows", "start", "end"])
def _mxfp4_embedding(
    W,
    S,
    I,
    O,
    rows,
    start,
    end,
    WS,
    SS,
    D: tl.constexpr,
    LOOKUP: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, B)
    if LOOKUP:
        index = tl.load(I + row).to(tl.int64)
        valid = (index >= start) & (index < end)
        source = index - start
    else:
        source = row
        valid = row < rows
    mask = valid & (d < D)
    packed = tl.load(W + source * WS + d // 2, mask, 0)
    bits = (packed >> ((d % 2) * 4)) & 15
    magnitude = bits & 7
    value = tl.where(
        magnitude < 2,
        magnitude.to(tl.float32) * 0.5,
        (1 + (magnitude & 1).to(tl.float32) * 0.5)
        * tl.exp2((magnitude // 2 - 1).to(tl.float32)),
    )
    value = tl.where((bits & 8) != 0, -value, value)
    exponent = tl.load(S + source * SS + d // 32, mask, 127).to(tl.int32)
    scale = tl.where(
        exponent == 0, 2.0**-127, (exponent << 23).to(tl.float32, bitcast=True)
    )
    scale = tl.where(exponent == 255, float("nan"), scale)
    tl.store(O + row * D + d, tl.where(valid, value * scale, 0.0), d < D)


@register_kernel(
    "embedding",
    "mxfp4_embedding",
    name="triton_mxfp4_embedding",
    solution="triton",
    signatures=[format_signature(weight=dense_tensor_format(torch.uint8))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def mxfp4_embedding(weight, scales, indices, row_start, row_end):
    indices = indices.contiguous()
    d = weight.shape[1] * 2
    out = torch.empty((*indices.shape, d), device=weight.device, dtype=torch.bfloat16)
    if indices.numel():
        _mxfp4_embedding[(indices.numel(),)](
            weight,
            scales,
            indices,
            out,
            indices.numel(),
            row_start,
            row_end,
            weight.stride(0),
            scales.stride(0),
            d,
            True,
            triton.next_power_of_2(d),
            num_warps=4,
        )
    return out


@register_kernel(
    "embedding",
    "mxfp4_embedding_decode",
    name="triton_mxfp4_embedding_decode",
    solution="triton",
    signatures=[format_signature(weight=dense_tensor_format(torch.uint8))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def mxfp4_embedding_decode(codes, scales):
    d = codes.shape[-1] * 2
    rows = codes.numel() // codes.shape[-1]
    out = torch.empty((*codes.shape[:-1], d), device=codes.device, dtype=torch.bfloat16)
    if rows:
        _mxfp4_embedding[(rows,)](
            codes,
            scales,
            codes,
            out,
            rows,
            0,
            rows,
            d // 2,
            d // 32,
            d,
            False,
            triton.next_power_of_2(d),
            num_warps=4,
        )
    return out
