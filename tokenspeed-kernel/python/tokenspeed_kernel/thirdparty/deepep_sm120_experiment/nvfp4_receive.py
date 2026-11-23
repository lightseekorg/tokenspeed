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


@triton.jit(do_not_specialize=["capacity", "rows"])
def _wire_nvfp4_decode(
    P,
    S,
    G,
    C,
    X,
    IDS,
    W,
    capacity,
    rows,
    H: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    dest = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, triton.next_power_of_2(E))
    counts = tl.load(C + experts, experts < E, 0)
    ends = tl.cumsum(counts)
    expert = tl.sum(((dest >= ends) & (experts < E)).to(tl.int32), 0)
    start = tl.sum(tl.where(experts < expert, counts, 0), 0)
    valid = dest < tl.sum(counts, 0)
    row = expert.to(tl.int64) * capacity + dest - start
    d = tl.program_id(1) * B + tl.arange(0, B)
    mask = valid & (d < H)
    packed = tl.load(P + row * (H // 2) + d // 2, mask, 0)
    code = (packed >> ((d % 2) * 4)) & 15
    mag = code & 7
    value = tl.where(
        mag < 2,
        mag.to(tl.float32) * 0.5,
        (1 + (mag & 1).to(tl.float32) * 0.5) * tl.exp2((mag // 2 - 1).to(tl.float32)),
    )
    value = tl.where((code & 8) != 0, -value, value)
    sf = (
        tl.load(S + row * (H // 16) + d // 16, mask, 0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    global_scale = tl.load(G + row, valid, 0)
    tl.store(X + dest * H + d, (value * sf) * global_scale, d < H)
    if tl.program_id(1) == 0:
        tl.store(IDS + dest, tl.where(valid, expert, -1))
        tl.store(W + dest, tl.where(valid, 1.0, 0.0))


def decode_wire_nvfp4(payload, scales, global_scales, counts, max_rows):
    """Decode only compact receive rows; communication payload stays NVFP4."""
    e, capacity, half_h = payload.shape
    h = half_h * 2
    rows = min(e * capacity, max_rows)
    x = torch.empty((rows, h), dtype=torch.bfloat16, device=payload.device)
    ids = torch.empty((rows, 1), dtype=torch.int32, device=payload.device)
    weights = torch.empty((rows, 1), dtype=torch.float32, device=payload.device)
    if rows:
        _wire_nvfp4_decode[(rows, triton.cdiv(h, 1024))](
            payload,
            scales.view(torch.uint8),
            global_scales,
            counts,
            x,
            ids,
            weights,
            capacity,
            rows,
            h,
            e,
            1024,
            num_warps=4,
        )
    return x, ids, weights


@triton.jit
def _elastic_fp8_decode(X, S, COUNT, O, H: tl.constexpr, XS, SS, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * B + tl.arange(0, B)
    valid = row < tl.load(COUNT)
    bits = tl.load(X + row * XS + d, valid & (d < H), 0)
    v = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    word = tl.load(S + row * SS + d // 512, valid & (d < H), 0)
    exponent = (word >> ((d // 128 % 4) * 8)) & 255
    scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(O + row * H + d, v * scale, d < H)


def decode_elastic_fp8(x, scales, counts):
    """GPU-decode valid Elastic FP8/G128 rows before calibrated NVFP4 GEMMs."""
    if x.dtype != torch.float8_e4m3fn or scales.dtype != torch.int32:
        raise TypeError(
            "Elastic NVFP4 adapter expects FP8 codes and packed UE8M0 scales"
        )
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    if x.shape[0]:
        _elastic_fp8_decode[(x.shape[0], triton.cdiv(x.shape[1], 1024))](
            x.view(torch.uint8),
            scales,
            counts[-1:],
            out,
            x.shape[1],
            x.stride(0),
            scales.stride(0),
            1024,
            num_warps=4,
        )
    return out
