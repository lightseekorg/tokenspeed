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


@triton.jit(do_not_specialize=["M", "C", "S0", "S1", "OM"])
def _pack_ue8m0(S, O, M, C, S0, S1, OM, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    packs = tl.cdiv(C, 4)
    row = i % OM
    col = i // OM
    valid = (row < M) & (col < packs)
    packed = tl.full((B,), 0, tl.uint32)
    for j in tl.static_range(4):
        c = col * 4 + j
        s = tl.load(S + row * S0 + c * S1, valid & (c < C), other=127).to(tl.float32)
        # Match the pinned CUDA torch.exp2 path before vendor bit packing.
        # Nonzero integer exponents are exact; byte 255 produces +inf.
        # Byte zero rounds 2**-127 down to FP32 bits 0x003fffff. The
        # vendor packer preserves those mantissa bits across byte lanes.
        bits = tl.where(s == 0, 0x003FFFFF, s.to(tl.uint32) << 23).to(tl.uint32)
        bits = tl.where(c < C, bits, 0)
        if j == 0:
            packed |= bits >> 23
        elif j == 1:
            packed |= bits >> 15
        elif j == 2:
            packed |= bits >> 7
        else:
            packed |= bits << 1
    tl.store(O + i, tl.where(row < M, packed, 0), i < OM * packs)


def pack_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    """Fuse canonical UE8M0 decoding and DeepGEMM column-major int32 packing."""
    if scales.ndim != 2 or scales.dtype != torch.uint8 or not scales.is_cuda:
        raise ValueError("Expected CUDA uint8 scales [rows, groups]")
    m, c = scales.shape
    aligned = triton.cdiv(m, 4) * 4
    storage = torch.empty(
        (triton.cdiv(c, 4), aligned), dtype=torch.int32, device=scales.device
    )
    if storage.numel():
        _pack_ue8m0[(triton.cdiv(storage.numel(), 256),)](
            scales, storage, m, c, *scales.stride(), aligned, 256
        )
    return storage.t()[:m]
