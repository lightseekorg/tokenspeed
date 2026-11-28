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

"""Group-32 MXFP8 packing and scatter for the shared MLA prologue."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _store(
    Cache,
    Slots,
    Nope,
    Rope,
    Mask,
    cache_stride,
    nope_stride,
    rope_stride,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    RECORD: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SANITIZE: tl.constexpr,
    LIMIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + row).to(tl.int64)
    valid = tl.full((), True, tl.int1)
    if HAS_MASK:
        # Some backends load torch.bool storage as a byte.
        valid = tl.load(Mask + row).to(tl.int1)
    group = tl.arange(0, GROUPS)
    channel = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    width: tl.constexpr = RANK + ROPE
    x = tl.load(
        Nope + row * nope_stride + channel, valid & (channel < RANK), other=0
    ).to(tl.float32)
    y = tl.load(
        Rope + row * rope_stride + channel - RANK,
        valid & (channel >= RANK) & (channel < width),
        other=0,
    ).to(tl.float32)
    x = tl.where(channel < RANK, x, y)
    if SANITIZE:
        x = tl.where(x != x, 0.0, tl.minimum(tl.maximum(x, -LIMIT), LIMIT))
    maximum = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-4)
    ratio = maximum * (1.0 / 448.0)
    bits = ratio.to(tl.int32, bitcast=True)
    code = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = tl.where(code == 0, 0x400000, code << 23).to(tl.float32, bitcast=True)
    payload = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0).to(
        tl.float8e4nv
    )
    dst = Cache + slot * cache_stride
    tl.store(
        dst + channel, payload.to(tl.uint8, bitcast=True), valid & (channel < width)
    )
    tl.store(dst + width + group, code.to(tl.uint8), valid & (group < width // 32))
    padding = tl.arange(0, 16)
    start: tl.constexpr = width + width // 32
    tl.store(dst + start + padding, 0, valid & (start + padding < RECORD))


def store_mxfp8_latent(cache, slots, k_nope, k_rope, *, sanitize, write_mask):
    """Write group-32 payload/scales and zero record padding at live slots.

    Called after descriptor validation by the prologue. The input parts may
    be strided views. Every slot must be valid, including masked rows; only
    the write mask suppresses writes, matching the LatentKVCache contract.
    No temporary or persistent device tensor is allocated.
    """
    if slots.numel() == 0:
        return
    rank, rope = k_nope.shape[-1], k_rope.shape[-1]
    _store[(slots.numel(),)](
        cache,
        slots,
        k_nope,
        k_rope,
        write_mask,
        cache.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        rank,
        rope,
        cache.shape[-1],
        triton.next_power_of_2((rank + rope) // 32),
        write_mask is not None,
        sanitize,
        torch.finfo(k_nope.dtype).max,
        num_warps=4,
    )
