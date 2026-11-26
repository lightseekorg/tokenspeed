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

"""Round-keyed Philox coins for speculative chain verification."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# A separate counter range from ordinary and draft sampling. Each round uses
# the 31-bit starting position and has its own depth counters.
_CHAIN_VERIFY_COUNTER_DOMAIN = tl.constexpr(1 << 63)
_MIN_COIN = tl.constexpr(torch.finfo(torch.float32).tiny)


@triton.jit
def _positive_uniform(seed, counter):
    bits = tl.randint(seed, counter).to(tl.uint32)
    # 24 bits convert exactly to fp32. Keep both endpoints out, including
    # zero, which the target-only verifier could accept for a zero-mass token.
    uniform = (bits >> 8).to(tl.float32) * (1.0 / (1 << 24))
    return tl.maximum(uniform, _MIN_COIN)


@triton.jit(
    do_not_specialize=["width", "slot_stride"],
    do_not_specialize_on_alignment=["width", "slot_stride"],
)
def _chain_verify_coins(
    slots_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    coins_ptr,
    final_coins_ptr,
    width,
    slot_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + row * slot_stride).to(tl.int64)
    seed = tl.load(seed_pool_ptr + slot)
    position = tl.load(offsets_pool_ptr + slot).to(tl.uint64)
    counter = (position << 32) | _CHAIN_VERIFY_COUNTER_DOMAIN

    depth = tl.arange(0, BLOCK)
    accept_coins = _positive_uniform(seed, counter | (depth.to(tl.uint64) + 1))
    tl.store(coins_ptr + row * width + depth, accept_coins, depth < width)
    tl.store(final_coins_ptr + row, _positive_uniform(seed, counter))


@register_kernel(
    "sampling",
    "chain_verify_coins",
    name="triton_chain_verify_coins",
    solution="triton",
    signatures=format_signatures("coins", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_chain_verify_coins(slots, seed_pool, offsets_pool, *, coins, final_coins):
    rows, width = coins.shape
    _chain_verify_coins[(rows,)](
        slots,
        seed_pool,
        offsets_pool,
        coins,
        final_coins,
        width,
        slots.stride(0),
        BLOCK=triton.next_power_of_2(width),
        num_warps=1,
    )
