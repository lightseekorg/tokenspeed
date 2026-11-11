# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton  # noqa.

cdiv, jit = triton.cdiv, triton.jit
from tokenspeed_kernel._triton import tl  # noqa.

builtin = tl.core.builtin
from tokenspeed_kernel._triton import libdevice  # noqa.

fast_expf, fast_logf = libdevice.fast_expf, libdevice.fast_logf


MAX_K: int = 128

NOISE_FLOOR: tl.constexpr = tl.constexpr(1.0e-7)
NOISE_STRIDE: tl.constexpr = tl.constexpr(256)


def next_power_of_2(value: int) -> int:

    active_bits = int.bit_count(value)

    return (1 << int.bit_length(value)) if active_bits > 1 else value


@jit
def gdc_wait():

    tl.inline_asm_elementwise(
        asm="griddepcontrol.wait;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@jit
def gdc_launch_dependents():

    tl.inline_asm_elementwise(
        asm="griddepcontrol.launch_dependents;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@jit
def philox_key(h_ptr, l_ptr, batch_id, step):

    # Key the stream by the slot's seed and offset, so draws advance with the request only.

    seed = tl.load(h_ptr + batch_id).to(tl.int64)
    offset = tl.load(l_ptr + batch_id).to(tl.int64)

    lo, hi, _, _ = tl.randint4x(seed, offset * NOISE_STRIDE + step)

    return (hi.to(tl.int64) << 32) | lo.to(tl.int64)


@jit
def philox_gumbel(key, indices):

    # Draw one Philox round per group of four consecutive token indices.

    u0, u1, u2, u3 = tl.rand4x(key, indices // 4)

    lane = indices % 4
    uniform = tl.where(
        lane == 0, u0, tl.where(lane == 1, u1, tl.where(lane == 2, u2, u3))
    )

    return tl.log(-tl.log(tl.maximum(uniform, NOISE_FLOOR)))


@jit
def philox_uniform(
    h_ptr,
    l_ptr,
    batch_id,
    span,
    mask,
    counter,
    block_g: tl.constexpr,
):

    tl.static_assert(block_g <= NOISE_STRIDE)

    uniform = tl.zeros((block_g,), dtype=tl.float32)

    for step in tl.static_range(block_g):

        # Draw each step's coin in (0, 1] past the counters its gumbel noise consumes.

        key = philox_key(h_ptr, l_ptr, batch_id, step)
        coin = tl.rand(key, counter + tl.zeros((1,), dtype=tl.int32))
        coin = tl.maximum(1.0 - coin, NOISE_FLOOR)

        uniform = tl.where(span == step, coin.broadcast_to((block_g,)), uniform)

    return tl.where(mask, uniform, 0.0)


@jit
def ones_like(other):

    return tl.full(other.shape, 1, dtype=other.dtype)


@jit
def log_softmax(logits, maximum):

    scores = logits.to(tl.float32) - maximum
    exp_sum = tl.sum(fast_expf(scores), axis=0)

    return scores - fast_logf(exp_sum)


@jit
def nucleus_masking(p_ptr, batch_id, logits, maximum, max_k: tl.constexpr):

    scores = fast_expf(logits.to(tl.float32) - maximum)
    ladder = tl.cumsum(scores, axis=0)

    scale = tl.gather(
        ladder, tl.full((1,), max_k - 1, dtype=tl.int32), axis=0
    ) * tl.load(p_ptr + batch_id).to(tl.float32)

    # Keep each token the mass ranked above it leaves short of p, so the one crossing p too.
    return ladder - scores < scale


@builtin
def to_scalar(value, _semantic=None, _generator=None):

    return _semantic.unsplat(value)


@jit
def cumulative_masking(
    k_ptr,
    p_ptr,
    n_ptr,
    batch_id,
    indicator,
    values,
    maximum,
    offsets,
    max_k: tl.constexpr,
):

    minima: tl.constexpr = -float("inf")

    if (indicator & 0x80) > 0:

        subset_mask = offsets < tl.load(k_ptr + batch_id)
        values = tl.where(subset_mask, values, minima)

    if (indicator & 0x100) > 0:

        nucleus_mask = nucleus_masking(p_ptr, batch_id, values, maximum, max_k) | (
            offsets == 0
        )

        values = tl.where(nucleus_mask, values, minima)

    if (indicator & 0x200) > 0:

        pivot = fast_logf(tl.load(n_ptr + batch_id).to(tl.float32)) + maximum
        values = tl.where(values >= pivot, values, minima)

    return values
