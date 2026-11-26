# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2026 SGLang contributors
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


"""Expert selection and the sorted union of the rows' experts.

Each row takes its top 16 of 896 experts by sigmoid score plus bias, with
normalized sigmoid weights; the union of all rows' experts, sorted, assigns
the routed-expert work.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/moe.py``.
"""

from tokenspeed_kernel_amd._extern import (
    wave_ballot_i64,
    wave_inclusive_sum_i32,
    wave_mbcnt,
    wave_popcount_i64,
)
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    put_pair,
    thread_ids,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.extern import (
    readlane,
    trailing_zeros,
    writelane,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import (
    load_f32,
    load_i32,
    store_f32,
    store_i32,
    store_mask_i32,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs


def _batcher(n):
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return tuple(pairs)


ROUTE_NET = gl.constexpr(_batcher(16))


ROW_NET = gl.constexpr(
    ((2, 1), (4, 2), (4, 1), (8, 4), (8, 2), (8, 1), (16, 8), (16, 4), (16, 2), (16, 1))
)


@gluon.jit
def replace(values, index: gl.constexpr, value):
    result = ()
    for i in gl.static_range(len(values)):
        result += (value if i == index else values[i],)
    return result


@gluon.jit
def compare_swap(values, A: gl.constexpr, B: gl.constexpr):
    first = values[A] > values[B]
    a, b = values[A], values[B]
    values = replace(values, A, gl.where(first, a, b))
    return replace(values, B, gl.where(first, b, a))


@gluon.jit
def shuffle(value, index):
    tile = gl.reshape(value, [8, 64])
    lane = gl.reshape(index, [8, 64])
    return gl.convert_layout(
        gl.reshape(gl.gather(tile, lane, 1), [512]), value.type.layout
    )


@gluon.jit
def wave_umax(value):
    tile = gl.reshape(value.to(gl.uint32), [8, 64])
    result = gl.max(tile, 1)[:, None] + gl.full([8, 64], 0, gl.uint32, tile.type.layout)
    return gl.convert_layout(gl.reshape(result, [512]), value.type.layout)


@gluon.jit
def route_parts(region, parts, token, tag, S: gl.constexpr, guard):
    tid = thread_ids()
    lane = tid % 64
    wave = tid // 64
    # Only waves 0-3 run this block, so its LDS stores use the barrier-free
    # helpers.
    if wave_pending(tid < 256):
        indices = ()
        for i in gl.static_range(14):
            e = lane + 64 * i
            indices += ((((e // 112 * 4 + wave) * S + token) * 112 + e % 112),)
        values = poll_pairs(region, indices, tag, tid < 256, "sys", guard)
        for i in gl.static_range(14):
            store_f32(parts, wave * 896 + lane + 64 * i, values[i])
    gl.barrier()


@gluon.jit
def extract(lane, keys, sigmoids, red):
    packed = ()
    for i in gl.static_range(14):
        store_f32(red, lane + 64 * i, sigmoids[i])
        packed += ((keys[i].to(gl.uint64) << 32) + (15 - i),)
    packed += (
        gl.full([512], 0, gl.uint64, lane.type.layout),
        gl.full([512], 0, gl.uint64, lane.type.layout),
    )
    for i in gl.static_range(len(ROUTE_NET)):
        packed = compare_swap(packed, ROUTE_NET[i][0], ROUTE_NET[i][1])
    ks = ()
    nibbles = gl.full([512], 0, gl.uint64, lane.type.layout)
    for i in gl.static_range(16):
        ks += ((packed[i] >> 32).to(gl.uint32),)
        nibbles = nibbles | ((15 - (packed[i] & 15)) << (4 * i))
    ks += (gl.full([512], 0, gl.uint32, lane.type.layout),)
    selected = gl.full([512], 0, gl.int32, lane.type.layout)
    for pick in gl.static_range(16):
        head = ks[0]
        maximum = wave_umax(head)
        top = head == maximum
        ballot = wave_ballot_i64(top.to(gl.int32))
        slot = (nibbles & 15).to(gl.int32)
        # The winner is the ballot's lowest set lane (one scalar s_ff1; Gluon has no
        # ctz). readlane returns its slot in a scalar register, where gl.gather
        # would add an LDS round trip.
        winner = trailing_zeros(ballot)
        expert = winner + 64 * readlane(slot, winner)
        tied = 1023 - wave_umax(
            gl.where(top, 1023 - (lane + 64 * slot), 0).to(gl.uint32)
        ).to(gl.int32)
        expert = gl.where(wave_popcount_i64(ballot) == 1, expert, tied)
        hit = lane == (expert & 63)
        next_ks = ()
        for i in gl.static_range(16):
            next_ks += (gl.where(hit, ks[i + 1], ks[i]),)
        ks = next_ks + (ks[-1],)
        nibbles = gl.where(hit, nibbles >> 4, nibbles)
        # writelane puts this pick into its lane in one instruction; gl.where
        # would add a compare and a select.
        selected = writelane(
            expert, gl.full([512], pick, gl.int32, lane.type.layout), selected
        )
    return selected


@gluon.jit
def rank_candidates(lane, keys, sigmoids, red):
    maximum = keys[0]
    for i in gl.static_range(14):
        store_f32(red, lane + 64 * i, sigmoids[i])
        if i > 0:
            maximum = gl.maximum(maximum, keys[i])
    store_i32(red, 896 + 2 * lane, gl.full([512], 0, gl.int32, lane.type.layout))
    store_i32(red, 896 + 2 * lane + 1, gl.full([512], 0, gl.int32, lane.type.layout))
    for i in gl.static_range(10):
        span = ROW_NET[i][0]
        distance = ROW_NET[i][1]
        partner = shuffle(maximum, lane ^ distance)
        if span < 16:
            keep = ((lane & distance) == 0) == ((lane & span) == 0)
        else:
            keep = (lane & distance) == 0
        maximum = gl.where(
            keep, gl.maximum(maximum, partner), gl.minimum(maximum, partner)
        )
    # readlane reads the fixed lanes straight into scalar registers.
    threshold = readlane(
        maximum.to(gl.int32), gl.full([512], 3, gl.int32, lane.type.layout)
    ).to(gl.uint32)
    for row in gl.static_range(1, 4):
        threshold = gl.minimum(
            threshold,
            readlane(
                maximum.to(gl.int32),
                gl.full([512], 16 * row + 3, gl.int32, lane.type.layout),
            ).to(gl.uint32),
        )
    total = gl.full([512], 0, gl.int32, lane.type.layout)
    for i in gl.static_range(14):
        hit = keys[i] >= threshold
        ballot = wave_ballot_i64(hit.to(gl.int32))
        at = total + wave_mbcnt(ballot)
        live = hit & (at < 64)
        store_mask_i32(red, 896 + 2 * at, 1023 - (lane + 64 * i), live)
        store_mask_i32(red, 896 + 2 * at + 1, keys[i].to(gl.int32), live)
        total = total + wave_popcount_i64(ballot)
    low = load_i32(red, 896 + 2 * lane)
    high = load_i32(red, 896 + 2 * lane + 1)
    mine = (high.to(gl.uint32).to(gl.uint64) << 32) | low.to(gl.uint32).to(gl.uint64)
    limit = gl.minimum(total, 64)
    rank = gl.full([512], 0, gl.int32, lane.type.layout)
    i = 0
    # The candidate count differs per wave, so this loop is per wave; readlane
    # broadcasts each other candidate without an LDS round trip.
    while wave_pending(i < limit):
        for u in gl.static_range(8):
            other_low = readlane(
                low, gl.full([512], 0, gl.int32, lane.type.layout) + i + u
            )
            other_high = readlane(
                high, gl.full([512], 0, gl.int32, lane.type.layout) + i + u
            )
            other = (other_high.to(gl.uint32).to(gl.uint64) << 32) | other_low.to(
                gl.uint32
            ).to(gl.uint64)
            rank = rank + (other > mine).to(gl.int32)
        i += 8
    expert = 1023 - (mine & 1023).to(gl.int32)
    store_mask_i32(red, 896 + rank, expert, (lane < limit) & (rank < 16))
    selected = load_i32(red, 896 + gl.minimum(lane, 15))
    # Only a wave with more than 64 candidates takes this fallback.
    if wave_pending(total > 64):
        selected = extract(lane, keys, sigmoids, red)
    return selected, load_i32(red, gl.minimum(gl.maximum(selected, 0), 895))


@gluon.jit
def stage_route(
    rlogp,
    parts,
    bias,
    route,
    red,
    token,
    tag,
    S: gl.constexpr,
    guard,
):
    tid = thread_ids()
    lane = tid % 64
    # The bias loads are in flight across the wait for the logits.
    biases = ()
    for i in gl.static_range(14):
        biases += (gl.load(bias + lane + 64 * i),)
    route_parts(rlogp, parts, token, tag, S, guard)
    # Wave 0 alone routes the rows, so the LDS accesses here and in
    # rank_candidates and extract use the barrier-free helpers.
    if wave_pending(tid < 64):
        logits = ()
        for i in gl.static_range(14):
            expert = lane + 64 * i
            value = load_f32(parts, expert)
            for q in gl.static_range(1, 4):
                value = value + load_f32(parts, q * 896 + expert)
            logits += (value,)
        keys = ()
        sigmoids = ()
        for i in gl.static_range(14):
            sigmoid = gl.extra.libdevice.fast_dividef(
                1.0, 1.0 + gl.exp2(logits[i] * -1.4426950408889634)
            )
            score = sigmoid + biases[i]
            score = gl.where(score == score, score, -3.0e38)
            bits = score.to(gl.int32, bitcast=True)
            keys += (gl.where(bits < 0, ~bits, bits ^ (-2147483648)).to(gl.uint32),)
            sigmoids += (gl.where(sigmoid == sigmoid, sigmoid, 0.0),)
        selected, probability_bits = rank_candidates(lane, keys, sigmoids, red)
        total = gl.full([512], 0, gl.float32, lane.type.layout)
        for pick in gl.static_range(16):
            # Sum the 16 picks' weights from their lanes in scalar registers.
            total = total + readlane(
                probability_bits, gl.full([512], pick, gl.int32, lane.type.layout)
            ).to(gl.float32, bitcast=True)
        inverse = gl.div_rn(1.0, total)
        put_pair(
            route,
            token * 32 + lane,
            gl.minimum(gl.maximum(selected, 0), 895),
            tag,
            tid < 16,
            "gpu",
        )
        put_pair(
            route,
            token * 32 + 16 + lane,
            probability_bits.to(gl.float32, bitcast=True) * inverse,
            tag,
            tid < 16,
            "gpu",
        )
    gl.barrier()


@gluon.jit
def load_route(route, rt_route, flag, experts, kof, tag, S: gl.constexpr, guard):
    tid = thread_ids()
    lane = tid % 64
    for i in gl.static_range(2):
        store_i32(kof, tid + 512 * i, gl.full([512], -1, gl.int32, tid.type.layout))
    store_mask_i32(flag, tid, gl.full([512], 0, gl.int32, tid.type.layout), tid < 28)
    value = poll_pairs(route, (tid,), tag, tid < S * 32, "gpu", guard)[0].to(
        gl.int32, bitcast=True
    )
    store_mask_i32(rt_route, tid, value, tid < S * 32)
    gl.barrier()
    token, pick = tid // 16, tid % 16
    expert = gl.minimum(
        gl.maximum(load_i32(rt_route, gl.minimum(token * 32 + pick, S * 32 - 1)), 0),
        895,
    )
    flag.atomic_scatter_or(
        gl.full([512], 1, gl.int32, tid.type.layout) << (expert & 31),
        expert >> 5,
        0,
        mask=tid < S * 16,
    )
    gl.barrier()
    # Wave 0 alone builds the per-expert offsets; its LDS accesses add no
    # barrier.
    if wave_pending(tid < 64):
        word = gl.where(lane < 28, load_i32(flag, gl.minimum(lane, 27)), 0)
        count = wave_popcount_i64(word.to(gl.uint32).to(gl.int64))
        inclusive = wave_inclusive_sum_i32(count)
        store_mask_i32(flag, 32 + lane, inclusive - count, lane < 28)
        store_mask_i32(
            flag, gl.full([512], 60, gl.int32, tid.type.layout), inclusive, lane == 27
        )
    gl.barrier()
    word = load_i32(flag, expert >> 5)
    below = word & ((gl.full([512], 1, gl.int32, tid.type.layout) << (expert & 31)) - 1)
    slot = load_i32(flag, 32 + (expert >> 5)) + wave_popcount_i64(
        below.to(gl.uint32).to(gl.int64)
    )
    store_mask_i32(experts, slot, expert, tid < S * 16)
    store_mask_i32(kof, slot * 8 + token, pick, tid < S * 16)
    # The union's size, from its scalar index (LDS word 60).
    total = load_i32(flag, gl.to_tensor(60))
    gl.barrier()
    return total
