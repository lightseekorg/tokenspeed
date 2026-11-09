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

"""Reduce K3 decode MoE outputs whose routed rows are still per-slot partials.

The decode MoE's second stage writes one routed row per (token, top-k slot).
The Lamport all-reduces' routed tiles (one-shot, and two-shot from 8 rows)
sum those slots themselves, in the separate top-k reduce's order, so that
reduce never launches.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_lamport_exchange_tile,
    _iris_lamport_twoshot_tile,
    _iris_sanitize_lamport_bf16,
)


@gluon.jit
def iris_k3moe_lamport_allreduce_topk(
    partials_ptr,
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    ROUTED_ELEMENTS: gl.constexpr,
    ROUTED_WIDTH: gl.constexpr,
    TOPK: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """``iris_lamport_allreduce`` whose first ROUTED_ELEMENTS come from top-k partials.

    ``partials_ptr`` holds contiguous ``[tokens * TOPK, ROUTED_WIDTH]`` BF16
    rows, token-major. Routed elements are their FP32 sum over the slots in
    slot order, rounded once to BF16, as ``gluon_mxfp4_moe_stage2_reduce_kernel``
    computes them; the remaining elements load from ``input_sym_ptr`` as the
    plain all-reduce does.
    """
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(TOTAL_ELEMENTS % 512 == 0 and ROUTED_WIDTH % 512 == 0)
    gl.static_assert(ROUTED_ELEMENTS % ROUTED_WIDTH == 0)
    gl.static_assert(ROUTED_ELEMENTS < TOTAL_ELEMENTS <= MAX_ELEMENTS)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    mask = offsets < TOTAL_ELEMENTS
    generation = (
        gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    )
    if gl.program_id(0) * 512 < ROUTED_ELEMENTS:
        token = gl.program_id(0) * 512 // ROUTED_WIDTH
        slot_offsets = token * (TOPK - 1) * ROUTED_WIDTH + offsets
        summed = gl.zeros([64, 8], gl.float32, layout)
        for slot in gl.static_range(TOPK):
            summed += gl.amd.cdna4.buffer_load(
                partials_ptr, slot_offsets + slot * ROUTED_WIDTH
            ).to(gl.float32)
        local = summed.to(gl.bfloat16)
    else:
        local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    _iris_lamport_exchange_tile(
        _iris_sanitize_lamport_bf16(local),
        region_sym_ptr,
        output_ptr,
        epochs,
        regions,
        generation,
        pack,
        offsets,
        mask,
        RANK,
        WORLD_SIZE,
        TOTAL_ELEMENTS,
        MAX_ELEMENTS,
        NUM_STAGES,
        PEER_SIGNAL_STORE_CACHE,
    )


@gluon.jit
def iris_k3moe_lamport_twoshot_topk(
    partials_ptr,
    input_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    ROUTED_ELEMENTS: gl.constexpr,
    ROUTED_WIDTH: gl.constexpr,
    TOPK: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """``iris_lamport_twoshot_bf16_allreduce`` whose first ROUTED_ELEMENTS
    come from top-k partials, as in ``iris_k3moe_lamport_allreduce_topk``.

    A program's 128-element tiles never straddle a routed row or the routed
    section's end: shards and routed rows are whole tiles.
    """
    gl.static_assert(WORLD_SIZE == 8)
    VECTOR: gl.constexpr = 2
    TILE: gl.constexpr = 64 * VECTOR
    gl.static_assert(TOTAL_ELEMENTS % (WORLD_SIZE * TILE) == 0)
    gl.static_assert((TOTAL_ELEMENTS // WORLD_SIZE) % TILE == 0)
    gl.static_assert(ROUTED_WIDTH % TILE == 0)
    gl.static_assert(ROUTED_ELEMENTS % ROUTED_WIDTH == 0)
    gl.static_assert(ROUTED_ELEMENTS < TOTAL_ELEMENTS)
    gl.static_assert(MAX_SHARD_ELEMENTS % TILE == 0)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, VECTOR, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * VECTOR + element[None, :]
    SHARD: gl.constexpr = TOTAL_ELEMENTS // WORLD_SIZE
    stage = gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    chunks = ()
    for delta in gl.static_range(WORLD_SIZE):
        owner = (RANK + delta) % WORLD_SIZE
        start = owner * SHARD + gl.program_id(0) * TILE
        if start < ROUTED_ELEMENTS:
            token = start // ROUTED_WIDTH
            slot_offsets = token * (TOPK - 1) * ROUTED_WIDTH + owner * SHARD + offsets
            summed = gl.zeros([64, VECTOR], gl.float32, layout)
            for slot in gl.static_range(TOPK):
                summed += gl.amd.cdna4.buffer_load(
                    partials_ptr, slot_offsets + slot * ROUTED_WIDTH
                ).to(gl.float32)
            local = summed.to(gl.bfloat16)
        else:
            local = gl.amd.cdna4.buffer_load(input_ptr + owner * SHARD, offsets)
        chunks += (_iris_sanitize_lamport_bf16(local),)
    _iris_lamport_twoshot_tile(
        chunks,
        region_sym_ptr,
        output_ptr,
        epochs,
        regions,
        offsets,
        SHARD,
        stage,
        layout,
        RANK,
        WORLD_SIZE,
        MAX_SHARD_ELEMENTS,
        NUM_STAGES,
        PEER_SIGNAL_STORE_CACHE,
    )
