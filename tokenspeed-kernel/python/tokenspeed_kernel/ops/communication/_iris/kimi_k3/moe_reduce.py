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

The gathered decode tail reads only its own column block of the shared
output, so ``iris_k3moe_lamport_twoshot_shared_rs`` all-reduces the routed
rows and only reduce-scatters the shared ones.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_lamport_exchange_tile,
    _iris_lamport_push,
    _iris_lamport_twoshot_tile,
    _iris_sanitize_lamport_bf16,
    _iris_wait_lamport_slots,
)


@gluon.jit(do_not_specialize=["TOTAL_ELEMENTS", "ROUTED_ELEMENTS"])
def iris_k3moe_lamport_allreduce_topk(
    partials_ptr,
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS,
    ROUTED_ELEMENTS,
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
    gl.static_assert(ROUTED_WIDTH % 512 == 0)
    # The launcher validates the complete-tile payload and mailbox bound.
    TOTAL_ELEMENTS = gl.multiple_of(TOTAL_ELEMENTS, 512)
    ROUTED_ELEMENTS = gl.multiple_of(ROUTED_ELEMENTS, 512)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    # Payload bounds are tile-aligned at the launcher; every lane in this
    # tile has the same predicate, preserving 16-byte peer transactions.
    mask = gl.full([64, 8], True, gl.int1, layout) & (
        gl.program_id(0) * 512 < TOTAL_ELEMENTS
    )
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


@gluon.jit
def _k3moe_routed_twoshot_program(
    partials_ptr,
    input_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    offsets,
    shard,
    stage,
    layout: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    ROUTED_WIDTH: gl.constexpr,
    TOPK: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Two-shot one tile of each ``shard``-element routed shard, as
    ``iris_k3moe_lamport_twoshot_topk`` (``TOPK``) or the plain two-shot does."""
    TILE: gl.constexpr = offsets.shape[0] * offsets.shape[1]
    chunks = ()
    for delta in gl.static_range(WORLD_SIZE):
        owner = (RANK + delta) % WORLD_SIZE
        if TOPK:
            token = (owner * shard + gl.program_id(0) * TILE) // ROUTED_WIDTH
            slot_offsets = token * (TOPK - 1) * ROUTED_WIDTH + owner * shard + offsets
            summed = gl.zeros(offsets.shape, gl.float32, layout)
            for slot in gl.static_range(TOPK):
                summed += gl.amd.cdna4.buffer_load(
                    partials_ptr, slot_offsets + slot * ROUTED_WIDTH
                ).to(gl.float32)
            local = summed.to(gl.bfloat16)
        else:
            local = gl.amd.cdna4.buffer_load(input_ptr + owner * shard, offsets)
        chunks += (_iris_sanitize_lamport_bf16(local),)
    _iris_lamport_twoshot_tile(
        chunks,
        region_sym_ptr,
        output_ptr,
        epochs,
        regions,
        offsets,
        shard,
        stage,
        layout,
        RANK,
        WORLD_SIZE,
        MAX_SHARD_ELEMENTS,
        NUM_STAGES,
        PEER_SIGNAL_STORE_CACHE,
    )


@gluon.jit
def _k3moe_shared_rs_program(
    shared_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    lane,
    offsets,
    tile,
    stage,
    layout: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    SHARED_WIDTH: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Reduce-scatter shared tile ``tile`` of every rank's column block.

    Block b of a row is its columns [b, b + 1) * SHARED_WIDTH / WORLD_SIZE.
    Rank RANK receives its block's tile from every peer through the two-shot's
    reduce-scatter mailboxes and stores the reduction only there; nothing is
    gathered. ``offsets`` are the program's mailbox elements.
    """
    TILE: gl.constexpr = offsets.shape[0] * offsets.shape[1]
    BLOCK: gl.constexpr = SHARED_WIDTH // WORLD_SIZE
    TILES: gl.constexpr = BLOCK // TILE
    first = (tile // TILES) * SHARED_WIDTH + (tile % TILES) * TILE
    chunks = ()
    for delta in gl.static_range(WORLD_SIZE):
        owner = (RANK + delta) % WORLD_SIZE
        chunks += (
            _iris_sanitize_lamport_bf16(
                gl.amd.cdna4.buffer_load(shared_ptr + first + owner * BLOCK, lane)
            ),
        )
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SHARD_ELEMENTS
    slots = stage * (2 * PHASE)
    _iris_lamport_push(
        chunks[1:],
        regions,
        slots + RANK * MAX_SHARD_ELEMENTS,
        offsets,
        None,
        gl.program_id(0),
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )
    valid = gl.full([offsets.shape[0]], True, gl.int1, gl.SliceLayout(1, layout))
    scattered = _iris_wait_lamport_slots(
        region_sym_ptr + slots,
        offsets,
        valid,
        RANK,
        WORLD_SIZE,
        MAX_SHARD_ELEMENTS,
        layout,
    )
    # The two-shot's per-element arithmetic: FP32 rank order, one BF16 rounding.
    for source in gl.static_range(0, WORLD_SIZE):
        if source == RANK:
            term = chunks[0]
        else:
            term = scattered[(source - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if source == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(
        _iris_sanitize_lamport_bf16(accumulator.to(gl.bfloat16)),
        output_ptr + first + RANK * BLOCK,
        lane,
    )

    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        source = (RANK + delta) % WORLD_SIZE
        gl.amd.cdna4.buffer_store(
            sentinel,
            region_sym_ptr + slots + source * MAX_SHARD_ELEMENTS,
            offsets,
            cache=".wt",
        )
    # Order the sentinel restores before this program's stage advance.
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((stage + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit(do_not_specialize=["ROUTED_ELEMENTS"])
def iris_k3moe_lamport_twoshot_shared_rs(
    partials_ptr,
    input_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    ROUTED_ELEMENTS,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    ROUTED_WIDTH: gl.constexpr,
    SHARED_WIDTH: gl.constexpr,
    TOPK: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """The routed/shared two-shot whose shared rows only reduce-scatter.

    ``input_ptr`` holds ``[rows, ROUTED_WIDTH]`` routed then ``[rows,
    SHARED_WIDTH]`` shared BF16 rows; with ``TOPK`` the routed rows instead
    sum ``partials_ptr``'s top-k slots as ``iris_k3moe_lamport_twoshot_topk``
    does. The first ROUTED_ELEMENTS // (WORLD_SIZE * 128) programs two-shot
    the routed section in WORLD_SIZE contiguous shards; the rest reduce-scatter
    the shared section by column block, so ``output_ptr``'s shared rows hold
    only this rank's block. Each element is the plain two-shot's: inputs and
    results sanitized, an FP32 sum in rank order, one BF16 rounding. Programs
    use the two-shot's mailboxes and epochs as it does, so the two interleave.
    """
    gl.static_assert(WORLD_SIZE == 8)
    VECTOR: gl.constexpr = 2
    TILE: gl.constexpr = 64 * VECTOR
    gl.static_assert(ROUTED_WIDTH % TILE == 0)
    gl.static_assert(SHARED_WIDTH % (WORLD_SIZE * TILE) == 0)
    gl.static_assert(MAX_SHARD_ELEMENTS % TILE == 0)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [64, 1], [1, 1], [0, 1])
    packs = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, VECTOR, layout=gl.SliceLayout(0, layout))
    lane = packs[:, None] * VECTOR + element[None, :]
    offsets = gl.program_id(0) * TILE + lane
    ROUTED_ELEMENTS = gl.multiple_of(ROUTED_ELEMENTS, WORLD_SIZE * TILE)
    shard = gl.multiple_of(ROUTED_ELEMENTS // WORLD_SIZE, TILE)
    routed_programs = ROUTED_ELEMENTS // (WORLD_SIZE * TILE)
    stage = gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    if gl.program_id(0) < routed_programs:
        _k3moe_routed_twoshot_program(
            partials_ptr,
            input_ptr,
            region_sym_ptr,
            output_ptr,
            epochs,
            regions,
            offsets,
            shard,
            stage,
            layout,
            RANK,
            WORLD_SIZE,
            ROUTED_WIDTH,
            TOPK,
            MAX_SHARD_ELEMENTS,
            NUM_STAGES,
            PEER_SIGNAL_STORE_CACHE,
        )
    else:
        _k3moe_shared_rs_program(
            input_ptr + ROUTED_ELEMENTS,
            region_sym_ptr,
            output_ptr + ROUTED_ELEMENTS,
            epochs,
            regions,
            lane,
            offsets,
            gl.program_id(0) - routed_programs,
            stage,
            layout,
            RANK,
            WORLD_SIZE,
            SHARED_WIDTH,
            MAX_SHARD_ELEMENTS,
            NUM_STAGES,
            PEER_SIGNAL_STORE_CACHE,
        )
