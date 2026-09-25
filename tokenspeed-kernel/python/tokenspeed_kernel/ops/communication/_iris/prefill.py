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

"""CDNA4 token-sharded MoE reduction and projected-output gathering."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.sync import _iris_drain_subgroup_vmem


@gluon.jit
def _prefill_store_completion(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Finish every subgroup's write-through stores before publishing completion.
    _iris_drain_subgroup_vmem()
    gl.barrier()
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    gl.store(peer_flags + block_id * 8 + RANK, epoch, mask=remote, cache_modifier=".wt")
    local = flags + block_id * 8 + peers
    seen = gl.load(local, mask=remote, other=epoch, cache_modifier=".cv", volatile=True)
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(
            local, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
        )
    gl.atomic_add(local, 0, mask=remote, sem="acquire", scope="sys")


@gluon.jit
def _prefill_entry_barrier(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Prior producers complete on the calling stream before this entry barrier.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    gl.atomic_xchg(
        peer_flags + block_id * 8 + RANK,
        epoch,
        mask=remote,
        sem="release",
        scope="sys",
    )
    local_flags = flags + block_id * 8 + peers
    seen = gl.load(
        local_flags, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
    )
    # Accept newer epochs across uint32 wraparound; peers stay within one call.
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(
            local_flags, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
        )
    # Acquire lowering already joins the workgroup after cache invalidation.
    gl.atomic_add(local_flags, 0, mask=remote, sem="acquire", scope="sys")


@gluon.jit
def _peer_buffers(pointer, heaps, RANK: gl.constexpr):
    offset = pointer.to(gl.uint64) - heaps[RANK]
    result = ()
    for peer in gl.static_range(8):
        result += (
            gl.multiple_of((heaps[peer] + offset).to(gl.pointer_type(gl.bfloat16)), 16),
        )
    return result


@gluon.jit
def _peer_flags(pointer, heaps, RANK: gl.constexpr, NUM_WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    bases = gl.full((8,), 0, gl.uint64, layout)
    for peer in gl.static_range(8):
        bases = gl.where(peers == peer, heaps[peer], bases)
    offset = pointer.to(gl.uint64) - heaps[RANK]
    return (bases + offset).to(gl.pointer_type(gl.uint32))


@gluon.jit
def iris_moe_reduce_scatter_gluon_kernel(
    input_ptr,
    scratch_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    ROWS: gl.constexpr,
    FIRST_WIDTH: gl.constexpr,
    SECOND_WIDTH: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    gl.static_assert(ROWS % 8 == 0)
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    inputs = _peer_buffers(input_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    block_id = gl.program_id(0)
    epoch = (
        gl.atomic_add(flags + block_id * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    )
    _prefill_entry_barrier(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)
    FIRST_ELEMENTS: gl.constexpr = ROWS // 8 * FIRST_WIDTH
    SECOND_ELEMENTS: gl.constexpr = ROWS // 8 * SECOND_WIDTH
    PARTITION_ELEMENTS: gl.constexpr = FIRST_ELEMENTS + SECOND_ELEMENTS
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(
        block_id, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS
    ):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        source = gl.where(
            offsets < FIRST_ELEMENTS,
            RANK * FIRST_ELEMENTS + offsets,
            ROWS * FIRST_WIDTH + RANK * SECOND_ELEMENTS + offsets - FIRST_ELEMENTS,
        )
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[(RANK + step) % 8], source, mask, 0, cache=".cg"
                ),
            )
        sums = ()
        for peer in gl.static_range(2):
            sums += (
                values[(peer - RANK) % 8].to(gl.float32)
                + values[(peer + 2 - RANK) % 8].to(gl.float32),
                values[(peer + 4 - RANK) % 8].to(gl.float32)
                + values[(peer + 6 - RANK) % 8].to(gl.float32),
            )
        reduced = ((sums[0] + sums[1]) + (sums[2] + sums[3])).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(reduced, scratch_ptr, offsets, mask, cache=".wt")


@gluon.jit
def iris_moe_add_push_gather_gluon_kernel(
    projection_ptr,
    shared_ptr,
    prefix_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    PARTITION_ELEMENTS: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
):
    # In-place prefixes are safe: ranks read then write disjoint rows.
    # Reduce-scatter entry waits for prior prefix consumers.
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    block_id = gl.program_id(0)
    epoch = (
        gl.atomic_add(flags + block_id * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    )
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(
        block_id, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS
    ):
        offsets = tile * BLOCK_ELEMENTS + lanes
        if PARTITION_ELEMENTS % BLOCK_ELEMENTS == 0:
            mask = gl.full((BLOCK_ELEMENTS,), True, gl.int1, layout)
        else:
            mask = offsets < PARTITION_ELEMENTS
        prefix_offsets = offsets
        if not PREFIX_IS_SHARDED:
            prefix_offsets += RANK * PARTITION_ELEMENTS
        a = gl.amd.cdna4.buffer_load(
            prefix_ptr, prefix_offsets, mask, 0, cache=".cg"
        ).to(gl.float32)
        b = gl.amd.cdna4.buffer_load(projection_ptr, offsets, mask, 0, cache=".cg").to(
            gl.float32
        )
        c = gl.amd.cdna4.buffer_load(shared_ptr, offsets, mask, 0, cache=".cg").to(
            gl.float32
        )
        result = (a + b + c).to(gl.bfloat16)
        # Precomputed peer bases keep stores consecutive without VMEM drains.
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                result,
                outputs[(RANK + step) % 8],
                offsets + RANK * PARTITION_ELEMENTS,
                mask,
                cache=".wt",
            )
    _prefill_store_completion(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)
