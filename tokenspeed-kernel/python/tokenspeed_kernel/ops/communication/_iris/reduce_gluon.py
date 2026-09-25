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

"""TP8 BF16 producer-direct reduction using one subgroup per workgroup."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.prefill import _peer_buffers, _peer_flags
from tokenspeed_kernel.ops.communication._iris.sync import _iris_drain_subgroup_vmem


def _reduce_metadata(grid, kernel, args):
    elements = args["PARTITION_ELEMENTS"]
    return {"name": kernel.name, "bytes": elements * 50, "flops32": elements * 7}


@gluon.jit
def _reduce_barrier(
    flags, peer_flags, block_id, epoch, RANK: gl.constexpr, ENTRY: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    if ENTRY:
        gl.atomic_xchg(
            peer_flags + block_id * 8 + RANK,
            epoch,
            mask=remote,
            sem="release",
            scope="sys",
        )
    else:
        # The only subgroup drains all payload stores before publishing them.
        _iris_drain_subgroup_vmem()
        gl.store(
            peer_flags + block_id * 8 + RANK,
            epoch,
            mask=remote,
            cache_modifier=".wt",
        )
    incoming = flags + block_id * 8 + peers
    seen = gl.load(
        incoming, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
    )
    while gl.min((seen - epoch).to(gl.int32), 0) < 0:
        seen = gl.load(
            incoming, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
        )
    gl.atomic_add(incoming, 0, mask=remote, sem="acquire", scope="sys")


@gluon.jit(launch_metadata=_reduce_metadata)
def iris_reduce_symmetric_register_gluon_kernel(
    input_ptr,
    scratch_ptr,
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
):
    """Reduce-scatter into private scratch, then pull into an owned output."""
    gl.static_assert(PARTITION_ELEMENTS % 8 == 0)
    gl.static_assert(input_ptr.dtype.element_ty == gl.bfloat16)
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
    scratch = _peer_buffers(scratch_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, 1)
    block_id = gl.program_id(0)
    epoch_ptr = flags + block_id * 8 + RANK
    # Each diagonal counter has one owner. Calls are ordered on one stream;
    # the non-diagonal flags publish this kernel's entry and completion.
    epoch = gl.load(epoch_ptr, cache_modifier=".cv", volatile=True) + 1
    _reduce_barrier(flags, peer_flags, block_id, epoch, RANK, True)

    layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    tile = block_id
    while tile < gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        source = RANK * PARTITION_ELEMENTS + offsets
        mask = offsets < PARTITION_ELEMENTS
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[(RANK + step) % 8], source, mask, 0, cache=".cv"
                ),
            )
        # Match the even/odd FP32 tree in the existing two-stage reduction.
        even = (
            values[(0 - RANK) % 8].to(gl.float32)
            + values[(2 - RANK) % 8].to(gl.float32)
        ) + (
            values[(4 - RANK) % 8].to(gl.float32)
            + values[(6 - RANK) % 8].to(gl.float32)
        )
        odd = (
            values[(1 - RANK) % 8].to(gl.float32)
            + values[(3 - RANK) % 8].to(gl.float32)
        ) + (
            values[(5 - RANK) % 8].to(gl.float32)
            + values[(7 - RANK) % 8].to(gl.float32)
        )
        gl.amd.cdna4.buffer_store(
            (even + odd).to(gl.bfloat16), scratch_ptr, offsets, mask, cache=".wt"
        )
        tile += NUM_PROGRAMS

    _reduce_barrier(flags, peer_flags, block_id, epoch + 1, RANK, False)
    tile = block_id
    while tile < gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    scratch[(RANK + step) % 8], offsets, mask, 0, cache=".cv"
                ),
            )
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                values[step],
                output_ptr,
                ((RANK + step) % 8) * PARTITION_ELEMENTS + offsets,
                mask,
                cache=".wb",
            )
        tile += NUM_PROGRAMS
    gl.store(epoch_ptr, epoch + 1, cache_modifier=".wt")
    # The intermediate rendezvous joins every input reader before input reuse.
    # A following scratch writer must rendezvous at entry, after each rank has
    # completed this gather on its calling stream.
