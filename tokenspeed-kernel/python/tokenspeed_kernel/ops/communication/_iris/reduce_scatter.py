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


"""TP4/TP8 pull reduce-scatter over consecutive, possibly uneven token rows."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_peer_pointers,
    _iris_rendezvous,
)


@gluon.jit(do_not_specialize=["ROWS"])
def iris_pull_reduce_scatter(
    input_ptr,
    output_ptr,
    ready_flags,
    heaps,
    ROWS,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    BLOCK: gl.constexpr,
    PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    EXIT_BARRIER: gl.constexpr,
    PRODUCER_WRITES_THROUGH: gl.constexpr,
    WAIT_VMEM: gl.constexpr,
):
    """Reduce BF16 contributions into consecutive, possibly uneven rows.

    Sum input_ptr's [ROWS, WIDTH] partials in FP32 and write a BF16 shard of
    ceil(ROWS / WORLD_SIZE) rows to output_ptr on each rank, zeroing the padding.
    ready_flags publish the inputs; EXIT_BARRIER waits for peer reads to finish.
    PRODUCER_WRITES_THROUGH requires WIDTH=4096 and EXIT_BARRIER; ordinary
    producers publish with release ordering and support the caller's width.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    if PRODUCER_WRITES_THROUGH:
        gl.static_assert(EXIT_BARRIER and WIDTH == 4096)
    inputs = _iris_peer_pointers(input_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    pid = gl.program_id(0)
    epoch = (
        gl.atomic_add(flags + pid * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu")
        + 1
    )
    _iris_rendezvous(
        flags, heaps, pid, epoch, RANK, NUM_WARPS, not PRODUCER_WRITES_THROUGH
    )
    local_rows = gl.cdiv(ROWS, WORLD_SIZE)
    elements = local_rows * WIDTH
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    for tile in range(pid, gl.cdiv(elements, BLOCK), PROGRAMS):
        offsets = tile * BLOCK + lanes
        source = RANK * elements + offsets
        mask = (offsets < elements) & (source < ROWS * WIDTH)
        values = ()
        for step in gl.static_range(WORLD_SIZE):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[(RANK + step) % WORLD_SIZE], source, mask, 0, cache=".cg"
                ).to(gl.float32),
            )
        # All owners use the same sum order and round the result to BF16.
        reduced = values[(0 - RANK) % WORLD_SIZE]
        for peer in gl.static_range(1, WORLD_SIZE):
            reduced += values[(peer - RANK) % WORLD_SIZE]
        gl.amd.cdna4.buffer_store(
            reduced.to(gl.bfloat16),
            output_ptr,
            offsets,
            offsets < elements,
            cache=".wt",
        )
    if EXIT_BARRIER:
        WAIT_VMEM(args=[], constraints="=r,~{memory}")
        gl.barrier()
        done = (
            gl.atomic_add(
                flags + pid * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
            )
            + 1
        )
        _iris_rendezvous(flags, heaps, pid, done, RANK, NUM_WARPS, False)
