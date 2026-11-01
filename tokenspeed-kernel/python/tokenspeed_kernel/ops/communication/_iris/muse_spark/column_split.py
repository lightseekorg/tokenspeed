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


"""Muse expert TP prefill with the down projection split by output columns.

Each rank's gate/up GEMM writes its intermediate slice as sorted E2M1 rows. The
gather assembles the full intermediate on every rank, each rank projects its
own output columns of every route, and the routed norm returns complete token
rows to their owners. See ``MuseIrisState.gather_expert_intermediate``.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_peer_pointers,
    _iris_rendezvous,
)


@gluon.jit(do_not_specialize=["rows"])
def iris_pull_allgather_sorted_mxfp4_slices(
    slice_data,
    slice_scales,
    sorted_ids,
    num_valid_ids,
    output,
    output_scales,
    ready_flags,
    heaps,
    rows,
    RANK: gl.constexpr,
    SLICE_WORDS: gl.constexpr,
    PANEL_WORDS: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_VMEM: gl.constexpr,
):
    """Concatenate every rank's sorted E2M1 K slice into full-K rows.

    slice_data holds this rank's sorted rows of SLICE_WORDS 32-bit words and
    slice_scales its CDNA4 scale panels of PANEL_WORDS words per 32 rows; both
    are symmetric. Rank r's words land at word offset r * SLICE_WORDS of each
    output row and r * PANEL_WORDS of each output panel, which is the full-K
    layout because the scale swizzle keeps each 8-column K block contiguous in
    a panel. Every rank must have sorted the same top-k IDs deterministically.
    Rows past num_valid_ids[0] and padding routes (token field >= rows) are not
    copied. ready_flags publish the producer's write-back stores and hold the
    slices until every peer has read them.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    program = gl.program_id(0)
    data_peers = _iris_peer_pointers(slice_data, heaps, RANK)
    scale_peers = _iris_peer_pointers(slice_scales, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    # The gate/up GEMM, an earlier kernel in this stream, wrote the slice
    # with ordinary stores; its end-of-kernel release makes them visible.
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    valid_rows = gl.load(num_valid_ids)
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    words = valid_rows * SLICE_WORDS
    for tile in range(program, gl.cdiv(words, BLOCK), gl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        row = offsets // SLICE_WORDS
        inside = offsets < words
        token = gl.load(sorted_ids + row, mask=inside, other=0) & 0xFFFFFF
        mask = inside & (token < rows)
        destination = row * (WORLD_SIZE * SLICE_WORDS) + offsets % SLICE_WORDS
        # Issue every peer's load before the first store so they overlap.
        values = ()
        for step in gl.static_range(WORLD_SIZE):
            values += (
                gl.amd.cdna4.buffer_load(
                    data_peers[(RANK + step) % WORLD_SIZE],
                    offsets,
                    mask,
                    0,
                    cache=".cg",
                ),
            )
        for step in gl.static_range(WORLD_SIZE):
            gl.amd.cdna4.buffer_store(
                values[step],
                output,
                destination + ((RANK + step) % WORLD_SIZE) * SLICE_WORDS,
                mask,
            )
    scale_words = gl.cdiv(valid_rows, 32) * PANEL_WORDS
    for tile in range(program, gl.cdiv(scale_words, BLOCK), gl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        mask = offsets < scale_words
        destination = (offsets // PANEL_WORDS) * (
            WORLD_SIZE * PANEL_WORDS
        ) + offsets % PANEL_WORDS
        values = ()
        for step in gl.static_range(WORLD_SIZE):
            values += (
                gl.amd.cdna4.buffer_load(
                    scale_peers[(RANK + step) % WORLD_SIZE],
                    offsets,
                    mask,
                    0,
                    cache=".cg",
                ),
            )
        for step in gl.static_range(WORLD_SIZE):
            gl.amd.cdna4.buffer_store(
                values[step],
                output_scales,
                destination + ((RANK + step) % WORLD_SIZE) * PANEL_WORDS,
                mask,
            )
    # The next gate/up GEMM may overwrite the slices once every peer is done.
    WAIT_VMEM(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _iris_rendezvous(
        ready_flags + FLAG_ROWS * WORLD_SIZE,
        heaps,
        program,
        epoch,
        RANK,
        NUM_WARPS,
        False,
    )


@gluon.jit(do_not_specialize=["rows"])
def iris_column_routed_rmsnorm_sum_return(
    routes,
    route_weights,
    norm_weight,
    partial_squares,
    slices,
    output,
    flags,
    heaps,
    rows,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
    WAIT_VMEM: gl.constexpr,
):
    """Normalize column-split routes and return complete rows to token owners.

    routes holds complete BF16 values of this rank's WIDTH / WORLD_SIZE
    columns for all ``rows * 8`` routes in token/slot order. Each route's
    RMSNorm scales by the learned gain first, then sums squares over all WIDTH
    columns: ranks exchange FP32 partial sums through partial_squares and add
    them in rank order, so every rank applies the same scale. Weighted slot
    sums are stored to slices, then owner rank o pulls rows
    [o * L, o * L + L), L = ceil(rows / WORLD_SIZE), from every rank into
    output [L, WIDTH], zeroing rows past ``rows``. Program p handles tokens
    t = p (mod programs) in every phase on every rank, so per-program flags
    order each exchange; the exit waits for peer reads of both buffers.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    COLUMNS: gl.constexpr = WIDTH // WORLD_SIZE
    gl.static_assert(COLUMNS % (64 * NUM_WARPS) == 0)
    program = gl.program_id(0)
    programs = gl.num_programs(0)
    square_peers = _iris_peer_pointers(partial_squares, heaps, RANK)
    slice_peers = _iris_peer_pointers(slices, heaps, RANK)
    epoch = (
        gl.atomic_add(
            flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    tile_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 64], [NUM_WARPS, 1], [1, 0]
    )
    # One element per lane slot: COLUMNS spans the workgroup exactly.
    row_layout: gl.constexpr = gl.BlockedLayout(
        [COLUMNS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    slots = gl.arange(0, 8, layout=gl.SliceLayout(1, tile_layout))
    tile_columns = gl.arange(0, COLUMNS, layout=gl.SliceLayout(0, tile_layout))
    columns = gl.arange(0, COLUMNS, layout=row_layout)
    gain = gl.load(norm_weight + RANK * COLUMNS + columns).to(gl.float32)
    gain = (gain + WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    tile_gain = gl.convert_layout(gain, gl.SliceLayout(0, tile_layout))

    # Phase 1: this rank's share of every route's sum of squares.
    for token in range(program, rows, programs):
        offsets = (token * 8 + slots[:, None]) * COLUMNS + tile_columns[None, :]
        values = gl.load(routes + offsets).to(gl.float32) * tile_gain[None, :]
        squares = gl.sum(values * values, 1)
        gl.amd.cdna4.buffer_store(
            squares, partial_squares, token * 8 + slots, cache=".wt"
        )
    WAIT_STORES(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _iris_rendezvous(flags, heaps, program, epoch, RANK, NUM_WARPS, False)

    # Phase 2: complete scales, then this rank's columns of each weighted sum.
    for token in range(program, rows, programs):
        total = gl.zeros([8], gl.float32, layout=gl.SliceLayout(1, tile_layout))
        for peer in gl.static_range(WORLD_SIZE):
            total += gl.amd.cdna4.buffer_load(
                square_peers[peer], token * 8 + slots, cache=".cg"
            )
        inverse = gl.rsqrt(total / WIDTH + EPS)
        weights = gl.load(route_weights + token * 8 + slots)
        combined = gl.zeros([COLUMNS], gl.float32, layout=row_layout)
        for slot in gl.static_range(8):
            values = gl.load(routes + (token * 8 + slot) * COLUMNS + columns)
            values = values.to(gl.float32) * gain
            slot_inverse = gl.sum(gl.where(slots == slot, inverse, 0.0), 0)
            slot_weight = gl.sum(gl.where(slots == slot, weights, 0.0), 0)
            combined = gl.fma(values * slot_inverse, slot_weight, combined)
        gl.amd.cdna4.buffer_store(
            combined.to(gl.bfloat16), slices, token * COLUMNS + columns, cache=".wt"
        )
    WAIT_STORES(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _iris_rendezvous(
        flags + FLAG_ROWS * WORLD_SIZE, heaps, program, epoch, RANK, NUM_WARPS, False
    )

    # Phase 3: pull this owner's rows from every rank.
    local_rows = gl.cdiv(rows, WORLD_SIZE)
    first = RANK * local_rows
    start = first + (program - first % programs + programs) % programs
    for token in range(start, first + local_rows, programs):
        destination = (token - first) * WIDTH + columns
        values = ()
        for step in gl.static_range(WORLD_SIZE):
            values += (
                gl.amd.cdna4.buffer_load(
                    slice_peers[(RANK + step) % WORLD_SIZE],
                    token * COLUMNS + columns,
                    (token < rows) & (columns < COLUMNS),
                    0.0,
                    cache=".cg",
                ),
            )
        for step in gl.static_range(WORLD_SIZE):
            gl.amd.cdna4.buffer_store(
                values[step],
                output,
                destination + ((RANK + step) % WORLD_SIZE) * COLUMNS,
            )
    # Peers rewrite both exchange buffers on their next call.
    WAIT_VMEM(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _iris_rendezvous(
        flags + 2 * FLAG_ROWS * WORLD_SIZE,
        heaps,
        program,
        epoch,
        RANK,
        NUM_WARPS,
        False,
    )
