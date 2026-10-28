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


"""Pack and gather Muse's 4096-feature, eight-route expert inputs."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_peer_pointers,
    _iris_rendezvous,
)


@gluon.jit
def _mxfp4_scale_offset(row, group):
    """Address CDNA4-swizzled scales for 4096-feature MXFP4 token rows."""
    swizzled = (
        ((group // 8 * 4 + group % 4) * 16 + row % 16) * 2 + group % 8 // 4
    ) * 2 + row % 32 // 16
    return row // 32 * 4096 + swizzled


@gluon.jit(do_not_specialize=["rows", "valid_rows"])
def iris_pack_moe_inputs(
    data,
    scales,
    ids,
    weights,
    packet,
    rows,
    valid_rows,
    QUANTIZED: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Pack 4096-feature token rows and eight expert IDs/weights per row.

    data, scales, ids and weights are byte views of the owned input tensors.
    QUANTIZED selects MXFP4 with CDNA4 scales; otherwise activations are BF16.
    packet holds rows entries, zeroing entries beyond valid_rows. Write-through
    stores let the following all-gather read packet without a staging copy.
    """
    DATA_BYTES: gl.constexpr = 2048 if QUANTIZED else 8192
    SCALE_BYTES: gl.constexpr = 128 if QUANTIZED else 0
    PACKET_BYTES: gl.constexpr = DATA_BYTES + SCALE_BYTES + 64
    offsets = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([16], [64], [NUM_WARPS], [0])
    )
    row = offsets // PACKET_BYTES
    field = offsets % PACKET_BYTES
    valid = row < valid_rows
    values = gl.amd.cdna4.buffer_load(
        data, row * DATA_BYTES + field, valid & (field < DATA_BYTES), 0
    )
    if QUANTIZED:
        group = field - DATA_BYTES
        scale = gl.amd.cdna4.buffer_load(
            scales,
            _mxfp4_scale_offset(row, group),
            valid & (group >= 0) & (group < 128),
            0,
        )
        values = gl.where(
            (field >= DATA_BYTES) & (field < DATA_BYTES + 128), scale, values
        )
    route_byte = field - DATA_BYTES - SCALE_BYTES
    id_byte = gl.amd.cdna4.buffer_load(
        ids, row * 32 + route_byte, valid & (route_byte >= 0) & (route_byte < 32), 0
    )
    weight_byte = gl.amd.cdna4.buffer_load(
        weights,
        row * 32 + route_byte - 32,
        valid & (route_byte >= 32) & (route_byte < 64),
        0,
    )
    values = gl.where(
        route_byte >= 0, gl.where(route_byte < 32, id_byte, weight_byte), values
    )
    gl.amd.cdna4.buffer_store(
        values, packet, offsets, offsets < rows * PACKET_BYTES, cache=".wt"
    )


@gluon.jit(do_not_specialize=["rows"])
def iris_pull_allgather_moe_inputs(
    packet,
    data,
    scales,
    ids,
    weights,
    ready_flags,
    heaps,
    rows,
    RANK: gl.constexpr,
    QUANTIZED: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_VMEM: gl.constexpr,
):
    """All-gather packed MoE rows and restore their original token order.

    Each rank supplies ceil(rows / WORLD_SIZE) packet rows. Unpack the 4096-feature
    activations and eight route IDs/weights into data, scales, ids and weights,
    ignoring the last owner's padding. Restore CDNA4 scales without requantization.
    ready_flags coordinate publication and finish peer reads before packet reuse.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    DATA_BYTES: gl.constexpr = 2048 if QUANTIZED else 8192
    SCALE_BYTES: gl.constexpr = 128 if QUANTIZED else 0
    PACKET_BYTES: gl.constexpr = DATA_BYTES + SCALE_BYTES + 64
    program = gl.program_id(0)
    peers = _iris_peer_pointers(packet, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    WAIT_VMEM(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, False)
    local_rows = gl.cdiv(rows, WORLD_SIZE)
    local_bytes = local_rows * PACKET_BYTES
    lanes = gl.arange(0, BLOCK, layout=gl.BlockedLayout([16], [64], [NUM_WARPS], [0]))
    for tile in range(program, gl.cdiv(local_bytes, BLOCK), gl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        local_row = offsets // PACKET_BYTES
        field = offsets % PACKET_BYTES
        for step in gl.static_range(WORLD_SIZE):
            row = ((RANK + step) % WORLD_SIZE) * local_rows + local_row
            valid = (offsets < local_bytes) & (row < rows)
            values = gl.amd.cdna4.buffer_load(
                peers[(RANK + step) % WORLD_SIZE], offsets, valid, 0, cache=".cg"
            )
            gl.amd.cdna4.buffer_store(
                values, data, row * DATA_BYTES + field, valid & (field < DATA_BYTES)
            )
            if QUANTIZED:
                group = field - DATA_BYTES
                gl.amd.cdna4.buffer_store(
                    values,
                    scales,
                    _mxfp4_scale_offset(row, group),
                    valid & (group >= 0) & (group < 128),
                )
            route_byte = field - DATA_BYTES - SCALE_BYTES
            gl.amd.cdna4.buffer_store(
                values,
                ids,
                row * 32 + route_byte,
                valid & (route_byte >= 0) & (route_byte < 32),
            )
            gl.amd.cdna4.buffer_store(
                values,
                weights,
                row * 32 + route_byte - 32,
                valid & (route_byte >= 32) & (route_byte < 64),
            )
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
