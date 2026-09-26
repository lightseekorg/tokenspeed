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

"""CDNA4 attention reduction and gathering through the prepared MoE workspace."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.prefill import (
    _peer_buffers,
    _peer_flags,
    _prefill_entry_barrier,
    _prefill_store_completion,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import _attn_res_mix_gfx950


def _reduce_metadata(grid, kernel, args):
    elements = args["PARTITION_ELEMENTS"]
    return {
        "name": kernel.name,
        "bytes": elements * (9 + int(args["HAS_RESIDUAL"])) * 2,
        "flops32": elements * (7 + int(args["HAS_RESIDUAL"])),
    }


def _gather_metadata(grid, kernel, args):
    return {"name": kernel.name, "bytes": args["PARTITION_ELEMENTS"] * 9 * 2}


def _mix_gather_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "bytes": args["LOCAL_ROWS"]
        * 7168
        * (args["NUM_VALID_BLOCKS"] + 10 + 2 * int(args["NUM_VALID_BLOCKS"] > 0))
        * 2,
    }


@gluon.jit(launch_metadata=_reduce_metadata)
def iris_attention_reduce_scatter_gluon_kernel(
    input_ptr,
    residual_ptr,
    prefix_ptr,
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
    HAS_RESIDUAL: gl.constexpr,
):
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
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    _prefill_entry_barrier(flags, peers, pid, epoch, RANK, NUM_WARPS)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        source = RANK * PARTITION_ELEMENTS + offsets
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[(RANK + step) % 8], source, mask, 0, cache=".cg"
                ),
            )
        # Preserve Iris's even/odd FP32 tree and the BF16 boundary before add.
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
        prefix = (even + odd).to(gl.bfloat16)
        if HAS_RESIDUAL:
            residual = gl.amd.cdna4.buffer_load(
                residual_ptr, source, mask, 0, cache=".ca"
            )
            prefix = (prefix.to(gl.float32) + residual.to(gl.float32)).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(prefix, prefix_ptr, offsets, mask, cache=".wb")
    # Only local storage was written. The gather's completion orders every
    # rank's input reads before the next producer reuses the symmetric input.


@gluon.jit(launch_metadata=_gather_metadata)
def iris_attention_push_gather_gluon_kernel(
    mixed_ptr,
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
):
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
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        value = gl.amd.cdna4.buffer_load(mixed_ptr, offsets, mask, 0, cache=".cg")
        # Scalar peer bases let all eight stores issue without intervening drains.
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                value,
                outputs[(RANK + step) % 8],
                RANK * PARTITION_ELEMENTS + offsets,
                mask,
                cache=".wt",
            )
    _prefill_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)


@gluon.jit(launch_metadata=_mix_gather_metadata)
def iris_attention_mix_push_gluon_kernel(
    prefix_ptr,
    output_ptr,
    block_residual,
    res_weight,
    rms_weight,
    out_norm_weight,
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
    LOCAL_ROWS: gl.constexpr,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
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
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, 8192, layout=layout)
    mask = hidden < 7168
    for row in range(pid, LOCAL_ROWS, NUM_PROGRAMS):
        token = RANK * LOCAL_ROWS + row
        prefix = gl.amd.cdna4.buffer_load(
            prefix_ptr, row * 7168 + hidden, mask, 0, cache=".ca"
        )
        mixed = _attn_res_mix_gfx950(
            prefix,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            token,
            hidden,
            mask,
            STRIDE_BLOCK_T,
            STRIDE_BLOCK_N,
            7168,
            NUM_VALID_BLOCKS + 1,
            SCORE_EPS,
            OUTPUT_EPS,
        )
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                mixed,
                outputs[(RANK + step) % 8],
                token * 7168 + hidden,
                mask,
                cache=".wt",
            )
    _prefill_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)
