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

"""Iris all-reduce with sharded K3 attn/MoE compute and shared result storage."""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris import allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    ProducerPull,
    iris_arch_policy,
)
from tokenspeed_kernel.ops.tuning import is_warming_up
from tokenspeed_kernel.platform import current_platform

REDUCE_PROGRAMS = 24
GATHER_PROGRAMS = 128
# Row-sharded MoE inputs: packed E2M1 latent bytes, CDNA4 scale bytes, and top-k.
MOE_INPUT_LATENT = 3584
MOE_INPUT_DATA_BYTES = MOE_INPUT_LATENT // 2
MOE_INPUT_SCALE_BYTES = MOE_INPUT_LATENT // 32
MOE_INPUT_TOPK = 16
# Each rank's rows fill whole 32-row CDNA4 scale blocks.
MOE_INPUT_ROW_ALIGN = 8 * 32
MOE_INPUT_GATHER_PROGRAMS = 64
# A push gather that runs beside a GEMM leaves it most of the CUs.
SIDE_GATHER_PROGRAMS = 32
# Leaves the 256-CU GEMM 224 CUs: one 256-row tile per owner per round (8 x 28 tiles).
PUBLISHED_SCATTER_PROGRAMS = 32
# One program per owned row of an 8192-row chunk: serial mixes need rows in flight.
_MIX_ROWS_PROGRAMS = 1024
# K3 MLA q_a + kv_a (+ RoPE key) rows, which the MoE tail can project ahead.
MLA_QKV_A_WIDTH = 2112
# K3.1 DSA prefill selection rows (index_topk), gathered as unsigned 16-bit slots.
PREFILL_TOPK = 2048
PREFILL_TOPK_GATHER_PROGRAMS = 64
# Row tile of the gfx950 large-M GEMM (LARGEM_BLOCK_M).
_LARGEM_ROW_TILE = 256
# A published projection's scatter overlaps the GEMM from the second owned row tile.
PUBLISHED_MIN_ROWS = 2 * 8 * _LARGEM_ROW_TILE
# Owned row tiles of an 8192-row chunk (LARGEM_GATHER_MAX_TILES).
_GATHER_ROW_TILES = 4

_attn_res_mix_gfx950 = None


def _gather_rows_gemm_enabled() -> bool:
    """Whether the KDA input projection gathers its rows inside the GEMM
    (experiment, default off; listed with the K3.1 switches in the runtime's
    ``utils/env.py``)."""
    return os.environ.get("TOKENSPEED_K31_GATHER_ROWS_GEMM", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@gluon.jit
def _row_partition_store_completion(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Complete local stores, update ready flags, and wait for peer completion."""
    WAIT_STORES(args=[], constraints="=r,~{memory}")
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
def _row_partition_entry_barrier(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Publish the entry epoch and acquire each peer's producer completion."""
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
    """Map a symmetric BF16 buffer to all eight peer heaps."""
    offset = pointer.to(gl.uint64) - heaps[RANK]
    result = ()
    for peer in gl.static_range(8):
        result += (
            gl.multiple_of((heaps[peer] + offset).to(gl.pointer_type(gl.bfloat16)), 16),
        )
    return result


@gluon.jit
def _peer_flags(pointer, heaps, RANK: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Map a symmetric flag buffer to all eight peer heaps."""
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    bases = gl.full((8,), 0, gl.uint64, layout)
    for peer in gl.static_range(8):
        bases = gl.where(peers == peer, heaps[peer], bases)
    offset = pointer.to(gl.uint64) - heaps[RANK]
    return (bases + offset).to(gl.pointer_type(gl.uint32))


@gluon.jit(do_not_specialize=["ROWS"])
def iris_k3moe_pull_scatter(
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
    ROWS,
    FIRST_WIDTH: gl.constexpr,
    SECOND_WIDTH: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Pull K3 routed/shared partials into each rank's consecutive row shard.

    For r = q * (ROWS / 8) + u, scratch_q[u] = BF16(sum_p FP32(partial_p[r])).
    A fixed FP32 tree reduces the routed and shared inputs, packed in that order.
    The matching push-gather completes the input-reuse barrier.
    """
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
    _row_partition_entry_barrier(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)
    FIRST_ELEMENTS = ROWS // 8 * FIRST_WIDTH
    SECOND_ELEMENTS = ROWS // 8 * SECOND_WIDTH
    PARTITION_ELEMENTS = FIRST_ELEMENTS + SECOND_ELEMENTS
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


@gluon.jit(do_not_specialize=["LOCAL_ROWS"])
def iris_k3moe_push_gather(
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Add K3 routed/shared outputs and residual, then push rows to every peer.

    For r = q * LOCAL_ROWS + u, use i = u for sharded prefixes, else i = r:
        output_p[r] = BF16((FP32(prefix_q[i]) + FP32(projected_q[u]))
                          + FP32(shared_q[u]))
    Wait for peer publication before reusing the producer buffers.
    """
    # Preserve row alignment for vector loads/stores without specializing M.
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
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
    _row_partition_store_completion(
        flags, peer_flags, block_id, epoch, RANK, NUM_WARPS, WAIT_STORES=WAIT_STORES
    )


@gluon.jit
def _peer_words(pointer, heaps, RANK: gl.constexpr):
    """Map a symmetric buffer, viewed as 32-bit words, to all eight peer heaps."""
    offset = pointer.to(gl.uint64) - heaps[RANK]
    result = ()
    for peer in gl.static_range(8):
        result += (
            gl.multiple_of((heaps[peer] + offset).to(gl.pointer_type(gl.int32)), 16),
        )
    return result


@gluon.jit
def _push_owned_words(
    local_ptr,
    heaps,
    words,
    pid,
    lanes,
    RANK: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    """Store this rank's ``words``-long slice of a symmetric buffer to its peers.

    ``words`` is a multiple of 4, so every slice starts 16-byte aligned.
    """
    peers = _peer_words(local_ptr, heaps, RANK)
    # Clearing two known-zero bits proves 16-byte alignment, so accesses vectorize.
    words = words // 4 * 4
    first = RANK * words
    for tile in range(pid, gl.cdiv(words, BLOCK), NUM_PROGRAMS):
        offsets = tile * BLOCK + lanes
        mask = offsets < words
        offsets += first
        value = gl.amd.cdna4.buffer_load(peers[RANK], offsets, mask, 0, cache=".cg")
        for step in gl.static_range(1, 8):
            gl.amd.cdna4.buffer_store(
                value, peers[(RANK + step) % 8], offsets, mask, cache=".wt"
            )


@gluon.jit(do_not_specialize=["DATA_WORDS", "SCALE_WORDS", "ROUTE_WORDS"])
def iris_k3moe_input_push_gather(
    data_ptr,
    scale_ptr,
    ids_ptr,
    weights_ptr,
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
    DATA_WORDS,
    SCALE_WORDS,
    ROUTE_WORDS,
    BLOCK: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Gather row-sharded MoE inputs: rank q owns the q-th slice of each buffer.

    The slices are the packed E2M1 latent rows, their CDNA4 scale row blocks,
    and the top-k ids and weights, each ``*_WORDS`` 32-bit words long (a
    multiple of 4). Peers
    write disjoint slices; wait for all peer stores before the MoE reads them.
    """
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
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    _push_owned_words(
        data_ptr, heaps, DATA_WORDS, pid, lanes, RANK, BLOCK, NUM_PROGRAMS
    )
    _push_owned_words(
        scale_ptr, heaps, SCALE_WORDS, pid, lanes, RANK, BLOCK, NUM_PROGRAMS
    )
    _push_owned_words(
        ids_ptr, heaps, ROUTE_WORDS, pid, lanes, RANK, BLOCK, NUM_PROGRAMS
    )
    _push_owned_words(
        weights_ptr, heaps, ROUTE_WORDS, pid, lanes, RANK, BLOCK, NUM_PROGRAMS
    )
    _row_partition_store_completion(
        flags, peers, pid, epoch, RANK, NUM_WARPS, WAIT_STORES=WAIT_STORES
    )


def _reduce_metadata(grid, kernel, args):
    elements = args["LOCAL_ROWS"] * 7168
    return {
        "name": kernel.name,
        "bytes": elements * (9 + int(args["HAS_RESIDUAL"])) * 2,
        "flops32": elements * (7 + int(args["HAS_RESIDUAL"])),
    }


def _gather_metadata(grid, kernel, args):
    return {"name": kernel.name, "bytes": args["LOCAL_ROWS"] * 7168 * 9 * 2}


def _mix_gather_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "bytes": args["LOCAL_ROWS"]
        * 7168
        * (args["NUM_VALID_BLOCKS"] + 10 + 2 * int(args["NUM_VALID_BLOCKS"] > 0))
        * 2,
    }


@gluon.jit(launch_metadata=_reduce_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3attn_pull_scatter(
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    RESIDUAL_IS_SHARDED: gl.constexpr,
):
    """Pull K3 attn rows into each rank's residual shard.

    For r = q * LOCAL_ROWS + u, with i = u for a sharded residual, else i = r:
        reduced_q[u] = BF16(sum_p FP32(partial_p[r]))
        prefix_q[u] = BF16(FP32(reduced_q[u]) + FP32(residual_q[i]))
    Without a residual, store reduced directly. Matching push-gather orders reuse.
    """
    # Whole rows preserve vector alignment without specializing the row count.
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
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
    _row_partition_entry_barrier(flags, peers, pid, epoch, RANK, NUM_WARPS)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        _attn_pull_reduce_block(
            inputs,
            residual_ptr,
            prefix_ptr,
            tile * BLOCK_ELEMENTS + lanes,
            PARTITION_ELEMENTS,
            RANK,
            HAS_RESIDUAL,
            RESIDUAL_IS_SHARDED,
        )


@gluon.jit(
    launch_metadata=_reduce_metadata,
    do_not_specialize=["LOCAL_ROWS", "TARGET_0", "TARGET_1", "TARGET_2", "TARGET_3"],
)
def iris_k3attn_pull_scatter_published(
    input_ptr,
    residual_ptr,
    prefix_ptr,
    counters,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    LOCAL_ROWS,
    TARGET_0,
    TARGET_1,
    TARGET_2,
    TARGET_3,
    TILE_ROWS: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    RESIDUAL_IS_SHARDED: gl.constexpr,
):
    """``iris_k3attn_pull_scatter`` on each owned row tile once peers publish it.

    The producers ran ``gluon_mm_a16w16_prefill_publish_rows_gfx950``: owned
    ``TILE_ROWS``-row tile ``c`` (at most 4) is complete on every peer once
    ``counters[c]`` reaches ``TARGET_c`` (int32s that wrap). No entry barrier:
    the publication counts replace it.
    """
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    TILE_ELEMENTS: gl.constexpr = TILE_ROWS * 7168
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
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for owned_tile in range(0, LOCAL_ROWS // TILE_ROWS):
        target = TARGET_0
        if owned_tile == 1:
            target = TARGET_1
        elif owned_tile == 2:
            target = TARGET_2
        elif owned_tile == 3:
            target = TARGET_3
        # Peers increment this rank's own counters, so the spin stays local.
        seen = gl.load(counters + owned_tile, cache_modifier=".cv", volatile=True)
        while seen - target < 0:
            seen = gl.load(counters + owned_tile, cache_modifier=".cv", volatile=True)
        gl.atomic_add(counters + owned_tile, 0, sem="acquire", scope="sys")
        for block in range(pid, TILE_ELEMENTS // BLOCK_ELEMENTS, NUM_PROGRAMS):
            _attn_pull_reduce_block(
                inputs,
                residual_ptr,
                prefix_ptr,
                owned_tile * TILE_ELEMENTS + block * BLOCK_ELEMENTS + lanes,
                PARTITION_ELEMENTS,
                RANK,
                HAS_RESIDUAL,
                RESIDUAL_IS_SHARDED,
            )


@gluon.jit
def _attn_pull_reduce_block(
    inputs,
    residual_ptr,
    prefix_ptr,
    offsets,
    PARTITION_ELEMENTS,
    RANK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    RESIDUAL_IS_SHARDED: gl.constexpr,
):
    """Reduce one block of owned attention elements into ``prefix_ptr``."""
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
        values[(0 - RANK) % 8].to(gl.float32) + values[(2 - RANK) % 8].to(gl.float32)
    ) + (values[(4 - RANK) % 8].to(gl.float32) + values[(6 - RANK) % 8].to(gl.float32))
    odd = (
        values[(1 - RANK) % 8].to(gl.float32) + values[(3 - RANK) % 8].to(gl.float32)
    ) + (values[(5 - RANK) % 8].to(gl.float32) + values[(7 - RANK) % 8].to(gl.float32))
    prefix = (even + odd).to(gl.bfloat16)
    if HAS_RESIDUAL:
        residual = gl.amd.cdna4.buffer_load(
            residual_ptr,
            offsets if RESIDUAL_IS_SHARDED else source,
            mask,
            0,
            cache=".ca",
        )
        prefix = (prefix.to(gl.float32) + residual.to(gl.float32)).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(prefix, prefix_ptr, offsets, mask, cache=".wb")


@gluon.jit(launch_metadata=_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3attn_push_gather(
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Push mixed K3 attn rows to every peer.

    Each owner q writes output_p[q * LOCAL_ROWS + u] = mixed_q[u] for every p.
    Wait for all peer stores before consuming the replicated output.
    """
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
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
    _row_partition_store_completion(
        flags, peers, pid, epoch, RANK, NUM_WARPS, WAIT_STORES=WAIT_STORES
    )


@gluon.jit(launch_metadata=_mix_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3attn_mix_push_gather(
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
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Mix K3 prefixes with AttnRes history, then push normalized rows.

    For r = q * LOCAL_ROWS + u, every peer p receives:
        output_p[r] = BF16(RMSNorm(BF16(AttnRes(prefix_q[u], history[r]))))
    Owners write disjoint rows and wait for all peer stores.
    """
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
    _row_partition_store_completion(
        flags, peers, pid, epoch, RANK, NUM_WARPS, WAIT_STORES=WAIT_STORES
    )


@gluon.jit(launch_metadata=_mix_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def k3attn_mix_rows(
    prefix_ptr,
    mixed_ptr,
    block_residual,
    res_weight,
    rms_weight,
    out_norm_weight,
    RANK: gl.constexpr,
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """``iris_k3attn_mix_push_gather``'s rows, stored locally: same bits.

    mixed[u] = BF16(RMSNorm(BF16(AttnRes(prefix[u], history[RANK * LOCAL_ROWS + u]))))
    """
    pid = gl.program_id(0)
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
        gl.amd.cdna4.buffer_store(mixed, mixed_ptr, row * 7168 + hidden, mask)


def _moe_mix_gather_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "bytes": args["LOCAL_ROWS"]
        * 7168
        * (args["NUM_VALID_BLOCKS"] + 12 + 2 * int(args["NUM_VALID_BLOCKS"] > 0))
        * 2,
    }


@gluon.jit(launch_metadata=_moe_mix_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3moe_mix_push_gather(
    projection_ptr,
    shared_ptr,
    prefix_ptr,
    prefix_out_ptr,
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
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Finish the MoE residual on owned rows, mix the next attention input, push it.

    For r = q * LOCAL_ROWS + u, with i = u for sharded prefixes, else i = r:
        prefix_out_q[u] = BF16((FP32(prefix_q[i]) + FP32(projected_q[u]))
                               + FP32(shared_q[u]))
        output_p[r] = BF16(RMSNorm(BF16(AttnRes(prefix_out_q[u], history[r]))))
    The residual stays on its owner. The mix runs in the row layout of the
    standalone AttnRes kernel, so its FP32 reductions keep their order.
    """
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
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [NUM_WARPS], [0])
    store_layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, 8192, layout=layout)
    mask = hidden < 7168
    store_hidden = gl.arange(0, 8192, layout=store_layout)
    store_mask = store_hidden < 7168
    for row in range(pid, LOCAL_ROWS, NUM_PROGRAMS):
        token = RANK * LOCAL_ROWS + row
        if PREFIX_IS_SHARDED:
            prefix_row = row
        else:
            prefix_row = token
        a = gl.amd.cdna4.buffer_load(
            prefix_ptr, prefix_row * 7168 + hidden, mask, 0, cache=".cg"
        ).to(gl.float32)
        b = gl.amd.cdna4.buffer_load(
            projection_ptr, row * 7168 + hidden, mask, 0, cache=".cg"
        ).to(gl.float32)
        c = gl.amd.cdna4.buffer_load(
            shared_ptr, row * 7168 + hidden, mask, 0, cache=".cg"
        ).to(gl.float32)
        prefix = (a + b + c).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(prefix, prefix_out_ptr, row * 7168 + hidden, mask)
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
        mixed = gl.convert_layout(mixed, store_layout)
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                mixed,
                outputs[(RANK + step) % 8],
                token * 7168 + store_hidden,
                store_mask,
                cache=".wt",
            )
    _row_partition_store_completion(
        flags, peers, pid, epoch, RANK, NUM_WARPS, WAIT_STORES=WAIT_STORES
    )


@gluon.jit(do_not_specialize=["LOCAL_ROWS"])
def k3moe_residual_mix_rows(
    projection_ptr,
    shared_ptr,
    prefix_ptr,
    prefix_out_ptr,
    mixed_ptr,
    block_residual,
    res_weight,
    rms_weight,
    out_norm_weight,
    RANK: gl.constexpr,
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """``iris_k3moe_mix_push_gather``'s residual and mix, stored on this rank only.

    ``mixed_ptr[u]`` receives row ``u`` of the mixed activation for a separate
    push gather; the arithmetic and its layouts are the fused kernel's.
    """
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [NUM_WARPS], [0])
    store_layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, 8192, layout=layout)
    mask = hidden < 7168
    store_hidden = gl.arange(0, 8192, layout=store_layout)
    store_mask = store_hidden < 7168
    for row in range(gl.program_id(0), LOCAL_ROWS, NUM_PROGRAMS):
        token = RANK * LOCAL_ROWS + row
        if PREFIX_IS_SHARDED:
            prefix_row = row
        else:
            prefix_row = token
        a = gl.amd.cdna4.buffer_load(
            prefix_ptr, prefix_row * 7168 + hidden, mask, 0, cache=".cg"
        ).to(gl.float32)
        b = gl.amd.cdna4.buffer_load(
            projection_ptr, row * 7168 + hidden, mask, 0, cache=".cg"
        ).to(gl.float32)
        c = gl.amd.cdna4.buffer_load(
            shared_ptr, row * 7168 + hidden, mask, 0, cache=".cg"
        ).to(gl.float32)
        prefix = (a + b + c).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(prefix, prefix_out_ptr, row * 7168 + hidden, mask)
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
        gl.amd.cdna4.buffer_store(
            gl.convert_layout(mixed, store_layout),
            mixed_ptr,
            row * 7168 + store_hidden,
            store_mask,
        )


def _partition_gather_metadata(grid, kernel, args):
    return {"name": kernel.name, "bytes": args["LOCAL_ROWS"] * args["WIDTH"] * 9 * 2}


@gluon.jit(launch_metadata=_partition_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3_push_gather_partition(
    source_ptr,
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
    LOCAL_ROWS,
    WIDTH: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Push this rank's ``[LOCAL_ROWS, WIDTH]`` rows to every peer's output.

    output_p[RANK * LOCAL_ROWS + u] = source[u] for every peer p; wait for all
    peers' stores before the replicated output is consumed. ``WIDTH`` must be
    a multiple of 8 so whole rows keep 16-byte vectors.
    """
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
    PARTITION_ELEMENTS = LOCAL_ROWS * WIDTH
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        value = gl.amd.cdna4.buffer_load(source_ptr, offsets, mask, 0, cache=".cg")
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                value,
                outputs[(RANK + step) % 8],
                RANK * PARTITION_ELEMENTS + offsets,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(
        flags, peers, pid, epoch, RANK, NUM_WARPS, WAIT_STORES=WAIT_STORES
    )


@gluon.jit
def _peer_buffers_like(pointer, heaps, RANK: gl.constexpr):
    """Map a symmetric buffer to all eight peer heaps, keeping its element type."""
    offset = pointer.to(gl.uint64) - heaps[RANK]
    result = ()
    for peer in gl.static_range(8):
        result += (gl.multiple_of((heaps[peer] + offset).to(pointer.dtype), 16),)
    return result


def _topk_gather_metadata(grid, kernel, args):
    rows = args["LOCAL_ROWS"]
    topk = args["TOPK"]
    return {"name": kernel.name, "bytes": rows * (topk * 6 + 8 + 16 * (topk * 2 + 4))}


@gluon.jit(launch_metadata=_topk_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3_pull_gather_topk_rows(
    selection_ptr,
    lens_ptr,
    publish_ptr,
    lens_publish_ptr,
    output_ptr,
    lens_output_ptr,
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
    LOCAL_ROWS,
    TOPK: gl.constexpr,
    MAX_LOCAL_ROWS: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Gather every rank's ``[LOCAL_ROWS, TOPK]`` int32 selection rows and lengths.

    Publishes int16(selection) (low 16 bits) and lens into this rank's
    symmetric buffers, then for every peer p and row u:
        output[p * LOCAL_ROWS + u] = publish_p[u]
        lens_output[p * LOCAL_ROWS + u] = lens_publish_p[u]
    Program b publishes and pulls the same tiles on every rank, so its entry
    barrier with the peers' program b orders each tile's store before its loads.
    Calls alternate between two publish slots: a rank publishing call n + 1
    passed call n's barrier, so no peer still reads call n - 1's slot.
    """
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
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    slot = (epoch & 1).to(gl.int64)
    publish_ptr += slot * MAX_LOCAL_ROWS * TOPK
    lens_publish_ptr += slot * MAX_LOCAL_ROWS
    PARTITION_ELEMENTS = LOCAL_ROWS * TOPK
    TILES = gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, TILES, NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        value = gl.amd.cdna4.buffer_load(
            selection_ptr, offsets, mask, 0, cache=".cg"
        ).to(gl.int16)
        gl.amd.cdna4.buffer_store(value, publish_ptr, offsets, mask, cache=".wt")
    if pid == NUM_PROGRAMS - 1:
        for start in range(0, LOCAL_ROWS, BLOCK_ELEMENTS):
            offsets = start + lanes
            mask = offsets < LOCAL_ROWS
            value = gl.amd.cdna4.buffer_load(lens_ptr, offsets, mask, 0, cache=".cg")
            gl.amd.cdna4.buffer_store(
                value, lens_publish_ptr, offsets, mask, cache=".wt"
            )
    WAIT_STORES(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _row_partition_entry_barrier(flags, peer_flags, pid, epoch, RANK, NUM_WARPS)
    publishes = _peer_buffers_like(publish_ptr, heaps, RANK)
    for tile in range(pid, TILES, NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    publishes[(RANK + step) % 8], offsets, mask, 0, cache=".cg"
                ),
            )
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                values[step],
                output_ptr,
                (RANK + step) % 8 * PARTITION_ELEMENTS + offsets,
                mask,
            )
    if pid == NUM_PROGRAMS - 1:
        lens_publishes = _peer_buffers_like(lens_publish_ptr, heaps, RANK)
        for start in range(0, LOCAL_ROWS, BLOCK_ELEMENTS):
            offsets = start + lanes
            mask = offsets < LOCAL_ROWS
            for step in gl.static_range(8):
                value = gl.amd.cdna4.buffer_load(
                    lens_publishes[(RANK + step) % 8], offsets, mask, 0, cache=".cg"
                )
                gl.amd.cdna4.buffer_store(
                    value,
                    lens_output_ptr,
                    (RANK + step) % 8 * LOCAL_ROWS + offsets,
                    mask,
                )


def _overlaps(tensor: torch.Tensor, buffer: torch.Tensor) -> bool:
    span = tensor.numel()
    if span == 0:
        return False
    # Collective buffers are contiguous; history can have padding between rows.
    if not tensor.is_contiguous():
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    begin = tensor.data_ptr()
    end = begin + span * tensor.element_size()
    return (
        begin < buffer.data_ptr() + buffer.numel() * buffer.element_size()
        and buffer.data_ptr() < end
    )


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    # High priority gets its own hardware queue, so side kernels run beside the forward.
    return torch.cuda.Stream(device=device, priority=-1)


def _published_tiles(max_rows: int) -> int:
    # The scatter takes one target per owned tile; 8192 rows own 4 tiles each.
    return max(1, min(4, max_rows // (8 * _LARGEM_ROW_TILE)))


@dataclass(frozen=True)
class _PublishedProjection:
    """An attention partial whose row tiles a GEMM publishes as it finishes them."""

    out: torch.Tensor
    # Recorded on the calling stream just before the GEMM launch.
    start: torch.cuda.Event
    # Per owned row tile: every peer's published column tiles so far (int32 wrap).
    targets: tuple[int, int, int, int]


@dataclass(frozen=True)
class _PendingGather:
    """An attention activation whose rows a side stream is still gathering."""

    # This rank's rows of ``output``, final on the calling stream.
    own_rows: torch.Tensor
    output: torch.Tensor
    # Recorded on the side stream after the gather.
    done: torch.cuda.Event


@dataclass(frozen=True)
class _ReducingMoeShared:
    """A MoE shared partial whose owned rows a side stream reduces into the scratch."""

    out: torch.Tensor
    # Recorded on the side stream after the reduction.
    done: torch.cuda.Event


class IrisAllReduceShardedState:
    """Own the shared result and borrow the producer input, scratch, and entry flags.

    Attention and MoE use these buffers in order on one stream. Their matching
    gathers finish all peer reads before the next producer can reuse the input.
    """

    @staticmethod
    def validate_capacity(
        max_rows: int, world_size: int, dtype: torch.dtype, producer_max_numel: int
    ) -> None:
        if not (
            current_platform().is_cdna4
            and world_size == 8
            and dtype == torch.bfloat16
            and 0 < max_rows <= 8192
            and max_rows % 8 == 0
            and max_rows * (7168 + 3584) <= producer_max_numel
        ):
            raise ValueError(
                "Iris row-sharded fusion requires TP8 BF16 producer capacity on CDNA4"
            )

    @staticmethod
    def heap_bytes(max_rows: int, world_size: int, dtype: torch.dtype) -> int:
        return (
            allocation_bytes((max_rows, 7168), dtype)
            + allocation_bytes((max_rows, MLA_QKV_A_WIDTH), dtype)
            + 2 * allocation_bytes((GATHER_PROGRAMS, world_size), torch.int32)
            + allocation_bytes((max_rows, MOE_INPUT_DATA_BYTES), torch.uint8)
            + allocation_bytes((max_rows * MOE_INPUT_SCALE_BYTES,), torch.uint8)
            + 2 * allocation_bytes((max_rows, MOE_INPUT_TOPK), torch.int32)
            + allocation_bytes((_published_tiles(max_rows),), torch.int32)
            + allocation_bytes((2, max_rows // world_size, PREFILL_TOPK), torch.int16)
            + allocation_bytes((2, max_rows // world_size), torch.int32)
            + allocation_bytes((PREFILL_TOPK_GATHER_PROGRAMS, world_size), torch.int32)
            + allocation_bytes((world_size * _GATHER_ROW_TILES,), torch.int32)
        )

    def __init__(self, pull: ProducerPull, *, max_rows: int) -> None:
        self._arch = iris_arch_policy()
        inputs = pull.producer.data
        assert inputs.numel() >= max_rows * (7168 + 3584)
        assert pull.scratch is not None
        assert pull.scratch.numel() >= max_rows // 8 * (7168 + 3584)
        assert pull.ready_flags.shape[0] >= REDUCE_PROGRAMS
        peers = pull.producer.peers
        self.pull = pull
        self.rank = peers.rank
        self.heap_bases = peers.addresses
        self.output = peers.ctx.empty((max_rows, 7168), dtype=inputs.dtype)
        self.gather_flags = peers.ctx.zeros((GATHER_PROGRAMS, 8), dtype=torch.int32)
        self.moe_input_data = peers.ctx.zeros(
            (max_rows, MOE_INPUT_DATA_BYTES), dtype=torch.uint8
        )
        self.moe_input_scale = peers.ctx.zeros(
            (max_rows * MOE_INPUT_SCALE_BYTES,), dtype=torch.uint8
        )
        self.moe_input_ids = peers.ctx.zeros(
            (max_rows, MOE_INPUT_TOPK), dtype=torch.int32
        )
        self.moe_input_weights = peers.ctx.zeros(
            (max_rows, MOE_INPUT_TOPK), dtype=torch.float32
        )
        self.qkv_output = peers.ctx.empty(
            (max_rows, MLA_QKV_A_WIDTH), dtype=inputs.dtype
        )
        self.qkv_gather_flags = peers.ctx.zeros((GATHER_PROGRAMS, 8), dtype=torch.int32)
        # Double-buffered selection rows that peers pull into their own memory.
        self.topk_publish = peers.ctx.empty(
            (2, max_rows // 8, PREFILL_TOPK), dtype=torch.int16
        )
        self.topk_lens_publish = peers.ctx.empty((2, max_rows // 8), dtype=torch.int32)
        self.topk_gather_flags = peers.ctx.zeros(
            (PREFILL_TOPK_GATHER_PROGRAMS, 8), dtype=torch.int32
        )
        self.gather_stream: torch.cuda.Stream | None = None
        # Peers add their published column tiles of each owned row tile here.
        self.published_counters = peers.ctx.zeros(
            (_published_tiles(max_rows),), dtype=torch.int32
        )
        # Column tiles every owned row tile has been published, as int32s.
        self._published_counts = [0] * 4
        self._published: _PublishedProjection | None = None
        self.scatter_stream: torch.cuda.Stream | None = None
        # Peers add their pushed workgroups of each of their row tiles here.
        self.gather_row_counters = peers.ctx.zeros(
            (8 * _GATHER_ROW_TILES,), dtype=torch.int32
        )
        # Pushed workgroups of every peer's row tile so far, as int32s.
        self._gather_row_counts = [0] * _GATHER_ROW_TILES
        # Output tiles handed out so far, as an int32.
        self.gather_row_tickets = torch.zeros(
            (1,), dtype=torch.int32, device=peers.device
        )
        self._gather_ticket_base = 0

    def project_attention(
        self, activation: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> bool:
        """Write the attention ``o_proj`` partial, publishing finished row tiles.

        Returns False, launching nothing, unless ``attn_mix`` can then reduce
        each owned row tile while the GEMM computes the next ones. The caller
        has verified that ``out`` is the borrowed producer input.
        """
        self._published = None
        if out.data_ptr() != self.pull.producer.data.data_ptr():
            return False
        published = self._publish_projection(activation, weight, out)
        if published is None:
            return False
        self._published = published
        return True

    def project_moe_shared(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        routed_out: torch.Tensor,
        shared_out: torch.Tensor,
    ) -> _ReducingMoeShared | None:
        """Write the MoE shared-expert partial, then reduce it beside the experts.

        ``routed_out`` and ``shared_out`` are the borrowed producer inputs of
        the MoE tail. The GEMM publishes its row tiles and a side stream reduces
        this rank's shared rows into the scratch as peers publish them, while
        the caller runs the routed experts. Pass the record to ``moe_tail`` or
        ``moe_tail_mix``, whose scatter then pulls only the routed rows, or to
        ``join_moe_shared`` before any other consumer of the producers. Returns
        None, launching nothing, unless the rows tile by ``8 * 256``.
        """
        rows = shared_out.shape[0] if shared_out.ndim == 2 else 0
        producer = self.pull.producer.data
        if (
            routed_out.ndim != 2
            or routed_out.shape != (rows, 3584)
            or routed_out.data_ptr() != producer.data_ptr()
            or shared_out.data_ptr()
            != producer.data_ptr() + rows * 3584 * producer.element_size()
            or self.pull.scratch.numel() < rows // 8 * (3584 + 7168)
        ):
            return None
        published = self._publish_projection(activation, weight, shared_out)
        if published is None:
            return None
        local_rows = rows // 8
        shared = self.pull.scratch[local_rows * 3584 : local_rows * (3584 + 7168)]
        done = torch.cuda.Event()
        self._launch_published_scatter(
            published, shared_out, None, shared.view(local_rows, 7168), False
        )
        done.record(self.scatter_stream)
        return _ReducingMoeShared(shared_out, done)

    def join_moe_shared(
        self, shared_reduced: _ReducingMoeShared | None, shared_partial: torch.Tensor
    ) -> None:
        """Order the calling stream after ``project_moe_shared``'s reduction."""
        if shared_reduced is None:
            return
        if shared_reduced.out.data_ptr() != shared_partial.data_ptr() or (
            shared_reduced.out.shape != shared_partial.shape
        ):
            raise RuntimeError("the reduced shared rows are not this partial's")
        torch.cuda.current_stream(shared_partial.device).wait_event(shared_reduced.done)

    def _publish_projection(
        self, activation: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> _PublishedProjection | None:
        """Launch the publishing GEMM into producer storage ``out``, or nothing."""
        rows = out.shape[0] if out.ndim == 2 else 0
        if (
            not current_platform().is_cdna4
            or torch.cuda.is_current_stream_capturing()
            or rows < PUBLISHED_MIN_ROWS
            or rows % (8 * _LARGEM_ROW_TILE)
            or rows > self.output.shape[0]
            or out.shape[1] != 7168
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or activation.ndim != 2
            or activation.shape[0] != rows
            or activation.dtype != out.dtype
            or weight.shape != (7168, activation.shape[1])
            or weight.dtype != out.dtype
            or activation.device != out.device
            or weight.device != out.device
        ):
            return None
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            launch_gluon_mm_a16w16_prefill_publish_rows_gfx950,
            supports_gluon_mm_a16w16_prefill_gfx950,
        )

        # Only where the registry GEMM is the large-M kernel, whose bits these match.
        if not supports_gluon_mm_a16w16_prefill_gfx950(rows, 7168, weight.shape[1]):
            return None
        start = torch.cuda.Event()
        start.record()
        if (
            launch_gluon_mm_a16w16_prefill_publish_rows_gfx950(
                activation,
                weight,
                out,
                self.published_counters,
                self.heap_bases,
                rank=self.rank,
            )
            is None
        ):
            return None
        per_call = 8 * (7168 // _LARGEM_ROW_TILE)
        for tile in range(rows // (8 * _LARGEM_ROW_TILE)):
            count = self._published_counts[tile] + per_call
            self._published_counts[tile] = (count + (1 << 31)) % (1 << 32) - (1 << 31)
        return _PublishedProjection(out, start, tuple(self._published_counts))

    def take_published(self, partial: torch.Tensor) -> _PublishedProjection | None:
        """Hand over the last ``project_attention`` if it published ``partial``."""
        published, self._published = self._published, None
        if (
            published is None
            or published.out.data_ptr() != partial.data_ptr()
            or published.out.shape != partial.shape
        ):
            return None
        return published

    def can_gather_moe_inputs(self, rows: int) -> bool:
        """Whether ``rows`` split into whole CDNA4 scale blocks per rank and fit."""
        return (
            current_platform().is_cdna4
            and 0 < rows <= self.output.shape[0]
            and rows % MOE_INPUT_ROW_ALIGN == 0
        )

    def _moe_input_scale_view(self, first_row: int, rows: int) -> torch.Tensor:
        # CDNA4-swizzled [K-scale bytes x 32, row blocks] with row blocks outermost.
        block_bytes = MOE_INPUT_SCALE_BYTES * 32
        return self.moe_input_scale[first_row * MOE_INPUT_SCALE_BYTES :].as_strided(
            (block_bytes, rows // 32), (1, block_bytes)
        )

    def moe_input_views(
        self, rows: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return this rank's and the full ``(data, scale, ids, weights)`` views.

        The caller fills the local views of its consecutive row shard, then
        ``gather_moe_inputs`` completes the full views on every rank. They stay
        valid until the next ``gather_moe_inputs``.
        """
        assert self.can_gather_moe_inputs(rows)
        local_rows = rows // 8
        first = self.rank * local_rows
        last = first + local_rows
        local = (
            self.moe_input_data[first:last],
            self._moe_input_scale_view(first, local_rows),
            self.moe_input_ids[first:last],
            self.moe_input_weights[first:last],
        )
        full = (
            self.moe_input_data[:rows],
            self._moe_input_scale_view(0, rows),
            self.moe_input_ids[:rows],
            self.moe_input_weights[:rows],
        )
        return local, full

    def moe_row_sharded_inputs(
        self,
        hidden_states: torch.Tensor,
        packed_weight: torch.Tensor,
        *,
        beta: float,
        linear_beta: float | None,
        route: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        attention_gather: _PendingGather | None,
    ) -> (
        tuple[
            tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
        ]
        | None
    ):
        """Project replicated MoE rows once per row instead of once per rank.

        Every rank holds the same ``[rows, 7168]`` input. Each projects the
        router and routed-latent rows of the packed ``[router; latent; shared
        gate; shared up]`` weight for its consecutive eighth of the rows,
        routes them with ``route(rows, logits) -> (weights, ids)`` and quantizes
        the latent to E2M1, then gathers those compact results. The rank's
        shared gate/up columns still project every row. Projections, routing
        and quantization are row-local, so the values match the replicated
        computation's.

        Returns ``((packed, scales), topk_weights, topk_ids, shared_input)``
        for all rows, valid until the next gather, or ``None`` before launching
        anything when the shape is unsupported. With ``attention_gather`` (the
        ``attn_mix`` that is still gathering ``hidden_states``) the owned rows
        project while the other rows arrive; it is joined before any return.
        """
        rows = hidden_states.shape[0]
        if attention_gather is not None and (
            attention_gather.output.data_ptr() != hidden_states.data_ptr()
            or attention_gather.output.shape != hidden_states.shape
        ):
            raise RuntimeError("the pending gather is not this activation's")
        if (
            not self.can_gather_moe_inputs(rows)
            or hidden_states.shape[1] != 7168
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or packed_weight.shape != (896 + MOE_INPUT_LATENT + 2 * 768, 7168)
            or packed_weight.dtype != torch.bfloat16
            or not packed_weight.is_contiguous()
        ):
            if attention_gather is not None:
                self.join_gather(attention_gather)
            return None
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_largem import (
            launch_gluon_latent_input_largem_routed_gfx950,
            launch_gluon_latent_input_largem_shared_gfx950,
        )
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
            _quantize_mxfp4_activation,
        )

        local, full = self.moe_input_views(rows)
        local_rows = rows // 8
        owned = (
            hidden_states[self.rank * local_rows : (self.rank + 1) * local_rows]
            if attention_gather is None
            else attention_gather.own_rows
        )
        logits, latent = launch_gluon_latent_input_largem_routed_gfx950(
            owned, packed_weight, warm_variants=is_warming_up()
        )
        # The prefill router allocates its outputs; copy them into the gather.
        weights, ids = route(owned, logits)
        local[3].copy_(weights)
        local[2].copy_(ids)
        _quantize_mxfp4_activation(latent, out=(local[0], local[1]))
        if attention_gather is not None:
            # Both gathers use the same flags; the shared rows need every row.
            self.join_gather(attention_gather)
        self.gather_moe_inputs(rows)
        shared = launch_gluon_latent_input_largem_shared_gfx950(
            hidden_states,
            packed_weight,
            beta=beta,
            linear_beta=linear_beta,
            warm_variants=is_warming_up(),
        )
        return (full[0], full[1]), full[3], full[2], shared

    def gather_moe_inputs(self, rows: int) -> None:
        """Push every rank's filled ``moe_input_views`` shard to all peers.

        The caller issues this after the previous MoE has passed its
        reduce-scatter entry barrier, which orders every peer's reads of the
        previous gather before these stores.
        """
        assert self.can_gather_moe_inputs(rows)
        local_rows = rows // 8
        num_warps = 4
        iris_k3moe_input_push_gather[(MOE_INPUT_GATHER_PROGRAMS,)](
            self.moe_input_data,
            self.moe_input_scale,
            self.moe_input_ids,
            self.moe_input_weights.view(torch.int32),
            self.gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            DATA_WORDS=local_rows * MOE_INPUT_DATA_BYTES // 4,
            SCALE_WORDS=local_rows * MOE_INPUT_SCALE_BYTES // 4,
            ROUTE_WORDS=local_rows * MOE_INPUT_TOPK,
            BLOCK=4 * 64 * num_warps,
            NUM_PROGRAMS=MOE_INPUT_GATHER_PROGRAMS,
            NUM_WARPS=num_warps,
            num_warps=num_warps,
            WAIT_STORES=self._arch.wait_stores,
        )

    def moe_tail(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        prefix_is_sharded: bool,
        norm_weight: torch.Tensor | None,
        eps: float | None,
        shared_reduced: _ReducingMoeShared | None,
    ) -> torch.Tensor | None:
        """Reduce and project local MoE rows, then gather into the borrowed result.

        The caller has verified producer ownership. Reject unsupported operands
        before publishing to peers; each rank handles consecutive rows.
        ``shared_reduced`` (from ``project_moe_shared``) already reduces the
        shared rows.
        """
        self.join_moe_shared(shared_reduced, shared_partial)
        if not self._moe_tail_supported(
            routed_partial,
            shared_partial,
            prefix,
            projection_weight,
            prefix_is_sharded=prefix_is_sharded,
            norm_weight=norm_weight,
            eps=eps,
        ):
            return None
        rows = routed_partial.shape[0]
        local_rows = rows // 8
        output = self.output[:rows]
        # Sharded prefixes project in place; a replicated prefix may occupy the result.
        projected = (
            output[self.rank * local_rows : (self.rank + 1) * local_rows]
            if prefix_is_sharded
            else torch.empty(
                (local_rows, 7168), device=prefix.device, dtype=prefix.dtype
            )
        )
        shared = self._moe_tail_reduce_project(
            rows,
            projection_weight,
            projected,
            norm_weight=norm_weight,
            eps=eps,
            shared_reduced=shared_reduced is not None,
        )
        iris_k3moe_push_gather[(GATHER_PROGRAMS,)](
            projected,
            shared,
            prefix,
            output,
            self.gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=GATHER_PROGRAMS,
            NUM_WARPS=4,
            PREFIX_IS_SHARDED=prefix_is_sharded,
            num_warps=4,
            WAIT_STORES=self._arch.wait_stores,
        )
        return output

    def moe_tail_mix(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        prefix_is_sharded: bool,
        norm_weight: torch.Tensor | None,
        eps: float | None,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        mix_eps: float,
        out_norm_weight: torch.Tensor,
        out_norm_eps: float,
        num_valid_blocks: int,
        block_write_idx: int,
        next_weight: torch.Tensor | None,
        gather_next_projection: bool,
        shared_reduced: _ReducingMoeShared | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Finish owned MoE residual rows and gather the next AttnRes mix instead.

        Returns ``(residual, activation, projection)``: this rank's consecutive
        rows of the updated residual (the ``block_write_idx`` snapshot rows when
        one is given) and the borrowed, replicated mix normalized by
        ``out_norm_weight``. ``block_residual`` is block-major ``[blocks, T,
        7168]``; only this rank's rows of it are read or written.

        Given the next layer's first ``[N, 7168]`` BF16 weight, ``projection``
        is ``activation @ next_weight.T`` (otherwise None), and each rank
        projects its own mixed rows while the activation gathers. A replicated
        weight (``gather_next_projection``, the MLA q_a/kv_a rows) is applied
        to the own rows only and the projection gathers into borrowed storage.
        A column-sharded weight (the KDA input projection) then projects the
        other rows once they arrive, with the large-M GEMM, into new storage;
        that needs the rows to tile by ``8 * 256``. ``shared_reduced`` (from
        ``project_moe_shared``) already reduces the shared rows.
        """
        self.join_moe_shared(shared_reduced, shared_partial)
        if not self._moe_tail_supported(
            routed_partial,
            shared_partial,
            prefix,
            projection_weight,
            prefix_is_sharded=prefix_is_sharded,
            norm_weight=norm_weight,
            eps=eps,
        ):
            return None
        rows = routed_partial.shape[0]
        weights = (res_weight, rms_weight, out_norm_weight)
        if (
            block_residual.ndim != 3
            or block_residual.shape[1] != rows
            or block_residual.shape[2] != 7168
            or block_residual.dtype != torch.bfloat16
            or block_residual.device != prefix.device
            or block_residual.stride(2) != 1
            or block_residual.stride(1) != 7168
            or (rows - 1) * 7168 + 7168 >= 1 << 30
            or not isinstance(num_valid_blocks, int)
            or not 0 <= num_valid_blocks <= min(11, block_residual.shape[0])
            or not (
                block_write_idx == -1
                or num_valid_blocks == block_write_idx < block_residual.shape[0]
            )
            or not math.isfinite(mix_eps)
            or mix_eps <= 0
            or not math.isfinite(out_norm_eps)
            or out_norm_eps <= 0
            or any(
                weight.shape != (7168,)
                or weight.dtype != torch.bfloat16
                or weight.device != prefix.device
                or not weight.is_contiguous()
                for weight in weights
            )
        ):
            return None
        if next_weight is not None and (
            next_weight.ndim != 2
            or next_weight.shape[1] != 7168
            or next_weight.dtype != torch.bfloat16
            or next_weight.device != prefix.device
            or next_weight.stride(1) != 1
            or (
                gather_next_projection
                and (
                    next_weight.shape[0] != MLA_QKV_A_WIDTH
                    or rows > self.qkv_output.shape[0]
                )
            )
            or (not gather_next_projection and rows % (8 * _LARGEM_ROW_TILE))
        ):
            return None
        protected = (
            self.pull.producer.data,
            self.pull.scratch,
            self.output,
            self.qkv_output,
        )
        operands = (block_residual, *weights)
        if next_weight is not None:
            operands += (next_weight,)
        if any(
            _overlaps(tensor, buffer) for tensor in operands for buffer in protected
        ) or (block_write_idx >= 0 and _overlaps(prefix, block_residual)):
            return None

        global _attn_res_mix_gfx950
        from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
            _attn_res_mix_gfx950,
        )

        local_rows = rows // 8
        first_row = self.rank * local_rows
        output = self.output[:rows]
        projected = torch.empty(
            (local_rows, 7168), device=prefix.device, dtype=prefix.dtype
        )
        residual = (
            block_residual[block_write_idx, first_row : first_row + local_rows]
            if block_write_idx >= 0
            else torch.empty_like(projected)
        )
        shared = self._moe_tail_reduce_project(
            rows,
            projection_weight,
            projected,
            norm_weight=norm_weight,
            eps=eps,
            shared_reduced=shared_reduced is not None,
        )
        if next_weight is not None:
            projection = self._mix_gather_project(
                projected,
                shared,
                prefix,
                residual,
                output,
                block_residual,
                weights,
                next_weight,
                gather_next_projection=gather_next_projection,
                prefix_is_sharded=prefix_is_sharded,
                mix_eps=mix_eps,
                out_norm_eps=out_norm_eps,
                num_valid_blocks=num_valid_blocks,
            )
            return residual, output, projection
        iris_k3moe_mix_push_gather[(GATHER_PROGRAMS,)](
            projected,
            shared,
            prefix,
            residual,
            output,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            self.gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            STRIDE_BLOCK_T=block_residual.stride(1),
            STRIDE_BLOCK_N=block_residual.stride(0),
            NUM_VALID_BLOCKS=num_valid_blocks,
            SCORE_EPS=mix_eps,
            OUTPUT_EPS=out_norm_eps,
            PREFIX_IS_SHARDED=prefix_is_sharded,
            NUM_PROGRAMS=GATHER_PROGRAMS,
            NUM_WARPS=4,
            num_warps=4,
            WAIT_STORES=self._arch.wait_stores,
        )
        return residual, output, None

    def _gather_side_stream(self, device: torch.device) -> torch.cuda.Stream:
        """The stream a gather overlaps on: the calling stream while capturing.

        A side-stream fork captured into a breakable prefill graph faults at
        replay on ROCm once the host runs ahead of the GPU.
        """
        if self.gather_stream is None:
            self.gather_stream = _side_stream(device)
        if torch.cuda.is_current_stream_capturing():
            return torch.cuda.current_stream(device)
        return self.gather_stream

    def _mix_gather_project(
        self,
        projected: torch.Tensor,
        shared: torch.Tensor,
        prefix: torch.Tensor,
        residual: torch.Tensor,
        output: torch.Tensor,
        block_residual: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        next_weight: torch.Tensor,
        *,
        gather_next_projection: bool,
        prefix_is_sharded: bool,
        mix_eps: float,
        out_norm_eps: float,
        num_valid_blocks: int,
    ) -> torch.Tensor:
        """Mix owned rows, gather them on a side stream, project them meanwhile.

        Only one collective runs at a time: a projection gather starts on the
        calling stream after the activation gather has finished.
        """
        local_rows = projected.shape[0]
        res_weight, rms_weight, out_norm_weight = weights
        mixed = torch.empty_like(projected)
        k3moe_residual_mix_rows[(_MIX_ROWS_PROGRAMS,)](
            projected,
            shared,
            prefix,
            residual,
            mixed,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            STRIDE_BLOCK_T=block_residual.stride(1),
            STRIDE_BLOCK_N=block_residual.stride(0),
            NUM_VALID_BLOCKS=num_valid_blocks,
            SCORE_EPS=mix_eps,
            OUTPUT_EPS=out_norm_eps,
            PREFIX_IS_SHARDED=prefix_is_sharded,
            NUM_PROGRAMS=_MIX_ROWS_PROGRAMS,
            NUM_WARPS=4,
            num_warps=4,
        )
        if not gather_next_projection:
            out = self._gather_rows_project(mixed, output, next_weight)
            if out is not None:
                return out
        main = torch.cuda.current_stream(mixed.device)
        side = self._gather_side_stream(mixed.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            iris_k3attn_push_gather[(SIDE_GATHER_PROGRAMS,)](
                mixed,
                output,
                self.gather_flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=local_rows,
                BLOCK_ELEMENTS=2048,
                NUM_PROGRAMS=SIDE_GATHER_PROGRAMS,
                NUM_WARPS=4,
                num_warps=4,
                WAIT_STORES=self._arch.wait_stores,
            )
        mixed.record_stream(side)
        rows = local_rows * 8
        if not gather_next_projection:
            from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
                launch_gluon_mm_a16w16_prefill_gfx950,
                launch_gluon_mm_a16w16_prefill_skip_rows_gfx950,
            )

            first_row = self.rank * local_rows
            out = torch.empty(
                (rows, next_weight.shape[0]), device=mixed.device, dtype=mixed.dtype
            )
            own = launch_gluon_mm_a16w16_prefill_gfx950(
                mixed,
                next_weight,
                mixed.dtype,
                out=out[first_row : first_row + local_rows],
            )
            main.wait_stream(side)
            others = launch_gluon_mm_a16w16_prefill_skip_rows_gfx950(
                output,
                next_weight,
                out,
                skip_start=first_row,
                skip_rows=local_rows,
            )
            if own is None or others is None:
                raise RuntimeError("the large-M GEMM declined a validated projection")
            return out
        qkv_rows = torch.nn.functional.linear(mixed, next_weight)
        main.wait_stream(side)
        qkv = self.qkv_output[:rows]
        iris_k3_push_gather_partition[(GATHER_PROGRAMS,)](
            qkv_rows,
            qkv,
            self.qkv_gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            WIDTH=MLA_QKV_A_WIDTH,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=GATHER_PROGRAMS,
            NUM_WARPS=4,
            num_warps=4,
            WAIT_STORES=self._arch.wait_stores,
        )
        return qkv

    def _gather_rows_project(
        self, mixed: torch.Tensor, output: torch.Tensor, next_weight: torch.Tensor
    ) -> torch.Tensor | None:
        """Gather ``mixed`` into ``output`` inside the projection GEMM, or nothing.

        Each row tile of a peer is projected as soon as it lands, so the
        gather hides behind the GEMM instead of preceding seven eighths of it.
        """
        local_rows = mixed.shape[0]
        tiles = local_rows // _LARGEM_ROW_TILE
        if (
            not _gather_rows_gemm_enabled()
            or torch.cuda.is_current_stream_capturing()
            or local_rows % _LARGEM_ROW_TILE
            or not 0 < tiles <= len(self._gather_row_counts)
        ):
            return None
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            LARGEM_GATHER_PUSH_PROGRAMS,
            gather_rows_mm_tickets,
            launch_gluon_mm_a16w16_prefill_gather_rows_gfx950,
        )

        counts = list(self._gather_row_counts)
        for tile in range(tiles):
            count = counts[tile] + LARGEM_GATHER_PUSH_PROGRAMS
            counts[tile] = (count + (1 << 31)) % (1 << 32) - (1 << 31)
        out = torch.empty(
            (output.shape[0], next_weight.shape[0]),
            device=mixed.device,
            dtype=mixed.dtype,
        )
        if (
            launch_gluon_mm_a16w16_prefill_gather_rows_gfx950(
                mixed,
                output,
                next_weight,
                out,
                self.gather_row_counters,
                self.gather_row_tickets,
                self.heap_bases,
                rank=self.rank,
                targets=tuple(counts),
                ticket_base=self._gather_ticket_base,
            )
            is None
        ):
            return None
        self._gather_row_counts = counts
        base = self._gather_ticket_base + gather_rows_mm_tickets(
            local_rows, next_weight.shape[0]
        )
        self._gather_ticket_base = (base + (1 << 31)) % (1 << 32) - (1 << 31)
        return out

    def _mix_side_gather(
        self,
        prefix: torch.Tensor,
        output: torch.Tensor,
        history: torch.Tensor,
        block_residual: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        eps: float,
        out_norm_eps: float,
        num_valid_blocks: int,
        fuse_mix: bool,
    ) -> _PendingGather:
        """Mix the owned rows locally, then push them to every peer on a side stream."""
        from tokenspeed_kernel.ops.residual import attn_res_fwd

        res_weight, rms_weight, out_norm_weight = weights
        rows = prefix.shape[0]
        if fuse_mix:
            # The fused gather's mix, stored locally, keeps its bits.
            mixed = torch.empty_like(prefix)
            num_subgroups = 8 if num_valid_blocks <= 7 else 4
            k3attn_mix_rows[(_MIX_ROWS_PROGRAMS,)](
                prefix,
                mixed,
                block_residual,
                res_weight,
                rms_weight,
                out_norm_weight,
                RANK=self.rank,
                LOCAL_ROWS=rows,
                STRIDE_BLOCK_T=block_residual.stride(1),
                STRIDE_BLOCK_N=block_residual.stride(0),
                NUM_VALID_BLOCKS=num_valid_blocks,
                SCORE_EPS=eps,
                OUTPUT_EPS=out_norm_eps,
                NUM_PROGRAMS=_MIX_ROWS_PROGRAMS,
                NUM_WARPS=num_subgroups,
                num_warps=num_subgroups,
            )
        else:
            mixed = attn_res_fwd(
                prefix,
                history,
                res_weight,
                rms_weight,
                eps,
                out_norm_weight=out_norm_weight,
                out_norm_eps=out_norm_eps,
                delta=None,
                num_valid_blocks=num_valid_blocks,
                block_write_idx=-1,
            )
        side = self._gather_side_stream(mixed.device)
        side.wait_stream(torch.cuda.current_stream(mixed.device))
        with torch.cuda.stream(side):
            iris_k3attn_push_gather[(SIDE_GATHER_PROGRAMS,)](
                mixed,
                output,
                self.gather_flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=rows,
                BLOCK_ELEMENTS=2048,
                NUM_PROGRAMS=SIDE_GATHER_PROGRAMS,
                NUM_WARPS=4,
                num_warps=4,
                WAIT_STORES=self._arch.wait_stores,
            )
        mixed.record_stream(side)
        done = torch.cuda.Event()
        done.record(side)
        return _PendingGather(mixed, output, done)

    def gather_prefill_topk(
        self, selection: torch.Tensor, lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """All-gather each rank's DSA prefill selection rows as int16, and their lengths.

        ``selection`` is this rank's ``[local_rows, PREFILL_TOPK]`` int32 slots
        below 0xFFFF (or -1) and ``lens`` its ``[local_rows]`` int32 lengths.
        Returns new ``[8 * local_rows, PREFILL_TOPK]`` int16 rows (the low 16
        bits of rank ``q``'s rows at ``q * local_rows``) and the gathered
        lengths; ``None`` when unsupported.
        """
        local_rows = selection.shape[0] if selection.ndim == 2 else 0
        if (
            not current_platform().is_cdna4
            or local_rows == 0
            or local_rows % 8
            or local_rows > self.topk_publish.shape[1]
            or selection.shape[1] != PREFILL_TOPK
            or selection.dtype != torch.int32
            or not selection.is_contiguous()
            or lens.shape != (local_rows,)
            or lens.dtype != torch.int32
            or not lens.is_contiguous()
            or selection.device != self.topk_publish.device
            or lens.device != self.topk_publish.device
            or any(
                _overlaps(tensor, buffer)
                for tensor in (selection, lens)
                for buffer in (self.topk_publish, self.topk_lens_publish)
            )
        ):
            return None
        rows = local_rows * 8
        output = torch.empty(
            (rows, PREFILL_TOPK), dtype=torch.int16, device=selection.device
        )
        lens_output = torch.empty((rows,), dtype=torch.int32, device=selection.device)
        iris_k3_pull_gather_topk_rows[(PREFILL_TOPK_GATHER_PROGRAMS,)](
            selection,
            lens,
            self.topk_publish,
            self.topk_lens_publish,
            output,
            lens_output,
            self.topk_gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            TOPK=PREFILL_TOPK,
            MAX_LOCAL_ROWS=self.topk_publish.shape[1],
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=PREFILL_TOPK_GATHER_PROGRAMS,
            NUM_WARPS=4,
            num_warps=4,
            WAIT_STORES=self._arch.wait_stores,
        )
        return output, lens_output

    def join_gather(self, pending: _PendingGather) -> torch.Tensor:
        """Order the calling stream after a deferred gather; return its output."""
        torch.cuda.current_stream(pending.output.device).wait_event(pending.done)
        return pending.output

    def _published_scatter(
        self,
        published: _PublishedProjection,
        partial: torch.Tensor,
        residual: torch.Tensor | None,
        prefix: torch.Tensor,
        residual_is_sharded: bool,
    ) -> None:
        """Scatter on a side stream that starts with the publishing GEMM."""
        if published.out.data_ptr() != partial.data_ptr() or (
            published.out.shape != partial.shape
        ):
            raise RuntimeError("the published projection is not this partial")
        self._launch_published_scatter(
            published, partial, residual, prefix, residual_is_sharded
        )
        # The calling stream allocated these and waits below, before any reuse.
        torch.cuda.current_stream(partial.device).wait_stream(self.scatter_stream)

    def _launch_published_scatter(
        self,
        published: _PublishedProjection,
        partial: torch.Tensor,
        residual: torch.Tensor | None,
        prefix: torch.Tensor,
        residual_is_sharded: bool,
    ) -> None:
        if self.scatter_stream is None:
            self.scatter_stream = _side_stream(partial.device)
        side = self.scatter_stream
        side.wait_event(published.start)
        with torch.cuda.stream(side):
            iris_k3attn_pull_scatter_published[(PUBLISHED_SCATTER_PROGRAMS,)](
                partial,
                residual,
                prefix,
                self.published_counters,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=partial.shape[0] // 8,
                TARGET_0=published.targets[0],
                TARGET_1=published.targets[1],
                TARGET_2=published.targets[2],
                TARGET_3=published.targets[3],
                TILE_ROWS=_LARGEM_ROW_TILE,
                BLOCK_ELEMENTS=2048,
                NUM_PROGRAMS=PUBLISHED_SCATTER_PROGRAMS,
                NUM_WARPS=4,
                HAS_RESIDUAL=residual is not None,
                RESIDUAL_IS_SHARDED=residual_is_sharded,
                num_warps=4,
            )

    def _moe_tail_supported(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        prefix_is_sharded: bool,
        norm_weight: torch.Tensor | None,
        eps: float | None,
    ) -> bool:
        """Whether the row-sharded MoE tail takes these operands."""
        if not current_platform().is_cdna4 or routed_partial.ndim != 2:
            return False
        rows, latent = routed_partial.shape
        if (
            rows <= 0
            or rows % 8 != 0
            or latent != 3584
            or shared_partial.shape != (rows, 7168)
            or prefix.shape != (rows // 8 if prefix_is_sharded else rows, 7168)
            or projection_weight.shape != (7168, 3584)
        ):
            return False
        tensors = (routed_partial, shared_partial, prefix, projection_weight)
        if norm_weight is not None:
            if (
                norm_weight.shape != (3584,)
                or eps is None
                or not math.isfinite(eps)
                or eps <= 0
            ):
                return False
            tensors += (norm_weight,)
        elif eps is not None:
            return False
        if any(
            not tensor.is_cuda
            or tensor.device != routed_partial.device
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            return False
        result_buffer = self.output
        if result_buffer.shape[0] < rows:
            return False
        # Reject unsafe overlaps before launching either collective.
        for tensor in tensors[2:]:
            for buffer in (self.pull.producer.data, self.pull.scratch, result_buffer):
                if _overlaps(tensor, buffer):
                    # Only exact prefix aliasing preserves row ownership.
                    if not (
                        buffer is result_buffer
                        and tensor is prefix
                        and tensor.data_ptr() == buffer.data_ptr()
                        and not prefix_is_sharded
                    ):
                        return False
        return True

    def _moe_tail_reduce_project(
        self,
        rows: int,
        projection_weight: torch.Tensor,
        projected: torch.Tensor,
        *,
        norm_weight: torch.Tensor | None,
        eps: float | None,
        shared_reduced: bool,
    ) -> torch.Tensor:
        """Reduce-scatter both MoE partials, then project the owned routed rows.

        With ``shared_reduced`` the scratch already holds the shared rows.
        Returns the owned shared-expert rows, a view of the scratch.
        """
        from tokenspeed_kernel.ops.gemm import kimi3_latent_projection
        from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

        local_rows = rows // 8
        routed_elements = local_rows * 3584
        shared_elements = local_rows * 7168
        scratch = self.pull.scratch
        routed = scratch[:routed_elements].view(local_rows, 3584)
        shared = scratch[routed_elements : routed_elements + shared_elements].view(
            local_rows, 7168
        )
        iris_k3moe_pull_scatter[(REDUCE_PROGRAMS,)](
            self.pull.producer.data,
            scratch,
            self.pull.ready_flags,
            *self.heap_bases,
            RANK=self.rank,
            ROWS=rows,
            FIRST_WIDTH=3584,
            SECOND_WIDTH=0 if shared_reduced else 7168,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=REDUCE_PROGRAMS,
            NUM_WARPS=4,
            num_warps=4,
        )
        normalized = (
            rmsnorm(routed, norm_weight, eps, residual=None, out=routed)
            if norm_weight is not None
            else routed
        )
        kimi3_latent_projection(
            normalized, projection_weight, out=projected, solution="auto"
        )
        return shared

    def attn_mix(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor | None,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        *,
        eps: float,
        out_norm_weight: torch.Tensor,
        out_norm_eps: float,
        num_valid_blocks: int,
        residual_is_sharded: bool,
        published: _PublishedProjection | None,
        defer_gather: bool,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, _PendingGather]
        | None
    ):
        """Return owned residual rows and a borrowed, replicated AttnRes mix.

        The caller has verified producer ownership. Validate the fusion before
        launching either collective; runtime policy chooses its token window.
        A sharded ``residual`` holds only this rank's consecutive rows.
        ``published`` (from ``take_published``) reduces each owned row tile
        beside the GEMM that publishes them. With ``defer_gather`` the mix is
        gathered on a side stream and a third value is returned: join it
        (``join_gather``) before reading rows of other ranks.
        """
        if (
            not current_platform().is_cdna4
            or partial.ndim != 2
            or partial.shape[0] <= 0
            or partial.shape[0] % 8 != 0
            or partial.shape[1] != 7168
            or not partial.is_cuda
            or partial.dtype != torch.bfloat16
            or not partial.is_contiguous()
            or block_residual.ndim != 3
            or block_residual.shape[1:] != partial.shape
            or block_residual.dtype != partial.dtype
            or block_residual.device != partial.device
            or block_residual.stride(-1) != 1
            or (partial.shape[0] - 1) * block_residual.stride(1) + 7168 >= 1 << 30
            or not isinstance(num_valid_blocks, int)
            or not 0 <= num_valid_blocks <= min(11, block_residual.shape[0])
            or not math.isfinite(eps)
            or eps <= 0
            or not math.isfinite(out_norm_eps)
            or out_norm_eps <= 0
        ):
            return None
        residual_shape = (
            (partial.shape[0] // 8, 7168) if residual_is_sharded else partial.shape
        )
        if residual is not None and (
            residual.shape != residual_shape
            or residual.dtype != partial.dtype
            or residual.device != partial.device
            or not residual.is_contiguous()
        ):
            return None
        weights = (res_weight, rms_weight, out_norm_weight)
        if any(
            weight.shape != (7168,)
            or weight.dtype != partial.dtype
            or weight.device != partial.device
            or not weight.is_contiguous()
            for weight in weights
        ):
            return None

        output_buffer = self.output
        flags = self.pull.ready_flags
        gather_flags = self.gather_flags
        if output_buffer.shape[0] < partial.shape[0]:
            return None
        protected = (self.pull.producer.data, self.pull.scratch, output_buffer)
        for tensor in (block_residual, *weights):
            if any(_overlaps(tensor, buffer) for buffer in protected):
                return None
        if residual is not None:
            for buffer in protected:
                if _overlaps(residual, buffer) and (
                    residual_is_sharded
                    or not (
                        buffer is output_buffer
                        and residual.data_ptr() == buffer.data_ptr()
                    )
                ):
                    return None

        from tokenspeed_kernel.ops.residual import attn_res_fwd, attn_res_fwd_available

        rows = partial.shape[0] // 8
        first_row = self.rank * rows
        history = block_residual[:, first_row : first_row + rows]
        if not attn_res_fwd_available(
            partial[:rows],
            history,
            res_weight,
            rms_weight,
            eps,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            delta=None,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=-1,
        ):
            return None

        # The ordinary mixer is faster in the middle token range. At larger sizes,
        # longer histories also need enough rows to amortize live peer pointers.
        fuse_mix = (partial.shape[0] < 1024 or partial.shape[0] >= 4096) and (
            num_valid_blocks <= 6
            or (partial.shape[0] >= 7680 and num_valid_blocks <= 8)
        )
        # Resolve the fused device function before either collective publishes.
        global _attn_res_mix_gfx950
        from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
            _attn_res_mix_gfx950,
        )

        prefix = torch.empty_like(partial[:rows])
        output = output_buffer[: partial.shape[0]]
        if published is not None:
            self._published_scatter(
                published, partial, residual, prefix, residual_is_sharded
            )
        else:
            programs = REDUCE_PROGRAMS
            iris_k3attn_pull_scatter[(programs,)](
                partial,
                residual,
                prefix,
                flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=rows,
                BLOCK_ELEMENTS=2048,
                NUM_PROGRAMS=programs,
                NUM_WARPS=4,
                HAS_RESIDUAL=residual is not None,
                RESIDUAL_IS_SHARDED=residual_is_sharded,
                num_warps=4,
            )
        if defer_gather:
            return (
                prefix,
                output,
                self._mix_side_gather(
                    prefix,
                    output,
                    history,
                    block_residual,
                    weights,
                    eps=eps,
                    out_norm_eps=out_norm_eps,
                    num_valid_blocks=num_valid_blocks,
                    fuse_mix=fuse_mix,
                ),
            )
        if fuse_mix:
            gather_programs = GATHER_PROGRAMS
            num_subgroups = 8 if num_valid_blocks <= 7 else 4
            iris_k3attn_mix_push_gather[(gather_programs,)](
                prefix,
                output,
                block_residual,
                res_weight,
                rms_weight,
                out_norm_weight,
                gather_flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=rows,
                STRIDE_BLOCK_T=block_residual.stride(1),
                STRIDE_BLOCK_N=block_residual.stride(0),
                NUM_VALID_BLOCKS=num_valid_blocks,
                SCORE_EPS=eps,
                OUTPUT_EPS=out_norm_eps,
                NUM_PROGRAMS=gather_programs,
                NUM_WARPS=num_subgroups,
                num_warps=num_subgroups,
                WAIT_STORES=self._arch.wait_stores,
            )
        else:
            # Longer histories favor the existing mixer without persistent peer
            # pointers occupying registers throughout the candidate reductions.
            mixed = attn_res_fwd(
                prefix,
                history,
                res_weight,
                rms_weight,
                eps,
                out_norm_weight=out_norm_weight,
                out_norm_eps=out_norm_eps,
                delta=None,
                num_valid_blocks=num_valid_blocks,
                block_write_idx=-1,
            )
            gather_programs = 32
            iris_k3attn_push_gather[(gather_programs,)](
                mixed,
                output,
                gather_flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=rows,
                BLOCK_ELEMENTS=2048,
                NUM_PROGRAMS=gather_programs,
                NUM_WARPS=4,
                num_warps=4,
                WAIT_STORES=self._arch.wait_stores,
            )
        return prefix, output
