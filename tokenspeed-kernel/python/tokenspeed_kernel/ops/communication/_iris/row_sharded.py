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

"""Row-sharded attention and MoE fusions sharing one Iris result workspace."""

import math

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_drain_subgroup_vmem,
)
from tokenspeed_kernel.platform import current_platform

REDUCE_PROGRAMS = 24
GATHER_PROGRAMS = 128


@gluon.jit
def _row_partition_store_completion(
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
def _row_partition_entry_barrier(
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


# Rank q owns L = ROWS/8 consecutive rows. For local row u and coordinate j,
# r = q*L + u, reduce that row from all eight producer ranks:
#
#   scratch_routed_q[u,j] = BF16(sum_p FP32(routed_partial_p[r,j]))
#   scratch_shared_q[u,j] = BF16(sum_p FP32(shared_partial_p[r,j]))
#
# The sum uses the kernel's fixed FP32 tree. Each peer's input packs all
# routed rows before all shared rows; scratch packs only q's reduced rows
# in that order.
@gluon.jit(do_not_specialize=["ROWS"])
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
    ROWS,
    FIRST_WIDTH: gl.constexpr,
    SECOND_WIDTH: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Reduce paired routed/shared inputs into this rank's consecutive rows."""
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
    # The matching gather waits for every rank after these reads complete.
    # Its completion must precede the next producer's symmetric-input writes.


# Rank q owns L = M/8 consecutive rows. For local row u, r = q*L + u,
# set i = r for a replicated prefix or i = u when PREFIX_IS_SHARDED.
# For every destination rank p:
#
#   output_p[r,j] = BF16((FP32(prefix_q[i,j])
#                         + FP32(projected_q[u,j]))
#                         + FP32(shared_reduced_q[u,j]))
#
# Rank q pushes its rows to every peer; other ranks write disjoint rows.
@gluon.jit(do_not_specialize=["LOCAL_ROWS"])
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
):
    """Add projection, shared reduction, and prefix, then publish complete rows."""
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
    _row_partition_store_completion(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)


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


# Rank q owns L = M/8 consecutive rows. For local row u, r = q*L + u,
# reduce that row from all eight attention producers:
#
#   reduced_q[u,j] = BF16(sum_p FP32(partial_p[r,j]))
#   prefix_q[u,j] = reduced_q[u,j]                         (no residual)
#                 = BF16(FP32(reduced_q[u,j])
#                      + FP32(residual_q[r,j]))            (with residual)
#
# The sum uses the even/odd FP32 tree. BF16 rounding precedes the residual
# add; only rank q's rows are stored in its local prefix.
@gluon.jit(launch_metadata=_reduce_metadata, do_not_specialize=["LOCAL_ROWS"])
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
):
    """Reduce attention rows and round before adding the optional residual."""
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


# Rank q has mixed its L = M/8 local rows. For u in [0,L), r = q*L + u,
# and every destination rank p:
#
#   output_p[r,j] = mixed_q[u,j]
#
# Each rank pushes its rows to every peer; other ranks write disjoint rows.
@gluon.jit(launch_metadata=_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Publish already mixed local rows and wait for every peer's publication."""
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
    _row_partition_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)


# Rank q owns local prefix row u for global row r = q*L + u, L = M/8.
# Mix that prefix with the first NUM_VALID_BLOCKS history candidates at r,
# round to BF16, apply output RMSNorm, and round to BF16 again. For every peer p:
#
#   output_p[r,j] = mixed_q[u,j]
#
# Each rank pushes its rows to every peer; other ranks write disjoint rows.
@gluon.jit(launch_metadata=_mix_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
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
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    MIX: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Mix and normalize local AttnRes rows, then publish them to all peers."""
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
        mixed = MIX(
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
    _row_partition_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)


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


class IrisRowShardedWorkspace:
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
            max_rows * 7168 * dtype.itemsize
            + GATHER_PROGRAMS * world_size * torch.int32.itemsize
        )

    def __init__(
        self,
        ctx,
        *,
        rank: int,
        heap_bases: tuple[int, ...],
        inputs: torch.Tensor,
        scratch: torch.Tensor,
        reduce_flags: torch.Tensor,
        max_rows: int,
    ) -> None:
        assert inputs.numel() >= max_rows * (7168 + 3584)
        assert scratch.numel() >= max_rows // 8 * (7168 + 3584)
        assert reduce_flags.shape[0] >= REDUCE_PROGRAMS
        self.rank = rank
        self.heap_bases = heap_bases
        self.inputs = inputs
        self.scratch = scratch
        self.reduce_flags = reduce_flags
        self.output = ctx.empty((max_rows, 7168), dtype=inputs.dtype)
        self.gather_flags = ctx.zeros((GATHER_PROGRAMS, 8), dtype=torch.int32)

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
    ) -> torch.Tensor | None:
        """Reduce and project local MoE rows, then gather into the borrowed result.

        The caller has verified producer ownership. Reject unsupported operands
        before publishing to peers; each rank handles consecutive rows.
        """
        if not current_platform().is_cdna4 or routed_partial.ndim != 2:
            return None
        rows, latent = routed_partial.shape
        if (
            rows <= 0
            or rows % 8 != 0
            or latent != 3584
            or shared_partial.shape != (rows, 7168)
            or prefix.shape != (rows // 8 if prefix_is_sharded else rows, 7168)
            or projection_weight.shape != (7168, 3584)
        ):
            return None
        tensors = (routed_partial, shared_partial, prefix, projection_weight)
        if norm_weight is not None:
            if (
                norm_weight.shape != (3584,)
                or eps is None
                or not math.isfinite(eps)
                or eps <= 0
            ):
                return None
            tensors += (norm_weight,)
        elif eps is not None:
            return None
        if any(
            not tensor.is_cuda
            or tensor.device != routed_partial.device
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            return None

        local_rows = rows // 8
        routed_elements = local_rows * 3584
        shared_elements = local_rows * 7168
        scratch = self.scratch
        flags = self.reduce_flags
        programs = REDUCE_PROGRAMS
        gather_programs = GATHER_PROGRAMS
        result_buffer = self.output
        gather_flags = self.gather_flags
        if result_buffer.shape[0] < rows:
            return None
        # Reject unsafe overlaps before launching either collective.
        for tensor in tensors[2:]:
            for buffer in (self.inputs, scratch, result_buffer):
                if _overlaps(tensor, buffer):
                    # Only exact prefix aliasing preserves row ownership.
                    if not (
                        buffer is result_buffer
                        and tensor is prefix
                        and tensor.data_ptr() == buffer.data_ptr()
                        and not prefix_is_sharded
                    ):
                        return None

        from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection
        from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

        routed = scratch[:routed_elements].view(local_rows, 3584)
        shared = scratch[routed_elements : routed_elements + shared_elements].view(
            local_rows, 7168
        )
        output = result_buffer[:rows]
        # A sharded prefix is disjoint from the result. Its owner can project into
        # its output rows, then consume them before the gather overwrites them.
        # A replicated prefix can already occupy the result and must be preserved.
        projected = (
            output[self.rank * local_rows : (self.rank + 1) * local_rows]
            if prefix_is_sharded
            else torch.empty(
                (local_rows, 7168), device=prefix.device, dtype=prefix.dtype
            )
        )
        iris_moe_reduce_scatter_gluon_kernel[(programs,)](
            self.inputs,
            scratch,
            flags,
            *self.heap_bases,
            RANK=self.rank,
            ROWS=rows,
            FIRST_WIDTH=3584,
            SECOND_WIDTH=7168,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=programs,
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
        iris_moe_add_push_gather_gluon_kernel[(gather_programs,)](
            projected,
            shared,
            prefix,
            output,
            gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=gather_programs,
            NUM_WARPS=4,
            PREFIX_IS_SHARDED=prefix_is_sharded,
            num_warps=4,
        )
        return output

    def attention_mix(
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
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return owned residual rows and a borrowed, replicated AttnRes mix.

        The caller has verified producer ownership. Validate the fusion before
        launching either collective; runtime policy chooses its token window.
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
        if residual is not None and (
            residual.shape != partial.shape
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
        flags = self.reduce_flags
        gather_flags = self.gather_flags
        if output_buffer.shape[0] < partial.shape[0]:
            return None
        protected = (self.inputs, self.scratch, output_buffer)
        for tensor in (block_residual, *weights):
            if any(_overlaps(tensor, buffer) for buffer in protected):
                return None
        if residual is not None:
            for buffer in protected:
                if _overlaps(residual, buffer) and not (
                    buffer is output_buffer and residual.data_ptr() == buffer.data_ptr()
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
        if fuse_mix:
            from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
                _attn_res_mix_gfx950,
            )

        prefix = torch.empty_like(partial[:rows])
        output = output_buffer[: partial.shape[0]]
        programs = REDUCE_PROGRAMS
        iris_attention_reduce_scatter_gluon_kernel[(programs,)](
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
            num_warps=4,
        )
        if fuse_mix:
            gather_programs = GATHER_PROGRAMS
            num_subgroups = 8 if num_valid_blocks <= 7 else 4
            iris_attention_mix_push_gluon_kernel[(gather_programs,)](
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
                MIX=_attn_res_mix_gfx950,
                SCORE_EPS=eps,
                OUTPUT_EPS=out_norm_eps,
                NUM_PROGRAMS=gather_programs,
                NUM_WARPS=num_subgroups,
                num_warps=num_subgroups,
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
            iris_attention_push_gather_gluon_kernel[(gather_programs,)](
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
            )
        return prefix, output
