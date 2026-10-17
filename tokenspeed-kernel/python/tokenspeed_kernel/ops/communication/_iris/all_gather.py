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


"""Column and row all-gathers through symmetric storage."""

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton
from tokenspeed_kernel.ops.communication._iris import allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    ProducerBuffer,
    _iris_drain_subgroup_vmem,
    _iris_group_heap_base,
    _iris_heap_base,
    _iris_peer_pointers,
    _iris_rendezvous,
    _iris_sync_peers,
    _iris_sync_rank_epoch,
)
from tokenspeed_kernel.ops.communication.iris import _PRODUCER_DIRECT_GL_DTYPES
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
        _mxfp4_quantize_tile_in_layout,
        _mxfp4_store_cdna4_scale,
        scaled_downcast_layout,
    )


class IrisAllGather:
    """Borrow producer storage; retain separate gather epochs and local results.

    Small results borrow ``output`` until the next collective using that buffer.
    Larger gathers process capacity-sized row chunks into independent storage.
    """

    @staticmethod
    def heap_bytes(world_size: int) -> int:
        return allocation_bytes((84, world_size), torch.int32)

    def __init__(self, producer: ProducerBuffer):
        self.producer = producer
        self.ready_flags = producer.peers.ctx.zeros(
            (84, producer.peers.world_size), dtype=torch.int32
        )
        self.output = torch.empty_like(producer.data)

    def can_run(self, tensor: torch.Tensor) -> bool:
        """Check prepared TP4/TP8 storage for a possibly chunked hidden-shard gather."""
        return (
            current_platform().is_cdna4
            and self.producer.peers.world_size in (4, 8)
            and tensor.ndim == 2
            and tensor.dtype == self.producer.data.dtype == torch.bfloat16
            and tensor.device == self.producer.data.device
            and tensor.is_contiguous()
            and tensor.data_ptr() % 8 == 0
            and tensor.shape[0] > 0
            and all_gather_columns_can_run(
                world_size=self.producer.peers.world_size,
                num_tokens=1,
                local_hidden_size=tensor.shape[1],
                dtype=tensor.dtype,
                max_bytes=self.producer.data.numel() * tensor.element_size(),
            )
        )

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        """Concatenate equal hidden-dimension TP4/TP8 shards through symmetric memory."""
        assert tensor.ndim == 2, "Iris column all-gather requires a 2D tensor"
        assert (
            tensor.device == self.producer.data.device
        ), f"Iris device mismatch: tensor={tensor.device}, state={self.producer.data.device}"
        assert (
            tensor.dtype == self.producer.data.dtype
        ), f"Iris dtype mismatch: tensor={tensor.dtype}, state={self.producer.data.dtype}"
        assert (
            tensor.is_contiguous()
        ), "Iris column all-gather requires contiguous input"
        num_tokens, local_hidden_size = tensor.shape
        total_hidden_size = local_hidden_size * self.producer.peers.world_size
        total_numel = num_tokens * total_hidden_size
        assert all_gather_columns_can_run(
            world_size=self.producer.peers.world_size,
            num_tokens=1,
            local_hidden_size=local_hidden_size,
            dtype=tensor.dtype,
            max_bytes=self.producer.data.numel() * tensor.element_size(),
        )
        assert num_tokens > 0
        assert self.producer.data is not None
        assert self.output is not None
        assert self.ready_flags is not None
        if total_numel <= self.producer.data.numel():
            output = self.output[:total_numel].view(num_tokens, total_hidden_size)
        else:
            output = tensor.new_empty((num_tokens, total_hidden_size))
        chunk_rows = self.producer.data.numel() // total_hidden_size
        for start in range(0, num_tokens, chunk_rows):
            local = tensor[start : start + chunk_rows]
            rows = local.shape[0]
            elements_per_word = 8 // self.producer.data.element_size()
            local_words = rows * local_hidden_size // elements_per_word
            block_words = 64 * 1 * 4
            num_tiles = triton.cdiv(local_words, block_words)
            num_programs = min(num_tiles, 84)
            published_input = (
                rows == 1
                and local.data_ptr()
                == self.producer.data.data_ptr()
                + self.producer.peers.rank * local_hidden_size * tensor.element_size()
            )
            iris_pull_allgather_columns[(num_programs,)](
                local,
                self.producer.data,
                output[start : start + chunk_rows],
                self.ready_flags,
                self.producer.peers.addresses,
                RANK=self.producer.peers.rank,
                WORLD_SIZE=self.producer.peers.world_size,
                LOCAL_WORDS=local_words,
                LOCAL_HIDDEN_WORDS=local_hidden_size // elements_per_word,
                TOTAL_HIDDEN_WORDS=total_hidden_size // elements_per_word,
                BLOCK_WORDS=block_words,
                NUM_WARPS=1,
                SUBGROUP_SIZE=64,
                WORDS_PER_LANE=4,
                INPUT_PUBLISHED=published_input,
                num_warps=1,
            )
        return output


def all_gather_columns_can_run(
    world_size: int,
    num_tokens: int,
    local_hidden_size: int,
    dtype: torch.dtype,
    max_bytes: int,
) -> bool:
    """Check whether Iris can concatenate equal TP4/TP8 hidden shards.

    Args:
        world_size: Number of ranks contributing equal-width shards.
        num_tokens: Rows in each rank-local shard.
        local_hidden_size: Elements in each rank-local row.
        dtype: Element type shared by every rank.
        max_bytes: Capacity of the symmetric input and local output buffers.

    Returns:
        Whether the CDNA4 Gluon kernel supports the collective geometry.
    """
    total_numel = num_tokens * local_hidden_size * world_size
    element_bytes = dtype.itemsize
    elements_per_word = 8 // element_bytes
    return (
        current_platform().is_cdna4
        and world_size in (4, 8)
        and num_tokens > 0
        and local_hidden_size > 0
        and dtype in _PRODUCER_DIRECT_GL_DTYPES
        and 8 % element_bytes == 0
        and local_hidden_size % elements_per_word == 0
        and total_numel * element_bytes <= max_bytes
    )


@gluon.jit(do_not_specialize=["LOCAL_WORDS"])
def iris_pull_allgather_columns(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,
    heaps,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WORDS,
    LOCAL_HIDDEN_WORDS: gl.constexpr,
    TOTAL_HIDDEN_WORDS: gl.constexpr,
    BLOCK_WORDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    INPUT_PUBLISHED: gl.constexpr,
):
    """Concatenate equal feature shards from four ranks into each local output.

    Copy input_ptr into input_sym_ptr unless INPUT_PUBLISHED says the producer
    already wrote there. Read each peer's shard and write full rows to output_ptr.
    Widths and element counts are measured in 64-bit words. Peer counters guard
    input publication and completion before the symmetric storage is reused.
    """
    block_id = gl.program_id(0)
    num_tiles = gl.cdiv(LOCAL_WORDS, BLOCK_WORDS)
    local_heap = _iris_heap_base(RANK, *heaps)
    layout: gl.constexpr = gl.BlockedLayout(
        [WORDS_PER_LANE], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    words = gl.arange(0, BLOCK_WORDS, layout=layout)
    input_words = tl.cast(input_ptr, gl.pointer_type(gl.uint64))
    symmetric_words = tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64))
    output_words = tl.cast(output_ptr, gl.pointer_type(gl.uint64))
    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap

    if not INPUT_PUBLISHED:
        tile_id = block_id
        while tile_id < num_tiles:
            local_word = tile_id * BLOCK_WORDS + words
            mask = local_word < LOCAL_WORDS
            row = local_word // LOCAL_HIDDEN_WORDS
            column = local_word % LOCAL_HIDDEN_WORDS
            symmetric_offset = (
                row * TOTAL_HIDDEN_WORDS + RANK * LOCAL_HIDDEN_WORDS + column
            )
            packed = gl.load(input_words + local_word, mask=mask, other=0)
            gl.amd.cdna4.buffer_store(
                packed,
                symmetric_words,
                symmetric_offset.to(gl.int32),
                mask=mask,
                cache=".wt",
            )
            tile_id += gl.num_programs(0)

    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    ready_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        ready_epoch,
        local_heap,
        *heaps,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
    )
    gl.atomic_add(epoch_ptr, 0, sem="acquire", scope="sys")

    tile_id = block_id
    while tile_id < num_tiles:
        local_word = tile_id * BLOCK_WORDS + words
        mask = local_word < LOCAL_WORDS
        row = local_word // LOCAL_HIDDEN_WORDS
        column = local_word % LOCAL_HIDDEN_WORDS
        for peer in gl.static_range(0, WORLD_SIZE):
            peer_heap = _iris_heap_base(peer, *heaps)
            peer_input = tl.cast(
                peer_heap + input_heap_offset, gl.pointer_type(gl.uint64)
            )
            output_offset = (
                row * TOTAL_HIDDEN_WORDS + peer * LOCAL_HIDDEN_WORDS + column
            )
            packed = gl.load(
                peer_input + output_offset,
                mask=mask,
                other=0,
                cache_modifier=".cg",
            )
            gl.amd.cdna4.buffer_store(
                packed,
                output_words,
                output_offset.to(gl.int32),
                mask=mask,
            )
        tile_id += gl.num_programs(0)

    done_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        done_epoch,
        local_heap,
        *heaps,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
    )


@gluon.jit(do_not_specialize=["rows"])
def iris_pull_allgather_rows(
    source,
    output,
    ready_flags,
    heaps,
    rows,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    PRODUCER_WRITES_THROUGH: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Gather consecutive BF16 row shards into every rank's output.

    Each source holds ceil(rows / WORLD_SIZE) rows of WIDTH features. Copy valid rows in
    rank order into output and mask padding beyond rows. Publish producer writes
    through ready_flags, then wait for all peer reads before source is reused.
    PRODUCER_WRITES_THROUGH selects the matching input-publication protocol.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    program = gl.program_id(0)
    peers = _iris_peer_pointers(source, heaps, RANK)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    if PRODUCER_WRITES_THROUGH:
        _iris_drain_subgroup_vmem()
        gl.barrier()
        _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, False)
    else:
        _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, True)
    local_rows = gl.cdiv(rows, WORLD_SIZE)
    local_elements = local_rows * WIDTH
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    for tile in range(program, gl.cdiv(local_elements, BLOCK), gl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        for step in gl.static_range(WORLD_SIZE):
            destination = ((RANK + step) % WORLD_SIZE) * local_elements + offsets
            mask = (offsets < local_elements) & (destination < rows * WIDTH)
            values = gl.amd.cdna4.buffer_load(
                peers[(RANK + step) % WORLD_SIZE], offsets, mask, 0, cache=".cg"
            )
            gl.amd.cdna4.buffer_store(values, output, destination, mask)
    # The next producer may reuse the owner scratch.
    _iris_drain_subgroup_vmem()
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


@gluon.jit
def iris_fused_pull_allgather_rmsnorm_mxfp4(
    local_input,
    symmetric_input,
    weight,
    packed_output,
    scale_output,
    ready_flags,
    heaps,
    rank: gl.constexpr,
    local_width: gl.constexpr,
    total_width: gl.constexpr,
    eps: gl.constexpr,
    weight_offset: gl.constexpr,
    scale_stride_linear,
    scale_stride_mblock,
    block_k: gl.constexpr,
    input_published: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Gather BF16 feature shards, apply RMSNorm, and quantize to MXFP4.

    Widths, RMSNorm epsilon/weight offset, and output-scale strides are supplied
    by the caller. Scales use the CDNA4 layout with 32-row and eight-group tiles.
    Write packed_output and scale_output directly, retaining FP32 normalized
    values until quantization. Peer counters protect symmetric_input reuse.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    row = gl.program_id(0)
    local_heap = _iris_group_heap_base(rank, heaps)
    input_offset = tl.cast(symmetric_input, gl.uint64) - local_heap

    layout: gl.constexpr = scaled_downcast_layout(1, block_k, num_warps)
    offsets = gl.arange(0, block_k, layout=gl.SliceLayout(0, layout))
    valid = offsets < total_width
    if not input_published:
        local = gl.load(
            local_input + row * local_width + offsets,
            mask=offsets < local_width,
            other=0.0,
        )
        gl.store(
            symmetric_input + row * total_width + rank * local_width + offsets,
            local,
            mask=offsets < local_width,
        )

    epoch_ptr = ready_flags + row * WORLD_SIZE + rank
    ready_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_peers(ready_flags, row, ready_epoch, heaps, rank, num_warps)

    values = gl.zeros((block_k,), gl.float32, layout=offsets.type.layout)
    for peer in gl.static_range(0, WORLD_SIZE):
        peer_heap = _iris_group_heap_base(peer, heaps)
        peer_input = tl.cast(
            peer_heap + input_offset,
            gl.pointer_type(gl.bfloat16),
        )
        peer_mask = (
            valid
            & (offsets >= peer * local_width)
            & (offsets < (peer + 1) * local_width)
        )
        values += gl.load(
            peer_input + row * total_width + offsets,
            mask=peer_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(gl.float32)

    variance = gl.sum(values * values, axis=0) / total_width
    values *= gl.rsqrt(variance + eps)
    norm_weight = gl.load(weight + offsets, mask=valid, other=0.0).to(gl.float32)
    values *= (norm_weight + weight_offset).to(gl.bfloat16).to(gl.float32)

    values = values.reshape((1, block_k))
    packed, scale_byte = _mxfp4_quantize_tile_in_layout(values)
    packed_offsets = gl.arange(
        0,
        block_k // 2,
        layout=gl.SliceLayout(0, packed.type.layout),
    )
    packed_rows = gl.arange(
        0,
        1,
        layout=gl.SliceLayout(1, packed.type.layout),
    )
    gl.store(
        packed_output
        + (row + packed_rows[:, None]) * (total_width // 2)
        + packed_offsets[None, :],
        packed,
        mask=packed_offsets[None, :] < total_width // 2,
    )

    scale_groups = gl.arange(
        0,
        block_k // 32,
        layout=gl.SliceLayout(0, scale_byte.type.layout),
    )
    _mxfp4_store_cdna4_scale(
        scale_output,
        scale_byte,
        row
        + gl.arange(
            0,
            1,
            layout=gl.SliceLayout(1, scale_byte.type.layout),
        )[:, None],
        scale_groups[None, :],
        scale_stride_linear,
        scale_stride_mblock,
        scale_groups[None, :] < total_width // 32,
        M_SWIZZLE=32,
        K_SWIZZLE=8,
    )

    done_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_peers(ready_flags, row, done_epoch, heaps, rank, num_warps)
