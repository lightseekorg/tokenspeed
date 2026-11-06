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

"""Project K3 decode rows' up-projection columns per rank and gather them.

The reduced routed latent is replicated on every rank after the MoE
all-reduce, so each rank normalizes it, projects only its column block of the
replicated ``[7168, 3584]`` up-projection (one eighth of the weight bytes),
adds the residual and the reduced shared-expert output, and pushes the block
through the K3 Lamport mailbox. One reducer program per row then waits for
every rank's block.

Past four rows the GEMV senders' per-row work outgrows the weight read, so
MFMA senders project 16 x 16 tiles instead and push each as one contiguous
block; reducers then gather 1024 elements of every slot at a time.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_lamport_push,
    _iris_sanitize_lamport_bf16,
    _iris_subgroup_any,
)

# Each row's block starts on its own pair of 512-element Lamport tiles.
K3_GATHER_ROW_STRIDE = 1024


@gluon.jit
def _k3moe_gather_send(
    latent_ptr,
    norm_weight_ptr,
    weight_ptr,
    prefix_ptr,
    shared_ptr,
    epochs,
    regions,
    program,
    M,
    latent_stride,
    prefix_stride,
    shared_stride,
    eps,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    ROW_STRIDE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    HAS_NORM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Project OUTPUTS columns of every row into slot RANK of every mailbox.

    The program stages its normalized latent rows in LDS once, so the latent
    is read from L2 once per program rather than once per output subgroup.
    """
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT
    VECTOR: gl.constexpr = BLOCK_K // (64 * WAVES_PER_OUTPUT)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M, 1, VECTOR],
        [1, 1, 64],
        [1, OUTPUTS, WAVES_PER_OUTPUT],
        [2, 1, 0],
    )
    w_layout: gl.constexpr = gl.SliceLayout(0, layout)
    x_layout: gl.constexpr = gl.SliceLayout(1, layout)
    first = program * OUTPUTS
    outputs = first + gl.arange(0, OUTPUTS, layout=gl.SliceLayout(1, w_layout))
    w_columns = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, w_layout))
    w_offsets = (outputs[:, None] * K + w_columns[None, :]).to(gl.int32)
    STEPS: gl.constexpr = K // BLOCK_K
    # One subgroup stages each K step of the latent, eight elements a lane.
    gl.static_assert(STEPS <= NUM_WARPS and BLOCK_K == 8 * 64)

    # Issue all weight loads first so the cold loads overlap the latent staging.
    weights = ()
    for step in gl.static_range(STEPS):
        weights += (
            gl.amd.cdna4.buffer_load(
                weight_ptr, w_offsets + step * BLOCK_K, cache=".cg"
            ),
        )

    # The epilogue operands and stages load behind the weights.
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, OUTPUTS], [64, 1], [NUM_WARPS, 1], [1, 0]
    )
    out_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
    out_columns = first + gl.arange(0, OUTPUTS, layout=gl.SliceLayout(0, store_layout))
    valid = (out_rows < M)[:, None] & (out_columns >= 0)[None, :]
    columns = RANK * LOCAL_WIDTH + out_columns
    prefix = gl.amd.cdna4.buffer_load(
        prefix_ptr,
        (gl.multiple_of(out_rows * prefix_stride, 8)[:, None] + columns[None, :]).to(
            gl.int32
        ),
        mask=valid,
        other=0.0,
    )
    shared = gl.amd.cdna4.buffer_load(
        shared_ptr,
        (gl.multiple_of(out_rows * shared_stride, 8)[:, None] + columns[None, :]).to(
            gl.int32
        ),
        mask=valid,
        other=0.0,
    )
    elements = out_rows[:, None] * ROW_STRIDE + out_columns[None, :]
    # OUTPUTS divides the 512-element tile, so one store never spans two stages.
    stage = gl.load(epochs + elements // 512, mask=valid, other=0).to(gl.uint32) % 3

    # Stage the BF16 latent rows in LDS once per program; subgroup s loads K step s.
    latent_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 1, 64], [NUM_WARPS, 1, 1], [2, 1, 0]
    )
    steps = gl.arange(
        0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, latent_layout))
    )
    latent_rows = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, latent_layout))
    )
    latent_columns = gl.arange(
        0, BLOCK_K, layout=gl.SliceLayout(0, gl.SliceLayout(1, latent_layout))
    )
    k_index = steps[:, None, None] * BLOCK_K + latent_columns[None, None, :]
    latent_valid = (steps < STEPS)[:, None, None] & (latent_rows < M)[None, :, None]
    # Strides are multiples of eight elements, so row loads stay 16-byte vectors.
    latent = gl.amd.cdna4.buffer_load(
        latent_ptr,
        (gl.multiple_of(latent_rows * latent_stride, 8)[None, :, None] + k_index).to(
            gl.int32
        ),
        mask=latent_valid,
        other=0.0,
    )
    if HAS_NORM:
        # Triton RMSNorm's FP32 ops and one BF16 rounding; only the sum order differs.
        x = latent.to(gl.float32)
        inverse_rms = gl.rsqrt(gl.sum(gl.sum(x * x, axis=2), axis=0) / K + eps)
        weight_steps = gl.arange(
            0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(1, latent_layout))
        )
        norm_weight = gl.amd.cdna4.buffer_load(
            norm_weight_ptr,
            (weight_steps[:, None] * BLOCK_K + latent_columns[None, :]).to(gl.int32),
            mask=(weight_steps < STEPS)[:, None],
            other=0.0,
        ).to(gl.float32)
        latent = ((x * inverse_rms[None, :, None]) * norm_weight[:, None, :]).to(
            gl.bfloat16
        )
    staged = gl.allocate_shared_memory(
        gl.bfloat16,
        [NUM_WARPS * BLOCK_M, BLOCK_K],
        gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
    )
    staged.store(gl.reshape(latent, [NUM_WARPS * BLOCK_M, BLOCK_K]))
    gl.barrier()

    accumulator = gl.zeros((BLOCK_M, OUTPUTS, BLOCK_K), gl.float32, layout=layout)
    for step in gl.static_range(STEPS):
        x = staged.slice(step * BLOCK_M, BLOCK_M, dim=0).load(x_layout).to(gl.float32)
        accumulator += x[:, None, :] * weights[step].to(gl.float32)[None, :, :]
    sums = gl.sum(accumulator, axis=2)

    # Add residual then shared output before one BF16 rounding, as the add3 epilogue.
    sums = gl.convert_layout(sums, store_layout)
    sums += prefix.to(gl.float32)
    sums += shared.to(gl.float32)
    values = _iris_sanitize_lamport_bf16(sums.to(gl.bfloat16))

    offsets = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS)).to(gl.int32)
        + RANK * MAX_ELEMENTS
        + elements
    )
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (values,)
    _iris_lamport_push(
        published, regions, 0, offsets, valid, RANK, WORLD_SIZE, PEER_SIGNAL_STORE_CACHE
    )
    # Write through: the reducer may poll the own slot from another XCD.
    own = gl.multiple_of(regions[RANK].to(gl.pointer_type(gl.bfloat16)), 16)
    gl.amd.cdna4.buffer_store(values, own, offsets, mask=valid, cache=".wt")


@gluon.jit
def _k3moe_mfma_send(
    latent_ptr,
    norm_weight_ptr,
    weight_ptr,
    prefix_ptr,
    shared_ptr,
    epochs,
    regions,
    program,
    M,
    latent_stride,
    prefix_stride,
    shared_stride,
    eps,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    CHUNK: gl.constexpr,
    HAS_NORM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Project a 16x16 tile of rows and columns into slot RANK of every mailbox.

    Subgroup s owns the CHUNK-wide K slices s, s + NUM_WARPS, ... of the tile:
    it copies their weights to LDS, loads their latent and moves it into MFMA
    operand registers, and runs one MFMA chain over them; the subgroups'
    partial tiles are then summed. A tile's whole weight block is in flight
    at once.
    """
    gl.static_assert(K % (NUM_WARPS * CHUNK) == 0)
    STEPS: gl.constexpr = K // (NUM_WARPS * CHUNK)
    COLUMN_TILES: gl.constexpr = LOCAL_WIDTH // 16
    first_row = (program // COLUMN_TILES) * 16
    first_column = (program % COLUMN_TILES) * 16
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)

    # L2 operands load before the weights, as loads complete in order; norm via LDS.
    if HAS_NORM:
        norm_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 8], [8, 1, 8], [1, 1, NUM_WARPS], [2, 1, 0]
        )
        n_slices = gl.arange(
            0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, norm_layout))
        )
        n_k = gl.arange(
            0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(1, norm_layout))
        )
        n_offsets = (n_slices[:, None, None] * CHUNK + n_k[None, None, :]).to(gl.int32)
        staged_norm = gl.allocate_shared_memory(
            norm_weight_ptr.dtype.element_ty,
            [STEPS, NUM_WARPS, 1, CHUNK],
            gl.SwizzledSharedLayout(1, 1, 1, order=[2, 1, 0]),
        )
        for step in gl.static_range(STEPS):
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                staged_norm.index(step),
                norm_weight_ptr,
                n_offsets + step * NUM_WARPS * CHUNK,
            )
        gl.amd.cdna4.async_copy.commit_group()
    # Every lane stores whole 32-bit words, so the reducers' empty test stays exact.
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [16, 4], [NUM_WARPS, 1], [1, 0]
    )
    tile_rows = gl.arange(0, 16, layout=gl.SliceLayout(1, store_layout))
    tile_columns = gl.arange(0, 16, layout=gl.SliceLayout(0, store_layout))
    out_rows = first_row + tile_rows
    out_columns = first_column + tile_columns
    valid = (out_rows < M)[:, None] & (out_columns >= 0)[None, :]
    columns = RANK * LOCAL_WIDTH + out_columns
    prefix = gl.amd.cdna4.buffer_load(
        prefix_ptr,
        (gl.multiple_of(out_rows * prefix_stride, 8)[:, None] + columns[None, :]).to(
            gl.int32
        ),
        mask=valid,
        other=0.0,
    )
    shared = gl.amd.cdna4.buffer_load(
        shared_ptr,
        (gl.multiple_of(out_rows * shared_stride, 8)[:, None] + columns[None, :]).to(
            gl.int32
        ),
        mask=valid,
        other=0.0,
    )
    # One contiguous 256-element tile: 512-byte stores per peer, in one Lamport tile.
    block = program * 256
    elements = block + tile_rows[:, None] * 16 + tile_columns[None, :]
    stage = gl.load(epochs + block // 512).to(gl.uint32) % 3

    # Load latent as whole 128-byte lines and move it to the operand layout through LDS.
    latent_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 8, 8], [NUM_WARPS, 1, 1], [2, 1, 0]
    )
    a_slices = gl.arange(
        0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, latent_layout))
    )
    a_rows = first_row + gl.arange(
        0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, latent_layout))
    )
    a_k = gl.arange(
        0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(1, latent_layout))
    )
    k_offsets = a_slices[:, None, None] * CHUNK + a_k[None, None, :]
    # Rows past M repeat row M - 1 instead of masking; rows never mix or get stored.
    a_rows = gl.minimum(a_rows, M - 1)
    # Strides are multiples of eight elements, so row loads stay 16-byte vectors.
    a_offsets = (
        gl.multiple_of(a_rows * latent_stride, 8)[None, :, None] + k_offsets
    ).to(gl.int32)
    latent = ()
    for step in gl.static_range(STEPS):
        latent += (
            gl.amd.cdna4.buffer_load(latent_ptr, a_offsets + step * NUM_WARPS * CHUNK),
        )
    operands = ()
    for step in gl.static_range(STEPS):
        operands += (gl.convert_layout(latent[step], dot_a),)
    latent = operands
    # Copy the weight block to LDS in whole 128-byte lines before the norm runs.
    gl.static_assert(CHUNK == 64)
    copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8, 1], [1, 8, 8], [NUM_WARPS, 1, 1], [1, 2, 0]
    )
    b_slices = gl.arange(
        0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, copy_layout))
    )
    b_k = gl.arange(0, CHUNK, layout=gl.SliceLayout(0, gl.SliceLayout(2, copy_layout)))
    b_columns = first_column + gl.arange(
        0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, copy_layout))
    )
    b_offsets = (
        b_slices[:, None, None] * CHUNK
        + b_k[None, :, None]
        + b_columns[None, None, :] * K
    ).to(gl.int32)
    staged_weights = ()
    for step in gl.static_range(STEPS):
        staged_weights += (
            gl.allocate_shared_memory(
                weight_ptr.dtype.element_ty,
                [NUM_WARPS, CHUNK, 16],
                gl.SwizzledSharedLayout(8, 1, 8, order=[1, 2, 0]),
            ),
        )
    for step in gl.static_range(STEPS):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            staged_weights[step],
            weight_ptr,
            b_offsets + step * NUM_WARPS * CHUNK,
            cache_modifier=".cg",
        )
        gl.amd.cdna4.async_copy.commit_group()

    if HAS_NORM:
        # Triton RMSNorm's FP32 ops and one BF16 rounding; only the sum order differs.
        gl.amd.cdna4.async_copy.wait_group(STEPS)
        gl.barrier()
        squares = gl.zeros((NUM_WARPS, 16), gl.float32, gl.SliceLayout(2, dot_a))
        for step in gl.static_range(STEPS):
            x = latent[step].to(gl.float32)
            squares += gl.sum(x * x, axis=2)
        inverse_rms = gl.rsqrt(gl.sum(squares, axis=0) / K + eps)
        normalized = ()
        for step in gl.static_range(STEPS):
            x = latent[step].to(gl.float32)
            normalized += (
                (
                    (x * inverse_rms[None, :, None])
                    * gl.amd.cdna4.async_copy.load_shared_relaxed(
                        staged_norm.index(step), dot_a
                    ).to(gl.float32)
                ).to(gl.bfloat16),
            )
        latent = normalized

    accumulator = gl.zeros((NUM_WARPS, 16, 16), gl.float32, mfma)
    for step in gl.static_range(STEPS):
        # Subgroup s reads only the slice it copied itself.
        gl.amd.cdna4.async_copy.wait_group(STEPS - 1 - step)
        weights = gl.amd.cdna4.async_copy.load_shared_relaxed(
            staged_weights[step], dot_b
        )
        accumulator = gl.amd.cdna4.mfma(latent[step], weights, accumulator)
    sums = gl.convert_layout(gl.sum(accumulator, axis=0), store_layout)

    # Add residual then shared output before one BF16 rounding, as the add3 epilogue.
    sums += prefix.to(gl.float32)
    sums += shared.to(gl.float32)
    values = _iris_sanitize_lamport_bf16(sums.to(gl.bfloat16))

    offsets = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS)).to(gl.int32)
        + RANK * MAX_ELEMENTS
        + elements
    )
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (values,)
    _iris_lamport_push(
        published, regions, 0, offsets, valid, RANK, WORLD_SIZE, PEER_SIGNAL_STORE_CACHE
    )
    # Write through: the reducer may poll the own slot from another XCD.
    own = gl.multiple_of(regions[RANK].to(gl.pointer_type(gl.bfloat16)), 16)
    gl.amd.cdna4.buffer_store(values, own, offsets, mask=valid, cache=".wt")


@gluon.jit
def _k3moe_gather_reduce(
    region,
    epochs,
    out_ptr,
    row,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    ROW_STRIDE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Wait for, copy out and clear every slot's block of ``row``."""
    gl.static_assert(WORLD_SIZE == 8)
    # One subgroup polls each slot.
    gl.static_assert(NUM_WARPS == WORLD_SIZE)
    gl.static_assert(ROW_STRIDE == 1024 and LOCAL_WIDTH <= ROW_STRIDE)
    SLOT_WORDS: gl.constexpr = ROW_STRIDE // 2
    WORDS: gl.constexpr = SLOT_WORDS // 64
    # Thread t polls WORDS words of slot t // 64, so a retry votes with one lane mask.
    layout: gl.constexpr = gl.BlockedLayout([1, WORDS], [64, 1], [NUM_WARPS, 1], [1, 0])
    thread = gl.arange(0, 64 * NUM_WARPS, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, WORDS, layout=gl.SliceLayout(0, layout))
    slot = thread // 64
    first = (thread % 64) * WORDS
    valid = (first < LOCAL_WIDTH // 2)[:, None] & (word >= 0)[None, :]
    # The block's two tiles may sit on different stages.
    tile = (row * ROW_STRIDE) // 512
    low_stage = gl.load(epochs + tile).to(gl.uint32) % 3
    high_stage = gl.load(epochs + tile + 1).to(gl.uint32) % 3
    stage = gl.where(first < 256, low_stage, high_stage)
    base = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS // 2)).to(gl.int32)
        + slot * (MAX_ELEMENTS // 2)
        + row * SLOT_WORDS
        + first
    )
    mailbox = gl.multiple_of(base, WORDS)[:, None] + word[None, :]
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    # Payloads exclude -0.0 and words are whole, so a high half of 0x8000 means empty.
    payload = gl.amd.cdna4.buffer_load(
        words_ptr, mailbox, mask=valid, other=0, cache=".cv"
    )
    while _iris_subgroup_any(
        gl.max((valid & ((payload >> 16) == 0x8000)).to(gl.int32), 1)
    ):
        payload = gl.amd.cdna4.buffer_load(
            words_ptr, mailbox, mask=valid, other=0, cache=".cv"
        )
    gl.amd.cdna4.buffer_store(
        gl.full_like(payload, 0x80000000), words_ptr, mailbox, mask=valid, cache=".wt"
    )
    out_words = gl.multiple_of(out_ptr.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    out_base = row * (WORLD_SIZE * LOCAL_WIDTH // 2) + slot * (LOCAL_WIDTH // 2) + first
    gl.amd.cdna4.buffer_store(
        payload,
        out_words,
        (gl.multiple_of(out_base, WORDS)[:, None] + word[None, :]).to(gl.int32),
        mask=valid,
    )
    # All subgroups have left the poll, so local senders have read this row's stages.
    gl.barrier()
    gl.store(epochs + tile, ((low_stage + 1) % 3).to(gl.int32))
    gl.store(epochs + tile + 1, ((high_stage + 1) % 3).to(gl.int32))


@gluon.jit
def _k3moe_tile_gather_reduce(
    region,
    epochs,
    out_ptr,
    reducer,
    M,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Wait for, copy out and clear every slot's 1024 elements of ``reducer``.

    A slot holds each 16-row tile of its rows as LOCAL_WIDTH // 16 blocks of
    16 x 16 elements, so block elements ``[reducer * 1024, reducer * 1024 +
    1024)`` are four column tiles of one row tile.
    """
    gl.static_assert(WORLD_SIZE == 8)
    # One subgroup polls each slot.
    gl.static_assert(NUM_WARPS == WORLD_SIZE)
    gl.static_assert((16 * LOCAL_WIDTH) % 1024 == 0)
    REDUCERS_PER_ROW_TILE: gl.constexpr = 16 * LOCAL_WIDTH // 1024
    # Thread t polls 8 words of slot t // 64, so a retry votes with one lane mask.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [NUM_WARPS, 1], [1, 0])
    thread = gl.arange(0, 64 * NUM_WARPS, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    slot = thread // 64
    lane = thread % 64
    within = (reducer % REDUCERS_PER_ROW_TILE) * 1024 + lane * 16
    row = (reducer // REDUCERS_PER_ROW_TILE) * 16 + (within % 256) // 16
    column = slot * LOCAL_WIDTH + (within // 256) * 16
    valid = (row < M)[:, None] & (word >= 0)[None, :]
    # The block's two tiles may sit on different stages.
    tile = reducer * 2
    low_stage = gl.load(epochs + tile).to(gl.uint32) % 3
    high_stage = gl.load(epochs + tile + 1).to(gl.uint32) % 3
    stage = gl.where(lane < 32, low_stage, high_stage)
    base = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS // 2)).to(gl.int32)
        + slot * (MAX_ELEMENTS // 2)
        + reducer * 512
        + lane * 8
    )
    mailbox = gl.multiple_of(base, 8)[:, None] + word[None, :]
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    # Payloads exclude -0.0 and words are whole, so a high half of 0x8000 means empty.
    payload = gl.amd.cdna4.buffer_load(
        words_ptr, mailbox, mask=valid, other=0, cache=".cv"
    )
    while _iris_subgroup_any(
        gl.max((valid & ((payload >> 16) == 0x8000)).to(gl.int32), 1)
    ):
        payload = gl.amd.cdna4.buffer_load(
            words_ptr, mailbox, mask=valid, other=0, cache=".cv"
        )
    gl.amd.cdna4.buffer_store(
        gl.full_like(payload, 0x80000000), words_ptr, mailbox, mask=valid, cache=".wt"
    )
    out_words = gl.multiple_of(out_ptr.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    out_base = (row * (WORLD_SIZE * LOCAL_WIDTH) + column) // 2
    gl.amd.cdna4.buffer_store(
        payload,
        out_words,
        (gl.multiple_of(out_base, 8)[:, None] + word[None, :]).to(gl.int32),
        mask=valid,
    )
    # All subgroups have left the poll, so local senders have read this block's stage.
    gl.barrier()
    gl.store(epochs + tile, ((low_stage + 1) % 3).to(gl.int32))
    gl.store(epochs + tile + 1, ((high_stage + 1) % 3).to(gl.int32))


@gluon.jit(do_not_specialize=("M", "latent_stride", "prefix_stride", "shared_stride"))
def iris_k3moe_gemv_lamport_gather(
    latent_ptr,
    norm_weight_ptr,
    weight_ptr,
    prefix_ptr,
    shared_ptr,
    out_ptr,
    region,
    epochs,
    regions,
    M,
    latent_stride,
    prefix_stride,
    shared_stride,
    eps,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    ROW_STRIDE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    HAS_NORM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Normalize, project, add and gather M <= BLOCK_M decode rows of the K3 tail.

    ``weight_ptr`` holds this rank's LOCAL_WIDTH rows of the ``[N, K]``
    up-projection. Grid ``(LOCAL_WIDTH // OUTPUTS + M,)`` with OUTPUTS =
    NUM_WARPS // WAVES_PER_OUTPUT: each sender projects OUTPUTS columns of
    every row and stores row m's block at elements
    ``[m * ROW_STRIDE, m * ROW_STRIDE + LOCAL_WIDTH)`` of slot RANK in every
    rank's Lamport mailbox, its own included. The last M programs, one per
    row, wait for all slots of their row, write it to ``out_ptr`` (contiguous
    ``[M, N]``), clear the slots and advance the row's two tile epochs. The
    mailbox and epochs are the routed/shared all-reduce's, used in the same
    stream order on every rank.
    """
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT
    gl.static_assert(OUTPUTS % 2 == 0 and 512 % OUTPUTS == 0)
    gl.static_assert(LOCAL_WIDTH % OUTPUTS == 0 and K % BLOCK_K == 0)
    gl.static_assert(BLOCK_K % (64 * WAVES_PER_OUTPUT) == 0)
    gl.static_assert(BLOCK_M * ROW_STRIDE <= MAX_ELEMENTS)
    program = gl.program_id(0)
    # Reducers are the last programs, so every sender dispatches before they spin.
    if program >= LOCAL_WIDTH // OUTPUTS:
        _k3moe_gather_reduce(
            region,
            epochs,
            out_ptr,
            program - LOCAL_WIDTH // OUTPUTS,
            WORLD_SIZE,
            LOCAL_WIDTH,
            ROW_STRIDE,
            MAX_ELEMENTS,
            NUM_WARPS,
        )
    else:
        _k3moe_gather_send(
            latent_ptr,
            norm_weight_ptr,
            weight_ptr,
            prefix_ptr,
            shared_ptr,
            epochs,
            regions,
            program,
            M,
            latent_stride,
            prefix_stride,
            shared_stride,
            eps,
            RANK,
            WORLD_SIZE,
            K,
            LOCAL_WIDTH,
            ROW_STRIDE,
            MAX_ELEMENTS,
            BLOCK_M,
            BLOCK_K,
            WAVES_PER_OUTPUT,
            HAS_NORM,
            NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE,
        )


@gluon.jit(do_not_specialize=("M", "latent_stride", "prefix_stride", "shared_stride"))
def iris_k3moe_mfma_lamport_gather(
    latent_ptr,
    norm_weight_ptr,
    weight_ptr,
    prefix_ptr,
    shared_ptr,
    out_ptr,
    region,
    epochs,
    regions,
    M,
    latent_stride,
    prefix_stride,
    shared_stride,
    eps,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    CHUNK: gl.constexpr,
    HAS_NORM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Normalize, project, add and gather M decode rows of the K3 tail with MFMA.

    The contract of ``iris_k3moe_gemv_lamport_gather`` for any M with
    ``cdiv(M, 16) * 16 * LOCAL_WIDTH <= MAX_ELEMENTS``, with grid
    ``(cdiv(M, 16) * LOCAL_WIDTH // 16 * 5 // 4,)``: each of the first
    ``cdiv(M, 16) * LOCAL_WIDTH // 16`` programs projects one 16-row,
    16-column tile and stores it as a contiguous 256-element block of its
    slot (tile ``t`` at ``t * 256``), and each of the rest gathers 1024
    elements of every slot.
    """
    gl.static_assert(LOCAL_WIDTH % 32 == 0)
    senders = (LOCAL_WIDTH // 16) * ((M + 15) // 16)
    program = gl.program_id(0)
    # Reducers are the last programs, so every sender dispatches before they spin.
    if program >= senders:
        _k3moe_tile_gather_reduce(
            region,
            epochs,
            out_ptr,
            program - senders,
            M,
            WORLD_SIZE,
            LOCAL_WIDTH,
            MAX_ELEMENTS,
            NUM_WARPS,
        )
    else:
        _k3moe_mfma_send(
            latent_ptr,
            norm_weight_ptr,
            weight_ptr,
            prefix_ptr,
            shared_ptr,
            epochs,
            regions,
            program,
            M,
            latent_stride,
            prefix_stride,
            shared_stride,
            eps,
            RANK,
            WORLD_SIZE,
            K,
            LOCAL_WIDTH,
            MAX_ELEMENTS,
            CHUNK,
            HAS_NORM,
            NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE,
        )
