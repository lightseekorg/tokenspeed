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

"""Fuse Muse's single-row post-expert projection, column gather and layer boundary."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_gather import _iris_tile_stage
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_lamport_push,
    _iris_sanitize_lamport_bf16,
    _iris_subgroup_any,
)


@gluon.jit
def _iris_muse_moetail_project(
    x_ptr,
    weight_ptr,
    program,
    K: gl.constexpr,
    OUTPUTS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
):
    """Project OUTPUTS consecutive outputs, WAVES_PER_OUTPUT subgroups per output.

    Each output keeps the reduction of gluon_rowcta_bf16_gemv_gfx950 with the
    same (BLOCK_K, WAVES_PER_OUTPUT) geometry: per-lane FP32 chunks accumulated in
    K order, then one sum over the output's subgroups.
    """
    layout: gl.constexpr = gl.BlockedLayout(
        [1, BLOCK_K // (64 * WAVES_PER_OUTPUT)],
        [1, 64],
        [OUTPUTS, WAVES_PER_OUTPUT],
        [1, 0],
    )
    outputs = program * OUTPUTS + gl.arange(
        0, OUTPUTS, layout=gl.SliceLayout(1, layout)
    )
    offsets = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, layout))
    accumulator = gl.zeros((OUTPUTS, BLOCK_K), gl.float32, layout=layout)
    for start in range(0, K, BLOCK_K):
        columns = start + offsets
        x = gl.amd.cdna4.buffer_load(x_ptr, columns.to(gl.int32)).to(gl.float32)
        weight = gl.amd.cdna4.buffer_load(
            weight_ptr,
            (outputs[:, None] * K + columns[None, :]).to(gl.int32),
            cache=".cg",
        ).to(gl.float32)
        accumulator += x[None, :] * weight
    return gl.sum(accumulator, axis=1)


@gluon.jit
def _iris_muse_moetail_send(
    x_ptr,
    weight_ptr,
    epochs,
    regions,
    program,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    OUTPUTS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Project OUTPUTS outputs and store them in slot RANK of every rank's mailbox."""
    layout: gl.constexpr = gl.BlockedLayout([OUTPUTS], [64], [NUM_WARPS], [0])
    offsets = program * OUTPUTS + gl.arange(0, OUTPUTS, layout=layout)
    # OUTPUTS divides the 512-element tile, so a program's outputs share one stage.
    stage = gl.load(epochs + program * OUTPUTS // 512).to(gl.uint32) % 3
    values = _iris_muse_moetail_project(
        x_ptr, weight_ptr, program, K, OUTPUTS, BLOCK_K, WAVES_PER_OUTPUT
    )
    values = _iris_sanitize_lamport_bf16(
        gl.convert_layout(values.to(gl.bfloat16), layout)
    )
    offset = stage * (WORLD_SIZE * MAX_ELEMENTS) + RANK * MAX_ELEMENTS
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (values,)
    # Every program owns OUTPUTS whole outputs; nothing is masked.
    whole = offsets >= 0
    _iris_lamport_push(
        published,
        regions,
        offset,
        offsets,
        whole,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )
    # Write through: the reducer may poll the own slot from another XCD. Like
    # the peer bases, the hinted own base keeps every 32-bit word one store at
    # any specialization; split halves could expose a stale low half.
    own = gl.multiple_of(regions[RANK].to(gl.pointer_type(gl.bfloat16)), 16)
    gl.amd.cdna4.buffer_store(values, own + offset, offsets, cache=".wt")


@gluon.jit
def _iris_muse_moetail_reduce(
    region,
    epochs,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_bf16_ptr,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    HAS_PRE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Wait for every slot, clear it, then update the residual and normalize.

    The norms repeat _post_norm_residual_pre_norm_kernel's operations in its
    layout (eight columns per thread, 16 subgroups) and order.
    """
    WIDTH: gl.constexpr = LOCAL_WIDTH * WORLD_SIZE
    THREADS: gl.constexpr = 64 * NUM_WARPS
    WORDS: gl.constexpr = WIDTH // 2 // THREADS
    TILES: gl.constexpr = LOCAL_WIDTH // 512
    gl.static_assert(WORDS * 2 * THREADS == WIDTH)
    layout: gl.constexpr = gl.BlockedLayout([2 * WORDS], [64], [NUM_WARPS], [0])
    columns = gl.arange(0, WIDTH, layout=layout)
    # Read the boundary operands while peers are still pushing.
    stream = gl.amd.cdna4.buffer_load(stream_ptr, columns)
    alpha = gl.amd.cdna4.buffer_load(alpha_ptr, columns)
    beta = gl.amd.cdna4.buffer_load(beta_ptr, columns)
    post_weight = gl.amd.cdna4.buffer_load(post_weight_ptr, columns)
    if HAS_PRE:
        pre_weight = gl.amd.cdna4.buffer_load(pre_weight_ptr, columns)
    stages = ()
    for tile in gl.static_range(TILES):
        stages += (gl.load(epochs + tile).to(gl.uint32) % 3,)

    # Thread t polls the BF16 pairs of columns [2 * WORDS * t, 2 * WORDS * (t + 1)),
    # which lie in one source slot and one 512-element tile.
    word_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS], [64, 1], [NUM_WARPS, 1], [1, 0]
    )
    column = gl.arange(0, THREADS, layout=gl.SliceLayout(1, word_layout)) * (2 * WORDS)
    word = gl.arange(0, WORDS, layout=gl.SliceLayout(0, word_layout))
    within = column % LOCAL_WIDTH
    base = (
        (_iris_tile_stage(stages, within // 512) * (WORLD_SIZE * MAX_ELEMENTS // 2)).to(
            gl.int32
        )
        + column // LOCAL_WIDTH * (MAX_ELEMENTS // 2)
        + within // 2
    )
    mailbox = gl.multiple_of(base, WORDS)[:, None] + word[None, :]
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    # One reader per word. Payloads never hold negative zero and 32-bit words
    # are written whole, so a word is empty exactly when its high half is 0x8000.
    words = gl.amd.cdna4.buffer_load(words_ptr, mailbox, cache=".cv")
    while _iris_subgroup_any(gl.max(((words >> 16) == 0x8000).to(gl.int32), 1)):
        words = gl.amd.cdna4.buffer_load(words_ptr, mailbox, cache=".cv")
    gl.amd.cdna4.buffer_store(
        gl.full_like(words, 0x80000000), words_ptr, mailbox, cache=".wt"
    )
    # Widening BF16 to FP32 places its bits in the high half of the word.
    even = (words << 16).to(gl.float32, bitcast=True)
    odd = (words & 0xFFFF0000).to(gl.float32, bitcast=True)
    values = gl.convert_layout(gl.reshape(gl.join(even, odd), [WIDTH]), layout)

    values *= (
        (post_weight.to(gl.float32) + POST_WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    )
    values *= gl.rsqrt(gl.sum(values * values, 0) / WIDTH + POST_EPS)
    stream = gl.fma(beta, values, alpha * stream)
    gl.amd.cdna4.buffer_store(stream, out_stream_ptr, columns)
    if HAS_PRE:
        pre = stream * gl.rsqrt(gl.sum(stream * stream, 0) / WIDTH + PRE_EPS)
        pre *= (
            (pre_weight.to(gl.float32) + PRE_WEIGHT_OFFSET)
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        gl.amd.cdna4.buffer_store(pre.to(gl.bfloat16), pre_bf16_ptr, columns)
    # Every subgroup has left its poll, so every local sender has read its
    # stage before pushing a polled own-slot word. The next use of these tiles
    # is a later launch, which also sees the restored markers.
    gl.barrier()
    for tile in gl.static_range(TILES):
        gl.store(epochs + tile, ((stages[tile] + 1) % 3).to(gl.int32))


@gluon.jit
def iris_fused_gemv_lamport_allgather_muse_moetail(
    x_ptr,
    weight_ptr,
    region,
    epochs,
    regions,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_bf16_ptr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    HAS_PRE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Project, gather and normalize one decode row of Muse's post-expert tail.

    This rank projects its column shard, gathers the row through Lamport
    mailboxes, then applies Muse's post-norm, gated residual and next input
    norm.

    weight_ptr holds this rank's LOCAL_WIDTH output rows of the [WIDTH, K]
    projection. Grid (LOCAL_WIDTH // OUTPUTS + 1,) with 16 subgroups, OUTPUTS =
    16 // WAVES_PER_OUTPUT: each sender projects OUTPUTS outputs and stores them at
    elements [0, LOCAL_WIDTH) of slot RANK in every rank's ProducerAllReduce
    one-shot mailbox, its own included; the last program waits for all slots,
    clears them, writes the FP32 residual and (HAS_PRE) the BF16 next input,
    and advances the per-512-element stage counters of tiles [0, LOCAL_WIDTH /
    512). Outputs match the replicated (BLOCK_K, WAVES_PER_OUTPUT) GEMV followed by
    _post_norm_residual_pre_norm_kernel, except that negative zero is sent as
    zero.
    """
    NUM_WARPS: gl.constexpr = 16
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT
    gl.static_assert(WORLD_SIZE == 4 or WORLD_SIZE == 8)
    # Senders store whole 32-bit words; the reducer reads a word's high half.
    gl.static_assert(OUTPUTS % 2 == 0 and 512 % OUTPUTS == 0)
    gl.static_assert(LOCAL_WIDTH % 512 == 0)
    gl.static_assert(MAX_ELEMENTS % 512 == 0 and LOCAL_WIDTH <= MAX_ELEMENTS)
    program = gl.program_id(0)
    # The reducer is the last program so every sender dispatches before it spins.
    if program == LOCAL_WIDTH // OUTPUTS:
        _iris_muse_moetail_reduce(
            region,
            epochs,
            stream_ptr,
            post_weight_ptr,
            alpha_ptr,
            beta_ptr,
            pre_weight_ptr,
            out_stream_ptr,
            pre_bf16_ptr,
            WORLD_SIZE,
            LOCAL_WIDTH,
            MAX_ELEMENTS,
            POST_EPS,
            POST_WEIGHT_OFFSET,
            PRE_EPS,
            PRE_WEIGHT_OFFSET,
            HAS_PRE,
            NUM_WARPS,
        )
    else:
        _iris_muse_moetail_send(
            x_ptr,
            weight_ptr,
            epochs,
            regions,
            program,
            RANK,
            WORLD_SIZE,
            K,
            MAX_ELEMENTS,
            OUTPUTS,
            BLOCK_K,
            WAVES_PER_OUTPUT,
            NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE,
        )
