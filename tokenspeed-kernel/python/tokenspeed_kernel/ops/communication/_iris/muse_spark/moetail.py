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

"""Fuse Muse's post-expert projection, column gather and layer boundary for decode rows.

The launches here are built from these device functions:

* senders: ``iris_muse_moetail_stage_rows``, ``iris_muse_moetail_project_weights``
  and ``iris_muse_moetail_project_staged`` (column GEMV) or
  ``iris_muse_moetail_load_columns`` (columns projected beforehand), then
  ``iris_muse_moetail_send`` (Lamport push of every row's columns to every
  rank; peer stores go through a caller-supplied sink);
* one reducer per row: ``iris_muse_moetail_boundary_operands`` (issued before
  the wait), ``iris_muse_moetail_reduce`` (wait), ``iris_muse_moetail_publish`` (boundary and outputs), then
  ``iris_muse_moetail_release`` (clear the mailbox after the outputs are out).
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_gather import _iris_tile_stage
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_lamport_push,
    _iris_sanitize_lamport_bf16,
    _iris_subgroup_any,
)
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    # Senders keep the replicated single-row GEMV's per-output reduction.
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        gluon_rowcta_bf16_gemv_rows_finish_lds_gfx950,
        gluon_rowcta_bf16_gemv_rows_stage_x_gfx950,
        gluon_rowcta_bf16_gemv_rows_weights_gfx950,
    )


@gluon.constexpr_function
def _send_layout(outputs: int, num_warps: int):
    """[rows, outputs] with up to 8 adjacent outputs (16 B) per lane.

    From 64 outputs on, 8 lanes cover a row's 128-byte line, so each push
    instruction writes whole lines; fewer outputs put one row per lane.
    """
    if outputs & (outputs - 1):
        raise ValueError(
            f"moetail senders need a power-of-two output count, got {outputs}"
        )
    per_lane = min(outputs, 8)
    lanes = min(max(outputs // 8, 1), 8)
    return gl.BlockedLayout([1, per_lane], [64 // lanes, lanes], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _row_layout(width: int, num_warps: int):
    return gl.BlockedLayout([width // (64 * num_warps)], [64], [num_warps], [0])


@gluon.jit
def iris_muse_moetail_project_weights(
    weight_ptr,
    first_output,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Issue the weights of outputs [first_output, first_output + OUTPUTS)
    for :func:`iris_muse_moetail_project_staged`."""
    return gluon_rowcta_bf16_gemv_rows_weights_gfx950(
        weight_ptr, first_output, K, BLOCK_M, BLOCK_K, WAVES_PER_OUTPUT, 1, NUM_WARPS
    )  # fmt: skip


@gluon.jit
def iris_muse_moetail_stage_rows(
    x_ptr,
    M,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Stage every row to project in LDS once per workgroup (16-byte copies);
    every sender program of the workgroup then reads it
    (:func:`iris_muse_moetail_project_staged`)."""
    return gluon_rowcta_bf16_gemv_rows_stage_x_gfx950(x_ptr, M, K, BLOCK_M, NUM_WARPS)


@gluon.jit
def iris_muse_moetail_project_staged(
    weights,
    x_smem,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    OUTPUTS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Project the issued weights' OUTPUTS outputs for every LDS-staged row.

    Each (row, output) keeps the replicated single-row GEMV's reduction.

    Returns:
        BF16 [BLOCK_M, OUTPUTS] in the send layout.
    """
    values = gluon_rowcta_bf16_gemv_rows_finish_lds_gfx950(
        weights, x_smem, K, BLOCK_M, BLOCK_K, WAVES_PER_OUTPUT, 1, NUM_WARPS
    )
    return gl.convert_layout(values.to(gl.bfloat16), _send_layout(OUTPUTS, NUM_WARPS))


@gluon.jit
def iris_muse_moetail_load_columns(
    y_ptr,
    first_output,
    M,
    LOCAL_WIDTH: gl.constexpr,
    BLOCK_M: gl.constexpr,
    OUTPUTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Load outputs [first_output, first_output + OUTPUTS) of every row of y [M, LOCAL_WIDTH].

    Returns:
        BF16 [BLOCK_M, OUTPUTS] in the send layout.
    """
    layout: gl.constexpr = _send_layout(OUTPUTS, NUM_WARPS)
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    outputs = first_output + gl.arange(0, OUTPUTS, layout=gl.SliceLayout(0, layout))
    elements = rows[:, None] * LOCAL_WIDTH + outputs[None, :]
    valid = (rows < M)[:, None] & (outputs >= 0)[None, :]
    return gl.amd.cdna4.buffer_load(y_ptr, elements, mask=valid, other=0.0)


@gluon.jit
def iris_muse_moetail_send(
    values,
    first_output,
    program,
    epochs,
    regions,
    M,
    sink: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    OUTPUTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_STORE_CACHE: gl.constexpr,
    OWN_STORE_CACHE: gl.constexpr,
):
    """Store ``values`` (outputs from ``first_output`` of every row) in slot RANK of every mailbox.

    Row m lands at elements [m * LOCAL_WIDTH, (m + 1) * LOCAL_WIDTH) of the
    copy its tile's stage word in ``epochs`` picks; a copy holds WORLD_SIZE
    slots of MAX_ELEMENTS.
    ``sink(published, regions, offset, offsets, mask, program, RANK,
    WORLD_SIZE, PEER_STORE_CACHE)`` stores the peers' copies, ``published[d -
    1]`` for peer RANK + d (``_iris_lamport_push``'s signature); ``program``
    picks the sender's first peer. The own copy is stored here with
    ``OWN_STORE_CACHE``.
    """
    layout: gl.constexpr = _send_layout(OUTPUTS, NUM_WARPS)
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    outputs = first_output + gl.arange(0, OUTPUTS, layout=gl.SliceLayout(0, layout))
    elements = rows[:, None] * LOCAL_WIDTH + outputs[None, :]
    valid = (rows < M)[:, None] & (outputs >= 0)[None, :]
    # OUTPUTS divides the 512-element tile, so a row's outputs share one stage.
    stage = gl.load(epochs + elements // 512, mask=valid, other=0).to(gl.uint32) % 3
    # Many lanes may hold each value; the store lowering masks the duplicates,
    # so one lane writes each 8-byte packet.
    values = _iris_sanitize_lamport_bf16(gl.convert_layout(values, layout))
    offsets = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS)).to(gl.int32)
        + RANK * MAX_ELEMENTS
        + elements
    )
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (values,)
    sink(
        published,
        regions,
        0,
        offsets,
        valid,
        program,
        RANK,
        WORLD_SIZE,
        PEER_STORE_CACHE,
    )
    # The reducer may poll the own slot from another XCD, so the own store
    # must reach memory (write through). Like the peer bases, the hinted own
    # base keeps every 32-bit word one store at any specialization; split
    # halves could expose a stale low half.
    own = gl.multiple_of(regions[RANK].to(gl.pointer_type(gl.bfloat16)), 16)
    gl.amd.cdna4.buffer_store(values, own, offsets, mask=valid, cache=OWN_STORE_CACHE)


@gluon.jit
def iris_muse_moetail_boundary_operands(
    stream_ptr,
    alpha_ptr,
    beta_ptr,
    post_weight_ptr,
    pre_weight_ptr,
    row,
    WIDTH: gl.constexpr,
    HAS_PRE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Load ``row``'s residual and the boundary vectors (issue before polling).

    Returns:
        (stream, alpha, beta, post_weight, pre_weight); pre_weight is
        post_weight without HAS_PRE.
    """
    columns = gl.arange(0, WIDTH, layout=_row_layout(WIDTH, NUM_WARPS))
    stream = gl.amd.cdna4.buffer_load(stream_ptr + row * WIDTH, columns)
    alpha = gl.amd.cdna4.buffer_load(alpha_ptr, columns)
    beta = gl.amd.cdna4.buffer_load(beta_ptr, columns)
    post_weight = gl.amd.cdna4.buffer_load(post_weight_ptr, columns)
    pre_weight = post_weight
    if HAS_PRE:
        pre_weight = gl.amd.cdna4.buffer_load(pre_weight_ptr, columns)
    return stream, alpha, beta, post_weight, pre_weight


@gluon.jit
def _iris_muse_moetail_mailbox(
    stages,
    row,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """32-bit word offsets of every slot's part of ``row``, [THREADS, WORDS]."""
    WIDTH: gl.constexpr = LOCAL_WIDTH * WORLD_SIZE
    THREADS: gl.constexpr = 64 * NUM_WARPS
    WORDS: gl.constexpr = WIDTH // 2 // THREADS
    # Thread t holds the BF16 pairs of columns [2 * WORDS * t, 2 * WORDS * (t + 1)),
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
        + (row * LOCAL_WIDTH + within) // 2
    )
    return gl.multiple_of(base, WORDS)[:, None] + word[None, :]


@gluon.jit
def iris_muse_moetail_row_stages(epochs, row, LOCAL_WIDTH: gl.constexpr):
    """The mailbox stage of each 512-element tile of ``row`` (a tuple)."""
    TILES: gl.constexpr = LOCAL_WIDTH // 512
    stages = ()
    for tile in gl.static_range(TILES):
        stages += (gl.load(epochs + row * TILES + tile).to(gl.uint32) % 3,)
    return stages


@gluon.jit
def iris_muse_moetail_reduce(
    region,
    stages,
    row,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    POLL_CACHE: gl.constexpr,
):
    """Wait for every slot's part of ``row`` and return the gathered FP32 row.

    One reader per word. Payloads never hold negative zero and 32-bit words
    are written whole, so a word is empty exactly when its high half is
    0x8000. Each wave votes once per retry; the loop holds no barrier.

    Returns:
        FP32 [LOCAL_WIDTH * WORLD_SIZE] in the boundary's row layout.
    """
    WIDTH: gl.constexpr = LOCAL_WIDTH * WORLD_SIZE
    mailbox = _iris_muse_moetail_mailbox(
        stages, row, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, NUM_WARPS
    )
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    words = gl.amd.cdna4.buffer_load(words_ptr, mailbox, cache=POLL_CACHE)
    while _iris_subgroup_any(gl.max(((words >> 16) == 0x8000).to(gl.int32), 1)):
        words = gl.amd.cdna4.buffer_load(words_ptr, mailbox, cache=POLL_CACHE)
    # Widening BF16 to FP32 places its bits in the high half of the word.
    even = (words << 16).to(gl.float32, bitcast=True)
    odd = (words & 0xFFFF0000).to(gl.float32, bitcast=True)
    return gl.convert_layout(
        gl.reshape(gl.join(even, odd), [WIDTH]), _row_layout(WIDTH, NUM_WARPS)
    )


@gluon.jit
def iris_muse_moetail_publish(
    values,
    operands,
    row,
    out_stream_ptr,
    pre_bf16_ptr,
    WIDTH: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    HAS_PRE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    OUT_CACHE: gl.constexpr,
):
    """Run ``row``'s boundary on the gathered ``values`` and store its outputs.

    Updates the FP32 residual and (HAS_PRE) stores the BF16 next input.
    ``operands`` is iris_muse_moetail_boundary_operands' tuple. The norms
    repeat _post_norm_residual_pre_norm_kernel's operations in its layout
    (eight columns per thread, 16 waves) and order.
    """
    stream, alpha, beta, post_weight, pre_weight = operands
    columns = gl.arange(0, WIDTH, layout=_row_layout(WIDTH, NUM_WARPS))
    values *= (
        (post_weight.to(gl.float32) + POST_WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    )
    values *= gl.rsqrt(gl.sum(values * values, 0) / WIDTH + POST_EPS)
    stream = gl.fma(beta, values, alpha * stream)
    gl.amd.cdna4.buffer_store(
        stream, out_stream_ptr + row * WIDTH, columns, cache=OUT_CACHE
    )
    if HAS_PRE:
        pre = stream * gl.rsqrt(gl.sum(stream * stream, 0) / WIDTH + PRE_EPS)
        pre *= (
            (pre_weight.to(gl.float32) + PRE_WEIGHT_OFFSET)
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        gl.amd.cdna4.buffer_store(
            pre.to(gl.bfloat16), pre_bf16_ptr + row * WIDTH, columns, cache=OUT_CACHE
        )


@gluon.jit
def iris_muse_moetail_release(
    region,
    epochs,
    stages,
    row,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    CLEAR_CACHE: gl.constexpr,
):
    """Restore ``row``'s mailbox words to empty and advance its tile stages.

    Call after iris_muse_moetail_reduce (and, to publish sooner, after
    iris_muse_moetail_publish): the next use of these words is a later
    launch on another stage, so clearing late is safe.
    """
    TILES: gl.constexpr = LOCAL_WIDTH // 512
    mailbox = _iris_muse_moetail_mailbox(
        stages, row, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, NUM_WARPS
    )
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    gl.amd.cdna4.buffer_store(
        gl.full(mailbox.shape, 0x80000000, gl.uint32, mailbox.type.layout),
        words_ptr,
        mailbox,
        cache=CLEAR_CACHE,
    )
    # Every wave has left its poll, so every local sender has read this
    # row's stages before pushing a polled own-slot word. The next use of
    # these tiles is a later launch, which also sees the restored markers.
    gl.barrier()
    for tile in gl.static_range(TILES):
        gl.store(epochs + row * TILES + tile, ((stages[tile] + 1) % 3).to(gl.int32))


@gluon.jit
def _iris_muse_moetail_row(
    region,
    epochs,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_bf16_ptr,
    row,
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
    """One reducer program of the standalone launches: wait, publish, then clear."""
    WIDTH: gl.constexpr = LOCAL_WIDTH * WORLD_SIZE
    operands = iris_muse_moetail_boundary_operands(
        stream_ptr, alpha_ptr, beta_ptr, post_weight_ptr, pre_weight_ptr, row, WIDTH,
        HAS_PRE, NUM_WARPS
    )  # fmt: skip
    stages = iris_muse_moetail_row_stages(epochs, row, LOCAL_WIDTH)
    values = iris_muse_moetail_reduce(
        region, stages, row, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, NUM_WARPS, ".cv"
    )  # fmt: skip
    iris_muse_moetail_publish(
        values, operands, row, out_stream_ptr, pre_bf16_ptr, WIDTH, POST_EPS,
        POST_WEIGHT_OFFSET, PRE_EPS, PRE_WEIGHT_OFFSET, HAS_PRE, NUM_WARPS, ""
    )  # fmt: skip
    iris_muse_moetail_release(
        region, epochs, stages, row, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, NUM_WARPS,
        ".wt"
    )  # fmt: skip


@gluon.jit(do_not_specialize=("M",))
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
    M,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    K: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WAVES_PER_OUTPUT: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    HAS_PRE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Project, gather and normalize M decode rows of Muse's post-expert tail.

    This rank projects its column shard of every row, gathers the rows through
    Lamport mailboxes, then applies Muse's post-norm, gated residual and next
    input norm to each.

    weight_ptr holds this rank's LOCAL_WIDTH output rows of the [WIDTH, K]
    projection. Grid (LOCAL_WIDTH // OUTPUTS + M,) with 16 waves, OUTPUTS =
    16 // WAVES_PER_OUTPUT, M <= BLOCK_M: each sender projects OUTPUTS outputs of
    every row and stores row m's at elements [m * LOCAL_WIDTH, (m + 1) *
    LOCAL_WIDTH) of slot RANK in every rank's ProducerAllReduce one-shot
    mailbox, its own included; the last M programs, one per row, wait for all
    slots of their row, write the FP32 residual and (HAS_PRE) the BF16 next
    input, then clear the slots and advance their row's per-512-element stage
    counters. Each row matches the replicated (BLOCK_K, WAVES_PER_OUTPUT)
    single-row GEMV followed by _post_norm_residual_pre_norm_kernel, except
    that negative zero is sent as zero.
    """
    NUM_WARPS: gl.constexpr = 16
    OUTPUTS: gl.constexpr = NUM_WARPS // WAVES_PER_OUTPUT
    gl.static_assert(WORLD_SIZE == 4 or WORLD_SIZE == 8)
    # Senders store whole 32-bit words; the reducer reads a word's high half.
    gl.static_assert(OUTPUTS % 2 == 0 and 512 % OUTPUTS == 0)
    gl.static_assert(LOCAL_WIDTH % 512 == 0)
    gl.static_assert(MAX_ELEMENTS % 512 == 0 and BLOCK_M * LOCAL_WIDTH <= MAX_ELEMENTS)
    program = gl.program_id(0)
    # The reducers are the last programs so every sender dispatches before
    # any of them spins.
    if program >= LOCAL_WIDTH // OUTPUTS:
        _iris_muse_moetail_row(
            region, epochs, stream_ptr, post_weight_ptr, alpha_ptr, beta_ptr,
            pre_weight_ptr, out_stream_ptr, pre_bf16_ptr, program - LOCAL_WIDTH // OUTPUTS,
            WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, POST_EPS, POST_WEIGHT_OFFSET, PRE_EPS,
            PRE_WEIGHT_OFFSET, HAS_PRE, NUM_WARPS
        )  # fmt: skip
    else:
        # The rows go to LDS in a few wide copies, then the weights issue.
        x_smem = iris_muse_moetail_stage_rows(x_ptr, M, K, BLOCK_M, NUM_WARPS)
        weights = iris_muse_moetail_project_weights(
            weight_ptr, program * OUTPUTS, K, BLOCK_M, BLOCK_K, WAVES_PER_OUTPUT, NUM_WARPS
        )  # fmt: skip
        values = iris_muse_moetail_project_staged(
            weights, x_smem, K, BLOCK_M, OUTPUTS, BLOCK_K, WAVES_PER_OUTPUT, NUM_WARPS
        )  # fmt: skip
        iris_muse_moetail_send(
            values, program * OUTPUTS, program, epochs, regions, M, _iris_lamport_push,
            RANK, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, BLOCK_M, OUTPUTS, NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE, ".wt"
        )  # fmt: skip


@gluon.jit(do_not_specialize=("M",))
def iris_columns_lamport_allgather_muse_moetail(
    y_ptr,
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
    M,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    LOCAL_WIDTH: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    OUTPUTS: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    HAS_PRE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Gather and normalize M decode rows from this rank's projected columns.

    The moetail launch with its projection done beforehand: y_ptr holds this
    rank's BF16 columns [M, LOCAL_WIDTH] of the post-expert projection (for
    example a decode GEMM over the column shard, which keeps the replicated
    GEMM's per-output reduction). Grid (LOCAL_WIDTH // OUTPUTS + M,) with 16
    waves; senders and reducers otherwise match
    iris_fused_gemv_lamport_allgather_muse_moetail.
    """
    NUM_WARPS: gl.constexpr = 16
    gl.static_assert(WORLD_SIZE == 4 or WORLD_SIZE == 8)
    gl.static_assert(OUTPUTS % 2 == 0 and 512 % OUTPUTS == 0)
    gl.static_assert(LOCAL_WIDTH % 512 == 0)
    gl.static_assert(MAX_ELEMENTS % 512 == 0 and BLOCK_M * LOCAL_WIDTH <= MAX_ELEMENTS)
    program = gl.program_id(0)
    if program >= LOCAL_WIDTH // OUTPUTS:
        _iris_muse_moetail_row(
            region, epochs, stream_ptr, post_weight_ptr, alpha_ptr, beta_ptr,
            pre_weight_ptr, out_stream_ptr, pre_bf16_ptr, program - LOCAL_WIDTH // OUTPUTS,
            WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, POST_EPS, POST_WEIGHT_OFFSET, PRE_EPS,
            PRE_WEIGHT_OFFSET, HAS_PRE, NUM_WARPS
        )  # fmt: skip
    else:
        values = iris_muse_moetail_load_columns(
            y_ptr, program * OUTPUTS, M, LOCAL_WIDTH, BLOCK_M, OUTPUTS, NUM_WARPS
        )
        iris_muse_moetail_send(
            values, program * OUTPUTS, program, epochs, regions, M, _iris_lamport_push,
            RANK, WORLD_SIZE, LOCAL_WIDTH, MAX_ELEMENTS, BLOCK_M, OUTPUTS, NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE, ".wt"
        )  # fmt: skip
