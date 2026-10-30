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


"""Fuse attention reduction with Muse's residual update and MoE input norms."""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_group_heap_base,
    _iris_rendezvous,
    _iris_sanitize_lamport_bf16x2,
    _iris_subgroup_any,
)


@gluon.jit
def _iris_muse_attntail_inputs(
    stream_ptr, alpha_ptr, beta_ptr, pre_weight_ptr, row, columns, WIDTH: gl.constexpr
):
    """Load the residual row and per-column coefficients ahead of the reduction."""
    offsets = (row * WIDTH + columns).to(gl.int32)
    stream = gl.amd.cdna4.buffer_load(stream_ptr, offsets)
    alpha = gl.amd.cdna4.buffer_load(alpha_ptr, columns)
    beta = gl.amd.cdna4.buffer_load(beta_ptr, columns)
    pre_weight = gl.amd.cdna4.buffer_load(pre_weight_ptr, columns)
    return stream, alpha, beta, pre_weight


@gluon.jit
def _iris_muse_attntail(
    branch,
    stream,
    alpha,
    beta,
    pre_weight,
    post_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    row,
    columns,
    WIDTH: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
):
    """Update the FP32 hidden-state residual and prepare both MoE inputs.

    After rounding the reduced attention output to BF16 and post-normalizing it,
    update the residual as alpha * residual + beta * normalized_attention.
    Pre-normalize that FP32 result for the router (FP32) and pre-expert projection (BF16).
    stream, alpha, beta, and pre_weight come from _iris_muse_attntail_inputs.
    """
    offsets = (row * WIDTH + columns).to(gl.int32)
    branch = branch.to(gl.bfloat16).to(gl.float32)
    if POST_HAS_WEIGHT:
        weight = gl.amd.cdna4.buffer_load(post_weight_ptr, columns).to(gl.float32)
        branch *= (weight + POST_WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    branch *= gl.rsqrt(gl.sum(branch * branch, 0) / WIDTH + POST_EPS)
    stream = gl.fma(beta, branch, alpha * stream)
    gl.amd.cdna4.buffer_store(stream, out_stream_ptr, offsets)
    pre = stream * gl.rsqrt(gl.sum(stream * stream, 0) / WIDTH + PRE_EPS)
    weight = pre_weight.to(gl.float32)
    pre *= (weight + PRE_WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(pre, pre_fp32_ptr, offsets)
    gl.amd.cdna4.buffer_store(pre.to(gl.bfloat16), pre_bf16_ptr, offsets)


@gluon.jit(do_not_specialize=["rows", "local_rows"])
def iris_fused_pull_allgather_muse_attntail(
    branch_sym_ptr,
    stream_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    ready_flags,
    heaps,
    rows,
    local_rows,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    FLAG_ROWS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WAIT_VMEM: gl.constexpr,
):
    """Gather reduced attention rows and update Muse's full-row residuals.

    Read BF16 shards from branch_sym_ptr across the group, masking padded rows.
    Apply post-normalization, the alpha/beta residual update, and MoE input
    normalization. Write the FP32 residual and router input plus the BF16 pre-expert projection
    input to the supplied outputs. Peer counters protect the source shard until
    every rank has finished reading it.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    gl.static_assert(WIDTH == 8192)
    program = gl.program_id(0)
    epoch = (
        gl.atomic_add(
            ready_flags + program * WORLD_SIZE + RANK, 1, sem="relaxed", scope="gpu"
        )
        + 1
    )
    # Attention scatter uses write-through stores; no cache writeback is needed at entry.
    WAIT_VMEM(args=[], constraints="=r,~{memory}")
    gl.barrier()
    _iris_rendezvous(ready_flags, heaps, program, epoch, RANK, NUM_WARPS, False)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [64], [NUM_WARPS], [0]
    )
    columns = gl.arange(0, WIDTH, layout=layout)
    branch_offset = branch_sym_ptr.to(gl.uint64) - heaps[RANK]
    # Interleave owners so the persistent grid uses every peer link together.
    for job in range(program, WORLD_SIZE * local_rows, gl.num_programs(0)):
        owner = job % WORLD_SIZE
        local_row = job // WORLD_SIZE
        row = owner * local_rows + local_row
        if row < rows:
            source_base = _iris_group_heap_base(owner, heaps)
            source = gl.multiple_of(
                (source_base + branch_offset).to(gl.pointer_type(gl.bfloat16)), 16
            )
            offsets = (local_row * WIDTH + columns).to(gl.int32)
            branch = gl.amd.cdna4.buffer_load(source, offsets, cache=".cg")
            stream, alpha, beta, pre_weight = _iris_muse_attntail_inputs(
                stream_ptr, alpha_ptr, beta_ptr, pre_weight_ptr, row, columns, WIDTH
            )
            _iris_muse_attntail(
                branch,
                stream,
                alpha,
                beta,
                pre_weight,
                branch_sym_ptr,
                out_stream_ptr,
                pre_fp32_ptr,
                pre_bf16_ptr,
                row,
                columns,
                WIDTH,
                POST_EPS,
                0.0,
                PRE_EPS,
                PRE_WEIGHT_OFFSET,
                False,
            )
    # Every rank consumed these rows before the next producer reuses its shard.
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


@gluon.jit
def _iris_muse_wait_mailbox(
    mailbox_ptr,
    slot,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    peer_stride,
    WORDS_PER_THREAD: gl.constexpr,
):
    """Reload peers' packed BF16 rows until no element holds the empty marker.

    slot holds each thread's WORDS_PER_THREAD consecutive 32-bit word offsets.
    Each subgroup polls on its own, without a workgroup-wide reduction, until
    its words from every peer have arrived.
    """
    THREADS: gl.constexpr = slot.shape[0] // WORDS_PER_THREAD
    slot = gl.reshape(slot, [THREADS, WORDS_PER_THREAD])
    received = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        received += (gl.zeros_like(slot).to(gl.uint32),)
    waiting = True
    while waiting:
        missing = gl.zeros_like(slot)
        loaded = ()
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            # Arrived words stay unchanged until this row's slot is cleared,
            # so every thread rereads all of its words.
            words = gl.amd.cdna4.buffer_load(
                mailbox_ptr, slot + peer * peer_stride, cache=".cv"
            )
            loaded += (words,)
            # Senders never send negative zero, so a word is still empty
            # exactly when its high half holds the marker.
            missing |= ((words >> 16) == 0x8000).to(gl.int32)
        received = loaded
        waiting = _iris_subgroup_any(gl.max(missing, 1))
    flat = ()
    for delta in gl.static_range(1, WORLD_SIZE):
        flat += (gl.reshape(received[delta - 1], [THREADS * WORDS_PER_THREAD]),)
    return flat


@gluon.jit
def _iris_muse_lamport_reduce_row(
    branch_ptr,
    words_ptr,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    row,
    stage,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    WIDTH: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Wait for a row's peer contributions, then reduce and normalize it."""
    PAIRS: gl.constexpr = WIDTH // 2
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [64], [NUM_WARPS], [0]
    )
    pairs: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD // 2], [64], [NUM_WARPS], [0]
    )
    pair = gl.arange(0, PAIRS, layout=pairs)
    local = _iris_sanitize_lamport_bf16x2(
        gl.amd.cdna4.buffer_load(
            branch_ptr.to(gl.pointer_type(gl.uint32), bitcast=True), row * PAIRS + pair
        )
    )
    # Overlap the residual and coefficient reads with the peer wait.
    columns = gl.arange(0, WIDTH, layout=layout)
    stream, alpha, beta, pre_weight = _iris_muse_attntail_inputs(
        stream_ptr, alpha_ptr, beta_ptr, pre_weight_ptr, row, columns, WIDTH
    )
    received = _iris_muse_wait_mailbox(
        words_ptr,
        (stage * WORLD_SIZE * MAX_ROWS + row) * PAIRS + pair,
        RANK,
        WORLD_SIZE,
        MAX_ROWS * PAIRS,
        ELEMENTS_PER_THREAD // 2,
    )
    # Widening BF16 to FP32 places its bits in the high half of the word.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            words = local
        else:
            words = received[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        even = (words << 16).to(gl.float32, bitcast=True)
        odd = (words & 0xFFFF0000).to(gl.float32, bitcast=True)
        if peer == 0:
            branch_even = even
            branch_odd = odd
        else:
            branch_even += even
            branch_odd += odd
    branch = gl.convert_layout(
        gl.reshape(gl.join(branch_even, branch_odd), [WIDTH]), layout
    )
    _iris_muse_attntail(
        branch,
        stream,
        alpha,
        beta,
        pre_weight,
        post_weight_ptr,
        out_stream_ptr,
        pre_fp32_ptr,
        pre_bf16_ptr,
        row,
        columns,
        WIDTH,
        POST_EPS,
        POST_WEIGHT_OFFSET,
        PRE_EPS,
        PRE_WEIGHT_OFFSET,
        POST_HAS_WEIGHT,
    )


@gluon.jit
def _iris_muse_lamport_send_chunk(
    branch_ptr,
    mailbox_ptr,
    words_ptr,
    heaps,
    row,
    stage,
    chunk,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    WIDTH: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    SENDERS: gl.constexpr,
    ROTATE_PEERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Push one column chunk of a row and clear its previously consumed slot."""
    PAIRS: gl.constexpr = WIDTH // 2
    PEER_STRIDE: gl.constexpr = MAX_ROWS * PAIRS
    gl.static_assert(PAIRS % SENDERS == 0)
    CHUNK: gl.constexpr = PAIRS // SENDERS
    THREADS: gl.constexpr = 64 * NUM_WARPS
    # Chunks smaller than the workgroup leave the remaining threads idle.
    WORDS: gl.constexpr = max(CHUNK // THREADS, 1)
    gl.static_assert(CHUNK % (WORDS * THREADS) == 0 or CHUNK < THREADS)
    layout: gl.constexpr = gl.BlockedLayout([WORDS], [64], [NUM_WARPS], [0])
    index = gl.arange(0, WORDS * THREADS, layout=layout)
    valid = index < CHUNK
    pair = chunk * CHUNK + index
    local = gl.amd.cdna4.buffer_load(
        branch_ptr.to(gl.pointer_type(gl.uint32), bitcast=True),
        row * PAIRS + pair,
        mask=valid,
    )
    local = _iris_sanitize_lamport_bf16x2(local)
    slot = (stage * WORLD_SIZE * MAX_ROWS + row) * PAIRS + pair
    mailbox_offset = mailbox_ptr.to(gl.uint64) - _iris_group_heap_base(RANK, heaps)
    for step in gl.static_range(WORLD_SIZE - 1):
        if ROTATE_PEERS:
            peer = (
                RANK + 1 + (step + row * SENDERS + chunk) % (WORLD_SIZE - 1)
            ) % WORLD_SIZE
        else:
            peer = (RANK + 1 + step) % WORLD_SIZE
        destination = gl.multiple_of(
            (_iris_group_heap_base(peer, heaps) + mailbox_offset).to(
                gl.pointer_type(gl.uint32)
            ),
            16,
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination,
            slot + RANK * PEER_STRIDE,
            mask=valid,
            cache=PEER_SIGNAL_STORE_CACHE,
        )
    # The previous call consumed the slot two stages ahead. Write-through
    # clears reach memory before this kernel completes, and peers rewrite
    # that slot only after this row's next reduction.
    previous = slot + ((stage + 2) % 3 - stage) * WORLD_SIZE * PEER_STRIDE
    empty = gl.full((WORDS * THREADS,), 0x80008000, gl.uint32, layout)
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        gl.amd.cdna4.buffer_store(
            empty,
            words_ptr,
            previous + peer * PEER_STRIDE,
            mask=valid,
            cache=".wt",
        )


@gluon.jit
def iris_fused_lamport_allreduce_muse_attntail(
    branch_ptr,
    mailbox_ptr,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    epochs,
    heaps,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    WIDTH: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
    SENDERS: gl.constexpr,
    ROTATE_PEERS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """All-reduce TP4/TP8 attention rows, then apply Muse's residual update and norms.

    Grid (rows, SENDERS + 1): sender workgroups push column chunks of each BF16
    row to peer mailboxes and clear the row's previously consumed slot of three;
    the last workgroup waits for every peer chunk, sums in FP32 rank order, and
    applies the post-norm, alpha/beta update and MoE input norm. ``epochs`` holds
    one stage per workgroup and advances only through this kernel. Negative zero
    is sent as zero. ROTATE_PEERS staggers each sender's peer order.
    """
    gl.static_assert(WORLD_SIZE == 4 or WORLD_SIZE == 8)
    gl.static_assert(WIDTH == 8192)
    gl.static_assert(ELEMENTS_PER_THREAD % 2 == 0)
    row = gl.program_id(0)
    role = gl.program_id(1)
    epoch = epochs + row * (SENDERS + 1) + role
    stage = (gl.load(epoch).to(gl.uint32) % 3).to(gl.int32)
    # Mailboxes hold BF16 pairs as 32-bit words; the low half is the even column.
    words_ptr = mailbox_ptr.to(gl.pointer_type(gl.uint32), bitcast=True)
    # The reducer must be the last role so that senders dispatch before any
    # reducer spins; otherwise peers could wait on senders that never run.
    if role == SENDERS:
        _iris_muse_lamport_reduce_row(
            branch_ptr,
            words_ptr,
            stream_ptr,
            post_weight_ptr,
            alpha_ptr,
            beta_ptr,
            pre_weight_ptr,
            out_stream_ptr,
            pre_fp32_ptr,
            pre_bf16_ptr,
            row,
            stage,
            RANK,
            WORLD_SIZE,
            WIDTH,
            MAX_ROWS,
            POST_EPS,
            POST_WEIGHT_OFFSET,
            PRE_EPS,
            PRE_WEIGHT_OFFSET,
            POST_HAS_WEIGHT,
            ELEMENTS_PER_THREAD,
            NUM_WARPS,
        )
    else:
        _iris_muse_lamport_send_chunk(
            branch_ptr,
            mailbox_ptr,
            words_ptr,
            heaps,
            row,
            stage,
            role,
            RANK,
            WORLD_SIZE,
            WIDTH,
            MAX_ROWS,
            SENDERS,
            ROTATE_PEERS,
            NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE,
        )
    gl.store(epoch, (stage + 1) % 3)
