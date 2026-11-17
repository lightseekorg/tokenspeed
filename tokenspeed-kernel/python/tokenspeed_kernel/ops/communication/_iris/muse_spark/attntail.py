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

from tokenspeed_kernel._triton import gl, gluon, tl
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_group_heap_base,
    _iris_rendezvous,
    _iris_sanitize_lamport_bf16,
    _iris_sanitize_lamport_bf16x2,
    _iris_subgroup_any,
    _iris_wait_lamport_slots,
    iris_lamport_twoshot_ag_push,
    iris_lamport_twoshot_rs_wait_reduce,
)
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    # The fused sliding-attention tail runs the gfx950 split-decode body as one role.
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        AttentionConfig,
        AttentionProgram,
        InputStrides,
        stream_split_pipelined,
        stream_split_single_buffer,
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
    WAIT_VMEM()
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
    WAIT_VMEM()
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
    # A wave that loads the stage after this store would push into the next
    # stage's slot. Every wave has used its stage before the barrier, so no
    # memory wait is needed.
    gl.barrier()
    gl.store(epoch, (stage + 1) % 3)


# ---------------------------------------------------------------------------
# Sliding attention + gated o-norm + o_proj + two-shot all-reduce + boundary
# ---------------------------------------------------------------------------

# An empty BF16 pair in the attention buffer: negative zero in both halves.
_ATT_BUFFER_EMPTY = gl.constexpr(0x80008000)

# The fused sliding tail's epoch words count launches modulo this period. A
# multiple of three keeps the stage (epoch % 3) rotating across the wrap; below
# 2**31 the tag (epoch + 1) never reaches 0, the tag of a never-written norm
# sum. Counting modulo 2**32 instead would repeat a stage at the wrap
# (2**32 - 1 and 0 are both 0 mod 3), and a signed count would turn negative.
_SLIDING_TAIL_EPOCH_PERIOD = gl.constexpr(3 << 29)


@gluon.jit
def _muse_attntail_wait_buffer(buffer_ptr, offsets, valid, LAYOUT: gl.constexpr):
    """Wait for this rank's BF16 words at offsets of a local Lamport mailbox.

    The mailbox is _iris_wait_lamport_slots' one-source case: rank 1 of 2
    polls source 0's slot at buffer_ptr.
    """
    return _iris_wait_lamport_slots(buffer_ptr, offsets, valid, 1, 2, 8, LAYOUT)[0]


@gluon.jit
def muse_attntail_store_attention(program, gate, att_buffer, EPS: gl.constexpr):
    """Merge this row block's splits, gate and normalize, and publish them.

    Only the last-arriving split calls it. As attention followed by
    weightless_rmsnorm_gated_sigmoid: the merged rows round to BF16, then per
    head x * rsqrt(mean(x^2) + EPS) * sigmoid(gate) in FP32, rounded to BF16.
    The rows go to att_buffer ([tokens, heads * head_dim] BF16, empty marker
    -0.0) written through, since readers on every XCD poll them; negative
    zero is stored as zero.
    """
    cfg = program.cfg
    x = program.merge_splits(att_buffer, False).to(gl.bfloat16).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 1) / cfg.HEAD_DIM + EPS)
    g = gate.to(gl.float32)
    rows = _iris_sanitize_lamport_bf16(
        (x * inv[:, None] * (1.0 / (1.0 + gl.exp(-g)))).to(gl.bfloat16)
    )
    # Eight contiguous BF16 per lane: whole 32-bit words in one 16-byte store.
    buffer_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [8, 8], [cfg.NUM_WARPS, 1], [1, 0]
    )
    rows = gl.convert_layout(rows, buffer_layout)
    m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, buffer_layout))
    d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, buffer_layout))
    base = gl.multiple_of(
        program.row_tokens(m) * (cfg.NUM_Q_HEADS * cfg.HEAD_DIM)
        + program.row_heads(m) * cfg.HEAD_DIM,
        cfg.HEAD_DIM,
    )
    gl.amd.cdna4.buffer_store(
        rows,
        gl.multiple_of(att_buffer.to(gl.pointer_type(gl.bfloat16), bitcast=True), 16),
        base[:, None] + d[None, :],
        mask=program.row_valid(m)[:, None] & (d >= 0)[None, :],
        cache=".wt",
    )


@gluon.jit
def muse_attntail_attention(
    request,
    head_block,
    split,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    counters_ptr,
    gate_ptr,
    gate_stride,
    att_buffer,
    cfg: gl.constexpr,
    EPS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """One sliding split of _mha_decode_sliding_split_atomic; the last split
    publishes the gated, normalized rows (muse_attntail_store_attention).
    NUM_STAGES pages stream in flight, as the split decode's launch config sets."""
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        request,
        head_block,
        split,
    )
    # Every split loads its row block's gate now; only the last arriver uses it.
    m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
    d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
    gate_rows = (
        program.row_tokens(m) * gate_stride + program.row_heads(m) * cfg.HEAD_DIM
    )
    gate = gl.amd.cdna4.buffer_load(
        gate_ptr,
        gate_rows[:, None] + d[None, :],
        mask=program.row_valid(m)[:, None] & (d >= 0)[None, :],
    )
    q = program.load_q()
    m_i, l_i, acc, _ = program.init_state(q_ptr, False)
    if NUM_STAGES == 1:
        k_smem = gl.allocate_shared_memory(
            k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
        )
        v_smem = gl.allocate_shared_memory(
            v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
        )
        m_i, l_i, acc = stream_split_single_buffer(
            program, q, m_i, l_i, acc, k_smem, v_smem
        )
    else:
        m_i, l_i, acc = stream_split_pipelined(
            program, q, m_i, l_i, acc, k_cache_ptr, v_cache_ptr, NUM_STAGES
        )
    program.store_split(acc, l_i, m_i)
    if program.arrive(counters_ptr) == cfg.NUM_KV_SPLITS - 1:
        muse_attntail_store_attention(program, gate, att_buffer, EPS)
        gl.store(counters_ptr + program.split_counter_offset(), 0)


@gluon.jit
def muse_attntail_o_proj_prefetch(w_ptr, block, K: gl.constexpr, OUTPUTS: gl.constexpr):
    """Issue o_proj rows [block * OUTPUTS, (block + 1) * OUTPUTS) of the
    [*, K] BF16 weight into registers, in the MFMA A layout (.cg).

    The waves split K. Returns one [waves, 16, K / waves] tile per 16 rows.
    """
    WAVES: gl.constexpr = gl.num_warps()
    KW: gl.constexpr = K // WAVES
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WAVES, 1, 1],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma, k_width=8
    )
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_layout)))
    row = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, a_layout)))
    k = gl.arange(0, KW, gl.SliceLayout(0, gl.SliceLayout(1, a_layout)))
    base = row[None, :, None] * K + (wave * KW)[:, None, None] + k[None, None, :]
    tiles = ()
    for t in gl.static_range(OUTPUTS // 16):
        tiles += (
            gl.amd.cdna4.buffer_load(
                w_ptr, base + (block * OUTPUTS + t * 16) * K, cache=".cg"
            ),
        )
    return tiles


@gluon.jit
def muse_attntail_send(
    tiles,
    block,
    att_buffer,
    rs_peers,
    stage,
    rows,
    K: gl.constexpr,
    OUTPUTS: gl.constexpr,
    WIDTH: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Project the attention buffer through this sender's o_proj rows and push
    the BF16 partials to their columns' owner.

    Waits for the buffer's rows (rows <= 16), multiplies them by the resident
    tiles (muse_attntail_o_proj_prefetch), sums the waves' K parts in a fixed
    order, and stores row r's OUTPUTS columns in reduce-scatter slot RANK of
    the rank owning them (WIDTH / WORLD_SIZE columns each), at r * (WIDTH /
    WORLD_SIZE) + column % (WIDTH / WORLD_SIZE): the two-shot region's layout
    with column shards.
    """
    WAVES: gl.constexpr = gl.num_warps()
    KW: gl.constexpr = K // WAVES
    TILES: gl.constexpr = OUTPUTS // 16
    gl.static_assert(TILES == 4)
    # The 16 x K buffer as 256 packs (one per lane of 4 waves), each polled
    # until it no longer holds the empty marker.
    PACK: gl.constexpr = 16 * K // 256
    packs: gl.constexpr = gl.BlockedLayout([1, PACK], [64, 1], [WAVES, 1], [1, 0])
    pack = gl.arange(0, 256, layout=gl.SliceLayout(1, packs))
    lane = gl.arange(0, PACK, layout=gl.SliceLayout(0, packs))
    buffer_ptr = att_buffer.to(gl.pointer_type(gl.bfloat16), bitcast=True)
    offsets = pack[:, None] * PACK + lane[None, :]
    valid = pack * PACK // K < rows
    # First wait on one word per merged row block (the first token's first head
    # of each KV head: 8 heads of 64 apart), so 128 senders do not stream the
    # whole buffer from memory while attention runs; then read every pack. The
    # probe reads one word per lane: one load per poll.
    words: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [1, 0])
    probe_pack = gl.arange(0, 256, layout=gl.SliceLayout(1, words))
    probe = (
        (probe_pack * PACK // K < rows)
        & (probe_pack * PACK % (8 * K) < K)
        & (probe_pack * PACK % (8 * 64) == 0)
    )
    probe_offsets = (
        probe_pack[:, None] * PACK
        + gl.arange(0, 2, layout=gl.SliceLayout(0, words))[None, :]
    )
    _muse_attntail_wait_buffer(buffer_ptr, probe_offsets, probe, words)
    attended = _muse_attntail_wait_buffer(buffer_ptr, offsets, valid, packs)
    # [16 tokens, K] -> the MFMA B operand [waves, K / waves, 16 tokens].
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WAVES, 1, 1],
    )
    b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma, k_width=8
    )
    x = gl.convert_layout(
        gl.permute(gl.reshape(attended, [16, WAVES, KW]), [1, 2, 0]), b_layout
    )
    sums_layout: gl.constexpr = gl.BlockedLayout(
        [WAVES, 1, 1], [1, 4, 16], [1, WAVES, 1], [2, 1, 0]
    )
    partials = ()
    for t in gl.static_range(TILES):
        buffer = gl.allocate_shared_memory(
            gl.float32, [WAVES, 16, 16], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
        )
        buffer.store(
            gl.amd.cdna4.mfma(tiles[t], x, gl.zeros((WAVES, 16, 16), gl.float32, mfma))
        )
        partials += (buffer,)
    gl.barrier()
    columns = ()
    for t in gl.static_range(TILES):
        # [16 outputs, 16 tokens] summed over the waves, as [tokens, outputs].
        columns += (gl.permute(gl.sum(partials[t].load(sums_layout), axis=0), [1, 0]),)
    out = gl.join(gl.join(columns[0], columns[1]), gl.join(columns[2], columns[3]))
    out = gl.reshape(gl.permute(out, [0, 3, 2, 1]), [16, OUTPUTS])
    # Each wave stores whole token rows (OUTPUTS BF16 = one 128 B line), so
    # no line crosses the link in partial writes.
    PUSH_VECTOR: gl.constexpr = 16 * OUTPUTS // (64 * WAVES)
    push: gl.constexpr = gl.BlockedLayout(
        [1, PUSH_VECTOR],
        [64 * PUSH_VECTOR // OUTPUTS, OUTPUTS // PUSH_VECTOR],
        [WAVES, 1],
        [1, 0],
    )
    out = _iris_sanitize_lamport_bf16(gl.convert_layout(out, push).to(gl.bfloat16))
    token = gl.arange(0, 16, gl.SliceLayout(1, push))
    column = block * OUTPUTS + gl.arange(0, OUTPUTS, gl.SliceLayout(0, push))
    # The column owner's reduce-scatter slot RANK: [rows, WIDTH / WORLD_SIZE].
    OWNED: gl.constexpr = WIDTH // WORLD_SIZE
    owner = block * OUTPUTS // OWNED
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SHARD_ELEMENTS
    offsets = (
        stage * (2 * PHASE)
        + RANK * MAX_SHARD_ELEMENTS
        + token[:, None] * OWNED
        + (column % OWNED)[None, :]
    )
    valid = (token < rows)[:, None] & (column >= 0)[None, :]
    for peer in gl.static_range(WORLD_SIZE):
        if owner == peer:
            destination = gl.multiple_of(
                rs_peers[peer].to(gl.pointer_type(gl.bfloat16)), 16
            )
            if peer == RANK:
                # This rank's own source slot, read on other XCDs.
                gl.amd.cdna4.buffer_store(
                    out, destination, offsets, mask=valid, cache=".wt"
                )
            else:
                gl.amd.cdna4.buffer_store(
                    out, destination, offsets, mask=valid, cache=PEER_SIGNAL_STORE_CACHE
                )


@gluon.jit
def _muse_attntail_pack_sums(tag, values):
    """64-bit [tag << 32 | FP32 bits] words: one store each, so a reader never
    sees a current tag next to an old value."""
    return (tag.to(gl.int64) << 32) | (
        values.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF
    )


@gluon.jit
def muse_attntail_own(
    owner,
    region,
    peers,
    sums_peers,
    tag,
    stage,
    rows,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    WIDTH: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    OWNER_COLUMNS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Reduce this rank's columns [owner * OWNER_COLUMNS, ...) of its column
    shard and publish them with their norm sums.

    At entry, restores the empty markers of every source's reduce-scatter
    slot for these columns in stage + 2 (written through), which a peer
    writes again only two launches on. Then waits for all WORLD_SIZE sources
    and sums them in rank order with one BF16 rounding
    (iris_lamport_twoshot_rs_wait_reduce), pushes b to every rank's
    all-gather slot RANK (its own included) and, per row, four FP32 norm sums
    of these columns as 64-bit tagged words: sum b^2, B = sum (alpha h)
    (beta b), A = sum (alpha h)^2, C = sum (beta b)^2, with b weighted by the
    post-norm when POST_HAS_WEIGHT.
    """
    WAVES: gl.constexpr = gl.num_warps()
    OWNED: gl.constexpr = WIDTH // WORLD_SIZE
    VECTOR: gl.constexpr = MAX_ROWS * OWNER_COLUMNS // (64 * WAVES)
    PACKS_PER_ROW: gl.constexpr = OWNER_COLUMNS // VECTOR
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [64, 1], [WAVES, 1], [1, 0])
    pack = gl.arange(0, 64 * WAVES, layout=gl.SliceLayout(1, layout))
    row = pack // PACKS_PER_ROW
    local = (
        owner * OWNER_COLUMNS
        + (pack % PACKS_PER_ROW)[:, None] * VECTOR
        + gl.arange(0, VECTOR, layout=gl.SliceLayout(0, layout))[None, :]
    )
    offsets = row[:, None] * OWNED + local
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SHARD_ELEMENTS
    empty = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for source in gl.static_range(WORLD_SIZE):
        gl.amd.cdna4.buffer_store(
            empty,
            region + ((stage + 2) % 3) * (2 * PHASE) + source * MAX_SHARD_ELEMENTS,
            offsets,
            cache=".wt",
        )
    columns = RANK * OWNED + local
    clamped = gl.minimum(row, rows - 1)
    h = gl.amd.cdna4.buffer_load(stream_ptr, clamped[:, None] * WIDTH + columns)
    alpha = gl.amd.cdna4.buffer_load(alpha_ptr, columns)
    beta = gl.amd.cdna4.buffer_load(beta_ptr, columns)
    if POST_HAS_WEIGHT:
        post = (
            (
                gl.amd.cdna4.buffer_load(post_weight_ptr, columns).to(gl.float32)
                + POST_WEIGHT_OFFSET
            )
            .to(gl.bfloat16)
            .to(gl.float32)
        )
    else:
        post = gl.full(columns.shape, 1.0, gl.float32, layout)
    valid = row < rows
    own = _muse_attntail_wait_buffer(
        region + stage * (2 * PHASE) + RANK * MAX_SHARD_ELEMENTS, offsets, valid, layout
    )
    b = iris_lamport_twoshot_rs_wait_reduce(
        region, stage, own, offsets, valid, RANK, WORLD_SIZE, MAX_SHARD_ELEMENTS, layout
    )
    iris_lamport_twoshot_ag_push(
        b,
        peers,
        stage,
        offsets,
        valid[:, None],
        gl.program_id(0),
        RANK,
        WORLD_SIZE,
        MAX_SHARD_ELEMENTS,
        PEER_SIGNAL_STORE_CACHE,
    )
    # The parts of this rank read its own columns from all-gather slot RANK too.
    gl.amd.cdna4.buffer_store(
        b,
        region + stage * (2 * PHASE) + PHASE + RANK * MAX_SHARD_ELEMENTS,
        offsets,
        mask=valid[:, None],
        cache=".wt",
    )
    weighted = b.to(gl.float32) * post
    alpha_h = alpha * h
    beta_b = beta * weighted
    four = gl.reshape(
        gl.join(
            gl.join(weighted * weighted, alpha_h * alpha_h),
            gl.join(alpha_h * beta_b, beta_b * beta_b),
        ),
        [MAX_ROWS, PACKS_PER_ROW, VECTOR, 4],
    )  # last axis: sum b^2, B, A, C
    # Two words per lane, consecutive lanes consecutive: whole 128 B lines.
    sums_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [WAVES, 1], [1, 0])
    sums = gl.convert_layout(gl.sum(gl.sum(four, 2), 1), sums_layout)  # [MAX_ROWS, 4]
    sum_row = gl.arange(0, MAX_ROWS, gl.SliceLayout(1, sums_layout))
    which = gl.arange(0, 4, gl.SliceLayout(0, sums_layout))
    OWNERS: gl.constexpr = OWNED // OWNER_COLUMNS
    # [every rank's owners][MAX_ROWS][4] on every rank: an owner's rows are
    # contiguous, so its store covers whole lines.
    slot = ((RANK * OWNERS + owner) * MAX_ROWS + sum_row[:, None]) * 4 + which[None, :]
    packed = _muse_attntail_pack_sums(tag, sums)
    mask = (sum_row < rows)[:, None] & (which >= 0)[None, :]
    for peer in gl.static_range(WORLD_SIZE):
        destination = gl.multiple_of(sums_peers[peer].to(gl.pointer_type(gl.int64)), 16)
        if peer == RANK:
            gl.amd.cdna4.buffer_store(packed, destination, slot, mask=mask, cache=".wt")
        else:
            gl.amd.cdna4.buffer_store(
                packed, destination, slot, mask=mask, cache=PEER_SIGNAL_STORE_CACHE
            )


@gluon.jit
def muse_attntail_finish(
    row,
    part,
    att_buffer,
    region,
    sums,
    tag,
    stage,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    PARTS: gl.constexpr,
    WIDTH: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    OWNER_COLUMNS: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
):
    """Finish columns [part * WIDTH / PARTS, ...) of row: the FP32 residual,
    router input and pre-expert input.

    Waits for the owners' reduced b of these columns (all-gather slots of
    their ranks) and every owner's four sums of this row, combines the sums
    in one fixed order (the same on every part and rank), then inv1 =
    rsqrt(sum b^2 / WIDTH + eps) and sum h'^2 = A + 2 inv1 B + inv1^2 C. No
    row-wide pass or join between parts. Afterwards restores the attention
    buffer's markers for its share of the row: every rank's owners pushed, so
    every local sender pushed and finished reading the buffer.
    """
    WAVES: gl.constexpr = gl.num_warps()
    OWNED: gl.constexpr = WIDTH // WORLD_SIZE
    COLUMNS: gl.constexpr = WIDTH // PARTS
    SOURCES: gl.constexpr = COLUMNS // OWNED
    VECTOR: gl.constexpr = OWNED // (64 * WAVES)
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [64, 1], [WAVES, 1], [1, 0])
    pack = gl.arange(0, 64 * WAVES, layout=gl.SliceLayout(1, layout))
    local = (
        pack[:, None] * VECTOR
        + gl.arange(0, VECTOR, layout=gl.SliceLayout(0, layout))[None, :]
    )
    valid = pack >= 0
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SHARD_ELEMENTS
    streams = ()
    alphas = ()
    betas = ()
    posts = ()
    pres = ()
    for s in gl.static_range(SOURCES):
        columns = (part * SOURCES + s) * OWNED + local
        streams += (gl.amd.cdna4.buffer_load(stream_ptr, row * WIDTH + columns),)
        alphas += (gl.amd.cdna4.buffer_load(alpha_ptr, columns),)
        betas += (gl.amd.cdna4.buffer_load(beta_ptr, columns),)
        if POST_HAS_WEIGHT:
            posts += (
                (
                    gl.amd.cdna4.buffer_load(post_weight_ptr, columns).to(gl.float32)
                    + POST_WEIGHT_OFFSET
                )
                .to(gl.bfloat16)
                .to(gl.float32),
            )
        else:
            posts += (gl.full(columns.shape, 1.0, gl.float32, layout),)
        pres += (
            (
                gl.amd.cdna4.buffer_load(pre_weight_ptr, columns).to(gl.float32)
                + PRE_WEIGHT_OFFSET
            )
            .to(gl.bfloat16)
            .to(gl.float32),
        )
    # Every owner's sums of this row: one tag half per lane, then the words.
    OWNERS: gl.constexpr = WORLD_SIZE * (OWNED // OWNER_COLUMNS)
    sums_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [WAVES, 1], [1, 0])
    owner = gl.arange(0, OWNERS, gl.SliceLayout(1, sums_layout))
    which = gl.arange(0, 4, gl.SliceLayout(0, sums_layout))
    index = (owner[:, None] * MAX_ROWS + row) * 4 + which[None, :]
    # System scope (SC0 SC1): the sums arrive from other GPUs and from other
    # XCDs of this one, and each XCD's L2 may hold an older copy of the line;
    # an SC0 SC1 load bypasses every L2 coherently (CDNA4 ISA 9.1.10.2).
    gl.atomic_poll(
        sums.to(gl.pointer_type(gl.int32), bitcast=True) + index * 2 + 1,
        tag,
        sem="relaxed",
        scope="sys",
    )
    # The poll is relaxed: the barrier orders the word loads after it.
    gl.barrier()
    words = gl.amd.cdna4.buffer_load(sums, index, cache=".cv")
    totals = gl.sum(
        words.to(gl.int32).to(gl.float32, bitcast=True), 0
    )  # [4]: sum b^2, B, A, C
    t0 = gl.sum(gl.where(which == 0, totals, 0.0), 0)
    t1 = gl.sum(gl.where(which == 1, totals, 0.0), 0)
    t2 = gl.sum(gl.where(which == 2, totals, 0.0), 0)
    t3 = gl.sum(gl.where(which == 3, totals, 0.0), 0)
    inv1 = gl.rsqrt(t0 / WIDTH + POST_EPS)
    inv2 = gl.rsqrt((t2 + 2.0 * inv1 * t1 + inv1 * inv1 * t3) / WIDTH + PRE_EPS)
    for s in gl.static_range(SOURCES):
        source = part * SOURCES + s
        b = _muse_attntail_wait_buffer(
            region + stage * (2 * PHASE) + PHASE + source * MAX_SHARD_ELEMENTS,
            row * OWNED + local,
            valid,
            layout,
        )
        columns = source * OWNED + local
        h = gl.fma(betas[s], b.to(gl.float32) * posts[s] * inv1, alphas[s] * streams[s])
        gl.amd.cdna4.buffer_store(h, out_stream_ptr, row * WIDTH + columns)
        x = h * inv2 * pres[s]
        gl.amd.cdna4.buffer_store(x, pre_fp32_ptr, row * WIDTH + columns)
        gl.amd.cdna4.buffer_store(
            x.to(gl.bfloat16), pre_bf16_ptr, row * WIDTH + columns
        )
    BUFFER_WORDS: gl.constexpr = 512 // PARTS
    word_layout: gl.constexpr = gl.BlockedLayout([1], [64], [WAVES], [0])
    word = gl.arange(0, 64 * WAVES, word_layout)
    gl.amd.cdna4.buffer_store(
        gl.full_like(word, _ATT_BUFFER_EMPTY).to(gl.uint32),
        att_buffer.to(gl.pointer_type(gl.uint32), bitcast=True),
        row * 512 + part * BUFFER_WORDS + word,
        mask=word < BUFFER_WORDS,
        cache=".wt",
    )


@gluon.jit
def muse_attntail_clear_gather(
    clearer,
    region,
    stage,
    CLEARERS: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
):
    """Restore 1 / CLEARERS of the all-gather slots of stage + 2 (written
    through): its readers are every rank's finishing parts, so no one reader
    can clear them after use; a peer writes that stage again only two
    launches on."""
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SHARD_ELEMENTS
    CHUNK: gl.constexpr = PHASE // CLEARERS
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    offsets = clearer * CHUNK + gl.arange(0, CHUNK, layout)
    gl.amd.cdna4.buffer_store(
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True),
        region + ((stage + 2) % 3) * (2 * PHASE) + PHASE,
        offsets,
        cache=".wt",
    )


@gluon.jit(do_not_specialize=("rows",))
def iris_fused_sliding_attention_twoshot_muse_attntail(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    # Const, as in the standalone split: LLVM follows a volatile page-index
    # load from non-const memory with vmcnt(0), which would drain the K/V
    # page copies the split keeps in flight.
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    counters_ptr,
    gate_ptr,
    gate_stride,
    weight_ptr,
    att_buffer,
    epochs,
    region,
    rs_peers,
    sums,
    sums_peers,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    rows,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    NUM_REQUESTS: gl.constexpr,
    WIDTH: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    PARTS: gl.constexpr,
    GATE_EPS: gl.constexpr,
    POST_EPS: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    PRE_EPS: gl.constexpr,
    PRE_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
    RESERVED_LDS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Sliding decode attention through Muse's attention boundary in one launch.

    Grid of 256 workgroups x num_warps, roles by program id (every wait points
    to a lower id, so the launch progresses without co-residency):
    - [0, A): the sliding attention splits (A = requests x KV heads x splits);
      the last split of a row block publishes its gated, normalized rows to
      att_buffer (muse_attntail_attention).
    - [A, A + 128): senders of 64 o_proj columns each; their weights load at
      entry, while attention runs (muse_attntail_o_proj_prefetch / _send).
    - [A + 128, A + 136): owners of 128 of this rank's WIDTH / WORLD_SIZE
      columns: the reduce-scatter half and the norm sums (muse_attntail_own).
    - [A + 136, A + 136 + rows x PARTS): (row, part) finishing the boundary
      from the all-gather (muse_attntail_finish).
    - The last 8: all-gather clearers (muse_attntail_clear_gather); the rest
      only advance their epoch.
    The two-shot region is #2568's layout ([stage][reduce-scatter,
    all-gather][source][shard]) with column shards of [rows, WIDTH /
    WORLD_SIZE]. Every workgroup reads its own epoch word at entry and stores
    it plus one (modulo _SLIDING_TAIL_EPOCH_PERIOD) at exit, so all agree on
    stage = epoch % 3 and tag = epoch + 1. rows is at most MAX_ROWS and PARTS
    divides WORLD_SIZE. RESERVED_LDS keeps every workgroup alone on its
    compute unit, so no sender shares one with an attention split.
    """
    gl.static_assert(WIDTH == 8192 and WORLD_SIZE == 8)
    cfg: gl.constexpr = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        gl.num_warps(),
        True,
        WINDOW_LEFT,
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
    )
    HEAD_BLOCKS: gl.constexpr = NUM_KV_HEADS * cfg.NUM_ROW_BLOCKS
    ATTENTION: gl.constexpr = NUM_REQUESTS * HEAD_BLOCKS * NUM_KV_SPLITS
    SENDERS: gl.constexpr = 128
    OUTPUTS: gl.constexpr = WIDTH // SENDERS
    OWNER_COLUMNS: gl.constexpr = 128
    OWNERS: gl.constexpr = WIDTH // WORLD_SIZE // OWNER_COLUMNS
    CLEARERS: gl.constexpr = 8
    # Each request verifies MAX_SEQLEN_Q rows; every role must fit the grid.
    gl.static_assert(
        ATTENTION + SENDERS + OWNERS + NUM_REQUESTS * MAX_SEQLEN_Q * PARTS + CLEARERS
        <= 256
    )
    K: gl.constexpr = NUM_Q_HEADS * HEAD_DIM
    program = gl.program_id(0)
    # Unsigned, so a word outside [0, period) still gives a stage in [0, 3).
    epoch = gl.load(epochs + program).to(gl.uint32)
    stage = (epoch % 3).to(gl.int32)
    tag = (epoch + 1).to(gl.int32)
    reserved = gl.allocate_shared_memory(
        gl.int32, [RESERVED_LDS // 4], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    if program < 0:
        # Never runs; keeps the reservation.
        reserved.store(
            gl.zeros(
                [RESERVED_LDS // 4],
                gl.int32,
                gl.BlockedLayout([1], [64], [gl.num_warps()], [0]),
            )
        )
    if program < ATTENTION:
        # The 3-D split grid's order: request fastest, then head block, then split.
        muse_attntail_attention(
            program % NUM_REQUESTS,
            (program // NUM_REQUESTS) % HEAD_BLOCKS,
            program // (NUM_REQUESTS * HEAD_BLOCKS),
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            cache_seqlens_ptr,
            mid_o_ptr,
            mid_lse_ptr,
            counters_ptr,
            gate_ptr,
            gate_stride,
            att_buffer,
            cfg,
            GATE_EPS,
            NUM_STAGES,
        )
    elif program < ATTENTION + SENDERS:
        block = program - ATTENTION
        tiles = muse_attntail_o_proj_prefetch(weight_ptr, block, K, OUTPUTS)
        muse_attntail_send(
            tiles,
            block,
            att_buffer,
            rs_peers,
            stage,
            rows,
            K,
            OUTPUTS,
            WIDTH,
            RANK,
            WORLD_SIZE,
            MAX_SHARD_ELEMENTS,
            PEER_SIGNAL_STORE_CACHE,
        )
    elif program < ATTENTION + SENDERS + OWNERS:
        muse_attntail_own(
            program - ATTENTION - SENDERS,
            region,
            rs_peers,
            sums_peers,
            tag,
            stage,
            rows,
            stream_ptr,
            post_weight_ptr,
            alpha_ptr,
            beta_ptr,
            WIDTH,
            RANK,
            WORLD_SIZE,
            MAX_SHARD_ELEMENTS,
            MAX_ROWS,
            OWNER_COLUMNS,
            POST_WEIGHT_OFFSET,
            POST_HAS_WEIGHT,
            PEER_SIGNAL_STORE_CACHE,
        )
    elif program < ATTENTION + SENDERS + OWNERS + rows * PARTS:
        index = program - ATTENTION - SENDERS - OWNERS
        muse_attntail_finish(
            index % rows,
            index // rows,
            att_buffer,
            region,
            sums,
            tag,
            stage,
            stream_ptr,
            post_weight_ptr,
            alpha_ptr,
            beta_ptr,
            pre_weight_ptr,
            out_stream_ptr,
            pre_fp32_ptr,
            pre_bf16_ptr,
            PARTS,
            WIDTH,
            WORLD_SIZE,
            MAX_SHARD_ELEMENTS,
            MAX_ROWS,
            OWNER_COLUMNS,
            POST_EPS,
            POST_WEIGHT_OFFSET,
            PRE_EPS,
            PRE_WEIGHT_OFFSET,
            POST_HAS_WEIGHT,
        )
    elif program >= gl.num_programs(0) - CLEARERS:
        muse_attntail_clear_gather(
            program - (gl.num_programs(0) - CLEARERS),
            region,
            stage,
            CLEARERS,
            WORLD_SIZE,
            MAX_SHARD_ELEMENTS,
        )
    # A wave loading the epoch after this store would use the next stage;
    # each has used its stage before this barrier.
    gl.barrier()
    gl.store(epochs + program, ((epoch + 1) % _SLIDING_TAIL_EPOCH_PERIOD).to(gl.int32))
