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
    _iris_drain_subgroup_vmem,
    _iris_group_heap_base,
    _iris_rendezvous,
)


@gluon.jit
def _iris_muse_attntail(
    branch,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
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
    """
    offsets = (row * WIDTH + columns).to(gl.int32)
    branch = branch.to(gl.bfloat16).to(gl.float32)
    if POST_HAS_WEIGHT:
        weight = gl.amd.cdna4.buffer_load(post_weight_ptr, columns).to(gl.float32)
        branch *= (weight + POST_WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    branch *= gl.rsqrt(gl.sum(branch * branch, 0) / WIDTH + POST_EPS)
    stream = gl.amd.cdna4.buffer_load(stream_ptr, offsets)
    alpha = gl.amd.cdna4.buffer_load(alpha_ptr, columns)
    beta = gl.amd.cdna4.buffer_load(beta_ptr, columns)
    stream = gl.fma(beta, branch, alpha * stream)
    gl.amd.cdna4.buffer_store(stream, out_stream_ptr, offsets)
    pre = stream * gl.rsqrt(gl.sum(stream * stream, 0) / WIDTH + PRE_EPS)
    weight = gl.amd.cdna4.buffer_load(pre_weight_ptr, columns).to(gl.float32)
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
    _iris_drain_subgroup_vmem()
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
            _iris_muse_attntail(
                branch,
                stream_ptr,
                branch_sym_ptr,
                alpha_ptr,
                beta_ptr,
                pre_weight_ptr,
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
def iris_fused_push_oneshot_allreduce_muse_attntail(
    branch_ptr,
    inbox_ptr,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    pre_fp32_ptr,
    pre_bf16_ptr,
    generations,
    ready_flags,
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
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """All-reduce TP4/TP8 attention contributions and update Muse's residuals.

    Push each BF16 branch_ptr row into peer inboxes, sum in FP32 rank order, and
    round to BF16 before post-normalization. Apply the alpha/beta residual update
    and MoE input norm, writing FP32 residual/router and BF16 pre-expert projection outputs.
    Per-row generations and two alternating inbox slots protect successive uses.
    """
    gl.static_assert(WORLD_SIZE == 4 or WORLD_SIZE == 8)
    gl.static_assert(WIDTH == 8192)
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [64], [NUM_WARPS], [0]
    )
    columns = gl.arange(0, WIDTH, layout=layout)
    offsets = (row * WIDTH + columns).to(gl.int32)
    local = gl.amd.cdna4.buffer_load(branch_ptr, offsets)
    generation = gl.load(generations + row) + 1
    slot = generation & 1
    local_heap = _iris_group_heap_base(RANK, heaps)
    inbox_offset = inbox_ptr.to(gl.uint64) - local_heap
    slot_offset = slot * WORLD_SIZE * MAX_ROWS * WIDTH
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        peer_heap = _iris_group_heap_base(peer, heaps)
        destination = gl.multiple_of(
            (peer_heap + inbox_offset).to(gl.pointer_type(gl.bfloat16)), 16
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination + slot_offset + RANK * MAX_ROWS * WIDTH,
            offsets,
            cache=".wt",
        )
    _iris_drain_subgroup_vmem()
    gl.barrier()
    _iris_rendezvous(
        ready_flags + slot * MAX_ROWS * WORLD_SIZE,
        heaps,
        row,
        generation,
        RANK,
        NUM_WARPS,
        False,
    )
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            value = local.to(gl.float32)
        else:
            value = gl.amd.cdna4.buffer_load(
                inbox_ptr + slot_offset + peer * MAX_ROWS * WIDTH,
                offsets,
                cache=".cg",
            ).to(gl.float32)
        if peer == 0:
            branch = value
        else:
            branch += value
    _iris_muse_attntail(
        branch,
        stream_ptr,
        post_weight_ptr,
        alpha_ptr,
        beta_ptr,
        pre_weight_ptr,
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
    # Two slots make the next entry handshake reclaim this row's old inbox.
    gl.store(generations + row, generation)
