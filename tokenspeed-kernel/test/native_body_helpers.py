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

"""Test kernels around the native decoder layer's device bodies (gfx950).

The bodies are device functions with no program ids of their own; these
kernels give them one role per program so the bodies can be compiled offline
(codegen tests) and run against today's kernels (numerics tests).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import InputStrides
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    AttentionConfig,
    AttentionProgram,
    mha_decode_sliding_split_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.prologue.gqa import (
    gqa_prologue_key_gfx950,
    gqa_prologue_query_gfx950,
    gqa_prologue_rope_gfx950,
    gqa_prologue_rows_layout,
    gqa_prologue_value_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv_splitk import (
    gemv_a16w16_splitk_finish_x_gfx950,
    gemv_a16w16_splitk_hi_lo_sums_gfx950,
    gemv_a16w16_splitk_publish_gfx950,
    gemv_a16w16_splitk_sum_gfx950,
    gemv_a16w16_splitk_wave_sums_gfx950,
    gemv_a16w16_splitk_weights_gfx950,
    gemv_a16w16_splitk_weights_paced_gfx950,
    gemv_a16w16_splitk_x_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    mm_a16w16_column_relay_gfx950,
    mm_a16w16_column_relay_weights_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.layernorm.deferred import (
    deferred_norm_rows_gfx950,
    deferred_norm_scales_gfx950,
    deferred_norm_terms_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
    sigmoid_bias_topk_keys_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent import (
    ERROR_ABORT,
    FIRST_KERNEL_SITE,
    Launch,
    fail_stop,
    poll_tags16,
)

_ERROR_ABORT = gl.constexpr(ERROR_ABORT)
_RECORD_SITE = gl.constexpr(FIRST_KERNEL_SITE)


def compile_gfx950(fn, signature: dict, constexprs: dict, num_warps: int, **options):
    """Compile ``fn`` for gfx950 without a GPU and return the compiled kernel.

    Every pointer argument is declared 16-byte aligned, as a launch with
    real tensors specializes it, so buffer and LDS-direct copies vectorize.
    """
    names = list(fn.arg_names)
    attrs = {
        (names.index(name),): [["tt.divisibility", 16]]
        for name, kind in signature.items()
        if kind.startswith("*")
    }
    full = dict(signature)
    full.update({name: "constexpr" for name in constexprs})
    source = gluon._runtime.GluonASTSource(fn, full, constexprs=constexprs, attrs=attrs)
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": num_warps, **options},
    )


@gluon.jit
def test_launch(error_ptr, tag, deadline_ticks):
    """A ``Launch`` for a test grid that has no persistent entry: the
    workgroup's linear program id as its identity, ``error_ptr`` (int32,
    ``ERROR_WORDS`` long, zeroed) for the failure record and abort word."""
    pid = (gl.program_id(0) * gl.num_programs(1) + gl.program_id(1)) * gl.num_programs(
        2
    ) + gl.program_id(2)
    zero = pid - pid
    return Launch(
        pid,
        pid % 8,
        pid // 8,
        zero.to(gl.int64),
        tag,
        zero,
        zero,
        error_ptr,
        error_ptr + _ERROR_ABORT,
        deadline_ticks,
        error_ptr,
    )


@gluon.jit
def wait_records_tags16(record_ptr, first, count: gl.constexpr, tag, launch):
    """The layer's record poll: ``poll_tags16`` (scalar ``s_load_dwordx16
    glc``) over each 64-byte line of ``count`` int32 records; on expiry the
    launch's ``fail_stop``."""
    gl.static_assert(count % 16 == 0)
    for line in gl.static_range(count // 16):
        # Every wave polls every line itself, so each ends with its own acquire.
        if not poll_tags16(record_ptr + first + 16 * line, tag, launch, True, False):
            fail_stop(launch, _RECORD_SITE, line)


@gluon.jit
def split_records_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    record_ptr,
    error_ptr,
    tag,
    deadline_ticks,
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
    NUM_WARPS: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """The sliding split decode with records: every program streams its split
    and publishes its record; split 0 merges once all records carry ``tag``.
    The grid is ``_mha_decode_sliding_split_atomic``'s; all programs must be
    co-resident."""
    cfg = AttentionConfig(
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
        NUM_WARPS,
        True,
        WINDOW_LEFT,
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        gl.program_id(0),
        gl.program_id(1),
        gl.program_id(2),
    )
    launch = test_launch(error_ptr, tag, deadline_ticks)
    mha_decode_sliding_split_gfx950(
        program, q_ptr, k_cache_ptr, v_cache_ptr, NUM_STAGES
    )
    program.publish_split_record(record_ptr, launch.tag)
    if program.split_id == 0:
        merged = program.merge_split_records(
            record_ptr, launch.tag, q_ptr, False, wait_records_tags16, launch
        )
        program.store_merged(out_ptr, merged)


@gluon.constexpr_function
def sums_layout(waves: int, rows: int, vector: int):
    """``gemv_a16w16_splitk_wave_sums_gfx950``'s ``OUT_LAYOUT`` for 16 tokens
    by ``rows`` columns: ``vector`` contiguous columns per thread."""
    lanes = rows // vector
    return gl.BlockedLayout(
        [waves, 1, vector], [1, 64 // lanes, lanes], [1, waves, 1], [2, 1, 0]
    )


@gluon.jit
def sender_kernel(
    w_ptr,
    x_ptr,
    out_ptr,
    M,
    K: gl.constexpr,
    OUTPUTS: gl.constexpr,
    VECTOR: gl.constexpr,
):
    """The native layer's o_proj sender (SND) on a plain x: program b issues
    weight rows ``[b * OUTPUTS, ...)`` over the whole K, multiplies them by
    the first 16 rows of x (rows past ``M`` reread row ``M - 1``) and stores
    the BF16 sums ``[16, OUTPUTS]`` at ``out_ptr`` row-major per program."""
    block = gl.program_id(0)
    tiles = gemv_a16w16_splitk_weights_gfx950(
        w_ptr, block, K, OUTPUTS, 1, gl.num_programs(0)
    )
    x = gemv_a16w16_splitk_x_gfx950(x_ptr, K, block, M, K, 1, gl.num_programs(0), "")
    layout: gl.constexpr = sums_layout(gl.num_warps(), OUTPUTS, VECTOR)
    sums = gemv_a16w16_splitk_wave_sums_gfx950(tiles, x, layout)
    out_l: gl.constexpr = gl.SliceLayout(0, layout)
    token = gl.arange(0, 16, gl.SliceLayout(1, out_l))
    column = gl.arange(0, OUTPUTS, gl.SliceLayout(0, out_l))
    gl.amd.cdna4.buffer_store(
        sums.to(gl.bfloat16),
        out_ptr,
        (block * 16 + token[:, None]) * OUTPUTS + column[None, :],
    )


@gluon.jit
def _kpart_x(
    b_ptr,
    sums_ptr,
    h_ptr,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    pre_ptr,
    task,
    M,
    WIDTH: gl.constexpr,
    PRODUCERS: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """A K-part consumer's x (FP32 B operand) from plain stand-ins of what the
    layer polls: the branch rows ``b`` (BF16 ``[M, WIDTH]``) and the
    producers' sums (FP32 ``[M, PRODUCERS, 4]``)."""
    WAVES: gl.constexpr = gl.num_warps()
    sums_l: gl.constexpr = gl.BlockedLayout(
        [16 // WAVES, 1, 4], [1, 64, 1], [WAVES, 1, 1], [2, 1, 0]
    )
    t = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, sums_l)))
    p = gl.arange(0, PRODUCERS, gl.SliceLayout(0, gl.SliceLayout(2, sums_l)))
    w = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, sums_l)))
    t = gl.minimum(t, M - 1)
    sums = gl.amd.cdna4.buffer_load(
        sums_ptr,
        (t[:, None, None] * PRODUCERS + p[None, :, None]) * 4 + w[None, None, :],
    )
    inv1, inv2 = deferred_norm_scales_gfx950(sums, WIDTH, POST_EPS, PRE_EPS)
    b = gemv_a16w16_splitk_x_gfx950(b_ptr, WIDTH, task, M, WIDTH, PARTS, GROUPS, "")
    h = gemv_a16w16_splitk_x_gfx950(h_ptr, WIDTH, task, M, WIDTH, PARTS, GROUPS, "")
    alpha = gemv_a16w16_splitk_x_gfx950(alpha_ptr, 0, task, M, WIDTH, PARTS, GROUPS, "")
    beta = gemv_a16w16_splitk_x_gfx950(beta_ptr, 0, task, M, WIDTH, PARTS, GROUPS, "")
    post = gemv_a16w16_splitk_x_gfx950(post_ptr, 0, task, M, WIDTH, PARTS, GROUPS, "")
    pre = gemv_a16w16_splitk_x_gfx950(pre_ptr, 0, task, M, WIDTH, PARTS, GROUPS, "")
    token_l: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, b.type.layout))
    inv1 = gl.convert_layout(inv1, token_l)[None, None, :]
    inv2 = gl.convert_layout(inv2, token_l)[None, None, :]
    return deferred_norm_rows_gfx950(b, h, alpha, beta, post, pre, inv1, inv2)


@gluon.jit
def kpart_router_kernel(
    b_ptr,
    sums_ptr,
    h_ptr,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    pre_ptr,
    hi_ptr,
    lo_ptr,
    part_ptr,
    tag,
    M,
    WIDTH: gl.constexpr,
    EXPERTS: gl.constexpr,
    PRODUCERS: gl.constexpr,
    PARTS: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """The router K-part task (RTR): 16 experts x WIDTH / PARTS of the FP32
    router as BF16 hi + lo, over the FP32 x this K part builds."""
    task = gl.program_id(0)
    GROUPS: gl.constexpr = EXPERTS // 16
    hi = gemv_a16w16_splitk_weights_gfx950(hi_ptr, task, WIDTH, 16, PARTS, GROUPS)
    lo = gemv_a16w16_splitk_weights_gfx950(lo_ptr, task, WIDTH, 16, PARTS, GROUPS)
    _, x = _kpart_x(
        b_ptr, sums_ptr, h_ptr, alpha_ptr, beta_ptr, post_ptr, pre_ptr, task, M,
        WIDTH, PRODUCERS, PARTS, GROUPS, POST_EPS, PRE_EPS,
    )  # fmt: skip
    layout: gl.constexpr = sums_layout(gl.num_warps(), 16, 2)
    logits = gemv_a16w16_splitk_hi_lo_sums_gfx950(hi, lo, x, layout)
    gemv_a16w16_splitk_publish_gfx950(
        logits, part_ptr, tag, (task % GROUPS) * 16, task // GROUPS, M, EXPERTS, ".wt"
    )


@gluon.jit
def kpart_pooler_kernel(
    b_ptr,
    sums_ptr,
    h_ptr,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    pre_ptr,
    w_ptr,
    part_ptr,
    tag,
    M,
    WIDTH: gl.constexpr,
    OUTPUTS: gl.constexpr,
    ROWS: gl.constexpr,
    PRODUCERS: gl.constexpr,
    PARTS: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """The pooler K-part task (POOL): ``ROWS`` pooled columns x WIDTH / PARTS
    of the BF16 pooler over the BF16-rounded x this K part builds."""
    task = gl.program_id(0)
    GROUPS: gl.constexpr = OUTPUTS // ROWS
    tiles = gemv_a16w16_splitk_weights_gfx950(w_ptr, task, WIDTH, ROWS, PARTS, GROUPS)
    _, x = _kpart_x(
        b_ptr, sums_ptr, h_ptr, alpha_ptr, beta_ptr, post_ptr, pre_ptr, task, M,
        WIDTH, PRODUCERS, PARTS, GROUPS, POST_EPS, PRE_EPS,
    )  # fmt: skip
    layout: gl.constexpr = sums_layout(gl.num_warps(), ROWS, 2)
    pooled = gemv_a16w16_splitk_wave_sums_gfx950(tiles, x.to(gl.bfloat16), layout)
    gemv_a16w16_splitk_publish_gfx950(
        pooled, part_ptr, tag, (task % GROUPS) * ROWS, task // GROUPS, M, OUTPUTS, ".wt"
    )


@gluon.jit
def topk_keys_kernel(
    logits_ptr,
    bias_ptr,
    ids_ptr,
    weights_ptr,
    M,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
):
    """The native layer's TOPK role: one wave per row, rows ``program *
    num_warps ..`` of FP32 logits ``[M, E]``; normalized weights."""
    WAVES: gl.constexpr = gl.num_warps()
    layout: gl.constexpr = gl.BlockedLayout([1, E // 64], [1, 64], [WAVES, 1], [1, 0])
    out_l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [WAVES, 1], [1, 0])
    row = gl.program_id(0) * WAVES + gl.arange(0, WAVES, gl.SliceLayout(1, layout))
    expert = gl.arange(0, E, gl.SliceLayout(0, layout))
    logits = gl.amd.cdna4.buffer_load(
        logits_ptr,
        gl.minimum(row, M - 1)[:, None] * E + expert[None, :],
    )
    ids, weights = sigmoid_bias_topk_keys_gfx950(
        logits, bias_ptr, E, TOPK, TOPK, True, ROUTED_SCALING_FACTOR, gl.float32, out_l
    )
    out_row = gl.program_id(0) * WAVES + gl.arange(0, WAVES, gl.SliceLayout(1, out_l))
    pick = gl.arange(0, TOPK, gl.SliceLayout(0, out_l))
    offsets = out_row[:, None] * TOPK + pick[None, :]
    mask = (out_row < M)[:, None] & (pick >= 0)[None, :]
    gl.amd.cdna4.buffer_store(ids, ids_ptr, offsets, mask=mask)
    gl.amd.cdna4.buffer_store(weights, weights_ptr, offsets, mask=mask)


@gluon.jit
def qkv_task_kernel(
    b_ptr,
    sums_ptr,
    h_ptr,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    pre_ptr,
    w_ptr,
    part_ptr,
    tag,
    M,
    WIDTH: gl.constexpr,
    N: gl.constexpr,
    PRODUCERS: gl.constexpr,
    WINDOW: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """The native layer's QKV task: one 144-row x WIDTH / 16 task per
    program, its weights issued at most ``WINDOW`` tiles ahead, x built from
    the FFN boundary's deferred norm, fp32 partials published for the head
    tasks."""
    task = gl.program_id(0)
    ROWS: gl.constexpr = N // 16
    tiles = gemv_a16w16_splitk_weights_paced_gfx950(
        w_ptr, task, WIDTH, ROWS, 16, 16, WINDOW
    )
    _, x = _kpart_x(
        b_ptr, sums_ptr, h_ptr, alpha_ptr, beta_ptr, post_ptr, pre_ptr, task, M,
        WIDTH, PRODUCERS, 16, 16, POST_EPS, PRE_EPS,
    )  # fmt: skip
    gemv_a16w16_splitk_finish_x_gfx950(
        tiles, x.to(gl.bfloat16), part_ptr, tag, task, M, N, ROWS, 16, ".wt"
    )


@gluon.jit
def _head_rows(
    part_ptr, tag, column0, launch, M, N: gl.constexpr, HEAD_DIM: gl.constexpr
):
    """One head's 16 x HEAD_DIM rows from the QKV tasks' 16 K-part partials,
    summed in part order and rounded once to BF16 (as FP32), in the
    prologue's rows layout."""
    rows_l: gl.constexpr = gqa_prologue_rows_layout(gl.num_warps())
    flat_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    index = gl.arange(0, 16 * HEAD_DIM, flat_l)
    token = index // HEAD_DIM
    sums = gemv_a16w16_splitk_sum_gfx950(
        part_ptr,
        tag,
        token * N + column0 + index % HEAD_DIM,
        token < M,
        M,
        N,
        16,
        launch,
        fail_stop,
        _RECORD_SITE,
    )
    rows = gl.reshape(sums.to(gl.bfloat16).to(gl.float32), [16, HEAD_DIM])
    return gl.convert_layout(rows, rows_l)


@gluon.jit
def head_task_kernel(
    part_ptr,
    q_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    error_ptr,
    tag,
    deadline_ticks,
    M,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    """The native layer's HEAD tasks: program h < NUM_Q_HEADS + NUM_KV_HEADS
    sums its head's QKV partials (the K-split task's tagged words) and runs
    the GQA prologue on them in registers (FP8 query, FP8 cache)."""
    N: gl.constexpr = (
        NUM_Q_HEADS + 2 * NUM_KV_HEADS
    ) * HEAD_DIM + NUM_Q_HEADS * HEAD_DIM
    head = gl.program_id(0)
    launch = test_launch(error_ptr, tag, deadline_ticks)
    rows_l: gl.constexpr = gqa_prologue_rows_layout(gl.num_warps())
    tokens = gl.arange(0, 16, gl.SliceLayout(1, rows_l)).to(gl.int64)
    row_mask = tokens < M
    # Operands first: they do not wait on the partials.
    cos, sin = gqa_prologue_rope_gfx950(
        cos_sin_ptr, HEAD_DIM, positions_ptr, tokens, row_mask, HEAD_DIM // 2
    )
    slot = gl.load(slots_ptr + tokens, mask=row_mask, other=0).to(gl.int64)
    if head < NUM_Q_HEADS:
        rows = _head_rows(part_ptr, tag, head * HEAD_DIM, launch, M, N, HEAD_DIM)
        gqa_prologue_query_gfx950(
            rows,
            q_out_ptr + tokens * (NUM_Q_HEADS * HEAD_DIM) + head * HEAD_DIM,
            q_weight_ptr,
            cos,
            sin,
            row_mask,
            True,
            1e-6,
            0.0,
        )
    else:
        kv = head - NUM_Q_HEADS
        k_rows = _head_rows(
            part_ptr, tag, (NUM_Q_HEADS + kv) * HEAD_DIM, launch, M, N, HEAD_DIM
        )
        gqa_prologue_key_gfx950(
            k_rows,
            k_cache_ptr + (slot * NUM_KV_HEADS + kv) * HEAD_DIM,
            k_weight_ptr,
            cos,
            sin,
            row_mask,
            1.0,
            True,
            1e-6,
            0.0,
            False,
        )
        v_rows = _head_rows(
            part_ptr,
            tag,
            (NUM_Q_HEADS + NUM_KV_HEADS + kv) * HEAD_DIM,
            launch,
            M,
            N,
            HEAD_DIM,
        )
        gqa_prologue_value_gfx950(
            v_rows,
            v_cache_ptr + (slot * NUM_KV_HEADS + kv) * HEAD_DIM,
            row_mask,
            1.0,
            False,
        )


@gluon.jit
def column_relay_kernel(x_ptr, w_ptr, out_ptr, M, K: gl.constexpr):
    """The native layer's post-expert sender (UNP): program b relays the
    16-column tile ``[16 b, 16 b + 16)`` of ``x [M, K] @ w.T`` across its
    waves and stores the rows below ``M`` in BF16."""
    WAVES: gl.constexpr = gl.num_warps()
    column0 = gl.program_id(0) * 16
    w = mm_a16w16_column_relay_weights_gfx950(w_ptr, column0, K)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WAVES, 1, 1],
    )
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    token = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    k = gl.arange(0, K // WAVES, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    s = gl.amd.cdna4.buffer_load(
        x_ptr,
        token[None, :, None] * K
        + (wave * (K // WAVES))[:, None, None]
        + k[None, None, :],
        mask=(token < M)[None, :, None] & (k >= 0)[None, None, :],
        other=0.0,
    )
    relay = gl.allocate_shared_memory(
        gl.float32, [WAVES, 16, 16], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    )
    relay_count = gl.allocate_shared_memory(
        gl.int32, [1], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    relay_count.store(
        gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [WAVES], [0]))
    )
    gl.barrier()
    out_l: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WAVES, 1], [1, 0])
    y = mm_a16w16_column_relay_gfx950(s, w, relay, relay_count, out_l)
    t = gl.arange(0, 16, gl.SliceLayout(1, out_l))
    c = gl.arange(0, 16, gl.SliceLayout(0, out_l))
    N = gl.num_programs(0) * 16
    gl.amd.cdna4.buffer_store(
        y.to(gl.bfloat16),
        out_ptr,
        t[:, None] * N + column0 + c[None, :],
        mask=(t < M)[:, None] & (c >= 0)[None, :],
    )


@gluon.jit
def sender_sums_kernel(
    x_ptr,
    w_ptr,
    h_ptr,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    sums_ptr,
    tag,
    M,
    K: gl.constexpr,
    N: gl.constexpr,
):
    """The post-expert sender's partial norm sums: the relayed 16-column
    tile rounded to BF16 (the H6 value), and per row the FFN boundary's four
    sums over those columns (``deferred_norm_terms_gfx950``), published as
    tagged FP32 words ``[program, row, 4]`` for the sum reducer."""
    WAVES: gl.constexpr = gl.num_warps()
    column0 = gl.program_id(0) * 16
    w = mm_a16w16_column_relay_weights_gfx950(w_ptr, column0, K)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WAVES, 1, 1],
    )
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    token = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    k = gl.arange(0, K // WAVES, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    s = gl.amd.cdna4.buffer_load(
        x_ptr,
        token[None, :, None] * K
        + (wave * (K // WAVES))[:, None, None]
        + k[None, None, :],
        mask=(token < M)[None, :, None] & (k >= 0)[None, None, :],
        other=0.0,
    )
    relay = gl.allocate_shared_memory(
        gl.float32, [WAVES, 16, 16], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    )
    relay_count = gl.allocate_shared_memory(
        gl.int32, [1], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    relay_count.store(
        gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [WAVES], [0]))
    )
    gl.barrier()
    out_l: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [WAVES, 1], [1, 0])
    y = mm_a16w16_column_relay_gfx950(s, w, relay, relay_count, out_l)
    t = gl.minimum(gl.arange(0, 16, gl.SliceLayout(1, out_l)), M - 1)
    c = column0 + gl.arange(0, 16, gl.SliceLayout(0, out_l))
    h = gl.amd.cdna4.buffer_load(h_ptr, t[:, None] * N + c[None, :])
    alpha = gl.amd.cdna4.buffer_load(alpha_ptr, c)[None, :]
    beta = gl.amd.cdna4.buffer_load(beta_ptr, c)[None, :]
    post = gl.amd.cdna4.buffer_load(post_ptr, c)[None, :]
    terms = deferred_norm_terms_gfx950(y.to(gl.bfloat16), h, alpha, beta, post)
    sums = gl.sum(terms, axis=1)  # [16, 4]: sum b^2, B, A, C over the 16 columns
    SL: gl.constexpr = sums.type.layout
    row = gl.arange(0, 16, gl.SliceLayout(1, SL))
    which = gl.arange(0, 4, gl.SliceLayout(0, SL))
    word = (tag.to(gl.int64) << 32) | (
        sums.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF
    )
    gl.amd.cdna4.buffer_store(
        word,
        sums_ptr,
        (gl.program_id(0) * 16 + row[:, None]) * 4 + which[None, :],
        mask=(row < M)[:, None] & (which >= 0)[None, :],
        cache=".wt",
    )
