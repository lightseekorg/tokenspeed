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
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv_splitk import (
    gemv_a16w16_splitk_hi_lo_sums_gfx950,
    gemv_a16w16_splitk_publish_gfx950,
    gemv_a16w16_splitk_wave_sums_gfx950,
    gemv_a16w16_splitk_weights_gfx950,
    gemv_a16w16_splitk_x_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.layernorm.deferred import (
    deferred_norm_rows_gfx950,
    deferred_norm_scales_gfx950,
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
        if not poll_tags16(record_ptr + first + 16 * line, tag, launch, True):
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
