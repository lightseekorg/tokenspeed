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

"""Warp-pipelined MLA prefill Gluon kernel for AMD GFX950.

The two-wave ping-pong design is inspired by and adapted from the flash
attention kernels in ROCm/gfx950-gluon-tutorials. The previous kernel stays in
``prefill.py``; this module reuses its per-program helpers and persistent work
scheduler.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.attention._common import (
    _INV_LN2,
    InputStrides,
    attention_layouts,
    max,
    padded_shared_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mla.prefill import (
    AttentionProgram,
    LaunchConfig,
    ProgramScheduler,
)

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy


# ===-----------------------------------------------------------------------===#
# Kernel Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class Prefill8WaveConfig:
    # Carries the fields AttentionProgram and ProgramScheduler read, with the
    # layouts of this kernel's 256-row query blocks and key tiles.
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    HEAD_DIM: gl.constexpr
    ROPE_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    IS_CAUSAL: gl.constexpr
    HAS_LSE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    BATCH_SIZE: gl.constexpr
    NUM_XCDS: gl.constexpr
    NUM_BLOCKS: gl.constexpr
    IS_FP8: gl.constexpr
    RESCALE_THRESHOLD: gl.constexpr
    q_strides: InputStrides
    k_strides: InputStrides
    v_strides: InputStrides
    o_strides: InputStrides
    lse_strides: InputStrides
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    q_pe_layout: gl.constexpr
    k_pe_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    load_pe_layout: gl.constexpr
    store_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    k_pe_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        ROPE_DIM,
        SM_SCALE,
        IS_CAUSAL,
        HAS_LSE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        BATCH_SIZE,
        IS_FP8,
        KV_DTYPE,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
    ):
        assert HEAD_DIM == 128
        assert ROPE_DIM == 64
        # Each of the 8 waves owns 32 query rows of one 32x32 MFMA row block.
        # Key tiles are as wide as the PV MFMA's K dimension allows while a
        # wave stays within 256 VGPRs (see the loop comment): 32 keys for
        # 16-bit inputs and 64, one K=64 MFMA step, for FP8.
        assert NUM_WARPS == 8
        assert BLOCK_M == 32 * NUM_WARPS
        assert BLOCK_N == (64 if IS_FP8 else 32)

        (
            qk_layout,
            pv_layout,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            load_layout,
            _,
            k_smem_layout,
            v_smem_layout,
        ) = attention_layouts(
            HEAD_DIM,
            BLOCK_N,
            IS_FP8,
            KV_DTYPE,
            num_warps=NUM_WARPS,
            instr_shape=[32, 32, 64] if IS_FP8 else [32, 32, 16],
        )
        if IS_FP8:
            # attention_layouts pairs FP8 P and V at k_width 8, which takes
            # lane-32 exchanges to form P from the score layout. At k_width 4
            # each lane's 4 consecutive score columns already are its P
            # operand, as for 16-bit inputs, and V keeps its 8-byte transposed
            # LDS reads.
            p_layout = gl.DotOperandLayout(0, pv_layout, k_width=4)
            v_layout = gl.DotOperandLayout(1, pv_layout, k_width=4)
            v_smem_layout = padded_shared_layout(
                v_layout, [BLOCK_N, HEAD_DIM], KV_DTYPE, is_k_contig=False
            )
        # Keep each MFMA wave's 32 rows during output narrowing. Only the
        # lane-32 partner exchanges columns to form eight-element stores.
        store_layout = gl.BlockedLayout([1, 8], [32, 2], [NUM_WARPS, 1], [0, 1])
        # A RoPE tile holds only 8 bytes per thread, so 128-bit copies would
        # leave half the waves issuing duplicate copies; 32-bit copies give
        # every wave its own rows instead.
        pe_vec = 32 // KV_DTYPE.primitive_bitwidth
        pe_threads = ROPE_DIM // pe_vec
        load_pe_layout = gl.BlockedLayout(
            [1, pe_vec], [64 // pe_threads, pe_threads], [NUM_WARPS, 1], [1, 0]
        )
        # The RoPE K dot operand reuses the NoPE k_layout, so pass it here too.
        k_pe_smem_layout = padded_shared_layout(
            k_layout, [BLOCK_N, ROPE_DIM], KV_DTYPE, is_k_contig=True
        )

        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.ROPE_DIM = gl.constexpr(ROPE_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.BATCH_SIZE = gl.constexpr(BATCH_SIZE)
        self.NUM_XCDS = gl.constexpr(8)
        self.NUM_BLOCKS = gl.constexpr(512)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        # Lazy rescale threshold in log2 units: the running maximum only moves
        # when a tile maximum exceeds it by more than this, so p = exp2(s - m)
        # stays below 2**threshold. 16-bit P tolerates 2**8. FP8 P keeps the
        # exact maximum (threshold 0), so P <= 1 before its FP8 conversion as
        # in prefill.py, and its dominant weight stays exactly representable.
        self.RESCALE_THRESHOLD = gl.constexpr(0.0 if IS_FP8 else 8.0)
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
        self.o_strides = o_strides
        self.lse_strides = lse_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.q_pe_layout = gl.constexpr(q_layout)
        self.k_pe_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.load_pe_layout = gl.constexpr(load_pe_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.k_pe_smem_layout = gl.constexpr(k_pe_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


@gluon.jit
def finish_query_block(program, m_i, l_i, acc):
    cfg = program.cfg
    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)[:, None]
    if program.output_ptr.dtype.element_ty.primitive_bitwidth < 32:
        # Narrow before exchanging columns within each wave so each lane can
        # form eight-element stores. Wider outputs retain their direct
        # accumulator-layout stores.
        output = output.to(program.output_ptr.dtype.element_ty)
        output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


# ===-----------------------------------------------------------------------===#
# Warp-pipelined tile loop
# ===-----------------------------------------------------------------------===#
#
# A workgroup of 8 waves owns 256 query rows, 32 per wave, and walks the key
# tiles in BLOCK_N steps. Each loop iteration j finishes output tile j and is
# cut into four warp-pipeline clusters:
#
#   load_k   read K(j + 1) from LDS; start the DMA of V(j + 2); rescale
#   qk_sum   QK MFMA of tile j + 1; exp2, row sum and P convert of tile j
#   load_v   read V(j) from LDS; start the DMA of K(j + 4)
#   pv_max   PV MFMA of tile j; row max and exp2 of tile j + 1
#
# The two waves on a SIMD run the clusters one apart, so one wave's MFMAs
# overlap the other wave's LDS reads and DMA issue. The softmax runs in the
# MFMA clusters, where its VALU work co-issues with the same wave's MFMAs; it is
# split so that both MFMA clusters carry vector work.
#
# BLOCK_N keeps a wave within 256 VGPRs while Q, the accumulator (64), the K
# operand (reused for V) and the score tiles are live together: 32 keys for
# 16-bit inputs (Q 48, K 48, scores 16 per tile) and 64 for FP8 (Q 24, K 48,
# scores 32). Either way a tile gives each MFMA cluster the same matrix time:
# qk_sum issues 12 K=16 16-bit or 6 K=64 FP8 MFMAs (384 cycles), and pv_max 8
# or 4 (256 cycles).
#
# K and V move through 4-slot LDS rings. With K issued four tiles ahead and V
# three ahead, every DMA has about two tile periods to land, and it overwrites
# a slot whose operands both waves already consumed in an earlier MFMA cluster,
# so no LDS read can still be in flight. Tiles past the visible range are loaded
# with every row masked, which zero-fills LDS without reading memory; the loop
# therefore runs the same body to the last tile and needs no drain.

KV_RING = gl.constexpr(4)


@gluon.jit
def _keep_in_cluster(x):
    # An empty side-effecting asm that reads and redefines x. Neither the IR
    # optimizer nor MachineSink moves it across the cluster barriers, so the
    # ops producing x stay in the cluster that computes them instead of drifting
    # toward their consumer in a later cluster. "=v,0" ties the result to the
    # input register, so no instruction is emitted; narrower values go in
    # groups that fill one 32-bit VGPR per asm operand.
    pack: gl.constexpr = 32 // x.dtype.primitive_bitwidth
    return gl.inline_asm_elementwise(
        asm="",
        constraints="=v,0",
        args=[x],
        dtype=x.dtype,
        is_pure=False,
        pack=pack,
    )


@gluon.jit
def _split_columns(x):
    # Register-only column halves: each half keeps x's MFMA layout.
    half: gl.constexpr = x.shape[1] // 2
    lo = gl.amd.slice(x, [x.shape[0], half], [0, 0])
    hi = gl.amd.slice(x, [x.shape[0], half], [0, half])
    return lo, hi


@gluon.jit
def _join_columns(lo, hi):
    # Inverse of _split_columns, also register-only. join/permute/reshape
    # places lo's columns before hi's; assert_trivial fails the build if the
    # result would need data movement.
    layout: gl.constexpr = lo.type.layout
    shape: gl.constexpr = [lo.shape[0], lo.shape[1] + hi.shape[1]]
    x = gl.join(lo, hi).permute([0, 2, 1]).reshape(shape)
    return gl.convert_layout(x, layout, assert_trivial=True)


@gluon.aggregate
class TileCopies:
    # Per-thread global offsets and rows of key tile 0. A later tile only adds
    # a scalar row shift, which keeps the per-tile address multiplies out of
    # the copy clusters.
    k_offsets: gl.tensor
    k_rows: gl.tensor
    k_pe_offsets: gl.tensor
    k_pe_rows: gl.tensor
    v_offsets: gl.tensor
    v_rows: gl.tensor

    @gluon.constexpr_function
    def __init__(self, k_offsets, k_rows, k_pe_offsets, k_pe_rows, v_offsets, v_rows):
        self.k_offsets = k_offsets
        self.k_rows = k_rows
        self.k_pe_offsets = k_pe_offsets
        self.k_pe_rows = k_pe_rows
        self.v_offsets = v_offsets
        self.v_rows = v_rows

    @gluon.jit
    def create(program):
        k_offsets, k_rows = program.make_k_offsets(0)
        k_pe_offsets, k_pe_rows = program.make_k_pe_offsets(0)
        v_offsets, v_rows = program.make_v_offsets(0)
        return TileCopies(k_offsets, k_rows, k_pe_offsets, k_pe_rows, v_offsets, v_rows)

    @gluon.jit
    def issue_k(self, program, k_smem, k_pe_smem, tile):
        cfg = program.cfg
        slot = tile % KV_RING
        shift = tile * cfg.BLOCK_N
        # rows < kv_len - shift is the tile-0 form of (rows + shift) < kv_len.
        rows_left = program.kv_len - shift
        # Masked rows read nothing and zero-fill their LDS rows.
        async_copy.buffer_load_to_shared(
            k_smem.index(slot),
            program.k_ptr,
            self.k_offsets + shift * cfg.k_strides.stride_t,
            mask=self.k_rows[:, None] < rows_left,
        )
        async_copy.buffer_load_to_shared(
            k_pe_smem.index(slot),
            program.k_ptr,
            self.k_pe_offsets + shift * cfg.k_strides.stride_t,
            mask=self.k_pe_rows[:, None] < rows_left,
        )
        async_copy.commit_group()

    @gluon.jit
    def issue_v(self, program, v_smem, tile):
        cfg = program.cfg
        shift = tile * cfg.BLOCK_N
        async_copy.buffer_load_to_shared(
            v_smem.index(tile % KV_RING),
            program.v_ptr,
            self.v_offsets + shift * cfg.v_strides.stride_t,
            mask=self.v_rows[:, None] < program.kv_len - shift,
        )
        async_copy.commit_group()


# The loop orders these reads after the DMA through wait_group and the
# warp-pipeline barriers, so the loads need no extra waits of their own.
@gluon.jit
def _load_k_operand(cfg, k_smem):
    return async_copy.load_shared_relaxed(k_smem.permute([1, 0]), cfg.k_layout)


@gluon.jit
def _load_k_pe_operand(cfg, k_pe_smem):
    return async_copy.load_shared_relaxed(k_pe_smem.permute([1, 0]), cfg.k_pe_layout)


@gluon.jit
def _load_v_operand(cfg, v_smem):
    return async_copy.load_shared_relaxed(v_smem, cfg.v_layout)


@gluon.jit
def _read_k_tile(program, k_smem, k_pe_smem, tile):
    slot = tile % KV_RING
    k = _load_k_operand(program.cfg, k_smem.index(slot))
    k_pe = _load_k_pe_operand(program.cfg, k_pe_smem.index(slot))
    return k, k_pe


@gluon.jit
def _mask_columns(program, scores, last_col):
    # Sets scores past each row's last visible column (last_col, relative to
    # the tile) to -inf. The column index is loop invariant, so LLVM hoists it
    # and keeps it live across the loop; comparing quarter by quarter against
    # one quarter-wide index keeps that to a quarter of the score registers.
    cfg = program.cfg
    width: gl.constexpr = cfg.BLOCK_N // 4
    cols = gl.arange(0, width, layout=gl.SliceLayout(0, cfg.qk_layout))[None, :]
    lo, hi = _split_columns(scores)
    q0, q1 = _split_columns(lo)
    q2, q3 = _split_columns(hi)
    q0 = gl.where(cols <= last_col[:, None], q0, -float("inf"))
    q1 = gl.where(cols <= (last_col - width)[:, None], q1, -float("inf"))
    q2 = gl.where(cols <= (last_col - 2 * width)[:, None], q2, -float("inf"))
    q3 = gl.where(cols <= (last_col - 3 * width)[:, None], q3, -float("inf"))
    return _join_columns(_join_columns(q0, q1), _join_columns(q2, q3))


@gluon.jit
def _softmax_max(program, scores, m_run, kv_start, bound, MASKED: gl.constexpr):
    # Row maximum and exponent argument of one score tile, plus exp2 of its
    # first quarter. The two MFMA clusters share the softmax: this one holds
    # the PV MFMAs, two thirds as many as the QK MFMAs of the other, so the
    # remaining three quarters are exponentiated by _softmax_sum. Every
    # returned value is computed here, into fresh registers, so the next QK
    # MFMA can reuse the score registers.
    cfg = program.cfg
    scale: gl.constexpr = cfg.SM_SCALE * _INV_LN2
    if MASKED:
        scores = _mask_columns(program, scores, bound - kv_start)
    row_max = max(scores, 1) * scale
    if MASKED:
        # A row with no visible key keeps a finite maximum, so exp2 of its
        # masked scores is 0 rather than NaN.
        row_max = gl.where(row_max == -float("inf"), -1.0e20, row_max)
    # Branch-free lazy update: the maximum only moves when the tile maximum
    # exceeds it by more than the threshold, so no separate running maximum is
    # needed. The rescale that it may skip runs in load_k.
    m_new = gl.where(row_max - m_run > cfg.RESCALE_THRESHOLD, row_max, m_run)
    shifted = gl.fma(scores, scale, -m_new[:, None])
    first_half, second_half = _split_columns(shifted)
    first, second = _split_columns(first_half)
    p_first = _keep_in_cluster(gl.exp2(first))
    second = _keep_in_cluster(second)
    second_half = _keep_in_cluster(second_half)
    return m_new, p_first, second, second_half


@gluon.jit
def _softmax_sum(program, p_first, second, second_half, l_i):
    cfg = program.cfg
    p = _join_columns(_join_columns(p_first, gl.exp2(second)), gl.exp2(second_half))
    l_i = l_i + gl.sum(p, axis=1)
    # With k_width 4 the PV operand matches the score layout register for
    # register, so the conversion emits no data movement.
    p = gl.convert_layout(p.to(program.q_ptr.dtype.element_ty), cfg.p_layout)
    if cfg.IS_FP8:
        # Pinning packs four FP8 values per asm operand, which the register
        # allocator rejects, so FP8 p stays unpinned.
        return p, l_i
    return _keep_in_cluster(p), l_i


@gluon.jit
def _wave_maxima_unchanged(m_old, m_new):
    # Returns 1 when no row of the wave moved its maximum, else 0, as one
    # wave-uniform SGPR value, so a branch on it is a scalar branch that skips
    # the rescale for the whole wave:
    #   v_cmp_eq_f32  VCC bit set for each lane whose maximum is unchanged
    #   s_cmp_eq_u64  SCC = every active lane set its bit (VCC == EXEC)
    #   s_cselect     SGPR result = SCC ? 1 : 0
    # "=s" asks for an SGPR result; VCC and SCC are declared clobbered.
    return gl.inline_asm_elementwise(
        asm="v_cmp_eq_f32_e64 vcc, $1, $2\n"
        "s_cmp_eq_u64 vcc, exec\n"
        "s_cselect_b32 $0, 1, 0",
        constraints="=s,v,v,~{vcc},~{scc}",
        args=[m_old, m_new],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _rescale_row(l, m_old, m_new, unchanged):
    # alpha stays 1 for a wave whose maxima did not move. One value per lane,
    # so LLVM may turn this branch into a select.
    alpha = gl.cast(1.0, gl.float32)
    if unchanged == 0:
        alpha = gl.exp2(m_old - m_new)
        l = l * alpha
    return l, alpha


@gluon.jit
def _rescale_accumulator_pack(*args):
    # map_elementwise with pack=64 passes a lane's 64 elements of each operand
    # in order: args[0:64] accumulator, args[64:128] alpha and args[128:192]
    # the vote, the last two broadcast along the row. In the transposed 32x32
    # MFMA layout all 64 accumulator values of a lane belong to one row, so a
    # single branch covers them. The plain multiplies become v_pk_mul_f32,
    # the cheapest form here, where no MFMA shares the cluster.
    values = args[:64]
    if args[128] == 0:
        updated = ()
        for i in gl.static_range(64):
            updated += (values[i] * args[64],)
        values = updated
    return values


@gluon.jit
def _rescale(acc, l_i, m_old, m_new):
    # Apply the deferred correction. A wave skips it when none of its 32 rows
    # moved its maximum, which is the common case after the first tiles.
    unchanged = _wave_maxima_unchanged(m_old, m_new)
    l_i, alpha = gl.map_elementwise(_rescale_row, l_i, m_old, m_new, unchanged)
    (acc,) = gl.map_elementwise(
        _rescale_accumulator_pack,
        acc,
        alpha[:, None],
        unchanged[:, None],
        pack=64,
    )
    return acc, l_i


@gluon.jit
def _pipelined_tiles(
    program,
    copies,
    k_smem,
    k_pe_smem,
    v_smem,
    q,
    q_pe,
    p_first,
    second,
    second_half,
    m_old,
    m_i,
    l_i,
    acc,
    bound,
    start,
    end,
    MASKED: gl.constexpr,
):
    # Finish output tiles [start, end). On entry, the softmax pieces and `m_i`
    # hold tile `start`'s softmax input, and the rescale from `m_old` to `m_i`
    # is still pending; the DMA ring holds K up to start + 3 and V up to
    # start + 2, with K(start + 1) and V(start) visible to every wave. MASKED
    # selects the score mask for the tile whose row maximum each iteration
    # computes.
    cfg = program.cfg
    # Copy clusters run at a higher s_setprio priority, so when both waves on
    # a SIMD have work ready, the copy wave's loads issue first.
    for j in range(start, end):
        # A wave waits only for its own copies, and the other wave on the SIMD
        # runs one cluster behind. Keeping three groups in flight completes
        # each tile a full cluster before its first read, so both waves' shares
        # have landed whichever wave reads it.
        async_copy.wait_group(3)
        with gl.amd.warp_pipeline_stage("load_k", priority=1):
            # K is read and consumed within one iteration, so no K operand is
            # loop carried; LLVM would repack loop-carried FP8 bytes at every
            # backedge.
            k, k_pe = _read_k_tile(program, k_smem, k_pe_smem, j + 1)
            copies.issue_v(program, v_smem, j + 2)
            # Control flow stays out of the MFMA clusters: a branch there is
            # scheduled ahead of the first MFMA and stalls the matrix core.
            acc, l_i = _rescale(acc, l_i, m_old, m_i)
        with gl.amd.warp_pipeline_stage("qk_sum", priority=0):
            scores = program.compute_qk(q, k, q_pe, k_pe)
            p, l_i = _softmax_sum(program, p_first, second, second_half, l_i)
        async_copy.wait_group(3)
        with gl.amd.warp_pipeline_stage("load_v", priority=1):
            v = _load_v_operand(cfg, v_smem.index(j % KV_RING))
            copies.issue_k(program, k_smem, k_pe_smem, j + 4)
        with gl.amd.warp_pipeline_stage("pv_max", priority=0):
            acc = program.compute_pv(p, v, acc)
            m_old = m_i
            m_i, p_first, second, second_half = _softmax_max(
                program, scores, m_i, (j + 1) * cfg.BLOCK_N, bound, MASKED
            )
    return p_first, second, second_half, m_old, m_i, l_i, acc


@gluon.jit
def process_query_block(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    k_pe_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
):
    cfg = program.cfg
    q = program.load_q_nope()
    q_pe = program.load_q_pe()
    m_i, l_i, acc = program.init_state()

    # bound[i] = highest key index visible to query row (q_start + i).
    rows = (program.q_causal_start + program.q_start) + gl.arange(
        0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
    )
    if cfg.IS_CAUSAL:
        # Combine the causal and KV-length bounds once per row. Tiles before
        # main_end are fully visible to every row of this block.
        bound = gl.minimum(rows, program.kv_len - 1)
        main_end = gl.minimum(
            (program.q_causal_start + program.q_start) // cfg.BLOCK_N,
            program.kv_len // cfg.BLOCK_N,
        )
        visible = gl.minimum(
            program.q_causal_start + program.q_start + cfg.BLOCK_M, program.kv_len
        )
        count = gl.cdiv(visible, cfg.BLOCK_N)
    else:
        # Broadcast the scalar bound into the row layout.
        bound = rows - rows + (program.kv_len - 1)
        main_end = program.kv_len // cfg.BLOCK_N
        count = gl.cdiv(program.kv_len, cfg.BLOCK_N)

    if count > 0:
        copies = TileCopies.create(program)
        # Fill the rings in the loop's steady-state commit order: iteration j
        # commits V(j + 2) and then K(j + 4).
        copies.issue_k(program, k_smem, k_pe_smem, 0)
        copies.issue_k(program, k_smem, k_pe_smem, 1)
        copies.issue_v(program, v_smem, 0)
        copies.issue_k(program, k_smem, k_pe_smem, 2)
        copies.issue_v(program, v_smem, 1)
        copies.issue_k(program, k_smem, k_pe_smem, 3)
        # Six groups are in flight; waiting down to five retires K(0). The
        # compiler's barrier after each wait makes every wave's share of the
        # tile visible before it is read.
        async_copy.wait_group(5)
        k, k_pe = _read_k_tile(program, k_smem, k_pe_smem, 0)
        scores = program.compute_qk(q, k, q_pe, k_pe)
        m_i, p_first, second, second_half = _softmax_max(
            program, scores, m_i, 0, bound, True
        )
        # Down to three retires K(1) and V(0) for the first iteration; the
        # barrier after the wait makes every wave's share visible before the
        # other wave, one cluster behind, has waited itself.
        async_copy.wait_group(3)
        # acc and l_i are still zero, so tile 0 needs no rescale.
        m_old = m_i

        # Iteration j masks tile j + 1, so unmasked iterations stop one tile
        # before main_end.
        split = gl.maximum(main_end - 1, 0)
        p_first, second, second_half, m_old, m_i, l_i, acc = _pipelined_tiles(
            program,
            copies,
            k_smem,
            k_pe_smem,
            v_smem,
            q,
            q_pe,
            p_first,
            second,
            second_half,
            m_old,
            m_i,
            l_i,
            acc,
            bound,
            0,
            split,
            False,
        )
        p_first, second, second_half, m_old, m_i, l_i, acc = _pipelined_tiles(
            program,
            copies,
            k_smem,
            k_pe_smem,
            v_smem,
            q,
            q_pe,
            p_first,
            second,
            second_half,
            m_old,
            m_i,
            l_i,
            acc,
            bound,
            split,
            count,
            True,
        )
        # Retire the zero-fill DMAs issued past the last tile before the next
        # query block reuses the rings.
        async_copy.wait_group(0)

    finish_query_block(program, m_i, l_i, acc)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_mla_prefill_8wave_gfx950(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    K_STRIDE_T: gl.constexpr,
    K_STRIDE_H: gl.constexpr,
    V_STRIDE_T: gl.constexpr,
    V_STRIDE_H: gl.constexpr,
    O_STRIDE_T: gl.constexpr,
    O_STRIDE_H: gl.constexpr,
    LSE_STRIDE_T: gl.constexpr,
    LSE_STRIDE_H: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    SM_SCALE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    HAS_LSE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    max_seqlen_q,
    IS_FP8: gl.constexpr,
):
    cfg = Prefill8WaveConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        ROPE_DIM,
        SM_SCALE,
        IS_CAUSAL,
        HAS_LSE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        BATCH_SIZE,
        IS_FP8,
        k_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, 1),
        InputStrides(K_STRIDE_T, K_STRIDE_H, 1),
        InputStrides(V_STRIDE_T, V_STRIDE_H, 1),
        InputStrides(O_STRIDE_T, O_STRIDE_H, 1),
        InputStrides(LSE_STRIDE_T, LSE_STRIDE_H, 1),
    )
    k_smem = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [KV_RING, cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.k_smem_layout,
    )
    k_pe_smem = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [KV_RING, cfg.BLOCK_N, cfg.ROPE_DIM],
        cfg.k_pe_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [KV_RING, cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.v_smem_layout,
    )

    # Swizzle only helps the triangular causal workload; non-causal tiles are
    # uniform cost, so use the simpler round-robin order there.
    scheduler = ProgramScheduler.create(cfg, BATCH_SIZE, max_seqlen_q, IS_CAUSAL)
    while scheduler.has_work():
        program, active = scheduler.get_program(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            lse_ptr,
            cu_seqlens_q_ptr,
            cu_seqlens_kv_ptr,
        )
        if active:
            process_query_block(program, k_smem, k_pe_smem, v_smem)
        scheduler = scheduler.advance()


# ===-----------------------------------------------------------------------===#
# Host wrapper
# ===-----------------------------------------------------------------------===#


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def get_config(*, q: torch.Tensor, k: torch.Tensor) -> LaunchConfig:
    return LaunchConfig(
        n_heads=q.shape[1],
        n_kv_heads=k.shape[1],
        head_dim=128,
        rope_dim=64,
        block_m=256,
        # The widest key tile that keeps a wave's operands within 256 VGPRs;
        # FP8 also needs 64 keys for one K=64 PV MFMA step.
        block_n=64 if q.dtype in _FP8_DTYPES else 32,
        num_warps=8,
        grid=(512,),
    )


def launch_gluon_mla_prefill_8wave_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    seq_lens_kv: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dense non-absorbed MLA prefill on AMD gfx950.

    ``q``/``k`` are ``[total_tokens, num_heads, 192]`` (128 NoPE + 64 RoPE),
    ``v`` is ``[total_tokens, num_kv_heads, 128]``, all FP16, BF16, FP8 E4M3
    or FP8 E5M2.
    Output is ``[total_tokens, num_heads, 128]``. ``seq_lens_kv`` is accepted
    for interface parity; ``cu_seqlens_kv`` already defines the KV lengths.
    """
    if logit_cap != 0.0:
        raise NotImplementedError(
            "gluon MLA prefill 8wave gfx950 does not support logit_cap"
        )
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must be 3D [tokens, heads, head_dim]")
    if q.shape[-1] != 192 or k.shape[-1] != 192:
        raise ValueError(
            f"gluon MLA prefill requires qk_head_dim=192, got {q.shape[-1]}"
        )
    if v.shape[-1] != 128:
        raise ValueError(
            f"gluon MLA prefill requires v_head_dim=128, got {v.shape[-1]}"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads, "
            f"got {q.shape[1]} and {k.shape[1]}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have contiguous last dimension")
    if q.dtype not in (torch.float16, torch.bfloat16, *_FP8_DTYPES):
        raise TypeError(f"unsupported MLA prefill 8wave dtype {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must use the same dtype")

    total_tokens, n_heads, _ = q.shape
    v_head_dim = v.shape[-1]
    if out is None:
        out = torch.empty(
            (total_tokens, n_heads, v_head_dim), dtype=torch.bfloat16, device=q.device
        )
    if out.shape != (total_tokens, n_heads, v_head_dim):
        raise ValueError(
            f"out shape must be {(total_tokens, n_heads, v_head_dim)}, "
            f"got {tuple(out.shape)}"
        )
    if out.stride(-1) != 1:
        raise ValueError("out must have contiguous last dimension")

    lse = (
        torch.empty((total_tokens, n_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    lse_arg = lse if lse is not None else out

    config = get_config(q=q, k=k)
    # No request can exceed the query buffer's token capacity. Keep this a
    # runtime bound so varying prompt lengths do not each compile a kernel.
    max_seqlen_q = min(max_seqlen_q, total_tokens)
    gluon_mla_prefill_8wave_gfx950[config.grid](
        q,
        k,
        v,
        out,
        lse_arg,
        cu_seqlens_q,
        cu_seqlens_kv,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        lse_arg.stride(0),
        lse_arg.stride(1),
        N_HEADS=config.n_heads,
        N_KV_HEADS=config.n_kv_heads,
        HEAD_DIM=config.head_dim,
        ROPE_DIM=config.rope_dim,
        SM_SCALE=softmax_scale,
        IS_CAUSAL=is_causal,
        HAS_LSE=return_lse,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        NUM_WARPS=config.num_warps,
        BATCH_SIZE=cu_seqlens_q.numel() - 1,
        max_seqlen_q=max_seqlen_q,
        IS_FP8=q.dtype in _FP8_DTYPES,
        num_warps=config.num_warps,
        num_stages=1,
        # Keep the overlapping matrix and softmax state in one register class.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )

    if return_lse:
        return out, lse
    return out
