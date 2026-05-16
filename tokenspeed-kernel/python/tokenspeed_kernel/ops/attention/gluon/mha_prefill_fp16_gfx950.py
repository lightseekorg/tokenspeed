"""MHA prefill Gluon kernel optimized for AMD GFX950."""

from __future__ import annotations

import math

import tokenspeed_triton as triton
import tokenspeed_triton.experimental.gluon.language as gl
import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_triton import language as tl
from tokenspeed_triton.experimental import gluon
from tokenspeed_triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load,
    buffer_store,
    mfma,
)
from tokenspeed_triton.experimental.gluon.language.amd.cdna4.async_copy import (
    buffer_load_to_shared,
    commit_group,
    wait_group,
)
from tokenspeed_triton.language.core import PropagateNan

_INV_LN2_VALUE = 1.4426950408889634
_INV_LN2 = tl.constexpr(_INV_LN2_VALUE)

# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


@gluon.jit
def maximum(a, b, propagate_nan: gl.constexpr = PropagateNan.ALL):
    return gl.maximum(a, b, propagate_nan=propagate_nan)


@gluon.jit
def max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, maximum, keep_dims=keep_dims)


# ===-----------------------------------------------------------------------===#
# Kernel Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionConfig:
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    HEAD_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    BATCH_SIZE: gl.constexpr
    MAX_SEQLEN: gl.constexpr
    HAS_SINK: gl.constexpr
    HAS_LSE: gl.constexpr
    IS_SLIDING: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    NUM_Q_BLOCKS: gl.constexpr
    NUM_TILES: gl.constexpr
    NUM_KV_TILES: gl.constexpr
    NUM_SMS: gl.constexpr
    NUM_XCDS: gl.constexpr
    NUM_BLOCKS: gl.constexpr
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    store_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        BATCH_SIZE,
        MAX_SEQLEN,
        HAS_SINK,
        HAS_LSE,
        IS_SLIDING,
        WINDOW_LEFT,
        NUM_Q_BLOCKS,
        NUM_TILES,
        NUM_SMS,
        NUM_XCDS,
        NUM_BLOCKS,
    ):
        assert HEAD_DIM == 64
        assert NUM_WARPS == 4
        assert NUM_SMS % NUM_XCDS == 0
        assert NUM_BLOCKS % NUM_XCDS == 0
        assert NUM_BLOCKS >= N_HEADS
        if IS_SLIDING:
            assert WINDOW_LEFT >= 0
        else:
            assert WINDOW_LEFT == -1

        qk_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 16],
            transposed=True,
            warps_per_cta=[NUM_WARPS, 1],
        )
        pv_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 16],
            transposed=True,
            warps_per_cta=[NUM_WARPS, 1],
        )

        load_layout = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        store_layout = load_layout

        k_smem_layout = gl.PaddedSharedLayout.with_identity_for(
            [[512, 8]], [BLOCK_N, HEAD_DIM], [1, 0]
        )
        v_smem_layout = gl.PaddedSharedLayout.with_identity_for(
            [[512, 32]], [BLOCK_N, HEAD_DIM], [1, 0]
        )
        if IS_SLIDING:
            num_kv_tiles = (BLOCK_M + WINDOW_LEFT + BLOCK_N - 1) // BLOCK_N
        else:
            num_kv_tiles = 0
        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.BATCH_SIZE = gl.constexpr(BATCH_SIZE)
        self.MAX_SEQLEN = gl.constexpr(MAX_SEQLEN)
        self.HAS_SINK = gl.constexpr(HAS_SINK)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.IS_SLIDING = gl.constexpr(IS_SLIDING)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.NUM_Q_BLOCKS = gl.constexpr(NUM_Q_BLOCKS)
        self.NUM_TILES = gl.constexpr(NUM_TILES)
        self.NUM_KV_TILES = gl.constexpr(num_kv_tiles)
        self.NUM_SMS = gl.constexpr(NUM_SMS)
        self.NUM_XCDS = gl.constexpr(NUM_XCDS)
        self.NUM_BLOCKS = gl.constexpr(NUM_BLOCKS)
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, qk_layout, k_width=8))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, qk_layout, k_width=8))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, pv_layout, k_width=4))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, pv_layout, k_width=4))
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


# ===-----------------------------------------------------------------------===#
# Kernel Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    cfg: AttentionConfig
    q_ptr: gl.tensor
    k_ptr: gl.tensor
    v_ptr: gl.tensor
    output_ptr: gl.tensor
    sink_ptr: gl.tensor
    lse_ptr: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    q_start: gl.tensor
    q_head: gl.tensor
    kv_head: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        seq_base,
        seq_len,
        q_start,
        q_head,
        kv_head,
    ):
        self.cfg = cfg
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.output_ptr = output_ptr
        self.sink_ptr = sink_ptr
        self.lse_ptr = lse_ptr
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.q_start = q_start
        self.q_head = q_head
        self.kv_head = kv_head

    @gluon.jit
    def initialize_from_state(
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        seq_base,
        seq_len,
        query_block,
        q_head,
    ):
        kv_head = q_head // (cfg.N_HEADS // cfg.N_KV_HEADS)
        q_start = query_block * cfg.BLOCK_M
        return AttentionProgram(
            cfg,
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            sink_ptr,
            lse_ptr,
            seq_base,
            seq_len,
            q_start,
            q_head,
            kv_head,
        )

    @gluon.jit
    def load_q(self):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        offsets = (
            ((self.seq_base + offs_m[:, None]) * cfg.N_HEADS + self.q_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        mask = offs_m[:, None] < self.seq_len
        return buffer_load(self.q_ptr, offsets, mask=mask)

    @gluon.jit
    def make_kv_offsets(self, kv_start):
        cfg = self.cfg
        offs_n = kv_start + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = (
            ((self.seq_base + offs_n[:, None]) * cfg.N_KV_HEADS + self.kv_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        return offsets, offs_n

    @gluon.jit
    def update_kv_offsets(self, offsets):
        cfg = self.cfg
        return offsets + cfg.BLOCK_N * cfg.N_KV_HEADS * cfg.HEAD_DIM

    @gluon.jit
    def issue_buffer_load_k(self, offsets, k_smem, mask=None, other=None):
        if mask is None:
            buffer_load_to_shared(k_smem, self.k_ptr, offsets)
        elif other is None:
            buffer_load_to_shared(k_smem, self.k_ptr, offsets, mask=mask)
        else:
            buffer_load_to_shared(k_smem, self.k_ptr, offsets, mask=mask, other=other)
        commit_group()

    @gluon.jit
    def issue_buffer_load_v(self, offsets, v_smem, mask=None, other=None):
        if mask is None:
            buffer_load_to_shared(v_smem, self.v_ptr, offsets)
        elif other is None:
            buffer_load_to_shared(v_smem, self.v_ptr, offsets, mask=mask)
        else:
            buffer_load_to_shared(v_smem, self.v_ptr, offsets, mask=mask, other=other)
        commit_group()

    @gluon.jit
    def shared_load_k(self, k_smem):
        cfg = self.cfg
        k_buffer = k_smem.permute([1, 0])
        return k_buffer.load(cfg.k_layout)

    @gluon.jit
    def shared_load_v(self, v_smem):
        cfg = self.cfg
        return v_smem.load(cfg.v_layout)

    @gluon.jit
    def compute_qk(self, q, k):
        cfg = self.cfg
        qk = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout
        )
        return mfma(q, k, qk)

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return mfma(p, v, acc)

    @gluon.jit
    def init_attention_state(self):
        cfg = self.cfg
        if cfg.HAS_SINK:
            sink = gl.load(self.sink_ptr + self.q_head).to(gl.float32)
            sink_unscaled = sink * _INV_LN2 / cfg.SM_SCALE
            m_i = gl.full(
                [cfg.BLOCK_M],
                value=0,
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout),
            )
            m_i += sink_unscaled
        else:
            m_i = gl.full(
                [cfg.BLOCK_M],
                value=-float("inf"),
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout),
            )
        l_i = gl.full(
            [cfg.BLOCK_M],
            value=0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout
        )
        return m_i, l_i, acc

    @gluon.jit
    def apply_sliding_mask(self, qk, offs_n):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
        )
        kv = gl.convert_layout(offs_n, gl.SliceLayout(0, cfg.qk_layout))
        valid = offs_m[:, None] < self.seq_len
        valid &= kv[None, :] < self.seq_len
        valid &= kv[None, :] <= offs_m[:, None]
        valid &= offs_m[:, None] <= kv[None, :] + cfg.WINDOW_LEFT
        qk = gl.where(valid, qk, -float("inf"))
        return qk

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        row_max = max(qk, 1)
        m_new = maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def apply_sinks(self, l_i, m_i):
        cfg = self.cfg
        if cfg.HAS_SINK:
            sink = gl.load(self.sink_ptr + self.q_head).to(gl.float32)
            l_i += gl.exp2(sink * _INV_LN2 - m_i * cfg.SM_SCALE)
        return l_i

    @gluon.jit
    def store_output(self, output):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        offsets = (
            ((self.seq_base + offs_m[:, None]) * cfg.N_HEADS + self.q_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        mask = offs_m[:, None] < self.seq_len
        output = output.to(self.output_ptr.dtype.element_ty)
        buffer_store(output, self.output_ptr, offsets, mask=mask)

    @gluon.jit
    def store_lse(self, l_i, m_i):
        cfg = self.cfg
        if cfg.HAS_LSE:
            offs_m = self.q_start + gl.arange(
                0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)
            )
            offsets = ((self.seq_base + offs_m) * cfg.N_HEADS + self.q_head).to(
                gl.int32
            )
            mask = offs_m < self.seq_len
            lse_l_i = gl.where(l_i > 0.0, l_i, 1.0)
            lse = m_i * cfg.SM_SCALE + gl.log2(lse_l_i)
            buffer_store(lse, self.lse_ptr, offsets, mask=mask)


@gluon.aggregate
class ProgramScheduler:
    cfg: AttentionConfig
    lane_valid: gl.tensor
    batch: gl.tensor
    q_head: gl.tensor
    q_lane: gl.tensor
    query_block: gl.tensor
    batch_slots: gl.constexpr
    q_lanes: gl.constexpr
    q_rounds_per_wave: gl.constexpr
    num_q_rounds: gl.constexpr

    @gluon.jit
    def create(cfg):
        # Physical pids are interleaved across XCDs by launch order. The remap
        # below deinterleaves them into contiguous logical pid ranges: local_pid
        # is the CTA's ordinal within one XCD, and logical_pid is the id used for
        # assigning batch/head/Q-block work while keeping adjacent heads grouped
        # per XCD.
        start_pid = gl.program_id(axis=0)
        if cfg.IS_SLIDING:
            pids_per_xcd: gl.constexpr = (
                cfg.NUM_TILES + cfg.NUM_XCDS - 1
            ) // cfg.NUM_XCDS
            xcd = start_pid % cfg.NUM_XCDS
            local_pid = start_pid // cfg.NUM_XCDS
            logical_pid = xcd * pids_per_xcd + local_pid
            lane_valid = logical_pid < cfg.NUM_TILES
            safe_pid = gl.where(lane_valid, logical_pid, 0)
            query_block = safe_pid % cfg.NUM_Q_BLOCKS
            head_batch = safe_pid // cfg.NUM_Q_BLOCKS
            q_head = head_batch % cfg.N_HEADS
            batch = head_batch // cfg.N_HEADS
            q_lane = safe_pid - safe_pid
            batch_slots: gl.constexpr = 1
            q_lanes: gl.constexpr = 1
            q_rounds_per_wave: gl.constexpr = 1
            num_q_rounds: gl.constexpr = 1
        else:
            pids_per_xcd: gl.constexpr = cfg.NUM_BLOCKS // cfg.NUM_XCDS
            xcd = start_pid % cfg.NUM_XCDS
            local_pid = start_pid // cfg.NUM_XCDS
            logical_pid = xcd * pids_per_xcd + local_pid

            max_batch_slots: gl.constexpr = cfg.NUM_BLOCKS // cfg.N_HEADS
            if cfg.BATCH_SIZE < max_batch_slots:
                batch_slots: gl.constexpr = cfg.BATCH_SIZE
            else:
                batch_slots: gl.constexpr = max_batch_slots

            q_lanes: gl.constexpr = cfg.NUM_BLOCKS // (batch_slots * cfg.N_HEADS)
            q_rounds_per_wave: gl.constexpr = (
                cfg.NUM_Q_BLOCKS + q_lanes - 1
            ) // q_lanes
            num_batch_waves: gl.constexpr = (
                cfg.BATCH_SIZE + batch_slots - 1
            ) // batch_slots
            num_q_rounds: gl.constexpr = num_batch_waves * q_rounds_per_wave

            active_lanes: gl.constexpr = batch_slots * cfg.N_HEADS * q_lanes
            lane_valid = logical_pid < active_lanes
            safe_pid = gl.where(lane_valid, logical_pid, 0)
            q_lane = safe_pid % q_lanes
            head_batch_slot = safe_pid // q_lanes
            q_head = head_batch_slot % cfg.N_HEADS
            batch = head_batch_slot // cfg.N_HEADS
            query_block = safe_pid - safe_pid

        return ProgramScheduler(
            cfg,
            lane_valid,
            batch,
            q_head,
            q_lane,
            query_block,
            batch_slots,
            q_lanes,
            q_rounds_per_wave,
            num_q_rounds,
        )

    @gluon.jit
    def get_program(
        self,
        q_round,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        cu_seqlens_ptr,
    ):
        cfg = self.cfg

        if cfg.IS_SLIDING:
            query_block = self.query_block
            batch = self.batch
            valid = self.lane_valid
        else:
            batch_wave = q_round // self.q_rounds_per_wave
            wave_round = q_round - batch_wave * self.q_rounds_per_wave
            query_block_inc = wave_round * self.q_lanes + self.q_lane
            query_block_dec = wave_round * self.q_lanes + (
                self.q_lanes - 1 - self.q_lane
            )
            query_block = gl.where(
                wave_round % 2 == 0, query_block_inc, query_block_dec
            )
            batch = batch_wave * self.batch_slots + self.batch
            valid = (
                self.lane_valid
                & (batch < cfg.BATCH_SIZE)
                & (query_block < cfg.NUM_Q_BLOCKS)
            )

        safe_batch = gl.where(valid, batch, 0)
        seq_base = gl.load(cu_seqlens_ptr + safe_batch)
        seq_end = gl.load(cu_seqlens_ptr + safe_batch + 1)
        seq_len = seq_end - seq_base
        program = AttentionProgram.initialize_from_state(
            cfg,
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            sink_ptr,
            lse_ptr,
            seq_base,
            seq_len,
            query_block,
            self.q_head,
        )
        return program, valid


@gluon.jit
def process_attention_tile(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
    boundary_mask0: gl.tensor,
    boundary_mask1: gl.tensor,
):
    cfg = program.cfg
    q = program.load_q()
    m_i, l_i, acc = program.init_attention_state()

    main_end = program.q_start // cfg.BLOCK_N
    base_offsets, base_offs_n = program.make_kv_offsets(0)

    kv_offsets = base_offsets
    offs_n = base_offs_n

    for _ in range(0, main_end):
        program.issue_buffer_load_k(kv_offsets, k_smem)
        program.issue_buffer_load_v(kv_offsets, v_smem)

        wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

        kv_offsets = program.update_kv_offsets(kv_offsets)
        offs_n = offs_n + cfg.BLOCK_N

    # The main loop handles prefix tiles; the two boundary tiles are causal.
    boundary_offset = main_end * cfg.BLOCK_N * cfg.N_KV_HEADS * cfg.HEAD_DIM
    boundary_offs_n = main_end * cfg.BLOCK_N

    kv_offsets = base_offsets + boundary_offset
    offs_n = base_offs_n + boundary_offs_n
    mask = offs_n[:, None] < program.seq_len
    program.issue_buffer_load_k(kv_offsets, k_smem, mask=mask)
    program.issue_buffer_load_v(kv_offsets, v_smem, mask=mask, other=0.0)

    wait_group(1)
    k = program.shared_load_k(k_smem)
    qk = program.compute_qk(q, k)
    qk = gl.where(boundary_mask0, qk, -float("inf"))
    p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

    wait_group(0)
    v = program.shared_load_v(v_smem)
    acc = program.compute_pv(p, v, acc)

    boundary_step = cfg.BLOCK_N * cfg.N_KV_HEADS * cfg.HEAD_DIM
    kv_offsets = base_offsets + boundary_offset + boundary_step
    offs_n = base_offs_n + boundary_offs_n + cfg.BLOCK_N
    mask = offs_n[:, None] < program.seq_len
    program.issue_buffer_load_k(kv_offsets, k_smem, mask=mask)
    program.issue_buffer_load_v(kv_offsets, v_smem, mask=mask, other=0.0)

    wait_group(1)
    k = program.shared_load_k(k_smem)
    qk = program.compute_qk(q, k)
    qk = gl.where(boundary_mask1, qk, -float("inf"))
    p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

    wait_group(0)
    v = program.shared_load_v(v_smem)
    acc = program.compute_pv(p, v, acc)

    l_i = program.apply_sinks(l_i, m_i)
    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


@gluon.jit
def process_sliding_attention_tile(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
):
    cfg = program.cfg
    q = program.load_q()
    m_i, l_i, acc = program.init_attention_state()

    kv_start = program.q_start - cfg.WINDOW_LEFT
    kv_start = gl.where(kv_start > 0, (kv_start // cfg.BLOCK_N) * cfg.BLOCK_N, 0)
    for _ in range(0, cfg.NUM_KV_TILES):
        offsets, offs_n = program.make_kv_offsets(kv_start)
        mask = offs_n[:, None] < program.seq_len
        program.issue_buffer_load_k(offsets, k_smem, mask=mask)
        program.issue_buffer_load_v(offsets, v_smem, mask=mask, other=0.0)

        wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = program.apply_sliding_mask(qk, offs_n)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)
        kv_start = kv_start + cfg.BLOCK_N

    l_i = program.apply_sinks(l_i, m_i)
    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


@gluon.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_ptr,
    output_ptr,
    sink_ptr,
    lse_ptr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SM_SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    MAX_SEQLEN: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    NUM_Q_BLOCKS: gl.constexpr,
    NUM_TILES: gl.constexpr,
    NUM_SMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
):
    cfg = AttentionConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        BATCH_SIZE,
        MAX_SEQLEN,
        HAS_SINK,
        HAS_LSE,
        IS_SLIDING,
        WINDOW_LEFT,
        NUM_Q_BLOCKS,
        NUM_TILES,
        NUM_SMS,
        NUM_XCDS,
        NUM_BLOCKS,
    )
    k_smem = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.k_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.v_smem_layout,
    )

    scheduler = ProgramScheduler.create(cfg)
    if cfg.IS_SLIDING:
        program, valid = scheduler.get_program(
            0,
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            sink_ptr,
            lse_ptr,
            cu_seqlens_ptr,
        )
        if valid:
            process_sliding_attention_tile(program, k_smem, v_smem)
    else:
        mask_offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
        mask_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
        boundary_mask0 = mask_offs_n[None, :] <= mask_offs_m[:, None]
        boundary_mask1 = (mask_offs_n[None, :] + cfg.BLOCK_N) <= mask_offs_m[:, None]

        for q_round in range(0, scheduler.num_q_rounds):
            program, valid = scheduler.get_program(
                q_round,
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sink_ptr,
                lse_ptr,
                cu_seqlens_ptr,
            )
            if valid:
                process_attention_tile(
                    program, k_smem, v_smem, boundary_mask0, boundary_mask1
                )


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def schedule_grid(
    *,
    batch_size: int,
    n_heads: int,
    max_seqlen: int,
    block_m: int,
    num_blocks: int,
    num_xcds: int,
    is_sliding: bool,
) -> tuple[int, ...]:
    num_q_blocks = triton.cdiv(max_seqlen, block_m)
    if is_sliding:
        num_tiles = num_q_blocks * n_heads * batch_size
        return (max(1, triton.cdiv(num_tiles, num_xcds) * num_xcds),)
    return (num_blocks,)


@register_kernel(
    "attention",
    "mha_prefill",
    name="gluon_mha_prefill_fp16_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    dtypes={torch.float16, torch.bfloat16},
    priority=Priority.SPECIALIZED,
    traits={
        "head_dim": frozenset({64}),
        "is_causal": frozenset({True}),
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    },
)
def gluon_mha_prefill_fp16_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    total_tokens, n_heads, head_dim = q.shape

    block_m = 128
    block_n = 64
    num_warps = 4
    batch_size = cu_seqlens_q.numel() - 1
    num_sms = 256
    num_xcds = 8
    num_blocks = num_sms * 2
    is_sliding = window_left >= 0
    window_left = window_left if is_sliding else -1
    num_q_blocks = triton.cdiv(max_seqlen_q, block_m)
    num_tiles = num_q_blocks * n_heads * batch_size
    grid = schedule_grid(
        batch_size=batch_size,
        n_heads=n_heads,
        max_seqlen=max_seqlen_q,
        block_m=block_m,
        num_blocks=num_blocks,
        num_xcds=num_xcds,
        is_sliding=is_sliding,
    )
    output = torch.empty_like(q)
    lse = (
        torch.empty((total_tokens, n_heads), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    scale = 1.0 / math.sqrt(head_dim)
    has_sink = sinks is not None
    has_lse = return_lse
    sink_arg = sinks if sinks is not None else q
    lse_arg = lse if lse is not None else q

    attention_kernel[grid](
        q,
        k,
        v,
        cu_seqlens_q,
        output,
        sink_arg,
        lse_arg,
        n_heads,
        k.shape[1],
        head_dim,
        scale * _INV_LN2_VALUE,
        block_m,
        block_n,
        num_warps,
        batch_size,
        max_seqlen_q,
        has_sink,
        has_lse,
        is_sliding,
        window_left,
        num_q_blocks,
        num_tiles,
        num_sms,
        num_xcds,
        num_blocks,
        num_warps=num_warps,
    )
    if return_lse:
        return output, lse
    return output
