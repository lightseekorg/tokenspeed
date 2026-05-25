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

"""MHA prefill Gluon kernel optimized for AMD GFX950."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.attention.gluon.utils import (
    _INV_LN2,
    _INV_LN2_VALUE,
    InputStrides,
    max,
    maximum,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy


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
    HAS_SINK: gl.constexpr
    HAS_LSE: gl.constexpr
    IS_SLIDING: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    NUM_XCDS: gl.constexpr
    NUM_BLOCKS: gl.constexpr
    q_strides: InputStrides
    k_strides: InputStrides
    v_strides: InputStrides
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
        HAS_SINK,
        HAS_LSE,
        IS_SLIDING,
        WINDOW_LEFT,
        q_strides,
        k_strides,
        v_strides,
    ):
        assert HEAD_DIM == 64
        assert NUM_WARPS == 4
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
        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.BATCH_SIZE = gl.constexpr(BATCH_SIZE)
        self.HAS_SINK = gl.constexpr(HAS_SINK)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.IS_SLIDING = gl.constexpr(IS_SLIDING)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.NUM_XCDS = gl.constexpr(8)
        self.NUM_BLOCKS = gl.constexpr(512)
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        q_layout = gl.DotOperandLayout(0, qk_layout, k_width=8)
        k_layout = gl.DotOperandLayout(1, qk_layout, k_width=8)
        p_layout = gl.DotOperandLayout(0, pv_layout, k_width=4)
        v_layout = gl.DotOperandLayout(1, pv_layout, k_width=4)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


# ===-----------------------------------------------------------------------===#
# Kernel Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    cfg: gl.constexpr
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
        self.cfg = gl.constexpr(cfg)
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
    def load_q(self, other=None):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        offsets = cfg.q_strides.offsets(
            self.seq_base + offs_m[:, None], self.q_head, offs_d[None, :]
        )
        mask = offs_m[:, None] < self.seq_len
        if other is None:
            return cdna4.buffer_load(self.q_ptr, offsets, mask=mask)
        return cdna4.buffer_load(self.q_ptr, offsets, mask=mask, other=other)

    @gluon.jit
    def make_k_offsets(self, kv_start):
        cfg = self.cfg
        offs_n = kv_start + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.k_strides.offsets(
            self.seq_base + offs_n[:, None], self.kv_head, offs_d[None, :]
        )
        return offsets, offs_n

    @gluon.jit
    def make_v_offsets(self, kv_start):
        cfg = self.cfg
        offs_n = kv_start + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.v_strides.offsets(
            self.seq_base + offs_n[:, None], self.kv_head, offs_d[None, :]
        )
        return offsets

    @gluon.jit
    def update_k_offsets(self, offsets):
        cfg = self.cfg
        return offsets + cfg.BLOCK_N * cfg.k_strides.stride_t

    @gluon.jit
    def update_v_offsets(self, offsets):
        cfg = self.cfg
        return offsets + cfg.BLOCK_N * cfg.v_strides.stride_t

    @gluon.jit
    def issue_buffer_load_k(self, offsets, k_smem, mask=None, other=None):
        if mask is None:
            async_copy.buffer_load_to_shared(k_smem, self.k_ptr, offsets)
        elif other is None:
            async_copy.buffer_load_to_shared(k_smem, self.k_ptr, offsets, mask=mask)
        else:
            async_copy.buffer_load_to_shared(
                k_smem, self.k_ptr, offsets, mask=mask, other=other
            )
        async_copy.commit_group()

    @gluon.jit
    def issue_buffer_load_v(self, offsets, v_smem, mask=None, other=None):
        if mask is None:
            async_copy.buffer_load_to_shared(v_smem, self.v_ptr, offsets)
        elif other is None:
            async_copy.buffer_load_to_shared(v_smem, self.v_ptr, offsets, mask=mask)
        else:
            async_copy.buffer_load_to_shared(
                v_smem, self.v_ptr, offsets, mask=mask, other=other
            )
        async_copy.commit_group()

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
        return cdna4.mfma(q, k, qk)

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return cdna4.mfma(p, v, acc)

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
        cdna4.buffer_store(output, self.output_ptr, offsets, mask=mask)

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
            cdna4.buffer_store(lse, self.lse_ptr, offsets, mask=mask)


@gluon.aggregate
class ProgramScheduler:
    cfg: gl.constexpr
    work: gl.tensor
    total_work: gl.tensor
    num_q_blocks: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg, work, total_work, num_q_blocks):
        self.cfg = gl.constexpr(cfg)
        self.work = work
        self.total_work = total_work
        self.num_q_blocks = num_q_blocks

    @gluon.jit
    def create(cfg, batch_size, max_seqlen_q):
        num_q_blocks = (max_seqlen_q + cfg.BLOCK_M - 1) // cfg.BLOCK_M
        total_work = batch_size * cfg.N_HEADS * num_q_blocks

        # Physical pids are interleaved across XCDs by launch order. The remap
        # below deinterleaves them into contiguous logical pid ranges: local_pid
        # is the CTA's ordinal within one XCD, and work is the first logical tile
        # assigned to this persistent CTA.
        start_pid = gl.program_id(axis=0)
        pids_per_xcd: gl.constexpr = cfg.NUM_BLOCKS // cfg.NUM_XCDS
        xcd = start_pid % cfg.NUM_XCDS
        local_pid = start_pid // cfg.NUM_XCDS
        work = xcd * pids_per_xcd + local_pid

        return ProgramScheduler(gl.constexpr(cfg), work, total_work, num_q_blocks)

    @gluon.jit
    def has_work(self):
        return self.work < self.total_work

    @gluon.jit
    def advance(self):
        return ProgramScheduler(
            gl.constexpr(self.cfg),
            self.work + self.cfg.NUM_BLOCKS,
            self.total_work,
            self.num_q_blocks,
        )

    @gluon.jit
    def get_program(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        cu_seqlens_ptr,
    ):
        cfg = self.cfg
        query_block = self.work % self.num_q_blocks
        head_batch = self.work // self.num_q_blocks
        q_head = head_batch % cfg.N_HEADS
        batch = head_batch // cfg.N_HEADS
        seq_base = gl.load(cu_seqlens_ptr + batch)
        seq_end = gl.load(cu_seqlens_ptr + batch + 1)
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
            q_head,
        )
        return program, program.q_start < program.seq_len


@gluon.aggregate
class FullCausalProgramScheduler:
    cfg: gl.constexpr
    lane_valid: gl.tensor
    batch_slot: gl.tensor
    q_head: gl.tensor
    q_lane: gl.tensor
    num_q_blocks: gl.tensor
    q_rounds_per_wave: gl.tensor
    num_q_rounds: gl.tensor
    batch_slots: gl.constexpr
    q_lanes: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        lane_valid,
        batch_slot,
        q_head,
        q_lane,
        num_q_blocks,
        q_rounds_per_wave,
        num_q_rounds,
        batch_slots,
        q_lanes,
    ):
        self.cfg = gl.constexpr(cfg)
        self.lane_valid = lane_valid
        self.batch_slot = batch_slot
        self.q_head = q_head
        self.q_lane = q_lane
        self.num_q_blocks = num_q_blocks
        self.q_rounds_per_wave = q_rounds_per_wave
        self.num_q_rounds = num_q_rounds
        self.batch_slots = gl.constexpr(batch_slots)
        self.q_lanes = gl.constexpr(q_lanes)

    @gluon.jit
    def create(cfg, max_seqlen_q):
        num_q_blocks = (max_seqlen_q + cfg.BLOCK_M - 1) // cfg.BLOCK_M
        max_batch_slots: gl.constexpr = cfg.NUM_BLOCKS // cfg.N_HEADS
        if cfg.BATCH_SIZE < max_batch_slots:
            batch_slots: gl.constexpr = cfg.BATCH_SIZE
        else:
            batch_slots: gl.constexpr = max_batch_slots
        q_lanes: gl.constexpr = cfg.NUM_BLOCKS // (batch_slots * cfg.N_HEADS)

        q_rounds_per_wave = (num_q_blocks + q_lanes - 1) // q_lanes
        num_batch_waves: gl.constexpr = (
            cfg.BATCH_SIZE + batch_slots - 1
        ) // batch_slots
        num_q_rounds = num_batch_waves * q_rounds_per_wave

        start_pid = gl.program_id(axis=0)
        pids_per_xcd: gl.constexpr = cfg.NUM_BLOCKS // cfg.NUM_XCDS
        xcd = start_pid % cfg.NUM_XCDS
        local_pid = start_pid // cfg.NUM_XCDS
        logical_pid = xcd * pids_per_xcd + local_pid

        active_lanes: gl.constexpr = batch_slots * cfg.N_HEADS * q_lanes
        lane_valid = logical_pid < active_lanes
        safe_pid = gl.where(lane_valid, logical_pid, 0)
        q_lane = safe_pid % q_lanes
        head_batch_slot = safe_pid // q_lanes
        q_head = head_batch_slot % cfg.N_HEADS
        batch_slot = head_batch_slot // cfg.N_HEADS

        return FullCausalProgramScheduler(
            gl.constexpr(cfg),
            lane_valid,
            batch_slot,
            q_head,
            q_lane,
            num_q_blocks,
            q_rounds_per_wave,
            num_q_rounds,
            batch_slots,
            q_lanes,
        )

    @gluon.jit
    def has_work(self, q_round):
        return q_round < self.num_q_rounds

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
        batch_wave = q_round // self.q_rounds_per_wave
        wave_round = q_round - batch_wave * self.q_rounds_per_wave
        query_block_inc = wave_round * self.q_lanes + self.q_lane
        query_block_dec = wave_round * self.q_lanes + (self.q_lanes - 1 - self.q_lane)
        query_block = gl.where(wave_round % 2 == 0, query_block_inc, query_block_dec)
        batch = batch_wave * self.batch_slots + self.batch_slot
        valid = self.lane_valid & (query_block < self.num_q_blocks)

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
        return program, valid & (program.q_start < program.seq_len)


@gluon.jit
def process_single_attention_tile(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
):
    cfg = program.cfg
    q = program.load_q(other=0.0)

    k_offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(1, cfg.k_layout))
    k_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.k_layout))
    k_offsets = cfg.k_strides.offsets(
        program.seq_base + k_offs_n[None, :], program.kv_head, k_offs_d[:, None]
    )
    k_mask = k_offs_n[None, :] < program.seq_len
    k = cdna4.buffer_load(program.k_ptr, k_offsets, mask=k_mask, other=0.0)

    v_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.v_layout))
    v_offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.v_layout))
    v_offsets = cfg.v_strides.offsets(
        program.seq_base + v_offs_n[:, None], program.kv_head, v_offs_d[None, :]
    )
    v_mask = v_offs_n[:, None] < program.seq_len
    v = cdna4.buffer_load(program.v_ptr, v_offsets, mask=v_mask, other=0.0)

    qk = program.compute_qk(q, k)

    mask_offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
    mask_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    valid = mask_offs_m[:, None] < program.seq_len
    valid &= mask_offs_n[None, :] < program.seq_len
    valid &= mask_offs_n[None, :] <= mask_offs_m[:, None]
    if cfg.IS_SLIDING:
        valid &= mask_offs_m[:, None] <= mask_offs_n[None, :] + cfg.WINDOW_LEFT

    qk = gl.where(valid, qk, -1.0e20)
    row_has_valid = gl.sum(valid.to(gl.int32), axis=1) > 0
    row_max = max(qk, 1)
    m_i = gl.where(row_has_valid, row_max, 0.0)
    m_i_scaled = m_i * cfg.SM_SCALE
    p = gl.where(valid, gl.exp2(qk * cfg.SM_SCALE - m_i_scaled[:, None]), 0.0)
    l_i = gl.sum(p, axis=1)
    if cfg.HAS_SINK:
        sink_log2 = gl.load(program.sink_ptr + program.q_head).to(gl.float32) * _INV_LN2
        l_i += gl.exp2(sink_log2 - m_i_scaled)

    acc = gl.zeros([cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout)
    p = p.to(program.q_ptr.dtype.element_ty)
    p = gl.convert_layout(p, cfg.p_layout)
    acc = program.compute_pv(p, v, acc)

    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


@gluon.jit
def process_attention_tile(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
    boundary_mask0=None,
    boundary_mask1=None,
):
    cfg = program.cfg
    q = program.load_q()
    m_i, l_i, acc = program.init_attention_state()

    if cfg.IS_SLIDING:
        kv_start = program.q_start - cfg.WINDOW_LEFT
        kv_start = gl.where(kv_start > 0, (kv_start // cfg.BLOCK_N) * cfg.BLOCK_N, 0)
        num_kv_tiles: gl.constexpr = (
            cfg.BLOCK_M + cfg.WINDOW_LEFT + cfg.BLOCK_N - 1
        ) // cfg.BLOCK_N
        for _ in range(0, num_kv_tiles):
            k_offsets, offs_n = program.make_k_offsets(kv_start)
            v_offsets = program.make_v_offsets(kv_start)
            mask = offs_n[:, None] < program.seq_len
            program.issue_buffer_load_k(k_offsets, k_smem, mask=mask)
            program.issue_buffer_load_v(v_offsets, v_smem, mask=mask, other=0.0)

            async_copy.wait_group(1)
            k = program.shared_load_k(k_smem)
            qk = program.compute_qk(q, k)
            qk = program.apply_sliding_mask(qk, offs_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

            async_copy.wait_group(0)
            v = program.shared_load_v(v_smem)
            acc = program.compute_pv(p, v, acc)
            kv_start = kv_start + cfg.BLOCK_N
    else:
        main_end = program.q_start // cfg.BLOCK_N
        base_k_offsets, base_offs_n = program.make_k_offsets(0)
        base_v_offsets = program.make_v_offsets(0)

        k_offsets = base_k_offsets
        v_offsets = base_v_offsets
        offs_n = base_offs_n

        for _ in range(0, main_end):
            program.issue_buffer_load_k(k_offsets, k_smem)
            program.issue_buffer_load_v(v_offsets, v_smem)

            async_copy.wait_group(1)
            k = program.shared_load_k(k_smem)
            qk = program.compute_qk(q, k)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

            async_copy.wait_group(0)
            v = program.shared_load_v(v_smem)
            acc = program.compute_pv(p, v, acc)

            k_offsets = program.update_k_offsets(k_offsets)
            v_offsets = program.update_v_offsets(v_offsets)
            offs_n = offs_n + cfg.BLOCK_N

        # The main loop handles prefix tiles; the two boundary tiles are causal.
        boundary_start = main_end * cfg.BLOCK_N
        k_offsets, offs_n = program.make_k_offsets(boundary_start)
        v_offsets = program.make_v_offsets(boundary_start)
        mask = offs_n[:, None] < program.seq_len
        program.issue_buffer_load_k(k_offsets, k_smem, mask=mask, other=0.0)
        program.issue_buffer_load_v(v_offsets, v_smem, mask=mask, other=0.0)

        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = gl.where(boundary_mask0, qk, -float("inf"))
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

        boundary_start = boundary_start + cfg.BLOCK_N
        k_offsets, offs_n = program.make_k_offsets(boundary_start)
        v_offsets = program.make_v_offsets(boundary_start)
        mask = offs_n[:, None] < program.seq_len
        program.issue_buffer_load_k(k_offsets, k_smem, mask=mask, other=0.0)
        program.issue_buffer_load_v(v_offsets, v_smem, mask=mask, other=0.0)

        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = gl.where(boundary_mask1, qk, -float("inf"))
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    l_i = program.apply_sinks(l_i, m_i)
    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _mha_prefill_fp16(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_ptr,
    output_ptr,
    sink_ptr,
    lse_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    K_STRIDE_T: gl.constexpr,
    K_STRIDE_H: gl.constexpr,
    K_STRIDE_D: gl.constexpr,
    V_STRIDE_T: gl.constexpr,
    V_STRIDE_H: gl.constexpr,
    V_STRIDE_D: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SM_SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    max_seqlen_q,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
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
        HAS_SINK,
        HAS_LSE,
        IS_SLIDING,
        WINDOW_LEFT,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(K_STRIDE_T, K_STRIDE_H, K_STRIDE_D),
        InputStrides(V_STRIDE_T, V_STRIDE_H, V_STRIDE_D),
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

    if cfg.IS_SLIDING:
        scheduler = ProgramScheduler.create(cfg, BATCH_SIZE, max_seqlen_q)
        while scheduler.has_work():
            program, active = scheduler.get_program(
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sink_ptr,
                lse_ptr,
                cu_seqlens_ptr,
            )
            if active:
                if program.seq_len < cfg.BLOCK_N:
                    if program.q_start == 0:
                        process_single_attention_tile(program, k_smem, v_smem)
                else:
                    process_attention_tile(program, k_smem, v_smem)
            scheduler = scheduler.advance()
    else:
        scheduler = FullCausalProgramScheduler.create(cfg, max_seqlen_q)
        mask_offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
        mask_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
        boundary_mask0 = mask_offs_n[None, :] <= mask_offs_m[:, None]
        boundary_mask1 = (mask_offs_n[None, :] + cfg.BLOCK_N) <= mask_offs_m[:, None]

        q_round = 0
        while scheduler.has_work(q_round):
            program, active = scheduler.get_program(
                q_round,
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sink_ptr,
                lse_ptr,
                cu_seqlens_ptr,
            )
            if active:
                if program.seq_len < cfg.BLOCK_N:
                    if program.q_start == 0:
                        process_single_attention_tile(program, k_smem, v_smem)
                else:
                    process_attention_tile(
                        program, k_smem, v_smem, boundary_mask0, boundary_mask1
                    )
            q_round += 1


class LaunchConfig(NamedTuple):
    n_heads: int
    n_kv_heads: int
    head_dim: int
    sm_scale: float
    block_m: int
    block_n: int
    num_warps: int
    batch_size: int
    max_seqlen: int
    is_sliding: bool
    window_left: int
    grid: tuple[int, ...]


def get_config(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    window_left: int,
) -> LaunchConfig:
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    block_m = 128
    block_n = 64
    num_warps = 4
    batch_size = cu_seqlens_q.numel() - 1
    is_sliding = window_left >= 0
    window_left = window_left if is_sliding else -1
    sm_scale = (1.0 / math.sqrt(head_dim)) * _INV_LN2_VALUE
    return LaunchConfig(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        sm_scale=sm_scale,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        batch_size=batch_size,
        max_seqlen=max_seqlen_q,
        is_sliding=is_sliding,
        window_left=window_left,
        grid=(512,),
    )


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
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    total_tokens, n_heads, _ = q.shape
    config = get_config(
        q=q,
        k=k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        window_left=window_left,
    )
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = (
        torch.empty((total_tokens, n_heads), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    has_sink = sinks is not None
    has_lse = return_lse
    sink_arg = sinks if sinks is not None else q
    lse_arg = lse if lse is not None else q

    _mha_prefill_fp16[config.grid](
        q,
        k,
        v,
        cu_seqlens_q,
        output,
        sink_arg,
        lse_arg,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.sm_scale,
        config.block_m,
        config.block_n,
        config.num_warps,
        config.batch_size,
        config.max_seqlen,
        has_sink,
        has_lse,
        config.is_sliding,
        config.window_left,
        num_warps=config.num_warps,
    )
    if return_lse:
        return output, lse
    return output
