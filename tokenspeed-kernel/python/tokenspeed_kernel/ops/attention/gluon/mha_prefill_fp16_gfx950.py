"""MHA prefill Gluon kernel optimized for AMD GFX950."""

from __future__ import annotations

import math

import tokenspeed_triton.experimental.gluon.language as gl
import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
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

# ===-----------------------------------------------------------------------===#
# Kernel Utilities
# ===-----------------------------------------------------------------------===#


@gluon.jit
def elementwise_max_prop_nan(a, b, propagate_nan: gl.constexpr = PropagateNan.ALL):
    return gl.maximum(a, b, propagate_nan=propagate_nan)


@gluon.jit
def reduce_max_prop_nan(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, elementwise_max_prop_nan, keep_dims=keep_dims)


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
        NUM_SMS,
        NUM_XCDS,
        NUM_BLOCKS,
    ):
        assert HEAD_DIM == 64
        assert NUM_WARPS == 4
        assert NUM_SMS % NUM_XCDS == 0
        assert NUM_BLOCKS % NUM_XCDS == 0
        assert NUM_BLOCKS >= N_HEADS

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
        self.MAX_SEQLEN = gl.constexpr(MAX_SEQLEN)
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
    def make_k_offset_tile(self):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        return (
            ((self.seq_base + offs_n[:, None]) * cfg.N_KV_HEADS + self.kv_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)

    @gluon.jit
    def make_v_offset_tile(self):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = (
            ((self.seq_base + offs_n[:, None]) * cfg.N_KV_HEADS + self.kv_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        return offsets, offs_n

    @gluon.jit
    def advance_offs_k(self, offsets):
        cfg = self.cfg
        return offsets + cfg.BLOCK_N * cfg.N_KV_HEADS * cfg.HEAD_DIM

    @gluon.jit
    def advance_offs_v(self, offsets):
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
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        row_max = reduce_max_prop_nan(qk, 1)
        m_new = elementwise_max_prop_nan(m_i, row_max)
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


@gluon.aggregate
class ProgramScheduler:
    cfg: AttentionConfig
    logical_pid: gl.tensor
    lane_valid: gl.tensor
    batch_slot: gl.tensor
    q_head: gl.tensor
    q_lane: gl.tensor
    num_q_blocks: gl.constexpr
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
        pids_per_xcd: gl.constexpr = cfg.NUM_BLOCKS // cfg.NUM_XCDS
        xcd = start_pid % cfg.NUM_XCDS
        local_pid = start_pid // cfg.NUM_XCDS
        logical_pid = xcd * pids_per_xcd + local_pid

        num_q_blocks: gl.constexpr = (cfg.MAX_SEQLEN + cfg.BLOCK_M - 1) // cfg.BLOCK_M

        max_batch_slots: gl.constexpr = cfg.NUM_BLOCKS // cfg.N_HEADS
        if cfg.BATCH_SIZE < max_batch_slots:
            batch_slots: gl.constexpr = cfg.BATCH_SIZE
        else:
            batch_slots: gl.constexpr = max_batch_slots

        q_lanes: gl.constexpr = cfg.NUM_BLOCKS // (batch_slots * cfg.N_HEADS)
        q_rounds_per_wave: gl.constexpr = (num_q_blocks + q_lanes - 1) // q_lanes
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
        batch_slot = head_batch_slot // cfg.N_HEADS

        return ProgramScheduler(
            cfg,
            logical_pid,
            lane_valid,
            batch_slot,
            q_head,
            q_lane,
            num_q_blocks,
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
        cu_seqlens_ptr,
    ):
        cfg = self.cfg

        batch_wave = q_round // self.q_rounds_per_wave
        wave_round = q_round - batch_wave * self.q_rounds_per_wave
        query_block_inc = wave_round * self.q_lanes + self.q_lane
        query_block_dec = wave_round * self.q_lanes + (self.q_lanes - 1 - self.q_lane)
        query_block = gl.where(wave_round % 2 == 0, query_block_inc, query_block_dec)
        batch = batch_wave * self.batch_slots + self.batch_slot
        valid = (
            self.lane_valid
            & (batch < cfg.BATCH_SIZE)
            & (query_block < self.num_q_blocks)
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
    acc = gl.zeros([cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout)

    main_end = program.q_start // cfg.BLOCK_N
    base_k_offsets = program.make_k_offset_tile()
    base_v_offsets, base_offs_n = program.make_v_offset_tile()

    k_offsets = base_k_offsets
    v_offsets = base_v_offsets
    offs_n = base_offs_n

    for _ in range(0, main_end):
        program.issue_buffer_load_k(k_offsets, k_smem)
        program.issue_buffer_load_v(v_offsets, v_smem)

        wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

        k_offsets = program.advance_offs_k(k_offsets)
        v_offsets = program.advance_offs_v(v_offsets)
        offs_n = offs_n + cfg.BLOCK_N

    # The main loop handles prefix tiles; the two boundary tiles are causal.
    boundary_offset = main_end * cfg.BLOCK_N * cfg.N_KV_HEADS * cfg.HEAD_DIM
    boundary_offs_n = main_end * cfg.BLOCK_N

    k_offsets = base_k_offsets + boundary_offset
    v_offsets = base_v_offsets + boundary_offset
    offs_n = base_offs_n + boundary_offs_n
    mask = offs_n[:, None] < program.seq_len
    program.issue_buffer_load_k(k_offsets, k_smem, mask=mask)
    program.issue_buffer_load_v(v_offsets, v_smem, mask=mask, other=0.0)

    wait_group(1)
    k = program.shared_load_k(k_smem)
    qk = program.compute_qk(q, k)
    qk = gl.where(boundary_mask0, qk, -float("inf"))
    p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

    wait_group(0)
    v = program.shared_load_v(v_smem)
    acc = program.compute_pv(p, v, acc)

    boundary_step = cfg.BLOCK_N * cfg.N_KV_HEADS * cfg.HEAD_DIM
    k_offsets = base_k_offsets + boundary_offset + boundary_step
    v_offsets = base_v_offsets + boundary_offset + boundary_step
    offs_n = base_offs_n + boundary_offs_n + cfg.BLOCK_N
    mask = offs_n[:, None] < program.seq_len
    program.issue_buffer_load_k(k_offsets, k_smem, mask=mask)
    program.issue_buffer_load_v(v_offsets, v_smem, mask=mask, other=0.0)

    wait_group(1)
    k = program.shared_load_k(k_smem)
    qk = program.compute_qk(q, k)
    qk = gl.where(boundary_mask1, qk, -float("inf"))
    p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

    wait_group(0)
    v = program.shared_load_v(v_smem)
    acc = program.compute_pv(p, v, acc)

    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


@gluon.jit
def _mha_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_ptr,
    output_ptr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SM_SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    MAX_SEQLEN: gl.constexpr,
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

    mask_offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
    mask_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    boundary_mask0 = mask_offs_n[None, :] <= mask_offs_m[:, None]
    boundary_mask1 = (mask_offs_n[None, :] + cfg.BLOCK_N) <= mask_offs_m[:, None]

    scheduler = ProgramScheduler.create(cfg)
    for q_round in range(0, scheduler.num_q_rounds):
        program, valid = scheduler.get_program(
            q_round, q_ptr, k_ptr, v_ptr, output_ptr, cu_seqlens_ptr
        )
        if valid:
            process_attention_tile(
                program, k_smem, v_smem, boundary_mask0, boundary_mask1
            )


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


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
        "sliding_window": frozenset({False}),
        "support_sinks": frozenset({False}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    },
    tags={"throughput"},
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
) -> torch.Tensor:
    if not is_causal:
        raise NotImplementedError(
            "gluon_mha_prefill_fp16_gfx950 requires causal prefill"
        )
    if window_left >= 0:
        raise NotImplementedError("sliding-window prefill is not implemented yet")
    if logit_cap != 0.0:
        raise NotImplementedError("logit_cap is not supported")
    if sinks is not None:
        raise NotImplementedError("attention sinks are not implemented yet")
    if return_lse:
        raise NotImplementedError("return_lse is not implemented yet")
    if max_seqlen_q != max_seqlen_k:
        raise ValueError("max_seqlen_q and max_seqlen_k must match")

    qkv = (q, k, v)
    if any(t.ndim != 3 for t in qkv):
        raise ValueError("q, k, and v must be 3D tensors")
    if any(not t.is_cuda for t in qkv):
        raise ValueError("q, k, and v must be CUDA tensors")
    if any(t.device != q.device for t in qkv):
        raise ValueError("q, k, and v must be on the same device")
    if any(t.dtype != q.dtype for t in qkv) or q.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("q, k, and v must have the same fp16 or bf16 dtype")
    if any(not t.is_contiguous() for t in qkv):
        raise ValueError("q, k, and v must be contiguous")

    total_tokens, n_heads, head_dim = q.shape
    if k.shape[0] != total_tokens or v.shape[0] != total_tokens:
        raise ValueError("q, k, and v must have the same flattened token count")
    if k.shape[1] != v.shape[1] or k.shape[2] != head_dim or v.shape[2] != head_dim:
        raise ValueError("q, k, and v must have compatible head shapes")
    if n_heads == 0 or k.shape[1] == 0 or head_dim == 0:
        raise ValueError("q, k, and v must have non-empty head dimensions")
    if n_heads % k.shape[1] != 0:
        raise ValueError("number of Q heads must be divisible by number of K/V heads")
    if head_dim != 64:
        raise ValueError("head_dim must be 64")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must be a 1D tensor with at least two elements")
    if max_seqlen_q < 0:
        raise ValueError("max_seqlen_q must be non-negative")

    cu_seqlens_q = cu_seqlens_q.to(
        device=q.device, dtype=torch.int32, non_blocking=True
    ).contiguous()

    block_m = 128
    block_n = 64
    num_warps = 4
    batch_size = cu_seqlens_q.numel() - 1
    num_sms = 256
    num_xcds = 8
    num_blocks = num_sms * 2
    grid = (num_blocks,)
    output = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    _mha_prefill_kernel[grid](
        q,
        k,
        v,
        cu_seqlens_q,
        output,
        n_heads,
        k.shape[1],
        head_dim,
        scale * 1.4426950408889634,
        block_m,
        block_n,
        num_warps,
        batch_size,
        max_seqlen_q,
        num_sms,
        num_xcds,
        num_blocks,
        num_warps=num_warps,
    )
    return output
