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

"""MHA extend (prefix-cache / chunked-prefill) Gluon kernel for AMD GFX950.

Ragged, multi-token queries against a paged KV cache. Each program owns one
kv-head and packs the GQA query-head group with ``BLOCK_Q`` query positions into
the MFMA ``M`` dimension (``BLOCK_M = BLOCK_Q * GROUP_SIZE``), so each KV tile is
loaded once and reused across the group. The grid is ragged and token-based,
``(total_num_q_blocks, n_kv_heads)``: each program binary-searches
``cu_seqlens_q`` to self-locate its request/q-block, keeping the launch
CUDA-graph static while a ``q=1`` request costs a single block.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.attention.gluon.utils import (
    _INV_LN2,
    _INV_LN2_VALUE,
    _LN2,
    InputStrides,
    attention_layouts,
    max,
    maximum,
    select_kv_splits,
)

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy


@gluon.jit
def _find_seq_idx(cu_seqlens_q_ptr, target_block, num_seqs, BLOCK_Q: gl.constexpr):
    # Map a ragged global q-block index to its request via binary search.
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(cu_seqlens_q_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_block:
            left = mid + 1
        else:
            right = mid
    return left - 1


@gluon.jit
def _resolve_grid(
    cu_seqlens_q_ptr,
    num_seqs,
    RAGGED: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    pid_qblock,
    pid_batch,
):
    # Resolve (batch, q_pos_base) from the launch grid:
    #   * rectangular ``(blocks_per_req, batch, n_kv_heads)``: ``pid_qblock`` is
    #     the block within the request and ``pid_batch`` the request.
    #   * ragged ``(total_num_q_blocks, ...)``: ``pid_qblock`` is a global q-block
    #     that binary-searches ``cu_seqlens_q`` to find its request.
    if RAGGED:
        batch = _find_seq_idx(cu_seqlens_q_ptr, pid_qblock, num_seqs, BLOCK_Q)
        seq_base = gl.load(cu_seqlens_q_ptr + batch)
        q_block_start = seq_base // BLOCK_Q + batch
        q_pos_base = (pid_qblock - q_block_start) * BLOCK_Q
    else:
        batch = pid_batch
        q_pos_base = pid_qblock * BLOCK_Q
    return batch, q_pos_base


# ===-----------------------------------------------------------------------===#
# Query-batched path
#
# This path tiles BLOCK_M query rows (of a single q-head) into the MFMA M
# dimension -- like the prefill kernel -- so each paged KV tile is loaded once
# and reused across all BLOCK_M rows. KV still comes from the paged cache
# (decode-style async page loads). Causal masking only touches the few KV tiles
# that reach the diagonal; the long prefix is a mask-free fast path.
# ===-----------------------------------------------------------------------===#

_EXTEND_BLOCK_N = 64
_EXTEND_SHORT_Q_BLOCK_M = 64
_EXTEND_SHORT_Q_NUM_WARPS = 2
_EXTEND_LONG_Q_BLOCK_M = 128
_EXTEND_LONG_Q_NUM_WARPS = 4


def _select_extend_tile(
    max_seqlen_q: int, group_size: int
) -> tuple[int, int, int, int]:
    # Return (BLOCK_M, BLOCK_Q, BLOCK_N, NUM_WARPS) for the given max query length.
    if max_seqlen_q <= _EXTEND_SHORT_Q_BLOCK_M:
        block_m, num_warps = _EXTEND_SHORT_Q_BLOCK_M, _EXTEND_SHORT_Q_NUM_WARPS
    else:
        block_m, num_warps = _EXTEND_LONG_Q_BLOCK_M, _EXTEND_LONG_Q_NUM_WARPS
    # Floor BLOCK_M down to a whole number of query positions.
    block_q = block_m // group_size
    if block_q < 1:
        block_q = 1
    block_m = block_q * group_size
    return block_m, block_q, _EXTEND_BLOCK_N, num_warps


@gluon.aggregate
class ExtendConfig:
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    GROUP_SIZE: gl.constexpr
    HEAD_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_Q: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    NUM_KV_SPLITS: gl.constexpr
    PAGE_SIZE: gl.constexpr
    PAGE_TABLE_STRIDE: gl.constexpr
    IS_CAUSAL: gl.constexpr
    HAS_SINK: gl.constexpr
    HAS_LSE: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    IS_FP8: gl.constexpr
    q_strides: InputStrides
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
        BLOCK_Q,
        BLOCK_N,
        NUM_WARPS,
        NUM_KV_SPLITS,
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        HAS_SINK,
        HAS_LSE,
        WINDOW_LEFT,
        IS_FP8,
        KV_DTYPE,
        q_strides,
    ):
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE
        assert BLOCK_M == BLOCK_Q * (N_HEADS // N_KV_HEADS)

        # Extend uses a [32, 32, 16] MFMA with NUM_WARPS warp tiling.
        (
            qk_layout,
            pv_layout,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            load_layout,
            store_layout,
            k_smem_layout,
            v_smem_layout,
        ) = attention_layouts(
            HEAD_DIM,
            BLOCK_N,
            IS_FP8,
            KV_DTYPE,
            num_warps=NUM_WARPS,
            instr_shape=[32, 32, 16],
        )

        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.GROUP_SIZE = gl.constexpr(N_HEADS // N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_Q = gl.constexpr(BLOCK_Q)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.NUM_KV_SPLITS = gl.constexpr(NUM_KV_SPLITS)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.PAGE_TABLE_STRIDE = gl.constexpr(PAGE_TABLE_STRIDE)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.HAS_SINK = gl.constexpr(HAS_SINK)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.q_strides = q_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


@gluon.aggregate
class ExtendProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    k_cache_ptr: gl.tensor
    v_cache_ptr: gl.tensor
    page_table_ptr: gl.tensor
    output_ptr: gl.tensor
    lse_ptr: gl.tensor
    sink_ptr: gl.tensor
    batch: gl.tensor
    kv_head: gl.tensor
    q_pos_base: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    prefix: gl.tensor
    cache_len: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        sink_ptr,
        batch,
        kv_head,
        q_pos_base,
        seq_base,
        seq_len,
        prefix,
        cache_len,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.k_cache_ptr = k_cache_ptr
        self.v_cache_ptr = v_cache_ptr
        self.page_table_ptr = page_table_ptr
        self.output_ptr = output_ptr
        self.lse_ptr = lse_ptr
        self.sink_ptr = sink_ptr
        self.batch = batch
        self.kv_head = kv_head
        self.q_pos_base = q_pos_base
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.prefix = prefix
        self.cache_len = cache_len

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        sink_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        batch,
        kv_head,
        q_pos_base,
    ):
        seq_base = gl.load(cu_seqlens_q_ptr + batch)
        seq_end = gl.load(cu_seqlens_q_ptr + batch + 1)
        seq_len = seq_end - seq_base
        cache_len = gl.load(cache_seqlens_ptr + batch)
        prefix = cache_len - seq_len
        return ExtendProgram(
            gl.constexpr(cfg),
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            output_ptr,
            lse_ptr,
            sink_ptr,
            batch,
            kv_head,
            q_pos_base,
            seq_base,
            seq_len,
            prefix,
            cache_len,
        )

    @gluon.jit
    def row_qpos(self, offs_m):
        # Head-minor packing: row m -> query position (m // GROUP_SIZE).
        return self.q_pos_base + offs_m // self.cfg.GROUP_SIZE

    @gluon.jit
    def row_qhead(self, offs_m):
        # Head-minor packing: row m -> group head (m % GROUP_SIZE).
        return self.kv_head * self.cfg.GROUP_SIZE + offs_m % self.cfg.GROUP_SIZE

    @gluon.jit
    def load_q(self):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        q_pos = self.row_qpos(offs_m)
        q_head = self.row_qhead(offs_m)
        row = self.seq_base + q_pos
        offsets = cfg.q_strides.offsets(row[:, None], q_head[:, None], offs_d[None, :])
        mask = q_pos[:, None] < self.seq_len
        return cdna4.buffer_load(self.q_ptr, offsets, mask=mask, other=0.0)

    @gluon.jit
    def load_page(self, start_n):
        cfg = self.cfg
        page_index = start_n // cfg.PAGE_SIZE
        valid = start_n < self.cache_len
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index,
            mask=valid,
            other=0,
        )

    @gluon.jit
    def issue_load_k(self, physical_page, k_smem):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        token_loc = physical_page.to(gl.int64) * cfg.PAGE_SIZE + offs_n.to(gl.int64)
        offsets = (
            token_loc[:, None] * cfg.N_KV_HEADS * cfg.HEAD_DIM
            + self.kv_head * cfg.HEAD_DIM
            + offs_d[None, :]
        )
        async_copy.global_load_to_shared(k_smem, self.k_cache_ptr + offsets)
        async_copy.commit_group()

    @gluon.jit
    def issue_load_v(self, physical_page, v_smem):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        token_loc = physical_page.to(gl.int64) * cfg.PAGE_SIZE + offs_n.to(gl.int64)
        offsets = (
            token_loc[:, None] * cfg.N_KV_HEADS * cfg.HEAD_DIM
            + self.kv_head * cfg.HEAD_DIM
            + offs_d[None, :]
        )
        async_copy.global_load_to_shared(v_smem, self.v_cache_ptr + offsets)
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
            offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
            q_head = self.row_qhead(offs_m)
            sink_log2 = gl.load(self.sink_ptr + q_head).to(gl.float32) * _INV_LN2
            m_i = sink_log2 / cfg.SM_SCALE
        else:
            sink_log2 = gl.full(
                [cfg.BLOCK_M],
                value=0.0,
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout),
            )
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
        return m_i, l_i, acc, sink_log2

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc, allow_invalid: gl.constexpr = False):
        cfg = self.cfg
        # In sliding window case, some rows can see fully masked tiles before
        # any valid KV. Guard the online softmax state so `-inf - -inf` does not
        # produce NaNs. This does not happen when having sink, because m_i
        # is initialized to sink value instead of -inf. Under split-K a whole
        # (non-empty) KV split can also sit above a row's causal diagonal, so
        # the split-K compute passes allow_invalid=True to enable the guard.
        HAS_INVALID: gl.constexpr = (
            cfg.WINDOW_LEFT >= 0 or allow_invalid
        ) and not cfg.HAS_SINK

        row_max = max(qk, 1)
        m_new = maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        if HAS_INVALID:
            invalid = m_new == -float("inf")
            m_new_scaled = gl.where(invalid, 0.0, m_new_scaled)

        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        if HAS_INVALID:
            m_diff = gl.where(invalid, 0.0, m_diff)

        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def apply_sinks(self, l_i, m_i, sink_log2):
        cfg = self.cfg
        if cfg.HAS_SINK:
            l_i += gl.exp2(sink_log2 - m_i * cfg.SM_SCALE)
        return l_i

    @gluon.jit
    def store_output(self, output):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        q_pos = self.row_qpos(offs_m)
        q_head = self.row_qhead(offs_m)
        offsets = (
            ((self.seq_base + q_pos[:, None]) * cfg.N_HEADS + q_head[:, None])
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        mask = q_pos[:, None] < self.seq_len
        output = output.to(self.output_ptr.dtype.element_ty)
        cdna4.buffer_store(output, self.output_ptr, offsets, mask=mask)

    @gluon.jit
    def store_lse(self, l_i, m_i):
        cfg = self.cfg
        if cfg.HAS_LSE:
            offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
            q_pos = self.row_qpos(offs_m)
            q_head = self.row_qhead(offs_m)
            offsets = ((self.seq_base + q_pos) * cfg.N_HEADS + q_head).to(gl.int32)
            mask = q_pos < self.seq_len
            lse_l_i = gl.where(l_i > 0.0, l_i, 1.0)
            # Softmax runs in base-2 (exp2 hardware fast path), so m_i*SM_SCALE +
            # log2(l_i) is the LSE in base-2 units. Convert to natural log (the
            # public op contract / torch.logsumexp convention) by scaling by ln2.
            lse = (m_i * cfg.SM_SCALE + gl.log2(lse_l_i)) * _LN2
            cdna4.buffer_store(lse, self.lse_ptr, offsets, mask=mask)

    @gluon.jit
    def store_partial(self, acc, l_i, m_i, split_id, mid_o_ptr, mid_lse_ptr):
        # Split-K partial store: mid_o[row, head, split] = acc / l_i (this
        # split's local softmax normalization) and mid_lse[row, head, split] =
        # m_i * SM_SCALE + log2(l_i) in base-2 units, matching the decode
        # kernel's store_split so a shared reduce can merge splits. Rows whose
        # split saw no valid key (l_i == 0, e.g. a split entirely above the
        # causal diagonal) store a -inf lse / zero output so the reduce ignores
        # them without producing NaNs.
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        q_pos = self.row_qpos(offs_m)
        q_head = self.row_qhead(offs_m)
        row = self.seq_base + q_pos
        valid = q_pos < self.seq_len
        has_kv = l_i > 0.0
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        m_i = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.store_layout))
        has_kv = gl.convert_layout(has_kv, gl.SliceLayout(1, cfg.store_layout))
        inv_l = gl.where(has_kv, 1.0 / gl.where(has_kv, l_i, 1.0), 0.0)
        part_o = acc * inv_l[:, None]
        part_lse = gl.where(
            has_kv,
            m_i * cfg.SM_SCALE + gl.log2(gl.where(has_kv, l_i, 1.0)),
            -float("inf"),
        )
        o_off = (
            (row[:, None] * cfg.N_HEADS + q_head[:, None]) * cfg.NUM_KV_SPLITS
            + split_id
        ) * cfg.HEAD_DIM + offs_d[None, :]
        lse_off = (row * cfg.N_HEADS + q_head) * cfg.NUM_KV_SPLITS + split_id
        cdna4.buffer_store(part_o, mid_o_ptr, o_off, mask=valid[:, None])
        cdna4.buffer_store(part_lse, mid_lse_ptr, lse_off, mask=valid)


@gluon.jit
def _mha_extend(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    output_ptr,
    lse_ptr,
    sink_ptr,
    cu_seqlens_q_ptr,
    cache_seqlens_ptr,
    num_seqs,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    RAGGED: gl.constexpr,
):
    cfg = ExtendConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_Q,
        BLOCK_N,
        NUM_WARPS,
        1,  # NUM_KV_SPLITS: single-pass path has no split-K
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        HAS_SINK,
        HAS_LSE,
        WINDOW_LEFT,
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
    )
    if RAGGED:
        kv_head = gl.program_id(1)
    else:
        kv_head = gl.program_id(2)
    batch, q_pos_base = _resolve_grid(
        cu_seqlens_q_ptr,
        num_seqs,
        RAGGED,
        cfg.BLOCK_Q,
        gl.program_id(0),
        gl.program_id(1),
    )
    program = ExtendProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        sink_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        batch,
        kv_head,
        q_pos_base,
    )
    # Over-provisioned tile past this request's real query rows: nothing to do.
    if program.q_pos_base >= program.seq_len:
        return
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_attention_state()

    # Row m packs (query position m // GROUP_SIZE, group head m % GROUP_SIZE);
    # the causal/sliding masks key off the per-row query position only (heads in
    # a group share visibility), so diag_row broadcasts across the packed heads.
    offs_m_q = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
    q_pos = program.q_pos_base + offs_m_q // cfg.GROUP_SIZE
    offs_n_q = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    diag_row = program.prefix + q_pos

    if IS_CAUSAL:
        # Deepest diagonal in this tile is the last packed query position.
        kv_end = min(
            program.cache_len, program.prefix + program.q_pos_base + cfg.BLOCK_Q
        )
    else:
        kv_end = program.cache_len

    # Sliding window (inclusive-left, matches flash-attn window_size=(W, 0)):
    # skip KV tiles entirely below the window's lower edge. The tile's top query
    # position sits at absolute position pos_top; its window opens at
    # pos_top - WINDOW_LEFT (that key is visible -> W + 1 keys total). The min()
    # form clamps to 0 without the shadowed builtin max().
    if cfg.WINDOW_LEFT >= 0:
        pos_top = program.prefix + program.q_pos_base
        kv_start = pos_top - min(pos_top, cfg.WINDOW_LEFT)
        kv_start = (kv_start // cfg.BLOCK_N) * cfg.BLOCK_N
    else:
        kv_start = 0

    for start_n in range(kv_start, kv_end, cfg.BLOCK_N):
        physical_page = program.load_page(start_n)
        program.issue_load_k(physical_page, k_smem)
        program.issue_load_v(physical_page, v_smem)

        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)

        if cfg.WINDOW_LEFT >= 0:
            # Window lower edge + cache bound always apply; causal upper edge only
            # when IS_CAUSAL (independent layering keeps non-causal + window correct).
            offs_n_abs = start_n + offs_n_q
            mask = (offs_n_abs[None, :] >= diag_row[:, None] - cfg.WINDOW_LEFT) & (
                offs_n_abs[None, :] < program.cache_len
            )
            if IS_CAUSAL:
                mask &= offs_n_abs[None, :] <= diag_row[:, None]
            qk = gl.where(mask, qk, -float("inf"))
        elif IS_CAUSAL:
            if start_n + cfg.BLOCK_N > program.prefix + program.q_pos_base:
                offs_n_abs = start_n + offs_n_q
                mask = (offs_n_abs[None, :] <= diag_row[:, None]) & (
                    offs_n_abs[None, :] < program.cache_len
                )
                qk = gl.where(mask, qk, -float("inf"))
        else:
            if start_n + cfg.BLOCK_N > program.cache_len:
                offs_n_abs = start_n + offs_n_q
                qk = gl.where(
                    offs_n_abs[None, :] < program.cache_len, qk, -float("inf")
                )

        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    l_i = program.apply_sinks(l_i, m_i, sink_log2)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)
    program.store_lse(l_i, m_i)


@gluon.jit
def _mha_extend_split(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    sink_ptr,
    cu_seqlens_q_ptr,
    cache_seqlens_ptr,
    num_seqs,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_FP8: gl.constexpr,
    RAGGED: gl.constexpr,
):
    # Split-K compute pass. Two grids share the body:
    #   * ragged (total_num_q_blocks, n_kv_heads, NUM_KV_SPLITS): program_id(0)
    #     self-locates its request/q-block, program_id(2) is the KV split.
    #   * rectangular (blocks_per_req * NUM_KV_SPLITS, batch, n_kv_heads):
    #     program_id(0) folds the query block and KV split.
    # Each program streams only its slice of the request's visible KV range and
    # writes a per-split partial (normalized O + base-2 LSE). Sinks are NOT folded
    # here -- a sink is a single global softmax entry, so the reduce adds it once
    # (folding it per split would count it NUM_KV_SPLITS times). HAS_SINK is
    # therefore False in this config so the online softmax starts from -inf.
    cfg = ExtendConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_Q,
        BLOCK_N,
        NUM_WARPS,
        NUM_KV_SPLITS,
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        False,  # HAS_SINK: added once in the reduce, not per split
        False,  # HAS_LSE: split partials store base-2 lse unconditionally
        -1,  # WINDOW_LEFT: split-K is full-attention only
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
    )
    if RAGGED:
        kv_head = gl.program_id(1)
        split_id = gl.program_id(2)
        pid_qblock = gl.program_id(0)
    else:
        kv_head = gl.program_id(2)
        split_id = gl.program_id(0) % NUM_KV_SPLITS
        pid_qblock = gl.program_id(0) // NUM_KV_SPLITS
    batch, q_pos_base = _resolve_grid(
        cu_seqlens_q_ptr,
        num_seqs,
        RAGGED,
        cfg.BLOCK_Q,
        pid_qblock,
        gl.program_id(1),
    )
    program = ExtendProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        sink_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        batch,
        kv_head,
        q_pos_base,
    )
    # Over-provisioned tile past this request's real query rows: nothing to do.
    if program.q_pos_base >= program.seq_len:
        return

    # Deepest key this query block can see (causal) or the whole cache.
    if IS_CAUSAL:
        kv_end = min(
            program.cache_len, program.prefix + program.q_pos_base + cfg.BLOCK_Q
        )
    else:
        kv_end = program.cache_len

    # Partition [0, kv_end) into NUM_KV_SPLITS page-aligned slices (BLOCK_N ==
    # PAGE_SIZE), mirroring the decode kernel's split math.
    num_pages = gl.cdiv(kv_end, cfg.PAGE_SIZE)
    pages_per_split = gl.cdiv(num_pages, cfg.NUM_KV_SPLITS)
    split_start_page = split_id * pages_per_split
    split_end_page = min(split_start_page + pages_per_split, num_pages)
    split_start = split_start_page * cfg.PAGE_SIZE
    split_end = min(split_end_page * cfg.PAGE_SIZE, kv_end)

    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_attention_state()

    offs_m_q = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
    q_pos = program.q_pos_base + offs_m_q // cfg.GROUP_SIZE
    offs_n_q = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    diag_row = program.prefix + q_pos

    for start_n in range(split_start, split_end, cfg.BLOCK_N):
        physical_page = program.load_page(start_n)
        program.issue_load_k(physical_page, k_smem)
        program.issue_load_v(physical_page, v_smem)

        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)

        if IS_CAUSAL:
            if start_n + cfg.BLOCK_N > program.prefix + program.q_pos_base:
                offs_n_abs = start_n + offs_n_q
                mask = (offs_n_abs[None, :] <= diag_row[:, None]) & (
                    offs_n_abs[None, :] < program.cache_len
                )
                qk = gl.where(mask, qk, -float("inf"))
        else:
            if start_n + cfg.BLOCK_N > program.cache_len:
                offs_n_abs = start_n + offs_n_q
                qk = gl.where(
                    offs_n_abs[None, :] < program.cache_len, qk, -float("inf")
                )

        # allow_invalid: a whole split can sit above a row's causal diagonal.
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc, allow_invalid=True)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    program.store_partial(acc, l_i, m_i, split_id, mid_o_ptr, mid_lse_ptr)


@gluon.jit
def _mha_extend_reduce(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    lse_out_ptr,
    sink_ptr,
    SM_SCALE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    N_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
):
    # Combine the NUM_KV_SPLITS partials for one (query token, head) with a
    # global softmax rescale. Empty splits carry -inf lse (written by the
    # compute pass) so they drop out. Grid is (total_q, N_HEADS).
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [1, HEAD_DIM // 64], [1, 64], [1, 1], [1, 0]
    )
    row = gl.program_id(0)
    q_head = gl.program_id(1)

    # SPLIT_TILE pads NUM_KV_SPLITS up to a power of 2 for the tensor shape.
    SPLIT_TILE: gl.constexpr = 1 << (NUM_KV_SPLITS - 1).bit_length()
    offs_s = gl.arange(0, SPLIT_TILE, layout=gl.SliceLayout(1, reduce_layout))
    offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, reduce_layout))
    split_valid = offs_s < NUM_KV_SPLITS
    base = (row * N_HEADS + q_head) * NUM_KV_SPLITS + offs_s
    part_lse = gl.load(mid_lse_ptr + base, mask=split_valid, other=-float("inf"))
    o_off = base[:, None] * HEAD_DIM + offs_d[None, :]
    part_o = cdna4.buffer_load(mid_o_ptr, o_off, mask=split_valid[:, None], other=0.0)

    m_i = max(part_lse, axis=0)
    if HAS_SINK:
        sink = gl.load(sink_ptr + q_head).to(gl.float32) * _INV_LN2
        m_i = maximum(m_i, sink)
    beta = gl.exp2(part_lse - m_i)
    l_i = gl.sum(beta, axis=0)
    if HAS_SINK:
        l_i = l_i + gl.exp2(sink - m_i)
    acc = gl.sum(part_o * beta[:, None], axis=0)

    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)
    output = output.to(out_ptr.dtype.element_ty)
    out_base = (row * N_HEADS + q_head) * HEAD_DIM
    cdna4.buffer_store(output, out_ptr, out_base + offs_d)
    if HAS_LSE:
        lse = (m_i + gl.log2(denom)) * _LN2
        gl.store(lse_out_ptr + (row * N_HEADS + q_head), lse)


_GFX950_SM_COUNT = 256


def gluon_mha_extend_gfx950(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 1,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[2]
    n_heads = q.shape[1]
    n_kv_heads = k_cache.shape[2]
    group_size = n_heads // n_kv_heads
    page_size = k_cache.shape[1]
    block_m, block_q, block_n, num_warps = _select_extend_tile(max_seqlen_q, group_size)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = softmax_scale * _INV_LN2_VALUE

    # Grid selection. Two layouts share the kernel body:
    #   * rectangular (blocks_per_req * batch blocks): tight when every request
    #     has the same query length (per-stage 3-launch dispatch).
    #   * ragged (total_q // block_q + batch blocks, AITER-style): each program
    #     binary-searches cu_seqlens_q to self-locate, so a q=1 request costs one
    #     block instead of the batch-wide max. Tight for mixed query lengths
    #     (the fused single call).
    # Pick whichever launches fewer blocks: the ragged bound only wins when query
    # lengths vary, and the rectangular bound only wins when they're uniform, so
    # this avoids the ragged +num_seqs padding regressing uniform-q launches
    # while still unlocking the fused mixed path.
    batch = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    safe_max_q = max_seqlen_q if max_seqlen_q > 0 else 1
    blocks_per_req = (safe_max_q + block_q - 1) // block_q
    rect_num_q_blocks = blocks_per_req * batch
    ragged_num_q_blocks = total_q // block_q + batch
    use_ragged = ragged_num_q_blocks < rect_num_q_blocks
    grid_num_q_blocks = ragged_num_q_blocks if use_ragged else rect_num_q_blocks
    has_sink = sinks is not None
    sink_arg = sinks if has_sink else q
    cu_q_i32 = cu_seqlens_q.to(torch.int32).contiguous()
    cache_i32 = cache_seqlens.to(torch.int32).contiguous()

    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    output = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    if return_lse:
        lse = torch.empty((q.shape[0], n_heads), device=q.device, dtype=torch.float32)
        lse_arg = lse
    else:
        lse = None
        lse_arg = q

    # Split-K (full attention only): when the single-pass grid under-fills the
    # machine (low batch / few query blocks) but the KV is long, split each
    # request's KV stream across programs and merge with a reduce -- the same
    # 2D/3D switch AITER's unified kernel and our decode kernel make.
    is_sliding = window_left >= 0
    num_kv_splits = 1
    # Unlike decode (one q-pos per program -> a single scalar window), an extend
    # tile packs many q-positions each with its own window, so the page-range
    # split can't express a per-row window: split-K is full-attention only here.
    if not is_sliding:
        num_pages = (max_seqlen_k + page_size - 1) // page_size
        num_kv_splits = select_kv_splits(
            base_ctas=grid_num_q_blocks * n_kv_heads,
            num_pages=num_pages,
            sm_count=_GFX950_SM_COUNT,
        )

    if num_kv_splits > 1:
        mid_o = torch.empty(
            (total_q, n_heads, num_kv_splits, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        mid_lse = torch.empty(
            (total_q, n_heads, num_kv_splits),
            device=q.device,
            dtype=torch.float32,
        )
        if use_ragged:
            split_grid = (ragged_num_q_blocks, n_kv_heads, num_kv_splits)
        else:
            split_grid = (blocks_per_req * num_kv_splits, batch, n_kv_heads)
        _mha_extend_split[split_grid](
            q,
            k_cache,
            v_cache,
            page_table,
            mid_o,
            mid_lse,
            sink_arg,
            cu_q_i32,
            cache_i32,
            batch,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            sm_scale,
            page_table.stride(0),
            page_size,
            n_heads,
            n_kv_heads,
            head_dim,
            block_m,
            block_q,
            block_n,
            num_warps,
            num_kv_splits,
            is_causal,
            is_fp8,
            use_ragged,
            num_warps=num_warps,
        )
        reduce_grid = (total_q, n_heads)
        _mha_extend_reduce[reduce_grid](
            mid_o,
            mid_lse,
            output,
            lse_arg,
            sink_arg,
            sm_scale,
            num_kv_splits,
            n_heads,
            head_dim,
            has_sink,
            return_lse,
            num_warps=1,
        )
        if return_lse:
            return output, lse
        return output

    if use_ragged:
        grid = (ragged_num_q_blocks, n_kv_heads)
    else:
        grid = (blocks_per_req, batch, n_kv_heads)
    _mha_extend[grid](
        q,
        k_cache,
        v_cache,
        page_table,
        output,
        lse_arg,
        sink_arg,
        cu_q_i32,
        cache_i32,
        batch,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sm_scale,
        page_table.stride(0),
        page_size,
        n_heads,
        n_kv_heads,
        head_dim,
        block_m,
        block_q,
        block_n,
        num_warps,
        is_causal,
        has_sink,
        return_lse,
        window_left,
        is_fp8,
        use_ragged,
        num_warps=num_warps,
    )
    if return_lse:
        return output, lse
    return output
