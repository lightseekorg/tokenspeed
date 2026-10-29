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

"""MHA decode Gluon kernel optimized for AMD GFX950."""

from __future__ import annotations

import builtins
import math
from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._scheduling import sched_compile_options, wave_uniform_i32
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import (
    _INV_LN2,
    _INV_LN2_VALUE,
    InputStrides,
    attention_layouts,
    fp8_lse_shift,
    fp8_probability_shift,
    max,
    maximum,
    reject_block_scales,
    select_kv_splits,
)

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy
buffer = cdna4.buffer
cdiv = gl.cdiv
_GFX950_SM_COUNT = 256
# Pipeline depth of the full-attention decode K/V page loads: LDS buffers per
# work-group, so NUM_STAGES - 1 pages are prefetched while one is consumed. At
# BF16 head dim 64 the three prefetched pages are 48 buffer_load ... lds copies
# for a one-wave work-group; with more waves, each wave issues its share. The
# loop is either this deep or single-buffered.
_FULL_DECODE_STAGES = 4
# s_waitcnt vmcnt(N) takes a 6-bit count: a wave can wait for all but at most
# 63 of its outstanding vector memory loads.
_GFX950_MAX_VMCNT = 63
# Full-attention decode takes its split count as a constexpr, so it is not held to
# the runtime-reduce bound MAX_KV_SPLITS. Above 64, the split reduce and the
# per-split prologue outweigh the extra waves.
_FULL_DECODE_MAX_SPLITS = 64
# Rows of a decode work-group: four waves of one 16-row MFMA tile each.
_DECODE_MAX_BLOCK_M = 64
# Sliding windows, or contexts, of at most this many pages are short: their
# rows take one-wave work-groups, and their split counts follow the pages.
_SHORT_WINDOW_PAGES = 16
_SLIDING_SPLIT_COUNTERS: dict[tuple[int, int, int, int], torch.Tensor] = {}


# ===-----------------------------------------------------------------------===#
# Kernel Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionConfig:
    SM_SCALE: gl.constexpr
    PAGE_TABLE_STRIDE: gl.constexpr
    PAGE_SIZE: gl.constexpr
    NUM_KV_SPLITS: gl.constexpr
    MAX_SEQLEN_Q: gl.constexpr
    NUM_Q_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    HEAD_DIM: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    IS_SLIDING: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    IS_FP8: gl.constexpr
    GROUP_SIZE: gl.constexpr
    ROWS: gl.constexpr
    NUM_ROW_BLOCKS: gl.constexpr
    q_strides: InputStrides
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    store_layout: gl.constexpr
    reduce_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
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
        IS_SLIDING,
        WINDOW_LEFT,
        IS_FP8,
        KV_DTYPE,
        q_strides,
    ):
        assert NUM_Q_HEADS % NUM_KV_HEADS == 0
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE
        # The waves split the rows into whole 16-row MFMA tiles.
        assert BLOCK_M % (16 * NUM_WARPS) == 0
        if IS_SLIDING:
            assert WINDOW_LEFT >= 0
        else:
            assert WINDOW_LEFT == -1

        # Decode uses a [16, 16, 32] MFMA, the waves tiling the rows.
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
            instr_shape=[16, 16, 32],
        )
        reduce_layout = gl.BlockedLayout([1, HEAD_DIM // 64], [1, 64], [1, 1], [1, 0])

        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.PAGE_TABLE_STRIDE = gl.constexpr(PAGE_TABLE_STRIDE)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.NUM_KV_SPLITS = gl.constexpr(NUM_KV_SPLITS)
        self.MAX_SEQLEN_Q = gl.constexpr(MAX_SEQLEN_Q)
        self.NUM_Q_HEADS = gl.constexpr(NUM_Q_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.IS_SLIDING = gl.constexpr(IS_SLIDING)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.GROUP_SIZE = gl.constexpr(NUM_Q_HEADS // NUM_KV_HEADS)
        # A request's queries for one KV head form MAX_SEQLEN_Q * GROUP_SIZE
        # rows, ordered by query position and then by query head, tiled into
        # BLOCK_M-row blocks.
        self.ROWS = gl.constexpr(MAX_SEQLEN_Q * self.GROUP_SIZE)
        self.NUM_ROW_BLOCKS = gl.constexpr((self.ROWS + BLOCK_M - 1) // BLOCK_M)
        self.q_strides = q_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.reduce_layout = gl.constexpr(reduce_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


# ===-----------------------------------------------------------------------===#
# Kernel Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    k_cache_ptr: gl.tensor
    v_cache_ptr: gl.tensor
    page_table_ptr: gl.tensor
    cache_seqlens_ptr: gl.tensor
    mid_o_ptr: gl.tensor
    mid_lse_ptr: gl.tensor
    batch: gl.tensor
    kv_head: gl.tensor
    row_start: gl.tensor
    split_id: gl.tensor
    seq_len: gl.tensor
    cache_len: gl.tensor
    kv_start: gl.tensor
    split_start: gl.tensor
    split_end: gl.tensor
    mask_start: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        batch,
        kv_head,
        row_start,
        split_id,
        seq_len,
        cache_len,
        kv_start,
        split_start,
        split_end,
        mask_start,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.k_cache_ptr = k_cache_ptr
        self.v_cache_ptr = v_cache_ptr
        self.page_table_ptr = page_table_ptr
        self.cache_seqlens_ptr = cache_seqlens_ptr
        self.mid_o_ptr = mid_o_ptr
        self.mid_lse_ptr = mid_lse_ptr
        self.batch = batch
        self.kv_head = kv_head
        self.row_start = row_start
        self.split_id = split_id
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.kv_start = kv_start
        self.split_start = split_start
        self.split_end = split_end
        self.mask_start = mask_start

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
    ):
        """Locate this program's request, KV head, row block and split.

        The rows of a block share one pass over the union of the keys they
        see: up to its last row's causal end and, under a sliding window, from
        its first row's window start. A request's cache length counts all of
        its MAX_SEQLEN_Q query tokens, so query position p sees the keys
        before seq_len - (MAX_SEQLEN_Q - 1 - p).
        """
        batch = gl.program_id(0)
        head_block = gl.program_id(1)
        kv_head = head_block // cfg.NUM_ROW_BLOCKS
        row_start = (head_block - kv_head * cfg.NUM_ROW_BLOCKS) * cfg.BLOCK_M
        split_id = gl.program_id(2)
        seq_len = gl.load(cache_seqlens_ptr + batch)
        if cfg.MAX_SEQLEN_Q == 1:
            first_end = maximum(seq_len, 0)
            cache_len = first_end
        else:
            last_row = min(row_start + cfg.BLOCK_M, cfg.ROWS) - 1
            first_pos = row_start // cfg.GROUP_SIZE
            first_end = maximum(seq_len - (cfg.MAX_SEQLEN_Q - 1 - first_pos), 0)
            last_pos = last_row // cfg.GROUP_SIZE
            cache_len = maximum(seq_len - (cfg.MAX_SEQLEN_Q - 1 - last_pos), 0)
        if cfg.IS_SLIDING:
            # WINDOW_LEFT is defined as exclusive (keys strictly to the left, not
            # counting the current token), so the window is WINDOW_LEFT + 1 keys
            # once the current token is included. E.g. with
            # WINDOW_LEFT = 127 and cache_len = 500 (current token at index 499),
            # kv_start = 500 - (127 + 1) = 372 keeps keys [372, 499] = 128 keys.
            # Without the + 1, kv_start = 373 would drop the leftmost key.
            kv_start = first_end - min(first_end, cfg.WINDOW_LEFT + 1)
        else:
            kv_start = cache_len - cache_len
        first_page = kv_start // cfg.PAGE_SIZE
        end_page = cdiv(cache_len, cfg.PAGE_SIZE)
        num_pages = end_page - first_page
        pages_per_split = cdiv(num_pages, cfg.NUM_KV_SPLITS)
        split_start_page = first_page + split_id * pages_per_split
        split_end_page = min(split_start_page + pages_per_split, end_page)
        split_start = split_start_page * cfg.PAGE_SIZE
        split_end = min(split_end_page * cfg.PAGE_SIZE, cache_len)
        # Under full attention, every row of the block sees the split's keys
        # before mask_start; the pipelined path that skips their mask is
        # full-attention only.
        mask_start = min(split_end, first_end)
        return AttentionProgram(
            gl.constexpr(cfg),
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            cache_seqlens_ptr,
            mid_o_ptr,
            mid_lse_ptr,
            batch,
            kv_head,
            row_start,
            split_id,
            seq_len,
            cache_len,
            kv_start,
            split_start,
            split_end,
            mask_start,
        )

    @gluon.jit
    def row_query_pos(self, offs_m):
        """Query position within the request of each row of the block."""
        cfg = self.cfg
        if cfg.MAX_SEQLEN_Q == 1:
            query_pos = offs_m * 0
        else:
            # Padding rows past ROWS clamp onto the last query position.
            query_pos = gl.minimum(
                (self.row_start + offs_m) // cfg.GROUP_SIZE, cfg.MAX_SEQLEN_Q - 1
            )
        return query_pos

    @gluon.jit
    def row_tokens(self, offs_m):
        """Query token (row of q and the output) of each row of the block."""
        return self.batch * self.cfg.MAX_SEQLEN_Q + self.row_query_pos(offs_m)

    @gluon.jit
    def row_heads(self, offs_m):
        """Query head of each row of the block."""
        cfg = self.cfg
        rows = self.row_start + offs_m
        if cfg.MAX_SEQLEN_Q > 1:
            rows = rows % cfg.GROUP_SIZE
        return self.kv_head * cfg.GROUP_SIZE + rows

    @gluon.jit
    def row_valid(self, offs_m):
        return (self.row_start + offs_m) < self.cfg.ROWS

    @gluon.jit
    def load_q(self):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        offsets = cfg.q_strides.offsets(
            self.row_tokens(offs_m)[:, None],
            self.row_heads(offs_m)[:, None],
            offs_d[None, :],
        )
        valid = self.row_valid(offs_m)
        return cdna4.buffer_load(self.q_ptr, offsets, mask=valid[:, None], other=0.0)

    @gluon.jit
    def init_state(self, sink_ptr, has_sink: gl.constexpr):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
        q_heads = self.row_heads(offs_m)
        valid = self.row_valid(offs_m)
        if has_sink:
            sink_log2 = (
                gl.load(sink_ptr + q_heads, mask=valid, other=0.0).to(gl.float32)
                * _INV_LN2
            )
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
            value=0.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout
        )
        return m_i, l_i, acc, sink_log2

    @gluon.jit
    def apply_sinks(self, l_i, m_i, sink_log2, has_sink: gl.constexpr):
        cfg = self.cfg
        if has_sink:
            l_i += gl.exp2(sink_log2 - fp8_lse_shift(m_i, cfg.SM_SCALE, cfg.IS_FP8))
        return l_i

    @gluon.jit
    def load_page(self, start_n):
        cfg = self.cfg
        page_index = start_n // cfg.PAGE_SIZE
        valid = start_n < self.split_end
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index,
            mask=valid,
            other=0,
        )

    @gluon.jit
    def load_page_indices(self, start_n, count: gl.constexpr):
        """Physical pages of the count pages starting at token start_n.

        Pages past the split repeat its last page instead of being masked: a
        masked copy compiles to a scalar branch around the buffer_load, and
        after it the compiler cannot count the wave's outstanding loads, so
        it waits for all of them (s_waitcnt vmcnt(0)), prefetched pages
        included. The repeated page was just loaded and hits in L2. An empty
        split reads the request's first entry; select_page replaces it with
        page 0 for a block without visible keys. vmcnt retires a wave's loads
        in issue order, so reading this result waits for every load issued
        before it; the caller loads indices well before it reads them.
        """
        cfg = self.cfg
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [cfg.NUM_WARPS], [0])
        offs = gl.arange(0, count, layout=layout)
        last_page = gl.where(
            self.split_start < self.split_end, (self.split_end - 1) // cfg.PAGE_SIZE, 0
        )
        page_index = gl.minimum(start_n // cfg.PAGE_SIZE + offs, last_page)
        # The page table is read-only (tl.const), so the compiler may move an
        # ordinary load of it down past the page copies issued after it, into
        # the block that first reads it; reading it would then wait for those
        # copies. A volatile load stays where it is issued. Volatile also makes
        # the load skip the CU and L2 caches (sc0 sc1); loading two iterations
        # ahead hides the extra latency.
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index,
            volatile=True,
        )

    @gluon.jit
    def select_page(self, pages, index: gl.constexpr):
        """Entry index of load_page_indices' result, the same in every lane.

        Every lane loaded the same entries; masking the others to 0 and
        max-reducing selects the entry. A block without visible keys maps to
        page 0.
        """
        offs = gl.arange(0, pages.shape[0], layout=pages.type.layout)
        page = max(gl.where(offs == index, pages, 0), axis=0)
        return gl.where(self.cache_len > 0, page, 0)

    @gluon.jit
    def page_tile_offsets(self):
        """Element offsets of a [BLOCK_N, HEAD_DIM] tile within a page, before
        the KV head's offset."""
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        return offs_n[:, None] * (cfg.NUM_KV_HEADS * cfg.HEAD_DIM) + offs_d[None, :]

    @gluon.jit
    def load_page_to_shared(self, cache_ptr, physical_page, tile_offsets, smem):
        """Copy this KV head's tile of one page into LDS; the caller commits."""
        cfg = self.cfg
        # One page holds BLOCK_N tokens. A descriptor based at the page keeps
        # the 64-bit page address in SGPRs, so each lane needs only its int32
        # offset within the page, which stays the same for every page.
        page_elems: gl.constexpr = cfg.PAGE_SIZE * cfg.NUM_KV_HEADS * cfg.HEAD_DIM
        elem_bytes: gl.constexpr = cache_ptr.dtype.element_ty.primitive_bitwidth // 8
        page = buffer.make_buffer_descriptor(
            cache_ptr + physical_page.to(gl.int64) * page_elems,
            page_elems * elem_bytes,
        )
        buffer.load_to_shared(
            smem, page, tile_offsets, soffset=self.kv_head * cfg.HEAD_DIM
        )

    @gluon.jit
    def issue_load_k(self, physical_page, k_smem):
        self.load_page_to_shared(
            self.k_cache_ptr, physical_page, self.page_tile_offsets(), k_smem
        )
        async_copy.commit_group()

    @gluon.jit
    def issue_load_v(self, physical_page, v_smem):
        self.load_page_to_shared(
            self.v_cache_ptr, physical_page, self.page_tile_offsets(), v_smem
        )
        async_copy.commit_group()

    @gluon.jit
    def issue_load_kv(self, physical_page, tile_offsets, k_smem, v_smem):
        """Copy one page of K and V into LDS as a single async group."""
        # v_readfirstlane moves the page into an SGPR. Left in a VGPR, the page
        # could differ across lanes for all the compiler knows, and a buffer
        # descriptor must be in SGPRs, so it would wrap each copy in a loop
        # that issues it once per distinct lane value (readfirstlane,
        # compare, exec mask).
        physical_page = wave_uniform_i32(physical_page)
        self.load_page_to_shared(self.k_cache_ptr, physical_page, tile_offsets, k_smem)
        self.load_page_to_shared(self.v_cache_ptr, physical_page, tile_offsets, v_smem)
        async_copy.commit_group()

    @gluon.jit
    def shared_load_k(self, k_smem):
        return k_smem.permute([1, 0]).load(self.cfg.k_layout)

    @gluon.jit
    def shared_load_v(self, v_smem):
        return v_smem.load(self.cfg.v_layout)

    @gluon.jit
    def compute_qk(self, q, k):
        cfg = self.cfg
        qk = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout
        )
        return cdna4.mfma(q, k, qk)

    @gluon.jit
    def apply_kv_mask(self, qk, start_n):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
        tokens = start_n + offs_n[None, :]
        mask = (tokens >= self.kv_start) & (tokens < self.split_end)
        if cfg.MAX_SEQLEN_Q > 1:
            # Each query position ends its keys, and starts its sliding window,
            # at its own offset within the block's range.
            offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
            query_pos = self.row_query_pos(offs_m)
            row_end = self.seq_len - (cfg.MAX_SEQLEN_Q - 1 - query_pos)
            mask &= tokens < row_end[:, None]
            if cfg.IS_SLIDING:
                mask &= tokens >= (row_end - (cfg.WINDOW_LEFT + 1))[:, None]
        return gl.where(mask, qk, -float("inf"))

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        row_max = max(qk, axis=1)
        row_max = gl.convert_layout(row_max, gl.SliceLayout(1, cfg.pv_layout))
        m_new = maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        if cfg.MAX_SEQLEN_Q > 1:
            # A row can meet a page without any of its keys before one with
            # them: a later query position's window starts past the block's
            # first pages, and an earlier one's causal end can precede a whole
            # split. Keep -inf - -inf from turning its state into NaN. With
            # one query position every row sees the same keys, so no row does.
            invalid = m_new == -float("inf")
            m_new_scaled = gl.where(invalid, 0.0, m_new_scaled)
        p_shift = fp8_probability_shift(m_new_scaled, cfg.IS_FP8)
        qk_shifted = qk * cfg.SM_SCALE - p_shift[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        if cfg.MAX_SEQLEN_Q > 1:
            m_diff = gl.where(invalid, 0.0, m_diff)
        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return cdna4.mfma(p, v, acc)

    @gluon.jit
    def store_split(self, acc, l_i, m_i):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        tokens = self.row_tokens(offs_m)
        q_heads = self.row_heads(offs_m)
        valid = self.row_valid(offs_m)
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        m_i = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.store_layout))
        # A row without keys in this split (an empty split, or one past the
        # row's causal end or before its window) stores a zero partial with
        # lse = -inf, which the merge ignores.
        has_kv = l_i > 0.0
        recip_l_i = 1.0 / l_i
        part_o = gl.where(has_kv[:, None], acc * recip_l_i[:, None], 0.0)
        part_lse = gl.where(
            has_kv,
            fp8_lse_shift(m_i, cfg.SM_SCALE, cfg.IS_FP8) + gl.log2(l_i),
            -float("inf"),
        )
        mid_o_offsets = (
            (tokens[:, None] * cfg.NUM_Q_HEADS + q_heads[:, None]) * cfg.NUM_KV_SPLITS
            + self.split_id
        ) * cfg.HEAD_DIM + offs_d[None, :]
        mid_lse_offsets = (
            tokens * cfg.NUM_Q_HEADS + q_heads
        ) * cfg.NUM_KV_SPLITS + self.split_id
        cdna4.buffer_store(part_o, self.mid_o_ptr, mid_o_offsets, mask=valid[:, None])
        cdna4.buffer_store(part_lse, self.mid_lse_ptr, mid_lse_offsets, mask=valid)

    @gluon.jit
    def store_output(self, acc, l_i):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        tokens = self.row_tokens(offs_m)
        q_heads = self.row_heads(offs_m)
        valid = self.row_valid(offs_m)
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        output = acc * (1.0 / l_i)[:, None]
        output = output.to(self.mid_o_ptr.dtype.element_ty)
        offsets = (tokens[:, None] * cfg.NUM_Q_HEADS + q_heads[:, None]) * cfg.HEAD_DIM
        offsets += offs_d[None, :]
        cdna4.buffer_store(output, self.mid_o_ptr, offsets, mask=valid[:, None])

    @gluon.jit
    def finalize_splits(self, out_ptr, sink_ptr, counters_ptr, has_sink: gl.constexpr):
        """Let the last arriving split merge this row block and reset its counter."""
        cfg = self.cfg
        counter_offset = (
            self.batch * (cfg.NUM_KV_HEADS * cfg.NUM_ROW_BLOCKS)
            + self.kv_head * cfg.NUM_ROW_BLOCKS
            + self.row_start // cfg.BLOCK_M
        )
        if cfg.NUM_WARPS > 1:
            # One thread performs the increment, and its release waits only
            # for its own wave's stores. So every wave runs s_waitcnt
            # vmcnt(0), which waits until its stores have reached L2 (CDNA4
            # ISA 4.4); S_BARRIER waits on no counter, so the barrier the
            # compiler places before the acq_rel atomic then orders every
            # wave's stores before the increment. The release then writes
            # back the L2 cache (BUFFER_WBL2), which makes all of those stores
            # visible to the other L2 caches.
            offs_m = gl.arange(
                0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout)
            )
            gl.inline_asm_elementwise(
                asm="s_waitcnt vmcnt(0)",
                constraints="=v,0",
                args=[offs_m],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        old = gl.atomic_add(
            counters_ptr + counter_offset,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        if old == cfg.NUM_KV_SPLITS - 1:
            offs_m = gl.arange(
                0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout)
            )
            offs_d = gl.arange(
                0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout)
            )
            tokens = self.row_tokens(offs_m)
            q_heads = self.row_heads(offs_m)
            valid = self.row_valid(offs_m)
            final_m = gl.full(
                [cfg.BLOCK_M],
                value=-float("inf"),
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.store_layout),
            )
            final_l = gl.zeros(
                [cfg.BLOCK_M],
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.store_layout),
            )
            final_acc = gl.zeros(
                [cfg.BLOCK_M, cfg.HEAD_DIM],
                dtype=gl.float32,
                layout=cfg.store_layout,
            )
            # Each split's waves stored their partials before its acq_rel
            # increment, so this split's increment orders the partial loads
            # after all of them, and the loads can be ordinary cached loads;
            # volatile loads would bypass the caches and slow the merge. Only
            # the incrementing wave acquires, but its BUFFER_INV invalidates
            # the CU's whole vector L1 cache, which all of the work-group's
            # waves share, and the L2's non-coherently cached lines (CDNA4
            # ISA 9.1.10.2, Table 52), and the barrier that
            # broadcasts the old count orders their loads after it.
            for split in range(0, cfg.NUM_KV_SPLITS):
                base = (
                    (tokens[:, None] * cfg.NUM_Q_HEADS + q_heads[:, None])
                    * cfg.NUM_KV_SPLITS
                    + split
                ) * cfg.HEAD_DIM + offs_d[None, :]
                part_o = gl.load(
                    self.mid_o_ptr + base,
                    mask=valid[:, None],
                    other=0.0,
                ).to(gl.float32)
                lse_offset = (
                    tokens * cfg.NUM_Q_HEADS + q_heads
                ) * cfg.NUM_KV_SPLITS + split
                part_lse = gl.load(
                    self.mid_lse_ptr + lse_offset,
                    mask=valid,
                    other=-float("inf"),
                ).to(gl.float32)
                next_m = maximum(final_m, part_lse)
                if cfg.MAX_SEQLEN_Q > 1:
                    # A later query position's first splits can all precede
                    # its window, leaving -inf - -inf.
                    next_m_safe = gl.where(next_m == -float("inf"), 0.0, next_m)
                    alpha = gl.exp2(final_m - next_m_safe)
                    beta = gl.exp2(part_lse - next_m_safe)
                else:
                    alpha = gl.exp2(final_m - next_m)
                    beta = gl.exp2(part_lse - next_m)
                final_acc = final_acc * alpha[:, None] + part_o * beta[:, None]
                final_l = final_l * alpha + beta
                final_m = next_m
            if has_sink:
                sink_log2 = (
                    gl.load(sink_ptr + q_heads, mask=valid, other=0.0).to(gl.float32)
                    * _INV_LN2
                )
                next_m = maximum(final_m, sink_log2)
                alpha = gl.exp2(final_m - next_m)
                beta = gl.exp2(sink_log2 - next_m)
                final_acc *= alpha[:, None]
                final_l = final_l * alpha + beta
            output = (final_acc / final_l[:, None]).to(out_ptr.dtype.element_ty)
            out_offsets = (
                tokens[:, None] * cfg.NUM_Q_HEADS + q_heads[:, None]
            ) * cfg.HEAD_DIM + offs_d[None, :]
            cdna4.buffer_store(output, out_ptr, out_offsets, mask=valid[:, None])
            gl.store(counters_ptr + counter_offset, 0)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _stream_split_single_buffer(program, q, m_i, l_i, acc, k_smem, v_smem):
    """Load, wait for and consume one K/V page at a time; enough when other
    waves of the grid cover the load latency."""
    cfg = program.cfg
    physical_page = program.load_page(program.split_start)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            program.issue_load_k(physical_page, k_smem)
            program.issue_load_v(physical_page, v_smem)
            physical_page = program.load_page(start_n + cfg.BLOCK_N)

        with gl.amd.warp_pipeline_stage("qk_softmax", priority=0):
            async_copy.wait_group(1)
            k = program.shared_load_k(k_smem)
            qk = program.compute_qk(q, k)
            qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        with gl.amd.warp_pipeline_stage("pv", priority=1):
            async_copy.wait_group(0)
            v = program.shared_load_v(v_smem)
            acc = program.compute_pv(p, v, acc)
    return m_i, l_i, acc


@gluon.jit
def _stream_split_pipelined(
    program, q, m_i, l_i, acc, k_cache_ptr, v_cache_ptr, NUM_STAGES: gl.constexpr
):
    cfg = program.cfg
    # Software-pipelined K/V page loads, NUM_STAGES deep: page p + NUM_STAGES - 1
    # is copied into the LDS buffer page p - 1 freed while page p is consumed,
    # so a wave keeps NUM_STAGES - 1 pages of loads outstanding and a long split
    # is not bound by one memory round trip per page. Each loop iteration
    # consumes NUM_STAGES pages, one per buffer. Page indices are loaded two
    # iterations before use: vmcnt retires in issue order, so reading an index
    # loaded after the prefetched copies would wait for all of them.
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty,
        [NUM_STAGES, cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.k_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty,
        [NUM_STAGES, cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.v_smem_layout,
    )

    iteration_tokens: gl.constexpr = NUM_STAGES * cfg.BLOCK_N
    # A split with fewer pages than buffers would spend more on the pipeline
    # prologue than it hides.
    if program.split_end - program.split_start < iteration_tokens:
        return _stream_split_single_buffer(
            program, q, m_i, l_i, acc, k_smem.index(0), v_smem.index(0)
        )
    # Prologue: prefetch the first NUM_STAGES - 1 pages.
    pages = program.load_page_indices(program.split_start, NUM_STAGES)
    next_pages = program.load_page_indices(
        program.split_start + iteration_tokens, NUM_STAGES
    )
    tile_offsets = program.page_tile_offsets()
    for stage in gl.static_range(NUM_STAGES - 1):
        program.issue_load_kv(
            program.select_page(pages, stage),
            tile_offsets,
            k_smem.index(stage),
            v_smem.index(stage),
        )

    for iteration_start in range(
        program.split_start, program.split_end, iteration_tokens
    ):
        later_pages = program.load_page_indices(
            iteration_start + 2 * iteration_tokens, NUM_STAGES
        )
        for stage in gl.static_range(NUM_STAGES):
            # Prefetch the page NUM_STAGES - 1 ahead into the buffer the
            # previous page freed.
            if stage == 0:
                prefetch_page = program.select_page(pages, NUM_STAGES - 1)
            else:
                prefetch_page = program.select_page(next_pages, stage - 1)
            program.issue_load_kv(
                prefetch_page,
                tile_offsets,
                k_smem.index((stage + NUM_STAGES - 1) % NUM_STAGES),
                v_smem.index((stage + NUM_STAGES - 1) % NUM_STAGES),
            )

            start_n = iteration_start + stage * cfg.BLOCK_N
            # Wait for this page; the NUM_STAGES - 1 prefetched pages stay
            # outstanding (s_waitcnt vmcnt(N), N = their copies).
            async_copy.wait_group(NUM_STAGES - 1)
            k = program.shared_load_k(k_smem.index(stage))
            qk = program.compute_qk(q, k)
            # Full attention starts at token 0; only pages reaching past the
            # split's end or the block's first causal end need a mask.
            if start_n + cfg.BLOCK_N > program.mask_start:
                qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)
            v = program.shared_load_v(v_smem.index(stage))
            acc = program.compute_pv(p, v, acc)
        pages = next_pages
        next_pages = later_pages

    # The loop exits with NUM_STAGES - 1 prefetched pages still copying into
    # LDS, and the layout conversions after it reuse that LDS. Wait for this
    # wave's copies: buffer loads to LDS count in vmcnt (CDNA4 ISA 4.4, 9.1.9).
    # The barrier the compiler emits after the wait then covers the other waves.
    async_copy.wait_group(0)
    return m_i, l_i, acc


@gluon.jit
def gluon_mha_decode_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
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
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
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
        IS_SLIDING,
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
    )
    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_state(q_ptr, False)
    if NUM_STAGES == 1:
        k_smem = gl.allocate_shared_memory(
            k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
        )
        v_smem = gl.allocate_shared_memory(
            v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
        )
        m_i, l_i, acc = _stream_split_single_buffer(
            program, q, m_i, l_i, acc, k_smem, v_smem
        )
    else:
        m_i, l_i, acc = _stream_split_pipelined(
            program, q, m_i, l_i, acc, k_cache_ptr, v_cache_ptr, NUM_STAGES
        )
    program.store_split(acc, l_i, m_i)


@gluon.jit
def gluon_mha_decode_sliding_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    out_ptr,
    sink_ptr,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    HAS_SINK: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        1,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        IS_SLIDING,
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
        out_ptr,
        out_ptr,
    )
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_state(sink_ptr, HAS_SINK)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        physical_page = program.load_page(start_n)
        program.issue_load_k(physical_page, k_smem)
        program.issue_load_v(physical_page, v_smem)
        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = program.apply_kv_mask(qk, start_n)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    l_i = program.apply_sinks(l_i, m_i, sink_log2, HAS_SINK)
    program.store_output(acc, l_i)


@gluon.jit
def _mha_decode_sliding_split_atomic(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    sink_ptr,
    counters_ptr,
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
    HAS_SINK: gl.constexpr,
    IS_FP8: gl.constexpr,
):
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
    )
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc, _ = program.init_state(sink_ptr, False)
    physical_page = program.load_page(program.split_start)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            program.issue_load_k(physical_page, k_smem)
            program.issue_load_v(physical_page, v_smem)
            physical_page = program.load_page(start_n + cfg.BLOCK_N)

        with gl.amd.warp_pipeline_stage("qk_softmax", priority=0):
            async_copy.wait_group(1)
            k = program.shared_load_k(k_smem)
            qk = program.compute_qk(q, k)
            qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        with gl.amd.warp_pipeline_stage("pv", priority=1):
            async_copy.wait_group(0)
            v = program.shared_load_v(v_smem)
            acc = program.compute_pv(p, v, acc)

    program.store_split(acc, l_i, m_i)
    program.finalize_splits(out_ptr, sink_ptr, counters_ptr, HAS_SINK)


@gluon.jit
def gluon_mha_decode_reduce_gfx950(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    sink_ptr,
    k_cache_ptr,  # unused for loads; only its dtype feeds the shared config
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HAS_SINK: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    # One wave merges one (query token, head); the attention row tiling does
    # not apply, so the config takes a single-wave tile.
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        16,  # BLOCK_M
        BLOCK_N,
        1,  # NUM_WARPS
        False,  # IS_SLIDING
        -1,  # WINDOW_LEFT
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(1, 1, 1),
    )
    q_index = gl.program_id(0)
    q_head = gl.program_id(1)

    # SPLIT_TILE pads NUM_KV_SPLITS up to a power of 2.
    SPLIT_TILE: gl.constexpr = 1 << (NUM_KV_SPLITS - 1).bit_length()
    offs_s = gl.arange(0, SPLIT_TILE, layout=gl.SliceLayout(1, cfg.reduce_layout))
    offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.reduce_layout))
    # Every split stores its partial, with lse = -inf when it has no tokens, so
    # the loads need not wait on cache_seqlens; only the padding tail is masked.
    split_valid = offs_s < cfg.NUM_KV_SPLITS
    base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.NUM_KV_SPLITS + offs_s
    part_lse = gl.load(mid_lse_ptr + base, mask=split_valid, other=-float("inf"))
    o_off = base[:, None] * cfg.HEAD_DIM + offs_d[None, :]
    part_o = cdna4.buffer_load(mid_o_ptr, o_off, mask=split_valid[:, None], other=0.0)

    # Global softmax max over all splits (folding in the sink).
    m_i = max(part_lse, axis=0)
    if HAS_SINK:
        sink = gl.load(sink_ptr + q_head).to(gl.float32) * _INV_LN2
        m_i = maximum(m_i, sink)
    # Weighted sum of the split partials, normalized by the total softmax mass.
    beta = gl.exp2(part_lse - m_i)
    l_i = gl.sum(beta, axis=0)
    if HAS_SINK:
        l_i = l_i + gl.exp2(sink - m_i)
    acc = gl.sum(part_o * beta[:, None], axis=0)

    out_base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.HEAD_DIM
    output = acc * (1.0 / l_i)
    output = output.to(out_ptr.dtype.element_ty)
    cdna4.buffer_store(output, out_ptr, out_base + offs_d)


class LaunchConfig(NamedTuple):
    num_q_heads: int
    num_kv_heads: int
    num_requests: int
    num_row_blocks: int
    head_dim: int
    page_size: int
    num_kv_splits: int
    num_stages: int
    block_m: int
    block_n: int
    num_warps: int
    sm_scale: float
    is_sliding: bool
    window_left: int


def _sliding_split_counters(
    q: torch.Tensor,
    num_requests: int,
    head_blocks: int,
) -> torch.Tensor:
    """Return stream-local counters reset by each fused split reduction.

    Decode graph capture performs four warmup forwards on its capture stream.
    Reusing that warmed allocation keeps the otherwise redundant ``zeros``
    launch out of every replay; the last-arriving split resets every counter.
    Batch size is part of the key, so independently captured graph shapes do
    not alias one another.
    """
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
    key = (device_index, stream_id, num_requests, head_blocks)
    counters = _SLIDING_SPLIT_COUNTERS.get(key)
    if counters is None:
        counters = torch.zeros(
            (num_requests, head_blocks),
            device=q.device,
            dtype=torch.int32,
        )
        _SLIDING_SPLIT_COUNTERS[key] = counters
    return counters


def _full_decode_stages(
    num_kv_splits: int,
    page_size: int,
    head_dim: int,
    element_size: int,
    num_warps: int,
) -> int:
    """Pipeline the K/V page loads only when select_kv_splits returned its
    maximum, _FULL_DECODE_MAX_SPLITS: even the most splits leave the grid below
    its occupancy target, so each work-group streams a long split.

    With fewer splits the grid already holds enough waves to cover the load
    latency, and the pipeline's LDS buffers would lower occupancy.
    """
    if num_kv_splits < _FULL_DECODE_MAX_SPLITS:
        return 1
    # The waves of a work-group share the copies of one KV head's K and V page
    # tiles, 16 bytes per lane per buffer_load_dwordx4.
    copies_per_page = 2 * page_size * head_dim * element_size // (64 * 16 * num_warps)
    if (_FULL_DECODE_STAGES - 1) * copies_per_page > _GFX950_MAX_VMCNT:
        return 1
    return _FULL_DECODE_STAGES


def _row_tile(max_seqlen_q: int, group_size: int) -> tuple[int, int]:
    """Return (BLOCK_M, NUM_WARPS) for a request's rows of one KV head.

    Those max_seqlen_q * group_size rows share the KV head's keys, so one
    work-group takes up to 64 of them, one 16-row MFMA tile per wave, and
    every K/V page it loads into LDS serves them all.
    """
    rows = max_seqlen_q * group_size
    block_m = builtins.max(16, min(_DECODE_MAX_BLOCK_M, triton.next_power_of_2(rows)))
    return block_m, block_m // 16


def get_config(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int,
    window_left: int,
    softmax_scale: float | None,
) -> LaunchConfig:
    head_dim = q.shape[2]
    page_size = k_cache.shape[1]
    block_n = 64
    num_kv_heads = k_cache.shape[2]
    group_size = q.shape[1] // num_kv_heads
    num_requests = q.shape[0] // max_seqlen_q
    is_sliding = window_left >= 0
    window_left = window_left if is_sliding else -1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    effective_seqlen_k = min(max_seqlen_k, window_left) if is_sliding else max_seqlen_k
    num_pages = (effective_seqlen_k + page_size - 1) // page_size
    short_window = is_sliding and num_pages <= _SHORT_WINDOW_PAGES
    if short_window:
        # A short window (or a shorter context) gives a four-wave work-group
        # too few pages to share, so the rows take one-wave work-groups of 16
        # and more of them load the pages side by side.
        block_m, num_warps = 16, 1
    else:
        block_m, num_warps = _row_tile(max_seqlen_q, group_size)
    num_row_blocks = math.ceil(max_seqlen_q * group_size / block_m)
    if short_window:
        # Each split work-group merges its partial through an atomic counter,
        # which costs about as much as streaming two pages, so a split covers
        # at least two pages and a window of 3 pages or fewer runs unsplit.
        # Past 4 requests the grid is busier, so divide the splits by the
        # request's row blocks, keeping its work-group count near the
        # 4-request one.
        num_kv_splits = num_pages // 2
        if num_requests > 4:
            num_kv_splits //= num_row_blocks
        num_kv_splits = builtins.max(1, min(8, num_kv_splits))
    elif is_sliding:
        target_splits = 8 if num_requests <= 4 else 4
        num_kv_splits = builtins.max(1, min(target_splits, num_pages))
    else:
        num_kv_splits = select_kv_splits(
            base_ctas=num_requests * num_kv_heads * num_row_blocks,
            num_pages=num_pages,
            sm_count=_GFX950_SM_COUNT,
            max_splits=_FULL_DECODE_MAX_SPLITS,
        )
    return LaunchConfig(
        num_q_heads=q.shape[1],
        num_kv_heads=num_kv_heads,
        num_requests=num_requests,
        num_row_blocks=num_row_blocks,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
        num_stages=_full_decode_stages(
            num_kv_splits, page_size, head_dim, k_cache.element_size(), num_warps
        ),
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        sm_scale=softmax_scale * _INV_LN2_VALUE,
        is_sliding=is_sliding,
        window_left=window_left,
    )


def launch_gluon_mha_decode_gfx950(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    reject_block_scales(
        "gluon_mha_decode_gfx950", q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
    )
    total_q = q.shape[0]

    has_sink = sinks is not None
    sink_arg = sinks if sinks is not None else q
    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        window_left=window_left,
        softmax_scale=softmax_scale,
    )

    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    output = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    # One work-group per request, KV head, row block and split.
    head_blocks = config.num_kv_heads * config.num_row_blocks

    if config.is_sliding and config.num_kv_splits == 1:
        grid = (config.num_requests, head_blocks, 1)
        gluon_mha_decode_sliding_gfx950[grid](
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            output,
            sink_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            config.sm_scale,
            page_table.stride(0),
            config.page_size,
            max_seqlen_q,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            config.num_warps,
            config.is_sliding,
            config.window_left,
            has_sink,
            is_fp8,
            num_warps=config.num_warps,
        )
    elif config.is_sliding:
        mid_o = torch.empty(
            (total_q, config.num_q_heads, config.num_kv_splits, config.head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        mid_lse = torch.empty(
            (total_q, config.num_q_heads, config.num_kv_splits),
            device=q.device,
            dtype=torch.float32,
        )
        counters = _sliding_split_counters(q, config.num_requests, head_blocks)
        grid = (config.num_requests, head_blocks, config.num_kv_splits)
        _mha_decode_sliding_split_atomic[grid](
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            mid_o,
            mid_lse,
            output,
            sink_arg,
            counters,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            config.sm_scale,
            page_table.stride(0),
            config.page_size,
            config.num_kv_splits,
            max_seqlen_q,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            config.num_warps,
            config.window_left,
            has_sink,
            is_fp8,
            num_warps=config.num_warps,
        )
    else:
        # Always use split-k for full attention
        mid_o = torch.empty(
            (total_q, config.num_q_heads, config.num_kv_splits, config.head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        mid_lse = torch.empty(
            (total_q, config.num_q_heads, config.num_kv_splits),
            device=q.device,
            dtype=torch.float32,
        )

        grid = (config.num_requests, head_blocks, config.num_kv_splits)
        gluon_mha_decode_gfx950[grid](
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            mid_o,
            mid_lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            config.sm_scale,
            page_table.stride(0),
            config.page_size,
            config.num_kv_splits,
            max_seqlen_q,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            config.num_warps,
            config.is_sliding,
            config.window_left,
            is_fp8,
            config.num_stages,
            num_warps=config.num_warps,
            **sched_compile_options(),
        )

        # Sink is a single global softmax entry, so split-k must merge it once in
        # reduce. Adding it in each split would count the sink once per split.
        grid = (total_q, config.num_q_heads)
        gluon_mha_decode_reduce_gfx950[grid](
            mid_o,
            mid_lse,
            output,
            sink_arg,
            k_cache,
            config.sm_scale,
            page_table.stride(0),
            config.num_kv_splits,
            max_seqlen_q,
            config.page_size,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_n,
            has_sink,
            is_fp8,
            num_warps=1,
        )
    return output
