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

import numpy as np
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
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import _remap_xcd

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

        Pages past the split repeat its last page (an L2 hit): a masked copy
        becomes a branch after which the compiler waits for every outstanding
        load (s_waitcnt vmcnt(0)). An empty split reads the request's first
        entry; select_page maps a block without visible keys to page 0. vmcnt
        retires in issue order, so the caller loads indices well before reading
        them.
        """
        cfg = self.cfg
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [cfg.NUM_WARPS], [0])
        offs = gl.arange(0, count, layout=layout)
        last_page = gl.where(
            self.split_start < self.split_end, (self.split_end - 1) // cfg.PAGE_SIZE, 0
        )
        page_index = gl.minimum(start_n // cfg.PAGE_SIZE + offs, last_page)
        # One entry per lane keeps this a vector load, retired in issue order
        # with the copies by the loop's vmcnt waits. Uniform addresses into the
        # tl.const table would compile to s_load, which can only be waited on
        # with lgkmcnt(0) (CDNA4 ISA 4.4, 8.3), placed just before each copy:
        # 1-3% slower on long pipelined splits (MI355X). Because the table is
        # tl.const, the compiler may sink a plain load below the copies;
        # volatile keeps it where it is issued, and its cache bypass (sc0 sc1)
        # is hidden by loading two iterations ahead.
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index,
            volatile=True,
        )

    @gluon.jit
    def select_page(self, pages, index: gl.constexpr):
        """Entry index of load_page_indices' result, the same in every lane.

        Each lane holds entry lane % count; masking the others to 0 and
        max-reducing across lanes leaves the entry in every lane, still in a
        VGPR. A block without visible keys maps to page 0.
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
        # load_page_indices leaves the page in a VGPR. A buffer descriptor
        # must be in SGPRs; without v_readfirstlane the compiler would wrap each
        # copy in a loop over distinct lane values.
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
    # Page p + NUM_STAGES - 1 is copied into the buffer page p - 1 freed while
    # page p is consumed, keeping NUM_STAGES - 1 pages in flight. Each iteration
    # consumes NUM_STAGES pages, one per buffer, and loads the page indices for
    # two iterations ahead.
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


# ===-----------------------------------------------------------------------===#
# Balanced Split Schedule
# ===-----------------------------------------------------------------------===#
#
# The split-grid kernel above divides every request's pages into the same
# number of splits, chosen on the host from the batch size and max_seqlen_k.
# Under a captured graph that is the padded batch and the model's context
# limit, so a batch of a few long requests among short or padding ones gives
# each long request as few splits as the padding ones and leaves CUs idle.
#
# The balanced path reads a split schedule that
# gluon_mha_decode_schedule_gfx950 computes from cache_seqlens once per step
# for every layer of the same shape. With row_blocks (KV head, row block)
# pairs per request, every split covers split_pages pages, except a request's
# last: the fewest pages per split whose work items fit in
# _BALANCED_TARGET_PROGRAMS (two work-groups per CU), at least
# _BALANCED_MIN_SPLIT_PAGES and at most _BALANCED_MAX_SPLITS splits per
# request. A padding request takes one split. A slot is one split of one
# request; its row_blocks work items are adjacent, and the slots of all
# requests follow in request order. The schedule, int32 [num_requests + 1, 2],
# holds:
#
#   row 0:     (split_pages, number of slots)
#   row 1 + r: (first slot, split count) of request r
#
# The decode kernel bounds each split by the live cache_seqlens, and a
# request's last split runs to its end, so a schedule computed from other
# lengths (a drafter's steps since) still covers every key; it only balances
# the work less evenly.
#
# prefers_gluon_mha_decode_balanced_gfx950() decides on the host, from the
# requests' page counts, when the balanced path beats the split grid; callers
# use the split grid otherwise.

_BALANCED_TARGET_PROGRAMS = 2 * _GFX950_SM_COUNT
_BALANCED_MIN_SPLIT_PAGES = 2
_BALANCED_MAX_SPLITS = 128
# Two LDS buffers: with one round of work-groups, deeper pipelines did not
# help while address translation dominated the load latency (MI350X).
_BALANCED_STAGES = 2
# Splits of at least this many pages are rounded to whole pipeline
# iterations; shorter ones would lose more to the rounding than they save.
_BALANCED_ALIGN_MIN_PAGES = 16
# The balanced path is considered only when the longest request spans at
# least this many pages (32k tokens at page size 64): below it, it never beat
# the split grid (measured on MI350X).
_BALANCED_MIN_PAGES = 512
# What a balanced work item's schedule lookup costs, in pages streamed, when
# weighing it against the split grid (measured on MI350X).
_BALANCED_ITEM_PAGES = 3
# Schedule rows one round of vector loads covers: a work-group finds its
# request with three loads (first slots, split counts, lengths) for batches of
# up to this many requests.
_SCHEDULE_CHUNK = gl.constexpr(64)


@gluon.jit(do_not_specialize=("num_requests",))
def gluon_mha_decode_schedule_gfx950(
    cache_seqlens_ptr,
    schedule_ptr,
    num_requests,
    ROW_BLOCKS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    TARGET_PROGRAMS: gl.constexpr,
    MIN_SPLIT_PAGES: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    PAGE_ALIGN: gl.constexpr,
    ALIGN_MIN_PAGES: gl.constexpr,
):
    """Write the balanced split schedule of ``num_requests`` requests; one
    wave."""
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offs = gl.arange(0, _SCHEDULE_CHUNK, layout=layout)
    total = num_requests * 0
    longest = total
    for start in range(0, num_requests, _SCHEDULE_CHUNK):
        index = start + offs
        seq_lens = gl.load(
            cache_seqlens_ptr + index, mask=index < num_requests, other=0
        )
        pages = cdiv(gl.maximum(seq_lens, 0), PAGE_SIZE)
        total += gl.sum(pages, axis=0)
        longest = gl.maximum(longest, gl.max(pages, axis=0))
    # The fewest pages per split whose items fit in TARGET_PROGRAMS: binary
    # search between the even share of all pages and one split per request.
    # The first chunk's pages stay in registers.
    first_valid = offs < num_requests
    first_pages = cdiv(
        gl.maximum(gl.load(cache_seqlens_ptr + offs, mask=first_valid, other=0), 0),
        PAGE_SIZE,
    )
    lo = gl.maximum(cdiv(total * ROW_BLOCKS, TARGET_PROGRAMS), MIN_SPLIT_PAGES)
    lo = gl.maximum(lo, cdiv(longest, MAX_SPLITS))
    hi = gl.maximum(lo, longest)
    for _ in range(16):
        mid = (lo + hi) // 2
        count = gl.sum(
            gl.where(first_valid, gl.maximum(cdiv(first_pages, mid), 1), 0), axis=0
        )
        for start in range(_SCHEDULE_CHUNK, num_requests, _SCHEDULE_CHUNK):
            index = start + offs
            valid = index < num_requests
            seq_lens = gl.load(cache_seqlens_ptr + index, mask=valid, other=0)
            pages = cdiv(gl.maximum(seq_lens, 0), PAGE_SIZE)
            count += gl.sum(gl.where(valid, gl.maximum(cdiv(pages, mid), 1), 0), axis=0)
        fits = count * ROW_BLOCKS <= TARGET_PROGRAMS
        hi = gl.where(fits, mid, hi)
        lo = gl.where(fits, lo, mid + 1)
        lo = gl.minimum(lo, hi)
    split_pages = hi
    # The pipelined loop consumes PAGE_ALIGN pages per iteration, so a longer
    # split takes a whole number of iterations instead of computing on pages
    # past its end.
    split_pages = gl.where(
        (split_pages > PAGE_ALIGN) & (split_pages >= ALIGN_MIN_PAGES),
        cdiv(split_pages, PAGE_ALIGN) * PAGE_ALIGN,
        split_pages,
    )
    first_slot = num_requests * 0
    for start in range(0, num_requests, _SCHEDULE_CHUNK):
        index = start + offs
        valid = index < num_requests
        seq_lens = gl.load(cache_seqlens_ptr + index, mask=valid, other=0)
        pages = cdiv(gl.maximum(seq_lens, 0), PAGE_SIZE)
        splits = gl.where(valid, gl.maximum(cdiv(pages, split_pages), 1), 0)
        ends = first_slot + gl.cumsum(splits, axis=0)
        row = schedule_ptr + 2 * (1 + index)
        gl.store(row, ends - splits, mask=valid)
        gl.store(row + 1, splits, mask=valid)
        first_slot += gl.sum(splits, axis=0)
    gl.store(schedule_ptr, split_pages)
    gl.store(schedule_ptr + 1, first_slot)


@gluon.jit
def _find_slot(
    schedule_ptr, cache_seqlens_ptr, num_requests, slot, layout: gl.constexpr
):
    """Return the request holding ``slot`` with its length, first slot and
    split count, read from the schedule (its last request when ``slot`` lies
    past the last slot)."""
    offs = gl.arange(0, _SCHEDULE_CHUNK, layout=layout)
    # Requests' first slots increase, so the request holding ``slot`` is the
    # last one whose first slot is at most ``slot``. The first chunk of
    # _SCHEDULE_CHUNK requests is read without a loop, so its loads issue
    # with the kernel's first ones; larger batches read the rest a chunk at
    # a time. Masked buffer loads return 0 without a branch around each load.
    valid = offs < num_requests
    row = 2 * (1 + offs)
    firsts = cdna4.buffer_load(schedule_ptr, row, mask=valid, other=0)
    counts = cdna4.buffer_load(schedule_ptr, row + 1, mask=valid, other=0)
    seq_lens = cdna4.buffer_load(cache_seqlens_ptr, offs, mask=valid, other=0)
    before = gl.sum(((firsts <= slot) & valid).to(gl.int32), axis=0)
    lane = offs == before - 1
    request = before - 1
    seq_len = gl.sum(gl.where(lane, seq_lens, 0), axis=0)
    first_slot = gl.sum(gl.where(lane, firsts, 0), axis=0)
    splits = gl.sum(gl.where(lane, counts, 0), axis=0)
    for start in range(_SCHEDULE_CHUNK, num_requests, _SCHEDULE_CHUNK):
        index = start + offs
        valid_c = index < num_requests
        row_c = 2 * (1 + index)
        firsts_c = cdna4.buffer_load(schedule_ptr, row_c, mask=valid_c, other=0)
        counts_c = cdna4.buffer_load(schedule_ptr, row_c + 1, mask=valid_c, other=0)
        lens_c = cdna4.buffer_load(cache_seqlens_ptr, index, mask=valid_c, other=0)
        before_c = gl.sum(((firsts_c <= slot) & valid_c).to(gl.int32), axis=0)
        lane_c = offs == before_c - 1
        found = before_c > 0
        request = gl.where(found, start + before_c - 1, request)
        seq_len = gl.where(found, gl.sum(gl.where(lane_c, lens_c, 0), axis=0), seq_len)
        first_slot = gl.where(
            found, gl.sum(gl.where(lane_c, firsts_c, 0), axis=0), first_slot
        )
        splits = gl.where(found, gl.sum(gl.where(lane_c, counts_c, 0), axis=0), splits)
    return request, seq_len, first_slot, splits


@gluon.jit
def _locate_balanced(
    cfg: gl.constexpr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    request,
    seq_len,
    head_block,
    split_id,
    splits,
    split_pages,
):
    """Bound the keys of one request's row block and balanced split.

    As AttentionProgram.create, except that a split covers split_pages pages
    and the request's last split runs to its end, so lengths that grew since
    the schedule was computed are still covered.
    """
    kv_head = head_block // cfg.NUM_ROW_BLOCKS
    row_start = (head_block - kv_head * cfg.NUM_ROW_BLOCKS) * cfg.BLOCK_M
    if cfg.MAX_SEQLEN_Q == 1:
        first_end = maximum(seq_len, 0)
        cache_len = first_end
    else:
        last_row = min(row_start + cfg.BLOCK_M, cfg.ROWS) - 1
        first_pos = row_start // cfg.GROUP_SIZE
        first_end = maximum(seq_len - (cfg.MAX_SEQLEN_Q - 1 - first_pos), 0)
        last_pos = last_row // cfg.GROUP_SIZE
        cache_len = maximum(seq_len - (cfg.MAX_SEQLEN_Q - 1 - last_pos), 0)
    end_page = cdiv(cache_len, cfg.PAGE_SIZE)
    split_start_page = split_id * split_pages
    split_end_page = gl.where(
        split_id == splits - 1,
        end_page,
        min(split_start_page + split_pages, end_page),
    )
    split_start = split_start_page * cfg.PAGE_SIZE
    split_end = min(split_end_page * cfg.PAGE_SIZE, cache_len)
    mask_start = min(split_end, first_end)
    return AttentionProgram(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        request,
        kv_head,
        row_start,
        split_id,
        seq_len,
        cache_len,
        cache_len - cache_len,
        split_start,
        split_end,
        mask_start,
    )


@gluon.jit
def _store_balanced(program, acc, l_i, m_i, block_item, splits):
    """Store the partial of a balanced work item. A row block's items hold
    BLOCK_M x splits partials, row major, from block_item on, so the reduce
    reads each row's partials from one contiguous range."""
    cfg = program.cfg
    offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
    offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
    valid = program.row_valid(offs_m)
    acc = gl.convert_layout(acc, cfg.store_layout)
    l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
    m_i = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.store_layout))
    # A row without keys in this split stores a zero partial with lse = -inf,
    # which the merge ignores.
    has_kv = l_i > 0.0
    part_o = gl.where(has_kv[:, None], acc * (1.0 / l_i)[:, None], 0.0)
    part_lse = gl.where(
        has_kv,
        fp8_lse_shift(m_i, cfg.SM_SCALE, cfg.IS_FP8) + gl.log2(l_i),
        -float("inf"),
    )
    rows = block_item * cfg.BLOCK_M + offs_m * splits + program.split_id
    cdna4.buffer_store(
        part_o,
        program.mid_o_ptr,
        rows[:, None] * cfg.HEAD_DIM + offs_d[None, :],
        mask=valid[:, None],
    )
    cdna4.buffer_store(part_lse, program.mid_lse_ptr, rows, mask=valid)


@gluon.jit
def _stream_balanced(
    program,
    q,
    m_i,
    l_i,
    acc,
    k_cache_ptr,
    v_cache_ptr,
    pipeline,
    NUM_STAGES: gl.constexpr,
):
    """Stream a balanced split, software-pipelined NUM_STAGES deep as
    _stream_split_pipelined when ``pipeline`` is set and the split spans at
    least one iteration, else single-buffered in the first LDS buffer.

    A deliberate copy of _stream_split_pipelined with the runtime
    ``pipeline`` choice, so the split-grid kernel stays byte-identical."""
    cfg = program.cfg
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
    short = program.split_end - program.split_start < iteration_tokens
    if short | (pipeline == 0):
        return _stream_split_single_buffer(
            program, q, m_i, l_i, acc, k_smem.index(0), v_smem.index(0)
        )
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
            async_copy.wait_group(NUM_STAGES - 1)
            k = program.shared_load_k(k_smem.index(stage))
            qk = program.compute_qk(q, k)
            if start_n + cfg.BLOCK_N > program.mask_start:
                qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)
            v = program.shared_load_v(v_smem.index(stage))
            acc = program.compute_pv(p, v, acc)
        pages = next_pages
        next_pages = later_pages
    # As _stream_split_pipelined: wait for this wave's prefetched copies
    # before the layout conversions reuse their LDS.
    async_copy.wait_group(0)
    return m_i, l_i, acc


@gluon.jit(do_not_specialize=("num_requests",))
def gluon_mha_decode_balanced_gfx950(
    # The compiler (LLVM AMDGPU kernel-argument preloading) preloads the
    # first 14 dwords (56 bytes) of arguments into SGPRs; the rest take a
    # scalar load. The ones the first loads need come first.
    q_ptr,
    cache_seqlens_ptr,
    num_requests,
    schedule_ptr: tl.const,
    page_table_ptr: tl.const,
    k_cache_ptr,
    v_cache_ptr,
    mid_o_ptr,
    mid_lse_ptr,
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
    IS_FP8: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    TARGET_PROGRAMS: gl.constexpr,
):
    """Full-attention split of one work item of a balanced schedule; the grid
    has one program per possible item, and the ones past the last item
    exit."""
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        1,  # NUM_KV_SPLITS: the schedule sets each request's splits
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        False,  # IS_SLIDING
        -1,  # WINDOW_LEFT
        IS_FP8,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
    )
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    row_blocks: gl.constexpr = NUM_KV_HEADS * cfg.NUM_ROW_BLOCKS
    split_pages = gl.load(schedule_ptr)
    num_slots = gl.load(schedule_ptr + 1)
    num_items = num_slots * row_blocks
    # Work-groups are dispatched round robin over the XCDs (8, as every
    # gfx950 kernel here assumes). Give each XCD a contiguous run of the
    # items, as the FP16 GEMM's _remap_xcd does, so the row blocks of a slot,
    # which read the same pages, share one XCD's L2 and address translations.
    pid = gl.program_id(0)
    item = gl.where(pid < num_items, _remap_xcd(pid, num_items, 8), num_items)
    slot = item // row_blocks
    head_block = item - slot * row_blocks
    request, seq_len, first_slot, splits = _find_slot(
        schedule_ptr, cache_seqlens_ptr, num_requests, slot, layout
    )
    split_id = slot - first_slot
    program = _locate_balanced(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        request,
        seq_len,
        head_block,
        split_id,
        splits,
        split_pages,
    )
    if NUM_WARPS == 1:
        # A one-wave work-group computes every page itself, and the
        # pipeline's extra per-page instructions cost more than the latency
        # they hide once two work-groups per CU stream the KV. Pipeline only
        # while the items leave CUs with at most one work-group, or for
        # splits of 32 pages or more, which spend longest waiting on loads.
        long_split = program.split_end - program.split_start >= 32 * PAGE_SIZE
        pipeline = (num_items * 2 <= TARGET_PROGRAMS) | long_split
    else:
        # Every multi-wave work-group pipelines.
        pipeline = True
    if item < num_items:
        q = program.load_q()
        m_i, l_i, acc, sink_log2 = program.init_state(q_ptr, False)
        m_i, l_i, acc = _stream_balanced(
            program, q, m_i, l_i, acc, k_cache_ptr, v_cache_ptr, pipeline, NUM_STAGES
        )
        # The row block's partials follow its first item, at its share of
        # the request's slots times row blocks.
        _store_balanced(
            program,
            acc,
            l_i,
            m_i,
            first_slot * row_blocks + head_block * splits,
            splits,
        )


@gluon.jit
def _merge_tiles(
    mid_o_ptr,
    mid_lse_ptr,
    first_row,
    splits,
    HEAD_DIM: gl.constexpr,
    layout: gl.constexpr,
    TILE: gl.constexpr,
):
    """Merge a row's ``splits`` partials stored from ``first_row`` on; returns
    the row max, sum and accumulator. The first TILE load at once, without a
    loop, so their loads all issue before the first wait; rows with more
    splits merge the rest TILE at a time. Masked buffer loads return 0 past
    the count without a select, and the affine offsets keep each partial one
    contiguous load. A row whose partials all have lse = -inf (no keys)
    returns max -inf, sum 0 and a zero accumulator."""
    offs_t = gl.arange(0, TILE, layout=gl.SliceLayout(1, layout))
    offs_e = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, layout))
    live = offs_t < splits
    rows = first_row + offs_t
    part_lse = cdna4.buffer_load(mid_lse_ptr, rows, mask=live, other=0.0)
    part_o = cdna4.buffer_load(
        mid_o_ptr,
        rows[:, None] * HEAD_DIM + offs_e[None, :],
        mask=live[:, None],
        other=0.0,
    )
    part_lse = gl.where(live, part_lse, -float("inf"))
    m_i = max(part_lse, axis=0)
    # Shift by 0 while every partial is empty (-inf), so -inf - -inf does not
    # make NaN.
    beta = gl.exp2(part_lse - gl.where(m_i == -float("inf"), 0.0, m_i))
    l_i = gl.sum(beta, axis=0)
    acc = gl.sum(part_o * beta[:, None], axis=0)
    for split_start in range(TILE, splits, TILE):
        split_c = split_start + offs_t
        live_c = split_c < splits
        rows_c = first_row + split_c
        lse_c = cdna4.buffer_load(mid_lse_ptr, rows_c, mask=live_c, other=0.0)
        o_c = cdna4.buffer_load(
            mid_o_ptr,
            rows_c[:, None] * HEAD_DIM + offs_e[None, :],
            mask=live_c[:, None],
            other=0.0,
        )
        lse_c = gl.where(live_c, lse_c, -float("inf"))
        next_m = maximum(m_i, max(lse_c, axis=0))
        shift = gl.where(next_m == -float("inf"), 0.0, next_m)
        alpha = gl.exp2(m_i - shift)
        beta_c = gl.exp2(lse_c - shift)
        l_i = l_i * alpha + gl.sum(beta_c, axis=0)
        acc = acc * alpha + gl.sum(o_c * beta_c[:, None], axis=0)
        m_i = next_m
    return m_i, l_i, acc


@gluon.jit
def gluon_mha_decode_balanced_reduce_gfx950(
    schedule_ptr: tl.const,
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    sink_ptr,
    k_cache_ptr,  # unused for loads; only its dtype feeds the shared config
    SM_SCALE: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HAS_SINK: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    """Merge the balanced split partials of one (query token, head); one
    wave. The config keeps the decode's BLOCK_M to locate the row's work
    items."""
    cfg = AttentionConfig(
        SM_SCALE,
        1,  # PAGE_TABLE_STRIDE: unused
        PAGE_SIZE,
        1,  # NUM_KV_SPLITS: the schedule sets each request's splits
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
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
    # The decode's row block and row of this (query token, head): rows run
    # over query positions, then over the KV head's query heads.
    request = q_index // cfg.MAX_SEQLEN_Q
    kv_head = q_head // cfg.GROUP_SIZE
    row = (q_index - request * cfg.MAX_SEQLEN_Q) * cfg.GROUP_SIZE + (
        q_head - kv_head * cfg.GROUP_SIZE
    )
    row_block = row // cfg.BLOCK_M
    head_block = kv_head * cfg.NUM_ROW_BLOCKS + row_block
    # The request's first slot and split count, one scalar load, place the
    # row block's partials.
    first_slot = gl.load(schedule_ptr + 2 * (1 + request))
    splits = gl.load(schedule_ptr + 2 * (1 + request) + 1)
    block_item = (
        first_slot * (cfg.NUM_KV_HEADS * cfg.NUM_ROW_BLOCKS) + head_block * splits
    )
    first_row = block_item * cfg.BLOCK_M + (row - row_block * cfg.BLOCK_M) * splits
    # Four partial values per lane: 16-byte loads.
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [256 // HEAD_DIM, HEAD_DIM // 4], [1, 1], [1, 0]
    )
    # Rows with at most 16 splits load one tile of 16: the loads of masked
    # tile rows still issue, so a tile of 64 costs short contexts.
    if splits <= 16:
        m_i, l_i, acc = _merge_tiles(
            mid_o_ptr, mid_lse_ptr, first_row, splits, HEAD_DIM, layout, 16
        )
    else:
        m_i, l_i, acc = _merge_tiles(
            mid_o_ptr, mid_lse_ptr, first_row, splits, HEAD_DIM, layout, 64
        )
    if HAS_SINK:
        # The sink is one more softmax entry, merged once for the row.
        sink = gl.load(sink_ptr + q_head).to(gl.float32) * _INV_LN2
        next_m = maximum(m_i, sink)
        alpha = gl.exp2(m_i - next_m)
        l_i = l_i * alpha + gl.exp2(sink - next_m)
        acc = acc * alpha
    offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, layout))
    # A row without keys and without a sink has no softmax mass: output 0.
    output = acc * gl.where(l_i > 0.0, 1.0 / l_i, 0.0)
    out_base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.HEAD_DIM
    cdna4.buffer_store(output.to(out_ptr.dtype.element_ty), out_ptr, out_base + offs_d)


def launch_gluon_mha_decode_schedule_gfx950(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    max_seqlen_q: int,
    window_left: int,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute the balanced split schedule (see Balanced Split Schedule) from
    the requests' current ``cache_seqlens``.

    Args:
        q: Query of the decode call, [num_requests * max_seqlen_q, heads, dim].
        k_cache: Paged key cache of the decode call.
        cache_seqlens: The decode call's cache lengths.
        max_seqlen_q: Query tokens per request of the decode call.
        window_left: The decode call's window; only full attention (-1)
            takes a schedule.
        out: int32 [>= num_requests + 1, 2] tensor to write the schedule into,
            or None to allocate one.

    Returns:
        The schedule.
    """
    if window_left >= 0:
        raise ValueError(
            "gluon_mha_decode_schedule_gfx950 schedules full attention, got "
            f"window_left={window_left}"
        )
    num_requests = q.shape[0] // max_seqlen_q
    if out is None:
        out = torch.empty((num_requests + 1, 2), device=q.device, dtype=torch.int32)
    else:
        _check_schedule(out, num_requests)
    if num_requests > 0:
        gluon_mha_decode_schedule_gfx950[(1,)](
            cache_seqlens,
            out,
            num_requests,
            _balanced_row_blocks_of(q.shape[1], k_cache.shape[2], max_seqlen_q),
            k_cache.shape[1],
            _BALANCED_TARGET_PROGRAMS,
            _BALANCED_MIN_SPLIT_PAGES,
            _BALANCED_MAX_SPLITS,
            _BALANCED_STAGES,
            _BALANCED_ALIGN_MIN_PAGES,
            num_warps=1,
        )
    return out


def _check_schedule(schedule: torch.Tensor, num_requests: int) -> None:
    if (
        schedule.dtype != torch.int32
        or schedule.dim() != 2
        or schedule.shape[0] < num_requests + 1
        or schedule.shape[1] != 2
        or not schedule.is_contiguous()
    ):
        raise ValueError(
            f"schedule needs a contiguous int32 [>= {num_requests + 1}, 2] "
            f"tensor, got {schedule.dtype} {tuple(schedule.shape)}"
        )


def launch_gluon_mha_decode_balanced_gfx950(
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
    *,
    schedule: torch.Tensor,
) -> torch.Tensor:
    """Full-attention paged decode split by a balanced schedule from
    launch_gluon_mha_decode_schedule_gfx950 (see Balanced Split Schedule);
    arguments otherwise as launch_gluon_mha_decode_gfx950."""
    # The schedule sizes the splits from the lengths; max_seqlen_k (the
    # split grid's sizing input) has nothing to size here.
    del max_seqlen_k
    if logit_cap != 0.0 or return_lse:
        raise ValueError(
            "gluon_mha_decode_balanced_gfx950 supports neither logit_cap nor "
            f"return_lse, got logit_cap={logit_cap}, return_lse={return_lse}"
        )
    reject_block_scales(
        "gluon_mha_decode_balanced_gfx950",
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if window_left >= 0:
        raise ValueError("gluon_mha_decode_balanced_gfx950 serves full attention only")
    num_requests = q.shape[0] // max_seqlen_q
    _check_schedule(schedule, num_requests)
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = q.shape[1] // num_kv_heads
    block_m, num_warps = _row_tile(max_seqlen_q, group_size)
    num_row_blocks = math.ceil(max_seqlen_q * group_size / block_m)
    head_blocks = num_kv_heads * num_row_blocks
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = softmax_scale * _INV_LN2_VALUE
    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    output = torch.empty(
        q.shape, device=q.device, dtype=torch.bfloat16 if is_fp8 else q.dtype
    )
    if num_requests == 0:
        return output
    # A balanced schedule has at most TARGET items, or one per row block.
    num_programs = builtins.max(_BALANCED_TARGET_PROGRAMS, num_requests * head_blocks)
    mid_o = torch.empty(
        (num_programs, block_m, head_dim), device=q.device, dtype=torch.float32
    )
    mid_lse = torch.empty((num_programs, block_m), device=q.device, dtype=torch.float32)
    gluon_mha_decode_balanced_gfx950[(num_programs,)](
        q,
        cache_seqlens,
        num_requests,
        schedule,
        page_table,
        k_cache,
        v_cache,
        mid_o,
        mid_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sm_scale,
        page_table.stride(0),
        k_cache.shape[1],
        max_seqlen_q,
        q.shape[1],
        num_kv_heads,
        head_dim,
        block_m,
        64,
        num_warps,
        is_fp8,
        _BALANCED_STAGES,
        _BALANCED_TARGET_PROGRAMS,
        num_warps=num_warps,
        **sched_compile_options(),
    )
    has_sink = sinks is not None
    gluon_mha_decode_balanced_reduce_gfx950[(q.shape[0], q.shape[1])](
        schedule,
        mid_o,
        mid_lse,
        output,
        sinks if has_sink else q,
        k_cache,
        sm_scale,
        max_seqlen_q,
        k_cache.shape[1],
        q.shape[1],
        num_kv_heads,
        head_dim,
        block_m,
        64,
        has_sink,
        is_fp8,
        num_warps=1,
    )
    return output


def supports_gluon_mha_decode_balanced_gfx950(
    max_seqlen_k: int, page_size: int
) -> bool:
    """Whether calls whose lengths reach ``max_seqlen_k`` (the model's
    context limit) can ever prefer the balanced path."""
    return max_seqlen_k >= _BALANCED_MIN_PAGES * page_size


def _balanced_row_blocks_of(
    num_q_heads: int, num_kv_heads: int, max_seqlen_q: int
) -> int:
    group_size = num_q_heads // num_kv_heads
    block_m, _ = _row_tile(max_seqlen_q, group_size)
    return num_kv_heads * math.ceil(max_seqlen_q * group_size / block_m)


def limits_gluon_mha_decode_balanced_gfx950(
    *, num_q_heads: int, num_kv_heads: int, max_seqlen_q: int
) -> tuple[int, int]:
    """(min_pages, max_requests): the balanced path can be preferred only
    for batches whose longest request spans at least min_pages pages and
    that hold at most max_requests requests. With more, every request's row
    blocks leave at most one work item per request to share out, too few
    for balancing to beat the split grid."""
    row_blocks = _balanced_row_blocks_of(num_q_heads, num_kv_heads, max_seqlen_q)
    return _BALANCED_MIN_PAGES, (_BALANCED_TARGET_PROGRAMS - 1) // row_blocks - 1


def prefers_gluon_mha_decode_balanced_gfx950(
    pages,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    page_size: int,
) -> bool:
    """Whether the balanced path beats the split grid for requests holding
    ``pages`` KV pages each (a padding request counts one), on the host.

    Both are weighed by their busiest CU: items over the CUs times the pages
    each streams. The split grid's split count comes from the padded batch
    and max_seqlen_k, as get_config chooses it; a balanced item also pays
    _BALANCED_ITEM_PAGES for its schedule lookup, and its split size is
    estimated by giving every request its first split before sharing the
    rest of the items.

    Args:
        pages: Per-request page counts of the (padded) batch.
        num_q_heads, num_kv_heads, max_seqlen_q: The decode call's shape.
        max_seqlen_k: The split grid's max_seqlen_k (the context limit under
            graphs).
        page_size: The KV cache's page size.
    """
    pages = np.asarray(pages, dtype=np.int64)
    num_requests = pages.size
    min_pages, max_requests = limits_gluon_mha_decode_balanced_gfx950(
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, max_seqlen_q=max_seqlen_q
    )
    if num_requests == 0 or num_requests > max_requests:
        return False
    longest = int(pages.max())
    if longest < min_pages:
        return False
    row_blocks = _balanced_row_blocks_of(num_q_heads, num_kv_heads, max_seqlen_q)
    grid_splits = select_kv_splits(
        base_ctas=num_requests * row_blocks,
        num_pages=triton.cdiv(max_seqlen_k, page_size),
        sm_count=_GFX950_SM_COUNT,
        max_splits=_FULL_DECODE_MAX_SPLITS,
    )
    grid_load = triton.cdiv(
        num_requests * row_blocks * grid_splits, _GFX950_SM_COUNT
    ) * triton.cdiv(longest, grid_splits)
    spare = _BALANCED_TARGET_PROGRAMS - row_blocks * num_requests
    split_pages = builtins.max(
        triton.cdiv(row_blocks * int(pages.sum()), spare),
        _BALANCED_MIN_SPLIT_PAGES,
        triton.cdiv(longest, _BALANCED_MAX_SPLITS),
    )
    slots = int(np.maximum(-(-pages // split_pages), 1).sum())
    balanced_load = triton.cdiv(row_blocks * slots, _GFX950_SM_COUNT) * (
        split_pages + _BALANCED_ITEM_PAGES
    )
    return balanced_load < grid_load
