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
    max,
    maximum,
)

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy
buffer = cdna4.buffer
cdiv = gl.cdiv
_GFX950_SM_COUNT = 256
# Pipeline depth of the full-attention decode K/V page loads: LDS buffers per
# work-group, so NUM_STAGES - 1 pages are prefetched while one is consumed.
# With the split schedule's one round of work-groups, two buffers measured
# best: deeper pipelines did not help while address translation dominated the
# load latency, and with fast translation they lost a little. One prefetched
# BF16 head-dim-128 page is 32 buffer_load ... lds copies for a one-wave
# work-group, well within the 63 loads s_waitcnt vmcnt can leave outstanding.
_FULL_DECODE_STAGES = 2
# Full-attention decode splits each request's keys in proportion to its share of
# the batch's pages, read from cache_seqlens when the kernel runs (see Split
# Schedule), into at most _FULL_DECODE_TARGET_PROGRAMS work items: as many
# work-groups as the GPU runs at once with two per CU. A split reads at least
# _FULL_DECODE_MIN_SPLIT_PAGES pages, and a request takes at most
# _FULL_DECODE_MAX_SPLITS splits per row block.
_FULL_DECODE_TARGET_PROGRAMS = 2 * _GFX950_SM_COUNT
_FULL_DECODE_MIN_SPLIT_PAGES = 2
_FULL_DECODE_MAX_SPLITS = 128
# Requests whose lengths one vector load reads while scheduling the splits.
_SCHEDULE_CHUNK = gl.constexpr(64)
# Rows of a decode work-group: four waves of one 16-row MFMA tile each.
_DECODE_MAX_BLOCK_M = 64
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
        # The split reduce loads four partial values per lane (16 bytes).
        reduce_layout = gl.BlockedLayout(
            [1, 4], [256 // HEAD_DIM, HEAD_DIM // 4], [1, 1], [1, 0]
        )

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
# Split Schedule
# ===-----------------------------------------------------------------------===#
#
# Full-attention decode sizes its splits from the lengths in cache_seqlens when
# it runs, so a captured graph adapts to every replay: a padding request (one
# page) gets one split per row block, and the requests with keys share the
# rest of the TARGET work items in proportion to their pages. With
# row_blocks (KV head, row block) pairs per request, every split reads
# split_pages pages, except a request's last one:
#
#   split_pages = max(cdiv(row_blocks * sum(pages), TARGET - row_blocks * num_requests),
#                     MIN_SPLIT_PAGES)
#   splits(r)   = clamp(cdiv(pages(r), split_pages), 1, MAX_SPLITS)
#
# Rounding each request's splits up adds at most one per row block, so the
# work items number at most TARGET, or row_blocks * num_requests when the
# batch alone exceeds it. A request's items are its row blocks times its
# splits, row block major, and the items of all requests follow in request
# order. The decode kernel records each row block's first item and split
# count, from which the split reduce finds its partials.


@gluon.jit
def _load_lengths(cache_seqlens_ptr, start, end, layout: gl.constexpr):
    """Lengths of requests [start, start + _SCHEDULE_CHUNK) before ``end``, 0
    past it."""
    index = start + gl.arange(0, _SCHEDULE_CHUNK, layout=layout)
    return gl.load(cache_seqlens_ptr + index, mask=index < end, other=0)


@gluon.jit
def _request_splits(
    seq_lens, split_pages, PAGE_SIZE: gl.constexpr, MAX_SPLITS: gl.constexpr
):
    pages = cdiv(gl.maximum(seq_lens, 0), PAGE_SIZE)
    return gl.minimum(gl.maximum(cdiv(pages, split_pages), 1), MAX_SPLITS)


@gluon.jit
def _split_pages(
    cache_seqlens_ptr,
    num_requests,
    first_lens,
    ROW_BLOCKS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    TARGET_PROGRAMS: gl.constexpr,
    MIN_SPLIT_PAGES: gl.constexpr,
    PAGE_ALIGN: gl.constexpr,
    layout: gl.constexpr,
):
    """Pages per split, from the lengths of the first _SCHEDULE_CHUNK requests
    (``first_lens``) and of the rest, loaded here."""
    total = gl.sum(cdiv(gl.maximum(first_lens, 0), PAGE_SIZE), axis=0)
    for start in range(_SCHEDULE_CHUNK, num_requests, _SCHEDULE_CHUNK):
        seq_lens = _load_lengths(cache_seqlens_ptr, start, num_requests, layout)
        total += gl.sum(cdiv(gl.maximum(seq_lens, 0), PAGE_SIZE), axis=0)
    # Items left once every (request, row block) has its first split.
    spare = gl.maximum(TARGET_PROGRAMS - ROW_BLOCKS * num_requests, 1)
    split_pages = gl.maximum(cdiv(total * ROW_BLOCKS, spare), MIN_SPLIT_PAGES)
    # The pipelined loop consumes PAGE_ALIGN pages per iteration, so a longer
    # split takes a whole number of iterations instead of computing on pages
    # past its end.
    return gl.where(
        split_pages > PAGE_ALIGN,
        cdiv(split_pages, PAGE_ALIGN) * PAGE_ALIGN,
        split_pages,
    )


@gluon.jit
def _find_item(
    cache_seqlens_ptr,
    num_requests,
    item,
    ROW_BLOCKS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    TARGET_PROGRAMS: gl.constexpr,
    MIN_SPLIT_PAGES: gl.constexpr,
    PAGE_ALIGN: gl.constexpr,
    layout: gl.constexpr,
):
    """Return the request holding work item ``item``, its length, its first
    item, its split count, the pages per split, and the batch's item count.
    The request is num_requests past the last item."""
    # One load covers batches of up to _SCHEDULE_CHUNK requests; larger ones
    # load the rest a chunk at a time.
    seq_lens = _load_lengths(cache_seqlens_ptr, 0, num_requests, layout)
    split_pages = _split_pages(
        cache_seqlens_ptr,
        num_requests,
        seq_lens,
        ROW_BLOCKS,
        PAGE_SIZE,
        TARGET_PROGRAMS,
        MIN_SPLIT_PAGES,
        PAGE_ALIGN,
        layout,
    )
    offs = gl.arange(0, _SCHEDULE_CHUNK, layout=layout)
    chunk_first = num_requests * 0
    request = num_requests
    seq_len = chunk_first
    first_item = chunk_first
    splits = chunk_first + 1
    for start in range(0, num_requests, _SCHEDULE_CHUNK):
        if start > 0:
            seq_lens = _load_lengths(cache_seqlens_ptr, start, num_requests, layout)
        valid = start + offs < num_requests
        request_splits = _request_splits(seq_lens, split_pages, PAGE_SIZE, MAX_SPLITS)
        items = gl.where(valid, request_splits * ROW_BLOCKS, 0)
        ends = chunk_first + gl.cumsum(items, axis=0)
        # Requests of this chunk whose items all precede ``item``. Lanes past
        # the batch end where the chunk ends, so they count only when
        # ``item`` lies past the chunk as well.
        before = gl.sum((ends <= item).to(gl.int32), axis=0)
        found = (start + before < num_requests) & (before < _SCHEDULE_CHUNK)
        found = found & (request == num_requests)
        lane = offs == before
        request = gl.where(found, start + before, request)
        seq_len = gl.where(found, gl.sum(gl.where(lane, seq_lens, 0), axis=0), seq_len)
        first_item = gl.where(
            found, gl.sum(gl.where(lane, ends - items, 0), axis=0), first_item
        )
        splits = gl.where(
            found, gl.sum(gl.where(lane, request_splits, 0), axis=0), splits
        )
        chunk_first += gl.sum(items, axis=0)
    return request, seq_len, first_item, splits, split_pages, chunk_first


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
    item: gl.tensor

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
        item,
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
        self.item = item

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
        """Locate this program's request, KV head, row block and split from its
        position in a (request, row block, split) grid."""
        batch = gl.program_id(0)
        return AttentionProgram.locate(
            cfg,
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            cache_seqlens_ptr,
            mid_o_ptr,
            mid_lse_ptr,
            batch,
            gl.load(cache_seqlens_ptr + batch),
            gl.program_id(1),
            gl.program_id(2),
            cfg.NUM_KV_SPLITS,
            1,
            gl.program_id(2),
        )

    @gluon.jit
    def create_scheduled(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        num_requests,
        TARGET_PROGRAMS: gl.constexpr,
        MIN_SPLIT_PAGES: gl.constexpr,
        PAGE_ALIGN: gl.constexpr,
    ):
        """Locate this program's work item in a one-dimensional grid.

        The items are each request's row blocks times its splits, in request
        order (see _find_item). Returns the program, whether the grid
        position holds an item (positions past the last item have no work),
        the batch's item count, and the request's split count.
        """
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [cfg.NUM_WARPS], [0])
        item = gl.program_id(0)
        request, seq_len, first_item, splits, split_pages, num_items = _find_item(
            cache_seqlens_ptr,
            num_requests,
            item,
            cfg.NUM_KV_HEADS * cfg.NUM_ROW_BLOCKS,
            cfg.PAGE_SIZE,
            cfg.NUM_KV_SPLITS,
            TARGET_PROGRAMS,
            MIN_SPLIT_PAGES,
            PAGE_ALIGN,
            layout,
        )
        has_work = request < num_requests
        head_block = (item - first_item) // splits
        split_id = item - first_item - head_block * splits
        program = AttentionProgram.locate(
            cfg,
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            cache_seqlens_ptr,
            mid_o_ptr,
            mid_lse_ptr,
            gl.minimum(request, num_requests - 1),
            seq_len,
            head_block,
            split_id,
            splits,
            split_pages,
            item,
        )
        return program, has_work, num_items, splits

    @gluon.jit
    def locate(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        batch,
        seq_len,
        head_block,
        split_id,
        num_splits,
        min_split_pages,
        item,
    ):
        """Bound the keys of one request's row block and split.

        The rows of a block share one pass over the union of the keys they
        see: up to its last row's causal end and, under a sliding window, from
        its first row's window start. A request's cache length counts all of
        its MAX_SEQLEN_Q query tokens, so query position p sees the keys
        before seq_len - (MAX_SEQLEN_Q - 1 - p).
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
        pages_per_split = gl.maximum(cdiv(num_pages, num_splits), min_split_pages)
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
            item,
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
            l_i += gl.exp2(sink_log2 - m_i * cfg.SM_SCALE)
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
        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
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
    def split_partial(self, acc, l_i, m_i):
        """This split's normalized output and log2-sum-exp per row, in the
        store layout.

        A row without keys in this split (an empty split, or one past the
        row's causal end or before its window) gets a zero partial with
        lse = -inf, which the merge ignores.
        """
        cfg = self.cfg
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        m_i = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.store_layout))
        has_kv = l_i > 0.0
        recip_l_i = 1.0 / l_i
        part_o = gl.where(has_kv[:, None], acc * recip_l_i[:, None], 0.0)
        part_lse = gl.where(
            has_kv,
            m_i * cfg.SM_SCALE + gl.log2(l_i),
            -float("inf"),
        )
        return part_o, part_lse

    @gluon.jit
    def store_split(self, acc, l_i, m_i):
        """Store the partial at [token, head, split] of a split grid."""
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        tokens = self.row_tokens(offs_m)
        q_heads = self.row_heads(offs_m)
        valid = self.row_valid(offs_m)
        part_o, part_lse = self.split_partial(acc, l_i, m_i)
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
    def store_item(self, acc, l_i, m_i, splits):
        """Store the partial of a scheduled grid's work item.

        A row block's items hold BLOCK_M x splits partials, row major, so the
        split reduce reads each row's partials from one contiguous range.
        """
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        valid = self.row_valid(offs_m)
        part_o, part_lse = self.split_partial(acc, l_i, m_i)
        first = (self.item - self.split_id) * cfg.BLOCK_M
        rows = first + offs_m * splits + self.split_id
        cdna4.buffer_store(
            part_o,
            self.mid_o_ptr,
            rows[:, None] * cfg.HEAD_DIM + offs_d[None, :],
            mask=valid[:, None],
        )
        cdna4.buffer_store(part_lse, self.mid_lse_ptr, rows, mask=valid)

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
    # prologue than it hides; ``pipeline`` false takes the single buffer too.
    short = program.split_end - program.split_start < iteration_tokens
    if short | (pipeline == 0):
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
def _decode_item(
    program,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    pipeline,
    splits,
    NUM_STAGES: gl.constexpr,
):
    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_state(q_ptr, False)
    m_i, l_i, acc = _stream_split_pipelined(
        program, q, m_i, l_i, acc, k_cache_ptr, v_cache_ptr, pipeline, NUM_STAGES
    )
    program.store_item(acc, l_i, m_i, splits)


@gluon.jit(do_not_specialize=("num_requests",))
def gluon_mha_decode_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    block_items_ptr,
    num_requests,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
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
    MIN_SPLIT_PAGES: gl.constexpr,
):
    """Full-attention split of one work item of the split schedule; the grid
    has one program per possible item, and the ones past the last item exit."""
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        MAX_SPLITS,
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
    program, has_work, num_items, splits = AttentionProgram.create_scheduled(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        num_requests,
        TARGET_PROGRAMS,
        MIN_SPLIT_PAGES,
        NUM_STAGES,
    )
    if NUM_WARPS == 1:
        # A one-wave work-group computes every page itself, and the pipeline's
        # extra per-page instructions cost more than the latency they hide
        # once two work-groups per CU stream the KV. Pipeline only while the
        # work items leave CUs with at most one work-group, or for splits of
        # 32 pages or more, which spend longest waiting on loads.
        long_split = program.split_end - program.split_start >= 32 * PAGE_SIZE
        pipeline = (num_items * 2 <= TARGET_PROGRAMS) | long_split
    else:
        # Every multi-wave work-group pipelines.
        pipeline = num_items >= 0
    if has_work:
        if program.split_id == 0:
            # The reduce finds a row block's partials from its first item and
            # split count, so it does not evaluate the schedule again.
            block = (
                program.batch * (NUM_KV_HEADS * cfg.NUM_ROW_BLOCKS)
                + program.kv_head * cfg.NUM_ROW_BLOCKS
                + program.row_start // BLOCK_M
            )
            gl.store(block_items_ptr + 2 * block, program.item)
            gl.store(block_items_ptr + 2 * block + 1, splits)
        _decode_item(
            program, q_ptr, k_cache_ptr, v_cache_ptr, pipeline, splits, NUM_STAGES
        )


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
    block_items_ptr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
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
    """Merge the split partials of one (query token, head).

    One wave merges one row; the config keeps the decode's BLOCK_M to locate
    the row's work items, with a single wave.
    """
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        MAX_SPLITS,
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
    # The decode kernel's row block and row of this (query token, head):
    # rows run over query positions, then over the KV head's query heads.
    request = q_index // cfg.MAX_SEQLEN_Q
    kv_head = q_head // cfg.GROUP_SIZE
    row = (q_index - request * cfg.MAX_SEQLEN_Q) * cfg.GROUP_SIZE + (
        q_head - kv_head * cfg.GROUP_SIZE
    )
    row_block = row // cfg.BLOCK_M
    # The row block's first work item and split count, from the decode kernel.
    block = (request * cfg.NUM_KV_HEADS + kv_head) * cfg.NUM_ROW_BLOCKS + row_block
    first_item = gl.load(block_items_ptr + 2 * block)
    splits = gl.load(block_items_ptr + 2 * block + 1)
    # The row's partials follow one another from first_row on.
    first_row = first_item * cfg.BLOCK_M + (row - row_block * cfg.BLOCK_M) * splits

    # A chunk holds 64 values per lane, so splits up to 64 (head dim 64) or
    # 32 (head dim 128) load at once.
    SPLIT_CHUNK: gl.constexpr = 4096 // HEAD_DIM
    offs_s = gl.arange(0, SPLIT_CHUNK, layout=gl.SliceLayout(1, cfg.reduce_layout))
    offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, cfg.reduce_layout))
    no_splits = gl.full(
        [SPLIT_CHUNK], -float("inf"), gl.float32, gl.SliceLayout(1, cfg.reduce_layout)
    )
    m_i = max(no_splits, axis=0)
    l_i = gl.sum(gl.exp2(no_splits), axis=0)
    acc = gl.zeros(
        [HEAD_DIM], dtype=gl.float32, layout=gl.SliceLayout(0, cfg.reduce_layout)
    )
    # Every split of the row stored its partial, with lse = -inf when it had
    # no keys. Splits past the row's count reread its last partial and get
    # lse = -inf: unlike masked loads, which the compiler waits on a few at a
    # time to select their fill value, these loads all issue at once.
    for split_start in range(0, splits, SPLIT_CHUNK):
        split = split_start + offs_s
        rows = first_row + gl.minimum(split, splits - 1)
        part_lse = gl.load(mid_lse_ptr + rows)
        part_o = cdna4.buffer_load(
            mid_o_ptr, rows[:, None] * HEAD_DIM + offs_d[None, :]
        )
        part_lse = gl.where(split < splits, part_lse, -float("inf"))
        next_m = maximum(m_i, max(part_lse, axis=0))
        # A chunk can hold only rows without keys (lse = -inf) while m_i is
        # still -inf; shift by 0 then to keep -inf - -inf from making NaN.
        shift = gl.where(next_m == -float("inf"), 0.0, next_m)
        alpha = gl.exp2(m_i - shift)
        beta = gl.exp2(part_lse - shift)
        l_i = l_i * alpha + gl.sum(beta, axis=0)
        acc = acc * alpha + gl.sum(part_o * beta[:, None], axis=0)
        m_i = next_m
    if HAS_SINK:
        # The sink is one more softmax entry, merged once for the row.
        sink = gl.load(sink_ptr + q_head).to(gl.float32) * _INV_LN2
        next_m = maximum(m_i, sink)
        alpha = gl.exp2(m_i - next_m)
        l_i = l_i * alpha + gl.exp2(sink - next_m)
        acc = acc * alpha
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
    # Sliding window: splits of every request. Full attention: the most
    # splits a request's row block can take (the schedule sets each count).
    num_kv_splits: int
    # Full attention: the scheduled grid, the most work items of any batch.
    num_programs: int
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
    block_m, num_warps = _row_tile(max_seqlen_q, group_size)
    num_row_blocks = math.ceil(max_seqlen_q * group_size / block_m)
    num_requests = q.shape[0] // max_seqlen_q
    is_sliding = window_left >= 0
    window_left = window_left if is_sliding else -1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    head_blocks = num_kv_heads * num_row_blocks
    if is_sliding:
        effective_seqlen_k = min(max_seqlen_k, window_left)
        num_pages = (effective_seqlen_k + page_size - 1) // page_size
        target_splits = 8 if num_requests <= 4 else 4
        num_kv_splits = builtins.max(1, min(target_splits, num_pages))
        num_programs = num_requests * head_blocks * num_kv_splits
    else:
        # The schedule reads the lengths when the kernel runs; max_seqlen_k
        # (the model's context limit under graph capture) does not size it.
        num_kv_splits = _FULL_DECODE_MAX_SPLITS
        num_programs = builtins.max(
            _FULL_DECODE_TARGET_PROGRAMS, num_requests * head_blocks
        )
    return LaunchConfig(
        num_q_heads=q.shape[1],
        num_kv_heads=num_kv_heads,
        num_requests=num_requests,
        num_row_blocks=num_row_blocks,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
        num_programs=num_programs,
        num_stages=_FULL_DECODE_STAGES,
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
    elif config.num_requests > 0:
        # Full attention always splits the keys, by the schedule in
        # _find_item; one partial of BLOCK_M rows per work item.
        mid_o = torch.empty(
            (config.num_programs, config.block_m, config.head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        mid_lse = torch.empty(
            (config.num_programs, config.block_m),
            device=q.device,
            dtype=torch.float32,
        )
        # Each row block's first work item and split count.
        block_items = torch.empty(
            (config.num_requests, head_blocks, 2), device=q.device, dtype=torch.int32
        )
        gluon_mha_decode_gfx950[(config.num_programs,)](
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            mid_o,
            mid_lse,
            block_items,
            config.num_requests,
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
            is_fp8,
            config.num_stages,
            _FULL_DECODE_TARGET_PROGRAMS,
            _FULL_DECODE_MIN_SPLIT_PAGES,
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
            block_items,
            config.sm_scale,
            page_table.stride(0),
            config.num_kv_splits,
            max_seqlen_q,
            config.page_size,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            has_sink,
            is_fp8,
            num_warps=1,
        )
    return output
