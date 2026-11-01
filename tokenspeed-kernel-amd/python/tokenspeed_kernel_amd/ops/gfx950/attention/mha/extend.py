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
the MFMA ``M`` dimension (``BLOCK_M = BLOCK_Q * PADDED_GROUP_SIZE``), so each KV tile is
loaded once and reused across the group. The grid is ragged and token-based,
``(total_num_q_blocks, n_kv_heads)``: each program binary-searches
``cu_seqlens_q`` to self-locate its request/q-block, keeping the launch
CUDA-graph static while a ``q=1`` request costs a single block.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import (
    _INV_LN2,
    _INV_LN2_VALUE,
    _LN2,
    MAX_KV_SPLITS,
    InputStrides,
    _join_columns,
    _keep_in_cluster,
    _split_columns,
    _wave_maxima_unchanged,
    attention_layouts,
    max,
    maximum,
    select_kv_splits,
)

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy
buffer = cdna4.buffer

# The reduce kernel reads all split LSEs at once, one per lane of its single
# wave, so the split count must fit in a wave.
_LSE_TILE = gl.constexpr(64)
assert MAX_KV_SPLITS <= _LSE_TILE.value
# Split partials the reduce kernel loads per loop iteration.
_SPLIT_CHUNK = gl.constexpr(8)


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
# (BLOCK_M, NUM_WARPS) of the query tiles, smallest first. The 8-wave tile
# runs the warp-pipelined page loop (_stream_kv_warp_pipelined); the launcher
# picks it by _use_warp_pipeline rather than by query length.
_EXTEND_TILES = ((64, 2), (128, 4), (256, 8))
_WARP_PIPELINE_NUM_WARPS = 8
# Limits of the 8-wave tile (see _use_warp_pipeline): the shortest cached
# prefix and sliding window, in keys, and the most rounds of work-groups a
# sliding-window launch may take. Measured on MI355X with 16 query and 2 KV
# heads (Muse Spark TP8), CUDA graph replay; re-checked on MI350X.
_WARP_PIPELINE_MIN_PREFIX = 8192
_WARP_PIPELINE_MIN_WINDOW = 2047
_WARP_PIPELINE_MAX_WINDOW_ROUNDS = 4


def _warp_pipeline_supported(
    *, head_dim: int, element_size: int, window_left: int
) -> bool:
    """Whether a layer can take the 8-wave tile: head dim 64 with 16-bit K/V,
    where a wave's operands fit in 256 VGPRs and the K/V rings in 64 KB of
    LDS, and full attention or a window of at least 2047 keys."""
    return (
        head_dim == 64
        and element_size == 2
        and (window_left < 0 or window_left >= _WARP_PIPELINE_MIN_WINDOW)
    )


def _use_warp_pipeline(
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    total_q: int,
    group_size: int,
    n_kv_heads: int,
    head_dim: int,
    element_size: int,
    window_left: int,
) -> bool:
    """Whether an extend launch takes the 8-wave warp-pipelined tile.

    The 8-wave loop finishes a page faster than the 4-wave one, but costs
    more per work-group: it fills its K and V rings before the first page,
    and one work-group takes a whole CU where two 4-wave ones fit. It wins
    when every work-group streams many cached keys: under full attention a
    cached prefix of at least _WARP_PIPELINE_MIN_PREFIX keys (split-K keeps
    the grid full); under a sliding window a full window in every tile, a
    work-group for every CU (windows never split), and at most
    _WARP_PIPELINE_MAX_WINDOW_ROUNDS rounds of work-groups. Over more rounds
    of one-window tiles the 4-wave tile measured as fast or faster.

    The cached prefix is read as max_seqlen_k - max_seqlen_q, so callers
    should pass the longest request's length as max_seqlen_k, as the
    serving backend does for pure extends. A looser bound (the context
    limit) can send short-prefix extends to the 8-wave tile, which then
    runs up to 14% slower.
    """
    if max_seqlen_q <= _EXTEND_TILES[0][0] or not _warp_pipeline_supported(
        head_dim=head_dim, element_size=element_size, window_left=window_left
    ):
        return False
    prefix = max_seqlen_k - max_seqlen_q
    if window_left < 0:
        return prefix >= _WARP_PIPELINE_MIN_PREFIX
    block_m, num_warps = _EXTEND_TILES[-1]
    block_q = _extend_tile(block_m, num_warps, group_size)[1]
    rounds = triton.cdiv(total_q, block_q) * n_kv_heads / _GFX950_SM_COUNT
    return prefix >= window_left and 1 <= rounds <= _WARP_PIPELINE_MAX_WINDOW_ROUNDS


def _extend_tile(
    block_m: int, num_warps: int, group_size: int
) -> tuple[int, int, int, int]:
    # Each query position owns a power-of-two head span. Mask unused head
    # lanes instead of shrinking MFMA/range extents for groups such as 5 or 6.
    padded_group_size = triton.next_power_of_2(group_size)
    if block_m < padded_group_size:
        block_m = padded_group_size
    block_q = block_m // padded_group_size
    return block_m, block_q, _EXTEND_BLOCK_N, num_warps


def _select_extend_tile(
    max_seqlen_q: int, group_size: int, warp_pipeline: bool
) -> tuple[int, int, int, int]:
    # Return (BLOCK_M, BLOCK_Q, BLOCK_N, NUM_WARPS): the 8-wave tile when
    # warp_pipeline, else the smallest other tile with at least max_seqlen_q
    # rows, or the largest one.
    tiles = [
        tile
        for tile in _EXTEND_TILES
        if (tile[1] == _WARP_PIPELINE_NUM_WARPS) == warp_pipeline
    ]
    for block_m, num_warps in tiles:
        if max_seqlen_q <= block_m:
            break
    return _extend_tile(block_m, num_warps, group_size)


@gluon.aggregate
class ExtendConfig:
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    GROUP_SIZE: gl.constexpr
    PADDED_GROUP_SIZE: gl.constexpr
    HEAD_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_Q: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
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
        assert N_HEADS % N_KV_HEADS == 0
        padded_group_size = triton.next_power_of_2(N_HEADS // N_KV_HEADS)
        assert BLOCK_M == BLOCK_Q * padded_group_size

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
        self.PADDED_GROUP_SIZE = gl.constexpr(padded_group_size)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_Q = gl.constexpr(BLOCK_Q)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
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
        return self.q_pos_base + offs_m // self.cfg.PADDED_GROUP_SIZE

    @gluon.jit
    def row_qhead(self, offs_m):
        return self.kv_head * self.cfg.GROUP_SIZE + offs_m % self.cfg.PADDED_GROUP_SIZE

    @gluon.jit
    def row_has_head(self, offs_m):
        return offs_m % self.cfg.PADDED_GROUP_SIZE < self.cfg.GROUP_SIZE

    @gluon.jit
    def load_q(self):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        q_pos = self.row_qpos(offs_m)
        q_head = self.row_qhead(offs_m)
        row = self.seq_base + q_pos
        offsets = cfg.q_strides.offsets(row[:, None], q_head[:, None], offs_d[None, :])
        mask = (q_pos[:, None] < self.seq_len) & self.row_has_head(offs_m)[:, None]
        return cdna4.buffer_load(self.q_ptr, offsets, mask=mask, other=0.0)

    @gluon.jit
    def load_page_indices(self, start_n, last_page, count: gl.constexpr):
        """Physical pages of the count pages starting at token start_n.

        Pages past last_page repeat it (an L2 hit): a masked copy becomes a
        branch after which the compiler waits for every outstanding load
        (s_waitcnt vmcnt(0)). Every lane loads all count entries of the
        read-only (tl.const) table, so these are scalar loads (s_load), counted
        by lgkmcnt and never waiting for the page copies. Each costs a
        scalar-cache round trip at its use.
        """
        cfg = self.cfg
        layout: gl.constexpr = gl.BlockedLayout([count], [64], [cfg.NUM_WARPS], [0])
        offs = gl.arange(0, count, layout=layout)
        page_index = gl.minimum(start_n // cfg.PAGE_SIZE + offs, last_page)
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index
        )

    @gluon.jit
    def select_page(self, pages, index: gl.constexpr):
        """Entry index of load_page_indices' result."""
        offs = gl.arange(0, pages.shape[0], layout=pages.type.layout)
        return max(gl.where(offs == index, pages, 0), axis=0)

    @gluon.jit
    def page_tile_offsets(self):
        """Element offsets of a [BLOCK_N, HEAD_DIM] tile within a page, before
        the KV head's offset."""
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        return offs_n[:, None] * (cfg.N_KV_HEADS * cfg.HEAD_DIM) + offs_d[None, :]

    @gluon.jit
    def load_page_to_shared(self, cache_ptr, physical_page, tile_offsets, smem):
        """Copy this KV head's tile of one page into LDS; the caller commits."""
        cfg = self.cfg
        # One page holds BLOCK_N tokens. A descriptor based at the page keeps
        # the 64-bit page address in SGPRs, so each lane needs only its int32
        # offset within the page, which stays the same for every page.
        page_elems: gl.constexpr = cfg.PAGE_SIZE * cfg.N_KV_HEADS * cfg.HEAD_DIM
        elem_bytes: gl.constexpr = cache_ptr.dtype.element_ty.primitive_bitwidth // 8
        page = buffer.make_buffer_descriptor(
            cache_ptr + physical_page.to(gl.int64) * page_elems,
            page_elems * elem_bytes,
        )
        buffer.load_to_shared(
            smem, page, tile_offsets, soffset=self.kv_head * cfg.HEAD_DIM
        )

    @gluon.jit
    def issue_load_kv(self, physical_page, tile_offsets, k_smem, v_smem):
        """Copy one page of K and V into LDS as one async group each. The page
        index is already in an SGPR from its scalar load."""
        self.load_page_to_shared(self.k_cache_ptr, physical_page, tile_offsets, k_smem)
        async_copy.commit_group()
        self.load_page_to_shared(self.v_cache_ptr, physical_page, tile_offsets, v_smem)
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
    def mask_kv(self, qk, start_n, kv_end):
        """Mask keys at or past kv_end, past each row's causal diagonal and
        before its sliding window."""
        cfg = self.cfg
        offs_n = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)
        )
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
        # The packed heads of a query position share its visibility.
        diag_row = self.prefix + self.row_qpos(offs_m)
        mask = offs_n[None, :] < kv_end
        if cfg.IS_CAUSAL:
            mask &= offs_n[None, :] <= diag_row[:, None]
        if cfg.WINDOW_LEFT >= 0:
            # Inclusive left edge, flash-attn window_size=(W, 0): W + 1 keys.
            mask &= offs_n[None, :] >= diag_row[:, None] - cfg.WINDOW_LEFT
        return gl.where(mask, qk, -float("inf"))

    @gluon.jit
    def init_attention_state(self):
        cfg = self.cfg
        if cfg.HAS_SINK:
            offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
            q_head = self.row_qhead(offs_m)
            sink_log2 = (
                gl.load(
                    self.sink_ptr + q_head, mask=self.row_has_head(offs_m), other=0.0
                ).to(gl.float32)
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
        mask = (q_pos[:, None] < self.seq_len) & self.row_has_head(offs_m)[:, None]
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
            mask = (q_pos < self.seq_len) & self.row_has_head(offs_m)
            lse_l_i = gl.where(l_i > 0.0, l_i, 1.0)
            # Softmax runs in base-2 (exp2 hardware fast path), so m_i*SM_SCALE +
            # log2(l_i) is the LSE in base-2 units. Convert to natural log (the
            # public op contract / torch.logsumexp convention) by scaling by ln2.
            lse = (m_i * cfg.SM_SCALE + gl.log2(lse_l_i)) * _LN2
            cdna4.buffer_store(lse, self.lse_ptr, offsets, mask=mask)

    @gluon.jit
    def store_partial(
        self, acc, l_i, m_i, split_id, num_kv_splits, mid_o_ptr, mid_lse_ptr
    ):
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
        valid = (q_pos < self.seq_len) & self.row_has_head(offs_m)
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
            (row[:, None] * cfg.N_HEADS + q_head[:, None]) * num_kv_splits + split_id
        ) * cfg.HEAD_DIM + offs_d[None, :]
        lse_off = (row * cfg.N_HEADS + q_head) * num_kv_splits + split_id
        cdna4.buffer_store(part_o, mid_o_ptr, o_off, mask=valid[:, None])
        cdna4.buffer_store(part_lse, mid_lse_ptr, lse_off, mask=valid)


@gluon.jit
def _needs_mask(cfg, start_n, window_end, mask_start):
    # Whether the page starting at key start_n holds a key that some row of
    # the tile must not see (see _stream_kv). The operands are the same for
    # every lane of the work-group, so all its waves take the same branch.
    needs = start_n + cfg.BLOCK_N > mask_start
    if cfg.WINDOW_LEFT >= 0:
        needs = needs | (start_n < window_end)
    return needs


# ===-----------------------------------------------------------------------===#
# Warp-pipelined page loop (8 waves)
#
# A work-group of 8 waves owns 256 query rows, 32 per wave, and walks the KV
# pages. Each loop iteration j finishes page j in four warp-pipeline clusters,
# as in mla/prefill_8wave.py:
#
#   load_k   read K(j + 1) from LDS; copy V(j + 2) into LDS; deferred rescale
#   qk_sum   QK MFMAs of page j + 1; exp2, row sum and P conversion of page j
#   load_v   read V(j) from LDS; copy K(j + 4) into LDS
#   pv_max   PV MFMAs of page j; row maximum and exp2 of page j + 1
#
# The two waves on a SIMD run one cluster apart, so one wave's MFMAs and
# softmax overlap the other wave's LDS reads and copy issue. The copy clusters
# run at a higher s_setprio priority, so when both waves have an instruction
# ready, the copy wave's issues first. K and V move through 4-slot LDS rings,
# K four pages ahead and V two; a copy overwrites a slot that both waves read
# in an earlier cluster. Copies past the range repeat its last page, and the
# last page is finished after the loop, so no page past the range is scored.
#
# The running maximum is updated lazily: it moves only when a page raises a
# row's maximum by more than 2**8, so exp2 of a score stays below 2**8 and most
# pages skip the accumulator rescale.
# ===-----------------------------------------------------------------------===#

# exp2(score - running maximum) stays below 2**_RESCALE_THRESHOLD, which a
# 16-bit P holds without losing relative precision.
_RESCALE_THRESHOLD = gl.constexpr(8.0)


@gluon.jit
def _rescale_accumulator(*args):
    # map_elementwise passes a lane's accumulator values, then alpha and the
    # wave's vote, each once per value. All of a lane's accumulator values
    # belong to one row, so one scalar branch skips the multiplies for a wave
    # whose rows all kept their maximum.
    n: gl.constexpr = len(args) // 3
    values = args[:n]
    if args[2 * n] == 0:
        scaled = ()
        for i in gl.static_range(n):
            scaled += (values[i] * args[n],)
        values = scaled
    return values


@gluon.jit
def _rescale(cfg, acc, l_i, m_old, m_new):
    # Apply the deferred correction from m_old to m_new.
    unchanged = _wave_maxima_unchanged(m_old, m_new)
    alpha = gl.where(unchanged == 0, gl.exp2(m_old - m_new), 1.0)
    l_i = l_i * alpha
    acc_per_lane: gl.constexpr = cfg.BLOCK_M * cfg.HEAD_DIM // (cfg.NUM_WARPS * 64)
    (acc,) = gl.map_elementwise(
        _rescale_accumulator,
        acc,
        alpha[:, None],
        unchanged[:, None],
        pack=acc_per_lane,
    )
    return acc, l_i


@gluon.jit
def _softmax_max(program, scores, m_run, start_n, kv_end, MASKED: gl.constexpr):
    # Row maximum and exponent argument of page start_n's scores in base-2
    # units, plus exp2 of their first quarter. _softmax_sum exponentiates the
    # rest, so both MFMA clusters carry vector work. Every returned value is
    # computed here, into fresh registers, so the next QK MFMA can reuse the
    # score registers.
    cfg = program.cfg
    if MASKED:
        scores = program.mask_kv(scores, start_n, kv_end)
    row_max = max(scores, 1) * cfg.SM_SCALE
    if MASKED:
        # A row with no visible key keeps a finite maximum, so exp2 of its
        # masked scores is 0 rather than NaN.
        row_max = gl.where(row_max == -float("inf"), -1.0e20, row_max)
    m_new = gl.where(row_max - m_run > _RESCALE_THRESHOLD, row_max, m_run)
    # A multiply-subtract rather than gl.fma packs into v_pk_fma_f32.
    shifted = scores * cfg.SM_SCALE - m_new[:, None]
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
    # With 16-bit P (k_width 4) the PV operand matches the score layout
    # register for register, so the conversion moves no data.
    p = gl.convert_layout(p.to(program.q_ptr.dtype.element_ty), cfg.p_layout)
    return _keep_in_cluster(p), l_i


@gluon.jit
def _pipelined_pages(
    program,
    q,
    k_smem,
    v_smem,
    tile_offsets,
    kv_start,
    kv_end,
    last_page,
    state,
    begin,
    end,
    MASKED: gl.constexpr,
):
    # Finish pages [begin, end). state carries the loop's values between
    # calls. On entry the softmax pieces (p_first, second, second_half) and
    # m_i hold page begin's softmax input, and the rescale from m_old to m_i
    # is still pending; K up to page begin + 3 and V up to begin + 1 are
    # copied, with K(begin + 1) and V(begin) visible to every wave. pages
    # holds the physical pages of begin + 2 to begin + 5. MASKED selects the
    # score mask of page j + 1, whose row maximum iteration j computes.
    cfg = program.cfg
    ring: gl.constexpr = k_smem.shape[0]
    pages, p_first, second, second_half, m_old, m_i, l_i, acc = state
    p2, p3, p4, p5 = pages
    for j in range(begin, end):
        # A wave waits only for its own copies, and the other wave on the SIMD
        # runs one cluster behind. Keeping three groups in flight completes
        # each page a full cluster before its first read, so both waves'
        # shares have landed whichever wave reads it.
        async_copy.wait_group(3)
        with gl.amd.warp_pipeline_stage("load_k", priority=1):
            # A scalar load, issued four iterations before its first use.
            p6 = program.select_page(
                program.load_page_indices(
                    kv_start + (j + 6) * cfg.BLOCK_N, last_page, 1
                ),
                0,
            )
            k = program.shared_load_k(k_smem.index((j + 1) % ring))
            program.load_page_to_shared(
                program.v_cache_ptr, p2, tile_offsets, v_smem.index((j + 2) % ring)
            )
            async_copy.commit_group()
            # Control flow stays out of the MFMA clusters: a branch there is
            # scheduled ahead of the first MFMA and stalls the matrix core.
            acc, l_i = _rescale(cfg, acc, l_i, m_old, m_i)
        with gl.amd.warp_pipeline_stage("qk_sum", priority=0):
            scores = program.compute_qk(q, k)
            p, l_i = _softmax_sum(program, p_first, second, second_half, l_i)
        async_copy.wait_group(3)
        with gl.amd.warp_pipeline_stage("load_v", priority=1):
            v = program.shared_load_v(v_smem.index(j % ring))
            program.load_page_to_shared(
                program.k_cache_ptr, p4, tile_offsets, k_smem.index((j + 4) % ring)
            )
            async_copy.commit_group()
        with gl.amd.warp_pipeline_stage("pv_max", priority=0):
            acc = program.compute_pv(p, v, acc)
            m_old = m_i
            m_i, p_first, second, second_half = _softmax_max(
                program, scores, m_i, kv_start + (j + 1) * cfg.BLOCK_N, kv_end, MASKED
            )
        p2, p3, p4, p5 = p3, p4, p5, p6
    return (p2, p3, p4, p5), p_first, second, second_half, m_old, m_i, l_i, acc


@gluon.jit
def _stream_kv_warp_pipelined(
    program,
    q,
    m_i,
    l_i,
    acc,
    kv_start,
    kv_end,
    window_end,
    mask_start,
    last_page,
    tile_offsets,
    k_smem,
    v_smem,
):
    """_stream_kv for 8-wave tiles. m_i is in raw score units on entry and
    return, as elsewhere; the loop keeps it in base-2 units."""
    cfg = program.cfg
    gl.static_assert(cfg.NUM_WARPS == 8 and cfg.BLOCK_M == 256)
    gl.static_assert(k_smem.shape[0] == 4, "the loop's copy distances need 4 slots")
    gl.static_assert(not cfg.IS_FP8, "FP8 uses the 4-wave tiles")
    count = gl.cdiv(kv_end - kv_start, cfg.BLOCK_N)
    m_i = m_i * cfg.SM_SCALE
    if count > 0:
        first = program.load_page_indices(kv_start, last_page, 4)
        # The loop's page queue, loaded before the first copies so its scalar
        # loads return while they are in flight.
        p2 = program.select_page(
            program.load_page_indices(kv_start + 2 * cfg.BLOCK_N, last_page, 1), 0
        )
        p3 = program.select_page(
            program.load_page_indices(kv_start + 3 * cfg.BLOCK_N, last_page, 1), 0
        )
        p4 = program.select_page(
            program.load_page_indices(kv_start + 4 * cfg.BLOCK_N, last_page, 1), 0
        )
        p5 = program.select_page(
            program.load_page_indices(kv_start + 5 * cfg.BLOCK_N, last_page, 1), 0
        )
        # Fill the rings in the loop's steady-state commit order: iteration j
        # commits V(j + 2) and then K(j + 4).
        page0 = program.select_page(first, 0)
        page1 = program.select_page(first, 1)
        page2 = program.select_page(first, 2)
        page3 = program.select_page(first, 3)
        k_ptr = program.k_cache_ptr
        v_ptr = program.v_cache_ptr
        program.load_page_to_shared(k_ptr, page0, tile_offsets, k_smem.index(0))
        async_copy.commit_group()
        program.load_page_to_shared(k_ptr, page1, tile_offsets, k_smem.index(1))
        async_copy.commit_group()
        program.load_page_to_shared(v_ptr, page0, tile_offsets, v_smem.index(0))
        async_copy.commit_group()
        program.load_page_to_shared(k_ptr, page2, tile_offsets, k_smem.index(2))
        async_copy.commit_group()
        program.load_page_to_shared(v_ptr, page1, tile_offsets, v_smem.index(1))
        async_copy.commit_group()
        program.load_page_to_shared(k_ptr, page3, tile_offsets, k_smem.index(3))
        async_copy.commit_group()
        # Six groups are in flight; waiting down to five retires K(0). The
        # compiler's barrier after each wait makes every wave's share of the
        # page visible before it is read.
        async_copy.wait_group(5)
        scores = program.compute_qk(q, program.shared_load_k(k_smem.index(0)))
        m_new, p_first, second, second_half = _softmax_max(
            program, scores, m_i, kv_start, kv_end, True
        )
        # Down to three retires K(1) and V(0) for the first iteration.
        async_copy.wait_group(3)
        # acc and l_i still hold no keys, so page 0 needs no rescale: the
        # pending rescale starts from m_new.
        state = ((p2, p3, p4, p5), p_first, second, second_half, m_new, m_new, l_i, acc)
        # The pages _needs_mask selects are the head_pages before the window's
        # lower edge and those from tail_page on. Iteration j scores page
        # j + 1, so each boundary moves one iteration back. The loops stop
        # before the last page, which is finished after them.
        head_pages = gl.cdiv(window_end - kv_start, cfg.BLOCK_N)
        tail_page = (mask_start - kv_start) // cfg.BLOCK_N
        head = gl.minimum(gl.maximum(head_pages - 1, 0), count - 1)
        tail = gl.minimum(gl.maximum(tail_page - 1, head), count - 1)
        args = (program, q, k_smem, v_smem, tile_offsets, kv_start, kv_end, last_page)
        if cfg.WINDOW_LEFT >= 0:
            state = _pipelined_pages(*args, state, 0, head, True)
        state = _pipelined_pages(*args, state, head, tail, False)
        state = _pipelined_pages(*args, state, tail, count - 1, True)
        _, p_first, second, second_half, m_old, m_i, l_i, acc = state
        # Finish the last page without scoring a page after it. Its V and the
        # copies past the range retire together.
        acc, l_i = _rescale(cfg, acc, l_i, m_old, m_i)
        p, l_i = _softmax_sum(program, p_first, second, second_half, l_i)
        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem.index((count - 1) % k_smem.shape[0]))
        acc = program.compute_pv(p, v, acc)
    return m_i / cfg.SM_SCALE, l_i, acc


@gluon.jit
def _stream_kv(
    program,
    q,
    m_i,
    l_i,
    acc,
    kv_start,
    kv_end,
    k_cache_ptr,
    v_cache_ptr,
    NUM_STAGES: gl.constexpr,
    ALLOW_INVALID: gl.constexpr,
):
    """Attend this tile's rows to the keys in [kv_start, kv_end).

    K/V page loads are pipelined NUM_STAGES deep: page p + NUM_STAGES - 1 is
    copied into the buffer page p - 1 freed while page p is consumed. Each
    iteration consumes NUM_STAGES pages, one per buffer. NUM_STAGES == 1, or a
    range with fewer pages than buffers, is single-buffered.

    8-wave tiles take the warp-pipelined loop instead
    (_stream_kv_warp_pipelined), with NUM_STAGES == 4 slots per ring.
    """
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
    # Keys in [window_end, mask_start) are visible to every row of the tile,
    # so pages inside that range need no mask. The causal diagonal and the
    # cache end only cut pages that end past mask_start (the first row's
    # position when causal, else the cache end). A sliding window only cuts
    # pages that start before window_end, the window's lower edge for the
    # tile's last row: a long window spans many pages, and only the pages
    # that hold some row's window edge need the mask.
    if cfg.IS_CAUSAL:
        mask_start = min(program.prefix + program.q_pos_base, kv_end)
    else:
        mask_start = kv_end
    if cfg.WINDOW_LEFT >= 0:
        window_end = (
            program.prefix + program.q_pos_base + cfg.BLOCK_Q - 1 - cfg.WINDOW_LEFT
        )
    else:
        window_end = kv_start
    last_page = (kv_end - 1) // cfg.PAGE_SIZE
    tile_offsets = program.page_tile_offsets()
    if cfg.NUM_WARPS == 8:
        return _stream_kv_warp_pipelined(
            program,
            q,
            m_i,
            l_i,
            acc,
            kv_start,
            kv_end,
            window_end,
            mask_start,
            last_page,
            tile_offsets,
            k_smem,
            v_smem,
        )
    iteration_tokens: gl.constexpr = NUM_STAGES * cfg.BLOCK_N

    if NUM_STAGES == 1 or kv_end - kv_start < iteration_tokens:
        for start_n in range(kv_start, kv_end, cfg.BLOCK_N):
            pages = program.load_page_indices(start_n, last_page, 1)
            program.issue_load_kv(
                program.select_page(pages, 0),
                tile_offsets,
                k_smem.index(0),
                v_smem.index(0),
            )
            async_copy.wait_group(1)
            k = program.shared_load_k(k_smem.index(0))
            qk = program.compute_qk(q, k)
            if _needs_mask(cfg, start_n, window_end, mask_start):
                qk = program.mask_kv(qk, start_n, kv_end)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc, ALLOW_INVALID)
            async_copy.wait_group(0)
            v = program.shared_load_v(v_smem.index(0))
            acc = program.compute_pv(p, v, acc)
        return m_i, l_i, acc

    # Prologue: prefetch the first NUM_STAGES - 1 pages.
    pages = program.load_page_indices(kv_start, last_page, NUM_STAGES)
    next_pages = program.load_page_indices(
        kv_start + iteration_tokens, last_page, NUM_STAGES
    )
    for stage in gl.static_range(NUM_STAGES - 1):
        program.issue_load_kv(
            program.select_page(pages, stage),
            tile_offsets,
            k_smem.index(stage),
            v_smem.index(stage),
        )
    for iteration_start in range(kv_start, kv_end, iteration_tokens):
        later_pages = program.load_page_indices(
            iteration_start + 2 * iteration_tokens, last_page, NUM_STAGES
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
            # The last iteration can run past kv_end: its prefetches repeat
            # the last page, and the pages past kv_end are not consumed.
            if start_n < kv_end:
                # Wait for this page's K; its V and the K and V groups of the
                # NUM_STAGES - 1 prefetched pages stay outstanding.
                async_copy.wait_group(2 * NUM_STAGES - 1)
                k = program.shared_load_k(k_smem.index(stage))
                qk = program.compute_qk(q, k)
                if _needs_mask(cfg, start_n, window_end, mask_start):
                    qk = program.mask_kv(qk, start_n, kv_end)
                p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc, ALLOW_INVALID)
                async_copy.wait_group(2 * NUM_STAGES - 2)
                v = program.shared_load_v(v_smem.index(stage))
                acc = program.compute_pv(p, v, acc)
        pages = next_pages
        next_pages = later_pages
    # Epilogue: wait for the prefetches past kv_end before LDS is reused.
    async_copy.wait_group(0)
    return m_i, l_i, acc


# The request count follows the batch. Without do_not_specialize, Triton would
# compile separate binaries for a count of 1 and for multiples of 16.
@gluon.jit(do_not_specialize=("num_seqs",))
def gluon_mha_extend_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
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
    NUM_STAGES: gl.constexpr,
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

    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_attention_state()

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

    m_i, l_i, acc = _stream_kv(
        program,
        q,
        m_i,
        l_i,
        acc,
        kv_start,
        kv_end,
        k_cache_ptr,
        v_cache_ptr,
        NUM_STAGES,
        False,
    )

    l_i = program.apply_sinks(l_i, m_i, sink_log2)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)
    program.store_lse(l_i, m_i)


@gluon.jit(do_not_specialize=("num_seqs",))
def gluon_mha_extend_split_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
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
    # Picked per batch by select_kv_splits; runtime so every batch shape shares
    # one binary.
    NUM_KV_SPLITS,
    IS_CAUSAL: gl.constexpr,
    IS_FP8: gl.constexpr,
    RAGGED: gl.constexpr,
    NUM_STAGES: gl.constexpr,
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
    pages_per_split = gl.cdiv(num_pages, NUM_KV_SPLITS)
    split_start_page = split_id * pages_per_split
    split_end_page = min(split_start_page + pages_per_split, num_pages)
    split_start = split_start_page * cfg.PAGE_SIZE
    split_end = min(split_end_page * cfg.PAGE_SIZE, kv_end)

    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_attention_state()
    # allow_invalid: a whole split can sit above a row's causal diagonal.
    m_i, l_i, acc = _stream_kv(
        program,
        q,
        m_i,
        l_i,
        acc,
        split_start,
        split_end,
        k_cache_ptr,
        v_cache_ptr,
        NUM_STAGES,
        True,
    )

    program.store_partial(
        acc, l_i, m_i, split_id, NUM_KV_SPLITS, mid_o_ptr, mid_lse_ptr
    )


@gluon.jit
def gluon_mha_extend_reduce_gfx950(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    lse_out_ptr,
    sink_ptr,
    NUM_KV_SPLITS,
    N_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
):
    # Combine the NUM_KV_SPLITS partials for one (query token, head) with a
    # global softmax rescale. Empty splits carry -inf lse (written by the
    # compute pass) so they drop out. Grid is (total_q, N_HEADS).
    #
    # NUM_KV_SPLITS is a runtime value so one binary serves every batch shape.
    # Pass 1 reads all split LSEs (one per lane) to get the global max and
    # denominator; pass 2 streams the partial outputs in _SPLIT_CHUNK-row tiles,
    # so registers stay bounded by the chunk rather than the largest split
    # count.
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [1, HEAD_DIM // 64], [1, 64], [1, 1], [1, 0]
    )
    lse_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    row = gl.program_id(0)
    q_head = gl.program_id(1)
    base = (row * N_HEADS + q_head) * NUM_KV_SPLITS

    offs_l = gl.arange(0, _LSE_TILE, layout=lse_layout)
    all_lse = gl.load(
        mid_lse_ptr + base + offs_l,
        mask=offs_l < NUM_KV_SPLITS,
        other=-float("inf"),
    )
    m_i = max(all_lse, axis=0)
    if HAS_SINK:
        sink = gl.load(sink_ptr + q_head).to(gl.float32) * _INV_LN2
        m_i = maximum(m_i, sink)
    l_i = gl.sum(gl.exp2(all_lse - m_i), axis=0)
    if HAS_SINK:
        l_i = l_i + gl.exp2(sink - m_i)

    offs_c = gl.arange(0, _SPLIT_CHUNK, layout=gl.SliceLayout(1, reduce_layout))
    offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, reduce_layout))
    acc = gl.zeros([HEAD_DIM], gl.float32, layout=gl.SliceLayout(0, reduce_layout))
    for chunk_start in range(0, NUM_KV_SPLITS, _SPLIT_CHUNK):
        offs_s = chunk_start + offs_c
        split_valid = offs_s < NUM_KV_SPLITS
        part_lse = gl.load(
            mid_lse_ptr + base + offs_s, mask=split_valid, other=-float("inf")
        )
        o_off = (base + offs_s)[:, None] * HEAD_DIM + offs_d[None, :]
        part_o = cdna4.buffer_load(
            mid_o_ptr, o_off, mask=split_valid[:, None], other=0.0
        )
        acc += gl.sum(part_o * gl.exp2(part_lse - m_i)[:, None], axis=0)

    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)
    output = output.to(out_ptr.dtype.element_ty)
    out_base = (row * N_HEADS + q_head) * HEAD_DIM
    cdna4.buffer_store(output, out_ptr, out_base + offs_d)
    if HAS_LSE:
        lse = (m_i + gl.log2(denom)) * _LN2
        gl.store(lse_out_ptr + (row * N_HEADS + q_head), lse)


_GFX950_SM_COUNT = 256
# Pipeline depth of the extend K/V page loads: LDS buffers per workgroup, so
# NUM_STAGES - 1 pages are prefetched while one is consumed.
_EXTEND_STAGES = 4
# The pipeline's K and V buffers may take at most 64 KB of the CU's 160 KB of
# LDS, so two workgroups stay resident: head dim 64 at 16 bits fits; 16-bit
# head dim 128 (128 KB) would leave one workgroup per CU and stays
# single-buffered.
_EXTEND_PIPELINE_MAX_LDS_BYTES = 64 * 1024
# Resident workgroups per CU at head dim 64 (4-wave, 128-row tile): two
# pipelined (LDS-bound); single-buffered, three for the unsplit kernel and two
# for the split kernel. VGPRs are allocated in groups of 8 (CDNA4 ISA 3.6.4):
# the unsplit kernel's 167 round to 168, so three waves fit a SIMD's 512; the
# split kernel's 169 round to 176. MI355X, 16 query / 2 KV heads over a 131k
# prefix: unsplit grids of 640 and 768 workgroups ran 5% faster
# single-buffered, 512 and 1024 7% faster pipelined.
_EXTEND_PIPELINED_WORKGROUPS_PER_CU = 2
_EXTEND_SINGLE_WORKGROUPS_PER_CU = 3


def _extend_stage_choices(
    *, num_warps: int, head_dim: int, page_size: int, element_size: int
) -> tuple[int, ...]:
    """Pipeline depths _extend_stages picks from, single-buffered first."""
    if num_warps == _WARP_PIPELINE_NUM_WARPS:
        # The warp-pipelined loop always streams through 4-slot rings.
        return (_EXTEND_STAGES,)
    pipeline_bytes = 2 * _EXTEND_STAGES * page_size * head_dim * element_size
    if pipeline_bytes > _EXTEND_PIPELINE_MAX_LDS_BYTES:
        return (1,)
    return (1, _EXTEND_STAGES)


def _extend_stages(
    *,
    num_workgroups: int,
    split: bool,
    num_warps: int,
    head_dim: int,
    page_size: int,
    element_size: int,
) -> int:
    """Pipeline depth of the K/V page loads for an extend launch.

    Pipelining hides load latency but can lower occupancy, so an unsplit
    launch of num_workgroups workgroups pipelines only when the grid then needs
    no more rounds of resident workgroups than single-buffered.
    """
    choices = _extend_stage_choices(
        num_warps=num_warps,
        head_dim=head_dim,
        page_size=page_size,
        element_size=element_size,
    )
    if split or len(choices) == 1:
        return choices[-1]
    pipelined_rounds = triton.cdiv(
        num_workgroups, _EXTEND_PIPELINED_WORKGROUPS_PER_CU * _GFX950_SM_COUNT
    )
    single_rounds = triton.cdiv(
        num_workgroups, _EXTEND_SINGLE_WORKGROUPS_PER_CU * _GFX950_SM_COUNT
    )
    return choices[-1] if pipelined_rounds <= single_rounds else choices[0]


def launch_gluon_mha_extend_gfx950(
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
    return _extend(
        q,
        cu_seqlens_q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        is_causal=is_causal,
        window_left=window_left,
        sinks=sinks,
        return_lse=return_lse,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        compile_all_variants=False,
    )


def warmup_gluon_mha_extend_gfx950(
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
) -> None:
    """Compile every kernel variant the extend launcher can pick, without running.

    Takes the launcher's arguments. The query tile, the K/V pipeline depth,
    the grid, whether full attention splits the KV stream and whether the
    split count is a multiple of 16 follow the batch; each choice is a
    separate binary. This compiles all of them for the model's heads, dtypes, window,
    sinks and LSE output, and for the tensors' strides and alignment, so a
    server can compile them at startup. The batch shape of the arguments
    (lengths, request count) does not matter.
    """
    _extend(
        q,
        cu_seqlens_q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        is_causal=is_causal,
        window_left=window_left,
        sinks=sinks,
        return_lse=return_lse,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        compile_all_variants=True,
    )


def _extend(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    is_causal: bool,
    window_left: int,
    sinks: torch.Tensor | None,
    return_lse: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    compile_all_variants: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    # Picks the variant for this batch and launches it, or, with
    # compile_all_variants, compiles every variant any batch can pick.
    head_dim = q.shape[2]
    n_heads = q.shape[1]
    n_kv_heads = k_cache.shape[2]
    group_size = n_heads // n_kv_heads
    page_size = k_cache.shape[1]
    total_q = q.shape[0]
    warp_pipeline = _use_warp_pipeline(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        total_q=total_q,
        group_size=group_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        element_size=k_cache.element_size(),
        window_left=window_left,
    )
    tile = _select_extend_tile(max_seqlen_q, group_size, warp_pipeline)
    block_q = tile[1]
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
            # An 8-wave work-group takes a whole CU, where two 4-wave ones
            # fit, so it counts twice toward filling the machine.
            base_ctas=grid_num_q_blocks * n_kv_heads * (2 if warp_pipeline else 1),
            num_pages=num_pages,
            sm_count=_GFX950_SM_COUNT,
        )

    # Bound on q-blocks with query rows: the dense count plus one partial
    # block for each further request.
    num_q_blocks = min(rect_num_q_blocks, triton.cdiv(total_q, block_q) + batch - 1)
    stage_args = dict(
        head_dim=head_dim, page_size=page_size, element_size=k_cache.element_size()
    )
    num_stages = _extend_stages(
        num_workgroups=num_q_blocks * n_kv_heads,
        split=num_kv_splits > 1,
        num_warps=tile[3],
        **stage_args,
    )

    def split_args(tile, num_stages, ragged, num_kv_splits, mid_o, mid_lse):
        block_m, block_q, block_n, num_warps = tile
        return (
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
            ragged,
            num_stages,
        )

    def reduce_args(num_kv_splits, mid_o, mid_lse):
        return (
            mid_o,
            mid_lse,
            output,
            lse_arg,
            sink_arg,
            num_kv_splits,
            n_heads,
            head_dim,
            has_sink,
            return_lse,
        )

    def unsplit_args(tile, num_stages, ragged):
        block_m, block_q, block_n, num_warps = tile
        return (
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
            ragged,
            num_stages,
        )

    def split_buffers(rows, num_kv_splits):
        mid_o = torch.empty(
            (rows, n_heads, num_kv_splits, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        mid_lse = torch.empty(
            (rows, n_heads, num_kv_splits),
            device=q.device,
            dtype=torch.float32,
        )
        return mid_o, mid_lse

    if compile_all_variants:
        # Every tile _select_extend_tile can return, with every pipeline depth
        # _extend_stages can pick for it, on both grids. Triton also compiles
        # an integer argument separately when it is a multiple of 16, so the
        # split kernels take one split count of each kind select_kv_splits
        # can return. The request count is not specialized.
        pipeline_supported = _warp_pipeline_supported(
            head_dim=head_dim,
            element_size=k_cache.element_size(),
            window_left=window_left,
        )
        tiles = {
            _extend_tile(m, w, group_size)
            for m, w in _EXTEND_TILES
            if w != _WARP_PIPELINE_NUM_WARPS or pipeline_supported
        }
        split_counts = {n % 16 == 0: n for n in range(2, MAX_KV_SPLITS + 1)}
        # Compiling needs only their dtype and alignment, not their size.
        mid_o, mid_lse = split_buffers(1, MAX_KV_SPLITS)
        for tile in sorted(tiles):
            options = dict(grid=(1,), num_warps=tile[3])
            stage_choices = _extend_stage_choices(num_warps=tile[3], **stage_args)
            split_stages = _extend_stages(
                num_workgroups=1, split=True, num_warps=tile[3], **stage_args
            )
            for ragged in (False, True):
                for stages in stage_choices:
                    gluon_mha_extend_gfx950.warmup(
                        *unsplit_args(tile, stages, ragged), **options
                    )
                if is_sliding:
                    continue
                for splits in split_counts.values():
                    gluon_mha_extend_split_gfx950.warmup(
                        *split_args(tile, split_stages, ragged, splits, mid_o, mid_lse),
                        **options,
                    )
        if not is_sliding:
            for splits in split_counts.values():
                gluon_mha_extend_reduce_gfx950.warmup(
                    *reduce_args(splits, mid_o, mid_lse), grid=(1,), num_warps=1
                )
        return None

    num_warps = tile[3]
    if num_kv_splits > 1:
        mid_o, mid_lse = split_buffers(total_q, num_kv_splits)
        if use_ragged:
            split_grid = (ragged_num_q_blocks, n_kv_heads, num_kv_splits)
        else:
            split_grid = (blocks_per_req * num_kv_splits, batch, n_kv_heads)
        gluon_mha_extend_split_gfx950[split_grid](
            *split_args(tile, num_stages, use_ragged, num_kv_splits, mid_o, mid_lse),
            num_warps=num_warps,
        )
        reduce_grid = (total_q, n_heads)
        gluon_mha_extend_reduce_gfx950[reduce_grid](
            *reduce_args(num_kv_splits, mid_o, mid_lse),
            num_warps=1,
        )
        if return_lse:
            return output, lse
        return output

    if use_ragged:
        grid = (ragged_num_q_blocks, n_kv_heads)
    else:
        grid = (blocks_per_req, batch, n_kv_heads)
    gluon_mha_extend_gfx950[grid](
        *unsplit_args(tile, num_stages, use_ragged),
        num_warps=num_warps,
    )
    if return_lse:
        return output, lse
    return output
