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

"""Draft-tree MHA decode Gluon kernel for AMD GFX950.

The split decode of ``decode.py`` with a tree window (``TREE_WINDOW``): each
request's last ``window`` keys are its draft-tree window, and query position
``r`` sees every key before the window (or its sliding window ending there)
and window key ``j`` when bit ``j % 64`` of its mask word ``j // 64`` is set.
One pass over the keys serves every row, so the cascade's causal prefix
decode, its log-sum-exp and the window merge become one kernel and the split
reduce.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import (
    _INV_LN2_VALUE,
    InputStrides,
    select_kv_splits,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    _DECODE_MAX_BLOCK_M,
    _FULL_DECODE_MAX_SPLITS,
    _GFX950_SM_COUNT,
    AttentionConfig,
    AttentionProgram,
    _full_decode_stages,
    _row_tile,
    gluon_mha_decode_reduce_gfx950,
    stream_split_pipelined,
    stream_split_single_buffer,
)

async_copy = gl.amd.cdna4.async_copy

# Window keys per mask word, and the most words a query row takes.
_MASK_WORD_BITS = 64
_MAX_TREE_WORDS = 2


# Requests x KV heads from which a wide (> 64-row) tree tile picks its row tile
# by cache dtype and sizes its splits by the work-groups per CU below.
_TREE_WIDE_MIN_HEADS = 32
# Work-groups per CU of the wide tiles' grid; deeper splits cost the reduce
# more than they gain once every CU holds this many.
_TREE_WIDE_BF16_CTAS_PER_CU = 1
_TREE_WIDE_FP8_CTAS_PER_CU = 3


def _tree_launch_config(
    rows_per_req: int,
    group_size: int,
    request_heads: int,
    kv_bytes: int,
    num_pages: int,
    page_size: int,
    head_dim: int,
) -> tuple[int, int, int, int, bool]:
    """Return (BLOCK_M, NUM_WARPS, splits, NUM_STAGES, REGISTER_PREFETCH).

    Up to 64 rows of a KV head take the decode's tile, splits and pipeline.
    More rows (target verify: 16 nodes x 6 query heads) need tiles where each
    wave reads a whole K/V page from LDS into registers (and widens an FP8
    one), so every wave holds about 500 registers and a CU runs one wave per
    SIMD. Those tiles read each page into one LDS buffer and issue the next
    page's copy as soon as a wave has read this one (REGISTER_PREFETCH).

    At 32 or more requests x KV heads the tile and split count follow the
    cache dtype. A 16-bit page fills twice the LDS of an FP8 one, so BF16 runs
    four-wave 128-row work-groups, one per CU, while FP8 runs one-wave 32-row
    work-groups (no padding rows for 96 rows), three per CU. On the
    Qwen3.8-27B verify shape at 32 requests and 8k keys this takes the BF16
    kernel from 285 to 247 us and FP8 from 297 to 228 us. Fewer heads keep a
    64-row two-wave tile with the decode's split rule, which splits deeper.
    """
    rows = rows_per_req * group_size
    if rows <= _DECODE_MAX_BLOCK_M:
        block_m, num_warps = _row_tile(rows_per_req, group_size)
        splits = select_kv_splits(
            base_ctas=request_heads * triton.cdiv(rows, block_m),
            num_pages=num_pages,
            sm_count=_GFX950_SM_COUNT,
            max_splits=_FULL_DECODE_MAX_SPLITS,
        )
        stages = _full_decode_stages(splits, page_size, head_dim, kv_bytes, num_warps)
        return block_m, num_warps, splits, stages, False
    if request_heads < _TREE_WIDE_MIN_HEADS:
        block_m, num_warps = _DECODE_MAX_BLOCK_M, 2
        splits = select_kv_splits(
            base_ctas=request_heads * triton.cdiv(rows, block_m),
            num_pages=num_pages,
            sm_count=_GFX950_SM_COUNT,
            max_splits=_FULL_DECODE_MAX_SPLITS,
        )
        return block_m, num_warps, splits, 1, True
    if kv_bytes == 1:
        block_m, num_warps, ctas_per_cu = 32, 1, _TREE_WIDE_FP8_CTAS_PER_CU
    else:
        block_m, num_warps, ctas_per_cu = 128, 4, _TREE_WIDE_BF16_CTAS_PER_CU
    base_ctas = request_heads * triton.cdiv(rows, block_m)
    splits = triton.cdiv(_GFX950_SM_COUNT * ctas_per_cu, base_ctas)
    splits = max(1, min(splits, num_pages, _FULL_DECODE_MAX_SPLITS))
    return block_m, num_warps, splits, 1, True


@gluon.jit
def stream_split_register_prefetch(program, q, m_i, l_i, acc, k_smem, v_smem):
    """Single-buffered page loop that loads the next page while this one is
    consumed.

    Each wave reads a whole K (then V) page tile from LDS into registers
    before its MFMAs, so the buffer is free as soon as that read is done: the
    next page's K copy is issued right after K is read, before QK and the
    softmax, and its V copy right after V is read, before PV. A hd256 page
    leaves room for one K and one V buffer per work-group at the occupancy
    this tile runs at, so this hides the copy latency that
    stream_split_single_buffer leaves exposed on every page.
    """
    cfg = program.cfg
    physical_page = program.load_page(program.split_start)
    program.issue_load_k(physical_page, k_smem)
    program.issue_load_v(physical_page, v_smem)
    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        next_page = program.load_page(start_n + cfg.BLOCK_N)
        # Outstanding: this page's K, then its V.
        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        program.issue_load_k(next_page, k_smem)
        qk = program.compute_qk(q, k)
        if cfg.IS_SLIDING or start_n + cfg.BLOCK_N > program.mask_start:
            qk = program.apply_kv_mask(qk, start_n)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)
        # Outstanding: this page's V, then the next page's K.
        async_copy.wait_group(1)
        v = program.clean_tree_window_v(program.shared_load_v(v_smem), start_n)
        program.issue_load_v(next_page, v_smem)
        acc = program.compute_pv(p, v, acc)
    # The copies past the split (of the page after it, or of page 0) are still
    # writing into LDS that the epilogue's layout conversions reuse.
    async_copy.wait_group(0)
    return m_i, l_i, acc


@gluon.jit
def gluon_mha_tree_decode_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    seq_lens_ptr,
    mask_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    ROWS: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    TREE_WINDOW: gl.constexpr,
    TREE_WORDS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    REGISTER_PREFETCH: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        ROWS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        IS_SLIDING,
        WINDOW_LEFT,
        IS_FP8,
        False,  # UNIFORM_ROWS
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
        TREE_WINDOW,
        TREE_WORDS,
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        seq_lens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        gl.program_id(0),
        gl.program_id(1),
        gl.program_id(2),
        mask_ptr,
    )
    q = program.load_q()
    m_i, l_i, acc, _ = program.init_state(q_ptr, False)
    if NUM_STAGES == 1:
        k_smem = gl.allocate_shared_memory(
            k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
        )
        v_smem = gl.allocate_shared_memory(
            v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
        )
        if REGISTER_PREFETCH:
            m_i, l_i, acc = stream_split_register_prefetch(
                program, q, m_i, l_i, acc, k_smem, v_smem
            )
        else:
            m_i, l_i, acc = stream_split_single_buffer(
                program, q, m_i, l_i, acc, k_smem, v_smem
            )
    else:
        m_i, l_i, acc = stream_split_pipelined(
            program, q, m_i, l_i, acc, k_cache_ptr, v_cache_ptr, NUM_STAGES
        )
    program.store_split(acc, l_i, m_i)


def _check_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    mask: torch.Tensor,
    rows_per_req: int,
    window: int,
) -> int:
    """Validate a tree decode call; return its request count."""
    if k_cache.dtype != v_cache.dtype:
        raise ValueError(f"K/V cache dtypes differ: {k_cache.dtype} vs {v_cache.dtype}")
    widened = q.dtype in (torch.float16, torch.bfloat16) and (
        k_cache.dtype == torch.float8_e4m3fn
    )
    if q.dtype != k_cache.dtype and not widened:
        raise ValueError(
            f"gluon_mha_tree_decode_gfx950 reads {q.dtype} queries over a "
            f"{k_cache.dtype} cache; it takes matching dtypes or 16-bit queries "
            "over unit-scale FP8 E4M3"
        )
    if rows_per_req <= 0 or q.shape[0] % rows_per_req:
        raise ValueError(f"{q.shape[0]} query rows are not whole {rows_per_req=}")
    num_requests = q.shape[0] // rows_per_req
    words = triton.cdiv(window, _MASK_WORD_BITS)
    if not 0 < words <= _MAX_TREE_WORDS:
        raise ValueError(
            f"a {window}-key tree window needs 1 to {_MAX_TREE_WORDS} mask words"
        )
    if mask.dtype != torch.int64 or mask.numel() != q.shape[0] * words:
        raise ValueError(
            f"the tree mask takes {words} int64 words per query row, got "
            f"{mask.numel()} {mask.dtype} for {q.shape[0]} rows"
        )
    if seq_lens.shape[0] != num_requests:
        raise ValueError(f"{seq_lens.shape[0]} seq_lens for {num_requests} requests")
    return num_requests


def launch_gluon_mha_tree_decode_gfx950(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    mask: torch.Tensor,
    *,
    rows_per_req: int,
    window: int,
    max_seqlen_k: int,
    window_left: int = -1,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Draft-tree attention over a paged cache, on the split grid.

    ``q`` is ``[requests * rows_per_req, num_q_heads, head_dim]``; ``seq_lens``
    counts each request's keys with its ``window`` tree keys; ``mask`` holds
    ``cdiv(window, 64)`` int64 words per query row. A 16-bit query reads an
    FP8 E4M3 cache at unit scale, widening each K/V tile after the LDS read.
    """
    num_requests = _check_inputs(
        q, k_cache, v_cache, seq_lens, mask, rows_per_req, window
    )
    num_q_heads, head_dim = q.shape[1], q.shape[2]
    page_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads
    is_sliding = window_left >= 0
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    # A sliding window ends at the prefix end, so a request reads at most
    # that window and the tree window, plus a partial leading page.
    keys = min(max_seqlen_k, window_left + 1 + window) if is_sliding else max_seqlen_k
    num_pages = triton.cdiv(keys, page_size) + int(is_sliding)
    block_m, num_warps, num_kv_splits, num_stages, register_prefetch = (
        _tree_launch_config(
            rows_per_req,
            group_size,
            num_requests * num_kv_heads,
            k_cache.element_size(),
            num_pages,
            page_size,
            head_dim,
        )
    )
    num_row_blocks = triton.cdiv(rows_per_req * group_size, block_m)
    is_fp8 = q.element_size() == 1
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    output = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    if num_requests == 0:
        return output
    total_q = q.shape[0]
    mid_o = torch.empty(
        (total_q, num_q_heads, num_kv_splits, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    mid_lse = torch.empty(
        (total_q, num_q_heads, num_kv_splits), device=q.device, dtype=torch.float32
    )
    sm_scale = softmax_scale * _INV_LN2_VALUE
    grid = (num_requests, num_kv_heads * num_row_blocks, num_kv_splits)
    gluon_mha_tree_decode_gfx950[grid](
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        mask,
        mid_o,
        mid_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sm_scale,
        page_table.stride(0),
        page_size,
        num_kv_splits,
        rows_per_req,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_m,
        page_size,  # BLOCK_N
        num_warps,
        is_sliding,
        window_left if is_sliding else -1,
        is_fp8,
        window,
        triton.cdiv(window, _MASK_WORD_BITS),
        num_stages,
        register_prefetch,
        num_warps=num_warps,
        **extern_compile_options(),
    )
    gluon_mha_decode_reduce_gfx950[(total_q, num_q_heads)](
        mid_o,
        mid_lse,
        output,
        q,  # sink_ptr (unread)
        q,  # gate_ptr (unread)
        0,
        k_cache,
        sm_scale,
        page_table.stride(0),
        num_kv_splits,
        rows_per_req,
        page_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,  # BLOCK_N
        False,  # HAS_SINK
        is_fp8,
        False,  # UNIFORM_ROWS
        0.0,  # GATE_EPS
        False,  # HAS_GATE
        True,  # EMPTY_ROWS
        num_warps=1,
    )
    return output
