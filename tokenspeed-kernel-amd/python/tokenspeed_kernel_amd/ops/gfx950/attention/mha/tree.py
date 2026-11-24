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

# Window keys per mask word, and the most words a query row takes.
_MASK_WORD_BITS = 64
_MAX_TREE_WORDS = 2


def _tree_row_tile(rows_per_req: int, group_size: int) -> tuple[int, int]:
    """Return (BLOCK_M, NUM_WARPS) for a request's rows of one KV head.

    Up to 64 rows take the decode's tile, one 16-row MFMA tile per wave. More
    rows (target verify: 16 nodes x 6 query heads) need several work-groups
    per KV head, each reading every K/V page, and each wave reads a whole page
    from LDS (and widens an FP8 one) for its rows. Two MFMA tiles per wave
    halve that per row: on the Qwen3.8-27B verify shape a 64-row two-wave tile
    beats one 16-row tile per wave by 20-35% (BF16) and 25-35% (FP8 E4M3).
    """
    rows = rows_per_req * group_size
    if rows <= _DECODE_MAX_BLOCK_M:
        return _row_tile(rows_per_req, group_size)
    return _DECODE_MAX_BLOCK_M, _DECODE_MAX_BLOCK_M // 32


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
    block_m, num_warps = _tree_row_tile(rows_per_req, group_size)
    num_row_blocks = triton.cdiv(rows_per_req * group_size, block_m)
    # A sliding window ends at the prefix end, so a request reads at most
    # that window and the tree window, plus a partial leading page.
    keys = min(max_seqlen_k, window_left + 1 + window) if is_sliding else max_seqlen_k
    num_pages = triton.cdiv(keys, page_size) + int(is_sliding)
    num_kv_splits = select_kv_splits(
        base_ctas=num_requests * num_kv_heads * num_row_blocks,
        num_pages=num_pages,
        sm_count=_GFX950_SM_COUNT,
        max_splits=_FULL_DECODE_MAX_SPLITS,
    )
    if block_m // num_warps > 16:
        # Two-tile waves hold twice the accumulators; the pipeline's LDS
        # buffers then cost them more occupancy than the loads they hide.
        num_stages = 1
    else:
        num_stages = _full_decode_stages(
            num_kv_splits, page_size, head_dim, k_cache.element_size(), num_warps
        )
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
