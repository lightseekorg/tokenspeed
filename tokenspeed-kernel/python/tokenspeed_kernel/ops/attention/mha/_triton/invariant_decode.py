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

"""Batch-invariant MHA decode.

The adaptive decode in ``decode.py`` splits a request's keys by batch size and
SM count, so the order key blocks fold into a row's sum depends on the batch.
This kernel folds each row's keys in index order in one program, with no split
dimension: a row's bits depend only on its own query, keys and window.

The grid is ``bs * kv_heads`` programs, so a small batch underfills the device.
Time against the adaptive leaf on H200 (GQA 32/8, head_dim 128, BF16, CUDA graph
replay): block draft with a 1024 window 1.75x at batch 1 and 0.90-0.95x from
batch 8; no window over 4K-128K keys 5-12x at batch 1 and parity from batch 64.
"""

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.mha._triton.decode import _folded_tiles
from tokenspeed_kernel.platform import current_platform


@triton.jit
def _invariant_decode_kernel(
    Q,
    K_Buffer,
    V_Buffer,
    O,
    Lse,
    sink_ptr,
    sm_scale_log2,
    page_table,
    cache_seqlens,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_buf_kp,
    stride_buf_vp,
    stride_obs,
    stride_oh,
    stride_lse_b,
    stride_lse_h,
    page_table_stride_b,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_SINK: tl.constexpr,
    kv_group_num: tl.constexpr,
    ROW_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Attend one block of folded rows ``(query position, head of the KV
    group)`` of one request over all of its visible keys.

    Query position ``t`` of ``MAX_SEQLEN_Q`` sees the request's first
    ``seq_len - (MAX_SEQLEN_Q - 1 - t)`` keys, the last ``WINDOW_LEFT + 1`` of
    them under a sliding window plus the leading ``WINDOW_SINK`` retained
    ones. Keys fold in index order in one pass, so the result depends on the
    row alone. A row with no key in reach gets a zero output and a ``-inf``
    log-sum-exp.
    """
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1) // ROW_BLOCKS
    row_block = tl.program_id(1) % ROW_BLOCKS

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < MAX_SEQLEN_Q * kv_group_num
    q_pos = rows // kv_group_num
    cur_head = cur_kv_head * kv_group_num + rows % kv_group_num
    # Query rows of one request are contiguous; the token index scales with
    # the batch, so it is computed in int64.
    token = cur_batch.to(tl.int64) * MAX_SEQLEN_Q + q_pos

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    seq_len = tl.load(cache_seqlens + cur_batch)
    # The table row's offset scales with the batch: advance its base in int64.
    table_row = page_table + cur_batch.to(tl.int64) * page_table_stride_b
    row_end = tl.maximum(seq_len - (MAX_SEQLEN_Q - 1 - q_pos), 0)
    # The whole request's key range: from the first row's window start. Every
    # row of the block walks it and masks the keys outside its own window, so
    # each row's surviving keys fold in the same order whatever the block
    # holds.
    first_end = tl.maximum(seq_len - (MAX_SEQLEN_Q - 1), 0)
    if WINDOW_LEFT >= 0:
        row_start = tl.maximum(row_end - (WINDOW_LEFT + 1), 0)
        kv_lo = tl.maximum(first_end - (WINDOW_LEFT + 1), 0)
    else:
        row_start = tl.zeros([BLOCK_M], dtype=tl.int32)
        kv_lo = 0
    sink_len = tl.minimum(WINDOW_SINK, kv_lo)
    visible_len = sink_len + seq_len - kv_lo

    q = tl.load(
        Q
        + token[:, None] * stride_qbs
        + cur_head[:, None] * stride_qh
        + offs_d[None, :],
        mask=row_ok[:, None] & mask_d[None, :],
        other=0.0,
    )
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    for start_n in range(0, visible_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_ok = offs_n < visible_len
        # The retained sink tokens come first, then the window's own range.
        token_indices = tl.where(offs_n < sink_len, offs_n, kv_lo + offs_n - sink_len)
        page_indices = token_indices // PAGE_SIZE
        physical_pages = tl.load(table_row + page_indices, mask=key_ok, other=0)
        # int64 to avoid address overflow for large (>int32 range) KV caches
        k_loc = (
            physical_pages.to(tl.int64) * stride_buf_kp
            + (token_indices % PAGE_SIZE) * stride_buf_kbs
        )
        v_loc = (
            physical_pages.to(tl.int64) * stride_buf_vp
            + (token_indices % PAGE_SIZE) * stride_buf_vbs
        )
        k = tl.load(
            K_Buffer + k_loc[:, None] + cur_kv_head * stride_buf_kh + offs_d[None, :],
            mask=key_ok[:, None] & mask_d[None, :],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k.to(q.dtype)))
        qk = qk * sm_scale_log2
        visible = (
            row_ok[:, None]
            & key_ok[None, :]
            & (
                (token_indices[None, :] >= row_start[:, None])
                | (token_indices[None, :] < WINDOW_SINK)
            )
            & (token_indices[None, :] < row_end[:, None])
        )
        qk = tl.where(visible, qk, float("-inf"))
        v = tl.load(
            V_Buffer + v_loc[:, None] + cur_kv_head * stride_buf_vh + offs_dv[None, :],
            mask=key_ok[:, None] & mask_dv[None, :],
            other=0.0,
        )
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        # A row that has seen no key yet keeps a -inf max; rescale from 0.
        safe_max = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
        re_scale = tl.exp2(e_max - safe_max)
        p = tl.exp2(qk - safe_max[:, None])
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    if HAS_SINK:
        # Sinks are natural-log logits; this block's scale is base 2.
        cur_sink = tl.load(sink_ptr + cur_head, mask=row_ok, other=0.0).to(tl.float32)
        cur_sink = cur_sink * 1.4426950408889634
        sink_max = tl.maximum(e_max, cur_sink)
        safe_sink_max = tl.where(sink_max == float("-inf"), 0.0, sink_max)
        scale = tl.exp2(tl.where(e_max == float("-inf"), 0.0, e_max) - safe_sink_max)
        acc = acc * tl.where(e_max == float("-inf"), 0.0, scale)[:, None]
        e_sum = e_sum * scale + tl.exp2(cur_sink - safe_sink_max)
        e_max = sink_max

    has_keys = e_sum > 0.0
    tl.store(
        O
        + token[:, None] * stride_obs
        + cur_head[:, None] * stride_oh
        + offs_dv[None, :],
        tl.where(has_keys[:, None], acc / tl.where(has_keys, e_sum, 1.0)[:, None], 0.0),
        mask=row_ok[:, None] & mask_dv[None, :],
    )
    if RETURN_LSE:
        tl.store(
            Lse + token * stride_lse_b + cur_head * stride_lse_h,
            tl.where(
                has_keys,
                (e_max + tl.log2(tl.where(has_keys, e_sum, 1.0))) * 0.6931471805599453,
                -float("inf"),
            ),
            mask=row_ok,
        )


def invariant_decode_attention_fwd(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    page_size: int,
    window_left: int,
    window_sink: int,
    sm_scale: float,
    sinks: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
) -> None:
    """Decode ``max_seqlen_q`` query rows per request, each row's keys folded
    in index order by one program.

    Args:
        q: ``[bs * max_seqlen_q, num_q_heads, Lk]`` queries.
        k_buffer: ``[slots, num_kv_heads, Lk]`` token rows of the paged cache.
        v_buffer: ``[slots, num_kv_heads, Lv]`` value rows.
        o: ``[bs * max_seqlen_q, num_q_heads, Lv]`` output, written in place.
        page_table: ``[bs, max_pages]`` int32 page ids.
        cache_seqlens: ``[bs]`` int32 keys per request, the query rows included.
        max_seqlen_q: query rows per request.
        page_size: tokens per page.
        window_left: exclusive left sliding window, ``-1`` for full attention.
        window_sink: leading keys retained outside the window.
        sm_scale: softmax scale applied to ``q . k``.
        sinks: ``[num_q_heads]`` attention sink logits, or ``None``.
        lse_out: ``[bs * max_seqlen_q, num_q_heads]`` float32 natural-log
            log-sum-exp output, or ``None``. Undefined with ``sinks``.
    """
    if sinks is not None and lse_out is not None:
        raise ValueError("the log-sum-exp of a decode with sinks is undefined")
    bs = cache_seqlens.shape[0]
    total_q, head_num, lk = q.shape
    lv = v_buffer.shape[-1]
    kv_head_num = k_buffer.shape[-2]
    kv_group_num = head_num // kv_head_num
    folded_rows = max_seqlen_q * kv_group_num
    block_dmodel = triton.next_power_of_2(lk)
    block_dv = triton.next_power_of_2(lv)
    tiles = _folded_tiles(folded_rows, max(block_dmodel, block_dv))
    block_m = tiles["block_m"]
    row_blocks = triton.cdiv(folded_rows, block_m)
    if bs == 0 or total_q == 0:
        return

    extra_kargs = {}
    if current_platform().is_amd:
        extra_kargs = {
            "waves_per_eu": tiles["waves_per_eu"],
            "matrix_instr_nonkdim": 16,
        }

    _invariant_decode_kernel[(bs, kv_head_num * row_blocks)](
        q,
        k_buffer,
        v_buffer,
        o,
        lse_out,
        sinks,
        sm_scale * 1.4426950408889634,
        page_table,
        cache_seqlens,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        v_buffer.stride(-3),
        v_buffer.stride(-2),
        k_buffer.stride(0) if k_buffer.ndim == 4 else k_buffer.stride(0) * page_size,
        v_buffer.stride(0) if v_buffer.ndim == 4 else v_buffer.stride(0) * page_size,
        o.stride(0),
        o.stride(1),
        0 if lse_out is None else lse_out.stride(0),
        0 if lse_out is None else lse_out.stride(1),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        MAX_SEQLEN_Q=max_seqlen_q,
        WINDOW_LEFT=window_left,
        WINDOW_SINK=window_sink,
        kv_group_num=kv_group_num,
        ROW_BLOCKS=row_blocks,
        BLOCK_M=block_m,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DV=block_dv,
        BLOCK_N=tiles["block_n"],
        Lk=lk,
        Lv=lv,
        HAS_SINK=sinks is not None,
        RETURN_LSE=lse_out is not None,
        num_warps=tiles["num_warps"],
        num_stages=tiles["num_stages"],
        **extra_kargs,
    )


def triton_invariant_mha_decode_with_kvcache_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    window_sink: int = 0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del max_seqlen_k
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    invariant_decode_attention_fwd(
        q,
        k_cache,
        v_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen_q,
        k_cache.shape[1],
        window_left,
        window_sink,
        sm_scale=softmax_scale,
        sinks=sinks,
        lse_out=lse,
    )
    return (out, lse) if return_lse else out
