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

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_MIN_BLOCK_KV = 32


@triton.jit
def _rel_mha_prefill_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    Rel_Logits,
    O_Extend,
    LSE_Extend,
    K_Buffer,
    V_Buffer,
    cu_seqlens_q,
    cache_seqlens,
    page_table,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_rel_t,
    stride_rel_h,
    stride_rel_e,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_KV_CACHE: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(cu_seqlens_q + cur_seq)
    cur_seq_len_extend = tl.load(cu_seqlens_q + cur_seq + 1) - cur_seq_extend_start_idx
    if HAS_KV_CACHE:
        cur_seq_len = tl.load(cache_seqlens + cur_seq)
    else:
        cur_seq_len = cur_seq_len_extend
    cur_q_start = tl.maximum(cur_seq_len - cur_seq_len_extend, 0)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_local = cur_block_m * BLOCK_M + offs_m
    q_global = cur_seq_extend_start_idx + q_local
    query_positions = cur_q_start + q_local[:, None]
    mask_m = q_local < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = q_global[:, None] * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    q = tl.load(Q_Extend + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_positions = start_n + offs_n[None, :]
        mask_n = (start_n + offs_n) < cur_seq_len
        rel_dist = query_positions - key_positions
        final_mask = mask_m[:, None] & mask_n[None, :] & (rel_dist >= 0)
        if WINDOW_LEFT >= 0:
            final_mask &= rel_dist <= WINDOW_LEFT

        skip_tile = False
        if WINDOW_LEFT >= 0:
            skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not skip_tile:
            if HAS_KV_CACHE:
                cache_token_indices = start_n + offs_n
                page_indices = cache_token_indices // PAGE_SIZE
                page_offsets = cache_token_indices - page_indices * PAGE_SIZE
                physical_pages = tl.load(
                    page_table + cur_seq * page_table_stride_b + page_indices,
                    mask=mask_n,
                    other=0,
                )
                kv_loc = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets
                offs_k = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
                k = tl.load(
                    K_Buffer + offs_k,
                    mask=mask_n[None, :] & mask_d[:, None],
                    other=0.0,
                )
            else:
                offs_k = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_d[:, None]
                )
                k = tl.load(
                    K_Extend + offs_k,
                    mask=mask_n[None, :] & mask_d[:, None],
                    other=0.0,
                )

            qk = tl.dot(q.to(k.dtype), k) * sm_scale

            rel_valid = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
            rel_idx = tl.maximum(rel_dist, 0)
            rel_idx = tl.minimum(rel_idx, REL_EXTENT - 1)
            rel_offsets = (
                q_global[:, None] * stride_rel_t
                + cur_head * stride_rel_h
                + rel_idx * stride_rel_e
            )
            rel_bias = tl.load(
                Rel_Logits + rel_offsets,
                mask=mask_m[:, None] & rel_valid,
                other=0.0,
            ).to(tl.float32)
            qk += rel_bias
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if HAS_KV_CACHE:
                offs_v = (
                    kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
                v = tl.load(
                    V_Buffer + offs_v,
                    mask=mask_n[:, None] & mask_dv[None, :],
                    other=0.0,
                )
            else:
                offs_v = (
                    (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                    + cur_kv_head * stride_vh
                    + offs_dv[None, :]
                )
                v = tl.load(
                    V_Extend + offs_v,
                    mask=mask_n[:, None] & mask_dv[None, :],
                    other=0.0,
                )
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max

    safe_deno = tl.where(deno > 0.0, deno, 1.0)
    offs_o = q_global[:, None] * stride_obs + cur_head * stride_oh + offs_dv[None, :]
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / safe_deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / safe_deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )

    if HAS_LSE:
        offs_lse = q_global * stride_lse_bs + cur_head * stride_lse_h
        lse = tl.where(deno > 0.0, tl.log(deno) + e_max, float("-inf"))
        tl.store(LSE_Extend + offs_lse, lse, mask=mask_m)


def _rel_mha_prefill_fwd(
    q_extend,
    k_extend,
    v_extend,
    rel_logits,
    o_extend,
    k_buffer,
    v_buffer,
    cu_seqlens_q,
    cache_seqlens,
    max_len_extend,
    sm_scale,
    window_left=-1,
    page_table=None,
    page_table_stride_b=0,
    page_size=1,
    has_kv_cache=False,
    lse_extend=None,
):
    head_dim = q_extend.shape[-1]
    value_dim = v_buffer.shape[-1] if has_kv_cache else v_extend.shape[-1]
    block_dmodel = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(value_dim)

    block_m = 32 if head_dim > 256 else 64
    block_n = page_size if has_kv_cache else 128
    num_warps = 4

    batch_size = cu_seqlens_q.shape[0] - 1
    kv_heads = k_buffer.shape[1] if has_kv_cache else k_extend.shape[1]
    kv_group_num = q_extend.shape[1] // kv_heads
    lse_arg = lse_extend if lse_extend is not None else o_extend
    page_table_arg = page_table if page_table is not None else cache_seqlens
    grid = (batch_size, q_extend.shape[1], triton.cdiv(max_len_extend, block_m))

    num_stages = 1

    _rel_mha_prefill_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        rel_logits,
        o_extend,
        lse_arg,
        k_buffer,
        v_buffer,
        cu_seqlens_q,
        cache_seqlens,
        page_table_arg,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        o_extend.stride(0),
        o_extend.stride(1),
        lse_arg.stride(0),
        lse_arg.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        page_table_stride_b,
        page_size,
        WINDOW_LEFT=window_left,
        Lq=head_dim,
        Lv=value_dim,
        REL_EXTENT=rel_logits.shape[2],
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAS_KV_CACHE=has_kv_cache,
        HAS_LSE=lse_extend is not None,
        STORE_TRANSPOSE=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _rel_mha_decode_stage1_kernel(
    Q,
    Rel_Logits,
    K_Buffer,
    V_Buffer,
    sm_scale,
    page_table,
    cache_seqlens,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_rel_t,
    stride_rel_h,
    stride_rel_e,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    REL_EXTENT: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_start_offset = cache_len - cur_batch_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(
            Q + cur_q * stride_qbs + cur_head * stride_qh + offs_d,
            mask=mask_d,
            other=0.0,
        )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            token_indices = kv_start_offset + offs_n
            page_indices = token_indices // PAGE_SIZE
            page_offsets = token_indices - page_indices * PAGE_SIZE
            physical_pages = tl.load(
                page_table + cur_batch * page_table_stride_b + page_indices,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets

            offs_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_k,
                mask=(offs_n[:, None] < split_kv_end) & mask_d[None, :],
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1) * sm_scale

            rel_dist = cache_len - 1 - token_indices
            rel_valid = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
            rel_idx = tl.maximum(rel_dist, 0)
            rel_idx = tl.minimum(rel_idx, REL_EXTENT - 1)
            rel_offsets = (
                cur_q * stride_rel_t + cur_head * stride_rel_h + rel_idx * stride_rel_e
            )
            rel_bias = tl.load(
                Rel_Logits + rel_offsets,
                mask=(offs_n < split_kv_end) & rel_valid,
                other=0.0,
            ).to(tl.float32)
            qk += rel_bias
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_v,
                mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :],
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_q * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_lse = (
            cur_q * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum))


@triton.jit
def _rel_mha_decode_grouped_stage1_kernel(
    Q,
    Rel_Logits,
    K_Buffer,
    V_Buffer,
    sm_scale,
    page_table,
    cache_seqlens,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_rel_t,
    stride_rel_h,
    stride_rel_e,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    REL_EXTENT: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h &= cur_head < q_head_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_start_offset = cache_len - cur_batch_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    offs_q = cur_q * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            token_indices = kv_start_offset + offs_n
            page_indices = token_indices // PAGE_SIZE
            page_offsets = token_indices - page_indices * PAGE_SIZE
            physical_pages = tl.load(
                page_table + cur_batch * page_table_stride_b + page_indices,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets

            offs_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_k,
                mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None],
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype)) * sm_scale

            rel_dist = cache_len - 1 - token_indices
            rel_valid = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
            rel_idx = tl.maximum(rel_dist, 0)
            rel_idx = tl.minimum(rel_idx, REL_EXTENT - 1)
            rel_offsets = (
                cur_q * stride_rel_t
                + cur_head[:, None] * stride_rel_h
                + rel_idx[None, :] * stride_rel_e
            )
            rel_bias = tl.load(
                Rel_Logits + rel_offsets,
                mask=mask_h[:, None] & rel_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            qk += rel_bias
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk,
                float("-inf"),
            )

            offs_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_v,
                mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :],
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_q * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=mask_h[:, None] & mask_dv[None, :],
        )

        offs_mid_lse = (
            cur_q * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _rel_mha_decode_stage2_kernel(
    Mid_O,
    Mid_Lse,
    Out,
    cache_seqlens,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_SEQLEN_Q: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head = tl.program_id(1)

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    offs_v = cur_q * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_lse = (cur_q * stride_mid_ob + cur_head * stride_mid_oh) // Lv

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os,
                mask=mask_d,
                other=0.0,
            )
            split_lse = tl.load(Mid_Lse + offs_lse + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(split_lse, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            split_scale = tl.exp(split_lse - n_e_max)
            acc = acc * old_scale + split_scale * tv
            e_sum = e_sum * old_scale + split_scale
            e_max = n_e_max

    tl.store(
        Out + cur_q * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_split_count(max_seqlen_k: int, window_left: int) -> int:
    if window_left >= 0:
        effective_len = max(1, min(max_seqlen_k, window_left + 1))
        return min(4, max(1, triton.cdiv(effective_len, 128)))
    return min(32, max(1, triton.cdiv(max(1, max_seqlen_k), 2048)))


def _rel_mha_decode_fwd(
    q,
    rel_logits,
    k_buffer,
    v_buffer,
    out,
    page_table,
    cache_seqlens,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    max_seqlen_q,
    page_table_stride_b,
    page_size,
    window_left,
    sm_scale,
):
    block_n = page_size
    head_dim = k_buffer.shape[-1]
    value_dim = v_buffer.shape[-1]
    block_dmodel = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(value_dim)
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    block_h = min(8, kv_group_num)

    num_stages = 1

    if kv_group_num == 1:
        stage1_grid = (q.shape[0], q.shape[1], max_kv_splits)
        _rel_mha_decode_stage1_kernel[stage1_grid](
            q,
            rel_logits,
            k_buffer,
            v_buffer,
            sm_scale,
            page_table,
            cache_seqlens,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            rel_logits.stride(0),
            rel_logits.stride(1),
            rel_logits.stride(2),
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            page_table_stride_b,
            page_size,
            MAX_SEQLEN_Q=max_seqlen_q,
            WINDOW_LEFT=window_left,
            kv_group_num=kv_group_num,
            BLOCK_DMODEL=block_dmodel,
            BLOCK_DV=block_dv,
            BLOCK_N=block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            Lk=head_dim,
            Lv=value_dim,
            REL_EXTENT=rel_logits.shape[2],
            num_warps=1,
            num_stages=num_stages,
        )
    else:
        stage1_grid = (
            q.shape[0],
            triton.cdiv(q.shape[1], min(block_h, kv_group_num)),
            max_kv_splits,
        )
        _rel_mha_decode_grouped_stage1_kernel[stage1_grid](
            q,
            rel_logits,
            k_buffer,
            v_buffer,
            sm_scale,
            page_table,
            cache_seqlens,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            rel_logits.stride(0),
            rel_logits.stride(1),
            rel_logits.stride(2),
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            page_table_stride_b,
            page_size,
            MAX_SEQLEN_Q=max_seqlen_q,
            WINDOW_LEFT=window_left,
            kv_group_num=kv_group_num,
            q_head_num=q.shape[1],
            BLOCK_DMODEL=block_dmodel,
            BLOCK_DV=block_dv,
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            Lk=head_dim,
            Lv=value_dim,
            REL_EXTENT=rel_logits.shape[2],
            num_warps=4,
            num_stages=num_stages,
        )

    stage2_grid = (q.shape[0], q.shape[1])
    _rel_mha_decode_stage2_kernel[stage2_grid](
        attn_logits,
        attn_lse,
        out,
        cache_seqlens,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        out.stride(0),
        out.stride(1),
        MAX_SEQLEN_Q=max_seqlen_q,
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=block_dv,
        Lv=value_dim,
        WINDOW_LEFT=window_left,
        num_warps=4,
        num_stages=2,
    )


@register_kernel(
    "attention",
    "rel_mha_prefill",
    name="triton_rel_mha_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_rel_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    cache_seqlens = torch.empty((0,), dtype=torch.int32, device=q.device)
    empty_k = torch.empty((0, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    empty_v = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
    _rel_mha_prefill_fwd(
        q,
        k,
        v,
        rel_logits,
        out,
        empty_k,
        empty_v,
        cu_seqlens,
        cache_seqlens,
        max_seqlen,
        softmax_scale,
        window_left=window_left,
        has_kv_cache=False,
        lse_extend=lse,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "rel_mha_extend_with_kvcache",
    name="triton_rel_mha_extend_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_rel_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    rel_logits: torch.Tensor,
    window_left: int = -1,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    enable_pdl: bool = False,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    k = torch.empty(
        (0, k_cache.shape[2], k_cache.shape[3]),
        dtype=k_cache.dtype,
        device=k_cache.device,
    )
    v = torch.empty(
        (0, v_cache.shape[2], v_cache.shape[3]),
        dtype=v_cache.dtype,
        device=v_cache.device,
    )
    out = torch.empty_like(q)
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    _rel_mha_prefill_fwd(
        q,
        k,
        v,
        rel_logits,
        out,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        cu_seqlens_q,
        cache_seqlens,
        max_seqlen_q,
        softmax_scale,
        window_left=window_left,
        page_table=page_table,
        page_table_stride_b=page_table.stride(0),
        page_size=k_cache.shape[1],
        has_kv_cache=True,
        lse_extend=lse,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "rel_mha_decode_with_kvcache",
    name="triton_rel_mha_decode_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_rel_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    rel_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    softmax_scale: float | None = None,
    enable_pdl: bool = False,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    max_kv_splits = _decode_split_count(max_seqlen_k, window_left)
    attn_logits = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        dtype=torch.float32,
        device=q.device,
    )
    num_kv_splits = torch.full(
        (cache_seqlens.shape[0],),
        max_kv_splits,
        dtype=torch.int32,
        device=q.device,
    )
    _rel_mha_decode_fwd(
        q,
        rel_logits,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        out,
        page_table,
        cache_seqlens,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        max_seqlen_q,
        page_table.stride(0),
        k_cache.shape[1],
        window_left,
        softmax_scale,
    )
    return out
