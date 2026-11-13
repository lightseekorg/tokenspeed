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

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform

_MIN_BLOCK_KV = 32
# Split the keys until the stage-1 grid has about this many programs per
# compute unit; a small batch otherwise leaves most of the GPU idle while each
# program walks its whole cache.
_PROGRAMS_PER_CU = 4


def _num_kv_splits(q: torch.Tensor, num_kv_heads: int, max_kv_splits: int) -> int:
    """Key splits per query: the batch shape alone decides, so one captured
    graph keeps one split count. The splits are a runtime tensor, so a new
    count reuses the compiled kernels."""
    group = q.shape[1] // num_kv_heads
    # Stage-1 programs per query: one per head, or per block of up to 16
    # grouped heads.
    heads = q.shape[1] if group == 1 else num_kv_heads * triton.cdiv(group, 16)
    programs = q.shape[0] * heads
    target = _PROGRAMS_PER_CU * current_platform().sm_count
    return max(1, min(max_kv_splits, triton.cdiv(target, programs)))


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
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
    stride_buf_kp,
    stride_buf_vp,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Page-table width follows the batch; runtime so every batch shape
    # shares one binary.
    page_table_stride_b,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_SINK: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    DIRECT_OUTPUT: tl.constexpr,
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
    # WINDOW_LEFT is exclusive of the current token, so keep WINDOW_LEFT + 1
    # keys (window_left left keys plus the current one).
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_start_offset = cache_len - cur_batch_seq_len
    # Sink rows the window does not already cover, walked before it as extra
    # iterations at absolute token indices.
    sink_len = tl.minimum(WINDOW_SINK, kv_start_offset)
    cur_batch_seq_len = cur_batch_seq_len + sink_len
    kv_splits = 1 if DIRECT_OUTPUT else tl.load(num_kv_splits + cur_batch)

    off_q = cur_q * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # First sink_len entries are tokens 0..sink_len-1, then the window.
            token_indices = tl.where(
                offs_n < sink_len,
                offs_n,
                kv_start_offset + offs_n - sink_len,
            )
            page_indices = token_indices // PAGE_SIZE
            page_offsets = token_indices - page_indices * PAGE_SIZE
            physical_pages = tl.load(
                page_table + cur_batch * page_table_stride_b + page_indices,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # int64 to avoid address overflow for large (>int32 range) KV caches
            k_loc = (
                physical_pages.to(tl.int64) * stride_buf_kp
                + page_offsets * stride_buf_kbs
            )
            v_loc = (
                physical_pages.to(tl.int64) * stride_buf_vp
                + page_offsets * stride_buf_vbs
            )
            offs_buf_k = k_loc[:, None] + cur_kv_head * stride_buf_kh + offs_d[None, :]
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = v_loc[:, None] + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
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

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        if not DIRECT_OUTPUT:
            offs_mid_o_1 = (
                cur_q * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
            ) // Lv

            tl.store(
                Att_Lse + offs_mid_o_1,
                e_max + tl.log(e_sum),
            )
    elif DIRECT_OUTPUT:
        offs_out = cur_q * stride_mid_ob + cur_head * stride_mid_oh + offs_dv
        tl.store(Att_Out + offs_out, 0.0, mask=mask_dv)


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    page_table,
    cache_seqlens,
    num_kv_splits,
    max_kv_splits,
    page_table_stride_b,
    page_size,
    max_seqlen_q,
    window_left,
    window_sink,
    sm_scale,
    logit_cap,
    *,
    direct_output=False,
):
    platform = current_platform()

    BLOCK = 64
    # MI3xx uses a smaller block size to stay within the SGPR limit.
    if platform.is_amd:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    total_q, head_num = q.shape[0], q.shape[1]

    grid = (total_q, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if platform.is_amd:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        page_table,
        cache_seqlens,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        v_buffer.stride(-3),
        v_buffer.stride(-2),
        k_buffer.stride(0) if k_buffer.ndim == 4 else k_buffer.stride(0) * page_size,
        v_buffer.stride(0) if v_buffer.ndim == 4 else v_buffer.stride(0) * page_size,
        att_out.stride(0),
        att_out.stride(1),
        0 if direct_output else att_out.stride(2),
        page_table_stride_b,
        page_size,
        MAX_SEQLEN_Q=max_seqlen_q,
        WINDOW_LEFT=window_left,
        WINDOW_SINK=window_sink,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        DIRECT_OUTPUT=direct_output,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
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
    stride_buf_kp,
    stride_buf_vp,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Page-table width follows the batch; runtime so every batch shape
    # shares one binary.
    page_table_stride_b,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_SINK: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    DIRECT_OUTPUT: tl.constexpr,
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
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    # WINDOW_LEFT is exclusive of the current token, so keep WINDOW_LEFT + 1
    # keys (window_left left keys plus the current one).
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    kv_start_offset = cache_len - cur_batch_seq_len
    # Sinks: see _fwd_kernel_stage1.
    sink_len = tl.minimum(WINDOW_SINK, kv_start_offset)
    cur_batch_seq_len = cur_batch_seq_len + sink_len
    kv_splits = 1 if DIRECT_OUTPUT else tl.load(num_kv_splits + cur_batch)

    offs_q = cur_q * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = cur_q * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # First sink_len entries are tokens 0..sink_len-1, then the window.
            token_indices = tl.where(
                offs_n < sink_len,
                offs_n,
                kv_start_offset + offs_n - sink_len,
            )
            page_indices = token_indices // PAGE_SIZE
            page_offsets = token_indices - page_indices * PAGE_SIZE
            physical_pages = tl.load(
                page_table + cur_batch * page_table_stride_b + page_indices,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # int64 to avoid address overflow for large (>int32 range) KV caches
            k_loc = (
                physical_pages.to(tl.int64) * stride_buf_kp
                + page_offsets * stride_buf_kbs
            )
            v_loc = (
                physical_pages.to(tl.int64) * stride_buf_vp
                + page_offsets * stride_buf_vbs
            )
            offs_buf_k = k_loc[None, :] + cur_kv_head * stride_buf_kh + offs_d[:, None]
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    k_loc[None, :] + cur_kv_head * stride_buf_kh + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = v_loc[:, None] + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
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
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        if not DIRECT_OUTPUT:
            offs_mid_o_1 = (
                cur_q * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
            ) // Lv

            tl.store(
                Att_Lse + offs_mid_o_1,
                e_max + tl.log(e_sum),
                mask=mask_h,
            )
    elif DIRECT_OUTPUT:
        offs_out = (
            cur_q * stride_mid_ob + cur_head[:, None] * stride_mid_oh + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_out, 0.0, mask=mask_h[:, None] & mask_dv[None, :]
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    page_table,
    cache_seqlens,
    num_kv_splits,
    max_kv_splits,
    page_table_stride_b,
    page_size,
    max_seqlen_q,
    window_left,
    window_sink,
    sm_scale,
    logit_cap,
    *,
    direct_output=False,
):
    platform = current_platform()

    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # MI3xx uses a smaller block size for large heads to stay within shmem limits.
    if platform.is_amd and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    total_q, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        total_q,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if platform.is_amd:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16}
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        page_table,
        cache_seqlens,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        v_buffer.stride(-3),
        v_buffer.stride(-2),
        k_buffer.stride(0) if k_buffer.ndim == 4 else k_buffer.stride(0) * page_size,
        v_buffer.stride(0) if v_buffer.ndim == 4 else v_buffer.stride(0) * page_size,
        att_out.stride(0),
        att_out.stride(1),
        0 if direct_output else att_out.stride(2),
        page_table_stride_b,
        page_size,
        MAX_SEQLEN_Q=max_seqlen_q,
        WINDOW_LEFT=window_left,
        WINDOW_SINK=window_sink,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        DIRECT_OUTPUT=direct_output,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    cache_seqlens,
    num_kv_splits,
    sink_ptr,
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
    WINDOW_SINK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_batch = cur_q // MAX_SEQLEN_Q
    q_pos = cur_q - cur_batch * MAX_SEQLEN_Q
    cur_head = tl.program_id(1)

    cache_len = tl.load(cache_seqlens + cur_batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = tl.maximum(cache_len, 0)
    # WINDOW_LEFT is exclusive of the current token, so keep WINDOW_LEFT + 1
    # keys (window_left left keys plus the current one).
    cur_batch_seq_len = (
        tl.minimum(cache_len, WINDOW_LEFT + 1) if WINDOW_LEFT >= 0 else cache_len
    )
    # Must match stage 1's row count, sinks included, or the reduction reads a
    # different number of splits than stage 1 wrote. Both branches above keep
    # cur_batch_seq_len <= cache_len, so the offset is non-negative.
    kv_start_offset = cache_len - cur_batch_seq_len
    sink_len = tl.minimum(WINDOW_SINK, kv_start_offset)
    cur_batch_seq_len = cur_batch_seq_len + sink_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_q * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_q * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    # A query with no keys in reach (an empty cache) has nothing to attend:
    # a zero output rather than 0 / 0.
    tl.store(
        O + cur_q * stride_obs + cur_head * stride_oh + offs_d,
        tl.where(e_sum > 0.0, acc / e_sum, 0.0),
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_buffer,
    cache_seqlens,
    num_kv_splits,
    max_kv_splits,
    max_seqlen_q,
    window_left,
    window_sink,
    sinks=None,
):
    platform = current_platform()
    total_q, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    extra_kargs = {}
    if platform.is_amd:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16}

    grid = (total_q, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        cache_seqlens,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_SEQLEN_Q=max_seqlen_q,
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        WINDOW_LEFT=window_left,
        WINDOW_SINK=window_sink,
        HAS_SINK=HAS_SINK,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
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
    window_sink,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        page_table,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        page_table_stride_b,
        page_size,
        max_seqlen_q,
        window_left,
        window_sink,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        max_seqlen_q,
        window_left,
        window_sink,
        sinks,
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
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
    window_sink,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        page_table,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        page_table_stride_b,
        page_size,
        max_seqlen_q,
        window_left,
        window_sink,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        cache_seqlens,
        num_kv_splits,
        max_kv_splits,
        max_seqlen_q,
        window_left,
        window_sink,
        sinks,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
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
    window_sink,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] == cache_seqlens.shape[0] * max_seqlen_q
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
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
            window_sink,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
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
            window_sink,
            sm_scale,
            logit_cap=logit_cap,
            sinks=sinks,
        )


# Folded decode: one program per (request, KV head, key split) attends every
# query row of the request, so a request's keys are read once per KV head
# rather than once per query row. Keys are split until the stage-1 grid has
# ``programs_per_cu`` (see ``_folded_tiles``) programs per compute unit, at
# most this many splits.
_FOLDED_MAX_KV_SPLITS = 16


@triton.jit
def _fwd_folded_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_log2,
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
    stride_buf_kp,
    stride_buf_vp,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
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
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """Partial attention of one key split for a block of ``BLOCK_M`` folded
    rows ``(query position, head of the KV group)`` of one request.

    Query position ``t`` of ``MAX_SEQLEN_Q`` sees the request's first
    ``seq_len - (MAX_SEQLEN_Q - 1 - t)`` keys (the last ``WINDOW_LEFT + 1`` of
    them under a sliding window). Writes each row's normalized partial and its
    base-2 log-sum-exp, ``-inf`` for a row that sees no key of the split.
    """
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1) // ROW_BLOCKS
    row_block = tl.program_id(1) % ROW_BLOCKS
    split_kv_id = tl.program_id(2)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < MAX_SEQLEN_Q * kv_group_num
    q_pos = rows // kv_group_num
    cur_head = cur_kv_head * kv_group_num + rows % kv_group_num
    # Query rows of one request are contiguous; the token index scales with the batch.
    token = cur_batch.to(tl.int64) * MAX_SEQLEN_Q + q_pos

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    seq_len = tl.load(cache_seqlens + cur_batch)
    row_end = tl.maximum(seq_len - (MAX_SEQLEN_Q - 1 - q_pos), 0)
    # Key range of the whole request: from the first row's window start.
    first_end = tl.maximum(seq_len - (MAX_SEQLEN_Q - 1), 0)
    if WINDOW_LEFT >= 0:
        row_start = tl.maximum(row_end - (WINDOW_LEFT + 1), 0)
        kv_lo = tl.maximum(first_end - (WINDOW_LEFT + 1), 0)
    else:
        row_start = tl.zeros([BLOCK_M], dtype=tl.int32)
        kv_lo = 0
    kv_splits = tl.load(num_kv_splits)
    sink_len = tl.minimum(WINDOW_SINK, kv_lo)
    visible_len = sink_len + seq_len - kv_lo
    kv_len_per_split = tl.cdiv(tl.cdiv(visible_len, kv_splits), BLOCK_N) * BLOCK_N
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, visible_len)

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
    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_ok = offs_n < split_end
        token_indices = tl.where(offs_n < sink_len, offs_n, kv_lo + offs_n - sink_len)
        page_indices = token_indices // PAGE_SIZE
        physical_pages = tl.load(
            page_table + cur_batch * page_table_stride_b + page_indices,
            mask=key_ok,
            other=0,
        )
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
        if logit_cap > 0:
            # The cap applies to natural-scale logits; sm_scale_log2 carries log2(e).
            qk = qk * (sm_scale_log2 * 0.6931471805599453)
            qk = logit_cap * tanh(qk / logit_cap) * 1.4426950408889634
        else:
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

    has_keys = e_sum > 0.0
    offs_mid = (
        token[:, None] * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_dv[None, :]
    )
    tl.store(
        Att_Out + offs_mid,
        tl.where(has_keys[:, None], acc / tl.where(has_keys, e_sum, 1.0)[:, None], 0.0),
        mask=row_ok[:, None] & mask_dv[None, :],
    )
    tl.store(
        Att_Lse + token * stride_lse_b + cur_head * stride_lse_h + split_kv_id,
        tl.where(
            has_keys, e_max + tl.log2(tl.where(has_keys, e_sum, 1.0)), -float("inf")
        ),
        mask=row_ok,
    )


@triton.jit
def _fwd_folded_kernel_stage2(
    Mid_O,
    Mid_Lse,
    O,
    Lse,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_obs,
    stride_oh,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Merge the key-split partials of one (query row, head) by their base-2
    log-sum-exp; a row with no key in reach gets a zero output and ``-inf``."""
    cur_q = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    kv_splits = tl.load(num_kv_splits)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    for split_kv_id in range(0, kv_splits):
        lse = tl.load(
            Mid_Lse + cur_q * stride_lse_b + cur_head * stride_lse_h + split_kv_id
        )
        if lse > -float("inf"):
            part = tl.load(
                Mid_O
                + cur_q * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_d,
                mask=mask_d,
                other=0.0,
            )
            n_e_max = tl.maximum(lse, e_max)
            old_scale = tl.exp2(e_max - n_e_max)
            weight = tl.exp2(lse - n_e_max)
            acc = acc * old_scale + weight * part
            e_sum = e_sum * old_scale + weight
            e_max = n_e_max

    if HAS_SINK:
        # Sinks are natural-log logits.
        cur_sink = tl.load(sink_ptr + cur_head).to(tl.float32) * 1.4426950408889634
        sink_max = tl.maximum(e_max, cur_sink)
        acc = acc * tl.exp2(e_max - sink_max)
        e_sum = e_sum * tl.exp2(e_max - sink_max) + tl.exp2(cur_sink - sink_max)
        e_max = sink_max

    has_keys = e_sum > 0.0
    tl.store(
        O + cur_q * stride_obs + cur_head * stride_oh + offs_d,
        tl.where(has_keys, acc / tl.where(has_keys, e_sum, 1.0), 0.0),
        mask=mask_d,
    )
    if RETURN_LSE:
        tl.store(
            Lse + cur_q * tl.num_programs(1) + cur_head,
            tl.where(
                has_keys,
                (e_max + tl.log2(tl.where(has_keys, e_sum, 1.0))) * 0.6931471805599453,
                -float("inf"),
            ),
        )


def _folded_tiles(folded_rows: int, head_dim: int) -> dict:
    """Stage-1 launch config of the folded decode (tuned on gfx950, head_dim
    256, 48 and 96 folded rows): ``block_m`` folded rows per program (one
    MFMA tile at least; up to 128 with 8 warps so a 16-row tree verify reads
    its keys once per KV head), ``block_n``, ``num_warps``, ``num_stages``,
    the AMD ``waves_per_eu`` hint and the stage-1 ``programs_per_cu`` target
    that sets the key-split count."""
    block_n = 32 if head_dim > 128 else 64
    if folded_rows > 64:
        return dict(
            block_m=128,
            block_n=block_n,
            num_warps=8,
            num_stages=2,
            waves_per_eu=1,
            programs_per_cu=1,
        )
    return dict(
        block_m=max(triton.next_power_of_2(folded_rows), 16),
        block_n=block_n,
        num_warps=4,
        num_stages=2,
        waves_per_eu=2,
        programs_per_cu=2,
    )


def _folded_kv_splits(bs: int, programs_per_split: int, programs_per_cu: int) -> int:
    """Key splits of the folded decode: the batch shape alone decides, so one
    captured graph keeps one count; the count is a runtime argument."""
    target = programs_per_cu * current_platform().sm_count
    return max(
        1,
        min(
            _FOLDED_MAX_KV_SPLITS, triton.cdiv(target, max(bs * programs_per_split, 1))
        ),
    )


def decode_attention_fwd_folded(
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
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
) -> None:
    """Decode ``max_seqlen_q`` causal query rows per request with every row of
    a request and its KV group folded into one program per key split.

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
        sm_scale: softmax scale applied to ``q . k``.
        logit_cap: soft cap on the scaled logits, ``0`` for none.
        sinks: ``[num_q_heads]`` attention sink logits, or ``None``.
        lse_out: ``[bs * max_seqlen_q, num_q_heads]`` float32 natural-log
            log-sum-exp output, or ``None``.
    """
    bs = cache_seqlens.shape[0]
    total_q, head_num, lk = q.shape
    lv = v_buffer.shape[-1]
    kv_head_num = k_buffer.shape[-2]
    kv_group_num = head_num // kv_head_num
    folded_rows = max_seqlen_q * kv_group_num
    block_dmodel = triton.next_power_of_2(lk)
    block_dv = triton.next_power_of_2(lv)
    tiles = _folded_tiles(folded_rows, max(block_dmodel, block_dv))
    block_m, block_n = tiles["block_m"], tiles["block_n"]
    row_blocks = triton.cdiv(folded_rows, block_m)

    splits = _folded_kv_splits(bs, kv_head_num * row_blocks, tiles["programs_per_cu"])
    num_kv_splits = torch.full((1,), splits, dtype=torch.int32, device=q.device)
    att_out = torch.empty(
        (total_q, head_num, splits, lv), dtype=torch.float32, device=q.device
    )
    # Sized for the most splits whatever the count, so its strides (kernel
    # arguments) stay fixed and never fall into Triton's stride-1 specialization.
    att_lse = torch.empty(
        (total_q, head_num, _FOLDED_MAX_KV_SPLITS), dtype=torch.float32, device=q.device
    )
    extra_kargs = {}
    if current_platform().is_amd:
        extra_kargs = {
            "waves_per_eu": tiles["waves_per_eu"],
            "matrix_instr_nonkdim": 16,
        }
    if bs > 0:
        _fwd_folded_kernel_stage1[(bs, kv_head_num * row_blocks, splits)](
            q,
            k_buffer,
            v_buffer,
            sm_scale * 1.4426950408889634,
            page_table,
            cache_seqlens,
            att_out,
            att_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(-3),
            k_buffer.stride(-2),
            v_buffer.stride(-3),
            v_buffer.stride(-2),
            (
                k_buffer.stride(0)
                if k_buffer.ndim == 4
                else k_buffer.stride(0) * page_size
            ),
            (
                v_buffer.stride(0)
                if v_buffer.ndim == 4
                else v_buffer.stride(0) * page_size
            ),
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            att_lse.stride(0),
            att_lse.stride(1),
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
            BLOCK_N=block_n,
            logit_cap=logit_cap,
            Lk=lk,
            Lv=lv,
            num_warps=tiles["num_warps"],
            num_stages=tiles["num_stages"],
            **extra_kargs,
        )
    if total_q > 0:
        _fwd_folded_kernel_stage2[(total_q, head_num)](
            att_out,
            att_lse,
            o,
            lse_out,
            num_kv_splits,
            sinks,
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            att_lse.stride(0),
            att_lse.stride(1),
            o.stride(0),
            o.stride(1),
            BLOCK_DV=block_dv,
            Lv=lv,
            HAS_SINK=sinks is not None,
            RETURN_LSE=lse_out is not None,
            num_warps=4,
        )


def _triton_mha_decode_with_kvcache_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    window_sink: int = 0,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    if max_seqlen_q > 1 or return_lse:
        # Several query rows per request (speculative verify) share the
        # request's keys: fold them into one program per KV head and split.
        if return_lse and sinks is not None:
            raise ValueError("the log-sum-exp of a decode with sinks is undefined")
        lse = (
            torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        decode_attention_fwd_folded(
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
            logit_cap=logit_cap,
            sinks=sinks,
            lse_out=lse,
        )
        return (out, lse) if return_lse else out
    max_kv_splits = 4
    splits = _num_kv_splits(q, k_cache.shape[2], max_kv_splits)
    if splits == 1 and sinks is None and window_sink == 0:
        # With one KV split and no sink, stage 2 only casts and stores the result.
        launch = (
            _decode_att_m_fwd
            if q.shape[1] == v_cache.shape[2]
            else _decode_grouped_att_m_fwd
        )
        launch(
            q,
            k_cache,
            v_cache,
            out,
            None,
            page_table,
            cache_seqlens,
            None,
            1,
            page_table.stride(0),
            k_cache.shape[1],
            max_seqlen_q,
            window_left,
            window_sink,
            softmax_scale,
            logit_cap,
            direct_output=True,
        )
        return out
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
        splits,
        dtype=torch.int32,
        device=q.device,
    )
    decode_attention_fwd(
        q,
        k_cache,
        v_cache,
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
        window_sink,
        sm_scale=softmax_scale,
        logit_cap=logit_cap,
        sinks=sinks,
    )
    return out
