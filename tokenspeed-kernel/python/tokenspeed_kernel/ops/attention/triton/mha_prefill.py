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

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    LSE_Extend,
    K_Buffer,
    V_Buffer,
    cu_seqlens_q,
    cache_seqlens,
    mask_ptr,
    sink_ptr,
    page_table,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
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
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    HAS_KV_CACHE: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
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

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(cu_seqlens_q + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # Compute attention over the full visible KV range. For causal cached
    # prefill, query tokens are treated as the suffix of the KV sequence.
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask
        elif IS_CAUSAL:
            query_positions = cur_q_start + cur_block_m * BLOCK_M + offs_m[:, None]
            key_positions = start_n + offs_n[None, :]
            final_mask &= query_positions >= key_positions
        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            query_positions = cur_q_start + cur_block_m * BLOCK_M + offs_m[:, None]
            key_positions = start_n + offs_n[None, :]
            window_mask = query_positions <= key_positions + SLIDING_WINDOW_SIZE
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            if HAS_KV_CACHE:
                cache_token_indices = start_n + offs_n
                page_indices = cache_token_indices // PAGE_SIZE
                page_offsets = cache_token_indices - page_indices * PAGE_SIZE
                physical_pages = tl.load(
                    page_table + cur_seq * page_table_stride_b + page_indices,
                    mask=mask_n,
                    other=0,
                )
                # int64 to avoid address overflow for large (>int32 range) KV caches
                offs_kv_loc = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets
                offs_k = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
                k = tl.load(
                    K_Buffer + offs_k,
                    mask=(mask_n[None, :]) & (mask_d[:, None]),
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
                    mask=(mask_n[None, :]) & (mask_d[:, None]),
                    other=0.0,
                )

            qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                if HAS_KV_CACHE:
                    offs_kpe = (
                        offs_kv_loc[None, :] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                    kpe = tl.load(
                        K_Buffer + offs_kpe,
                        mask=mask_n[None, :],
                        other=0.0,
                    )
                else:
                    offs_kpe = (
                        (cur_seq_extend_start_idx + start_n + offs_n[None, :])
                        * stride_kbs
                        + cur_kv_head * stride_kh
                        + offs_dpe[:, None]
                    )
                    kpe = tl.load(
                        K_Extend + offs_kpe,
                        mask=mask_n[None, :],
                        other=0.0,
                    )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if HAS_KV_CACHE:
                offs_v = (
                    offs_kv_loc[:, None] * stride_buf_vbs
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
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    safe_deno = tl.where(deno > 0.0, deno, 1.0)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
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
        offs_lse = (
            cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m
        ) * stride_lse_bs + cur_head * stride_lse_h
        lse = tl.where(deno > 0.0, tl.log(deno) + e_max, float("-inf"))
        tl.store(LSE_Extend + offs_lse, lse, mask=mask_m)


def prefill_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    cu_seqlens_q,
    cache_seqlens,
    custom_mask,
    is_causal,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    page_table=None,
    page_table_stride_b=0,
    page_size=1,
    has_kv_cache=False,
    lse_extend=None,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    platform = current_platform()

    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if platform.is_amd:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if platform.is_hopper_plus:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif platform.is_ampere_plus:
            # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
            if platform.arch_version.minor == 9 or platform.arch_version.minor == 6:
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (64, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 32)
            else:
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (128, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = cu_seqlens_q.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    HAS_SINK = sinks is not None
    HAS_LSE = lse_extend is not None
    lse_arg = lse_extend if lse_extend is not None else o_extend
    page_table_arg = page_table if page_table is not None else cache_seqlens

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if platform.is_amd:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16}

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        lse_arg,
        k_buffer,
        v_buffer,
        cu_seqlens_q,
        cache_seqlens,
        custom_mask,
        sinks,
        page_table_arg,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
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
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        HAS_KV_CACHE=has_kv_cache,
        HAS_SINK=HAS_SINK,
        HAS_LSE=HAS_LSE,
        STORE_TRANSPOSE=platform.is_amd,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
