# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the LICENSE file
# of https://github.com/fla-org/flash-linear-attention (v0.5.1); vendored here
# with dual-index in-place state-pool addressing added for the flat KV cache.

"""KDA fused recurrent decode with in-kernel state-pool addressing.

Math is byte-identical to ``fla.ops.kda.fused_recurrent`` (v0.5.1). The only
change is state I/O: instead of gather -> kernel -> scatter round-trips through
``at::index_elementwise``, the kernel reads ``h0_pool[read_idx]`` and writes
``h0_pool[write_idx]`` directly. ``write_idx < 0`` (graph padding) skips the
store; read and write indices may differ (flat-KV page-boundary crossing).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_pool_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    lower_bound,
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    stride_g_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_tok + i_h * K + o_k
    p_k = k + bos * stride_k_tok + i_h * K + o_k
    p_v = v + bos * stride_v_tok + i_hv * V + o_v
    p_beta = beta + bos * stride_beta_tok + i_hv
    p_g = g + bos * stride_g_tok + i_hv * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_read = tl.load(read_indices + i_n).to(tl.int64)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_h0 = (
        h_pool
        + b_read * stride_state_page
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h += tl.load(p_h0, mask=mask_h & (b_read >= 0), other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy="evict_first").to(
            tl.float32
        )

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g, eviction_policy="evict_last").to(tl.float32)

        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32)
            if HAS_DT_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(
                    tl.float32
                )
                b_g = b_g + b_bias
            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
            else:
                b_gk = -tl.exp(b_A) * _softplus(b_g)
        else:
            b_gk = b_g

        b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
            eviction_policy="evict_first",
        )

        p_q += stride_q_tok
        p_k += stride_k_tok
        p_o += HV * V
        p_v += stride_v_tok
        p_g += stride_g_tok
        p_beta += stride_beta_tok

    b_write = tl.load(write_indices + i_n).to(tl.int64)
    p_ht = (
        h_pool
        + b_write * stride_state_page
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & (b_write >= 0))


def fused_recurrent_kda_pool(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    h_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
) -> torch.Tensor:
    """Single-step KDA decode reading/writing the state pool in-kernel.

    Args:
        q/k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``; g: ``[B, T, HV, K]`` raw
            gate input; beta: ``[B, T, HV]`` raw logits.
        A_log: ``[HV]`` per-head log decay; dt_bias: ``[HV*K]`` or ``[HV, K]``.
        h_pool: ``[num_pages, HV, K, V]`` full state pool (fp32), updated
            in place at ``write_indices``.
        read_indices / write_indices: ``[N]`` page ids; negative ids read
            zeros / skip the write (graph padding, page-boundary crossing).
        scale: attention scale; defaults to ``K ** -0.5``.
        cu_seqlens: varlen offsets ``[N+1]`` (decode: one token per sequence).
        lower_bound: safe-gate lower bound (K3: ``gate_lower_bound``).

    Returns:
        o: ``[B, T, HV, V]`` attention output (same dtype as ``v``).
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if scale is None:
        scale = K**-0.5

    # q/k/v are typically zero-copy slices of the packed conv output: honor
    # their token strides (inner head*dim layout must still be dense).
    for t, d in ((q, K), (k, K), (v, V), (g, K)):
        assert t.stride(-1) == 1 and t.stride(-2) == d, "inner dims must be dense"
    out = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)
    grid = (triton.cdiv(V, 32) * N * HV,)
    fused_recurrent_kda_pool_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h_pool=h_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        stride_q_tok=q.stride(1),
        stride_k_tok=k.stride(1),
        stride_v_tok=v.stride(1),
        stride_g_tok=g.stride(1),
        stride_beta_tok=beta.stride(1),
        scale=scale,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=32,
        stride_state_page=h_pool.stride(0),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_mtp_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h_pool,
    h_pool_out,
    read_indices,
    write_indices,
    lower_bound,
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    stride_g_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_state_out_page: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    num_stages: tl.constexpr,
):
    """MTP target-verify variant of the pool kernel: dense ``[N, T]`` batch,
    the evolved state is stored after EVERY step to its own row of
    ``h_pool_out`` (``write_indices[n, t]``) so a partial accept can resume
    from the state at the accepted position. ``h_pool_out`` may be the pool
    itself or a dedicated verify scratch (flat path)."""
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = i_n * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_tok + i_h * K + o_k
    p_k = k + bos * stride_k_tok + i_h * K + o_k
    p_v = v + bos * stride_v_tok + i_hv * V + o_v
    p_beta = beta + bos * stride_beta_tok + i_hv
    p_g = g + bos * stride_g_tok + i_hv * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_read = tl.load(read_indices + i_n).to(tl.int64)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_h0 = (
        h_pool
        + b_read * stride_state_page
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h += tl.load(p_h0, mask=mask_h & (b_read >= 0), other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy="evict_first").to(
            tl.float32
        )

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g, eviction_policy="evict_last").to(tl.float32)

        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32)
            if HAS_DT_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(
                    tl.float32
                )
                b_g = b_g + b_bias
            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
            else:
                b_gk = -tl.exp(b_A) * _softplus(b_g)
        else:
            b_gk = b_g

        b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
            eviction_policy="evict_first",
        )

        b_write = tl.load(write_indices + i_n * T + i_t).to(tl.int64)
        p_ht = (
            h_pool_out
            + b_write * stride_state_out_page
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & (b_write >= 0))

        p_q += stride_q_tok
        p_k += stride_k_tok
        p_o += HV * V
        p_v += stride_v_tok
        p_g += stride_g_tok
        p_beta += stride_beta_tok


def fused_recurrent_kda_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    h_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    h_pool_out: torch.Tensor | None = None,
    scale: float | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
) -> torch.Tensor:
    """MTP target-verify KDA decode: per-step state pages for rollback.

    Args:
        q/k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``; g: ``[B, T, HV, K]``;
            beta: ``[B, T, HV]`` — dense request-major verify batch,
            ``T`` = draft_token_num.
        h_pool: ``[num_pages, HV, K, V]`` fp32 state pool (read side).
        read_indices: ``[B]`` initial-state page per request (< 0 reads zeros).
        write_indices: ``[B, T]`` per-position output rows (< 0 skips).
        h_pool_out: optional write-side pool (verify scratch); defaults to
            ``h_pool``.

    Returns:
        o: ``[B, T, HV, V]`` attention output (same dtype as ``v``).
    """
    if h_pool_out is None:
        h_pool_out = h_pool
    B, T, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    if scale is None:
        scale = K**-0.5
    assert write_indices.shape == (B, T) or write_indices.numel() == B * T
    for t, d in ((q, K), (k, K), (v, V), (g, K)):
        assert t.stride(-1) == 1 and t.stride(-2) == d, "inner dims must be dense"
    out = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)
    grid = (triton.cdiv(V, 32) * B * HV,)
    fused_recurrent_kda_mtp_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h_pool=h_pool,
        h_pool_out=h_pool_out,
        read_indices=read_indices,
        write_indices=write_indices.reshape(-1),
        lower_bound=lower_bound,
        stride_q_tok=q.stride(1),
        stride_k_tok=k.stride(1),
        stride_v_tok=v.stride(1),
        stride_g_tok=g.stride(1),
        stride_beta_tok=beta.stride(1),
        scale=scale,
        N=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=32,
        stride_state_page=h_pool.stride(0),
        stride_state_out_page=h_pool_out.stride(0),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        HAS_DT_BIAS=dt_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_megafuse_fwd_kernel(
    qkv_raw,  # [T, 3*P] pre-conv packed projections (token-strided)
    conv_w,  # [3*P, 4] fused conv bank
    conv_pool,  # [pages, 3*P, 3] conv state
    f_a,  # [T, D_fa] low-rank gate input
    w_fb,  # [P, D_fa] f_b weight
    beta,
    A_log,
    dt_bias,
    o,
    h_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    lower_bound,
    stride_raw_tok: tl.constexpr,
    stride_fa_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    P: tl.constexpr,  # proj_local = HV * K
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_conv_page: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_read = tl.load(read_indices + i_n).to(tl.int64)
    b_write = tl.load(write_indices + i_n).to(tl.int64)
    read_ok = b_read >= 0

    # --- fused conv (4-tap depthwise + silu) over this program's features ---
    qf = i_h * K + o_k  # q features in section 0
    kf = P + i_h * K + o_k  # k features in section 1
    vf = 2 * P + i_hv * V + o_v  # v features in section 2

    x_q = tl.load(qkv_raw + bos * stride_raw_tok + qf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_k = tl.load(qkv_raw + bos * stride_raw_tok + kf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_v = tl.load(qkv_raw + bos * stride_raw_tok + vf, mask=mask_v, other=0.0).to(
        tl.float32
    )

    acc_q = x_q * tl.load(conv_w + qf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    acc_k = x_k * tl.load(conv_w + kf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    acc_v = x_v * tl.load(conv_w + vf * 4 + 3, mask=mask_v, other=0.0).to(tl.float32)
    s_q0 = tl.zeros([BK], dtype=tl.float32)
    s_k0 = tl.zeros([BK], dtype=tl.float32)
    s_v0 = tl.zeros([BV], dtype=tl.float32)
    s_q1 = tl.zeros([BK], dtype=tl.float32)
    s_k1 = tl.zeros([BK], dtype=tl.float32)
    s_v1 = tl.zeros([BV], dtype=tl.float32)
    s_q2 = tl.zeros([BK], dtype=tl.float32)
    s_k2 = tl.zeros([BK], dtype=tl.float32)
    s_v2 = tl.zeros([BV], dtype=tl.float32)
    if read_ok:
        cp = conv_pool + b_read * stride_conv_page
        s_q0 = tl.load(cp + qf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_q1 = tl.load(cp + qf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_q2 = tl.load(cp + qf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_k0 = tl.load(cp + kf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_k1 = tl.load(cp + kf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_k2 = tl.load(cp + kf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_v0 = tl.load(cp + vf * 3 + 0, mask=mask_v, other=0.0).to(tl.float32)
        s_v1 = tl.load(cp + vf * 3 + 1, mask=mask_v, other=0.0).to(tl.float32)
        s_v2 = tl.load(cp + vf * 3 + 2, mask=mask_v, other=0.0).to(tl.float32)
    acc_q += (
        s_q0 * tl.load(conv_w + qf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
        + s_q1 * tl.load(conv_w + qf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
        + s_q2 * tl.load(conv_w + qf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    )
    acc_k += (
        s_k0 * tl.load(conv_w + kf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
        + s_k1 * tl.load(conv_w + kf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
        + s_k2 * tl.load(conv_w + kf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    )
    acc_v += (
        s_v0 * tl.load(conv_w + vf * 4 + 0, mask=mask_v, other=0.0).to(tl.float32)
        + s_v1 * tl.load(conv_w + vf * 4 + 1, mask=mask_v, other=0.0).to(tl.float32)
        + s_v2 * tl.load(conv_w + vf * 4 + 2, mask=mask_v, other=0.0).to(tl.float32)
    )
    b_q = acc_q * tl.sigmoid(acc_q)  # silu
    b_k = acc_k * tl.sigmoid(acc_k)
    b_v = acc_v * tl.sigmoid(acc_v)

    # conv state update (shift window); q/k dupes across NV write same values.
    if b_write >= 0:
        cw = conv_pool + b_write * stride_conv_page
        tl.store(cw + qf * 3 + 0, s_q1.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + qf * 3 + 1, s_q2.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + qf * 3 + 2, x_q.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 0, s_k1.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 1, s_k2.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 2, x_k.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + vf * 3 + 0, s_v1.to(cw.dtype.element_ty), mask=mask_v)
        tl.store(cw + vf * 3 + 1, s_v2.to(cw.dtype.element_ty), mask=mask_v)
        tl.store(cw + vf * 3 + 2, x_v.to(cw.dtype.element_ty), mask=mask_v)

    # --- fused f_b: g_raw[c] = w_fb[c, :] . f_a for this head's K features ---
    o_fa = tl.arange(0, D_FA)
    fa = tl.load(f_a + bos * stride_fa_tok + o_fa).to(tl.float32)
    gc = i_hv * K + o_k  # gate feature = same head slice of P
    wfb = tl.load(
        w_fb + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_g = tl.sum(wfb * fa[None, :], axis=1)

    # --- recurrence (T=1 decode) ---
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_A = tl.load(A_log + i_hv).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias
    if USE_LOWER_BOUND:
        b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
    else:
        b_gk = -tl.exp(b_A) * tl.where(
            b_g < 20.0, tl.math.log(1 + tl.math.exp(b_g)), b_g
        )

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_h0 = (
        h_pool
        + b_read * stride_state_page
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h += tl.load(p_h0, mask=mask_h & read_ok, other=0.0).to(tl.float32)

    b_h *= tl.exp(b_gk[:, None])
    b_v = b_v - tl.sum(b_h * b_k[:, None], 0)
    b_beta = tl.load(beta + bos * stride_beta_tok + i_hv).to(tl.float32)
    b_beta = tl.sigmoid(b_beta)
    b_v *= b_beta
    b_h += b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], 0)
    tl.store(
        o + (bos * HV + i_hv) * V + o_v,
        b_o.to(o.dtype.element_ty),
        mask=mask_v,
    )
    p_ht = (
        h_pool
        + b_write * stride_state_page
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & (b_write >= 0))


@triton.heuristics({"USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None})
@triton.jit
def fused_recurrent_kda_verify_megafuse_fwd_kernel(
    qkv_raw,  # [N*T, 3*P] pre-conv packed projections (token-strided)
    conv_w,  # [3*P, 4] fused conv bank
    conv_pool,  # [pages, 3*P, 3] committed conv state (read-only here)
    conv_out,  # [rows, 3*P, 3] per-position conv windows (verify scratch)
    f_a,  # [N*T, D_FA] low-rank gate input
    w_fb,  # [P, D_FA] f_b weight
    beta,
    A_log,
    dt_bias,
    o,
    h_pool,  # committed recurrent state (read-only here)
    h_pool_out,  # per-position recurrent states (verify scratch)
    read_indices,  # [N] committed page per request (-1 = fresh)
    write_indices,  # [N*T] scratch rows, request-major
    lower_bound,
    stride_raw_tok: tl.constexpr,
    stride_fa_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    P: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_state_out_page: tl.constexpr,
    stride_conv_page: tl.constexpr,
    stride_conv_out_page: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """Target-verify variant of the KDA megafusion: conv(+silu), the f_b gate
    GEMV, and the delta-rule recurrence run per draft position, with BOTH the
    rolled conv window and the evolved recurrent state stored to their verify
    scratch row after every step (``write_indices[n*T + t]``) so a partial
    accept can commit the state at the accepted position."""
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = i_n * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_read = tl.load(read_indices + i_n).to(tl.int64)
    read_ok = b_read >= 0

    qf = i_h * K + o_k
    kf = P + i_h * K + o_k
    vf = 2 * P + i_hv * V + o_v

    # conv taps (weights are loop-invariant)
    w_q0 = tl.load(conv_w + qf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_q1 = tl.load(conv_w + qf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_q2 = tl.load(conv_w + qf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_q3 = tl.load(conv_w + qf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    w_k0 = tl.load(conv_w + kf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_k1 = tl.load(conv_w + kf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_k2 = tl.load(conv_w + kf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_k3 = tl.load(conv_w + kf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    w_v0 = tl.load(conv_w + vf * 4 + 0, mask=mask_v, other=0.0).to(tl.float32)
    w_v1 = tl.load(conv_w + vf * 4 + 1, mask=mask_v, other=0.0).to(tl.float32)
    w_v2 = tl.load(conv_w + vf * 4 + 2, mask=mask_v, other=0.0).to(tl.float32)
    w_v3 = tl.load(conv_w + vf * 4 + 3, mask=mask_v, other=0.0).to(tl.float32)

    # initial conv window from the committed page (zeros for fresh requests)
    s_q0 = tl.zeros([BK], dtype=tl.float32)
    s_q1 = tl.zeros([BK], dtype=tl.float32)
    s_q2 = tl.zeros([BK], dtype=tl.float32)
    s_k0 = tl.zeros([BK], dtype=tl.float32)
    s_k1 = tl.zeros([BK], dtype=tl.float32)
    s_k2 = tl.zeros([BK], dtype=tl.float32)
    s_v0 = tl.zeros([BV], dtype=tl.float32)
    s_v1 = tl.zeros([BV], dtype=tl.float32)
    s_v2 = tl.zeros([BV], dtype=tl.float32)
    if read_ok:
        cp = conv_pool + b_read * stride_conv_page
        s_q0 = tl.load(cp + qf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_q1 = tl.load(cp + qf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_q2 = tl.load(cp + qf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_k0 = tl.load(cp + kf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_k1 = tl.load(cp + kf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_k2 = tl.load(cp + kf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_v0 = tl.load(cp + vf * 3 + 0, mask=mask_v, other=0.0).to(tl.float32)
        s_v1 = tl.load(cp + vf * 3 + 1, mask=mask_v, other=0.0).to(tl.float32)
        s_v2 = tl.load(cp + vf * 3 + 2, mask=mask_v, other=0.0).to(tl.float32)

    # initial recurrent state from the committed page
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_h0 = (
        h_pool
        + b_read * stride_state_page
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h += tl.load(p_h0, mask=mask_h & read_ok, other=0.0).to(tl.float32)

    b_A = tl.load(A_log + i_hv).to(tl.float32)
    o_fa = tl.arange(0, D_FA)
    gc = i_hv * K + o_k
    wfb = tl.load(
        w_fb + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )

    for i_t in range(T):
        tok = bos + i_t
        x_q = tl.load(qkv_raw + tok * stride_raw_tok + qf, mask=mask_k, other=0.0).to(
            tl.float32
        )
        x_k = tl.load(qkv_raw + tok * stride_raw_tok + kf, mask=mask_k, other=0.0).to(
            tl.float32
        )
        x_v = tl.load(qkv_raw + tok * stride_raw_tok + vf, mask=mask_v, other=0.0).to(
            tl.float32
        )

        acc_q = x_q * w_q3 + s_q0 * w_q0 + s_q1 * w_q1 + s_q2 * w_q2
        acc_k = x_k * w_k3 + s_k0 * w_k0 + s_k1 * w_k1 + s_k2 * w_k2
        acc_v = x_v * w_v3 + s_v0 * w_v0 + s_v1 * w_v1 + s_v2 * w_v2
        b_q = acc_q * tl.sigmoid(acc_q)
        b_k = acc_k * tl.sigmoid(acc_k)
        b_v = acc_v * tl.sigmoid(acc_v)

        # roll the window: after consuming x_t the taps are (x_{t-2}, x_{t-1}, x_t)
        s_q0, s_q1, s_q2 = s_q1, s_q2, x_q
        s_k0, s_k1, s_k2 = s_k1, s_k2, x_k
        s_v0, s_v1, s_v2 = s_v1, s_v2, x_v

        b_write = tl.load(write_indices + i_n * T + i_t).to(tl.int64)
        write_ok = b_write >= 0
        # per-position conv window (q/k dupes across NV write same values)
        cw = conv_out + b_write * stride_conv_out_page
        tl.store(cw + qf * 3 + 0, s_q0.to(cw.dtype.element_ty), mask=mask_k & write_ok)
        tl.store(cw + qf * 3 + 1, s_q1.to(cw.dtype.element_ty), mask=mask_k & write_ok)
        tl.store(cw + qf * 3 + 2, s_q2.to(cw.dtype.element_ty), mask=mask_k & write_ok)
        tl.store(cw + kf * 3 + 0, s_k0.to(cw.dtype.element_ty), mask=mask_k & write_ok)
        tl.store(cw + kf * 3 + 1, s_k1.to(cw.dtype.element_ty), mask=mask_k & write_ok)
        tl.store(cw + kf * 3 + 2, s_k2.to(cw.dtype.element_ty), mask=mask_k & write_ok)
        tl.store(cw + vf * 3 + 0, s_v0.to(cw.dtype.element_ty), mask=mask_v & write_ok)
        tl.store(cw + vf * 3 + 1, s_v1.to(cw.dtype.element_ty), mask=mask_v & write_ok)
        tl.store(cw + vf * 3 + 2, s_v2.to(cw.dtype.element_ty), mask=mask_v & write_ok)

        # f_b gate GEMV for this token
        fa = tl.load(f_a + tok * stride_fa_tok + o_fa).to(tl.float32)
        b_g = tl.sum(wfb * fa[None, :], axis=1)
        if HAS_DT_BIAS:
            b_g = b_g + b_bias
        if USE_LOWER_BOUND:
            b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
        else:
            b_gk = -tl.exp(b_A) * tl.where(
                b_g < 20.0, tl.math.log(1 + tl.math.exp(b_g)), b_g
            )

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h *= tl.exp(b_gk[:, None])
        b_v = b_v - tl.sum(b_h * b_k[:, None], 0)
        b_beta = tl.load(beta + tok * stride_beta_tok + i_hv).to(tl.float32)
        b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(
            o + (tok * HV + i_hv) * V + o_v,
            b_o.to(o.dtype.element_ty),
            mask=mask_v,
        )

        p_ht = (
            h_pool_out
            + b_write * stride_state_out_page
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & write_ok)


def fused_recurrent_kda_verify_megafuse(
    qkv_raw: torch.Tensor,
    conv_w: torch.Tensor,
    conv_pool: torch.Tensor,
    conv_out: torch.Tensor,
    f_a: torch.Tensor,
    w_fb: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    h_pool: torch.Tensor,
    h_pool_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    scale: float | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Target-verify KDA megafusion: conv1d(+silu), f_b gate GEMV, and the
    per-position recurrence in one launch, with per-position conv windows and
    recurrent states stored to the verify scratch for partial-accept commit.

    Args:
        qkv_raw: ``[N*T, 3*P]`` pre-conv packed q|k|v, request-major.
        conv_w: ``[3*P, 4]`` fused conv kernel bank (contiguous).
        conv_pool: ``[pages, 3*P, 3]`` committed conv state (read-only).
        conv_out: ``[rows, 3*P, 3]`` verify conv scratch (per-position writes).
        f_a: ``[N*T, D]`` gate input; w_fb: ``[P, D]`` up weight.
        beta: ``[N*T, HV]`` raw logits (sigmoid in-kernel).
        h_pool / h_pool_out: committed recurrent slab / verify scratch.
        read_indices: ``[N]`` committed page per request (-1 = fresh).
        write_indices: ``[N, T]`` or ``[N*T]`` scratch row ids.
        num_heads/head_dim: per-rank head geometry (P = num_heads*head_dim).
        draft_token_num: draft positions per request (T).

    Returns:
        o: ``[N*T, HV, V]`` attention output in ``qkv_raw``'s dtype.
    """
    total = qkv_raw.shape[0]
    T = draft_token_num
    N = total // T
    HV = num_heads
    K = V = head_dim
    P = HV * K
    D = f_a.shape[-1]
    if scale is None:
        scale = K**-0.5
    assert total == N * T
    assert qkv_raw.stride(-1) == 1 and conv_w.is_contiguous() and w_fb.is_contiguous()
    assert write_indices.numel() == N * T
    out = torch.empty(total, HV, V, dtype=qkv_raw.dtype, device=qkv_raw.device)
    BV = 32
    grid = (triton.cdiv(V, BV) * N * HV,)
    fused_recurrent_kda_verify_megafuse_fwd_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_w,
        conv_pool=conv_pool,
        conv_out=conv_out,
        f_a=f_a,
        w_fb=w_fb,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h_pool=h_pool,
        h_pool_out=h_pool_out,
        read_indices=read_indices,
        write_indices=write_indices.reshape(-1),
        lower_bound=lower_bound,
        stride_raw_tok=qkv_raw.stride(0),
        stride_fa_tok=f_a.stride(0),
        stride_beta_tok=beta.stride(0),
        scale=scale,
        T=T,
        H=HV,
        HV=HV,
        K=K,
        V=V,
        P=P,
        D_FA=D,
        BK=triton.next_power_of_2(K),
        BV=BV,
        stride_state_page=h_pool.stride(0),
        stride_state_out_page=h_pool_out.stride(0),
        stride_conv_page=conv_pool.stride(0),
        stride_conv_out_page=conv_out.stride(0),
        HAS_DT_BIAS=dt_bias is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_recurrent_kda_megafuse(
    qkv_raw: torch.Tensor,
    conv_w: torch.Tensor,
    conv_pool: torch.Tensor,
    f_a: torch.Tensor,
    w_fb: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    h_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Single-step KDA decode with conv1d(+silu) and the f_b gate GEMV fused in.

    Args:
        qkv_raw: ``[T, 3*P]`` pre-conv packed q|k|v (token-strided slice ok).
        conv_w: ``[3*P, 4]`` fused conv kernel bank (contiguous).
        conv_pool: ``[pages, 3*P, 3]`` conv state (updated in place, dual-index).
        f_a: ``[T, D]`` low-rank decay-gate input; w_fb: ``[P, D]`` up weight.
        beta: ``[T, HV]`` raw logits (sigmoid in-kernel).
        h_pool / read_indices / write_indices: as in the pool kernel.
        num_heads/head_dim: per-rank head geometry (P = num_heads*head_dim).

    Returns:
        o: ``[T, HV, V]`` attention output (bf16).
    """
    T = qkv_raw.shape[0]
    HV = num_heads
    K = V = head_dim
    P = HV * K
    D = f_a.shape[-1]
    if scale is None:
        scale = K**-0.5
    assert qkv_raw.stride(-1) == 1 and conv_w.is_contiguous() and w_fb.is_contiguous()
    N = T if cu_seqlens is None else len(cu_seqlens) - 1
    out = torch.empty(T, HV, V, dtype=qkv_raw.dtype, device=qkv_raw.device)
    # One program per (request, head): NV == 1 leaves the q/k taps unshared.
    BV = triton.next_power_of_2(V)
    grid = (N * HV,)
    fused_recurrent_kda_megafuse_fwd_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_w,
        conv_pool=conv_pool,
        f_a=f_a,
        w_fb=w_fb,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h_pool=h_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        stride_raw_tok=qkv_raw.stride(0),
        stride_fa_tok=f_a.stride(0),
        stride_beta_tok=beta.stride(0),
        scale=scale,
        N=N,
        T=1,
        H=HV,
        HV=HV,
        K=K,
        V=V,
        P=P,
        D_FA=D,
        BK=triton.next_power_of_2(K),
        BV=BV,
        stride_state_page=h_pool.stride(0),
        stride_conv_page=conv_pool.stride(0),
        num_warps=4,
        num_stages=2,
    )
    return out
