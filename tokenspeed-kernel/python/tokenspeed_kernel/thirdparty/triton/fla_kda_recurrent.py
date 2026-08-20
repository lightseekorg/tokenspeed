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

``fused_recurrent_kda_megafuse`` additionally accepts a tokenspeed-only gated
RMSNorm epilogue (``output_gate``/``norm_weight``/``norm_eps``); with it unset
the kernel is unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.platform import current_platform

_IS_NVIDIA = current_platform().is_nvidia


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
        "FUSE_OUTPUT_NORM": lambda args: args["gate"] is not None,
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
    gate,  # tokenspeed extension: [T, HV*V] raw output-gate logits, or None
    norm_w,  # tokenspeed extension: [V] gated-RMSNorm weight, or None
    norm_eps,
    h_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    lower_bound,
    stride_raw_tok: tl.constexpr,
    stride_fa_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    stride_gate_tok: tl.constexpr,
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
    FUSE_OUTPUT_NORM: tl.constexpr,
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
    # tokenspeed extension: NV == 1, so b_o is the whole row the norm would reload.
    if FUSE_OUTPUT_NORM:
        b_rsig = tl.math.rsqrt(tl.sum(b_o * b_o) / V + norm_eps)
        b_nw = tl.load(norm_w + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_gate = tl.load(
            gate + bos * stride_gate_tok + i_hv * V + o_v, mask=mask_v, other=0.0
        ).to(tl.float32)
        b_o = b_o * b_rsig * b_nw * tl.sigmoid(b_gate)
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
    STORE_STATES: tl.constexpr,
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

        if STORE_STATES:
            b_write = tl.load(write_indices + i_n * T + i_t).to(tl.int64)
            write_ok = b_write >= 0
            # per-position conv window (q/k dupes across NV write same values)
            cw = conv_out + b_write * stride_conv_out_page
            tl.store(
                cw + qf * 3 + 0, s_q0.to(cw.dtype.element_ty), mask=mask_k & write_ok
            )
            tl.store(
                cw + qf * 3 + 1, s_q1.to(cw.dtype.element_ty), mask=mask_k & write_ok
            )
            tl.store(
                cw + qf * 3 + 2, s_q2.to(cw.dtype.element_ty), mask=mask_k & write_ok
            )
            tl.store(
                cw + kf * 3 + 0, s_k0.to(cw.dtype.element_ty), mask=mask_k & write_ok
            )
            tl.store(
                cw + kf * 3 + 1, s_k1.to(cw.dtype.element_ty), mask=mask_k & write_ok
            )
            tl.store(
                cw + kf * 3 + 2, s_k2.to(cw.dtype.element_ty), mask=mask_k & write_ok
            )
            tl.store(
                cw + vf * 3 + 0, s_v0.to(cw.dtype.element_ty), mask=mask_v & write_ok
            )
            tl.store(
                cw + vf * 3 + 1, s_v1.to(cw.dtype.element_ty), mask=mask_v & write_ok
            )
            tl.store(
                cw + vf * 3 + 2, s_v2.to(cw.dtype.element_ty), mask=mask_v & write_ok
            )

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

        if STORE_STATES:
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
    store_states: bool = True,
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
        store_states: Whether to write per-position conv and recurrent tapes.

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
    assert not store_states or write_indices.numel() == N * T
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
        STORE_STATES=store_states,
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
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
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
        output_gate: optional ``[T, HV*V]`` (or ``[T, HV, V]``) raw gate logits;
            a row-strided slice of a wider packed projection is accepted.
            Supplying it, with ``norm_weight`` and ``norm_eps``, folds the gated
            RMSNorm epilogue ``rmsnorm(o)*norm_weight*sigmoid(gate)`` into this
            kernel instead of running it as a second pass.
        norm_weight: ``[V]`` RMSNorm weight; required with ``output_gate``.
        norm_eps: RMSNorm epsilon; required with ``output_gate``.

    Returns:
        o: ``[T, HV, V]`` attention output (bf16), normalized and gated when
        ``output_gate`` is given.
    """
    T = qkv_raw.shape[0]
    HV = num_heads
    K = V = head_dim
    P = HV * K
    D = f_a.shape[-1]
    if scale is None:
        scale = K**-0.5
    assert qkv_raw.stride(-1) == 1 and conv_w.is_contiguous() and w_fb.is_contiguous()
    if (output_gate is None) != (norm_weight is None):
        raise ValueError("output_gate and norm_weight must be given together")
    if output_gate is not None:
        if norm_eps is None:
            raise ValueError("norm_eps is required with output_gate")
        if output_gate.stride(-1) != 1 or output_gate.numel() != T * HV * V:
            raise ValueError(
                f"output_gate must be row-contiguous over {T * HV * V} elements, "
                f"got shape {tuple(output_gate.shape)} stride "
                f"{tuple(output_gate.stride())}"
            )
        if output_gate.dim() == 3 and output_gate.stride(1) != V:
            raise ValueError("output_gate heads must be V-strided")
        if norm_weight.numel() != V:
            raise ValueError(f"norm_weight must be [{V}], got {norm_weight.shape}")
        stride_gate_tok = output_gate.stride(0)
    else:
        stride_gate_tok = 0
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
        gate=output_gate,
        norm_w=norm_weight,
        norm_eps=0.0 if norm_eps is None else norm_eps,
        stride_gate_tok=stride_gate_tok,
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


@triton.jit
def _kda_window_step(
    qkv_raw,
    beta,
    tok,
    stride_raw_tok,
    stride_beta_tok,
    i_hv,
    qf,
    kf,
    vf,
    mask_k,
    mask_v,
    w_q0,
    w_q1,
    w_q2,
    w_q3,
    w_k0,
    w_k1,
    w_k2,
    w_k3,
    w_v0,
    w_v1,
    w_v2,
    w_v3,
    s_q0,
    s_q1,
    s_q2,
    s_k0,
    s_k1,
    s_k2,
    s_v0,
    s_v1,
    s_v2,
    b_h,
    g_at,
    scale,
    COMPUTE_OUT: tl.constexpr,
):
    """One KDA token step: conv(+silu), f_b gate GEMV, delta-rule update.

    The single source of the per-token math. Every consumer -- plain verify,
    standalone replay, and the replay prefix fused into the next verify --
    inlines this same body, so their states cannot drift apart numerically.

    Returns the rolled conv window registers, the updated state, and the
    attention output (zeros unless ``COMPUTE_OUT``; replayed positions never
    need it, and skipping it drops the output dot from the critical path).
    """
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

    # f_b gate for this token: read it, or run the GEMV inline. Reading wins
    # whenever the caller can amortize the [K, D_FA] weight tile -- held live
    # across the recurrence it costs 128 registers per thread and spills.
    b_gk = tl.load(g_at, mask=mask_k, other=0.0)

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_h *= tl.exp(b_gk[:, None])
    b_v = b_v - tl.sum(b_h * b_k[:, None], 0)
    b_beta = tl.load(beta + tok * stride_beta_tok + i_hv).to(tl.float32)
    b_beta = tl.sigmoid(b_beta)
    b_v *= b_beta
    b_h += b_k[:, None] * b_v[None, :]
    b_o = tl.zeros_like(b_v)
    if COMPUTE_OUT:
        b_o = tl.sum(b_h * b_q[:, None], 0)
    return s_q0, s_q1, s_q2, s_k0, s_k1, s_k2, s_v0, s_v1, s_v2, b_h, b_o


@triton.jit
def fused_recurrent_kda_window_fwd_kernel(
    qkv_raw,  # [N*T, 3*P] pre-conv packed projections (token-strided)
    conv_w,  # [3*P, 4] fused conv bank
    conv_pool,  # [pages, 3*P, 3] committed conv state (read-only here)
    beta,
    o,  # [N*T, HV, V] per-position output (WRITE_OUTPUT only)
    h_pool,  # committed recurrent state (read-only here)
    h_pool_out,  # recurrent commit destination (STORE_FINAL / replay prefix)
    read_indices,  # [N] anchor page per request (-1 = fresh)
    write_indices,  # [N] destination page per request (STORE_FINAL only)
    n_steps,  # [N] tokens to consume per request (HAS_N_STEPS only)
    prev_qkv,  # previous window's payload (HAS_REPLAY_PREFIX only)
    prev_beta,
    prev_base,  # [N] base payload row of the pending window (-1 = none)
    prev_steps,  # [N] accepted tokens of the pending window
    commit_indices,  # [N] page for the deferred state commit (-1 = skip)
    stride_raw_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    stride_prev_raw_tok: tl.constexpr,
    stride_prev_beta_tok: tl.constexpr,
    g,
    prev_g,
    stride_g_tok: tl.constexpr,
    stride_prev_g_tok: tl.constexpr,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    P: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_state_out_page: tl.constexpr,
    stride_conv_page: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    HAS_N_STEPS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    HAS_REPLAY_PREFIX: tl.constexpr,
):
    """Run a window of KDA decode steps from a committed page: conv(+silu),
    the f_b gate GEMV, and the delta-rule recurrence, per token.

    One kernel serves every half of speculative decoding, so the paths can
    never drift apart numerically:

    - **target verify** (``WRITE_OUTPUT``, no ``STORE_FINAL``) consumes all
      ``T`` draft positions and emits their outputs. The evolved state stays
      in registers and is thrown away: verification is tentative, and its
      final state ``S_{N+T}`` is only correct if every draft token is
      accepted.
    - **replay commit** (``STORE_FINAL``, ``HAS_N_STEPS``, no output)
      re-consumes the first ``n_steps[n]`` positions of the same window from
      the same committed page, and stores the resulting conv window and
      recurrent state to ``write_indices[n]``. That reconstructs exactly the
      state a non-speculative decode of the accepted tokens would have
      reached, without verification ever having to materialize (and the
      caller ever having to keep) a state per draft position.
    - **fused replay + verify** (``HAS_REPLAY_PREFIX`` + ``WRITE_OUTPUT``)
      folds the previous round's replay commit into this round's verify:
      it anchors at the page from BEFORE the previous window, replays that
      window's accepted prefix from its captured payload (register-resident,
      no output), stores the result to ``commit_indices[n]`` -- which IS the
      previous round's deferred commit -- and rolls straight into the new
      window's ``T`` verify steps. Relative to a standalone replay followed
      by a verify this saves one full state read plus one kernel's fixed
      front per layer per round, and it matches that pair up to compiler
      FMA contraction (~1 ulp fp32; see the test header): the committed
      fp32 state merely stays in registers instead of round-tripping
      through memory, and the conv window entries are raw bf16 either way. ``prev_base[n] < 0`` (no pending window: a fresh
      request, or a padded slot) degenerates to a plain verify.

    ``n_steps[n] == 0`` / ``prev_steps[n] == 0`` are meaningful and must stay
    supported: they commit the anchor state unchanged, which is what an
    all-rejected window needs when the destination page differs from the
    source page.

    Page-ownership precondition (NOT checked in-kernel): every commit /
    write page must be exclusively owned by its request for this launch.
    Two rows committing to one page interleave per-program slices into a
    torn mixture, and one row's commit page doubling as another row's
    anchor makes the reader's view scheduling-defined (old or new state
    depending on grid position). The FlatKV pager upholds exclusivity by
    construction; any other caller must too.
    """
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
    # The state tile is touched exactly once per round; keeping it out of L2
    # leaves the cache to the MoE GEMM weights this kernel runs beside.
    b_h += tl.load(
        p_h0, mask=mask_h & read_ok, other=0.0, eviction_policy="evict_first"
    ).to(tl.float32)
    if ENABLE_PDL:
        # State and conv taps are independent of the gate, so they load while
        # the precompute is still running; only the gate reads below need it.
        tl.extra.cuda.gdc_wait()

    g_off = i_hv * K + o_k

    if HAS_REPLAY_PREFIX:
        # Deferred commit: replay the accepted prefix from the anchor, publish, continue -- state and conv window stay in registers across the seam.
        p_base = tl.load(prev_base + i_n).to(tl.int64)
        r_steps = 0
        if p_base >= 0:
            # prev_steps uses the previous window; staging flushes mismatched windows, so prev_steps <= T.
            r_steps = tl.minimum(
                tl.maximum(tl.load(prev_steps + i_n).to(tl.int32), 0), T
            )
        for i_t in range(r_steps):
            ptok = p_base + i_t
            (
                s_q0,
                s_q1,
                s_q2,
                s_k0,
                s_k1,
                s_k2,
                s_v0,
                s_v1,
                s_v2,
                b_h,
                _,
            ) = _kda_window_step(
                prev_qkv,
                prev_beta,
                ptok,
                stride_prev_raw_tok,
                stride_prev_beta_tok,
                i_hv,
                qf,
                kf,
                vf,
                mask_k,
                mask_v,
                w_q0,
                w_q1,
                w_q2,
                w_q3,
                w_k0,
                w_k1,
                w_k2,
                w_k3,
                w_v0,
                w_v1,
                w_v2,
                w_v3,
                s_q0,
                s_q1,
                s_q2,
                s_k0,
                s_k1,
                s_k2,
                s_v0,
                s_v1,
                s_v2,
                b_h,
                prev_g + (i_n * T + i_t) * stride_prev_g_tok + g_off,
                scale,
                COMPUTE_OUT=False,
            )
        # Gated on BOTH indices: p_base < 0 must write nothing even if the commit slot holds a stale page id.
        b_commit = tl.load(commit_indices + i_n).to(tl.int64)
        if p_base >= 0 and b_commit >= 0:
            # In-place safe: each program touches only its own [BK, BV] slice; the conv window commits via kda_commit_conv_window.
            p_hc = (
                h_pool_out
                + b_commit * stride_state_out_page
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_hc, b_h.to(p_hc.dtype.element_ty), mask=mask_h)

    steps = T
    if HAS_N_STEPS:
        # Clamped by the caller; a request may consume fewer than T tokens.
        steps = tl.load(n_steps + i_n).to(tl.int32)

    for i_t in range(steps):
        tok = bos + i_t
        (
            s_q0,
            s_q1,
            s_q2,
            s_k0,
            s_k1,
            s_k2,
            s_v0,
            s_v1,
            s_v2,
            b_h,
            b_o,
        ) = _kda_window_step(
            qkv_raw,
            beta,
            tok,
            stride_raw_tok,
            stride_beta_tok,
            i_hv,
            qf,
            kf,
            vf,
            mask_k,
            mask_v,
            w_q0,
            w_q1,
            w_q2,
            w_q3,
            w_k0,
            w_k1,
            w_k2,
            w_k3,
            w_v0,
            w_v1,
            w_v2,
            w_v3,
            s_q0,
            s_q1,
            s_q2,
            s_k0,
            s_k1,
            s_k2,
            s_v0,
            s_v1,
            s_v2,
            b_h,
            g + tok * stride_g_tok + g_off,
            scale,
            COMPUTE_OUT=WRITE_OUTPUT,
        )
        if WRITE_OUTPUT:
            tl.store(
                o + (tok * HV + i_hv) * V + o_v,
                b_o.to(o.dtype.element_ty),
                mask=mask_v,
            )

    if STORE_FINAL:
        b_write = tl.load(write_indices + i_n).to(tl.int64)
        write_ok = b_write >= 0
        # Recurrent state only; disjoint [BK, BV] slices make in-place writes safe.
        p_ht = (
            h_pool_out
            + b_write * stride_state_out_page
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(
            p_ht,
            b_h.to(p_ht.dtype.element_ty),
            mask=mask_h & write_ok,
            eviction_policy="evict_first",
        )
    if ENABLE_PDL:
        # State and output are visible; the conv-window commit may begin.
        tl.extra.cuda.gdc_launch_dependents()


@triton.heuristics(
    {
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit
def kda_gate_precompute_kernel(
    f_a,
    prev_f_a,
    w_fb,
    A_log,
    dt_bias,
    src_rows,
    gk_out,
    lower_bound,
    stride_fa_tok,
    stride_prev_fa_tok,
    stride_gk_tok,
    rows,
    split,
    T: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    INDEXED_SRC: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Materialize the f_b gate for every token: ``gk[row, i_hv*K + k]``.

    One program per (head, token block) loads the head's ``[K, D_FA]`` weight
    tile once and reuses it for BT tokens, which is exactly the amortization
    the recurrence kernel cannot do -- there the tile stays live across the
    whole state loop and spills.

    The reduction is the same ``tl.sum(wfb * fa[None, :], axis=1)`` over the
    same tile shape as the in-kernel GEMV, so both paths agree bit for bit.
    ``INDEXED_SRC`` reads row ``src_rows[row // T] + row % T`` instead of
    ``row``, which is how the replay prefix addresses the payload ring.
    """
    if ENABLE_PDL:
        # f_a comes from the projection immediately upstream; every launch_pdl
        # kernel here pairs the launch with a wait before touching producer data.
        tl.extra.cuda.gdc_wait()
    i_hv = tl.program_id(0)
    i_tb = tl.program_id(1)
    i_kb = tl.program_id(2)
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    wfb = tl.load(
        w_fb + gc[:, None] * D_FA + o_fa[None, :], mask=mask_k[:, None], other=0.0
    ).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = 0.0
    if HAS_DT_BIAS:
        b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
    # Splitting K only narrows the tile each program holds; every row's
    # reduction still runs over the whole D_FA axis, so the result is
    # unchanged bit for bit.
    for j in range(BT):
        row = i_tb * BT + j
        if row < rows:
            if INDEXED_SRC and row >= split:
                # Replay half: rows address the payload ring through the
                # per-request base. Rows without a pending window read row 0
                # and are ignored -- the consumer never replays them.
                r = row - split
                src = tl.maximum(tl.load(src_rows + r // T).to(tl.int64), 0) + r % T
                fa = tl.load(prev_f_a + src * stride_prev_fa_tok + o_fa).to(tl.float32)
            else:
                fa = tl.load(f_a + row * stride_fa_tok + o_fa).to(tl.float32)
            b_g = tl.sum(wfb * fa[None, :], axis=1)
            if HAS_DT_BIAS:
                b_g = b_g + b_bias
            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
            else:
                b_gk = -tl.exp(b_A) * tl.where(
                    b_g < 20.0, tl.math.log(1 + tl.math.exp(b_g)), b_g
                )
            tl.store(gk_out + row * stride_gk_tok + gc, b_gk, mask=mask_k)
    if ENABLE_PDL:
        # The recurrence can start loading state the moment these land.
        tl.extra.cuda.gdc_launch_dependents()


_SM_COUNT: dict[int, int] = {}


def _gate_tiling(rows: int, num_heads: int, head_dim: int, device):
    """Rows and gate channels per program, widest grid that still fills the GPU.

    Both knobs only decide which rows a program owns -- every row still
    reduces over the whole D_FA axis -- so this never moves a result. What it
    does move is whether the weight-tile load has any parallelism to hide
    behind: at one request the widest tiling leaves 48 programs and the kernel
    costs 12.6us for eight rows of work, against 6.1us once the grid covers
    the machine. At 256 rows both land at 14.3us.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SM_COUNT.get(index)
    if sms is None:
        sms = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = sms
    for block_t, block_k in ((8, 32), (4, 32), (2, 16), (1, 16)):
        programs = (
            num_heads * triton.cdiv(rows, block_t) * triton.cdiv(head_dim, block_k)
        )
        if programs >= 4 * sms:
            return block_t, min(block_k, triton.next_power_of_2(head_dim))
    return 1, min(16, triton.next_power_of_2(head_dim))


def kda_gate_precompute(
    f_a: torch.Tensor,
    w_fb: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    gk_out: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    prev_f_a: torch.Tensor | None = None,
    prev_base: torch.Tensor | None = None,
    draft_token_num: int = 1,
    enable_pdl: bool = False,
) -> None:
    """Fill ``gk_out`` with the per-token f_b gate.

    Rows ``[0, split)`` come from ``f_a`` packed token-major; when a replay
    payload is given, rows ``[split, 2*split)`` come from ``prev_f_a`` at
    ``prev_base[request] + position``. Both halves ride one launch because at
    these sizes the kernel is launch-bound, not work-bound.

    Args:
        f_a: ``[tokens, D_FA]`` this round's low-rank gate input.
        w_fb: ``[HV*K, D_FA]`` f_b weight.
        A_log: ``[HV]`` per-head log decay; dt_bias: ``[HV*K]`` or None.
        gk_out: ``[rows, HV*K]`` fp32 destination.
        num_heads: value heads HV; head_dim: K.
        lower_bound: safe-gate lower bound, or None for the softplus form.
        prev_f_a: payload ring holding the pending window's gate inputs.
        prev_base: ``[requests]`` first ring row per request.
        draft_token_num: rows per request in the replay half.

    Returns:
        None. ``gk_out`` is written in place.
    """
    enable_pdl = bool(enable_pdl and _IS_NVIDIA)
    rows = gk_out.shape[0]
    if rows == 0:
        return
    indexed = prev_base is not None
    split = rows // 2 if indexed else rows
    block_t, block_k = _gate_tiling(rows, num_heads, head_dim, gk_out.device)
    kda_gate_precompute_kernel[
        (num_heads, triton.cdiv(rows, block_t), triton.cdiv(head_dim, block_k))
    ](
        f_a=f_a,
        prev_f_a=prev_f_a if indexed else f_a,
        w_fb=w_fb,
        A_log=A_log,
        dt_bias=dt_bias,
        src_rows=prev_base if indexed else f_a,
        gk_out=gk_out,
        lower_bound=lower_bound,
        stride_fa_tok=f_a.stride(0),
        stride_prev_fa_tok=prev_f_a.stride(0) if indexed else 0,
        stride_gk_tok=gk_out.stride(0),
        rows=rows,
        split=split,
        T=draft_token_num,
        K=head_dim,
        D_FA=f_a.shape[-1],
        BK=block_k,
        BT=block_t,
        INDEXED_SRC=indexed,
        ENABLE_PDL=enable_pdl,
        num_warps=1 if block_t >= 4 else 2,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


# Gate scratch, one buffer per device, grown only outside CUDA-graph capture:
# a captured graph bakes the address, so a later reallocation would leave it
# pointing at freed memory.
_GATE_SCRATCH: dict[int, torch.Tensor] = {}


def kda_gate_scratch(rows: int, width: int, device) -> torch.Tensor:
    """Fallback gate buffer for callers that bring none of their own.

    The runtime passes ``gate_scratch`` carved from its shared workspace
    pool, whose freeze handles graph-address stability; this module-local
    buffer serves direct kernel callers (tests, sweeps) only.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    buf = _GATE_SCRATCH.get(index)
    if buf is None or buf.shape[0] < rows or buf.shape[1] != width:
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "KDA gate scratch must be reserved before graph capture"
        with torch.inference_mode(False):
            buf = torch.empty((rows, width), dtype=torch.float32, device=device)
        _GATE_SCRATCH[index] = buf
    return buf[:rows]


# (BV, num_warps, num_stages). Splitting V costs redundancy -- every block
# re-loads its (request, head) conv window -- so the tile is as wide as the
# grid allows. Once the f_b gate moved out of the recurrence the spread across
# tiles collapsed to a few percent (BV=32/64/128 within 3% at every batch
# size), which is worth less than the property a single tile buys: BV changes
# how the [BK, BV] state tile is spread over threads, and with it the order of
# the reduction along K, so mixing tiles across rounds would make a request's
# committed state depend on the batch size it happened to run at.
_WINDOW_TILE = (64, 4, 2)  # the window kernel's (BV, num_warps, num_stages)


def _launch_kda_window(
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
    *,
    h_pool_out: torch.Tensor | None,
    write_indices: torch.Tensor | None,
    n_steps: torch.Tensor | None,
    out: torch.Tensor | None,
    pending: tuple | None = None,
    enable_pdl: bool = False,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    scale: float | None,
    lower_bound: float | None,
    gate_scratch: torch.Tensor | None = None,
) -> None:
    """Shared launch for the verify / replay / fused halves of the window
    kernel. ``pending`` is ``(prev_qkv, prev_f_a, prev_beta, prev_base,
    prev_steps, commit_indices, commit_pool)`` for the fused replay prefix.
    ``gate_scratch`` is caller-provided fp32 scratch for the hoisted gate
    (rows x ``num_heads*head_dim``, transient within this launch pair);
    ``None`` falls back to the module-local buffer."""
    enable_pdl = bool(enable_pdl and _IS_NVIDIA)
    total = qkv_raw.shape[0]
    T = draft_token_num
    N = total // T
    HV = num_heads
    K = V = head_dim
    if scale is None:
        scale = K**-0.5
    assert total == N * T
    assert qkv_raw.stride(-1) == 1 and conv_w.is_contiguous() and w_fb.is_contiguous()
    store_final = h_pool_out is not None
    assert store_final == (write_indices is not None)
    if store_final:
        assert write_indices.numel() == N
    if n_steps is not None:
        assert n_steps.numel() == N
    if pending is None:
        prev_qkv = prev_f_a = prev_beta = None
        prev_base = prev_steps = commit_indices = None
        commit_pool = h_pool_out
    else:
        assert not store_final, "fused replay commits via commit_indices"
        (
            prev_qkv,
            prev_f_a,
            prev_beta,
            prev_base,
            prev_steps,
            commit_indices,
            commit_pool,
        ) = pending
        assert prev_qkv.stride(-1) == 1
        for t in (prev_base, prev_steps, commit_indices):
            assert t.numel() == N
    BV, warps, stages = _WINDOW_TILE
    grid = (triton.cdiv(V, BV) * N * HV,)
    # Hoist the f_b gate: computed here once per token, it stops costing the
    # recurrence kernel a live [K, D_FA] register tile (and its spills).
    # Always hoisted -- measured faster at every size (8 rows: 36.9us vs
    # 39.1us inline; 256 rows: 74.8 vs 117.9), and bit-identical.
    rows = N * T * (2 if pending is not None else 1)
    if gate_scratch is not None:
        assert gate_scratch.shape[0] >= rows and gate_scratch.shape[1] == HV * K
        assert gate_scratch.dtype == torch.float32
        gate = gate_scratch[:rows]
    else:
        gate = kda_gate_scratch(rows, HV * K, qkv_raw.device)
    kda_gate_precompute(
        f_a,
        w_fb,
        A_log,
        dt_bias,
        gate,
        num_heads=HV,
        head_dim=K,
        lower_bound=lower_bound,
        prev_f_a=prev_f_a,
        prev_base=prev_base,
        draft_token_num=T,
        enable_pdl=enable_pdl,
    )
    g_cur = gate[: N * T]
    g_prev = gate[N * T :] if pending is not None else g_cur
    fused_recurrent_kda_window_fwd_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_w,
        conv_pool=conv_pool,
        beta=beta,
        o=out,
        h_pool=h_pool,
        h_pool_out=commit_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        n_steps=n_steps,
        prev_qkv=prev_qkv,
        prev_beta=prev_beta,
        prev_base=prev_base,
        prev_steps=prev_steps,
        commit_indices=commit_indices,
        stride_raw_tok=qkv_raw.stride(0),
        stride_beta_tok=beta.stride(0),
        stride_prev_raw_tok=prev_qkv.stride(0) if pending is not None else 0,
        stride_prev_beta_tok=prev_beta.stride(0) if pending is not None else 0,
        g=g_cur,
        prev_g=g_prev,
        stride_g_tok=g_cur.stride(0),
        stride_prev_g_tok=g_prev.stride(0),
        scale=scale,
        T=T,
        H=HV,
        HV=HV,
        K=K,
        V=V,
        P=HV * K,
        BK=triton.next_power_of_2(K),
        BV=BV,
        stride_state_page=h_pool.stride(0),
        stride_state_out_page=commit_pool.stride(0) if commit_pool is not None else 0,
        stride_conv_page=conv_pool.stride(0),
        WRITE_OUTPUT=out is not None,
        STORE_FINAL=store_final,
        HAS_N_STEPS=n_steps is not None,
        ENABLE_PDL=enable_pdl,
        HAS_REPLAY_PREFIX=pending is not None,
        num_warps=warps,
        num_stages=stages,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


def fused_recurrent_kda_replay_commit(
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
    accepted_length: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    scale: float | None = None,
    lower_bound: float | None = None,
    gate_scratch: torch.Tensor | None = None,
) -> None:
    """Commit the accepted prefix of a verified draft window by replaying it.

    Re-runs ``accepted_length[n]`` steps of the same window that
    ``fused_recurrent_kda_verify_megafuse`` just verified, from the same
    committed page, and stores the resulting conv window and recurrent state
    to ``write_indices[n]``. Because it is the same kernel doing the same
    arithmetic in the same order, the committed state is exactly the one a
    non-speculative decode of those tokens would have produced.

    The projections in ``qkv_raw`` / ``f_a`` / ``beta`` must be the ones the
    **target** model computed during verification, not the draft model's. For
    a position inside the accepted prefix every preceding position was also
    accepted, so the hidden state those projections came from is the true
    one; positions past the accepted prefix are stale and are never replayed.

    Args:
        qkv_raw: ``[N*T, 3*P]`` pre-conv packed q|k|v captured during verify.
        conv_w: ``[3*P, 4]`` fused conv kernel bank (contiguous).
        conv_pool: ``[pages, 3*P, 3]`` committed conv state (read).
        conv_out: conv destination slab; may alias ``conv_pool``.
        f_a: ``[N*T, D]`` gate input; w_fb: ``[P, D]`` up weight.
        beta: ``[N*T, HV]`` raw logits (sigmoid in-kernel).
        h_pool: ``[pages, HV, K, V]`` committed recurrent slab (read).
        h_pool_out: recurrent destination slab; may alias ``h_pool``.
        read_indices: ``[N]`` committed page per request (-1 = fresh).
        write_indices: ``[N]`` destination page per request (-1 skips).
        accepted_length: ``[N]`` tokens to replay, in ``[0, T]``.
        num_heads/head_dim: per-rank head geometry (P = num_heads*head_dim).
        draft_token_num: draft positions per request (T).
        scale: q scale; defaults to ``head_dim ** -0.5``.
        lower_bound: gate lower bound; ``None`` selects the softplus gate.
        gate_scratch: caller-provided fp32 gate scratch,
            ``[>= N*T, num_heads*head_dim]``; ``None`` falls back to a
            module-local buffer.

    Returns:
        None. The destination pages are written in place.
    """
    steps = accepted_length.to(torch.int32).clamp(0, draft_token_num)
    _launch_kda_window(
        qkv_raw,
        conv_w,
        conv_pool,
        f_a,
        w_fb,
        beta,
        A_log,
        dt_bias,
        h_pool,
        read_indices,
        h_pool_out=h_pool_out,
        write_indices=write_indices,
        n_steps=steps,
        out=None,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        scale=scale,
        lower_bound=lower_bound,
        gate_scratch=gate_scratch,
    )
    kda_commit_conv_window(
        qkv_raw,
        conv_pool,
        conv_out,
        read_indices,
        write_indices,
        steps,
        conv_dim=3 * num_heads * head_dim,
        draft_token_num=draft_token_num,
    )


@triton.jit
def batched_kda_gate_precompute_kernel(
    addresses,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Descriptor-table form of the established KDA gate precompute."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    ab = i_l * 10
    f_a = tl.load(addresses + ab + 3).to(tl.pointer_type(tl.bfloat16))
    f_b = tl.load(addresses + ab + 4).to(tl.pointer_type(tl.bfloat16))
    A_log = tl.load(addresses + ab + 6).to(tl.pointer_type(tl.float32))
    dt_bias = tl.load(addresses + ab + 7).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    for j in range(BT):
        row = i_tb * BT + j
        if row < rows:
            fa = tl.load(f_a + row * stride_fa + o_fa).to(tl.float32)
            gate = tl.sum(wfb * fa[None, :], axis=1) + b_bias
            gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
            tl.store(gate_scratch + row * stride_gate + gc, gate, mask=mask_k)


@triton.jit
def batched_recurrent_kda_replay_commit_kernel(
    addresses,
    read_indices,
    write_indices,
    accepted_length,
    B,
    T: tl.constexpr,
    STRIDE_QKV: tl.constexpr,
    STRIDE_CONV: tl.constexpr,
    STRIDE_BETA: tl.constexpr,
    STRIDE_STATE: tl.constexpr,
    STRIDE_GATE: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    LAYERS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    D_FA: tl.constexpr,
    BV: tl.constexpr,
):
    """Replay one KDA layer/request/head per program via pointer descriptors."""
    i_l, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # address columns: qkv, conv_w, conv_pool, f_a, f_b, beta, A, dt, state, gate
    ab = i_l * 10
    qkv = tl.load(addresses + ab + 0).to(tl.pointer_type(tl.bfloat16))
    conv_w = tl.load(addresses + ab + 1).to(tl.pointer_type(tl.bfloat16))
    conv_pool = tl.load(addresses + ab + 2).to(tl.pointer_type(tl.bfloat16))
    beta = tl.load(addresses + ab + 5).to(tl.pointer_type(tl.bfloat16))
    state = tl.load(addresses + ab + 8).to(tl.pointer_type(tl.float32))
    gate_scratch = tl.load(addresses + ab + 9).to(tl.pointer_type(tl.float32))
    group = i_l // LAYERS_PER_GROUP

    read_page = tl.load(read_indices + group * B + i_n).to(tl.int64)
    write_page = tl.load(write_indices + group * B + i_n).to(tl.int64)
    steps = tl.load(accepted_length + i_n).to(tl.int32)
    steps = tl.minimum(tl.maximum(steps, 0), T)
    read_ok, write_ok = read_page >= 0, write_page >= 0
    P: tl.constexpr = HV * K
    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    qf = i_hv * K + o_k
    kf = P + qf

    w_q0 = tl.load(conv_w + qf * CONV_WIDTH + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_q1 = tl.load(conv_w + qf * CONV_WIDTH + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_q2 = tl.load(conv_w + qf * CONV_WIDTH + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_q3 = tl.load(conv_w + qf * CONV_WIDTH + 3, mask=mask_k, other=0.0).to(tl.float32)
    w_k0 = tl.load(conv_w + kf * CONV_WIDTH + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_k1 = tl.load(conv_w + kf * CONV_WIDTH + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_k2 = tl.load(conv_w + kf * CONV_WIDTH + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_k3 = tl.load(conv_w + kf * CONV_WIDTH + 3, mask=mask_k, other=0.0).to(tl.float32)
    cp = conv_pool + read_page * STRIDE_CONV
    s_q0 = tl.load(cp + qf * (CONV_WIDTH - 1) + 0, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_q1 = tl.load(cp + qf * (CONV_WIDTH - 1) + 1, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_q2 = tl.load(cp + qf * (CONV_WIDTH - 1) + 2, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_k0 = tl.load(cp + kf * (CONV_WIDTH - 1) + 0, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_k1 = tl.load(cp + kf * (CONV_WIDTH - 1) + 1, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )
    s_k2 = tl.load(cp + kf * (CONV_WIDTH - 1) + 2, mask=mask_k & read_ok, other=0.0).to(
        tl.float32
    )

    # Keep the established BV=64 recurrence reduction order. Each program
    # owns both V tiles sequentially. Convolution publication follows in a
    # separate launch after every committed-window read has completed.
    for i_v in tl.range(0, K, BV, loop_unroll_factor=1):
        o_v = i_v + tl.arange(0, BV)
        mask_v = o_v < K
        mask_h = mask_k[:, None] & mask_v[None, :]
        vf = 2 * P + i_hv * K + o_v
        w_v0 = tl.load(conv_w + vf * CONV_WIDTH + 0, mask=mask_v, other=0.0).to(
            tl.float32
        )
        w_v1 = tl.load(conv_w + vf * CONV_WIDTH + 1, mask=mask_v, other=0.0).to(
            tl.float32
        )
        w_v2 = tl.load(conv_w + vf * CONV_WIDTH + 2, mask=mask_v, other=0.0).to(
            tl.float32
        )
        w_v3 = tl.load(conv_w + vf * CONV_WIDTH + 3, mask=mask_v, other=0.0).to(
            tl.float32
        )
        s_v0 = tl.load(
            cp + vf * (CONV_WIDTH - 1) + 0, mask=mask_v & read_ok, other=0.0
        ).to(tl.float32)
        s_v1 = tl.load(
            cp + vf * (CONV_WIDTH - 1) + 1, mask=mask_v & read_ok, other=0.0
        ).to(tl.float32)
        s_v2 = tl.load(
            cp + vf * (CONV_WIDTH - 1) + 2, mask=mask_v & read_ok, other=0.0
        ).to(tl.float32)
        ph = (
            state
            + read_page * STRIDE_STATE
            + i_hv * K * K
            + o_k[:, None] * K
            + o_v[None, :]
        )
        # Match fused_recurrent_kda_window_fwd_kernel's initialization exactly;
        # keeping the same TTIR expression preserves its fp32 replay bits.
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        b_h += tl.load(
            ph, mask=mask_h & read_ok, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)

        tq0, tq1, tq2 = s_q0, s_q1, s_q2
        tk0, tk1, tk2 = s_k0, s_k1, s_k2
        for i_t in range(steps):
            tok = i_n * T + i_t
            tq0, tq1, tq2, tk0, tk1, tk2, s_v0, s_v1, s_v2, b_h, _ = _kda_window_step(
                qkv,
                beta,
                tok,
                STRIDE_QKV,
                STRIDE_BETA,
                i_hv,
                qf,
                kf,
                vf,
                mask_k,
                mask_v,
                w_q0,
                w_q1,
                w_q2,
                w_q3,
                w_k0,
                w_k1,
                w_k2,
                w_k3,
                w_v0,
                w_v1,
                w_v2,
                w_v3,
                tq0,
                tq1,
                tq2,
                tk0,
                tk1,
                tk2,
                s_v0,
                s_v1,
                s_v2,
                b_h,
                gate_scratch + tok * STRIDE_GATE + qf,
                K**-0.5,
                COMPUTE_OUT=False,
            )
        po = (
            state
            + write_page * STRIDE_STATE
            + i_hv * K * K
            + o_k[:, None] * K
            + o_v[None, :]
        )
        tl.store(po, b_h, mask=mask_h & write_ok, eviction_policy="evict_first")


@triton.jit
def batched_kda_commit_conv_window_kernel(
    addresses,
    read_indices,
    write_indices,
    accepted_length,
    B,
    T: tl.constexpr,
    STRIDE_QKV: tl.constexpr,
    STRIDE_CONV: tl.constexpr,
    CONV_DIM: tl.constexpr,
    LAYERS_PER_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Publish convolution windows after every recurrent program has read."""
    i_l, i_n, i_cb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ab = i_l * 10
    qkv = tl.load(addresses + ab).to(tl.pointer_type(tl.bfloat16))
    conv_pool = tl.load(addresses + ab + 2).to(tl.pointer_type(tl.bfloat16))
    group = i_l // LAYERS_PER_GROUP
    read_page = tl.load(read_indices + group * B + i_n).to(tl.int64)
    write_page = tl.load(write_indices + group * B + i_n).to(tl.int64)
    steps = tl.load(accepted_length + i_n).to(tl.int32)
    steps = tl.minimum(tl.maximum(steps, 0), T)
    offsets = i_cb * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < CONV_DIM
    src = conv_pool + read_page * STRIDE_CONV + offsets * 3
    s0 = tl.load(src, mask=mask & (read_page >= 0), other=0.0)
    s1 = tl.load(src + 1, mask=mask & (read_page >= 0), other=0.0)
    s2 = tl.load(src + 2, mask=mask & (read_page >= 0), other=0.0)
    for i_t in range(steps):
        x = tl.load(
            qkv + (i_n * T + i_t) * STRIDE_QKV + offsets,
            mask=mask,
            other=0.0,
        )
        s0, s1, s2 = s1, s2, x
    dst = conv_pool + write_page * STRIDE_CONV + offsets * 3
    tl.store(dst, s0, mask=mask & (write_page >= 0))
    tl.store(dst + 1, s1, mask=mask & (write_page >= 0))
    tl.store(dst + 2, s2, mask=mask & (write_page >= 0))


def batched_recurrent_kda_replay_commit(
    addresses: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    f_a_dim: int,
    qkv_stride: int,
    conv_stride: int,
    f_a_stride: int,
    beta_stride: int,
    state_stride: int,
    gate_stride: int,
    conv_width: int,
    layers_per_group: int,
    lower_bound: float,
) -> None:
    """Commit all descriptor-table KDA layers with constant launch count."""
    if head_dim != 128:
        raise ValueError("batched KDA replay currently requires head_dim=128")
    layers = addresses.shape[0]
    batch = accepted_length.numel()
    rows = batch * draft_token_num
    block_t, block_k = _gate_tiling(rows, num_heads, head_dim, addresses.device)
    batched_kda_gate_precompute_kernel[
        (
            layers * num_heads,
            triton.cdiv(rows, block_t),
            triton.cdiv(head_dim, block_k),
        )
    ](
        addresses,
        rows,
        stride_fa=f_a_stride,
        stride_gate=gate_stride,
        lower_bound=lower_bound,
        HV=num_heads,
        K=head_dim,
        D_FA=f_a_dim,
        BK=block_k,
        BT=block_t,
        num_warps=1 if block_t >= 4 else 2,
    )
    batched_recurrent_kda_replay_commit_kernel[(layers, batch, num_heads)](
        addresses,
        read_indices,
        write_indices,
        accepted_length,
        batch,
        T=draft_token_num,
        STRIDE_QKV=qkv_stride,
        STRIDE_CONV=conv_stride,
        STRIDE_BETA=beta_stride,
        STRIDE_STATE=state_stride,
        STRIDE_GATE=gate_stride,
        CONV_WIDTH=conv_width,
        LAYERS_PER_GROUP=layers_per_group,
        NUM_GROUPS=read_indices.shape[0],
        HV=num_heads,
        K=head_dim,
        BK=triton.next_power_of_2(head_dim),
        D_FA=f_a_dim,
        BV=64,
        num_warps=4,
        num_stages=2,
    )
    batched_kda_commit_conv_window_kernel[(layers, batch, 1)](
        addresses,
        read_indices,
        write_indices,
        accepted_length,
        batch,
        T=draft_token_num,
        STRIDE_QKV=qkv_stride,
        STRIDE_CONV=conv_stride,
        CONV_DIM=3 * num_heads * head_dim,
        LAYERS_PER_GROUP=layers_per_group,
        BLOCK=triton.next_power_of_2(3 * num_heads * head_dim),
        num_warps=8,
    )


@triton.jit
def kda_commit_conv_window_kernel(
    qkv_raw,  # payload ring rows [2*cap, 3*P]: prev parity read, cur written
    conv_pool,  # [pages, 3*P, 3] committed conv window (read)
    conv_out,  # [pages, 3*P, 3] destination (may alias conv_pool)
    read_indices,  # [N] committed page per request (-1 = fresh)
    write_indices,  # [N] destination page per request (-1 skips)
    n_steps,  # [N] tokens consumed per request
    row_base,  # [N] first payload row per request (-1 skips), parity-offset
    src_qkv,  # [N*T, 3*P] THIS round's projections (capture source)
    src_fa,  # [N*T, W_FA]
    src_beta,  # [N*T, W_BETA]
    dst_fa,  # ring [2*cap, W_FA]
    dst_beta,  # ring [2*cap, W_BETA]
    capture_base,  # [1] int64: this round's parity * cap (device-carried)
    stride_raw_tok: tl.constexpr,
    stride_conv_page: tl.constexpr,
    stride_conv_out_page: tl.constexpr,
    stride_src_tok: tl.constexpr,
    stride_sfa_tok: tl.constexpr,
    stride_sbeta_tok: tl.constexpr,
    stride_dfa_tok: tl.constexpr,
    stride_dbeta_tok: tl.constexpr,
    T: tl.constexpr,
    CONV_DIM: tl.constexpr,
    W_FA: tl.constexpr,
    W_BETA: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    CAPTURE: tl.constexpr,
):
    """Roll a committed conv window forward by ``n_steps`` raw tokens.

    The 4-tap causal conv keeps the three preceding raw projections per
    channel, so committing the window after a replayed prefix is just a shift
    of the last ``min(n_steps, 3)`` raw values in -- no convolution needed.

    This is a separate launch from the recurrence because the window is
    indexed by channel alone: in the recurrence kernel every program of the
    NV column split shares (and would rewrite) the same q/k channels, which
    races against their own reads as soon as the destination page is the
    source page. Here one program owns a channel block outright, so the
    in-place case is safe by construction.
    """
    i_n = tl.program_id(0)
    i_c = tl.program_id(1)

    if CAPTURE:
        # Stage THIS round's inputs into the ring's current parity half; the
        # conv roll below reads the OTHER half, so the two never collide even
        # when the batch re-packs slots between rounds. Runs before the
        # early return: fresh requests still owe next round a payload.
        cb = tl.load(capture_base)
        offs_c = i_c * BLOCK + tl.arange(0, BLOCK)
        m_q = offs_c < CONV_DIM
        for i_t in range(T):
            row = i_n * T + i_t
            v = tl.load(src_qkv + row * stride_src_tok + offs_c, mask=m_q, other=0)
            tl.store(qkv_raw + (cb + row) * stride_raw_tok + offs_c, v, mask=m_q)
            # The gate payloads ride the same column blocks as qkv rather
            # than one privileged program, so BLOCK can shrink to give small
            # batches a grid without stranding f_a on a single program.
            if i_c * BLOCK < W_FA:
                m_f = offs_c < W_FA
                vf = tl.load(src_fa + row * stride_sfa_tok + offs_c, mask=m_f, other=0)
                tl.store(dst_fa + (cb + row) * stride_dfa_tok + offs_c, vf, mask=m_f)
            if i_c * BLOCK < W_BETA:
                m_b = offs_c < W_BETA
                vb = tl.load(
                    src_beta + row * stride_sbeta_tok + offs_c, mask=m_b, other=0
                )
                tl.store(
                    dst_beta + (cb + row) * stride_dbeta_tok + offs_c, vb, mask=m_b
                )

    b_write = tl.load(write_indices + i_n).to(tl.int64)
    base = tl.load(row_base + i_n).to(tl.int64)
    if b_write < 0 or base < 0:
        return
    b_read = tl.load(read_indices + i_n).to(tl.int64)
    # Same precondition as the fused clamp: steps are in this window's units.
    steps = tl.minimum(tl.maximum(tl.load(n_steps + i_n).to(tl.int32), 0), T)

    offsets = i_c * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < CONV_DIM

    src = conv_pool + b_read * stride_conv_page + offsets * 3
    s0 = tl.load(src + 0, mask=mask & (b_read >= 0), other=0.0)
    s1 = tl.load(src + 1, mask=mask & (b_read >= 0), other=0.0)
    s2 = tl.load(src + 2, mask=mask & (b_read >= 0), other=0.0)

    for i_t in range(steps):
        x = tl.load(
            qkv_raw + (base + i_t) * stride_raw_tok + offsets, mask=mask, other=0.0
        )
        s0, s1, s2 = s1, s2, x

    dst = conv_out + b_write * stride_conv_out_page + offsets * 3
    if ENABLE_PDL:
        # In place these stores overwrite slots the recurrence kernel reads;
        # wait for it before the first store (loads above are safe early).
        tl.extra.cuda.gdc_wait()
    tl.store(dst + 0, s0, mask=mask)
    tl.store(dst + 1, s1, mask=mask)
    tl.store(dst + 2, s2, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def kda_commit_conv_window(
    qkv_raw: torch.Tensor,
    conv_pool: torch.Tensor,
    conv_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    n_steps: torch.Tensor,
    *,
    conv_dim: int,
    draft_token_num: int,
    row_base: torch.Tensor | None = None,
    enable_pdl: bool = False,
    capture: tuple | None = None,
) -> None:
    """Commit each request's conv window after replaying ``n_steps`` tokens.

    Args:
        qkv_raw: ``[N*T, 3*P]`` pre-conv packed q|k|v, request-major.
        conv_pool: ``[pages, 3*P, 3]`` committed conv window (read).
        conv_out: destination slab; may be ``conv_pool`` itself.
        read_indices: ``[N]`` committed page per request (-1 = fresh).
        write_indices: ``[N]`` destination page per request (-1 skips).
        n_steps: ``[N]`` tokens consumed, in ``[0, draft_token_num]``.
        conv_dim: channel count ``3 * num_heads * head_dim``.
        draft_token_num: draft positions per request (T).
        row_base: ``[N]`` first payload row per request (-1 skips). Defaults
            to the request-major layout ``n * draft_token_num``; the fused
            lazy commit passes explicit bases because a request may sit at a
            different batch slot than when its payload was captured.

    Returns:
        None. The destination windows are written in place.
    """
    enable_pdl = bool(enable_pdl and _IS_NVIDIA)
    n = write_indices.numel()
    if row_base is None:
        row_base = (
            torch.arange(n, device=write_indices.device, dtype=torch.int32)
            * draft_token_num
        )
    # Narrow the column block until the grid covers the machine: at one
    # request the widest block leaves 18 programs for 4608 channels.
    index = write_indices.device.index
    if index is None:
        index = torch.cuda.current_device()
    sms = _SM_COUNT.get(index)
    if sms is None:
        sms = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT[index] = sms
    # Narrowing past 64 stops paying: the columns each program owns get too
    # few to amortize its own setup (one request, 4608 channels: 10.2us at
    # 256, 9.0us at 128 and 64, back to 10.2us at 32).
    block = 256
    while block > 64 and n * triton.cdiv(conv_dim, block) < 4 * sms:
        block //= 2
    if capture is not None:
        src_qkv, src_fa, src_beta, dst_fa, dst_beta, capture_base = capture
    else:
        src_qkv = src_fa = src_beta = dst_fa = dst_beta = qkv_raw
        capture_base = read_indices  # unused placeholder, same device
    kda_commit_conv_window_kernel[(n, triton.cdiv(conv_dim, block))](
        qkv_raw=qkv_raw,
        conv_pool=conv_pool,
        conv_out=conv_out,
        read_indices=read_indices,
        write_indices=write_indices,
        n_steps=n_steps,
        row_base=row_base,
        src_qkv=src_qkv,
        src_fa=src_fa,
        src_beta=src_beta,
        dst_fa=dst_fa,
        dst_beta=dst_beta,
        capture_base=capture_base,
        stride_raw_tok=qkv_raw.stride(0),
        stride_conv_page=conv_pool.stride(0),
        stride_conv_out_page=conv_out.stride(0),
        stride_src_tok=src_qkv.stride(0),
        stride_sfa_tok=src_fa.stride(0),
        stride_sbeta_tok=src_beta.stride(0),
        stride_dfa_tok=dst_fa.stride(0),
        stride_dbeta_tok=dst_beta.stride(0),
        T=draft_token_num,
        CONV_DIM=conv_dim,
        W_FA=src_fa.shape[-1],
        W_BETA=src_beta.shape[-1],
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        CAPTURE=capture is not None,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
