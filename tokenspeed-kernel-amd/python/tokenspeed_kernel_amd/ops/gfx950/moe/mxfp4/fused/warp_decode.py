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

"""Small-M (M<=4) FP8 x MXFP4 warp-decode MoE: fused dense top-k +
gate_up GEMM + SwiGLU + FP8 requant stage 1, and the split-K stage 2."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _make_dummy,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (
    _load_layout,
    _moe_partial_reduce,
    _situ_reduce,
    _swiglu_reduce,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.pipelined_program import (
    AsyncCopyDescriptor,
    MoEConfig,
    MoEPipelinedProgram,
    WPreshuffledLdsDescriptor,
    _make_preshuffled_w_full_offsets,
    _preshuffled_w_copy_layout,
    _preshuffled_w_read_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.routing import (
    _ROUTE_GL_DTYPE,
    _route_next_pow2,
    gluon_route_supported,
)


def _gluon_mxfp4_fp8_warp_decode_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight,
    w2_weight,
    *,
    w13_bias=None,
    w2_bias=None,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    out_dtype: torch.dtype,
    top_k: int,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
) -> torch.Tensor | None:
    """Small-M direct warp-decode MoE for GPT-OSS FP8 x MXFP4 path."""
    assert hidden_states.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    if hidden_states.ndim != 2 or router_logits.ndim != 2:
        return None
    n_tokens = int(router_logits.shape[0])
    n_experts = int(router_logits.shape[1])
    if n_tokens > WARP_DECODE_MAX_M:
        return None
    if not gluon_route_supported(router_logits, top_k, router_logits.dtype):
        return None

    # Use the optional Gluon dot-layout preshuffled attachments when they match
    # the layout this warp-decode path knows how to consume.
    w13_raw_candidate = _extract_gluon_raw_w(w13_weight)
    w13_preshuffled = (
        isinstance(w13_raw_candidate, torch.Tensor)
        and bool(getattr(w13_raw_candidate, "is_shuffled_for_gluon_dot", False))
        and int(getattr(w13_raw_candidate, "gluon_dot_block_k_pk", 0)) == 128
        and int(getattr(w13_raw_candidate, "gluon_dot_block_n", 0)) == 128
    )
    w13_raw = w13_raw_candidate if w13_preshuffled else _extract_gluon_raw_s(w13_weight)
    w2_raw_candidate = _extract_gluon_raw_w(w2_weight)
    w2_preshuffled = (
        isinstance(w2_raw_candidate, torch.Tensor)
        and bool(getattr(w2_raw_candidate, "is_shuffled_for_gluon_dot", False))
        and int(getattr(w2_raw_candidate, "gluon_dot_block_k_pk", 0)) == 128
        and int(getattr(w2_raw_candidate, "gluon_dot_block_n", 0)) == 128
    )
    w2_raw = w2_raw_candidate if w2_preshuffled else _extract_gluon_raw_s(w2_weight)
    w13_scale = _extract_gluon_raw_s(w13_mx_scale)
    w2_scale = _extract_gluon_raw_s(w2_mx_scale)
    if not all(
        isinstance(t, torch.Tensor) for t in (w13_raw, w2_raw, w13_scale, w2_scale)
    ):
        return None
    if w13_raw.ndim != 3 or w2_raw.ndim != 3:
        return None
    if w13_raw.dtype != torch.uint8 or w2_raw.dtype != torch.uint8:
        return None
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        return None

    D = int(hidden_states.shape[1])
    w13_k_pk = int(getattr(w13_raw, "original_k_pk", int(w13_raw.shape[1])))
    if w13_k_pk * 2 != D:
        return None
    two_i = int(w13_raw.shape[2])
    if two_i % 2 != 0:
        return None
    if w13_preshuffled and two_i % 128 != 0:
        return None
    i_dim = two_i // 2
    w2_k_pk = int(getattr(w2_raw, "original_k_pk", int(w2_raw.shape[1])))
    if w2_k_pk * 2 != i_dim:
        return None
    w2_n_phys = int(w2_raw.shape[2])
    N = int(getattr(w2_raw, "original_n", w2_n_phys))
    if w2_preshuffled and (N > w2_n_phys or w2_n_phys % 128 != 0):
        return None

    # Stage1 computes the dense top-k inside the kernel; allocate its outputs.
    router_logits_c = router_logits.contiguous()
    topk_ids = torch.empty(
        (n_tokens, top_k), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (n_tokens, top_k), dtype=router_logits.dtype, device=router_logits.device
    )

    # Current GPT-OSS path uses FP8 E4M3 activations with per-tensor scale.
    x_fp8 = hidden_states
    # Pass the FP8 tensor straight to Gluon.  ``view(torch.uint8)`` materializes a
    # copy for float8 tensors on this stack and dominates small-M latency.

    out = torch.empty((n_tokens, N), dtype=out_dtype, device=hidden_states.device)

    # The kernels only read the bias pointer when HAS_BIAS; allocate the
    # placeholder solely for the absent ones.
    dummy_bias = (
        _make_dummy(hidden_states.device, torch.float32, 1)
        if (w13_bias is None or w2_bias is None)
        else None
    )
    b13 = w13_bias if w13_bias is not None else dummy_bias
    b2 = w2_bias if w2_bias is not None else dummy_bias

    BLOCK_K = 128
    S2_BLOCK_N = 32 if n_tokens >= 16 else (8 if n_tokens <= 1 else 16)
    S2_M_DUP = 4
    # Tuned stage2 K=I split factor by batch size (8/4 raise small-M occupancy, off for M>=5).
    if n_tokens <= 2:
        s2_split_k = 8
    elif n_tokens <= 4:
        s2_split_k = 4
    else:
        s2_split_k = 1

    inter = torch.empty(
        (n_tokens * top_k, i_dim), dtype=x_fp8.dtype, device=hidden_states.device
    )
    # Cooperative-LDS, num_warps=4, software-pipelined stage1 -- the smallest-M
    # decode path (this wrapper is only entered for n_tokens <= WARP_DECODE_MAX_M).
    # BLOCK_N=64 maximizes CTAs/CU for non-preshuffled W. Preshuffled W13 is
    # laid out as 128-wide CTA tiles, so the first decode preshuffle path
    # consumes full 128-wide tiles and lets SwiGLU reduce them to 64 columns.
    COOP_NUM_WARPS = 4
    COOP_BLOCK_N = 128 if w13_preshuffled else 64
    COOP_BLOCK_K = 256
    coop_k_iters = (D + COOP_BLOCK_K - 1) // COOP_BLOCK_K
    coop_even_k = D % COOP_BLOCK_K == 0
    # decode_pipeline() prefetches NUM_BUFFERS-1 tiles before the main loop.
    # For short synthetic shapes (for example D=256) the fixed 3-buffer GPT-OSS
    # schedule over-prefetches past the only real K tile. Keep the production
    # 3-buffer schedule when possible, but shrink the pipeline for short K.
    COOP_NUM_BUFFERS = min(3, coop_k_iters + (1 if coop_even_k else 0))
    coop_grid = (n_tokens * ((2 * i_dim + COOP_BLOCK_N - 1) // COOP_BLOCK_N) * top_k,)
    # X is stored as raw i8 in LDS and bitcast to e4m3 in mfma_scaled; pass the
    # uint8 view (an fp8 LDS buffer fails to lower).
    x_uint8 = x_fp8.view(torch.uint8)
    # fmt: off
    _warp_decode_topk_stage1_coop_kernel[coop_grid](
        x_uint8, router_logits_c, w13_raw, w13_scale, topk_ids, topk_weights, inter,
        n_tokens, n_experts, D, i_dim,
        x_uint8.stride(0), x_uint8.stride(1),
        router_logits_c.stride(0), topk_ids.stride(0), topk_weights.stride(0),
        w13_raw.stride(0), w13_raw.stride(-2), w13_raw.stride(-1),
        w13_scale.stride(0), w13_scale.stride(-2), w13_scale.stride(-1),
        inter.stride(0), inter.stride(1),
        w13_act_scale, w2_act_scale, b13,
        D_PACKED=D // 2, TOPK=top_k,
        # EP/TKP: padded widths of the [tokens, experts] logits tile and the
        # top-k selection tile; >= 64*num_warps keeps the blocked layout valid.
        EP=max(_route_next_pow2(n_experts), 64 * COOP_NUM_WARPS), TKP=64 * COOP_NUM_WARPS,
        X_DTYPE=_ROUTE_GL_DTYPE[router_logits.dtype],
        BLOCK_K=COOP_BLOCK_K, BLOCK_N=COOP_BLOCK_N, BLOCK_M=16,
        NUM_BUFFERS=COOP_NUM_BUFFERS, NUM_WARPS=COOP_NUM_WARPS,
        W_PRESHUFFLED=w13_preshuffled,
        EVEN_K=coop_even_k,
        HAS_BIAS=w13_bias is not None,
        SWIGLU_ALPHA=float(swiglu_alpha), SWIGLU_LIMIT=float(swiglu_limit),
        SWIGLU_BETA=float(swiglu_beta),
        num_warps=COOP_NUM_WARPS,
    )
    # fmt: on

    n_tiles2 = (N + S2_BLOCK_N - 1) // S2_BLOCK_N
    if s2_split_k > 1:
        out_partial = torch.empty(
            (s2_split_k, n_tokens, N), dtype=torch.float32, device=hidden_states.device
        )
        s2_dst = out_partial
        s2_stride_om = out_partial.stride(1)
        s2_stride_on = out_partial.stride(2)
        s2_stride_ok = out_partial.stride(0)
        s2_grid = (n_tokens * n_tiles2 * s2_split_k,)
    else:
        s2_dst = out
        s2_stride_om = out.stride(0)
        s2_stride_on = out.stride(1)
        s2_stride_ok = 0
        s2_grid = (n_tokens * n_tiles2,)
    # fmt: off
    _warp_decode_stage2_fp8_mxfp4_kernel[s2_grid](
        inter, w2_raw, w2_scale, topk_ids, topk_weights, s2_dst,
        n_tokens, N, w2_n_phys, i_dim,
        inter.stride(0), inter.stride(1),
        w2_raw.stride(0), w2_raw.stride(-2), w2_raw.stride(-1),
        w2_scale.stride(0), w2_scale.stride(-2), w2_scale.stride(-1),
        s2_stride_om, s2_stride_on, s2_stride_ok,
        w2_act_scale, b2,
        I_PACKED=i_dim // 2, TOPK=top_k,
        BLOCK_K=BLOCK_K, BLOCK_N=S2_BLOCK_N, M_DUP=S2_M_DUP,
        W_PRESHUFFLED=w2_preshuffled,
        HAS_BIAS=w2_bias is not None, SPLIT_K=s2_split_k,
        num_warps=1,
    )
    # fmt: on
    if s2_split_k > 1:
        R_BLOCK_N = 256
        r_grid = (n_tokens * ((N + R_BLOCK_N - 1) // R_BLOCK_N),)
        # fmt: off
        _moe_partial_reduce[r_grid](
            out_partial, out, n_tokens, N,
            out_partial.stride(0), out_partial.stride(1), out_partial.stride(2),
            out.stride(0), out.stride(1),
            SPLIT_K=s2_split_k, BLOCK_N=R_BLOCK_N, num_warps=1,
        )
        # fmt: on
    return out


# Warp-decode is only for the smallest decode regime. M>=8 should use the
# medium-decode direct path selected by the ragged matmul path below.
WARP_DECODE_MAX_M = 4


@gluon.jit
def _add_expert_bias(acc, bias_base, col, bound, mfma_layout: gl.constexpr):
    """Broadcast-add a per-expert column bias into an MFMA accumulator.

    The bias is loaded along N then converted into the accumulator's column
    slice layout, which keeps the broadcast-add convert-compatible with acc.
    """
    b = gl.load(bias_base + col, mask=bound, other=0.0).to(gl.float32)
    b = gl.convert_layout(b, gl.SliceLayout(0, mfma_layout))
    return acc + b[None, :]


@gluon.constexpr_function
def _warp_decode_mfma_layouts(m_dup, block_n, block_k_scale):
    """MFMA + dot-operand + e8m0 scale layouts shared by the warp-decode kernels.

    get_mfma_layout is not reused: it asserts num_warps in (4, 8), whereas warp
    decode runs a single warp ([1, 1] warps_per_cta).
    """
    mfma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1]
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [m_dup, block_k_scale])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [block_n, block_k_scale])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _warp_decode_stage1_coop_compute(
    token,
    slot,
    expert,
    pid_n,
    X,
    W,
    WScale,
    Y,
    M,
    D,
    i_dim,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_ym,
    stride_yn,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    w13_bias,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    EVEN_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr = False,
):
    """Cooperative gate_up GEMM + bias + SwiGLU + fp8-quant + store for one
    (token, slot, expert).  N runs over the INTERLEAVED gate_up rows (2*I);
    ``_swiglu_reduce`` splits even=gate / odd=up.  Mirrors the plain path of
    ``_pipelined_moe_tile_compute`` (W_TRANSPOSE=False, swizzled w-scale,
    per-tensor x scale) but specialized to a single decode token (row 0 of the
    BLOCK_M tile).
    """
    N = 2 * i_dim
    off_n = pid_n * BLOCK_N
    # Keep base offsets int32 (buffer_load_to_shared requires int32/uint32
    # offsets); expert * stride fits int32 for GPT-OSS shapes.
    w_base_offset = expert * stride_we
    ws_base_offset = expert * stride_wse

    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        "e4m3",  # X format (fp8 activations)
        "e2m1",  # W format (mxfp4 weights)
        32,  # SCALE_BLOCK
        NUM_BUFFERS,
        not W_PRESHUFFLED,  # W_TRANSPOSE for non-preshuffled K-packed-contiguous W
        False,  # WITH_X_MX_SCALE (per-tensor x scale only)
        True,  # WITH_W_MX_SCALE (e8m0 block scales)
        "swizzle",  # SCALE_LOAD_MODE -> W_SCALE_VIA_LDS unswizzle
        gl.int32,
        (1, 1, 1),  # NUM_SUBTILES
        EVEN_K,
        False,  # USE_GATHER
        NUM_WARPS,
        W_PRESHUFFLED=W_PRESHUFFLED,
        W_VIA_VGPR=False,
        W_PREFETCH=True,
    )

    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // 2
    W_CACHE_MODIFIER: gl.constexpr = ".cg" if BLOCK_M <= 32 else ""

    X_ELEM_BITS: gl.constexpr = X.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = W.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_X, BLOCK_M, NUM_WARPS, [1, 0], X_ELEM_BITS
    )
    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))

    valid = (token < M) & (expert >= 0)
    # One decode token per CTA: row 0 of the BLOCK_M tile carries the token,
    # the remaining rows are clamped/masked (buffer OOB -> 0 in LDS).
    rows_m = gl.where(offs_xm == 0, token, gl.zeros_like(offs_xm))
    mask_m = (offs_xm == 0) & valid

    k_limit_x = gl.multiple_of(D // cfg.DIV_FACTOR_X, 16)
    k_limit_w = gl.multiple_of(D // cfg.DIV_FACTOR_W, 16)

    x_desc = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        X,
        rows_m,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m[:, None],
        k_limit_x,
    )
    if W_PRESHUFFLED:
        gl.static_assert(
            BLOCK_N == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
            "warp_decode preshuffled W13 path assumes 128x128 W tiles "
            "and NUM_WARPS=4; re-derive the copy/read layouts for other shapes.",
        )
        LOAD_W_LAYOUT: gl.constexpr = _preshuffled_w_read_layout(
            BLOCK_N // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS
        )
        LOAD_W_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            BLOCK_N // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, True
        )
        offsets_w, base_off_w = _make_preshuffled_w_full_offsets(
            w_base_offset,
            pid_n,
            LOAD_W_COPY_LAYOUT,
            BLOCK_N,
            BLOCK_N,
            BLOCK_K_W,
        )
        w_desc = WPreshuffledLdsDescriptor(
            cfg,
            BLOCK_K_W,
            W,
            W.dtype.element_ty,
            gl.to_tensor(N),
            offsets_w + base_off_w,
            pred=gl.to_tensor(True),
            load_layout=LOAD_W_LAYOUT,
            cache_modifier=W_CACHE_MODIFIER,
        )
    else:
        # K-contig W (W_TRANSPOSE=True): vectorise the contiguous K_packed axis
        # (mirrors the W_TRANSPOSE branch of _pipelined_moe_tile_compute).
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, BLOCK_N, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        mask_n = (off_n + offs_wn) < N
        w_desc = AsyncCopyDescriptor.initialize(
            cfg,
            0,
            BLOCK_K_W,
            W,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[:, None],
            k_limit_w,
            base_offset=w_base_offset,
            cache_modifier=W_CACHE_MODIFIER,
        )

    # W e8m0 scales -> LDS in the post-swizzle HBM shape; issue_local_load_unswizzle
    # reconstructs [BLOCK_N, BLOCK_K_SCALE] (the 7-D reshape/permute).
    BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
    BLOCK_K_S_PS_W: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
    LW_S: gl.constexpr = cfg.load_layout_w_scale
    offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))
    offs_ws_k = gl.arange(0, BLOCK_K_S_PS_W, layout=gl.SliceLayout(0, LW_S))
    rows_n_scale = off_n // cfg.PRESHUFFLE_FACTOR + offs_ws_n
    row_limit_w_s = (N + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
    # Suppress the K-mask: the swizzle packs K with N; the W K-mask already
    # zeroes the OOB product regardless of scale value.
    k_limit_ws_load = ((D // cfg.SCALE_BLOCK + 7) // 8 * 8) * cfg.PRESHUFFLE_FACTOR
    w_scale_desc = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_S_PS_W,
        WScale,
        rows_n_scale,
        offs_ws_k,
        stride_wsn,
        stride_wsk,
        rows_n_scale[:, None] < row_limit_w_s,
        k_limit_ws_load,
        base_offset=ws_base_offset,
    )

    pgm = MoEPipelinedProgram.initialize(cfg, x_desc, w_desc, 0, w_scale_desc)
    # Preserve the upstream small-M decode schedule: it uses a three-buffer
    # local-prefetch pipeline and different tail masking from the optimized
    # prefill GEMM pipeline above.
    acc = pgm.decode_pipeline(D)

    # Per-tensor activation scale.
    x_scale = gl.load(x_global_scale_ptr).to(gl.float32)
    acc = acc * x_scale

    if HAS_BIAS:
        # Bias is laid out [E, 2*I] (interleaved gate/up rows); add before the
        # SwiGLU even/odd split, matching the num_warps=1 path.
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
        bias_mask = bias_offs < N
        bias = gl.load(
            w13_bias + expert.to(gl.int64) * N + bias_offs,
            mask=bias_mask,
            other=0.0,
        )
        acc = acc + bias[None, :].to(gl.float32)

    if DO_SITU:
        out = _situ_reduce(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, OUT_BLOCK_N)
    else:
        out = _swiglu_reduce(
            acc,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            OUT_BLOCK_N,
            cfg.acc_layout,
        )
    out_inv_scale = 1.0 / gl.load(out_quant_scale_ptr).to(gl.float32)
    out = (out * out_inv_scale).to(Y.dtype.element_ty)
    STORE_LAYOUT: gl.constexpr = out.type.layout

    offs_y_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT))
    row = token * TOPK + slot
    # Only tile-row 0 holds the token's result; all valid columns map to the
    # single Y row (row*stride_ym).
    y_offs = (
        row.to(gl.int64) * stride_ym
        + offs_y_n[None, :].to(gl.int64) * stride_yn
        + offs_y_m[:, None].to(gl.int64) * 0
    )
    mask_y = (offs_y_m[:, None] == 0) & valid & (offs_y_n[None, :] < i_dim)
    gl.store(Y + y_offs, out, mask=mask_y)


@gluon.jit
def _warp_decode_topk_stage1_coop_kernel(
    X,
    Logits,
    W,
    WScale,
    TopkIdsOut,
    TopkWeightsOut,
    Y,
    M,
    E,
    D,
    i_dim,
    stride_xm,
    stride_xk,
    stride_lm,
    stride_tim,
    stride_twm,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_ym,
    stride_yn,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    w13_bias,
    D_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    X_DTYPE: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    EVEN_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    """Cooperative (multi-warp) fused dense top-k + gate_up stage1.

    The slot dimension is folded into the grid -- one gate_up GEMM (one
    MoEPipelinedProgram / LDS buffer set) per program, so LDS is not multiplied
    by TOPK. Routing layouts span all warps (EP/TKP padded to 64*NUM_WARPS).
    """
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(2 * i_dim, BLOCK_N)
    slot = pid % TOPK
    rest = pid // TOPK
    pid_n = rest % num_pid_n
    token = rest // num_pid_n

    # ---- direct top-k for this token (replicated per (N tile, slot)) ----
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    LT: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    cur = gl.load(
        Logits + token.to(gl.int64) * stride_lm + e,
        mask=(token < M) & emask,
        other=float("-inf"),
    ).to(gl.float32)
    t = gl.arange(0, TKP, layout=LT)
    val_t = gl.full([TKP], -1e30, gl.float32, layout=LT)
    idx_t = gl.zeros([TKP], gl.int32, layout=LT)
    live = (token < M) & emask
    topmask = gl.full([EP], 0x80000000, gl.uint32, layout=LE)
    fullmask = gl.full([EP], 0xFFFFFFFF, gl.uint32, layout=LE)
    zero_pack = gl.full([EP], 0, gl.uint64, layout=LE)
    for r in gl.static_range(TOPK):
        raw = cur.to(gl.uint32, bitcast=True)
        value_key = raw ^ gl.where((raw & topmask) != 0, fullmask, topmask)
        index_key = (EP - e).to(gl.uint32)
        packed = (value_key.to(gl.uint64) << 16) | index_key.to(gl.uint64)
        packed = gl.where(live, packed, zero_pack)
        best = gl.max(packed, axis=0)
        amax_key = (best & 0xFFFF).to(gl.int32)
        amax = (EP - amax_key).to(gl.int32)
        chosen = live & (e == amax)
        vmax = gl.sum(gl.where(chosen, cur, gl.zeros_like(cur)), axis=0)
        sel = t == r
        val_t = gl.where(sel, vmax, val_t)
        idx_t = gl.where(sel, amax, idx_t)
        live = live & (e != amax)
    rmax = gl.max(val_t, axis=0)
    num = gl.exp(val_t - rmax)
    den = gl.sum(num, axis=0)
    gate_t = gl.fdiv(num, den)
    if (pid_n == 0) & (slot == 0):
        gl.store(
            TopkIdsOut + token.to(gl.int64) * stride_tim + t,
            idx_t,
            mask=(token < M) & (t < TOPK),
        )
        gl.store(
            TopkWeightsOut + token.to(gl.int64) * stride_twm + t,
            gate_t.to(TopkWeightsOut.dtype.element_ty),
            mask=(token < M) & (t < TOPK),
        )

    slot_sel = t == slot
    expert = gl.sum(
        gl.where(slot_sel, idx_t, gl.zeros([TKP], gl.int32, layout=LT)), axis=0
    )
    # Grouped by role: coords / tensors / shapes / strides / scalars / constexpr.
    # fmt: off
    _warp_decode_stage1_coop_compute(
        token, slot, expert, pid_n,
        X, W, WScale, Y,
        M, D, i_dim,
        stride_xm, stride_xk,
        stride_we, stride_wk, stride_wn,
        stride_wse, stride_wsk, stride_wsn,
        stride_ym, stride_yn,
        x_global_scale_ptr, out_quant_scale_ptr, w13_bias,
        TOPK, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, NUM_WARPS,
        W_PRESHUFFLED, EVEN_K, HAS_BIAS, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA,
    )

    # fmt: on


@gluon.jit
def _warp_decode_precomputed_situ_stage1_kernel(
    X,
    W,
    WScale,
    TopkIds,
    Y,
    M,
    D,
    i_dim,
    stride_xm,
    stride_xk,
    stride_tim,
    stride_tik,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_ym,
    stride_yn,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    w13_bias,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    EVEN_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
):
    """Route-direct FP8xMXFP4 W13 with fused SiTU and FP8 store."""
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(2 * i_dim, BLOCK_N)
    slot = pid % TOPK
    rest = pid // TOPK
    pid_n = rest % num_pid_n
    token = rest // num_pid_n
    expert = gl.load(
        TopkIds + token.to(gl.int64) * stride_tim + slot * stride_tik,
        mask=token < M,
        other=-1,
    ).to(gl.int32)
    _warp_decode_stage1_coop_compute(
        token,
        slot,
        expert,
        pid_n,
        X,
        W,
        WScale,
        Y,
        M,
        D,
        i_dim,
        stride_xm,
        stride_xk,
        stride_we,
        stride_wk,
        stride_wn,
        stride_wse,
        stride_wsk,
        stride_wsn,
        stride_ym,
        stride_yn,
        x_global_scale_ptr,
        out_quant_scale_ptr,
        w13_bias,
        TOPK,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        NUM_WARPS,
        W_PRESHUFFLED,
        EVEN_K,
        HAS_BIAS,
        SITU_BETA,
        SITU_LINEAR_BETA,
        0.0,
        DO_SITU=True,
    )


@gluon.jit
def _warp_decode_stage2_preshuffled_w_offset(
    w_expert_off,
    k_pack,
    n_col,
    N_PHYS,
):
    k_in_block = k_pack % 128
    n_in_block = n_col % 128

    k_within = k_in_block % 16
    k_quad = (k_in_block // 16) % 4
    k_block = k_in_block // 64
    n_in_sub = n_in_block % 16
    n_block = n_in_block // 16

    in_tile = (
        n_block.to(gl.int64) * 2048
        + k_block.to(gl.int64) * 1024
        + k_quad.to(gl.int64) * 256
        + n_in_sub.to(gl.int64) * 16
        + k_within.to(gl.int64)
    )
    n_tiles = N_PHYS // 128
    tile_id = (k_pack // 128).to(gl.int64) * n_tiles + (n_col // 128).to(gl.int64)
    return (w_expert_off + tile_id * (128 * 128) + in_tile).to(gl.int32)


@gluon.jit
def _warp_decode_stage2_load_tile(
    kt,
    ak,
    bk,
    bsk,
    am,
    X,
    W,
    WScale,
    x_row_off,
    w_expert_off,
    w_n_off,
    ws_expert_off,
    scale_row_off,
    n_cols,
    stride_xk,
    stride_wk,
    stride_wsk,
    N_PHYS,
    i_dim,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
    I_PACKED: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    MASK_TAIL: gl.constexpr = False,
):
    k_elem = kt * BLOCK_K + ak
    k_pack = kt * BLOCK_K_PACKED + bk
    a_off = (x_row_off + k_elem.to(gl.int64) * stride_xk + am.to(gl.int64) * 0).to(
        gl.int32
    )
    if W_PRESHUFFLED:
        b_off = _warp_decode_stage2_preshuffled_w_offset(
            w_expert_off, k_pack, n_cols, N_PHYS
        )
    else:
        b_off = (w_n_off + k_pack.to(gl.int64) * stride_wk).to(gl.int32)
    if BLOCK_K_SCALE == 4:
        scale_k_lin = (kt // 2) * 256 + (kt % 2) * 2 + bsk * 64
    else:
        sk = kt * BLOCK_K_SCALE + bsk
        scale_k_lin = (sk // 8) * 256 + (sk % 4) * 64 + ((sk % 8) // 4) * 2
    scale_k_off = scale_k_lin.to(gl.int64) * stride_wsk
    s_off = (ws_expert_off + scale_row_off + scale_k_off).to(gl.int32)
    if MASK_TAIL:
        # Partial / odd final K-tile (K = intermediate dim I): mask out-of-range
        # K lanes to 0 so they contribute nothing and never over-read.
        sk_valid = (kt * BLOCK_K_SCALE + bsk) < (i_dim // 32)
        a = gl.amd.cdna4.buffer_load(
            ptr=X, offsets=a_off, mask=k_elem < i_dim, other=0.0
        )
        b_mask = k_pack < I_PACKED
        b = gl.amd.cdna4.buffer_load(ptr=W, offsets=b_off, mask=b_mask, other=0)
        s = gl.amd.cdna4.buffer_load(ptr=WScale, offsets=s_off, mask=sk_valid, other=0)
    else:
        a = gl.amd.cdna4.buffer_load(ptr=X, offsets=a_off)
        b = gl.amd.cdna4.buffer_load(ptr=W, offsets=b_off)
        s = gl.amd.cdna4.buffer_load(ptr=WScale, offsets=s_off)
    return a, b, s


@gluon.jit
def _warp_decode_stage2_load_pair(
    kt,
    ak,
    bk,
    bsk,
    am,
    X,
    W,
    WScale,
    x_row_off,
    w_expert_off,
    w_n_off,
    ws_expert_off,
    scale_row_off,
    n_cols,
    stride_xk,
    stride_wk,
    stride_wsk,
    N_PHYS,
    i_dim,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
    I_PACKED: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
):
    """Load the even (kt) and odd (kt+1) K-tiles of one pipeline step."""
    # fmt: off
    a_even, b_even, s_even = _warp_decode_stage2_load_tile(
        kt, ak, bk, bsk, am, X, W, WScale,
        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
    )
    a_odd, b_odd, s_odd = _warp_decode_stage2_load_tile(
        kt + 1, ak, bk, bsk, am, X, W, WScale,
        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
    )
    # fmt: on
    return a_even, b_even, s_even, a_odd, b_odd, s_odd


@gluon.jit
def _warp_decode_stage2_mfma_pair(
    acc, a_even, b_even, s_even, a_odd, b_odd, s_odd, a_scale
):
    """Accumulate the scaled-MFMA of one even+odd K-tile pair (fp8 x mxfp4)."""
    # fmt: off
    acc = gl.amd.cdna4.mfma_scaled(
        a=a_even, a_scale=a_scale, a_format="e4m3",
        b=b_even, b_scale=s_even, b_format="e2m1", acc=acc,
    )
    acc = gl.amd.cdna4.mfma_scaled(
        a=a_odd, a_scale=a_scale, a_format="e4m3",
        b=b_odd, b_scale=s_odd, b_format="e2m1", acc=acc,
    )
    # fmt: on
    return acc


@gluon.jit
def _warp_decode_stage2_fp8_mxfp4_kernel(
    X,
    W,
    WScale,
    TopkIds,
    TopkWeights,
    Out,
    M,
    N,
    N_PHYS,
    i_dim,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_wse,
    stride_wsk,
    stride_wsn,
    stride_om,
    stride_on,
    stride_ok,
    x_global_scale_ptr,
    w2_bias,
    I_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    M_DUP: gl.constexpr,
    W_PRESHUFFLED: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SPLIT_K: gl.constexpr,
    SPLIT_TOPK: gl.constexpr = False,
):
    """Direct top-k stage2: FP8 intermediate x MXFP4 W2 -> BF16 output.

    With SPLIT_K > 1 the K (intermediate) reduction is partitioned across
    SPLIT_K CTAs per output tile; each writes an fp32 partial into slice
    ``pid_k`` of the destination, reduced by ``_moe_partial_reduce``.
    Bias is added only by the first slice so it is not counted SPLIT_K times.
    """
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    if W_PRESHUFFLED:
        gl.static_assert(
            128 % BLOCK_N == 0 and BLOCK_K_PACKED == 64,
            "warp_decode preshuffled W2 expects BLOCK_N to divide the "
            "128-wide shuffled tile and BLOCK_K_PACKED=64 so two stage2 "
            "iterations cover one 128-packed-byte K tile.",
        )
    gl.static_assert(
        not (SPLIT_K > 1 and SPLIT_TOPK),
        "stage2 cannot split K and top-k simultaneously",
    )
    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(N, BLOCK_N)
    if SPLIT_TOPK:
        pid_k = 0
        pid_token = pid // (TOPK * num_n)
        rem = pid % (TOPK * num_n)
        pid_slot = rem // num_n
        pid_n = rem % num_n
    elif SPLIT_K == 1:
        pid_k = 0
        pid_slot = 0
        pid_token = pid // num_n
        pid_n = pid % num_n
    else:
        pid_slot = 0
        per_k = M * num_n
        pid_k = pid // per_k
        rem = pid % per_k
        pid_token = rem // num_n
        pid_n = rem % num_n
    # Full + partial K-tile coverage (K = intermediate dim I). The old
    # `num_kt = I // BLOCK_K` dropped the partial final tile, miscomputing any
    # I not a multiple of BLOCK_K (GPT-OSS I=2880 lost K=2816..2879).
    num_full = i_dim // BLOCK_K
    total_kt = (i_dim + BLOCK_K - 1) // BLOCK_K
    kt_per = (total_kt + SPLIT_K - 1) // SPLIT_K
    kt_start = pid_k * kt_per
    kt_stop = gl.minimum(kt_start + kt_per, total_kt)
    full_stop = gl.minimum(kt_stop, num_full)
    _layouts: gl.constexpr = _warp_decode_mfma_layouts(M_DUP, BLOCK_N, BLOCK_K_SCALE)
    mfma_layout: gl.constexpr = _layouts[0]
    dot_a_layout: gl.constexpr = _layouts[1]
    dot_b_layout: gl.constexpr = _layouts[2]
    a_scale_layout: gl.constexpr = _layouts[3]
    b_scale_layout: gl.constexpr = _layouts[4]
    am = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
    ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
    bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    bsn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[:, None]
    bsk = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, b_scale_layout))[None, :]
    n_cols = pid_n * BLOCK_N + bn
    n_cols_s = pid_n * BLOCK_N + bsn
    a_scale = gl.full((M_DUP, BLOCK_K_SCALE), 127, gl.uint8, layout=a_scale_layout)
    acc_total = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    if pid_token < M:
        NUM_SLOTS: gl.constexpr = 1 if SPLIT_TOPK else TOPK
        for slot_iter in gl.static_range(0, NUM_SLOTS):
            slot = pid_slot if SPLIT_TOPK else slot_iter
            expert = gl.load(
                TopkIds + pid_token * TOPK + slot, mask=pid_token < M, other=-1
            )
            gate = gl.load(
                TopkWeights + pid_token * TOPK + slot,
                mask=pid_token < M,
                other=0.0,
            ).to(gl.float32)
            if expert >= 0:
                row = pid_token * TOPK + slot
                x_row_off = row.to(gl.int64) * stride_xm
                w_expert_off = expert.to(gl.int64) * stride_we
                ws_expert_off = expert.to(gl.int64) * stride_wse
                w_n_off = w_expert_off + n_cols.to(gl.int64) * stride_wn
                scale_row = n_cols_s.to(gl.uint32)
                scale_row_off = (scale_row // 32).to(gl.int64) * stride_wsn + (
                    (scale_row % 16) * 4 + ((scale_row % 32) // 16)
                ).to(gl.int64) * stride_wsk
                acc = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
                main_end = kt_start + ((full_stop - kt_start) // 2) * 2

                # Software-pipeline the main paired K-loop one step ahead:
                # prefetch the first pair, then each iteration loads the next pair
                # before MFMA-ing the current one (prefetch depth 2).
                main_kt = main_end - kt_start
                # fmt: off
                if main_kt > 0:
                    (a_even, b_even, s_even,
                     a_odd, b_odd, s_odd) = _warp_decode_stage2_load_pair(
                        kt_start, ak, bk, bsk, am, X, W, WScale,
                        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
                        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
                        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
                    )
                    for kt in range(kt_start, main_end - 2, 2):
                        (nxt_a_even, nxt_b_even, nxt_s_even,
                         nxt_a_odd, nxt_b_odd, nxt_s_odd) = _warp_decode_stage2_load_pair(
                            kt + 2, ak, bk, bsk, am, X, W, WScale,
                            x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
                            n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
                            BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
                        )
                        acc = _warp_decode_stage2_mfma_pair(
                            acc, a_even, b_even, s_even, a_odd, b_odd, s_odd, a_scale
                        )
                        a_even, b_even, s_even, a_odd, b_odd, s_odd = (
                            nxt_a_even, nxt_b_even, nxt_s_even,
                            nxt_a_odd, nxt_b_odd, nxt_s_odd,
                        )
                    # Epilogue: MFMA the final prefetched pair.
                    acc = _warp_decode_stage2_mfma_pair(
                        acc, a_even, b_even, s_even, a_odd, b_odd, s_odd, a_scale
                    )
                # Masked remainder: leftover odd/partial K-tile(s) in this split.
                for kt in range(main_end, kt_stop):
                    a_t, b_t, s_t = _warp_decode_stage2_load_tile(
                        kt, ak, bk, bsk, am, X, W, WScale,
                        x_row_off, w_expert_off, w_n_off, ws_expert_off, scale_row_off,
                        n_cols, stride_xk, stride_wk, stride_wsk, N_PHYS, i_dim,
                        BLOCK_K, BLOCK_K_PACKED, BLOCK_K_SCALE, I_PACKED, W_PRESHUFFLED,
                        MASK_TAIL=True,
                    )
                    acc = gl.amd.cdna4.mfma_scaled(
                        a=a_t, a_scale=a_scale, a_format="e4m3",
                        b=b_t, b_scale=s_t, b_format="e2m1", acc=acc,
                    )
                # fmt: on
                acc = acc * gl.load(x_global_scale_ptr).to(gl.float32)
                if HAS_BIAS:
                    bias_n = pid_n * BLOCK_N + gl.arange(
                        0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
                    )
                    w2_base = w2_bias + expert.to(gl.int64) * N
                    if SPLIT_K == 1:
                        bias_bound = bias_n < N
                    else:
                        bias_bound = (bias_n < N) & (pid_k == 0)
                    acc = _add_expert_bias(
                        acc, w2_base, bias_n, bias_bound, mfma_layout
                    )
                acc_total += gate * acc
    sm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, mfma_layout))[:, None]
    sn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))[None, :]
    col = pid_n * BLOCK_N + sn
    out_row = pid_token * TOPK + pid_slot if SPLIT_TOPK else pid_token
    out_base = (
        Out
        + out_row.to(gl.int64) * stride_om
        + col.to(gl.int64) * stride_on
        + sm.to(gl.int64) * 0
    )
    if SPLIT_K > 1:
        out_base = out_base + pid_k.to(gl.int64) * stride_ok
    gl.store(
        out_base,
        acc_total.to(Out.dtype.element_ty),
        mask=(pid_token < M) & (sm == 0) & (col < N),
    )
