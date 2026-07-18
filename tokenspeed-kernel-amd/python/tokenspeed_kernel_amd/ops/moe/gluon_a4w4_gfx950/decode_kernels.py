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


"""Small-M decode kernels for the gfx950 A4W4 MoE package.

The production entry points in this module consume packed E2M1 activations,
E8M0 activation scales, and gdot128-shuffled MXFP4 weights/scales,
then execute direct CDNA4 MFMA for both MoE stages.

The older ``invoke_stage*_warp_decode_gluon`` helpers consume BF16
activations and scalar-dequantize only the MXFP4 weights.  They are retained
solely for historical microbenchmarks through ``mxfp4_warp_decode_gfx950``;
the public ``gluon_a4w4_gfx950`` package and production dispatch do not export
or call them.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

_LANES = gl.constexpr(64)  # wavefront width (reduction lanes)
_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ROUTE_MAX_E = 1024
_ROUTE_MAX_G = 64
_ROUTE_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


@gluon.jit
def _e2m1_to_f32(n):
    """Decode a 4-bit e2m1 (fp4) code (0..15) to f32.

    magnitude by (e,m): e==0 -> m*0.5 (0 or 0.5); e>0 -> 2^(e-1) * (1 + 0.5*m)
    => {0,0.5,1,1.5,2,3,4,6}; sign bit negates.
    """
    sign = (n >> 3) & 1
    idx = n & 7
    m = (idx & 1).to(gl.float32)
    e = idx >> 1
    mag = gl.where(e == 0, m * 0.5, gl.exp2((e - 1).to(gl.float32)) * (1.0 + 0.5 * m))
    return gl.where(sign == 1, -mag, mag)


@gluon.jit
def _gluon_dot_preshuffled_w_offset(w_expert_off, k_pack, n_col, n_phys):
    """Return flat byte offset for the 128x128 Gluon-dot W layout."""
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
    n_tiles = n_phys // 128
    tile_id = (k_pack // 128).to(gl.int64) * n_tiles + (n_col // 128).to(gl.int64)
    return w_expert_off + tile_id * (128 * 128) + in_tile


@gluon.jit
def _cdna4_swizzled_mxfp4_scale_offset(
    scale_expert_off,
    n_col,
    k_scale,
    stride_slin,
    stride_snb,
):
    """Return byte offset for CDNA4-swizzled e8m0 scales.

    The preprocessor stores scales as ``(E, K_scale_padded * 32, N_padded / 32)``
    with ``stride(-2) == 1``.  This is the scalar form of the unswizzle used by
    the reference MFMA path.
    """
    n_block = n_col // 32
    n_mix = (n_col % 16) * 4 + ((n_col % 32) // 16)
    k_lin = (
        (k_scale // 8).to(gl.int64) * 256
        + (k_scale % 4).to(gl.int64) * 64
        + ((k_scale % 8) // 4).to(gl.int64) * 2
        + n_mix.to(gl.int64)
    )
    return scale_expert_off + k_lin * stride_slin + n_block.to(gl.int64) * stride_snb


# ---------------------------------------------------------------------------
# Route-owned decode: produce top-k ids/weights in Gluon.
#
# The dynamic MXFP4 route-owned path must not bounce through torch.softmax /
# torch.topk before entering warp-decode.  These kernels produce the same
# ``topk_ids`` / ``topk_weights`` contract consumed by stage1/stage2.
# ---------------------------------------------------------------------------
@gluon.jit
def _softmax_topk_route_gluon_kernel(
    logits_ptr,  # (M, E)
    bias_ptr,  # (E), read only when HAS_BIAS
    topk_ids_ptr,  # (M, TOPK) int32
    topk_weights_ptr,  # (M, TOPK) float32
    stride_lm,
    stride_le,
    stride_be,
    stride_tim,
    stride_tik,
    stride_twm,
    stride_twk,
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    MP: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    NEG: gl.constexpr = float("-inf")
    lt: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])
    row = gl.expand_dims(gl.arange(0, MP, layout=gl.SliceLayout(1, lt)), 1)
    col = gl.expand_dims(gl.arange(0, EP, layout=gl.SliceLayout(0, lt)), 0)
    mask = (row < M) & (col < E)

    logits = gl.load(
        logits_ptr + row.to(gl.int64) * stride_lm + col.to(gl.int64) * stride_le,
        mask=mask,
        other=NEG,
    ).to(gl.float32)
    rmax = gl.max(logits, axis=1, keep_dims=True)
    num = gl.exp(logits - rmax)
    den = gl.sum(num, axis=1, keep_dims=True)
    scores = gl.fdiv(num, den)
    choice = scores
    if HAS_BIAS:
        bias = gl.load(
            bias_ptr + col.to(gl.int64) * stride_be,
            mask=col < E,
            other=0.0,
        ).to(gl.float32)
        choice = choice + bias
    choice = gl.where(mask, choice, NEG)

    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, lt)), 0)
    val_t = gl.zeros([MP, TKP], gl.float32, layout=lt)
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=lt)
    big_e = gl.full([MP, EP], E, gl.int32, layout=lt)
    cur = choice
    for r in gl.static_range(TOPK):
        vmax = gl.max(cur, axis=1, keep_dims=True)
        ismax = (cur == vmax) & mask
        amax = gl.min(gl.where(ismax, col, big_e), axis=1, keep_dims=True)
        gate = gl.sum(gl.where(col == amax, scores, gl.zeros_like(scores)), axis=1)
        sel = tcol == r
        val_t = gl.where(sel, gl.expand_dims(gate, 1), val_t)
        idx_t = gl.where(sel, amax, idx_t)
        cur = gl.where(col == amax, NEG, cur)

    if NORMALIZE_TOPK_WEIGHTS:
        denom = gl.sum(val_t, axis=1, keep_dims=True)
        denom = gl.where(denom != 0.0, denom, 1.0)
        val_t = gl.fdiv(val_t, denom)
    val_t = val_t * ROUTED_SCALING_FACTOR

    m = gl.arange(0, MP, layout=gl.SliceLayout(1, lt))
    zero_i = gl.zeros([MP, TKP], gl.int32, layout=lt)
    zero_f = gl.zeros([MP, TKP], gl.float32, layout=lt)
    for r in gl.static_range(TOPK):
        sel = tcol == r
        idx_r = gl.sum(gl.where(sel, idx_t, zero_i), axis=1)
        val_r = gl.sum(gl.where(sel, val_t, zero_f), axis=1)
        valid_m = m < M
        gl.store(
            topk_ids_ptr + m.to(gl.int64) * stride_tim + r * stride_tik,
            idx_r,
            mask=valid_m,
        )
        gl.store(
            topk_weights_ptr + m.to(gl.int64) * stride_twm + r * stride_twk,
            val_r,
            mask=valid_m,
        )


@gluon.jit
def _sigmoid_bias_topk_route_gluon_kernel(
    logits_ptr,  # (M, E)
    bias_ptr,  # (E)
    topk_ids_ptr,  # (M, TOPK) int32
    topk_weights_ptr,  # (M, TOPK) float32
    stride_lm,
    stride_le,
    stride_be,
    stride_tim,
    stride_tik,
    stride_twm,
    stride_twk,
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    MP: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    X_DTYPE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    NEG: gl.constexpr = float("-inf")
    lt: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])
    row = gl.expand_dims(gl.arange(0, MP, layout=gl.SliceLayout(1, lt)), 1)
    col = gl.expand_dims(gl.arange(0, EP, layout=gl.SliceLayout(0, lt)), 0)
    mask = (row < M) & (col < E)

    logits = gl.load(
        logits_ptr + row.to(gl.int64) * stride_lm + col.to(gl.int64) * stride_le,
        mask=mask,
        other=NEG,
    ).to(gl.float32)
    # Match the reference grouped-biased route: sigmoid scores are rounded to the
    # router dtype before the bias add, top-k choice, and optional normalization.
    scores = gl.fdiv(1.0, 1.0 + gl.exp(-logits)).to(X_DTYPE)
    bias = gl.load(
        bias_ptr + col.to(gl.int64) * stride_be,
        mask=col < E,
        other=0.0,
    ).to(gl.float32)
    cur = gl.where(mask, scores.to(gl.float32) + bias, NEG)

    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, lt)), 0)
    val_t = gl.zeros([MP, TKP], gl.float32, layout=lt)
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=lt)
    big_e = gl.full([MP, EP], E, gl.int32, layout=lt)
    for r in gl.static_range(TOPK):
        vmax = gl.max(cur, axis=1, keep_dims=True)
        ismax = (cur == vmax) & mask
        amax = gl.min(gl.where(ismax, col, big_e), axis=1, keep_dims=True)
        gate = gl.sum(gl.where(col == amax, scores, gl.zeros_like(scores)), axis=1)
        sel = tcol == r
        val_t = gl.where(sel, gl.expand_dims(gate, 1), val_t)
        idx_t = gl.where(sel, amax, idx_t)
        cur = gl.where(col == amax, NEG, cur)

    if NORMALIZE_TOPK_WEIGHTS:
        val_t = val_t.to(X_DTYPE)
        denom = gl.sum(val_t, axis=1, keep_dims=True)
        denom = gl.where(denom != 0.0, denom, 1.0)
        val_t = gl.fdiv(val_t, denom) * ROUTED_SCALING_FACTOR

    m = gl.arange(0, MP, layout=gl.SliceLayout(1, lt))
    zero_i = gl.zeros([MP, TKP], gl.int32, layout=lt)
    zero_f = gl.zeros([MP, TKP], gl.float32, layout=lt)
    for r in gl.static_range(TOPK):
        sel = tcol == r
        idx_r = gl.sum(gl.where(sel, idx_t, zero_i), axis=1)
        val_r = gl.sum(gl.where(sel, val_t, zero_f), axis=1)
        valid_m = m < M
        gl.store(
            topk_ids_ptr + m.to(gl.int64) * stride_tim + r * stride_tik,
            idx_r,
            mask=valid_m,
        )
        gl.store(
            topk_weights_ptr + m.to(gl.int64) * stride_twm + r * stride_twk,
            val_r,
            mask=valid_m,
        )


def _route_supported(router_logits: torch.Tensor, topk: int) -> bool:
    if router_logits.ndim != 2 or router_logits.dtype not in _ROUTE_DTYPES:
        return False
    M, E = router_logits.shape
    return 0 < topk <= E <= _ROUTE_MAX_E and M * topk <= _ROUTE_MAX_G


def invoke_softmax_topk_route_gluon(
    router_logits: torch.Tensor,
    topk: int,
    *,
    correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route with full-row softmax semantics.

    Selection is by ``softmax(logits) + correction_bias`` when a bias is
    supplied; stored weights are the unbiased selected full-row softmax scores,
    optionally renormalized across the selected experts and always scaled.
    """
    if not _route_supported(router_logits, topk):
        raise ValueError("unsupported MXFP4 warp-decode softmax route shape")
    router_logits = router_logits.contiguous()
    if correction_bias is not None:
        if (
            correction_bias.ndim != 1
            or correction_bias.shape[0] != router_logits.shape[1]
        ):
            raise ValueError("correction_bias must be a rank-1 tensor with E elements")
        correction_bias = correction_bias.contiguous()
    M, E = router_logits.shape
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=router_logits.device)
    topk_weights = torch.empty(
        (M, topk), dtype=torch.float32, device=router_logits.device
    )
    bias = correction_bias if correction_bias is not None else topk_weights
    nw = 1 if M <= 2 else 4
    _softmax_topk_route_gluon_kernel[(1,)](
        router_logits,
        bias,
        topk_ids,
        topk_weights,
        router_logits.stride(0),
        router_logits.stride(1),
        bias.stride(0),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        M=M,
        E=E,
        TOPK=topk,
        MP=_next_pow2(M),
        EP=_next_pow2(E),
        TKP=_next_pow2(topk),
        HAS_BIAS=correction_bias is not None,
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NUM_WARPS=nw,
        num_warps=nw,
    )
    return topk_ids, topk_weights


def invoke_sigmoid_bias_topk_route_gluon(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route with DeepSeekV3/Kimi noaux_tc semantics for a single group.

    Selection is by ``sigmoid(logits) + correction_bias``; stored weights are
    the unbiased selected sigmoid scores.  With Kimi's ``n_group=1`` and
    ``topk_group=1`` the grouped route degenerates to this global top-k.
    """
    if not _route_supported(router_logits, topk):
        raise ValueError("unsupported MXFP4 warp-decode sigmoid route shape")
    if correction_bias.ndim != 1 or correction_bias.shape[0] != router_logits.shape[1]:
        raise ValueError("correction_bias must be a rank-1 tensor with E elements")
    router_logits = router_logits.contiguous()
    correction_bias = correction_bias.contiguous()
    M, E = router_logits.shape
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=router_logits.device)
    topk_weights = torch.empty(
        (M, topk), dtype=torch.float32, device=router_logits.device
    )
    nw = 1 if M <= 2 else 4
    _sigmoid_bias_topk_route_gluon_kernel[(1,)](
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
        router_logits.stride(0),
        router_logits.stride(1),
        correction_bias.stride(0),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        M=M,
        E=E,
        TOPK=topk,
        MP=_next_pow2(M),
        EP=_next_pow2(E),
        TKP=_next_pow2(topk),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        X_DTYPE=_ROUTE_GL_DTYPE[router_logits.dtype],
        NUM_WARPS=nw,
        num_warps=nw,
    )
    return topk_ids, topk_weights


# ---------------------------------------------------------------------------
# Stage 1: bf16 x fp4 W13 (gdot128 runtime layout) + e8m0 -> SwiGLU.
# ---------------------------------------------------------------------------
@gluon.jit
def _stage1_mxfp4_warp_gemv_gluon(
    x_ptr,  # hidden (num_tokens, D)         bf16
    w1_ptr,  # w1     (E, D//2 padded, 2*I_r) uint8 gdot128-shuffled
    w1s_ptr,  # w1s    (E, Kscale*32, N/32)    uint8 CDNA4-swizzled e8m0
    out_ptr,  # inter  (num_tokens*topk, I_r)  bf16
    topk_ids_ptr,  # (num_tokens, topk) int32
    D,
    I_r,
    num_tokens,
    top_k,
    stride_xm,
    stride_xk,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    N_PHYS: gl.constexpr,
    BLOCK_N: gl.constexpr,  # neurons per program
    BLOCK_KB: gl.constexpr,  # weight bytes per reduction tile (= h/2)
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_n = gl.cdiv(I_r, BLOCK_N)
    slot = pid // num_pid_n
    pid_n = pid % num_pid_n
    token = slot // top_k
    e = gl.load(topk_ids_ptr + token * stride_tit + (slot % top_k) * stride_tis)

    # warps span the neurons, lanes span the byte reduction -> gl.sum(axis=1) warp-reduces
    blk: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_KB // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, blk)
    k_layout: gl.constexpr = gl.SliceLayout(0, blk)

    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    n_mask = offs_n < I_r
    Dh = D // 2
    x_row = x_ptr + token.to(gl.int64) * stride_xm
    w_expert_off = e.to(gl.int64) * stride_we
    s_expert_off = e.to(gl.int64) * stride_se
    # gdot128 W13 runtime layout is interleaved by output column:
    # even columns are gate, odd columns are up.
    g_col = 2 * offs_n
    u_col = g_col + 1

    acc_g = gl.zeros([BLOCK_N], gl.float32, n_layout)
    acc_u = gl.zeros([BLOCK_N], gl.float32, n_layout)
    for kb0 in range(0, Dh, BLOCK_KB):
        offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=k_layout)
        kb_mask = offs_kb < Dh
        grp = offs_kb // 16  # 32 h == 16 bytes -> one e8m0 group
        tile_mask = n_mask[:, None] & kb_mask[None, :]
        # even/odd h packed in each byte: h = 2*byte (low nibble), 2*byte+1 (high)
        xe = gl.load(x_row + (2 * offs_kb) * stride_xk, mask=kb_mask, other=0.0).to(
            gl.float32
        )
        xo = gl.load(x_row + (2 * offs_kb + 1) * stride_xk, mask=kb_mask, other=0.0).to(
            gl.float32
        )
        wk_g = _gluon_dot_preshuffled_w_offset(
            w_expert_off, offs_kb[None, :], g_col[:, None], N_PHYS
        )
        wk_u = _gluon_dot_preshuffled_w_offset(
            w_expert_off, offs_kb[None, :], u_col[:, None], N_PHYS
        )
        sk_g = _cdna4_swizzled_mxfp4_scale_offset(
            s_expert_off, g_col[:, None], grp[None, :], stride_slin, stride_snb
        )
        sk_u = _cdna4_swizzled_mxfp4_scale_offset(
            s_expert_off, u_col[:, None], grp[None, :], stride_slin, stride_snb
        )
        bg = gl.load(w1_ptr + wk_g, mask=tile_mask, other=0).to(gl.int32)
        bu = gl.load(w1_ptr + wk_u, mask=tile_mask, other=0).to(gl.int32)
        sg = gl.exp2(
            (gl.load(w1s_ptr + sk_g, mask=tile_mask, other=127).to(gl.int32) - 127).to(
                gl.float32
            )
        )
        su = gl.exp2(
            (gl.load(w1s_ptr + sk_u, mask=tile_mask, other=127).to(gl.int32) - 127).to(
                gl.float32
            )
        )
        acc_g += gl.sum(
            sg
            * (
                _e2m1_to_f32(bg & 0xF) * xe[None, :]
                + _e2m1_to_f32((bg >> 4) & 0xF) * xo[None, :]
            ),
            axis=1,
        )
        acc_u += gl.sum(
            su
            * (
                _e2m1_to_f32(bu & 0xF) * xe[None, :]
                + _e2m1_to_f32((bu >> 4) & 0xF) * xo[None, :]
            ),
            axis=1,
        )

    inter = acc_g * (1.0 / (1.0 + gl.exp(-acc_g))) * acc_u  # SwiGLU: silu(gate) * up
    gl.store(
        out_ptr + slot.to(gl.int64) * stride_om + offs_n * stride_on,
        inter.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


def invoke_stage1_warp_decode_gluon(
    hidden_states,  # (num_tokens, D)   bf16
    w1,  # (E, D//2 padded, 2*I_r) uint8 gdot128-shuffled
    w1_scale,  # (E, Kscale*32, ceil(2*I_r/32)) uint8 CDNA4-swizzled
    topk_ids,  # (num_tokens, topk) int
    out,  # (num_tokens*topk, I_r) bf16
    topk,
    BLOCK_N: int = 8,
    BLOCK_KB: int = 128,
    num_warps: int = 8,
):
    assert hidden_states.dtype == torch.bfloat16 and w1.dtype == torch.uint8
    assert w1_scale.dtype == torch.uint8 and out.dtype == torch.bfloat16
    num_tokens, D = hidden_states.shape
    E, Dh_phys, two_I = w1.shape
    Dh = int(getattr(w1, "original_k_pk", Dh_phys))
    assert Dh == D // 2 and two_I % 2 == 0 and Dh % BLOCK_KB == 0
    assert bool(getattr(w1, "is_shuffled_for_gluon_dot", False))
    assert int(getattr(w1, "gluon_dot_block_k_pk", 0)) == 128
    assert int(getattr(w1, "gluon_dot_block_n", 0)) == 128
    assert w1_scale.stride(-2) == 1
    I_r = two_I // 2
    assert out.shape == (num_tokens * topk, I_r)
    topk_ids = topk_ids.to(torch.int32)
    grid = (num_tokens * topk * triton.cdiv(I_r, BLOCK_N),)
    _stage1_mxfp4_warp_gemv_gluon[grid](
        hidden_states,
        w1,
        w1_scale,
        out,
        topk_ids,
        D,
        I_r,
        num_tokens,
        topk,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        w1.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_KB=BLOCK_KB,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Stage 2: bf16 inter x fp4 W2 (gdot128 runtime layout) + e8m0, fused topk combine.
# ---------------------------------------------------------------------------
@gluon.jit
def _stage2_mxfp4_warp_gemv_gluon(
    inter_ptr,  # inter (num_tokens*topk, I_r)  bf16
    w2_ptr,  # w2    (E, I_r//2 padded, D)    uint8 gdot128-shuffled
    w2s_ptr,  # w2s   (E, Kscale*32, D/32)     uint8 CDNA4-swizzled e8m0
    y_ptr,  # out   (num_tokens, D)         bf16
    topk_ids_ptr,  # (num_tokens, topk) int32
    topk_weights_ptr,  # (num_tokens, topk) float32
    D,
    N_PHYS: gl.constexpr,
    I_r,
    num_tokens,
    top_k,
    stride_im,
    stride_ik,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    stride_yt,
    stride_yd,
    stride_tit,
    stride_tis,
    stride_twt,
    stride_tws,
    BLOCK_D: gl.constexpr,  # output dims per program
    BLOCK_KB: gl.constexpr,  # inter bytes per reduction tile (= n/2)
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_pid_d = gl.cdiv(D, BLOCK_D)
    token = pid // num_pid_d
    pid_d = pid % num_pid_d

    # warps span the output dims, lanes span the byte reduction -> gl.sum(axis=1) warp-reduces
    blk: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_D + NUM_WARPS - 1) // NUM_WARPS, BLOCK_KB // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    d_layout: gl.constexpr = gl.SliceLayout(1, blk)
    k_layout: gl.constexpr = gl.SliceLayout(0, blk)

    offs_d = pid_d * BLOCK_D + gl.arange(0, BLOCK_D, layout=d_layout)
    d_mask = offs_d < D
    Ih = I_r // 2

    acc = gl.zeros([BLOCK_D], gl.float32, d_layout)
    for s in range(0, top_k):  # fuse the topk combine
        e = gl.load(topk_ids_ptr + token * stride_tit + s * stride_tis)
        prob = gl.load(topk_weights_ptr + token * stride_twt + s * stride_tws).to(
            gl.float32
        )
        in_row = inter_ptr + (token * top_k + s).to(gl.int64) * stride_im
        w_expert_off = e.to(gl.int64) * stride_we
        s_expert_off = e.to(gl.int64) * stride_se
        dot = gl.zeros([BLOCK_D], gl.float32, d_layout)
        for kb0 in range(0, Ih, BLOCK_KB):
            offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=k_layout)
            kb_mask = offs_kb < Ih
            grp = offs_kb // 16
            tile_mask = d_mask[:, None] & kb_mask[None, :]
            ve = gl.load(
                in_row + (2 * offs_kb) * stride_ik, mask=kb_mask, other=0.0
            ).to(gl.float32)
            vo = gl.load(
                in_row + (2 * offs_kb + 1) * stride_ik, mask=kb_mask, other=0.0
            ).to(gl.float32)
            wk = _gluon_dot_preshuffled_w_offset(
                w_expert_off, offs_kb[None, :], offs_d[:, None], N_PHYS
            )
            sk = _cdna4_swizzled_mxfp4_scale_offset(
                s_expert_off, offs_d[:, None], grp[None, :], stride_slin, stride_snb
            )
            b = gl.load(
                w2_ptr + wk,
                mask=tile_mask,
                other=0,
            ).to(gl.int32)
            sc = gl.exp2(
                (
                    gl.load(
                        w2s_ptr + sk,
                        mask=tile_mask,
                        other=127,
                    ).to(gl.int32)
                    - 127
                ).to(gl.float32)
            )
            dot += gl.sum(
                sc
                * (
                    _e2m1_to_f32(b & 0xF) * ve[None, :]
                    + _e2m1_to_f32((b >> 4) & 0xF) * vo[None, :]
                ),
                axis=1,
            )
        acc += prob * dot

    gl.store(
        y_ptr + token * stride_yt + offs_d * stride_yd,
        acc.to(y_ptr.dtype.element_ty),
        mask=d_mask,
    )


def invoke_stage2_warp_decode_gluon(
    inter_states,  # (num_tokens*topk, I_r) bf16
    w2,  # (E, I_r//2 padded, D padded) uint8 gdot128-shuffled
    w2_scale,  # (E, Kscale*32, ceil(D/32)) uint8 CDNA4-swizzled
    topk_ids,  # (num_tokens, topk) int
    topk_weights,  # (num_tokens, topk) float
    out,  # (num_tokens, D) bf16
    topk,
    BLOCK_D: int = 8,
    BLOCK_KB: int = 128,
    num_warps: int = 4,
):
    assert inter_states.dtype == torch.bfloat16 and w2.dtype == torch.uint8
    assert w2_scale.dtype == torch.uint8 and out.dtype == torch.bfloat16
    _, Ih_phys, N_phys = w2.shape
    Ih = int(getattr(w2, "original_k_pk", Ih_phys))
    I_r = Ih * 2
    num_tokens, D = out.shape
    assert out.shape == (num_tokens, D)
    assert inter_states.shape == (num_tokens * topk, I_r) and Ih % BLOCK_KB == 0
    assert D <= N_phys
    assert bool(getattr(w2, "is_shuffled_for_gluon_dot", False))
    assert int(getattr(w2, "gluon_dot_block_k_pk", 0)) == 128
    assert int(getattr(w2, "gluon_dot_block_n", 0)) == 128
    assert w2_scale.stride(-2) == 1
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    grid = (num_tokens * triton.cdiv(D, BLOCK_D),)
    _stage2_mxfp4_warp_gemv_gluon[grid](
        inter_states,
        w2,
        w2_scale,
        out,
        topk_ids,
        topk_weights,
        D,
        N_phys,
        I_r,
        num_tokens,
        topk,
        inter_states.stride(0),
        inter_states.stride(1),
        w2.stride(0),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_D=BLOCK_D,
        BLOCK_KB=BLOCK_KB,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Experimental stage 2: MXFP4 inter x MXFP4 W2 direct MFMA + fused topk combine.
#
# This is the direct top-k variant of the reference combine GEMM.  It assumes the
# intermediate rows are already in (token, topk-slot) order, so it does not
# consume ragged metadata / scatter indices.  Keep it out of default dispatch
# until the matching direct stage1 path writes this row order with MXFP4 output.
# ---------------------------------------------------------------------------
@gluon.constexpr_function
def _direct_mxfp4_mfma_layouts(m_dup, block_n, block_k_scale):
    mfma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1]
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [m_dup, block_k_scale])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [block_n, block_k_scale])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _direct_mxfp4_load_tile(
    kt,
    ak,
    bk,
    ask,
    bsk,
    am,
    asm,
    x_ptr,
    x_scale_ptr,
    w_ptr,
    w_scale_ptr,
    x_row_off,
    w_expert_off,
    s_expert_off,
    n_cols,
    n_cols_s,
    x_scale_row,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    stride_slin,
    stride_snb,
    N_PHYS,
    K_DIM,
    N_DIM,
    K_PACKED: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
):
    """Load one direct MXFP4xMXFP4 K tile into MFMA operand layouts."""
    k_pack_a = kt * BLOCK_K_PACKED + ak
    k_pack_b = kt * BLOCK_K_PACKED + bk
    k_scale_a = kt * BLOCK_K_SCALE + ask
    k_scale_b = kt * BLOCK_K_SCALE + bsk
    a_off = x_row_off + k_pack_a.to(gl.int64) * stride_xk + am.to(gl.int64) * 0
    b_off = _gluon_dot_preshuffled_w_offset(w_expert_off, k_pack_b, n_cols, N_PHYS)
    a_scale_off = _cdna4_swizzled_mxfp4_scale_offset(
        0,
        x_scale_row + asm * 0,
        k_scale_a,
        stride_xslin,
        stride_xsnb,
    )
    b_scale_off = _cdna4_swizzled_mxfp4_scale_offset(
        s_expert_off,
        n_cols_s,
        k_scale_b,
        stride_slin,
        stride_snb,
    )
    a = gl.amd.cdna4.buffer_load(
        ptr=x_ptr,
        offsets=a_off.to(gl.int32),
        mask=k_pack_a < K_PACKED,
        other=0,
    )
    b = gl.amd.cdna4.buffer_load(
        ptr=w_ptr,
        offsets=b_off.to(gl.int32),
        mask=(n_cols < N_DIM) & (k_pack_b < K_PACKED),
        other=0,
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=x_scale_ptr,
        offsets=a_scale_off.to(gl.int32),
        mask=k_scale_a < (K_DIM // 32),
        other=127,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=w_scale_ptr,
        offsets=b_scale_off.to(gl.int32),
        mask=(n_cols_s < N_DIM) & (k_scale_b < (K_DIM // 32)),
        other=127,
    )
    return a, b, a_scale, b_scale


@gluon.jit
def _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale):
    return gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_scale,
        a_format="e2m1",
        b=b,
        b_scale=b_scale,
        b_format="e2m1",
        acc=acc,
    )


@gluon.constexpr_function
def _direct_swiglu_split_layout(
    block_m: int, block_n_full: int, num_warps: int
) -> gl.constexpr:
    del block_m, block_n_full
    threads_per_warp = 64
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, threads_per_warp // 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@gluon.jit
def _direct_swiglu_reduce(
    acc,
    alpha: gl.constexpr,
    limit: gl.constexpr,
    beta: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
):
    BLOCK_M: gl.constexpr = acc.shape[0]
    BLOCK_N_FULL: gl.constexpr = acc.shape[1]
    split_layout: gl.constexpr = _direct_swiglu_split_layout(
        BLOCK_M, BLOCK_N_FULL, gl.num_warps()
    )
    acc = gl.convert_layout(acc, split_layout)
    reshaped = acc.reshape((BLOCK_M, OUT_BLOCK_N, 2))
    gate, linear = gl.split(reshaped)
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.clamp(linear, -limit, limit)
    s = gate / (1.0 + gl.exp(-alpha * gate))
    return s * (linear + beta)


@gluon.jit
def _stage1_mxfp4_direct_mfma_gluon(
    hidden_ptr,  # (M, D/2) uint8 e2m1 packed, token order
    hidden_scale_ptr,  # (Kscale_pad*32, ceil(M/32)) uint8 CDNA4 swizzled
    w1_ptr,  # (E, D/2 padded, 2*I) uint8 gdot128-shuffled
    w1s_ptr,  # (E, Kscale*32, ceil((2*I)/32)) uint8 CDNA4-swizzled
    topk_ids_ptr,  # (M, TOPK) int32
    out_ptr,  # (M*TOPK, I) bf16, slot order
    M,
    D,
    TWO_I,
    N_PHYS: gl.constexpr,
    stride_xm,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    K_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    M_DUP: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // 2
    gl.static_assert(
        BLOCK_K == 128 and BLOCK_K_PACKED == 64,
        "direct MXFP4 stage1 currently assumes one CDNA4 scaled-MFMA K tile",
    )
    gl.static_assert(BLOCK_N % 2 == 0, "SwiGLU stage1 needs even BLOCK_N")
    gl.static_assert(
        128 % BLOCK_N == 0,
        "direct MXFP4 stage1 BLOCK_N must divide the gdot128 128-wide W tile",
    )

    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(TWO_I, BLOCK_N)
    slot_flat = pid // num_n
    pid_n = pid % num_n
    token = slot_flat // TOPK
    slot = slot_flat - token * TOPK
    expert = gl.load(topk_ids_ptr + token * stride_tit + slot * stride_tis)

    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(M_DUP, BLOCK_N, BLOCK_K_SCALE)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]

    am = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
    ak = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
    bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    asm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, a_scale_layout))[:, None]
    ask = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, a_scale_layout))[None, :]
    bsn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[:, None]
    bsk = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, b_scale_layout))[None, :]

    n_cols = pid_n * BLOCK_N + bn
    n_cols_s = pid_n * BLOCK_N + bsn
    x_row_off = token.to(gl.int64) * stride_xm
    w_expert_off = expert.to(gl.int64) * stride_we
    s_expert_off = expert.to(gl.int64) * stride_se
    # Keep the trip count compile-time.  With runtime ``D`` Gluon emitted one
    # load/wait/MFMA group per K tile; the constexpr bound lets the backend
    # overlap adjacent groups and cuts the 56-tile Kimi stage1 wait chain.
    TOTAL_KT: gl.constexpr = gl.cdiv(K_PACKED, BLOCK_K_PACKED)
    acc = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    if token < M:
        for kt in range(0, TOTAL_KT):
            a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                kt,
                ak,
                bk,
                ask,
                bsk,
                am,
                asm,
                hidden_ptr,
                hidden_scale_ptr,
                w1_ptr,
                w1s_ptr,
                x_row_off,
                w_expert_off,
                s_expert_off,
                n_cols,
                n_cols_s,
                token,
                stride_xk,
                stride_xslin,
                stride_xsnb,
                stride_slin,
                stride_snb,
                N_PHYS,
                D,
                TWO_I,
                K_PACKED,
                BLOCK_K_PACKED,
                BLOCK_K_SCALE,
            )
            acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)

    out_tile = _direct_swiglu_reduce(
        acc,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        SWIGLU_BETA,
        OUT_BLOCK_N,
    )
    sm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, out_tile.type.layout))[:, None]
    sn = gl.arange(0, OUT_BLOCK_N, layout=gl.SliceLayout(0, out_tile.type.layout))[
        None, :
    ]
    out_col = pid_n * OUT_BLOCK_N + sn
    gl.store(
        out_ptr
        + slot_flat.to(gl.int64) * stride_om
        + out_col.to(gl.int64) * stride_on
        + sm.to(gl.int64) * 0,
        out_tile.to(out_ptr.dtype.element_ty),
        mask=(token < M) & (sm == 0) & (out_col < (TWO_I // 2)),
    )


def invoke_stage1_mxfp4_mfma_decode_gluon(
    hidden_states_mxfp4,  # (num_tokens, D//2) uint8 e2m1, token order
    hidden_scale,  # CDNA4-swizzled e8m0 scales from _quantize_mxfp4_activation
    w1,  # (E, D//2 padded, 2*I_r) uint8 gdot128-shuffled
    w1_scale,  # (E, Kscale*32, ceil(2*I_r/32)) uint8 CDNA4-swizzled
    topk_ids,
    out,  # (num_tokens*topk, I_r) bf16
    topk: int,
    BLOCK_N: int = 32,
    BLOCK_K: int = 128,
    M_DUP: int = 4,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
):
    assert hidden_states_mxfp4.dtype == torch.uint8
    assert hidden_scale.dtype == torch.uint8
    assert w1.dtype == torch.uint8 and w1_scale.dtype == torch.uint8
    assert out.dtype == torch.bfloat16
    num_tokens = int(hidden_states_mxfp4.shape[0])
    E, Dh_phys, two_I = w1.shape
    del E
    Dh = int(getattr(w1, "original_k_pk", Dh_phys))
    D = Dh * 2
    I_r = two_I // 2
    assert two_I % 2 == 0
    assert hidden_states_mxfp4.shape == (num_tokens, Dh)
    assert out.shape == (num_tokens * topk, I_r)
    assert bool(getattr(w1, "is_shuffled_for_gluon_dot", False))
    assert int(getattr(w1, "gluon_dot_block_k_pk", 0)) == 128
    assert int(getattr(w1, "gluon_dot_block_n", 0)) == 128
    assert w1_scale.stride(-2) == 1
    assert hidden_scale.stride(-2) == 1
    topk_ids = topk_ids.to(torch.int32)
    grid = (num_tokens * topk * triton.cdiv(two_I, BLOCK_N),)
    _stage1_mxfp4_direct_mfma_gluon[grid](
        hidden_states_mxfp4,
        hidden_scale,
        w1,
        w1_scale,
        topk_ids,
        out,
        num_tokens,
        D,
        two_I,
        w1.shape[2],
        hidden_states_mxfp4.stride(0),
        hidden_states_mxfp4.stride(1),
        hidden_scale.stride(0),
        hidden_scale.stride(1),
        w1.stride(0),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        K_PACKED=Dh,
        TOPK=topk,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        M_DUP=M_DUP,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        SWIGLU_BETA=float(swiglu_beta),
        num_warps=1,
    )
    return out


@gluon.jit
def _stage2_mxfp4_direct_mfma_gluon(
    inter_ptr,  # (M*TOPK, I/2) uint8 e2m1 packed, slot order
    inter_scale_ptr,  # (Kscale_pad*32, ceil((M*TOPK)/32)) uint8 CDNA4 swizzled
    w2_ptr,  # (E, I/2 padded, D padded) uint8 gdot128-shuffled
    w2s_ptr,  # (E, Kscale*32, ceil(D/32)) uint8 CDNA4-swizzled
    topk_ids_ptr,  # (M, TOPK) int32
    topk_weights_ptr,  # (M, TOPK) float32
    out_ptr,  # (M, D) bf16
    M,
    D,
    N_PHYS: gl.constexpr,
    I_DIM,
    stride_im,
    stride_ik,
    stride_xslin,
    stride_xsnb,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    stride_om,
    stride_on,
    stride_tit,
    stride_tis,
    stride_twt,
    stride_tws,
    I_PACKED: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    M_DUP: gl.constexpr,
    PIPELINE_K: gl.constexpr,
):
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    gl.static_assert(
        BLOCK_K == 128 and BLOCK_K_PACKED == 64,
        "direct MXFP4 stage2 currently assumes one CDNA4 scaled-MFMA K tile",
    )
    gl.static_assert(
        128 % BLOCK_N == 0,
        "direct MXFP4 stage2 BLOCK_N must divide the gdot128 128-wide W tile",
    )

    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(D, BLOCK_N)
    token = pid // num_n
    pid_n = pid % num_n

    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(M_DUP, BLOCK_N, BLOCK_K_SCALE)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]

    am = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
    ak = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
    bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    asm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, a_scale_layout))[:, None]
    ask = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, a_scale_layout))[None, :]
    bsn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[:, None]
    bsk = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, b_scale_layout))[None, :]

    n_cols = pid_n * BLOCK_N + bn
    n_cols_s = pid_n * BLOCK_N + bsn
    TOTAL_KT: gl.constexpr = gl.cdiv(I_PACKED, BLOCK_K_PACKED)
    acc_total = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    if token < M:
        for slot in gl.static_range(0, TOPK):
            expert = gl.load(topk_ids_ptr + token * stride_tit + slot * stride_tis)
            gate = gl.load(
                topk_weights_ptr + token * stride_twt + slot * stride_tws
            ).to(gl.float32)
            row = token * TOPK + slot
            x_row_off = row.to(gl.int64) * stride_im
            w_expert_off = expert.to(gl.int64) * stride_we
            s_expert_off = expert.to(gl.int64) * stride_se
            acc = gl.zeros((M_DUP, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

            if PIPELINE_K and TOTAL_KT > 1:
                # Hold one K tile in VGPRs while issuing the next tile's four
                # global loads.  Kimi stage2 has only four tiles, so this small
                # lookahead hides most VMEM waits without an LDS round trip.
                a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                    0,
                    ak,
                    bk,
                    ask,
                    bsk,
                    am,
                    asm,
                    inter_ptr,
                    inter_scale_ptr,
                    w2_ptr,
                    w2s_ptr,
                    x_row_off,
                    w_expert_off,
                    s_expert_off,
                    n_cols,
                    n_cols_s,
                    row,
                    stride_ik,
                    stride_xslin,
                    stride_xsnb,
                    stride_slin,
                    stride_snb,
                    N_PHYS,
                    I_DIM,
                    D,
                    I_PACKED,
                    BLOCK_K_PACKED,
                    BLOCK_K_SCALE,
                )
                for kt in range(0, TOTAL_KT - 1):
                    next_a, next_b, next_a_scale, next_b_scale = (
                        _direct_mxfp4_load_tile(
                            kt + 1,
                            ak,
                            bk,
                            ask,
                            bsk,
                            am,
                            asm,
                            inter_ptr,
                            inter_scale_ptr,
                            w2_ptr,
                            w2s_ptr,
                            x_row_off,
                            w_expert_off,
                            s_expert_off,
                            n_cols,
                            n_cols_s,
                            row,
                            stride_ik,
                            stride_xslin,
                            stride_xsnb,
                            stride_slin,
                            stride_snb,
                            N_PHYS,
                            I_DIM,
                            D,
                            I_PACKED,
                            BLOCK_K_PACKED,
                            BLOCK_K_SCALE,
                        )
                    )
                    acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
                    a, b, a_scale, b_scale = (
                        next_a,
                        next_b,
                        next_a_scale,
                        next_b_scale,
                    )
                acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
            else:
                for kt in range(0, TOTAL_KT):
                    a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                        kt,
                        ak,
                        bk,
                        ask,
                        bsk,
                        am,
                        asm,
                        inter_ptr,
                        inter_scale_ptr,
                        w2_ptr,
                        w2s_ptr,
                        x_row_off,
                        w_expert_off,
                        s_expert_off,
                        n_cols,
                        n_cols_s,
                        row,
                        stride_ik,
                        stride_xslin,
                        stride_xsnb,
                        stride_slin,
                        stride_snb,
                        N_PHYS,
                        I_DIM,
                        D,
                        I_PACKED,
                        BLOCK_K_PACKED,
                        BLOCK_K_SCALE,
                    )
                    acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
            # Match the reference combine epilogue ordering.  Its GEMM kernel first
            # rounds each expert partial to the output dtype, multiplies by a
            # routed weight in that same dtype, stores BF16 partial rows, and
            # only then reduces top-k.  Keeping ``gate * acc`` in FP32 until
            # the final store changes thousands of Kimi decode elements by a
            # BF16 ULP even when routing, quantization, and stage 1 are exact.
            partial = acc.to(out_ptr.dtype.element_ty)
            routed_weight = gate.to(partial.dtype)
            acc_total += (partial * routed_weight).to(gl.float32)

    sm = gl.arange(0, M_DUP, layout=gl.SliceLayout(1, mfma_layout))[:, None]
    sn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))[None, :]
    col = pid_n * BLOCK_N + sn
    gl.store(
        out_ptr
        + token.to(gl.int64) * stride_om
        + col.to(gl.int64) * stride_on
        + sm.to(gl.int64) * 0,
        acc_total.to(out_ptr.dtype.element_ty),
        mask=(token < M) & (sm == 0) & (col < D),
    )


def invoke_stage2_mxfp4_mfma_decode_gluon(
    inter_states_mxfp4,  # (num_tokens*topk, I_r//2) uint8 e2m1, slot order
    inter_scale,  # CDNA4-swizzled e8m0 scales from _quantize_mxfp4_activation
    w2,  # (E, I_r//2 padded, D padded) uint8 gdot128-shuffled
    w2_scale,  # (E, Kscale*32, ceil(D/32)) uint8 CDNA4-swizzled
    topk_ids,
    topk_weights,
    out,  # (num_tokens, D) bf16
    topk: int,
    BLOCK_N: int = 16,
    BLOCK_K: int = 128,
    M_DUP: int = 4,
    PIPELINE_K: bool = True,
):
    assert inter_states_mxfp4.dtype == torch.uint8
    assert inter_scale.dtype == torch.uint8
    assert w2.dtype == torch.uint8 and w2_scale.dtype == torch.uint8
    assert out.dtype == torch.bfloat16
    _, Ih_phys, N_phys = w2.shape
    Ih = int(getattr(w2, "original_k_pk", Ih_phys))
    I_r = Ih * 2
    num_tokens, D = out.shape
    assert inter_states_mxfp4.shape == (num_tokens * topk, Ih)
    assert D <= N_phys
    assert bool(getattr(w2, "is_shuffled_for_gluon_dot", False))
    assert int(getattr(w2, "gluon_dot_block_k_pk", 0)) == 128
    assert int(getattr(w2, "gluon_dot_block_n", 0)) == 128
    assert w2_scale.stride(-2) == 1
    assert inter_scale.stride(-2) == 1
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    grid = (num_tokens * triton.cdiv(D, BLOCK_N),)
    _stage2_mxfp4_direct_mfma_gluon[grid](
        inter_states_mxfp4,
        inter_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        out,
        num_tokens,
        D,
        N_phys,
        I_r,
        inter_states_mxfp4.stride(0),
        inter_states_mxfp4.stride(1),
        inter_scale.stride(0),
        inter_scale.stride(1),
        w2.stride(0),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        I_PACKED=Ih,
        TOPK=topk,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        M_DUP=M_DUP,
        PIPELINE_K=PIPELINE_K,
        num_warps=1,
    )
    return out


def _legacy_a16w4_decode_block_n1(num_tokens: int) -> int:
    # Runtime gdot128-shuffled indexing has enough fixed address math that the
    # 8-neuron tile wins for the decode M=1/2 path.
    if num_tokens < 16:
        return 8
    return 8


def _legacy_a16w4_moe_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_n1: int | None = None,
    block_d2: int = 8,
) -> torch.Tensor:
    """Legacy BF16-activation x MXFP4-weight scalar decode benchmark.

    Args:
        hidden_states: ``(M, D)`` bf16 activations.
        w1: gdot128 runtime ``(E, D//2 padded, 2*I)`` shuffled uint8 weights.
        w1_scale: CDNA4-swizzled uint8 e8m0 scales.
        w2: gdot128 runtime ``(E, I//2 padded, D padded)`` shuffled uint8 weights.
        w2_scale: CDNA4-swizzled uint8 e8m0 scales.
        topk_ids/topk_weights: precomputed routing output.
    """
    assert hidden_states.dtype == torch.bfloat16
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1_scale.dtype == torch.uint8 and w2_scale.dtype == torch.uint8
    num_tokens, D = hidden_states.shape
    E, w1_k_phys, two_I = w1.shape
    w1_k = int(getattr(w1, "original_k_pk", w1_k_phys))
    assert w1_k == D // 2
    I_r = two_I // 2
    assert two_I % 2 == 0
    assert w2.shape[0] == E
    w2_k = int(getattr(w2, "original_k_pk", int(w2.shape[1])))
    assert w2_k * 2 == I_r
    assert int(getattr(w2, "original_n", int(w2.shape[2]))) == D
    topk = topk_ids.shape[1]
    if block_n1 is None:
        block_n1 = _legacy_a16w4_decode_block_n1(num_tokens)
    inter = torch.empty(
        (num_tokens * topk, I_r), dtype=torch.bfloat16, device=hidden_states.device
    )
    invoke_stage1_warp_decode_gluon(
        hidden_states, w1, w1_scale, topk_ids, inter, topk, BLOCK_N=block_n1
    )
    out = torch.empty(
        (num_tokens, D), dtype=torch.bfloat16, device=hidden_states.device
    )
    invoke_stage2_warp_decode_gluon(
        inter, w2, w2_scale, topk_ids, topk_weights, out, topk, BLOCK_D=block_d2
    )
    return out
