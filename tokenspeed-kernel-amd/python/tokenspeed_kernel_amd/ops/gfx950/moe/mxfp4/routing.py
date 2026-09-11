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


"""Fused dense top-k routing kernels for gfx950 A4W4 decode and package prefill."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna4 = gl.amd.cdna4
_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ROUTE_MAX_E = 1024
# Kimi K3's top-k16 decode reaches 128 routed slots at M=8. The single-CTA
# route remains faster than the generic torch fallback at that shape.
_ROUTE_MAX_G = 128
_ROUTE_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


# ---------------------------------------------------------------------------
# Route-owned dense top-k routing: produce top-k ids/weights in Gluon.
#
# The dynamic MXFP4 route-owned path must not bounce through torch.softmax /
# torch.topk before entering decode or package prefill. These kernels produce
# the same dense ``topk_ids`` / ``topk_weights`` contract consumed by
# stage1/stage2 and package prefill.
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
def _route_score_to_u32_bits(score, element_ty: gl.constexpr):
    gl.static_assert(
        element_ty == gl.float16
        or element_ty == gl.bfloat16
        or element_ty == gl.float32,
        "routing score dtype must be fp16, bf16, or fp32",
    )
    if element_ty == gl.float32:
        return score.to(gl.uint32, bitcast=True)
    else:
        return score.to(gl.uint16, bitcast=True).to(gl.uint32)


@gluon.jit
def _route_u32_bits_to_f32(bits, element_ty: gl.constexpr):
    gl.static_assert(
        element_ty == gl.float16
        or element_ty == gl.bfloat16
        or element_ty == gl.float32,
        "routing score dtype must be fp16, bf16, or fp32",
    )
    if element_ty == gl.float32:
        return bits.to(gl.float32, bitcast=True)
    else:
        return bits.to(gl.uint16).to(element_ty, bitcast=True).to(gl.float32)


@gluon.jit(
    do_not_specialize=("M",),
    do_not_specialize_on_alignment=("M",),
)
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
    M,
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
    live = mask
    topmask = gl.full([MP, EP], 0x80000000, gl.uint32, layout=lt)
    fullmask = gl.full([MP, EP], 0xFFFFFFFF, gl.uint32, layout=lt)
    zero_pack = gl.full([MP, EP], 0, gl.uint64, layout=lt)
    score_raw = _route_score_to_u32_bits(scores, X_DTYPE)
    zero_score_raw = gl.full([MP, EP], 0, gl.uint32, layout=lt)
    raw = cur.to(gl.uint32, bitcast=True)
    value_key = raw ^ gl.where((raw & topmask) != 0, fullmask, topmask)
    index_key = (EP - col).to(gl.uint32)
    packed_key = (value_key.to(gl.uint64) << 16) | index_key.to(gl.uint64)
    for r in gl.static_range(TOPK):
        packed = gl.where(live, packed_key, zero_pack)
        best = gl.max(packed, axis=1, keep_dims=True)
        amax_key = (best & 0xFFFF).to(gl.int32)
        amax = (EP - amax_key).to(gl.int32)
        chosen = live & (col == amax)
        # Gather through integer bits so signed zero and NaN payloads survive.
        gate_raw = gl.sum(gl.where(chosen, score_raw, zero_score_raw), axis=1)
        gate = _route_u32_bits_to_f32(gate_raw, X_DTYPE)
        sel = tcol == r
        val_t = gl.where(sel, gl.expand_dims(gate, 1), val_t)
        idx_t = gl.where(sel, amax, idx_t)
        live = live & (col != amax)

    if NORMALIZE_TOPK_WEIGHTS:
        # Materialize the routing dtype after reduction, division, and scaling.
        selected = val_t.to(X_DTYPE)
        denom = gl.sum(selected.to(gl.float32), axis=1, keep_dims=True).to(X_DTYPE)
        normalized = gl.div_rn(selected.to(gl.float32), denom.to(gl.float32)).to(
            X_DTYPE
        )
        scale = gl.full([MP, TKP], ROUTED_SCALING_FACTOR, gl.float32, layout=lt)
        val_t = (normalized.to(gl.float32) * scale).to(X_DTYPE).to(gl.float32)

    m = gl.arange(0, MP, layout=gl.SliceLayout(1, lt))
    zero_i = gl.zeros([MP, TKP], gl.int32, layout=lt)
    val_raw = val_t.to(gl.uint32, bitcast=True)
    zero_val_raw = gl.full([MP, TKP], 0, gl.uint32, layout=lt)
    for r in gl.static_range(TOPK):
        sel = tcol == r
        idx_r = gl.sum(gl.where(sel, idx_t, zero_i), axis=1)
        val_r_raw = gl.sum(gl.where(sel, val_raw, zero_val_raw), axis=1)
        val_r = val_r_raw.to(gl.float32, bitcast=True)
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


def gluon_topk_route_supported(router_logits: torch.Tensor, topk: int) -> bool:
    """Whether the dense-output Gluon top-k kernels support this route shape."""
    if (
        router_logits.ndim != 2
        or router_logits.dtype not in _ROUTE_DTYPES
        or not router_logits.is_cuda
    ):
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
    if not gluon_topk_route_supported(router_logits, topk):
        raise ValueError("unsupported MXFP4 Gluon softmax route shape")
    if not router_logits.is_contiguous():
        router_logits = router_logits.contiguous()
    if correction_bias is not None:
        if (
            correction_bias.ndim != 1
            or correction_bias.shape[0] != router_logits.shape[1]
        ):
            raise ValueError("correction_bias must be a rank-1 tensor with E elements")
        if not correction_bias.is_contiguous():
            correction_bias = correction_bias.contiguous()
    M, E = router_logits.shape
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=router_logits.device)
    topk_weights = torch.empty(
        (M, topk), dtype=torch.float32, device=router_logits.device
    )
    bias = correction_bias if correction_bias is not None else topk_weights
    nw = min(_next_pow2(M), 4)
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


def _launch_sigmoid_bias_topk_route_gluon(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, E = router_logits.shape
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=router_logits.device)
    topk_weights = torch.empty(
        (M, topk), dtype=torch.float32, device=router_logits.device
    )
    nw = 1 if M == 1 else min(max(_next_pow2(M), 4), 8)
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
        M,
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
    the unbiased selected sigmoid scores. With Kimi's ``n_group=1`` and
    ``topk_group=1``, grouped routing reduces to this global top-k. All
    supported dtypes compute sigmoid, stable top-k selection, and routed-weight
    normalization in this kernel while preserving the selected values' bits.
    """
    if not gluon_topk_route_supported(router_logits, topk):
        raise ValueError("unsupported MXFP4 Gluon sigmoid route shape")
    if correction_bias.ndim != 1 or correction_bias.shape[0] != router_logits.shape[1]:
        raise ValueError("correction_bias must be a rank-1 tensor with E elements")
    if not router_logits.is_contiguous():
        router_logits = router_logits.contiguous()
    if not correction_bias.is_contiguous():
        correction_bias = correction_bias.contiguous()
    return _launch_sigmoid_bias_topk_route_gluon(
        router_logits,
        correction_bias,
        topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )


# ---------------------------------------------------------------------------
# Large-M package prefill routing: one independent CTA per token.
# ---------------------------------------------------------------------------
@gluon.jit
def _sigmoid_bias_topk_route_prefill_kernel(
    logits_ptr,
    bias_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    stride_lm,
    stride_le,
    stride_be,
    stride_tim,
    stride_tik,
    stride_twm,
    stride_twk,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    X_DTYPE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    topk_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    expert = gl.arange(0, EP, layout=expert_layout)
    expert_mask = expert < E

    logits = cdna4.buffer_load(
        logits_ptr,
        (token * stride_lm + expert * stride_le).to(gl.int32),
        mask=expert_mask,
        other=-float("inf"),
    ).to(gl.float32)
    scores = gl.fdiv(1.0, 1.0 + gl.exp(-logits)).to(X_DTYPE)
    bias = cdna4.buffer_load(
        bias_ptr,
        (expert * stride_be).to(gl.int32),
        mask=expert_mask,
        other=0.0,
    ).to(gl.float32)
    choice = gl.where(expert_mask, scores.to(gl.float32) + bias, -float("inf"))

    topk_lane = gl.arange(0, TKP, layout=topk_layout)
    selected_ids = gl.zeros([TKP], gl.int32, topk_layout)
    selected_weights = gl.zeros([TKP], gl.float32, topk_layout)
    sentinel = gl.full([EP], E, gl.int32, expert_layout)
    for rank in gl.static_range(TOPK):
        maximum = gl.max(choice, axis=0)
        selected = gl.min(
            gl.where((choice == maximum) & expert_mask, expert, sentinel), axis=0
        )
        weight = gl.sum(gl.where(expert == selected, scores, 0.0), axis=0)
        selected_ids = gl.where(topk_lane == rank, selected, selected_ids)
        selected_weights = gl.where(topk_lane == rank, weight, selected_weights)
        choice = gl.where(expert == selected, -float("inf"), choice)

    if NORMALIZE_TOPK_WEIGHTS:
        selected_weights = selected_weights.to(X_DTYPE)
        denominator = gl.sum(selected_weights, axis=0)
        denominator = gl.where(denominator != 0.0, denominator, 1.0)
        selected_weights = selected_weights.to(gl.float32) * (
            ROUTED_SCALING_FACTOR / denominator
        )

    topk_mask = topk_lane < TOPK
    cdna4.buffer_store(
        selected_ids,
        topk_ids_ptr,
        (token * stride_tim + topk_lane * stride_tik).to(gl.int32),
        mask=topk_mask,
    )
    cdna4.buffer_store(
        selected_weights,
        topk_weights_ptr,
        (token * stride_twm + topk_lane * stride_twk).to(gl.int32),
        mask=topk_mask,
    )


def invoke_sigmoid_bias_topk_route_prefill_gluon(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Large-M Kimi noaux_tc routing with one independent CTA per token."""
    if (
        router_logits.ndim != 2
        or router_logits.dtype not in _ROUTE_DTYPES
        or not router_logits.is_cuda
        or not 0 < topk <= 16
        or router_logits.shape[1] > 1024
    ):
        raise ValueError("unsupported gfx950 prefill sigmoid-route shape")
    if correction_bias.shape != (router_logits.shape[1],):
        raise ValueError("correction_bias must have shape [num_experts]")

    if not router_logits.is_contiguous():
        router_logits = router_logits.contiguous()
    if not correction_bias.is_contiguous():
        correction_bias = correction_bias.contiguous()
    tokens, experts = router_logits.shape
    topk_ids = torch.empty(
        (tokens, topk), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=router_logits.device
    )
    num_warps = 1
    _sigmoid_bias_topk_route_prefill_kernel[(tokens,)](
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
        E=experts,
        TOPK=topk,
        EP=_next_pow2(experts),
        TKP=_next_pow2(topk),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        X_DTYPE=_ROUTE_GL_DTYPE[router_logits.dtype],
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return topk_ids, topk_weights


__all__ = [
    "gluon_topk_route_supported",
    "invoke_sigmoid_bias_topk_route_gluon",
    "invoke_sigmoid_bias_topk_route_prefill_gluon",
    "invoke_softmax_topk_route_gluon",
]
