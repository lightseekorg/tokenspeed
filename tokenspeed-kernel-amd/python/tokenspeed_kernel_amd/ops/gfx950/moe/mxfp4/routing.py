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
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.simulation import permute_experts

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


@gluon.jit
def _local_route_map_kernel(
    topk_ids_ptr,
    expanded_idx_ptr,
    num_routes,
    expert_start,
    expert_end,
    BLOCK: gl.constexpr,
):
    route = gl.program_id(0) * BLOCK + gl.arange(
        0,
        BLOCK,
        layout=gl.BlockedLayout([1], [64], [4], [0]),
    )
    valid = route < num_routes
    expert = gl.load(topk_ids_ptr + route, mask=valid, other=-1)
    owned = valid & (expert >= expert_start) & (expert < expert_end)
    gl.store(expanded_idx_ptr + route, gl.where(owned, route, -1), mask=valid)


def gluon_local_route_map(
    topk_ids: torch.Tensor,
    expert_start: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Map local routes to canonical rows and mark remote EP routes with -1.

    Args:
        topk_ids: Global expert IDs with shape ``[tokens, top_k]``.
        expert_start: First global expert ID owned by this rank.
        num_local_experts: Number of contiguous experts owned by this rank.

    Returns:
        A flattened int32 route map. Local routes map to their canonical row;
        routes owned by another EP rank contain ``-1``.
    """

    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [tokens, top_k]")
    if expert_start < 0 or num_local_experts <= 0:
        raise ValueError("the local expert range must be non-empty and non-negative")
    topk_ids = topk_ids.to(torch.int32).contiguous()
    num_routes = topk_ids.numel()
    expanded_idx = torch.empty(
        (num_routes,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    if num_routes == 0:
        return expanded_idx
    block = 256
    _local_route_map_kernel[(triton.cdiv(num_routes, block),)](
        topk_ids,
        expanded_idx,
        num_routes,
        expert_start,
        expert_start + num_local_experts,
        BLOCK=block,
        num_warps=4,
    )
    return expanded_idx


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
    do_not_specialize=("M", "simulation_seed"),
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
    simulation_seed,
    SIMULATE: gl.constexpr,
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

    if SIMULATE:
        idx_t = permute_experts(idx_t, row, simulation_seed, E)

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
    simulation_seed: int | None,
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
        simulation_seed=0 if simulation_seed is None else simulation_seed,
        SIMULATE=simulation_seed is not None,
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
    simulation_seed: int | None,
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
        simulation_seed=simulation_seed,
    )


# ---------------------------------------------------------------------------
# Large-M package prefill routing: one independent CTA per token.
# ---------------------------------------------------------------------------
@gluon.jit
def _max_unique_key(keys):
    """Cross-lane max of 64-bit keys whose low words are distinct (or zero).

    Two 32-bit maxima (the high words, then the low words of the lanes that
    hold the highest) pick the same key as a 64-bit max, but each reduction
    step is one fused DPP max instead of a 64-bit compare and two selects.
    """
    high = (keys >> 32).to(gl.uint32)
    best_high = gl.max(high, axis=0)
    low = gl.where(high == best_high, keys.to(gl.uint32), 0)
    best_low = gl.max(low, axis=0)
    return (best_high.to(gl.uint64) << 32) | best_low.to(gl.uint64)


@gluon.jit
def sigmoid_bias_topk_row_gfx950(
    token,
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
    simulation_seed,
    SIMULATE: gl.constexpr,
):
    """Select one token's experts as a tournament over per-lane candidates.

    Each lane owns a column of experts and keeps its six best as sorted
    heads, so a selection round is a single cross-lane max over one key per
    lane instead of a reduction over every expert. A lane that gives up all
    six heads may hold further winners, so the token then repeats the
    selection over all experts; with four heads that happened often enough
    to slow a 64-row batch down to its slowest token. Routers of at most
    320 experts hold at most five per lane, so they extract only as many
    heads as a lane holds and never repeat the selection.

    The body of :func:`gluon_sigmoid_bias_topk_gfx950` for row ``token``, so a
    kernel that runs other work in its other programs can route rows too.
    """
    LANES: gl.constexpr = 64 * NUM_WARPS
    ROWS: gl.constexpr = (EP + LANES - 1) // LANES
    expert_layout: gl.constexpr = gl.BlockedLayout(
        [ROWS, 1], [1, 64], [1, NUM_WARPS], [0, 1]
    )
    lane_layout: gl.constexpr = gl.SliceLayout(0, expert_layout)
    topk_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    row = gl.arange(0, ROWS, layout=gl.SliceLayout(1, expert_layout))
    lane = gl.arange(0, LANES, layout=lane_layout)
    expert = row[:, None] * LANES + lane[None, :]
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
    # Order-preserving choice bits over an inverted id: ties in max go to the lower id.
    choice_bits = (scores.to(gl.float32) + bias).to(gl.uint32, bitcast=True)
    choice_bits ^= gl.where((choice_bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(
        gl.uint32
    )
    keys = (choice_bits.to(gl.uint64) << 32) | (ROWS * LANES - 1 - expert).to(gl.uint64)
    keys = gl.where(expert_mask, keys, 0)

    # Heads past a lane's experts stay empty; only LANE_EXPERTS > 5 can exhaust all six.
    LANE_EXPERTS: gl.constexpr = (E + LANES - 1) // LANES
    remaining = keys
    head_0 = gl.max(remaining, axis=0)
    head_1 = gl.zeros([LANES], gl.uint64, lane_layout)
    head_2 = gl.zeros([LANES], gl.uint64, lane_layout)
    head_3 = gl.zeros([LANES], gl.uint64, lane_layout)
    head_4 = gl.zeros([LANES], gl.uint64, lane_layout)
    head_5 = gl.zeros([LANES], gl.uint64, lane_layout)
    if LANE_EXPERTS > 1:
        remaining = gl.where(remaining == head_0[None, :], 0, remaining)
        head_1 = gl.max(remaining, axis=0)
    if LANE_EXPERTS > 2:
        remaining = gl.where(remaining == head_1[None, :], 0, remaining)
        head_2 = gl.max(remaining, axis=0)
    if LANE_EXPERTS > 3:
        remaining = gl.where(remaining == head_2[None, :], 0, remaining)
        head_3 = gl.max(remaining, axis=0)
    if LANE_EXPERTS > 4:
        remaining = gl.where(remaining == head_3[None, :], 0, remaining)
        head_4 = gl.max(remaining, axis=0)
    if LANE_EXPERTS > 5:
        remaining = gl.where(remaining == head_4[None, :], 0, remaining)
        head_5 = gl.max(remaining, axis=0)
    taken = gl.zeros([LANES], gl.int32, lane_layout)

    topk_lane = gl.arange(0, TKP, layout=topk_layout)
    selected_keys = gl.zeros([TKP], gl.uint64, topk_layout)
    for rank in gl.static_range(TOPK):
        best = _max_unique_key(head_0)
        selected_keys = gl.where(topk_lane == rank, best, selected_keys)
        won = head_0 == best
        taken += won.to(gl.int32)
        head_0 = gl.where(won, head_1, head_0)
        head_1 = gl.where(won, head_2, head_1)
        head_2 = gl.where(won, head_3, head_2)
        head_3 = gl.where(won, head_4, head_3)
        head_4 = gl.where(won, head_5, head_4)
        head_5 = gl.where(won, 0, head_5)
    if LANE_EXPERTS > 5:
        if gl.max(taken, axis=0) >= 6:
            for rank in gl.static_range(TOPK):
                best = _max_unique_key(gl.max(keys, axis=0))
                selected_keys = gl.where(topk_lane == rank, best, selected_keys)
                keys = gl.where(keys == best, 0, keys)

    topk_mask = topk_lane < TOPK
    selected_ids = gl.where(
        topk_mask,
        (ROWS * LANES - 1 - (selected_keys & 0xFFFFFFFF)).to(gl.int32),
        0,
    )
    # Recomputing the selected scores reproduces their bits exactly.
    selected_logits = cdna4.buffer_load(
        logits_ptr,
        (token * stride_lm + selected_ids * stride_le).to(gl.int32),
        mask=topk_mask,
        other=0.0,
    ).to(gl.float32)
    selected_weights = gl.fdiv(1.0, 1.0 + gl.exp(-selected_logits)).to(X_DTYPE)
    selected_weights = gl.where(topk_mask, selected_weights.to(gl.float32), 0.0)

    if NORMALIZE_TOPK_WEIGHTS:
        selected_weights = selected_weights.to(X_DTYPE)
        denominator = gl.sum(selected_weights, axis=0)
        denominator = gl.where(denominator != 0.0, denominator, 1.0)
        selected_weights = selected_weights.to(gl.float32) * (
            ROUTED_SCALING_FACTOR / denominator
        )
    else:
        selected_weights *= ROUTED_SCALING_FACTOR

    if SIMULATE:
        selected_ids = permute_experts(selected_ids, token, simulation_seed, E)

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


@gluon.jit(do_not_specialize=["simulation_seed"])
def gluon_sigmoid_bias_topk_gfx950(
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
    simulation_seed,
    SIMULATE: gl.constexpr,
):
    sigmoid_bias_topk_row_gfx950(
        gl.program_id(0),
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
        E,
        TOPK,
        EP,
        TKP,
        NORMALIZE_TOPK_WEIGHTS,
        ROUTED_SCALING_FACTOR,
        X_DTYPE,
        NUM_WARPS,
        simulation_seed,
        SIMULATE,
    )


def invoke_sigmoid_bias_topk_route_prefill_gluon(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
    simulation_seed: int | None,
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
    gluon_sigmoid_bias_topk_gfx950[(tokens,)](
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
        simulation_seed=0 if simulation_seed is None else simulation_seed,
        SIMULATE=simulation_seed is not None,
        num_warps=num_warps,
    )
    return topk_ids, topk_weights


__all__ = [
    "gluon_topk_route_supported",
    "sigmoid_bias_topk_row_gfx950",
    "invoke_sigmoid_bias_topk_route_gluon",
    "invoke_sigmoid_bias_topk_route_prefill_gluon",
    "invoke_softmax_topk_route_gluon",
]
