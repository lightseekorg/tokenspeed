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

"""Fused MoE routing: single-launch Gluon route kernels producing ragged
metadata + gather/scatter indices + gate scales for bounded M, torch
reference implementations, and capability predicates.

Distinct from mxfp4/routing.py, which wraps the staged package
top-k-only route kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
    RaggedTensorMetadata,
    make_ragged_tensor_metadata,
    topk,
)

_ROUTING_METHOD_RENORMALIZE = 1


def _uses_grouped_routing(n_group: int, topk_group: int) -> bool:
    return n_group > 0 and topk_group > 0


def _has_incomplete_grouped_routing(n_group: int, topk_group: int) -> bool:
    return (n_group > 0) != (topk_group > 0)


def _normalize_route_weights(
    topk_weights: torch.Tensor,
    *,
    normalize_topk_weights: bool,
    routed_scaling_factor: float,
    scale_when_unnormalized: bool,
) -> torch.Tensor:
    if normalize_topk_weights:
        tiny = torch.finfo(topk_weights.dtype).tiny
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            tiny
        )
    if normalize_topk_weights or scale_when_unnormalized:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights


def _stable_topk_smaller_index(
    values: torch.Tensor,
    k: int,
    *,
    dim: int = -1,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k with the same exact-tie rule as the reference streaming top-k.

    The reference ranks a packed ``(ordered float bits, inverse index)`` integer key,
    so equal floating-point values select the smaller expert id.  ``torch.topk``
    does not define which index wins an exact tie, which is observable for BF16
    router logits after sigmoid.  Pack the same key here while gathering the
    original values so non-tied ordering and route weights remain unchanged.
    """
    if values.dtype == torch.float32:
        integer_dtype = torch.int32
        value_mask = 0xFFFFFFFF
        sign_mask = 0x80000000
    elif values.dtype in (torch.float16, torch.bfloat16):
        integer_dtype = torch.int16
        value_mask = 0xFFFF
        sign_mask = 0x8000
    else:
        raise TypeError(
            "stable route top-k supports float16, bfloat16, and float32; "
            f"got {values.dtype}"
        )

    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise IndexError(f"top-k dimension {dim} is invalid for rank {values.ndim}")
    width = int(values.shape[dim])
    if not 0 < k <= width:
        raise ValueError(f"top-k requires 0 < k <= {width}; got {k}")
    if width >= 1 << 16:
        raise ValueError(
            f"stable route top-k supports fewer than 65536 values: {width}"
        )

    raw = values.contiguous().view(integer_dtype).to(torch.int64) & value_mask
    # Build the flip masks on-device with ``full_like`` rather than
    # ``raw.new_tensor(<python int>)``: the latter materializes a CPU tensor and
    # copies it to the GPU, which is illegal during CUDA-graph capture.
    ordered = raw ^ torch.where(
        (raw & sign_mask) != 0,
        torch.full_like(raw, value_mask),
        torch.full_like(raw, sign_mask),
    )
    index_shape = [1] * values.ndim
    index_shape[dim] = width
    index = torch.arange(width, device=values.device, dtype=torch.int64).view(
        index_shape
    )
    packed = (ordered << 16) | (width - index)
    _, topk_ids = torch.topk(packed, k=k, dim=dim, sorted=sorted)
    return values.gather(dim, topk_ids), topk_ids


def _softmax_topk_reference(
    logits: torch.Tensor,
    topk: int,
    *,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(logits.float(), dim=-1)
    scores_for_choice = scores
    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.to(scores.dtype).unsqueeze(0)
    _, topk_ids = _stable_topk_smaller_index(
        scores_for_choice, k=topk, dim=-1, sorted=True
    )
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = _normalize_route_weights(
        topk_weights,
        normalize_topk_weights=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        scale_when_unnormalized=True,
    )
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _grouped_topk_reference(
    logits: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(logits.float(), dim=-1)
    n_tokens, n_experts = scores.shape
    group_scores = scores.view(n_tokens, n_group, -1).max(dim=-1).values
    _, group_idx = _stable_topk_smaller_index(
        group_scores, k=topk_group, dim=-1, sorted=False
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(n_tokens, n_group, n_experts // n_group)
        .reshape(n_tokens, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    topk_weights, topk_ids = _stable_topk_smaller_index(
        tmp_scores, k=topk, dim=-1, sorted=False
    )
    topk_weights = _normalize_route_weights(
        topk_weights,
        normalize_topk_weights=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        scale_when_unnormalized=False,
    )
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def default_scaled_route(
    logits: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_weights, topk_ids = _softmax_topk_reference(
        logits,
        topk,
        correction_bias=None,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )
    return _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts=logits.shape[1],
        dtype=dtype,
    )


def default_packed_topk_route(
    logits: torch.Tensor,
    topk: int,
    *,
    normalize_topk_weights: bool,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_logits, topk_ids = _stable_topk_smaller_index(
        logits, k=topk, dim=-1, sorted=True
    )
    topk_weights = topk_logits.exp()
    topk_weights = _normalize_route_weights(
        topk_weights,
        normalize_topk_weights=normalize_topk_weights,
        routed_scaling_factor=1.0,
        scale_when_unnormalized=False,
    )
    return _route_from_topk(
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        num_experts=logits.shape[1],
        dtype=dtype,
    )


def default_biased_route(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_weights, topk_ids = _softmax_topk_reference(
        logits,
        topk,
        correction_bias=correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )
    return _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts=logits.shape[1],
        dtype=dtype,
    )


def default_grouped_route(
    logits: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_weights, topk_ids = _grouped_topk_reference(
        logits,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )
    return _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts=logits.shape[1],
        dtype=dtype,
    )


# ===========================================================================
# Small-M (decode) fused MoE routing in Gluon.
#
# Decode routing is launch-overhead bound. For route shapes satisfying both
# ``M <= SMALLM_MAX_M`` and ``G = M*topk <= GLUON_ROUTE_MAX_G`` this replaces
# the generic ``triton_kernels_routing`` pipeline (~12 kernel launches) with a
# single Gluon kernel, producing output bit-for-bit identical to the generic
# path. Larger M or G falls back; the caller gates on both bounds.
#
# Why the bounds make this exact: ``M <= 16`` means every nonzero expert holds
# exactly one RaggedTensorMetadata block (single-block collapse), and
# ``G = M*topk <= GLUON_ROUTE_MAX_G`` keeps the register-only counting sort in
# the supported rank-tile regime. The kernel fuses in-kernel top-k,
# histogram/cumsum, single-block schedule, and counting sort, reproducing
# ``moe_route(traits={"output_type": "ragged_metadata"})``:
# ``RaggedTensorMetadata`` + gather_indx/scatter_indx/gate_scal of length
# ``G``. Metadata shapes are queried from ``RaggedTensorMetadata`` so they match
# ``make_ragged_tensor_metadata`` on HIP and non-HIP alike.
# ===========================================================================

# Number of block-size rows in RaggedTensorMetadata for the active platform
# ([16,32,64,128,256] -> 5 on HIP, [16,32,64,128] -> 4 otherwise). Derived
# from the library so the metadata shapes match make_ragged_tensor_metadata
# exactly on every target.
_ROUTE_NB = len(RaggedTensorMetadata.block_sizes())


# Token-count bound for single-block collapse. 16 == the smallest
# RaggedTensorMetadata block size, so for M <= 16 every expert's token count is
# ``col_sum <= M <= 16``. The flat gate count ``G = M*topk`` is bounded
# separately below; callers must satisfy both bounds to use the Gluon route.
SMALLM_MAX_M = 16


# Backwards-compatible alias for the small-M bound.
FUSED_ROUTE_MAX_M = SMALLM_MAX_M


# Configs the Gluon routing path supports; everything else falls back to the
# generic triton_kernels_routing pipeline.
GLUON_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


GLUON_ROUTE_MAX_E = 1024  # next_pow2(E) bins / EP-wide tiles stay bounded


# Flat gate-count bound, where ``G = M*topk``. The stable-sort rank tile is
# [GP, GP] and the kernel's layouts assume the single-wavefront regime
# (GP <= 64); configs that exceed it fall back to the generic pipeline.
GLUON_ROUTE_MAX_G = 64


# torch gate dtype -> gluon element type (for the in-kernel softmax cast that
# reproduces topk_forward's ``softmax(...).to(x_dtype)`` rounding exactly).
_ROUTE_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


@gluon.jit
def _route_add(a, b):
    return a + b


@gluon.jit
def _fused_topk(
    Logits,  # [M, E]   X_DTYPE   (raw routing logits)
    stride_lm,  # logits row stride
    gmask,  # [GP]   bool     g < G
    tok,  # [GP]      int32    g // TOPK
    slot,  # [GP]     int32    g %  TOPK
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    MP: gl.constexpr,  # next_pow2(M)
    EP: gl.constexpr,  # next_pow2(E)
    GP: gl.constexpr,  # next_pow2(M*topk)
    TKP: gl.constexpr,  # next_pow2(topk)
    X_DTYPE: gl.constexpr,  # gate element type (logits dtype)
    L1: gl.constexpr,  # 1D blocked layout used by the consuming kernel
    LT: gl.constexpr,  # 2D blocked layout for the [MP, EP] logits tile
):
    """Fused in-kernel top-k matching ``topk_forward(apply_softmax=True)``.

    Selects, per token row, the top ``TOPK`` experts by logit value (ties to
    the smaller expert id, descending value order) and     the softmax gate over
    the selected logits -- reproducing the triton kernels ``_topk_forward``
    semantics without a separate launch or a ``y_vals``/``y_indx`` global
    round-trip.
    Returns flat ``(idx[GP] int32, vals[GP] X_DTYPE)`` in token-major gate
    order (``g = token*TOPK + slot``), ready for the counting sort.
    """
    NEG: gl.constexpr = float("-inf")
    # ---- load the [MP, EP] logits tile (invalid lanes -> -inf) -------------
    row = gl.expand_dims(gl.arange(0, MP, layout=gl.SliceLayout(1, LT)), 1)  # [MP,1]
    col = gl.expand_dims(gl.arange(0, EP, layout=gl.SliceLayout(0, LT)), 0)  # [1,EP]
    lmask = (row < M) & (col < E)
    cur = gl.load(Logits + row * stride_lm + col, mask=lmask, other=NEG).to(gl.float32)

    # ---- iterative arg-max top-k (descending value, smaller-id tie-break) --
    # Equivalent to streaming_topk's packed sort: max value wins, ties resolve
    # to the smaller expert index; the iteration emits experts in descending
    # value order, matching topk_forward's output slot order. Results are
    # written column-by-column into [MP, TKP] tiles (no python lists, which
    # gluon tracing does not support).
    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, LT)), 0)  # [1,TKP]
    val_t = gl.full([MP, TKP], -1e30, gl.float32, layout=LT)  # finite -inf-ish
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=LT)
    live = lmask
    topmask = gl.full([MP, EP], 0x80000000, gl.uint32, layout=LT)
    fullmask = gl.full([MP, EP], 0xFFFFFFFF, gl.uint32, layout=LT)
    zero_pack = gl.full([MP, EP], 0, gl.uint64, layout=LT)
    for _r in gl.static_range(TOPK):
        # Match the generic Triton top-k strategy: rank a packed key that
        # carries both the float value ordering and the expert index. This keeps
        # the selected index valid even for NaN/inf logits, without remapping the
        # original selected value used by the softmax below.
        raw = cur.to(gl.uint32, bitcast=True)
        value_key = raw ^ gl.where((raw & topmask) != 0, fullmask, topmask)
        index_key = (EP - col).to(gl.uint32)
        packed = (value_key.to(gl.uint64) << 16) | index_key.to(gl.uint64)
        packed = gl.where(live, packed, zero_pack)
        best = gl.max(packed, axis=1, keep_dims=True)
        amax_key = (best & 0xFFFF).to(gl.int32)
        amax = (EP - amax_key).to(gl.int32)  # [MP,1]
        chosen = live & (col == amax)
        vmax = gl.sum(gl.where(chosen, cur, gl.zeros_like(cur)), axis=1, keep_dims=True)
        sel = tcol == _r  # [1,TKP]
        val_t = gl.where(sel, vmax, val_t)  # write column _r
        idx_t = gl.where(sel, amax, idx_t)
        live = live & (col != amax)  # drop chosen expert

    # ---- softmax over the selected logits (matches tl.softmax in fp32) -----
    # z = x - max(x); num = exp(z); den = sum(num); gate = fdiv(num, den).
    # Padding columns (TOPK..TKP) hold -1e30 -> exp(-) == 0 -> ignored.
    rmax = gl.max(val_t, axis=1, keep_dims=True)  # [MP,1]
    num = gl.exp(val_t - rmax)  # [MP,TKP]
    den = gl.sum(num, axis=1, keep_dims=True)  # [MP,1]
    gate_t = gl.fdiv(num, den)  # [MP,TKP] fp32

    # ---- flatten per-slot columns into the flat [GP] gate order -----------
    z_i = gl.zeros([MP, TKP], gl.int32, layout=LT)
    z_f = gl.zeros([MP, TKP], gl.float32, layout=LT)
    idx = gl.zeros([GP], gl.int32, layout=L1)
    valsf = gl.zeros([GP], gl.float32, layout=L1)
    for _r in gl.static_range(TOPK):
        sel = tcol == _r  # [1,TKP]
        idx_r = gl.convert_layout(gl.sum(gl.where(sel, idx_t, z_i), axis=1), L1)
        gat_r = gl.convert_layout(gl.sum(gl.where(sel, gate_t, z_f), axis=1), L1)
        take = (slot == _r) & gmask
        idx = gl.where(take, gl.gather(idx_r, tok, axis=0), idx)
        valsf = gl.where(take, gl.gather(gat_r, tok, axis=0), valsf)
    # cast like topk_forward's softmax(...).to(x_dtype) before the gate store.
    return idx, valsf.to(X_DTYPE)


@gluon.jit
def _fused_biased_grouped_topk(
    Logits,  # [M, E]   X_DTYPE   (raw routing logits)
    CorrectionBias,  # [E]      fp32      expert correction bias
    stride_lm,  # logits row stride
    gmask,  # [GP]     bool      g < G
    tok,  # [GP]     int32     g // TOPK
    slot,  # [GP]     int32     g % TOPK
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    N_GROUP: gl.constexpr,
    TOPK_GROUP: gl.constexpr,
    EXPERTS_PER_GROUP: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    MP: gl.constexpr,
    EP: gl.constexpr,
    GP: gl.constexpr,
    TKP: gl.constexpr,
    NGP: gl.constexpr,
    X_DTYPE: gl.constexpr,
    L1: gl.constexpr,
    LT: gl.constexpr,
):
    NEG: gl.constexpr = float("-inf")

    row = gl.expand_dims(gl.arange(0, MP, layout=gl.SliceLayout(1, LT)), 1)
    col = gl.expand_dims(gl.arange(0, EP, layout=gl.SliceLayout(0, LT)), 0)
    lmask = (row < M) & (col < E)

    logits = gl.load(Logits + row * stride_lm + col, mask=lmask, other=NEG).to(
        gl.float32
    )
    bias = gl.load(CorrectionBias + col, mask=col < E, other=NEG).to(gl.float32)
    scores = gl.fdiv(1.0, 1.0 + gl.exp(-logits)).to(X_DTYPE)
    choice = gl.where(lmask, scores.to(gl.float32) + bias, NEG)

    gcol = gl.expand_dims(gl.arange(0, NGP, layout=gl.SliceLayout(0, LT)), 0)
    group_scores = gl.full([MP, NGP], NEG, gl.float32, layout=LT)
    big_e = gl.full([MP, EP], E, gl.int32, layout=LT)
    expert_group = col // EXPERTS_PER_GROUP

    for _g in gl.static_range(N_GROUP):
        in_group = lmask & (expert_group == _g)
        best1 = gl.max(gl.where(in_group, choice, NEG), axis=1, keep_dims=True)
        best1_expert = gl.min(
            gl.where(in_group & (choice == best1), col, big_e),
            axis=1,
            keep_dims=True,
        )
        choice2 = gl.where(col == best1_expert, NEG, choice)
        best2 = gl.max(gl.where(in_group, choice2, NEG), axis=1, keep_dims=True)
        group_scores = gl.where(gcol == _g, best1 + best2, group_scores)

    group_cur = group_scores
    group_selected = gl.zeros([MP, NGP], gl.int32, layout=LT)
    big_g = gl.full([MP, NGP], N_GROUP, gl.int32, layout=LT)
    for _r in gl.static_range(TOPK_GROUP):
        gmax = gl.max(group_cur, axis=1, keep_dims=True)
        gbest = gl.min(
            gl.where(group_cur == gmax, gcol, big_g),
            axis=1,
            keep_dims=True,
        )
        group_selected = gl.where(gcol == gbest, 1, group_selected)
        group_cur = gl.where(gcol == gbest, NEG, group_cur)

    expert_selected = gl.zeros([MP, EP], gl.int32, layout=LT)
    zero_groups = gl.zeros([MP, NGP], gl.int32, layout=LT)
    for _g in gl.static_range(N_GROUP):
        selected = gl.sum(
            gl.where(gcol == _g, group_selected, zero_groups),
            axis=1,
            keep_dims=True,
        )
        expert_selected = gl.where(expert_group == _g, selected, expert_selected)

    cur = gl.where((expert_selected > 0) & lmask, choice, NEG)

    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, LT)), 0)
    val_t = gl.zeros([MP, TKP], gl.float32, layout=LT)
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=LT)
    for _r in gl.static_range(TOPK):
        vmax = gl.max(cur, axis=1, keep_dims=True)
        ismax = (cur == vmax) & (col < E)
        amax = gl.min(gl.where(ismax, col, big_e), axis=1, keep_dims=True)
        gate = gl.max(gl.where(col == amax, scores, 0.0), axis=1, keep_dims=True)
        sel = tcol == _r
        val_t = gl.where(sel, gate, val_t)
        idx_t = gl.where(sel, amax, idx_t)
        cur = gl.where(col == amax, NEG, cur)

    if NORMALIZE_TOPK_WEIGHTS:
        # Match the Python grouped route for bf16 router logits: selected gates
        # are bf16 and the per-token normalization is performed in that dtype.
        val_t = val_t.to(X_DTYPE)
        den = gl.sum(val_t, axis=1, keep_dims=True)
        den = gl.where(den != 0.0, den, 1.0)
        val_t = gl.fdiv(val_t, den) * ROUTED_SCALING_FACTOR

    z_i = gl.zeros([MP, TKP], gl.int32, layout=LT)
    z_f = gl.zeros([MP, TKP], gl.float32, layout=LT)
    idx = gl.zeros([GP], gl.int32, layout=L1)
    valsf = gl.zeros([GP], gl.float32, layout=L1)
    for _r in gl.static_range(TOPK):
        sel = tcol == _r
        idx_r = gl.convert_layout(gl.sum(gl.where(sel, idx_t, z_i), axis=1), L1)
        gat_r = gl.convert_layout(gl.sum(gl.where(sel, val_t, z_f), axis=1), L1)
        take = (slot == _r) & gmask
        idx = gl.where(take, gl.gather(idx_r, tok, axis=0), idx)
        valsf = gl.where(take, gl.gather(gat_r, tok, axis=0), valsf)
    return idx, valsf.to(X_DTYPE)


# ===========================================================================
# Small route shapes: M <= 16 and G=M*topk <= 64.
# Single-workgroup, stable-order, single-block collapse.
# ===========================================================================
@gluon.jit
def _fused_route_small_m(
    Logits,  # [M, E]       X_DTYPE (raw routing logits)
    SliceSizes,  # [E]          int32
    SliceOffs,  # [E+1]         int32
    BlockOffs,  # [NB, E+1]     int32
    BlockSched,  # [NB, MAXBLK] int32
    GatherIndx,  # [G]          int32
    ScatterIndx,  # [G]         int32
    GateScal,  # [G]           dtype
    stride_lm,  # logits row stride
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    MP: gl.constexpr,  # next_pow2(M)
    GP: gl.constexpr,  # next_pow2(M*topk)
    EP: gl.constexpr,  # next_pow2(E)
    TKP: gl.constexpr,  # next_pow2(topk)
    MAXBLK: gl.constexpr,  # == M*topk
    MAXBLKP: gl.constexpr,  # next_pow2(MAXBLK)
    NB_C: gl.constexpr,  # number of block-size rows (NB)
    X_DTYPE: gl.constexpr,  # gate element type (logits dtype)
    NW_C: gl.constexpr,  # num_warps (1 for the M<=2 decode hot path, else 4)
    bo_stride: gl.constexpr,  # block_offs row stride  == E+1
    bs_stride: gl.constexpr,  # block_sched row stride == MAXBLK
):
    G: gl.constexpr = M * TOPK
    # Layouts are parametric in NW_C. At M<=2 a single warp (NW_C=1) removes the
    # cross-warp s_barrier stalls (LDS reductions over 4 warps) that dominated
    # the decode hot path; for larger small-M the O(G^2) rank tile + top-k want
    # 4 warps, so NW_C=4 there.
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])  # 1D (EP)
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])  # 1D (GP)
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])  # 1D (MAXBLKP)
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW_C, 1], [1, 0])  # 2D

    # ---- fused top-k: compute (expert id, softmax gate) for each of the
    # G=M*TOPK flat gates in-kernel,
    # replacing the separate topk_forward launch + y_vals/y_indx round-trip.
    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    tok = (g // TOPK).to(gl.int32)
    slot = (g % TOPK).to(gl.int32)
    idx, vals = _fused_topk(
        Logits,
        stride_lm,
        gmask,
        tok,
        slot,
        M,
        E,
        TOPK,
        MP,
        EP,
        GP,
        TKP,
        X_DTYPE,
        LG,
        LT,
    )

    # ---- histogram -> slice_sizes -----------------------------------------
    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    hist = gl.histogram(idx, EP, mask=gmask, layout=LE)
    gl.store(SliceSizes + e, hist, mask=emask)

    # ---- slice_offs = [0] + cumsum(slice_sizes) ---------------------------
    # Store exclusive prefixes at 0..E-1; index E (the total) is the only entry
    # the inclusive scan uniquely supplies, so write just that one element
    # rather than re-writing 1..E-1 with identical values.
    incl = gl.associative_scan(hist, 0, _route_add)
    col_offs = incl - hist
    last = e == (E - 1)
    gl.store(SliceOffs + e, col_offs, mask=emask)
    gl.store(SliceOffs + e + 1, incl, mask=emask & last)

    # ---- block_offs_data / block_schedule_data ----------------------------
    # Single-block collapse: M<=16 bounds every per-expert token count to one
    # block, while the separate G=M*TOPK bound keeps the rank tile small. All
    # NB rows are identical, and the packed block value is just the expert id.
    n_blk = (hist > 0).to(gl.int32)
    blk_incl = gl.associative_scan(n_blk, 0, _route_add)
    blk_excl = blk_incl - n_blk
    n_total = gl.sum(n_blk, 0)
    jb = gl.arange(0, MAXBLKP, layout=LB)
    jbmask = jb < MAXBLK
    neg_fill = gl.full([MAXBLKP], -1, gl.int32, layout=LB)
    for k in gl.static_range(NB_C):
        gl.store(BlockOffs + k * bo_stride + e, blk_excl, mask=emask)
        gl.store(BlockOffs + k * bo_stride + e + 1, blk_incl, mask=emask & last)
        # Fill -1 only in the tail (jb >= n_total). It is disjoint from the
        # scatter targets [0, n_total) below, so the compiler cannot reorder
        # the two stores into an alias that clobbers scattered ids.
        gl.store(
            BlockSched + k * bs_stride + jb,
            neg_fill,
            mask=jbmask & (jb >= n_total),
        )
        # Packed value is the bare expert id (single block, so block index 0).
        gl.store(
            BlockSched + k * bs_stride + blk_excl,
            e,
            mask=(hist > 0) & emask,
        )

    # ---- stable per-expert rank -------------------------------------------
    # rank[g] = #{j<g : idx[j]==idx[g]}. idx is in registers post-fuse, so use
    # a [GP,GP] compare tile reduced over j; cheap since GP <= 64.
    idx_row = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(1, LT)), 1)
    idx_col = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(0, LT)), 0)
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row)).to(gl.int32)
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)

    # ---- scatter to destination = slice_offs[expert] + rank ---------------
    pos = gl.gather(col_offs, idx, axis=0) + rank
    gl.store(GatherIndx + pos, tok, mask=gmask)
    gl.store(ScatterIndx + pos, g.to(gl.int32), mask=gmask)
    gl.store(GateScal + pos, vals, mask=gmask)


@gluon.jit
def _fused_biased_grouped_route_small_m(
    Logits,  # [M, E]       X_DTYPE (raw routing logits)
    CorrectionBias,  # [E]          fp32
    SliceSizes,  # [E]          int32
    SliceOffs,  # [E+1]        int32
    BlockOffs,  # [NB, E+1]    int32
    BlockSched,  # [NB, MAXBLK] int32
    GatherIndx,  # [G]          int32
    ScatterIndx,  # [G]         int32
    GateScal,  # [G]           dtype
    stride_lm,  # logits row stride
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    N_GROUP: gl.constexpr,
    TOPK_GROUP: gl.constexpr,
    EXPERTS_PER_GROUP: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    MP: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    NGP: gl.constexpr,
    MAXBLK: gl.constexpr,
    MAXBLKP: gl.constexpr,
    NB_C: gl.constexpr,
    X_DTYPE: gl.constexpr,
    NW_C: gl.constexpr,
    bo_stride: gl.constexpr,
    bs_stride: gl.constexpr,
):
    G: gl.constexpr = M * TOPK
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW_C, 1], [1, 0])

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    tok = (g // TOPK).to(gl.int32)
    slot = (g % TOPK).to(gl.int32)
    idx, vals = _fused_biased_grouped_topk(
        Logits,
        CorrectionBias,
        stride_lm,
        gmask,
        tok,
        slot,
        M,
        E,
        TOPK,
        N_GROUP,
        TOPK_GROUP,
        EXPERTS_PER_GROUP,
        NORMALIZE_TOPK_WEIGHTS,
        ROUTED_SCALING_FACTOR,
        MP,
        EP,
        GP,
        TKP,
        NGP,
        X_DTYPE,
        LG,
        LT,
    )

    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    hist = gl.histogram(idx, EP, mask=gmask, layout=LE)
    gl.store(SliceSizes + e, hist, mask=emask)

    incl = gl.associative_scan(hist, 0, _route_add)
    col_offs = incl - hist
    last = e == (E - 1)
    gl.store(SliceOffs + e, col_offs, mask=emask)
    gl.store(SliceOffs + e + 1, incl, mask=emask & last)

    n_blk = (hist > 0).to(gl.int32)
    blk_incl = gl.associative_scan(n_blk, 0, _route_add)
    blk_excl = blk_incl - n_blk
    n_total = gl.sum(n_blk, 0)
    jb = gl.arange(0, MAXBLKP, layout=LB)
    jbmask = jb < MAXBLK
    neg_fill = gl.full([MAXBLKP], -1, gl.int32, layout=LB)
    for k in gl.static_range(NB_C):
        gl.store(BlockOffs + k * bo_stride + e, blk_excl, mask=emask)
        gl.store(BlockOffs + k * bo_stride + e + 1, blk_incl, mask=emask & last)
        gl.store(
            BlockSched + k * bs_stride + jb,
            neg_fill,
            mask=jbmask & (jb >= n_total),
        )
        gl.store(
            BlockSched + k * bs_stride + blk_excl,
            e,
            mask=(hist > 0) & emask,
        )

    idx_row = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(1, LT)), 1)
    idx_col = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(0, LT)), 0)
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row)).to(gl.int32)
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)

    pos = gl.gather(col_offs, idx, axis=0) + rank
    gl.store(GatherIndx + pos, tok, mask=gmask)
    gl.store(ScatterIndx + pos, g.to(gl.int32), mask=gmask)
    gl.store(GateScal + pos, vals, mask=gmask)


@gluon.jit
def _precomputed_topk_route_small_m(
    TopkIds,  # [M, TOPK] int32
    TopkWeights,  # [M, TOPK] fp/bf
    SliceSizes,  # [E] int32
    SliceOffs,  # [E+1] int32
    BlockOffs,  # [NB, E+1] int32
    BlockSched,  # [NB, MAXBLK] int32
    GatherIndx,  # [G] int32
    ScatterIndx,  # [G] int32
    GateScal,  # [G] dtype
    stride_tim,
    stride_tik,
    stride_twm,
    stride_twk,
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    MAXBLK: gl.constexpr,
    MAXBLKP: gl.constexpr,
    NB_C: gl.constexpr,
    X_DTYPE: gl.constexpr,
    NW_C: gl.constexpr,
    bo_stride: gl.constexpr,
    bs_stride: gl.constexpr,
):
    G: gl.constexpr = M * TOPK
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [NW_C], [0])
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW_C, 1], [1, 0])

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    tok = (g // TOPK).to(gl.int32)
    slot = (g % TOPK).to(gl.int32)

    idx_raw = gl.load(
        TopkIds + tok.to(gl.int64) * stride_tim + slot.to(gl.int64) * stride_tik,
        mask=gmask,
        other=0,
    ).to(gl.int32)
    valid = gmask & (idx_raw >= 0) & (idx_raw < E)
    idx = gl.where(valid, idx_raw, gl.zeros([GP], gl.int32, layout=LG))
    vals = gl.load(
        TopkWeights + tok.to(gl.int64) * stride_twm + slot.to(gl.int64) * stride_twk,
        mask=valid,
        other=0.0,
    ).to(gl.float32)

    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    hist = gl.histogram(idx, EP, mask=valid, layout=LE)
    gl.store(SliceSizes + e, hist, mask=emask)

    incl = gl.associative_scan(hist, 0, _route_add)
    col_offs = incl - hist
    last = e == (E - 1)
    gl.store(SliceOffs + e, col_offs, mask=emask)
    gl.store(SliceOffs + e + 1, incl, mask=emask & last)

    n_blk = (hist > 0).to(gl.int32)
    blk_incl = gl.associative_scan(n_blk, 0, _route_add)
    blk_excl = blk_incl - n_blk
    n_total = gl.sum(n_blk, 0)
    jb = gl.arange(0, MAXBLKP, layout=LB)
    jbmask = jb < MAXBLK
    neg_fill = gl.full([MAXBLKP], -1, gl.int32, layout=LB)
    for k in gl.static_range(NB_C):
        gl.store(BlockOffs + k * bo_stride + e, blk_excl, mask=emask)
        gl.store(BlockOffs + k * bo_stride + e + 1, blk_incl, mask=emask & last)
        gl.store(
            BlockSched + k * bs_stride + jb,
            neg_fill,
            mask=jbmask & (jb >= n_total),
        )
        gl.store(
            BlockSched + k * bs_stride + blk_excl,
            e,
            mask=(hist > 0) & emask,
        )

    idx_row = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(1, LT)), 1)
    idx_col = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(0, LT)), 0)
    valid_row = gl.expand_dims(gl.convert_layout(valid, gl.SliceLayout(1, LT)), 1)
    valid_col = gl.expand_dims(gl.convert_layout(valid, gl.SliceLayout(0, LT)), 0)
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = (valid_row & valid_col & (idx_row == idx_col) & (g_col < g_row)).to(
        gl.int32
    )
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)

    pos = gl.gather(col_offs, idx, axis=0) + rank
    gl.store(GatherIndx + pos, tok, mask=valid)
    gl.store(ScatterIndx + pos, g.to(gl.int32), mask=valid)
    gl.store(GateScal + pos, vals.to(X_DTYPE), mask=valid)


@gluon.jit
def _precomputed_topk_route_m1_flat(
    TopkIds,  # [1, TOPK] int32
    TopkWeights,  # [1, TOPK] fp/bf
    SliceSizes,  # [E] int32
    SliceOffs,  # [E+1] int32
    BlockOffs,  # [NB, E+1] int32
    BlockSched,  # [NB, TOPK] int32
    GatherIndx,  # [TOPK] int32
    ScatterIndx,  # [TOPK] int32
    GateScal,  # [TOPK] dtype
    stride_tik,
    stride_twk,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    EP: gl.constexpr,
    TKP: gl.constexpr,
    NB_C: gl.constexpr,
    X_DTYPE: gl.constexpr,
    bo_stride: gl.constexpr,
    bs_stride: gl.constexpr,
):
    """Flat precomputed route for M=1.

    For one token, torch/top-k returns unique expert ids, so each active expert
    owns exactly one flat slot row. The matmul block-schedule path only needs
    the active experts' slice offsets and a compact schedule; it does not need
    the full histogram/prefix/rank route used for M>=2.
    """
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    LT: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])

    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    gl.store(SliceSizes + e, gl.full([EP], 0, gl.int32, layout=LE), mask=emask)
    gl.store(SliceOffs + e, gl.full([EP], 0, gl.int32, layout=LE), mask=emask)
    last = e == (E - 1)
    gl.store(SliceOffs + e + 1, TOPK, mask=emask & last)
    for k in gl.static_range(NB_C):
        gl.store(
            BlockOffs + k * bo_stride + e,
            gl.full([EP], 0, gl.int32, layout=LE),
            mask=emask,
        )
        gl.store(BlockOffs + k * bo_stride + e + 1, TOPK, mask=emask & last)

    slot = gl.arange(0, TKP, layout=LT)
    smask = slot < TOPK
    expert = gl.load(TopkIds + slot * stride_tik, mask=smask, other=0).to(gl.int32)
    valid = smask & (expert >= 0) & (expert < E)
    weight = gl.load(TopkWeights + slot * stride_twk, mask=valid, other=0.0).to(
        gl.float32
    )

    gl.store(SliceSizes + expert, 1, mask=valid)
    gl.store(SliceOffs + expert, slot.to(gl.int32), mask=valid)
    for k in gl.static_range(NB_C):
        gl.store(
            BlockSched + k * bs_stride + slot,
            expert,
            mask=valid,
        )
    gl.store(GatherIndx + slot, 0, mask=smask)
    gl.store(ScatterIndx + slot, slot.to(gl.int32), mask=smask)
    gl.store(GateScal + slot, weight.to(X_DTYPE), mask=valid)


# ===========================================================================
# Host wrappers for the small-M fused route
# ===========================================================================
def _route_next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def _route_small_m(logits, topk, dtype):
    """1-kernel stable-order fused route for bounded M and G=M*topk."""
    M, E = logits.shape
    G = M * topk
    device = logits.device
    logits = logits.contiguous()

    slice_sizes = torch.empty(E, dtype=torch.int32, device=device)
    slice_offs = torch.empty(E + 1, dtype=torch.int32, device=device)
    block_offs_data = torch.empty(_ROUTE_NB, E + 1, dtype=torch.int32, device=device)
    # Query the library for the block-schedule width so it stays exact on any
    # platform rather than hardcoding the small-M value.
    maxblk = RaggedTensorMetadata.max_n_blocks(E, G)
    block_schedule_data = torch.empty(
        _ROUTE_NB, maxblk, dtype=torch.int32, device=device
    )
    gather_indx = torch.empty(G, dtype=torch.int32, device=device)
    scatter_indx = torch.empty(G, dtype=torch.int32, device=device)
    gate_scal = torch.empty(G, dtype=dtype, device=device)

    # M<=2 is the launch-bound decode hot path: a single warp removes the
    # cross-warp s_barrier stalls. Larger small-M has enough work (O(G^2) rank
    # tile + top-k) to benefit from 4 warps.
    nw = 1 if M <= 2 else 4

    _fused_route_small_m[(1,)](
        logits,
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
        logits.stride(0),
        M=M,
        E=E,
        TOPK=topk,
        MP=_route_next_pow2(M),
        GP=_route_next_pow2(G),
        EP=_route_next_pow2(E),
        TKP=_route_next_pow2(topk),
        MAXBLK=maxblk,
        MAXBLKP=_route_next_pow2(maxblk),
        NB_C=_ROUTE_NB,
        X_DTYPE=_ROUTE_GL_DTYPE[logits.dtype],
        NW_C=nw,
        bo_stride=block_offs_data.stride(0),
        bs_stride=block_schedule_data.stride(0),
        num_warps=nw,
    )

    ragged = RaggedTensorMetadata(
        slice_sizes, slice_offs, block_offs_data, block_schedule_data
    )
    return ragged, gather_indx, scatter_indx, gate_scal


def gluon_precomputed_topk_route_supported(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    dtype: torch.dtype,
) -> bool:
    if (
        topk_weights.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or dtype not in GLUON_ROUTE_DTYPES
    ):
        return False
    M, topk = topk_ids.shape
    G = M * topk
    return (
        M <= SMALLM_MAX_M
        and G <= GLUON_ROUTE_MAX_G
        and 0 < topk <= num_experts <= GLUON_ROUTE_MAX_E
    )


def gluon_precomputed_topk_flat_m1_route(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Specialized flat precomputed route for a single decode token.

    The output row order remains the caller's top-k slot order. This is valid
    for M=1 because top-k ids are unique within a token, so each active expert
    has one contiguous row. M>=2 still uses the general expert-grouped route
    because experts can repeat across tokens.
    """
    if dtype is None:
        dtype = topk_weights.dtype
    if not gluon_precomputed_topk_route_supported(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=dtype,
    ):
        raise ValueError("unsupported precomputed-topk Gluon route configuration")
    if int(topk_ids.shape[0]) != 1:
        raise ValueError("flat M=1 route requires exactly one token")

    device = topk_ids.device
    topk = int(topk_ids.shape[1])
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    slice_sizes = torch.empty(num_experts, dtype=torch.int32, device=device)
    slice_offs = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    block_offs_data = torch.empty(
        _ROUTE_NB, num_experts + 1, dtype=torch.int32, device=device
    )
    block_schedule_data = torch.empty(_ROUTE_NB, topk, dtype=torch.int32, device=device)
    gather_indx = torch.empty(topk, dtype=torch.int32, device=device)
    scatter_indx = torch.empty(topk, dtype=torch.int32, device=device)
    gate_scal = torch.empty(topk, dtype=dtype, device=device)

    _precomputed_topk_route_m1_flat[(1,)](
        topk_ids,
        topk_weights,
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
        topk_ids.stride(1),
        topk_weights.stride(1),
        E=num_experts,
        TOPK=topk,
        EP=_route_next_pow2(num_experts),
        TKP=_route_next_pow2(topk),
        NB_C=_ROUTE_NB,
        X_DTYPE=_ROUTE_GL_DTYPE[dtype],
        bo_stride=block_offs_data.stride(0),
        bs_stride=block_schedule_data.stride(0),
        num_warps=1,
    )
    ragged = RaggedTensorMetadata(
        slice_sizes, slice_offs, block_offs_data, block_schedule_data
    )
    return ragged, gather_indx, scatter_indx, gate_scal


def _route_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    # Ids outside [0, num_experts) are unrouted -- expert parallelism marks
    # every pair owned by another rank this way. They must sort *after* all
    # real experts: parking them at expert 0 instead would interleave them with
    # genuine expert-0 rows, while the slice sizes below count only valid
    # pairs, so every expert's ragged slice would then address the wrong rows.
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    safe_ids = torch.where(valid, flat_ids, flat_ids.new_full((), num_experts))
    sort_order = torch.argsort(safe_ids, stable=True)

    top_k = topk_ids.shape[1]
    # sort_order defines the expert-sorted ragged row order. GEMM1 gathers
    # source token rows; GEMM2 scatters back to flat token/top-k rows.
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    gate_scal = topk_weights.reshape(-1)[sort_order]
    gate_scal = torch.where(valid[sort_order], gate_scal, torch.zeros_like(gate_scal))
    if dtype is not None and gate_scal.dtype != dtype:
        gate_scal = gate_scal.to(dtype)

    # One extra bucket absorbs the unrouted sentinel; it accumulates zero
    # because those entries contribute ``valid == 0``, and is dropped before
    # building the ragged metadata.
    col_sum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=safe_ids.device)
    col_sum.scatter_add_(0, safe_ids, valid.to(torch.int32))
    col_sum = col_sum[:num_experts]
    ragged_metadata = make_ragged_tensor_metadata(col_sum, int(sort_order.numel()))
    return ragged_metadata, gather_indx, scatter_indx, gate_scal


@gluon.jit
def _fused_precomputed_topk_route_small_m(
    TopkWeights,  # [M, TOPK] dtype
    TopkIds,  # [M, TOPK] int32
    SliceSizes,  # [E] int32
    SliceOffs,  # [E+1] int32
    BlockOffs,  # [NB, E+1] int32
    BlockSched,  # [NB, MAXBLK] int32
    GatherIndx,  # [G] int32
    ScatterIndx,  # [G] int32
    GateScal,  # [G] dtype
    stride_wm,
    stride_im,
    M: gl.constexpr,
    E: gl.constexpr,
    TOPK: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    MAXBLK: gl.constexpr,
    MAXBLKP: gl.constexpr,
    NB_C: gl.constexpr,
    bo_stride: gl.constexpr,
    bs_stride: gl.constexpr,
):
    G: gl.constexpr = M * TOPK
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    LB: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    tok = (g // TOPK).to(gl.int32)
    slot = (g % TOPK).to(gl.int32)
    idx = gl.load(TopkIds + tok * stride_im + slot, mask=gmask, other=0).to(gl.int32)
    valid = gmask & (idx >= 0) & (idx < E)
    safe_idx = gl.where(valid, idx, 0)
    vals = gl.load(TopkWeights + tok * stride_wm + slot, mask=gmask, other=0.0)
    vals = gl.where(valid, vals, 0.0)

    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    hist = gl.histogram(safe_idx, EP, mask=valid, layout=LE)
    gl.store(SliceSizes + e, hist, mask=emask)

    incl = gl.associative_scan(hist, 0, _route_add)
    col_offs = incl - hist
    last = e == (E - 1)
    gl.store(SliceOffs + e, col_offs, mask=emask)
    gl.store(SliceOffs + e + 1, incl, mask=emask & last)

    n_blk = (hist > 0).to(gl.int32)
    blk_incl = gl.associative_scan(n_blk, 0, _route_add)
    blk_excl = blk_incl - n_blk
    n_total = gl.sum(n_blk, 0)
    jb = gl.arange(0, MAXBLKP, layout=LB)
    jbmask = jb < MAXBLK
    neg_fill = gl.full([MAXBLKP], -1, gl.int32, layout=LB)
    for k in gl.static_range(NB_C):
        gl.store(BlockOffs + k * bo_stride + e, blk_excl, mask=emask)
        gl.store(BlockOffs + k * bo_stride + e + 1, blk_incl, mask=emask & last)
        gl.store(
            BlockSched + k * bs_stride + jb,
            neg_fill,
            mask=jbmask & (jb >= n_total),
        )
        gl.store(
            BlockSched + k * bs_stride + blk_excl,
            e,
            mask=(hist > 0) & emask,
        )

    idx_row = gl.expand_dims(gl.convert_layout(safe_idx, gl.SliceLayout(1, LT)), 1)
    idx_col = gl.expand_dims(gl.convert_layout(safe_idx, gl.SliceLayout(0, LT)), 0)
    valid_row = gl.expand_dims(gl.convert_layout(valid, gl.SliceLayout(1, LT)), 1)
    valid_col = gl.expand_dims(gl.convert_layout(valid, gl.SliceLayout(0, LT)), 0)
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & valid_row & valid_col & (g_col < g_row)).to(
        gl.int32
    )
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)

    pos = gl.gather(col_offs, safe_idx, axis=0) + rank
    gl.store(GatherIndx + pos, tok, mask=valid)
    gl.store(ScatterIndx + pos, g.to(gl.int32), mask=valid)
    gl.store(GateScal + pos, vals, mask=valid)


def gluon_precomputed_topk_fused_route(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if dtype is None:
        dtype = topk_weights.dtype
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must both be rank-2 with same shape"
        )
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if topk_weights.dtype not in GLUON_ROUTE_DTYPES or dtype not in GLUON_ROUTE_DTYPES:
        raise ValueError(
            f"unsupported topk weight dtype: {topk_weights.dtype}, output dtype: {dtype}"
        )

    M, topk = topk_ids.shape
    if M < 1 or M > SMALLM_MAX_M:
        raise ValueError(
            f"precomputed fused route requires 1 <= M <= {SMALLM_MAX_M}, got {M}"
        )
    if topk < 1 or topk > num_experts:
        raise ValueError(f"invalid topk={topk} for num_experts={num_experts}")
    if num_experts < 1 or num_experts > GLUON_ROUTE_MAX_E:
        raise ValueError(
            f"precomputed fused route supports 1 <= num_experts <= {GLUON_ROUTE_MAX_E}, "
            f"got {num_experts}"
        )
    if M * topk > GLUON_ROUTE_MAX_G:
        raise ValueError(
            f"precomputed fused route requires M*topk <= {GLUON_ROUTE_MAX_G}, "
            f"got {M * topk}"
        )
    G = M * topk
    device = topk_ids.device
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()

    slice_sizes = torch.empty(num_experts, dtype=torch.int32, device=device)
    slice_offs = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    block_offs_data = torch.empty(
        _ROUTE_NB, num_experts + 1, dtype=torch.int32, device=device
    )
    maxblk = RaggedTensorMetadata.max_n_blocks(num_experts, G)
    block_schedule_data = torch.empty(
        _ROUTE_NB, maxblk, dtype=torch.int32, device=device
    )
    gather_indx = torch.empty(G, dtype=torch.int32, device=device)
    scatter_indx = torch.empty(G, dtype=torch.int32, device=device)
    gate_scal = torch.empty(G, dtype=dtype, device=device)

    _fused_precomputed_topk_route_small_m[(1,)](
        topk_weights,
        topk_ids,
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
        topk_weights.stride(0),
        topk_ids.stride(0),
        M=M,
        E=num_experts,
        TOPK=topk,
        GP=_route_next_pow2(G),
        EP=_route_next_pow2(num_experts),
        MAXBLK=maxblk,
        MAXBLKP=_route_next_pow2(maxblk),
        NB_C=_ROUTE_NB,
        bo_stride=block_offs_data.stride(0),
        bs_stride=block_schedule_data.stride(0),
        num_warps=1,
    )
    ragged = RaggedTensorMetadata(
        slice_sizes, slice_offs, block_offs_data, block_schedule_data
    )
    return ragged, gather_indx, scatter_indx, gate_scal


def _biased_grouped_topk_reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits.sigmoid()
    n_tokens, n_experts = scores.shape
    scores_for_choice = scores + correction_bias.unsqueeze(0)
    group_top2, _ = _stable_topk_smaller_index(
        scores_for_choice.view(n_tokens, n_group, -1),
        k=2,
        dim=-1,
        sorted=True,
    )
    group_scores = group_top2.sum(dim=-1)
    _, group_idx = _stable_topk_smaller_index(
        group_scores, k=topk_group, dim=-1, sorted=True
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(n_tokens, n_group, n_experts // n_group)
        .reshape(n_tokens, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_ids = _stable_topk_smaller_index(tmp_scores, k=topk, dim=-1, sorted=True)
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights *= routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _biased_grouped_route_small_m(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    dtype: torch.dtype,
):
    M, E = logits.shape
    G = M * topk
    device = logits.device
    logits = logits.contiguous()
    correction_bias = correction_bias.contiguous()

    slice_sizes = torch.empty(E, dtype=torch.int32, device=device)
    slice_offs = torch.empty(E + 1, dtype=torch.int32, device=device)
    block_offs_data = torch.empty(_ROUTE_NB, E + 1, dtype=torch.int32, device=device)
    maxblk = RaggedTensorMetadata.max_n_blocks(E, G)
    block_schedule_data = torch.empty(
        _ROUTE_NB, maxblk, dtype=torch.int32, device=device
    )
    gather_indx = torch.empty(G, dtype=torch.int32, device=device)
    scatter_indx = torch.empty(G, dtype=torch.int32, device=device)
    gate_scal = torch.empty(G, dtype=dtype, device=device)

    nw = 1 if M <= 2 else 4
    _fused_biased_grouped_route_small_m[(1,)](
        logits,
        correction_bias,
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
        logits.stride(0),
        M=M,
        E=E,
        TOPK=topk,
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=E // n_group,
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        MP=_route_next_pow2(M),
        GP=_route_next_pow2(G),
        EP=_route_next_pow2(E),
        TKP=_route_next_pow2(topk),
        NGP=_route_next_pow2(n_group),
        MAXBLK=maxblk,
        MAXBLKP=_route_next_pow2(maxblk),
        NB_C=_ROUTE_NB,
        X_DTYPE=_ROUTE_GL_DTYPE[logits.dtype],
        NW_C=nw,
        bo_stride=block_offs_data.stride(0),
        bs_stride=block_schedule_data.stride(0),
        num_warps=nw,
    )

    ragged = RaggedTensorMetadata(
        slice_sizes, slice_offs, block_offs_data, block_schedule_data
    )
    return ragged, gather_indx, scatter_indx, gate_scal


def gluon_route_supported(
    logits: torch.Tensor,
    topk: int,
    dtype: torch.dtype | None = None,
) -> bool:
    """Whether the unified Gluon routing path supports this configuration.

    Guards the structural assumptions the Gluon kernels make so unsupported
    configs fall back to the generic ``triton_kernels_routing`` pipeline:
    a 2D float ``logits`` tensor,     a supported gate ``dtype``, a sane ``topk``
    and an expert count whose ``next_pow2`` keeps the histogram bins / EP-wide
    tiles bounded.
    """
    if logits.ndim != 2:
        return False
    if dtype is None:
        dtype = logits.dtype
    if logits.dtype not in GLUON_ROUTE_DTYPES or dtype not in GLUON_ROUTE_DTYPES:
        return False
    M, E = logits.shape
    if topk < 1 or topk > E:
        return False
    if E < 1 or E > GLUON_ROUTE_MAX_E:
        return False
    # G = M*topk drives the [GP, GP] rank tile / single-wavefront layouts.
    if M * topk > GLUON_ROUTE_MAX_G:
        return False
    return True


def gluon_biased_grouped_route_supported(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    dtype: torch.dtype | None = None,
) -> bool:
    if not gluon_route_supported(logits, topk, dtype):
        return False
    if correction_bias.ndim != 1 or correction_bias.shape[0] != logits.shape[1]:
        return False
    _, E = logits.shape
    if n_group <= 0 or topk_group <= 0:
        return False
    if E % n_group != 0 or E // n_group < 2:
        return False
    if topk_group > n_group:
        return False
    return True


def gluon_fused_route(
    logits: torch.Tensor,
    topk: int,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small-M (decode) fused MoE routing.

    Reproduces ``moe_route(traits={"output_type": "ragged_metadata"})`` in a
    single Gluon kernel, returning ``(ragged_metadata, gather_indx,
    scatter_indx, gate_scal)`` bit-for-bit identical to the generic pipeline.
    Valid when both ``M <= SMALLM_MAX_M`` and
    ``G = M*topk <= GLUON_ROUTE_MAX_G`` hold; callers gate on both bounds and
    fall back to the generic pipeline otherwise.
    """
    if dtype is None:
        dtype = logits.dtype
    M = logits.shape[0]
    if M > SMALLM_MAX_M:
        raise ValueError(
            f"gluon_fused_route requires M <= {SMALLM_MAX_M} "
            f"(single-block-collapse regime); got M={M}. Route larger M "
            "through the generic triton_kernels_routing pipeline."
        )
    return _route_small_m(logits, topk, dtype)


def gluon_biased_grouped_fused_route(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if dtype is None:
        dtype = logits.dtype
    M = logits.shape[0]
    if M > SMALLM_MAX_M:
        raise ValueError(
            f"gluon_biased_grouped_fused_route requires M <= {SMALLM_MAX_M}, "
            f"got M={M}"
        )
    if not gluon_biased_grouped_route_supported(
        logits,
        correction_bias,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        dtype=dtype,
    ):
        raise ValueError("unsupported grouped-biased Gluon route configuration")
    return _biased_grouped_route_small_m(
        logits,
        correction_bias,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        dtype=dtype,
    )


def default_biased_grouped_route(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    topk_weights, topk_ids = _biased_grouped_topk_reference(
        logits,
        correction_bias,
        topk,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )
    return _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts=logits.shape[1],
        dtype=dtype,
    )


def default_route(
    logits: torch.Tensor,
    n_expts_act: int,
    dtype: torch.dtype | None = None,
) -> tuple[RaggedTensorMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = logits.dtype

    assert logits.ndim == 2, "router_logits must be (n_tokens, n_expts_tot)"
    n_tokens, _ = logits.shape

    sparse = topk(logits, n_expts_act, apply_softmax=True)
    mask_metadata = sparse.mask_metadata

    col_sorted = mask_metadata.col_sorted_indx
    gather_indx = col_sorted // n_expts_act
    scatter_indx = col_sorted

    vals_flat = sparse.vals.reshape(-1)
    if dtype is not None and vals_flat.dtype != dtype:
        vals_flat = vals_flat.to(dtype)
    gate_scal = vals_flat[scatter_indx]

    n_total_rows = n_tokens * n_expts_act
    ragged_metadata = make_ragged_tensor_metadata(mask_metadata.col_sum, n_total_rows)

    return ragged_metadata, gather_indx, scatter_indx, gate_scal
