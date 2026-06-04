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

# ===========================================================================
# Small-M (decode) fused MoE routing, implemented in Gluon.
#
# For small token counts (``M <= SMALLM_MAX_M``) this replaces the generic
# ``triton_kernels_routing`` post-topk pipeline -- bitmatrix metadata
# stage1/stage2, row-sum, ragged-metadata memset/compute, gather/scatter index
# build, the ``//topk`` divide, the gate_scal gather and dtype copies (roughly
# a dozen kernel launches) -- with a SINGLE Gluon kernel. Decode routing is
# launch-overhead bound, so collapsing ~12 launches into one is a large win
# (~6x cheaper routing, ~1.1-1.4x end-to-end at M=1..16). Larger M keeps the
# generic pipeline; the caller gates on the bound (one ``if``).
#
# Routing is a COUNTING SORT: gates ``g in [0, M*topk)`` are bucketed by their
# selected expert, producing ``RaggedTensorMetadata`` plus the gather/scatter
# index tensors -- reproducing, field-for-field, the contract that
# ``moe_route(traits={"output_type": "ragged_metadata"})`` produces:
#   RaggedTensorMetadata(slice_sizes, slice_offs, block_offs_data,
#                        block_schedule_data),
#   gather_indx [G] int32, scatter_indx [G] int32, gate_scal [G] dtype
#   where G = M * topk.
# The result is BIT-FOR-BIT identical to the generic pipeline (``M <= 16`` is
# the single-block-collapse regime, so the index tensors are stable too).
#
# The single kernel (``_fused_route_smallM``) fuses: in-kernel top-k (selection
# + softmax gate, matching ``topk_forward``, so there is no separate topk
# launch and no ``y_vals``/``y_indx`` global round-trip); the histogram and
# slice-offset cumsum; the single-block schedule (``col_sum <= M <= 16 == min
# block size`` so every nonzero expert is exactly one block); and a stable
# register-only [G,G] rank tile keeping the gather/scatter/gate order
# bit-identical.
#
# Gluon symbols are imported through the project ``_triton`` indirection so the
# underlying triton distribution can be swapped in one place; block-size counts
# and ``max_n_blocks`` are queried from ``RaggedTensorMetadata`` so the metadata
# shapes match ``make_ragged_tensor_metadata`` on HIP and non-HIP alike.
# ===========================================================================
from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon, redirect_triton_to_tokenspeed_triton

with redirect_triton_to_tokenspeed_triton():
    from triton_kernels.tensor import RaggedTensorMetadata
    from triton_kernels.topk import topk_forward

__all__ = [
    "FUSED_ROUTE_MAX_M",
    "SMALLM_MAX_M",
    "GLUON_ROUTE_DTYPES",
    "GLUON_ROUTE_MAX_E",
    "gluon_fused_route",
    "gluon_route_supported",
]

# Number of block-size rows in RaggedTensorMetadata for the active platform
# ([16,32,64,128,256] -> 5 on HIP, [16,32,64,128] -> 4 otherwise). Derived
# from the library so the metadata shapes match make_ragged_tensor_metadata
# exactly on every target.
NB = len(RaggedTensorMetadata.block_sizes())

# Token-count bound for the small-M fused route. 16 == the smallest
# RaggedTensorMetadata block size, so for M <= 16 every expert's token count is
# ``col_sum <= M <= 16``, i.e. exactly one block, and the single-block schedule
# collapse is exact. The caller dispatches only ``M <= SMALLM_MAX_M`` to the
# Gluon kernel (decode, where it wins ~6x on routing) and keeps the generic
# ``triton_kernels_routing`` pipeline for larger M.
SMALLM_MAX_M = 16
# Backwards-compatible alias for the small-M bound.
FUSED_ROUTE_MAX_M = SMALLM_MAX_M

# Configs the Gluon routing path supports; everything else falls back to the
# generic triton_kernels_routing pipeline.
GLUON_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
GLUON_ROUTE_MAX_E = 1024  # next_pow2(E) bins / EP-wide tiles stay bounded

# torch gate dtype -> gluon element type (for the in-kernel softmax cast that
# reproduces topk_forward's ``softmax(...).to(x_dtype)`` rounding exactly).
_GL_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
}


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _fused_topk(
    Logits,  # [M, E]   X_DTYPE   (raw routing logits)
    stride_lm,  # logits row stride
    g,  # [GP]      int32    flat gate index (token-major, == arange)
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
    the smaller expert id, descending value order) and the softmax gate over
    the selected logits -- reproducing the OAI ``_topk_forward`` semantics
    without a separate launch or a ``y_vals``/``y_indx`` global round-trip.
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
    big = gl.full([MP, EP], E, gl.int32, layout=LT)
    tcol = gl.expand_dims(gl.arange(0, TKP, layout=gl.SliceLayout(0, LT)), 0)  # [1,TKP]
    val_t = gl.full([MP, TKP], -1e30, gl.float32, layout=LT)  # finite -inf-ish
    idx_t = gl.zeros([MP, TKP], gl.int32, layout=LT)
    for _r in gl.static_range(TOPK):
        vmax = gl.max(cur, axis=1, keep_dims=True)  # [MP,1]
        ismax = (cur == vmax) & (col < E)
        amax = gl.min(gl.where(ismax, col, big), axis=1, keep_dims=True)  # [MP,1]
        sel = tcol == _r  # [1,TKP]
        val_t = gl.where(sel, vmax, val_t)  # write column _r
        idx_t = gl.where(sel, amax, idx_t)
        cur = gl.where(col == amax, NEG, cur)  # drop chosen expert

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


# ===========================================================================
# Small-M (M <= 16): single-workgroup, stable-order, single-block collapse.
# ===========================================================================
@gluon.jit
def _fused_route_smallM(
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
    NB_C: gl.constexpr,  # 5 (HIP)
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

    # ---- fused top-k: compute (expert id, softmax gate) per gate in-kernel,
    # replacing the separate topk_forward launch + y_vals/y_indx round-trip.
    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    tok = (g // TOPK).to(gl.int32)
    slot = (g % TOPK).to(gl.int32)
    idx, vals = _fused_topk(
        Logits,
        stride_lm,
        g,
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

    # ---- histogram (col_sum / slice_sizes) --------------------------------
    e = gl.arange(0, EP, layout=LE)
    emask = e < E
    hist = gl.histogram(idx, EP, mask=gmask, layout=LE)  # invalid excluded
    gl.store(SliceSizes + e, hist, mask=emask)

    # ---- slice_offs = [0] + cumsum(slice_sizes) ---------------------------
    # Two-store trick: exclusive scan -> indices 0..E-1; inclusive scan shifted
    # by +1 -> indices 1..E (writes the trailing total at index E).
    incl = gl.associative_scan(hist, 0, _add)  # inclusive cumsum [EP]
    col_offs = incl - hist  # exclusive cumsum [EP]
    gl.store(SliceOffs + e, col_offs, mask=emask)  # idx 0..E-1
    gl.store(SliceOffs + e + 1, incl, mask=emask)  # idx 1..E

    # ---- block_offs_data / block_schedule_data (single-block collapse) -----
    # M <= 16 == smallest block size, so col_sum <= M and every nonzero expert
    # is exactly ONE block at every block size -- all NB rows are identical and
    # the packed block value is just the expert id.
    n_blk = (hist > 0).to(gl.int32)  # 1 per nonzero expert
    blk_incl = gl.associative_scan(n_blk, 0, _add)
    blk_excl = blk_incl - n_blk  # position of expert's block
    n_total = gl.sum(n_blk, 0)  # #nonzero experts (scalar)
    jb = gl.arange(0, MAXBLKP, layout=LB)
    jbmask = jb < MAXBLK
    for k in gl.static_range(NB_C):
        # block_offs row k (same two-store trick)
        gl.store(BlockOffs + k * bo_stride + e, blk_excl, mask=emask)
        gl.store(BlockOffs + k * bo_stride + e + 1, blk_incl, mask=emask)
        # Fill -1 ONLY in the unused tail (jb >= n_total). This is DISJOINT
        # from the scatter positions below (0..n_total-1), so the two stores
        # never alias and the compiler is free to reorder them safely. (A full
        # 0..MAXBLK-1 fill would overlap the scatter and get reordered after
        # it, clobbering scattered ids -> wrong rows.)
        gl.store(
            BlockSched + k * bs_stride + jb,
            gl.full([MAXBLKP], -1, gl.int32, layout=LB),
            mask=jbmask & (jb >= n_total),
        )
        # scatter packed block ids: blk==0 => packed value == expert id e
        gl.store(
            BlockSched + k * bs_stride + blk_excl,
            e,
            mask=(hist > 0) & emask,
        )

    # ---- stable sort by expert (rank via register-only 2D all-pairs) ------
    # rank[g] = #{j : idx[j]==idx[g], j<g}. idx now lives in registers (no
    # global Idx after the fuse), so compute it as a [GP, GP] comparison tile
    # reduced over j -- cheap since GP <= 64 in the small-M tier.
    idx_row = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(1, LT)), 1)
    idx_col = gl.expand_dims(gl.convert_layout(idx, gl.SliceLayout(0, LT)), 0)
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row)).to(gl.int32)  # [GP,GP]
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)  # [GP]

    # ---- destination position = col_offs[expert] + rank -------------------
    base = gl.gather(col_offs, idx, axis=0)  # [GP]
    pos = base + rank

    # ---- scatter the three index tensors ----------------------------------
    gl.store(GatherIndx + pos, tok, mask=gmask)
    gl.store(ScatterIndx + pos, g.to(gl.int32), mask=gmask)
    gl.store(GateScal + pos, vals, mask=gmask)


# ===========================================================================
# Host wrappers
# ===========================================================================
def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def _route_smallM(logits, topk, dtype):
    """M <= 16: 1-kernel stable-order fused route (top-k fused in-kernel)."""
    M, E = logits.shape
    G = M * topk
    device = logits.device
    logits = logits.contiguous()

    slice_sizes = torch.empty(E, dtype=torch.int32, device=device)
    slice_offs = torch.empty(E + 1, dtype=torch.int32, device=device)
    block_offs_data = torch.empty(NB, E + 1, dtype=torch.int32, device=device)
    # max_n_blocks(E, G) == G in the small-M regime (G <= E so every gate is
    # its own block row); query the library to stay exact on any platform.
    maxblk = RaggedTensorMetadata.max_n_blocks(E, G)
    block_schedule_data = torch.empty(NB, maxblk, dtype=torch.int32, device=device)
    gather_indx = torch.empty(G, dtype=torch.int32, device=device)
    scatter_indx = torch.empty(G, dtype=torch.int32, device=device)
    gate_scal = torch.empty(G, dtype=dtype, device=device)

    # M<=2 is the launch-bound decode hot path: a single warp removes the
    # cross-warp s_barrier stalls. Larger small-M has enough work (O(G^2) rank
    # tile + top-k) to benefit from 4 warps.
    nw = 1 if M <= 2 else 4

    _fused_route_smallM[(1,)](
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
        MP=_next_pow2(M),
        GP=_next_pow2(G),
        EP=_next_pow2(E),
        TKP=_next_pow2(topk),
        MAXBLK=maxblk,
        MAXBLKP=_next_pow2(maxblk),
        NB_C=NB,
        X_DTYPE=_GL_DTYPE[logits.dtype],
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
    a 2D float ``logits`` tensor, a supported gate ``dtype``, a sane ``topk``
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
    Valid for ``M <= SMALLM_MAX_M`` (the single-block-collapse regime); callers
    gate on that bound and fall back to the generic pipeline for larger ``M``.
    """
    if dtype is None:
        dtype = logits.dtype
    return _route_smallM(logits, topk, dtype)
