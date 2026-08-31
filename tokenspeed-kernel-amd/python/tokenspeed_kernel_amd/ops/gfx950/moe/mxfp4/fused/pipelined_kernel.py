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

"""The pipelined ragged MoE GEMM kernel: per-W-layout tile runners, the
tile compute body, and the kernel entry point."""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (
    _group_m_swizzle,
    _load_layout,
    _store_layout,
    _swiglu_reduce,
    _xcd_chiplet_swizzle,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.medium_decode import (
    _medium_decode_body,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.pipelined_program import (
    AsyncCopyDescriptor,
    MoEConfig,
    MoEPipelinedProgram,
    MoESliceMNProgram,
    MoESliceNProgram,
    WPreshuffledLdsDescriptor,
    WVgprDescriptor,
    _make_moe_x_desc,
    _make_nonpreshuffled_w_full_desc,
    _make_nonpreshuffled_w_half_descs,
    _make_preshuffled_w_full_offsets,
    _make_preshuffled_w_slice_offsets,
    _make_preshuffled_w_x_desc,
    _make_slice_mn_x_descs,
    _make_swizzled_scale_direct_desc,
    _preshuffled_w_copy_layout,
    _preshuffled_w_read_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
    _mxfp4_store_cdna4_scale,
)


@gluon.jit
def _run_moe_tile_w_via_vgpr(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    M_X,
    N,
    K,
    off_m,
    m_limit,
    rows_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    N_LIMIT: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    gl.static_assert(
        cfg.W_PRESHUFFLED,
        "W_VIA_VGPR consumes the preshuffled Gluon-dot W layout.",
    )
    gl.static_assert(
        BLOCK_K_W == 128 and NUM_WARPS == 4 and not USE_SLICE_MN,
        "W_VIA_VGPR layout bases assume BLOCK_K_W=128, NUM_WARPS=4, "
        "and USE_SLICE_MN=False. Re-derive bases for other shapes.",
    )

    x_desc = _make_preshuffled_w_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        M_X,
        k_limit_x,
        BLOCK_K_X,
        HAS_GATHER,
    )

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        gl.static_assert(
            SUB_BN == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
            "USE_SLICE_N + W_VIA_VGPR requires SUB_BN=BLOCK_K_W=128 "
            "and NUM_WARPS=4; the half-tile LOAD_W_LAYOUT bases assume "
            "this shape (re-derive otherwise).",
        )
        LOAD_W_HALF_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            SUB_BN // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, False
        )
        (
            offsets_h,
            base_off_top,
            base_off_bot,
            w_k_stride,
            bottom_valid,
        ) = _make_preshuffled_w_slice_offsets(
            w_base_offset,
            pid_n,
            N,
            LOAD_W_HALF_COPY_LAYOUT,
            N_LIMIT,
            SUB_BN,
            BLOCK_K_W,
        )
        w_desc_top = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_k_stride,
            offsets_h + base_off_top,
            pred=gl.to_tensor(True),
            LOAD_BN=SUB_BN,
        )
        w_desc_bot = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_k_stride,
            offsets_h + base_off_bot,
            pred=bottom_valid,
            LOAD_BN=SUB_BN,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)
    else:
        gl.static_assert(
            BLOCK_N == 128,
            "W_VIA_VGPR full-tile layout bases assume BLOCK_N=128. "
            "Re-derive bases for other shapes.",
        )
        BLOCK_N_LAYOUT: gl.constexpr = BLOCK_N
        LOAD_W_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            BLOCK_N_LAYOUT // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, False
        )
        offsets_b_vgpr, base_off_b_vgpr = _make_preshuffled_w_full_offsets(
            w_base_offset,
            pid_n,
            LOAD_W_COPY_LAYOUT,
            BLOCK_N_LAYOUT,
            BLOCK_N,
            BLOCK_K_W,
        )
        w_desc = WVgprDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            gl.to_tensor(N),  # K-iter advance step: idx * BK_W * N
            offsets_b_vgpr + base_off_b_vgpr,
            pred=gl.to_tensor(True),  # full-tile path: always in-bounds
            LOAD_BN=BLOCK_N_LAYOUT,
        )
        pgm = MoEPipelinedProgram.initialize(
            cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
        )
        return pgm.run(K, USE_WARP_PIPELINE)


@gluon.jit
def _run_moe_tile_preshuffled_lds_w(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    M_X,
    N,
    K,
    off_m,
    m_limit,
    rows_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    N_LIMIT: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    gl.static_assert(
        BLOCK_K_W == 128
        and (BLOCK_N == 128 or USE_SLICE_N)
        and NUM_WARPS == 4
        and not USE_SLICE_MN,
        "preshuffled W layout bases assume BLOCK_K_W=128, "
        "BLOCK_N=128 (or USE_SLICE_N=True for half-tile path), "
        "NUM_WARPS=4, and USE_SLICE_MN=False. Re-derive bases for "
        "other shapes.",
    )

    x_desc = _make_preshuffled_w_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        M_X,
        k_limit_x,
        BLOCK_K_X,
        HAS_GATHER,
    )

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        gl.static_assert(
            SUB_BN == 128 and BLOCK_K_W == 128 and NUM_WARPS == 4,
            "USE_SLICE_N + preshuffled W requires SUB_BN=BLOCK_K_W=128 "
            "and NUM_WARPS=4; the half-tile LOAD_W_LAYOUT bases assume "
            "this shape (re-derive otherwise).",
        )
        LOAD_W_HALF_LAYOUT: gl.constexpr = _preshuffled_w_read_layout(
            SUB_BN // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS
        )
        LOAD_W_HALF_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
            SUB_BN // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, True
        )
        (
            offsets_h,
            base_off_top,
            base_off_bot,
            w_k_stride,
            bottom_valid,
        ) = _make_preshuffled_w_slice_offsets(
            w_base_offset,
            pid_n,
            N,
            LOAD_W_HALF_COPY_LAYOUT,
            N_LIMIT,
            SUB_BN,
            BLOCK_K_W,
        )
        w_desc_top = WPreshuffledLdsDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_ptr.dtype.element_ty,
            w_k_stride,
            offsets_h + base_off_top,
            pred=gl.to_tensor(True),
            load_layout=LOAD_W_HALF_LAYOUT,
            LOAD_BN=SUB_BN,
            cache_modifier=W_CACHE_MODIFIER,
        )
        w_desc_bot = WPreshuffledLdsDescriptor(
            cfg,
            BLOCK_K_W,
            w_ptr,
            w_ptr.dtype.element_ty,
            w_k_stride,
            offsets_h + base_off_bot,
            pred=bottom_valid,
            load_layout=LOAD_W_HALF_LAYOUT,
            LOAD_BN=SUB_BN,
            cache_modifier=W_CACHE_MODIFIER,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)

    # Gluon still type-checks the code below when USE_SLICE_N returns above.
    # Keep the original half-tile layout in that specialization so the
    # preshuffled copy/read layouts remain valid during compilation.
    BLOCK_N_LAYOUT: gl.constexpr = (BLOCK_N // 2) if USE_SLICE_N else BLOCK_N
    LOAD_W_LAYOUT: gl.constexpr = _preshuffled_w_read_layout(
        BLOCK_N_LAYOUT // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS
    )
    LOAD_W_COPY_LAYOUT: gl.constexpr = _preshuffled_w_copy_layout(
        BLOCK_N_LAYOUT // 16, BLOCK_K_W, cfg.W_SCALE_VIA_LDS, True
    )
    offsets_b_vgpr, base_off_b_vgpr = _make_preshuffled_w_full_offsets(
        w_base_offset,
        pid_n,
        LOAD_W_COPY_LAYOUT,
        BLOCK_N_LAYOUT,
        BLOCK_N,
        BLOCK_K_W,
    )
    w_desc = WPreshuffledLdsDescriptor(
        cfg,
        BLOCK_K_W,
        w_ptr,
        w_ptr.dtype.element_ty,
        gl.to_tensor(N),
        offsets_b_vgpr + base_off_b_vgpr,
        pred=gl.to_tensor(True),
        load_layout=LOAD_W_LAYOUT,
        cache_modifier=W_CACHE_MODIFIER,
    )
    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )
    return pgm.run(K, USE_WARP_PIPELINE)


@gluon.jit
def _run_moe_tile_transposed_w(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    M_X,
    N,
    K,
    off_m,
    off_n,
    m_limit,
    rows_m_x,
    mask_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    X_ELEM_BITS: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    x_desc = _make_moe_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m_x[:, None],
        k_limit_x,
        BLOCK_K_X,
    )

    if USE_SLICE_MN:
        SUB_BN_MN: gl.constexpr = BLOCK_N // 2
        x_desc_top_mn, x_desc_bot_mn = _make_slice_mn_x_descs(
            cfg,
            x_ptr,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            off_m,
            m_limit,
            k_limit_x,
            BLOCK_M,
            BLOCK_K_X,
            NUM_WARPS,
            HAS_GATHER,
            X_ELEM_BITS,
        )

        w_desc_left_mn, w_desc_right_mn = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN_MN,
            BLOCK_K_W,
            NUM_WARPS,
            True,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        slice_mn_pgm = MoESliceMNProgram.initialize(
            cfg,
            x_desc_top_mn,
            x_desc_bot_mn,
            w_desc_left_mn,
            w_desc_right_mn,
            x_scale_desc,
            w_scale_desc,
        )
        return slice_mn_pgm.pipeline(K)

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        bottom_valid = gl.to_tensor(True)
        w_desc_top, w_desc_bot = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN,
            BLOCK_K_W,
            NUM_WARPS,
            True,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)

    w_desc = _make_nonpreshuffled_w_full_desc(
        cfg,
        w_ptr,
        stride_wn,
        stride_wk,
        N,
        off_n,
        k_limit_w,
        w_base_offset,
        BLOCK_N,
        BLOCK_K_W,
        NUM_WARPS,
        True,
        W_ELEM_BITS,
        W_CACHE_MODIFIER,
    )
    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )
    return pgm.run(K, USE_WARP_PIPELINE)


@gluon.jit
def _run_moe_tile_ncontig_w(
    cfg,
    x_ptr,
    w_ptr,
    x_scale_desc,
    w_scale_desc,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    M_X,
    N,
    K,
    off_m,
    off_n,
    m_limit,
    rows_m_x,
    mask_m_x,
    offs_xk,
    k_limit_x,
    k_limit_w,
    w_base_offset,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    USE_SLICE_MN: gl.constexpr,
    USE_SLICE_N: gl.constexpr,
    USE_WARP_PIPELINE: gl.constexpr,
    X_ELEM_BITS: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    x_desc = _make_moe_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        mask_m_x[:, None],
        k_limit_x,
        BLOCK_K_X,
    )

    if USE_SLICE_MN:
        SUB_BN_MN: gl.constexpr = BLOCK_N // 2
        x_desc_top_mn, x_desc_bot_mn = _make_slice_mn_x_descs(
            cfg,
            x_ptr,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            off_m,
            m_limit,
            k_limit_x,
            BLOCK_M,
            BLOCK_K_X,
            NUM_WARPS,
            HAS_GATHER,
            X_ELEM_BITS,
        )

        w_desc_left_mn, w_desc_right_mn = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN_MN,
            BLOCK_K_W,
            NUM_WARPS,
            False,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        slice_mn_pgm = MoESliceMNProgram.initialize(
            cfg,
            x_desc_top_mn,
            x_desc_bot_mn,
            w_desc_left_mn,
            w_desc_right_mn,
            x_scale_desc,
            w_scale_desc,
        )
        return slice_mn_pgm.pipeline(K)

    if USE_SLICE_N:
        SUB_BN: gl.constexpr = BLOCK_N // 2
        bottom_valid = gl.to_tensor(True)
        w_desc_top, w_desc_bot = _make_nonpreshuffled_w_half_descs(
            cfg,
            w_ptr,
            stride_wn,
            stride_wk,
            N,
            off_n,
            k_limit_w,
            w_base_offset,
            SUB_BN,
            BLOCK_K_W,
            NUM_WARPS,
            False,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
        pgm = MoESliceNProgram.initialize(
            cfg,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )
        return pgm.pipeline(K)

    w_desc = _make_nonpreshuffled_w_full_desc(
        cfg,
        w_ptr,
        stride_wn,
        stride_wk,
        N,
        off_n,
        k_limit_w,
        w_base_offset,
        BLOCK_N,
        BLOCK_K_W,
        NUM_WARPS,
        False,
        W_ELEM_BITS,
        W_CACHE_MODIFIER,
    )
    pgm = MoEPipelinedProgram.initialize(
        cfg, x_desc, w_desc, x_scale_desc, w_scale_desc
    )
    return pgm.run(K, USE_WARP_PIPELINE)


@gluon.jit
def _pipelined_moe_tile_compute(
    # Tensors --------------------------------------------------------
    x_ptr,
    w_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    y_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    gate_scal_ptr,
    slice_offs_ptr,
    slice_sizes_ptr,
    x_scale_block_offs_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_xsm,
    stride_xsk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_yn,
    stride_ym,
    stride_be,
    stride_bn,
    M,
    M_X,
    N,
    K,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    out_mx_scale_ptr,
    stride_out_mxs_kswizzled,
    stride_out_mxs_mblock,
    compact_idx,
    block_in_expert,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKS_PER_EXPERT: gl.constexpr,
    X_FORMAT: gl.constexpr,
    W_FORMAT: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    HAS_X_BLOCK_SCALE: gl.constexpr,
    HAS_W_BLOCK_SCALE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    HAS_SCATTER: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    APPLY_GATE_SCAL: gl.constexpr,
    HAS_RAGGED_OFFS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_LOAD_MODE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
    K_ITERS: gl.constexpr = 0,
    N_CONST: gl.constexpr = 0,
    Y_N_CONST: gl.constexpr = 0,
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    HAS_FP8_QUANT_OUT: gl.constexpr = False,
    HAS_MXFP4_QUANT_OUT: gl.constexpr = False,
    W_PRESHUFFLED: gl.constexpr = False,
    W_VIA_VGPR: gl.constexpr = False,
    W_PREFETCH: gl.constexpr = True,
    W_CACHE_CG: gl.constexpr = False,
    X_SCALE_VIA_LDS: gl.constexpr = False,
    W_SCALE_VIA_LDS: gl.constexpr = False,
    USE_NARROW_N_STORE_LAYOUT: gl.constexpr = False,
    X_SCALE_RAGGED_PADDED: gl.constexpr = False,
):
    expert_id = compact_idx

    USE_GATHER: gl.constexpr = HAS_GATHER

    BLOCK_SCALE_FACTOR: gl.constexpr = 32
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // BLOCK_SCALE_FACTOR

    if HAS_RAGGED_OFFS:
        # X experts are packed back-to-back at slice_offs[expert_id];
        # boundary is slice_sizes[expert_id] (NOT padded to BLOCK_M).
        m_base = gl.load(slice_offs_ptr + expert_id).to(gl.int32)
        m_size = gl.load(slice_sizes_ptr + expert_id).to(gl.int32)
        off_m = m_base + block_in_expert * BLOCK_M
        m_limit = m_base + m_size
    else:
        off_m = compact_idx * BLOCKS_PER_EXPERT * BLOCK_M + block_in_expert * BLOCK_M
        m_limit = M
    off_n = pid_n * BLOCK_N
    if W_PRESHUFFLED:
        w_base_offset = expert_id * stride_we
        ws_base_offset = expert_id * stride_wse
    else:
        w_base_offset = expert_id.to(gl.int64) * stride_we
        ws_base_offset = expert_id.to(gl.int64) * stride_wse
    N_LIMIT: gl.constexpr = N_CONST if N_CONST else 0

    STORE: gl.constexpr = _store_layout(
        NUM_WARPS,
        block_m=BLOCK_M,
        w_via_vgpr=W_VIA_VGPR or W_PRESHUFFLED,
        use_narrow_n_layout=USE_NARROW_N_STORE_LAYOUT,
    )

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        X_FORMAT,
        W_FORMAT,
        BLOCK_SCALE_FACTOR,
        NUM_BUFFERS,
        W_TRANSPOSE,
        HAS_X_BLOCK_SCALE,
        HAS_W_BLOCK_SCALE,
        SCALE_LOAD_MODE,
        index_type,
        NUM_SUBTILES,
        EVEN_K,
        K_ITERS,
        USE_GATHER,
        NUM_WARPS,
        W_PRESHUFFLED=W_PRESHUFFLED,
        W_VIA_VGPR=W_VIA_VGPR,
        W_PREFETCH=W_PREFETCH,
        X_SCALE_VIA_LDS=X_SCALE_VIA_LDS,
        W_SCALE_VIA_LDS=W_SCALE_VIA_LDS,
    )
    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

    W_CACHE_MODIFIER: gl.constexpr = ".cg" if W_CACHE_CG else ""

    X_ELEM_BITS: gl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = w_ptr.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_X, BLOCK_M, NUM_WARPS, [1, 0], X_ELEM_BITS
    )

    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))

    rows_m = off_m + offs_xm
    if not cfg.W_VIA_VGPR and USE_SLICE_N and BLOCK_M == 128:
        src_offs_xm = (offs_xm % 4) * 32 + (offs_xm // 4)
    elif not cfg.W_VIA_VGPR and USE_SLICE_N and BLOCK_M == 64:
        src_offs_xm = (offs_xm % 4) * 16 + (offs_xm // 4)
    elif not cfg.W_VIA_VGPR and USE_SLICE_N and BLOCK_M == 32:
        src_offs_xm = (offs_xm % 2) * 16 + (offs_xm // 2)
    else:
        src_offs_xm = offs_xm
    rows_m_x = off_m + src_offs_xm
    # m_limit = per-expert tail (HAS_RAGGED_OFFS) or global M.
    pre_gather_mask = rows_m < m_limit
    pre_gather_mask_x = rows_m_x < m_limit
    if HAS_GATHER:
        rows_m_safe = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        rows_m = gl.load(
            gather_idx_ptr + rows_m_safe, mask=pre_gather_mask, other=0
        ).to(gl.int32)
        rows_m_x_safe = gl.where(pre_gather_mask_x, rows_m_x, gl.zeros_like(rows_m_x))
        rows_m_x = gl.load(
            gather_idx_ptr + rows_m_x_safe, mask=pre_gather_mask_x, other=0
        ).to(gl.int32)
        # Post-gather rows_m is in global token-id space (size M_X);
        # mask out junk gather_idx values too. Don't conflate M_X with
        # ``M`` (= dispatched tile count, can exceed M_X for top-k>1).
        mask_m_x = pre_gather_mask_x & (rows_m_x < M_X)
    else:
        # Clamp OOB lanes to 0 so the buffer_load address stays in
        # bounds during HIP graph warm-up; mask still filters.
        rows_m = gl.where(pre_gather_mask, rows_m, gl.zeros_like(rows_m))
        rows_m_x = gl.where(pre_gather_mask_x, rows_m_x, gl.zeros_like(rows_m_x))
        mask_m_x = pre_gather_mask_x

    k_limit_x = gl.multiple_of(K // cfg.DIV_FACTOR_X, 16)
    k_limit_w = gl.multiple_of(K // cfg.DIV_FACTOR_W, 16)

    # Swizzled scale loads use post-swizzle HBM shape via buffer_load_to_shared;
    # direct scale loads use G->VGPR gl.load and can follow gathered X rows.
    if HAS_X_BLOCK_SCALE:
        if cfg.X_SCALE_VIA_LDS:
            BLOCK_M_PS: gl.constexpr = cfg.BLOCK_M_PRESHUFFLED
            BLOCK_K_S_PS: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
            LX_S: gl.constexpr = cfg.load_layout_x_scale
            offs_xs_m = gl.arange(0, BLOCK_M_PS, layout=gl.SliceLayout(1, LX_S))
            offs_xs_k = gl.arange(0, BLOCK_K_S_PS, layout=gl.SliceLayout(0, LX_S))
            row_base_x_s = off_m // cfg.PRESHUFFLE_FACTOR
            rows_m_scale = row_base_x_s + offs_xs_m
            row_limit_x_s = (M_X + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
            # Suppress the K-mask: the swizzle packs K with N so a
            # K-mask on the packed column scrambles both. The host
            # pads with e8m0=0 and the W K-mask zeros the OOB product
            # regardless of scale value.
            k_limit_xs_load = (
                (K // cfg.SCALE_BLOCK + 7) // 8 * 8
            ) * cfg.PRESHUFFLE_FACTOR
            x_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_S_PS,
                x_scale_ptr,
                rows_m_scale,
                offs_xs_k,
                stride_xsm,
                stride_xsk,
                rows_m_scale[:, None] < row_limit_x_s,
                k_limit_xs_load,
            )
        else:
            offs_xs_m = gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, cfg.layout_x_scale)
            )
            offs_xs_k = gl.arange(
                0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, cfg.layout_x_scale)
            )
            rows_m_scale = off_m + offs_xs_m
            if X_SCALE_RAGGED_PADDED:
                local_rows_m_scale = block_in_expert * BLOCK_M + offs_xs_m
                scale_base = (
                    gl.load(x_scale_block_offs_ptr + expert_id).to(gl.int32)
                    * cfg.PRESHUFFLE_FACTOR
                )
                rows_m_scale = scale_base + local_rows_m_scale
                mask_m_scale = local_rows_m_scale < m_size
            elif HAS_GATHER:
                pre_gather_mask_scale = rows_m_scale < m_limit
                rows_m_scale_safe = gl.where(
                    pre_gather_mask_scale,
                    rows_m_scale,
                    gl.zeros_like(rows_m_scale),
                )
                rows_m_scale = gl.load(
                    gather_idx_ptr + rows_m_scale_safe,
                    mask=pre_gather_mask_scale,
                    other=0,
                ).to(gl.int32)
                mask_m_scale = pre_gather_mask_scale & (rows_m_scale < M_X)
            else:
                mask_m_scale = rows_m_scale < m_limit
                rows_m_scale = gl.where(
                    mask_m_scale,
                    rows_m_scale,
                    gl.zeros_like(rows_m_scale),
                )
            if SCALE_LOAD_MODE == "swizzle":
                x_scale_desc = _make_swizzled_scale_direct_desc(
                    cfg,
                    x_scale_ptr,
                    rows_m_scale,
                    offs_xs_k,
                    stride_xsm,
                    stride_xsk,
                    mask_m_scale,
                    K // cfg.SCALE_BLOCK,
                    BLOCK_K_SCALE,
                )
            else:
                x_scale_desc = AsyncCopyDescriptor.initialize(
                    cfg,
                    0,
                    BLOCK_K_SCALE,
                    x_scale_ptr,
                    rows_m_scale,
                    offs_xs_k,
                    stride_xsm,
                    stride_xsk,
                    mask_m_scale[:, None],
                    K // cfg.SCALE_BLOCK,
                )
    else:
        x_scale_desc: gl.constexpr = 0

    if HAS_W_BLOCK_SCALE:
        if cfg.W_SCALE_VIA_LDS:
            BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
            BLOCK_K_S_PS_W: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
            LW_S: gl.constexpr = cfg.load_layout_w_scale
            offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))
            offs_ws_k = gl.arange(0, BLOCK_K_S_PS_W, layout=gl.SliceLayout(0, LW_S))
            row_base_w_s = off_n // cfg.PRESHUFFLE_FACTOR
            rows_n_scale = row_base_w_s + offs_ws_n
            if N_LIMIT:
                row_limit_w_s: gl.constexpr = (
                    N_LIMIT + cfg.PRESHUFFLE_FACTOR - 1
                ) // cfg.PRESHUFFLE_FACTOR
            else:
                row_limit_w_s = (N + cfg.PRESHUFFLE_FACTOR - 1) // cfg.PRESHUFFLE_FACTOR
            # See x_scale: suppress K-mask, OOB product is zero.
            k_limit_ws_load = (
                (K // cfg.SCALE_BLOCK + 7) // 8 * 8
            ) * cfg.PRESHUFFLE_FACTOR
            w_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_S_PS_W,
                w_scale_ptr,
                rows_n_scale,
                offs_ws_k,
                stride_wsn,
                stride_wsk,
                rows_n_scale[:, None] < row_limit_w_s,
                k_limit_ws_load,
                base_offset=ws_base_offset,
            )
        else:
            offs_ws_n = gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(1, cfg.layout_w_scale)
            )
            offs_ws_k = gl.arange(
                0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, cfg.layout_w_scale)
            )
            w_scale_desc = AsyncCopyDescriptor.initialize(
                cfg,
                0,
                BLOCK_K_SCALE,
                w_scale_ptr,
                off_n + offs_ws_n,
                offs_ws_k,
                stride_wsn,
                stride_wsk,
                (off_n + offs_ws_n)[:, None] < N,
                K // cfg.SCALE_BLOCK,
                base_offset=ws_base_offset,
            )
    else:
        w_scale_desc: gl.constexpr = 0

    if cfg.W_VIA_VGPR:
        acc = _run_moe_tile_w_via_vgpr(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            N,
            K,
            off_m,
            m_limit,
            rows_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            N_LIMIT,
            W_CACHE_MODIFIER,
        )
    elif cfg.W_PRESHUFFLED:
        acc = _run_moe_tile_preshuffled_lds_w(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            M_X,
            N,
            K,
            off_m,
            m_limit,
            rows_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            N_LIMIT,
            W_CACHE_MODIFIER,
        )
    elif W_TRANSPOSE:
        acc = _run_moe_tile_transposed_w(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            stride_wn,
            stride_wk,
            M_X,
            N,
            K,
            off_m,
            off_n,
            m_limit,
            rows_m_x,
            mask_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            X_ELEM_BITS,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )
    else:
        acc = _run_moe_tile_ncontig_w(
            cfg,
            x_ptr,
            w_ptr,
            x_scale_desc,
            w_scale_desc,
            gather_idx_ptr,
            stride_xm,
            stride_xk,
            stride_wn,
            stride_wk,
            M_X,
            N,
            K,
            off_m,
            off_n,
            m_limit,
            rows_m_x,
            mask_m_x,
            offs_xk,
            k_limit_x,
            k_limit_w,
            w_base_offset,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K_X,
            BLOCK_K_W,
            NUM_WARPS,
            HAS_GATHER,
            USE_SLICE_MN,
            USE_SLICE_N,
            USE_WARP_PIPELINE,
            X_ELEM_BITS,
            W_ELEM_BITS,
            W_CACHE_MODIFIER,
        )

    if APPLY_X_GLOBAL_SCALE and not HAS_X_BLOCK_SCALE:
        x_global_scale = gl.load(x_global_scale_ptr)
        acc = acc * x_global_scale

    if HAS_BIAS:
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
        if Y_N_CONST and not DO_SWIGLU:
            BIAS_N: gl.constexpr = Y_N_CONST
            bias_mask = bias_offs < BIAS_N
        else:
            bias_mask = bias_offs < N
        # Masked lanes still need in-bounds addresses; W2 preshuffle can
        # tile over padded physical N while bias remains logical N.
        bias_offs_safe = gl.where(bias_mask, bias_offs, gl.zeros_like(bias_offs))
        bias = gl.load(
            bias_ptr + expert_id * stride_be + bias_offs_safe,
            mask=bias_mask,
            other=0.0,
        )
        acc = acc + bias[None, :].to(gl.float32)

    if DO_SWIGLU:
        out = _swiglu_reduce(
            acc,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            OUT_BLOCK_N,
            cfg.acc_layout,
        )
        if HAS_MXFP4_QUANT_OUT:
            packed, scale_byte = _mxfp4_quantize_tile(out)
            packed = packed.reshape((BLOCK_M, OUT_BLOCK_N // 2))
            PACK_LAYOUT: gl.constexpr = packed.type.layout
            offs_pack_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, PACK_LAYOUT))
            y_m_in_bounds = offs_pack_m < m_limit
            offs_pack_m_safe = gl.where(
                y_m_in_bounds, offs_pack_m, gl.zeros_like(offs_pack_m)
            )
            y_cols = pid_n * (OUT_BLOCK_N // 2) + gl.arange(
                0, OUT_BLOCK_N // 2, gl.SliceLayout(0, PACK_LAYOUT)
            )
            if Y_N_CONST:
                ACTUAL_PACKED_N: gl.constexpr = Y_N_CONST // 2
                n_in_bounds = y_cols < ACTUAL_PACKED_N
            elif N_LIMIT:
                ACTUAL_PACKED_N: gl.constexpr = N_LIMIT // 4
                n_in_bounds = y_cols < ACTUAL_PACKED_N
            else:
                n_in_bounds = y_cols < (N // 4)
            y_cols_safe = gl.where(n_in_bounds, y_cols, gl.zeros_like(y_cols))
            y_offs = (
                offs_pack_m_safe[:, None].to(gl.int64) * stride_ym
                + y_cols_safe[None, :].to(gl.int64) * stride_yn
            )
            mask_y = y_m_in_bounds[:, None] & n_in_bounds[None, :]
            gl.store(y_ptr + y_offs, packed, mask=mask_y)

            SCALE_LAYOUT: gl.constexpr = scale_byte.type.layout
            scale_offsets_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, SCALE_LAYOUT))
            if HAS_RAGGED_OFFS:
                local_scale_m = block_in_expert * BLOCK_M + scale_offsets_m
                scale_base = (
                    gl.load(x_scale_block_offs_ptr + expert_id).to(gl.int32) * 32
                )
                scale_m = scale_base + local_scale_m
                scale_m_in_bounds = local_scale_m < m_size
            else:
                scale_m = off_m + scale_offsets_m
                scale_m_in_bounds = scale_m < m_limit
            scale_k = pid_n * (OUT_BLOCK_N // 32) + gl.arange(
                0, OUT_BLOCK_N // 32, gl.SliceLayout(0, SCALE_LAYOUT)
            )
            if Y_N_CONST:
                scale_k_in_bounds = scale_k < (Y_N_CONST // 32)
            elif N_LIMIT:
                scale_k_in_bounds = scale_k < (N_LIMIT // 64)
            else:
                scale_k_in_bounds = scale_k < (N // 64)
            _mxfp4_store_cdna4_scale(
                out_mx_scale_ptr,
                scale_byte,
                scale_m[:, None],
                scale_k[None, :],
                stride_out_mxs_kswizzled,
                stride_out_mxs_mblock,
                scale_m_in_bounds[:, None] & scale_k_in_bounds[None, :],
                M_SWIZZLE=32,
                K_SWIZZLE=8,
            )
            return
        if HAS_FP8_QUANT_OUT:
            scale = gl.load(out_quant_scale_ptr).to(gl.float32)
            inv_scale = 1.0 / scale
            out = out * inv_scale
        out = out.to(y_ptr.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = out.type.layout
    else:
        out = acc.to(y_ptr.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = STORE
        out = gl.convert_layout(out, STORE_LAYOUT)

    offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT))

    # Clamp offs_y_m to m_limit before any pointer arithmetic; AMD/HIP
    # faults on the masked-off lanes if the address goes OOB even
    # under a predicated load.
    y_m_in_bounds = offs_y_m < m_limit
    offs_y_m_safe = gl.where(y_m_in_bounds, offs_y_m, gl.zeros_like(offs_y_m))

    if APPLY_GATE_SCAL:
        scal = gl.load(
            gate_scal_ptr + offs_y_m_safe,
            mask=y_m_in_bounds,
            other=1.0,
        )
        out = out * scal[:, None].to(out.dtype)

    if Y_N_CONST:
        ACTUAL_N: gl.constexpr = Y_N_CONST
    elif N_LIMIT:
        ACTUAL_N: gl.constexpr = (N_LIMIT // 2) if DO_SWIGLU else N_LIMIT
    else:
        actual_n = (N // 2) if DO_SWIGLU else N
    if Y_N_CONST or N_LIMIT:
        n_in_bounds = offs_y_n < ACTUAL_N
    else:
        n_in_bounds = offs_y_n < actual_n
    # Clamp masked-off N lanes before pointer arithmetic; masked GPU
    # memory ops may still fault on OOB addresses.
    offs_y_n_safe = gl.where(n_in_bounds, offs_y_n, gl.zeros_like(offs_y_n))
    if HAS_SCATTER:
        rows_y = gl.load(scatter_idx_ptr + offs_y_m_safe, mask=y_m_in_bounds, other=M)
        rows_y_in_bounds = y_m_in_bounds & (rows_y < M)
        mask_y = rows_y_in_bounds[:, None] & n_in_bounds[None, :]
        rows_y_safe = gl.where(rows_y_in_bounds, rows_y, gl.zeros_like(rows_y))
        y_offs = rows_y_safe[:, None] * stride_ym + offs_y_n_safe[None, :] * stride_yn
    else:
        mask_y = y_m_in_bounds[:, None] & n_in_bounds[None, :]
        offs_y_m_2d_safe = offs_y_m_safe[:, None]
        y_offs = offs_y_m_2d_safe * stride_ym + offs_y_n_safe[None, :] * stride_yn

    gl.store(y_ptr + y_offs, out, mask=mask_y)


def _pipelined_moe_kernel_repr(specialization) -> str:
    """Distinct rocprof names for schedule vs no-schedule specialization."""
    if bool(specialization.constants.get("IS_MEDIUM_DECODE", False)):
        if bool(specialization.constants.get("MEDIUM_COMBINE", False)):
            return "_moe_medium_decode_combine"
        return "_moe_medium_decode_dispatch"
    use_block_schedule = bool(specialization.constants.get("USE_BLOCK_SCHEDULE", False))
    if use_block_schedule:
        return "_pipelined_moe_kernel_scaled_block_schedule"
    return "_pipelined_moe_kernel_scaled"


@gluon.jit(repr=_pipelined_moe_kernel_repr)
def _pipelined_moe_kernel_scaled(
    x_ptr,
    w_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    y_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    gate_scal_ptr,
    slice_offs_ptr,
    slice_sizes_ptr,
    x_scale_block_offs_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_xsm,
    stride_xsk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_yn,
    stride_ym,
    stride_be,
    stride_bn,
    M,
    M_X,
    N,
    K,
    x_global_scale_ptr,
    out_quant_scale_ptr,
    out_mx_scale_ptr,
    stride_out_mxs_kswizzled,
    stride_out_mxs_mblock,
    NUM_TILES,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKS_PER_EXPERT: gl.constexpr,
    X_FORMAT: gl.constexpr,
    W_FORMAT: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    HAS_X_BLOCK_SCALE: gl.constexpr,
    HAS_W_BLOCK_SCALE: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    HAS_SCATTER: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    APPLY_GATE_SCAL: gl.constexpr,
    HAS_RAGGED_OFFS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_LOAD_MODE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
    K_ITERS: gl.constexpr = 0,
    N_CONST: gl.constexpr = 0,
    Y_N_CONST: gl.constexpr = 0,
    APPLY_X_GLOBAL_SCALE: gl.constexpr = True,
    USE_WARP_PIPELINE: gl.constexpr = False,
    USE_SLICE_MN: gl.constexpr = False,
    USE_SLICE_N: gl.constexpr = False,
    HAS_FP8_QUANT_OUT: gl.constexpr = False,
    HAS_MXFP4_QUANT_OUT: gl.constexpr = False,
    USE_BLOCK_SCHEDULE: gl.constexpr = False,
    N_EXPTS_TOT: gl.constexpr = 0,
    GRID_N: gl.constexpr = 0,
    GROUP_M: gl.constexpr = 1,
    XCD_SWIZZLE: gl.constexpr = 1,
    W_PRESHUFFLED: gl.constexpr = False,
    W_VIA_VGPR: gl.constexpr = False,
    W_PREFETCH: gl.constexpr = True,
    W_CACHE_CG: gl.constexpr = False,
    X_SCALE_VIA_LDS: gl.constexpr = False,
    W_SCALE_VIA_LDS: gl.constexpr = False,
    USE_NARROW_N_STORE_LAYOUT: gl.constexpr = False,
    IS_MEDIUM_DECODE: gl.constexpr = False,
    MEDIUM_COMBINE: gl.constexpr = False,
    X_SCALE_RAGGED_PADDED: gl.constexpr = False,
):
    if IS_MEDIUM_DECODE:
        # Medium-decode (M=8/16) reuses this kernel's signature but runs the
        # single-buffer direct-load body instead of the pipelined prefill loop.
        # The constexpr guard DCEs this branch for the existing/default path.
        _medium_decode_body(
            x_ptr,
            w_ptr,
            w_scale_ptr,
            gather_idx_ptr,
            scatter_idx_ptr,
            gate_scal_ptr,
            slice_sizes_ptr,
            slice_offs_ptr,
            block_offs_ptr,
            block_schedule_ptr,
            y_ptr,
            M,
            M_X,
            N,
            K,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            stride_ym,
            stride_yn,
            stride_be,
            stride_bn,
            x_global_scale_ptr,
            out_quant_scale_ptr,
            bias_ptr,
            N_EXPERTS=N_EXPTS_TOT,
            NUM_TILES=NUM_TILES,
            GRID_N=GRID_N,
            GROUP_M=GROUP_M,
            XCD_SWIZZLE=XCD_SWIZZLE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            HAS_BIAS=HAS_BIAS,
            SWIGLU_ALPHA=SWIGLU_ALPHA,
            SWIGLU_LIMIT=SWIGLU_LIMIT,
            SWIGLU_BETA=SWIGLU_BETA,
            Y_N_CONST=Y_N_CONST,
            MEDIUM_COMBINE=MEDIUM_COMBINE,
        )
        return

    if GRID_N > 0:
        grid_n: gl.constexpr = GRID_N
        tiles_per_expert: gl.constexpr = BLOCKS_PER_EXPERT * GRID_N
    else:
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        tiles_per_expert = BLOCKS_PER_EXPERT * grid_n

    if USE_BLOCK_SCHEDULE:
        unpadded_m = gl.load(block_offs_ptr + N_EXPTS_TOT).to(gl.int32)
        loop_tiles = unpadded_m * grid_n
    else:
        loop_tiles = NUM_TILES

    for tile_idx in range(gl.program_id(0), loop_tiles, gl.num_programs(0)):
        if USE_BLOCK_SCHEDULE:
            swizzled = _xcd_chiplet_swizzle(tile_idx, loop_tiles, XCD_SWIZZLE)
            pid_m, pid_n = _group_m_swizzle(swizzled, unpadded_m, grid_n, GROUP_M)
            schedule_raw = gl.load(block_schedule_ptr + pid_m).to(
                gl.uint32, bitcast=True
            )
            compact_idx = (schedule_raw & 0x0000FFFF).to(gl.int32)
            block_in_expert = (schedule_raw >> 16).to(gl.int32)
        else:
            # Dense path: tile_idx packs (compact_idx, intra-expert pid);
            # GROUP_M applies WITHIN one expert (W only reusable per expert).
            swizzled = _xcd_chiplet_swizzle(tile_idx, NUM_TILES, XCD_SWIZZLE)
            compact_idx = swizzled // tiles_per_expert
            local = swizzled % tiles_per_expert
            block_in_expert, pid_n = _group_m_swizzle(
                local, BLOCKS_PER_EXPERT, grid_n, GROUP_M
            )

        _pipelined_moe_tile_compute(
            x_ptr,
            w_ptr,
            x_scale_ptr,
            w_scale_ptr,
            bias_ptr,
            y_ptr,
            gather_idx_ptr,
            scatter_idx_ptr,
            gate_scal_ptr,
            slice_offs_ptr,
            slice_sizes_ptr,
            x_scale_block_offs_ptr,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_xsm,
            stride_xsk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            stride_yn,
            stride_ym,
            stride_be,
            stride_bn,
            M,
            M_X,
            N,
            K,
            x_global_scale_ptr,
            out_quant_scale_ptr,
            out_mx_scale_ptr,
            stride_out_mxs_kswizzled,
            stride_out_mxs_mblock,
            compact_idx,
            block_in_expert,
            pid_n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCKS_PER_EXPERT=BLOCKS_PER_EXPERT,
            X_FORMAT=X_FORMAT,
            W_FORMAT=W_FORMAT,
            UPCAST_INDICES=UPCAST_INDICES,
            HAS_X_BLOCK_SCALE=HAS_X_BLOCK_SCALE,
            HAS_W_BLOCK_SCALE=HAS_W_BLOCK_SCALE,
            HAS_BIAS=HAS_BIAS,
            HAS_GATHER=HAS_GATHER,
            HAS_SCATTER=HAS_SCATTER,
            DO_SWIGLU=DO_SWIGLU,
            SWIGLU_ALPHA=SWIGLU_ALPHA,
            SWIGLU_LIMIT=SWIGLU_LIMIT,
            SWIGLU_BETA=SWIGLU_BETA,
            OUT_BLOCK_N=OUT_BLOCK_N,
            APPLY_GATE_SCAL=APPLY_GATE_SCAL,
            HAS_RAGGED_OFFS=HAS_RAGGED_OFFS,
            NUM_WARPS=NUM_WARPS,
            NUM_BUFFERS=NUM_BUFFERS,
            SCALE_LOAD_MODE=SCALE_LOAD_MODE,
            W_TRANSPOSE=W_TRANSPOSE,
            NUM_SUBTILES=NUM_SUBTILES,
            EVEN_K=EVEN_K,
            K_ITERS=K_ITERS,
            N_CONST=N_CONST,
            Y_N_CONST=Y_N_CONST,
            APPLY_X_GLOBAL_SCALE=APPLY_X_GLOBAL_SCALE,
            USE_WARP_PIPELINE=USE_WARP_PIPELINE,
            USE_SLICE_MN=USE_SLICE_MN,
            USE_SLICE_N=USE_SLICE_N,
            HAS_FP8_QUANT_OUT=HAS_FP8_QUANT_OUT,
            HAS_MXFP4_QUANT_OUT=HAS_MXFP4_QUANT_OUT,
            W_PRESHUFFLED=W_PRESHUFFLED,
            W_VIA_VGPR=W_VIA_VGPR,
            W_PREFETCH=W_PREFETCH,
            W_CACHE_CG=W_CACHE_CG,
            X_SCALE_VIA_LDS=X_SCALE_VIA_LDS,
            W_SCALE_VIA_LDS=W_SCALE_VIA_LDS,
            USE_NARROW_N_STORE_LAYOUT=USE_NARROW_N_STORE_LAYOUT,
            X_SCALE_RAGGED_PADDED=X_SCALE_RAGGED_PADDED,
        )
