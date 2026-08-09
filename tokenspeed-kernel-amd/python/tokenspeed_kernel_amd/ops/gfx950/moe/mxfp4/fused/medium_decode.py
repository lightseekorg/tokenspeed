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

"""Medium-decode (M=8/16) single-buffer direct-load body. Reuses the
pipelined kernel's signature via the IS_MEDIUM_DECODE constexpr switch
in pipelined_kernel."""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (
    _group_m_swizzle,
    _load_layout,
    _load_w_scale_tile_direct_cdna4,
    _swiglu_reduce,
    _xcd_chiplet_swizzle,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.pipelined_program import (
    MoEConfig,
)


@gluon.jit
def _moe_masked_store(out, y_ptr, y_offs, mask, USE_BUFFER_STORE: gl.constexpr):
    """Shared masked store for the prefill and medium-decode epilogues.

    ``USE_BUFFER_STORE`` selects the AMD ``buffer_store`` intrinsic (the medium
    dispatch path's fast store) vs generic ``gl.store`` (prefill + medium
    combine). The caller owns all address/mask computation so each path keeps
    its own addressing; only the final emit is shared.
    """
    if USE_BUFFER_STORE:
        gl.amd.cdna4.buffer_store(out, y_ptr, y_offs.to(gl.int32), mask=mask)
    else:
        gl.store(y_ptr + y_offs, out, mask=mask)


@gluon.jit
def _decode_block_schedule(block_schedule_ptr, pid_m):
    """Unpack the packed ``block_in_expert << 16 | expert`` schedule word."""
    schedule_raw = gl.load(block_schedule_ptr + pid_m).to(gl.uint32, bitcast=True)
    expert = (schedule_raw & 0x0000FFFF).to(gl.int32)
    block_in_expert = (schedule_raw >> 16).to(gl.int32)
    return expert, block_in_expert


@gluon.constexpr_function
def _medium_decode_moe_config(BLOCK_M, BLOCK_N, BLOCK_K):
    """Frozen single-buffer direct-load profile shared by the medium-decode stages."""
    return MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        "e4m3",
        "e2m1",
        32,
        1,
        True,
        False,
        True,
        "swizzle",
        gl.int32,
        (1, 1, 1),
        False,
        True,
        4,
        W_VIA_VGPR=False,
        W_PREFETCH=False,
    )


@gluon.jit
def _medium_decode_mfma_tile(
    X,
    W,
    WScale,
    token_m_l_addr,
    expert,
    kt,
    off_n,
    lk,
    lwk,
    lwn,
    acc,
    a_scale,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    K,
    N,
    cfg: gl.constexpr,
    DIRECT_SHARED_X: gl.constexpr,
    DIRECT_SHARED_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    MASK_K: gl.constexpr,
    MASK_N: gl.constexpr,
):
    s = _load_w_scale_tile_direct_cdna4(
        WScale, expert, kt, off_n, stride_wse, stride_wsk, stride_wsn, cfg
    )
    k_elem_l = kt * BLOCK_K + lk
    k_pack_l = kt * BLOCK_K_PACKED + lwk
    n_cols_l = off_n + lwn
    x_off_l = (
        gl.expand_dims(token_m_l_addr, 1).to(gl.int64) * stride_xm
        + gl.expand_dims(k_elem_l, 0).to(gl.int64) * stride_xk
    )
    w_off_l = (
        expert.to(gl.int64) * stride_we
        + k_pack_l.to(gl.int64) * stride_wk
        + n_cols_l.to(gl.int64) * stride_wn
    )
    # MASK_K guards the ragged K tail; MASK_N guards N tiles that do not divide
    # BLOCK_N. The full-K stage1 tile (N a multiple of BLOCK_N) takes neither,
    # which keeps its loads bit-identical to the unmasked baseline.
    if MASK_K:
        a_l = gl.amd.cdna4.buffer_load(
            ptr=X,
            offsets=x_off_l.to(gl.int32),
            mask=gl.expand_dims(k_elem_l, 0) < K,
            other=0.0,
        )
    else:
        a_l = gl.amd.cdna4.buffer_load(ptr=X, offsets=x_off_l.to(gl.int32))
    if MASK_N and MASK_K:
        b_l = gl.amd.cdna4.buffer_load(
            ptr=W,
            offsets=w_off_l.to(gl.int32),
            mask=(n_cols_l < N) & (k_pack_l < (K // 2)),
            other=0,
        )
    elif MASK_N:
        b_l = gl.amd.cdna4.buffer_load(
            ptr=W,
            offsets=w_off_l.to(gl.int32),
            mask=n_cols_l < N,
            other=0,
        )
    else:
        b_l = gl.amd.cdna4.buffer_load(ptr=W, offsets=w_off_l.to(gl.int32))
    a_smem = gl.allocate_shared_memory(
        X.dtype.element_ty, [BLOCK_M, BLOCK_K], DIRECT_SHARED_X, a_l
    )
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem, cfg.dot_layout_x)
    b_smem = gl.allocate_shared_memory(
        W.dtype.element_ty, [BLOCK_K_PACKED, BLOCK_N], DIRECT_SHARED_W, b_l
    )
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem, cfg.dot_layout_w)
    return gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_scale,
        a_format="e4m3",
        b=b,
        b_scale=s,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _medium_decode_body(
    X,
    W,
    WScale,
    Gather,
    Scatter,
    Gate,
    SliceSizes,
    SliceOffs,
    BlockOffs,
    BlockSchedule,
    Y,
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
    N_EXPERTS: gl.constexpr,
    NUM_TILES: gl.constexpr,
    GRID_N: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    Y_N_CONST: gl.constexpr,
    MEDIUM_COMBINE: gl.constexpr,
):
    """Shared AITER-style direct-load grouped body for M=8/16 decode.

    ``MEDIUM_COMBINE=False`` is stage1 (gather X rows, SwiGLU + fp8-quant store);
    ``MEDIUM_COMBINE=True`` is stage2/combine (contiguous X rows, gate + scatter
    store). This is selected only for the additive medium-decode path; the
    existing prefill/default pipeline remains the fallback for all other shapes.
    """
    pid_raw = gl.program_id(axis=0)
    grid_n: gl.constexpr = GRID_N
    if XCD_SWIZZLE > 1:
        pid = _xcd_chiplet_swizzle(pid_raw, NUM_TILES, XCD_SWIZZLE)
    else:
        pid = pid_raw
    grid_m_padded: gl.constexpr = NUM_TILES // GRID_N
    pid_m, pid_n = _group_m_swizzle(pid, grid_m_padded, grid_n, GROUP_M)

    unpadded_m = gl.load(BlockOffs + N_EXPERTS).to(gl.int32)
    if pid_m >= unpadded_m:
        return

    expert, block_in_expert = _decode_block_schedule(BlockSchedule, pid_m)
    gl.assume(M >= 0)
    if not MEDIUM_COMBINE:
        gl.assume(M_X >= 0)
    gl.assume(N >= 0)
    gl.assume(K >= 0)
    gl.assume(expert >= 0)
    gl.assume(block_in_expert >= 0)
    gl.assume(stride_xm >= 0)
    gl.assume(stride_xk >= 0)
    gl.assume(stride_we >= 0)
    gl.assume(stride_wn >= 0)
    gl.assume(stride_wk >= 0)
    gl.assume(stride_wse >= 0)
    gl.assume(stride_wsn >= 0)
    gl.assume(stride_wsk >= 0)
    gl.assume(stride_ym >= 0)
    gl.assume(stride_yn >= 0)
    gl.assume(stride_be >= 0)
    gl.assume(stride_bn >= 0)

    cfg = _medium_decode_moe_config(BLOCK_M, BLOCK_N, BLOCK_K)
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32
    gl.static_assert(
        BLOCK_K_SCALE == 8, "M=8/16 direct WScale path assumes BLOCK_K=256"
    )
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // 2

    X_ELEM_BITS: gl.constexpr = X.dtype.element_ty.primitive_bitwidth
    W_ELEM_BITS: gl.constexpr = W.dtype.element_ty.primitive_bitwidth
    LOAD_X_LAYOUT: gl.constexpr = _load_layout(BLOCK_K, BLOCK_M, 4, [1, 0], X_ELEM_BITS)
    LOAD_W_LAYOUT: gl.constexpr = _load_layout(
        BLOCK_K_PACKED, BLOCK_N, 4, [0, 1], W_ELEM_BITS
    )
    DIRECT_SHARED_X: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, order=[1, 0])
    DIRECT_SHARED_W: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])

    m_base = gl.load(SliceOffs + expert).to(gl.int32)
    m_size = gl.load(SliceSizes + expert).to(gl.int32)
    gl.assume(m_base >= 0)
    gl.assume(m_size > 0)
    if not MEDIUM_COMBINE:
        off_m = m_base + block_in_expert * BLOCK_M
        gl.assume(off_m >= 0)
    off_n = pid_n * BLOCK_N
    gl.assume(off_n >= 0)

    lk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))
    lwk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))[:, None]
    lwn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))[None, :]

    local_m_l = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    valid_m_l = local_m_l < m_size
    safe_local_m_l = gl.where(valid_m_l, local_m_l, gl.zeros_like(local_m_l))
    sorted_l = m_base + safe_local_m_l
    if MEDIUM_COMBINE:
        # Combine reads the sorted (contiguous) X rows directly.
        row_addr = sorted_l
    else:
        # Dispatch gathers the routed token rows for this expert block.
        token_m_l = gl.amd.cdna4.buffer_load(
            ptr=Gather,
            offsets=sorted_l.to(gl.int32),
            mask=valid_m_l,
            other=0,
        ).to(gl.int32)
        row_addr = token_m_l.to(gl.uint32)

    a_scale = gl.full(
        (BLOCK_M, BLOCK_K_SCALE), 127, gl.uint8, layout=cfg.layout_x_scale
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)

    total_kt = gl.cdiv(K, BLOCK_K)
    gl.assume(total_kt >= 0)
    num_full = K // BLOCK_K
    gl.assume(num_full >= 0)
    for kt in range(0, num_full):
        gl.assume(kt >= 0)
        acc = _medium_decode_mfma_tile(
            X,
            W,
            WScale,
            row_addr,
            expert,
            kt,
            off_n,
            lk,
            lwk,
            lwn,
            acc,
            a_scale,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            K,
            N,
            cfg,
            DIRECT_SHARED_X,
            DIRECT_SHARED_W,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_K_PACKED,
            MASK_K=False,
            MASK_N=MEDIUM_COMBINE,
        )

    if total_kt > num_full:
        kt = num_full
        gl.assume(kt >= 0)
        acc = _medium_decode_mfma_tile(
            X,
            W,
            WScale,
            row_addr,
            expert,
            kt,
            off_n,
            lk,
            lwk,
            lwn,
            acc,
            a_scale,
            stride_xm,
            stride_xk,
            stride_we,
            stride_wn,
            stride_wk,
            stride_wse,
            stride_wsn,
            stride_wsk,
            K,
            N,
            cfg,
            DIRECT_SHARED_X,
            DIRECT_SHARED_W,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_K_PACKED,
            MASK_K=True,
            MASK_N=True,
        )

    acc = acc * gl.load(x_global_scale_ptr).to(gl.float32)
    if MEDIUM_COMBINE:
        if HAS_BIAS:
            bias_n = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
            if Y_N_CONST:
                bias_bound = bias_n < Y_N_CONST
            else:
                bias_bound = bias_n < N
            bias = gl.load(
                bias_ptr
                + expert.to(gl.int64) * stride_be
                + bias_n.to(gl.int64) * stride_bn,
                mask=bias_bound,
                other=0.0,
            ).to(gl.float32)
            bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
            acc = acc + bias[None, :]

        out = acc.to(Y.dtype.element_ty)
        STORE_LAYOUT: gl.constexpr = out.type.layout
        n_out = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT))
        local_store_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
        valid_store = local_store_m < m_size
        safe_store_m = gl.where(
            valid_store, local_store_m, gl.zeros_like(local_store_m)
        )
        sorted_store = m_base + safe_store_m
        scatter_row = gl.load(Scatter + sorted_store, mask=valid_store, other=0).to(
            gl.int32
        )
        gate = gl.load(Gate + sorted_store, mask=valid_store, other=0.0).to(
            Y.dtype.element_ty
        )
        out = out * gate[:, None]
        y_offs = (
            scatter_row[:, None].to(gl.int64) * stride_ym
            + n_out[None, :].to(gl.int64) * stride_yn
        )
        if Y_N_CONST:
            store_mask = valid_store[:, None] & (n_out[None, :] < Y_N_CONST)
        else:
            store_mask = valid_store[:, None] & (n_out[None, :] < N)
        _moe_masked_store(out, Y, y_offs, store_mask, USE_BUFFER_STORE=False)
    else:
        if HAS_BIAS:
            bias_n = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
            if Y_N_CONST:
                bias_bound = bias_n < Y_N_CONST
            else:
                bias_bound = bias_n < N
            bias = gl.load(
                bias_ptr
                + expert.to(gl.int64) * stride_be
                + bias_n.to(gl.int64) * stride_bn,
                mask=bias_bound,
                other=0.0,
            ).to(gl.float32)
            bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
            acc = acc + bias[None, :]

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
        sm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT))
        n_out = pid_n * OUT_BLOCK_N + gl.arange(
            0, OUT_BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT)
        )
        local_store_m = block_in_expert * BLOCK_M + sm
        sorted_store = m_base + local_store_m
        valid_store = local_store_m < m_size
        y_off = (
            sorted_store[:, None].to(gl.int64) * stride_ym
            + n_out[None, :].to(gl.int64) * stride_yn
        )
        if Y_N_CONST:
            store_mask = valid_store[:, None] & (n_out[None, :] < (Y_N_CONST // 2))
        else:
            store_mask = valid_store[:, None] & (n_out[None, :] < (N // 2))
        _moe_masked_store(out, Y, y_off, store_mask, USE_BUFFER_STORE=True)
