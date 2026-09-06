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


from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    MoEConfig,
    MoEPipelinedProgram,
    _enforce_wave_uniform_i32,
    _situ_gfx1250,
    _swiglu_gfx1250,
    compute_offsets,
    compute_pids,
    create_descriptor,
    get_blocked_layout,
    get_scaled_dot_format_string,
    get_tdm_gather_scatter_idx_layout,
)


@gluon.jit
def _matmul_decode(
    Y,
    stride_y_k,
    stride_y_z,
    stride_y_m,
    stride_y_n,
    XGlobalScale,
    X,
    stride_x_z,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_z,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    W_TRANSPOSE: gl.constexpr,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,  # Bias
    M,
    N,
    K,
    KW,  # shapes
    GatherIndx,
    WriteBackIndx,
    writeback_size,
    RAGGED_DIMENSION: gl.constexpr,  #
    XSliceSizes,
    XSliceOffs,
    XBlockOffs,
    XBlockSchedule,
    X_EXPECTED_SLICE_SIZE: gl.constexpr,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,  #
    WSliceSizes,
    WSliceOffs,
    WBlockOffs,
    WBlockSchedule,
    W_EXPECTED_SLICE_SIZE: gl.constexpr,
    _W_SLICE_SIZES_DIVISIBILITY: gl.constexpr,  #
    # true grid size
    batch_size,
    grid_m,
    grid_n,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    # MoE config
    N_EXPTS_TOT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
    NUM_BUFFERS: gl.constexpr = 2,
    SCALE_BLOCK: gl.constexpr = 32,
    SCHEDULE: gl.constexpr = "baseline",
    PINGPONG: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
):
    # Decode is a small-M, M-ragged MoE GEMM with a fixed baseline schedule.
    gl.static_assert(
        RAGGED_DIMENSION == "M",
        "decode kernel only supports M-ragged MoE GEMMs",
    )
    gl.static_assert(
        SCHEDULE == "baseline",
        "decode kernel does not have schedule variants",
    )
    gl.static_assert(
        not PINGPONG,
        "decode kernel does not have a ping-pong variant",
    )
    gl.static_assert(
        _W_SLICE_SIZES_DIVISIBILITY is None,
        "decode kernel does not support K-ragged weights",
    )
    SPLIT_K: gl.constexpr = 1

    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)

    if GatherIndx is not None:
        # In triton_kernels, when indices exceed int32 range, they are upcasted to int64. TDM Gather doesn't
        # support int64 indices. Only int16 or int32 are supported. In that case, we need to fall back to
        # AsyncCopy. Fortunately in the GPT-OSS example, we don't need to upcast.
        gl.static_assert(
            not UPCAST_INDICES,
            "TDM Gather doesn't support int64 indices. Only int16 or int32 are supported.",
        )

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    USE_GATHER: gl.constexpr = GatherIndx is not None

    SCALE_PRESHUFFLE: gl.constexpr = (
        SWIZZLE_MX_SCALE is not None and SWIZZLE_MX_SCALE != "STRIDED"
    )

    WITH_X_MX_SCALE: gl.constexpr = XMxScale is not None
    WITH_W_MX_SCALE: gl.constexpr = WMxScale is not None

    NUM_SUBTILES: gl.constexpr = (1, 1, 1)

    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        SCALE_BLOCK=SCALE_BLOCK,
        NUM_BUFFERS=NUM_BUFFERS,
        W_TRANSPOSE=W_TRANSPOSE,
        WITH_X_MX_SCALE=WITH_X_MX_SCALE,
        WITH_W_MX_SCALE=WITH_W_MX_SCALE,
        SCALE_PRESHUFFLE=SCALE_PRESHUFFLE,
        index_type=index_type,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        USE_GATHER=USE_GATHER,
        NUM_WARPS=NUM_WARPS,
    )

    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_W

    W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = 1

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    padding_m = grid_m - gl.load(XBlockOffs + N_EXPTS_TOT)

    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K

    if padding_m > 0 and pid >= total_actual_tiles:
        return

    _, pid_m, pid_n, pid_k = compute_pids(
        pid, unpadded_m, grid_n, total_actual_tiles, XCD_SWIZZLE, GROUP_M, SPLIT_K
    )

    expt_id, _, _, start_m, _, off_m, off_k_x, _ = compute_offsets(
        0,
        pid_m,
        pid_k,
        XBlockSchedule,
        XSliceOffs,
        XBlockOffs,
        X_SLICE_SIZES_DIVISIBILITY,
        WBlockSchedule,
        WSliceOffs,
        W_SLICE_SIZES_DIVISIBILITY,
        RAGGED_DIMENSION,
        BLOCK_M,
        BLOCK_K,
        PACKED_BLOCK_K_W,
        SPLIT_K,
    )
    if X_SLICE_SIZES_DIVISIBILITY is not None:
        off_k_x = off_k_x // X_SLICE_SIZES_DIVISIBILITY * X_SLICE_SIZES_DIVISIBILITY

    eM = gl.multiple_of(gl.load(XSliceSizes + expt_id), X_SLICE_SIZES_DIVISIBILITY)

    expt_id, off_m = expt_id.to(cfg.index_type), off_m.to(cfg.index_type)
    start_m = start_m.to(cfg.index_type)
    pid_n, pid_k = pid_n.to(cfg.index_type), pid_k.to(cfg.index_type)

    X_ptr = X
    if not cfg.USE_GATHER:
        X_ptr += start_m * stride_x_m

    W_ptr = W + expt_id * stride_w_e
    w_offs = pid_n * BLOCK_N * stride_w_n

    if cfg.WITH_X_MX_SCALE:
        XMxScale_ptr = XMxScale
        if not cfg.USE_GATHER:
            XMxScale_ptr += start_m * stride_x_mx_m
    else:
        XMxScale_ptr = XMxScale

    if cfg.WITH_W_MX_SCALE:
        WMxScale_ptr = WMxScale + expt_id * stride_w_mx_e
        w_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n
    else:
        WMxScale_ptr = WMxScale
        w_scale_offs = 0

    descriptor_m = M
    if not cfg.USE_GATHER:
        descriptor_m = eM - off_m
    x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m = create_descriptor(
        cfg,
        X_ptr,
        W_ptr,
        XMxScale_ptr,
        WMxScale_ptr,
        off_m,
        off_k_x,
        w_offs,
        w_scale_offs,
        descriptor_m,
        N,
        K,
        stride_x_m,
        stride_x_k,
        stride_w_k,
        stride_w_n,
        stride_x_mx_m,
        stride_x_mx_k,
        stride_w_mx_n,
        stride_w_mx_k,
        GatherIndx,
        start_m,
    )

    Y_ptr = Y

    pgm = MoEPipelinedProgram.initialize(
        cfg,
        x_desc,
        w_desc,
        x_scale_desc,
        w_scale_desc,
        gathered_m,
        off_k_x // cfg.DIV_FACTOR_X,
    )

    loop_k = K - off_k_x
    acc = pgm.pipeline(loop_k)
    if XGlobalScale is not None and not cfg.WITH_X_MX_SCALE:
        acc *= gl.load(XGlobalScale).to(gl.float32)

    # bias
    b_dtype = B.dtype if B is not None else gl.float32
    BLOCKED_LAYOUT_BIAS: gl.constexpr = get_blocked_layout(
        [BLOCK_N], b_dtype, cfg.NUM_WARPS, 1
    )
    offs_bias_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, BLOCKED_LAYOUT_BIAS)
    mask_bias_n = offs_bias_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_bias_n
        bias = gl.load(BPtrs, mask=mask_bias_n, other=0)
    else:
        bias = gl.full([BLOCK_N], 0, dtype=gl.float32, layout=BLOCKED_LAYOUT_BIAS)

    bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
    acc += bias[None, :]

    gl.static_assert(
        not (DO_SWIGLU and DO_SITU),
        "SwiGLU and SiTU cannot both be enabled",
    )
    if DO_SITU:
        out = _situ_gfx1250(acc, SITU_BETA, SITU_LINEAR_BETA)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    elif DO_SWIGLU:
        out = _swiglu_gfx1250(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    else:
        out = acc
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )

    BLOCKED_LAYOUT_Y: gl.constexpr = get_blocked_layout(
        [BLOCK_M, OUT_BLOCK_N], Y.dtype, cfg.NUM_WARPS
    )
    out = out.to(Y.dtype.element_ty)
    out = gl.convert_layout(out, BLOCKED_LAYOUT_Y)

    if WriteBackIndx is not None:
        WriteBackIndx += start_m

        SCATTER_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )

        IDX_BASE_LAYOUT: gl.constexpr = get_tdm_gather_scatter_idx_layout(
            BLOCK_M, cfg.NUM_WARPS
        )
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(1, IDX_BASE_LAYOUT)

        idx_offs = gl.arange(0, BLOCK_M, IDX_LAYOUT)
        idx_mask = (off_m + idx_offs < eM) & (
            start_m + off_m + idx_offs < writeback_size
        )
        dst_row_indices = gl.load(
            WriteBackIndx + off_m + idx_offs, mask=idx_mask, other=writeback_size
        )
        dst_row_indices = dst_row_indices.to(cfg.index_type)

        out_smem = gl.allocate_shared_memory(
            Y.dtype.element_ty, (BLOCK_M, OUT_BLOCK_N), SCATTER_SHARED_LAYOUT
        )
        out_smem.store(out)

        y_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=Y_ptr,
            shape=(writeback_size, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=SCATTER_SHARED_LAYOUT,
        )

        col_offset = (OUT_BLOCK_N * _enforce_wave_uniform_i32(pid_n)).to(cfg.index_type)
        y_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            y_desc, add_offsets=[0, col_offset], clamp_bounds=True
        )
        gl.amd.gfx1250.tdm.async_scatter(y_desc, dst_row_indices, out_smem)
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, BLOCKED_LAYOUT_Y))
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(
            0, OUT_BLOCK_N, gl.SliceLayout(0, BLOCKED_LAYOUT_Y)
        )
        mask_m = offs_y_m < eM
        mask_n = offs_y_n < yN

        Y_ptr += start_m * stride_y_m

        y_offs = (
            offs_y_m.to(cfg.index_type)[:, None] * stride_y_m
            + offs_y_n.to(cfg.index_type)[None, :] * stride_y_n
        )
        y_mask = mask_m[:, None] & mask_n[None, :]
        gl.amd.gfx1250.buffer_store(out, Y_ptr, y_offs, mask=y_mask)
