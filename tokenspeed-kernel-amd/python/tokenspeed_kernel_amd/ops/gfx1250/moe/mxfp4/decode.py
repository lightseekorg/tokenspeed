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

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    MoEConfig,
    MoEPipelinedProgram,
    _enforce_wave_uniform_i32,
    _situ_gfx1250,
    _swiglu_gfx1250,
    compute_pids,
    get_blocked_layout,
    get_scaled_dot_format_string,
    get_tdm_gather_scatter_idx_layout,
    partial_tdm_warp_hint,
    reverse_tdm_warp_used_hint,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.quantize import (
    store_mxfp4_tile_gfx1250,
)

# Reduction length from which the decode tile streams 256-byte weight rows.
_DECODE_LONG_K = 2048


def decode_tile_config(k: int) -> dict[str, int]:
    """Return the decode tile for a projection with reduction length ``k``.

    Decode streams each touched expert's weights once, so the tile is chosen
    for HBM streaming rather than reuse. 128-column tiles give a single-token
    step enough programs to occupy the device. A long K reads 256-byte MXFP4
    row segments (``block_k=512``), which keeps more bytes in flight per TDM
    operation. A short K uses ``block_k=128`` so the three buffers hold the
    whole reduction in a small LDS footprint, letting more programs stay
    resident to hide each program's fixed latency.
    """
    return {
        "block_n": 128,
        "block_k": 512 if k >= _DECODE_LONG_K else 128,
        "num_buffers": 3,
        "num_warps": 4,
    }


def _decode_launch_metadata(grid, kernel, args):
    """Report routed-row work and modeled traffic to Proton.

    The number of distinct experts is only known on the device, so the weight
    traffic assumes every routed row selects its own expert, capped at the
    expert count. That bound is exact for a single-token step and an
    over-estimate once rows share experts.
    """
    m, n, k = args["M"], args["N"], args["K"]
    experts = min(m, args["N_EXPTS_TOT"])

    def row_bytes(values, scales):
        # uint8 operands pack two e2m1 values per byte; E8M0 scales add one
        # byte per SCALE_BLOCK values.
        packed = values.dtype == torch.uint8
        data = k // 2 if packed else k * values.element_size()
        return data + (k // args["SCALE_BLOCK"] if scales is not None else 0)

    w_bytes = n * row_bytes(args["W"], args["WMxScale"])
    x_bytes = m * row_bytes(args["X"], args["XMxScale"])
    y_values = m * (n // args["ACTIVATION_REDUCTION_N"])
    if args["YMxScale"] is None:
        y_bytes = y_values * args["Y"].element_size()
    else:
        y_bytes = y_values // 2 + y_values // args["SCALE_BLOCK"]
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}]",
        "flops8": 2 * m * n * k,
        "bytes": experts * w_bytes + x_bytes * args["grid_n"] + y_bytes,
    }


@gluon.constexpr_function
def _tdm_index_layout(num_indices, num_warps):
    """Layout of the row indices a TDM gather or scatter consumes."""
    IDX_BASE_LAYOUT = get_tdm_gather_scatter_idx_layout(num_indices, num_warps)
    return gl.SliceLayout(0, IDX_BASE_LAYOUT)


@gluon.jit
def _create_w_descriptor(
    cfg: MoEConfig,
    w_ptr,
    w_scale_ptr,
    w_offs,
    w_scale_offs,
    N,
    K,
    stride_wk,
    stride_wn,
    stride_w_scale_n,
    stride_w_scale_k,
):
    gl.static_assert(cfg.W_TRANSPOSE, "decode kernel expects N-major weights")
    w_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=w_ptr + w_offs,
        shape=(N, K // cfg.DIV_FACTOR_W),
        strides=(stride_wn, stride_wk),
        block_shape=(cfg.BLOCK_N, cfg.BLOCK_K // cfg.DIV_FACTOR_W),
        layout=cfg.shared_layout_w,
    )
    if cfg.WITH_W_MX_SCALE:
        PRESHUFFLE_FACTOR: gl.constexpr = cfg.PRESHUFFLE_FACTOR
        SCALE_BLOCK: gl.constexpr = cfg.SCALE_BLOCK
        SCALE_KWIDTH: gl.constexpr = cfg.SCALE_KWIDTH
        # The padded shape makes TDM load all of the preshuffled scales.
        N_PADDED = (N + PRESHUFFLE_FACTOR - 1) // PRESHUFFLE_FACTOR * PRESHUFFLE_FACTOR
        K_SCALE = (K + SCALE_BLOCK - 1) // SCALE_BLOCK
        K_SCALE_PADDED = (K_SCALE + SCALE_KWIDTH - 1) // SCALE_KWIDTH * SCALE_KWIDTH
        w_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=w_scale_ptr + w_scale_offs,
            shape=(N_PADDED // PRESHUFFLE_FACTOR, K_SCALE_PADDED * PRESHUFFLE_FACTOR),
            strides=(stride_w_scale_n, stride_w_scale_k),
            block_shape=(cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED),
            layout=cfg.shared_layout_w_scale,
        )
    else:
        w_scale_desc = gl.constexpr(0)
    return w_desc, w_scale_desc


@gluon.jit
def _create_x_descriptor(
    cfg: MoEConfig,
    x_ptr,
    x_scale_ptr,
    off_m,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_x_scale_m,
    stride_x_scale_k,
    GatherIndx,
    start_m,
):
    SCALE_BLOCK: gl.constexpr = cfg.SCALE_BLOCK
    if cfg.USE_GATHER:
        IDX_LAYOUT: gl.constexpr = _tdm_index_layout(cfg.BLOCK_M, cfg.NUM_WARPS)
        offs_m_gather = off_m + gl.arange(0, cfg.BLOCK_M, IDX_LAYOUT)
        gathered_m = gl.load(
            GatherIndx + start_m + offs_m_gather,
            mask=start_m + offs_m_gather < M,
            other=0,
        ).to(cfg.index_type)
        x_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=x_ptr,
            shape=(M, K // cfg.DIV_FACTOR_X),
            strides=(stride_xm, stride_xk),
            block_shape=(cfg.BLOCK_M, cfg.BLOCK_K // cfg.DIV_FACTOR_X),
            layout=cfg.shared_layout_x,
        )
        if cfg.WITH_X_MX_SCALE:
            x_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
                base=x_scale_ptr,
                shape=(M, K // SCALE_BLOCK),
                strides=(stride_x_scale_m, stride_x_scale_k),
                block_shape=(cfg.BLOCK_M, cfg.BLOCK_K // SCALE_BLOCK),
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_desc = gl.constexpr(0)
    else:
        gathered_m = gl.constexpr(0)
        x_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=x_ptr + off_m * stride_xm,
            shape=(M, K // cfg.DIV_FACTOR_X),
            strides=(stride_xm, stride_xk),
            block_shape=(cfg.BLOCK_M, cfg.BLOCK_K // cfg.DIV_FACTOR_X),
            layout=cfg.shared_layout_x,
        )
        if cfg.WITH_X_MX_SCALE:
            x_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
                base=x_scale_ptr + off_m * stride_x_scale_m,
                shape=(M, K // SCALE_BLOCK),
                strides=(stride_x_scale_m, stride_x_scale_k),
                block_shape=(cfg.BLOCK_M, cfg.BLOCK_K // SCALE_BLOCK),
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_desc = gl.constexpr(0)
    return x_desc, x_scale_desc, gathered_m


@gluon.jit
def _allocate_w_buffers(cfg: MoEConfig, w_desc):
    w_buffer = gl.allocate_shared_memory(
        w_desc.dtype,
        shape=[cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K // cfg.DIV_FACTOR_W],
        layout=cfg.shared_layout_w,
    )
    if cfg.WITH_W_MX_SCALE:
        w_scale_buffer = gl.allocate_shared_memory(
            gl.uint8,
            shape=[
                cfg.NUM_BUFFERS,
                cfg.BLOCK_N_PRESHUFFLED,
                cfg.BLOCK_K_SCALE_PRESHUFFLED,
            ],
            layout=cfg.shared_layout_w_scale,
        )
    else:
        w_scale_buffer = gl.constexpr(0)
    return w_buffer, w_scale_buffer


@gluon.jit
def _allocate_x_buffers(cfg: MoEConfig, x_desc):
    x_buffer = gl.allocate_shared_memory(
        x_desc.dtype,
        shape=[cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K // cfg.DIV_FACTOR_X],
        layout=cfg.shared_layout_x,
    )
    if cfg.WITH_X_MX_SCALE:
        x_scale_buffer = gl.allocate_shared_memory(
            gl.uint8,
            shape=[cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K // cfg.SCALE_BLOCK],
            layout=cfg.shared_layout_x_scale,
        )
    else:
        x_scale_buffer = gl.constexpr(0)
    return x_buffer, x_scale_buffer


@gluon.constexpr_function
def _reversed_partial_tdm_hint(num_warps):
    return reverse_tdm_warp_used_hint(partial_tdm_warp_hint(num_warps), num_warps)


@gluon.jit
def _issue_w_loads(
    cfg: MoEConfig, w_desc, w_scale_desc, w_buffer, w_scale_buffer, load_idx, pred
):
    buf = load_idx % cfg.NUM_BUFFERS
    w_offsets = [0, load_idx * (cfg.BLOCK_K // cfg.DIV_FACTOR_W)]
    w_scale_offsets = [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED]
    # Decode issues the weights before the activation rows are known, so
    # partial TDM pairs each weight load with its scale load: half the warps
    # carry each, as one fused operation.
    FUSE: gl.constexpr = cfg.FUSE_X_W or cfg.FUSE_W_W_SCALE
    if FUSE:
        W_HINT: gl.constexpr = partial_tdm_warp_hint(cfg.NUM_WARPS)
        W_SCALE_HINT: gl.constexpr = _reversed_partial_tdm_hint(cfg.NUM_WARPS)
        gl.amd.cdna5.tdm.async_load_fused(
            [
                (
                    gl.amd.cdna5.tdm.update_tensor_descriptor(
                        w_desc, add_offsets=w_offsets, pred=pred, clamp_bounds=True
                    ),
                    w_buffer.index(buf),
                    W_HINT,
                ),
                (
                    gl.amd.cdna5.tdm.update_tensor_descriptor(
                        w_scale_desc,
                        add_offsets=w_scale_offsets,
                        pred=pred,
                        clamp_bounds=True,
                    ),
                    w_scale_buffer.index(buf),
                    W_SCALE_HINT,
                ),
            ]
        )
    else:
        gl.amd.cdna5.tdm.async_load(w_desc, w_offsets, w_buffer.index(buf), pred=pred)
        if cfg.WITH_W_MX_SCALE:
            gl.amd.cdna5.tdm.async_load(
                w_scale_desc, w_scale_offsets, w_scale_buffer.index(buf), pred=pred
            )


@gluon.jit
def _issue_x_loads(
    cfg: MoEConfig,
    x_desc,
    x_scale_desc,
    x_buffer,
    x_scale_buffer,
    gathered_m,
    load_idx,
    pred,
):
    buf = load_idx % cfg.NUM_BUFFERS
    BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    if cfg.USE_GATHER:
        x_desc_k = gl.amd.cdna5.tdm.update_tensor_descriptor(
            x_desc,
            add_offsets=[0, load_idx * BLOCK_K_PACKED_X],
            pred=pred,
            clamp_bounds=True,
        )
        gl.amd.cdna5.tdm.async_gather(x_desc_k, gathered_m, x_buffer.index(buf))
    else:
        gl.amd.cdna5.tdm.async_load(
            x_desc, [0, load_idx * BLOCK_K_PACKED_X], x_buffer.index(buf), pred=pred
        )
    if cfg.WITH_X_MX_SCALE:
        if cfg.USE_GATHER:
            x_scale_desc_k = gl.amd.cdna5.tdm.update_tensor_descriptor(
                x_scale_desc,
                add_offsets=[0, load_idx * (cfg.BLOCK_K // cfg.SCALE_BLOCK)],
                pred=pred,
                clamp_bounds=True,
            )
            gl.amd.cdna5.tdm.async_gather(
                x_scale_desc_k, gathered_m, x_scale_buffer.index(buf)
            )
        else:
            gl.amd.cdna5.tdm.async_load(
                x_scale_desc,
                [0, load_idx * (cfg.BLOCK_K // cfg.SCALE_BLOCK)],
                x_scale_buffer.index(buf),
                pred=pred,
            )


@gluon.jit
def _decode_pipeline(pgm: MoEPipelinedProgram, K):
    """Run the K loop after the prologue issued every weight load first.

    The prologue issues W(0..NB-2) and then X(0..NB-2); each loop iteration
    issues X then W for NB-1 steps ahead. Only the first NB-2 iterations see a
    different number of later TDM operations, so they are peeled with their
    own wait counts.
    """
    cfg = pgm.cfg
    NB: gl.constexpr = cfg.NUM_BUFFERS
    NX: gl.constexpr = 2 if cfg.WITH_X_MX_SCALE else 1
    # Partial TDM fuses each weight load with its scale load.
    FUSE_W: gl.constexpr = cfg.FUSE_X_W or cfg.FUSE_W_W_SCALE
    NW: gl.constexpr = 1 if FUSE_W or not cfg.WITH_W_MX_SCALE else 2
    LOADS: gl.constexpr = NX + NW

    accumulator = gl.zeros(
        (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
    )
    loop_ub = gl.cdiv(K, cfg.BLOCK_K)
    gl.assume(loop_ub > 0)
    for i in gl.static_range(NB - 2):
        if i < loop_ub:
            pred = (i + NB - 1 < loop_ub).to(gl.int32)
            _issue_x_loads(
                cfg,
                pgm.x_desc,
                pgm.x_scale_desc,
                pgm.x_buffer,
                pgm.x_scale_buffer,
                pgm.gathered_m,
                i + NB - 1,
                pred,
            )
            _issue_w_loads(
                cfg,
                pgm.w_desc,
                pgm.w_scale_desc,
                pgm.w_buffer,
                pgm.w_scale_buffer,
                i + NB - 1,
                pred,
            )
            gl.amd.cdna5.tdm.async_wait((NB - 2 - i) * NX + (i + 1) * LOADS)
            x, w, scale_x, scale_w = pgm.issue_local_loads(i)
            accumulator = pgm.wmma(x, scale_x, w, scale_w, accumulator)
    LOOP_START: gl.constexpr = NB - 2 if NB > 2 else 0
    for i in range(LOOP_START, loop_ub):
        pred = (i + NB - 1 < loop_ub).to(gl.int32)
        _issue_x_loads(
            cfg,
            pgm.x_desc,
            pgm.x_scale_desc,
            pgm.x_buffer,
            pgm.x_scale_buffer,
            pgm.gathered_m,
            i + NB - 1,
            pred,
        )
        _issue_w_loads(
            cfg,
            pgm.w_desc,
            pgm.w_scale_desc,
            pgm.w_buffer,
            pgm.w_scale_buffer,
            i + NB - 1,
            pred,
        )
        gl.amd.cdna5.tdm.async_wait((NB - 1) * LOADS)
        x, w, scale_x, scale_w = pgm.issue_local_loads(i)
        accumulator = pgm.wmma(x, scale_x, w, scale_w, accumulator)
    return accumulator


@gluon.jit(launch_metadata=_decode_launch_metadata)
def _matmul_decode(
    Y,
    stride_y_k,
    stride_y_z,
    stride_y_m,
    stride_y_n,
    YGlobalScale,
    YMxScale,
    stride_y_mx_m,
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
    INDEX_TYPE: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
    NUM_BUFFERS: gl.constexpr = 2,
    SCALE_BLOCK: gl.constexpr = 32,
    SCHEDULE: gl.constexpr = "baseline",
    PINGPONG: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
    PARTIAL_TDM: gl.constexpr = False,
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
    gl.static_assert(
        not PARTIAL_TDM or WMxScale is not None,
        "decode partial TDM fuses weight loads with their scale loads",
    )
    SPLIT_K: gl.constexpr = 1

    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)

    # Width of pointer arithmetic only; TDM row indices use INDEX_TYPE.
    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
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
        index_type=INDEX_TYPE,
        PARTIAL_TDM=PARTIAL_TDM,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        USE_GATHER=USE_GATHER,
        NUM_WARPS=NUM_WARPS,
    )

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    # Map over the padded grid so the tile does not wait on the active block
    # count; padding blocks carry a -1 schedule entry and exit here.
    _, pid_m, pid_n, _ = compute_pids(
        pid, grid_m, grid_n, grid_m * grid_n, XCD_SWIZZLE, GROUP_M, SPLIT_K
    )
    block_schedule = gl.load(XBlockSchedule + pid_m)
    if block_schedule == -1:
        return
    expt_id = block_schedule & 0x0000FFFF
    off_m = BLOCK_M * (block_schedule >> 16)

    expt_id, off_m = expt_id.to(address_index_type), off_m.to(address_index_type)
    pid_n = pid_n.to(address_index_type)

    # The weight tiles depend only on the expert, so start streaming them
    # before the row metadata and gather indices arrive.
    W_ptr = W + expt_id * stride_w_e
    w_offs = pid_n * BLOCK_N * stride_w_n
    if cfg.WITH_W_MX_SCALE:
        WMxScale_ptr = WMxScale + expt_id * stride_w_mx_e
        w_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n
    else:
        WMxScale_ptr = WMxScale
        w_scale_offs = 0
    w_desc, w_scale_desc = _create_w_descriptor(
        cfg,
        W_ptr,
        WMxScale_ptr,
        w_offs,
        w_scale_offs,
        # The weight descriptors start at this N tile, so their bounds must too.
        (N - pid_n * BLOCK_N).to(gl.int32),
        K,
        stride_w_k,
        stride_w_n,
        stride_w_mx_n,
        stride_w_mx_k,
    )
    w_buffer, w_scale_buffer = _allocate_w_buffers(cfg, w_desc)
    for i in gl.static_range(NUM_BUFFERS - 1):
        _issue_w_loads(cfg, w_desc, w_scale_desc, w_buffer, w_scale_buffer, i, 1)

    start_m = gl.load(XSliceOffs + expt_id).to(address_index_type)
    eM = gl.multiple_of(gl.load(XSliceSizes + expt_id), X_SLICE_SIZES_DIVISIBILITY)

    X_ptr = X
    if not cfg.USE_GATHER:
        X_ptr += start_m * stride_x_m
    if cfg.WITH_X_MX_SCALE:
        XMxScale_ptr = XMxScale
        if not cfg.USE_GATHER:
            XMxScale_ptr += start_m * stride_x_mx_m
    else:
        XMxScale_ptr = XMxScale

    descriptor_m = M
    if not cfg.USE_GATHER:
        # Rows left in this expert fit i32 even when the weight slab needs the
        # wide index type.
        descriptor_m = (eM - off_m).to(gl.int32)
    x_desc, x_scale_desc, gathered_m = _create_x_descriptor(
        cfg,
        X_ptr,
        XMxScale_ptr,
        off_m,
        descriptor_m,
        K,
        stride_x_m,
        stride_x_k,
        stride_x_mx_m,
        stride_x_mx_k,
        GatherIndx,
        start_m,
    )
    x_buffer, x_scale_buffer = _allocate_x_buffers(cfg, x_desc)
    for i in gl.static_range(NUM_BUFFERS - 1):
        _issue_x_loads(
            cfg, x_desc, x_scale_desc, x_buffer, x_scale_buffer, gathered_m, i, 1
        )

    if WriteBackIndx is not None:
        # Fetch the scatter rows now; their latency hides behind the K loop.
        IDX_LAYOUT: gl.constexpr = _tdm_index_layout(BLOCK_M, cfg.NUM_WARPS)
        idx_offs = gl.arange(0, BLOCK_M, IDX_LAYOUT)
        idx_mask = (off_m + idx_offs < eM) & (
            start_m + off_m + idx_offs < writeback_size
        )
        dst_row_indices = gl.load(
            WriteBackIndx + start_m + off_m + idx_offs,
            mask=idx_mask,
            other=writeback_size,
        ).to(cfg.index_type)

    pgm = MoEPipelinedProgram(
        cfg,
        x_buffer,
        w_buffer,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        x_scale_desc,
        w_scale_desc,
        gathered_m,
        gl.to_tensor(0),
    )
    acc = _decode_pipeline(pgm, K)
    Y_ptr = Y
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

    if YMxScale is not None:
        gl.static_assert(
            WriteBackIndx is None and YGlobalScale is None,
            "MXFP4 output supports neither scatter nor a global output scale",
        )
        store_mxfp4_tile_gfx1250(
            out,
            Y_ptr,
            stride_y_m,
            YMxScale,
            stride_y_mx_m,
            start_m + off_m,
            eM - off_m,
            OUT_BLOCK_N * pid_n,
            yN,
            cfg.NUM_WARPS,
        )
        return

    BLOCKED_LAYOUT_Y: gl.constexpr = get_blocked_layout(
        [BLOCK_M, OUT_BLOCK_N], Y.dtype, cfg.NUM_WARPS
    )
    if YGlobalScale is not None:
        out = out * (1.0 / gl.load(YGlobalScale).to(gl.float32))
    out = out.to(Y.dtype.element_ty)
    out = gl.convert_layout(out, BLOCKED_LAYOUT_Y)

    OUTPUT_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    out_smem = gl.allocate_shared_memory(
        Y.dtype.element_ty, (BLOCK_M, OUT_BLOCK_N), OUTPUT_SHARED_LAYOUT
    )
    out_smem.store(out)

    if WriteBackIndx is not None:
        y_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=Y_ptr,
            shape=(writeback_size, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=OUTPUT_SHARED_LAYOUT,
        )

        # TDM descriptor offsets do not support i64
        col_offset = OUT_BLOCK_N * _enforce_wave_uniform_i32(pid_n.to(gl.int32))
        y_desc_s = gl.amd.cdna5.tdm.update_tensor_descriptor(
            y_desc, add_offsets=[0, col_offset], clamp_bounds=True
        )
        gl.amd.cdna5.tdm.async_scatter(y_desc_s, dst_row_indices, out_smem)
        gl.amd.cdna5.tdm.async_wait(0)
    else:
        y_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=Y_ptr + start_m * stride_y_m,
            shape=(eM, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=OUTPUT_SHARED_LAYOUT,
        )
        # TDM descriptor offsets do not support i64
        gl.amd.cdna5.tdm.async_store(
            y_desc,
            [off_m.to(gl.int32), (OUT_BLOCK_N * pid_n).to(gl.int32)],
            out_smem,
        )
        gl.amd.cdna5.tdm.async_wait(0)
