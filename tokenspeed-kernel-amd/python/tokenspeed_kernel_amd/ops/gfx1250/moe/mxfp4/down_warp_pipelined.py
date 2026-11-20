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

"""Warp-pipelined down GEMM of an FP8 x MXFP4 MoE layer.

The input is the gate/up output, ``BLOCK_M`` contiguous expert-sorted FP8
rows per tile; each 512-column output tile is computed as two 256-column
accumulators, and each 256-wide K tile as two 128-wide halves. The eight warps
form two groups of four that run one warp-pipeline stage apart. Results are
written to ``Y[WriteBackIndx[r]]`` (the route's row) or, without
``WriteBackIndx``, to row ``r``.

``gluon_moe_down_warp_pipelined_gfx1250`` runs one tile per program and stages
the output in LDS freed by the ring for one TDM scatter.
``gluon_moe_down_warp_pipelined_persistent_gfx1250`` keeps the ring across
tiles, issues the next tile's leading K tiles before the epilogue, and stores
each tile straight from registers.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    compute_offsets,
    compute_pids,
    get_scaled_dot_format_string,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._warp_pipeline_common import (
    RingProgram,
    WarpPipelineConfig,
    allocate_ring,
    create_descriptors,
    epilogue_math,
    get_tdm_gather_scatter_idx_layout,
    load_bias,
    load_scale,
    load_w_scale_slice,
    warp_pipeline_launch_metadata,
    wmma_scaled,
)

tdm = gl.amd.cdna5.tdm

# X, W, and the W scales share one fused TDM load per tile. Fused members need
# disjoint warp sets: X from warp 0, the W scales from warp 1, W from warps 2-3.
# Every producer must be in the leading warp-pipeline group (warps 0-3): each
# producer waits only on its own TDM ops, and the leading group starts reading
# a tile one stage before the trailing group reaches its wait, so a producer in
# warps 4-7 would let the leading group read unfinished data.
_TDM_HINT_X = gl.constexpr(0b00000001)
_TDM_HINT_W_SCALE = gl.constexpr(0b00000010)
_TDM_HINT_W = gl.constexpr(0b00001100)
# With partition conflict resolution, X and W are split across the two LDS
# partitions. A partitioned copy needs one warp per piece (X has two, W four),
# so each K tile takes two TDM ops: X with the W scales, then W on its own.
_TDM_HINT_X_PARTITIONED = gl.constexpr(0b00000011)
_TDM_HINT_W_SCALE_PARTITIONED = gl.constexpr(0b00001100)
_TDM_HINT_W_PARTITIONED = gl.constexpr(0b00001111)


@gluon.constexpr_function
def _tdm_ops_per_tile(cfg):
    return 2 if cfg.RESOLVE_PARTITION_CONFLICTS else 1


@gluon.jit
def _issue_global_loads(pgm, load_idx, slot, pred=1):
    cfg = pgm.cfg
    block_k_packed_w: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
    x_desc = tdm.update_tensor_descriptor(
        pgm.x_desc,
        add_offsets=[0, load_idx * cfg.BLOCK_K],
        pred=pred,
        clamp_bounds=True,
    )
    w_desc = tdm.update_tensor_descriptor(
        pgm.w_desc,
        add_offsets=[0, load_idx * block_k_packed_w],
        pred=pred,
        clamp_bounds=True,
    )
    w_scale_desc = tdm.update_tensor_descriptor(
        pgm.w_scale_desc,
        add_offsets=[0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
        pred=pred,
        clamp_bounds=True,
    )
    x_dest = pgm.x_buffer.index(slot)
    w_dest = pgm.w_buffer.index(slot)
    w_scale_dest = pgm.w_scale_buffer.index(slot)
    if cfg.RESOLVE_PARTITION_CONFLICTS:
        # The ring waits count TDM ops in issue order, so a slot's two ops are
        # issued back to back.
        tdm.async_load_fused(
            [
                (x_desc, x_dest, _TDM_HINT_X_PARTITIONED),
                (w_scale_desc, w_scale_dest, _TDM_HINT_W_SCALE_PARTITIONED),
            ],
            cache_modifier=cfg.W_CACHE_MODIFIER,
        )
        tdm.async_load(
            w_desc,
            dest=w_dest,
            warp_used_hint=_TDM_HINT_W_PARTITIONED,
            cache_modifier=cfg.W_CACHE_MODIFIER,
        )
    else:
        tdm.async_load_fused(
            [
                (x_desc, x_dest, _TDM_HINT_X),
                (w_scale_desc, w_scale_dest, _TDM_HINT_W_SCALE),
                (w_desc, w_dest, _TDM_HINT_W),
            ],
            cache_modifier=cfg.W_CACHE_MODIFIER,
        )


@gluon.jit
def _wait_global_loads(cfg, num_tiles: gl.constexpr):
    tdm.async_wait(num_tiles * _tdm_ops_per_tile(cfg))


@gluon.jit
def _local_load_k_half(cfg, x_slot, w_slot, w_scale_slot, k_idx: gl.constexpr):
    """Read one 128-wide K half of a ring slot: X, both N halves' scales, W.

    The scales are read right after X so the first WMMA does not wait on the
    W reads queued behind them.
    """
    sub_k: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES_K
    sub_n: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES_N
    x = x_slot.slice(k_idx * sub_k, sub_k, 1).load(layout=cfg.dot_layout_x)
    scale_w0 = load_w_scale_slice(cfg, w_scale_slot, k_idx * sub_k, sub_k, 0, sub_n)
    scale_w1 = load_w_scale_slice(cfg, w_scale_slot, k_idx * sub_k, sub_k, sub_n, sub_n)
    w_k = w_slot.slice(k_idx * sub_k // cfg.DIV_FACTOR_W, sub_k // cfg.DIV_FACTOR_W, 1)
    w0 = w_k.slice(0, sub_n, 0).permute([1, 0]).load(layout=cfg.dot_layout_w)
    w1 = w_k.slice(sub_n, sub_n, 0).permute([1, 0]).load(layout=cfg.dot_layout_w)
    return x, w0, scale_w0, w1, scale_w1


@gluon.jit
def _mma_k_half(cfg, x, w0, scale_w0, w1, scale_w1, c0, c1):
    c0 = wmma_scaled(cfg, x, w0, scale_w0, c0)
    c1 = wmma_scaled(cfg, x, w1, scale_w1, c1)
    return c0, c1


@gluon.jit
def _consume_and_refill(
    pgm, slot: gl.constexpr, refill_slot: gl.constexpr, refill_idx, c0, c1
):
    """Refill ``refill_slot`` with tile ``refill_idx`` and consume ``slot``.

    Four stages: refill/load low -> compute low -> load high/arrive -> compute
    high/cluster wait. ``refill_slot`` is the slot the previous step consumed.
    """
    cfg = pgm.cfg
    x_slot = pgm.x_buffer.index(slot)
    w_slot = pgm.w_buffer.index(slot)
    w_scale_slot = pgm.w_scale_buffer.index(slot)

    _wait_global_loads(cfg, cfg.NUM_BUFFERS - 2)
    with gl.amd.warp_pipeline_stage("down_refill_load_low"):
        _issue_global_loads(pgm, refill_idx, refill_slot)
        x0, w00, scale_w00, w01, scale_w01 = _local_load_k_half(
            cfg, x_slot, w_slot, w_scale_slot, 0
        )
    with gl.amd.warp_pipeline_stage("down_compute_low"):
        c0, c1 = _mma_k_half(cfg, x0, w00, scale_w00, w01, scale_w01, c0, c1)
    with gl.amd.warp_pipeline_stage("down_load_high_arrive"):
        x1, w10, scale_w10, w11, scale_w11 = _local_load_k_half(
            cfg, x_slot, w_slot, w_scale_slot, 1
        )
        if cfg.CTA_M > 1:
            gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("down_compute_high_wait"):
        c0, c1 = _mma_k_half(cfg, x1, w10, scale_w10, w11, scale_w11, c0, c1)
        if cfg.CTA_M > 1:
            gl.amd.cdna5.cluster.wait()
    return c0, c1


@gluon.jit
def _consume_tail(pgm, slot, c0, c1):
    """Consume one resident tile without refilling it."""
    cfg = pgm.cfg
    x_slot = pgm.x_buffer.index(slot)
    w_slot = pgm.w_buffer.index(slot)
    w_scale_slot = pgm.w_scale_buffer.index(slot)
    for k_idx in gl.static_range(cfg.NUM_SUBTILES_K):
        with gl.amd.warp_pipeline_stage("down_tail_load"):
            x, w0, scale_w0, w1, scale_w1 = _local_load_k_half(
                cfg, x_slot, w_slot, w_scale_slot, k_idx
            )
        with gl.amd.warp_pipeline_stage("down_tail_compute"):
            c0, c1 = _mma_k_half(cfg, x, w0, scale_w0, w1, scale_w1, c0, c1)
    return c0, c1


@gluon.jit
def _tdm_prefetch(pgm, pred=1):
    """Issue a tile's leading K tiles 0 .. NUM_BUFFERS - 2 into the slots of
    the same index; the K loop fills the last slot."""
    for slot in gl.static_range(pgm.cfg.NUM_BUFFERS - 1):
        _issue_global_loads(pgm, slot, slot, pred)


@gluon.jit
def _k_loop_bounds(pgm, loop_k):
    """Refilling steps of the steady loop and its whole ring-size rounds.

    Step t consumes K tile t and issues K tile t + NUM_BUFFERS - 1.
    """
    cfg = pgm.cfg
    iter_max = gl.cdiv(loop_k, cfg.BLOCK_K)
    gl.assume(iter_max >= cfg.NUM_BUFFERS)
    num_steps = iter_max - cfg.NUM_BUFFERS + 1
    return num_steps, num_steps // cfg.NUM_BUFFERS


@gluon.jit
def _k_loop_steady(pgm, loop_k):
    """Run the refilling K loop of one output tile; ``_k_loop_tail`` drains.

    The caller has already issued the first NUM_BUFFERS - 1 K tiles with
    ``_tdm_prefetch``.
    """
    cfg = pgm.cfg
    num_buffers: gl.constexpr = cfg.NUM_BUFFERS
    num_steps, num_rounds = _k_loop_bounds(pgm, loop_k)
    sub_n: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES_N
    c0 = gl.zeros((cfg.BLOCK_M, sub_n), dtype=gl.float32, layout=cfg.acc_layout)
    c1 = gl.zeros((cfg.BLOCK_M, sub_n), dtype=gl.float32, layout=cfg.acc_layout)

    _wait_global_loads(cfg, num_buffers - 2)
    pgm.cluster_sync()
    # Unroll by the ring size so every slot index is a compile-time constant.
    for round_idx in range(0, num_rounds):
        for slot in gl.static_range(num_buffers):
            tile_idx = round_idx * num_buffers + slot
            c0, c1 = _consume_and_refill(
                pgm,
                slot,
                (slot + num_buffers - 1) % num_buffers,
                tile_idx + num_buffers - 1,
                c0,
                c1,
            )
    n_full = num_rounds * num_buffers
    for slot in gl.static_range(num_buffers - 1):
        for _ in range(gl.minimum(gl.maximum(num_steps - n_full - slot, 0), 1)):
            c0, c1 = _consume_and_refill(
                pgm,
                slot,
                (slot + num_buffers - 1) % num_buffers,
                n_full + slot + num_buffers - 1,
                c0,
                c1,
            )
    # Retire the first drained tile.
    _wait_global_loads(cfg, num_buffers - 2)
    return c0, c1


@gluon.jit
def _k_loop_tail(pgm, loop_k, c0, c1):
    """Drain the last NUM_BUFFERS - 1 K tiles through flat pipeline stages.

    Flat stages are illegal directly inside a loop; the persistent tile loop
    wraps this call in a one-trip loop.
    """
    cfg = pgm.cfg
    num_buffers: gl.constexpr = cfg.NUM_BUFFERS
    num_steps, _ = _k_loop_bounds(pgm, loop_k)
    for drain_idx in gl.static_range(num_buffers - 1):
        c0, c1 = _consume_tail(pgm, (num_steps + drain_idx) % num_buffers, c0, c1)
        if drain_idx < num_buffers - 2:
            _wait_global_loads(cfg, num_buffers - 3 - drain_idx)
            pgm.cluster_sync()
    # Peers may still be multicasting into this workgroup's ring; the epilogue
    # reuses that LDS.
    pgm.cluster_sync()
    return c0, c1


@gluon.jit
def _merge_accumulator(cfg, c0, c1):
    # c0 holds output columns [0, BLOCK_N / 2), c1 the rest.
    accumulator = gl.join(c0, c1)
    accumulator = accumulator.permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
    return gl.convert_layout(accumulator, cfg.acc_layout, assert_trivial=True)


@gluon.jit
def _tile_setup(
    cfg,
    tile_id,
    unpadded_m,
    total_tiles,
    grid_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    N,
    K,
    XSliceSizes,
    XSliceOffs,
    XBlockOffs,
    XBlockSchedule,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    GROUP_M: gl.constexpr,
    address_index_type: gl.constexpr,
):
    """Coordinates and TDM descriptors of one output tile."""
    pid_s, pid_m, pid_n, pid_k = compute_pids(
        tile_id, unpadded_m, grid_n, total_tiles, XCD_SWIZZLE, GROUP_M, 1
    )
    expt_id, _, _, start_m, _, off_m, off_k_x, _ = compute_offsets(
        pid_s,
        pid_m,
        pid_k,
        XBlockSchedule,
        XSliceOffs,
        XBlockOffs,
        X_SLICE_SIZES_DIVISIBILITY,
        None,
        None,
        1,
        "M",
        cfg.BLOCK_M,
        cfg.BLOCK_K,
        cfg.BLOCK_K // cfg.DIV_FACTOR_W,
        1,
    )
    eM = gl.multiple_of(gl.load(XSliceSizes + expt_id), X_SLICE_SIZES_DIVISIBILITY)

    expt_id, off_m = expt_id.to(address_index_type), off_m.to(address_index_type)
    start_m = start_m.to(address_index_type)
    pid_n = pid_n.to(address_index_type)

    x_desc, w_desc, w_scale_desc, gathered_m = create_descriptors(
        cfg,
        X + start_m * stride_x_m,
        W + expt_id * stride_w_e,
        WMxScale + expt_id * stride_w_mx_e,
        off_m,
        pid_n * cfg.BLOCK_N * stride_w_n,
        pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n,
        0,
        eM,
        (N - pid_n * cfg.BLOCK_N).to(gl.int32),
        K,
        stride_x_m,
        stride_x_k,
        stride_w_k,
        stride_w_n,
        stride_w_mx_n,
        stride_w_mx_k,
        None,
        start_m,
    )
    return expt_id, start_m, off_m, off_k_x, pid_n, eM, x_desc, w_desc, w_scale_desc


@gluon.jit
def _make_config(
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    X,
    W,
    NUM_BUFFERS: gl.constexpr,
    INDEX_TYPE: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    RESOLVE_PARTITION_CONFLICTS: gl.constexpr,
):
    gl.static_assert(BLOCK_N == 512)
    gl.static_assert(BLOCK_K == 256)
    # Two slots let K spans of two tiles (K <= 512) run, with each tile's load
    # overlapping only the previous tile's WMMAs.
    gl.static_assert(NUM_BUFFERS >= 2 and NUM_BUFFERS <= 4)
    # Four N tiles per warp give each lane scale rows n and n + 32, which sit
    # in adjacent bytes of the preshuffled scales: one ds_load_b64 per W
    # subtile instead of two ds_load_b32.
    return WarpPipelineConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        get_scaled_dot_format_string(X.dtype.element_ty),
        get_scaled_dot_format_string(W.dtype.element_ty),
        NUM_BUFFERS=NUM_BUFFERS,
        index_type=INDEX_TYPE,
        NUM_SUBTILES=(1, 2, 2),
        USE_GATHER=False,
        NUM_WARPS=8,
        RESOLVE_PARTITION_CONFLICTS=RESOLVE_PARTITION_CONFLICTS,
        CTA_M=gl.num_ctas(),
        W_CACHE_MODIFIER=W_CACHE_MODIFIER,
        WMMA_N_TILES_PER_WARP=4,
    )


# Token and routed-row counts, and the routing buffers whose alignment follows
# them, vary per batch; specializing on them would recompile while serving.
_PER_BATCH_ARGS = [
    "num_rows",
    "writeback_size",
    "WriteBackIndx",
    "XSliceSizes",
    "XSliceOffs",
    "XBlockOffs",
    "XBlockSchedule",
]


@gluon.jit(
    launch_metadata=warp_pipeline_launch_metadata, do_not_specialize=_PER_BATCH_ARGS
)
def gluon_moe_down_warp_pipelined_gfx1250(
    Y,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XGlobalScale,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,
    N,
    K,
    num_rows,
    WriteBackIndx: tl.const,
    writeback_size,
    XSliceSizes: tl.const,
    XSliceOffs: tl.const,
    XBlockOffs: tl.const,
    XBlockSchedule: tl.const,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,
    grid_n,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    N_EXPTS_TOT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    INDEX_TYPE: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    RESOLVE_PARTITION_CONFLICTS: gl.constexpr,
):
    """One output tile per program; ``Y[dst] = X[r] @ W[e].T * x_scale + B[e]``."""
    gl.static_assert(ACTIVATION_REDUCTION_N == 1)
    cfg = _make_config(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        X,
        W,
        NUM_BUFFERS,
        INDEX_TYPE,
        W_CACHE_MODIFIER,
        RESOLVE_PARTITION_CONFLICTS,
    )
    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    pid = gl.program_id(0)
    unpadded_m = gl.load(XBlockOffs + N_EXPTS_TOT)
    gl.assume(unpadded_m >= 0)
    total_tiles = unpadded_m * grid_n
    if pid >= total_tiles:
        return

    expt_id, start_m, off_m, off_k_x, pid_n, eM, x_desc, w_desc, w_scale_desc = (
        _tile_setup(
            cfg,
            pid,
            unpadded_m,
            total_tiles,
            grid_n,
            X,
            stride_x_m,
            stride_x_k,
            W,
            stride_w_e,
            stride_w_k,
            stride_w_n,
            WMxScale,
            stride_w_mx_e,
            stride_w_mx_k,
            stride_w_mx_n,
            N,
            K,
            XSliceSizes,
            XSliceOffs,
            XBlockOffs,
            XBlockSchedule,
            X_SLICE_SIZES_DIVISIBILITY,
            XCD_SWIZZLE,
            GROUP_M,
            address_index_type,
        )
    )
    # Allocate the ring per tile so its live range ends with the K loop and
    # the output staging buffer below can reuse that LDS.
    x_buffer, w_buffer, w_scale_buffer = allocate_ring(
        cfg, X.dtype.element_ty, W.dtype.element_ty
    )
    pgm = RingProgram(
        cfg,
        x_buffer,
        w_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        w_scale_desc,
        gl.constexpr(0),
        off_k_x,
    )
    _tdm_prefetch(pgm)
    # Issue the epilogue's scalar loads before the K loop so their latency
    # hides under it.
    x_scale = load_scale(XGlobalScale)

    loop_k = K - off_k_x
    c0, c1 = _k_loop_steady(pgm, loop_k)
    bias = load_bias(cfg, B, stride_b_e, expt_id, pid_n, N, 0, BLOCK_N)
    c0, c1 = _k_loop_tail(pgm, loop_k, c0, c1)

    out = epilogue_math(
        _merge_accumulator(cfg, c0, c1),
        bias,
        x_scale,
        1.0,
        Y.dtype.element_ty,
        False,
        0.0,
        0.0,
        0.0,
        False,
        0.0,
        0.0,
    )
    OUTPUT_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_M, BLOCK_N], [1, 0], cfg.ACC_CGA_LAYOUT
    )
    out_smem = gl.allocate_shared_memory(
        Y.dtype.element_ty, (BLOCK_M, BLOCK_N), OUTPUT_SHARED_LAYOUT
    )
    out_smem.store(out)
    col_offset = (BLOCK_N * pid_n).to(gl.int32)
    if WriteBackIndx is not None:
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
            0, get_tdm_gather_scatter_idx_layout(BLOCK_M, cfg.NUM_WARPS, cfg.CTA_M)
        )
        idx_offs = gl.arange(0, BLOCK_M, IDX_LAYOUT)
        idx_mask = (off_m + idx_offs < eM) & (
            start_m + off_m + idx_offs < writeback_size
        )
        # Rows without a destination point one row past the output, which the
        # clamped descriptor drops.
        dst_rows = gl.load(
            WriteBackIndx + start_m + off_m + idx_offs,
            mask=idx_mask,
            other=writeback_size,
        )
        dst_rows = gl.where(dst_rows == -1, writeback_size, dst_rows).to(cfg.index_type)
        y_desc = tdm.make_tensor_descriptor(
            base=Y,
            shape=(writeback_size, N),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=OUTPUT_SHARED_LAYOUT,
        )
        y_desc = tdm.update_tensor_descriptor(
            y_desc, add_offsets=[0, col_offset], clamp_bounds=True
        )
        tdm.async_scatter(y_desc, dst_rows, out_smem)
    else:
        y_desc = tdm.make_tensor_descriptor(
            base=Y + start_m * stride_y_m,
            shape=(eM, N),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=OUTPUT_SHARED_LAYOUT,
        )
        tdm.async_store(y_desc, [off_m.to(gl.int32), col_offset], out_smem)
    tdm.async_wait(0)


@gluon.jit(
    launch_metadata=warp_pipeline_launch_metadata, do_not_specialize=_PER_BATCH_ARGS
)
def gluon_moe_down_warp_pipelined_persistent_gfx1250(
    Y,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XGlobalScale,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,
    N,
    K,
    num_rows,
    WriteBackIndx: tl.const,
    writeback_size,
    XSliceSizes: tl.const,
    XSliceOffs: tl.const,
    XBlockOffs: tl.const,
    XBlockSchedule: tl.const,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,
    grid_n,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    N_EXPTS_TOT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    INDEX_TYPE: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
    RESOLVE_PARTITION_CONFLICTS: gl.constexpr,
):
    """Persistent variant: the ring stays allocated across tiles.

    Once a tile's K loop has drained, the next tile's leading K tiles are
    issued before the epilogue runs, so their latency hides under it. The
    epilogue stores straight from the accumulator layout, which needs no LDS
    beyond the ring.
    """
    gl.static_assert(ACTIVATION_REDUCTION_N == 1)
    cfg = _make_config(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        X,
        W,
        NUM_BUFFERS,
        INDEX_TYPE,
        W_CACHE_MODIFIER,
        RESOLVE_PARTITION_CONFLICTS,
    )
    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    # program_id counts clusters while num_programs counts workgroups.
    pid = gl.program_id(0)
    nprog = gl.num_programs(0) // cfg.CTA_M
    unpadded_m = gl.load(XBlockOffs + N_EXPTS_TOT)
    gl.assume(unpadded_m >= 0)
    total_tiles = unpadded_m * grid_n

    x_buffer, w_buffer, w_scale_buffer = allocate_ring(
        cfg, X.dtype.element_ty, W.dtype.element_ty
    )
    ROW_LAYOUT: gl.constexpr = gl.SliceLayout(1, cfg.acc_layout)
    COL_LAYOUT: gl.constexpr = gl.SliceLayout(0, cfg.acc_layout)
    if pid < total_tiles:
        expt_id, start_m, off_m, off_k_x, pid_n, eM, x_desc, w_desc, w_scale_desc = (
            _tile_setup(
                cfg,
                pid,
                unpadded_m,
                total_tiles,
                grid_n,
                X,
                stride_x_m,
                stride_x_k,
                W,
                stride_w_e,
                stride_w_k,
                stride_w_n,
                WMxScale,
                stride_w_mx_e,
                stride_w_mx_k,
                stride_w_mx_n,
                N,
                K,
                XSliceSizes,
                XSliceOffs,
                XBlockOffs,
                XBlockSchedule,
                X_SLICE_SIZES_DIVISIBILITY,
                XCD_SWIZZLE,
                GROUP_M,
                address_index_type,
            )
        )
        pgm = RingProgram(
            cfg,
            x_buffer,
            w_buffer,
            w_scale_buffer,
            x_desc,
            w_desc,
            w_scale_desc,
            gl.constexpr(0),
            off_k_x,
        )
        _tdm_prefetch(pgm)
        # Kernel-wide scales are loaded once; a per-tile scalar load would
        # make the compiler drain the previous tile's stores.
        x_scale = load_scale(XGlobalScale)

        for tile in range(pid, total_tiles, nprog):
            loop_k = K - off_k_x
            c0, c1 = _k_loop_steady(pgm, loop_k)
            bias = load_bias(cfg, B, stride_b_e, expt_id, pid_n, N, 0, BLOCK_N)
            rows = gl.arange(0, BLOCK_M, ROW_LAYOUT)
            row_mask = (off_m + rows < eM) & (start_m + off_m + rows < writeback_size)
            if WriteBackIndx is not None:
                dst_rows = gl.load(
                    WriteBackIndx + start_m + off_m + rows, mask=row_mask, other=-1
                )
                row_mask = dst_rows != -1
            else:
                dst_rows = start_m + off_m + rows
            # The warp-pipeline pass would merge drain stages placed directly
            # in the tile loop into one pipelined body; a one-trip loop with a
            # runtime bound keeps them a flat drain.
            for _ in range(gl.minimum(gl.cdiv(loop_k, BLOCK_K), 1)):
                c0, c1 = _k_loop_tail(pgm, loop_k, c0, c1)

            next_tile = tile + nprog
            (
                n_expt_id,
                n_start_m,
                n_off_m,
                n_off_k_x,
                n_pid_n,
                n_eM,
                n_x_desc,
                n_w_desc,
                n_w_scale_desc,
            ) = _tile_setup(
                cfg,
                gl.minimum(next_tile, total_tiles - 1),
                unpadded_m,
                total_tiles,
                grid_n,
                X,
                stride_x_m,
                stride_x_k,
                W,
                stride_w_e,
                stride_w_k,
                stride_w_n,
                WMxScale,
                stride_w_mx_e,
                stride_w_mx_k,
                stride_w_mx_n,
                N,
                K,
                XSliceSizes,
                XSliceOffs,
                XBlockOffs,
                XBlockSchedule,
                X_SLICE_SIZES_DIVISIBILITY,
                XCD_SWIZZLE,
                GROUP_M,
                address_index_type,
            )
            next_pgm = RingProgram(
                cfg,
                x_buffer,
                w_buffer,
                w_scale_buffer,
                n_x_desc,
                n_w_desc,
                n_w_scale_desc,
                gl.constexpr(0),
                n_off_k_x,
            )
            # Predicated rather than branched: TDM ops under scf.if make the
            # wait-count analysis walk every path combination across the loop.
            _tdm_prefetch(next_pgm, (next_tile < total_tiles).to(gl.int32))
            # Keep the epilogue math below the issue; LLVM otherwise sinks the
            # TDM ops past it and the loads no longer overlap the epilogue.
            gl.amd.hint.sched_barrier()

            out = epilogue_math(
                _merge_accumulator(cfg, c0, c1),
                bias,
                x_scale,
                1.0,
                Y.dtype.element_ty,
                False,
                0.0,
                0.0,
                0.0,
                False,
                0.0,
                0.0,
            )
            cols = (BLOCK_N * pid_n).to(address_index_type) + gl.arange(
                0, BLOCK_N, COL_LAYOUT
            )
            y_ptrs = (
                Y
                + dst_rows.to(address_index_type)[:, None] * stride_y_m
                + cols[None, :] * stride_y_n
            )
            gl.store(y_ptrs, out, mask=row_mask[:, None] & (cols < N)[None, :])

            pgm = next_pgm
            expt_id, start_m, off_m, off_k_x, pid_n, eM = (
                n_expt_id,
                n_start_m,
                n_off_m,
                n_off_k_x,
                n_pid_n,
                n_eM,
            )
        # The last tile's predicated-off prefetch is still counted.
        tdm.async_wait(0)
    x_buffer._keep_alive()
    w_buffer._keep_alive()
    w_scale_buffer._keep_alive()
