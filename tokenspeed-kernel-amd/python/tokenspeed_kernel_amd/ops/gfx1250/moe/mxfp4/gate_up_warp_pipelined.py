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

"""Persistent warp-pipelined gate/up GEMM of an FP8 x MXFP4 MoE layer.

Each program (or cluster of ``CTA_M`` workgroups sharing W by multicast) walks
output tiles ``pid, pid + nprog, ...`` with a ``NUM_BUFFERS``-slot LDS ring
that stays allocated across tiles. A tile is ``BLOCK_M`` gathered rows by 512
interleaved gate/up columns; each 256-wide K tile is consumed as two 128-wide
halves. The eight warps form two groups of four that run one warp-pipeline
stage apart, so one group reads LDS while the other issues WMMAs. The epilogue
fuses the gate activation and the FP8 cast of the down GEMM's input, stages
the 256-column output in a ring slot and stores it with TDM, while the next
tile's first K tile is already loading.
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
    load_bias,
    load_scale,
    load_w_scale_slice,
    warp_pipeline_launch_metadata,
    wmma_scaled,
)

tdm = gl.amd.cdna5.tdm

# Gather rows addressable by 16-bit indices, leaving room for the inactive-lane
# sentinel one row past the end.
MAX_16BIT_GATHER_ROWS = gl.constexpr((1 << 16) - 2)
# W takes warps 0-1 and its scales warps 2-3 of the fused TDM copy.
_TDM_HINT_W = gl.constexpr(0b00000011)
_TDM_HINT_W_SCALE = gl.constexpr(0b00001100)


@gluon.jit
def _gather_row_span(cfg, GatherIndx, off_m, eM):
    """Return the lowest gathered row of a cluster tile and the rows it spans.

    Routing lists each expert's rows in ascending token order, so the tile's
    first and last valid indices bound it. Every workgroup of the cluster
    reads the same two scalars, so they agree on the index width.
    """
    lo = gl.load(GatherIndx + off_m)
    hi = gl.load(GatherIndx + gl.minimum(off_m + cfg.BLOCK_M, eM) - 1)
    return lo, hi - lo + 1


@gluon.jit
def _with_16bit_gather(pgm):
    return RingProgram(
        pgm.cfg,
        pgm.x_buffer,
        pgm.w_buffer,
        pgm.w_scale_buffer,
        pgm.x_desc,
        pgm.w_desc,
        pgm.w_scale_desc,
        pgm.gathered_m.to(gl.int16),
        pgm.off_k_x,
    )


@gluon.jit
def _issue_global_loads(pgm, load_idx, slot, pred=1):
    """Issue K tile ``load_idx`` into ring slot ``slot``.

    Every warp executes every TDM instruction (the warp mask only selects
    which warps move data), but only warps 0-3, the leading warp-pipeline
    group, may move data into the ring: a wait covers the executing warp's
    own copies, and warps 4-7 reach their wait one stage after warps 0-3
    start reading. A copy into the partitioned W buffer needs a multiple of
    its two pieces as warps, so W takes warps 0-1 and still fuses with the
    scales.
    """
    cfg = pgm.cfg
    block_k_packed_w: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
    x_desc = tdm.update_tensor_descriptor(
        pgm.x_desc,
        add_offsets=[0, pgm.off_k_x + load_idx * cfg.BLOCK_K],
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
    tdm.async_gather(x_desc, pgm.gathered_m, pgm.x_buffer.index(slot))
    tdm.async_load_fused(
        [
            (w_desc, pgm.w_buffer.index(slot), _TDM_HINT_W),
            (w_scale_desc, pgm.w_scale_buffer.index(slot), _TDM_HINT_W_SCALE),
        ],
        cache_modifier=cfg.W_CACHE_MODIFIER,
    )


@gluon.jit
def _issue_first_global_loads(pgm, use_16bit_gather, pred=1):
    """Issue K tile 0 into slot 0 with the tile's gather index width."""
    if use_16bit_gather:
        _issue_global_loads(_with_16bit_gather(pgm), 0, 0, pred)
    else:
        _issue_global_loads(pgm, 0, 0, pred)


@gluon.jit
def _wait_global_loads(num_tiles: gl.constexpr):
    # Two TDM ops per K tile: the X gather and the fused W/W-scale copy.
    tdm.async_wait(num_tiles * 2)


@gluon.jit
def _load_k_half(pgm, slot, k_idx: gl.constexpr):
    """Load one 128-wide K half of a ring slot: X and the W/scale N subtiles.

    Each N subtile's scales are read ahead of its W values, so the first WMMA
    waits for X, the scales and its first W registers instead of every W
    read. With one N subtile (partitioned W) the single 512-row W slice is
    returned as both halves.
    """
    cfg = pgm.cfg
    sub_k: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES_K
    sub_n: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES_N
    x = (
        pgm.x_buffer.index(slot)
        .slice(k_idx * sub_k, sub_k, 1)
        .load(layout=cfg.dot_layout_x)
    )
    w_slot = pgm.w_buffer.index(slot)
    w_scale_slot = pgm.w_scale_buffer.index(slot)
    scale_w0 = load_w_scale_slice(cfg, w_scale_slot, k_idx * sub_k, sub_k, 0, sub_n)
    w0 = (
        w_slot.slice(0, sub_n, 0)
        .slice(k_idx * sub_k // cfg.DIV_FACTOR_W, sub_k // cfg.DIV_FACTOR_W, 1)
        .permute([1, 0])
        .load(layout=cfg.dot_layout_w)
    )
    if cfg.NUM_SUBTILES_N == 1:
        result = (x, w0, scale_w0, w0, scale_w0)
    else:
        scale_w1 = load_w_scale_slice(
            cfg, w_scale_slot, k_idx * sub_k, sub_k, sub_n, sub_n
        )
        w1 = (
            w_slot.slice(sub_n, sub_n, 0)
            .slice(k_idx * sub_k // cfg.DIV_FACTOR_W, sub_k // cfg.DIV_FACTOR_W, 1)
            .permute([1, 0])
            .load(layout=cfg.dot_layout_w)
        )
        result = (x, w0, scale_w0, w1, scale_w1)
    return result


@gluon.jit
def _wmma_k_half(cfg, x, w0, scale_w0, w1, scale_w1, c0, c1):
    """The WMMAs of one K half. With one N subtile, c0 is the whole tile and
    c1 passes through."""
    c0 = wmma_scaled(cfg, x, w0, scale_w0, c0)
    if cfg.NUM_SUBTILES_N == 2:
        c1 = wmma_scaled(cfg, x, w1, scale_w1, c1)
    return c0, c1


@gluon.jit
def _consume_and_refill(
    pgm, slot: gl.constexpr, refill_slot: gl.constexpr, refill_idx, c0, c1
):
    """Refill ``refill_slot`` with K tile ``refill_idx`` and consume ``slot``.

    Four stages: refill/load low -> compute low -> load high/arrive -> compute
    high/cluster wait. ``refill_slot`` is the slot the previous step consumed.
    """
    cfg = pgm.cfg
    _wait_global_loads(cfg.NUM_BUFFERS - 2)
    with gl.amd.warp_pipeline_stage("gate_up_refill_load_low"):
        _issue_global_loads(pgm, refill_idx, refill_slot)
        x0, w00, scale_w00, w01, scale_w01 = _load_k_half(pgm, slot, 0)
    with gl.amd.warp_pipeline_stage("gate_up_compute_low"):
        c0, c1 = _wmma_k_half(cfg, x0, w00, scale_w00, w01, scale_w01, c0, c1)
    with gl.amd.warp_pipeline_stage("gate_up_load_high_arrive"):
        x1, w10, scale_w10, w11, scale_w11 = _load_k_half(pgm, slot, 1)
        if cfg.CTA_M > 1:
            gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("gate_up_compute_high_wait"):
        c0, c1 = _wmma_k_half(cfg, x1, w10, scale_w10, w11, scale_w11, c0, c1)
        if cfg.CTA_M > 1:
            gl.amd.cdna5.cluster.wait()
    return c0, c1


@gluon.jit
def _consume_tail(pgm, tile_idx, c0, c1):
    """Consume one resident K tile without refilling its slot."""
    cfg = pgm.cfg
    slot = tile_idx % cfg.NUM_BUFFERS
    for k_idx in gl.static_range(cfg.NUM_SUBTILES_K):
        with gl.amd.warp_pipeline_stage("gate_up_tail_load"):
            x, w0, scale_w0, w1, scale_w1 = _load_k_half(pgm, slot, k_idx)
        with gl.amd.warp_pipeline_stage("gate_up_tail_compute"):
            c0, c1 = _wmma_k_half(cfg, x, w0, scale_w0, w1, scale_w1, c0, c1)
    return c0, c1


@gluon.jit
def _drain(pgm, iter_max, c0, c1):
    """Consume the last NUM_BUFFERS - 1 K tiles; the first is already resident."""
    cfg = pgm.cfg
    for drain_idx in gl.static_range(cfg.NUM_BUFFERS - 1):
        tile_idx = iter_max - (cfg.NUM_BUFFERS - 1) + drain_idx
        c0, c1 = _consume_tail(pgm, tile_idx, c0, c1)
        if drain_idx < cfg.NUM_BUFFERS - 2:
            _wait_global_loads(cfg.NUM_BUFFERS - 3 - drain_idx)
            pgm.cluster_sync()
    return c0, c1


@gluon.jit
def _merge_accumulator(cfg, c0, c1):
    if cfg.NUM_SUBTILES_N == 1:
        accumulator = c0
    else:
        # c0 holds output columns [0, BLOCK_N / 2), c1 the rest.
        accumulator = gl.join(c0, c1)
        accumulator = accumulator.permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        accumulator = gl.convert_layout(
            accumulator, cfg.acc_layout, assert_trivial=True
        )
    return accumulator


@gluon.jit
def _k_loop(pgm, loop_k):
    """K loop of one output tile.

    The caller already issued K tile 0 into slot 0 (during the previous tile's
    epilogue), and the previous tile's output store from ring slot
    NUM_BUFFERS - 1 may still be in flight behind it. TDM loads and stores
    retire in order on the shared tensor counter, so the wait for K tile 0
    also retires that store before a K tile overwrites the slot.
    """
    cfg = pgm.cfg
    # Slot 1 is free; its loads overlap the wait for K tile 0 and the previous
    # store, which are the older outstanding ops.
    _issue_global_loads(pgm, 1, 1)
    _wait_global_loads(1)
    pgm.cluster_sync()
    for prefetch_idx in gl.static_range(2, cfg.NUM_BUFFERS - 1):
        _issue_global_loads(pgm, prefetch_idx, prefetch_idx)

    iter_max = gl.cdiv(loop_k, cfg.BLOCK_K)
    num_steps = iter_max - cfg.NUM_BUFFERS + 1
    gl.assume(num_steps > 0)

    subtile_n: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES_N
    c0 = gl.zeros((cfg.BLOCK_M, subtile_n), dtype=gl.float32, layout=cfg.acc_layout)
    c1 = gl.zeros((cfg.BLOCK_M, subtile_n), dtype=gl.float32, layout=cfg.acc_layout)

    # Unrolled by the ring size so every slot index is a constant.
    n_full = num_steps // cfg.NUM_BUFFERS * cfg.NUM_BUFFERS
    for tile_base in range(0, n_full, cfg.NUM_BUFFERS):
        for slot in gl.static_range(cfg.NUM_BUFFERS):
            c0, c1 = _consume_and_refill(
                pgm,
                slot,
                (slot + cfg.NUM_BUFFERS - 1) % cfg.NUM_BUFFERS,
                tile_base + slot + cfg.NUM_BUFFERS - 1,
                c0,
                c1,
            )
    # Remainder steps start at a multiple of the ring size, so their slots are
    # constants too; one-trip loops keep stage borders in loops.
    for slot in gl.static_range(cfg.NUM_BUFFERS - 1):
        for _ in range(gl.minimum(gl.maximum(num_steps - n_full - slot, 0), 1)):
            c0, c1 = _consume_and_refill(
                pgm,
                slot,
                (slot + cfg.NUM_BUFFERS - 1) % cfg.NUM_BUFFERS,
                n_full + slot + cfg.NUM_BUFFERS - 1,
                c0,
                c1,
            )

    # The first drained K tile.
    _wait_global_loads(cfg.NUM_BUFFERS - 2)
    # The warp-pipeline pass would sweep drain stages placed directly in the
    # tile loop into one pipelined body; a one-trip loop with a runtime bound
    # keeps them in their own pipeline.
    for _ in range(gl.minimum(iter_max, 1)):
        c0, c1 = _drain(pgm, iter_max, c0, c1)

    # Peers may still be multicasting into this workgroup's ring; the epilogue
    # reuses that LDS.
    pgm.cluster_sync()
    return _merge_accumulator(cfg, c0, c1)


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
    M,
    N,
    K,
    GatherIndx,
    XSliceSizes,
    XSliceOffs,
    XBlockOffs,
    XBlockSchedule,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    GROUP_M: gl.constexpr,
    address_index_type: gl.constexpr,
    RELATIVE_GATHER: gl.constexpr,
):
    """Coordinates and TDM descriptors of one output tile.

    With RELATIVE_GATHER the gather indices are relative to the tile's lowest
    row, and ``use_16bit_gather`` tells whether they fit in 16 bits; otherwise
    it is constexpr False.
    """
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

    W_ptr = W + expt_id * stride_w_e
    w_offs = pid_n * cfg.BLOCK_N * stride_w_n
    WMxScale_ptr = WMxScale + expt_id * stride_w_mx_e
    w_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n
    rem_n = (N - pid_n * cfg.BLOCK_N).to(gl.int32)

    if RELATIVE_GATHER:
        # X has too many rows for 16-bit gather indices. Index relative to the
        # tile's lowest row instead, which halves the TDM gather instructions
        # whenever the tile spans few enough rows.
        gather_base, gather_rows = _gather_row_span(
            cfg, GatherIndx + start_m, off_m, eM
        )
        use_16bit_gather = gather_rows <= MAX_16BIT_GATHER_ROWS
    else:
        gather_base = None
        gather_rows = None
        use_16bit_gather = gl.constexpr(False)

    x_desc, w_desc, w_scale_desc, gathered_m = create_descriptors(
        cfg,
        X,
        W_ptr,
        WMxScale_ptr,
        off_m,
        w_offs,
        w_scale_offs,
        M,
        eM,
        rem_n,
        K,
        stride_x_m,
        stride_x_k,
        stride_w_k,
        stride_w_n,
        stride_w_mx_n,
        stride_w_mx_k,
        GatherIndx,
        start_m,
        gather_base,
        gather_rows,
    )
    return (
        expt_id,
        start_m,
        off_m,
        off_k_x,
        pid_n,
        eM,
        x_desc,
        w_desc,
        w_scale_desc,
        gathered_m,
        use_16bit_gather,
    )


# Token and routed-row counts, and the routing buffers whose alignment follows
# them, vary per batch; specializing on them would recompile while serving.
_PER_BATCH_ARGS = [
    "M",
    "num_rows",
    "GatherIndx",
    "XSliceSizes",
    "XSliceOffs",
    "XBlockOffs",
    "XBlockSchedule",
]


@gluon.jit(
    launch_metadata=warp_pipeline_launch_metadata, do_not_specialize=_PER_BATCH_ARGS
)
def gluon_moe_gate_up_warp_pipelined_gfx1250(
    Y,
    stride_y_m,
    stride_y_n,
    YGlobalScale,
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
    M,
    N,
    K,
    num_rows,
    GatherIndx: tl.const,
    XSliceSizes: tl.const,
    XSliceOffs: tl.const,
    XBlockOffs: tl.const,
    XBlockSchedule: tl.const,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,
    grid_n,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
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
    RELATIVE_GATHER: gl.constexpr,
):
    """Persistent gate/up kernel; each tile's epilogue overlaps the next
    tile's prologue.

    ``Y[r, :]`` is the FP8 activation output of gathered row ``r`` (expert
    sorted), ``act(X[GatherIndx[r]] @ W[e].T * x_scale + B[e]) / y_scale``.
    Once a tile's K loop has drained, the next tile's K tile 0 is issued into
    slot 0 before the epilogue runs, so its latency hides under it. The
    output of a tile is staged in ring slot NUM_BUFFERS - 1 and its TDM store
    is left in flight across the next tile's setup; see ``_k_loop`` for how
    that store is retired.
    """
    CTA_M: gl.constexpr = gl.num_ctas()
    gl.static_assert(BLOCK_N == 512)
    gl.static_assert(BLOCK_K == 256)
    gl.static_assert(NUM_BUFFERS == 3 or NUM_BUFFERS == 4)
    gl.static_assert(ACTIVATION_REDUCTION_N == 2)
    gl.static_assert(DO_SWIGLU or DO_SITU)
    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)
    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    unpadded_m = gl.load(XBlockOffs + N_EXPTS_TOT)
    gl.assume(unpadded_m >= 0)
    total_tiles = unpadded_m * grid_n

    # The gathered X and W are each split over the two LDS partitions, and
    # each K half is one 128 x 512 dot.
    NUM_SUBTILES: gl.constexpr = (1, 1, 2) if RESOLVE_PARTITION_CONFLICTS else (1, 2, 2)
    cfg = WarpPipelineConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        NUM_BUFFERS=NUM_BUFFERS,
        index_type=INDEX_TYPE,
        NUM_SUBTILES=NUM_SUBTILES,
        USE_GATHER=True,
        NUM_WARPS=8,
        RESOLVE_PARTITION_CONFLICTS=RESOLVE_PARTITION_CONFLICTS,
        GATHER_PRODUCER_WARPS=4,
        CTA_M=CTA_M,
        W_CACHE_MODIFIER=W_CACHE_MODIFIER,
    )
    # The output tile is staged in an X ring slot, in X's (possibly
    # partitioned) layout, so Y must match X's tile shape and element type.
    gl.static_assert(
        OUT_BLOCK_N == BLOCK_K and Y.dtype.element_ty == X.dtype.element_ty
    )
    if pid >= total_tiles:
        return

    # program_id counts clusters while num_programs counts workgroups.
    nprog = gl.num_programs(0) // CTA_M
    x_buffer, w_buffer, w_scale_buffer = allocate_ring(
        cfg, X.dtype.element_ty, W.dtype.element_ty
    )
    out_smem = x_buffer.index(NUM_BUFFERS - 1)
    x_scale = load_scale(XGlobalScale)
    y_scale_inv = 1.0 / load_scale(YGlobalScale)

    # The coordinates and descriptors of the tile about to run are carried
    # across iterations; K tile 0 of each tile is issued before the epilogue
    # of the tile before it.
    (
        expt_id,
        start_m,
        off_m,
        off_k_x,
        pid_n,
        eM,
        x_desc,
        w_desc,
        w_scale_desc,
        gathered_m,
        use_16bit_gather,
    ) = _tile_setup(
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
        M,
        N,
        K,
        GatherIndx,
        XSliceSizes,
        XSliceOffs,
        XBlockOffs,
        XBlockSchedule,
        X_SLICE_SIZES_DIVISIBILITY,
        XCD_SWIZZLE,
        GROUP_M,
        address_index_type,
        RELATIVE_GATHER,
    )
    pgm = RingProgram(
        cfg,
        x_buffer,
        w_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        w_scale_desc,
        gathered_m,
        off_k_x,
    )
    _issue_first_global_loads(pgm, use_16bit_gather)

    for tile in range(pid, total_tiles, nprog):
        # Both index widths issue the same TDM ops, so the wait counts do not
        # depend on the branch.
        if use_16bit_gather:
            acc = _k_loop(_with_16bit_gather(pgm), K - off_k_x)
        else:
            acc = _k_loop(pgm, K - off_k_x)
        bias = load_bias(cfg, B, stride_b_e, expt_id, pid_n, N, 0, BLOCK_N)

        # Issue the next tile's K tile 0 into the drained slot 0 so its
        # latency hides under this tile's epilogue.
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
            n_gathered_m,
            n_use_16bit_gather,
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
            M,
            N,
            K,
            GatherIndx,
            XSliceSizes,
            XSliceOffs,
            XBlockOffs,
            XBlockSchedule,
            X_SLICE_SIZES_DIVISIBILITY,
            XCD_SWIZZLE,
            GROUP_M,
            address_index_type,
            RELATIVE_GATHER,
        )
        next_pgm = RingProgram(
            cfg,
            x_buffer,
            w_buffer,
            w_scale_buffer,
            n_x_desc,
            n_w_desc,
            n_w_scale_desc,
            n_gathered_m,
            n_off_k_x,
        )
        # Predicated rather than branched: TDM ops under scf.if make the
        # wait-count analysis walk every path combination across the tile loop.
        has_next = (next_tile < total_tiles).to(gl.int32)
        _issue_first_global_loads(next_pgm, n_use_16bit_gather, has_next)
        # Keep the epilogue math below the issue; LLVM otherwise sinks the TDM
        # ops past it and the load no longer overlaps the epilogue.
        gl.amd.hint.sched_barrier()

        out = epilogue_math(
            acc,
            bias,
            x_scale,
            y_scale_inv,
            Y.dtype.element_ty,
            DO_SWIGLU,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            DO_SITU,
            SITU_BETA,
            SITU_LINEAR_BETA,
        )
        # Stage straight from the accumulator layout: a layout conversion
        # would need scratch LDS, and the ring stays allocated here.
        out_smem.store(out)
        y_desc = tdm.make_tensor_descriptor(
            base=Y + start_m * stride_y_m,
            shape=(eM, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=cfg.shared_layout_x,
        )
        tdm.async_store(
            y_desc,
            [off_m.to(gl.int32), (OUT_BLOCK_N * pid_n).to(gl.int32)],
            out_smem,
        )

        pgm = next_pgm
        use_16bit_gather = n_use_16bit_gather
        expt_id, start_m, off_m, off_k_x, pid_n, eM = (
            n_expt_id,
            n_start_m,
            n_off_m,
            n_off_k_x,
            n_pid_n,
            n_eM,
        )
    tdm.async_wait(0)
    x_buffer._keep_alive()
    w_buffer._keep_alive()
    w_scale_buffer._keep_alive()
