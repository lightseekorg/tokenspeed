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

"""Persistent small-M (decode) FP8 x MXFP4 MoE GEMM for both projections.

With ``GatherIndx`` the kernel runs the gate/up GEMM: it gathers token rows
and stores the activation output as expert-sorted rows. Without it, it runs
the down GEMM over expert-sorted rows and writes each row to the route's
output row ``WriteBackIndx[r]``.

Each program walks tiles ``pid, pid + nprog, ...`` with a NUM_BUFFERS-slot
LDS ring allocated once. Within a tile, step t consumes K tile t and refills
the slot of step t - 1 with K tile t + NUM_BUFFERS - 1; with WARP_PIPELINE,
warps 4-7 run one stage behind warps 0-3, so one group reads LDS while the
other issues WMMAs. Once a tile's K loop has drained, the next tile's first
NUM_BUFFERS - 1 K tiles are issued before the epilogue, so their latency hides
under it. Outputs are stored from registers, which keeps the tensor counter
to ring loads only; with ASYNC_STORE (down GEMM) the output tile is staged in
LDS and copied out asynchronously, and the copies drain under the next tile's
K loop.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, tl
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
    get_blocked_layout,
    load_bias,
    load_scale,
    load_w_scale_slice,
    warp_pipeline_launch_metadata,
    wmma_scaled,
)

tdm = gl.amd.cdna5.tdm
async_copy = gl.amd.cdna5.async_copy

ALL_WARPS = 0xFF


@aggregate
class DecodeSched:
    """Warp-pipeline switch and the TDM producer warps of one K tile.

    The down GEMM copies X, the W scales and W (masks in that order); the
    gate/up GEMM gathers X and copies the W scales and W (masks in that order,
    then optionally the number of gather warps, 4 or 8). ALL_WARPS spreads a
    copy over every warp. Copies with disjoint masks share one fused TDM op:
    all of them when possible, else (down) X with the scales while W gets its
    own op; the rest are separate ops. With the warp pipeline every producer
    must be in the leading group (warps 0-3): a TDM wait covers only the
    executing warp's own copies, and warps 4-7 reach their wait one stage
    after warps 0-3 start reading a slot.
    """

    WARP_PIPELINE: gl.constexpr
    X_MASK: gl.constexpr
    W_SCALE_MASK: gl.constexpr
    W_MASK: gl.constexpr
    # "all": one fused op; "x_scale": X and the scales fused, W alone;
    # "none": one op per copy.
    FUSION: gl.constexpr
    GATHER_WARPS: gl.constexpr
    OPS_PER_K_TILE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, WARP_PIPELINE, USE_GATHER, TDM_MASKS):
        def disjoint(*masks):
            return ALL_WARPS not in masks and all(
                a & b == 0 for i, a in enumerate(masks) for b in masks[i + 1 :]
            )

        gather_warps = 0
        if USE_GATHER:
            # W scales on warps 2-3 and W on warps 0-1.
            masks = tuple(TDM_MASKS or (0b00001100, 0b00000011))
            gather_warps = masks[2] if len(masks) > 2 else 4
            x_mask = None
            w_scale_mask, w_mask = masks[:2]
            fusion = "all" if disjoint(w_scale_mask, w_mask) else "none"
            ops = 1 + (1 if fusion == "all" else 2)
            producers = [w_scale_mask, w_mask, (1 << gather_warps) - 1]
        else:
            # X on warp 0, the scales on warp 1, W on warps 2-3.
            x_mask, w_scale_mask, w_mask = TDM_MASKS or (
                0b00000001,
                0b00000010,
                0b00001100,
            )
            if disjoint(x_mask, w_scale_mask, w_mask):
                fusion, ops = "all", 1
            elif disjoint(x_mask, w_scale_mask):
                fusion, ops = "x_scale", 2
            else:
                fusion, ops = "none", 3
            producers = [x_mask, w_scale_mask, w_mask]
        assert gather_warps in (0, 4, 8)
        if WARP_PIPELINE:
            assert all(m & ~0x0F == 0 for m in producers), "producers must be warps 0-3"
        self.WARP_PIPELINE = gl.constexpr(WARP_PIPELINE)
        self.X_MASK = gl.constexpr(None if x_mask == ALL_WARPS else x_mask)
        self.W_SCALE_MASK = gl.constexpr(
            None if w_scale_mask == ALL_WARPS else w_scale_mask
        )
        self.W_MASK = gl.constexpr(None if w_mask == ALL_WARPS else w_mask)
        self.FUSION = gl.constexpr(fusion)
        self.GATHER_WARPS = gl.constexpr(gather_warps)
        self.OPS_PER_K_TILE = gl.constexpr(ops)


@gluon.jit
def _issue_k_tile(pgm, sched, load_idx, slot, pred=1):
    """Issue K tile ``load_idx`` into ring slot ``slot``; ``pred`` = 0 issues
    ops that move no data."""
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
    x_dest = pgm.x_buffer.index(slot)
    w_dest = pgm.w_buffer.index(slot)
    w_scale_dest = pgm.w_scale_buffer.index(slot)
    if cfg.USE_GATHER:
        tdm.async_gather(x_desc, pgm.gathered_m, x_dest)
        if sched.FUSION == "all":
            tdm.async_load_fused(
                [
                    (w_scale_desc, w_scale_dest, sched.W_SCALE_MASK),
                    (w_desc, w_dest, sched.W_MASK),
                ],
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
        else:
            tdm.async_load(
                w_scale_desc,
                dest=w_scale_dest,
                warp_used_hint=sched.W_SCALE_MASK,
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
            tdm.async_load(
                w_desc,
                dest=w_dest,
                warp_used_hint=sched.W_MASK,
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
    else:
        if sched.FUSION == "all":
            tdm.async_load_fused(
                [
                    (x_desc, x_dest, sched.X_MASK),
                    (w_scale_desc, w_scale_dest, sched.W_SCALE_MASK),
                    (w_desc, w_dest, sched.W_MASK),
                ],
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
        elif sched.FUSION == "x_scale":
            tdm.async_load_fused(
                [
                    (x_desc, x_dest, sched.X_MASK),
                    (w_scale_desc, w_scale_dest, sched.W_SCALE_MASK),
                ],
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
            tdm.async_load(
                w_desc,
                dest=w_dest,
                warp_used_hint=sched.W_MASK,
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
        else:
            tdm.async_load(x_desc, dest=x_dest, warp_used_hint=sched.X_MASK)
            tdm.async_load(
                w_scale_desc,
                dest=w_scale_dest,
                warp_used_hint=sched.W_SCALE_MASK,
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )
            tdm.async_load(
                w_desc,
                dest=w_dest,
                warp_used_hint=sched.W_MASK,
                cache_modifier=cfg.W_CACHE_MODIFIER,
            )


@gluon.jit
def _wait_k_tiles(sched, num_tiles: gl.constexpr):
    """Wait until at most ``num_tiles`` K tiles are in flight."""
    tdm.async_wait(num_tiles * sched.OPS_PER_K_TILE)


@gluon.jit
def _tdm_prefetch(pgm, sched, pred=1):
    """Issue a tile's leading K tiles 0 .. NUM_BUFFERS - 2 into the slots of
    the same index."""
    for slot in gl.static_range(pgm.cfg.NUM_BUFFERS - 1):
        _issue_k_tile(pgm, sched, slot, slot, pred)


@gluon.jit
def _load_k_sub(pgm, slot, k_idx: gl.constexpr):
    """Read one K subtile of a ring slot: X, the W scales, then W."""
    cfg = pgm.cfg
    sub_k: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES_K
    x = (
        pgm.x_buffer.index(slot)
        .slice(k_idx * sub_k, sub_k, 1)
        .load(layout=cfg.dot_layout_x)
    )
    w_scale = load_w_scale_slice(
        cfg, pgm.w_scale_buffer.index(slot), k_idx * sub_k, sub_k, 0, cfg.BLOCK_N
    )
    w = (
        pgm.w_buffer.index(slot)
        .slice(k_idx * sub_k // cfg.DIV_FACTOR_W, sub_k // cfg.DIV_FACTOR_W, 1)
        .permute([1, 0])
        .load(layout=cfg.dot_layout_w)
    )
    return x, w, w_scale


@gluon.jit
def _consume_and_refill(
    pgm, slot: gl.constexpr, refill_slot: gl.constexpr, refill_idx, acc, sched
):
    """Refill ``refill_slot`` with K tile ``refill_idx`` and consume ``slot``.

    Two stages per K subtile, load -> compute; the refill joins the first
    load. ``refill_slot`` is the slot the previous step consumed.
    """
    cfg = pgm.cfg
    _wait_k_tiles(sched, cfg.NUM_BUFFERS - 2)
    for k_idx in gl.static_range(cfg.NUM_SUBTILES_K):
        if sched.WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("decode_load"):
                if k_idx == 0:
                    _issue_k_tile(pgm, sched, refill_idx, refill_slot)
                x, w, w_scale = _load_k_sub(pgm, slot, k_idx)
            with gl.amd.warp_pipeline_stage("decode_compute"):
                acc = wmma_scaled(cfg, x, w, w_scale, acc)
        else:
            if k_idx == 0:
                _issue_k_tile(pgm, sched, refill_idx, refill_slot)
            x, w, w_scale = _load_k_sub(pgm, slot, k_idx)
            acc = wmma_scaled(cfg, x, w, w_scale, acc)
    return acc


@gluon.jit
def _consume_tail(pgm, slot, acc, sched):
    """Consume one resident K tile without refilling its slot."""
    cfg = pgm.cfg
    for k_idx in gl.static_range(cfg.NUM_SUBTILES_K):
        if sched.WARP_PIPELINE:
            with gl.amd.warp_pipeline_stage("decode_tail_load"):
                x, w, w_scale = _load_k_sub(pgm, slot, k_idx)
            with gl.amd.warp_pipeline_stage("decode_tail_compute"):
                acc = wmma_scaled(cfg, x, w, w_scale, acc)
        else:
            x, w, w_scale = _load_k_sub(pgm, slot, k_idx)
            acc = wmma_scaled(cfg, x, w, w_scale, acc)
    return acc


@gluon.jit
def _k_loop_bounds(cfg, loop_k):
    """Refilling steps of the steady loop and its whole ring-size rounds.

    Step t consumes K tile t and issues K tile t + NUM_BUFFERS - 1.
    """
    iter_max = gl.cdiv(loop_k, cfg.BLOCK_K)
    gl.assume(iter_max >= cfg.NUM_BUFFERS)
    num_steps = iter_max - cfg.NUM_BUFFERS + 1
    return num_steps, num_steps // cfg.NUM_BUFFERS


@gluon.jit
def _k_loop_steady(pgm, loop_k, sched):
    """The refilling K loop of one output tile; the caller already issued K
    tiles 0 .. NUM_BUFFERS - 2."""
    cfg = pgm.cfg
    num_buffers: gl.constexpr = cfg.NUM_BUFFERS
    num_steps, num_rounds = _k_loop_bounds(cfg, loop_k)
    acc = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)
    # Unroll by the ring size so every slot index is a compile-time constant.
    for round_idx in range(0, num_rounds):
        for slot in gl.static_range(num_buffers):
            acc = _consume_and_refill(
                pgm,
                slot,
                (slot + num_buffers - 1) % num_buffers,
                round_idx * num_buffers + slot + num_buffers - 1,
                acc,
                sched,
            )
    # Remainder steps start at a multiple of the ring size, so their slots are
    # constants too; one-trip loops keep stage borders in loops.
    n_full = num_rounds * num_buffers
    for slot in gl.static_range(num_buffers - 1):
        for _ in range(gl.minimum(gl.maximum(num_steps - n_full - slot, 0), 1)):
            acc = _consume_and_refill(
                pgm,
                slot,
                (slot + num_buffers - 1) % num_buffers,
                n_full + slot + num_buffers - 1,
                acc,
                sched,
            )
    # Retire the first drained K tile.
    _wait_k_tiles(sched, num_buffers - 2)
    return acc


@gluon.jit
def _k_loop_tail(pgm, loop_k, acc, sched):
    """Drain the last NUM_BUFFERS - 1 K tiles.

    Flat pipeline stages are illegal directly inside a loop; the tile loop
    wraps this call in a one-trip loop.
    """
    cfg = pgm.cfg
    num_buffers: gl.constexpr = cfg.NUM_BUFFERS
    num_steps, _ = _k_loop_bounds(cfg, loop_k)
    for drain_idx in gl.static_range(num_buffers - 1):
        acc = _consume_tail(pgm, (num_steps + drain_idx) % num_buffers, acc, sched)
        if drain_idx < num_buffers - 2:
            _wait_k_tiles(sched, num_buffers - 3 - drain_idx)
    return acc


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

    X_ptr = X
    if not cfg.USE_GATHER:
        X_ptr += start_m * stride_x_m
    x_desc, w_desc, w_scale_desc, gathered_m = create_descriptors(
        cfg,
        X_ptr,
        W + expt_id * stride_w_e,
        WMxScale + expt_id * stride_w_mx_e,
        off_m,
        pid_n * cfg.BLOCK_N * stride_w_n,
        pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n,
        M,
        eM,
        (N - pid_n * cfg.BLOCK_N).to(gl.int32),
        K,
        stride_x_m,
        stride_x_k,
        stride_w_k,
        stride_w_n,
        stride_w_mx_n,
        stride_w_mx_k,
        GatherIndx,
        start_m,
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
    )


@gluon.jit
def _load_dst_rows(
    WriteBackIndx,
    start_m,
    off_m,
    eM,
    writeback_size,
    BLOCK_M: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
):
    """Destination rows of one output tile; -1 marks rows not to store."""
    rows = gl.arange(0, BLOCK_M, ROW_LAYOUT)
    in_bounds = (off_m + rows < eM) & (start_m + off_m + rows < writeback_size)
    return gl.load(WriteBackIndx + start_m + off_m + rows, mask=in_bounds, other=-1)


# Token and routed-row counts, and the routing buffers whose alignment follows
# them, vary per batch; specializing on them would recompile while serving.
_PER_BATCH_ARGS = [
    "M",
    "num_rows",
    "writeback_size",
    "GatherIndx",
    "WriteBackIndx",
    "XSliceSizes",
    "XSliceOffs",
    "XBlockOffs",
    "XBlockSchedule",
]


@gluon.jit(
    launch_metadata=warp_pipeline_launch_metadata, do_not_specialize=_PER_BATCH_ARGS
)
def gluon_moe_decode_warp_pipelined_gfx1250(
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
    WriteBackIndx: tl.const,
    writeback_size,
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
    NUM_K_SUBTILES: gl.constexpr,
    WARP_PIPELINE: gl.constexpr,
    TDM_MASKS: gl.constexpr,
    W_PARTITION: gl.constexpr,
    X_PARTITION: gl.constexpr,
    ASYNC_STORE: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    INDEX_TYPE: gl.constexpr,
    UPCAST_INDICES: gl.constexpr,
):
    """Persistent decode kernel for the gate/up (gather) or down GEMM."""
    NUM_WARPS: gl.constexpr = 8
    gl.static_assert(gl.num_ctas() == 1)
    USE_GATHER: gl.constexpr = GatherIndx is not None
    # The gate/up GEMM stores expert-sorted rows; the down GEMM writes each
    # row to its route's output row.
    gl.static_assert((WriteBackIndx is None) == USE_GATHER)
    gl.static_assert(not ASYNC_STORE or not USE_GATHER)
    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)
    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N

    unpadded_m = gl.load(XBlockOffs + N_EXPTS_TOT)
    gl.assume(unpadded_m >= 0)
    total_tiles = unpadded_m * grid_n

    sched = DecodeSched(WARP_PIPELINE, USE_GATHER, TDM_MASKS)
    cfg = WarpPipelineConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        NUM_BUFFERS=NUM_BUFFERS,
        index_type=INDEX_TYPE,
        NUM_SUBTILES=(1, 1, NUM_K_SUBTILES),
        USE_GATHER=USE_GATHER,
        NUM_WARPS=NUM_WARPS,
        RESOLVE_PARTITION_CONFLICTS=False,
        GATHER_PRODUCER_WARPS=sched.GATHER_WARPS,
        W_CACHE_MODIFIER=W_CACHE_MODIFIER,
        DECODE_W_PARTITION=W_PARTITION,
        DECODE_X_PARTITION=X_PARTITION,
    )
    BLOCKED_LAYOUT_Y: gl.constexpr = get_blocked_layout(
        [BLOCK_M, OUT_BLOCK_N], Y.dtype, NUM_WARPS, max_vector_bytes=16
    )

    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    x_buffer, w_buffer, w_scale_buffer = allocate_ring(
        cfg, X.dtype.element_ty, W.dtype.element_ty
    )
    ELEM_BYTES: gl.constexpr = Y.dtype.element_ty.primitive_bitwidth // 8
    if ASYNC_STORE:
        # 16 bytes per lane: 256 lanes cover the 16 x 128 tile once, and each
        # half-wave reads two whole rows, one per half of the banks.
        COPY_LAYOUT: gl.constexpr = gl.BlockedLayout(
            [1, 16 // ELEM_BYTES], [2, 16], [NUM_WARPS, 1], [1, 0]
        )
        gl.static_assert(BLOCK_M == 16 and OUT_BLOCK_N * ELEM_BYTES == 256)
        out_smem = gl.allocate_shared_memory(
            Y.dtype.element_ty,
            [BLOCK_M, OUT_BLOCK_N],
            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        )
        # Rows without a destination are masked; the copy needs no mask when
        # every row of a tile has one and the rows are 16-byte aligned.
        aligned_rows = (stride_y_n == 1) & ((stride_y_m * ELEM_BYTES) % 16 == 0)
        COPY_ROW_LAYOUT: gl.constexpr = gl.SliceLayout(1, COPY_LAYOUT)
        COPY_COL_LAYOUT: gl.constexpr = gl.SliceLayout(0, COPY_LAYOUT)

    if pid < total_tiles:
        x_scale = load_scale(XGlobalScale)
        y_scale_inv = 1.0 / load_scale(YGlobalScale)
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
        _tdm_prefetch(pgm, sched)

        if ASYNC_STORE:
            # Later tiles get their destination rows from the previous epilogue.
            dst_rows = _load_dst_rows(
                WriteBackIndx,
                start_m,
                off_m,
                eM,
                writeback_size,
                BLOCK_M,
                COPY_ROW_LAYOUT,
            )
        for tile in range(pid, total_tiles, nprog):
            loop_k = K - off_k_x
            acc = _k_loop_steady(pgm, loop_k, sched)
            bias = load_bias(cfg, B, stride_b_e, expt_id, pid_n, N, 0, BLOCK_N)
            # The warp-pipeline pass would merge drain stages placed directly
            # in the tile loop into one pipelined body; a one-trip loop with a
            # runtime bound keeps them a flat drain.
            for _ in range(gl.minimum(gl.cdiv(loop_k, BLOCK_K), 1)):
                acc = _k_loop_tail(pgm, loop_k, acc, sched)

            # Issue the next tile's leading K tiles into the drained ring so
            # their latency hides under this tile's epilogue. Predicated
            # rather than branched: TDM ops under scf.if make the wait-count
            # analysis walk every path combination across the tile loop.
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
            _tdm_prefetch(next_pgm, sched, (next_tile < total_tiles).to(gl.int32))
            # Keep the epilogue math below the issue; LLVM otherwise sinks the
            # TDM ops past it and the loads no longer overlap the epilogue.
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
            if ASYNC_STORE:
                row_mask = dst_rows != -1
                cols = (OUT_BLOCK_N * pid_n).to(address_index_type) + gl.arange(
                    0, OUT_BLOCK_N, COPY_COL_LAYOUT
                )
                y_ptrs = (
                    Y
                    + gl.maximum(dst_rows, 0).to(address_index_type)[:, None]
                    * stride_y_m
                    + cols[None, :] * stride_y_n
                )
                # The previous tile's copies were issued a whole K loop ago;
                # they must be done before the staging tile is reused.
                if tile != pid:
                    async_copy.wait_group(0)
                # Stage straight from the accumulator layout.
                out_smem.store(out)
                all_rows = gl.min(row_mask.to(gl.int32), axis=0) != 0
                # Apply the contiguity hint at the pointers' definition: on a
                # helper's argument it is dropped and the copy uses byte stores.
                if aligned_rows & all_rows:
                    async_copy.shared_to_global(
                        gl.max_contiguous(
                            gl.multiple_of(y_ptrs, [1, 16 // ELEM_BYTES]),
                            [1, 16 // ELEM_BYTES],
                        ),
                        out_smem,
                    )
                else:
                    async_copy.shared_to_global(
                        y_ptrs, out_smem, mask=row_mask[:, None]
                    )
                async_copy.commit_group()
                dst_rows = _load_dst_rows(
                    WriteBackIndx,
                    n_start_m,
                    n_off_m,
                    n_eM,
                    writeback_size,
                    BLOCK_M,
                    COPY_ROW_LAYOUT,
                )
            else:
                out = gl.convert_layout(out, BLOCKED_LAYOUT_Y)
                rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, BLOCKED_LAYOUT_Y))
                cols = (OUT_BLOCK_N * pid_n).to(address_index_type) + gl.arange(
                    0, OUT_BLOCK_N, gl.SliceLayout(0, BLOCKED_LAYOUT_Y)
                )
                if USE_GATHER:
                    # A TDM store would share the tensor counter with the next
                    # tile's prefetched loads, so their waits would also wait
                    # for it.
                    dst = start_m + off_m + rows.to(address_index_type)
                    row_mask = off_m + rows < eM
                else:
                    dst_rows = _load_dst_rows(
                        WriteBackIndx,
                        start_m,
                        off_m,
                        eM,
                        writeback_size,
                        BLOCK_M,
                        gl.SliceLayout(1, BLOCKED_LAYOUT_Y),
                    )
                    row_mask = dst_rows != -1
                    dst = dst_rows.to(address_index_type)
                y_ptrs = Y + dst[:, None] * stride_y_m + cols[None, :] * stride_y_n
                gl.store(
                    y_ptrs,
                    out,
                    mask=row_mask[:, None]
                    & (cols < N // ACTIVATION_REDUCTION_N)[None, :],
                )

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
        if ASYNC_STORE:
            async_copy.wait_group(0)
    x_buffer._keep_alive()
    w_buffer._keep_alive()
    w_scale_buffer._keep_alive()
