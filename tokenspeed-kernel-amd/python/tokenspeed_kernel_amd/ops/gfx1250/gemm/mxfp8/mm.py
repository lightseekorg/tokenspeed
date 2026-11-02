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

"""Clustered warp-pipeline MXFP8 projections for gfx1250 batches above 16 rows.

The BK512 and BK256 kernels adapt the MXFP8 variants of Triton's CDNA5
warp-pipeline GEMM example
(``third_party/amd/python/examples/gluon/gemm_warp_pipeline_cdna5.py``) to
serving shapes: every TDM descriptor is bounded at the real M, N and K, so
ragged edges zero-fill on load and clip on store, and tiles follow a runtime
XCD order.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    _NUM_CUS,
    _largem_warp_pipeline_xcd_tile,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.decode_mm import (
    _REDUCE_BLOCK_N,
    _mxfp8_split_reduce_gfx1250,
)

_NUM_XCDS = 8
_GROUP_M = 4
_SCALE_BLOCK = 32
_BK512 = 512
_BK256 = 256
_BK256_TILE = 256
_BK256_MIN_M = 512


@gluon.constexpr_function
def _bk512_layouts(tile, cluster, fp32_scales):
    """LDS and WMMA layouts of a ``cluster x cluster`` group of square tiles."""
    block = tile * cluster
    # CTA-id bits walk M first, then N; the same bases as
    # make_cga_layout([cluster, cluster], [cluster, cluster], [0, 1]).
    bits = cluster.bit_length() - 1
    cga = [[1 << i, 0] for i in range(bits)] + [[0, 1 << i] for i in range(bits)]
    local = gl.PaddedSharedLayout.with_identity_for(
        [[_BK512, 16]], [block, _BK512], [1, 0]
    )
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
        block,
        block,
        local,
        local,
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=tile,
    )
    wmma = gl.amd.AMDWMMALayout(
        3,
        local_wmma.transposed,
        local_wmma.warp_bases,
        local_wmma.reg_bases,
        local_wmma.instr_shape,
        cga,
    )
    cga_a = gl.DotOperandLayout(0, wmma, 16).cga_layout
    # B sits in LDS as [N, K], so its CGA bases swap the operand's (K, N).
    cga_b = [[x[1], x[0]] for x in gl.DotOperandLayout(1, wmma, 16).cga_layout]
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(
        tile,
        tile,
        gl.PaddedSharedLayout.with_identity_for(
            [[_BK512, 16]], [block, _BK512], [1, 0], cga_a
        ),
        gl.PaddedSharedLayout.with_identity_for(
            [[_BK512, 16]], [block, _BK512], [1, 0], cga_b
        ),
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=tile,
    )
    if fp32_scales:
        scale_cols = _BK512 // 128
        shared_a_scale = gl.PaddedSharedLayout.with_identity_for(
            [[scale_cols, 2]], [block, scale_cols], [1, 0], cga_a
        )
        # One weight-scale row per 128 output columns; CTAs inside one
        # weight block share it.
        shared_b_scale = gl.PaddedSharedLayout.with_identity_for(
            [[scale_cols, 2]],
            [triton.cdiv(block, 128), scale_cols],
            [1, 0],
            [[n // 128, k] for n, k in cga_b],
        )
    else:
        scale_cols = _BK512 // _SCALE_BLOCK
        shared_a_scale = gl.PaddedSharedLayout.with_identity_for(
            [[scale_cols, 8]], [block, scale_cols], [1, 0], cga_a
        )
        shared_b_scale = gl.PaddedSharedLayout.with_identity_for(
            [[scale_cols, 8]], [block, scale_cols], [1, 0], cga_b
        )
    shared_out = gl.PaddedSharedLayout.with_identity_for(
        [[tile, 8]], [block, block], [1, 0], cga
    )
    return shared_a, shared_b, shared_a_scale, shared_b_scale, wmma, shared_out


@gluon.constexpr_function
def _bk256_layouts(cluster):
    """Layouts of a ``cluster x cluster`` group of 256 x 256 tiles.

    Each CTA computes its tile as two 128-column halves. B is staged as a
    ``[cluster, 256, 256]`` partitioned box per CTA column, and both raw
    scale planes as ``[rows, 8]`` per 256-wide K tile.
    """
    tile = _BK256_TILE
    half = tile // 2
    block = tile * cluster
    # CTA-id bits walk M first, then N; the same bases as
    # make_cga_layout([cluster, cluster], [cluster, cluster], [0, 1]).
    bits = cluster.bit_length() - 1
    cga = [[1 << i, 0] for i in range(bits)] + [[0, 1 << i] for i in range(bits)]
    local = gl.PaddedSharedLayout.with_identity_for(
        [[_BK256, 16]], [block, _BK256], [1, 0]
    )
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
        block,
        block,
        local,
        local,
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=half,
    )
    load_wmma = gl.amd.AMDWMMALayout(
        3,
        local_wmma.transposed,
        local_wmma.warp_bases,
        local_wmma.reg_bases,
        local_wmma.instr_shape,
        cga,
    )
    cga_a = gl.DotOperandLayout(0, load_wmma, 16).cga_layout
    # B sits in LDS as [N, K], so its CGA bases swap the operand's (K, N).
    cga_b = [[x[1], x[0]] for x in gl.DotOperandLayout(1, load_wmma, 16).cga_layout]
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(
        tile,
        tile,
        gl.PaddedSharedLayout.with_identity_for(
            [[_BK256, 16]], [block, _BK256], [1, 0], cga_a
        ),
        gl.PaddedSharedLayout.with_identity_for(
            [[_BK256, 16]], [block, _BK256], [1, 0], cga_b
        ),
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=half,
    )
    _, _, half_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
        block,
        block // 2,
        local,
        gl.PaddedSharedLayout.with_identity_for(
            [[_BK256, 16]], [block // 2, _BK256], [1, 0]
        ),
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=half,
    )
    wmma = gl.amd.AMDWMMALayout(
        3,
        half_wmma.transposed,
        half_wmma.warp_bases,
        half_wmma.reg_bases,
        half_wmma.instr_shape,
        cga,
    )
    # A 12-byte row pitch spreads the 4-byte scale reads of consecutive rows
    # across distinct banks.
    scale_cols = _BK256 // _SCALE_BLOCK
    shared_a_scale = gl.PaddedSharedLayout.with_identity_for(
        [[scale_cols, 4]], [block, scale_cols], [1, 0], cga_a
    )
    shared_b_scale = gl.PaddedSharedLayout.with_identity_for(
        [[scale_cols, 4]], [block, scale_cols], [1, 0], cga_b
    )
    b_parent_cga = [
        [basis[0], 0, basis[1]] for basis in shared_b.partition_layout.cga_layout
    ]
    shared_b2 = gl.amd.cdna5.PartitionedSharedLayout(
        2,
        2,
        1,
        gl.PaddedSharedLayout.with_identity_for(
            [[_BK256, 16]], [cluster, 64, _BK256], [2, 1, 0], b_parent_cga
        ),
    )
    b2_load = gl.DistributedLinearLayout(
        reg_bases=[
            [1, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [8, 0, 0],
            [32, 0, 0],
            [64, 0, 0],
            [0, 0, 16],
            [0, 0, 32],
        ],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [16, 0, 0]],
        warp_bases=[[0, 0, 64], [0, 0, 0], [0, 0, 0]],
        block_bases=[[0, 0, 0]] * bits + [[0, 1 << i, 0] for i in range(bits)],
        shape=[half, cluster, half],
    )
    # The output is viewed as [N / 256, M, 256], so the per-CTA unit dimension
    # of the box leads, as TDM requires descending shared order.
    shared_out = gl.PaddedSharedLayout.with_identity_for(
        [[half, 8]],
        [cluster, block, half],
        [2, 1, 0],
        [[basis[1], basis[0], 0] for basis in cga],
    )
    return (
        shared_a,
        shared_b2,
        shared_a_scale,
        shared_b_scale,
        wmma,
        load_wmma,
        b2_load,
        shared_out,
    )


def _cluster_launch_metadata(grid, kernel, args, block_m, block_n):
    m, n, k = args["M"], args["N"], args["K"]
    grid_m = triton.cdiv(m, block_m)
    grid_n = triton.cdiv(n, block_n)
    # Each cluster streams its A row panel and B column panel once; the
    # multicast hands each to all of the cluster's CTAs.
    value_bytes = (grid_n * m + grid_m * n) * k
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}]",
        "flops8": 2 * m * n * k,
        "bytes": value_bytes * (1 + 1 / _SCALE_BLOCK) + 2 * m * n,
    }


def _bk512_launch_metadata(grid, kernel, args):
    """Report useful FLOPs, the per-cluster A/B panel reads and the store."""
    block = args["TILE"] * args["CLUSTER"]
    return _cluster_launch_metadata(grid, kernel, args, block, block)


def _bk512_repr(specialization):
    constants = specialization.constants
    scales = "_fp32" if constants.get("FP32_SCALES") else ""
    walk = "_k1" if constants.get("SINGLE_K_TILE") else ""
    return (
        f"gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250{scales}_t{constants.get('TILE')}_"
        f"c{constants.get('CLUSTER')}_sk{constants.get('SPLIT_K')}{walk}"
    )


@gluon.jit
def _bk512_refill(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    slot,
    BLOCK_K: gl.constexpr,
    SCALE_K: gl.constexpr,
):
    # Waves 0-3 issue A and B; the fused scale load splits its two planes
    # across waves 0-1 and 2-3.
    gl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem.index(slot), warp_used_hint=15)
    gl.amd.cdna5.tdm.async_load(b_desc, [0, 0], b_smem.index(slot), warp_used_hint=15)
    gl.amd.cdna5.tdm.async_load_fused(
        [
            (a_scale_desc, a_scale_smem.index(slot), 3),
            (b_scale_desc, b_scale_smem.index(slot), 12),
        ]
    )
    # Advancing also shrinks the bounds, so a partial last K tile zero-fills
    # past K instead of reading the next row.
    a_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, BLOCK_K], clamp_bounds=True
    )
    b_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, BLOCK_K], clamp_bounds=True
    )
    a_scale_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        a_scale_desc, add_offsets=[0, BLOCK_K // SCALE_K], clamp_bounds=True
    )
    b_scale_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        b_scale_desc, add_offsets=[0, BLOCK_K // SCALE_K], clamp_bounds=True
    )
    return a_desc, b_desc, a_scale_desc, b_scale_desc


@gluon.jit
def _bk512_operands(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    slot,
    k: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    SCALE_K: gl.constexpr,
):
    a = a_smem.index(slot).slice(k, 128, 1).load(layout=DOT_A)
    a_scale = (
        a_scale_smem.index(slot)
        .slice(k // SCALE_K, 128 // SCALE_K, 1)
        .load(layout=SCALE_A)
    )
    b = b_smem.index(slot).slice(k, 128, 1).permute([1, 0]).load(layout=DOT_B)
    if SCALE_K == 128:
        # Widen each weight-scale row to the 128 output columns it covers.
        cols: gl.constexpr = b_smem.type.shape[1]
        rows: gl.constexpr = b_scale_smem.type.shape[1]
        grid: gl.constexpr = (
            gl.zeros([1, cols], gl.float32, SCALE_B)
            .reshape([1, rows, cols // rows])
            .type.layout
        )
        b_rows = (
            b_scale_smem.index(slot)
            .slice(k // SCALE_K, 1, 1)
            .permute([1, 0])
            .load(layout=gl.SliceLayout(2, grid))
        )
        b_scale = gl.convert_layout(
            gl.expand_dims(b_rows, 2)
            .broadcast_to([1, rows, cols // rows])
            .reshape([1, cols]),
            SCALE_B,
            assert_trivial=True,
        )
    else:
        b_scale = (
            b_scale_smem.index(slot)
            .slice(k // SCALE_K, 128 // SCALE_K, 1)
            .load(layout=SCALE_B)
        )
    return a, a_scale, b, b_scale


@gluon.jit
def _bk512_mma(a, a_scale, b, b_scale, acc, SCALE_K: gl.constexpr):
    if SCALE_K == 128:
        # FP32 block scales: scale the unscaled 128-wide partial, activation
        # scale first, before accumulating.
        partial = gl.amd.cdna5.wmma(a, b, gl.zeros_like(acc))
        acc = acc + partial * a_scale * b_scale
    else:
        acc = gl.amd.cdna5.wmma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
    return acc


@gluon.jit
def _bk512_pin(acc, SCALE_K: gl.constexpr):
    # LLVM sinks pure math to its first use when a branch separates them. The
    # warp pipeline's closing barrier and the one-wave cluster arrive both
    # branch, so the FP32 scale math would leave its stage and every partial
    # would stay live and spill. An empty asm with side effects consumes the
    # sum inside the stage and emits no instruction.
    if SCALE_K == 128:
        acc = gl.inline_asm_elementwise(
            "", "=v,0", [acc], dtype=gl.float32, is_pure=False, pack=1
        )
    return acc


@gluon.jit
def _bk512_consume_refill(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    slot,
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    acc,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    BLOCK_K: gl.constexpr,
    CLUSTER_SYNC: gl.constexpr,
    SCALE_K: gl.constexpr,
):
    # One 512-wide K tile in five stages: the two waves sharing a SIMD
    # alternate LDS reads of one half with the scaled WMMAs of the other, and
    # the last stage refills this slot with the tile two ahead.
    gl.amd.cdna5.tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage("stage0_load_k0", phase_gap=1):
        a0, as0, b0, bs0 = _bk512_operands(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            slot,
            0,
            DOT_A,
            DOT_B,
            SCALE_A,
            SCALE_B,
            SCALE_K,
        )
        a1, as1, b1, bs1 = _bk512_operands(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            slot,
            128,
            DOT_A,
            DOT_B,
            SCALE_A,
            SCALE_B,
            SCALE_K,
        )
    with gl.amd.warp_pipeline_stage("stage1_compute_k0"):
        acc = _bk512_mma(a0, as0, b0, bs0, acc, SCALE_K)
        acc = _bk512_pin(_bk512_mma(a1, as1, b1, bs1, acc, SCALE_K), SCALE_K)
    with gl.amd.warp_pipeline_stage("stage2_load_k2"):
        a2, as2, b2, bs2 = _bk512_operands(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            slot,
            256,
            DOT_A,
            DOT_B,
            SCALE_A,
            SCALE_B,
            SCALE_K,
        )
        a3, as3, b3, bs3 = _bk512_operands(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            slot,
            384,
            DOT_A,
            DOT_B,
            SCALE_A,
            SCALE_B,
            SCALE_K,
        )
        if CLUSTER_SYNC:
            # Keep the cluster's CTAs within one K tile of each other so each
            # multicast refill finds every destination slot free.
            gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("stage3_compute_k2"):
        acc = _bk512_mma(a2, as2, b2, bs2, acc, SCALE_K)
        acc = _bk512_pin(_bk512_mma(a3, as3, b3, bs3, acc, SCALE_K), SCALE_K)
    with gl.amd.warp_pipeline_stage("stage4_wait_refill"):
        if CLUSTER_SYNC:
            gl.amd.cdna5.cluster.wait()
        a_desc, b_desc, a_scale_desc, b_scale_desc = _bk512_refill(
            a_desc,
            b_desc,
            a_scale_desc,
            b_scale_desc,
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            slot,
            BLOCK_K,
            SCALE_K,
        )
    return a_desc, b_desc, a_scale_desc, b_scale_desc, acc


@gluon.jit
def _bk512_consume_tail(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    slot,
    acc,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SCALE_K: gl.constexpr,
):
    for k in gl.static_range(0, BLOCK_K, 128):
        with gl.amd.warp_pipeline_stage("tail_load"):
            a, a_scale, b, b_scale = _bk512_operands(
                a_smem,
                b_smem,
                a_scale_smem,
                b_scale_smem,
                slot,
                k,
                DOT_A,
                DOT_B,
                SCALE_A,
                SCALE_B,
                SCALE_K,
            )
        with gl.amd.warp_pipeline_stage("tail_compute"):
            acc = _bk512_pin(_bk512_mma(a, a_scale, b, b_scale, acc, SCALE_K), SCALE_K)
    return acc


# The live row count varies with the batch, so it must not split the compile
# cache on its divisibility.
@gluon.jit(
    launch_metadata=_bk512_launch_metadata,
    repr=_bk512_repr,
    do_not_specialize=["M"],
)
def gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    stride_am,
    stride_bn,
    stride_asm,
    stride_bsn,
    stride_ok,
    stride_om,
    M,
    N,
    K,
    TILE: gl.constexpr,
    CLUSTER: gl.constexpr,
    SPLIT_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    FP32_SCALES: gl.constexpr,
    SINGLE_K_TILE: gl.constexpr,
):
    """Clustered eight-wave E4M3 GEMM over 512-wide K tiles.

    Scales are raw ``[1,32]`` UE8M0 bytes for the scaled WMMA or, with
    ``FP32_SCALES``, canonical FP32 ``[128,128]`` block scales: each 128-wide
    raw WMMA partial takes its rows' activation scales, then its weight
    block's scale, before accumulation. FP32 weight scales hold one row per
    128 output columns.

    Each program is a ``CLUSTER x CLUSTER`` group of CTAs, one
    ``TILE x TILE`` output tile each, that share A, B and their scale
    planes through TDM multicast into two LDS slots. A walk of one K tile
    (``SINGLE_K_TILE``) loads it alone; longer walks cover at least two
    tiles, and a partial last tile and any tile past K zero-fill. CTAs wholly
    past the M or N edge still run the loop, so every
    cluster barrier completes. With ``SPLIT_K`` above one, unclustered
    programs on grid axis 1 each walk ``K / SPLIT_K`` and store FP32
    partials ``stride_ok`` apart.
    """
    gl.static_assert(gl.num_warps() == 8, "the warp pipeline needs two waves per SIMD")
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER, "one CTA per cluster tile")
    gl.static_assert(SPLIT_K == 1 or CLUSTER == 1, "split K runs unclustered")
    BLOCK: gl.constexpr = TILE * CLUSTER
    BLOCK_K: gl.constexpr = 512
    # K columns per scale, and output columns per weight-scale row.
    SCALE_K: gl.constexpr = 128 if FP32_SCALES else 32
    B_ROW: gl.constexpr = 128 if FP32_SCALES else 1
    B_SCALE_ROWS: gl.constexpr = (BLOCK + B_ROW - 1) // B_ROW
    SCALE_COLS: gl.constexpr = BLOCK_K // SCALE_K
    CLUSTER_SYNC: gl.constexpr = CLUSTER > 1
    layouts: gl.constexpr = _bk512_layouts(TILE, CLUSTER, FP32_SCALES)
    SHARED_A: gl.constexpr = layouts[0]
    SHARED_B: gl.constexpr = layouts[1]
    SHARED_A_SCALE: gl.constexpr = layouts[2]
    SHARED_B_SCALE: gl.constexpr = layouts[3]
    WMMA_LAYOUT: gl.constexpr = layouts[4]
    SHARED_OUT: gl.constexpr = layouts[5]
    pid_m, pid_n = _largem_warp_pipeline_xcd_tile(
        gl.program_id(0), gl.cdiv(M, BLOCK), gl.cdiv(N, BLOCK), NUM_XCDS, GROUP_M
    )
    off_m = pid_m * BLOCK
    off_n = pid_n * BLOCK
    k_split = K // SPLIT_K
    k_start = gl.program_id(1) * k_split
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    if FP32_SCALES:
        scale_a: gl.constexpr = WMMA_LAYOUT
        scale_b: gl.constexpr = WMMA_LAYOUT
    else:
        scale_a: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK, 4])
        scale_b: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK, 4])
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, BLOCK, BLOCK_K], SHARED_A
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, BLOCK, BLOCK_K], SHARED_B
    )
    a_scale_smem = gl.allocate_shared_memory(
        a_scale_ptr.dtype.element_ty, [2, BLOCK, SCALE_COLS], SHARED_A_SCALE
    )
    b_scale_smem = gl.allocate_shared_memory(
        b_scale_ptr.dtype.element_ty, [2, B_SCALE_ROWS, SCALE_COLS], SHARED_B_SCALE
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am + k_start,
        shape=(M - off_m, k_split),
        strides=(stride_am, 1),
        block_shape=(BLOCK, BLOCK_K),
        layout=SHARED_A,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + off_n * stride_bn + k_start,
        shape=(N - off_n, k_split),
        strides=(stride_bn, 1),
        block_shape=(BLOCK, BLOCK_K),
        layout=SHARED_B,
    )
    a_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_scale_ptr + off_m * stride_asm + k_start // SCALE_K,
        shape=(M - off_m, k_split // SCALE_K),
        strides=(stride_asm, 1),
        block_shape=(BLOCK, SCALE_COLS),
        layout=SHARED_A_SCALE,
    )
    b_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_scale_ptr + off_n // B_ROW * stride_bsn + k_start // SCALE_K,
        shape=(gl.cdiv(N - off_n, B_ROW), k_split // SCALE_K),
        strides=(stride_bsn, 1),
        block_shape=(B_SCALE_ROWS, SCALE_COLS),
        layout=SHARED_B_SCALE,
    )
    for slot in gl.static_range(1 if SINGLE_K_TILE else 2):
        a_desc, b_desc, a_scale_desc, b_scale_desc = _bk512_refill(
            a_desc,
            b_desc,
            a_scale_desc,
            b_scale_desc,
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            slot,
            BLOCK_K,
            SCALE_K,
        )
    acc = gl.zeros((BLOCK, BLOCK), dtype=gl.float32, layout=WMMA_LAYOUT)
    if SINGLE_K_TILE:
        gl.amd.cdna5.tdm.async_wait(0)
        acc = _bk512_consume_tail(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            0,
            acc,
            dot_a,
            dot_b,
            scale_a,
            scale_b,
            BLOCK_K,
            SCALE_K,
        )
    else:
        num_k_tiles = gl.maximum(gl.cdiv(k_split, BLOCK_K), 2)
        gl.assume(num_k_tiles >= 2)
        # Unrolling six tiles keeps the two-slot parity static in the body.
        for _ in range((num_k_tiles - 2) // 6):
            for inner in gl.static_range(6):
                a_desc, b_desc, a_scale_desc, b_scale_desc, acc = _bk512_consume_refill(
                    a_smem,
                    b_smem,
                    a_scale_smem,
                    b_scale_smem,
                    inner % 2,
                    a_desc,
                    b_desc,
                    a_scale_desc,
                    b_scale_desc,
                    acc,
                    dot_a,
                    dot_b,
                    scale_a,
                    scale_b,
                    BLOCK_K,
                    CLUSTER_SYNC,
                    SCALE_K,
                )
        for tile in range((num_k_tiles - 2) // 6 * 6, num_k_tiles - 2):
            a_desc, b_desc, a_scale_desc, b_scale_desc, acc = _bk512_consume_refill(
                a_smem,
                b_smem,
                a_scale_smem,
                b_scale_smem,
                tile % 2,
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                acc,
                dot_a,
                dot_b,
                scale_a,
                scale_b,
                BLOCK_K,
                CLUSTER_SYNC,
                SCALE_K,
            )
        gl.amd.cdna5.tdm.async_wait(3)
        acc = _bk512_consume_tail(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            num_k_tiles % 2,
            acc,
            dot_a,
            dot_b,
            scale_a,
            scale_b,
            BLOCK_K,
            SCALE_K,
        )
        gl.amd.cdna5.tdm.async_wait(0)
        acc = _bk512_consume_tail(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            (num_k_tiles - 1) % 2,
            acc,
            dot_a,
            dot_b,
            scale_a,
            scale_b,
            BLOCK_K,
            SCALE_K,
        )
    if CLUSTER_SYNC:
        gl.amd.cdna5.cluster.arrive()
        gl.amd.cdna5.cluster.wait()
    a_smem._keep_alive()
    b_smem._keep_alive()
    a_scale_smem._keep_alive()
    b_scale_smem._keep_alive()
    out_smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [BLOCK, BLOCK], SHARED_OUT
    )
    out_smem.store(acc.to(out_ptr.dtype.element_ty))
    out_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=out_ptr + gl.program_id(1) * stride_ok + off_m * stride_om + off_n,
        shape=(M - off_m, N - off_n),
        strides=(stride_om, 1),
        block_shape=(BLOCK, BLOCK),
        layout=SHARED_OUT,
    )
    gl.amd.cdna5.tdm.async_store(out_desc, [0, 0], out_smem)
    gl.amd.cdna5.tdm.async_wait(0)


def _bk256_launch_metadata(grid, kernel, args):
    """Report useful FLOPs, the per-cluster A/B panel reads and the store."""
    block = _BK256_TILE * args["CLUSTER"]
    return _cluster_launch_metadata(grid, kernel, args, block, block)


def _bk256_repr(specialization):
    return (
        "gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250_"
        f"c{specialization.constants.get('CLUSTER')}"
    )


@gluon.jit
def _bk256_scales(
    a_scale_smem,
    b_scale_smem,
    slot,
    start_k: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    SCALE_B_LOAD: gl.constexpr,
    BLOCK: gl.constexpr,
    CLUSTER: gl.constexpr,
):
    a_scale = a_scale_smem.index(slot).slice(start_k, 4, 1).load(layout=SCALE_A)
    b_scale = b_scale_smem.index(slot).slice(start_k, 4, 1).load(layout=SCALE_B_LOAD)
    # Each CTA's 256 B rows split into the low and high 128-column halves.
    grid = b_scale.reshape((CLUSTER, 256, 4))
    b_lo = gl.convert_layout(
        gl.amd.slice(grid, [CLUSTER, 128, 4], [0, 0, 0]).reshape((BLOCK // 2, 4)),
        SCALE_B,
        assert_trivial=True,
    )
    b_hi = gl.convert_layout(
        gl.amd.slice(grid, [CLUSTER, 128, 4], [0, 128, 0]).reshape((BLOCK // 2, 4)),
        SCALE_B,
        assert_trivial=True,
    )
    return a_scale, b_lo, b_hi


@gluon.jit
def _bk256_load_b(
    b_smem,
    slot,
    start_n: gl.constexpr,
    start_k: gl.constexpr,
    B2_LOAD: gl.constexpr,
    DOT_B: gl.constexpr,
):
    rank3 = (
        b_smem.index(slot)
        .slice(start_n, 128, 1)
        .slice(start_k, 128, 2)
        .permute([2, 0, 1])
        .load(layout=B2_LOAD)
    )
    return gl.convert_layout(rank3.reshape((128, b_smem.type.shape[1] * 128)), DOT_B)


@gluon.jit
def _bk256_issue_data(a_desc, b_desc, a_smem, b_smem, slot):
    gl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem.index(slot), warp_used_hint=15)
    gl.amd.cdna5.tdm.async_load(
        b_desc, [0, 0, 0], b_smem.index(slot), warp_used_hint=15
    )


@gluon.jit
def _bk256_advance_data(a_desc, b_desc):
    # Advancing also shrinks the bounds, so K tiles past K zero-fill instead
    # of reading the next row.
    a_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, 256], clamp_bounds=True
    )
    b_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, 0, 256], clamp_bounds=True
    )
    return a_desc, b_desc


@gluon.jit
def _bk256_issue_scales(a_scale_desc, b_scale_desc, a_scale_smem, b_scale_smem, slot):
    gl.amd.cdna5.tdm.async_load_fused(
        [
            (a_scale_desc, a_scale_smem.index(slot), 3),
            (b_scale_desc, b_scale_smem.index(slot), 12),
        ]
    )


@gluon.jit
def _bk256_advance_scales(a_scale_desc, b_scale_desc):
    a_scale_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        a_scale_desc, add_offsets=[0, 256 // 32], clamp_bounds=True
    )
    b_scale_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        b_scale_desc, add_offsets=[0, 256 // 32], clamp_bounds=True
    )
    return a_scale_desc, b_scale_desc


@gluon.jit
def _bk256_half(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    data_slot,
    scale_slot,
    start_k: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    B2_LOAD: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    SCALE_B_LOAD: gl.constexpr,
    BLOCK: gl.constexpr,
    CLUSTER: gl.constexpr,
):
    a_scale, b_scale_lo, b_scale_hi = _bk256_scales(
        a_scale_smem,
        b_scale_smem,
        scale_slot,
        start_k // 32,
        SCALE_A,
        SCALE_B,
        SCALE_B_LOAD,
        BLOCK,
        CLUSTER,
    )
    a = a_smem.index(data_slot).slice(start_k, 128, 1).load(layout=DOT_A)
    b_lo = _bk256_load_b(b_smem, data_slot, 0, start_k, B2_LOAD, DOT_B)
    b_hi = _bk256_load_b(b_smem, data_slot, 128, start_k, B2_LOAD, DOT_B)
    return a, a_scale, b_lo, b_scale_lo, b_hi, b_scale_hi


@gluon.jit
def _bk256_consume_refill(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    data_slot,
    scale_slot,
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    acc0,
    acc1,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    B2_LOAD: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    SCALE_B_LOAD: gl.constexpr,
    BLOCK: gl.constexpr,
    CLUSTER: gl.constexpr,
    REFILL_SCALES: gl.constexpr,
    CLUSTER_SYNC: gl.constexpr,
):
    # One 256-wide K tile in five stages over its two 128-wide halves. Data
    # rides two LDS slots and scales three; the last stage refills this data
    # slot with the tile two ahead and this scale slot with the tile three
    # ahead.
    gl.amd.cdna5.tdm.async_wait(4)
    next_a_desc = a_desc
    next_b_desc = b_desc
    next_a_scale_desc = a_scale_desc
    next_b_scale_desc = b_scale_desc
    with gl.amd.warp_pipeline_stage("stage0_load_low", phase_gap=1):
        a0, as0, b0_lo, bs0_lo, b0_hi, bs0_hi = _bk256_half(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            data_slot,
            scale_slot,
            0,
            DOT_A,
            DOT_B,
            B2_LOAD,
            SCALE_A,
            SCALE_B,
            SCALE_B_LOAD,
            BLOCK,
            CLUSTER,
        )
    with gl.amd.warp_pipeline_stage("stage1_compute_low"):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("stage2_load_high_advance"):
        a1, as1, b1_lo, bs1_lo, b1_hi, bs1_hi = _bk256_half(
            a_smem,
            b_smem,
            a_scale_smem,
            b_scale_smem,
            data_slot,
            scale_slot,
            128,
            DOT_A,
            DOT_B,
            B2_LOAD,
            SCALE_A,
            SCALE_B,
            SCALE_B_LOAD,
            BLOCK,
            CLUSTER,
        )
        if CLUSTER_SYNC:
            # Keep the cluster's CTAs within one K tile of each other so each
            # multicast refill finds every destination slot free.
            gl.amd.cdna5.cluster.arrive()
        next_a_desc, next_b_desc = _bk256_advance_data(a_desc, b_desc)
        if REFILL_SCALES:
            next_a_scale_desc, next_b_scale_desc = _bk256_advance_scales(
                a_scale_desc, b_scale_desc
            )
    with gl.amd.warp_pipeline_stage("stage3_compute_high"):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("stage4_wait_refill"):
        if CLUSTER_SYNC:
            gl.amd.cdna5.cluster.wait()
        _bk256_issue_data(a_desc, b_desc, a_smem, b_smem, data_slot)
        if REFILL_SCALES:
            _bk256_issue_scales(
                a_scale_desc, b_scale_desc, a_scale_smem, b_scale_smem, scale_slot
            )
    return (
        next_a_desc,
        next_b_desc,
        next_a_scale_desc,
        next_b_scale_desc,
        acc0,
        acc1,
    )


@gluon.jit
def _bk256_consume_tail(
    a_smem,
    b_smem,
    a_scale_smem,
    b_scale_smem,
    data_slot,
    scale_slot,
    acc0,
    acc1,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    B2_LOAD: gl.constexpr,
    SCALE_A: gl.constexpr,
    SCALE_B: gl.constexpr,
    SCALE_B_LOAD: gl.constexpr,
    BLOCK: gl.constexpr,
    CLUSTER: gl.constexpr,
):
    for start_k in gl.static_range(0, 256, 128):
        with gl.amd.warp_pipeline_stage("tail_load"):
            a, a_scale, b_lo, bs_lo, b_hi, bs_hi = _bk256_half(
                a_smem,
                b_smem,
                a_scale_smem,
                b_scale_smem,
                data_slot,
                scale_slot,
                start_k,
                DOT_A,
                DOT_B,
                B2_LOAD,
                SCALE_A,
                SCALE_B,
                SCALE_B_LOAD,
                BLOCK,
                CLUSTER,
            )
        with gl.amd.warp_pipeline_stage("tail_compute"):
            acc0 = gl.amd.cdna5.wmma_scaled(
                a, a_scale, "e4m3", b_lo, bs_lo, "e4m3", acc0
            )
            acc1 = gl.amd.cdna5.wmma_scaled(
                a, a_scale, "e4m3", b_hi, bs_hi, "e4m3", acc1
            )
    return acc0, acc1


# The live row count varies with the batch, so it must not split the compile
# cache on its divisibility.
@gluon.jit(
    launch_metadata=_bk256_launch_metadata,
    repr=_bk256_repr,
    do_not_specialize=["M"],
)
def gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    stride_am,
    stride_bn,
    stride_asm,
    stride_bsn,
    stride_om,
    M,
    N,
    K,
    CLUSTER: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Clustered eight-wave ``[1,32]`` E4M3/UE8M0 GEMM over 256-wide K tiles.

    Each program is a ``CLUSTER x CLUSTER`` group of CTAs, one 256 x 256 BF16
    output tile each, that share A, B and their raw UE8M0 scale planes through
    TDM multicast. N must be a multiple of 256. The K walk rounds up to an
    even count of at least four tiles, and tiles past K zero-fill. CTAs wholly
    past the M or N edge still run the loop, so every cluster barrier
    completes.
    """
    gl.static_assert(gl.num_warps() == 8, "the warp pipeline needs two waves per SIMD")
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER, "one CTA per cluster tile")
    BLOCK: gl.constexpr = 256 * CLUSTER
    SCALE_COLS: gl.constexpr = 256 // 32
    CLUSTER_SYNC: gl.constexpr = CLUSTER > 1
    layouts: gl.constexpr = _bk256_layouts(CLUSTER)
    SHARED_A: gl.constexpr = layouts[0]
    SHARED_B2: gl.constexpr = layouts[1]
    SHARED_A_SCALE: gl.constexpr = layouts[2]
    SHARED_B_SCALE: gl.constexpr = layouts[3]
    WMMA_LAYOUT: gl.constexpr = layouts[4]
    LOAD_WMMA_LAYOUT: gl.constexpr = layouts[5]
    B2_LOAD: gl.constexpr = layouts[6]
    SHARED_OUT: gl.constexpr = layouts[7]
    pid_m, pid_n = _largem_warp_pipeline_xcd_tile(
        gl.program_id(0), gl.cdiv(M, BLOCK), gl.cdiv(N, BLOCK), NUM_XCDS, GROUP_M
    )
    off_m = pid_m * BLOCK
    off_n = pid_n * BLOCK
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_a: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_a, [BLOCK, 4])
    scale_b: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(dot_b, [BLOCK // 2, 4])
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b_load, [BLOCK, 4]
    )
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, BLOCK, 256], SHARED_A
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, CLUSTER, 256, 256], SHARED_B2
    )
    a_scale_smem = gl.allocate_shared_memory(
        a_scale_ptr.dtype.element_ty, [3, BLOCK, SCALE_COLS], SHARED_A_SCALE
    )
    b_scale_smem = gl.allocate_shared_memory(
        b_scale_ptr.dtype.element_ty, [3, BLOCK, SCALE_COLS], SHARED_B_SCALE
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am,
        shape=(M - off_m, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK, 256),
        layout=SHARED_A,
    )
    # B is viewed as [N / 256, 256, K] so each CTA column's rows form a box.
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + off_n * stride_bn,
        shape=((N - off_n) // 256, 256, K),
        strides=(256 * stride_bn, stride_bn, 1),
        block_shape=(CLUSTER, 256, 256),
        layout=SHARED_B2,
    )
    a_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_scale_ptr + off_m * stride_asm,
        shape=(M - off_m, K // 32),
        strides=(stride_asm, 1),
        block_shape=(BLOCK, SCALE_COLS),
        layout=SHARED_A_SCALE,
    )
    b_scale_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_scale_ptr + off_n * stride_bsn,
        shape=(N - off_n, K // 32),
        strides=(stride_bsn, 1),
        block_shape=(BLOCK, SCALE_COLS),
        layout=SHARED_B_SCALE,
    )
    for slot in gl.static_range(2):
        _bk256_issue_scales(
            a_scale_desc, b_scale_desc, a_scale_smem, b_scale_smem, slot
        )
        a_scale_desc, b_scale_desc = _bk256_advance_scales(a_scale_desc, b_scale_desc)
        _bk256_issue_data(a_desc, b_desc, a_smem, b_smem, slot)
        a_desc, b_desc = _bk256_advance_data(a_desc, b_desc)
    _bk256_issue_scales(a_scale_desc, b_scale_desc, a_scale_smem, b_scale_smem, 2)
    a_scale_desc, b_scale_desc = _bk256_advance_scales(a_scale_desc, b_scale_desc)
    acc0 = gl.zeros((BLOCK, BLOCK // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK, BLOCK // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    num_k_tiles = gl.cdiv(K, 256)
    num_k_tiles = gl.maximum(num_k_tiles + num_k_tiles % 2, 4)
    gl.assume(num_k_tiles >= 4)
    gl.assume(num_k_tiles % 2 == 0)
    # Six tiles cycle both the two data slots and the three scale slots.
    for _ in range(0, (num_k_tiles - 4) // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, a_scale_desc, b_scale_desc, acc0, acc1 = (
                _bk256_consume_refill(
                    a_smem,
                    b_smem,
                    a_scale_smem,
                    b_scale_smem,
                    inner % 2,
                    inner % 3,
                    a_desc,
                    b_desc,
                    a_scale_desc,
                    b_scale_desc,
                    acc0,
                    acc1,
                    dot_a,
                    dot_b,
                    B2_LOAD,
                    scale_a,
                    scale_b,
                    scale_b_load,
                    BLOCK,
                    CLUSTER,
                    True,
                    CLUSTER_SYNC,
                )
            )
    for pair in range(0, (num_k_tiles - 4) % 6 // 2):
        for half in gl.static_range(2):
            a_desc, b_desc, a_scale_desc, b_scale_desc, acc0, acc1 = (
                _bk256_consume_refill(
                    a_smem,
                    b_smem,
                    a_scale_smem,
                    b_scale_smem,
                    half,
                    (pair * 2 + half) % 3,
                    a_desc,
                    b_desc,
                    a_scale_desc,
                    b_scale_desc,
                    acc0,
                    acc1,
                    dot_a,
                    dot_b,
                    B2_LOAD,
                    scale_a,
                    scale_b,
                    scale_b_load,
                    BLOCK,
                    CLUSTER,
                    True,
                    CLUSTER_SYNC,
                )
            )
    # The last two refilling tiles bring in the final data tiles; only the
    # first of them still has a scale tile left to load.
    scale_slot0 = (num_k_tiles - 4) % 3
    scale_slot1 = (num_k_tiles - 3) % 3
    a_desc, b_desc, a_scale_desc, b_scale_desc, acc0, acc1 = _bk256_consume_refill(
        a_smem,
        b_smem,
        a_scale_smem,
        b_scale_smem,
        0,
        scale_slot0,
        a_desc,
        b_desc,
        a_scale_desc,
        b_scale_desc,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD,
        scale_a,
        scale_b,
        scale_b_load,
        BLOCK,
        CLUSTER,
        True,
        CLUSTER_SYNC,
    )
    a_desc, b_desc, a_scale_desc, b_scale_desc, acc0, acc1 = _bk256_consume_refill(
        a_smem,
        b_smem,
        a_scale_smem,
        b_scale_smem,
        1,
        scale_slot1,
        a_desc,
        b_desc,
        a_scale_desc,
        b_scale_desc,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD,
        scale_a,
        scale_b,
        scale_b_load,
        BLOCK,
        CLUSTER,
        False,
        CLUSTER_SYNC,
    )
    gl.amd.cdna5.tdm.async_wait(3)
    acc0, acc1 = _bk256_consume_tail(
        a_smem,
        b_smem,
        a_scale_smem,
        b_scale_smem,
        0,
        (num_k_tiles - 2) % 3,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD,
        scale_a,
        scale_b,
        scale_b_load,
        BLOCK,
        CLUSTER,
    )
    gl.amd.cdna5.tdm.async_wait(0)
    acc0, acc1 = _bk256_consume_tail(
        a_smem,
        b_smem,
        a_scale_smem,
        b_scale_smem,
        1,
        (num_k_tiles - 1) % 3,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD,
        scale_a,
        scale_b,
        scale_b_load,
        BLOCK,
        CLUSTER,
    )
    if CLUSTER_SYNC:
        gl.amd.cdna5.cluster.arrive()
        gl.amd.cdna5.cluster.wait()
    a_smem._keep_alive()
    b_smem._keep_alive()
    a_scale_smem._keep_alive()
    b_scale_smem._keep_alive()
    # The output is viewed as [N / 256, M, 256]; the two 128-column halves of
    # every CTA tile are one box each, clipped at the M and N edges.
    out_smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [CLUSTER, BLOCK, 128], SHARED_OUT
    )
    out_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=out_ptr + off_m * stride_om + off_n,
        shape=((N - off_n) // 256, M - off_m, 256),
        strides=(256, stride_om, 1),
        block_shape=(CLUSTER, BLOCK, 128),
        layout=SHARED_OUT,
    )
    out_smem.store(
        acc0.reshape((BLOCK, CLUSTER, 128))
        .permute((1, 0, 2))
        .to(out_ptr.dtype.element_ty)
    )
    gl.amd.cdna5.tdm.async_store(out_desc, [0, 0, 0], out_smem)
    gl.amd.cdna5.tdm.async_wait(0)
    out_smem.store(
        acc1.reshape((BLOCK, CLUSTER, 128))
        .permute((1, 0, 2))
        .to(out_ptr.dtype.element_ty)
    )
    gl.amd.cdna5.tdm.async_store(out_desc, [0, 0, 128], out_smem)
    gl.amd.cdna5.tdm.async_wait(0)


def _bk512_split_k(m: int, n: int, k: int) -> int:
    k_tiles = k // _BK512
    if m > 64 or k % _BK512 or k_tiles < 8:
        return 1
    ctas = triton.cdiv(n, 64)
    return next(
        (
            s
            for s in range(k_tiles // 2, 1, -1)
            if k_tiles % s == 0 and ctas * s <= _NUM_CUS
        ),
        1,
    )


def _cluster_mm_config(
    m: int, n: int, k: int, *, fp32_scales: bool
) -> tuple[int, int, int]:
    """Return the clustered kernel's ``(tile, cluster, split_k)`` for a shape.

    A 256 tile is the BK256 kernel, which needs UE8M0 scales and N divisible
    by 256; it takes batches from 512 rows whose 256 x 256 tiles exceed half
    the CUs. The BK512 kernel takes the rest. It uses 64-row tiles while
    those fit one wave of CUs, then 128-row tiles. Both run unclustered up to
    128 rows, where the cluster barriers cost more than the shared loads
    save. 128-row tiles also run unclustered for two K tiles, too few to hide
    the cluster's first multicast; a single K tile clusters, which measured
    faster from 512 rows. A single row of 64-row tiles over at least eight K
    tiles splits K into the most partitions of two or more tiles that still
    fit one wave of CUs, and an FP32 reduction writes the output.
    """
    if (
        not fp32_scales
        and m >= _BK256_MIN_M
        and n % _BK256_TILE == 0
        and triton.cdiv(m, 512) * triton.cdiv(n, 512) * 4 > _NUM_CUS // 2
    ):
        return _BK256_TILE, 2, 1
    if triton.cdiv(m, 128) * triton.cdiv(n, 128) * 4 <= _NUM_CUS:
        if m > 128:
            return 64, 2, 1
        return 64, 1, _bk512_split_k(m, n, k)
    return 128, 1 if m <= 128 or _BK512 < k <= 2 * _BK512 else 2, 1


def _launch_cluster_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    tile: int,
    cluster: int,
    split_k: int,
    fp32_scales: bool,
) -> None:
    """Run the clustered kernel ``_cluster_mm_config`` chose into ``out``."""
    m, k = A.shape
    n = B.shape[0]
    if tile == _BK256_TILE and (fp32_scales or n % _BK256_TILE):
        raise ValueError(
            f"BK256 requires UE8M0 scales and N divisible by {_BK256_TILE}, got N={n}"
        )
    if split_k > 1 and (
        tile == _BK256_TILE or cluster > 1 or k % _BK512 or (k // _BK512) % split_k
    ):
        raise ValueError(
            f"split-K {split_k} needs an unclustered BK512 tile and K divisible "
            f"into 512-wide partitions, got tile={tile}, cluster={cluster}, K={k}"
        )
    block = tile * cluster
    grid = (triton.cdiv(m, block) * triton.cdiv(n, block), split_k)
    if tile != _BK256_TILE:
        partials = out
        if split_k > 1:
            partials = torch.empty(
                (split_k, m, n), dtype=torch.float32, device=A.device
            )
        gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250[grid](
            A,
            B,
            A_scales,
            B_scales,
            partials,
            A.stride(0),
            B.stride(0),
            A_scales.stride(0),
            B_scales.stride(0),
            partials.stride(0) if split_k > 1 else 0,
            partials.stride(-2),
            m,
            n,
            k,
            TILE=tile,
            CLUSTER=cluster,
            SPLIT_K=split_k,
            GROUP_M=_GROUP_M,
            NUM_XCDS=_NUM_XCDS,
            FP32_SCALES=fp32_scales,
            SINGLE_K_TILE=k // split_k <= _BK512,
            num_warps=8,
            num_ctas=cluster * cluster,
            waves_per_eu=2,
        )
        if split_k > 1:
            _mxfp8_split_reduce_gfx1250[(m * triton.cdiv(n, _REDUCE_BLOCK_N),)](
                partials,
                out,
                partials.stride(0),
                partials.stride(1),
                partials.stride(2),
                out.stride(0),
                out.stride(1),
                M=m,
                N=n,
                SPLIT_K=split_k,
                BLOCK_N=_REDUCE_BLOCK_N,
                num_warps=1,
                waves_per_eu=1,
            )
        return
    gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250[grid](
        A,
        B,
        A_scales,
        B_scales,
        out,
        A.stride(0),
        B.stride(0),
        A_scales.stride(0),
        B_scales.stride(0),
        out.stride(0),
        m,
        n,
        k,
        CLUSTER=cluster,
        GROUP_M=_GROUP_M,
        NUM_XCDS=_NUM_XCDS,
        num_warps=8,
        num_ctas=cluster * cluster,
        waves_per_eu=2,
    )


def launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project E4M3 matrices with row-major ``[1,32]`` raw UE8M0 scales.

    Args:
        A: Row-major E4M3 activation matrix ``[M,K]``; K divisible by 32.
        B: Row-major E4M3 weight matrix ``[N,K]``; N divisible by 16.
        A_scales: Raw UE8M0 bytes ``[M,K/32]`` with unit column stride.
        B_scales: Raw UE8M0 bytes ``[N,K/32]`` with unit column stride.
        out_dtype: Explicit output dtype; only BF16 is supported.
        alpha: Unsupported global output scale.
        block_size: Explicit scale block, which must be ``[1,32]``.
        out: Optional BF16 output buffer ``[M,N]`` with unit column stride.

    Returns:
        The caller-owned ``out`` tensor or a newly allocated BF16 result.
    """
    if alpha is not None:
        raise ValueError("gfx1250 MXFP8 GEMM does not support alpha")
    if block_size is None or tuple(block_size) != (1, 32):
        raise ValueError("gfx1250 UE8M0 MXFP8 GEMM requires block_size=[1, 32]")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    m, k = A.shape
    n = B.shape[0]
    if B.shape[1] != k:
        raise ValueError(f"B must be [N, {k}], got {tuple(B.shape)}")
    for name, tensor in (("A", A), ("B", B)):
        if (
            tensor.dtype != torch.float8_e4m3fn
            or not tensor.is_cuda
            or tensor.device != A.device
            or tensor.stride(-1) != 1
        ):
            raise ValueError(f"{name} must be row-major GPU E4M3 colocated with A")
    if A_scales is None or B_scales is None:
        raise ValueError("A_scales and B_scales are required")
    for name, scales, shape in (
        ("A_scales", A_scales, (m, triton.cdiv(k, 32))),
        ("B_scales", B_scales, (n, triton.cdiv(k, 32))),
    ):
        if (
            scales.dtype != torch.uint8
            or tuple(scales.shape) != shape
            or scales.device != A.device
            or scales.stride(-1) != 1
        ):
            raise ValueError(
                f"{name} must be row-major GPU uint8 {shape} colocated with A"
            )
    if out_dtype != torch.bfloat16:
        raise ValueError(f"gfx1250 MXFP8 GEMM requires BF16 output, got {out_dtype}")
    if m < 1 or k < 32 or k % 32 or n % 16:
        raise ValueError(
            "gfx1250 MXFP8 GEMM requires M >= 1, K divisible by 32 and N "
            f"divisible by 16, got M={m}, N={n}, K={k}"
        )
    if out is None:
        out = A.new_empty((m, n), dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or out.stride(-1) != 1
    ):
        raise ValueError(
            f"out must be row-major GPU {out_dtype} ({m}, {n}) colocated with A"
        )

    tile, cluster, split_k = _cluster_mm_config(m, n, k, fp32_scales=False)
    _launch_cluster_mm(
        A,
        B,
        A_scales,
        B_scales,
        out,
        tile=tile,
        cluster=cluster,
        split_k=split_k,
        fp32_scales=False,
    )
    return out


def launch_gluon_mm_fp8_blockscale_largem_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project E4M3 matrices with canonical FP32 ``[128,128]`` block scales.

    Args:
        A: Row-major E4M3 activation matrix ``[M,K]``; K divisible by 512.
        B: Row-major E4M3 weight matrix ``[N,K]``; N divisible by 128.
        A_scales: Per-row FP32 scales ``[M,K/128]`` with unit column stride.
        B_scales: FP32 scales ``[N/128,K/128]`` with unit column stride.
        out_dtype: Explicit output dtype; only BF16 is supported.
        alpha: Unsupported global output scale.
        block_size: Explicit scale block, which must be ``[128,128]``.
        out: Optional BF16 output buffer ``[M,N]`` with unit column stride.

    Returns:
        The caller-owned ``out`` tensor or a newly allocated BF16 result.
    """
    if alpha is not None:
        raise ValueError("gfx1250 FP8 block-scale GEMM does not support alpha")
    if block_size is None or tuple(block_size) != (128, 128):
        raise ValueError("gfx1250 FP8 block-scale GEMM requires block_size=[128, 128]")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    m, k = A.shape
    n = B.shape[0]
    if B.shape[1] != k:
        raise ValueError(f"B must be [N, {k}], got {tuple(B.shape)}")
    for name, tensor in (("A", A), ("B", B)):
        if (
            tensor.dtype != torch.float8_e4m3fn
            or not tensor.is_cuda
            or tensor.device != A.device
            or tensor.stride(-1) != 1
        ):
            raise ValueError(f"{name} must be row-major GPU E4M3 colocated with A")
    if m < 1 or k < _BK512 or k % _BK512 or n % 128:
        raise ValueError(
            "gfx1250 FP8 block-scale GEMM requires M >= 1, N divisible by 128 "
            f"and K by {_BK512}, got M={m}, N={n}, K={k}"
        )
    if A_scales is None or B_scales is None:
        raise ValueError("A_scales and B_scales are required")
    for name, scales, shape in (
        ("A_scales", A_scales, (m, k // 128)),
        ("B_scales", B_scales, (n // 128, k // 128)),
    ):
        if (
            scales.dtype != torch.float32
            or tuple(scales.shape) != shape
            or scales.device != A.device
            or scales.stride(-1) != 1
        ):
            raise ValueError(
                f"{name} must be row-major GPU FP32 {shape} colocated with A"
            )
    if out_dtype != torch.bfloat16:
        raise ValueError(
            f"gfx1250 FP8 block-scale GEMM requires BF16 output, got {out_dtype}"
        )
    if out is None:
        out = A.new_empty((m, n), dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or out.stride(-1) != 1
    ):
        raise ValueError(
            f"out must be row-major GPU {out_dtype} ({m}, {n}) colocated with A"
        )
    tile, cluster, split_k = _cluster_mm_config(m, n, k, fp32_scales=True)
    _launch_cluster_mm(
        A,
        B,
        A_scales,
        B_scales,
        out,
        tile=tile,
        cluster=cluster,
        split_k=split_k,
        fp32_scales=True,
    )
    return out


__all__ = [
    "launch_gluon_mm_fp8_blockscale_largem_gfx1250",
    "launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250",
]
