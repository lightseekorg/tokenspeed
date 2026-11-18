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

"""Shared pieces of the gfx1250 warp-pipelined FP8 x MXFP4 MoE kernels.

The kernels keep a ring of ``NUM_BUFFERS`` LDS slots per operand (FP8 X,
packed MXFP4 W, and W's preshuffled UE8M0 scales) and stream K tiles into it
with TDM. X may be gathered by routing index (gate/up) or read as contiguous
expert-sorted rows (down). A cluster of ``CTA_M`` workgroups may share one
expert's W through multicast; each workgroup then owns ``BLOCK_M / CTA_M`` rows
of the cluster tile.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon

tdm = gl.amd.cdna5.tdm

# MXFP4 weights carry one UE8M0 scale per 32 K values; the gfx1250 scale
# layout groups 128 N rows and four K scales (see weight_preprocess.py).
SCALE_BLOCK = 32
SCALE_PRESHUFFLE_N = 128
SCALE_PRESHUFFLE_K = 4


@gluon.jit
def load_scale(scale_ptr):
    return 1.0 if scale_ptr is None else gl.load(scale_ptr)


@gluon.constexpr_function
def _bitwidth(dtype):
    if isinstance(dtype, gl.pointer_type):
        dtype = dtype.element_ty
    return dtype.primitive_bitwidth


@gluon.constexpr_function
def get_blocked_layout(shape, dtype, num_warps, cga_layout=(), max_vector_bytes=None):
    """Row-major register layout for storing a 2-D output tile.

    ``max_vector_bytes`` caps the contiguous bytes per lane; stores straight to
    global memory use 16 so that one b128 store covers whole 128-byte lines.
    """
    inner_dim = shape[-1]
    assert inner_dim in (32, 64, 128, 256, 512)
    bitwidth = _bitwidth(dtype)
    vector_size = max(inner_dim // (8 * (bitwidth // 8)), 4)
    if max_vector_bytes is not None:
        vector_size = max(min(vector_size, max_vector_bytes * 8 // bitwidth), 1)
    return gl.BlockedLayout(
        [1, vector_size],
        [4, 8],
        [num_warps // 2, 2],
        [1, 0],
        cga_layout=list(cga_layout),
    )


@gluon.constexpr_function
def make_cluster_layout(cta_m, axis, broadcast):
    """CGA bases that split ``axis`` over ``cta_m`` workgroups, or replicate."""
    assert cta_m in (1, 2, 4, 8)
    bases = []
    value = 1
    while value < cta_m:
        base = [0, 0]
        if not broadcast:
            base[axis] = value
        bases.append(base)
        value *= 2
    return bases


@gluon.constexpr_function
def get_tdm_gather_scatter_idx_layout(num_indices, num_warps, cta_m=1):
    """Partition TDM row indices over the warps and the cluster workgroups."""
    assert num_indices % cta_m == 0
    local_num_indices = num_indices // cta_m
    assert local_num_indices % num_warps == 0
    return gl.BlockedLayout(
        [1, local_num_indices // num_warps],
        [32, 1],
        [1, num_warps],
        [0, 1],
        cga_layout=make_cluster_layout(cta_m, axis=1, broadcast=False),
    )


@gluon.constexpr_function
def _append_power_of_two_bases(bases, count, axis, base_stride):
    value = 1
    stride = base_stride
    while value < count:
        base = [0, 0]
        base[axis] = stride
        bases.append(base)
        value *= 2
        stride *= 2
    return bases


@gluon.constexpr_function
def _get_wmma_layout(block_m, block_n, num_warps, instr_shape, n_tiles_per_warp):
    """WMMA layout with 2 x ``n_tiles_per_warp`` register tiles per warp."""
    assert num_warps in (4, 8)
    if block_m == 16 and block_n == 128 and num_warps == 8:
        reg_bases = []
        warp_bases = [[0, 1], [0, 2], [0, 4]]
    else:
        tiles_per_warp_m = 2
        tiles_per_warp_n = n_tiles_per_warp or tiles_per_warp_m
        reg_bases = _append_power_of_two_bases([], tiles_per_warp_n, 1, 1)
        reg_bases = _append_power_of_two_bases(reg_bases, tiles_per_warp_m, 0, 1)
        warp_bases = _append_power_of_two_bases([], num_warps // 2, 1, tiles_per_warp_n)
        warp_bases = _append_power_of_two_bases(warp_bases, 2, 0, tiles_per_warp_m)
    return gl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape)


@gluon.constexpr_function
def _get_compact_partitioned_wmma_layout(
    slice_m, slice_n, num_warps, instr_shape, n_tiles_per_warp
):
    """Partition-aware WMMA layout that keeps the compact per-warp footprint.

    Like ``make_partitioned_dot_layouts``, warp-id bit 0 moves half a slice in
    both M and N, so the two warps that issue together read opposite halves
    of the M- and N-partitioned operands. Each warp still computes
    ``n_tiles_per_warp`` consecutive N tiles by the remaining M tiles, so
    per-warp LDS reads and registers match the unpartitioned layout.
    """
    tiles_m = slice_m // instr_shape[0]
    tiles_n = slice_n // instr_shape[1]
    warp_split_n = num_warps // 2
    assert tiles_n == warp_split_n * n_tiles_per_warp
    reg_bases = _append_power_of_two_bases([], n_tiles_per_warp, 1, 1)
    reg_bases = _append_power_of_two_bases(reg_bases, tiles_m // 2, 0, 1)
    warp_bases = [[tiles_m // 2, tiles_n // 2]]
    warp_bases = _append_power_of_two_bases(
        warp_bases, warp_split_n // 2, 1, n_tiles_per_warp
    )
    warp_bases = _append_power_of_two_bases(warp_bases, 2, 1, tiles_n // 2)
    return gl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape)


@aggregate
class WarpPipelineConfig:
    """Tile shape, ring depth and layouts of one FP8 x MXFP4 MoE kernel.

    X is FP8 E4M3 with K contiguous; W is packed MXFP4 stored N-major with K
    contiguous (``W_TRANSPOSE``) and preshuffled UE8M0 scales.
    """

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    NUM_WARPS: gl.constexpr
    DIV_FACTOR_X: gl.constexpr
    DIV_FACTOR_W: gl.constexpr
    DTYPE_X: gl.constexpr
    DTYPE_W: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    W_CACHE_MODIFIER: gl.constexpr
    GATHER_PRODUCER_WARPS: gl.constexpr
    RESOLVE_PARTITION_CONFLICTS: gl.constexpr

    SCALE_BLOCK: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    SCALE_KWIDTH: gl.constexpr

    NUM_SUBTILES_M: gl.constexpr
    NUM_SUBTILES_N: gl.constexpr
    NUM_SUBTILES_K: gl.constexpr
    USE_GATHER: gl.constexpr
    CTA_M: gl.constexpr

    shared_layout_x: gl.constexpr
    dot_layout_x: gl.constexpr
    shared_layout_w: gl.constexpr
    dot_layout_w: gl.constexpr
    shared_layout_w_scale: gl.constexpr
    layout_w_scale: gl.constexpr
    acc_layout: gl.constexpr
    ACC_CGA_LAYOUT: gl.constexpr

    # TDM descriptor row-index width, not the width of pointer arithmetic.
    index_type: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        NUM_BUFFERS,
        index_type,
        NUM_SUBTILES,
        USE_GATHER,
        NUM_WARPS,
        RESOLVE_PARTITION_CONFLICTS,
        GATHER_PRODUCER_WARPS=0,
        CTA_M=1,
        W_CACHE_MODIFIER="",
        WMMA_N_TILES_PER_WARP=None,
        DECODE_W_PARTITION=0,
        DECODE_X_PARTITION=0,
    ):
        assert DTYPE_X == "e4m3" and DTYPE_W == "e2m1"
        assert CTA_M in (1, 2, 4, 8) and BLOCK_M % CTA_M == 0
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.DIV_FACTOR_X = gl.constexpr(1)
        self.DIV_FACTOR_W = gl.constexpr(2)
        self.DTYPE_X = gl.constexpr(DTYPE_X)
        self.DTYPE_W = gl.constexpr(DTYPE_W)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.W_CACHE_MODIFIER = gl.constexpr(W_CACHE_MODIFIER or "")
        self.GATHER_PRODUCER_WARPS = gl.constexpr(GATHER_PRODUCER_WARPS)
        self.RESOLVE_PARTITION_CONFLICTS = gl.constexpr(RESOLVE_PARTITION_CONFLICTS)
        self.NUM_SUBTILES_M = gl.constexpr(NUM_SUBTILES[0])
        self.NUM_SUBTILES_N = gl.constexpr(NUM_SUBTILES[1])
        self.NUM_SUBTILES_K = gl.constexpr(NUM_SUBTILES[2])
        self.USE_GATHER = gl.constexpr(USE_GATHER)
        self.CTA_M = gl.constexpr(CTA_M)
        self.index_type = gl.constexpr(index_type)

        block_k_scale = BLOCK_K // SCALE_BLOCK
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.SCALE_KWIDTH = gl.constexpr(min(SCALE_PRESHUFFLE_K, block_k_scale))
        self.PRESHUFFLE_FACTOR = gl.constexpr(SCALE_PRESHUFFLE_N)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // SCALE_PRESHUFFLE_N)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(
            block_k_scale * SCALE_PRESHUFFLE_N
        )

        pad_vec = 16
        dot_k_width = 16
        instr_m = 16
        block_k_packed_x = BLOCK_K
        block_k_packed_w = BLOCK_K // 2
        if USE_GATHER:
            # TDM gather needs the padding interval to divide the K tile.
            pad_interval_x = block_k_packed_x
        else:
            pad_interval_x = 256 if block_k_packed_x <= 256 else block_k_packed_x
        pad_interval_w = 256 if block_k_packed_w <= 256 else block_k_packed_w
        cga_layout_x = make_cluster_layout(CTA_M, axis=0, broadcast=False)
        # Every workgroup of a cluster reads the same W tile (multicast).
        cga_layout_w = make_cluster_layout(CTA_M, axis=0, broadcast=True)

        padded_x = gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval_x, pad_vec]],
            [BLOCK_M, block_k_packed_x],
            [1, 0],
            cga_layout_x,
        )
        padded_w = gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval_w, pad_vec]],
            [BLOCK_N, block_k_packed_w],
            [1, 0],
            cga_layout_w,
        )

        local_block_m = BLOCK_M // CTA_M
        if RESOLVE_PARTITION_CONFLICTS:
            # Split each operand tile along M (X) or N (W) into pieces placed in
            # the two LDS partitions, with a WMMA layout whose paired warps read
            # opposite partitions.
            slice_m = local_block_m // NUM_SUBTILES[0]
            slice_n = BLOCK_N // NUM_SUBTILES[1]
            shared_x, shared_w, wmma_layout = gl.amd.cdna5.make_partitioned_dot_layouts(
                local_block_m,
                BLOCK_N,
                padded_x,
                padded_w,
                NUM_WARPS,
                [instr_m, 16, 128],
                a_transposed=False,
                b_transposed=True,
                slice_m=slice_m,
                slice_n=slice_n,
            )
            if WMMA_N_TILES_PER_WARP is not None:
                # Down projection: partitioned X and W with the compact
                # per-warp footprint.
                wmma_layout = _get_compact_partitioned_wmma_layout(
                    slice_m,
                    slice_n,
                    NUM_WARPS,
                    [instr_m, 16, 128],
                    WMMA_N_TILES_PER_WARP,
                )
            elif USE_GATHER and NUM_WARPS == 8 and slice_m == 128 and slice_n == 512:
                # One 128 x 512 dot per K slice (gate/up): the gathered X splits
                # into its two 64-row halves and W into its two 256-row halves,
                # one per LDS partition, so the W copy takes two warps and still
                # fuses with the W-scale copy. Each warp keeps 4 x 8 WMMA tiles;
                # warp bit 0 steps M and N by one half each, so the paired warps
                # 0 and 1 read different partitions of X and W in every
                # instruction. The register N steps are 32 and 64 rows (warp bit 1
                # steps 128), so a thread's four W-scale rows are adjacent words
                # of the preshuffled layout and load with one ds_load_b128.
                wmma_layout = gl.amd.AMDWMMALayout(
                    3,
                    True,
                    [[4, 16], [0, 8], [2, 0]],
                    [[0, 1], [1, 0], [0, 2], [4, 0], [0, 4]],
                    [instr_m, 16, 128],
                )
            shared_layout_x = shared_x
            shared_layout_w = shared_w
        elif DECODE_W_PARTITION or DECODE_X_PARTITION:
            # Decode 16 x 128 tile, one 16-column N tile per warp. Two LDS
            # requests are served in one cycle only when they target different
            # partitions, so the operands are spread over both:
            #   W_PARTITION 1: W's two N halves in opposite partitions, and
            #                  warp bit 0 steps half of BLOCK_N, so the two warps
            #                  that issue together read different partitions.
            #   W_PARTITION 2 / X_PARTITION 2: four K pieces alternating between
            #                  the partitions, so consecutive WMMA K steps of a
            #                  wave read opposite partitions.
            #   X_PARTITION 1: the two K halves in opposite partitions.
            assert local_block_m == 16 and BLOCK_N == 128 and NUM_WARPS == 8
            assert DECODE_W_PARTITION in (0, 1, 2) and DECODE_X_PARTITION in (0, 1, 2)
            assert not (
                DECODE_X_PARTITION and USE_GATHER
            ), "TDM gathers cannot write a partitioned X"
            warp_bases = (
                [[0, 4], [0, 1], [0, 2]]
                if DECODE_W_PARTITION == 1
                else [[0, 1], [0, 2], [0, 4]]
            )
            wmma_layout = gl.amd.AMDWMMALayout(
                3, True, warp_bases, [], [instr_m, 16, 128]
            )

            def k_pieces(rows, packed_k, groups, cga_layout):
                piece = packed_k // (2 * groups)
                inner = gl.PaddedSharedLayout.with_identity_for(
                    [[piece, pad_vec]], [rows, piece], [1, 0], cga_layout
                )
                return gl.amd.cdna5.PartitionedSharedLayout(2, groups, 1, inner)

            if DECODE_W_PARTITION == 1:
                inner_w = gl.PaddedSharedLayout.with_identity_for(
                    [[pad_interval_w, pad_vec]],
                    [BLOCK_N // 2, block_k_packed_w],
                    [1, 0],
                    cga_layout_w,
                )
                shared_layout_w = gl.amd.cdna5.PartitionedSharedLayout(2, 1, 0, inner_w)
            elif DECODE_W_PARTITION == 2:
                shared_layout_w = k_pieces(BLOCK_N, block_k_packed_w, 2, cga_layout_w)
            else:
                shared_layout_w = padded_w
            if DECODE_X_PARTITION:
                shared_layout_x = k_pieces(
                    BLOCK_M, block_k_packed_x, DECODE_X_PARTITION, cga_layout_x
                )
            else:
                shared_layout_x = padded_x
        else:
            wmma_layout = _get_wmma_layout(
                local_block_m,
                BLOCK_N,
                NUM_WARPS,
                [instr_m, 16, 128],
                WMMA_N_TILES_PER_WARP,
            )
            shared_layout_x = padded_x
            shared_layout_w = padded_w
        # Packed E2M1 operands use a 64-wide K instruction tile.
        wmma_layout_packed = gl.amd.AMDWMMALayout(
            3,
            True,
            wmma_layout.warp_bases,
            wmma_layout.reg_bases,
            [instr_m, 16, 64],
            cga_layout_x,
        )
        wmma_layout = gl.amd.AMDWMMALayout(
            3,
            True,
            wmma_layout.warp_bases,
            wmma_layout.reg_bases,
            wmma_layout.instr_shape,
            cga_layout_x,
        )
        self.shared_layout_x = gl.constexpr(shared_layout_x)
        self.shared_layout_w = gl.constexpr(shared_layout_w)
        self.dot_layout_x = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=0, parent=wmma_layout, k_width=dot_k_width
            )
        )
        self.dot_layout_w = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=1, parent=wmma_layout_packed, k_width=dot_k_width
            )
        )
        self.layout_w_scale = gl.constexpr(
            gl.amd.cdna5.get_wmma_scale_layout(
                self.dot_layout_w,
                [BLOCK_N // NUM_SUBTILES[1], block_k_scale // NUM_SUBTILES[2]],
            )
        )
        self.acc_layout = gl.constexpr(wmma_layout)
        self.ACC_CGA_LAYOUT = gl.constexpr(cga_layout_x)
        self.shared_layout_w_scale = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[256, 8]],
                [self.BLOCK_N_PRESHUFFLED, self.BLOCK_K_SCALE_PRESHUFFLED],
                [1, 0],
                cga_layout_w,
            )
        )


@gluon.jit
def create_descriptors(
    cfg: WarpPipelineConfig,
    x_ptr,
    w_ptr,
    w_scale_ptr,
    off_m,
    w_offs,
    w_scale_offs,
    M,
    eM,
    rem_n,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_w_scale_n,
    stride_w_scale_k,
    GatherIndx,
    start_m,
    gather_base=None,
    gather_rows=None,
):
    """TDM descriptors of one output tile's X, W and W scales.

    ``rem_n`` is the number of W rows from this N tile to the end of N, so a
    trailing partial N tile reads zeros past it. Gathered X rows of a tile
    past the expert's end point one row beyond the source, which TDM fills
    with zeros, instead of touching real rows.
    """
    if cfg.USE_GATHER:
        NUM_INDICES: gl.constexpr = cfg.BLOCK_M
        if cfg.GATHER_PRODUCER_WARPS > 0:
            # Partition the indices over one warp-pipeline group and make the
            # other group a redundant copy. The gather lowering predicates the
            # redundant warp dimension off, so only the low producer group
            # writes the LDS ring.
            gl.static_assert(cfg.NUM_WARPS % cfg.GATHER_PRODUCER_WARPS == 0)
            gl.static_assert(NUM_INDICES % (cfg.GATHER_PRODUCER_WARPS * cfg.CTA_M) == 0)
            REDUNDANT_WARP_COPIES: gl.constexpr = (
                cfg.NUM_WARPS // cfg.GATHER_PRODUCER_WARPS
            )
            IDX_PARENT_LAYOUT: gl.constexpr = gl.BlockedLayout(
                [NUM_INDICES // (cfg.GATHER_PRODUCER_WARPS * cfg.CTA_M), 1],
                [1, 32],
                [cfg.GATHER_PRODUCER_WARPS, REDUNDANT_WARP_COPIES],
                # The index dimension is minor so its warp bits are warp-id bits
                # 0/1; slicing the dummy dimension then makes bit 2 redundant,
                # selecting the contiguous warps 0..3.
                [0, 1],
                cga_layout=make_cluster_layout(cfg.CTA_M, axis=0, broadcast=False),
            )
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(1, IDX_PARENT_LAYOUT)
        else:
            IDX_BASE_LAYOUT: gl.constexpr = get_tdm_gather_scatter_idx_layout(
                NUM_INDICES, cfg.NUM_WARPS, cfg.CTA_M
            )
            IDX_LAYOUT: gl.constexpr = gl.SliceLayout(0, IDX_BASE_LAYOUT)

        offs_m_gather = off_m + gl.arange(0, NUM_INDICES, IDX_LAYOUT)
        if gather_base is None:
            inactive_row = M + 1
        else:
            # Indices relative to the tile's lowest row; inactive lanes again
            # point one row past the end of that view.
            inactive_row = gather_base + gather_rows + 1
        if off_m + NUM_INDICES <= eM:
            # Every row of the tile is active: one unmasked load, without a
            # compare and branch per index.
            gathered_m = gl.load(GatherIndx + start_m + offs_m_gather)
        else:
            gathered_m = gl.load(
                GatherIndx + start_m + offs_m_gather,
                mask=offs_m_gather < eM,
                other=inactive_row,
            )

        if gather_base is None:
            gathered_m = gathered_m.to(cfg.index_type)
            x_desc = tdm.make_tensor_descriptor(
                base=x_ptr,
                shape=(M, K),
                strides=(stride_xm, stride_xk),
                block_shape=(NUM_INDICES, cfg.BLOCK_K),
                layout=cfg.shared_layout_x,
            )
        else:
            gathered_m = (gathered_m - gather_base).to(cfg.index_type)
            x_desc = tdm.make_tensor_descriptor(
                base=x_ptr + gather_base.to(off_m.dtype) * stride_xm,
                shape=(gather_rows, K),
                strides=(stride_xm, stride_xk),
                block_shape=(NUM_INDICES, cfg.BLOCK_K),
                layout=cfg.shared_layout_x,
            )
    else:
        gathered_m = gl.constexpr(0)
        x_desc = tdm.make_tensor_descriptor(
            base=x_ptr + off_m * stride_xm,
            shape=(eM - off_m.to(gl.int32), K),
            strides=(stride_xm, stride_xk),
            block_shape=(cfg.BLOCK_M, cfg.BLOCK_K),
            layout=cfg.shared_layout_x,
        )

    w_desc = tdm.make_tensor_descriptor(
        base=w_ptr + w_offs,
        shape=(rem_n, K // cfg.DIV_FACTOR_W),
        strides=(stride_wn, stride_wk),
        block_shape=(cfg.BLOCK_N, cfg.BLOCK_K // cfg.DIV_FACTOR_W),
        layout=cfg.shared_layout_w,
    )
    # The preshuffled scales are padded to whole 128-row groups and to a
    # multiple of four K scales; the descriptor must cover that padding.
    k_scale = gl.cdiv(K, cfg.SCALE_BLOCK)
    k_scale_padded = gl.cdiv(k_scale, cfg.SCALE_KWIDTH) * cfg.SCALE_KWIDTH
    w_scale_desc = tdm.make_tensor_descriptor(
        base=w_scale_ptr + w_scale_offs,
        shape=(
            gl.cdiv(rem_n, cfg.PRESHUFFLE_FACTOR),
            k_scale_padded * cfg.PRESHUFFLE_FACTOR,
        ),
        strides=(stride_w_scale_n, stride_w_scale_k),
        block_shape=(cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED),
        layout=cfg.shared_layout_w_scale,
    )
    return x_desc, w_desc, w_scale_desc, gathered_m


@aggregate
class RingProgram:
    """The LDS ring and TDM descriptors of the output tile being computed."""

    cfg: WarpPipelineConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    w_scale_buffer: gl.shared_memory_descriptor
    x_desc: tdm.tensor_descriptor
    w_desc: tdm.tensor_descriptor
    w_scale_desc: tdm.tensor_descriptor
    gathered_m: gl.tensor | gl.constexpr
    off_k_x: gl.tensor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        x_buffer,
        w_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        w_scale_desc,
        gathered_m,
        off_k_x,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.w_scale_buffer = w_scale_buffer
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.w_scale_desc = w_scale_desc
        # Constexpr zeros arrive here unwrapped.
        self.gathered_m = (
            gathered_m
            if isinstance(gathered_m, gl.tensor)
            else gl.constexpr(gathered_m)
        )
        self.off_k_x = (
            off_k_x if isinstance(off_k_x, gl.tensor) else gl.constexpr(off_k_x)
        )

    @gluon.jit
    def cluster_sync(self):
        if self.cfg.CTA_M > 1:
            gl.amd.cdna5.cluster.arrive()
            gl.amd.cdna5.cluster.wait()


@gluon.jit
def allocate_ring(
    cfg: WarpPipelineConfig, x_dtype: gl.constexpr, w_dtype: gl.constexpr
):
    x_buffer = gl.allocate_shared_memory(
        x_dtype, [cfg.NUM_BUFFERS, cfg.BLOCK_M, cfg.BLOCK_K], cfg.shared_layout_x
    )
    w_buffer = gl.allocate_shared_memory(
        w_dtype,
        [cfg.NUM_BUFFERS, cfg.BLOCK_N, cfg.BLOCK_K // cfg.DIV_FACTOR_W],
        cfg.shared_layout_w,
    )
    w_scale_buffer = gl.allocate_shared_memory(
        gl.uint8,
        [cfg.NUM_BUFFERS, cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
        cfg.shared_layout_w_scale,
    )
    return x_buffer, w_buffer, w_scale_buffer


@gluon.jit
def load_w_scale_slice(
    cfg: WarpPipelineConfig,
    w_scale_slot,
    k_start: gl.constexpr,
    k_len: gl.constexpr,
    n_start: gl.constexpr,
    n_len: gl.constexpr,
):
    """Undo the preshuffle of one ring slot's scales and load a K/N slice."""
    block_k_scale: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
    w_scale = w_scale_slot.reshape(
        (
            cfg.BLOCK_N_PRESHUFFLED,
            block_k_scale // cfg.SCALE_KWIDTH,
            cfg.PRESHUFFLE_FACTOR // 4,
            4,
            cfg.SCALE_KWIDTH,
        )
    )
    w_scale = w_scale.permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, block_k_scale))
    w_scale = w_scale.slice(n_start, n_len, 0)
    w_scale = w_scale.slice(k_start // cfg.SCALE_BLOCK, k_len // cfg.SCALE_BLOCK, 1)
    return w_scale.load(layout=cfg.layout_w_scale)


@gluon.jit
def wmma_scaled(cfg: WarpPipelineConfig, x, w, w_scale, acc):
    return gl.amd.cdna5.wmma_scaled(x, None, cfg.DTYPE_X, w, w_scale, cfg.DTYPE_W, acc)


@gluon.jit
def load_bias(
    cfg: WarpPipelineConfig,
    B,
    stride_b_e,
    expt_id,
    pid_n,
    N,
    START: gl.constexpr,
    SIZE: gl.constexpr,
):
    """Bias columns [START, START + SIZE) of the tile in the accumulator's
    column layout, or zeros."""
    BIAS_LAYOUT: gl.constexpr = gl.SliceLayout(0, cfg.acc_layout)
    if B is None:
        bias = gl.full([SIZE], 0, dtype=gl.float32, layout=BIAS_LAYOUT)
    else:
        cols = cfg.BLOCK_N * pid_n + START + gl.arange(0, SIZE, BIAS_LAYOUT)
        bias = gl.load(B + expt_id * stride_b_e + cols, mask=cols < N, other=0)
    return bias


@gluon.jit
def _tanh(x):
    # 1 - 2 / (exp(2x) + 1) saturates exactly at both ends. libdevice tanh
    # costs more time and, on newer Triton, about 400 more live VGPRs across
    # the 128 x 512 epilogue.
    return 1.0 - gl.fdiv(2.0, gl.exp(2.0 * x) + 1.0, approx=True)


@gluon.jit
def _swiglu(acc, alpha: gl.constexpr, limit: gl.constexpr, beta: gl.constexpr):
    """SwiGLU of interleaved gate/up columns, as ``_swiglu_gfx1250`` but with
    an approximate (2.5 ULP) sigmoid division."""
    BLOCK_M: gl.constexpr = acc.shape[0]
    OUT_BLOCK_N: gl.constexpr = acc.shape[1] // 2
    gate, linear = gl.split(acc.reshape((BLOCK_M, OUT_BLOCK_N, 2)))
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.clamp(linear, -limit, limit)
    sigmoid = gl.fdiv(1.0, 1.0 + gl.exp(-alpha * gate), approx=True)
    return gate * sigmoid * (linear + beta)


@gluon.jit
def _situ(acc, beta: gl.constexpr, linear_beta: gl.constexpr):
    """SiTU of interleaved gate/up columns, as ``_situ_gfx1250`` but with
    approximate (2.5 ULP) divisions, which the FP8 output cast absorbs."""
    BLOCK_M: gl.constexpr = acc.shape[0]
    OUT_BLOCK_N: gl.constexpr = acc.shape[1] // 2
    inv_beta: gl.constexpr = 1.0 / beta
    inv_linear_beta: gl.constexpr = 1.0 / linear_beta
    # Match Kimi-K3's BF16 gate/up boundary before applying SiTU.
    acc = acc.to(gl.bfloat16).to(gl.float32)
    gate, linear = gl.split(acc.reshape((BLOCK_M, OUT_BLOCK_N, 2)))
    sigmoid = gl.fdiv(1.0, 1.0 + gl.exp(-gate), approx=True)
    gate = beta * _tanh(gate * inv_beta) * sigmoid
    linear = linear_beta * _tanh(linear * inv_linear_beta)
    return gate * linear


@gluon.jit
def epilogue_math(
    acc,
    bias,
    x_scale,
    y_scale_inv,
    out_dtype: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
):
    """Activation scale, bias, fused gate activation and the output cast.

    The result is divided by the output scale (multiplied by its reciprocal)
    before the cast, so an FP8 output feeds the next GEMM directly.
    """
    gl.static_assert(not (DO_SWIGLU and DO_SITU))
    # Explicit fma: gfx1250 lowers mulf/addf to packed ops without contraction.
    acc = gl.fma(acc, x_scale, bias[None, :])
    if DO_SITU:
        out = _situ(acc, SITU_BETA, SITU_LINEAR_BETA)
    elif DO_SWIGLU:
        out = _swiglu(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA)
    else:
        out = acc
    out *= y_scale_inv
    return out.to(out_dtype)


def warp_pipeline_launch_metadata(grid, kernel, args):
    """Report a routed FP8 x MXFP4 GEMM's work and modeled traffic to Proton.

    ``bytes`` counts each operand once: the routed FP8 rows, every expert's
    packed weights and scales, and the output rows.
    """
    m, n, k = args["num_rows"], args["N"], args["K"]
    w, y = args["W"], args["Y"]
    w_values = w.numel() * 2
    y_values = m * (n // args["ACTIVATION_REDUCTION_N"])
    return {
        "name": f"{kernel.name} [M={m}, N={n}, K={k}]",
        "flops8": 2 * m * n * k,
        "bytes": m * k
        + w.numel()
        + w_values // SCALE_BLOCK
        + y_values * y.element_size(),
    }


def num_compute_units(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
