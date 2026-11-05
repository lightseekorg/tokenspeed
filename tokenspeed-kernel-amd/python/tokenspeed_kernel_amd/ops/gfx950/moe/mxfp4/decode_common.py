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


"""Direct CDNA4 MFMA primitives shared by the gfx950 A4W4 decode stages.

Tile addressing, MXFP4 operand loads, and the scaled-MFMA op used by both
production decode stages (``decode_stage1.py`` / ``decode_stage2.py``). They
assume the intermediate rows are in (token, topk-slot) order, so no ragged
metadata / scatter indices are consumed. ``situ_decode.py`` reuses the CDNA4
scale-offset helper and the compact scaled-upcast scale tile for its in-situ
dequantization path.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import MXFP4_BLOCK


@gluon.jit
def _gluon_dot_preshuffled_w_offset(w_expert_off, k_pack, n_col, n_phys):
    """Return flat byte offset for the 128x128 Gluon-dot W layout."""
    k_in_block = k_pack % 128
    n_in_block = n_col % 128

    k_within = k_in_block % 16
    k_quad = (k_in_block // 16) % 4
    k_block = k_in_block // 64
    n_in_sub = n_in_block % 16
    n_block = n_in_block // 16

    in_tile = (
        n_block.to(gl.int64) * 2048
        + k_block.to(gl.int64) * 1024
        + k_quad.to(gl.int64) * 256
        + n_in_sub.to(gl.int64) * 16
        + k_within.to(gl.int64)
    )
    n_tiles = n_phys // 128
    tile_id = (k_pack // 128).to(gl.int64) * n_tiles + (n_col // 128).to(gl.int64)
    return w_expert_off + tile_id * (128 * 128) + in_tile


@gluon.jit
def _cdna4_swizzled_mxfp4_scale_offset(
    scale_expert_off,
    n_col,
    k_scale,
    stride_slin,
    stride_snb,
):
    """Return byte offset for CDNA4-swizzled e8m0 scales.

    The preprocessor stores scales as ``(E, K_scale_padded * 32, N_padded / 32)``
    with ``stride(-2) == 1``.  This is the scalar form of the unswizzle used by
    the reference MFMA path.
    """
    n_block = n_col // 32
    n_mix = (n_col % 16) * 4 + ((n_col % 32) // 16)
    k_lin = (
        (k_scale // 8).to(gl.int64) * 256
        + (k_scale % 4).to(gl.int64) * 64
        + ((k_scale % 8) // 4).to(gl.int64) * 2
        + n_mix.to(gl.int64)
    )
    return scale_expert_off + k_lin * stride_slin + n_block.to(gl.int64) * stride_snb


# One v_cvt_scalef32_pk_bf16_fp4 converts eight packed FP4 values under a
# single scale, so a lane must cover whole groups of that size.
_SCALED_UPCAST_GROUP = 8


@gluon.constexpr_function
def _compact_mxfp4_scale_tile(expanded_layout, axis):
    """Return the compact e8m0 scale tile for a CDNA4 scaled-upcast result.

    ``gl.amd.cdna4.scaled_upcast`` takes a scale tensor whose extent along the
    scaled axis merely has to divide the result's, so a lane can hold one scale
    per ``v_cvt_scalef32_pk_bf16_fp4`` group instead of one per upcast element.
    The op infers that operand's layout from the result's by folding away the
    register bases the division makes redundant, which for a blocked result
    layout is the same layout with a smaller ``size_per_thread`` on ``axis``.

    A lane that already covers a whole 32-element MXFP4 block keeps one scale
    per block, so the tile is exactly the block scales the tile needs. A lane
    covering fewer elements keeps one scale per lane instead, and the lanes
    within a block re-read its byte; that is still one load and one register
    per lane rather than one per upcast element.

    Args:
        expanded_layout: ``gl.BlockedLayout`` of the scaled-upcast result tile.
        axis: Scaled axis of the upcast, as passed to ``scaled_upcast``.

    Returns:
        A ``(layout, group)`` pair: the layout the compact scale tile must be
        loaded in, and the number of result elements each of its entries
        covers, so the caller can address one scale byte per ``group``
        elements.

    Raises:
        ValueError: If a lane does not cover whole scaled-upcast groups along
            ``axis``, which leaves a group without a single scale to apply.
    """
    per_lane = expanded_layout.size_per_thread[axis]
    if per_lane % _SCALED_UPCAST_GROUP:
        raise ValueError(
            "scaled upcast needs every lane to cover whole "
            f"{_SCALED_UPCAST_GROUP}-element groups along axis {axis}, but "
            f"each lane covers {per_lane} elements"
        )
    group = min(per_lane, MXFP4_BLOCK)
    size_per_thread = list(expanded_layout.size_per_thread)
    size_per_thread[axis] = per_lane // group
    return (
        gl.BlockedLayout(
            size_per_thread,
            expanded_layout.threads_per_warp,
            expanded_layout.warps_per_cta,
            expanded_layout.order,
        ),
        group,
    )


@gluon.constexpr_function
def _direct_mxfp4_mfma_layouts(m_dup, block_n, block_k_scale, num_warps):
    mfma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [m_dup, block_k_scale])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [block_n, block_k_scale])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _direct_mxfp4_load_tile(
    kt,
    ak,
    bk,
    ask,
    bsk,
    am,
    asm,
    x_ptr,
    x_scale_ptr,
    w_ptr,
    w_scale_ptr,
    x_row_off,
    w_expert_off,
    s_expert_off,
    n_cols,
    n_cols_s,
    x_scale_row,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    stride_slin,
    stride_snb,
    N_PHYS,
    K_DIM,
    N_DIM,
    K_PACKED: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
):
    """Load one direct MXFP4xMXFP4 K tile into MFMA operand layouts."""
    a, a_scale = _direct_mxfp4_load_activation_tile(
        kt,
        ak,
        ask,
        am,
        asm,
        x_ptr,
        x_scale_ptr,
        x_row_off,
        x_scale_row,
        stride_xk,
        stride_xslin,
        stride_xsnb,
        K_DIM,
        K_PACKED,
        BLOCK_K_PACKED,
        BLOCK_K_SCALE,
    )
    b, b_scale = _direct_mxfp4_load_weight_tile(
        kt,
        bk,
        bsk,
        w_ptr,
        w_scale_ptr,
        w_expert_off,
        s_expert_off,
        n_cols,
        n_cols_s,
        stride_slin,
        stride_snb,
        N_PHYS,
        K_DIM,
        N_DIM,
        K_PACKED,
        BLOCK_K_PACKED,
        BLOCK_K_SCALE,
    )
    return a, b, a_scale, b_scale


@gluon.jit
def _direct_mxfp4_load_weight_tile(
    kt,
    bk,
    bsk,
    w_ptr,
    w_scale_ptr,
    w_expert_off,
    s_expert_off,
    n_cols,
    n_cols_s,
    stride_slin,
    stride_snb,
    N_PHYS,
    K_DIM,
    N_DIM,
    K_PACKED: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
):
    """Load one K tile of the gdot128-shuffled weight and its scales."""
    k_pack_b = kt * BLOCK_K_PACKED + bk
    k_scale_b = kt * BLOCK_K_SCALE + bsk
    b_off = _gluon_dot_preshuffled_w_offset(w_expert_off, k_pack_b, n_cols, N_PHYS)
    b_scale_off = _cdna4_swizzled_mxfp4_scale_offset(
        s_expert_off,
        n_cols_s,
        k_scale_b,
        stride_slin,
        stride_snb,
    )
    b = gl.amd.cdna4.buffer_load(
        ptr=w_ptr,
        offsets=b_off.to(gl.int32),
        mask=(n_cols < N_DIM) & (k_pack_b < K_PACKED),
        other=0,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=w_scale_ptr,
        offsets=b_scale_off.to(gl.int32),
        mask=(n_cols_s < N_DIM) & (k_scale_b < (K_DIM // 32)),
        other=127,
    )
    return b, b_scale


@gluon.jit
def _direct_mxfp4_load_activation_tile(
    kt,
    ak,
    ask,
    am,
    asm,
    x_ptr,
    x_scale_ptr,
    x_row_off,
    x_scale_row,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    K_DIM,
    K_PACKED: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
):
    """Load one K tile of the packed activation rows and their scales."""
    k_pack_a = kt * BLOCK_K_PACKED + ak
    k_scale_a = kt * BLOCK_K_SCALE + ask
    a_off = x_row_off + k_pack_a.to(gl.int64) * stride_xk + am.to(gl.int64) * 0
    a_scale_off = _cdna4_swizzled_mxfp4_scale_offset(
        0,
        x_scale_row + asm * 0,
        k_scale_a,
        stride_xslin,
        stride_xsnb,
    )
    a = gl.amd.cdna4.buffer_load(
        ptr=x_ptr,
        offsets=a_off.to(gl.int32),
        mask=k_pack_a < K_PACKED,
        other=0,
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=x_scale_ptr,
        offsets=a_scale_off.to(gl.int32),
        mask=k_scale_a < (K_DIM // 32),
        other=127,
    )
    return a, a_scale


@gluon.jit
def _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale):
    return gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_scale,
        a_format="e2m1",
        b=b,
        b_scale=b_scale,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _direct_mxfp4_k_loop(
    acc,
    ak,
    bk,
    ask,
    bsk,
    am,
    asm,
    x_ptr,
    x_scale_ptr,
    w_ptr,
    w_scale_ptr,
    x_row_off,
    w_expert_off,
    s_expert_off,
    n_cols,
    n_cols_s,
    x_scale_row,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    stride_slin,
    stride_snb,
    N_PHYS,
    K_DIM,
    N_DIM,
    K_PACKED: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
    PREFETCH_TILES: gl.constexpr,
):
    """Accumulate every K tile in order into ``acc``.

    ``PREFETCH_TILES`` is how many K tiles' loads are in flight ahead of the
    MFMA: 0 loads and multiplies one tile at a time; 1 holds one tile in
    VGPRs while the next tile's loads are issued; more unrolls the loop and,
    in the prologue, issues all prefetched weight tiles before the activation
    tiles, since their addresses never wait on the activation rows. The MFMA
    order is the same either way, so results are bitwise identical.
    """
    NUM_KT: gl.constexpr = K_PACKED // BLOCK_K_PACKED
    if PREFETCH_TILES > 1 and NUM_KT > 1:
        DEPTH: gl.constexpr = min(PREFETCH_TILES, NUM_KT - 1)
        weights = ()
        for kt in gl.static_range(DEPTH):
            weights = weights + (
                _direct_mxfp4_load_weight_tile(
                    kt,
                    bk,
                    bsk,
                    w_ptr,
                    w_scale_ptr,
                    w_expert_off,
                    s_expert_off,
                    n_cols,
                    n_cols_s,
                    stride_slin,
                    stride_snb,
                    N_PHYS,
                    K_DIM,
                    N_DIM,
                    K_PACKED,
                    BLOCK_K_PACKED,
                    BLOCK_K_SCALE,
                ),
            )
        activations = ()
        for kt in gl.static_range(DEPTH):
            activations = activations + (
                _direct_mxfp4_load_activation_tile(
                    kt,
                    ak,
                    ask,
                    am,
                    asm,
                    x_ptr,
                    x_scale_ptr,
                    x_row_off,
                    x_scale_row,
                    stride_xk,
                    stride_xslin,
                    stride_xsnb,
                    K_DIM,
                    K_PACKED,
                    BLOCK_K_PACKED,
                    BLOCK_K_SCALE,
                ),
            )
        for kt in gl.static_range(NUM_KT):
            if kt + DEPTH < NUM_KT:
                weights = weights + (
                    _direct_mxfp4_load_weight_tile(
                        kt + DEPTH,
                        bk,
                        bsk,
                        w_ptr,
                        w_scale_ptr,
                        w_expert_off,
                        s_expert_off,
                        n_cols,
                        n_cols_s,
                        stride_slin,
                        stride_snb,
                        N_PHYS,
                        K_DIM,
                        N_DIM,
                        K_PACKED,
                        BLOCK_K_PACKED,
                        BLOCK_K_SCALE,
                    ),
                )
                activations = activations + (
                    _direct_mxfp4_load_activation_tile(
                        kt + DEPTH,
                        ak,
                        ask,
                        am,
                        asm,
                        x_ptr,
                        x_scale_ptr,
                        x_row_off,
                        x_scale_row,
                        stride_xk,
                        stride_xslin,
                        stride_xsnb,
                        K_DIM,
                        K_PACKED,
                        BLOCK_K_PACKED,
                        BLOCK_K_SCALE,
                    ),
                )
            acc = _direct_mxfp4_mfma(
                acc,
                activations[kt][0],
                weights[kt][0],
                activations[kt][1],
                weights[kt][1],
            )
    elif PREFETCH_TILES == 1 and NUM_KT > 1:
        a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
            0,
            ak,
            bk,
            ask,
            bsk,
            am,
            asm,
            x_ptr,
            x_scale_ptr,
            w_ptr,
            w_scale_ptr,
            x_row_off,
            w_expert_off,
            s_expert_off,
            n_cols,
            n_cols_s,
            x_scale_row,
            stride_xk,
            stride_xslin,
            stride_xsnb,
            stride_slin,
            stride_snb,
            N_PHYS,
            K_DIM,
            N_DIM,
            K_PACKED,
            BLOCK_K_PACKED,
            BLOCK_K_SCALE,
        )
        for kt in range(0, NUM_KT - 1):
            next_a, next_b, next_a_scale, next_b_scale = _direct_mxfp4_load_tile(
                kt + 1,
                ak,
                bk,
                ask,
                bsk,
                am,
                asm,
                x_ptr,
                x_scale_ptr,
                w_ptr,
                w_scale_ptr,
                x_row_off,
                w_expert_off,
                s_expert_off,
                n_cols,
                n_cols_s,
                x_scale_row,
                stride_xk,
                stride_xslin,
                stride_xsnb,
                stride_slin,
                stride_snb,
                N_PHYS,
                K_DIM,
                N_DIM,
                K_PACKED,
                BLOCK_K_PACKED,
                BLOCK_K_SCALE,
            )
            acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
            a, b, a_scale, b_scale = next_a, next_b, next_a_scale, next_b_scale
        acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
    else:
        for kt in range(NUM_KT):
            a, b, a_scale, b_scale = _direct_mxfp4_load_tile(
                kt,
                ak,
                bk,
                ask,
                bsk,
                am,
                asm,
                x_ptr,
                x_scale_ptr,
                w_ptr,
                w_scale_ptr,
                x_row_off,
                w_expert_off,
                s_expert_off,
                n_cols,
                n_cols_s,
                x_scale_row,
                stride_xk,
                stride_xslin,
                stride_xsnb,
                stride_slin,
                stride_snb,
                N_PHYS,
                K_DIM,
                N_DIM,
                K_PACKED,
                BLOCK_K_PACKED,
                BLOCK_K_SCALE,
            )
            acc = _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale)
    return acc


@gluon.constexpr_function
def _route_layout(num_warps):
    return gl.BlockedLayout([1], [64], [num_warps], [0])


@gluon.jit
def _route_groups_from_ids(
    ids,
    rows,
    EXPERT_START: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    TOPK: gl.constexpr,
    MAX_ROWS: gl.constexpr,
):
    """Group the routes of at most ``MAX_ROWS`` tokens by local expert,
    straight from their top-k IDs.

    A token lists an expert at most once, so each expert holds at most
    ``MAX_ROWS`` routes. Groups follow local expert order. Returns each
    route's local expert (-1 for another rank's routes and past ``rows``),
    the per-expert route counts, each expert's group index and the number of
    groups.
    """
    ROUTE_BLOCK: gl.constexpr = triton.next_power_of_2(MAX_ROWS * TOPK)
    EXPERT_PAD: gl.constexpr = triton.next_power_of_2(NUM_EXPERTS)
    layout: gl.constexpr = _route_layout(gl.num_warps())
    route = gl.arange(0, ROUTE_BLOCK, layout=layout)
    expert = gl.load(ids + route, route < rows * TOPK, other=-1) - EXPERT_START
    valid = (route < rows * TOPK) & (expert >= 0) & (expert < NUM_EXPERTS)
    counts = gl.histogram(
        gl.where(valid, expert, 0), EXPERT_PAD, mask=valid, layout=layout
    )
    active = (counts > 0).to(gl.int32)
    group_index = gl.cumsum(active, 0) - active
    return gl.where(valid, expert, -1), counts, group_index, gl.sum(active, 0)


@gluon.jit
def _group_routes(
    route_expert,
    counts,
    group_index,
    group,
    BLOCK_M: gl.constexpr,
):
    """Return ``(expert, count, routes)`` of one group: its local expert,
    its route count and its routes in route order, padded with its first
    route (``[BLOCK_M]``, at least the group's count)."""
    EXPERT_PAD: gl.constexpr = counts.shape[0]
    ROUTE_BLOCK: gl.constexpr = route_expert.shape[0]
    layout: gl.constexpr = _route_layout(gl.num_warps())
    member_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 64], [gl.num_warps(), 1], [1, 0]
    )
    in_group = (group_index == group) & (counts > 0)
    expert = gl.sum(gl.where(in_group, gl.arange(0, EXPERT_PAD, layout=layout), 0), 0)
    count = gl.sum(gl.where(in_group, counts, 0), 0)
    selected = (route_expert == expert).to(gl.int32)
    # Each selected route's position in the group; the others match no row.
    rank = gl.where(selected != 0, gl.cumsum(selected, 0) - selected, BLOCK_M)
    route = gl.arange(0, ROUTE_BLOCK, layout=layout)
    first = gl.min(gl.where(selected != 0, route, ROUTE_BLOCK), 0)
    rank = gl.convert_layout(rank, gl.SliceLayout(0, member_layout))[None, :]
    route = gl.arange(0, ROUTE_BLOCK, layout=gl.SliceLayout(0, member_layout))[None, :]
    row = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, member_layout))
    routes = gl.sum(gl.where(rank == row[:, None], route, 0), 1)
    return expert, count, gl.where(row < count, routes, first)
