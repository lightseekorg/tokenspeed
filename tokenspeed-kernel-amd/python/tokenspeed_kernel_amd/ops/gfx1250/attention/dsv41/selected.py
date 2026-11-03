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

"""Fused two-reader DeepSeek V4.1 selected attention for GFX1250.

Wave32 WMMA port of the GFX950 kernel. Reads page-planar SWA (E4M3/E8M0,
528B) and global (E2M1/E4M3, 288B) caches directly. One online softmax
includes both segments and a single sink.

A workgroup owns one query row, up to 64 heads and a contiguous share of the
row's selected tiles, so each selected row is loaded and dequantized once per
16-head group that needs it. Head groups whose queries are all zero (the
64/128-head padding of the decode layout) skip the score WMMAs: their scores
are exactly zero, so their output is the V row sum scaled per head by the
sink, computed once for all such groups. Small batches split the selected
tiles over workgroups and merge the partial softmax states in a reduce kernel.
See the "DeepSeek V4.1 selected attention" section of ``ops/README.md``.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["launch_gluon_dsv41_selected_attention_gfx1250"]

_SWA_ROW_BYTES = 528
_GLOBAL_ROW_BYTES = 288
_PAGE_ROWS = 64
_HEAD_DIM = 512
_BLOCK_H = 16
# Four waves split a tile's rows for QK, 16 rows each.
_TILE_K = 64
# Rows are split over workgroups only while the grid still fits the 256 CUs
# at two workgroups each.
_TARGET_CTAS = 512
_MAX_SPLITS = 16
# The reduce splits head dims over workgroups too: one workgroup per token and
# 16-head group alone is too few to stream a small batch's partials.
_REDUCE_DIM_BLOCK = 64
# One workgroup covers at most 64 query heads (four 16-head groups).
_HEAD_GROUPS_PER_CTA = 4
# Fewer head groups per workgroup re-read each selected row once more per head
# block. That pays off while rows x head blocks x tiles stays within this
# budget (about 2.5 tiles per CU).
_HEAD_BLOCK_TILES = 640
# Raw-row LDS ring depth; deeper rings measured slower at decode shapes.
_STAGES = 2


@gluon.jit
def _load_slots(
    swa_slots,
    swa_row,
    swa_width,
    global_slots,
    global_row,
    global_width,
    SLOT_TILES: gl.constexpr,
    SWA_TILES: gl.constexpr,
    TILE_K: gl.constexpr,
    HAS_GLOBAL: gl.constexpr,
):
    """Load a query row's SWA then global slots as [SLOT_TILES, TILE_K].

    Row ``r < SWA_TILES`` holds SWA tile ``r``, row ``SWA_TILES + j`` global
    tile ``j``; positions past a width read -1. The loads are unmasked, at
    clamped positions: a mask on a tile row is uniform per register and
    would become a branch around every load.
    """
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    tiles = gl.arange(0, SLOT_TILES, layout=gl.SliceLayout(1, layout))
    is_swa = (tiles < SWA_TILES)[:, None]
    positions = (gl.where(tiles < SWA_TILES, tiles, tiles - SWA_TILES) * TILE_K)[
        :, None
    ] + gl.arange(0, TILE_K, layout=gl.SliceLayout(0, layout))[None, :]
    width = swa_width
    pointers = swa_slots + swa_row + gl.minimum(positions, swa_width - 1)
    if HAS_GLOBAL:
        # With no global slots, every row reads the SWA slots and is masked.
        use_global = (~is_swa) & (global_width > 0)
        width = gl.where(use_global, global_width, gl.where(is_swa, swa_width, 0))
        pointers = gl.where(
            use_global,
            global_slots + global_row + gl.minimum(positions, global_width - 1),
            pointers,
        )
    else:
        width = gl.where(is_swa, swa_width, 0)
    slot = gl.where(positions < width, gl.load(pointers), -1)
    return slot, positions, is_swa


@gluon.jit
def _stage_slots(
    slots_smem,
    slot,
    positions,
    is_swa,
    gather_rows_smem,
    swa_len,
    swa_capacity,
    swa_rows_per_page,
    global_len,
    global_capacity,
    global_rows_per_page,
):
    """Store loaded slots into LDS rows of TILE_K.

    gather_rows_smem gets each position's value row in the TDM row view of
    its cache, -1 if unusable.
    """
    slots_smem.store(slot)
    length = gl.where(is_swa, swa_len, global_len)
    capacity = gl.where(is_swa, swa_capacity, global_capacity)
    rows_per_page = gl.where(is_swa, swa_rows_per_page, global_rows_per_page)
    valid = (positions < length) & (slot >= 0) & (slot < capacity)
    gather_rows_smem.store(
        gl.where(valid, (slot >> 6) * rows_per_page + (slot & 63), -1)
    )


@gluon.jit
def _staged_slots(slots_smem, smem_row, first, bound, capacity, TILE_K, LAYOUT):
    """Return (clamped slot, valid) of one staged tile in LAYOUT.

    ``first`` is the tile's first position in its segment and ``bound`` the
    segment's end for this split (0 for a padding tile).
    """
    positions = first + gl.arange(0, TILE_K, layout=LAYOUT)
    slot = slots_smem.index(smem_row).load(LAYOUT)
    valid = (positions < bound) & (slot >= 0) & (slot < capacity)
    return gl.where(valid, slot, 0), valid


@gluon.constexpr_function
def _warp_split_row_layout(tile_k, num_warps, warp_size):
    # TDM gathers take lane-uniform row indices; each warp gathers its own rows.
    rows_per_warp = tile_k // num_warps
    return gl.DistributedLinearLayout(
        reg_bases=[[1 << i] for i in range(rows_per_warp.bit_length() - 1)],
        lane_bases=[[0]] * (warp_size.bit_length() - 1),
        warp_bases=[[rows_per_warp << i] for i in range(num_warps.bit_length() - 1)],
        block_bases=[],
        shape=[tile_k],
    )


@gluon.jit
def _gather_values(
    value_desc,
    value_rows,
    gather_rows_smem,
    smem_row,
    padding,
    values_smem,
    ROW_LAYOUT: gl.constexpr,
):
    """Start a TDM gather of one tile's staged value rows into LDS.

    The cache is viewed as ``value_rows`` rows of the value width. Unusable
    rows (-1) and every row of a padding tile index one past the end, which
    the gather zero-fills.
    """
    rows = gather_rows_smem.index(smem_row).load(ROW_LAYOUT)
    # Branch-free clamp: TDM takes the rows in SGPRs, and a lane-mask select
    # per row would cost a VALU->SALU dependency each.
    rows = gl.minimum((rows | padding).to(gl.uint32), value_rows.to(gl.uint32))
    gl.amd.cdna5.tdm.async_gather(value_desc, rows.to(gl.int32), values_smem)


@gluon.jit
def _copy_scales(
    scales_smem,
    is_swa,
    swa_cache,
    swa_page_stride,
    global_cache,
    global_page_stride,
    slots_smem,
    smem_row,
    first,
    bound,
    capacity,
    TILE_K: gl.constexpr,
    HAS_GLOBAL: gl.constexpr,
):
    """Start one commit group copying a tile's scales into LDS [TILE_K, 32].

    Scale rows are too narrow for an int32 TDM row view of a large cache, so
    they take one unmasked async copy (a masked one would branch around the
    copy and break the group count). Invalid rows read row 0, whose scales
    the dequant keeps finite. SWA rows fill both halves with their 16 scales.
    """
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [4, 1], [1, 0])
    slot, _ = _staged_slots(
        slots_smem,
        smem_row,
        first,
        bound,
        capacity,
        TILE_K,
        gl.SliceLayout(1, layout),
    )
    page = (slot // 64).to(tl.int64)[:, None]
    row = (slot % 64)[:, None]
    cols = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))[None, :]
    source = swa_cache + page * swa_page_stride + 64 * 512 + row * 16 + cols % 16
    if HAS_GLOBAL:
        source = gl.where(
            is_swa,
            source,
            global_cache + page * global_page_stride + 64 * 256 + row * 32 + cols,
        )
    gl.amd.cdna5.async_copy.global_to_shared(scales_smem, source)
    gl.amd.cdna5.async_copy.commit_group()


@gluon.jit
def _global_stage(values_ring, buffer, LAYOUT: gl.constexpr):
    """Ring stage ``buffer`` viewed as a global tile [TILE_K, 256].

    SWA and global tiles share each stage, so a global tile at a split's
    SWA-to-global boundary never overlaps the other stage.
    """
    return values_ring.index(buffer).reinterpret(
        gl.uint8, [values_ring.shape[1], 256], LAYOUT
    )


@gluon.jit
def _tile_coords(
    tile,
    num_tiles,
    swa_tiles,
    swa_first,
    swa_bound,
    swa_capacity,
    global_first,
    global_bound,
    global_capacity,
    SWA_TILES: gl.constexpr,
    GLOBAL_TILES: gl.constexpr,
    TILE_K: gl.constexpr,
):
    """Map the split's tile index to (slot LDS row, first, bound, capacity).

    The split's first ``swa_tiles`` tiles are SWA tiles; tiles from
    ``num_tiles`` on are padding with bound 0.
    """
    is_swa = tile < swa_tiles
    swa_tile = swa_first + tile
    global_tile = global_first + tile - swa_tiles
    smem_row = gl.where(
        is_swa,
        gl.minimum(swa_tile, SWA_TILES - 1),
        SWA_TILES + gl.minimum(gl.maximum(global_tile, 0), GLOBAL_TILES - 1),
    )
    first = gl.where(is_swa, swa_tile, global_tile) * TILE_K
    bound = gl.where(tile < num_tiles, gl.where(is_swa, swa_bound, global_bound), 0)
    capacity = gl.where(is_swa, swa_capacity, global_capacity)
    return smem_row, first, bound, capacity


@gluon.jit
def _issue_tile(
    tile,
    num_tiles,
    swa_tiles,
    swa_first,
    swa_bound,
    global_first,
    global_bound,
    slots_smem,
    gather_rows_smem,
    swa_capacity,
    global_capacity,
    swa_cache,
    swa_page_stride,
    global_cache,
    global_page_stride,
    swa_value_desc,
    swa_value_rows,
    global_value_desc,
    global_value_rows,
    values_ring,
    scales_ring,
    SWA_TILES: gl.constexpr,
    GLOBAL_TILES: gl.constexpr,
    STAGES: gl.constexpr,
    TILE_K: gl.constexpr,
    HAS_GLOBAL: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
    GLOBAL_LAYOUT: gl.constexpr,
):
    """Start the split's tile into its ring buffer: one TDM gather of values
    from the tile's own segment (padding tiles gather only zero-filled rows)
    and one scale commit group."""
    smem_row, first, bound, capacity = _tile_coords(
        tile,
        num_tiles,
        swa_tiles,
        swa_first,
        swa_bound,
        swa_capacity,
        global_first,
        global_bound,
        global_capacity,
        SWA_TILES,
        GLOBAL_TILES,
        TILE_K,
    )
    buffer = tile % STAGES
    is_swa = tile < swa_tiles
    padding = -(tile >= num_tiles).to(gl.int32)
    if HAS_GLOBAL:
        if is_swa:
            _gather_values(
                swa_value_desc,
                swa_value_rows,
                gather_rows_smem,
                smem_row,
                padding,
                values_ring.index(buffer),
                ROW_LAYOUT,
            )
        else:
            _gather_values(
                global_value_desc,
                global_value_rows,
                gather_rows_smem,
                smem_row,
                padding,
                _global_stage(values_ring, buffer, GLOBAL_LAYOUT),
                ROW_LAYOUT,
            )
    else:
        _gather_values(
            swa_value_desc,
            swa_value_rows,
            gather_rows_smem,
            smem_row,
            padding,
            values_ring.index(buffer),
            ROW_LAYOUT,
        )
    _copy_scales(
        scales_ring.index(buffer),
        is_swa,
        swa_cache,
        swa_page_stride,
        global_cache,
        global_page_stride,
        slots_smem,
        smem_row,
        first,
        bound,
        capacity,
        TILE_K,
        HAS_GLOBAL,
    )


@gluon.jit
def _dequant_swa_tile(values_smem, scales_smem, WMMA_LAYOUT: gl.constexpr):
    """E4M3 [TILE_K, 512] times E8M0 per 32 dims into the QK operand [512, TILE_K].

    The upcast runs at a uniform unit scale: a per-lane E8M0 scale reaches the
    wrong lanes on GFX1250. The power-of-two scale then multiplies exactly.
    """
    k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )
    values = values_smem.permute([1, 0]).load(k_layout)
    unit = gl.full(values.shape, 127, dtype=gl.uint8, layout=k_layout)
    kv = gl.amd.cdna5.scaled_upcast(
        values.to(gl.float8e4nv, bitcast=True), unit, gl.bfloat16
    )
    kv = kv.reshape([16, 32, kv.shape[1]])
    scale_layout: gl.constexpr = gl.SliceLayout(1, kv.type.layout)
    scales = scales_smem.permute([1, 0]).slice(0, 16, 0).load(scale_layout)
    # E8M0 byte s is 2^(s - 127): BF16 exponent bits, with s = 0 as 2^-127.
    # The NaN byte 0xFF (unwritten rows) takes 2^-127 too, so the zero values
    # of invalid rows stay zero.
    bits = gl.where((scales == 0) | (scales == 0xFF), 0x40, scales.to(gl.int16) << 7)
    kv = kv * bits.to(gl.bfloat16, bitcast=True)[:, None, :]
    return gl.convert_layout(
        kv.reshape([512, kv.shape[2]]), k_layout, assert_trivial=True
    )


@gluon.jit
def _dequant_global_tile(values_smem, scales_smem, WMMA_LAYOUT: gl.constexpr):
    """E2M1 [TILE_K, 256] (even dim in the low nibble) times E4M3 per 16 dims
    into the QK operand [512, TILE_K].

    The upcast runs at unit E8M0 scale. E4M3 converts exactly to BF16, and an
    E2M1 value times an E4M3 scale needs at most five significant bits, so the
    product is exact before its single BF16 rounding.
    """
    k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )
    packed_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=4
    )
    values = values_smem.permute([1, 0]).load(packed_layout)
    unit = gl.full(
        [2 * values.shape[0], values.shape[1]], 127, dtype=gl.uint8, layout=k_layout
    )
    kv = gl.amd.cdna5.scaled_upcast(values, unit, gl.bfloat16, axis=0)
    kv = kv.reshape([32, 16, kv.shape[1]])
    scale_layout: gl.constexpr = gl.SliceLayout(1, kv.type.layout)
    scales = scales_smem.permute([1, 0]).load(scale_layout)
    # NaN scales (0x7F, 0xFF; unwritten rows) read as zero, so the zero values
    # of invalid rows stay zero.
    scales = gl.where((scales & 0x7F) == 0x7F, 0, scales)
    kv = kv * scales.to(gl.float8e4nv, bitcast=True).to(gl.bfloat16)[:, None, :]
    return gl.convert_layout(
        kv.reshape([512, kv.shape[2]]), k_layout, assert_trivial=True
    )


@gluon.jit
def _attend_tile(
    k_dot,
    kv_shared,
    valid_rows,
    q_dot,
    valid_heads,
    softmax_scale,
    max_value,
    denominator,
    accumulator,
    row_sum,
    row_count,
    active,
    with_row_sum,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
):
    """Fold a dequantized tile into the softmax.

    k_dot is its BF16 QK operand [512, TILE_K] and kv_shared holds the same
    tile as [TILE_K, 512] for PV.
    """
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    # V is read from LDS only after QK, so the K and V operands are never live
    # together.
    # Invalid rows dequantize to zero, so all-one weights give the V row sum.
    ones = gl.full([BLOCK_H, TILE_K], 1.0, dtype=gl.bfloat16, layout=p_dot_layout)
    if active:
        scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=WMMA_LAYOUT)
        scores = gl.amd.cdna5.wmma(q_dot, k_dot, scores) * softmax_scale
        scores = gl.where(
            valid_heads[:, None] & valid_rows[None, :], scores, -float("inf")
        )
        tile_max = gl.max(scores, axis=1)
        next_max = gl.maximum(max_value, tile_max)
        safe_next_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        previous_scale = gl.exp(max_value - safe_next_max)
        probabilities = gl.exp(scores - safe_next_max[:, None])
        denominator = previous_scale * denominator + gl.sum(probabilities, axis=1)
        accumulator *= previous_scale[:, None]
        p_dot = gl.convert_layout(probabilities.to(gl.bfloat16), p_dot_layout)
        v_dot = kv_shared.load(k_dot_layout)
        accumulator = gl.amd.cdna5.wmma(p_dot, v_dot, accumulator)
        max_value = next_max
        if with_row_sum:
            row_sum = gl.amd.cdna5.wmma(ones, v_dot, row_sum)
    elif with_row_sum:
        row_sum = gl.amd.cdna5.wmma(ones, kv_shared.load(k_dot_layout), row_sum)
    if with_row_sum:
        row_count += gl.sum(valid_rows.to(gl.int32), axis=0)
    return max_value, denominator, accumulator, row_sum, row_count


@gluon.jit
def _bit_or(a, b):
    return a | b


@gluon.jit
def _inverse_rope(values, dims, rows, rope_row, dim_base, ROPE_DIM: gl.constexpr):
    """Rotate the interleaved RoPE tail of BF16 [rows, dims] output tiles.

    dims are absolute head dims; dim_base localizes the gather to the tile.
    Only the last ROPE_DIM of 512 rotate. rope_row points at the token's
    inverse cos/sin table row.
    """
    start: gl.constexpr = 512 - ROPE_DIM
    rope = dims >= start
    pair = gl.where(rope, (dims - start) // 2, 0)
    cosine = gl.load(rope_row + pair, mask=rope, other=1.0)
    sine = gl.load(rope_row + ROPE_DIM // 2 + pair, mask=rope, other=0.0)
    value = values.to(gl.float32)
    partner = gl.gather(
        value, ((dims - dim_base) ^ 1)[None, :] + rows[:, None] * 0, axis=1
    )
    first = value * cosine[None, :]
    second = partner * sine[None, :]
    rotated = gl.where(((dims & 1) != 0)[None, :], first + second, first - second)
    return gl.where(rope[None, :], rotated, value).to(gl.bfloat16)


def _selected_attention_metadata(grid, kernel, args):
    """Report the attention FLOPs and DRAM bytes of one launch.

    Device lengths are not read, so the counts are a capacity bound: every
    selected slot is valid and every head is computed. Split launches store
    one FP16 ``[heads, 512]`` partial per split instead of the output.
    """
    tokens = args["q"].shape[0]
    heads = args["num_heads"]
    splits = grid[1]
    swa_width = args["swa_width"]
    global_width = args["global_width"] if args["HAS_GLOBAL"] else 0
    output_bytes = heads * _HEAD_DIM * 2 * splits
    return {
        "name": kernel.name,
        "flops16": 4 * tokens * heads * (swa_width + global_width) * _HEAD_DIM,
        "bytes": tokens
        * (
            swa_width * (_SWA_ROW_BYTES + 4)
            + global_width * (_GLOBAL_ROW_BYTES + 4)
            + heads * _HEAD_DIM * 2
            + output_bytes
        ),
    }


def _selected_attention_reduce_metadata(grid, kernel, args):
    """Report the split reduce's DRAM bytes: per token and head, it reads
    ``num_splits`` FP16 partial rows and writes one BF16 output row."""
    tokens, heads = grid[0], args["num_heads"]
    return {
        "name": kernel.name,
        "bytes": tokens * heads * _HEAD_DIM * 2 * (args["num_splits"] + 1),
    }


@gluon.jit(launch_metadata=_selected_attention_metadata)
def gluon_dsv41_selected_attention_gfx1250(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    out,
    part_stats,
    part_acc,
    part_row_sum,
    part_count,
    part_active,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    swa_page_stride: tl.int64,
    swa_slot_stride: tl.int64,
    swa_width,
    global_page_stride: tl.int64,
    global_slot_stride: tl.int64,
    global_width,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    softmax_scale: tl.float32,
    swa_capacity,
    global_capacity,
    swa_value_rows,
    global_value_rows,
    num_heads,
    HAS_GLOBAL: gl.constexpr,
    HEAD_GROUPS: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SWA_TILES: gl.constexpr,
    GLOBAL_TILES: gl.constexpr,
    SLOT_TILES: gl.constexpr,
    STAGES: gl.constexpr,
):
    # Program (t, s, b) covers query row t, the s-th contiguous share of its
    # selected tiles (SWA tiles first, then global tiles) and head block b.
    # With one split it adds the sink and writes the output; otherwise it
    # writes partial softmax state for the reduce kernel. Four waves split a
    # tile's rows for QK and the head dims for PV.
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    gather_row_layout: gl.constexpr = _warp_split_row_layout(TILE_K, 4, 32)
    raw_shared_1d: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    raw_shared_2d: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    # The QK operand reads 16 value rows at once; an 8-byte pad per 256 bytes
    # puts them in different banks. Both views of the shared value ring need
    # the same padding.
    swa_ring_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [TILE_K, 512], [1, 0]
    )
    global_ring_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [TILE_K, 256], [1, 0]
    )
    score_row_layout: gl.constexpr = gl.SliceLayout(0, wmma)
    head_layout: gl.constexpr = gl.SliceLayout(1, wmma)
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [TILE_K, HEAD_DIM], [1, 0]
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma, k_width=8
    )

    token_idx = gl.program_id(axis=0)
    split_idx = gl.program_id(axis=1)
    num_splits = gl.num_programs(axis=1)
    head_block = gl.program_id(axis=2)
    head_base = head_block * (HEAD_GROUPS * BLOCK_H)
    single = num_splits == 1
    swa_row = token_idx.to(tl.int64) * swa_slot_stride
    if HAS_GLOBAL:
        global_row = token_idx.to(tl.int64) * global_slot_stride
    else:
        global_row = token_idx.to(tl.int64) * 0
    # The zero-query check reads every group's query as int32 words at once.
    # A mask on the head rows is uniform per register and would become a
    # branch around every load, so out-of-range rows load a clamped head and
    # are dropped.
    slot, slot_positions, slot_is_swa = _load_slots(
        swa_slots,
        swa_row,
        swa_width,
        global_slots,
        global_row,
        global_width,
        SLOT_TILES,
        SWA_TILES,
        TILE_K,
        HAS_GLOBAL,
    )
    check_layout: gl.constexpr = gl.BlockedLayout(
        [HEAD_GROUPS * BLOCK_H // 4, 4], [1, 32], [4, 1], [1, 0]
    )
    check_rows = gl.arange(
        0, HEAD_GROUPS * BLOCK_H, layout=gl.SliceLayout(1, check_layout)
    )
    check_heads = head_base + check_rows
    check_words = gl.arange(0, HEAD_DIM // 2, layout=gl.SliceLayout(0, check_layout))
    q_words = gl.load(
        q.to(gl.pointer_type(gl.int32), bitcast=True)
        + token_idx.to(tl.int64) * (stride_q_t // 2)
        + gl.minimum(check_heads, num_heads - 1)[:, None].to(tl.int64)
        * (stride_q_h // 2)
        + check_words[None, :]
    )
    swa_len = gl.minimum(gl.maximum(gl.load(swa_lens + token_idx), 0), swa_width)
    swa_tiles = gl.cdiv(swa_len, TILE_K)
    if HAS_GLOBAL:
        global_len = gl.minimum(
            gl.maximum(gl.load(global_lens + token_idx), 0), global_width
        )
    else:
        global_len = 0
    total_tiles = swa_tiles + gl.cdiv(global_len, TILE_K)
    tile_begin = split_idx * total_tiles // num_splits
    tile_end = (split_idx + 1) * total_tiles // num_splits
    # This split's SWA tiles [swa_first, swa_stop) and global tiles
    # [global_first, global_stop); rows past its share read as invalid.
    swa_first = gl.minimum(tile_begin, swa_tiles)
    swa_stop = gl.minimum(tile_end, swa_tiles)
    swa_bound = gl.minimum(swa_len, swa_stop * TILE_K)
    global_first = gl.maximum(tile_begin, swa_tiles) - swa_tiles
    global_stop = gl.maximum(tile_end, swa_tiles) - swa_tiles
    global_bound = gl.minimum(global_len, global_stop * TILE_K)
    swa_tiles_here = swa_stop - swa_first
    num_tiles = swa_tiles_here + global_stop - global_first

    # Every copy reads its slots from LDS, so issuing a tile never waits on a
    # global load.
    slots_table = gl.allocate_shared_memory(
        gl.int32, [SLOT_TILES, TILE_K], layout=raw_shared_2d
    )
    slots_smem = slots_table.reinterpret(gl.int32, [SLOT_TILES, TILE_K], raw_shared_1d)
    gather_rows_table = gl.allocate_shared_memory(
        gl.int32, [SLOT_TILES, TILE_K], layout=raw_shared_2d
    )
    gather_rows_smem = gather_rows_table.reinterpret(
        gl.int32, [SLOT_TILES, TILE_K], raw_shared_1d
    )
    _stage_slots(
        slots_table,
        slot,
        slot_positions,
        slot_is_swa,
        gather_rows_table,
        swa_len,
        swa_capacity,
        (swa_page_stride // 512).to(gl.int32),
        global_len,
        global_capacity,
        (global_page_stride // 256).to(gl.int32),
    )
    # SWA and global tiles share one value ring, which keeps the LDS of a
    # workgroup within half a CU.
    values_ring = gl.allocate_shared_memory(
        gl.uint8, [STAGES, TILE_K, 512], layout=swa_ring_layout
    )
    scales_ring = gl.allocate_shared_memory(
        gl.uint8, [STAGES, TILE_K, 32], layout=raw_shared_2d
    )
    swa_value_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=swa_cache,
        shape=[swa_value_rows, 512],
        strides=[512, 1],
        block_shape=[TILE_K, 512],
        layout=swa_ring_layout,
    )
    if HAS_GLOBAL:
        global_value_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=global_cache,
            shape=[global_value_rows, 256],
            strides=[256, 1],
            block_shape=[TILE_K, 256],
            layout=global_ring_layout,
        )
    else:
        global_value_desc = swa_value_desc

    # The split's tiles, SWA first, run through a ring of STAGES raw LDS
    # tiles: copies run STAGES - 1 tiles ahead of the tile being dequantized,
    # one value gather and one scale group per tile. The first pass's
    # leading copies overlap the query checks.
    for stage in gl.static_range(STAGES - 1):
        _issue_tile(
            stage,
            num_tiles,
            swa_tiles_here,
            swa_first,
            swa_bound,
            global_first,
            global_bound,
            slots_smem,
            gather_rows_smem,
            swa_capacity,
            global_capacity,
            swa_cache,
            swa_page_stride,
            global_cache,
            global_page_stride,
            swa_value_desc,
            swa_value_rows,
            global_value_desc,
            global_value_rows,
            values_ring,
            scales_ring,
            SWA_TILES,
            GLOBAL_TILES,
            STAGES,
            TILE_K,
            HAS_GLOBAL,
            gather_row_layout,
            global_ring_layout,
        )
    # Bit g of active_groups marks a head group with a nonzero query, which
    # needs its own pass over the selected rows. A -0.0 counts as nonzero,
    # which only costs that group the full pass.
    row_bits = gl.reduce(q_words, 1, _bit_or)
    active_groups = gl.reduce(
        gl.where(
            (row_bits != 0) & (check_heads < num_heads),
            1 << (check_rows // BLOCK_H),
            0,
        ),
        0,
        _bit_or,
    )
    need_row_sum = active_groups != (1 << HEAD_GROUPS) - 1
    kv_shared = gl.allocate_shared_memory(
        gl.bfloat16, [TILE_K, HEAD_DIM], layout=kv_shared_layout
    )

    # Outputs and partials store straight from the WMMA layout, where a lane
    # holds 8 contiguous dims of one head row.
    out_row = out + token_idx.to(tl.int64) * stride_o_t
    acc_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, wmma))
    # Partial rows are [token, split, head] ([token, split] for row sums).
    part_row = (token_idx * num_splits + split_idx).to(tl.int64)
    row_sum = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=wmma)
    row_count = 0

    for group in range(HEAD_GROUPS):
        active = ((active_groups >> group) & 1) != 0
        # Group 0 runs even when inactive: with with_row_sum, its pass also
        # accumulates the row_sum and row_count that the zero-query groups'
        # closed-form output below needs.
        if active or group == 0:
            with_row_sum = need_row_sum and group == 0
            dot_heads = (
                head_base
                + group * BLOCK_H
                + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_dot_layout))
            )
            dot_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_dot_layout))
            q_dot = gl.load(
                q
                + token_idx.to(tl.int64) * stride_q_t
                + dot_heads[:, None].to(tl.int64) * stride_q_h
                + dot_dims[None, :],
                mask=(dot_heads < num_heads)[:, None],
                other=0.0,
            )
            score_heads = (
                head_base + group * BLOCK_H + gl.arange(0, BLOCK_H, layout=head_layout)
            )
            valid_heads = score_heads < num_heads
            # One split starts from the sink (logit sink, weight 1); split
            # partials start empty and the reduce kernel adds the sink once.
            sink = gl.load(attn_sink + score_heads, mask=valid_heads, other=0.0).to(
                gl.float32
            )
            max_value = gl.where(single, sink, -float("inf"))
            denominator = gl.where(single, 1.0, 0.0) + gl.zeros(
                [BLOCK_H], dtype=gl.float32, layout=head_layout
            )
            accumulator = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=wmma)

            if group > 0:
                for stage in gl.static_range(STAGES - 1):
                    _issue_tile(
                        stage,
                        num_tiles,
                        swa_tiles_here,
                        swa_first,
                        swa_bound,
                        global_first,
                        global_bound,
                        slots_smem,
                        gather_rows_smem,
                        swa_capacity,
                        global_capacity,
                        swa_cache,
                        swa_page_stride,
                        global_cache,
                        global_page_stride,
                        swa_value_desc,
                        swa_value_rows,
                        global_value_desc,
                        global_value_rows,
                        values_ring,
                        scales_ring,
                        SWA_TILES,
                        GLOBAL_TILES,
                        STAGES,
                        TILE_K,
                        HAS_GLOBAL,
                        gather_row_layout,
                        global_ring_layout,
                    )
            for tile in range(num_tiles):
                _issue_tile(
                    tile + STAGES - 1,
                    num_tiles,
                    swa_tiles_here,
                    swa_first,
                    swa_bound,
                    global_first,
                    global_bound,
                    slots_smem,
                    gather_rows_smem,
                    swa_capacity,
                    global_capacity,
                    swa_cache,
                    swa_page_stride,
                    global_cache,
                    global_page_stride,
                    swa_value_desc,
                    swa_value_rows,
                    global_value_desc,
                    global_value_rows,
                    values_ring,
                    scales_ring,
                    SWA_TILES,
                    GLOBAL_TILES,
                    STAGES,
                    TILE_K,
                    HAS_GLOBAL,
                    gather_row_layout,
                    global_ring_layout,
                )
                gl.amd.cdna5.tdm.async_wait(STAGES - 1)
                gl.amd.cdna5.async_copy.wait_group(STAGES - 1)
                buffer = tile % STAGES
                smem_row, _, _, _ = _tile_coords(
                    tile,
                    num_tiles,
                    swa_tiles_here,
                    swa_first,
                    swa_bound,
                    swa_capacity,
                    global_first,
                    global_bound,
                    global_capacity,
                    SWA_TILES,
                    GLOBAL_TILES,
                    TILE_K,
                )
                # Tiles dequantize straight into the QK operand; LDS keeps a
                # BF16 copy only for the transposed PV read.
                if HAS_GLOBAL:
                    if tile < swa_tiles_here:
                        k_dot = _dequant_swa_tile(
                            values_ring.index(buffer),
                            scales_ring.index(buffer),
                            wmma,
                        )
                    else:
                        k_dot = _dequant_global_tile(
                            _global_stage(values_ring, buffer, global_ring_layout),
                            scales_ring.index(buffer),
                            wmma,
                        )
                else:
                    k_dot = _dequant_swa_tile(
                        values_ring.index(buffer), scales_ring.index(buffer), wmma
                    )
                kv_shared.permute([1, 0]).store(k_dot)
                score_valid = (
                    gather_rows_smem.index(smem_row).load(score_row_layout) >= 0
                )
                max_value, denominator, accumulator, row_sum, row_count = _attend_tile(
                    k_dot,
                    kv_shared,
                    score_valid,
                    q_dot,
                    valid_heads,
                    softmax_scale,
                    max_value,
                    denominator,
                    accumulator,
                    row_sum,
                    row_count,
                    active,
                    with_row_sum,
                    BLOCK_H,
                    TILE_K,
                    wmma,
                )
            # Retire the padding copies issued by the last steps.
            gl.amd.cdna5.tdm.async_wait(0)
            gl.amd.cdna5.async_copy.wait_group(0)

            if active:
                normalize = gl.where(denominator > 0.0, 1.0 / denominator, 0.0)
                if single:
                    gl.store(
                        out_row
                        + score_heads[:, None].to(tl.int64) * stride_o_h
                        + acc_dims[None, :],
                        (accumulator * normalize[:, None]).to(gl.bfloat16),
                        mask=valid_heads[:, None],
                    )
                else:
                    stats = part_stats + (part_row * num_heads + score_heads) * 2
                    gl.store(stats, max_value, mask=valid_heads)
                    gl.store(stats + 1, denominator, mask=valid_heads)
                    gl.store(
                        part_acc
                        + (part_row * num_heads + score_heads[:, None]) * HEAD_DIM
                        + acc_dims[None, :],
                        (accumulator * normalize[:, None]).to(gl.float16),
                        mask=valid_heads[:, None],
                    )

    if need_row_sum:
        if single:
            # A zero query scores exactly zero on every valid row, so its output
            # is row_sum * e^-m / (count * e^-m + e^(sink - m)), m = max(sink, 0).
            count = row_count.to(gl.float32)
            for group in range(HEAD_GROUPS):
                if ((active_groups >> group) & 1) == 0:
                    score_heads = (
                        head_base
                        + group * BLOCK_H
                        + gl.arange(0, BLOCK_H, layout=head_layout)
                    )
                    sink = gl.load(
                        attn_sink + score_heads,
                        mask=score_heads < num_heads,
                        other=0.0,
                    ).to(gl.float32)
                    frame = gl.maximum(sink, 0.0)
                    weight = gl.exp(-frame)
                    total = count * weight + gl.exp(sink - frame)
                    coefficient = gl.where(total > 0.0, weight / total, 0.0)
                    gl.store(
                        out_row
                        + score_heads[:, None].to(tl.int64) * stride_o_h
                        + acc_dims[None, :],
                        (row_sum * coefficient[:, None]).to(gl.bfloat16),
                        mask=(score_heads < num_heads)[:, None],
                    )
        else:
            # Every row of row_sum holds the same sum; head row 0 is stored.
            sum_heads = gl.arange(0, BLOCK_H, layout=head_layout)
            gl.store(
                part_row_sum
                + part_row * HEAD_DIM
                + acc_dims[None, :]
                + sum_heads[:, None] * 0,
                row_sum,
                mask=(sum_heads == 0)[:, None],
            )
            gl.store(part_count + part_row, row_count)
    if not single:
        if split_idx == 0:
            gl.store(
                part_active + token_idx * gl.num_programs(axis=2) + head_block,
                active_groups,
            )


@gluon.constexpr_function
def _reduce_layout(dim_block):
    """[splits, heads, dims] layout of the split reduce: 4 waves over heads."""
    return gl.BlockedLayout(
        [1, 1, 4], [1, 32 // (dim_block // 4), dim_block // 4], [1, 4, 1], [2, 1, 0]
    )


@gluon.jit
def _merge_partials(
    part_stats,
    part_acc,
    attn_sink,
    num_heads,
    num_splits,
    token_idx,
    group_idx,
    dim_base,
    BLOCK_H: gl.constexpr,
    SPLIT_BLOCK: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    DIM_BLOCK: gl.constexpr,
):
    """BF16 [heads, dims] merge of 16 heads' split partials and their sink.

    Partials are FP16 and normalized by their split's denominator, so each
    split weighs in with its denominator times its max rescale. All splits
    (at most SPLIT_BLOCK, a power of two) load at once, unmasked, at clamped
    splits and heads: splits and heads sit in registers, and a mask uniform
    per register would become a branch and a full wait around every load. A
    clamped split's weight is zero; clamped heads are not stored.
    """
    layout: gl.constexpr = _reduce_layout(DIM_BLOCK)
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    heads = group_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(0, stat_layout)
    )
    valid_head = heads < num_heads
    heads = gl.minimum(heads, num_heads - 1)
    sink = gl.where(valid_head, gl.load(attn_sink + heads).to(gl.float32), 0.0)
    splits = gl.arange(0, SPLIT_BLOCK, layout=gl.SliceLayout(1, stat_layout))
    valid = (splits < num_splits)[:, None] & valid_head[None, :]
    rows = token_idx * num_splits + gl.minimum(splits, num_splits - 1)
    stats = part_stats + (rows[:, None].to(tl.int64) * num_heads + heads[None, :]) * 2
    split_max = gl.where(valid, gl.load(stats), -float("inf"))
    split_sum = gl.where(valid, gl.load(stats + 1), 0.0)
    frame = gl.maximum(gl.max(split_max, axis=0), sink)
    frame = gl.where(frame > -float("inf"), frame, 0.0)
    weight = gl.exp(split_max - frame[None, :])
    total = gl.exp(sink - frame) + gl.sum(split_sum * weight, axis=0)
    value_dims = dim_base + gl.arange(
        0, DIM_BLOCK, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    values = gl.load(
        part_acc
        + (rows[:, None, None].to(tl.int64) * num_heads + heads[None, :, None])
        * HEAD_DIM
        + value_dims[None, None, :]
    ).to(gl.float32)
    merged = gl.sum(values * (split_sum * weight)[:, :, None], axis=0)
    scale = gl.convert_layout(
        gl.where(total > 0.0, 1.0 / total, 0.0), gl.SliceLayout(1, out_layout)
    )
    return (merged * scale[:, None]).to(gl.bfloat16)


@gluon.jit
def _merge_zero_query(
    part_row_sum,
    part_count,
    attn_sink,
    num_heads,
    num_splits,
    token_idx,
    group_idx,
    dim_base,
    BLOCK_H: gl.constexpr,
    SPLIT_BLOCK: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    DIM_BLOCK: gl.constexpr,
):
    """BF16 [heads, dims] output of 16 zero-query heads.

    Every split scores its rows exactly zero, so the merged output is the
    single-split form row_sum * e^-m / (count * e^-m + e^(sink - m)),
    m = max(sink, 0), over the row sum and count of all splits. The heads
    share both, so each split's row sum loads once rather than per head.
    """
    layout: gl.constexpr = gl.SliceLayout(1, _reduce_layout(DIM_BLOCK))
    out_layout: gl.constexpr = gl.SliceLayout(0, _reduce_layout(DIM_BLOCK))
    splits = gl.arange(0, SPLIT_BLOCK, layout=gl.SliceLayout(1, layout))
    valid = splits < num_splits
    rows = token_idx * num_splits + gl.minimum(splits, num_splits - 1)
    value_dims = dim_base + gl.arange(0, DIM_BLOCK, layout=gl.SliceLayout(0, layout))
    sums = gl.load(
        part_row_sum + rows[:, None].to(tl.int64) * HEAD_DIM + value_dims[None, :]
    )
    row_sum = gl.sum(gl.where(valid[:, None], sums, 0.0), axis=0)
    count = gl.sum(gl.where(valid, gl.load(part_count + rows), 0), axis=0)
    heads = group_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, out_layout)
    )
    sink = gl.where(
        heads < num_heads,
        gl.load(attn_sink + gl.minimum(heads, num_heads - 1)).to(gl.float32),
        0.0,
    )
    frame = gl.maximum(sink, 0.0)
    weight = gl.exp(-frame)
    total = count.to(gl.float32) * weight + gl.exp(sink - frame)
    coefficient = gl.where(total > 0.0, weight / total, 0.0)
    row_sum = gl.convert_layout(row_sum, gl.SliceLayout(0, out_layout))
    return (row_sum[None, :] * coefficient[:, None]).to(gl.bfloat16)


@gluon.jit(launch_metadata=_selected_attention_reduce_metadata)
def gluon_dsv41_selected_attention_reduce_gfx1250(
    part_stats,
    part_acc,
    part_row_sum,
    part_count,
    part_active,
    attn_sink,
    out,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    num_heads,
    num_splits,
    head_blocks,
    rope_positions,
    rope_cos_sin,
    stride_rope: tl.int64,
    BLOCK_H: gl.constexpr,
    HEAD_GROUPS: gl.constexpr,
    SPLIT_BLOCK: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    DIM_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
):
    """Merge one token's split partials, adding the sink once.

    Each program covers 16 heads and ``DIM_BLOCK`` dims. Groups with a nonzero
    query merge their per-head partials; zero-query groups only need the
    shared per-split row sums and counts.
    """
    out_layout: gl.constexpr = gl.SliceLayout(0, _reduce_layout(DIM_BLOCK))
    token_idx = gl.program_id(axis=0)
    group_idx = gl.program_id(axis=1)
    dim_base = gl.program_id(axis=2) * DIM_BLOCK
    active_bits = gl.load(
        part_active + token_idx * head_blocks + group_idx // HEAD_GROUPS
    )
    if ((active_bits >> (group_idx % HEAD_GROUPS)) & 1) != 0:
        result = _merge_partials(
            part_stats,
            part_acc,
            attn_sink,
            num_heads,
            num_splits,
            token_idx,
            group_idx,
            dim_base,
            BLOCK_H,
            SPLIT_BLOCK,
            HEAD_DIM,
            DIM_BLOCK,
        )
    else:
        result = _merge_zero_query(
            part_row_sum,
            part_count,
            attn_sink,
            num_heads,
            num_splits,
            token_idx,
            group_idx,
            dim_base,
            BLOCK_H,
            SPLIT_BLOCK,
            HEAD_DIM,
            DIM_BLOCK,
        )
    out_heads = group_idx * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, out_layout)
    )
    dims = dim_base + gl.arange(0, DIM_BLOCK, layout=gl.SliceLayout(0, out_layout))
    if ROPE_DIM > 0:
        if dim_base + DIM_BLOCK > HEAD_DIM - ROPE_DIM:
            rope_row = (
                rope_cos_sin
                + gl.maximum(gl.load(rope_positions + token_idx), 0).to(tl.int64)
                * stride_rope
            )
            result = _inverse_rope(
                result, dims, out_heads, rope_row, dim_base, ROPE_DIM
            )
    gl.store(
        out
        + token_idx.to(tl.int64) * stride_o_t
        + out_heads[:, None].to(tl.int64) * stride_o_h
        + dims[None, :],
        result,
        mask=(out_heads < num_heads)[:, None],
    )


def _output(out, shape, dtype, device):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must have the specified shape, dtype, device and be contiguous"
        )
    return out


def _cache(cache, row_bytes, value_bytes, name):
    """Validate a page-planar cache; return its rows of value_bytes."""
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (64, row_bytes)
    ):
        raise ValueError(f"{name} cache must be uint8 [pages, 64, {row_bytes}]")
    # TDM gathers view the cache as rows of one page's value bytes, which
    # needs contiguous page bytes and a page stride of whole rows.
    if (
        cache.stride(2) != 1
        or cache.stride(1) != row_bytes
        or cache.stride(0) % value_bytes != 0
        or cache.data_ptr() % 16 != 0
    ):
        raise ValueError(
            f"{name} cache pages must be contiguous, 16-byte aligned and "
            f"{value_bytes}-byte strided"
        )
    if cache.shape[0] == 0:
        return 0
    rows = (cache.shape[0] - 1) * (cache.stride(0) // value_bytes) + _PAGE_ROWS
    # Gather row indices are int32.
    if rows >= 2**31:
        raise ValueError(f"{name} cache exceeds 2**31 rows of {value_bytes} bytes")
    return rows


def _integers(x, shape, name):
    if x.shape != shape or x.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32/int64 with shape {shape}")


@gluon.jit
def gluon_dsv41_output_rope_gfx1250(
    out,
    rope_positions,
    rope_cos_sin,
    stride_o_t,
    stride_o_h,
    stride_rope: tl.int64,
    num_heads,
    BLOCK_H: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    ROPE_DIM: gl.constexpr,
):
    """Rotate the RoPE tail of BLOCK_H heads of one token's output in place.

    Even and odd dims load separately, so the rotation needs no lane swap.
    """
    gl.static_assert(ROPE_DIM >= 2 and ROPE_DIM <= 64, "RoPE tail of 2 to 64")
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [64 // ROPE_DIM, ROPE_DIM // 2], [4, 1], [1, 0]
    )
    token_idx = gl.program_id(axis=0)
    heads = gl.program_id(axis=1) * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, layout)
    )
    pairs = gl.arange(0, ROPE_DIM // 2, layout=gl.SliceLayout(0, layout))
    token_out = out + token_idx.to(tl.int64) * stride_o_t + (HEAD_DIM - ROPE_DIM)
    even_offsets = heads[:, None] * stride_o_h + 2 * pairs[None, :]
    mask = (heads < num_heads)[:, None]
    rope_row = (
        rope_cos_sin
        + gl.maximum(gl.load(rope_positions + token_idx), 0).to(tl.int64) * stride_rope
    )
    cosine = gl.load(rope_row + pairs)[None, :]
    sine = gl.load(rope_row + ROPE_DIM // 2 + pairs)[None, :]
    even = gl.load(token_out + even_offsets, mask=mask, other=0.0).to(gl.float32)
    odd = gl.load(token_out + even_offsets + 1, mask=mask, other=0.0).to(gl.float32)
    gl.store(
        token_out + even_offsets,
        (even * cosine - odd * sine).to(gl.bfloat16),
        mask=mask,
    )
    gl.store(
        token_out + even_offsets + 1,
        (odd * cosine + even * sine).to(gl.bfloat16),
        mask=mask,
    )


def launch_gluon_dsv41_selected_attention_gfx1250(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
    rope_positions=None,
    rope_cos_sin=None,
):
    """Fused SWA+global selected attention over V4.1 page-planar caches.

    With ``rope_positions`` [tokens] and an FP32 ``rope_cos_sin`` table
    [positions, rope_dim] (inverse signs), the output tail is also rotated, as
    ``rope_inplace`` would do afterwards: in the split reduction, or in a
    per-token pass when the rows do not split. Paged decode inputs only.
    """
    attn_sink = attn_sink[: q.shape[1]]
    if prefill_kv is not None:
        if prefill_indices is None:
            raise ValueError("prefill_kv requires prefill_indices")
        if rope_positions is not None or rope_cos_sin is not None:
            raise ValueError("output RoPE needs paged decode inputs")
        from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

        out = _output(out, q.shape, q.dtype, q.device)
        lengths = torch.full(
            (q.shape[0],),
            prefill_indices.shape[-1],
            dtype=torch.int32,
            device=q.device,
        )
        dsv4_prefill(
            q=q.contiguous(),
            kv=prefill_kv,
            indices=prefill_indices,
            lens=lengths,
            attn_sink=attn_sink.contiguous(),
            softmax_scale=softmax_scale,
            out=out,
            override=None,
            solution=None,
        )
        return out
    if prefill_indices is not None:
        raise ValueError("prefill_indices requires prefill_kv")
    rope = rope_cos_sin is not None
    if rope:
        if (
            rope_cos_sin.ndim != 2
            or rope_cos_sin.dtype != torch.float32
            or rope_cos_sin.stride(1) != 1
            or rope_cos_sin.shape[1] not in (2, 4, 8, 16, 32, 64)
        ):
            raise ValueError("rope_cos_sin must be FP32 [positions, rope_dim]")
        _integers(rope_positions, (q.shape[0],), "rope_positions")
    elif rope_positions is not None:
        raise ValueError("rope_positions requires rope_cos_sin")
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16 or q.shape[1] < 1:
        raise ValueError("q must be BF16 [tokens, heads, 512]")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    extras = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        out,
    )
    if not q.is_cuda or any(t.device != q.device for t in extras if t is not None):
        raise ValueError("all tensors must share a CUDA/ROCm device")
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError("attn_sink must be FP32 [heads]")
    swa_value_rows = _cache(swa_cache, _SWA_ROW_BYTES, 512, "swa")
    if swa_slots.ndim != 2 or swa_slots.shape[0] != q.shape[0]:
        raise ValueError("slots must have one row per query")
    _integers(swa_slots, swa_slots.shape, "slots")
    _integers(swa_lens, (q.shape[0],), "lens")
    has_global = global_cache is not None
    if not has_global:
        if global_slots is not None or global_lens is not None:
            raise ValueError(
                "absent global cache requires None global slots and lengths"
            )
        global_value_rows = 0
    else:
        if global_slots is None or global_lens is None:
            raise ValueError("global cache requires global slots and lengths")
        global_value_rows = _cache(global_cache, _GLOBAL_ROW_BYTES, 256, "global")
        if global_slots.ndim != 2 or global_slots.shape[0] != q.shape[0]:
            raise ValueError("slots must have one row per query")
        _integers(global_slots, global_slots.shape, "slots")
        _integers(global_lens, (q.shape[0],), "lens")
    out = _output(out, q.shape, q.dtype, q.device)
    width = swa_slots.shape[1] + (global_slots.shape[1] if has_global else 0)
    if q.shape[0] == 0 or width == 0:
        return out.zero_()
    # Scale copies are unmasked: invalid rows read page 0.
    if swa_cache.shape[0] == 0 or (has_global and global_cache.shape[0] == 0):
        raise ValueError("caches must have at least one page")

    scale = float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    q = q.contiguous()
    # The zero-query check loads q as int32 words.
    if q.data_ptr() % 4 != 0:
        q = q.clone()
    sink = attn_sink.contiguous()
    swa_slots_i = swa_slots.to(torch.int32).contiguous()
    swa_lens_i = swa_lens.to(torch.int32).contiguous()
    if has_global:
        global_slots_i = global_slots.to(torch.int32).contiguous()
        global_lens_i = global_lens.to(torch.int32).contiguous()
        global_cache_u8 = global_cache
        global_width = global_slots_i.shape[1]
        global_capacity = global_cache.shape[0] * _PAGE_ROWS
        global_page_stride = global_cache.stride(0)
        global_slot_stride = global_slots_i.stride(0)
    else:
        global_slots_i = swa_slots_i
        global_lens_i = swa_lens_i
        global_cache_u8 = swa_cache
        global_width = 0
        global_capacity = 0
        global_page_stride = 0
        global_slot_stride = 0

    tokens = q.shape[0]
    swa_tiles = triton.cdiv(swa_slots_i.shape[1], _TILE_K)
    global_tiles = triton.cdiv(global_width, _TILE_K)
    max_tiles = swa_tiles + global_tiles
    # The token count only picks one of three head-group widths.
    head_groups = 1
    while (
        head_groups < _HEAD_GROUPS_PER_CTA
        and tokens * triton.cdiv(q.shape[1], head_groups * _BLOCK_H) * max_tiles
        > _HEAD_BLOCK_TILES
    ):
        head_groups *= 2
    head_groups = min(head_groups, triton.cdiv(q.shape[1], _BLOCK_H))
    head_blocks = triton.cdiv(q.shape[1], head_groups * _BLOCK_H)
    # Split each row's selected tiles over workgroups so small batches still
    # fill the GPU, at least two tiles per split: a one-tile split costs more
    # in the reduce than it saves. The split count depends only on shapes,
    # never on lengths.
    splits = max(
        1,
        min(
            _MAX_SPLITS,
            max_tiles // 2,
            _TARGET_CTAS // (tokens * head_blocks),
        ),
    )
    # The reduce loads every split at once. Its width bounds the split count
    # from the slot widths alone, so a new batch size never recompiles it.
    split_block = triton.next_power_of_2(max(1, min(_MAX_SPLITS, max_tiles // 2)))
    if splits > 1:
        heads = q.shape[1]
        part_stats = torch.empty(
            (tokens, splits, heads, 2), dtype=torch.float32, device=q.device
        )
        # Normalized by its split's denominator, a partial is bounded by the
        # largest value row, so FP16 holds it without overflow.
        part_acc = torch.empty(
            (tokens, splits, heads, _HEAD_DIM), dtype=torch.float16, device=q.device
        )
        part_row_sum = torch.empty(
            (tokens, splits, _HEAD_DIM), dtype=torch.float32, device=q.device
        )
        part_count = torch.empty((tokens, splits), dtype=torch.int32, device=q.device)
        part_active = torch.empty(
            (tokens, head_blocks), dtype=torch.int32, device=q.device
        )
    else:
        # Placeholders: a single split never touches the partials. Their
        # element types are compile keys, so they match the split buffers.
        part_stats = part_row_sum = sink
        part_acc = sink.view(torch.float16)
        part_count = part_active = swa_lens_i
    gluon_dsv41_selected_attention_gfx1250[(tokens, splits, head_blocks)](
        q,
        swa_cache,
        swa_slots_i,
        swa_lens_i,
        global_cache_u8,
        global_slots_i,
        global_lens_i,
        sink,
        out,
        part_stats,
        part_acc,
        part_row_sum,
        part_count,
        part_active,
        q.stride(0),
        q.stride(1),
        swa_cache.stride(0),
        swa_slots_i.stride(0),
        swa_slots_i.shape[1],
        global_page_stride,
        global_slot_stride,
        global_width,
        out.stride(0),
        out.stride(1),
        scale,
        swa_cache.shape[0] * _PAGE_ROWS,
        global_capacity,
        swa_value_rows,
        global_value_rows,
        q.shape[1],
        HAS_GLOBAL=has_global,
        HEAD_GROUPS=head_groups,
        BLOCK_H=_BLOCK_H,
        TILE_K=_TILE_K,
        HEAD_DIM=_HEAD_DIM,
        SWA_TILES=swa_tiles,
        GLOBAL_TILES=global_tiles,
        SLOT_TILES=triton.next_power_of_2(max_tiles),
        STAGES=_STAGES,
        num_warps=4,
        num_stages=1,
    )
    if splits > 1:
        gluon_dsv41_selected_attention_reduce_gfx1250[
            (tokens, triton.cdiv(q.shape[1], _BLOCK_H), _HEAD_DIM // _REDUCE_DIM_BLOCK)
        ](
            part_stats,
            part_acc,
            part_row_sum,
            part_count,
            part_active,
            sink,
            out,
            out.stride(0),
            out.stride(1),
            q.shape[1],
            splits,
            head_blocks,
            rope_positions if rope else swa_lens_i,
            rope_cos_sin if rope else sink,
            rope_cos_sin.stride(0) if rope else 0,
            BLOCK_H=_BLOCK_H,
            HEAD_GROUPS=head_groups,
            SPLIT_BLOCK=split_block,
            HEAD_DIM=_HEAD_DIM,
            DIM_BLOCK=_REDUCE_DIM_BLOCK,
            ROPE_DIM=rope_cos_sin.shape[1] if rope else 0,
            num_warps=4,
        )
    elif rope:
        # Rotating in the single-split epilogue costs this register-heavy
        # kernel more than a separate pass.
        gluon_dsv41_output_rope_gfx1250[(tokens, triton.cdiv(q.shape[1], _BLOCK_H))](
            out,
            rope_positions,
            rope_cos_sin,
            out.stride(0),
            out.stride(1),
            rope_cos_sin.stride(0),
            q.shape[1],
            BLOCK_H=_BLOCK_H,
            HEAD_DIM=_HEAD_DIM,
            ROPE_DIM=rope_cos_sin.shape[1],
            num_warps=4,
        )
    return out
