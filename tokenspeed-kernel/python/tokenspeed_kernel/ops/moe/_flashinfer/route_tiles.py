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

"""Route tiling for the TRT-LLM-gen batched GEMMs, in Triton.

These kernels lay the routed rows out in the batched GEMM's token tiles,
write its dynamic-batch metadata, and gather the GEMM rows back to route
rows. They reuse the portable quantizers of ``_triton/block_scaled.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe._triton.block_scaled import (
    _mxfp8_chunk_codes,
    _nvfp4_chunk_codes,
    _routed_quantized_input,
    _scale_offset,
    _store_mxfp8_chunk,
    _store_nvfp4_chunk,
    _trigger_after_stores,
)
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit(do_not_specialize=["route_count", "num_programs", "tile", "layout"])
def _tile_routes_kernel(
    Routes,
    Counts,
    Values,
    Scales,
    Amax,
    WAmax,
    WDen,
    TiledValues,
    TiledScales,
    RowScale,
    RouteRows,
    TileExpert,
    TileLimit,
    NumTiles,
    NumRows,
    route_count,
    num_programs,
    tile,
    layout,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    WORDS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    NVFP4: tl.constexpr,
    ROWS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Lay the local routes out for a batched expert GEMM: each expert's
    routes in route order, padded to ``tile`` rows, and the dynamic-batch
    metadata of that layout (tiles, each tile's expert and row limit, padded
    rows). A route's row receives its token's quantized row (``WORDS``
    32-bit words) and scales (in ``layout``, ``_scale_offset``), and an
    NVFP4 row its FP32 output scale ``(x_amax * w_amax) * rcp(1536 * den)``;
    ``RouteRows`` maps the route to the row. Padding rows stay unspecified:
    their outputs are never read. Under PDL it releases the GEMM only after
    its stores."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    experts = tl.arange(0, triton.next_power_of_2(EXPERTS))
    counts = tl.load(Counts + experts, experts < EXPERTS, other=0)
    tiles = tl.cdiv(counts, tile)
    tile_end = tl.cumsum(tiles, 0)
    tile_start = tile_end - tiles
    route_end = tl.cumsum(counts, 0)
    total_tiles = tl.sum(tiles, 0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(NumTiles, total_tiles)
        tl.store(NumRows, total_tiles * tile)
    for base in range(pid * ROWS, total_tiles, num_programs * ROWS):
        index = base + tl.arange(0, ROWS)
        valid = index < total_tiles
        expert = tl.sum((tile_end[None, :] <= index[:, None]).to(tl.int32), 1)
        mine = experts[None, :] == expert[:, None]
        start = tl.sum(tl.where(mine, tile_start[None, :], 0), 1) * tile
        count = tl.sum(tl.where(mine, counts[None, :], 0), 1)
        tl.store(TileExpert + index, expert, valid)
        tl.store(
            TileLimit + index, tl.minimum((index + 1) * tile, start + count), valid
        )
    words = tl.arange(0, triton.next_power_of_2(WORDS))
    scale_words = tl.arange(0, triton.next_power_of_2(SCALE_COLS // 4))
    total_routes = tl.sum(counts, 0)
    for base in range(pid * ROWS, total_routes, num_programs * ROWS):
        local = base + tl.arange(0, ROWS)
        valid = local < total_routes
        # Each local route's expert, rank among the expert's routes and tiled row.
        expert = tl.sum((route_end[None, :] <= local[:, None]).to(tl.int32), 1)
        mine = experts[None, :] == expert[:, None]
        rank = local - tl.sum(tl.where(mine, (route_end - counts)[None, :], 0), 1)
        start = tl.sum(tl.where(mine, tile_start[None, :], 0), 1) * tile
        route = tl.load(
            Routes + expert.to(tl.int64) * route_count + rank, valid, other=0
        )
        row = (start + rank).to(tl.int64)
        token = (route // TOP_K).to(tl.int64)
        mask = valid[:, None] & (words < WORDS)[None, :]
        payload = tl.load(Values + token[:, None] * WORDS + words[None, :], mask)
        tl.store(TiledValues + row[:, None] * WORDS + words[None, :], payload, mask)
        # Four scale columns per word: one word stays contiguous in every layout.
        mask = valid[:, None] & (scale_words < SCALE_COLS // 4)[None, :]
        scales = tl.load(
            Scales + token[:, None] * (SCALE_COLS // 4) + scale_words[None, :], mask
        )
        offset = (
            _scale_offset(row[:, None], 4 * scale_words[None, :], SCALE_COLS, layout)
            // 4
        )
        tl.store(TiledScales + offset, scales, mask)
        if NVFP4:
            rcp = tl.div_rn(1.0, 1536.0 * tl.load(WDen))
            amax = tl.load(Amax + token, valid, other=0.0)
            w_amax = tl.load(WAmax + expert, valid, other=0.0)
            tl.store(RowScale + row, (amax * w_amax) * rcp, valid)
        tl.store(RouteRows + route, row.to(tl.int32), valid)
    if ENABLE_PDL:
        _trigger_after_stores()


class TiledRoutes(NamedTuple):
    """Expert-tiled first-projection inputs and their dynamic-batch metadata.

    ``values`` and ``scales`` hold 32-bit words of each row's quantized
    payload and block scales (in the GEMM's layout), ``row_scale`` the NVFP4
    rows' FP32 output scales (None for MXFP8), ``route_rows`` each local
    route's row. ``num_tiles``, ``num_rows``, ``tile_expert`` and
    ``tile_limit`` are the GEMM's tile count, padded row count, and each
    tile's local expert and row limit.
    """

    values: torch.Tensor
    scales: torch.Tensor
    row_scale: torch.Tensor | None
    route_rows: torch.Tensor
    num_tiles: torch.Tensor
    num_rows: torch.Tensor
    tile_expert: torch.Tensor
    tile_limit: torch.Tensor


def _tile_routes(
    routes: torch.Tensor,
    counts: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    amax: torch.Tensor | None,
    w_amax: torch.Tensor | None,
    w_denominator: torch.Tensor | None,
    *,
    tile: int,
    layout: int,
    top_k: int,
) -> TiledRoutes:
    """Tile ``_routed_quantized_input``'s per-token rows for a batched GEMM.

    ``routes`` and ``counts`` are the local experts' route lists; ``values``
    and ``scales`` the tokens' quantized rows, ``amax`` their NVFP4 maxima
    (None for MXFP8) and ``w_amax``, ``w_denominator`` the weights' NVFP4
    outer scales. ``tile`` is the GEMM's token tile and ``layout`` its scale
    layout (``_scale_offset``).
    """
    experts, route_count = routes.shape
    device = routes.device
    nvfp4 = amax is not None
    # Every route local plus a partial tile per expert, in whole 128-row scale tiles.
    capacity = triton.cdiv(route_count + experts * (tile - 1), 128) * 128
    words = values.shape[1] * values.element_size() // 4
    scale_cols = scales.shape[1]
    tile_count = route_count // tile + experts + 1
    tiled = TiledRoutes(
        values=torch.empty((capacity, words), device=device, dtype=torch.int32),
        scales=torch.empty(
            (capacity * scale_cols // 4,), device=device, dtype=torch.int32
        ),
        row_scale=(
            torch.empty(capacity, device=device, dtype=torch.float32) if nvfp4 else None
        ),
        route_rows=torch.empty(route_count, device=device, dtype=torch.int32),
        num_tiles=torch.empty(1, device=device, dtype=torch.int32),
        num_rows=torch.empty(1, device=device, dtype=torch.int32),
        tile_expert=torch.empty(tile_count, device=device, dtype=torch.int32),
        tile_limit=torch.empty(tile_count, device=device, dtype=torch.int32),
    )
    rows_per_step = 16
    # At most every route is local: small batches take few programs.
    programs = min(
        torch.cuda.get_device_properties(device).multi_processor_count * 4,
        triton.cdiv(route_count, rows_per_step),
    )
    enable_pdl = pdl_enabled()
    unused = tiled.route_rows
    _tile_routes_kernel[(programs,)](
        routes,
        counts,
        values.view(torch.int32),
        scales.view(torch.int32),
        unused if amax is None else amax,
        unused if w_amax is None else w_amax,
        unused if w_denominator is None else w_denominator,
        tiled.values,
        tiled.scales,
        unused if tiled.row_scale is None else tiled.row_scale,
        tiled.route_rows,
        tiled.tile_expert,
        tiled.tile_limit,
        tiled.num_tiles,
        tiled.num_rows,
        route_count,
        programs,
        tile,
        layout,
        experts,
        top_k,
        words,
        scale_cols,
        nvfp4,
        rows_per_step,
        enable_pdl,
        num_warps=8,
        enable_fp_fusion=False,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return tiled


# Up to this many routes one launch routes, tiles and quantizes (_route_tile_quantize).
_FUSED_TILE_MAX_ROUTES = 1024


@triton.jit(do_not_specialize=["route_count", "expert_start", "tile", "layout"])
def _route_tile_quantize_kernel(
    X,
    Ids,
    WAmax,
    WDen,
    TiledValues,
    TiledScales,
    RowScale,
    RouteRows,
    TileExpert,
    TileLimit,
    NumTiles,
    NumRows,
    route_count,
    expert_start,
    tile,
    layout,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    width: tl.constexpr,
    NVFP4: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTES: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """``_routed_quantized_input`` and ``_tile_routes`` in one launch for
    batches of at most ``ROUTES`` routes, one program per (token, column
    chunk). Every program counts the batch's local routes per expert,
    finds its token's routes' tiled rows (an expert's routes in route
    order) and stores its quantized chunk once per local route; chunk 0
    stores the routes' rows and NVFP4 output scales, and the first program
    the tile metadata. Under PDL it quantizes its chunk before it waits:
    ``X`` is the router's input, complete once the router has waited, and
    the router releases its successors only then. It releases the GEMM once
    the tile metadata is stored: the GEMM reads only that and its weights
    before its own wait, so it streams weights while the rows are stored."""
    token = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    amax = 0.0
    if NVFP4:
        # The row's maximum is exact in any order.
        rcol = tl.arange(0, triton.next_power_of_2(width))
        whole = tl.load(X + token * width + rcol, rcol < width, other=0.0)
        amax = tl.max(tl.abs(whole.to(tl.float32)), 0)
    col = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + token * width + col, col < width, other=0.0).to(tl.float32)
    if NVFP4:
        packed, scale = _nvfp4_chunk_codes(x, amax, BLOCK)
    else:
        values, exponent = _mxfp8_chunk_codes(x, BLOCK)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    every = tl.arange(0, ROUTES)
    valid = every < route_count
    expert_of = tl.load(Ids + every, valid, other=-1) - expert_start
    local = valid & (expert_of >= 0) & (expert_of < EXPERTS)
    BINS: tl.constexpr = triton.next_power_of_2(EXPERTS + 1)
    experts = tl.arange(0, BINS)
    # Remote routes land in bin EXPERTS, which counts nothing.
    counts = tl.histogram(tl.where(local, expert_of, EXPERTS), BINS)
    counts = tl.where(experts < EXPERTS, counts, 0)
    tiles = tl.cdiv(counts, tile)
    tile_end = tl.cumsum(tiles, 0)
    tile_start = tile_end - tiles
    if (token == 0) & (chunk == 0):
        total_tiles = tl.sum(tiles, 0)
        tl.store(NumTiles, total_tiles)
        tl.store(NumRows, total_tiles * tile)
        for base in range(0, total_tiles, 32):
            index = base + tl.arange(0, 32)
            tile_expert = tl.sum((tile_end[None, :] <= index[:, None]).to(tl.int32), 1)
            hit = experts[None, :] == tile_expert[:, None]
            begin = tl.sum(tl.where(hit, tile_start[None, :], 0), 1) * tile
            count = tl.sum(tl.where(hit, counts[None, :], 0), 1)
            limit = tl.minimum((index + 1) * tile, begin + count)
            tl.store(TileExpert + index, tile_expert, index < total_tiles)
            tl.store(TileLimit + index, limit, index < total_tiles)
    if ENABLE_PDL:
        _trigger_after_stores()
    slot = tl.arange(0, triton.next_power_of_2(TOP_K))
    route = token * TOP_K + slot
    expert = tl.load(Ids + route, slot < TOP_K, other=-1) - expert_start
    own = (slot < TOP_K) & (expert >= 0) & (expert < EXPERTS)
    earlier = (
        (expert_of[None, :] == expert[:, None])
        & local[None, :]
        & (every[None, :] < route[:, None])
    )
    rank = tl.sum(earlier.to(tl.int32), 1)
    mine = experts[None, :] == expert[:, None]
    row = (tl.sum(tl.where(mine, tile_start[None, :], 0), 1) * tile + rank).to(tl.int64)
    for k in tl.static_range(TOP_K):
        if tl.sum(tl.where(slot == k, own.to(tl.int32), 0), 0) != 0:
            row_k = tl.sum(tl.where(slot == k, row, 0), 0)
            if NVFP4:
                _store_nvfp4_chunk(
                    TiledValues,
                    TiledScales,
                    row_k,
                    chunk,
                    packed,
                    scale,
                    width,
                    BLOCK,
                    layout,
                )
            else:
                _store_mxfp8_chunk(
                    TiledValues,
                    TiledScales,
                    row_k,
                    chunk,
                    values,
                    exponent,
                    width,
                    BLOCK,
                    layout,
                )
    if chunk == 0:
        tl.store(RouteRows + route, row.to(tl.int32), own)
        if NVFP4:
            rcp = tl.div_rn(1.0, 1536.0 * tl.load(WDen))
            w_amax = tl.load(WAmax + expert, own, other=0.0)
            tl.store(RowScale + row, (amax * w_amax) * rcp, own)


def _route_tile_quantize(
    x: torch.Tensor,
    ids: torch.Tensor,
    w_amax: torch.Tensor | None,
    w_denominator: torch.Tensor | None,
    *,
    experts: int,
    expert_start: int,
    fmt: str,
    tile: int,
    layout: int,
) -> TiledRoutes:
    """``_tile_routes(*_routed_quantized_input(...))`` in one launch, for at
    most ``_FUSED_TILE_MAX_ROUTES`` routes; the same rows and metadata."""
    tokens, width = x.shape
    top_k = ids.shape[1]
    route_count = ids.numel()
    if route_count > _FUSED_TILE_MAX_ROUTES:
        raise ValueError("the fused route tiling takes at most 1024 routes")
    device = x.device
    nvfp4 = fmt == "nvfp4"
    capacity = triton.cdiv(route_count + experts * (tile - 1), 128) * 128
    words = (width // 2 if nvfp4 else width) // 4
    scale_cols = width // (16 if nvfp4 else 32)
    tile_count = route_count // tile + experts + 1
    tiled = TiledRoutes(
        values=torch.empty((capacity, words), device=device, dtype=torch.int32),
        scales=torch.empty(
            (capacity * scale_cols // 4,), device=device, dtype=torch.int32
        ),
        row_scale=(
            torch.empty(capacity, device=device, dtype=torch.float32) if nvfp4 else None
        ),
        route_rows=torch.empty(route_count, device=device, dtype=torch.int32),
        num_tiles=torch.empty(1, device=device, dtype=torch.int32),
        num_rows=torch.empty(1, device=device, dtype=torch.int32),
        tile_expert=torch.empty(tile_count, device=device, dtype=torch.int32),
        tile_limit=torch.empty(tile_count, device=device, dtype=torch.int32),
    )
    if not tokens:
        return tiled
    enable_pdl = pdl_enabled()
    unused = tiled.route_rows
    _route_tile_quantize_kernel[(tokens, triton.cdiv(width, 256))](
        x,
        ids.to(torch.int32).contiguous(),
        unused if w_amax is None else w_amax,
        unused if w_denominator is None else w_denominator,
        tiled.values.view(torch.uint8 if nvfp4 else torch.float8_e4m3fn),
        tiled.scales.view(torch.float8_e4m3fn if nvfp4 else torch.uint8),
        unused if tiled.row_scale is None else tiled.row_scale,
        tiled.route_rows,
        tiled.tile_expert,
        tiled.tile_limit,
        tiled.num_tiles,
        tiled.num_rows,
        route_count,
        expert_start,
        tile,
        layout,
        experts,
        top_k,
        width,
        nvfp4,
        256,
        # Every program scans the batch's routes: a block of the batch's size
        # (a power of two, compiled once per bucket) instead of the bound.
        max(64, triton.next_power_of_2(route_count)),
        enable_pdl,
        # Every program histograms the batch's routes: on B300, 8 warps route
        # 8 / 32 tokens 0.7 / 0.8 us sooner than 4 behind the router and top-k.
        num_warps=8,
        enable_fp_fusion=False,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return tiled


@triton.jit(do_not_specialize=["route_count"])
def _gather_route_rows_kernel(
    Rows,
    RouteRows,
    Ids,
    Output,
    route_count,
    expert_start,
    num_experts,
    hidden: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
    ZERO_REMOTE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Copy ``ROUTES`` local routes' columns ``BLOCK`` of their expert-tiled
    rows back to their route rows; with ``ZERO_REMOTE`` remote routes' rows
    read zero for the EP reduction."""
    route = tl.program_id(0).to(tl.int64) * ROUTES + tl.arange(0, ROUTES)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = route < route_count
    expert = tl.load(Ids + route, valid, other=-1) - expert_start
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    local = valid & (expert >= 0) & (expert < num_experts)
    written = valid if ZERO_REMOTE else local
    row = tl.load(RouteRows + route, local, other=0).to(tl.int64)
    rows = tl.load(
        Rows + row[:, None] * hidden + col[None, :],
        local[:, None] & (col < hidden)[None, :],
        other=0.0,
    )
    tl.store(
        Output + route[:, None] * hidden + col[None, :],
        rows,
        written[:, None] & (col < hidden)[None, :],
    )


def _gather_route_rows(
    rows: torch.Tensor,
    route_rows: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    *,
    expert_start: int,
    num_experts: int,
    zero_remote_routes: bool,
) -> None:
    """``output[route] = rows[route_rows[route]]`` for the local routes."""
    route_count, hidden = output.shape
    enable_pdl = pdl_enabled()
    _gather_route_rows_kernel[(triton.cdiv(route_count, 4), triton.cdiv(hidden, 1024))](
        rows,
        route_rows,
        ids.contiguous(),
        output,
        route_count,
        expert_start,
        num_experts,
        hidden,
        4,
        1024,
        zero_remote_routes,
        enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
