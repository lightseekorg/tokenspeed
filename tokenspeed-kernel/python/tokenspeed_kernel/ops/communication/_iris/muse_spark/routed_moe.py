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


"""Muse's per-route RMSNorm and weighted sums fused with Iris communication.

Apply the learned feature gain before each route's RMSNorm, then weight and
sum the normalized routes. The Lamport and write-through variants specialize
4096-feature outputs with eight routes per token; route maps mask nonlocal work.
"""

from tokenspeed_kernel._triton import gl, gluon, tl
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_drain_subgroup_vmem,
    _iris_group_heap_base,
    _iris_sanitize_lamport_bf16,
    _iris_sync_peers,
    _iris_wait_lamport_peers,
)


@gluon.jit
def iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum(
    routes,
    route_map,
    route_weights,
    norm_weight,
    symmetric_input,
    output,
    ready_flags,
    heaps,
    rank: gl.constexpr,
    width: gl.constexpr,
    row_stride,
    topk: gl.constexpr,
    eps: gl.constexpr,
    weight_offset: gl.constexpr,
    block: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Normalize and weight local expert routes, then all-reduce their token sums.

    route_map locates rows in routes; negative entries identify nonlocal routes.
    Normalize each route before weighting and summing in FP32, then publish its
    BF16 token sum in symmetric_input. Pull peer sums and write BF16 output.
    ready_flags coordinate input publication and completion before storage reuse.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    token = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [block // (64 * num_warps)],
        [64],
        [num_warps],
        [0],
    )
    columns = gl.arange(0, block, layout=layout)
    valid = columns < width
    weight = gl.load(norm_weight + columns, mask=valid, other=0.0).to(gl.float32)
    weight = (weight + weight_offset).to(gl.bfloat16).to(gl.float32)
    combined = gl.zeros((block,), gl.float32, layout=layout)

    for slot in gl.static_range(0, topk):
        route = token * topk + slot
        source_row = gl.load(route_map + route)
        owned = source_row >= 0
        if owned:
            source_offset = gl.maximum(source_row, 0).to(gl.int64) * row_stride
            values = gl.load(
                routes + source_offset + columns,
                mask=valid & owned,
                other=0.0,
            ).to(gl.float32)
            values *= weight
            variance = gl.sum(values * values, axis=0) / width
            scale = gl.rsqrt(variance + eps) * gl.load(route_weights + route).to(
                gl.float32
            )
            combined += values * scale

    row_offsets = token * width + columns
    local = combined.to(gl.bfloat16)
    gl.store(symmetric_input + row_offsets, local, mask=valid)

    local_heap = _iris_group_heap_base(rank, heaps)
    epoch_ptr = ready_flags + token * WORLD_SIZE + rank
    ready_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_peers(ready_flags, token, ready_epoch, heaps, rank, num_warps)

    input_offset = tl.cast(symmetric_input, gl.uint64) - local_heap
    reduced = local.to(gl.float32)
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer != rank:
            peer_input = tl.cast(
                _iris_group_heap_base(peer, heaps) + input_offset,
                gl.pointer_type(gl.bfloat16),
            )
            reduced += gl.load(
                peer_input + row_offsets,
                mask=valid,
                other=0.0,
                cache_modifier=".cg",
            ).to(gl.float32)
    gl.store(output + row_offsets, reduced.to(gl.bfloat16), mask=valid)

    done_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_peers(ready_flags, token, done_epoch, heaps, rank, num_warps)


@gluon.jit
def iris_fused_lamport_allreduce_routed_rmsnorm_sum(
    routes,
    route_map,
    route_weights,
    norm_weight,
    output,
    region,
    epochs,
    regions,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    row_stride,
    EPS: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    PORTABLE_MATH: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Normalize eight local routes per token and all-reduce their weighted sum.

    Launch one single-wave program for each 512-feature tile of each token.
    route_map selects local rows in routes; negative entries are ignored. Apply
    per-route RMSNorm before weighting and summing, round locally to BF16, and
    exchange sums through sentinel-marked mailboxes. Write BF16 output and clear
    the consumed mailbox slot. PORTABLE_MATH preserves the portable path's
    multiplication and rank-addition order; epochs select the rotating slot.
    """
    WORLD_SIZE: gl.constexpr = len(regions)
    TILES: gl.constexpr = WIDTH // 512
    # Preserve the arithmetic of each existing routed normalization path.
    gl.static_assert(WIDTH == 4096)
    gl.static_assert(gl.num_warps() == 1)
    program = gl.program_id(0)
    token = program // TILES
    tile = program % TILES
    stage = gl.load(epochs + program).to(gl.uint32) % 3
    # Every program reads whole routes for their variance. As in an eight-warp
    # row, lane l holds features t * 512 + l * 8 + [0, 8) of each tile t.
    row_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, 1], [1, 0])
    tiles = gl.arange(0, TILES, layout=gl.SliceLayout(1, row_layout))
    columns = (
        tiles[:, None] * 512
        + gl.arange(0, 512, layout=gl.SliceLayout(0, row_layout))[None, :]
    )
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.arange(0, 64, layout=gl.SliceLayout(1, packed_layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, packed_layout))
    tile_columns = tile * 512 + pack[:, None] * 8 + element[None, :]
    weight = gl.load(norm_weight + columns).to(gl.float32)
    tile_weight = gl.load(norm_weight + tile_columns).to(gl.float32)
    source_rows = ()
    route_weight_values = ()
    for slot in gl.static_range(8):
        source_rows += (gl.load(route_map + token * 8 + slot),)
        route_weight_values += (gl.load(route_weights + token * 8 + slot),)
    weight = (weight + 1.0).to(gl.bfloat16).to(gl.float32)
    tile_weight = (tile_weight + 1.0).to(gl.bfloat16).to(gl.float32)
    combined = gl.zeros((64, 8), gl.float32, layout=packed_layout)
    for slot in gl.static_range(8):
        if source_rows[slot] >= 0:
            source = routes + source_rows[slot].to(gl.int64) * row_stride
            values = gl.load(source + columns).to(gl.float32) * weight
            tile_values = gl.load(source + tile_columns).to(gl.float32) * tile_weight
            partials = gl.sum(values * values, 1)
            # Add tile partials in the eight-warp reduction's butterfly order.
            tile_sums = ()
            for t in gl.static_range(TILES):
                tile_sums += (gl.sum(gl.where(tiles == t, partials, 0.0), 0),)
            variance = (
                ((tile_sums[0] + tile_sums[4]) + (tile_sums[2] + tile_sums[6]))
                + ((tile_sums[1] + tile_sums[5]) + (tile_sums[3] + tile_sums[7]))
            ) / WIDTH
            inv_rms = gl.rsqrt(variance + EPS)
            route_weight = route_weight_values[slot].to(gl.float32)
            if PORTABLE_MATH:
                combined += (tile_values * inv_rms) * route_weight
            else:
                combined += tile_values * (inv_rms * route_weight)
    offsets = token * WIDTH + tile_columns
    stage_base = (stage * (WORLD_SIZE * MAX_ELEMENTS)).to(gl.int32)
    local = _iris_sanitize_lamport_bf16(combined.to(gl.bfloat16))
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        address = _iris_group_heap_base(peer, regions)
        target = gl.multiple_of(address.to(gl.pointer_type(gl.bfloat16)), 16)
        gl.amd.cdna4.buffer_store(
            local,
            target,
            stage_base + RANK * MAX_ELEMENTS + offsets,
            cache=PEER_SIGNAL_STORE_CACHE,
        )
    received = _iris_wait_lamport_peers(
        region,
        stage.to(gl.uint64),
        offsets,
        pack >= 0,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        packed_layout,
    )
    if PORTABLE_MATH:
        # The portable two-pass path rounds locally, then sums global rank order.
        for peer in gl.static_range(WORLD_SIZE):
            if peer == RANK:
                term = local
            else:
                term = received[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
            if peer == 0:
                reduced = term.to(gl.float32)
            else:
                reduced += term.to(gl.float32)
    else:
        # Preserve local first, followed by increasing global peer rank.
        reduced = local.to(gl.float32)
        for peer in gl.static_range(WORLD_SIZE):
            if peer != RANK:
                reduced += received[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1].to(
                    gl.float32
                )
    gl.amd.cdna4.buffer_store(reduced.to(gl.bfloat16), output, offsets)
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        gl.amd.cdna4.buffer_store(
            sentinel, region, stage_base + peer * MAX_ELEMENTS + offsets, cache=".wt"
        )
    # Finish resetting the mailbox before advancing this tile's epoch.
    _iris_drain_subgroup_vmem()
    gl.store(epochs + program, ((stage + 1) % 3).to(gl.int32))


@gluon.jit(do_not_specialize=["row_stride"])
def iris_routed_rmsnorm_weighted_sum_write_through(
    routes,
    route_map,
    route_weights,
    norm_weight,
    output,
    row_stride,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
):
    """Normalize and sum weighted expert routes using write-through output stores.

    route_map selects local rows from routes; normalize each route before
    weighting and summing into BF16 token rows of width 4096 in output. Finish
    previous peer reads and flush earlier cached output writes before launch.
    Input row strides are divisible by eight. Stores finish before return so
    the following reduce-scatter can read output without a staging copy.
    """
    gl.static_assert(gl.num_warps() == 8)
    token = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [8], [0])
    columns = gl.arange(0, 4096, layout=layout)
    weight = gl.load(norm_weight + columns).to(gl.float32)
    weight = (weight + WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    combined = gl.zeros((4096,), gl.float32, layout=layout)
    for slot in gl.static_range(8):
        route = token * 8 + slot
        source_row = gl.load(route_map + route)
        if source_row >= 0:
            offset = source_row.to(gl.int64) * gl.multiple_of(row_stride, 8)
            values = gl.load(routes + offset + columns).to(gl.float32)
            values *= weight
            variance = gl.sum(values * values, 0) / 4096
            inverse = gl.rsqrt(variance + EPS)
            route_weight = gl.load(route_weights + route).to(gl.float32)
            combined = gl.fma(values * inverse, route_weight, combined)
    gl.amd.cdna4.buffer_store(
        combined.to(gl.bfloat16), output, token * 4096 + columns, cache=".wt"
    )
    # Finish every subgroup's write-through stores before returning.
    _iris_drain_subgroup_vmem()
    gl.barrier()
