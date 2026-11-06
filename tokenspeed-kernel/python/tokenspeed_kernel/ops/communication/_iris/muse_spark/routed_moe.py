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
    _iris_group_heap_base,
    _iris_sanitize_lamport_bf16,
    _iris_sanitize_lamport_bf16x2,
    _iris_subgroup_any,
    _iris_sync_peers,
    _iris_wait_lamport_peers,
    _iris_wait_lamport_slots,
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
    WAIT_STORES: gl.constexpr,
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
    WAIT_STORES(args=[], constraints="=r,~{memory}")
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
    WAIT_STORES: gl.constexpr,
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
    WAIT_STORES(args=[], constraints="=r,~{memory}")
    gl.barrier()


@gluon.jit(do_not_specialize=["rows"])
def iris_fused_routed_rmsnorm_sum_lamport_allgather(
    routes,
    route_weights,
    norm_weight,
    output,
    region,
    epochs,
    regions,
    rows,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Normalize and sum each owned token's routes, then gather every token's sum.

    Rank r owns tokens [r * L, (r + 1) * L), L = ceil(rows / WORLD_SIZE);
    routes holds their complete routes as [L, 8, WIDTH] and route_weights is
    [rows, 8]. Launch one eight-subgroup program per owned token. Its routed
    RMSNorm and weighted sum follow the portable routed sum's operation
    order and agree with it within BF16 rounding (FMA contraction and
    reduction order). Subgroup w then pushes features [512w, 512w + 512)
    to every peer through the Lamport one-shot mailbox, whose 512-element
    tile index and stage it shares with the other one-shot uses, and polls
    the peers' tiles on its own. A padding token pushes zeros, so every rank
    waits on every peer. Writes BF16 [rows, WIDTH] output.
    """
    WORLD_SIZE: gl.constexpr = len(regions)
    gl.static_assert(WIDTH == 4096)
    gl.static_assert(gl.num_warps() == 8)
    local_row = gl.program_id(0)
    local_rows = gl.cdiv(rows, WORLD_SIZE)
    token = RANK * local_rows + local_row
    # The portable routed norm's eight-subgroup row: lane l of subgroup w
    # holds features 512w + 8l + [0, 8).
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [8], [0])
    columns = gl.arange(0, WIDTH, layout=layout)
    weight = gl.load(norm_weight + columns).to(gl.float32)
    weight = (weight + WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    combined = gl.zeros((WIDTH,), gl.float32, layout=layout)
    if token < rows:
        for slot in gl.static_range(8):
            values = gl.load(routes + (local_row * 8 + slot) * WIDTH + columns).to(
                gl.float32
            )
            values = values * weight
            variance = gl.sum(values * values, 0) / WIDTH
            inv_rms = gl.rsqrt(variance + EPS)
            route_weight = gl.load(route_weights + token * 8 + slot).to(gl.float32)
            combined += (values * inv_rms) * route_weight
    # The same ownership as [WIDTH / 8 packs, 8]: pack p is lane p % 64 of
    # subgroup p // 64.
    packed: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [8, 1], [1, 0])
    pack = gl.arange(0, WIDTH // 8, layout=gl.SliceLayout(1, packed))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, packed))
    features = pack[:, None] * 8 + element[None, :]
    local = gl.convert_layout(
        gl.reshape(
            _iris_sanitize_lamport_bf16(combined.to(gl.bfloat16)), [WIDTH // 8, 8]
        ),
        packed,
    )
    tiles = local_row * (WIDTH // 512) + pack // 64
    stage = gl.load(epochs + tiles).to(gl.uint32) % 3
    offsets = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS)).to(gl.int32)[:, None]
        + local_row * WIDTH
        + features
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        address = _iris_group_heap_base((RANK + delta) % WORLD_SIZE, regions)
        target = gl.multiple_of(address.to(gl.pointer_type(gl.bfloat16)), 16)
        gl.amd.cdna4.buffer_store(
            local,
            target,
            RANK * MAX_ELEMENTS + offsets,
            cache=PEER_SIGNAL_STORE_CACHE,
        )
    gl.amd.cdna4.buffer_store(
        local, output, token * WIDTH + features, mask=(features >= 0) & (token < rows)
    )
    received = _iris_wait_lamport_slots(
        region, offsets, pack >= 0, RANK, WORLD_SIZE, MAX_ELEMENTS, packed
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer_token = ((RANK + delta) % WORLD_SIZE) * local_rows + local_row
        gl.amd.cdna4.buffer_store(
            received[delta - 1],
            output,
            peer_token * WIDTH + features,
            mask=(features >= 0) & (peer_token < rows),
        )
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        gl.amd.cdna4.buffer_store(
            sentinel,
            region,
            ((RANK + delta) % WORLD_SIZE) * MAX_ELEMENTS + offsets,
            cache=".wt",
        )
    # Finish each subgroup's mailbox reset before it advances its tile's stage.
    WAIT_STORES(args=[], constraints="=r,~{memory}")
    gl.store(epochs + tiles, ((stage + 1) % 3).to(gl.int32), mask=pack % 64 == 0)


# Column-owned route completion. A route row holds 4096 BF16 features, 2048
# 32-bit words in the mailboxes; each of the eight ranks owns a 512-feature
# slice (256 words) of every route.
_ROUTE_FEATURES = gl.constexpr(4096)
_ROUTE_WORDS = gl.constexpr(2048)
_SLICE_FEATURES = gl.constexpr(512)
_SLICE_WORDS = gl.constexpr(256)


@gluon.jit
def _iris_column_routes_offsets():
    """Lane l's four 32-bit words (BF16 columns 8l + [0, 8)) of a 512-column slice."""
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [1, 1], [1, 0])
    lane = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    return lane[:, None] * 4 + word[None, :]


@gluon.jit
def _iris_column_routes_send(
    routes,
    epochs,
    regions,
    program,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    STAGE_WORDS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Push this rank's partial of one route's 512 columns to the rank owning them."""
    PEERS: gl.constexpr = WORLD_SIZE - 1
    delta = program % PEERS + 1
    row = program // PEERS
    token = row // 8
    owner = (RANK + delta) % WORLD_SIZE
    words = _iris_column_routes_offsets()
    partial = _iris_sanitize_lamport_bf16x2(
        gl.amd.cdna4.buffer_load(
            routes.to(gl.pointer_type(gl.uint32), bitcast=True),
            row * _ROUTE_WORDS + owner * _SLICE_WORDS + words,
        )
    )
    stage = gl.load(epochs + token * WORLD_SIZE + owner).to(gl.uint32) % 3
    # The owner keeps one slot per other rank, indexed by distance - 1.
    offsets = (
        (stage * STAGE_WORDS).to(gl.int32)
        + (PEERS - delta) * (MAX_ROWS * _ROUTE_WORDS)
        + row * _SLICE_WORDS
        + words
    )
    target = gl.multiple_of(
        _iris_group_heap_base(owner, regions).to(gl.pointer_type(gl.uint32)), 16
    )
    gl.amd.cdna4.buffer_store(partial, target, offsets, cache=PEER_SIGNAL_STORE_CACHE)


@gluon.jit
def _iris_column_routes_complete(
    routes,
    norm_weight,
    region,
    epochs,
    regions,
    row,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    STAGE_WORDS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Complete this rank's 512 columns of one route and publish them to every rank.

    Sums the eight partials in FP32 rank order with one BF16 rounding, as the
    Lamport two-shot all-reduce does, and publishes the BF16 columns and the
    partial sum of squares of the gained route that the portable inverse-RMS
    kernel's subgroup RANK computes.
    """
    PEERS: gl.constexpr = WORLD_SIZE - 1
    P1_WORDS: gl.constexpr = PEERS * MAX_ROWS * _ROUTE_WORDS
    P2_WORDS: gl.constexpr = WORLD_SIZE * MAX_ROWS * _ROUTE_WORDS
    token = row // 8
    route = row % 8
    stage = gl.load(epochs + token * WORLD_SIZE + RANK).to(gl.uint32) % 3
    base = (stage * STAGE_WORDS).to(gl.int32)
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    words = _iris_column_routes_offsets()
    own = _iris_sanitize_lamport_bf16x2(
        gl.amd.cdna4.buffer_load(
            routes.to(gl.pointer_type(gl.uint32), bitcast=True),
            row * _ROUTE_WORDS + RANK * _SLICE_WORDS + words,
        )
    )
    slices = base + row * _SLICE_WORDS + words
    received = ()
    missing = gl.zeros_like(slices)
    for slot in gl.static_range(PEERS):
        loaded = gl.amd.cdna4.buffer_load(
            words_ptr, slices + slot * (MAX_ROWS * _ROUTE_WORDS), cache=".cv"
        )
        received += (loaded,)
        missing |= ((loaded >> 16) == 0x8000).to(gl.int32)
    while _iris_subgroup_any(gl.max(missing, 1)):
        # Arrived words stay unchanged until cleared; reread them all.
        received = ()
        missing = gl.zeros_like(slices)
        for slot in gl.static_range(PEERS):
            loaded = gl.amd.cdna4.buffer_load(
                words_ptr, slices + slot * (MAX_ROWS * _ROUTE_WORDS), cache=".cv"
            )
            received += (loaded,)
            missing |= ((loaded >> 16) == 0x8000).to(gl.int32)
    for source in gl.static_range(WORLD_SIZE):
        if source == RANK:
            loaded = own
        else:
            loaded = received[(source - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        # Widening BF16 to FP32 places its bits in the high half of the word.
        even = (loaded << 16).to(gl.float32, bitcast=True)
        odd = (loaded & 0xFFFF0000).to(gl.float32, bitcast=True)
        if source == 0:
            sum_even = even
            sum_odd = odd
        else:
            sum_even += even
            sum_odd += odd
    completed = _iris_sanitize_lamport_bf16(
        gl.reshape(gl.join(sum_even, sum_odd), [_SLICE_FEATURES]).to(gl.bfloat16)
    )
    # The inverse-RMS kernel's subgroup layout: lane l holds columns 8l + [0, 8).
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
    columns = gl.arange(0, _SLICE_FEATURES, layout=layout)
    values = gl.convert_layout(completed, layout).to(gl.float32)
    gain = gl.load(norm_weight + RANK * _SLICE_FEATURES + columns).to(gl.float32)
    values *= (gain + WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)
    partial = gl.sum(values * values, 0)

    # Publish the completed columns in slot RANK of every rank, this one's
    # included (write-through: its readers may run on another XCD).
    published = gl.convert_layout(
        gl.reshape(completed, [64, 8]),
        gl.BlockedLayout([1, 8], [64, 1], [1, 1], [1, 0]),
    )
    p_lane = gl.arange(0, 64, layout=gl.SliceLayout(1, published.type.layout))
    p_elem = gl.arange(0, 8, layout=gl.SliceLayout(0, published.type.layout))
    p_offsets = 2 * (
        base + P1_WORDS + RANK * (MAX_ROWS * _ROUTE_WORDS) + row * _SLICE_WORDS
    ) + (p_lane[:, None] * 8 + p_elem[None, :])
    # Every rank's eight column owners each get a copy of the sum of squares:
    # words [token][source][route][owner].
    owners = gl.arange(0, WORLD_SIZE, layout=gl.BlockedLayout([1], [64], [1], [0]))
    squares = gl.full(owners.shape, 0, gl.float32, owners.type.layout) + partial
    s_offsets = (
        base
        + P1_WORDS
        + P2_WORDS
        + ((token * WORLD_SIZE + RANK) * 8 + route) * WORLD_SIZE
        + owners
    )
    own_base = _iris_group_heap_base(RANK, regions)
    gl.amd.cdna4.buffer_store(
        published,
        gl.multiple_of(own_base.to(gl.pointer_type(gl.bfloat16)), 16),
        p_offsets,
        cache=".wt",
    )
    gl.amd.cdna4.buffer_store(
        squares.to(gl.uint32, bitcast=True),
        gl.multiple_of(own_base.to(gl.pointer_type(gl.uint32)), 16),
        s_offsets,
        cache=".wt",
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        target = _iris_group_heap_base((RANK + delta) % WORLD_SIZE, regions)
        gl.amd.cdna4.buffer_store(
            published,
            gl.multiple_of(target.to(gl.pointer_type(gl.bfloat16)), 16),
            p_offsets,
            cache=PEER_SIGNAL_STORE_CACHE,
        )
        gl.amd.cdna4.buffer_store(
            squares.to(gl.uint32, bitcast=True),
            gl.multiple_of(target.to(gl.pointer_type(gl.uint32)), 16),
            s_offsets,
            cache=PEER_SIGNAL_STORE_CACHE,
        )
    # Restore the consumed markers.
    empty = gl.full_like(slices, 0x80000000).to(gl.uint32)
    for slot in gl.static_range(PEERS):
        gl.amd.cdna4.buffer_store(
            empty, words_ptr, slices + slot * (MAX_ROWS * _ROUTE_WORDS), cache=".wt"
        )


@gluon.jit
def _iris_column_routes_finish(
    route_weights,
    norm_weight,
    output,
    region,
    epochs,
    program,
    WORLD_SIZE: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    STAGE_WORDS: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
):
    """Normalize, weight and sum one token's routes over one rank's 512 columns.

    Polls that rank's completed columns of the token's eight routes and every
    rank's partial sums of squares, which only this program reads and
    clears, then advances the (token, owner) stage it alone advances.
    """
    PEERS: gl.constexpr = WORLD_SIZE - 1
    P1_WORDS: gl.constexpr = PEERS * MAX_ROWS * _ROUTE_WORDS
    P2_WORDS: gl.constexpr = WORLD_SIZE * MAX_ROWS * _ROUTE_WORDS
    token = program // WORLD_SIZE
    owner = program % WORLD_SIZE
    counter = epochs + token * WORLD_SIZE + owner
    stage = gl.load(counter).to(gl.uint32) % 3
    base = (stage * STAGE_WORDS).to(gl.int32)
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    # Every lane holds all eight routes of its four words.
    poll: gl.constexpr = gl.BlockedLayout([8, 1, 4], [1, 64, 1], [1, 1, 1], [2, 1, 0])
    route = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, poll)))
    lane = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, poll)))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, poll)))
    slices = (
        base
        + P1_WORDS
        + owner * (MAX_ROWS * _ROUTE_WORDS)
        + (token * 8 + route)[:, None, None] * _SLICE_WORDS
        + (lane[None, :, None] * 4 + word[None, None, :])
    )
    # Lane l reads the sum of squares of source l // 8, route l % 8.
    square_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    index = gl.arange(0, 64, layout=square_layout)
    sums = base + P1_WORDS + P2_WORDS + (token * 64 + index) * WORLD_SIZE + owner
    loaded = gl.amd.cdna4.buffer_load(words_ptr, slices, cache=".cv")
    squares = gl.amd.cdna4.buffer_load(words_ptr, sums, cache=".cv")
    # Payloads never hold negative zero (sums of squares are never negative),
    # so a word is empty exactly when its high half is 0x8000.
    missing = gl.max(gl.max(((loaded >> 16) == 0x8000).to(gl.int32), 2), 0)
    missing |= gl.convert_layout(
        ((squares >> 16) == 0x8000).to(gl.int32), missing.type.layout
    )
    while _iris_subgroup_any(missing):
        loaded = gl.amd.cdna4.buffer_load(words_ptr, slices, cache=".cv")
        squares = gl.amd.cdna4.buffer_load(words_ptr, sums, cache=".cv")
        missing = gl.max(gl.max(((loaded >> 16) == 0x8000).to(gl.int32), 2), 0)
        missing |= gl.convert_layout(
            ((squares >> 16) == 0x8000).to(gl.int32), missing.type.layout
        )
    gl.amd.cdna4.buffer_store(
        gl.full_like(loaded, 0x80000000), words_ptr, slices, cache=".wt"
    )
    gl.amd.cdna4.buffer_store(
        gl.full_like(squares, 0x80000000), words_ptr, sums, cache=".wt"
    )
    # Only a later launch reads the restored markers or this stage, so the
    # advance need not wait for the clears.
    gl.store(counter, ((stage + 1) % 3).to(gl.int32))

    # Add each route's eight partial sums (source s = 4a + 2b + c) as the
    # inverse-RMS kernel's cross-subgroup butterfly does:
    # ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)).
    partials = gl.convert_layout(
        gl.reshape(squares.to(gl.float32, bitcast=True), [2, 2, 2, 8]),
        gl.BlockedLayout([2, 2, 2, 8], [1, 1, 1, 64], [1, 1, 1, 1], [3, 2, 1, 0]),
    )
    a0, a1 = gl.split(gl.permute(partials, (1, 2, 3, 0)))
    b0, b1 = gl.split(gl.permute(a0 + a1, (1, 2, 0)))
    c0, c1 = gl.split(gl.permute(b0 + b1, (1, 0)))
    inv_rms = gl.rsqrt((c0 + c1) / _ROUTE_FEATURES + EPS)

    # Weighted route sum in route order, as the portable routed sum does.
    even = (loaded << 16).to(gl.float32, bitcast=True)
    odd = (loaded & 0xFFFF0000).to(gl.float32, bitcast=True)
    values = gl.reshape(gl.join(even, odd), [8, _SLICE_FEATURES])
    layout: gl.constexpr = values.type.layout
    v_column = gl.arange(0, _SLICE_FEATURES, layout=gl.SliceLayout(0, layout))
    gain = gl.load(norm_weight + owner * _SLICE_FEATURES + v_column).to(gl.float32)
    values *= (gain + WEIGHT_OFFSET).to(gl.bfloat16).to(gl.float32)[None, :]
    values *= gl.convert_layout(inv_rms, gl.SliceLayout(1, layout))[:, None]
    # Route r = 4a + 2b + c; splitting off c, then b, then a leaves each
    # route's columns in its own tensor.
    split_c = gl.split(gl.reshape(gl.permute(values, (1, 0)), [_SLICE_FEATURES, 4, 2]))
    split_cb = ()
    for c in gl.static_range(2):
        split_cb += (gl.split(gl.reshape(split_c[c], [_SLICE_FEATURES, 2, 2])),)
    for route in gl.static_range(8):
        selected = gl.split(split_cb[route % 2][route // 2 % 2])[route // 4]
        weight = gl.load(route_weights + token * 8 + route).to(gl.float32)
        if route == 0:
            combined = gl.zeros_like(selected)
        combined = gl.fma(selected, weight, combined)
    v_column = gl.arange(0, _SLICE_FEATURES, layout=combined.type.layout)
    gl.store(
        output + token * _ROUTE_FEATURES + owner * _SLICE_FEATURES + v_column,
        combined.to(gl.bfloat16),
    )


@gluon.jit(do_not_specialize=["rows"])
def iris_fused_lamport_column_routed_rmsnorm_sum(
    routes,
    route_weights,
    norm_weight,
    output,
    region,
    epochs,
    regions,
    rows,
    RANK: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Sum expert TP8 partial routes, apply the routed RMSNorm and weighted sum.

    routes holds this rank's partial routes [rows * 8, 4096] in token and
    route order; rank q owns columns [512q, 512q + 512) of every route.
    One-subgroup programs, grid (72 * rows,):
    1. 56 * rows senders each push this rank's partial of one route's 512
       columns to their owner.
    2. 8 * rows completers, one per route, sum the owned columns' eight
       partials in FP32 rank order with one BF16 rounding (the Lamport
       two-shot all-reduce's values) and publish them, with their partial sum
       of squares after the gain, to every rank.
    3. 8 * rows finishers, one per (token, column owner), add the token's
       partial sums of squares in the portable inverse-RMS kernel's order
       (so the inverse RMS matches it bit for bit) and accumulate
       fma(normalized route, route weight, sum) over the routes in order for
       those columns, writing output [rows, 4096]. The portable routed sum
       contracts that multiply-add only in some specializations, so the
       sums agree with it within one BF16 ULP.
    Every rank writes every column, so ranks agree bit for bit.
    region holds [stage][slices from peers, completed slices from every rank,
    sums of squares] of 32-bit words for MAX_ROWS tokens. epochs[t, q] is the
    stage of token t's mailboxes for owner q's columns; only finisher (t, q)
    advances it, after every local reader has used it. Negative zero is sent
    as zero.
    """
    WORLD_SIZE: gl.constexpr = len(regions)
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(gl.num_warps() == 1)
    STAGE_WORDS: gl.constexpr = MAX_ROWS * (
        (2 * WORLD_SIZE - 1) * _ROUTE_WORDS + WORLD_SIZE * WORLD_SIZE * 8
    )
    program = gl.program_id(0)
    senders = rows * 8 * (WORLD_SIZE - 1)
    # Waiting programs follow the programs they wait on, so every sender and
    # completer dispatches before a program that depends on it spins.
    if program < senders:
        _iris_column_routes_send(
            routes,
            epochs,
            regions,
            program,
            RANK,
            WORLD_SIZE,
            MAX_ROWS,
            STAGE_WORDS,
            PEER_SIGNAL_STORE_CACHE,
        )
    elif program < senders + rows * 8:
        _iris_column_routes_complete(
            routes,
            norm_weight,
            region,
            epochs,
            regions,
            program - senders,
            RANK,
            WORLD_SIZE,
            MAX_ROWS,
            STAGE_WORDS,
            WEIGHT_OFFSET,
            PEER_SIGNAL_STORE_CACHE,
        )
    else:
        _iris_column_routes_finish(
            route_weights,
            norm_weight,
            output,
            region,
            epochs,
            program - senders - rows * 8,
            WORLD_SIZE,
            MAX_ROWS,
            STAGE_WORDS,
            EPS,
            WEIGHT_OFFSET,
        )
