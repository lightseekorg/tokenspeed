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
def _iris_lamport_poll_bf16(addresses):
    """Wait for every peer BF16 pack and return its packed 32-bit words."""
    # Native Gluon waits for the whole workgroup; each lane reads 16 aligned bytes.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [0, 1])
    source_layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [8, 1], [0, 1])
    element = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    pointers = ()
    values = ()
    missing = gl.full((512, 4), False, gl.int1, layout)
    for peer in gl.static_range(len(addresses)):
        address = gl.convert_layout(addresses[peer], gl.SliceLayout(1, layout))
        pointer = gl.multiple_of(address.to(gl.pointer_type(gl.uint32)), 16)
        pointers += (pointer,)
        value = gl.load(
            pointer[:, None] + element[None, :], cache_modifier=".cv", volatile=True
        )
        values += (value,)
        missing |= ((value & 0xFFFF) == 0x8000) | ((value >> 16) == 0x8000)
    while gl.sum(gl.sum(missing.to(gl.int32), 1), 0) != 0:
        values = ()
        missing = gl.full((512, 4), False, gl.int1, layout)
        for peer in gl.static_range(len(addresses)):
            value = gl.load(
                pointers[peer][:, None] + element[None, :],
                cache_modifier=".cv",
                volatile=True,
            )
            values += (value,)
            missing |= ((value & 0xFFFF) == 0x8000) | ((value >> 16) == 0x8000)
    words = ()
    for peer in gl.static_range(len(addresses)):
        even, odd = gl.split(values[peer].reshape(512, 2, 2))
        w0, w2 = gl.split(even)
        w1, w3 = gl.split(odd)
        words += (
            (
                gl.convert_layout(w0, gl.SliceLayout(1, source_layout)),
                gl.convert_layout(w1, gl.SliceLayout(1, source_layout)),
                gl.convert_layout(w2, gl.SliceLayout(1, source_layout)),
                gl.convert_layout(w3, gl.SliceLayout(1, source_layout)),
            ),
        )
    return words


@gluon.jit
def _iris_lamport_unpack_bf16(w0, w1, w2, w3, element):
    """Extract BF16 elements from four packed 32-bit words."""
    bits = gl.where(
        element[None, :] < 2,
        w0[:, None],
        gl.where(
            element[None, :] < 4,
            w1[:, None],
            gl.where(element[None, :] < 6, w2[:, None], w3[:, None]),
        ),
    )
    return (
        (bits >> ((element[None, :] % 2) * 16))
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )


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
):
    """Normalize eight local routes per token and all-reduce their weighted sum.

    route_map selects local rows in routes; negative entries are ignored. Apply
    per-route RMSNorm before weighting and summing, round locally to BF16, and
    exchange sums through sentinel-marked mailboxes. Write BF16 output and clear
    the consumed mailbox slot. PORTABLE_MATH preserves the portable path's
    multiplication and rank-addition order; epochs select the rotating slot.
    """
    WORLD_SIZE: gl.constexpr = len(regions)
    # Preserve the arithmetic of each existing routed normalization path.
    gl.static_assert(WIDTH == 4096)
    gl.static_assert(gl.num_warps() == 8)
    token = gl.program_id(0)
    norm_layout: gl.constexpr = gl.BlockedLayout([8], [64], [8], [0])
    columns = gl.arange(0, WIDTH, layout=norm_layout)
    weight = gl.load(norm_weight + columns).to(gl.float32)
    weight = (weight + 1.0).to(gl.bfloat16).to(gl.float32)
    combined = gl.zeros((WIDTH,), gl.float32, layout=norm_layout)
    for slot in gl.static_range(8):
        route = token * 8 + slot
        source_row = gl.load(route_map + route)
        if source_row >= 0:
            values = gl.load(
                routes + source_row.to(gl.int64) * row_stride + columns
            ).to(gl.float32)
            values *= weight
            variance = gl.sum(values * values, 0) / WIDTH
            inv_rms = gl.rsqrt(variance + EPS)
            route_weight = gl.load(route_weights + route).to(gl.float32)
            if PORTABLE_MATH:
                combined += (values * inv_rms) * route_weight
            else:
                combined += values * (inv_rms * route_weight)
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [8, 1], [0, 1])
    pack = gl.arange(0, WIDTH // 8, layout=gl.SliceLayout(1, packed_layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, packed_layout))
    offsets = token * WIDTH + pack[:, None] * 8 + element[None, :]
    tile = token * (WIDTH // 512) + pack // 64
    stage = gl.load(epochs + tile).to(gl.uint32).to(gl.uint64) % 3
    slot = stage[:, None] * WORLD_SIZE * MAX_ELEMENTS
    local = _iris_sanitize_lamport_bf16(
        gl.convert_layout(
            gl.reshape(combined.to(gl.bfloat16), [WIDTH // 8, 8]), packed_layout
        )
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        address = _iris_group_heap_base(peer, regions)
        target = gl.multiple_of(address.to(gl.pointer_type(gl.bfloat16)), 16)
        gl.amd.cdna4.buffer_store(
            local,
            target,
            (slot + RANK * MAX_ELEMENTS + offsets).to(gl.int32),
            cache=".wt",
        )
    addresses = ()
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        addresses += (
            (
                region
                + stage * WORLD_SIZE * MAX_ELEMENTS
                + peer * MAX_ELEMENTS
                + token * WIDTH
                + pack * 8
            ).to(gl.uint64),
        )
    words = _iris_lamport_poll_bf16(addresses)
    values = ()
    for peer in gl.static_range(WORLD_SIZE - 1):
        word = words[peer]
        values += (
            _iris_lamport_unpack_bf16(word[0], word[1], word[2], word[3], element),
        )
    if PORTABLE_MATH:
        # The portable two-pass path rounds locally, then sums global rank order.
        for peer in gl.static_range(WORLD_SIZE):
            if peer == RANK:
                term = local
            else:
                term = values[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
            if peer == 0:
                reduced = term.to(gl.float32)
            else:
                reduced += term.to(gl.float32)
    else:
        # Preserve local first, followed by increasing global peer rank.
        reduced = local.to(gl.float32)
        for peer in gl.static_range(WORLD_SIZE):
            if peer != RANK:
                reduced += values[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1].to(
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
            sentinel,
            region,
            (slot + peer * MAX_ELEMENTS + offsets).to(gl.int32),
            cache=".wt",
        )
    # Every subgroup finishes resetting the mailbox before the next use.
    _iris_drain_subgroup_vmem()
    gl.barrier()
    gl.store(epochs + tile, ((stage + 1) % 3).to(gl.int32), mask=pack % 64 == 0)


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
