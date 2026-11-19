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

"""Retained state utilities with runtime arena and request strides."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit(
    do_not_specialize=[
        "state_strides",
        "u_strides",
        "k_strides",
        "d_strides",
        "length_stride",
        "source_stride",
        "pages",
    ]
)
def retained_materialize_kernel(
    state,
    u,
    key,
    decay,
    length,
    sources,
    state_strides,
    u_strides,
    k_strides,
    d_strides,
    length_stride,
    source_stride,
    pages,
    K: tl.constexpr,
    V: tl.constexpr,
    CAPACITY: tl.constexpr,
    V_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source = tl.load(sources + row * source_stride).to(tl.int64)
    valid = (source > 0) & (source < pages)
    count = tl.load(length + source * length_stride, mask=valid, other=0)
    valid = valid & (count >= 0) & (count <= CAPACITY)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    v, k = (offsets // K).to(tl.int64), (offsets % K).to(tl.int64)
    if V_MAJOR:
        ptr = (
            state
            + row * state_strides[0]
            + head * state_strides[1]
            + v * state_strides[2]
            + k * state_strides[3]
        )
    else:
        ptr = (
            state
            + row * state_strides[0]
            + head * state_strides[1]
            + k * state_strides[2]
            + v * state_strides[3]
        )
    mask = valid & (v < V)
    value = tl.load(ptr, mask=mask, other=0).to(tl.float32)
    for token in range(tl.where(valid, count, 0)):
        uv = tl.load(
            u
            + source * u_strides[0]
            + head * u_strides[1]
            + token * u_strides[2]
            + v * u_strides[3],
            mask=mask,
            other=0,
        )
        kv = tl.load(
            key
            + source * k_strides[0]
            + head * k_strides[1]
            + token * k_strides[2]
            + k * k_strides[3],
            mask=mask,
            other=0,
        )
        dv = tl.load(
            decay
            + source * d_strides[0]
            + head * d_strides[1]
            + token * d_strides[2]
            + k * d_strides[3],
            mask=mask,
            other=0,
        )
        value = value * dv + uv * kv
    tl.store(ptr, value, mask=mask)


@triton.jit(do_not_specialize=["rows", "pages", "length_stride", "index_stride"])
def retained_reset_length_kernel(
    length, indices, rows, pages, length_stride, index_stride, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(indices + row * index_stride, mask=row < rows, other=0).to(tl.int64)
    tl.store(
        length + page * length_stride,
        0,
        mask=(row < rows) & (page > 0) & (page < pages),
    )


@triton.jit
def retained_batched_accept(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    metadata,
    stamps,
    layer,
    row,
    group_count,
    read_strides,
    write_strides,
    accepted_stride,
    query_stride,
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
):
    group = tl.load(groups + layer).to(tl.int64)
    valid = (group >= 0) & (group < group_count)
    source = tl.load(
        reads + group * read_strides[0] + row * read_strides[1], mask=valid, other=0
    ).to(tl.int64)
    target = tl.load(
        writes + group * write_strides[0] + row * write_strides[1], mask=valid, other=0
    ).to(tl.int64)
    pages = tl.load(descriptors + layer * 9 + 3).to(tl.int64)
    length = tl.load(descriptors + layer * 9 + 2).to(tl.pointer_type(tl.int32))
    length_stride = tl.load(descriptors + layer * 9 + 6).to(tl.int64)
    begin = tl.load(starts + row * query_stride).to(tl.int64)
    count = tl.load(starts + (row + 1) * query_stride).to(tl.int64) - begin
    valid = (
        valid
        & (source > 0)
        & (source < pages)
        & (target > 0)
        & (target < pages)
        & (begin >= 0)
        & (count > 0)
        & (count <= WINDOW)
    )
    take = tl.minimum(
        tl.maximum(tl.load(accepted + row * accepted_stride), 0),
        tl.minimum(count, WINDOW),
    )
    old = tl.load(length + target * length_stride, mask=valid, other=0)
    valid = valid & (old >= 0) & (old + take <= CAPACITY)
    slot = tl.load(metadata + row * 3).to(tl.int64)
    generation = tl.load(metadata + row * 3 + 1)
    absolute = tl.load(metadata + row * 3 + 2)
    valid = (
        valid & (slot >= 0) & (slot < stamp_slots) & (generation >= 0) & (absolute >= 0)
    )
    offsets = tl.arange(0, triton.next_power_of_2(WINDOW))
    positions = absolute + offsets
    ptr = (
        stamps
        + layer * stamp_strides[0]
        + slot * stamp_strides[1]
        + (positions % stamp_capacity) * stamp_strides[2]
    )
    mask = valid & (offsets < take)
    position = tl.load(ptr, mask=mask, other=-1)
    stamp_generation = tl.load(ptr + stamp_strides[3], mask=mask, other=-1)
    fresh = (
        tl.sum(
            (mask & ((position != positions) | (stamp_generation != generation))).to(
                tl.int32
            ),
            0,
        )
        == 0
    )
    return source, target, begin, take, old, valid & fresh, length, length_stride


@triton.jit(
    do_not_specialize=[
        "group_count",
        "read_strides",
        "write_strides",
        "accepted_stride",
        "query_stride",
        "stamp_strides",
        "stamp_slots",
        "stamp_capacity",
    ]
)
def retained_batched_conv_kernel(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    metadata,
    stamps,
    group_count,
    read_strides,
    write_strides,
    accepted_stride,
    query_stride,
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    CHANNELS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    CONV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source, target, begin, take, old, valid, length, length_stride = (
        retained_batched_accept(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            starts,
            metadata,
            stamps,
            layer,
            row,
            group_count,
            read_strides,
            write_strides,
            accepted_stride,
            query_stride,
            stamp_strides,
            stamp_slots,
            stamp_capacity,
            CAPACITY,
            WINDOW,
        )
    )
    if not valid:
        return
    base = descriptors + layer * 9
    original = tl.load(base).to(tl.pointer_type(ACTIVATION))
    conv = tl.load(base + 1).to(tl.pointer_type(CONV))
    qkv_stride = tl.load(base + 4).to(tl.int64)
    conv_stride = tl.load(base + 5).to(tl.int64)
    channel_stride = tl.load(base + 7).to(tl.int64)
    tap_stride = tl.load(base + 8).to(tl.int64)
    channel = (tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = channel < CHANNELS
    ptr = conv + source * conv_stride + channel * channel_stride
    h0 = tl.load(ptr, mask=mask, other=0)
    h1 = tl.load(ptr + tap_stride, mask=mask, other=0)
    h2 = tl.load(ptr + 2 * tap_stride, mask=mask, other=0)
    for token in tl.static_range(WINDOW):
        x = tl.load(
            original + (begin + token) * qkv_stride + channel,
            mask=mask & (token < take),
            other=0,
        ).to(CONV)
        h0 = tl.where(token < take, h1, h0)
        h1 = tl.where(token < take, h2, h1)
        h2 = tl.where(token < take, x, h2)
    ptr = conv + target * conv_stride + channel * channel_stride
    tl.store(ptr, h0, mask=mask)
    tl.store(ptr + tap_stride, h1, mask=mask)
    tl.store(ptr + 2 * tap_stride, h2, mask=mask)


@triton.jit(
    do_not_specialize=[
        "group_count",
        "read_strides",
        "write_strides",
        "accepted_stride",
        "query_stride",
        "stamp_strides",
        "stamp_slots",
        "stamp_capacity",
    ]
)
def retained_batched_publish_kernel(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    metadata,
    stamps,
    group_count,
    read_strides,
    write_strides,
    accepted_stride,
    query_stride,
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
):
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source, target, begin, take, old, valid, length, length_stride = (
        retained_batched_accept(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            starts,
            metadata,
            stamps,
            layer,
            row,
            group_count,
            read_strides,
            write_strides,
            accepted_stride,
            query_stride,
            stamp_strides,
            stamp_slots,
            stamp_capacity,
            CAPACITY,
            WINDOW,
        )
    )
    if valid:
        tl.store(length + target * length_stride, old + take)
