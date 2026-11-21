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

# Row width of the retained commit descriptor ABI
# (KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS).
_DESCRIPTOR_COLUMNS = tl.constexpr(14)


@triton.jit(
    do_not_specialize=[
        "state_strides",
        "u_strides",
        "k_strides",
        "d_strides",
        "length_stride",
        "source_stride",
        "slot_stride",
        "slots",
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
    record_slots,
    state_strides,
    u_strides,
    k_strides,
    d_strides,
    length_stride,
    source_stride,
    slot_stride,
    pages,
    slots,
    PAGED: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CAPACITY: tl.constexpr,
    V_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source = tl.load(sources + row * source_stride).to(tl.int64)
    slot = tl.load(record_slots + row * slot_stride).to(tl.int64)
    valid = (source > 0) & (slot >= 0) & (slot < slots)
    if PAGED:
        valid = valid & (source < pages)
    count = tl.load(length + slot * length_stride, mask=valid, other=0)
    if count <= 0 or count > CAPACITY:
        return
    state_row = source if PAGED else row
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    v, k = (offsets // K).to(tl.int64), (offsets % K).to(tl.int64)
    if V_MAJOR:
        ptr = (
            state
            + state_row * state_strides[0]
            + head * state_strides[1]
            + v * state_strides[2]
            + k * state_strides[3]
        )
    else:
        ptr = (
            state
            + state_row * state_strides[0]
            + head * state_strides[1]
            + k * state_strides[2]
            + v * state_strides[3]
        )
    mask = v < V
    value = tl.load(ptr, mask=mask, other=0).to(tl.float32)
    for token in range(count):
        uv = tl.load(
            u
            + slot * u_strides[0]
            + head * u_strides[1]
            + token * u_strides[2]
            + v * u_strides[3],
            mask=mask,
            other=0,
        )
        kv = tl.load(
            key
            + slot * k_strides[0]
            + head * k_strides[1]
            + token * k_strides[2]
            + k * k_strides[3],
            mask=mask,
            other=0,
        )
        dv = tl.load(
            decay
            + slot * d_strides[0]
            + head * d_strides[1]
            + token * d_strides[2]
            + k * d_strides[3],
            mask=mask,
            other=0,
        )
        value = value * dv + uv * kv
    tl.store(ptr, value, mask=mask)


@triton.jit(
    do_not_specialize=[
        "rows",
        "length_stride",
        "index_stride",
        "slot_stride",
        "slot_count",
        "pages",
    ]
)
def retained_reset_length_kernel(
    length,
    indices,
    record_slots,
    slot_stride,
    slot_count,
    pages,
    rows,
    length_stride,
    index_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(indices + row * index_stride, mask=row < rows, other=0).to(tl.int64)
    slot = tl.load(record_slots + row * slot_stride, mask=row < rows, other=-1).to(
        tl.int64
    )
    # Padding rows carry a page at or below 0, or one past the pool, as the
    # materialize kernel reads them; their slots may belong to others.
    valid = (
        (row < rows) & (page > 0) & (page < pages) & (slot >= 0) & (slot < slot_count)
    )
    tl.store(
        length + slot * length_stride,
        0,
        mask=valid,
    )


@triton.jit
def retained_batched_accept(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    record_slots,
    metadata,
    stamps,
    layer,
    row,
    group_count,
    read_strides,
    write_strides,
    accepted_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
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
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    pages = tl.load(base + 3).to(tl.int64)
    length = tl.load(base + 2).to(tl.pointer_type(tl.int32))
    length_stride = tl.load(base + 6).to(tl.int64)
    begin = tl.load(starts + row * query_stride).to(tl.int64)
    count = tl.load(starts + (row + 1) * query_stride).to(tl.int64) - begin
    record_slot = tl.load(record_slots + row * record_slot_stride).to(tl.int64)
    valid = (
        valid
        & (source > 0)
        & (source < pages)
        & (target > 0)
        & (target < pages)
        & (record_slot >= 0)
        & (record_slot < record_slot_count)
        & (begin >= 0)
        & (count > 0)
        & (count <= WINDOW)
    )
    take = tl.minimum(
        tl.maximum(tl.load(accepted + row * accepted_stride), 0),
        tl.minimum(count, WINDOW),
    )
    # The request's accepted count lives in its slot, not in the cache page.
    length = length + record_slot * length_stride
    old = tl.load(length, mask=valid, other=0)
    valid = valid & (old >= 0) & (old + take <= CAPACITY)
    stamp_slot = tl.load(metadata + row * 3).to(tl.int64)
    generation = tl.load(metadata + row * 3 + 1)
    absolute = tl.load(metadata + row * 3 + 2)
    valid = (
        valid
        & (stamp_slot >= 0)
        & (stamp_slot < stamp_slots)
        & (generation >= 0)
        & (absolute >= 0)
    )
    offsets = tl.arange(0, triton.next_power_of_2(WINDOW))
    positions = absolute + offsets
    ptr = (
        stamps
        + layer * stamp_strides[0]
        + stamp_slot * stamp_strides[1]
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
    return source, target, begin, take, old, valid & fresh, length, record_slot


@triton.jit
def retained_fold_ended_request(
    descriptors,
    correction,
    retained_key,
    retained_decay,
    layer,
    target,
    record_slot,
    count,
    correction_strides,
    key_strides,
    decay_strides,
    STATE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold a slot's first ``count`` records into its live page's dense state.

    The heads are split across axis 2 of the launch grid; each program
    applies the records to its heads' ``[value, key]`` planes in order.
    """
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    state = tl.load(base + 9).to(tl.pointer_type(STATE))
    page_stride = tl.load(base + 10).to(tl.int64)
    head_stride = tl.load(base + 11).to(tl.int64)
    value_stride = tl.load(base + 12).to(tl.int64)
    key_stride = tl.load(base + 13).to(tl.int64)
    u = correction + record_slot * correction_strides[0] + layer * correction_strides[1]
    k = retained_key + record_slot * key_strides[0] + layer * key_strides[1]
    d = retained_decay + record_slot * decay_strides[0] + layer * decay_strides[1]
    page = state + target * page_stride
    lanes = tl.arange(0, BLOCK)
    for head in range(tl.program_id(2), HEADS, tl.num_programs(2)):
        for start in range(0, DIM * DIM, BLOCK):
            offsets = start + lanes
            v, c = (offsets // DIM).to(tl.int64), (offsets % DIM).to(tl.int64)
            mask = v < DIM
            ptr = page + head * head_stride + v * value_stride + c * key_stride
            value = tl.load(ptr, mask=mask, other=0).to(tl.float32)
            for token in range(count):
                uv = tl.load(
                    u
                    + head * correction_strides[2]
                    + token * correction_strides[3]
                    + v * correction_strides[4],
                    mask=mask,
                    other=0,
                )
                kv = tl.load(
                    k
                    + head * key_strides[2]
                    + token * key_strides[3]
                    + c * key_strides[4],
                    mask=mask,
                    other=0,
                )
                dv = tl.load(
                    d
                    + head * decay_strides[2]
                    + token * decay_strides[3]
                    + c * decay_strides[4],
                    mask=mask,
                    other=0,
                )
                value = value * dv + uv * kv
            tl.store(ptr, value, mask=mask)


_BATCHED_RUNTIME_ARGUMENTS = [
    "group_count",
    "read_strides",
    "write_strides",
    "accepted_stride",
    "query_stride",
    "record_slot_stride",
    "record_slot_count",
    "stamp_strides",
    "stamp_slots",
    "stamp_capacity",
    "stopped_count",
]


@triton.jit(
    do_not_specialize=_BATCHED_RUNTIME_ARGUMENTS
    + ["correction_strides", "key_strides", "decay_strides"]
)
def retained_batched_conv_kernel(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    record_slots,
    metadata,
    stamps,
    request_stopped,
    group_count,
    read_strides,
    write_strides,
    accepted_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    stopped_count,
    correction,
    retained_key,
    retained_decay,
    correction_strides,
    key_strides,
    decay_strides,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    CHANNELS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    CONV: tl.constexpr,
    STATE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source, target, begin, take, old, valid, length, record_slot = (
        retained_batched_accept(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            starts,
            record_slots,
            metadata,
            stamps,
            layer,
            row,
            group_count,
            read_strides,
            write_strides,
            accepted_stride,
            query_stride,
            record_slot_stride,
            record_slot_count,
            stamp_strides,
            stamp_slots,
            stamp_capacity,
            CAPACITY,
            WINDOW,
        )
    )
    if not valid:
        return
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
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
    # The request ended in this forward: its live page becomes a dense
    # checkpoint others may read. The publish launch clears the count after
    # every program of this one has folded its heads.
    if tl.load(
        request_stopped + record_slot, mask=record_slot < stopped_count, other=0
    ):
        retained_fold_ended_request(
            descriptors,
            correction,
            retained_key,
            retained_decay,
            layer,
            target,
            record_slot,
            # Validated against the history capacity above.
            (old + take).to(tl.int32),
            correction_strides,
            key_strides,
            decay_strides,
            STATE,
            HEADS,
            DIM,
            BLOCK,
        )


@triton.jit(do_not_specialize=_BATCHED_RUNTIME_ARGUMENTS)
def retained_batched_publish_kernel(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    record_slots,
    metadata,
    stamps,
    request_stopped,
    group_count,
    read_strides,
    write_strides,
    accepted_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    stopped_count,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
):
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source, target, begin, take, old, valid, length, record_slot = (
        retained_batched_accept(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            starts,
            record_slots,
            metadata,
            stamps,
            layer,
            row,
            group_count,
            read_strides,
            write_strides,
            accepted_stride,
            query_stride,
            record_slot_stride,
            record_slot_count,
            stamp_strides,
            stamp_slots,
            stamp_capacity,
            CAPACITY,
            WINDOW,
        )
    )
    if valid:
        stopped = tl.load(
            request_stopped + record_slot, mask=record_slot < stopped_count, other=0
        )
        # An ended request's records are folded into its live page.
        tl.store(length, tl.where(stopped, 0, old + take))
