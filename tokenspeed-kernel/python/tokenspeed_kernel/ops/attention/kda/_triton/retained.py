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

# The retained commit descriptor ABI (KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS):
# one UINT64 row per layer, these columns.
_DESCRIPTOR_COLUMNS = tl.constexpr(12)
_QKV = tl.constexpr(0)
_CONV = tl.constexpr(1)
_PAGES = tl.constexpr(2)
_QKV_ROW_STRIDE = tl.constexpr(3)
_CONV_PAGE_STRIDE = tl.constexpr(4)
_CONV_CHANNEL_STRIDE = tl.constexpr(5)
_CONV_TAP_STRIDE = tl.constexpr(6)
_STATE = tl.constexpr(7)
_STATE_PAGE_STRIDE = tl.constexpr(8)
_STATE_HEAD_STRIDE = tl.constexpr(9)
_STATE_VALUE_STRIDE = tl.constexpr(10)
_STATE_KEY_STRIDE = tl.constexpr(11)
# A layer's verdict on a request row's commit (commit_verdicts): the layer does
# not carry the row, rejects its commit, or accepts it.
_VERDICT_IDLE = tl.constexpr(0)
_VERDICT_REJECTED = tl.constexpr(1)
_VERDICT_ACCEPTED = tl.constexpr(2)
# Records a fold applies per unrolled step, so their loads issue together.
_FOLD_RECORDS = tl.constexpr(8)


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


@triton.jit(
    do_not_specialize=[
        "rows",
        "group_count",
        "live_group_stride",
        "live_row_stride",
        "slot_stride",
        "slot_count",
        "length_stride",
        "start_stride",
        "pages",
    ]
)
def retained_snapshot_kernel(
    length,
    start,
    record_slots,
    lives,
    rows,
    group_count,
    live_group_stride,
    live_row_stride,
    slot_stride,
    slot_count,
    length_stride,
    start_stride,
    pages,
    BLOCK: tl.constexpr,
):
    """Copy each prefill row's count to its start, then clear stored rows' counts.

    A row's scan stores a dense state into its live page in every group or,
    as a padding row, in none: the count the groups share cannot describe a
    partial store, so a row with live pages in some groups only fails.
    """
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_batch = row < rows
    slot = tl.load(record_slots + row * slot_stride, mask=in_batch, other=-1).to(
        tl.int64
    )
    resident = in_batch & (slot >= 0) & (slot < slot_count)
    count = tl.load(length + slot * length_stride, mask=resident, other=0)
    tl.store(start + slot * start_stride, count, mask=resident)
    stored_groups = tl.zeros([BLOCK], dtype=tl.int32)
    for group in range(group_count):
        live = tl.load(
            lives + group.to(tl.int64) * live_group_stride + row * live_row_stride,
            mask=in_batch,
            other=0,
        ).to(tl.int64)
        stored_groups += ((live > 0) & (live < pages)).to(tl.int32)
    stored = stored_groups == group_count
    tl.device_assert(
        stored | (stored_groups == 0) | (resident == 0),
        "a prefill row has live pages in some state groups only",
    )
    tl.store(length + slot * length_stride, 0, mask=resident & stored)


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
    replay_start,
    layer,
    row,
    group_count,
    read_group_stride,
    read_row_stride,
    write_group_stride,
    write_row_stride,
    accepted_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
    start_stride,
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """One layer's checks of a request row's commit.

    ``live`` says the layer carries the row: its group's source and live
    pages lie in the pool, the record slot is resident and the window is
    non-empty. ``valid`` adds the capacity of the round's start count and
    every accepted candidate's stamp.
    """
    group = tl.load(groups + layer).to(tl.int64)
    valid = (group >= 0) & (group < group_count)
    source = tl.load(
        reads + group * read_group_stride + row * read_row_stride, mask=valid, other=0
    ).to(tl.int64)
    target = tl.load(
        writes + group * write_group_stride + row * write_row_stride,
        mask=valid,
        other=0,
    ).to(tl.int64)
    pages = tl.load(descriptors + layer * _DESCRIPTOR_COLUMNS + _PAGES).to(tl.int64)
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
    # The row carries a request of this forward, whatever its commit holds.
    live = valid
    take = tl.minimum(
        tl.maximum(tl.load(accepted + row * accepted_stride), 0),
        tl.minimum(count, WINDOW),
    )
    # The records the round's verify read before its candidates: the count
    # the prepare left for the request's slot, the same in every layer.
    old = tl.load(replay_start + record_slot * start_stride, mask=valid, other=0)
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
    return source, target, begin, take, old, valid & fresh, live, record_slot


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
    state = tl.load(base + _STATE).to(tl.pointer_type(STATE))
    page_stride = tl.load(base + _STATE_PAGE_STRIDE).to(tl.int64)
    head_stride = tl.load(base + _STATE_HEAD_STRIDE).to(tl.int64)
    value_stride = tl.load(base + _STATE_VALUE_STRIDE).to(tl.int64)
    key_stride = tl.load(base + _STATE_KEY_STRIDE).to(tl.int64)
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


# Triton specializes each element of a tuple argument whatever
# do_not_specialize lists, so a stride that follows the batch size -- the row
# stride of a [groups, batch] page table -- is a scalar here. The tuples left
# (stamps and records) hold strides fixed by their storage layout.
_BATCHED_RUNTIME_ARGUMENTS = [
    "group_count",
    "read_group_stride",
    "read_row_stride",
    "write_group_stride",
    "write_row_stride",
    "accepted_stride",
    "query_stride",
    "record_slot_stride",
    "record_slot_count",
    "start_stride",
    "verdict_stride",
    "stamp_slots",
    "stamp_capacity",
    "stopped_count",
]


@triton.jit(do_not_specialize=_BATCHED_RUNTIME_ARGUMENTS)
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
    replay_start,
    verdicts,
    group_count,
    read_group_stride,
    read_row_stride,
    write_group_stride,
    write_row_stride,
    accepted_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
    start_stride,
    verdict_stride,
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
    source, target, begin, take, old, valid, live, record_slot = (
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
            replay_start,
            layer,
            row,
            group_count,
            read_group_stride,
            read_row_stride,
            write_group_stride,
            write_row_stride,
            accepted_stride,
            query_stride,
            record_slot_stride,
            record_slot_count,
            start_stride,
            stamp_strides,
            stamp_slots,
            stamp_capacity,
            CAPACITY,
            WINDOW,
        )
    )
    if tl.program_id(2) == 0:
        # The publish launch reduces every layer's verdict on the row.
        verdict = tl.where(
            valid, _VERDICT_ACCEPTED, tl.where(live, _VERDICT_REJECTED, _VERDICT_IDLE)
        )
        tl.store(verdicts + row * verdict_stride + layer, verdict.to(tl.int8))
    if not valid:
        return
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    original = tl.load(base + _QKV).to(tl.pointer_type(ACTIVATION))
    conv = tl.load(base + _CONV).to(tl.pointer_type(CONV))
    qkv_stride = tl.load(base + _QKV_ROW_STRIDE).to(tl.int64)
    conv_stride = tl.load(base + _CONV_PAGE_STRIDE).to(tl.int64)
    channel_stride = tl.load(base + _CONV_CHANNEL_STRIDE).to(tl.int64)
    tap_stride = tl.load(base + _CONV_TAP_STRIDE).to(tl.int64)
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


@triton.jit(
    do_not_specialize=[
        "layers",
        "accepted_stride",
        "query_stride",
        "record_slot_stride",
        "length_stride",
        "start_stride",
        "verdict_stride",
        "stopped_count",
    ]
)
def retained_batched_publish_kernel(
    accepted,
    starts,
    record_slots,
    request_stopped,
    replay_length,
    replay_start,
    verdicts,
    request_ends,
    layers,
    accepted_stride,
    query_stride,
    record_slot_stride,
    length_stride,
    start_stride,
    verdict_stride,
    stopped_count,
    WINDOW: tl.constexpr,
    LAYERS: tl.constexpr,
    INCOMPLETE: tl.constexpr,
):
    """Publish one request row's count from every layer's commit verdict.

    Every layer's records follow the one count the request's slot keeps, so
    it advances only when every layer accepted the round. A request still
    decoding whose commit some layers accepted and others did not fails here
    rather than replay records some layers never wrote. A request that ended
    in this forward keeps the ``request_ends`` contract instead: a layer that
    carried it but rejected its commit left its records outside the live
    page, so the row reports ``INCOMPLETE``.
    """
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, LAYERS)
    verdict = tl.load(
        verdicts + row * verdict_stride + lanes,
        mask=lanes < layers,
        other=_VERDICT_IDLE,
    )
    accepting = tl.sum((verdict == _VERDICT_ACCEPTED).to(tl.int32), 0)
    rejecting = tl.sum((verdict == _VERDICT_REJECTED).to(tl.int32), 0)
    # A layer that carries the row validated its record slot.
    carried = (accepting + rejecting) > 0
    record_slot = tl.load(record_slots + row * record_slot_stride).to(tl.int64)
    begin = tl.load(starts + row * query_stride).to(tl.int64)
    count = tl.load(starts + (row + 1) * query_stride).to(tl.int64) - begin
    take = tl.minimum(
        tl.maximum(tl.load(accepted + row * accepted_stride), 0),
        tl.minimum(count, WINDOW),
    )
    old = tl.load(replay_start + record_slot * start_stride, mask=carried, other=0)
    stopped = tl.load(
        request_stopped + record_slot,
        mask=carried & (record_slot < stopped_count),
        other=0,
    )
    tl.device_assert(
        (stopped != 0) | (accepting == 0) | (accepting == layers),
        "a retained commit accepted by some KDA layers only",
    )
    if accepting == layers:
        # An ended request's records are folded into its live page.
        tl.store(
            replay_length + record_slot * length_stride,
            tl.where(stopped != 0, 0, old + take).to(tl.int32),
        )
    tl.store(request_ends + row, INCOMPLETE, mask=(stopped != 0) & (rejecting > 0))


# Readying retained history for a verify forward, every layer in one launch.
# Shared by both vendors. As for the commit, the page tables' strides are
# scalars, since a table's row stride follows the padded batch size; the
# record stride tuples are fixed by the records' storage.
_PREPARE_RUNTIME_ARGUMENTS = [
    "group_count",
    "source_group_stride",
    "source_row_stride",
    "live_group_stride",
    "live_row_stride",
    "query_stride",
    "record_slot_stride",
    "record_slot_count",
    "length_stride",
    "start_stride",
]


@triton.jit(do_not_specialize=_PREPARE_RUNTIME_ARGUMENTS)
def retained_batched_prepare_kernel(
    descriptors,
    groups,
    sources,
    lives,
    starts,
    record_slots,
    replay_length,
    replay_start,
    correction,
    retained_key,
    retained_decay,
    group_count,
    source_group_stride,
    source_row_stride,
    live_group_stride,
    live_row_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
    length_stride,
    start_stride,
    correction_strides,
    key_strides,
    decay_strides,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    GROUPS: tl.constexpr,
    STATE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KEYS: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Pick a request row's fold, fold it into this layer's live page, snapshot.

    Every layer of a row decides on the same inputs: the count L of its
    record slot and the source and live pages of every state group. A row
    takes part when its pages lie inside the pool in some group, its window
    is non-empty and its record slot is resident; then 0 <= L <= CAPACITY
    and 0 < T <= CAPACITY. It folds when it moves to a private live page in
    any group or when L plus the larger of T and the verify's WINDOW exceeds
    CAPACITY: the verify kernels read at most CAPACITY - WINDOW records, so a
    row whose window is shorter than the verify's folds no later than a full
    one. Program (layer, row, split) writes source state plus records [0, L)
    into this layer's live page for every head h with h % splits == split,
    in ROWS x KEYS tiles; KEYS is DIM rounded up to a power of two. The
    first program of a row stores the round's start count -- 0 after a fold,
    else L -- which the verify and the commit read; the count itself and the
    records stay untouched.
    """
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    pages = tl.load(descriptors + layer * _DESCRIPTOR_COLUMNS + _PAGES).to(tl.int64)
    group_lanes = tl.arange(0, GROUPS)
    listed = group_lanes < group_count
    group_rows = group_lanes.to(tl.int64)
    group_sources = tl.load(
        sources + group_rows * source_group_stride + row * source_row_stride,
        mask=listed,
        other=0,
    ).to(tl.int64)
    group_lives = tl.load(
        lives + group_rows * live_group_stride + row * live_row_stride,
        mask=listed,
        other=0,
    ).to(tl.int64)
    paged = (
        listed
        & (group_sources > 0)
        & (group_sources < pages)
        & (group_lives > 0)
        & (group_lives < pages)
    )
    moved = tl.sum((paged & (group_sources != group_lives)).to(tl.int32), 0) > 0
    begin = tl.load(starts + row * query_stride).to(tl.int64)
    tokens = tl.load(starts + (row + 1) * query_stride).to(tl.int64) - begin
    slot = tl.load(record_slots + row * record_slot_stride).to(tl.int64)
    active = (
        (tl.sum(paged.to(tl.int32), 0) > 0)
        & (tokens != 0)
        & (slot >= 0)
        & (slot < record_slot_count)
    )
    # The accepted count belongs to the request slot, not to either page.
    count = tl.load(replay_length + slot * length_stride, mask=active, other=0)
    valid = (count >= 0) & (count <= CAPACITY) & (tokens > 0) & (tokens <= CAPACITY)
    fold = active & valid & (moved | (count + tl.maximum(tokens, WINDOW) > CAPACITY))
    if (layer == 0) & (tl.program_id(2) == 0):
        tl.device_assert(valid | (active == 0), "invalid retained history interval")
        tl.store(
            replay_start + slot * start_stride,
            tl.where(fold, 0, count),
            mask=active & valid,
        )
    mine = group_lanes == tl.load(groups + layer)
    if not (fold & (tl.sum((mine & paged).to(tl.int32), 0) > 0)):
        return
    source = tl.sum(tl.where(mine, group_sources, 0), 0)
    live = tl.sum(tl.where(mine, group_lives, 0), 0)
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    state = tl.load(base + _STATE).to(tl.pointer_type(STATE))
    page_stride = tl.load(base + _STATE_PAGE_STRIDE).to(tl.int64)
    head_stride = tl.load(base + _STATE_HEAD_STRIDE).to(tl.int64)
    value_stride = tl.load(base + _STATE_VALUE_STRIDE).to(tl.int64)
    key_stride = tl.load(base + _STATE_KEY_STRIDE).to(tl.int64)
    source_page = state + source * page_stride
    live_page = state + live * page_stride
    u = correction + slot * correction_strides[0] + layer * correction_strides[1]
    k = retained_key + slot * key_strides[0] + layer * key_strides[1]
    d = retained_decay + slot * decay_strides[0] + layer * decay_strides[1]
    keys = tl.arange(0, KEYS)
    key_mask = keys < DIM
    for head in range(tl.program_id(2), HEADS, tl.num_programs(2)):
        for start in range(0, DIM, ROWS):
            values = start + tl.arange(0, ROWS)
            value_mask = values < DIM
            mask = value_mask[:, None] & key_mask[None, :]
            tile = (
                head * head_stride
                + values[:, None] * value_stride
                + keys[None, :] * key_stride
            )
            running = tl.load(source_page + tile, mask=mask, other=0).to(tl.float32)
            # Positions past the count are identity records (decay 1, u 0,
            # k -0), which leave every value unchanged, a zero's sign included.
            for first in range(0, count, _FOLD_RECORDS):
                for offset in tl.static_range(_FOLD_RECORDS):
                    position = first + offset
                    present = position < count
                    uv = tl.load(
                        u
                        + head * correction_strides[2]
                        + position * correction_strides[3]
                        + values * correction_strides[4],
                        mask=value_mask & present,
                        other=0.0,
                    )
                    kv = tl.load(
                        k
                        + head * key_strides[2]
                        + position * key_strides[3]
                        + keys * key_strides[4],
                        mask=key_mask & present,
                        other=-0.0,
                    )
                    dv = tl.load(
                        d
                        + head * decay_strides[2]
                        + position * decay_strides[3]
                        + keys * decay_strides[4],
                        mask=key_mask & present,
                        other=1.0,
                    )
                    running = running * dv[None, :] + uv[:, None] * kv[None, :]
            tl.store(live_page + tile, running.to(STATE), mask=mask)
