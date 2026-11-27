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
_DESCRIPTOR_COLUMNS = tl.constexpr(13)
_RING = tl.constexpr(0)
_CONV = tl.constexpr(1)
_PAGES = tl.constexpr(2)
_RING_ROW_STRIDE = tl.constexpr(3)
_CONV_PAGE_STRIDE = tl.constexpr(4)
_CONV_CHANNEL_STRIDE = tl.constexpr(5)
_CONV_TAP_STRIDE = tl.constexpr(6)
_STATE = tl.constexpr(7)
_STATE_PAGE_STRIDE = tl.constexpr(8)
_STATE_HEAD_STRIDE = tl.constexpr(9)
_STATE_VALUE_STRIDE = tl.constexpr(10)
_STATE_KEY_STRIDE = tl.constexpr(11)
_RING_SLOT_STRIDE = tl.constexpr(12)
# The pre-convolution ring keeps the convolution's 3 past inputs ahead of a
# verify window's candidates: RING = WINDOW + 3 rows per request slot.
_RING_HISTORY = tl.constexpr(3)
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
    replay_state,
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
    # The count and the ring state clear together: an unseeded ring reseeds
    # from the page that now holds the dense state.
    tl.store(replay_state + slot * length_stride, 0, mask=valid)
    tl.store(replay_state + slot * length_stride + 1, 0, mask=valid)


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
    snapshot,
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
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    WINDOW: tl.constexpr,
):
    """One layer's checks of a request row's commit.

    ``live`` says the layer carries the row: its group's source and live
    pages lie in the pool, the record slot is resident and the window is
    non-empty. ``valid`` adds every accepted candidate's stamp. ``old`` is
    the count of records the round's verify read before its candidates and
    ``head`` the ring row of its first candidate, both from the prepare's
    snapshot of the row.
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
    # Every layer of the request shares its slot's count. The verify read it
    # whole when it lies in [0, CAPACITY - WINDOW], else none (the prepare
    # folded the records into the live page), so old + take <= CAPACITY; the
    # prepare's snapshot holds that applied count and the ring head.
    old = tl.load(snapshot + row * 2, mask=valid, other=0)
    head = tl.load(snapshot + row * 2 + 1, mask=valid, other=0).to(tl.int64)
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
    return source, target, begin, take, old, head, valid & fresh, live, record_slot


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
    "stamp_slots",
    "stamp_capacity",
    "stopped_count",
]


# A mixed forward's decode rows follow its extend rows in the stop report, so
# request_ends may start at any INT32 element.
@triton.jit(
    do_not_specialize=_BATCHED_RUNTIME_ARGUMENTS,
    do_not_specialize_on_alignment=["request_ends"],
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
    snapshot,
    request_ends,
    group_count,
    read_group_stride,
    read_row_stride,
    write_group_stride,
    write_row_stride,
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
    WINDOW: tl.constexpr,
    CHANNELS: tl.constexpr,
    CONV: tl.constexpr,
    STATE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    ENDED: tl.constexpr,
    INCOMPLETE: tl.constexpr,
):
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source, target, begin, take, old, head, valid, live, record_slot = (
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
            snapshot,
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
            stamp_strides,
            stamp_slots,
            stamp_capacity,
            WINDOW,
        )
    )
    if not valid:
        # A request that ended in this forward keeps records outside its live
        # page; its finish must not publish that page. The stop conditions
        # report ENDED for exactly the rows whose request ended, so the row's
        # own entry stands in for its stop flag.
        if live & (tl.program_id(2) == 0):
            tl.atomic_cas(request_ends + row, ENDED, INCOMPLETE)
        return
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    ring = tl.load(base + _RING).to(tl.pointer_type(CONV))
    conv = tl.load(base + _CONV).to(tl.pointer_type(CONV))
    ring_row_stride = tl.load(base + _RING_ROW_STRIDE).to(tl.int64)
    ring_slot_stride = tl.load(base + _RING_SLOT_STRIDE).to(tl.int64)
    conv_stride = tl.load(base + _CONV_PAGE_STRIDE).to(tl.int64)
    channel_stride = tl.load(base + _CONV_CHANNEL_STRIDE).to(tl.int64)
    tap_stride = tl.load(base + _CONV_TAP_STRIDE).to(tl.int64)
    channel = (tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = channel < CHANNELS
    ptr = conv + source * conv_stride + channel * channel_stride
    h0 = tl.load(ptr, mask=mask, other=0)
    h1 = tl.load(ptr + tap_stride, mask=mask, other=0)
    h2 = tl.load(ptr + 2 * tap_stride, mask=mask, other=0)
    # The verify wrote candidate t to ring row (head + t) mod RING. The row
    # index stays below RING, so its INT32 math cannot wrap; it is widened
    # before the row stride multiplies it.
    slot_rows = ring + record_slot * ring_slot_stride
    head32 = head.to(tl.int32)
    for token in tl.static_range(WINDOW):
        ring_row = head32 + token
        ring_row = tl.where(
            ring_row >= WINDOW + _RING_HISTORY,
            ring_row - (WINDOW + _RING_HISTORY),
            ring_row,
        )
        x = tl.load(
            slot_rows + ring_row.to(tl.int64) * ring_row_stride + channel,
            mask=mask & (token < take),
            other=0,
        )
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
        "batch",
        "layers",
        "group_count",
        "read_group_stride",
        "read_row_stride",
        "write_group_stride",
        "write_row_stride",
        "accepted_stride",
        "query_stride",
        "record_slot_stride",
        "record_slot_count",
        "state_stride",
        "stopped_count",
    ]
)
def retained_batched_publish_kernel(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    starts,
    record_slots,
    request_stopped,
    snapshot,
    replay_state,
    batch,
    layers,
    group_count,
    read_group_stride,
    read_row_stride,
    write_group_stride,
    write_row_stride,
    accepted_stride,
    query_stride,
    record_slot_stride,
    record_slot_count,
    state_stride,
    stopped_count,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write each request row's count and ring state once every layer's commit has read them.

    A row the commit carries -- its source and live pages lie in the pool in
    a state group one of the ``layers`` launched layers lists, its record
    slot is resident and its window is non-empty -- sets its slot's count to
    the records its verify read plus the round's acceptance and its ring
    state to the ring row after the last accepted candidate, plus one; or
    both to 0 when its request ended in this forward (the commit folded its
    records into its live page). The prepare's snapshot holds the applied
    count and the ring head of each row. Other rows (graph padding, masked
    overlap rows, rows paged only in groups no launched layer lists) leave
    every slot alone. Real rows have distinct slots, so no two programs write
    the same slot.
    """
    lanes = tl.arange(0, BLOCK)
    row = tl.program_id(0).to(tl.int64) * BLOCK + lanes
    in_batch = row < batch
    # Every layer shares the State pool's page count.
    pages = tl.load(descriptors + _PAGES).to(tl.int64)
    # Issued ahead of the group loop: their latency overlaps the page loads'.
    begin = tl.load(starts + row * query_stride, mask=in_batch, other=0).to(tl.int64)
    count = (
        tl.load(starts + (row + 1) * query_stride, mask=in_batch, other=0).to(tl.int64)
        - begin
    )
    record_slot = tl.load(
        record_slots + row * record_slot_stride, mask=in_batch, other=-1
    ).to(tl.int64)
    paged = tl.zeros([BLOCK], dtype=tl.int32)
    for group in range(group_count):
        source = tl.load(
            reads + group.to(tl.int64) * read_group_stride + row * read_row_stride,
            mask=in_batch,
            other=0,
        ).to(tl.int64)
        target = tl.load(
            writes + group.to(tl.int64) * write_group_stride + row * write_row_stride,
            mask=in_batch,
            other=0,
        ).to(tl.int64)
        # A group carries rows only when a launched layer lists it; the page
        # loads above do not wait for this check.
        hits = tl.zeros([BLOCK], dtype=tl.int32)
        for first in range(0, layers, BLOCK):
            hits += (
                tl.load(groups + first + lanes, mask=first + lanes < layers, other=-1)
                == group
            ).to(tl.int32)
        paged += (
            (tl.max(hits, 0) > 0)
            & (source > 0)
            & (source < pages)
            & (target > 0)
            & (target < pages)
        ).to(tl.int32)
    carried = (
        in_batch
        & (paged > 0)
        & (record_slot >= 0)
        & (record_slot < record_slot_count)
        & (begin >= 0)
        & (count > 0)
        & (count <= WINDOW)
    )
    take = tl.minimum(
        tl.maximum(tl.load(accepted + row * accepted_stride, mask=carried, other=0), 0),
        tl.minimum(count, WINDOW),
    )
    old = tl.load(snapshot + row * 2, mask=carried, other=0)
    head = tl.load(snapshot + row * 2 + 1, mask=carried, other=0) + take
    head = tl.where(
        head >= WINDOW + _RING_HISTORY, head - (WINDOW + _RING_HISTORY), head
    )
    stopped = tl.load(
        request_stopped + record_slot,
        mask=carried & (record_slot < stopped_count),
        other=0,
    )
    slot_state = replay_state + record_slot * state_stride
    tl.store(slot_state, tl.where(stopped, 0, old + take).to(tl.int32), mask=carried)
    tl.store(slot_state + 1, tl.where(stopped, 0, head + 1).to(tl.int32), mask=carried)


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
    "state_stride",
]


@triton.jit(do_not_specialize=_PREPARE_RUNTIME_ARGUMENTS)
def retained_batched_prepare_kernel(
    descriptors,
    groups,
    sources,
    lives,
    starts,
    record_slots,
    replay_state,
    snapshot,
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
    state_stride,
    correction_strides,
    key_strides,
    decay_strides,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    CHANNELS: tl.constexpr,
    CONV: tl.constexpr,
    STATE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KEYS: tl.constexpr,
    ROWS: tl.constexpr,
    SEED_BLOCK: tl.constexpr,
):
    """Ready one layer's live page and ring of a request row for the round's verify.

    A row takes part in a layer when the layer's group is listed, its source
    and live pages lie inside the pool, its window is non-empty and its
    record slot is resident. The count H every layer of the request shares
    overflows when it lies outside [0, CAPACITY - WINDOW]: the verify reads
    at most CAPACITY - WINDOW records, and the verify and the commit then
    apply none. An overflowing row gets its source state with records [0, H)
    applied in its live page (in place when the two are one page; an H past
    CAPACITY applies none). Otherwise a row whose live page differs from its
    source gets the source state copied, with no record applied: the H
    records stay valid on top of the copy. Program (layer, row, split)
    handles every head h with h % splits == split, in ROWS x KEYS tiles;
    KEYS is DIM rounded up to a power of two.

    A slot whose ring state is 0 holds no pre-convolution rows yet: the row
    seeds ring rows 0..2 of the layer from its source page's three
    convolution taps, and the verify starts its candidates at ring row 3.
    Program (layer 0, row, split 0) writes the row's snapshot: the count the
    verify applies (H, or 0 when H overflows) and the ring row of its first
    candidate, which the verify and the commit read. Counts, ring states and
    records stay untouched; the commit alone writes the counts and ring
    states.
    """
    layer, row = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    pages = tl.load(base + _PAGES).to(tl.int64)
    group = tl.load(groups + layer).to(tl.int64)
    listed = (group >= 0) & (group < group_count)
    source = tl.load(
        sources + group * source_group_stride + row * source_row_stride,
        mask=listed,
        other=0,
    ).to(tl.int64)
    live = tl.load(
        lives + group * live_group_stride + row * live_row_stride,
        mask=listed,
        other=0,
    ).to(tl.int64)
    begin = tl.load(starts + row * query_stride).to(tl.int64)
    tokens = tl.load(starts + (row + 1) * query_stride).to(tl.int64) - begin
    slot = tl.load(record_slots + row * record_slot_stride).to(tl.int64)
    resident = (slot >= 0) & (slot < record_slot_count)
    active = (
        listed
        & (source > 0)
        & (source < pages)
        & (live > 0)
        & (live < pages)
        & (tokens != 0)
        & resident
    )
    # The accepted count and the ring state belong to the request slot, not
    # to either page.
    count = tl.load(replay_state + slot * state_stride, mask=resident, other=0)
    ring_state = tl.load(replay_state + slot * state_stride + 1, mask=resident, other=0)
    overflow = (count < 0) | (count > CAPACITY - WINDOW)
    if (layer == 0) & (tl.program_id(2) == 0):
        tl.store(snapshot + row * 2, tl.where(overflow, 0, count))
        tl.store(
            snapshot + row * 2 + 1,
            tl.where(ring_state > 0, ring_state - 1, _RING_HISTORY),
        )
    if active & (ring_state == 0):
        # Seed ring rows 0..2 with the taps the verify read from the page
        # before the ring held them, oldest first.
        ring = tl.load(base + _RING).to(tl.pointer_type(CONV))
        conv = tl.load(base + _CONV).to(tl.pointer_type(CONV))
        ring_row_stride = tl.load(base + _RING_ROW_STRIDE).to(tl.int64)
        ring_slot_stride = tl.load(base + _RING_SLOT_STRIDE).to(tl.int64)
        conv_stride = tl.load(base + _CONV_PAGE_STRIDE).to(tl.int64)
        channel_stride = tl.load(base + _CONV_CHANNEL_STRIDE).to(tl.int64)
        tap_stride = tl.load(base + _CONV_TAP_STRIDE).to(tl.int64)
        taps = conv + source * conv_stride
        rows = ring + slot * ring_slot_stride
        lanes = tl.arange(0, SEED_BLOCK)
        first = tl.program_id(2) * SEED_BLOCK
        for start in range(first, CHANNELS, tl.num_programs(2) * SEED_BLOCK):
            channel = (start + lanes).to(tl.int64)
            mask = channel < CHANNELS
            for tap in tl.static_range(_RING_HISTORY):
                value = tl.load(
                    taps + channel * channel_stride + tap * tap_stride,
                    mask=mask,
                    other=0,
                )
                tl.store(rows + tap * ring_row_stride + channel, value, mask=mask)
    if not (active & (overflow | (source != live))):
        return
    # The bound keeps the fold loop's trip count known to the compiler.
    count = tl.minimum(
        tl.where(overflow & (count > 0) & (count <= CAPACITY), count, 0), CAPACITY
    )
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
