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
        "live_stride",
        "conv_pages",
        "ring_slot_stride",
        "ring_row_stride",
        "ring_state_stride",
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
    conv,
    ring,
    ring_state,
    lives,
    live_stride,
    conv_pages,
    conv_strides,
    ring_slot_stride,
    ring_row_stride,
    ring_state_stride,
    PAGED: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CAPACITY: tl.constexpr,
    V_MAJOR: tl.constexpr,
    CHANNELS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    source = tl.load(sources + row * source_stride).to(tl.int64)
    slot = tl.load(record_slots + row * slot_stride).to(tl.int64)
    valid = (source > 0) & (slot >= 0) & (slot < slots)
    if PAGED:
        valid = valid & (source < pages)
    count = tl.load(length + slot * length_stride, mask=valid, other=0)
    # A seeded ring holds the slot's last three accepted inputs, which the
    # destination page's convolution state may lack: the row's programs copy
    # them, BLOCK channels at a time. The destination is the page the
    # recurrent state folds into: the listed page of a paged fold, the
    # request's live page (which holds the gathered taps) otherwise.
    if PAGED:
        live = source
    else:
        live = tl.load(lives + row * live_stride, mask=valid, other=0).to(tl.int64)
    seeded = tl.load(ring_state + slot * ring_state_stride, mask=valid, other=0)
    if valid & (seeded > 0) & (live > 0) & (live < conv_pages):
        blocks = tl.num_programs(2)
        retained_materialize_conv(
            ring + slot * ring_slot_stride,
            ring_row_stride,
            conv + live * conv_strides[0],
            conv_strides[1],
            conv_strides[2],
            seeded - 1,
            (head * blocks + tl.program_id(2)).to(tl.int32) * BLOCK,
            (tl.num_programs(1) * blocks).to(tl.int32) * BLOCK,
            CHANNELS,
            WINDOW,
            BLOCK,
        )
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
    first_head,
    correction_strides,
    key_strides,
    decay_strides,
    STATE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold a slot's first ``count`` records into its live page's dense state.

    The program applies the records, in order, to the ``[value, key]`` plane
    of every head ``first_head + i * SPLITS``.
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
    for head in range(first_head, HEADS, SPLITS):
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


@triton.jit
def retained_ring_row(row, WINDOW: tl.constexpr):
    """Wrap a ring row in [-RING, 2 * RING) into [0, RING), RING = WINDOW + 3.

    Rows stay below 2 * RING, so the INT32 math cannot wrap; callers widen
    the result before a stride multiplies it.
    """
    row = tl.where(row >= WINDOW + _RING_HISTORY, row - (WINDOW + _RING_HISTORY), row)
    return tl.where(row < 0, row + (WINDOW + _RING_HISTORY), row)


@triton.jit
def retained_materialize_conv(
    ring_rows,
    ring_row_stride,
    conv_page,
    channel_stride,
    tap_stride,
    last,
    first_channel,
    channel_step,
    CHANNELS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write a slot's last three ring inputs, oldest first, as a page's taps.

    ``ring_rows`` points at the slot's ring row 0 of the layer and ``last``
    is the ring row of the slot's next input, so rows ``last - 3 .. last - 1``
    (mod RING) hold the convolution's window. The program copies channels
    ``first_channel + i * channel_step`` blocks of BLOCK.
    """
    lanes = tl.arange(0, BLOCK)
    row0 = retained_ring_row(last - 3, WINDOW).to(tl.int64) * ring_row_stride
    row1 = retained_ring_row(last - 2, WINDOW).to(tl.int64) * ring_row_stride
    row2 = retained_ring_row(last - 1, WINDOW).to(tl.int64) * ring_row_stride
    for start in range(first_channel, CHANNELS, channel_step):
        channel = (start + lanes).to(tl.int64)
        mask = channel < CHANNELS
        x0 = tl.load(ring_rows + row0 + channel, mask=mask, other=0)
        x1 = tl.load(ring_rows + row1 + channel, mask=mask, other=0)
        x2 = tl.load(ring_rows + row2 + channel, mask=mask, other=0)
        ptr = conv_page + channel * channel_stride
        tl.store(ptr, x0, mask=mask)
        tl.store(ptr + tap_stride, x1, mask=mask)
        tl.store(ptr + 2 * tap_stride, x2, mask=mask)


@triton.jit
def retained_publish_rows(
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
    block,
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
    stopped_stride,
    WINDOW: tl.constexpr,
    GROUPS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write the count and ring state of each request row of a block of BLOCK rows.

    A row the commit carries -- its source and live pages lie in the pool in
    a state group one of the ``layers`` launched layers lists, its record
    slot is resident and its window is non-empty -- sets its slot's count to
    the records its verify read plus the round's acceptance and its ring
    state to the ring row after the last accepted candidate, plus one; or
    both to 0 when its request ended in this forward (the end programs fold
    its records and write its convolution window into its live page). The
    prepare's snapshot holds the applied count and the ring head of each
    row. Other rows (graph padding, masked overlap rows, rows paged only in
    groups no launched layer lists) leave every slot alone. Real rows have
    distinct slots, so no two programs write the same slot.

    GROUPS and LAYERS bound ``group_count`` and ``layers`` (powers of two):
    every load but the stop flags' issues at once, ahead of any check.
    """
    lanes = tl.arange(0, BLOCK)
    row = block.to(tl.int64) * BLOCK + lanes
    in_batch = row < batch
    # Every layer shares the State pool's page count.
    pages = tl.load(descriptors + _PAGES).to(tl.int64)
    begin = tl.load(starts + row * query_stride, mask=in_batch, other=0).to(tl.int64)
    count = (
        tl.load(starts + (row + 1) * query_stride, mask=in_batch, other=0).to(tl.int64)
        - begin
    )
    record_slot = tl.load(
        record_slots + row * record_slot_stride, mask=in_batch, other=-1
    ).to(tl.int64)
    take = tl.load(accepted + row * accepted_stride, mask=in_batch, other=0)
    old = tl.load(snapshot + row * 2, mask=in_batch, other=0)
    head = tl.load(snapshot + row * 2 + 1, mask=in_batch, other=0)
    layer = tl.arange(0, LAYERS)
    layer_groups = tl.load(groups + layer, mask=layer < layers, other=-1)
    paged = tl.zeros([BLOCK], dtype=tl.int32)
    for group in tl.static_range(GROUPS):
        present = in_batch & (group < group_count)
        source = tl.load(
            reads + read_group_stride.to(tl.int64) * group + row * read_row_stride,
            mask=present,
            other=0,
        ).to(tl.int64)
        target = tl.load(
            writes + write_group_stride.to(tl.int64) * group + row * write_row_stride,
            mask=present,
            other=0,
        ).to(tl.int64)
        # A group carries rows only when a launched layer lists it.
        listed = tl.max((layer_groups == group).to(tl.int32), 0) > 0
        paged += (
            listed & (source > 0) & (source < pages) & (target > 0) & (target < pages)
        ).to(tl.int32)
    # The stop flags are a strided view; the INT64 slot keeps the offset wide.
    stopped = tl.load(
        request_stopped + record_slot * stopped_stride,
        mask=in_batch & (record_slot >= 0) & (record_slot < stopped_count),
        other=0,
    )
    carried = (
        in_batch
        & (paged > 0)
        & (record_slot >= 0)
        & (record_slot < record_slot_count)
        & (begin >= 0)
        & (count > 0)
        & (count <= WINDOW)
    )
    take = tl.minimum(tl.maximum(take, 0), tl.minimum(count, WINDOW))
    head = head + take
    head = tl.where(
        head >= WINDOW + _RING_HISTORY, head - (WINDOW + _RING_HISTORY), head
    )
    slot_state = replay_state + record_slot * state_stride
    tl.store(slot_state, tl.where(stopped, 0, old + take).to(tl.int32), mask=carried)
    tl.store(slot_state + 1, tl.where(stopped, 0, head + 1).to(tl.int32), mask=carried)


@triton.jit
def retained_end_rows(
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
    layer,
    split,
    batch,
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
    stopped_stride,
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
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    ENDED: tl.constexpr,
    INCOMPLETE: tl.constexpr,
):
    """Materialize one layer's share ``split`` of every request that ended in this forward.

    The program loads the stop flags of BLOCK rows at once and visits only
    the rows whose request ended: a row that did not end costs its lane of
    those loads. For an ended row the layer carries and accepts (every
    accepted candidate's stamp is fresh), the program writes its share of
    the ring's last three accepted inputs into the live page's convolution
    state and folds records [0, applied count + acceptance) into its share
    of the live page's heads. A layer that carries the row but rejects its
    commit raises the row's ``request_ends`` entry to INCOMPLETE once.
    """
    lanes = tl.arange(0, BLOCK)
    for first in range(0, batch, BLOCK):
        rows = first + lanes
        in_batch = rows < batch
        slots = tl.load(
            record_slots + rows.to(tl.int64) * record_slot_stride,
            mask=in_batch,
            other=-1,
        ).to(tl.int64)
        ended = (
            tl.load(
                request_stopped + slots * stopped_stride,
                mask=in_batch & (slots >= 0) & (slots < stopped_count),
                other=0,
            )
            != 0
        )
        row = tl.min(tl.where(ended, rows, batch), 0)
        while row < batch:
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
                    row.to(tl.int64),
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
            if valid:
                base = descriptors + layer * _DESCRIPTOR_COLUMNS
                ring = tl.load(base + _RING).to(tl.pointer_type(CONV))
                conv = tl.load(base + _CONV).to(tl.pointer_type(CONV))
                ring_row_stride = tl.load(base + _RING_ROW_STRIDE).to(tl.int64)
                ring_slot_stride = tl.load(base + _RING_SLOT_STRIDE).to(tl.int64)
                conv_stride = tl.load(base + _CONV_PAGE_STRIDE).to(tl.int64)
                channel_stride = tl.load(base + _CONV_CHANNEL_STRIDE).to(tl.int64)
                tap_stride = tl.load(base + _CONV_TAP_STRIDE).to(tl.int64)
                # The verify wrote candidate t to ring row (head + t) mod RING:
                # the window after the accepted ones ends at row head + take.
                retained_materialize_conv(
                    ring + record_slot * ring_slot_stride,
                    ring_row_stride,
                    conv + target * conv_stride,
                    channel_stride,
                    tap_stride,
                    head.to(tl.int32) + take.to(tl.int32),
                    split * BLOCK,
                    SPLITS * BLOCK,
                    CHANNELS,
                    WINDOW,
                    BLOCK,
                )
                retained_fold_ended_request(
                    descriptors,
                    correction,
                    retained_key,
                    retained_decay,
                    layer,
                    target,
                    record_slot,
                    (old + take).to(tl.int32),
                    split,
                    correction_strides,
                    key_strides,
                    decay_strides,
                    STATE,
                    HEADS,
                    DIM,
                    SPLITS,
                    BLOCK,
                )
            elif live & (split == 0):
                # The request keeps records and inputs outside its live page;
                # its finish must not publish that page.
                tl.atomic_cas(request_ends + row, ENDED, INCOMPLETE)
            row = tl.min(tl.where(ended & (rows > row), rows, batch), 0)


# Triton specializes each element of a tuple argument whatever
# do_not_specialize lists, so a stride that follows the batch size -- the row
# stride of a [groups, batch] page table -- is a scalar here. The tuples left
# (stamps and records) hold strides fixed by their storage layout.
_COMMIT_RUNTIME_ARGUMENTS = [
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
    "stamp_slots",
    "stamp_capacity",
    "stopped_count",
    "stopped_stride",
]


# A mixed forward's decode rows follow its extend rows in the stop report, so
# request_ends may start at any INT32 element.
@triton.jit(
    do_not_specialize=_COMMIT_RUNTIME_ARGUMENTS,
    do_not_specialize_on_alignment=["request_ends"],
)
def retained_batched_commit_kernel(
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
    stamp_strides,
    stamp_slots,
    stamp_capacity,
    stopped_count,
    stopped_stride,
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
    SPLITS: tl.constexpr,
    GROUPS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
    ENDED: tl.constexpr,
    INCOMPLETE: tl.constexpr,
):
    """Publish every request's count and ring state and materialize ended requests, in one launch.

    Programs [0, cdiv(batch, BLOCK)) publish BLOCK rows each
    (retained_publish_rows); program cdiv(batch, BLOCK) + layer * SPLITS +
    split materializes share ``split`` of every ended request of ``layer``
    (retained_end_rows). Publish programs write only the replay state; end
    programs read the prepare's snapshot, the records, the stamps and the
    ring, never the replay state, and write live pages and request_ends, so
    no program reads what another writes.
    """
    program = tl.program_id(0)
    publishers = tl.cdiv(batch, BLOCK)
    if program < publishers:
        retained_publish_rows(
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
            program,
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
            stopped_stride,
            WINDOW,
            GROUPS,
            LAYERS,
            BLOCK,
        )
    else:
        program -= publishers
        retained_end_rows(
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
            (program // SPLITS).to(tl.int64),
            program % SPLITS,
            batch,
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
            stopped_stride,
            correction,
            retained_key,
            retained_decay,
            correction_strides,
            key_strides,
            decay_strides,
            WINDOW,
            CHANNELS,
            CONV,
            STATE,
            HEADS,
            DIM,
            SPLITS,
            BLOCK,
            ENDED,
            INCOMPLETE,
        )


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
