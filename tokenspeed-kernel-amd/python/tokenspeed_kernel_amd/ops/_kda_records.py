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

"""Gluon helpers for request-owned KDA records and their accepted counts."""

from tokenspeed_kernel_amd._triton import gl, gluon, triton

# Row width of the batched retained commit descriptors: the pre-convolution
# ring and convolution fields, the v-major recurrent state pointer and
# strides, then the ring's record-slot stride.
_DESCRIPTOR_COLUMNS = gl.constexpr(13)
_RING = gl.constexpr(0)
_CONV = gl.constexpr(1)
_PAGES = gl.constexpr(2)
_RING_ROW_STRIDE = gl.constexpr(3)
_CONV_PAGE_STRIDE = gl.constexpr(4)
_CONV_CHANNEL_STRIDE = gl.constexpr(5)
_CONV_TAP_STRIDE = gl.constexpr(6)
_STATE = gl.constexpr(7)
_STATE_PAGE_STRIDE = gl.constexpr(8)
_STATE_HEAD_STRIDE = gl.constexpr(9)
_STATE_VALUE_STRIDE = gl.constexpr(10)
_STATE_KEY_STRIDE = gl.constexpr(11)
_RING_SLOT_STRIDE = gl.constexpr(12)
# The pre-convolution ring keeps the convolution's 3 past inputs ahead of a
# verify window's candidates: WINDOW + 3 rows per request slot.
_RING_HISTORY = gl.constexpr(3)


def kda_records_launch_metadata(grid, kernel, args):
    return {"name": kernel.name}


@gluon.jit
def kda_retained_panel(
    running,
    correction,
    key,
    decay,
    slot,
    head,
    length,
    u_slot,
    u_head,
    u_pos,
    u_channel,
    k_slot,
    k_head,
    k_pos,
    k_channel,
    d_slot,
    d_head,
    d_pos,
    d_channel,
    values,
    keys,
):
    """Reconstruct one owned [value, key] panel from accepted FP32 records."""
    slot = slot.to(gl.int64)
    head = head.to(gl.int64)
    values = values.to(gl.int64)
    keys = keys.to(gl.int64)
    for position in range(length):
        history_position = position.to(gl.int64)
        u = gl.load(
            correction
            + slot * u_slot
            + head * u_head
            + history_position * u_pos
            + values * u_channel
        )
        k = gl.load(
            key
            + slot * k_slot
            + head * k_head
            + history_position * k_pos
            + keys * k_channel
        )
        d = gl.load(
            decay
            + slot * d_slot
            + head * d_head
            + history_position * d_pos
            + keys * d_channel
        )
        running = running * d[None, :] + u[:, None] * k[None, :]
    return running


def _elements_are_disjoint(tensor) -> bool:
    """Whether distinct indices of ``tensor`` address distinct elements.

    Dimensions of extent above one, taken by increasing stride, must each
    step past every element the smaller ones reach: dense tensors and views
    selecting from a larger one (one layer of a ring) pass, expanded and
    overlapping views fail.
    """
    reach = 0
    for size, stride in sorted(
        (
            (size, stride)
            for size, stride in zip(tensor.shape, tensor.stride())
            if size > 1
        ),
        key=lambda dim: dim[1],
    ):
        if stride <= reach:
            return False
        reach += (size - 1) * stride
    return True


def validate_kda_retained_verify(
    mixed,
    ring,
    ring_heads,
    weights,
    raw_g,
    f_a,
    f_b,
    beta,
    a_log,
    bias,
    state,
    reads,
    starts,
    correction,
    key,
    decay,
    length,
    live,
    record_slots,
    metadata,
    stamps,
    heads,
    dim,
    window,
):
    """Validate producer/request correspondence for pointer-only AMD launchers."""
    import torch

    batch, tokens = reads.numel(), mixed.shape[0]
    if (
        dim != 128
        or window <= 0
        or window > correction.shape[2]
        or tokens > batch * window
    ):
        raise ValueError("retained KDA requires D=128 and a bounded candidate window")
    if mixed.shape != (tokens, 3 * heads * dim) or beta.shape != (tokens, heads):
        raise ValueError("retained KDA producer geometry does not match heads")
    if raw_g is None:
        if (
            f_a is None
            or f_b is None
            or f_a.shape != (tokens, dim)
            or f_b.shape != (heads * dim, dim)
            or not f_b.is_contiguous()
        ):
            raise ValueError("retained KDA requires both gate projection inputs")
        gate_inputs = (f_a, f_b)
    else:
        if f_a is not None or f_b is not None or raw_g.shape != (tokens, heads * dim):
            raise ValueError("retained KDA raw gate excludes projection inputs")
        gate_inputs = (raw_g,)
    if mixed.dtype not in (torch.bfloat16, torch.float16) or any(
        x.dtype != mixed.dtype or x.stride(-1) != 1 for x in (*gate_inputs, beta, mixed)
    ):
        raise ValueError(
            "retained KDA producers require matching activation dtype and contiguous channels"
        )
    if weights.shape != (3 * heads * dim, 4) or state.stride()[1:] != (
        dim * dim,
        dim,
        1,
    ):
        raise ValueError("retained KDA requires four-tap convolution and V-major state")
    if (
        ring.ndim != 3
        or ring.shape != (correction.shape[0], window + 3, 3 * heads * dim)
        or ring.stride(2) != 1
        or ring.dtype not in (mixed.dtype, torch.bfloat16)
        or ring_heads.shape != (batch,)
        or ring_heads.dtype != torch.int32
    ):
        raise ValueError(
            "retained KDA requires a [record slots, window + 3, channels] "
            "pre-convolution ring and one INT32 ring head per request"
        )
    if not _elements_are_disjoint(ring):
        # A verify reads some rows of a slot while it writes others, and
        # requests of different slots write concurrently.
        raise ValueError("pre-convolution ring rows and slots must not share memory")
    if (
        a_log.shape != (heads,)
        or bias.numel() != heads * dim
        or any(x.dtype != torch.float32 or not x.is_contiguous() for x in (a_log, bias))
    ):
        raise ValueError("retained KDA gate parameters must be contiguous FP32")
    if (
        reads.shape != (batch,)
        or live.shape != (batch,)
        or record_slots.shape != (batch,)
        or starts.shape != (batch + 1,)
    ):
        raise ValueError("retained KDA request indices must match query boundaries")
    if (metadata is None) != (stamps is None):
        raise ValueError("retained KDA metadata and stamps must be supplied together")
    tensors = (
        weights,
        ring,
        ring_heads,
        *gate_inputs,
        beta,
        a_log,
        bias,
        state,
        reads,
        starts,
        correction,
        key,
        decay,
        length,
        live,
        record_slots,
    )
    if metadata is not None:
        if (
            metadata.shape != (batch, 3)
            or stamps.ndim != 3
            or (stamps.shape[1] < window or stamps.shape[2] != 2)
            or any(
                x.dtype != torch.int64 or not x.is_contiguous()
                for x in (metadata, stamps)
            )
        ):
            raise ValueError(
                "retained KDA stamps must cover request slots and history capacity"
            )
        tensors += (metadata, stamps)
    destination = ring.untyped_storage()
    begin, end = destination.data_ptr(), destination.data_ptr() + destination.nbytes()
    for producer in (mixed, beta, *gate_inputs):
        storage = producer.untyped_storage()
        if begin < storage.data_ptr() + storage.nbytes() and storage.data_ptr() < end:
            raise ValueError("the pre-convolution ring must not overlap producers")
    if any(x.device != mixed.device for x in tensors):
        raise ValueError("retained KDA inputs must be on the same device")


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
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
    ],
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
    PAGED: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    CAPACITY: gl.constexpr,
    V_MAJOR: gl.constexpr,
    CHANNELS: gl.constexpr,
    WINDOW: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
):
    row, head = gl.program_id(0).to(gl.int64), gl.program_id(1).to(gl.int64)
    source = gl.load(sources + row * source_stride).to(gl.int64)
    slot = gl.load(record_slots + row * slot_stride).to(gl.int64)
    valid = (source > 0) & (slot >= 0) & (slot < slots)
    if PAGED:
        valid = valid & (source < pages)
    count = gl.load(length + slot * length_stride, mask=valid, other=0)
    # A seeded ring holds the slot's last three accepted inputs, which the
    # destination page's convolution state may lack: the row's programs copy
    # them, BLOCK channels at a time. The destination is the page the
    # recurrent state folds into: the listed page of a paged fold, the
    # request's live page (which holds the gathered taps) otherwise.
    if PAGED:
        live = source
    else:
        live = gl.load(lives + row * live_stride, mask=valid, other=0).to(gl.int64)
    seeded = gl.load(ring_state + slot * ring_state_stride, mask=valid, other=0)
    if valid & (seeded > 0) & (live > 0) & (live < conv_pages):
        blocks = gl.num_programs(2)
        retained_materialize_conv(
            ring + slot * ring_slot_stride,
            ring_row_stride,
            conv + live * conv_strides[0],
            conv_strides[1],
            conv_strides[2],
            seeded - 1,
            (head * blocks + gl.program_id(2)).to(gl.int32) * BLOCK,
            (gl.num_programs(1) * blocks).to(gl.int32) * BLOCK,
            CHANNELS,
            WINDOW,
            BLOCK,
            WAVE,
        )
    if count <= 0 or count > CAPACITY:
        return
    state_row = source if PAGED else row
    offsets = gl.program_id(2) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0])
    )
    v, k = (offsets // K).to(gl.int64), (offsets % K).to(gl.int64)
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
    value = gl.load(ptr, mask=mask, other=0).to(gl.float32)
    for token in range(count):
        uv = gl.load(
            u
            + slot * u_strides[0]
            + head * u_strides[1]
            + token * u_strides[2]
            + v * u_strides[3],
            mask=mask,
            other=0,
        )
        kv = gl.load(
            key
            + slot * k_strides[0]
            + head * k_strides[1]
            + token * k_strides[2]
            + k * k_strides[3],
            mask=mask,
            other=0,
        )
        dv = gl.load(
            decay
            + slot * d_strides[0]
            + head * d_strides[1]
            + token * d_strides[2]
            + k * d_strides[3],
            mask=mask,
            other=0,
        )
        value = value * dv + uv * kv
    gl.store(ptr, value.to(state.dtype.element_ty), mask=mask)


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "rows",
        "length_stride",
        "index_stride",
        "slot_stride",
        "slot_count",
        "pages",
    ],
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
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
):
    row = gl.program_id(0).to(gl.int64) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0])
    )
    page = gl.load(indices + row * index_stride, mask=row < rows, other=0).to(gl.int64)
    slot = gl.load(record_slots + row * slot_stride, mask=row < rows, other=-1).to(
        gl.int64
    )
    # Padding rows carry a page at or below 0, or one past the pool, as the
    # materialize kernel reads them; their slots may belong to others.
    valid = (
        (row < rows) & (page > 0) & (page < pages) & (slot >= 0) & (slot < slot_count)
    )
    # The count and the ring state clear together: an unseeded ring reseeds
    # from the page that now holds the dense state.
    gl.store(replay_state + slot * length_stride, 0, mask=valid)
    gl.store(replay_state + slot * length_stride + 1, 0, mask=valid)


@gluon.jit
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
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
):
    """One layer's checks of a request row's commit.

    ``live`` says the layer carries the row: its group's source and live
    pages lie in the pool, the record slot is resident and the window is
    non-empty. ``valid`` adds every accepted candidate's stamp. ``old`` is
    the count of records the round's verify read before its candidates and
    ``head`` the ring row of its first candidate, both from the prepare's
    snapshot of the row.
    """
    group = gl.load(groups + layer).to(gl.int64)
    valid = (group >= 0) & (group < group_count)
    source = gl.load(
        reads + group * read_group_stride + row * read_row_stride, mask=valid, other=0
    ).to(gl.int64)
    target = gl.load(
        writes + group * write_group_stride + row * write_row_stride,
        mask=valid,
        other=0,
    ).to(gl.int64)
    pages = gl.load(descriptors + layer * _DESCRIPTOR_COLUMNS + _PAGES).to(gl.int64)
    begin = gl.load(starts + row * query_stride).to(gl.int64)
    count = gl.load(starts + (row + 1) * query_stride).to(gl.int64) - begin
    record_slot = gl.load(record_slots + row * record_slot_stride).to(gl.int64)
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
    take = gl.minimum(
        gl.maximum(gl.load(accepted + row * accepted_stride), 0),
        gl.minimum(count, WINDOW),
    )
    # Every layer of the request shares its slot's count. The verify read it
    # whole when it lies in [0, CAPACITY - WINDOW], else none (the prepare
    # folded the records into the live page), so old + take <= CAPACITY; the
    # prepare's snapshot holds that applied count and the ring head.
    old = gl.load(snapshot + row * 2, mask=valid, other=0)
    head = gl.load(snapshot + row * 2 + 1, mask=valid, other=0).to(gl.int64)
    slot = gl.load(metadata + row * 3).to(gl.int64)
    generation = gl.load(metadata + row * 3 + 1)
    absolute = gl.load(metadata + row * 3 + 2)
    valid = (
        valid & (slot >= 0) & (slot < stamp_slots) & (generation >= 0) & (absolute >= 0)
    )
    offsets = gl.arange(
        0,
        triton.next_power_of_2(WINDOW),
        layout=gl.BlockedLayout([1], [WAVE], [4], [0]),
    )
    positions = absolute + offsets
    ptr = (
        stamps
        + layer * stamp_strides[0]
        + slot * stamp_strides[1]
        + (positions % stamp_capacity) * stamp_strides[2]
    )
    mask = valid & (offsets < take)
    position = gl.load(ptr, mask=mask, other=-1)
    stamp_generation = gl.load(ptr + stamp_strides[3], mask=mask, other=-1)
    fresh = (
        gl.sum(
            (mask & ((position != positions) | (stamp_generation != generation))).to(
                gl.int32
            ),
            0,
        )
        == 0
    )
    return source, target, begin, take, old, head, valid & fresh, live, record_slot


@gluon.jit
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
    STATE: gl.constexpr,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
):
    """Fold a slot's first ``count`` records into its live page's dense state.

    The program applies the records, in order, to the ``[value, key]`` plane
    of every head ``first_head + i * SPLITS``.
    """
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    state = gl.load(base + _STATE).to(gl.pointer_type(STATE))
    page_stride = gl.load(base + _STATE_PAGE_STRIDE).to(gl.int64)
    head_stride = gl.load(base + _STATE_HEAD_STRIDE).to(gl.int64)
    value_stride = gl.load(base + _STATE_VALUE_STRIDE).to(gl.int64)
    key_stride = gl.load(base + _STATE_KEY_STRIDE).to(gl.int64)
    u = correction + record_slot * correction_strides[0] + layer * correction_strides[1]
    k = retained_key + record_slot * key_strides[0] + layer * key_strides[1]
    d = retained_decay + record_slot * decay_strides[0] + layer * decay_strides[1]
    page = state + target * page_stride
    lanes = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0]))
    for head in range(first_head, HEADS, SPLITS):
        for start in range(0, DIM * DIM, BLOCK):
            offsets = start + lanes
            v, c = (offsets // DIM).to(gl.int64), (offsets % DIM).to(gl.int64)
            mask = v < DIM
            ptr = page + head * head_stride + v * value_stride + c * key_stride
            value = gl.load(ptr, mask=mask, other=0).to(gl.float32)
            for token in range(count):
                uv = gl.load(
                    u
                    + head * correction_strides[2]
                    + token * correction_strides[3]
                    + v * correction_strides[4],
                    mask=mask,
                    other=0,
                )
                kv = gl.load(
                    k
                    + head * key_strides[2]
                    + token * key_strides[3]
                    + c * key_strides[4],
                    mask=mask,
                    other=0,
                )
                dv = gl.load(
                    d
                    + head * decay_strides[2]
                    + token * decay_strides[3]
                    + c * decay_strides[4],
                    mask=mask,
                    other=0,
                )
                value = value * dv + uv * kv
            gl.store(ptr, value.to(STATE), mask=mask)


@gluon.jit
def retained_ring_row(row, WINDOW: gl.constexpr):
    """Wrap a ring row in [-RING, 2 * RING) into [0, RING), RING = WINDOW + 3.

    Rows stay below 2 * RING, so the INT32 math cannot wrap; callers widen
    the result before a stride multiplies it.
    """
    row = gl.where(row >= WINDOW + _RING_HISTORY, row - (WINDOW + _RING_HISTORY), row)
    return gl.where(row < 0, row + (WINDOW + _RING_HISTORY), row)


@gluon.jit
def retained_materialize_conv(
    ring_rows,
    ring_row_stride,
    conv_page,
    channel_stride,
    tap_stride,
    last,
    first_channel,
    channel_step,
    CHANNELS: gl.constexpr,
    WINDOW: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
):
    """Write a slot's last three ring inputs, oldest first, as a page's taps.

    ``ring_rows`` points at the slot's ring row 0 of the layer and ``last``
    is the ring row of the slot's next input, so rows ``last - 3 .. last - 1``
    (mod RING) hold the convolution's window. The program copies channels
    ``first_channel + i * channel_step`` blocks of BLOCK.
    """
    lanes = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0]))
    row0 = retained_ring_row(last - 3, WINDOW).to(gl.int64) * ring_row_stride
    row1 = retained_ring_row(last - 2, WINDOW).to(gl.int64) * ring_row_stride
    row2 = retained_ring_row(last - 1, WINDOW).to(gl.int64) * ring_row_stride
    for start in range(first_channel, CHANNELS, channel_step):
        channel = (start + lanes).to(gl.int64)
        mask = channel < CHANNELS
        x0 = gl.load(ring_rows + row0 + channel, mask=mask, other=0)
        x1 = gl.load(ring_rows + row1 + channel, mask=mask, other=0)
        x2 = gl.load(ring_rows + row2 + channel, mask=mask, other=0)
        ptr = conv_page + channel * channel_stride
        gl.store(ptr, x0, mask=mask)
        gl.store(ptr + tap_stride, x1, mask=mask)
        gl.store(ptr + 2 * tap_stride, x2, mask=mask)


@gluon.jit
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
    WINDOW: gl.constexpr,
    GROUPS: gl.constexpr,
    LAYERS: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
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
    layout: gl.constexpr = gl.BlockedLayout([1], [WAVE], [4], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    row = block.to(gl.int64) * BLOCK + lanes
    in_batch = row < batch
    # Every layer shares the State pool's page count.
    pages = gl.load(descriptors + _PAGES).to(gl.int64)
    begin = gl.load(starts + row * query_stride, mask=in_batch, other=0).to(gl.int64)
    count = (
        gl.load(starts + (row + 1) * query_stride, mask=in_batch, other=0).to(gl.int64)
        - begin
    )
    record_slot = gl.load(
        record_slots + row * record_slot_stride, mask=in_batch, other=-1
    ).to(gl.int64)
    take = gl.load(accepted + row * accepted_stride, mask=in_batch, other=0)
    old = gl.load(snapshot + row * 2, mask=in_batch, other=0)
    head = gl.load(snapshot + row * 2 + 1, mask=in_batch, other=0)
    layer = gl.arange(0, LAYERS, layout=layout)
    layer_groups = gl.load(groups + layer, mask=layer < layers, other=-1)
    paged = gl.zeros([BLOCK], gl.int32, layout)
    for group in gl.static_range(GROUPS):
        present = in_batch & (group < group_count)
        source = gl.load(
            reads + read_group_stride.to(gl.int64) * group + row * read_row_stride,
            mask=present,
            other=0,
        ).to(gl.int64)
        target = gl.load(
            writes + write_group_stride.to(gl.int64) * group + row * write_row_stride,
            mask=present,
            other=0,
        ).to(gl.int64)
        # A group carries rows only when a launched layer lists it.
        listed = gl.max((layer_groups == group).to(gl.int32), 0) > 0
        paged += (
            listed & (source > 0) & (source < pages) & (target > 0) & (target < pages)
        ).to(gl.int32)
    # The stop flags are a strided view; the INT64 slot keeps the offset wide.
    stopped = gl.load(
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
    take = gl.minimum(gl.maximum(take, 0), gl.minimum(count, WINDOW))
    head = head + take
    head = gl.where(
        head >= WINDOW + _RING_HISTORY, head - (WINDOW + _RING_HISTORY), head
    )
    slot_state = replay_state + record_slot * state_stride
    gl.store(slot_state, gl.where(stopped, 0, old + take).to(gl.int32), mask=carried)
    gl.store(slot_state + 1, gl.where(stopped, 0, head + 1).to(gl.int32), mask=carried)


@gluon.jit
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
    WINDOW: gl.constexpr,
    CHANNELS: gl.constexpr,
    CONV: gl.constexpr,
    STATE: gl.constexpr,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    ENDED: gl.constexpr,
    INCOMPLETE: gl.constexpr,
    WAVE: gl.constexpr,
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
    lanes = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0]))
    for first in range(0, batch, BLOCK):
        rows = first + lanes
        in_batch = rows < batch
        slots = gl.load(
            record_slots + rows.to(gl.int64) * record_slot_stride,
            mask=in_batch,
            other=-1,
        ).to(gl.int64)
        ended = (
            gl.load(
                request_stopped + slots * stopped_stride,
                mask=in_batch & (slots >= 0) & (slots < stopped_count),
                other=0,
            )
            != 0
        )
        row = gl.min(gl.where(ended, rows, batch), 0)
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
                    row.to(gl.int64),
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
                    WAVE,
                )
            )
            if valid:
                base = descriptors + layer * _DESCRIPTOR_COLUMNS
                ring = gl.load(base + _RING).to(gl.pointer_type(CONV))
                conv = gl.load(base + _CONV).to(gl.pointer_type(CONV))
                ring_row_stride = gl.load(base + _RING_ROW_STRIDE).to(gl.int64)
                ring_slot_stride = gl.load(base + _RING_SLOT_STRIDE).to(gl.int64)
                conv_stride = gl.load(base + _CONV_PAGE_STRIDE).to(gl.int64)
                channel_stride = gl.load(base + _CONV_CHANNEL_STRIDE).to(gl.int64)
                tap_stride = gl.load(base + _CONV_TAP_STRIDE).to(gl.int64)
                # The verify wrote candidate t to ring row (head + t) mod RING:
                # the window after the accepted ones ends at row head + take.
                retained_materialize_conv(
                    ring + record_slot * ring_slot_stride,
                    ring_row_stride,
                    conv + target * conv_stride,
                    channel_stride,
                    tap_stride,
                    head.to(gl.int32) + take.to(gl.int32),
                    split * BLOCK,
                    SPLITS * BLOCK,
                    CHANNELS,
                    WINDOW,
                    BLOCK,
                    WAVE,
                )
                retained_fold_ended_request(
                    descriptors,
                    correction,
                    retained_key,
                    retained_decay,
                    layer,
                    target,
                    record_slot,
                    (old + take).to(gl.int32),
                    split,
                    correction_strides,
                    key_strides,
                    decay_strides,
                    STATE,
                    HEADS,
                    DIM,
                    SPLITS,
                    BLOCK,
                    WAVE,
                )
            elif live & (split == 0):
                # The request keeps records and inputs outside its live page;
                # its finish must not publish that page.
                gl.atomic_cas(request_ends + row, ENDED, INCOMPLETE)
            row = gl.min(gl.where(ended & (rows > row), rows, batch), 0)


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
@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
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
    WINDOW: gl.constexpr,
    CHANNELS: gl.constexpr,
    CONV: gl.constexpr,
    STATE: gl.constexpr,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUPS: gl.constexpr,
    LAYERS: gl.constexpr,
    BLOCK: gl.constexpr,
    ENDED: gl.constexpr,
    INCOMPLETE: gl.constexpr,
    WAVE: gl.constexpr,
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
    program = gl.program_id(0)
    publishers = (batch + BLOCK - 1) // BLOCK
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
            WAVE,
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
            (program // SPLITS).to(gl.int64),
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
            WAVE,
        )
