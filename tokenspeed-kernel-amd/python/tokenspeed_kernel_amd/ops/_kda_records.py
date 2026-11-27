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
    PAGED: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    CAPACITY: gl.constexpr,
    V_MAJOR: gl.constexpr,
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
    correction_strides,
    key_strides,
    decay_strides,
    STATE: gl.constexpr,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
):
    """Fold a slot's first ``count`` records into its live page's dense state.

    The heads are split across axis 2 of the launch grid; each program
    applies the records to its heads' ``[value, key]`` planes in order.
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
    for head in range(gl.program_id(2), HEADS, gl.num_programs(2)):
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
@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
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
    WINDOW: gl.constexpr,
    CHANNELS: gl.constexpr,
    CONV: gl.constexpr,
    STATE: gl.constexpr,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    BLOCK: gl.constexpr,
    ENDED: gl.constexpr,
    INCOMPLETE: gl.constexpr,
    WAVE: gl.constexpr,
):
    layer, row = gl.program_id(0).to(gl.int64), gl.program_id(1).to(gl.int64)
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
            WAVE,
        )
    )
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    ring = gl.load(base + _RING).to(gl.pointer_type(CONV))
    conv = gl.load(base + _CONV).to(gl.pointer_type(CONV))
    ring_row_stride = gl.load(base + _RING_ROW_STRIDE).to(gl.int64)
    ring_slot_stride = gl.load(base + _RING_SLOT_STRIDE).to(gl.int64)
    conv_stride = gl.load(base + _CONV_PAGE_STRIDE).to(gl.int64)
    channel_stride = gl.load(base + _CONV_CHANNEL_STRIDE).to(gl.int64)
    tap_stride = gl.load(base + _CONV_TAP_STRIDE).to(gl.int64)
    channel = (
        gl.program_id(2) * BLOCK
        + gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0]))
    ).to(gl.int64)
    mask = (channel < CHANNELS) & valid
    ptr = conv + source * conv_stride + channel * channel_stride
    h0 = gl.load(ptr, mask=mask, other=0)
    h1 = gl.load(ptr + tap_stride, mask=mask, other=0)
    h2 = gl.load(ptr + 2 * tap_stride, mask=mask, other=0)
    # The verify wrote candidate t to ring row (head + t) mod RING.
    rows = ring + record_slot * ring_slot_stride + channel
    for token in gl.static_range(WINDOW):
        ring_row = head + token
        ring_row = gl.where(
            ring_row >= WINDOW + _RING_HISTORY,
            ring_row - (WINDOW + _RING_HISTORY),
            ring_row,
        )
        x = gl.load(
            rows + ring_row * ring_row_stride,
            mask=mask & (token < take),
            other=0,
        )
        h0 = gl.where(token < take, h1, h0)
        h1 = gl.where(token < take, h2, h1)
        h2 = gl.where(token < take, x, h2)
    # Reject only after the window loads, which ``valid`` masks off for a
    # rejected row: a store ahead of them, even on the rejecting branch, makes
    # LLVM AMDGPU fetch the uniform descriptor fields with vector instead of
    # scalar loads, an extra memory round trip in every program.
    if not valid:
        # A request that ended in this forward keeps records outside its live
        # page; its finish must not publish that page. The stop conditions
        # report ENDED for exactly the rows whose request ended, so the row's
        # own entry stands in for its stop flag, with no load on this path.
        if live & (gl.program_id(2) == 0):
            gl.atomic_cas(request_ends + row, ENDED, INCOMPLETE)
        return
    ptr = conv + target * conv_stride + channel * channel_stride
    gl.store(ptr, h0, mask=mask)
    gl.store(ptr + tap_stride, h1, mask=mask)
    gl.store(ptr + 2 * tap_stride, h2, mask=mask)
    # The request ended in this forward: its live page becomes a dense
    # checkpoint others may read. The publish launch clears the count after
    # every program of this one has folded its heads.
    if gl.load(
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
            (old + take).to(gl.int32),
            correction_strides,
            key_strides,
            decay_strides,
            STATE,
            HEADS,
            DIM,
            BLOCK,
            WAVE,
        )


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
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
    ],
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
    WINDOW: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
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
    layout: gl.constexpr = gl.BlockedLayout([1], [WAVE], [4], [0])
    lanes = gl.arange(0, BLOCK, layout=layout)
    row = gl.program_id(0).to(gl.int64) * BLOCK + lanes
    in_batch = row < batch
    # Every layer shares the State pool's page count.
    pages = gl.load(descriptors + _PAGES).to(gl.int64)
    # Issued ahead of the group loop: their latency overlaps the page loads'.
    begin = gl.load(starts + row * query_stride, mask=in_batch, other=0).to(gl.int64)
    count = (
        gl.load(starts + (row + 1) * query_stride, mask=in_batch, other=0).to(gl.int64)
        - begin
    )
    record_slot = gl.load(
        record_slots + row * record_slot_stride, mask=in_batch, other=-1
    ).to(gl.int64)
    paged = gl.zeros([BLOCK], gl.int32, layout)
    for group in range(group_count):
        source = gl.load(
            reads + group.to(gl.int64) * read_group_stride + row * read_row_stride,
            mask=in_batch,
            other=0,
        ).to(gl.int64)
        target = gl.load(
            writes + group.to(gl.int64) * write_group_stride + row * write_row_stride,
            mask=in_batch,
            other=0,
        ).to(gl.int64)
        # A group carries rows only when a launched layer lists it; the page
        # loads above do not wait for this check.
        hits = gl.zeros([BLOCK], gl.int32, layout)
        for first in range(0, layers, BLOCK):
            hits += (
                gl.load(groups + first + lanes, mask=first + lanes < layers, other=-1)
                == group
            ).to(gl.int32)
        paged += (
            (gl.max(hits, 0) > 0)
            & (source > 0)
            & (source < pages)
            & (target > 0)
            & (target < pages)
        ).to(gl.int32)
    carried = (
        in_batch
        & (paged > 0)
        & (record_slot >= 0)
        & (record_slot < record_slot_count)
        & (begin >= 0)
        & (count > 0)
        & (count <= WINDOW)
    )
    take = gl.minimum(
        gl.maximum(gl.load(accepted + row * accepted_stride, mask=carried, other=0), 0),
        gl.minimum(count, WINDOW),
    )
    old = gl.load(snapshot + row * 2, mask=carried, other=0)
    head = gl.load(snapshot + row * 2 + 1, mask=carried, other=0) + take
    head = gl.where(
        head >= WINDOW + _RING_HISTORY, head - (WINDOW + _RING_HISTORY), head
    )
    stopped = gl.load(
        request_stopped + record_slot,
        mask=carried & (record_slot < stopped_count),
        other=0,
    )
    slot_state = replay_state + record_slot * state_stride
    gl.store(slot_state, gl.where(stopped, 0, old + take).to(gl.int32), mask=carried)
    gl.store(slot_state + 1, gl.where(stopped, 0, head + 1).to(gl.int32), mask=carried)
