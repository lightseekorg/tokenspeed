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

"""Gluon helpers for request-owned KDA records and their accepted lengths."""

from tokenspeed_kernel_amd._triton import gl, gluon, triton

# Row width of the batched retained commit descriptors: the QKV, convolution
# and length fields, then the v-major recurrent state pointer and strides.
_DESCRIPTOR_COLUMNS = gl.constexpr(14)


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


def validate_kda_retained_verify(
    mixed,
    original,
    weights,
    conv,
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
    if (
        weights.shape != (3 * heads * dim, 4)
        or conv.shape != (state.shape[0], 3 * heads * dim, 3)
        or conv.stride()[1:] != (3, 1)
        or state.stride()[1:] != (dim * dim, dim, 1)
    ):
        raise ValueError("retained KDA requires four-tap convolution and V-major state")
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
        conv,
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
    if original is not None:
        if (
            original.shape != mixed.shape
            or original.dtype != mixed.dtype
            or original.stride(-1) != 1
            or original.stride(0) < original.shape[-1]
        ):
            raise ValueError("Original QKV destination must match the input rows")
        destination = original.untyped_storage()
        begin, end = (
            destination.data_ptr(),
            destination.data_ptr() + destination.nbytes(),
        )
        for producer in (mixed, beta, *gate_inputs):
            storage = producer.untyped_storage()
            if (
                begin < storage.data_ptr() + storage.nbytes()
                and storage.data_ptr() < end
            ):
                raise ValueError("Original QKV destination must not overlap producers")
        tensors += (original,)
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
    length,
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
    gl.store(
        length + slot * length_stride,
        0,
        mask=valid,
    )


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
    CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
):
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
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    pages = gl.load(base + 3).to(gl.int64)
    length = gl.load(base + 2).to(gl.pointer_type(gl.int32))
    length_stride = gl.load(base + 6).to(gl.int64)
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
    # The request's accepted count lives in its slot, not in the cache page.
    length = length + record_slot * length_stride
    old = gl.load(length, mask=valid, other=0)
    valid = valid & (old >= 0) & (old + take <= CAPACITY)
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
    return source, target, begin, take, old, valid & fresh, live, length, record_slot


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
    state = gl.load(base + 9).to(gl.pointer_type(STATE))
    page_stride = gl.load(base + 10).to(gl.int64)
    head_stride = gl.load(base + 11).to(gl.int64)
    value_stride = gl.load(base + 12).to(gl.int64)
    key_stride = gl.load(base + 13).to(gl.int64)
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


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=_BATCHED_RUNTIME_ARGUMENTS,
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
    CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    CHANNELS: gl.constexpr,
    ACTIVATION: gl.constexpr,
    CONV: gl.constexpr,
    STATE: gl.constexpr,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    BLOCK: gl.constexpr,
    WAVE: gl.constexpr,
):
    layer, row = gl.program_id(0).to(gl.int64), gl.program_id(1).to(gl.int64)
    source, target, begin, take, old, valid, _, length, record_slot = (
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
            CAPACITY,
            WINDOW,
            WAVE,
        )
    )
    if not valid:
        return
    base = descriptors + layer * _DESCRIPTOR_COLUMNS
    original = gl.load(base).to(gl.pointer_type(ACTIVATION))
    conv = gl.load(base + 1).to(gl.pointer_type(CONV))
    qkv_stride = gl.load(base + 4).to(gl.int64)
    conv_stride = gl.load(base + 5).to(gl.int64)
    channel_stride = gl.load(base + 7).to(gl.int64)
    tap_stride = gl.load(base + 8).to(gl.int64)
    channel = (
        gl.program_id(2) * BLOCK
        + gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [WAVE], [4], [0]))
    ).to(gl.int64)
    mask = channel < CHANNELS
    ptr = conv + source * conv_stride + channel * channel_stride
    h0 = gl.load(ptr, mask=mask, other=0)
    h1 = gl.load(ptr + tap_stride, mask=mask, other=0)
    h2 = gl.load(ptr + 2 * tap_stride, mask=mask, other=0)
    for token in gl.static_range(WINDOW):
        x = gl.load(
            original + (begin + token) * qkv_stride + channel,
            mask=mask & (token < take),
            other=0,
        ).to(CONV)
        h0 = gl.where(token < take, h1, h0)
        h1 = gl.where(token < take, h2, h1)
        h2 = gl.where(token < take, x, h2)
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
            # Validated against the history capacity above.
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
    do_not_specialize=_BATCHED_RUNTIME_ARGUMENTS,
)
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
    request_ends,
    CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
    INCOMPLETE: gl.constexpr,
):
    layer, row = gl.program_id(0).to(gl.int64), gl.program_id(1).to(gl.int64)
    source, target, begin, take, old, valid, live, length, record_slot = (
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
            CAPACITY,
            WINDOW,
            WAVE,
        )
    )
    stopped = gl.load(
        request_stopped + record_slot,
        mask=live & (record_slot < stopped_count),
        other=0,
    )
    if valid:
        # An ended request's records are folded into its live page.
        gl.store(length, gl.where(stopped, 0, old + take))
    # A request that ended in this forward but whose commit this layer
    # rejected keeps records outside its live page; its finish must not
    # publish that page.
    gl.store(request_ends + row, INCOMPLETE, mask=live & (stopped != 0) & (valid == 0))
