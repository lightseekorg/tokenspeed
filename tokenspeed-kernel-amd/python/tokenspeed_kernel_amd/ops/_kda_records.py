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

"""Gluon helpers for current-window KDA correction records."""

from tokenspeed_kernel_amd._triton import gl, gluon, triton


def kda_records_launch_metadata(grid, kernel, args):
    return {"name": kernel.name}


@gluon.jit
def kda_record_interval(
    metadata,
    stamps,
    batch_row,
    layer,
    steps,
    SLOTS,
    CAPACITY: gl.constexpr,
    stamp_layer_stride,
    MAX_STEPS: gl.constexpr,
    WAVE: gl.constexpr,
):
    slot = batch_row.to(gl.int64)
    start = slot * 0
    generation = start
    if metadata is not None:
        slot = gl.load(metadata + batch_row * 3)
        generation = gl.load(metadata + batch_row * 3 + 1)
        start = gl.load(metadata + batch_row * 3 + 2)
    valid = (
        (slot >= 0)
        & (slot < SLOTS)
        & (start >= 0)
        & (generation >= 0)
        & (steps >= 0)
        & (steps <= MAX_STEPS)
        & (steps <= CAPACITY)
    )
    gl.device_assert(valid, "invalid replay interval")
    if stamps is not None:
        # Mask every access when metadata is invalid, before reducing stamps.
        offsets = gl.arange(
            0,
            triton.next_power_of_2(CAPACITY),
            layout=gl.BlockedLayout([1], [WAVE], [gl.num_warps()], [0]),
        )
        positions = start + offsets
        ptr = (
            stamps
            + layer * stamp_layer_stride
            + (slot * CAPACITY + positions % CAPACITY) * 2
        )
        live = valid & (offsets < steps)
        old_pos = gl.load(ptr, mask=live, other=-1)
        old_gen = gl.load(ptr + 1, mask=live, other=-1)
        fresh = (
            gl.sum(
                (live & ((old_pos != positions) | (old_gen != generation))).to(
                    gl.int32
                ),
                0,
            )
            == 0
        )
        gl.device_assert(fresh, "stale replay record")
        valid = valid & fresh
    return slot, start, valid


def validate_kda_records_verify(
    conv_qkv,
    original_qkv,
    raw_g,
    beta,
    state,
    reads,
    correction,
    key_gate,
    metadata,
    stamps,
    query_start_loc,
    a_log,
    bias,
    heads,
    dim,
    window,
    *,
    f_a_out,
    f_b_weight,
):
    """Check the shared records layout before a pointer-based verify launch."""
    import torch

    batch, tokens = reads.numel(), conv_qkv.shape[0]
    if raw_g is None:
        if f_a_out is None or f_b_weight is None:
            raise ValueError(
                "KDA verify requires either a raw gate or both projection inputs"
            )
        if (
            f_a_out.shape != (tokens, dim)
            or f_b_weight.shape != (heads * dim, dim)
            or not f_b_weight.is_contiguous()
        ):
            raise ValueError(
                "KDA gate projection requires [tokens, D] and [heads * D, D]"
            )
        gate_inputs = (f_a_out, f_b_weight)
    else:
        if f_a_out is not None or f_b_weight is not None:
            raise ValueError("A supplied raw gate excludes projection inputs")
        if raw_g.shape != (tokens, heads * dim):
            raise ValueError("KDA raw gate shape does not match the head geometry")
        gate_inputs = (raw_g,)

    if original_qkv is not None:
        if (
            original_qkv.shape != conv_qkv.shape
            or original_qkv.dtype != conv_qkv.dtype
            or original_qkv.device != conv_qkv.device
            or original_qkv.stride(-1) != 1
            or original_qkv.stride(0) < original_qkv.shape[-1]
        ):
            raise ValueError("Original QKV destination must match the input rows")
        destination = original_qkv.untyped_storage()
        begin, end = (
            destination.data_ptr(),
            destination.data_ptr() + destination.nbytes(),
        )
        for source in (conv_qkv, beta, *gate_inputs):
            storage = source.untyped_storage()
            if (
                begin < storage.data_ptr() + storage.nbytes()
                and storage.data_ptr() < end
            ):
                raise ValueError("Original QKV destination must not overlap producers")

    if dim != 128 or window <= 0 or tokens > batch * window:
        raise ValueError("KDA verify needs D=128 and a positive bounded window")
    if conv_qkv.shape != (tokens, 3 * heads * dim) or beta.shape != (tokens, heads):
        raise ValueError("KDA verify producer shapes do not match the head geometry")
    if (
        state.ndim != 4
        or state.shape[1] < heads
        or state.shape[2:] != (dim, dim)
        or state.dtype != torch.float32
    ):
        raise ValueError("KDA checkpoint must be FP32 [page, head, value, key]")
    if (
        correction.ndim != 4
        or correction.shape[1] != heads
        or correction.shape[3] != dim
        or correction.dtype != torch.float32
    ):
        raise ValueError(
            "KDA correction buffer must be FP32 [slot, head, capacity, value]"
        )
    slots, _, capacity, _ = correction.shape
    if (
        slots < batch
        or capacity < window
        or key_gate.shape != (slots, heads, capacity, 2 * dim)
    ):
        raise ValueError(
            "KDA records must cover the runtime batch and candidate window"
        )
    if conv_qkv.dtype not in (torch.bfloat16, torch.float16) or any(
        x.dtype != conv_qkv.dtype for x in (*gate_inputs, beta, key_gate)
    ):
        raise ValueError(
            "KDA producers and K/raw-g records must use the activation dtype"
        )
    if any(
        x.stride(-1) != 1 for x in (conv_qkv, *gate_inputs, beta, correction, key_gate)
    ):
        raise ValueError("KDA producers and records need contiguous channels")
    if query_start_loc.shape != (batch + 1,) or reads.shape != (batch,):
        raise ValueError(
            "KDA query boundaries and checkpoint indices must match the batch"
        )
    if metadata.shape != (batch, 3) or stamps.shape != (slots, capacity, 2):
        raise ValueError("KDA record metadata and stamps have incompatible shapes")
    if any(x.dtype != torch.int64 or not x.is_contiguous() for x in (metadata, stamps)):
        raise ValueError("KDA record metadata and stamps must be contiguous INT64")
    if (
        a_log.shape != (heads,)
        or bias.numel() != heads * dim
        or any(x.dtype != torch.float32 or not x.is_contiguous() for x in (a_log, bias))
    ):
        raise ValueError(
            "KDA gate parameters must be contiguous FP32 with matching heads"
        )
    if any(
        x.device != conv_qkv.device
        for x in (
            *gate_inputs,
            beta,
            state,
            reads,
            correction,
            key_gate,
            metadata,
            stamps,
            query_start_loc,
            a_log,
            bias,
        )
    ):
        raise ValueError("KDA verify inputs must be on the same device")


def validate_kda_records_commit(
    descriptors, groups, reads, writes, accepted, metadata, stamps, window
):
    """Check shared descriptor and record metadata shapes before recovery."""
    import torch

    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != 10
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("KDA descriptors must be contiguous uint64 [layers, 10]")
    layers, batch = descriptors.shape[0], accepted.numel()
    if (
        groups.shape != (layers,)
        or accepted.shape != (batch,)
        or reads.ndim != 2
        or reads.shape[1] != batch
        or reads.shape != writes.shape
    ):
        raise ValueError(
            "KDA layer groups, page tables and accepted counts have incompatible shapes"
        )
    if any(
        x.dtype not in (torch.int32, torch.int64) or not x.is_contiguous()
        for x in (groups, reads, writes, accepted)
    ):
        raise ValueError(
            "KDA group/page indices and accepted counts must be contiguous integers"
        )
    if (metadata is None) != (stamps is None):
        raise ValueError("KDA record metadata and stamps must be supplied together")
    if window <= 0:
        raise ValueError("KDA candidate window must be positive")
    tensors = (groups, reads, writes, accepted)
    if metadata is not None:
        if (
            stamps.ndim != 4
            or stamps.shape[0] != layers
            or stamps.shape[-1] != 2
            or stamps.shape[1] < batch
            or stamps.shape[2] < window
            or metadata.shape != (batch, 3)
        ):
            raise ValueError(
                "KDA record metadata must cover all layers, rows and candidate positions"
            )
        if any(
            x.dtype != torch.int64 or not x.is_contiguous() for x in (metadata, stamps)
        ):
            raise ValueError("KDA record metadata and stamps must be contiguous INT64")
        tensors += (metadata, stamps)
    if any(x.device != descriptors.device for x in tensors):
        raise ValueError("KDA recovery metadata must be on the descriptor device")


@gluon.jit
def kda_retained_panel(
    running,
    correction,
    key,
    decay,
    page,
    head,
    length,
    u_page,
    u_head,
    u_pos,
    u_channel,
    k_page,
    k_head,
    k_pos,
    k_channel,
    d_page,
    d_head,
    d_pos,
    d_channel,
    values,
    keys,
):
    """Reconstruct one owned [value, key] panel from accepted FP32 records."""
    page = page.to(gl.int64)
    head = head.to(gl.int64)
    values = values.to(gl.int64)
    keys = keys.to(gl.int64)
    for position in range(length):
        history_position = position.to(gl.int64)
        u = gl.load(
            correction
            + page * u_page
            + head * u_head
            + history_position * u_pos
            + values * u_channel
        )
        k = gl.load(
            key
            + page * k_page
            + head * k_head
            + history_position * k_pos
            + keys * k_channel
        )
        d = gl.load(
            decay
            + page * d_page
            + head * d_head
            + history_position * d_pos
            + keys * d_channel
        )
        running = running * d[None, :] + u[:, None] * k[None, :]
    return running


@gluon.jit
def kda_retained_request(
    reads,
    live,
    starts,
    length,
    row,
    read_stride,
    live_stride,
    query_stride,
    length_stride,
    pages,
    CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
):
    row = row.to(gl.int64)
    source = gl.load(reads + row * read_stride).to(gl.int64)
    target = gl.load(live + row * live_stride).to(gl.int64)
    begin = gl.load(starts + row * query_stride).to(gl.int64)
    count = gl.load(starts + (row + 1) * query_stride).to(gl.int64) - begin
    valid = (
        (source > 0)
        & (source < pages)
        & (target > 0)
        & (target < pages)
        & (count > 0)
        & (count <= WINDOW)
    )
    old = gl.load(length + source * length_stride, mask=valid, other=0)
    valid = valid & (old >= 0) & (old <= CAPACITY)
    return source, target, begin, count, old, valid


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "state_stride",
        "u_page",
        "u_head",
        "u_pos",
        "u_channel",
        "k_page",
        "k_head",
        "k_pos",
        "k_channel",
        "d_page",
        "d_head",
        "d_pos",
        "d_channel",
        "read_stride",
        "live_stride",
        "query_stride",
        "length_stride",
        "pages",
    ],
)
def kda_retained_consolidate(
    state,
    correction,
    key,
    decay,
    length,
    reads,
    live,
    starts,
    state_stride,
    u_page,
    u_head,
    u_pos,
    u_channel,
    k_page,
    k_head,
    k_pos,
    k_channel,
    d_page,
    d_head,
    d_pos,
    d_channel,
    read_stride,
    live_stride,
    query_stride,
    length_stride,
    pages,
    HEADS: gl.constexpr,
    DIM: gl.constexpr,
    CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
):
    row, head, panel = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    source, target, begin, count, old, valid = kda_retained_request(
        reads,
        live,
        starts,
        length,
        row,
        read_stride,
        live_stride,
        query_stride,
        length_stride,
        pages,
        CAPACITY,
        WINDOW,
    )
    if not valid or not ((source != target) | (old + count > CAPACITY)):
        return
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [WAVE // 4, 4], [4, 1], [1, 0])
    values = panel * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    keys = gl.arange(0, DIM, layout=gl.SliceLayout(0, layout))
    offset = head * DIM * DIM + values[:, None] * DIM + keys[None, :]
    running = gl.load(state + source.to(gl.int64) * state_stride + offset)
    running = kda_retained_panel(
        running,
        correction,
        key,
        decay,
        source,
        head,
        old,
        u_page,
        u_head,
        u_pos,
        u_channel,
        k_page,
        k_head,
        k_pos,
        k_channel,
        d_page,
        d_head,
        d_pos,
        d_channel,
        values,
        keys,
    )
    gl.store(state + target.to(gl.int64) * state_stride + offset, running)


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "read_stride",
        "live_stride",
        "query_stride",
        "length_stride",
        "pages",
    ],
)
def kda_retained_reset(
    length,
    reads,
    live,
    starts,
    read_stride,
    live_stride,
    query_stride,
    length_stride,
    pages,
    CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
):
    row = gl.program_id(0)
    source, target, begin, count, old, valid = kda_retained_request(
        reads,
        live,
        starts,
        length,
        row,
        read_stride,
        live_stride,
        query_stride,
        length_stride,
        pages,
        CAPACITY,
        WINDOW,
    )
    if valid and ((source != target) | (old + count > CAPACITY)):
        gl.store(length + target * length_stride, 0)


@gluon.jit
def kda_retained_accept(
    reads,
    live,
    starts,
    accepted,
    length,
    metadata,
    stamps,
    row,
    read_stride,
    live_stride,
    query_stride,
    accepted_stride,
    length_stride,
    pages,
    stamp_slots,
    CAPACITY: gl.constexpr,
    STAMP_CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
):
    source, target, begin, count, old_source, valid = kda_retained_request(
        reads,
        live,
        starts,
        length,
        row,
        read_stride,
        live_stride,
        query_stride,
        length_stride,
        pages,
        CAPACITY,
        WINDOW,
    )
    take = gl.minimum(
        gl.maximum(gl.load(accepted + row.to(gl.int64) * accepted_stride), 0),
        gl.minimum(count, WINDOW),
    )
    slot, start, fresh = kda_record_interval(
        metadata, stamps, row, 0, take, stamp_slots, STAMP_CAPACITY, 0, WINDOW, WAVE
    )
    old = gl.load(length + target * length_stride, mask=valid, other=0)
    valid = valid & fresh & (old >= 0) & (old + take <= CAPACITY)
    return source, target, begin, take, old, valid


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "qkv_stride",
        "conv_stride",
        "conv_channel",
        "conv_tap",
        "read_stride",
        "live_stride",
        "query_stride",
        "accepted_stride",
        "length_stride",
        "pages",
        "stamp_slots",
    ],
)
def kda_retained_conv_commit(
    original,
    conv,
    reads,
    live,
    starts,
    accepted,
    length,
    metadata,
    stamps,
    qkv_stride,
    conv_stride,
    conv_channel,
    conv_tap,
    read_stride,
    live_stride,
    query_stride,
    accepted_stride,
    length_stride,
    pages,
    stamp_slots,
    CHANNELS: gl.constexpr,
    CAPACITY: gl.constexpr,
    STAMP_CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
):
    row = gl.program_id(0)
    source, target, begin, take, old, valid = kda_retained_accept(
        reads,
        live,
        starts,
        accepted,
        length,
        metadata,
        stamps,
        row,
        read_stride,
        live_stride,
        query_stride,
        accepted_stride,
        length_stride,
        pages,
        stamp_slots,
        CAPACITY,
        STAMP_CAPACITY,
        WINDOW,
        WAVE,
    )
    if not valid:
        return
    offsets = (
        gl.program_id(1) * 256
        + gl.arange(0, 256, layout=gl.BlockedLayout([1], [WAVE], [4], [0]))
    ).to(gl.int64)
    mask = offsets < CHANNELS
    base = conv + source.to(gl.int64) * conv_stride + offsets * conv_channel
    # All source taps and accepted inputs are loaded before any aliasing store.
    h0 = gl.load(base, mask=mask, other=0)
    h1 = gl.load(base + conv_tap, mask=mask, other=0)
    h2 = gl.load(base + 2 * conv_tap, mask=mask, other=0)
    for token in gl.static_range(WINDOW):
        x = gl.load(
            original + (begin + token) * qkv_stride + offsets,
            mask=mask & (token < take),
            other=0,
        ).to(conv.dtype.element_ty)
        h0 = gl.where(token < take, h1, h0)
        h1 = gl.where(token < take, h2, h1)
        h2 = gl.where(token < take, x, h2)
    destination = conv + target.to(gl.int64) * conv_stride + offsets * conv_channel
    gl.store(destination, h0, mask=mask)
    gl.store(destination + conv_tap, h1, mask=mask)
    gl.store(destination + 2 * conv_tap, h2, mask=mask)


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "read_stride",
        "live_stride",
        "query_stride",
        "accepted_stride",
        "length_stride",
        "pages",
        "stamp_slots",
    ],
)
def kda_retained_publish(
    length,
    reads,
    live,
    starts,
    accepted,
    metadata,
    stamps,
    read_stride,
    live_stride,
    query_stride,
    accepted_stride,
    length_stride,
    pages,
    stamp_slots,
    CAPACITY: gl.constexpr,
    STAMP_CAPACITY: gl.constexpr,
    WINDOW: gl.constexpr,
    WAVE: gl.constexpr,
):
    row = gl.program_id(0)
    source, target, begin, take, old, valid = kda_retained_accept(
        reads,
        live,
        starts,
        accepted,
        length,
        metadata,
        stamps,
        row,
        read_stride,
        live_stride,
        query_stride,
        accepted_stride,
        length_stride,
        pages,
        stamp_slots,
        CAPACITY,
        STAMP_CAPACITY,
        WINDOW,
        WAVE,
    )
    if valid:
        gl.store(length + target * length_stride, old + take)


def prepare_kda_retained(
    state,
    correction,
    key,
    decay,
    length,
    reads,
    live,
    starts,
    *,
    heads,
    dim,
    window,
    wave,
):
    """Materialize cross-page/overflow endpoints, then reset live lengths."""
    batch = reads.numel()
    if not batch:
        return
    capacity = correction.shape[2]
    kda_retained_consolidate[(batch, heads, dim // 16)](
        state,
        correction,
        key,
        decay,
        length,
        reads,
        live,
        starts,
        state.stride(0),
        *correction.stride(),
        *key.stride(),
        *decay.stride(),
        reads.stride(0),
        live.stride(0),
        starts.stride(0),
        length.stride(0),
        state.shape[0],
        HEADS=heads,
        DIM=dim,
        CAPACITY=capacity,
        WINDOW=window,
        WAVE=wave,
        num_warps=4,
        debug=False,
    )
    kda_retained_reset[(batch,)](
        length,
        reads,
        live,
        starts,
        reads.stride(0),
        live.stride(0),
        starts.stride(0),
        length.stride(0),
        state.shape[0],
        CAPACITY=capacity,
        WINDOW=window,
        num_warps=4,
        debug=False,
    )


def launch_kda_retained_commit(
    *,
    original_qkv,
    conv_pool,
    read_indices,
    live_indices,
    query_start_loc,
    accepted_length,
    retained_length,
    history_capacity,
    record_metadata,
    record_stamps,
    num_heads,
    head_dim,
    draft_token_num,
    wave,
):
    """Commit only convolution taps and publish accepted retained length."""
    batch = read_indices.numel()
    if not batch:
        return
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("KDA metadata and stamps must be supplied together")
    stamp_slots = batch if record_stamps is None else record_stamps.shape[0]
    stamp_capacity = (
        history_capacity if record_stamps is None else record_stamps.shape[1]
    )
    common = (
        read_indices.stride(0),
        live_indices.stride(0),
        query_start_loc.stride(0),
        accepted_length.stride(0),
        retained_length.stride(0),
        conv_pool.shape[0],
        stamp_slots,
    )
    kda_retained_conv_commit[(batch, triton.cdiv(3 * num_heads * head_dim, 256))](
        original_qkv,
        conv_pool,
        read_indices,
        live_indices,
        query_start_loc,
        accepted_length,
        retained_length,
        record_metadata,
        record_stamps,
        original_qkv.stride(0),
        *conv_pool.stride(),
        *common,
        CHANNELS=3 * num_heads * head_dim,
        CAPACITY=history_capacity,
        STAMP_CAPACITY=stamp_capacity,
        WINDOW=draft_token_num,
        WAVE=wave,
        num_warps=4,
        debug=False,
    )
    # Separate launches provide the cross-workgroup completion boundary.
    kda_retained_publish[(batch,)](
        retained_length,
        read_indices,
        live_indices,
        query_start_loc,
        accepted_length,
        record_metadata,
        record_stamps,
        *common,
        CAPACITY=history_capacity,
        STAMP_CAPACITY=stamp_capacity,
        WINDOW=draft_token_num,
        WAVE=wave,
        num_warps=4,
        debug=False,
    )


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
    )
    if metadata is not None:
        if (
            metadata.shape != (batch, 3)
            or stamps.ndim != 3
            or stamps.shape[0] < batch
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
