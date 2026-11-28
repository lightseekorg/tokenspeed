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

"""Validation and launch boundary for request-owned KDA records."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda import (
    KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS as DESCRIPTOR_COLUMNS,
)
from tokenspeed_kernel.ops.metadata.stop_conditions import (
    REQUEST_ENDED,
    REQUEST_ENDED_INCOMPLETE,
)
from tokenspeed_kernel.platform import current_platform

# Request rows a commit program publishes, and the end programs' row and
# channel block.
_COMMIT_BLOCK = 256
# At most this many end programs per layer share an ended request's heads.
_END_SPLITS = 16


def _kernels():
    if current_platform().is_amd:
        from tokenspeed_kernel_amd.ops import _kda_records

        return _kda_records, {"WAVE": 64 if current_platform().is_cdna4 else 32}
    from tokenspeed_kernel.ops.attention.kda._triton import retained

    return retained, {}


def _check_replay_state(replay_state, *, slots, device):
    # Launches write each slot's row independently, so rows must not share
    # elements: a row pitch of at least the two columns (no expanded or
    # overlapping rows).
    if (
        replay_state.shape != (slots, 2)
        or replay_state.stride(1) != 1
        or (slots > 1 and replay_state.stride(0) < 2)
        or replay_state.dtype != torch.int32
        or replay_state.device != device
    ):
        raise ValueError(
            "replay_state must be INT32 [record slots, 2] (count, ring state) "
            "with adjacent columns and disjoint rows"
        )


def _byte_span(tensor):
    """The [first, last) addresses a non-empty tensor's elements lie within."""
    first = tensor.data_ptr()
    reach = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return first, first + (reach + 1) * tensor.element_size()


def _check_separate_replay(replay_state, snapshot):
    """Reject a snapshot that shares an element with the replay state.

    The prepare reads replay_state while it writes the snapshot; the
    commit's publish programs write replay_state while its end programs
    read the snapshot. Disjoint views of one allocation are allowed,
    including a snapshot in the padding between strided replay rows.
    Callers check both layouts first (_check_replay_state,
    _check_snapshot): the snapshot is contiguous, so its bytes are one
    interval, and replay_state's are its rows, each of adjacent elements,
    at a pitch of at least a row when there are several.
    """
    if replay_state.numel() == 0 or snapshot.numel() == 0:
        return
    state_first, state_last = _byte_span(replay_state)
    snapshot_first, snapshot_last = _byte_span(snapshot)
    if snapshot_last <= state_first or state_last <= snapshot_first:
        return
    # The spans meet. Row i holds [state_first + i * pitch, + row), and it
    # meets the snapshot exactly when first <= i <= last below.
    rows = replay_state.shape[0]
    row = replay_state.shape[1] * replay_state.element_size()
    pitch = replay_state.stride(0) * replay_state.element_size()
    if rows > 1 and pitch >= row:
        first = max(0, (snapshot_first - state_first - row + pitch) // pitch)
        last = min(rows - 1, (snapshot_last - state_first - 1) // pitch)
        if first > last:
            return
    raise ValueError("replay_state and replay_snapshot must not share memory")


def _check_snapshot(snapshot, *, batch, device):
    if (
        snapshot.shape != (batch, 2)
        or not snapshot.is_contiguous()
        or snapshot.dtype != torch.int32
        or snapshot.device != device
    ):
        raise ValueError(
            "replay_snapshot must be contiguous INT32 [requests, 2] "
            "(applied count, ring head)"
        )


def materialize(
    recurrent,
    *,
    history,
    source_indices,
    recurrent_layout,
    paged,
    conv_pool,
    preconv,
    ring_state,
):
    slots, heads, capacity, values = history.correction.shape
    keys = history.key.shape[-1]
    expected = (values, keys) if recurrent_layout == "v_major" else (keys, values)
    if recurrent_layout not in ("v_major", "k_major"):
        raise ValueError("unsupported recurrent layout")
    rows = source_indices.numel()
    # A paged fold writes pool pages; a gathered fold writes one row per source.
    pages = recurrent.shape[0]
    if (
        recurrent.shape[1:] != (heads, *expected)
        or (not paged and pages != rows)
        or history.record_slots.numel() < rows
        or recurrent.dtype not in (torch.float32, torch.bfloat16)
        or source_indices.ndim != 1
        or source_indices.dtype not in (torch.int32, torch.int64)
        or source_indices.device != recurrent.device
        or history.correction.device != recurrent.device
    ):
        raise ValueError(
            "gathered recurrent state and source indices must match retained history"
        )
    if (
        preconv.ndim != 3
        or preconv.shape[0] != slots
        or preconv.shape[1] <= 3
        or preconv.stride(2) != 1
        or conv_pool.ndim != 3
        or conv_pool.shape[1:] != (preconv.shape[2], 3)
        or (paged and conv_pool.shape[0] != pages)
        or conv_pool.dtype != preconv.dtype
        or ring_state.shape != (slots, 1)
        or ring_state.dtype != torch.int32
        or any(x.device != recurrent.device for x in (conv_pool, preconv, ring_state))
    ):
        raise ValueError(
            "convolution pool [pages, channels, 3], pre-convolution ring "
            "[record slots, window + 3, channels] and INT32 [record slots, 1] "
            "ring state must match the retained history"
        )
    if source_indices.numel() == 0:
        return
    kernels, options = _kernels()
    kernels.retained_materialize_kernel[
        (source_indices.numel(), heads, triton.cdiv(keys * values, 256))
    ](
        recurrent,
        history.correction,
        history.key,
        history.decay,
        history.length,
        source_indices,
        history.record_slots,
        recurrent.stride(),
        history.correction.stride(),
        history.key.stride(),
        history.decay.stride(),
        history.length.stride(0),
        source_indices.stride(0),
        history.record_slots.stride(0),
        pages,
        slots,
        conv_pool,
        preconv,
        ring_state,
        history.live_state_indices,
        history.live_state_indices.stride(0),
        conv_pool.shape[0],
        conv_pool.stride(),
        preconv.stride(0),
        preconv.stride(1),
        ring_state.stride(0),
        PAGED=paged,
        K=keys,
        V=values,
        CAPACITY=capacity,
        V_MAJOR=recurrent_layout == "v_major",
        CHANNELS=preconv.shape[2],
        WINDOW=preconv.shape[1] - 3,
        BLOCK=256,
        num_warps=4,
        **options,
    )


def reset_length(replay_state, *, write_indices, record_slots, page_count):
    """Zero the count and ring state of each row whose destination page lies in the pool."""
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 1
    ):
        raise ValueError("page_count must be a positive page-pool size")
    _check_replay_state(
        replay_state, slots=replay_state.shape[0], device=replay_state.device
    )
    if (
        write_indices.ndim != 1
        or record_slots.ndim != 1
        or record_slots.numel() < write_indices.numel()
        or any(
            x.dtype not in (torch.int32, torch.int64) or x.device != replay_state.device
            for x in (write_indices, record_slots)
        )
    ):
        raise ValueError("pages and slots must be integer request vectors")
    if write_indices.numel() == 0:
        return
    kernels, options = _kernels()
    kernels.retained_reset_length_kernel[(triton.cdiv(write_indices.numel(), 256),)](
        replay_state,
        write_indices,
        record_slots,
        record_slots.stride(0),
        replay_state.shape[0],
        page_count,
        write_indices.numel(),
        replay_state.stride(0),
        write_indices.stride(0),
        BLOCK=256,
        num_warps=4,
        **options,
    )


def batched_commit(
    *,
    descriptors,
    group_indices,
    read_indices,
    write_indices,
    accepted_length,
    query_start_loc,
    record_slots,
    record_slot_count,
    record_metadata,
    record_stamps,
    retained_correction,
    retained_key,
    retained_decay,
    replay_state,
    replay_snapshot,
    request_stopped,
    request_ends,
    draft_token_num,
    history_capacity,
    num_heads,
    head_dim,
    conv_width,
    conv_state_dtype,
    recurrent_state_dtype,
):
    """Publish each request's count and ring state and materialize ended requests, in one launch.

    descriptors is contiguous UINT64 [layers, 13]: the pre-convolution ring
    and convolution pointers; page count; ring row, convolution page,
    convolution channel and convolution tap element strides; the v-major
    recurrent state pointer and its page, head, value and key element
    strides; the ring's record-slot element stride. The ring holds
    draft_token_num + 3 rows of the convolution state's dtype per record
    slot and layer, with unit channel stride. record_slots selects each
    request's record slot of record_slot_count. group_indices maps layers
    into [groups, requests] page tables. Each live destination is private
    and cannot overlap another active source. Per-layer descriptor values
    are validated by the cache owner when binding its arena; the launch
    validates shared tensor geometry without GPU reads.

    query_start_loc is the packed [requests + 1] boundary vector. Acceptance is
    clamped to the actual interval and draft_token_num. metadata is contiguous
    INT64 [requests, 3]; stamps is INT64 [layers, slots, ring capacity, 2],
    including strided layer subsets. history_capacity and stamp ring
    capacity are independent.

    retained_correction, retained_key and retained_decay are the FP32 records
    [record slots, layers, heads, history_capacity, head_dim]. Every layer of
    a request follows one replay state: replay_state is INT32 [record slots,
    2], the count H of records accepted since each request's dense base and
    the ring state (0 before the ring holds the request's inputs, else the
    ring row of its next candidate plus one). replay_snapshot is the
    contiguous INT32 [requests, 2] the round's prepare wrote: the count its
    verify applied (H within [0, history_capacity - draft_token_num], else
    0: the prepare folded the records) and the ring row of its first
    candidate. The two must not share memory, and replay_state's rows must
    be disjoint; either violation raises ValueError before the launch.

    The launch's publish programs write each carried request's count once
    -- carried rows are those paged in a group a launched layer lists --
    to the applied count plus its acceptance, and its ring state past the
    accepted candidates; they do not inspect stamps. The ring keeps a
    request's convolution inputs between materializations, so no layer
    writes a convolution state for a request that keeps decoding.
    request_stopped is a bool per request slot, of any element stride, set
    for a request that ended in this forward. Per layer, the launch's end
    programs check every accepted stamp of such a request, then fold its
    accepted records, through this round's acceptance, into its live page's
    dense state and write the ring's last three accepted inputs, oldest
    first, as that page's convolution state; the publish clears its count
    and ring state instead of advancing them. No other dense checkpoint or record is
    copied. request_ends is the contiguous INT32 [requests]
    ``apply_stop_conditions`` report for these rows, ``REQUEST_ENDED``
    exactly for the requests set in request_stopped: a request that ended
    but whose commit a layer that carried it rejected (stale stamps) keeps
    records and inputs outside its live page, and that layer raises its
    entry to ``REQUEST_ENDED_INCOMPLETE`` so its finish publishes nothing.
    End programs read only the snapshot, stamps, records and ring, and
    publish programs write only the replay state, so the launch has no
    read-after-write hazard.
    """
    if (
        num_heads <= 0
        or head_dim <= 0
        or conv_width != 4
        or draft_token_num <= 0
        or history_capacity < draft_token_num
    ):
        raise ValueError(
            "retained commit requires positive heads/window, positive head_dim, conv_width=4 and sufficient capacity"
        )
    types = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    if conv_state_dtype not in types or recurrent_state_dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("unsupported retained convolution or recurrent dtype")
    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != DESCRIPTOR_COLUMNS
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError(
            f"retained descriptors must be contiguous UINT64 [layers, {DESCRIPTOR_COLUMNS}]"
        )
    layers = descriptors.shape[0]
    batch = accepted_length.numel()
    if (
        group_indices.shape != (layers,)
        or not group_indices.is_contiguous()
        or read_indices.ndim != 2
        or read_indices.shape[1] != batch
        or write_indices.shape != read_indices.shape
        or accepted_length.shape != (batch,)
        or query_start_loc.shape != (batch + 1,)
        or record_slots.shape != (batch,)
        or record_slot_count <= 0
    ):
        raise ValueError(
            "retained commit requires layer groups, [groups, batch] pages, "
            "packed boundaries and one record slot per request"
        )
    if any(
        x.dtype not in (torch.int32, torch.int64)
        for x in (
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            query_start_loc,
            record_slots,
        )
    ):
        raise ValueError(
            "retained routing and accepted lengths must be integer tensors"
        )
    if (
        record_metadata.shape != (batch, 3)
        or not record_metadata.is_contiguous()
        or record_stamps.ndim != 4
        or record_stamps.shape[0] != layers
        or record_stamps.shape[-1] != 2
        or record_stamps.shape[2] < draft_token_num
        or record_metadata.dtype != torch.int64
        or record_stamps.dtype != torch.int64
    ):
        raise ValueError(
            "retained metadata/stamps must be INT64 [batch,3] and [layers,slots,capacity,2]"
        )
    records = (retained_correction, retained_key, retained_decay)
    if any(
        x.shape != (record_slot_count, layers, num_heads, history_capacity, head_dim)
        or x.dtype != torch.float32
        for x in records
    ):
        raise ValueError(
            "retained records must be FP32 [record slots, layers, heads, "
            "capacity, head_dim]"
        )
    _check_replay_state(
        replay_state, slots=record_slot_count, device=descriptors.device
    )
    _check_snapshot(replay_snapshot, batch=batch, device=descriptors.device)
    _check_separate_replay(replay_state, replay_snapshot)
    if request_stopped.ndim != 1 or request_stopped.dtype != torch.bool:
        raise ValueError("request stop flags must be a bool vector")
    if (
        request_ends.shape != (batch,)
        or request_ends.dtype != torch.int32
        or not request_ends.is_contiguous()
    ):
        raise ValueError("request ends must be contiguous INT32 [requests]")
    if any(
        x.device != descriptors.device
        for x in (
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            query_start_loc,
            record_slots,
            record_metadata,
            record_stamps,
            *records,
            request_stopped,
            request_ends,
        )
    ):
        raise ValueError("retained commit tensors must share a device")
    if not layers or not batch:
        return
    kernels, options = _kernels()
    # Publish programs first, then SPLITS end programs per layer: each loads
    # the stop flags of the batch and materializes its share (heads h with
    # h % SPLITS == split, and a channel slice) of the requests that ended.
    splits = min(num_heads, _END_SPLITS)
    kernels.retained_batched_commit_kernel[
        (triton.cdiv(batch, _COMMIT_BLOCK) + layers * splits,)
    ](
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        query_start_loc,
        record_slots,
        record_metadata,
        record_stamps,
        request_stopped,
        replay_snapshot,
        request_ends,
        replay_state,
        batch,
        layers,
        read_indices.shape[0],
        *read_indices.stride(),
        *write_indices.stride(),
        accepted_length.stride(0),
        query_start_loc.stride(0),
        record_slots.stride(0),
        record_slot_count,
        replay_state.stride(0),
        record_stamps.stride(),
        record_stamps.shape[1],
        record_stamps.shape[2],
        request_stopped.numel(),
        request_stopped.stride(0),
        *records,
        *(x.stride() for x in records),
        WINDOW=draft_token_num,
        CHANNELS=3 * num_heads * head_dim,
        CONV=types[conv_state_dtype],
        STATE=types[recurrent_state_dtype],
        HEADS=num_heads,
        DIM=head_dim,
        SPLITS=splits,
        # Model-fixed bounds of the group and layer counts.
        GROUPS=triton.next_power_of_2(read_indices.shape[0]),
        LAYERS=triton.next_power_of_2(layers),
        BLOCK=_COMMIT_BLOCK,
        ENDED=REQUEST_ENDED,
        INCOMPLETE=REQUEST_ENDED_INCOMPLETE,
        num_warps=4,
        **options,
    )


def batched_prepare(
    *,
    descriptors,
    group_indices,
    source_indices,
    live_indices,
    query_start_loc,
    record_slots,
    record_slot_count,
    retained_correction,
    retained_key,
    retained_decay,
    replay_state,
    replay_snapshot,
    history_capacity,
    draft_token_num,
    num_heads,
    head_dim,
    conv_state_dtype,
    recurrent_state_dtype,
):
    """Ready every layer's retained history and ring for a verify forward in one launch.

    descriptors, group_indices, record_slots, replay_state and the FP32
    records [record slots, layers, heads, history_capacity, head_dim] follow
    batched_commit; every layer must share the page count. source_indices
    and live_indices are integer [groups, requests] tables of the pages
    verify reads from and the private pages it reads its dense base from;
    query_start_loc is the packed [requests + 1] boundary vector;
    draft_token_num is the window the verify that follows was built for.

    Every layer of a request shares its count H and ring state (replay_state,
    INT32 [record slots, 2]), which the launch only reads. H overflows when
    it lies outside [0, history_capacity - draft_token_num]: the verify
    kernels apply at most that many records, and they and the commit then
    apply none. Per layer, a request whose H overflows gets its source state
    with records [0, H) applied in its live page (in place when the two are
    one page; an H past history_capacity applies none); otherwise a request
    whose live page differs from its source in the layer's group gets the
    source state copied without any record, since the H records stay valid
    on top of it. A request whose ring state is 0 gets ring rows 0..2 of
    every layer seeded from its source page's convolution taps. Padding rows
    (pages at or below 0 or at or past the pool, an empty window or a slot
    outside the record slots) keep their pages and ring rows.

    replay_snapshot (contiguous INT32 [requests, 2]) receives each row's
    applied count (H, or 0 when it overflows) and the ring row of its first
    candidate (3 for a ring seeded here, else the ring state minus one): the
    verify and the commit of the round read these. It must not share memory
    with replay_state, whose rows must be disjoint (ValueError otherwise).
    """
    if num_heads <= 0 or head_dim <= 0 or history_capacity <= 0:
        raise ValueError(
            "retained preparation requires positive heads, head_dim and capacity"
        )
    if not 0 < draft_token_num <= history_capacity:
        raise ValueError(
            "retained preparation requires a verify window within the capacity"
        )
    states = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    convs = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    if recurrent_state_dtype not in states or conv_state_dtype not in convs:
        raise ValueError("unsupported retained convolution or recurrent state dtype")
    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != DESCRIPTOR_COLUMNS
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError(
            f"retained descriptors must be contiguous UINT64 [layers, {DESCRIPTOR_COLUMNS}]"
        )
    layers = descriptors.shape[0]
    batch = record_slots.numel()
    if (
        group_indices.shape != (layers,)
        or not group_indices.is_contiguous()
        or source_indices.ndim != 2
        or source_indices.shape[0] < 1
        or source_indices.shape[1] != batch
        or live_indices.shape != source_indices.shape
        or query_start_loc.shape != (batch + 1,)
        or record_slots.shape != (batch,)
        or record_slot_count <= 0
    ):
        raise ValueError(
            "retained preparation requires layer groups, [groups, batch] source "
            "and live pages, packed boundaries and one record slot per request"
        )
    routing = (group_indices, source_indices, live_indices, query_start_loc)
    if any(x.dtype not in (torch.int32, torch.int64) for x in (*routing, record_slots)):
        raise ValueError("retained routing must be integer tensors")
    records = (retained_correction, retained_key, retained_decay)
    if any(
        x.shape != (record_slot_count, layers, num_heads, history_capacity, head_dim)
        or x.dtype != torch.float32
        for x in records
    ):
        raise ValueError(
            "retained records must be FP32 [record slots, layers, heads, "
            "capacity, head_dim]"
        )
    _check_replay_state(
        replay_state, slots=record_slot_count, device=descriptors.device
    )
    _check_snapshot(replay_snapshot, batch=batch, device=descriptors.device)
    _check_separate_replay(replay_state, replay_snapshot)
    if any(x.device != descriptors.device for x in (*routing, record_slots, *records)):
        raise ValueError("retained preparation tensors must share a device")
    if not layers or not batch:
        return
    # Both vendors share this Triton kernel.
    from tokenspeed_kernel.ops.attention.kda._triton import retained

    keys = triton.next_power_of_2(head_dim)
    # Split a request's heads over about 16 programs per SM, so every SM keeps
    # loads in flight when few requests fold (a batch-1 fold of 69 layers takes
    # 52 us on GB300, 70 us at 2 programs per SM); a large batch keeps one
    # program per request so rows that do not fold cost one early exit each.
    splits = min(
        num_heads,
        max(1, triton.cdiv(16 * current_platform().sm_count, layers * batch)),
    )
    retained.retained_batched_prepare_kernel[(layers, batch, splits)](
        descriptors,
        group_indices,
        source_indices,
        live_indices,
        query_start_loc,
        record_slots,
        replay_state,
        replay_snapshot,
        *records,
        source_indices.shape[0],
        *source_indices.stride(),
        *live_indices.stride(),
        query_start_loc.stride(0),
        record_slots.stride(0),
        record_slot_count,
        replay_state.stride(0),
        *(x.stride() for x in records),
        CAPACITY=history_capacity,
        WINDOW=draft_token_num,
        CHANNELS=3 * num_heads * head_dim,
        CONV=convs[conv_state_dtype],
        STATE=states[recurrent_state_dtype],
        HEADS=num_heads,
        DIM=head_dim,
        KEYS=keys,
        ROWS=min(triton.next_power_of_2(head_dim), max(1, 2048 // keys)),
        SEED_BLOCK=512,
        num_warps=4,
    )
