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
from tokenspeed_kernel.ops.metadata.stop_conditions import REQUEST_ENDED_INCOMPLETE
from tokenspeed_kernel.platform import current_platform


def _kernels():
    if current_platform().is_amd:
        from tokenspeed_kernel_amd.ops import _kda_records

        return _kda_records, {"WAVE": 64 if current_platform().is_cdna4 else 32}
    from tokenspeed_kernel.ops.attention.kda._triton import retained

    return retained, {}


def materialize(recurrent, *, history, source_indices, recurrent_layout, paged):
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
        PAGED=paged,
        K=keys,
        V=values,
        CAPACITY=capacity,
        V_MAJOR=recurrent_layout == "v_major",
        BLOCK=256,
        num_warps=4,
        **options,
    )
    if paged:
        reset_length(
            history.length,
            write_indices=source_indices,
            record_slots=history.record_slots,
            page_count=pages,
        )


def reset_length(length, *, write_indices, record_slots, page_count):
    """Zero the slot length of each row whose destination page lies in the pool."""
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 1
    ):
        raise ValueError("page_count must be a positive page-pool size")
    if (
        length.ndim != 2
        or length.shape[1] != 1
        or length.dtype != torch.int32
        or write_indices.ndim != 1
        or record_slots.ndim != 1
        or record_slots.numel() < write_indices.numel()
        or any(
            x.dtype not in (torch.int32, torch.int64) or x.device != length.device
            for x in (write_indices, record_slots)
        )
    ):
        raise ValueError(
            "length must be INT32 [slot, 1] and pages/slots integer request vectors"
        )
    if write_indices.numel() == 0:
        return
    kernels, options = _kernels()
    kernels.retained_reset_length_kernel[(triton.cdiv(write_indices.numel(), 256),)](
        length,
        write_indices,
        record_slots,
        record_slots.stride(0),
        length.shape[0],
        page_count,
        write_indices.numel(),
        length.stride(0),
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
    request_stopped,
    request_ends,
    draft_token_num,
    history_capacity,
    num_heads,
    head_dim,
    conv_width,
    activation_dtype,
    conv_state_dtype,
    recurrent_state_dtype,
):
    """Update all layers' convolution, then publish their accepted lengths.

    descriptors is contiguous UINT64 [layers, 14]: original QKV, convolution,
    length pointers; page count; QKV row, convolution page, length slot,
    convolution channel, convolution tap element strides; the v-major
    recurrent state pointer and its page, head, value and key element
    strides. QKV channels have unit stride. Every layer's length holds
    record_slot_count request slots; record_slots selects each request's slot.
    group_indices maps layers into [groups, requests] page tables.
    Each live destination is private and cannot overlap another active source.
    Per-layer descriptor values are validated by the cache owner when binding
    its arena; the launch validates shared tensor geometry without GPU reads.

    query_start_loc is the packed [requests + 1] boundary vector. Acceptance is
    clamped to the actual interval and draft_token_num. metadata is contiguous
    INT64 [requests, 3]; stamps is INT64 [layers, slots, ring capacity, 2],
    including strided layer subsets. Every accepted stamp must match before
    either phase writes. history_capacity and ring capacity are independent.

    retained_correction, retained_key and retained_decay are the FP32 records
    [record slots, layers, heads, history_capacity, head_dim]. request_stopped
    is a bool per request slot, set for a request that ended in this forward.
    Such a request's accepted records, through this round's acceptance, are
    folded into its live page's dense state, and its length is cleared
    instead of advanced. No other dense checkpoint or record is copied.
    request_ends is the contiguous INT32 [requests] ``apply_stop_conditions``
    report for these rows: a request that ended but whose commit a layer
    rejected (stale stamps, an interval past the capacity) keeps records
    outside its live page, and its entry becomes ``REQUEST_ENDED_INCOMPLETE``
    so its finish publishes nothing.
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
    if (
        activation_dtype not in (torch.bfloat16, torch.float16)
        or conv_state_dtype not in types
        or recurrent_state_dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError(
            "unsupported retained activation, convolution or recurrent dtype"
        )
    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != 14
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("retained descriptors must be contiguous UINT64 [layers, 14]")
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
    args = (
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
        read_indices.shape[0],
        *read_indices.stride(),
        *write_indices.stride(),
        accepted_length.stride(0),
        query_start_loc.stride(0),
        record_slots.stride(0),
        record_slot_count,
        record_stamps.stride(),
        record_stamps.shape[1],
        record_stamps.shape[2],
        request_stopped.numel(),
    )
    kernels.retained_batched_conv_kernel[
        (layers, batch, triton.cdiv(3 * num_heads * head_dim, 256))
    ](
        *args,
        *records,
        *(x.stride() for x in records),
        CAPACITY=history_capacity,
        WINDOW=draft_token_num,
        CHANNELS=3 * num_heads * head_dim,
        ACTIVATION=types[activation_dtype],
        CONV=types[conv_state_dtype],
        STATE=types[recurrent_state_dtype],
        HEADS=num_heads,
        DIM=head_dim,
        BLOCK=256,
        num_warps=4,
        **options,
    )
    kernels.retained_batched_publish_kernel[(layers, batch)](
        *args,
        request_ends,
        CAPACITY=history_capacity,
        WINDOW=draft_token_num,
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
    history_capacity,
    num_heads,
    head_dim,
    recurrent_state_dtype,
):
    """Ready every layer's retained history for a verify forward in two launches.

    descriptors, group_indices, record_slots and the FP32 records
    [record slots, layers, heads, history_capacity, head_dim] follow
    batched_commit; the launch reads only the length and recurrent-state
    columns of each descriptor. source_indices and live_indices are integer
    [groups, requests] tables of the pages verify reads from and the private
    pages it reads its dense base from; query_start_loc is the packed
    [requests + 1] boundary vector.

    A request folds when its live page differs from its source or when its
    accepted count L plus its window T exceeds history_capacity: the first
    launch writes the source state with records [0, L) applied into the live
    page, and a second, ordered launch clears L once every tile has read it.
    Padding rows (a page at or below 0 or at or past the pool, an empty
    window or a slot outside the record slots) keep their pages and lengths;
    an L or T out of range on any other row fails a device assertion.
    """
    if num_heads <= 0 or head_dim <= 0 or history_capacity <= 0:
        raise ValueError(
            "retained preparation requires positive heads, head_dim and capacity"
        )
    states = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    if recurrent_state_dtype not in states:
        raise ValueError("unsupported retained recurrent state dtype")
    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != 14
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("retained descriptors must be contiguous UINT64 [layers, 14]")
    layers = descriptors.shape[0]
    batch = record_slots.numel()
    if (
        group_indices.shape != (layers,)
        or not group_indices.is_contiguous()
        or source_indices.ndim != 2
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
    if any(x.device != descriptors.device for x in (*routing, record_slots, *records)):
        raise ValueError("retained preparation tensors must share a device")
    if not layers or not batch:
        return
    # Both vendors share these Triton kernels.
    from tokenspeed_kernel.ops.attention.kda._triton import retained

    args = (
        descriptors,
        group_indices,
        source_indices,
        live_indices,
        query_start_loc,
        record_slots,
    )
    rows = (
        batch,
        source_indices.shape[0],
        *source_indices.stride(),
        *live_indices.stride(),
        query_start_loc.stride(0),
        record_slots.stride(0),
        record_slot_count,
    )
    keys = triton.next_power_of_2(head_dim)
    # Split a request's heads over about 16 programs per SM, so every SM keeps
    # loads in flight when few requests fold (a batch-1 fold of 69 layers takes
    # 52 us on GB300, 70 us at 2 programs per SM); a large batch keeps one
    # program per request so rows that do not fold cost one early exit each.
    splits = min(
        num_heads,
        max(1, triton.cdiv(16 * current_platform().sm_count, layers * batch)),
    )
    retained.retained_batched_consolidate_kernel[(layers, batch, splits)](
        *args,
        *records,
        *rows,
        *(x.stride() for x in records),
        CAPACITY=history_capacity,
        STATE=states[recurrent_state_dtype],
        HEADS=num_heads,
        DIM=head_dim,
        KEYS=keys,
        ROWS=min(triton.next_power_of_2(head_dim), max(1, 2048 // keys)),
        num_warps=4,
    )
    retained.retained_batched_reset_kernel[(layers, triton.cdiv(batch, 128))](
        *args,
        *rows,
        CAPACITY=history_capacity,
        BLOCK=128,
        num_warps=4,
        # Keeps the interval assertion.
        debug=True,
    )
