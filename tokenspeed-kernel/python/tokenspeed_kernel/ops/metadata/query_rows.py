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

"""Build packed query layouts and gather row-aligned model tensors."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit(
    do_not_specialize=["stride0", "stride1", "stride2", "stride3"],
)
def _gather_query_rows_kernel(
    Inputs,
    Indices,
    Valid,
    Outputs,
    stride0,
    stride1,
    stride2,
    stride3,
    WIDTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    input_strides = (stride0, stride1, stride2, stride3)
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(Indices + row).to(tl.int64)
    valid = tl.load(Valid + row)
    for i in tl.static_range(len(WIDTHS)):
        values = tl.load(
            Inputs[i] + source * input_strides[i] + columns,
            mask=valid & (columns < WIDTHS[i]),
            other=0,
        )
        tl.store(
            Outputs[i] + row * WIDTHS[i] + columns, values, mask=columns < WIDTHS[i]
        )


def gather_query_rows(
    values: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Gather rows of aligned tensors, writing zero for invalid destinations.

    Args:
        values: One to four tensors with a shared first row dimension.
            Inner dimensions must be contiguous; the row stride may differ.
        indices: Contiguous int64 source-row indices, one per output row.
            Indices for valid rows must be in range; invalid rows are not read.
        valid: Contiguous bool mask with the same shape as indices.

    Returns:
        Contiguous tensors with ``indices.numel()`` rows and the input inner
        shapes and dtypes. Values are copied without changing arithmetic.
    """
    if not 1 <= len(values) <= 4 or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("query gather needs tensors and int64 row indices")
    if valid.shape != indices.shape or valid.dtype != torch.bool:
        raise ValueError("query gather mask must be bool and match row indices")
    if not indices.is_contiguous() or not valid.is_contiguous():
        raise ValueError("query gather maps must be contiguous")
    if values[0].ndim < 1 or values[0].shape[0] == 0:
        raise ValueError("query gather needs nonempty input rows")
    rows = values[0].shape[0]
    widths = tuple(math.prod(value.shape[1:]) for value in values)
    for value in values:
        if value.ndim < 1 or value.shape[0] != rows:
            raise ValueError("query gather tensors must share their row count")
        if value.device != indices.device or valid.device != indices.device:
            raise ValueError("query gather tensors and maps must share a device")
        expected = 1
        for size, stride in zip(
            reversed(value.shape[1:]), reversed(value.stride()[1:])
        ):
            if size > 1 and stride != expected:
                raise ValueError("query gather inner dimensions must be contiguous")
            expected *= size
    if not indices.is_cuda:
        return tuple(
            torch.where(
                valid.view((-1,) + (1,) * (value.ndim - 1)),
                value.index_select(0, indices.clamp(0, max(rows - 1, 0))),
                torch.zeros((), dtype=value.dtype, device=value.device),
            )
            for value in values
        )
    outputs = tuple(
        value.new_empty((indices.numel(), *value.shape[1:])) for value in values
    )
    if indices.numel() and max(widths):
        _gather_query_rows_kernel[(indices.numel(), triton.cdiv(max(widths), 256))](
            values,
            indices,
            valid,
            outputs,
            *(tuple(value.stride(0) for value in values) + (0,) * (4 - len(values))),
            widths,
            256,
        )
    return outputs


VERIFY_LAYOUT_MAX_TOKENS = 4096


@triton.jit(
    do_not_specialize=["BS", "CAPACITY", "STRIDE", "LATEST_STRIDE", "LATEST_ROWS"]
)
def _verify_layout_kernel(
    Scores,
    Latest,
    Slots,
    ActiveCount,
    Active,
    Ready,
    Rows,
    Indptr,
    Counts,
    Valid,
    BS,
    CAPACITY,
    STRIDE,
    LATEST_STRIDE,
    LATEST_ROWS,
    DRAFT: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_LATEST: tl.constexpr,
    HAS_ACTIVE_COUNT: tl.constexpr,
    PACKED: tl.constexpr,
):
    index = tl.arange(0, BLOCK).to(tl.int64)
    request = index // DRAFT
    column = index % DRAFT
    if PACKED:
        live = tl.load(Active + request, request < BS, False)
        eligible = live & tl.load(Ready + request, request < BS, False)
    else:
        active_bs = tl.load(ActiveCount).to(tl.int32) if HAS_ACTIVE_COUNT else BS
        live = request < active_bs
        eligible = live
    if USE_LATEST:
        slot = tl.load(Slots + request, request < BS, 0).to(tl.int64)
        slot_valid = live & (slot >= 0) & (slot < LATEST_ROWS)
        latest_finite = slot_valid
        for j in tl.static_range(DRAFT):
            value = tl.load(Latest + slot * LATEST_STRIDE + j, slot_valid, 0)
            latest_finite = latest_finite & (tl.abs(value) < float("inf"))
    finite = tl.full((BLOCK,), True, tl.int1)
    survival = tl.full((BLOCK,), 1.0, tl.float32)
    for j in tl.static_range(DRAFT):
        score = tl.load(Scores + request * STRIDE + j, request < BS, 0)
        if USE_LATEST:
            value = tl.load(Latest + slot * LATEST_STRIDE + j, slot_valid, 0)
            score = tl.where(latest_finite, value, score)
        if PACKED:
            probability = tl.where(
                tl.abs(score) < float("inf"),
                tl.minimum(tl.maximum(score, 0.0), 1.0),
                0.0,
            )
        else:
            finite = finite & (tl.abs(score) < float("inf"))
            score = tl.minimum(tl.maximum(score, -60.0), 60.0)
            probability = tl.div_rn(1.0, 1.0 + libdevice.exp(-score))
        survival = survival * tl.where(j <= column, probability, 1.0)
    if PACKED:
        gain = tl.abs(survival)  # Treat signed zeros as the same score.
    else:
        gain = tl.where(finite, survival, float("inf"))
    # Positive IEEE float bits sort in numeric order. The low word makes
    # ties deterministic in request/token order, preserving each prefix.
    key = (gain.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | (
        0xFFFFFFFF - index.to(tl.uint64)
    )
    key = tl.where(eligible, key, 0)
    ordered = tl.sort(key, descending=True)
    if PACKED:
        anchors = tl.load(Active + index, index < BS, False)
        extra = CAPACITY - tl.sum(anchors.to(tl.int32), 0)
    else:
        anchors = index < BS
        extra = CAPACITY - BS
    cutoff = tl.sum(tl.where(index == extra - 1, ordered, 0), 0)
    selected = eligible & (extra > 0) & (key >= cutoff)
    if not PACKED:
        unknown = tl.sum((live & ~finite & (column == 0)).to(tl.int32), 0)
        valid = unknown * DRAFT <= extra
    counts = tl.full((BLOCK,), 1, tl.int32)
    for j in tl.static_range(DRAFT):
        selected_index = tl.minimum(index * DRAFT + j, BLOCK - 1).to(tl.int32)
        counts += tl.gather(selected.to(tl.int32), selected_index, axis=0)
    if PACKED:
        counts = tl.where(anchors, counts, 0)
    else:
        counts = tl.where(anchors, tl.where(valid, counts, 1), 0)
    ends = tl.cumsum(counts, 0)
    tl.store(Counts + index, counts.to(tl.int64), index < BS)
    tl.store(Indptr + index + 1, ends.to(tl.int64), index < BS)
    tl.store(Indptr, 0)
    if not PACKED:
        tl.store(Valid, valid)
    # Upper bound over cumulative lengths maps each packed output to its owner.
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), BS, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.gather(ends, tl.minimum(mid, BLOCK - 1), axis=0)
        active = lo < hi
        right = end <= index
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    owner = tl.minimum(lo, BS - 1)
    before = tl.gather(ends, tl.maximum(owner - 1, 0), axis=0)
    before = tl.where(owner == 0, 0, before)
    total = tl.sum(counts, 0)
    rows = owner.to(tl.int64) * (DRAFT + 1) + index - before
    tl.store(
        Rows + index,
        tl.where(index < total, rows, -1 if PACKED else 0),
        index < CAPACITY,
    )


def launch_verify_layout(
    scores,
    query_capacity,
    rows_out,
    indptr_out,
    counts_out,
    valid_out,
    *,
    latest_scores,
    request_slots,
    active_count,
    active,
    ready,
):
    bs, draft = scores.shape
    block = triton.next_power_of_2(bs * (draft + 1))
    _verify_layout_kernel[(1,)](
        scores,
        latest_scores,
        request_slots,
        active_count,
        active,
        ready,
        rows_out,
        indptr_out,
        counts_out,
        valid_out,
        bs,
        query_capacity,
        scores.stride(0),
        latest_scores.stride(0) if latest_scores is not None else 0,
        latest_scores.shape[0] if latest_scores is not None else 0,
        DRAFT=draft,
        BLOCK=block,
        USE_LATEST=latest_scores is not None,
        HAS_ACTIVE_COUNT=active_count is not None,
        PACKED=active is not None,
        num_warps=4,
    )


def verify_query_prefixes(
    rates: torch.Tensor,
    active: torch.Tensor,
    ready: torch.Tensor,
    *,
    rows_out: torch.Tensor,
    offsets_out: torch.Tensor,
    lengths_out: torch.Tensor,
) -> None:
    """Allocate GPU query prefixes from conditional probabilities in place.

    The FP32 rates have shape [B, W-1]. Boolean masks [B] identify active
    requests and whether they have draft candidates. Each active request gets
    its anchor; only ready requests compete for extra queries. Non-finite
    probabilities contribute zero. The capacity is the length of rows_out.

    Outputs are contiguous int64 source rows (with -1 for filler), int32 CSR
    offsets [B+1], and int32 prefix lengths [B]. All tensors share a GPU.
    """
    batch, draft = rates.shape
    capacity = rows_out.numel()
    if (
        not rates.is_cuda
        or rates.dtype != torch.float32
        or rates.stride(1) != 1
        or min(batch, draft) < 1
        or not batch <= capacity <= batch * (draft + 1) <= VERIFY_LAYOUT_MAX_TOKENS
    ):
        raise ValueError("unsupported packed verification layout")
    for tensor, shape, dtype in (
        (active, (batch,), torch.bool),
        (ready, (batch,), torch.bool),
        (rows_out, (capacity,), torch.int64),
        (offsets_out, (batch + 1,), torch.int32),
        (lengths_out, (batch,), torch.int32),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != rates.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("packed verification buffers must match the layout")
    kernel = select_kernel(
        "metadata",
        "verify_query_prefixes",
        format_signature(x=dense_tensor_format(rates.dtype)),
    )
    kernel(rates, active, ready, rows_out, offsets_out, lengths_out)


@register_kernel(
    "metadata",
    "verify_query_prefixes",
    name="triton_verify_query_prefixes",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=[format_signature(x=dense_tensor_format(torch.float32))],
    priority=Priority.PORTABLE,
)
def triton_verify_query_prefixes(rates, active, ready, rows, offsets, lengths):
    launch_verify_layout(
        rates,
        rows.numel(),
        rows,
        offsets,
        lengths,
        None,
        latest_scores=None,
        request_slots=None,
        active_count=None,
        active=active,
        ready=ready,
    )
